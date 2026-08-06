"""ComfyUI nodes for the MiniMax-H3 Turbo LoRA (4-step audio-video).

Drops into the stock MiniMax-H3 workflow (t2v and i2v):

  MiniMaxH3TurboLoRA    MODEL -> MODEL   applies the turbo LoRA
  MiniMaxH3TurboSampler       -> SAMPLER 4-step sampler for SamplerCustomAdvanced

The sampler steps the video and audio streams on their own flow schedules
(video shift 12, audio shift 3). A stock single-schedule sampler over-steps the
audio at 4 steps and the audio breaks; this one keeps each stream on its own
clock so 4 steps stays clean.
"""

import math
import os

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.lora
import comfy.utils
import comfy.patcher_extension
import folder_paths

SHIFT_V, SHIFT_A = 12.0, 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv):
    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)


def _audio_slope(sv):
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model):
    """[video_shape, audio_shape] the sampler is packing over — video latent is
    flattened first, then audio, so we need the split point."""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   **kwargs):
    extra_args = {} if extra_args is None else extra_args
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov               # video on its own sigma
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock
        x = torch.cat([xv, xa], dim=-1)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


# --- pruned / curve-mode base support -------------------------------------
# Pruned H3 checkpoints replace the time embedder + full-width adaln with a small
# 8-dim curve (adaln_t_table), so the LoRA's adaln update (which lives in the
# 2688-dim silu(t_emb) space) can't be applied as a weight patch. Instead we
# re-inject it at run time: a shared silu(t_emb) is interpolated from a bundled
# grid each forward, and each adaln projection adds B @ A @ silu(t_emb) to its
# output. The backbone (attn/mlp/refiner) is patched normally.

_EGRID = None


def _egrid():
    global _EGRID
    if _EGRID is None:
        p = os.path.join(os.path.dirname(__file__), "h3_silu_temb_grid.safetensors")
        _EGRID = comfy.utils.load_torch_file(p)["silu_t_emb_grid"]   # [1025, 2688]
    return _EGRID


VISUAL_COND_T, AUDIO_COND_T = 0.999, 1.0


def _cond_kinds(payload):
    """Conditioning segment kinds present in the packed sequence.

    The model decides its timestep rows from the packed layout's segments, not
    from the raw keyframes/refs, and that layout rides in the payload — so read
    it rather than re-deriving it. The keyframes/refs path mirrors PackedLayout
    for the case where the layout wasn't prebuilt."""
    layout = payload.get("layout")
    segments = getattr(layout, "segments", None)
    if segments is not None:
        return {k for _, _, k in segments}
    kinds = set()
    if payload.get("keyframes"):
        kinds.add("cond")
    for blk in payload.get("refs") or []:
        kind = blk.get("kind")
        if kind in ("image", "video", "video_audio"):
            kinds.add("ref_img")
        if kind in ("audio", "video", "video_audio") and blk.get("ref_audio_t", 0) > 0:
            kinds.add("ref_audio")
    return kinds


def _unique_t(timestep, shift_v, shift_a, payload):
    """The model's distinct per-row timesteps, in its row order.

    Must match comfy.ldm.minimax.model._forward exactly — these index the adaln
    rows our delta is added to, so a single extra or missing row misaligns every
    injection. Arithmetic stays on the fp32 tensor for the same reason: t_v is
    compared against the cond timestep, and doing the subtract in float64
    instead can split a value the model collapsed (or vice versa)."""
    sv = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sv)
    t_a = float(1.0 - _time_shift_sigma(sv, shift_v, shift_a))
    kinds = _cond_kinds(payload)
    s = {t_v, t_a}
    if kinds & {"cond", "ref_img"}:
        s.add(max(t_v, float(payload.get("visual_cond_noise_aug", VISUAL_COND_T))))
    if "ref_audio" in kinds:
        s.add(max(t_a, float(payload.get("audio_cond_noise_aug", AUDIO_COND_T))))
    return sorted(s)


def _interp_egrid(unique_t, E, device, dtype):
    n = E.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(E[i0].float(), E[i0 + 1].float(), pos - i0))
    # interpolate on the cpu-resident grid, ship only the [M, 2688] result
    return torch.stack(rows).to(device=device, dtype=dtype)


class _AdalnDelta(torch.nn.Module):
    """Curve-mode AdalnProj wrapper: adds B @ A @ silu(t_emb) to the projection
    output before the reference view/chunk."""

    def __init__(self, base, a, b, shared):
        super().__init__()
        self.base = base
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self.shared = shared

    def forward(self, t_emb):
        base = self.base
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = self.shared.get("silu_temb")
        if st is not None:
            if st.shape[0] != x.shape[0]:
                # our row set disagrees with the model's; adding it would line the
                # LoRA's video update up against an audio/cond row
                raise RuntimeError(
                    "MiniMaxH3TurboLoRA: expected {} timestep rows, model built "
                    "{}. The LoRA's time-conditioning can't be aligned — please "
                    "report the workflow.".format(st.shape[0], x.shape[0]))
            a, b = self.a.to(x.dtype), self.b.to(x.dtype)
            x = x + (b @ (a @ st.to(a.device, x.dtype).T)).T          # [M, out]
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)


# --- lora key naming ------------------------------------------------------
# The turbo LoRA ships with its keys already prefixed for ComfyUI
# ("diffusion_model.blocks.0.attn..."), but other exports name the same modules
# bare ("blocks.0.attn..."). Strip any prefix so we can rebuild the model-side
# path ourselves instead of doubling it.

_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def _dit_path(module):
    for p in _LORA_PREFIXES:
        if module.startswith(p):
            return module[len(p):]
    return module


def _weight_patch(new_model, lora, to_load, strength, log_missing=True):
    """add_patches drops keys the model doesn't have, so check what landed."""
    applied = new_model.add_patches(
        comfy.lora.load_lora(lora, to_load, log_missing=log_missing), strength)
    if to_load and not applied:
        k = next(iter(to_load))
        raise RuntimeError(
            "MiniMaxH3TurboLoRA: none of the LoRA's {} modules exist in the "
            "loaded model (looked for '{}'). Is this a MiniMax-H3 LoRA, and is "
            "the model a MiniMax-H3 base?".format(len(to_load), to_load[k]))
    return applied


class MiniMaxH3TurboSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = ("4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "
                   "SamplerCustomAdvanced and set the scheduler to 4 steps.")

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(_turbo_sampler),)


class MiniMaxH3TurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                   "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = "Apply the MiniMax-H3 Turbo LoRA to the H3 diffusion model."

    def apply_lora(self, model, lora_name, strength):
        path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})
        dm = model.model.diffusion_model
        pruned = getattr(dm, "use_adaln_curves", False)
        new_model = model.clone()

        if not pruned:
            to_load = {m: "diffusion_model.{}.weight".format(_dit_path(m))
                       for m in modules}
            applied = _weight_patch(new_model, lora, to_load, strength)
            print(f"[MiniMaxH3TurboLoRA] {len(applied)}/{len(to_load)} modules "
                  f"patched", flush=True)
            return (new_model,)

        # pruned/curve base: backbone via weight patches, adaln via run-time delta
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
        to_load = {m: "diffusion_model.{}.weight".format(_dit_path(m))
                   for m in backbone}
        # the adaln keys are handled below, so don't let load_lora cry about them
        applied = _weight_patch(new_model, lora, to_load, strength,
                                log_missing=False)

        E = _egrid()
        shared = {"silu_temb": None}
        shift_v = float(getattr(dm, "sigma_shift_video", SHIFT_V))
        shift_a = float(getattr(dm, "sigma_shift_audio", SHIFT_A))

        def wrap(executor, *args, **kwargs):
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            ctx = args[2] if len(args) > 2 else kwargs.get("context")
            payload = kwargs.get("minimax_payload") or {}
            to = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
            sv = float(to.get("minimax_h3_sigma_shift_video", shift_v))
            sa = float(to.get("minimax_h3_sigma_shift_audio", shift_a))
            us = _unique_t(ts, sv, sa, payload)
            # curve-mode adaln runs in fp32, so keep the grid rows there too
            shared["silu_temb"] = _interp_egrid(us, E, ctx.device, torch.float32)
            return executor(*args, **kwargs)

        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)
        for name in adaln:                       # name = "....adaln_proj.linear"
            a = lora[name + ".lora_A.weight"]
            b = lora[name + ".lora_B.weight"] * strength
            key = "diffusion_model." + _dit_path(name).rsplit(".linear", 1)[0]
            try:
                base = new_model.get_model_object(key)
            except AttributeError:
                raise RuntimeError(
                    "MiniMaxH3TurboLoRA: the model has no '{}' to inject the "
                    "LoRA's adaln update into.".format(key)) from None
            new_model.add_object_patch(key, _AdalnDelta(base, a, b, shared))
        print(f"[MiniMaxH3TurboLoRA] pruned base: {len(applied)}/{len(backbone)} "
              f"backbone patched + {len(adaln)} adaln injected at run time",
              flush=True)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboLoRA": MiniMaxH3TurboLoRA,
    "MiniMaxH3TurboSampler": MiniMaxH3TurboSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboLoRA": "MiniMax-H3 Turbo LoRA",
    "MiniMaxH3TurboSampler": "MiniMax-H3 Turbo Sampler (4-step)",
}
