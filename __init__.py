"""ComfyUI nodes for the MiniMax-H3 Turbo LoRA (4-step audio-video).

Drops into the stock MiniMax-H3 workflow (t2v and i2v):

  MiniMaxH3TurboLoRA    MODEL -> MODEL   applies the turbo LoRA
  MiniMaxH3TurboSampler       -> SAMPLER 4-step sampler for SamplerCustomAdvanced

On older ComfyUI the sampler steps the video and audio streams on their own flow
schedules (video shift 12, audio shift 3), because a stock single-schedule sampler
over-steps the audio at 4 steps and it breaks. Recent ComfyUI resolves that dual
schedule natively (ModelSamplingAV), so this sampler auto-detects it and falls back
to a plain single-schedule step, avoiding a double-shift that would corrupt the
audio. Either way it drops into the same workflow slot.
"""

import math
import os

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.model_sampling
import comfy.lora
import comfy.weight_adapter
import comfy.utils
import comfy.patcher_extension
import folder_paths

try:                                       # what the stock samplers step with, so
    from comfy.utils import model_trange as _trange   # the console bar and the
except ImportError:                        # dynamic-VRAM warmup readout match
    from tqdm.auto import trange as _trange

SHIFT_V, SHIFT_A = 12.0, 3.0

# MINIMAX_H3_TURBO_DEBUG=1 turns on the per-step sampler trace and reports the
# size of the injected adaln update against the projection it lands on. A ratio
# well above 1 means the injection is overpowering the model's time conditioning.
_DEBUG = bool(os.environ.get("MINIMAX_H3_TURBO_DEBUG"))


def _dbg(*a, **k):
    if _DEBUG:
        print(*a, **k)


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


def _model_sampling(model):
    """The model's model_sampling instance, reached from the object a KSAMPLER
    hands the sampler function: KSamplerX0Inpaint -> CFGGuider -> predictor, where
    the predictor carries .model_sampling (comfy/samplers.py accesses exactly
    model_wrap.inner_model.model_sampling)."""
    for chain in (("inner_model", "inner_model", "model_sampling"),
                  ("inner_model", "model_sampling"),
                  ("model_sampling",)):
        o = model
        try:
            for a in chain:
                o = getattr(o, a)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def _native_av_schedule(model):
    """True when this ComfyUI resolves the MiniMax-H3 audio/video dual flow
    schedule natively via ModelSamplingAV.

    Recent ComfyUI carries the audio latent scaled onto the video schedule
    (ModelSamplingAV), so the packed latent is an ordinary single-schedule flow
    latent and a plain flow step is correct. Re-applying the audio shift here, as
    older ComfyUI required, would double-shift and corrupt the audio (node issues
    #6 / #18 / #19, HF discussions #17 / #19). Older ComfyUI has no ModelSamplingAV
    and still needs the manual dual-schedule step, so this sampler adapts to
    whichever ComfyUI it runs under."""
    ms = _model_sampling(model)
    if ms is None:
        return False
    if getattr(ms, "audio_shift", None) is not None:
        return True
    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    _rms = lambda t: float(t.float().pow(2).mean().sqrt())

    if _native_av_schedule(model):
        # Recent ComfyUI: ModelSamplingAV already carries the audio stream scaled
        # onto the video schedule, so the pack is an ordinary single-schedule flow
        # latent. Step the whole pack with a plain flow (Euler) update — the model
        # and ModelSamplingAV handle the audio clock. Manually re-shifting the audio
        # here (the legacy path below) would double-apply and corrupt the audio.
        _dbg(f"[H3TURBO sampler] native ModelSamplingAV -> single-schedule Euler  "
              f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "
              f"dtype={x.dtype}", flush=True)
        for i in _trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            _dbg(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  "
                  f"denoised_rms={_rms(denoised):.4f} x_rms={_rms(x):.4f} d_rms={_rms(d):.4f}",
                  flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        return x

    # Legacy ComfyUI without ModelSamplingAV: video and audio ride separate flow
    # schedules (video shift 12, audio shift 3); step each on its own clock. A stock
    # single-schedule sampler over-steps the audio at 4 steps and breaks it — that
    # is the reason this node's sampler exists on older ComfyUI.
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]
    a_numel = (x.shape[-1] - v_numel)
    _dbg(f"[H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  "
          f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "
          f"dtype={x.dtype}  v_numel={v_numel} a_numel={a_numel}  shapes={shapes}", flush=True)
    for i in _trange(len(sigmas) - 1, disable=disable):   # tqdm it/s bar, like stock
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov               # video on its own sigma
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock
        x = torch.cat([xv, xa], dim=-1)
        _dbg(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  denoised_rms={_rms(denoised):.4f}  "
              f"video: x_rms={_rms(xv):.4f} v_rms={_rms(ov):.4f}  "
              f"audio: x_rms={_rms(xa):.4f} v_rms={_rms(oa):.4f} slope={sl:.4f}", flush=True)
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


def _make_adaln_forward(base, a, b, shared):
    """Curve-mode adaln injection as a *forward-attribute* patch: returns a
    replacement AdalnProj.forward that adds B @ A @ silu(t_emb) to the projection
    before the reference view/chunk. Installed via add_object_patch on the
    "<adaln_proj>.forward" attribute, so the module tree is left untouched.

    Why not a wrapper module: replacing the whole AdalnProj with an nn.Module that
    holds the original under .base injects a `.base` submodule (and its
    `.base.linear.weight`) into the model's parameter/buffer tree. ComfyUI's
    dynamic-VRAM streaming loader records every such path in its backup and, on
    unload, restores it by that path via set_attr_param/set_attr_buffer — but the
    object-patch has by then reverted AdalnProj to the plain module, so `.base`
    no longer resolves and it crashes with
    `AttributeError: 'AdalnProj' object has no attribute 'base'` (issue #4).
    Patching only the .forward attribute keeps adaln_proj.linear.weight at its
    natural path, so streaming backup/restore behaves exactly as unpatched.

    a/b are held as plain captured tensors (never registered), so they never enter
    the tree; they're cast to x's device/dtype per call, which also covers the
    VRAM-offload case where the projection runs on GPU while a/b sit on CPU."""

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = shared.get("silu_temb")
        if st is not None:
            if st.shape[0] != x.shape[0]:
                # our row set disagrees with the model's; adding it would line the
                # LoRA's video update up against an audio/cond row
                raise RuntimeError(
                    "MiniMaxH3TurboLoRA: expected {} timestep rows, model built "
                    "{}. The LoRA's time-conditioning can't be aligned — please "
                    "report the workflow.".format(st.shape[0], x.shape[0]))
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            delta = (bv @ (av @ sv.T)).T                              # [M, out]
            if _DEBUG and shared.pop("debug_step", False):
                xn, dn = float(x.detach().norm()), float(delta.detach().norm())
                print("[MiniMaxH3TurboLoRA] t={} |adaln|={:.4f} |delta|={:.4f} "
                      "ratio={:.4f}".format(
                          [round(t, 4) for t in shared.get("t", ())],
                          xn, dn, dn / max(xn, 1e-6)), flush=True)
            x = x + delta
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


class MiniMaxH3TurboSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = ("4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "
                   "SamplerCustomAdvanced and set the scheduler to 4 steps. "
                   "Auto-adapts to the ComfyUI version: on recent builds that "
                   "handle the audio schedule natively (ModelSamplingAV) it steps "
                   "as a plain single-schedule sampler; on older builds it steps "
                   "video and audio on their separate clocks.")

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(_turbo_sampler),)


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """LoRA bypass adapter with a memory-frugal additive path.

    ComfyUI's default bypass is g(base_out + h(x)); for LoRA, h(x) allocates the
    full-size projection twice (`out` and `out * scale`) and the outer add
    allocates a third, so each bypassed layer holds ~3× its output activation
    transiently. On the DiT's MLP down-projection (fc2, out = hidden, over a
    ~46k-token sequence) that is ~1.5 GB of avoidable peak per block and is what
    OOMs low-VRAM / pruned-fp8 runs that used to fit under the old merge path
    (issue #4). Overriding bypass_forward to accumulate up(down(x))*scale straight
    into base_out in place keeps one temporary instead of three. base_out is the
    module's fresh output, so the in-place add is safe. Numerically identical to
    the stock LoRA bypass. Linear-only (all H3 lora modules are Linear); anything
    else falls back to the stock path."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)


# --- lora key naming ------------------------------------------------------
# The published turbo LoRA ships with its keys already prefixed for ComfyUI
# ("diffusion_model.blocks.0.attn..."), while other exports name the same modules
# bare ("blocks.0.attn..."). Everything below rebuilds the model-side path from a
# module name, so strip any prefix first or the result doubles it
# ("diffusion_model.diffusion_model....") and matches nothing.

_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def _dit_path(module):
    for p in _LORA_PREFIXES:
        if module.startswith(p):
            return module[len(p):]
    return module


def _apply_bypass_lora(new_model, lora, modules, strength):
    """Apply the low-rank update at RUN TIME (output = base(x) + lora(x)) via
    ComfyUI's bypass injection, so it is never folded into the weights. The delta
    is tiny relative to the base weight — merging rounds it away in bf16 and
    requantizes it away in int8/fp8 — whereas bypass runs the base's own
    (possibly quantized) forward and adds the bf16 update in activation space,
    exactly like the standalone generate.py reference. The stock
    model_lora_keys_unet does not recognise the H3 lora naming, so build the key
    map directly (module -> diffusion_model.<module>.weight). Adapters are wrapped
    in _FrugalLoRA for the in-place additive path (see its docstring)."""
    key_map = {m: "diffusion_model.{}.weight".format(_dit_path(m)) for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(new_model.model.state_dict().keys())
    n = 0
    for key, adapter in loaded.items():
        if key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        n += 1
    injections = manager.create_injections(new_model.model)
    if manager.get_hook_count() > 0:
        new_model.set_injections("bypass_lora", injections)
    return n


def _apply_merge_lora(new_model, lora, modules, strength):
    """Low-VRAM path: fold the low-rank update into the weights (add_patches, the
    same call ComfyUI's own load_lora_for_models makes), so nothing extra is
    computed at forward time. This is the cheapest on peak VRAM and lets small
    GPUs run, but on a quantized base the delta is partly rounded away when it is
    merged back into int8/fp8 (and on bf16 it sits near the ULP), i.e. softer than
    the bypass path — the sharpness/VRAM trade the low_vram switch exposes."""
    key_map = {m: "diffusion_model.{}.weight".format(_dit_path(m)) for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    return len(new_model.add_patches(loaded, strength))


def _int8_fused_fc2(dm, modules):
    """MLP fc2 modules whose base weight rides ComfyUI's fused int8 matmul.

    comfy.ops.linear_input_act (minimax MLP.forward) folds the swiglu activation
    into an INT8 activation quantizer and calls the fused int8 kernel on
    linear.weight DIRECTLY — it never calls the module's forward, so a
    BypassForwardHook installed on fc2.forward never fires and that fc2's LoRA is
    silently dropped (measured: on int8_convrot the 50 DiT-block fc2 hooks fire 0
    times). Those fc2 must instead go through the merge/weight-function path, where
    ComfyUI dequantizes the int8 weight and applies the LoRA during the weight cast
    (delta preserved in fp32; ~one fc2 weight dequantized transiently per call, no
    resident cost). fc2 on bf16 / fp8 bases is left on bypass — there the fused int8
    path isn't taken (the eager `linear(swiglu(x))` fallback runs) so the hook fires
    normally."""
    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, _dit_path(m) + ".weight")
        except Exception:
            continue
        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(w, "_params", None), "transposed", False)):
            fused.append(m)
    return fused


def _inject_adaln_egrid(new_model, dm, lora, adaln, strength):
    """Pruned/curve base only: the adaln update lives in the 2688-dim silu(t_emb)
    space, which the pruned base has collapsed into a small curve, so it can be
    neither a bypass adapter nor a merged weight patch. Re-inject it at run time —
    a shared silu(t_emb) interpolated from the bundled E-grid each forward, plus a
    forward-attribute patch on each adaln projection that adds B @ A @ silu(t_emb)
    (see _make_adaln_forward). Peak memory is negligible (M <= 3 rows), so this is
    identical in both the bypass and low_vram modes."""
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
        shared["t"] = us
        shared["debug_step"] = _DEBUG      # first adaln of the step reports
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)
    for name in adaln:                       # name = "....adaln_proj.linear"
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + _dit_path(name).rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(new_model.get_model_object(key), a, b, shared))


def _add_dbg_wrapper(new_model, dm, tag, mode):
    """Observability: at diffusion-model forward time, log that the lora is
    actually active this forward, plus the timestep and video/audio input rms.
    Only the first few calls are printed to avoid flooding.

    The activity canary depends on `mode`. In bypass mode a lora'd module's
    forward is taken over by BypassForwardHook, so qkv_proj.forward_owner reads
    `BypassForwardHook` iff the lora is live. In merge mode the delta is folded
    into the weights and the forward stays the base Linear, so the owner is
    expected to be the base module — activity is instead reflected by the layer
    carrying a patch (weight_function), which we report separately."""
    st = {"n": 0}

    def wrap(executor, *args, **kwargs):
        if st["n"] < 6:
            st["n"] += 1
            try:
                m0 = dm.blocks[0].attn.qkv_proj
                owner = type(getattr(m0.forward, "__self__", None)).__name__
                has_wf = bool(getattr(m0, "weight_function", None)) or \
                    getattr(m0, "weight_lowvram_function", None) is not None
            except Exception as e:                       # noqa
                owner, has_wf = "err:%s" % e, "?"
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            xx = args[0] if args else kwargs.get("x")
            try:
                vr = float(xx[0].float().pow(2).mean().sqrt())
                ar = float(xx[1].float().pow(2).mean().sqrt())
                dt = str(xx[0].dtype)
            except Exception:
                vr = ar = -1.0
                dt = "?"
            tsv = float(ts.flatten()[0]) if ts is not None else -1
            if mode == "merge":
                canary = (f"qkv_proj.forward_owner={owner} weight_patched={has_wf} "
                          f"(merge: delta folded into weights; owner is the base "
                          f"Linear, patch presence => lora ACTIVE)")
            else:
                canary = (f"qkv_proj.forward_owner={owner} "
                          f"(BypassForwardHook => lora ACTIVE; else => BASE ONLY!)")
            _dbg(f"[H3TURBO fwd {tag}/{mode}] call#{st['n']}  {canary}  is_injected="
                  f"{getattr(new_model, 'is_injected', '?')}  timestep={tsv:.2f}  "
                  f"video_rms={vr:.4f} audio_rms={ar:.4f} dtype={dt}", flush=True)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo_dbg", wrap)


class MiniMaxH3TurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                   "step": 0.01}),
            "low_vram": ("BOOLEAN", {
                "default": False,
                "label_on": "merge (low VRAM, softer)",
                "label_off": "bypass (sharp, more VRAM)",
                "tooltip": "OFF (default): apply the LoRA at run time (bypass) — "
                           "sharpest, but costs extra peak VRAM. ON: merge the "
                           "LoRA into the weights — lowest VRAM so small GPUs can "
                           "run, but softer on quantized bases (the delta is "
                           "partly rounded away). Turn ON only if you OOM."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = "Apply the MiniMax-H3 Turbo LoRA to the H3 diffusion model."

    def apply_lora(self, model, lora_name, strength, low_vram=False):
        path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        dm = model.model.diffusion_model
        pruned = getattr(dm, "use_adaln_curves", False)
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
        new_model = model.clone()
        mode = "merge" if low_vram else "bypass"

        # On the pruned base the adaln update can't be a weight patch (it lives in
        # the collapsed silu(t_emb) curve), so it is always re-injected at run
        # time regardless of mode; everything else is the "backbone", which takes
        # the bypass or merge path per low_vram.
        if pruned:
            backbone = [m for m in modules if "adaln_proj" not in m]
            adaln = [m for m in modules if "adaln_proj" in m]
        else:
            backbone, adaln = modules, []

        n_fc2 = 0
        if low_vram:
            n = _apply_merge_lora(new_model, lora, backbone, strength)
        else:
            # int8-fused fc2 is invisible to the bypass hook — apply those via merge,
            # the rest via bypass (see _int8_fused_fc2).
            fc2_fused = set(_int8_fused_fc2(dm, backbone))
            bypass_mods = [m for m in backbone if m not in fc2_fused]
            n = _apply_bypass_lora(new_model, lora, bypass_mods, strength)
            if fc2_fused:
                n_fc2 = _apply_merge_lora(new_model, lora, sorted(fc2_fused), strength)
                n += n_fc2
        if backbone and not n:
            # every module was dropped: wrong lora, wrong model, or a key naming
            # this build doesn't map. Silent no-op reads as "lora applied".
            raise RuntimeError(
                "MiniMaxH3TurboLoRA: none of the LoRA's {} backbone modules exist "
                "in the loaded model (looked for '{}'). Is this a MiniMax-H3 "
                "LoRA, and is the model a MiniMax-H3 base?".format(
                    len(backbone),
                    "diffusion_model.{}.weight".format(_dit_path(backbone[0]))))
        if pruned and adaln:
            _inject_adaln_egrid(new_model, dm, lora, adaln, strength)

        try:
            p0 = dm.blocks[0].attn.qkv_proj.weight
            wdt, wdev = str(p0.dtype), str(p0.device)
        except Exception:
            wdt, wdev = "?", "?"
        if low_vram:
            detail = f"{n} weights patched (merged)"
        else:
            injs = new_model.injections.get("bypass_lora", [])
            detail = f"{n - n_fc2} bypass adapters, {len(injs)} injections"
            if n_fc2:
                detail += f", {n_fc2} int8 fc2 via merge"
        extra = f" + {len(adaln)} adaln injected at run time" if adaln else ""
        print(f"[MiniMaxH3TurboLoRA] {'pruned' if pruned else 'full'} base [{mode}]: "
              f"lora={lora_name} strength={strength} | {len(backbone)} backbone "
              f"modules, {detail}{extra} | model={type(new_model.model).__name__} "
              f"weight_dtype={wdt} weight_dev={wdev}", flush=True)
        _add_dbg_wrapper(new_model, dm, "pruned" if pruned else "full", mode)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboLoRA": MiniMaxH3TurboLoRA,
    "MiniMaxH3TurboSampler": MiniMaxH3TurboSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboLoRA": "MiniMax-H3 Turbo LoRA",
    "MiniMaxH3TurboSampler": "MiniMax-H3 Turbo Sampler (4-step)",
}
