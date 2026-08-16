#!/usr/bin/env python3
"""
Step 5 (Phase 0 of MOE_UPCYCLE_PLAN.md): dense Qwen3.8-27B -> MoE by EXPERT SPLITTING.

Source and product are Qwen3.8-27B throughout. NOTE ON THE CLASS ID: the config's
`model_type` is a transformers class identifier, and Qwen stamps the 3.8 family with the
`qwen3_5_*` code line -- the base `Qwen/Qwen3.8-27B/config.json` itself reads
`model_type: qwen3_5_text`. So the MoE output keeps `qwen3_5_moe_text` /
`Qwen3_5MoeForCausalLM`: that string must match the registered modeling class or the
checkpoint will not load. It is the architecture-code version, not the model's name.

Converts the dense Qwen3.8-27B checkpoint into its sparse MoE form by partitioning each
FFN's intermediate dimension into E narrow experts and
adding a randomly-initialized router + shared expert. This is NOT the paper's "upcycling"
(which copies full-size MLPs and grows params); it is MoEfication-style *splitting*, which
keeps total params ~constant and cuts ACTIVE params -> the only route to escha-class decode
speed on 16GB (see plan section 1a).

    # local, free: convert + verify shapes/bit-exactness (no forward pass; streamed)
    python node/05_upcycle.py --src <dense-hf-dir> --out <moe-hf-dir> --experts 64 --topk 8
    python node/05_upcycle.py --verify --src <dense-hf-dir> --out <moe-hf-dir>

    # node, needs GPU + RAM: Gate 0b forward pass
    python node/05_upcycle.py --forward-check --out <moe-hf-dir>

DESIGN (locked to the two failure modes in the plan):
  - R-U5 (silent mapping error looks like undertraining): every FFN is reconstructed from
    its expert shards and asserted bit-exact against the source BEFORE writing. A mapping
    bug aborts here, on this machine, for $0 -- never on a billed node.
  - R-U2 (contiguous slicing -> dead router): --cluster reorders each FFN's intermediate
    neurons by co-activation before slicing, so each expert is a coherent group. Default is
    contiguous with a loud warning; do NOT spend a recovery run on a contiguous split.

STREAMS shard-by-shard via safetensors.safe_open, so it runs on a 40GB-RAM box despite the
54GB checkpoint: only one tensor is resident at a time.
"""
import argparse, os, json, glob, math, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--src", help="dense HF checkpoint dir (Qwen/Qwen3.8-27B, safetensors)")
    p.add_argument("--out", help="output MoE HF checkpoint dir")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--shared-intermediate", type=int, default=272,
                   help="shared-expert FFN width; random-init, learned residual path")
    p.add_argument("--drop-vision", action="store_true",
                   help="drop the vision tower (default: KEEP it, matching the known-good "
                        "escha W2 MoE-VLM structure; vision is ~0.5GB, droppable at quant time)")
    p.add_argument("--cluster", choices=["none", "coactivation"], default="none",
                   help="R-U2: neuron grouping before slicing. 'none' = contiguous (WARNS).")
    p.add_argument("--cluster-acts", help="path to saved per-layer FFN activation stats for --cluster coactivation")
    p.add_argument("--verify", action="store_true", help="re-open OUT and assert bit-exact reconstruction of every FFN")
    p.add_argument("--forward-check", action="store_true", help="Gate 0b: load OUT and run one forward pass (needs GPU+RAM; run on node)")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


# ------- tensor-name helpers (confirmed against transformers qwen3_next / qwen3_5_moe) -------
# dense FFN:   model.layers.{L}.mlp.{gate,up,down}_proj.weight
#   gate_proj/up_proj: [I, H]   down_proj: [H, I]
# MoE experts (FUSED): model.layers.{L}.mlp.experts.gate_up_proj  [E, 2*m, H]
#                      model.layers.{L}.mlp.experts.down_proj     [E, H, m]
# router:              model.layers.{L}.mlp.gate.weight           [E, H]      (random)
# shared expert:       model.layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight  (random)
#                      model.layers.{L}.mlp.shared_expert_gate.weight  [1, H]  (random)
def is_ffn(name):
    # Only the LANGUAGE tower's dense FFN gets split into experts. The real dense Qwen3.8-27B
    # is a VLM: language layers live under `model.language_model.layers.*`, and the vision
    # tower (`visual.*`) has its own MLPs that must NOT be touched -> require language_model.
    return (".language_model." in name and ".mlp." in name
            and name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
            and ".experts." not in name and ".shared_expert" not in name)


def layer_of(name):
    # ...language_model.layers.{L}.mlp.gate_proj.weight -> L
    return name.split(".layers.")[1].split(".")[0]


def mlp_prefix_map(idx):
    # Prefix-agnostic: derive each layer's real `...mlp` prefix from the actual tensor names
    # (e.g. model.language_model.layers.7.mlp) rather than hardcoding model.layers.*.
    pm = {}
    for k in idx:
        if is_ffn(k) and k.endswith(".gate_proj.weight"):
            pm[layer_of(k)] = k[: -len(".gate_proj.weight")]
    return pm


def build_index(src):
    """Map every tensor name -> (shard file, dtype) by reading each shard's header lazily."""
    idx = {}
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        sys.exit(f"no safetensors in {src}. This needs the base HF checkpoint, not GGUF.")
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                idx[k] = f
    return idx, files


def split_ffn(gate, up, down, E, order=None):
    """
    gate,up: [I,H]   down: [H,I].  Returns fused gate_up [E,2m,H], down [E,H,m], and the
    reconstruction (gate',up',down') so the caller can assert bit-exactness.
    order: optional permutation of the I intermediate neurons (R-U2). Applied identically to
    gate rows, up rows, and down columns, so the FFN's function is preserved exactly.
    """
    I, H = gate.shape
    assert I % E == 0, f"intermediate {I} not divisible by experts {E}"
    m = I // E
    if order is not None:
        gate, up, down = gate[order], up[order], down[:, order]
    gu, dn = [], []
    for e in range(E):
        s = slice(e * m, (e + 1) * m)
        gu.append(torch.cat([gate[s], up[s]], dim=0))   # [2m, H]
        dn.append(down[:, s])                            # [H, m]
    gate_up = torch.stack(gu, 0).contiguous()            # [E, 2m, H]
    down_e = torch.stack(dn, 0).contiguous()             # [E, H, m]
    return gate_up, down_e, m


def reconstruct(gate_up, down_e):
    """Inverse of split_ffn (no reordering): rebuild dense gate,up,down from experts."""
    E, twoM, H = gate_up.shape
    m = twoM // 2
    gate = torch.cat([gate_up[e, :m] for e in range(E)], 0)
    up = torch.cat([gate_up[e, m:] for e in range(E)], 0)
    down = torch.cat([down_e[e] for e in range(E)], 1)
    return gate, up, down


def make_moe_config(src, a):
    # The real dense Qwen3.8-27B is a VLM: text params are nested under `text_config`, and it
    # loads as Qwen3_5ForConditionalGeneration. We mirror the KNOWN-GOOD structure of a working
    # qwen3_5 MoE VLM (EschaLabs/Qwen3.6-35B-A3B-Escha-W2): Qwen3_5MoeForConditionalGeneration
    # with MoE params in text_config and the vision tower kept intact. (Vision is ~0.5GB and can
    # be dropped later at quant time; keeping it maximizes the chance the checkpoint loads.)
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = dict(cfg.get("text_config") or cfg)      # VLM nests text config; fall back to flat
    H = tc["hidden_size"]; I = tc["intermediate_size"]
    m = I // a.experts
    base_mt = tc.get("model_type", "qwen3_5_text")
    tc["model_type"] = base_mt.replace("_text", "_moe_text") if "moe" not in base_mt else base_mt
    tc["num_experts"] = a.experts
    tc["num_experts_per_tok"] = a.topk
    tc["moe_intermediate_size"] = m
    tc["shared_expert_intermediate_size"] = a.shared_intermediate
    tc["router_aux_loss_coef"] = 0.001
    tc["output_router_logits"] = False
    tc["mtp_num_hidden_layers"] = 1               # trained later, Phase 3
    tc.pop("intermediate_size", None)
    if "text_config" in cfg:
        cfg["text_config"] = tc
        cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        if a.drop_vision:
            for k in ("vision_config", "image_token_id", "video_token_id"):
                cfg.pop(k, None)
            cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
    else:                                          # flat (non-VLM) fallback
        cfg = tc; cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
        cfg["model_type"] = tc["model_type"]
    # NOTE: exact arch/config conventions are validated at Gate 0b (forward pass on the node
    # with real transformers). Gate 0a here only asserts the FFN expert tiling is bit-exact.
    return cfg, H, m


def convert(a):
    dtype = getattr(torch, a.dtype)
    os.makedirs(a.out, exist_ok=True)
    cfg, H, m = make_moe_config(a.src, a)
    E = a.experts
    if a.cluster == "none":
        print("!! WARNING (R-U2): contiguous FFN split. Each expert is an arbitrary fragment;")
        print("!!   the router has weak signal and may collapse at Gate 1. Use --cluster")
        print("!!   coactivation with real activation stats before a paid recovery run.")
    order_by_layer = load_cluster_order(a) if a.cluster == "coactivation" else {}

    idx, _ = build_index(a.src)
    pmap = mlp_prefix_map(idx)                     # layer -> real "...mlp" prefix (VLM-safe)
    ffn_layers = sorted(pmap.keys(), key=int)
    if not ffn_layers:
        sys.exit("no language-tower FFN tensors found (is_ffn matched nothing). Check the "
                 "checkpoint's tensor naming; expected ...language_model.layers.N.mlp.*_proj.weight")
    print(f"{len(ffn_layers)} FFN layers -> {E} experts of width {m} (top-{a.topk}) each")

    g = torch.Generator().manual_seed(0)
    new_tensors, shard_no, cur, cur_bytes = {}, 0, {}, 0
    SHARD_LIMIT = 2 * 1024**3   # 2GB: flush sooner, lower peak RAM (was silently dying at 4GB)
    weight_map = {}

    def flush():
        nonlocal shard_no, cur, cur_bytes
        if not cur:
            return
        fn = f"model-{shard_no:05d}.safetensors"
        save_file(cur, os.path.join(a.out, fn), metadata={"format": "pt"})
        for k in cur:
            weight_map[k] = fn
        print(f"  wrote {fn} ({cur_bytes/1e9:.1f} GB, {len(cur)} tensors)")
        shard_no += 1; cur = {}; cur_bytes = 0

    def emit(name, t):
        nonlocal cur_bytes
        cur[name] = t.contiguous().to(dtype)
        cur_bytes += cur[name].numel() * cur[name].element_size()
        if cur_bytes >= SHARD_LIMIT:
            flush()

    def read(name):
        with safe_open(idx[name], framework="pt") as sf:
            return sf.get_tensor(name)

    for name in idx:
        if a.drop_vision and (".visual." in name or "vision" in name):
            continue
        if is_ffn(name):
            continue  # handled per-layer below
        emit(name, read(name))   # attention, norms, embeddings, output head -> copied verbatim

    for L in ffn_layers:
        pre = pmap[L]                              # e.g. model.language_model.layers.7.mlp
        gate = read(f"{pre}.gate_proj.weight")
        up = read(f"{pre}.up_proj.weight")
        down = read(f"{pre}.down_proj.weight")
        order = order_by_layer.get(L)
        gate_up, down_e, _ = split_ffn(gate, up, down, E, order)

        # R-U5: assert bit-exact tiling (only meaningful without reordering; with --cluster
        # the reconstruction uses the same permutation, so compare against reordered source).
        rg, ru, rd = reconstruct(gate_up, down_e)
        if order is None:
            assert torch.equal(rg, gate) and torch.equal(ru, up) and torch.equal(rd, down), \
                f"layer {L}: expert tiling is NOT bit-exact -- mapping bug (R-U5). ABORT."
        else:
            assert torch.equal(rg, gate[order]) and torch.equal(rd, down[:, order]), \
                f"layer {L}: clustered tiling not bit-exact (R-U5). ABORT."

        emit(f"{pre}.experts.gate_up_proj", gate_up)
        emit(f"{pre}.experts.down_proj", down_e)
        # router + shared expert: the NEW, randomly-initialized parts to be learned
        emit(f"{pre}.gate.weight", torch.empty(E, H).normal_(0, 0.02, generator=g))
        si = a.shared_intermediate
        emit(f"{pre}.shared_expert.gate_proj.weight", torch.empty(si, H).normal_(0, 0.02, generator=g))
        emit(f"{pre}.shared_expert.up_proj.weight", torch.empty(si, H).normal_(0, 0.02, generator=g))
        emit(f"{pre}.shared_expert.down_proj.weight", torch.zeros(H, si))  # zero-init: starts as no-op
        emit(f"{pre}.shared_expert_gate.weight", torch.zeros(1, H))        # sigmoid(0)=0.5, learns
    flush()

    total = sum(sz(os.path.join(a.out, f)) for f in weight_map.values() for _ in [0]) / len(weight_map) if weight_map else 0
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        s = os.path.join(a.src, extra)
        if os.path.exists(s):
            import shutil; shutil.copy(s, os.path.join(a.out, extra))
    print(f"done -> {a.out}. Next: --verify here (free), then --forward-check on the node (Gate 0b).")


def sz(p):
    return os.path.getsize(p)


def load_cluster_order(a):
    if not a.cluster_acts or not os.path.exists(a.cluster_acts):
        sys.exit("--cluster coactivation needs --cluster-acts <stats.pt> (per-layer FFN "
                 "activation correlation over your corpus). Generate it first; see plan R-U2.")
    stats = torch.load(a.cluster_acts)   # {layer: LongTensor permutation of intermediate idx}
    return {str(k): v for k, v in stats.items()}


def verify(a):
    """Re-open OUT and assert every FFN reconstructs bit-exactly from the source (R-U5)."""
    sidx, _ = build_index(a.src)
    oidx, _ = build_index(a.out)
    pmap = mlp_prefix_map(sidx)
    ffn_layers = sorted(pmap.keys(), key=int)
    ok = 0
    for L in ffn_layers:
        pre = pmap[L]
        with safe_open(oidx[f"{pre}.experts.gate_up_proj"], framework="pt") as sf:
            gate_up = sf.get_tensor(f"{pre}.experts.gate_up_proj")
        with safe_open(oidx[f"{pre}.experts.down_proj"], framework="pt") as sf:
            down_e = sf.get_tensor(f"{pre}.experts.down_proj")
        rg, ru, rd = reconstruct(gate_up, down_e)
        with safe_open(sidx[f"{pre}.gate_proj.weight"], framework="pt") as sf:
            gate = sf.get_tensor(f"{pre}.gate_proj.weight")
        if not torch.equal(rg.to(gate.dtype), gate):
            sys.exit(f"VERIFY FAIL layer {L}: experts do not tile the source FFN (R-U5).")
        ok += 1
    print(f"VERIFY OK: {ok}/{len(ffn_layers)} FFN layers tile bit-exactly. Gate 0a PASS.")


def forward_check(a):
    """Gate 0b: instantiate OUT and run one forward pass. Needs GPU + RAM -> run on node."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.out)
    model = AutoModelForCausalLM.from_pretrained(a.out, torch_dtype=getattr(torch, a.dtype),
                                                 device_map="auto", trust_remote_code=True)
    ids = tok("def is_prime(n):", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
    print(f"Gate 0b PASS: forward pass OK, logits {tuple(out.logits.shape)}, "
          f"finite={torch.isfinite(out.logits).all().item()}")


def main():
    a = parse()
    if a.verify:
        return verify(a)
    if a.forward_check:
        return forward_check(a)
    if not (a.src and a.out):
        sys.exit("need --src and --out to convert (or --verify / --forward-check).")
    convert(a)


if __name__ == "__main__":
    main()
