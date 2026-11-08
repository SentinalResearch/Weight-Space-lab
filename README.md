# weight-space-lab

**How far can you go with model weights without training?**

Experiments with Qwen3-0.6B (596M params): random init, weight merging, MoE
upcycling, router distillation, and VQ compression — every claim backed by a
measured perplexity number. Code and blog for [sentinalresearch.com] (https://sentinalresearch.com).

## The thesis

Everything a model "knows" is a *direction* in its 596M-dimensional weight
space. Training-free math can **recombine and compress** those directions;
only gradient descent **creates** new ones. We put a ppl number on both sides
of that line.

## Results (sandbox-measured, CPU; GPU reruns in `results/`)
   experiment | result | lesson |
 |---|---|---|
 | random init, seed=0 | ppl 123,594 ≈ vocab; logit std 0.641 = 0.02·√1024 (exact) | a model = `(config, seed)`; fingerprint reproducible |
 | soup base+instruct | ppl **8** (parents 28/10) | same-basin merges are free (cos 0.991) |
 | split merge bottom=base, top=instruct | ppl **4** | layer ranges matter |
 | layer concat 14+14 | ppl 10.4 | depth concat works within a basin |
 | soup of two random models | logit std 0.641→0.453 = /√2 | orthogonal vectors cancel |
 | mix 25% random into trained | ppl 10→739 | sharp knee: 25% noise kills facts |
 | delta = trained−random | ppl 1515, norm weights → −0.83 | subtraction doesn't extract relations |
 | width expand trained+random (μP) | ppl 280→120 | scale matters, still degraded |
 | **MoE upcycle 0.6B→4.82B (E=17, top-2), expert0=dense** | pre-ft ppl 151; router distilled 377→126 | capacity via structure, router is 2 MB |
 | **VQ weight compression (codes+int8 residual)** | 2.00× smaller, ppl 12.2 vs 11.8 ' Paris' | delta coding = lossless-ish; codebook itself useless at 0.6B (weights ~Gaussian, recon err 0.999) |
 | **VQ memory + entity retrieval** | context 869→96 tokens, ppl 1.26→1.1, answer kept | activations cluster (3/128 codes) → 10× context |
 | cross-size task vector 1.7B→0.6B (SVD rank-32, α) | α=0.1: ppl 11.8→7.0; α=1.0: 547 | projected deltas work at low dose only |
 | frozen LLM + 66K-param routed neurons | ppl 11.8→1.0, router α flips per domain | hypernetwork-style dynamic weights work |

## Repo layout
weight-space-lab/
├── README.md                 # the blog post / project overview
├── requirements.txt          # torch, transformers, accelerate, safetensors, ...
├── bench.py                  # one results row per artifact: wikitext ppl, needle@ctx, tok/s, VRAM, disk
├── build_model.py            # random model from (config, seed) + fingerprint
├── inspect_init.py           # verify init stats (std 0.02, norms 1.0, tied lm_head)
├── merge.py                  # soups + layer-range splices (base/instruct)
├── upcycle_moe.py            # dense 0.6B -> MoE 4.82B, expert0 = dense MLP
├── distill_router.py         # KL-distill the 2 MB router from the dense teacher
├── save_vq.py                # weights -> VQ codes + int8 residuals artifact
├── load_vq.py                # artifact -> runnable HF checkpoint
├── compressive_memory.py     # long context -> VQ triggers + entity retrieval
├── ean_neurons.py            # frozen LLM + routed low-rank neurons (optional)
├── figures.py                # plots from results.json
├── configs/
│   └── qwen3_06b.py          # exact 0.6B config as data
└── results/
├── results.json          # every bench.py row, appended automatically
└── figs/                 # knee.png, router_alpha.png, context_curve.png, storage.png, vram.png

## Run it

```console
pip install torch transformers accelerate safetensors
python bench.py --model Qwen/Qwen3-0.6B --tag original
python bench.py --model Qwen/Qwen3-0.6B --4bit --tag nf4
python save_vq.py --model Qwen/Qwen3-0.6B --out vq_artifact
python load_vq.py --artifact vq_artifact --out vq_model
python bench.py --model vq_model --tag vq-decoded
python upcycle_moe.py
python distill_router.py --steps 8000 --out qwen3-moe-distill3   # nohup on GPU
python compressive_memory.py --ctx 8192 --K 16
```

## Blog outline (sentinalresearch.com)

1. **Hook: a 596M-param model generated from the number 42**
   `(config, seed)` = the whole model; SHA-256 fingerprint reproducible across machines.

2. **What random init actually is**
   Every Linear/Embedding weight ~ N(0, 0.02²); RMSNorm weights = 1.0; tied lm_head.
   Measured: logit std 0.641 = 0.02·√1024 exactly; ppl 123,594 ≈ vocab; repetition loops from tied embeddings.

3. **The weight-surgery scoreboard**
   Table of every training-free operation with ppl: soups (8), split merges (4), concat (10.4),
   random soup (/√2 law), 25% mix knee (10→739), delta=trained−random (1515, norms flip),
   μP width expansion (280→120). Figure: ppl vs % random contamination.

4. **Growing a model without training**
   MoE upcycle 0.6B→4.82B (E=17 experts, top-2, expert 0 = bit-exact dense copy);
   2 MB router distilled from the dense teacher: ppl 377→126. Figure: router α per layer/domain.

5. **Compression as a format**
   VQ codes + int8 residual = delta coding. Weights at 0.6B are ~Gaussian (codebook recon err 0.999,
   residual carries everything → 2.00× smaller, ppl 12.2 vs 11.8, same answer).
   Activations cluster (3/128 codes used) → context 869→96 tokens, ppl 1.26→1.1.
   Figures: storage bar chart, context-compression curve. NF4 row for comparison.

6. **Memory that doesn't look back**
   Entity-keyed retrieval over VQ trigger memory: needle-in-a-haystack @ 2k/8k accuracy vs context length;
   where it breaks (mid-context facts, position loss, retrieval key quality).

7. **The boundary**
   Training-free math = recombine/rescale/compress within the convex hull of trained checkpoints
   (plus permutations per Git Re-Basin). New capability requires gradient descent.
   Future work: neuron-router (EAN), cross-size task vectors at α≈0.1, weight codebooks at 30B+ scale.

8. **Repro appendix**
   Repo layout, bench.py commands, hardware, sandbox numbers as quick-repro table.

---

## References

- Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023
- Wortsman et al., *Model soups*, 2022
- Ainslie et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*, 2023
- Yadav et al., *TIES-Merging*, 2023; Yu et al., *DARE*, 2024
- Meng et al., *Locating and Editing Factual Associations in GPT (ROME)*, 2022; *MEMIT*, 2023
- Ha et al., *HyperNetworks*, 2016
- Hu et al., *LoRA*, 2022; Dettmers et al., *QLoRA / NF4*, 2023; Lin et al., *AWQ*, 2023; Frantar et al., *GPTQ*, 2022
- van den Oord et al., *VQ-VAE*, 2017
- Ge et al., *ICAE: Context Compression via Autoencoders*, 2023; Wu et al., *Memorizing Transformer*, 2022
- Lewis et al., *RAG*, 2020; Meng et al., *causal tracing*, 2022
- Yang & Hu, *μP / maximal update parametrization*, 2021; Chen et al., *Net2Net*, 2015
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015
- H2O / SnapKV (KV-cache eviction/compression), 2023
- Qwen3 Technical Report, 2025

  
