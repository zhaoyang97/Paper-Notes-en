---
title: >-
  [Paper Note] Fast-MIA: Efficient and Scalable Membership Inference for LLMs
description: >-
  [ACL 2026][LLM Safety][Membership Inference] Fast-MIA integrates 9 mainstream LLM Membership Inference Attack (MIA) methods into a single vLLM batch inference engine and adds a cross-method log-prob cache layer. This ach…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Membership Inference"
  - "vLLM"
  - "Cross-method Caching"
  - "WikiMIA"
  - "MIMIR"
date: 2026-05-08
content_hash: e45c770b2f0875f4
---

# Fast-MIA: Efficient and Scalable Membership Inference for LLMs

**Conference**: ACL 2026  
**arXiv**: [2510.23074](https://arxiv.org/abs/2510.23074)  
**Code**: https://github.com/Nikkei/fast-mia (Available)  
**Area**: LLM Safety / Privacy / Membership Inference Attack / Evaluation Tools  
**Keywords**: Membership Inference, vLLM, Cross-method Caching, WikiMIA, MIMIR

## TL;DR
Fast-MIA integrates 9 mainstream LLM Membership Inference Attack (MIA) methods into a single vLLM batch inference engine and adds a cross-method log-prob cache layer. This achieves an overall acceleration of approximately 5× on LLaMA-30B / WikiMIA (with SaMIA single-method acceleration of 19.5×) while maintaining near-identical AUC, making large-scale MIA auditing practical for the first time.

## Background & Motivation
**Background**: LLM memorization of training data poses risks in three categories: privacy leakage, copyright infringement, and benchmark contamination. MIA (Membership Inference Attack) is the standard tool for auditing these risks: given a model $f$ and a sample $x$, determine if $x$ was in the training set. Various methods exist, including LOSS, Min-K% Prob, DC-PDD, ReCaLL, Con-ReCaLL, PAC, and SaMIA.

**Limitations of Prior Work**: 1) New methods are becoming increasingly "heavy"—text perturbation methods (ReCaLL/Con-ReCaLL) require multiple prefixes per sample; black-box methods (SaMIA) require multiple generation samples; Puerto et al. demonstrated that MIA is truly effective only when aggregated at the dataset level, further raising the computational barrier. 2) Implementations for each paper are independent; methods like PPL/zlib, Min-K%, and DC-PDD share the same log-probs but recalculate them individually. 3) Existing toolkits (e.g., LLM-Sanitize) are either unmaintained or lack caching mechanisms.

**Key Challenge**: MIA evaluation is a triple product of "heavy computation × multiple methods × large-scale datasets." Current implementations link the latter two dimensions linearly, making them inefficient. There is a lack of a unified framework that shares intermediate results at the system level and leverages vLLM to maximize single-inference throughput.

**Goal**: Develop an open-source Python library where a single inference pass suffices for the common sweep of "same model × same data × multiple MIA methods."

**Key Insight**: The authors noted the mathematical "shared substrate" of MIA methods—most token-distribution / text-alternation methods are essentially different aggregations or comparisons of $\log p(c_t \mid c_{1..t-1})$. By caching token-level log-probs for each sample once, the "first inference pass" for almost all methods becomes free.

**Core Idea**: Combine vLLM high-throughput batch inference + cross-method log-prob caching + modular method registration, allowing 9 types of MIA to be executed via a unified YAML configuration.

## Method

### Overall Architecture
Fast-MIA is a YAML-config-driven evaluation library consisting of six components: (1) Data Loader supporting CSV/JSON/JSONL/Parquet/HF formats; (2) Model Loader for HF models or LoRA adapters via vLLM (supporting quantization); (3) Evaluator responsible for triggering on-demand inference, maintaining the cache, and calling methods; (4) MIA Method Registry for registering attacks via the `BaseMethod` base class; (5) YAML Config Interface for declarative specification of models, data, methods, and sampling; (6) Output of timestamped directories containing metrics, ROC, score distributions, and git/cache metadata.

### Key Designs

1.  **vLLM High-throughput Batch Inference Backend**:
    - **Function**: Computes all token-level log-probs and generated tokens in one pass, replacing the original per-sample HF Transformers implementations.
    - **Mechanism**: Utilizes vLLM's PagedAttention for KV cache page storage and dynamic batching, allowing prompts from different samples to share a compute pool; for methods like LOSS / Min-K / DC-PDD that only require prompt log-probs, it sets `max_tokens=1, prompt_logprobs=0`. For SaMIA, which requires multiple generations, it rewrites "per-sample loops + multiple generations" into vLLM's "batched multi-output generation" to avoid independent spawning for each sample.
    - **Design Motivation**: Transformers' `generate` lacks throughput benefits outside of batching; vLLM is an industrial-grade LLM serving core with low migration costs and direct gains (measured at ~5×, and up to 19.5× for SaMIA).

2.  **Cross-method log-prob Cache**:
    - **Function**: Caches the token-level log-probs generated during the first run of any method, allowing subsequent methods sharing those log-probs to reuse them with zero additional inference.
    - **Mechanism**: The Evaluator maintains a dependency table of "method → required inference type" and keys inference results by the triplet `(sample_id, prompt_variant, model_id)`. LOSS / PPL/zlib / all Min-K%-$K$ / DC-PDD share the same original log-prob; Lowercase triggers a new cache key for "lowercased prompt"; ReCaLL/Con-ReCall trigger "prefixed" versions. When a cache hit occurs, the number of inference calls $n_\text{infer}=0$.
    - **Design Motivation**: Authors observed that "hyperparameter sweeps" like Min-K% ($K \in \{0.1, 0.2, 0.3, 0.5, 0.8, 1.0\}$) wastefully perform 6 inferences when they only need different aggregations of the same log-prob. This optimization was missing in previous toolkits.

3.  **Modular Method Registration + Multilingual Support**:
    - **Function**: Each MIA method inherits from `BaseMethod`, implementing only `process_output` and `run`, then registers via `factory.py`.
    - **Mechanism**: The base class abstracts "trigger inference + use cache + calculate score," so method authors focus only on aggregating intermediate results. A `space_delimited_language` flag is exposed for languages without space-based segmentation (e.g., Chinese/Japanese).
    - **Design Motivation**: MIA is a rapidly evolving field; hardcoding methods leads to obsolescence. Modularity allows the community to add new methods within a week. The multilingual switch addresses different trends in Japanese MIA shown in prior work.

### Loss & Training
No training involved. All methods are inference-only. Evaluation metrics include AUC, FPR@95 (FPR at 95% TPR), and TPR@5 (TPR at 5% FPR), with the latter two following Carlini 2022's recommendation for low-FPR performance.

## Key Experimental Results

### Main Results
LLaMA-30B, WikiMIA, token length=32, NVIDIA A100 80GB, Left = Fast-MIA / Right = HF Transformers:

| Method | AUC (FM / Tr) | Time (FM / Tr) | Gain | FPR@95 |
| :--- | :--- | :--- | :--- | :--- |
| LOSS | 69.4 / 69.4 | 12s / 57s | ×4.75 | 84.3 / 84.3 |
| Min-K% Prob (K=0.2) | 69.3 / 69.3 | 12s / 57s | ×4.75 | 82.3 / 82.3 |
| DC-PDD | 67.4 / 67.4 | 12s / 57s | ×4.75 | 84.8 / 84.8 |
| Lowercase | 64.1 / 64.1 | 25s / 1m59s | ×4.76 | 83.5 / 83.8 |
| PAC | 73.3 / 73.4 | 1m17s / 6m24s | ×4.99 | 82.3 / 77.9 |
| ReCaLL | 90.7 / 90.3 | 55s / 2m10s | ×2.36 | 28.5 / 34.7 |
| Con-ReCaLL | 96.8 / 96.1 | 1m53s / 3m30s | ×1.86 | 10.8 / 12.9 |
| SaMIA | 65.5 / 64.5 | 2h3m / 40h10m | **×19.5** | 90.5 / 90.7 |

AUC is almost identical (zero difference for baseline/token-distribution classes; <1 point fluctuation for generation classes due to sampling randomness).

### Ablation Study
Comparing vLLM acceleration only vs. full features vs. HF baseline (excluding SaMIA due to slowness):

| Config | Total Time | Total Inferences | Description |
| :--- | :--- | :--- | :--- |
| Fast-MIA w/ cache | **3m54s** | **10** | Full solution |
| Fast-MIA w/o cache | 5m18s | 17 | vLLM acceleration only |
| Transformers (per-paper impl) | 17m51s | 17 | Original baseline |

Breaking it down: vLLM batching reduces 17m51s → 5m18s (≈3.4× system acceleration), and cross-method cache reduces 17 inferences to 10, 5m18s → 3m54s (≈1.4× algorithm acceleration). Combined, end-to-end gain is ≈4.6×.

### Key Findings
- **PPL/zlib / Min-K%-(0.1..1.0) / DC-PDD take 0 seconds once cached** — these methods share original log-probs with LOSS, effectively flattening both "hyperparameter sweep" and "method sweep" dimensions.
- **SaMIA shows the highest gain (×19.5)** because it replaces serial loops of "5 generations per sample" with vLLM's batched multi-output — generation-heavy methods benefit far more than prompt-only methods.
- **Negligible AUC loss** indicates that speedups stem from system/caching optimizations rather than numerical approximations, allowing safe large-scale re-evaluation.

## Highlights & Insights
- The "cross-method cache" idea essentially reduces MIA evaluation from $O(\text{methods} \times \text{samples})$ to $O(\text{unique prompt variants} \times \text{samples})$. Gain is further amplified during hyperparameter sweeps, a trick transferable to any evaluation library.
- Using vLLM as an evaluation backend rather than HF Transformers is an engineering reality many prior libraries ignored; Fast-MIA proves this is a "free lunch."
- Unified YAML files facilitate reproducibility by including experiment specs, timestamped outputs, and git/cache metadata, providing a reasonable baseline tool for a field often plagued by reproducibility crises.

## Limitations & Future Work
- Method coverage is limited to 9; dataset-level MIA (Maini/Puerto etc.) is not yet integrated.
- Model support depends on the vLLM backend, excluding encoder-only / encoder-decoder models; closed-source APIs are theoretically limited to black-box methods.
- Evaluation was conducted on a 1 model × 1 dataset × 1 length setup; scaling curves for model size / context length / hardware were not explicitly swept and may vary.
- Implementation remains "custom plug-and-play," where custom metrics/reports still require modifying the main loop; future plans include full YAML integration.

## Related Work & Insights
- **vs LLM-Sanitize (Ravaut 2025)**: Also a multi-method toolkit, but LLM-Sanitize is locked to vLLM 0.3.3 and unmaintained since 2024; Fast-MIA uses vLLM 0.15.1 and adds cross-method caching, offering superior maintainability.
- **vs MIMIR (Duan 2024) / Privacy Meter (Murakonda 2020)**: These are batch implementations within research projects but lack vLLM + cache integration; Fast-MIA integrates them as benchmarks.
- **vs Chen 2025 Survey Implementation**: The survey offers comprehensive method comparisons but no open-source code; Fast-MIA makes that comparison matrix practically executable.

## Rating
- Novelty: ⭐⭐⭐ Primarily engineering integration, no new attacks or metrics, but cross-method caching is a novel contribution in this context.
- Experimental Thoroughness: ⭐⭐⭐ Single model/dataset suffices for speedup claims but lacks scaling curves across backbones.
- Writing Quality: ⭐⭐⭐⭐ Table 1 clearly compares capabilities across toolkits; YAML examples are directly reproducible.
- Value: ⭐⭐⭐⭐⭐ A rare "install and get 5x faster" utility for the MIA / data contamination auditing research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[ICML 2026\] Efficient DP-SGD for LLMs with Randomized Clipping](../../ICML2026/llm_safety/efficient_dp-sgd_for_llms_with_randomized_clipping.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](../../NeurIPS2025/llm_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)

</div>

<!-- RELATED:END -->
