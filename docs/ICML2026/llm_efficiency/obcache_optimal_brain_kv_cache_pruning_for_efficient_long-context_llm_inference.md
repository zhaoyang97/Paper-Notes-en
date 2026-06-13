---
title: >-
  [Paper Note] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference
description: >-
  [ICML2026][LLM Efficiency][KV cache eviction] Ours reformulates KV cache eviction as a "layer-wise structured pruning" problem. By leveraging the second-order Taylor approximation from Optimal Brain Damage…
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "KV cache eviction"
  - "Optimal Brain Damage"
  - "Long-context inference"
  - "Second-order Taylor approximation"
  - "output-aware saliency"
date: 2026-05-08
content_hash: 7af3e3311b7a01c9
---

# OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

**Conference**: ICML2026  
**arXiv**: [2510.07651](https://arxiv.org/abs/2510.07651)  
**Code**: https://github.com/DreamSoul-AI/OBCache  
**Area**: LLM Efficiency / KV Cache Compression  
**Keywords**: KV cache eviction, Optimal Brain Damage, Long-context inference, Second-order Taylor approximation, output-aware saliency

## TL;DR
Ours reformulates KV cache eviction as a "layer-wise structured pruning" problem. By leveraging the second-order Taylor approximation from Optimal Brain Damage, the authors derive closed-form saliency scores for independent value units, independent key units, and joint key-value units. These scores serve as plug-and-play "saliency replacements" for existing attention-only eviction frameworks like H2O, TOVA, SnapKV, and AdaKV. Ours achieves consistent gains on RULER and LongBench for LLaMA-3.1 / Qwen-2.5 (e.g., AdaKV's accuracy increases by nearly 15% on query-agnostic RULER-4K with a 30% budget).

## Background & Motivation

**Background**: The primary bottleneck in long-context LLM (128K–1M tokens) inference is the KV cache, which scales linearly with sequence length and batch size—LLaMA-3.1-8B requires over 120GB of KV cache for a 1M context, far exceeding the memory of mainstream GPUs. A common mitigation is **training-free cache eviction**: methods like H2O, TOVA, and SnapKV discard unimportant tokens permanently based on saliency scores, observing that only a few tokens significantly affect the output.

**Limitations of Prior Work**: Existing eviction methods rely almost exclusively on accumulated attention weights as saliency (H2O accumulates historical weights, TOVA looks at the latest query, and SnapKV uses a local window with pooling), completely ignoring the contribution of the value state to the final output. Intuitively, this can be error-prone—a token with high attention weight but a near-zero value vector might have no effect on the output yet be retained, while a token with low attention weight but a unique value direction might be incorrectly evicted.

**Key Challenge**: The "gold standard" for saliency should be the **perturbation of the actual output $\mathbf{O}$ after removing a token**, whereas attention weight is merely a coarse approximation of this perturbation. While VATP and CriticalKV recognized that the value norm should be included, they either lack a formalized framework or require additional assumptions about attention distributions.

**Goal**: (1) Provide a unified theoretical framework for cache eviction that incorporates both attention-only and value-aware scores; (2) Derive **closed-form saliency scores that explicitly include value/key information** under this framework; (3) Ensure the new scores can be used as plug-and-play replacements for saliency terms in any existing method.

**Key Insight**: The authors note that LeCun’s 1989 Optimal Brain Damage (OBD) provides an elegant closed-form solution for the second-order Taylor approximation of task loss after pruning weights. If the cached KV is viewed as "dynamic weights," eviction becomes structured pruning, making the second-order expansion of OBD applicable.

**Core Idea**: Formulate cache eviction as a layer-wise pruning problem to minimize the Frobenius error between the pruned attention output $\widehat{\mathbf{O}}$ and the original $\mathbf{O}$. By applying a second-order Taylor expansion to this error, **closed-form scores are derived for three pruning units**: values, keys, and KV-pairs.

## Method

### Overall Architecture
OBCache does not change the eviction scheduling of H2O, TOVA, SnapKV, or AdaKV (i.e., "when to trigger pruning" and "how to allocate budget across heads"). It only replaces their scoring functions with its "saliency." The overall pipeline:

1.  **Modeling**: Define the perturbation at token position $p$ as $\widehat{\mathbf{V}} = \mathbf{V} + \delta\mathbf{V}$ and $\widehat{\mathbf{K}} = \mathbf{K} + \delta\mathbf{K}$. Evicting $p$ implies $\mathbf{e}_p^\top [\widehat{\mathbf{V}}\ \widehat{\mathbf{K}}] = \mathbf{0}$.
2.  **Objective**: Minimize the *pruning-induced eviction error* $\mathcal{L}(\widehat{\mathbf{V}}, \widehat{\mathbf{K}}) = \| \sigma(\mathbf{Q}\widehat{\mathbf{K}}^\top/\sqrt{d})\widehat{\mathbf{V}} - \sigma(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})\mathbf{V} \|_F^2$ as a proxy for the true, unobservable *eviction error* (which affects future $\mathbf{o}_{s+1},\dots$).
3.  **Second-order Expansion**: Expand at point $(\mathbf{V},\mathbf{K})$. The first-order term vanishes because $\widehat{\mathbf{O}}-\mathbf{O}=\mathbf{0}$, yielding $\mathcal{L} = \tfrac{1}{2}\delta\mathbf{V}^\top \mathbf{H}^{vv} \delta\mathbf{V} + \tfrac{1}{2}\delta\mathbf{K}^\top \mathbf{H}^{kk} \delta\mathbf{K} + \delta\mathbf{V}^\top \mathbf{H}^{vk} \delta\mathbf{K} + \mathcal{O}(\|\cdot\|^3)$.
4.  **OBD Diagonal Assumption**: Reuse the diagonalization from OBD by taking only the $(p,p)$ sub-block of $\mathbf{H}$ to obtain the closed-form saliency for token $p$.
5.  **Substitution**: Replace the attention-accumulation scores in H2O, TOVA, SnapKV, or AdaKV with the derived $\mathbf{S}_p$, keeping other processes (recent window, head budgets, etc.) intact.

The closed-form scores for the three core pruning units (Propositions 4.3–4.5) are as follows.

### Key Designs

1.  **Value-Pruning Score ($\mathbf{S}_p^{\text{value}}$)**:
    *   **Function**: Treats only the value as the pruning unit ($\mathbf{e}_p^\top \widehat{\mathbf{V}} = \mathbf{0}$), measuring the perturbation of removing $\mathbf{v}_p$ on the output.
    *   **Mechanism**: Derived from the first term of the second-order Taylor expansion, the closed-form result is $\mathbf{S}_p^{\text{value}} = \sum_i |\mathbf{A}_{i,p}|^2 \|\mathbf{v}_p\|^2$, which is the product of the $\ell_2$-norm squared of the attention matrix column and the value norm squared. The authors prove that the value-aware scores proposed by VATP/CriticalKV are special cases using the $\ell_1$-norm, showing that the inclusion of the value norm is naturally derived from the OBD framework.
    *   **Design Motivation**: Introduce value-state information while maintaining nearly the same efficiency as attention-only scores, and provide a theoretical explanation for existing value-aware heuristics.

2.  **Key-Pruning Score ($\mathbf{S}_p^{\text{key}}$)**:
    *   **Function**: Treats only the key as the pruning unit ($\mathbf{e}_p^\top \widehat{\mathbf{K}} = \mathbf{0}$), measuring the perturbation of removing $\mathbf{k}_p$. This perturbation is typically much larger than the value perturbation because changing a key affects the entire attention distribution.
    *   **Mechanism**: The closed-form score is $\mathbf{S}_p^{\text{key}} = \sum_i |\mathbf{A}_{i,p} \mathbf{Z}_{i,p}|^2 \|\mathbf{v}_p - \mathbf{o}_i\|^2$, where $\mathbf{Z}$ represents the pre-softmax logits and $\mathbf{o}_i$ is the attention output at the $i$-th query position. Intuitively, high-scoring tokens are those where the value differs greatly from the current output direction and both attention and logits are non-negligible. Evicting such tokens causes a significant shift in $\mathbf{O}$ due to softmax renormalization.
    *   **Design Motivation**: Existing scores fail to explicitly account for the cascading effects of softmax row renormalization after key eviction; this is the primary source of Ours' gains over prior methods.

3.  **Joint Key-Value Score ($\mathbf{S}_p^{\text{joint}}$)**:
    *   **Function**: Treats $(\mathbf{k}_p,\mathbf{v}_p)$ as a joint pruning unit, providing the most comprehensive estimate of the true eviction error.
    *   **Mechanism**: The score is $\mathbf{S}_p^{\text{joint}} = \mathbf{S}_p^{\text{value}} + \mathbf{S}_p^{\text{key}} + 2 \sum_i |\mathbf{A}_{i,p}|^2 \mathbf{Z}_{i,p} (\|\mathbf{v}_p\|^2 - \mathbf{v}_p^\top \mathbf{o}_i)$, where the extra cross-Hessian term $\mathbf{H}^{vk}$ captures the interaction effects between the key and value.
    *   **Design Motivation**: Achieve "theoretical completeness" within the framework and allow users to choose between OBCache-V, -K, or -V&K based on their computational budget.

**Unification across existing methods**: By relaxing the objective from "output error" to "attention matrix row error $\|\widehat{\mathbf{A}}_{w:s} - \mathbf{A}_{w:s}\|_{1,1}$" and simplifying the pruning unit to a single column of the attention matrix, the framework degenerates to $\mathbf{S}_p^{\text{attn}} = \sum_{i=w}^s |\mathbf{A}_{i,p}|$. This matches the accumulation scores used by H2O ($w=1$), TOVA ($w=s$), and SnapKV ($w \gg 1$), characterizing them as special cases of Ours under different "perturbation windows" $w$.

### Loss & Training
OBCache is a **training-free inference-time method** requiring no training or fine-tuning. All Hessian sub-blocks can be computed during a standard forward pass (requiring attention weights $\mathbf{A}$, pre-softmax logits $\mathbf{Z}$, values $\mathbf{V}$, and outputs $\mathbf{O}$). For FlashAttention-2 implementations, this adds almost zero memory overhead during prefill. Eviction is performed greedily once during the prefill phase to reach the target budget; during the decoding phase, $\mathbf{S}_p$ is updated in real-time to support dynamic eviction. Independent derivations are provided for GQA models (Appendix A.5).

## Key Experimental Results

### Main Results
Setup: LLaMA-3.1-8B-Instruct / Qwen-2.5-7B-Instruct using the KVPress framework. Pre-fill phase evaluated on RULER (4K/32K) and LongBench; decoding phase evaluated for perplexity on PG19 (~70K tokens). Cache budget set at 10–40% of prompt length. Comparisons conducted in both query-aware and query-agnostic settings.

| Baseline (LLaMA-3.1-8B, RULER-4K) | Setting | Avg Acc | + OBCache-K | + OBCache-V&K | Gain |
|------|------|------|------|------|------|
| H2O | Q-Aware | 57.5 | 67.6 | 67.8 | **+10.3** |
| H2O | Q-Agnostic | 31.7 | 38.9 | 40.0 | +8.3 |
| TOVA | Q-Aware | 74.5 | 76.5 | 76.7 | +2.2 |
| SnapKV | Q-Aware | 72.4 | 73.9 | 73.6 | +1.5 |
| SnapKV | Q-Agnostic | 37.9 | 42.1 | 41.9 | +4.2 |
| AdaKV | Q-Aware | 75.7 | 81.6 | 81.9 | +6.2 |
| AdaKV | Q-Agnostic | 43.0 | 55.0 | **55.2** | **+12.2** |

> Gains are also observed on RULER-32K, where AdaKV + OBCache-V&K improves query-agnostic accuracy from 45.5 to 55.1 (+9.6). On LongBench, AdaKV with a 10% budget and OBCache-K shows gains of +1.2 (query-aware) and +2.6 (query-agnostic).

### Ablation Study

| Configuration (RULER-4K, AdaKV baseline, Q-Agnostic) | Avg Acc | Description |
|------|------|------|
| AdaKV (attention-only) | 43.0 | Strongest attention-only baseline |
| + OBCache-V | 51.4 | Value info only (similar to VATP/CriticalKV, but $\ell_2$) |
| + OBCache-K | 55.0 | Includes key sensitivity; largest contributor |
| + OBCache-V&K | 55.2 | Adds cross-term; marginal improvement over V |
| VATP (OBCache-V-L1) | < OBCache-V | $\ell_1$-norm value-only; outperformed by OBCache-V |
| CriticalKV | < OBCache-K | Only approaches Ours at 40% budget |

### Key Findings
- **Key-pruning scores yield much higher gains than value-pruning scores**: Evicting keys forces a renormalization of the entire softmax row, creating a naturally larger perturbation. OBCache-K outperforms OBCache-V across all baselines and budgets.
- **OBCache is highly complementary to strong baselines**: Improvements of ~10% on H2O and ~12% on query-agnostic AdaKV suggest that attention-only signals leave an "information gap" in all budget allocation strategies.
- **Second-order Taylor approximation is nearly lossless**: Needle-in-A-Haystack experiments (Section 4.4) show that the recall of the top-$k$ oracle using OBCache closed-form scores is nearly identical to the "exact recalculation of Eq.(1)" but at a lower cost.
- **Structural bias**: Larger perturbation windows can cause early tokens to accumulate excessively high scores because they are attended to by more queries. Using H2O’s "fixed recent window" strategy (e.g., 20 tokens) mitigates this.
- **Decoding scenario**: On PG19 with a fixed 1024-token budget and 4 sinks, OBCache-K maintains lower perplexity than H2O across the 1–32K range. OBCache-V&K did not exceed OBCache-K, suggesting that simple addition might not be the optimal way to fuse terms.

## Highlights & Insights
- **Elegant application of 1989 OBD to 2026 KV cache**: Cached KV effectively acts as "on-demand dynamic weights." The machinery of second-order analysis for structured pruning is immediately applicable without reinvention.
- **Valuable unification of prior work**: Proving that H2O/TOVA/SnapKV are special cases of the same objective under different windows, and that VATP/CriticalKV are $\ell_1$ variants of value-only pruning, is highly convincing.
- **OBCache-V is nearly free**: Adding the $\|\mathbf{v}_p\|^2$ factor to attention-only scores costs almost nothing but provides gains on most baselines, offering high deployment value.
- **Strong transferability**: The paradigm can extend to channel-wise KV pruning, cache merging (moving from "evict to 0" to "merge into another token"), and KV factorization in weight-sharing scenarios.

## Limitations & Future Work
- **Second-order expansion holds only for "small perturbations"**: At extremely low budgets (e.g., 5%), where many tokens are pruned simultaneously, both the diagonal and small-perturbation assumptions weaken. Results below 10% budget were not reported.
- **Weak cross-term gains**: OBCache-V&K barely outperforms OBCache-K, suggesting the additive form does not fully exploit key/value interactions—a space the authors leave for future refinement.
- **Lack of head-wise/channel-wise pruning research**: While the framework is applicable to channel dimensions (as OBD originally was), this study only covers token-wise eviction.
- **Dependence on accessing logits $\mathbf{Z}$ and outputs $\mathbf{O}$**: Custom inference backends that fuse softmax into the kernel may require extra effort to expose intermediate variables.
- **Lack of cross-model-family generalization**: Validated only on LLaMA-3.1 and Qwen-2.5. Stability on MoE models (like DeepSeek-V3) or models with distilled KV caches remains to be verified.

## Related Work & Insights
- **vs H2O (NeurIPS 2023)**: H2O accumulates all historical attention. Ours views this as a special case ($w=1$ window, minimizing attention row error) and adds value/key info, gaining a stable 10% under the same budget scheduler.
- **vs SnapKV (NeurIPS 2024)**: SnapKV uses a local window and 1D pooling to smooth attention scores. Ours provides more accurate saliency; the marginal benefit of pooling is reduced when using Ours.
- **vs VATP / CriticalKV (2024–2025)**: These heuristically multiply attention scores by $\|\mathbf{v}_p\|$. Ours re-derives similar formulas under OBD (proving $\ell_2$ is superior for deriving key-pruning terms) and introduces key/joint scores, outperforming them.
- **vs Wanda / SparseGPT (2023)**: These represent a resurgence of the OBD paradigm for static weight pruning. Ours generalizes this to "dynamic KV weights" by shifting the Hessian derivation from the weight dimension to the token dimension.
- **Transferable Insights**: (1) Any attention/router with softmax can use this framework (e.g., MoE expert eviction, chunk eviction in RAG, vision token pruning). (2) Proving that a heuristic score is a second-order solution to a well-defined objective is a powerful narrative for "unification" papers.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying OBD to dynamic KV caches is a clean conceptual reuse; the unification of existing methods is particularly strong. The value-aware part primarily provides theoretical grounding for VATP/CriticalKV.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across 4 baselines, 2 models, and 2 long-context benchmarks at multiple budgets, including prefill/decoding. Missing very low budget and MoE validation.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from OBD theory to closed-form scores and the unification of special cases. GQA/multi-head notation can be dense.
- Value: ⭐⭐⭐⭐ Plug-and-play saliency replacement with immediate utility for long-context LLM deployment. OBCache-V is particularly cost-effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[ICML 2026\] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)

</div>

<!-- RELATED:END -->
