---
title: >-
  [Paper Note] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective
description: >-
  [ICML 2026][LLM Efficiency][KV cache eviction] The authors formalize the empirical question of "which KV cache entries are critical" as an optimization problem of minimizing attention output perturbation. They derive an…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "KV cache eviction"
  - "output perturbation upper bound"
  - "long-sequence inference"
  - "attention weights"
  - "projected value norm"
date: 2026-05-08
content_hash: c14b3b9c2a0d8b4b
---

# CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective

**Conference**: ICML 2026  
**arXiv**: [2502.03805](https://arxiv.org/abs/2502.03805)  
**Code**: https://github.com/FFY0/DefensiveKV (Available)  
**Area**: LLM Efficiency / KV Cache Compression / Model Inference Optimization  
**Keywords**: KV cache eviction, output perturbation upper bound, long-sequence inference, attention weights, projected value norm  

## TL;DR
The authors formalize the empirical question of "which KV cache entries are critical" as an optimization problem of minimizing attention output perturbation. They derive an analytical upper bound for perturbation (incorporating both attention weights and value norms projected via $W^O$) and design a plug-and-play two-stage greedy selection algorithm. This approach reduces the average compression loss of three SOTA methods—SnapKV, AdaKV, and HeadKV—by more than half across 29 long-context datasets.

## Background & Motivation

**Background**: As context length grows, the KV cache of Transformer self-attention expands linearly, becoming a memory and I/O bottleneck for long-sequence inference. The mainstream mitigation strategy is KV cache eviction: selecting and retaining $b$ "critical" KV entries under a fixed budget $b$ while discarding the rest. H2O and Scissorhands observed power-law distributions in attention weights; SnapKV introduced "observation windows + max pooling" to accumulate weights stably; and AdaKV/HeadKV dynamically allocate budgets across different heads.

**Limitations of Prior Work**: All these methods essentially assume that "entries with high attention weights are critical." However, the definition of "critical" has never been formally established, relying instead on empirical observations like the power-law distribution. This leaves two questions unanswered: What is the selection criterion? Are attention weights sufficient?

**Key Challenge**: Starting from the most fundamental goal—minimizing the perturbation of the attention output after eviction—the authors discovered that this perturbation is not determined solely by attention weights. From the structure of the output $o = AVW^O$, the impact of discarded entries on the final output depends simultaneously on the attention weight $A_i$ and the norm of the value projected by $W^O$, i.e., $\lVert (VW^O)_i \rVert$. Relying only on weights ignores half of the signal.

**Goal**: Formulate "critical entry identification" as an optimization problem minimizing output perturbation, derive a computable upper bound for this problem, and provide a selection algorithm that can be integrated into existing eviction pipelines without additional computational overhead.

**Key Insight**: In the field of pruning, Wanda successfully guided weight pruning using a similar logic of "discarding what causes the least impact on output." This paper is the first to transfer the "perturbation analysis → selection metric" paradigm to the KV cache.

**Core Idea**: Select critical entries by minimizing the worst-case upper bound of output perturbation, using the product of "attention weight × projected value norm" as a new importance metric to replace pure attention weight scoring.

## Method

### Overall Architecture
The output of single-head self-attention is denoted as $o = AVW^O$, where $A = \mathrm{softmax}(qK^\top/\sqrt{d})$. Given a cache budget $b$, the goal is to select $b$ KV entries from $n$ total entries to form $\langle \hat K, \hat V \rangle$, such that the $L_1$ distance between the approximated output $\hat o$ and the original output $o$, $\mathcal{L} = \lVert o - \hat o \rVert_1$, is minimized. The authors encode "which entries to discard" into a multiplicative mask $\mathcal{N} \in \{0,1\}^n$, derive an analytical perturbation upper bound $\theta$ for $\mathcal{N}$, and use a two-stage greedy algorithm to minimize $\theta$ within each head. This selection process directly replaces the "Top-K by weight" step in existing SnapKV/AdaKV/HeadKV pipelines.

### Key Designs

1.  **Analytical Upper Bound $\theta$ of Output Perturbation**:
    - **Function**: Translates "how far $\hat o$ is from $o$" into a scalar that depends only on $\mathcal{N}$, attention weights $A$, and projected value norms $\lVert \bm{\mathcal{V}}_{i,:} \rVert_1$, bypassing the difficulty of directly optimizing the norm of matrix differences.
    - **Mechanism**: First proves that the remaining softmax entries undergo renormalization as $A' = (\mathcal{N} \odot A) / \sum_i \mathcal{N}_i A_i$. Then, using the triangle inequality, it derives $\mathcal{L} \leq \theta = C - (2 - 1/\sum_i \mathcal{N}_i A_i)\sum_i \mathcal{N}_i A_i \lVert \bm{\mathcal{V}}_{i,:} \rVert_1$, where $\bm{\mathcal{V}} = V W^O$ and $C$ is a constant independent of $\mathcal{N}$.
    - **Design Motivation**: This bound simultaneously includes "attention weight + value norm projected by $W^O$" in the metric for the first time, theoretically demonstrating that $A_i$ alone is insufficient; $\lVert (VW^O)_i \rVert_1$ must be included to reflect the true output impact.

2.  **Two-Stage Greedy Selection Algorithm**:
    - **Function**: Approximates the minimization of $\theta$ without exponential search while ensuring the second stage of optimization is valid.
    - **Mechanism**: Splits the budget into $b' = \alpha b$ and $b'' = (1-\alpha) b$. Stage 1 selects Top-$b'$ entries based on pure attention weights $A$ (typically $\alpha=0.5$) to ensure the cumulative weight of selected entries $\sigma > 0.5$. Stage 2 selects the remaining Top-$b''$ entries using a composite score $\mathcal{A}_i = (A_i + \epsilon) \cdot \lVert \bm{\mathcal{V}}_{i,:} \rVert_1$. The two stages combined directly minimize the second-stage upper bound $\hat\theta$.
    - **Design Motivation**: Stage 1 ensures that the cumulative weight exceeds half, so $1/\sigma < 2$, keeping the coefficients in the upper bound non-negative and the greedy direction valid. Stage 2 then allows "weight × projected norm" to serve as a unified score. The fixed value $\alpha=0.5$ satisfies the assumption in over 99% of heads and avoids the deployment complexity of hyperparameter searching.

3.  **Integration as a General Plugin for Existing Eviction Pipelines**:
    - **Function**: Replaces the original "Top-K based on cumulative attention weight" in SnapKV, AdaKV, and HeadKV with the two-stage selection described above, while keeping other components (observation window, max pooling, cross-head budget allocation) unchanged.
    - **Mechanism**: The authors abstract SnapKV, AdaKV, and HeadKV into a unified template of "budget allocation + observation window weight accumulation + selection" (Algorithm 2 in the paper). The new method only modifies the "selection" step.
    - **Design Motivation**: This ensures: (1) strict orthogonality to works like AdaKV and HeadKV which focus on "inter-head budget allocation," allowing for additive gains; (2) negligible overhead, as it only requires calculating the already existing $\lVert VW^O \rVert$; (3) drop-in replacement during inference without retraining or offline profiling.

### Loss & Training
The method occurs entirely during inference and requires no training or fine-tuning of the model. The only additional runtime computation is the $L_1$ norm of each row in $\bm{\mathcal{V}} = V W^O$. The hyperparameter $\alpha$ is fixed at 0.5.

## Key Experimental Results

### Main Results
On 29 datasets from Ruler (13 tasks) and LongBench (16 tasks), integrated testing was performed with SnapKV, AdaKV, and HeadKV using Llama-3.1-8B, Mistral-7B, and Qwen2.5-32B at a fixed 40% cache budget. The following table shows representative Ruler average scores (Full Cache baseline is 100%, arrows indicate drop relative to Full Cache):

| Model | Method | Ruler Avg ↑ | Drop relative to Full Cache ↓ |
| :--- | :--- | :--- | :--- |
| Llama-3.1-8B | Full Cache | 91.05 | 0% |
| Llama-3.1-8B | SnapKV | 67.93 | 25.4% |
| Llama-3.1-8B | SnapKV + Ours | **76.89** | **15.6%** |
| Llama-3.1-8B | AdaKV | 78.38 | 13.9% |
| Llama-3.1-8B | AdaKV + Ours | **86.28** | **5.2%** |
| Llama-3.1-8B | HeadKV | 79.98 | 12.2% |
| Llama-3.1-8B | HeadKV + Ours | **89.29** | **1.9%** |
| Mistral-7B | AdaKV | 34.88 | 55.4% |
| Mistral-7B | AdaKV + Ours | **69.17** | **11.6%** |

Three Findings: (1) After integration, the performance drop is generally reduced by more than half; HeadKV + Ours reduces Llama's loss to 1.9%, nearly matching Full Cache. (2) Models like Mistral, which were previously crushed by SnapKV/AdaKV (55%+ drop), benefit most significantly, with AdaKV integration pulling the score from 34.88 back to 69.17. (3) Gains become more pronounced as the base method strengthens, indicating the perturbation perspective is truly orthogonal to the budget allocation perspective.

### Ablation Study

| Configuration | Key Observation | Description |
| :--- | :--- | :--- |
| Attention Weight Only (Orig. SnapKV) | Baseline drop | Verifies that $A_i$ alone is insufficient in Stage 2. |
| Stage 1 $\alpha=0.5$ + Stage 2 Weight × Norm | Consistently optimal | Complete two-stage method. |
| Varying $\alpha \in \{0.25, 0.5, 0.75\}$ | Stable performance | Insensitive to $\alpha$ (Appendix C.1). |
| Head-level Perturbation Stats | Over 92% of Llama heads show reduced perturbation | Consistent with theory. |
| Layer-level Perturbation Accumulation | Hidden state perturbation in final layer significantly reduced | Advantages compound across layers. |
| Different Cache Budgets | Consistent wins across all budgets | Benefits deployments under varying VRAM constraints. |
| $L_2$ Distance instead of $L_1$ | Similar gains | Framework is robust to distance metrics (Appendix C.3). |

### Key Findings
- The composite score $A_i \cdot \lVert (VW^O)_i \rVert_1$ in Stage 2 is the core driver of performance gains, confirming that the "projected value norm" is a critical signal missed by existing methods.
- The assumption $\sigma > 0.5$ is a very loose condition, satisfied by over 99% of heads in practical LLMs.
- Improvements are quantifiable at both head and layer levels: 92% of heads show reduced perturbation, and final hidden state perturbation continues to decrease, indicating the method truly optimizes the theoretical objective rather than overfitting the dataset.

## Highlights & Insights
- **KV Eviction from First Principles**: Elevates "entry criticality" from empirical power-law observations to an optimization objective of "minimizing output perturbation," providing a new theoretical starting point for the cache eviction roadmap.
- **The Overlooked $W^O$ Projection**: While prior methods used $K$ and $V$ for scoring, this paper points out that the norm of the value after the output projection $W^O$ is what truly determines output impact. This insight can transfer to assigning bit widths for different tokens during quantization or pre-screening entries during the prefill phase.
- **Orthogonal Plugin Stance**: By abstracting a three-stage template of "budget allocation + weight accumulation + selection," the new method only modifies the "selection" phase. This allows it to naturally stack with allocation-focused methods like AdaKV/HeadKV, proving that compression loss stems primarily from selection strategy rather than allocation strategy.

## Limitations & Future Work
- The upper bound $\theta$ is derived and optimized per head; inter-head coupling (e.g., perturbation in one head being compensated by another) is not part of the objective, leaving theoretical room for further tightening.
- A fixed $\alpha = 0.5$ is used for Stage 1. The authors acknowledge that adapting $\alpha$ by model, budget, or head might yield better results, but the search cost is high.
- Experiments are concentrated on English long-context models ≥ 7B; whether the perturbation assumptions hold for smaller models, multilingual contexts, or vision-language long contexts remains to be verified.
- The method assumes the decoder sees the full KV at once. Solutions for "re-selection when new entries are added dynamically" in purely streaming or chunked prefill scenarios have not yet been provided.

## Related Work & Insights
- **vs SnapKV/AdaKV/HeadKV**: These only optimize "allocation + weight accumulation," while selection remains pure Top-K(A). This paper uses the same observation window weights to produce superior selection, yielding 5–40 point improvements when integrated.
- **vs H2O / Scissorhands**: H2O uses cumulative attention weights to select heavy-hitters empirically. This work is the first to provide a formal definition (output perturbation minimization) and an optimizable upper bound for "critical entries."
- **vs Wanda (Weight Pruning)**: Shared philosophy—"discard what causes the least impact on output." This paper transfers this perturbation metric from static weight pruning to dynamic KV cache selection and introduces the projected value norm as a new metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ICML 2026\] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)
- [\[ICLR 2026\] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](../../ICLR2026/llm_efficiency/randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)
- [\[ICML 2026\] Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Schemes](do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)

</div>

<!-- RELATED:END -->
