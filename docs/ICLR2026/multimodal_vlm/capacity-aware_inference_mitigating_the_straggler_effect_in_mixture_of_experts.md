---
title: >-
  [Paper Note] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts
description: >-
  [ICLR2026][Multimodal VLM][Mixture of Experts] To address the Straggler Effect in MoE inference—where the most heavily loaded expert determines overall latency due to uneven token distribution—this paper proposes Capacit…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "Mixture of Experts"
  - "reasoning efficiency"
  - "straggler effect"
  - "token drop"
  - "expert parallelism"
date: 2026-05-08
content_hash: 1dd9a94f457d37e5
---

# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts

**Conference**: ICLR2026
**arXiv**: [2503.05066](https://arxiv.org/abs/2503.05066)  
**Code**: [https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE](https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE)  
**Area**: Multimodal VLM
**Keywords**: Mixture of Experts, reasoning efficiency, straggler effect, token drop, expert parallelism

## TL;DR
To address the Straggler Effect in MoE inference—where the most heavily loaded expert determines overall latency due to uneven token distribution—this paper proposes Capacity-Aware Token Drop (discarding low-scoring tokens from overloaded experts) and Expanded Drop (re-routing overflow tokens to lightly loaded local experts). The approach achieves a 1.85× speedup on Mixtral-8×7B with a 0.2% performance improvement.

## Background & Motivation

**Background**: MoE is a key architecture for scaling LLMs, balancing performance and efficiency through sparse activation of multiple experts. Under expert parallelism, experts are distributed across multiple GPUs for parallel computation.

**Limitations of Prior Work**: The auxiliary balance loss used during training does not guarantee load balance at inference time. Empirical measurements show that the most heavily loaded expert can handle more than 7× the average number of tokens at inference time—lightly loaded experts finish first and must then wait for heavily loaded ones to synchronize, causing severe latency.

**Key Challenge**: This is the **Straggler Effect**—the latency of an MoE layer is determined by the most heavily loaded expert ($L \propto \max(\{N_i\})$), not the average load. Existing solutions (e.g., expert replication in DeepSeek-V3) require additional GPU resources.

**Goal**: To mitigate the Straggler Effect at inference time through intelligent token scheduling, without increasing GPU resource consumption.

**Key Insight**: Two complementary strategies—imposing a capacity upper bound on overloaded experts to drop low-importance tokens, and expanding the candidate set for lightly loaded experts to absorb overflow tokens.

**Core Idea**: Use gating scores as an importance metric to limit the number of tokens processed by heavily loaded experts, while re-routing overflow tokens to lightly loaded experts on the same GPU, achieving load balancing and inference speedup simultaneously.

## Method

### Overall Architecture
During MoE inference, the router assigns each token to its top-k experts. Capacity-Aware Inference inserts a two-step procedure after the router: (1) **Token Drop**—checks whether each expert's load exceeds the capacity limit $C = \gamma \bar{N}$, and drops excess tokens with low gating scores from overloaded experts; (2) **Expanded Drop**—re-routes dropped tokens by expanding the candidate set to include other lightly loaded experts on the same GPU, utilizing idle capacity. The entire process is completed before the All-to-All communication, incurring zero additional communication overhead.

### Key Designs

1. **Capacity-Aware Token Drop (handling overloaded experts)**:

    - **Function**: For experts exceeding the capacity limit, drop excess low-importance tokens.
    - **Mechanism**: Define a capacity factor $\gamma$ such that each expert processes at most $C = \gamma \bar{N}$ tokens, where $\bar{N} = tk/n$ is the expected average. When expert $j$'s load $N_j > C$, a scoring function $\mathcal{S}$ ranks all tokens assigned to that expert, retaining the top-$C$ and dropping the rest. Four scoring functions are compared—Order, Reverse Order, Random, and Score (gating score)—with Score substantially outperforming the others.
    - **Design Motivation**: Although dropping tokens appears to harm performance, the vast majority of tokens in overloaded experts are redundant—dropping 12% of overflow tokens yields 85% of the speedup on Mixtral. The gating score naturally reflects token–expert affinity, making it the most principled criterion for selecting which tokens to retain.

2. **Capacity-Aware Expanded Drop (utilizing lightly loaded experts)**:

    - **Function**: Re-route tokens dropped by Token Drop to lightly loaded experts on the same GPU.
    - **Mechanism**: For each token, in addition to its original top-k experts, $m$ local experts on the same GPU are added to the candidate pool (totaling $k+m$ candidates). Tokens rejected by their original experts after Token Drop can be accepted by lightly loaded local experts. Since all candidates reside on the same GPU, no additional cross-device communication is required.
    - **Design Motivation**: After Token Drop, lightly loaded experts still wait for synchronization, leaving their idle compute unused. Expanded Drop exploits this spare capacity to process dropped tokens. The gating score decays gradually beyond the top-k experts (Figure 8), indicating that re-routing to slightly lower-ranked experts does not significantly degrade output quality.

3. **Device-Level Capacity (advanced variant)**:

    - **Function**: Impose capacity constraints at the device level rather than the individual expert level.
    - **Mechanism**: When multiple experts are deployed on a single GPU, the constraint becomes $N_1 + N_2 + \cdots + N_{n_l} \leq n_l \cdot \gamma \bar{N}$, allowing load transfer among experts on the same GPU.
    - **Design Motivation**: Expert-level constraints may be overly strict—when one expert exceeds its limit but other experts on the same GPU have ample spare capacity, tokens are dropped unnecessarily.

### Loss & Training
This method is a purely inference-time technique that **requires no retraining**. It is applied directly to pre-trained MoE models with zero training cost.

## Key Experimental Results

### Main Results (Expanded Drop vs. Token Drop vs. Expert Drop vs. Baseline)

| Model | Method | Avg. Performance | vs. Baseline |
|-------|--------|-----------------|--------------|
| **Mixtral-8×7B-Instruct** | Baseline | 74.3 | - |
| | Token Drop ($\gamma$=1.5) | 73.8 | -0.5% |
| | Expanded Drop ($\gamma$=1.5) | **74.5** | **+0.2%** |
| | Expert Drop | 72.2 | -2.1% |
| **OLMoE-Instruct** | Baseline | 63.5 | - |
| | Token Drop ($\gamma$=2.0) | 62.3 | -1.2% |
| | Expanded Drop ($\gamma$=2.0) | **63.2** | -0.3% |
| | Expert Drop | 60.5 | -3.0% |
| **DeepSeek-V2-Lite-Chat** | Baseline | 69.3 | - |
| | Token Drop ($\gamma$=2.0) | 68.2 | -1.1% |
| | Expanded Drop ($\gamma$=2.0) | **68.9** | -0.4% |

### Ablation Study (Scoring Function Comparison for Token Drop, OLMoE, $\gamma$=1.0)

| Scoring Function | OBQA | PIQA | MMLU | Avg. |
|-----------------|------|------|------|------|
| Order | 36.0 | 60.2 | 36.9 | 51.8 |
| Reverse Order | 36.2 | 59.5 | 38.7 | 52.0 |
| Random | 34.0 | 63.1 | 35.7 | 53.1 |
| **Score** | **41.6** | **76.0** | **47.8** | **61.1** |

### Key Findings
- **Score substantially outperforms other strategies**: At $\gamma$=1.0, average performance is 61.1 vs. 53.1 for Random (+8%), confirming that gating score is an effective proxy for token importance.
- **Lightly loaded experts are critical**: Expert Drop (skipping the 10% lightest experts) removes only 2% of tokens yet causes a 3% performance drop, whereas Token Drop removes 12% of tokens with only a 0.9% drop—demonstrating that each expert, even when lightly loaded, encodes unique knowledge.
- **Expanded Drop can surpass the baseline**: On Mixtral, Expanded Drop exceeds the unconstrained baseline by 0.2%, suggesting that re-routing tokens to a broader set of experts actually improves representation quality.
- **Speedup depends on the GPU-to-expert ratio**: Speedup is greatest when 1–2 experts are deployed per GPU (1.85× on Mixtral) and diminishes with 8 experts per GPU, as aggregated load dilutes single-expert bottlenecks.
- **Image tokens in multimodal models can be aggressively compressed**: Visual MoE models tolerate $\gamma$=0.5 with negligible performance loss, indicating substantial redundancy in image tokens across experts.

## Highlights & Insights
- **Training-free load balancing at inference time**: Without retraining, load balance is achieved at inference time purely through capacity constraints and re-routing—directly applicable to deployed MoE models such as Mixtral and DeepSeek.
- **Locality-preserving design of Expanded Drop**: Expanding the candidate pool only within the same GPU entirely avoids cross-device communication overhead. This is a simple yet crucial engineering insight: leveraging idle synchronization time for productive computation.
- **Observation of a flat gating score tail**: Figure 8 shows that expert gating scores decay gradually beyond the top-k selection, providing theoretical support for re-routing—tokens assigned to slightly lower-ranked experts still exhibit reasonable affinity.
- **Large speedup from dropping a small fraction of tokens**: On Mixtral, dropping 12% of overflow tokens yields 85% of the total speedup, reflecting the long-tail distribution of the Straggler Effect, where minimal intervention produces disproportionate gains.

## Limitations & Future Work
- **Impact of token dropping on generation quality is not assessed**: Evaluation is conducted solely on classification and multiple-choice benchmarks; whether token dropping degrades coherence in open-ended text generation remains untested.
- **Static capacity factor**: $\gamma$ is a single global constant. Different layers and different inputs may warrant different capacity policies—adaptive $\gamma$ scheduling could yield further improvements.
- **Only inference scenarios are examined**: The interaction between inference-time Token Drop and the auxiliary loss used during training is not investigated.
- **KV cache implications are not discussed**: When a token is dropped at one layer, how the absence of its information propagates through residual connections in subsequent layers warrants further analysis.

## Related Work & Insights
- **vs. Expert replication in DeepSeek-V3**: DeepSeek-V3 mitigates load imbalance by replicating heavily loaded experts across multiple devices, requiring additional GPU resources. The proposed method incurs zero additional hardware overhead and is thus more practical.
- **vs. Switch-Transformer Token Drop**: Switch-Transformer applies Token Drop with an Order strategy during training. This work demonstrates that the Score strategy substantially outperforms Order (+9%) and is the first to systematically apply Token Drop at inference time.
- **vs. Expert pruning**: Although skipping lightly loaded experts also reduces computation, it incurs severe performance degradation. The comparative results clearly show that lightly loaded experts cannot be safely removed.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The explicit formulation and systematic analysis of the Straggler Effect is a meaningful contribution; the Expanded Drop mechanism for exploiting idle capacity is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four MoE models, multimodal experiments, scoring function ablations, efficiency analysis, and device-level variants.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear, mathematical derivations are complete, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Direct practical utility for inference optimization of deployed MoE models; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/multimodal_vlm/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](../../CVPR2026/multimodal_vlm/moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](../../CVPR2026/multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)

</div>

<!-- RELATED:END -->
