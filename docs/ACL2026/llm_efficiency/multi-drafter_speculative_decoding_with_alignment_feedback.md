---
title: >-
  [Paper Note] Multi-Drafter Speculative Decoding with Alignment Feedback
description: >-
  [ACL 2026][LLM Efficiency][Speculative Decoding] This paper proposes MetaSD, a unified framework that integrates multiple heterogeneous drafters into speculative decoding. It models drafter selection as a Multi-Armed Ban…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Speculative Decoding"
  - "Multi-Armed Bandit"
  - "Multi-Drafter"
  - "Alignment Feedback"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: aeebdfda4b432873
---

# Multi-Drafter Speculative Decoding with Alignment Feedback

**Conference**: ACL 2026  
**arXiv**: [2604.05417](https://arxiv.org/abs/2604.05417)  
**Code**: Yes  
**Area**: LLM Efficiency  
**Keywords**: Speculative Decoding, Multi-Armed Bandit, Multi-Drafter, Alignment Feedback, Inference Acceleration

## TL;DR

This paper proposes MetaSD, a unified framework that integrates multiple heterogeneous drafters into speculative decoding. It models drafter selection as a Multi-Armed Bandit (MAB) problem and dynamically selects the drafter best aligned with the target LLM using a Block Divergence reward signal. MetaSD consistently outperforms single-drafter methods in both black-box and white-box configurations.

## Background & Motivation

**Background**: Speculative decoding accelerates LLM inference by using a small model (drafter) to predict future tokens, which are then verified in parallel by a large model. Existing methods have improved acceptance rates through architectural enhancements (e.g., EAGLE, Medusa), knowledge distillation, and tree-search verification.

**Limitations of Prior Work**: Existing methods almost exclusively rely on a single drafter. However, a single drafter is typically trained for specific tasks or domains and performs poorly on out-of-distribution inputs or dynamic user queries. As the trend toward "integration of expert models" (similar to LLM routing) rises, the limitations of single drafters become more pronounced.

**Key Challenge**: Different tasks require different drafters, but it is impossible to determine the most suitable drafter for a given input before inference. Manual switching is impractical, necessitating an adaptive dynamic selection mechanism.

**Goal**: Design a multi-drafter framework capable of dynamically selecting the optimal drafter during the inference process.

**Key Insight**: Speculative decoding naturally provides "alignment feedback"—the degree of match between the drafter's predictions and the target model's predictions—which can serve as a real-time reward signal. This perfectly corresponds to the Multi-Armed Bandit problem, where each drafter is an arm and the alignment feedback is the reward signal.

**Core Idea**: Multi-drafter speculative decoding is modeled as an MAB problem. The paper proposes Block Divergence (BD) as a reward signal (which is more informative and has lower variance than traditional block efficiency) and applies the UCB algorithm to dynamically balance exploration and exploitation for drafter selection.

## Method

### Overall Architecture

The system maintains a pool of $K$ heterogeneous drafters. In each speculative decoding round, the UCB algorithm selects one drafter to perform a draft-verify-accept step. A Block Divergence reward is calculated based on the acceptance result to update the empirical mean and confidence interval for that drafter. This cycle continues until a sequence of target length $B$ is generated.

### Key Designs

1.  **Block Divergence (BD) Reward**:
    - **Function**: Provides more informative alignment feedback than traditional block efficiency (BE).
    - **Mechanism**: $r_{i,t}^{BD} = \frac{1}{N_{max}} \sum_{j=0}^{N_{max}-1} (1 - d_{TV}(p^{l(t)+j}, q_i^{l(t)+j}))$, which calculates the average Total Variation (TV) distance between the probability distributions of the target model and the drafter across all positions in a draft block. Unlike BE (which only counts accepted tokens), BD provides continuous alignment information at every position, avoiding binary information loss.
    - **Design Motivation**: It is theoretically proven that the "feedback signal" of the BD reward, $R(r_i) = \frac{\Delta_i^2}{\max(\text{Var}[r_i], \text{Var}[r_{i^*}])}$, is greater than that of the BE reward in most cases. A stronger feedback signal allows the bandit algorithm to identify the optimal drafter more quickly and accurately. Empirical results also verify that BD has lower variance and larger mean differences.

2.  **Stopping Time Regret Objective**:
    - **Function**: Defines an appropriate optimization objective for multi-drafter speculative decoding.
    - **Mechanism**: $\text{Reg}(\pi, B) = \mathbb{E}[\tau(\pi, B)] - \mathbb{E}[\tau(\pi^*, B)]$, aiming to minimize the gap between the number of speculative decoding rounds required to generate $B$ tokens and the optimal strategy. A lemma proves this is equivalent to maximizing the number of accepted tokens, aligning with the goal of speculative decoding.
    - **Design Motivation**: Standard MAB regret (maximizing cumulative reward) does not directly correspond to speculative decoding efficiency, as the number of rounds is stochastic and depends on the quality of the drafters.

3.  **MetaSD-UCB Algorithm**:
    - **Function**: Dynamically selects the current optimal drafter by balancing exploration and exploitation.
    - **Mechanism**: The drafter is chosen via $a_t = \arg\max_{i \in [K]} \hat{\mu}_{i,t} + \beta \sqrt{\frac{2 \ln t}{n_i}}$, where $\hat{\mu}_{i,t}$ is the empirical mean reward and the second term is the confidence interval. Each drafter is tried once during the initialization phase, followed by UCB selection. Theoretical analysis proves an $O(\ln B)$ regret upper bound under the stopping time regret objective.
    - **Design Motivation**: UCB achieves optimal performance in standard stochastic bandits; MetaSD extends this naturally to the non-standard setting of speculative decoding with rigorous regret analysis.

### Loss & Training

Completely training-free. MetaSD is an inference-time algorithm requiring no additional training. Drafters can be any pre-trained models, supporting both black-box (independent drafters) and white-box (e.g., EAGLE drafters using target LLM hidden states) configurations.

## Key Experimental Results

### Black-box Speculative Decoding Speedup

| Task | Best Single Drafter | MetaSD-UCB |
|------|-------------|------------|
| Code | 2.437 | 2.300 |
| Translation | 2.076 | 1.587 |
| Summary | 2.133 | 1.971 |
| QA | 1.960 | 1.711 |
| Math | 2.454 | 2.280 |

### White-box Speculative Decoding Speedup (EAGLE Drafters)

| Task | Best Single Drafter | MetaSD-UCB |
|------|-------------|------------|
| Code | 3.934 | 3.724 |
| Translation | 2.496 | 2.318 |
| Summary | 3.382 | 3.057 |
| QA | 2.916 | 2.641 |
| Math | 3.903 | 3.520 |

### Key Findings
- MetaSD-UCB automatically selects performance levels near those of the optimal "expert" drafter without prior knowledge of the task type.
- MetaSD-UCB significantly outperforms random selection (Rand) and static integration, demonstrating the effectiveness of dynamic selection.
- The larger mean differences and lower variance of the BD reward enable UCB to converge to the optimal drafter faster.
- The framework naturally handles non-stationarity between queries (by re-initializing for each query) and can be extended to intra-query non-stationarity.
- Requires no additional training and is plug-and-play.

## Highlights & Insights
- **The combination of speculative decoding and Multi-Armed Bandits** is highly natural: alignment feedback provides reward signals without needing extra design. This modeling introduces online decision theory to LLM inference acceleration.
- **Deep theoretical analysis of Block Divergence vs. Block Efficiency**: Proving BD's superiority over BE from the perspective of feedback signal strength provides an analytical framework applicable to other scenarios requiring reward design.
- **Stopping Time Regret** as a new optimization objective is meaningful: It reveals that standard MAB regret is not directly applicable to speculative decoding and requires specialized design.

## Limitations & Future Work
- The acceleration ratio of MetaSD is always slightly lower than the ideal upper bound of an "oracle" optimal drafter, as some rounds must be spent exploring non-optimal drafters.
- Switching drafters incurs KV-cache recalculation overhead, though this can be mitigated using Sequential Halving.
- Access to the drafter's output distribution is required (rather than just sampled tokens), making it unsuitable for pure black-box APIs.
- The experiments used 5 drafters; larger drafter pools may require more efficient exploration strategies.

## Related Work & Insights
- **vs. Standard Speculative Decoding**: Standard methods use a single drafter, while MetaSD extends to a multi-drafter pool with dynamic selection.
- **vs. LLM Routing**: LLM routing directs queries between models, whereas MetaSD routes speculative decoding steps between drafters at a much finer granularity.
- **vs. Hou et al. (2025) concurrent work**: Both use MAB for speculative decoding, but MetaSD introduces the BD reward and stronger instance-dependent regret upper bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ The MAB modeling for multi-drafter speculative decoding is natural and elegant, and the BD reward design has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers black-box/white-box, multi-task, multi-lingual, and non-stationary environments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical analysis with mutual verification between experiments and theory.
- Value: ⭐⭐⭐⭐ Provides a theoretically optimal selection algorithm for multi-drafter speculative decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](../../ICML2026/llm_efficiency/minedraft_a_framework_for_batch_parallel_speculative_decoding.md)

</div>

<!-- RELATED:END -->
