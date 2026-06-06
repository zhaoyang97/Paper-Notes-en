---
title: >-
  [Paper Note] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling
description: >-
  [ICLR 2026][LLM Reasoning][large language reasoning models] This paper treats reflection tokens (e.g., "wait", "but") in the reasoning process as schedulable "resources" and…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "large language reasoning models"
  - "reflection token scheduling"
  - "test-time scaling"
  - "cyclical learning rate"
  - "decoding strategy"
date: 2026-05-08
content_hash: 21c88b6102bbb003
---

# CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling

**Conference**: ICLR 2026
**arXiv**: [2506.11077](https://arxiv.org/abs/2506.11077)  
**Code**: [https://github.com/OPTML-Group/CyclicReflex](https://github.com/OPTML-Group/CyclicReflex)  
**Area**: LLM Reasoning
**Keywords**: large language reasoning models, reflection token scheduling, test-time scaling, cyclical learning rate, decoding strategy

## TL;DR
This paper treats reflection tokens (e.g., "wait", "but") in the reasoning process as schedulable "resources" and, inspired by cyclical learning rate scheduling in optimization, proposes CyclicReflex — a training-free decoding strategy that dynamically modulates the logits of reflection tokens via a triangular waveform. CyclicReflex consistently improves the accuracy of 1.5B–8B models across multiple mathematical reasoning benchmarks (MATH500, AIME2024/2025, AMC2023).

## Background & Motivation
Large reasoning models (LRMs) such as OpenAI o1 and DeepSeek-R1 tackle complex problems through multi-step reasoning, guided by "reflection tokens" (e.g., "wait", "but", "alternatively"). These tokens serve as pivotal turning points and self-evaluation signals within reasoning trajectories.

However, existing LRMs exhibit two symmetric failure modes:
- **Under-reflection**: Too few reflection tokens cause the model to terminate reasoning prematurely, preventing sufficient exploration of solution paths — analogous to a learning rate that is too small, leading to premature convergence.
- **Over-reflection**: Excessive reflection tokens cause the model to loop repeatedly (e.g., continuously outputting "wait"), wasting computational resources and failing to converge to the correct answer — analogous to a learning rate that is too large, causing optimization divergence.

Existing methods such as TIP (Thought switching penalty) can only unidirectionally suppress reflection tokens with a fixed logit penalty, making them incapable of simultaneously addressing under- and over-reflection across problems of varying difficulty. The authors pose the core question: **How can a resource allocation strategy dynamically regulate the frequency and placement of reflection tokens?** Their key insight is to draw an analogy between reflection token scheduling and learning rate scheduling in optimization, specifically borrowing the "stepsize hedging" idea from cyclical learning rates.

## Method

### Overall Architecture
CyclicReflex is a training-free decoding strategy that dynamically adjusts the logits of reflection tokens based on the current token position during autoregressive generation. Given a question $\mathbf{x}$, it produces a reasoning trace $\mathbf{r}$ and a final answer $\mathbf{y}$. The method requires no modification of model parameters and operates purely at inference time.

### Key Designs

1. **Formalization of reflection token resource allocation**: Reflection tokens ("wait", "but", "alternatively", etc.) are treated as schedulable resources whose frequency and placement directly influence reasoning quality. Extended experiments with TIP (allowing positive and negative $\alpha$) reveal that TIP($-3$) yields the largest gain on hard problems but severely degrades performance on easy ones, while TIP($+1$) provides a slight improvement on easy problems. This demonstrates that a single fixed strategy cannot generalize across varying difficulty levels.

2. **Validation via thought landscape analogy**: The Landscape of Thoughts visualization tool is used to project reasoning steps into a 2D space, validating three distinct patterns:
    - Under-reflection: the reasoning trajectory is overly conservative and fails to move far from the starting point.
    - Desired-reflection: the trajectory is well-structured and converges to the correct answer.
    - Over-reflection: the model reaches a region close to the correct answer (e.g., "Alternatively, perhaps the correct answer is...") but excessive reflection causes it to overshoot and ultimately deviate from the correct answer.

3. **Core formulation of CyclicReflex**: A position-dependent bidirectional triangular waveform is used to modulate the logits of reflection tokens:

$$\hat{z}_{t,v} = \begin{cases} z_{t,v} + \delta(t) & \text{if } v \in \hat{V} \\ z_{t,v} & \text{otherwise} \end{cases}$$

$$\delta(t) = A \left| \frac{4 \cdot (t - C/4) \bmod C}{C} - 2 \right| - A$$

where $A$ is the amplitude (controlling adjustment magnitude), $C$ is the period (controlling oscillation frequency), and $\hat{V}$ is the set of reflection tokens. The schedule yields $\delta(C/4) = A > 0$, which promotes reflection, and $\delta(3C/4) = -A < 0$, which suppresses reflection.

4. **Key distinctions from TIP**:
    - TIP is **unidirectional and static** (fixed $\alpha \leq 0$), only suppressing reflection tokens.
    - CyclicReflex is **bidirectional and dynamic**, alternately promoting and suppressing reflection.
    - The ascending phase encourages exploration (switching reasoning directions), while the descending phase facilitates convergence (stabilizing the reasoning process).
    - This mirrors the stepsize hedging strategy of cyclical learning rates.

### Loss & Training
The method requires no training whatsoever and is a purely inference-time strategy. Hyperparameters are determined via grid search: $A \in [1, 10]$ and $C \in [200, 2000]$ (varying by dataset).

## Key Experimental Results

### Main Results

| Dataset | Model | Metric | Original | TIP | S1 | Silver | CyclicReflex |
|--------|------|------|----------|-----|----|----|-------------|
| MATH500 | Qwen-7B | Acc | 0.86 | 0.87 | 0.83 | 0.88 | **0.89** |
| AIME2024 | Qwen-7B | Acc | 0.43 | 0.43 | 0.33 | 0.37 | **0.50** |
| AIME2025 | Qwen-7B | Acc | 0.31 | 0.30 | 0.33 | 0.30 | **0.37** |
| AMC2023 | Qwen-7B | Acc | 0.81 | 0.85 | 0.85 | 0.85 | **0.90** |
| AIME2024 | Llama-8B | Acc | 0.42 | 0.47 | 0.43 | 0.47 | **0.53** |
| AMC2023 | Llama-8B | Acc | 0.81 | 0.85 | 0.75 | 0.85 | **0.90** |
| MATH500 | Qwen-1.5B | Acc | 0.74 | 0.75 | 0.73 | 0.75 | **0.77** |
| AIME2024 | Qwen-1.5B | Acc | 0.23 | 0.23 | 0.17 | 0.27 | **0.30** |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Different difficulty levels | Gains across Easy/Medium/Hard | TIP only effective on Hard; degrades on Easy |
| + Best-of-N ($N=8$) | Consistent improvement in BoN accuracy | Compatible and complementary with external test-time methods |
| + Beam Search | Consistent improvement in BS accuracy | Gains more pronounced under low budget |
| Initial phase $\phi=0$ optimal | — | Encouraging reflection early and suppressing it later is most effective |
| Period $C$ more influential | Accuracy more sensitive to $C$ | $C=600$ optimal for Qwen-7B on MATH500 |
| Amplitude $A$ controls length | Primarily affects number of reflection tokens and generation length | Larger $A$ yields longer reasoning traces |

### Key Findings
- CyclicReflex consistently improves performance across all model scales (1.5B–8B) and all datasets, while maintaining generation lengths comparable to the original decoding strategy.
- Self-correction capability is substantially enhanced: given an erroneous reasoning trace (100% length prefix), CyclicReflex's correction rate substantially exceeds that of TIP and the original decoding strategy.
- Generated thought landscapes are more concentrated with fewer distractor regions, and reasoning trajectories converge more directly to the correct answer.
- S1 (forcing insertion of "Wait") severely degrades performance on AMC2023, demonstrating that naively increasing reflection tokens is insufficient.

## Highlights & Insights
- **Precise analogical thinking**: The analogy between reflection token scheduling and learning rate scheduling is highly apt — under-reflection ↔ learning rate too small ↔ premature convergence; over-reflection ↔ learning rate too large ↔ oscillatory divergence. This analogy is not only intuitively compelling but is also well-validated through thought landscape visualizations.
- **Minimalist yet effective design**: The entire method reduces to a single triangular waveform function with no learnable parameters, making it trivially implementable with zero additional overhead.
- **Bidirectionality as the key innovation**: In contrast to TIP's unidirectional suppression, CyclicReflex's ability to alternately promote and suppress reflection enables it to adapt to problems of varying difficulty.
- **Seamless compatibility with external test-time scaling methods**: Combinations with both Best-of-N and Beam Search yield further improvements.

## Limitations & Future Work
- The theoretical foundations remain relatively weak: the root causes of over- and under-reflection in LRMs are not fully explained.
- Hyperparameters ($A$ and $C$) require dataset-specific grid search, with no adaptive mechanism proposed.
- Validation is limited to mathematical reasoning tasks; generalization to code generation, logical reasoning, and other reasoning scenarios has not been tested.
- The definition of reflection tokens ("wait", "but", etc.) is heuristic in nature, and reflection patterns may differ across models.
- The optimality of the initial phase $\phi = 0$ hints at deeper underlying dynamics of the reasoning process, which warrants further investigation.

## Related Work & Insights
- **TIP** (Wang et al., 2025a): Suppresses reflection tokens via a fixed penalty to address overthinking; serves as the direct baseline for this work.
- **S1** (Muennighoff et al., 2025): Forces insertion of "Wait" after the thinking tag, but yields unstable results.
- **Silver Stepsize Schedule** (Altschuler & Parrilo, 2024): A stepsize hedging strategy from optimization theory, theoretically proven to accelerate convergence.
- **Cyclical Learning Rates** (Smith, 2017): The cyclical learning rate strategy in deep learning, which serves as the core inspiration for this work.
- Insight: Scheduling strategies from optimization theory may offer broader guidance for the reasoning processes of LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ (the analogy is novel, though the method itself is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multiple models and datasets, thorough ablations, well-executed visualizations)
- Writing Quality: ⭐⭐⭐⭐⭐ (smooth narrative, clear analogies, excellent figures)
- Value: ⭐⭐⭐⭐ (strong practical utility, though theoretical foundations warrant further development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[AAAI 2026\] Improving Value-based Process Verifier via Low-Cost Variance Reduction](../../AAAI2026/llm_reasoning/improving_value-based_process_verifier_via_low-cost_variance_reduction.md)
- [\[ACL 2026\] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning](../../ACL2026/llm_reasoning/delta_dynamic_layer-aware_token_attention_for_efficient_long-context_reasoning.md)
- [\[ICLR 2026\] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)

</div>

<!-- RELATED:END -->
