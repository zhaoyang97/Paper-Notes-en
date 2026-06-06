---
title: >-
  [Paper Note] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning
description: >-
  [ACL 2026][Reinforcement Learning][Overthinking] Ours proposes AttnPO, a low-overhead process-supervised RL framework that utilizes the model's intrinsic attention signals for step-level credit assignment. By identifying…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Overthinking"
  - "Process Supervision"
  - "Attention Mechanism"
  - "Reasoning Efficiency"
date: 2026-05-08
content_hash: 7fb6f237ef4b225f
---

# AttnPO: Attention-Guided Process Supervision for Efficient Reasoning

**Conference**: ACL 2026  
**arXiv**: [2602.09953](https://arxiv.org/abs/2602.09953)  
**Code**: [GitHub](https://github.com/NieSYsc20/AttnPO)  
**Area**: Reinforcement Learning / Efficient Reasoning  
**Keywords**: Overthinking, Process Supervision, Attention Mechanism, Reinforcement Learning, Reasoning Efficiency

## TL;DR

Ours proposes AttnPO, a low-overhead process-supervised RL framework that utilizes the model's intrinsic attention signals for step-level credit assignment. By identifying Key-Focus Heads (KFH) to distinguish between redundant and critical reasoning steps, it significantly improves accuracy while substantially reducing reasoning length.

## Background & Motivation

**Background**: Large Reasoning Models (LRM) trained on RLVR, such as DeepSeek-R1, perform excellently on complex reasoning tasks but suffer from severe "overthinking"—generating lengthy reasoning processes even for simple operations, which wastes computational resources.

**Limitations of Prior Work**: (1) Trajectory-level length penalties treat all reasoning steps uniformly and cannot distinguish between redundant and necessary steps, often leading to a decrease in accuracy; (2) Sampling-based process supervision methods (Monte Carlo sampling) involve high computational overhead; (3) Model-based methods (training a reward model to locate the first correct answer) provide imprecise credit assignment.

**Key Challenge**: Need for fine-grained step-level supervision to distinguish redundant from critical steps, but existing methods are either high-overhead (extra sampling/models) or inaccurate in credit assignment.

**Goal**: Achieve fine-grained step-level process supervision with near-zero additional resource cost, relying solely on the model's intrinsic signals.

**Key Insight**: In-depth analysis of the model's attention mechanism reveals specific attention heads that naturally focus on critical steps when the final answer is generated.

**Core Idea**: Key-Focus Heads (KFH) naturally assign high attention to critical reasoning steps and low attention to redundant steps during final answer generation, which can be directly used for step-level credit assignment.

## Method

### Overall Architecture

Based on the GRPO/RLOO framework, AttnPO uses attention scores from KFH to scale the outcome-level advantage at a step level: for correct responses with a positive advantage, it attenuates the positive advantage of redundant steps (reducing over-encouragement); for correct responses with a negative advantage, it attenuates the negative advantage of critical steps (avoiding over-punishment).

### Key Designs

1.  **Key-Focus Heads (KFH) Discovery and Validation**:

    - **Function**: Identify attention heads capable of distinguishing critical vs. redundant reasoning steps.
    - **Mechanism**: Define step scores $\mathcal{S}_{s_k}^{l,h} = \frac{1}{|s_k|}\sum_{m \in \mathcal{F}}\sum_{n \in s_k} a_{m \to n}^{l,h}$ (attention from final answer to reasoning steps), and measure differentiation capability using Step Ranking Accuracy (SRA)—the best heads reach 95-96% SRA.
    - **Design Motivation**: An LRM must select critical information from lengthy reasoning when generating the final answer; the attention mechanism is a natural information selection tool.

2.  **Positive Advantage Redundancy Attenuation**:

    - **Function**: Reduce over-encouragement of redundant steps to mitigate overthinking.
    - **Mechanism**: When $A^i > 0$ and $\mathcal{S}_{s_k}^i < \mathcal{S}_{\text{base}}^i$, the advantage is decayed using a scaling factor $\gamma_{s_k}^i = (1-\delta) \cdot p_i^\lambda \cdot (\mathcal{S}_{s_k}^i / \mathcal{S}_{\text{base}}^i) + \delta$; the baseline score $\mathcal{S}_{\text{base}}^i = p_i^\beta \cdot \frac{|\mathcal{F}_i|}{|o_i|}$ is difficulty-aware.
    - **Design Motivation**: Positive advantages strengthen the generation probability of all steps; redundant steps need to be selectively weakened.

3.  **Negative Advantage Critical Step Protection**:

    - **Function**: Avoid over-penalizing critical steps in correct reasoning paths.
    - **Mechanism**: When $A^i < 0$ and $\mathcal{S}_{s_k}^i > \mathcal{S}_{\text{base}}^i$, $\gamma_{s_k}^i$ is set to 0 (full exemption from punishment), concentrating the penalty on redundant steps.
    - **Design Motivation**: Critical steps in correct replies with negative advantages should not be punished, as this would harm the model's reasoning capability.

### Loss & Training

The reward function is defined as $r_i = \mathbb{I}[o_i \text{ correct}](1 - \alpha \cdot \sigma(f(o_i)))$, where $f(o_i) = \sigma((\text{len}(o_i) - \text{mean}(q)) / \text{std}(q))$. The RLOO advantage estimator $A^i = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j$ is used. KFH are selected from the top-N heads ranked by SRA, and their behavior remains stable during RL training (Pearson correlation > 0.85).

## Key Experimental Results

### Main Results (1.5B Model)

| Method | GSM8K Acc | MATH500 Acc | AIME24 Acc | AIME25 Acc | Average Acc | Average Token |
|------|----------|------------|-----------|-----------|---------|-----------|
| DS-R1-1.5B Baseline | 78.8 | 82.1 | 28.1 | 22.8 | 54.5 | 8005 |
| AutoThink | 83.0 | 84.0 | 34.6 | 21.8 | 57.0 | 5056 |
| AdaptThink | 83.1 | 82.0 | - | - | - | - |
| AttnPO (Ours) | **Significant Gain** | **Significant Gain** | **Significant Gain** | - | **+7.3pts** | **-60%** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Pos-Adv Attenuation Only | Effectively reduces length but limited accuracy gain |
| Neg-Adv Protection Only | Effectively protects accuracy but limited length reduction |
| Combination (AttnPO) | Simultaneously achieves large reduction and accuracy gain |
| Remove high SRA steps vs low SRA steps | Removing high SRA steps significantly lowers pass@32, low SRA steps have minimal impact |

### Key Findings

- KFH are primarily located in the middle and late layers; a small number of heads (SRA > 0.9) is sufficient, with diminishing returns for more heads.
- KFH behavior is highly stable during the RL training process, showing robust functional roles.
- KFH identified on non-difficult problems generalize well to difficult ones (e.g., AIME24).
- On DeepSeek-R1-Distill-Qwen-1.5B, achieved an average +7.3 points accuracy increase and 60% reduction in reasoning length across 6 math benchmarks.

## Highlights & Insights

- First to reveal the existence of Key-Focus Heads in LRMs—naturally focusing on critical steps during final answer generation.
- Near-zero extra overhead: No additional sampling or reward models required; only utilizes existing attention scores.
- The two complementary strategies (Pos-Adv Attenuation + Neg-Adv Protection) are elegantly designed to fulfill their respective roles.
- The difficulty-aware mechanism ($p_i^\beta$ and delayed scheduling $t > T \cdot p_i$) ensures sufficient exploration space for difficult problems.

## Limitations & Future Work

- Reasoning step segmentation relies on predefined special phrases, which may not be universal.
- Performance of KFH on larger models (>7B) has not been fully verified.
- Evaluated only on mathematical reasoning tasks; coding/logic tasks remain to be explored.
- Calculation of attention scores introduces extra overhead during inference (though negligible during training).

## Related Work & Insights

- GRPO / DeepSeek-R1 (Guo et al., 2025): Foundations for outcome-supervised RL.
- TLMRE (Arora & Zanette, 2025): Trajectory-level length penalty methods.
- Monte Carlo sampling methods (Dai et al., 2025; Yue et al., 2025): High-overhead process supervision.
- Attention head functional differentiation (Zheng et al., 2024; Li et al., 2025): Research on the functional specialization of attention heads.
- The discovery of KFH provides a new perspective for understanding the internal working mechanisms of LRMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of KFH is highly insightful, and the idea of using intrinsic signals for process supervision is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 benchmarks, thorough probing analysis, and ablation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative from discovery to application, rigorous formulas.
- Value: ⭐⭐⭐⭐⭐ +7.3pts accuracy + 60% length reduction, extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ACL 2026\] Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning](good_reasoning_makes_good_demonstrations_implicit_reasoning_quality_supervision_.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](../../ICLR2026/reinforcement_learning/regret-guided_search_control_for_efficient_learning_in_alphazero.md)

</div>

<!-- RELATED:END -->
