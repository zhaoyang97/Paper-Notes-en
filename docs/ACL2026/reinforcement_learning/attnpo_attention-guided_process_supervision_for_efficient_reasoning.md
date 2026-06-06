---
title: >-
  [Paper Note] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning
description: >-
  [ACL 2026][Reinforcement Learning][overthinking] This paper proposes AttnPO, a low-overhead process supervision RL framework that leverages intrinsic attention signals for step-level credit assignment. By identifying Key…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "overthinking"
  - "process supervision"
  - "attention mechanism"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: bc14b613bb42f360
---

# AttnPO: Attention-Guided Process Supervision for Efficient Reasoning

**Conference**: ACL 2026
**arXiv**: [2602.09953](https://arxiv.org/abs/2602.09953)  
**Code**: [GitHub](https://github.com/NieSYsc20/AttnPO)  
**Area**: Reinforcement Learning / Efficient Reasoning
**Keywords**: overthinking, process supervision, attention mechanism, reinforcement learning, reasoning efficiency

## TL;DR

This paper proposes AttnPO, a low-overhead process supervision RL framework that leverages intrinsic attention signals for step-level credit assignment. By identifying Key-Focus Heads (KFH) to distinguish redundant from critical reasoning steps, AttnPO substantially reduces reasoning length while significantly improving accuracy.

## Background & Motivation

**Background**: Large reasoning models (LRMs) trained via RLVR, such as DeepSeek-R1, achieve strong performance on complex reasoning tasks but suffer from severe "overthinking"—generating unnecessarily verbose reasoning chains even for simple operations, wasting computational resources.

**Limitations of Prior Work**: (1) Trajectory-level length penalties treat all reasoning steps uniformly, failing to distinguish redundant from necessary steps and often causing accuracy degradation; (2) Sampling-based process supervision methods (Monte Carlo sampling) incur substantial computational overhead; (3) Model-based approaches (training reward models to locate the first correct answer) yield imprecise credit assignment.

**Key Challenge**: Fine-grained step-level supervision is needed to differentiate redundant from critical steps, yet existing methods are either computationally expensive (requiring additional sampling or models) or inaccurate in credit assignment.

**Goal**: Achieve fine-grained step-level process supervision at near-zero additional resource cost, relying solely on intrinsic model signals.

**Key Insight**: A deeper analysis of the model's attention mechanism reveals the existence of specialized attention heads that naturally focus on critical steps during final answer generation.

**Core Idea**: Key-Focus Heads (KFH) naturally allocate high attention to critical reasoning steps and low attention to redundant steps when generating the final answer, and can thus be directly used for step-level credit assignment.

## Method

### Overall Architecture

AttnPO builds upon the GRPO/RLOO framework and uses KFH attention scores to perform step-level scaling of outcome-level advantages: for correct responses with positive advantage, the positive advantage of redundant steps is attenuated (reducing over-encouragement); for correct responses with negative advantage, the negative advantage of critical steps is attenuated (preventing over-penalization).

### Key Designs

1. **Key-Focus Heads (KFH) Discovery and Validation**:

    - Function: Identify attention heads capable of distinguishing critical from redundant reasoning steps.
    - Mechanism: Define the step score as $\mathcal{S}_{s_k}^{l,h} = \frac{1}{|s_k|}\sum_{m \in \mathcal{F}}\sum_{n \in s_k} a_{m \to n}^{l,h}$ (attention from the final answer to reasoning steps), and measure discriminative ability via Step Ranking Accuracy (SRA)—the top-performing heads achieve SRA of 95–96%.
    - Design Motivation: When generating the final answer, LRMs must select critical information from lengthy reasoning chains; the attention mechanism serves as a natural information selection tool.

2. **Positive Advantage Attenuation for Redundant Steps**:

    - Function: Reduce over-encouragement of redundant steps to mitigate overthinking.
    - Mechanism: When $A^i > 0$ and $\mathcal{S}_{s_k}^i < \mathcal{S}_{\text{base}}^i$, the advantage is scaled by $\gamma_{s_k}^i = (1-\delta) \cdot p_i^\lambda \cdot (\mathcal{S}_{s_k}^i / \mathcal{S}_{\text{base}}^i) + \delta$; the baseline score $\mathcal{S}_{\text{base}}^i = p_i^\beta \cdot \frac{|\mathcal{F}_i|}{|o_i|}$ incorporates difficulty awareness.
    - Design Motivation: Positive advantages reinforce the generation probability of all steps; selective attenuation of redundant steps is therefore necessary.

3. **Negative Advantage Protection for Critical Steps**:

    - Function: Prevent over-penalization of critical steps in correct reasoning chains.
    - Mechanism: When $A^i < 0$ and $\mathcal{S}_{s_k}^i > \mathcal{S}_{\text{base}}^i$, the scaling factor is set to $\gamma_{s_k}^i = 0$ (full exemption from penalty), concentrating penalties on redundant steps.
    - Design Motivation: In correct responses with negative advantage, penalizing critical steps would impair the model's reasoning capability.

### Loss & Training

The reward function is defined as $r_i = \mathbb{I}[o_i \text{ correct}](1 - \alpha \cdot \sigma(f(o_i)))$, where $f(o_i) = \sigma((\text{len}(o_i) - \text{mean}(q)) / \text{std}(q))$. The RLOO advantage estimator is $A^i = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j$. KFH selection takes the top-$N$ heads by SRA, and their behavior remains stable throughout RL training (Pearson correlation > 0.85).

## Key Experimental Results

### Main Results (1.5B Model)

| Method | GSM8K Acc | MATH500 Acc | AIME24 Acc | AIME25 Acc | Avg. Acc | Avg. Tokens |
|--------|-----------|-------------|------------|------------|----------|-------------|
| DS-R1-1.5B Baseline | 78.8 | 82.1 | 28.1 | 22.8 | 54.5 | 8005 |
| AutoThink | 83.0 | 84.0 | 34.6 | 21.8 | 57.0 | 5056 |
| AdaptThink | 83.1 | 82.0 | — | — | — | — |
| AttnPO (Ours) | **significant gain** | **significant gain** | **significant gain** | — | **+7.3 pts** | **−60%** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Pos-Adv attenuation only | Effectively reduces length but limited accuracy gain |
| Neg-Adv protection only | Effectively preserves accuracy but limited length reduction |
| Both combined (AttnPO) | Simultaneously achieves substantial length reduction and accuracy improvement |
| Removing high-SRA vs. low-SRA steps | Removing high-SRA steps significantly reduces pass@32; low-SRA steps have minimal impact |

### Key Findings

- KFH are predominantly located in middle-to-late layers; a small number of heads with SRA > 0.9 suffices, with diminishing returns upon adding more.
- KFH behavior remains highly stable throughout RL training, with robust functional roles.
- KFH identified on non-difficult problems generalize effectively to hard problems (AIME24).
- On DeepSeek-R1-Distill-Qwen-1.5B, AttnPO achieves an average accuracy gain of +7.3 points and a 60% reduction in reasoning length across six mathematical benchmarks.

## Highlights & Insights

- This work is the first to reveal the existence of Key-Focus Heads in LRMs—attention heads that naturally concentrate on critical steps during final answer generation.
- Near-zero additional overhead: no extra sampling or reward model is required; only the model's existing attention scores are utilized.
- The two complementary strategies (Pos-Adv attenuation and Neg-Adv protection) are elegantly designed with distinct, non-overlapping roles.
- Difficulty-aware mechanisms ($p_i^\beta$ and delayed scheduling $t > T \cdot p_i$) ensure sufficient exploration space for difficult problems.

## Limitations & Future Work

- Reasoning step segmentation relies on predefined special phrases, which may lack generality.
- The behavior of KFH on larger models (>7B) has not been thoroughly validated.
- Evaluation is limited to mathematical reasoning; tasks such as coding and logical reasoning remain unexplored.
- Computing attention scores incurs additional overhead at inference time (though negligible during training).

## Related Work & Insights

- GRPO / DeepSeek-R1 (Guo et al., 2025): foundational outcome-supervised RL framework.
- TLMRE (Arora & Zanette, 2025): trajectory-level length penalty approach.
- Monte Carlo sampling methods (Dai et al., 2025; Yue et al., 2025): high-overhead process supervision.
- Functional specialization of attention heads (Zheng et al., 2024; Li et al., 2025): research on role differentiation among attention heads.
- The discovery of KFH offers a new perspective for understanding the internal working mechanisms of LRMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of KFH is highly insightful; leveraging intrinsic signals for process supervision is a genuinely novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 9 benchmarks with comprehensive probing analyses and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative from discovery to application is fluid, and the formulations are rigorous.
- Value: ⭐⭐⭐⭐⭐ +7.3 pts accuracy and 60% length reduction demonstrate exceptional practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](../../ICLR2026/reinforcement_learning/regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)

</div>

<!-- RELATED:END -->
