---
title: >-
  [Paper Note] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation
description: >-
  [NeurIPS 2025][Reinforcement Learning][Visual Reasoning] This paper proposes NoisyRollout, a data augmentation method with zero additional training cost. During GRPO-based VLM training…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Visual Reasoning"
  - "Policy Exploration"
  - "Data Augmentation"
  - "GRPO"
  - "Noise Annealing"
date: 2026-05-08
content_hash: 98fc89d31f9adb87
---

# NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation

**Conference**: NeurIPS 2025
**arXiv**: [2504.13055](https://arxiv.org/abs/2504.13055)
**Code**: [GitHub](https://github.com/NoisyRollout)
**Area**: Reinforcement Learning / VLM Reasoning
**Keywords**: Visual Reasoning, Policy Exploration, Data Augmentation, GRPO, Noise Annealing

## TL;DR

This paper proposes NoisyRollout, a data augmentation method with zero additional training cost. During GRPO-based VLM training, it mixes rollouts from clean and moderately perturbed images to enhance policy exploration diversity. Using only 2.1K samples, it achieves state-of-the-art performance among open-source RL fine-tuned models across five out-of-domain benchmarks.

## Background & Motivation

- Scaling test-time computation (inference) via reinforcement learning is an important direction for enhancing model intelligence, but VLMs face unique challenges:
    - **Insufficient policy exploration**: Conventional methods such as raising the sampling temperature introduce only superficial diversity and fail to guide policies toward more robust behaviors.
    - **Visual perception deficiencies**: VLMs frequently exhibit perception errors that propagate into subsequent reasoning steps.
- Existing VLM-RL works largely transplant methods from the LLM domain without accounting for the specific challenges of visual perception.
- Core insight: **If a model can successfully reason over perturbed images, its reasoning path is more robust; reward discrepancies between clean and perturbed images serve as implicit contrastive signals that improve perception.**

## Method

### Overall Architecture

For each training sample $(I, \mathbf{q})$, the old policy generates two sets of rollouts: $n_1$ rollouts from the clean image and $n_2$ rollouts from the perturbed image $\tilde{I} = T_{\alpha_t}(I)$. All rollouts are mixed to compute the reward baseline and advantage values. **Crucially**, policy updates are conditioned solely on the clean image; perturbed images are used only for collecting diverse rollouts. A noise annealing schedule progressively reduces the perturbation intensity throughout training.

### Key Designs

1. **Mixed Rollout Strategy**:
    - Function: Mixes reasoning trajectories from clean and perturbed images for GRPO optimization.
    - Mechanism: $n_1$ clean rollouts and $n_2$ noisy rollouts jointly form a single group, sharing a unified reward mean and standard deviation as the normalization baseline.
    - Design Motivation:
        - Successful trajectories on perturbed images provide alternative, more robust reasoning paths.
        - Reward discrepancies between clean and perturbed inputs expose perceptual fragility, functioning as implicit contrastive learning.

2. **Noise Annealing Schedule**:
    - Function: Gradually reduces image perturbation intensity during training.
    - Mechanism: Employs a sigmoid-shaped annealing schedule $\alpha_t = \alpha_0 \cdot \left(1 - \frac{1}{1 + e^{-\lambda(t-\gamma)/t_{max}}}\right)$.
    - Design Motivation: High noise in early training encourages exploration; low noise in later stages reduces distributional shift and ensures stable convergence.

3. **Policy Updates Conditioned on Clean Inputs Only**:
    - Function: Although rollouts are collected from perturbed images, the policy gradient is computed using $\frac{\pi_\theta(\mathbf{o}_i | I, \mathbf{q})}{\pi_{\theta_{old}}(\mathbf{o}_i | I, \mathbf{q})}$.
    - Design Motivation: Prevents the policy from learning noise-dependent behaviors, ensuring optimal performance on clean inputs at inference time.

### Loss & Training

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{n_1+n_2}\sum_{i=1}^{n_1+n_2} \min\left(\frac{\pi_\theta(\mathbf{o}_i | I, \mathbf{q})}{\pi_{\theta_{old}}(\mathbf{o}_i | I, \mathbf{q})}\hat{A}_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

- Rule-based rewards are used (correct = 1, incorrect = 0) with no KL divergence penalty.
- Default configuration: Gaussian noise, $n_1 = 6$, $n_2 = 6$ (total rollouts = 12, unchanged).
- The visual encoder is frozen; learning rate is set to 1e-6.

## Key Experimental Results

### Main Results

Qwen2.5-VL-7B-Instruct, trained on only 2.1K Geometry3K samples:

| Method | MathVerse | MathVision | MathVista | WeMath | HallusionBench |
|--------|-----------|------------|-----------|--------|----------------|
| Qwen2.5-VL-7B (base) | 46.2 | 25.0 | 67.5 | 63.1 | 64.6 |
| + Vanilla GRPO | 50.8 | 27.3 | 70.5 | 67.4 | 69.8 |
| + **NoisyRollout** | **53.2** | **28.5** | **72.6** | **69.6** | **72.1** |

### Ablation Study

- **Rollout diversity analysis**: NoisyRollout significantly increases rollout cosine distance diversity in early training, with effects comparable to raising the temperature to 1.2.
- **Temperature comparison**: NoisyRollout (temperature 1.0) consistently outperforms vanilla GRPO at any temperature (0.8–1.4) across all benchmarks, demonstrating more targeted diversity.
- **Noise type**: Both Gaussian noise and rotation are effective; Gaussian noise performs marginally better.
- **Ratio experiment**: $n_1 = 6, n_2 = 6$ (50% noisy rollouts) is the optimal configuration.
- **32B model**: NoisyRollout remains effective at larger scale (MathVision 41.6 vs. GRPO 40.0).

### Key Findings

- Using only 2.1K training samples, NoisyRollout surpasses competitors trained on 15K–260K samples (e.g., OpenVLThinker, R1-VL), demonstrating exceptional data efficiency.
- Improvements on HallusionBench (+2.3%) indicate that NoisyRollout enhances not only reasoning but also visual perception.
- Noise annealing is critical for training stability—fixed perturbation intensity leads to instability in later training stages.
- Consistent gains are observed across different datasets (Geometry3K vs. MMK12) and model scales (7B vs. 32B).

## Highlights & Insights

- The design is remarkably simple ("free lunch"): no additional training cost, no modification to the RL objective, and no increase in total rollout count.
- Using visual perturbation as a policy exploration tool is a novel idea—it exploits the visual perception characteristics of VLMs to provide meaningful diversity.
- The implicit contrastive learning mechanism is elegant: reward discrepancies between clean and perturbed inputs naturally regularize perceptual behavior.
- Data efficiency is striking: 2.1K samples suffice to achieve state-of-the-art results on five out-of-domain benchmarks.

## Limitations & Future Work

- The perturbation types (Gaussian noise, rotation) are relatively simple; more complex augmentations (e.g., occlusion, style transfer) remain unexplored.
- The reasons for the failure of certain strategies such as cropping are not thoroughly analyzed.
- Hyperparameter selection for the noise annealing schedule ($\alpha_0, \lambda, \gamma$) remains largely manual.
- Applicability to non-visual reasoning tasks (e.g., pure text reasoning) is not discussed.

## Related Work & Insights

- NoisyRollout is complementary to concurrent works such as DeepVideo-R1: NoisyRollout improves exploration strategy while DeepVideo-R1 refines the optimization objective.
- The mixed rollout concept is generalizable to other RL fine-tuning scenarios (e.g., code generation, mathematical reasoning).
- Noise annealing aligns with the principle of curriculum learning: transitioning from broad exploration to narrow exploitation.

## Rating

- ⭐⭐⭐⭐⭐ — The method is concise, efficient, and demonstrates strong generalization, representing a practical contribution to the VLM-RL field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [\[NeurIPS 2025\] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](../../ICLR2026/reinforcement_learning/mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](../../ICLR2026/reinforcement_learning/dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)

</div>

<!-- RELATED:END -->
