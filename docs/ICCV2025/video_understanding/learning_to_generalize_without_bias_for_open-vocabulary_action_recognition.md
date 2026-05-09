---
title: >-
  [Paper Note] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition
description: >-
  [ICCV 2025][Video Understanding][Open-vocabulary action recognition] This paper proposes Open-MeDe, a meta-learning-based framework for open-vocabulary action recognition (OVAR). By simulating "known-to-open" generalization tasks via cross-batch meta-optimization and stabilizing training with a Gaussian weight averaging strategy, the framework improves generalization in both in-context and out-of-context settings without relying on CLIP regularization.
tags:
  - ICCV 2025
  - Video Understanding
  - Open-vocabulary action recognition
  - meta-learning
  - static bias elimination
  - CLIP adaptation
  - self-ensemble
date: 2026-05-08
content_hash: 7ad365990323ca2c
---

# Learning to Generalize Without Bias for Open-Vocabulary Action Recognition

**Conference**: ICCV 2025
**arXiv**: [2502.20158](https://arxiv.org/abs/2502.20158)
**Code**: [GitHub](https://github.com/Mia-YatingYu/Open-MeDe)
**Area**: Video Understanding
**Keywords**: Open-vocabulary action recognition, meta-learning, static bias elimination, CLIP adaptation, self-ensemble

## TL;DR
This paper proposes Open-MeDe, a meta-learning-based framework for open-vocabulary action recognition (OVAR). By simulating "known-to-open" generalization tasks via cross-batch meta-optimization and stabilizing training with a Gaussian weight averaging strategy, the framework improves generalization in both in-context and out-of-context settings without relying on CLIP regularization.

## Background & Motivation
Open-vocabulary action recognition (OVAR) requires models to recognize action categories unseen during training, placing stringent demands on generalization and zero-shot capability. Current mainstream approaches adapt CLIP for video understanding, but face a core tension:

**Pain Point 1: Standard fine-tuning leads to overfitting.** Directly fine-tuning CLIP-based video learners tends to overfit to training categories, yielding strong performance on seen classes but degraded zero-shot generalization.

**Pain Point 2: CLIP regularization introduces static bias.** Methods such as Open-VCLIP and FROSTER apply regularization to prevent the model from deviating from CLIP's generalization capacity, achieving reasonable in-context performance. However, since CLIP is an image-pretrained model, such regularization causes video learners to over-rely on **shortcut static features** (e.g., scene backgrounds) while neglecting critical motion cues. When evaluated in out-of-context settings (e.g., with video backgrounds replaced), performance drops sharply.

**Key Challenge**: CLIP's static generalization capacity is a double-edged sword—it facilitates in-context generalization while hindering the learning of motion cues and harming out-of-context generalization.

**Key Insight**: Drawing on the meta-learning perspective of "learning to generalize," the video learner is explicitly encouraged during training to rapidly generalize to arbitrary subsequent data, thereby minimizing inherent bias toward seen data and static cues. **Core Idea: cross-batch meta-optimization naturally simulates the known-to-open generalization task, achieving static debiasing without CLIP regularization.**

## Method

### Overall Architecture
Open-MeDe consists of two core components: (1) a cross-batch meta-optimization scheme that simulates the known-to-open generalization task via support–query dual-batch training; and (2) Gaussian Weight Averaging (GWA), which performs weighted averaging over the optimization trajectory to obtain robust generalization parameters. The framework is model-agnostic and can be integrated into any CLIP-based video learner.

### Key Designs
1. **Cross-batch Meta-optimization**:

    - **Function**: Expands each training step into a dual-batch operation—the current batch serves as the support set (meta-train), and the subsequent batch serves as the query set (meta-test).
    - **Mechanism**:
        - **Inner loop (meta-train)**: Fast weights are computed by updating on the support batch $\mathcal{S}$ with standard cross-entropy loss: $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta)$
        - **Outer loop (meta-test + meta-optimization)**: Generalization performance is evaluated on the query batch $\mathcal{Q}$ using fast weights, followed by joint optimization over support and query losses: $\min_\theta (\mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta) + \mathcal{L}_{\mathcal{T}_i}^{\mathcal{Q}}(\theta_i'))$
        - First-order approximation (FOMAML) is adopted to avoid second-order gradient computation.
    - **Design Motivation**: Different batches naturally exhibit different class distributions, so cross-batch evaluation inherently simulates a "known-to-open" generalization scenario. Unlike conventional MAML, which requires constructing explicit N-way K-shot tasks, this approach introduces no additional overhead—it directly exploits the stochasticity of the training data sampler.

2. **Gaussian Weight Averaging (GWA)**:

    - **Function**: Performs weighted averaging of model parameters along the optimization trajectory to obtain a final parameter set with stronger generalization.
    - **Mechanism**: A Gaussian distribution $w_t = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(t-\mu)^2}{2\sigma^2}}$ assigns weights to each epoch's parameters. After normalization $\alpha_t = w_t / \sum_i w_i$, parameters are updated via a running average: $\theta_{\text{GWA}} \leftarrow \frac{\sum_{i=1}^{t-1} w_i}{\sum_{i=1}^{t} w_i} \cdot \theta_{\text{GWA}} + \frac{w_t}{\sum_{i=1}^{t} w_i} \cdot \theta_t$
    - **Design Motivation**: Early-epoch parameters retain excessive CLIP static bias, while late-epoch parameters are overly specialized. GWA assigns higher weight to intermediate epochs and excludes CLIP's original weights $\theta_0$, achieving static debiasing while preserving generalization.

3. **Implicit Debiasing Without CLIP Regularization**:

    - **Function**: Naturally eliminates static bias during meta-optimization, without explicit CLIP regularization terms.
    - **Mechanism**: The virtual evaluation mechanism in meta-learning compels the model to learn genuinely generalizable video features rather than relying on CLIP's static priors. Feedback from the query batch encourages the model to capture motion cues rather than static shortcuts.
    - **Design Motivation**: Explicit CLIP regularization not only incurs additional computational cost but also forces the retention of static features, which is the root cause of out-of-context performance degradation.

### Loss & Training
- The base loss is the standard visual-language cross-entropy loss $\mathcal{L}_{CE}$.
- Meta-optimization objective: $\theta \leftarrow \theta - \beta \sum_{i=1}^{N} (\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta) + \delta \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{Q}}(\theta_i'))$
- The text encoder is frozen; only the temporal adaptation modules within the visual encoder are optimized.
- Training is conducted on K400, with the first 2 epochs serving as a warm-up phase.

## Key Experimental Results

### Main Results

| Dataset | Metric | Open-MeDe | Open-VCLIP | FROSTER | Frozen CLIP |
|---------|--------|-----------|------------|---------|-------------|
| K400 (Novel) | Top-1 Acc | **63.8** | 62.3 | 61.9 | 53.4 |
| K400 (HM) | Top-1 Acc | **69.9** | 68.6 | 68.3 | 57.5 |
| HMDB (Novel) | Top-1 Acc | **56.4** | 50.2 | 49.9 | 46.8 |
| UCF (Novel) | Top-1 Acc | **78.5** | 77.2 | 76.9 | 63.6 |
| SSv2 (HM) | Top-1 Acc | **14.3** | 12.9 | 12.4 | 5.1 |

Cross-dataset zero-shot evaluation (trained on K400, tested on other datasets):

| Dataset | Open-MeDe | Open-VCLIP | FROSTER | Frozen CLIP |
|---------|-----------|------------|---------|-------------|
| UCF | **83.7±1.3** | 83.3±1.4 | 82.9±0.6 | 73.8±0.6 |
| HMDB | **54.6±1.1** | 53.8±1.5 | 53.4±1.2 | 47.9±0.5 |
| K600 | **73.7±0.9** | 73.0±0.8 | 71.1±0.8 | 68.1±1.1 |

### Ablation Study

| Configuration | UCF (in-ctx) | UCF-SCUBA (out-ctx) | HM | Note |
|---------------|-------------|--------------------|----|------|
| Frozen CLIP | 73.8 | 42.0 | 54.1 | Baseline, no fine-tuning |
| Open-VCLIP | 83.3 | 33.2 | 47.4 | CLIP regularization hurts out-ctx |
| FROSTER | 82.9 | 34.1 | 48.1 | Same issue |
| Open-MeDe (w/o GWA) | 82.5 | 41.8 | 55.2 | Meta-optimization only |
| Open-MeDe (full) | **83.7** | **44.7** | **57.6** | Meta-optimization + GWA |

### Key Findings
- **CLIP regularization is a double-edged sword**: Open-VCLIP and FROSTER show clear in-context gains but perform worse than frozen CLIP in out-of-context settings (UCF-SCUBA), confirming the severity of static bias.
- Meta-optimization alone, even without GWA, approaches frozen CLIP performance in out-of-context settings while maintaining high in-context performance.
- GWA further improves out-of-context performance by approximately 2.9 percentage points.
- Gains are particularly pronounced on the temporally sensitive SSv2 dataset (HM: 14.3 vs. 12.9), indicating that the method genuinely enhances the learning of motion cues.
- The method is model-agnostic and applicable to different video adapter architectures.

## Highlights & Insights
- **Precise problem formulation**: The paper is the first to analyze the out-of-context degradation of CLIP regularization methods from the perspective of static bias.
- **Elegant meta-learning design**: No additional task distribution construction is required; the approach directly exploits the natural stochasticity of mini-batches to simulate generalization tasks.
- **No additional computational overhead**: Unlike CLIP regularization methods (which require extra forward passes for distillation loss), meta-optimization requires only one additional gradient computation step on existing batches.
- **GWA is a simple yet effective ensemble strategy**: The Gaussian prior assigns maximum weight to intermediate epochs, avoiding both early-epoch static bias and late-epoch over-specialization.

## Limitations & Future Work
- Meta-optimization requires twice the batch data (support + query); while theoretically introducing no extra overhead, memory consumption increases in practice.
- Absolute performance on SSv2 remains low (HM: 14.3), suggesting that static debiasing alone may be insufficient for temporal reasoning tasks.
- The $\mu$ and $\sigma$ hyperparameters of GWA require manual tuning and lack an adaptive mechanism.
- Validation on larger CLIP models (e.g., ViT-L) has not been conducted.

## Related Work & Insights
- The paper offers a new perspective on open-vocabulary problems by advocating "learning to generalize" rather than "learning to regularize."
- The cross-batch meta-optimization idea can be extended to other vision-language adaptation scenarios (e.g., open-vocabulary detection).
- The GWA strategy can serve as a general fine-tuning stabilization technique, as an alternative to EMA or SWA.

## Rating
- Novelty: ⭐⭐⭐⭐ The meta-learning perspective for OVAR is novel, though MAML and weight averaging are both well-established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers diverse in-context and out-of-context settings with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is insightful, though notation is occasionally redundant.
- Value: ⭐⭐⭐⭐ Provides practical guidance for addressing the static bias problem in the OVAR field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[CVPR 2026\] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](../../CVPR2026/video_understanding/decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)

</div>

<!-- RELATED:END -->
