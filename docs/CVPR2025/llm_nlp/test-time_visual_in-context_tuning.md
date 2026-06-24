---
title: >-
  [Paper Note] Test-Time Visual In-Context Tuning
description: >-
  [CVPR 2025][LLM (Other)][In-context learning] This paper proposes Visual In-Context Tuning (VICT), which performs one-shot adaptation of visual in-context learning models (e.g., Painter) at test time by flipping the roles of task prompts and test samples and utilizing cycle consistency loss, significantly improving generalization under distribution shifts.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "In-context learning"
  - "Test-time training"
  - "Cycle consistency"
  - "Distribution shift"
  - "Painter"
date: 2026-05-08
content_hash: 101a631aa3bb1e9a
---

# Test-Time Visual In-Context Tuning

**Conference**: CVPR 2025  
**arXiv**: [2503.21777](https://arxiv.org/abs/2503.21777)  
**Code**: [https://github.com/Jiahao000/VICT](https://github.com/Jiahao000/VICT)  
**Area**: LLM/NLP  
**Keywords**: In-context learning, Test-time training, Cycle consistency, Distribution shift, Painter

## TL;DR
This paper proposes Visual In-Context Tuning (VICT), which performs one-shot adaptation of visual in-context learning models (e.g., Painter) at test time by flipping the roles of task prompts and test samples and utilizing cycle consistency loss, significantly improving generalization under distribution shifts.

## Background & Motivation

**Background**: Visual In-Context Learning (VICL) is a new paradigm in computer vision that formulates various visual tasks as image inpainting by stitching input-output exemplar pairs and test images into a grid. The representative method, Painter, has demonstrated few-shot adaptation capabilities across multiple tasks.

**Limitations of Prior Work**: VICL models are frozen during deployment, but test distributions often differ from training distributions (e.g., image corruptions). Empirical findings show that Painter generalizes poorly under distribution shifts. More surprisingly, even when task prompts from the test distribution are provided (one-shot setting), the performance deteriorates further, indicating that the generalization ability of current VICL models is severely deficient.

**Key Challenge**: VICL models need to adapt to new distributions at inference time, but existing models are frozen and cannot exploit the distribution information inherent in test samples. Meanwhile, traditional test-time training (TTT) methods rely on specific self-supervised tasks (e.g., rotation prediction, MAE reconstruction), which are not universally applicable in the multi-task context of VICL.

**Goal**: This paper aims to design a task-agnostic test-time training method that enables VICL models to adapt to new distributions at inference time using a single test sample.

**Key Insight**: VICL models naturally possess the ability to "predict outputs given prompts." If a model understands the test distribution, it should be able to reconstruct the original task prompt output starting from its own prediction.

**Core Idea**: The predicted test output is fed back to the model as a new "prompt," requiring the model to reconstruct the original task prompt output, thereby establishing cycle consistency. This signal naturally exists within the VICL framework without requiring extra data or annotations, making it applicable to any task.

## Method

### Overall Architecture
Given a task prompt pair $(x,y)$ and a test input $x_t$, a grid $I=(x,y,x_t,\varnothing)$ is first constructed to let the model predict $\hat{y}_t$. Subsequently, the roles are flipped to construct $I'=(x,\varnothing,x_t,\hat{y}_t)$, allowing the model to predict $\hat{y}$. The model weights are optimized using the regression loss between $\hat{y}$ and the ground-truth $y$. Each test sample is optimized independently (re-initializing from the pre-trained weights).

### Key Designs

1. **Cycle-Consistent Self-Supervised Signal**:

    - **Function**: Provides a task-agnostic supervision signal for test-time training.
    - **Mechanism**: Forward propagation predicts the test output $\hat{y}_t = f_\theta(x,y,x_t,\varnothing)$, followed by a role flip to reconstruct the task prompt output $\hat{y} = f_\theta(x,\varnothing,x_t,\hat{y}_t)$, minimizing the smooth-$\ell_1$ loss $\mathcal{L}(\hat{y}, y)$. Key insight: If the model truly adapts to the test distribution, it should be capable of reconstructing the known task prompt via its own prediction.
    - **Design Motivation**: Traditional TTT methods (rotation prediction, MAE) are only suitable for specific scenarios, whereas the grid structure of VICL naturally allows role flipping to generate cost-free supervision signals.

2. **Weight Reset Strategy**:

    - **Function**: Ensures that each test sample adapts independently.
    - **Mechanism**: The model weights are reset back to the pre-trained state $\theta_0$ for every new test input processed, preventing the accumulation of overfitting to prior test samples.
    - **Design Motivation**: Does not assume test samples arise from the same distribution, thereby maximizing flexibility.

3. **Zero-Shot and One-Shot Dual Modes**:

    - **Function**: Covers two practical scenarios.
    - **Mechanism**: Two settings: zero-shot (where task prompts are from the training distribution/clean images) and one-shot (where task prompts are from the test distribution/corrupted images). VICT significantly improves performance under both settings.
    - **Design Motivation**: Annotations from the target distribution may or may not be available during actual deployment.

### Loss & Training
Smooth-$\ell_1$ loss is utilized for optimization, following Painter's design. The model is optimized for a few steps per test sample before inference. The model is built on Painter, using its pre-trained weights as the starting point.

## Key Experimental Results

### Main Results

| Task/Dataset | Painter (zero-shot) | VICT (zero-shot) | Painter (one-shot) | VICT (one-shot) |
|---|---|---|---|---|
| Depth Estimation NYUv2-C (A.Rel↓) | 0.392 | 0.365 | 0.537 | Significant Improvement |
| Semantic Segmentation ADE20K-C | Performance Drop | Significant Recovery | Worse | Significant Improvement |
| Panoptic Segmentation COCO-C | Performance Drop | Significant Recovery | Worse | Significant Improvement |
| Image Denoising SIDD-C | Performance Drop | Significant Recovery | Worse | Significant Improvement |
| Image Deraining | Performance Drop | Significant Recovery | Worse | Significant Improvement |
| Low-Light Enhancement LoL-C | Performance Drop | Significant Recovery | Worse | Significant Improvement |

### Ablation Study

| Configuration | Effect | Explanation |
|---|---|---|
| No TTT (Naive Painter) | Baseline | Poor performance under distribution shifts |
| One-shot Prompt (No TTT) | Worse than zero-shot | VICL generalizes poorly to new distributions |
| VICT Zero-shot | Significant improvement | Cycle consistency is effective |
| VICT One-shot | Further improvement | Distribution-matched prompt + TTT is optimal |

### Key Findings
- Painter exhibits severe performance drops under 15 types of image corruptions, particularly Gaussian noise, impulse noise, etc.
- Providing task prompts from the corrupted domain (one-shot) unexpectedly performs worse than clean prompts (zero-shot), exposing a generalization defect of VICL.
- VICT's zero-shot or one-shot modes can even outperform Painter trained with more few-shot corrupted samples.
- The proposed method can generalize to handling unseen tasks at test time (e.g., transferring from depth estimation to normal estimation).

## Highlights & Insights
- **Elegant Application of Cycle Consistency**: Exploits the property of the VICL grid structure that naturally supports role flipping, constructing a self-supervised signal at zero cost. This insight is highly ingenious, as it requires no external pretext tasks.
- **Revealing Generalization Defects of VICL**: This work is the first to systematically evaluate VICL's performance under distribution shifts, uncovering the counter-intuitive phenomenon that "one-shot is worse than zero-shot," which serves as an important warning to the VICL community.
- **High Versatility**: Demonstrated on 6 vision tasks (ranging from high-level semantic understanding to low-level image processing) across 15 corruptions, the method is entirely task-agnostic.

## Limitations & Future Work
- Multi-step optimization is required for each test sample, significantly increasing inference overhead.
- Evaluation is limited to Painter; the effectiveness on more recent VICL models (e.g., LVM) remains to be verified.
- Cycle consistency assumes the model's forward prediction is reasonable enough to provide a useful optimization signal; this assumption may fail under extremely severe corruption scenarios.
- Batch-level TTT strategies could be explored to enhance inference efficiency.

## Related Work & Insights
- **vs TTT-MAE**: TTT-MAE utilizes MAE reconstruction as a self-supervised signal, which is restricted to classification tasks, whereas VICT leverages cycle consistency to apply to any dense visual task.
- **vs Painter**: VICT significantly improves generalization over Painter through test-time tuning without requiring additional training data.
- The concept of cycle consistency can be transferred to other image-inpainting-based frameworks (e.g., language-guided image editing).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce test-time training to VICL; the cycle consistency insight is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 tasks $\times$ 15 types of corruptions, spanning both zero-shot and one-shot settings.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Significantly advances robustness research in VICL, with a highly versatile method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ICLR 2026\] Best-of-∞: Asymptotic Performance of Test-Time LLM Ensembling](../../ICLR2026/llm_nlp/best-of-infinity_asymptotic_performance_of_test-time_llm_ensembling.md)
- [\[ACL 2025\] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](../../ACL2025/llm_nlp/testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](../../ACL2025/llm_nlp/chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] Mixtures of In-Context Learners](../../ACL2025/llm_nlp/mixtures_of_in-context_learners.md)

</div>

<!-- RELATED:END -->
