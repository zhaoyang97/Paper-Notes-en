---
title: >-
  [Paper Note] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models
description: >-
  [ICLR 2026][Self-Supervised Learning][task vector] This paper proposes GradFix, a method that constructs a binary mask from gradient signs computed on a minimal number of samples from the target pre-trained model…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "task vector"
  - "model merging"
  - "gradient masking"
  - "foundation models"
  - "transfer learning"
date: 2026-05-08
content_hash: a6e21ad324b22520
---

# Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models

**Conference**: ICLR 2026
**arXiv**: [2510.09658](https://arxiv.org/abs/2510.09658)  
**Code**: [GitHub](https://github.com/fillo-rinaldi/GradFix)  
**Area**: Self-Supervised Learning / Model Merging / Transfer Learning
**Keywords**: task vector, model merging, gradient masking, foundation models, transfer learning

## TL;DR

This paper proposes GradFix, a method that constructs a binary mask from gradient signs computed on a minimal number of samples from the target pre-trained model, and uses it to filter the source model's task vector coordinate-wise, retaining only components aligned with the descent direction of the target loss landscape. Without any fine-tuning, GradFix enables task knowledge transfer across pre-trained models, provides a rigorous first-order descent guarantee in theory, and substantially outperforms both naive transfer and few-shot fine-tuning on vision and language benchmarks.

## Background & Motivation

**Background**: Deep learning has fully embraced the pre-train-then-fine-tune paradigm. Task Arithmetic demonstrates that task vectors (the difference between fine-tuned and pre-trained parameters, $\tau = \theta^{ft} - \theta^{0}$) can be linearly composed to combine multiple capabilities on the same pre-trained model. Model Rebasin research attempts to map different pre-trained models into the same loss basin via permutation alignment, enabling cross-model parameter comparison.

**Limitations of Prior Work**: Foundation models are continuously updated (with more data and improved training strategies), requiring users to re-fine-tune on the new model after each update—previously accumulated fine-tuning results on older models cannot be directly reused. Naively adding a source task vector $\tau_A$ to a new model $\theta_B$ is largely ineffective, yielding performance close to zero-shot level, because the parameter spaces of the two pre-trained models are misaligned and many components of the task vector correspond to harmful directions in the new model's loss landscape, increasing rather than decreasing the loss.

**Key Challenge**: The task vector $\tau_A$ encodes genuinely valuable task adaptation information, but beneficial and harmful components are intermixed. Existing approaches such as TransFusion attempt parameter alignment via permutation matching, but yield limited improvement at high computational cost. The core question is: **how to efficiently identify which coordinates are transferable and which induce negative transfer?**

**Key Insight**: The key insight draws from the distributed optimization literature—SignSGD shows that gradient sign information alone suffices to indicate reliable descent directions. Since the negative gradient $-\mathbf{g}$ on the target model $\theta_B$ indicates the locally optimal update direction, retaining only those coordinates of the task vector whose signs agree with $-\mathbf{g}$ filters out all harmful components. Furthermore, majority voting enables stable estimation of gradient signs from very few labeled samples.

**Core Idea**: Use gradient signs estimated from a small number of samples on the target model to construct a binary mask, retaining sign-aligned components of the source task vector and discarding misaligned ones, achieving cross-model task transfer in a single step without any fine-tuning.

## Method

### Overall Architecture

The input consists of three parts: the source pre-trained model $\theta_A$ and its fine-tuned version $\theta_A^{ft}$, the target pre-trained model $\theta_B$, and a small labeled dataset $\mathcal{D}_s$ for the target task. The pipeline proceeds in four steps: (1) compute the source task vector $\tau_A = \theta_A^{ft} - \theta_A$; (2) compute per-parameter gradient signs on $\theta_B$ for each sample in $\mathcal{D}_s$; (3) aggregate via majority voting to obtain a per-coordinate negative gradient sign estimate $\hat{s}_i$ and construct a binary mask $m_i = \mathbb{1}\{\text{sign}(\tau_{A,i}) = \hat{s}_i\}$; (4) add the masked task vector scaled by $\alpha$ to the target model: $\theta_B^{trans} = \theta_B + \alpha(\mathbf{m} \odot \tau_A)$. The entire process involves no parameter updates or fine-tuning and requires only a single forward-backward pass to compute gradients.

### Key Designs

1. **Gradient-Sign Mask**:

    - Function: Determines coordinate-wise whether each component of the source task vector aligns with the descent direction of the target model, constructing a binary filter mask.
    - Mechanism: Ideally, if the target task vector $\tau_B$ (the "oracle") were available, one could directly filter by sign agreement. However, obtaining $\tau_B$ requires fully fine-tuning the target model—precisely what the method aims to avoid. The key observation is that if the target model takes a single full-batch gradient descent step, its task vector is proportional to $-\mathbf{g}$; thus the negative gradient sign serves as a proxy for the sign of $\tau_B$. The mask is defined as $m_i = \mathbb{1}\{\text{sign}(\tau_{A,i}) = \text{sign}(-g_i)\}$, preserving aligned coordinates and zeroing out misaligned ones.
    - Design Motivation: This design guarantees strict first-order descent. Since each retained coordinate satisfies $g_i \cdot (m_i \tau_{A,i}) = -|g_i||\tau_{A,i}| \leq 0$, the overall inner product $\mathbf{g}^\top \delta^A = -\alpha \sum_i m_i |g_i||\tau_{A,i}| \leq 0$, ensuring that for sufficiently small $\alpha$ the update direction necessarily reduces the target loss.

2. **Majority-Vote Sign Estimator**:

    - Function: Robustly estimates the true gradient sign when only a small number of labeled samples (1–5 per class) are available.
    - Mechanism: For each sample in $\mathcal{D}_s$, per-sample gradient signs are computed independently and treated as individual votes; coordinate-wise majority voting then yields $\hat{s}_i = \text{sign}(-\sum_n \text{sign}(\nabla_\theta \ell(f_{\theta_B}(x_n), y_n)))$. Compared to mean aggregation (averaging gradients before taking the sign), majority voting depends only on sign frequencies rather than magnitudes and is therefore robust to outliers.
    - Design Motivation: Theoretical analysis using Hoeffding's inequality shows that when the per-sample gradient sign is correct with probability $p_i > 1/2$, majority voting recovers the true sign with probability at least $1 - \exp(-2N(p_i - 1/2)^2)$, i.e., accuracy converges exponentially in the number of samples. This explains why the method remains effective with as few as 1–2 samples per class.

3. **Scaling Factor $\alpha$ and Application**:

    - Function: Controls the step size of the transferred update, balancing transfer magnitude against the validity of the local approximation.
    - Mechanism: The final update is $\theta_B^{trans} = \theta_B + \alpha(\mathbf{m} \odot \tau_A)$ with $\alpha \in (0, 1]$ selected via a validation set. Majority voting maintains stable performance across a wide range of $\alpha$ without sudden drops, whereas mean aggregation is prone to sharp degradation at larger $\alpha$ due to sign flips.
    - Design Motivation: The first-order descent guarantee is strictly valid only for "sufficiently small $\alpha$" (beyond which higher-order terms may dominate), but experiments show the effective range is quite wide, reducing the burden of hyperparameter tuning.

### Loss & Training

GradFix involves no training in the conventional sense—it is a single-step transfer method with no parameter update loop. The entire pipeline requires only one forward-backward pass over $|\mathcal{D}_s|$ samples on $\theta_B$ to collect gradient signs, followed by coordinate-wise masking and addition. Compared to few-shot fine-tuning, GradFix requires no learning rate schedule or iteration count, and is one to two orders of magnitude cheaper computationally.

## Key Experimental Results

### Main Results: Vision Cross-Model Transfer (ViT-B/16, 2 samples per class)

| Method | EuroSAT | SVHN | GTSRB | RESISC45 | DTD | Note |
|--------|---------|------|-------|----------|-----|------|
| $\theta_B$ zero-shot | 49.41 | 50.58 | 48.29 | 67.98 | 55.96 | Lower bound |
| $\theta_B + \tau_A$ (direct add) | 49.58 | 50.84 | 49.31 | 67.87 | 56.27 | Essentially ineffective |
| TransFusion | 50.12 | 53.26 | 50.24 | 67.99 | 56.70 | Permutation alignment, marginal gain |
| Few-shot fine-tuning $\theta_B^{opt}$ | 59.49 | 62.01 | 61.70 | 71.20 | 57.00 | Fine-tuning with equal data budget |
| **GradFix** $\theta_B + \delta^A$ | **65.07** | **70.19** | **64.33** | **71.42** | **58.51** | Outperforms few-shot fine-tuning across all tasks |
| Oracle $\theta_B + \delta^\star$ | 95.06 | 92.04 | 82.92 | 87.06 | 71.44 | Upper bound (requires known $\tau_B$) |
| $\theta_B$ full fine-tuning | 98.70 | 97.45 | 98.65 | 95.66 | 83.19 | Theoretical upper bound |

### NLP Cross-Model Transfer (T5v1.1 → FLAN-T5, 50 samples per class)

| Method | SNLI | MNLI | RTE | QNLI | SCITAIL | AVG |
|--------|------|------|-----|------|---------|-----|
| $\theta_B$ zero-shot | 34.24 | 35.21 | 47.20 | 50.54 | 50.38 | 43.51 |
| $\theta_B + \tau_A$ | 31.61 | 30.75 | 47.36 | 50.52 | 50.46 | 42.12 |
| Few-shot fine-tuning | 35.09 | 26.05 | 47.29 | 51.45 | 51.78 | 42.33 |
| **GradFix** | **68.06** | **49.68** | **54.25** | **60.50** | **59.89** | **58.48** |
| $\theta_B$ full fine-tuning | 88.20 | 86.30 | 84.40 | 92.79 | 95.32 | 89.40 |

### Ablation Study: Masking Strategies (ViT-B/16, 1 sample per class)

| Masking Strategy | EuroSAT | GTSRB | SVHN | AVG | Note |
|-----------------|---------|-------|------|-----|------|
| Sign agreement (GradFix) | 61.94 | 60.89 | 71.07 | 64.45 | Retain only aligned components |
| Sign forcing | 61.32 | 60.91 | 70.52 | 64.18 | Flip misaligned signs, amplifies noise |
| Magnitude-weighted | 49.51 | 49.20 | 50.71 | 54.70 | Joint magnitude matching, over-filters |
| Random mask | 49.49 | 48.41 | 50.54 | 54.50 | No-signal baseline |

### Key Findings

- **Naive transfer completely fails**: Directly adding $\tau_A$ to $\theta_B$ performs comparably to zero-shot, confirming that parameter space misalignment leads to negative transfer.
- **GradFix consistently surpasses few-shot fine-tuning**: Under the same labeling budget, GradFix outperforms few-shot fine-tuning $\theta_B^{opt}$ on both ViT-B/16 and ViT-L/14 with lower variance and greater robustness.
- **Larger gains in NLP**: In the T5v1.1→FLAN-T5 transfer setting, GradFix shows even more pronounced improvement over naive transfer, suggesting that sign-based filtering is more valuable when the source–target pre-training gap is larger.
- **Magnitude information is not transferable**: Even in the Oracle setting, masking strategies that exploit magnitudes fall far short of pure sign masking. This reveals a key insight: direction information is transferable across pre-trained models, while magnitude is highly dependent on each model's own loss geometry.
- **Majority voting outperforms mean aggregation**: Performance remains consistently stable over a wide range of $\alpha$ and is not susceptible to sign flips caused by individual outlier gradients.
- **Compatible with model merging**: GradFix can be combined with Task Arithmetic and TIES-Merging. In the multi-task setting, Merge-then-Mask achieves the best result (AVG 66.02); in the multi-source setting, Mask-then-Merge is optimal (AVG 67.41).

## Highlights & Insights

- **Tight theory–practice alignment**: The first-order descent guarantee follows from a Taylor expansion in just a few lines, yet it precisely predicts the experimental behavior—all retained coordinates indeed reduce the loss. This style of "clean theory + strong empirical validation" is instructive.
- **"Direction transfers, magnitude does not"**: This is among the paper's most profound findings. It suggests that different pre-trained models encode similar information about *which direction* a task should move in parameter space, while *how far* to move is highly local to each model's own loss basin. This insight can guide all future parameter-manipulation-based model merging and transfer methods.
- **Zero-fine-tuning design is highly practical**: The method requires no parameter update loop and completes in a single forward-backward pass, making it extremely valuable in compute-constrained or rapid-adaptation scenarios—such as fast task transfer following a foundation model update in edge deployment settings.

## Limitations & Future Work

- **Filtering only, no correction**: GradFix discards all sign-misaligned coordinates, but some of these may contain valuable task information that simply needs adjustment in magnitude or direction. Future work could explore soft masking or directional correction of misaligned coordinates rather than simple zeroing.
- **Requires identical architectures**: Source and target models must share the same architecture (with a one-to-one correspondence in parameter count and structure), limiting applicability to cross-architecture transfer.
- **$\alpha$ still requires a validation set**: The scaling factor is the sole hyperparameter; while majority voting reduces its sensitivity, a validation set is still needed for selection.
- **Large gap to Oracle remains**: GradFix (AVG ~65) still trails the Oracle (AVG ~86) by more than 20 percentage points, indicating that few-shot gradient sign estimation remains a coarse approximation; improved sign estimation strategies could further close this gap.
- **Interaction with parameter-efficient fine-tuning unexplored**: Task vectors are derived from full-parameter fine-tuning; whether low-rank task vectors from LoRA-style fine-tuning transfer differently remains an open question.

## Related Work & Insights

- **vs. TransFusion**: TransFusion performs parameter space alignment via permutation matching (rebasin), which is computationally heavier but yields only marginal improvement over direct addition. GradFix bypasses parameter alignment entirely and instead performs local filtering in the target model's loss landscape, achieving both greater simplicity and stronger performance.
- **vs. TIES-Merging**: TIES resolves parameter conflicts among multiple task vectors via sign consistency, but its "signs" are derived from the task vectors themselves. GradFix's "signs" come from the target model's gradients—a fundamental distinction: the former resolves inter-task conflicts, while the latter addresses cross-model alignment.
- **vs. SignSGD**: GradFix can be viewed as a creative application of the SignSGD idea to the domain of model merging and transfer. SignSGD compresses gradient communication using sign information; GradFix filters task vectors using sign information. Both exploit the shared insight that *signs are more robust than magnitudes*.

## Rating

- Novelty: ⭐⭐⭐⭐ The gradient-sign masking idea is elegant and concise, though it is essentially a transfer of the SignSGD concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two domains (vision + language), three settings (single-task, multi-task, multi-source), four masking strategy comparisons, and complete theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation–theory–experiment chain is clear; the transition from Oracle derivation to the practical method is natural and well-motivated.
- Value: ⭐⭐⭐⭐ Highly practical and lightweight, though applicability is limited to same-architecture transfer and a notable gap to the Oracle remains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](../../CVPR2026/self_supervised/shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)
- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)

</div>

<!-- RELATED:END -->
