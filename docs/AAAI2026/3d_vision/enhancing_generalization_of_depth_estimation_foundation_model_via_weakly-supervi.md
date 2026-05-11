---
title: >-
  [Paper Note] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization
description: >-
  [AAAI 2026][3D Vision][Monocular Depth Estimation] This paper proposes WeSTAR, a framework that synergistically combines semantics-aware hierarchical depth normalization self-training…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Domain Adaptation"
  - "Weak Supervision"
  - "LoRA"
  - "Self-Training"
date: 2026-05-08
content_hash: d0002036000c2e19
---

# Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization

**Conference**: AAAI 2026
**arXiv**: [2511.14238](https://arxiv.org/abs/2511.14238)
**Code**: None
**Area**: 3D Vision
**Keywords**: Monocular Depth Estimation, Domain Adaptation, Weak Supervision, LoRA, Self-Training

## TL;DR
This paper proposes WeSTAR, a framework that synergistically combines semantics-aware hierarchical depth normalization self-training, sparse pairwise ordinal weak supervision, and LoRA weight regularization to enhance the generalization of depth estimation foundation models (Depth Anything V2) on unseen domains and corrupted data in a parameter-efficient manner, achieving state-of-the-art results on multiple OOD benchmarks.

## Background & Motivation

**Background**: Foundation models such as the Depth Anything series have achieved strong zero-shot monocular depth estimation generalization through large-scale training. However, performance still has room for improvement when facing distribution shifts in downstream tasks (e.g., adverse weather, sensor noise, out-of-distribution scenes).

**Limitations of Prior Work**:
- Self-training on regression tasks suffers from confirmation bias—inaccurate pseudo-labels reinforce model errors.
- When the baseline model is already strong, self-training yields only marginal gains.
- Aggressive adaptation may cause catastrophic forgetting, degrading pretrained generalization knowledge.
- Full fine-tuning is computationally expensive and prone to overfitting.

**Key Challenge**: How can downstream data be leveraged to improve performance while preserving the model's generalization capability?

**Goal**: Design a parameter-efficient and robust adaptation framework that enhances the generalization of depth foundation models under limited target-domain data (unlabeled or weakly labeled).

**Key Insight**: A three-pronged approach—self-training provides dense structural supervision; weak supervision provides sparse but independent ordinal constraints to break confirmation bias; weight regularization anchors pretrained knowledge to prevent forgetting.

**Core Idea**: Jointly employ semantics-aware hierarchical depth normalization self-training, low-cost pairwise ordinal weak supervision, and LoRA regularization to safely adapt depth foundation models to new domains.

## Method

### Overall Architecture
The input consists of a small number of unlabeled (or sparsely annotated with ordinal depth pairs) target-domain RGB images, and the output is the adapted depth estimation model. A teacher–student architecture is adopted: the teacher is updated via EMA, while the student is fine-tuned through LoRA adapters. Weakly augmented images are fed to the teacher to generate pseudo-labels, and strongly augmented images are fed to the student for depth prediction.

### Key Designs

1. **Semantics-Aware Hierarchical Depth Normalization (SA-HDN)**:

    - **Function**: Resolves scale/shift ambiguity between teacher pseudo-labels and student predictions during self-training.
    - **Mechanism**: Traditional hierarchical depth normalization (HDN) partitions images into fixed grids for normalization, ignoring semantic information and potentially splitting the same object. This work uses SAM2 to automatically generate instance masks, constructing a two-level hierarchy: global context $\mathcal{C}_{global}$ (all pixels) and instance context $\mathcal{C}_{ins}^k$ (pixels of the $k$-th object).
    - **Normalization formula**: $\Phi(d_p, \mathcal{C}_p) = \frac{d_p - t(\mathcal{C}_p)}{s(\mathcal{C}_p) + \epsilon}$, where $t$ and $s$ denote the median and MAD, respectively.
    - **Design Motivation**: Semantics-aware partitioning ensures normalization statistics are computed at the object level, preventing depth discontinuities across objects from corrupting the normalization.

2. **Weakly Supervised Adaptation**:

    - **Function**: Uses minimal-cost pairwise ordinal depth annotations to break the confirmation bias of self-training.
    - **Mechanism**: Each weak label $w_j = \{p_{jn}^+, p_{jn}^-, l_{jn}\}$ encodes an ordinal depth relationship (farther/equal/closer) between two pixels. A margin ranking loss enforces the model's predictions to satisfy these constraints.
    - **Sampling Strategy**: Five structured samples are drawn per image, each starting from an anchor point with farther and closer points selected to form transitivity-consistent pairwise constraints.
    - **Design Motivation**: Sparse annotations that are independent of the model provide additional supervision signals capable of correcting local topological errors that pseudo-labels cannot identify.

3. **LoRA Weight Regularization**:

    - **Function**: Constrains the magnitude of model updates to prevent overfitting and catastrophic forgetting.
    - **Mechanism**: Low-rank adapters $\Theta_a + UV$ are injected into the attention layers of the encoder, with only $U$ and $V$ updated. An additional regularization loss $\mathcal{L}_{reg} = \sum \|\frac{\alpha}{r} U_{tk} V_{tk}\|_2^2$ penalizes large deviations from initialization.
    - **Design Motivation**: While LoRA itself constrains the parameter space, severe domain shifts can still introduce confirmation bias. Weight regularization ensures that parameters are updated only when evidence from the target domain is sufficiently strong.

### Loss & Training
Total loss: $\mathcal{L} = \lambda_{st} \mathcal{L}_{st} + \lambda_w \mathcal{L}_{weak} + \lambda_r \mathcal{L}_{reg}$

Weights: $\lambda_{st}=1.0, \lambda_w=0.001, \lambda_r=1.0$. AdamW optimizer with cosine annealing learning rate schedule; EMA decay factor 0.996; LoRA rank=8, alpha=16. Trained on a single RTX 3090 with batch size 4.

## Key Experimental Results

### Main Results
Evaluated on 9 unseen real-world datasets (NYU, KITTI, Sintel, DIODE, NuScenes, DrivingStereo, etc.):

| Method | NYU δ₁↑ | KITTI δ₁↑ | Sintel δ₁↑ | NuScenes δ₁↑ | D-Rainy δ₁↑ |
|------|---------|----------|-----------|-------------|-------------|
| Source (zero-shot) | 97.7 | 93.4 | 74.8 | 74.4 | 84.8 |
| TTAC | 97.7 | 93.4 | 75.0 | 74.4 | 84.5 |
| SGRL | 97.6 | 94.1 | 76.9 | 75.8 | 85.3 |
| **WeSTAR** | **98.2** | **95.1** | **82.2** | **78.1** | **87.4** |

WeSTAR achieves the best results on all datasets, with a δ₁ gain of 7.4% on Sintel (74.8→82.2).

### Ablation Study (Corrupted Datasets)

| Method | NYU-C δ₁↑ | KITTI-C δ₁↑ | Sintel-C δ₁↑ |
|------|-----------|------------|-------------|
| Source | 87.4 | 83.2 | 60.3 |
| iBOT* | 92.1 | 85.6 | 62.7 |
| SGRL | 92.4 | 87.4 | 66.5 |
| **WeSTAR** | **94.6** | **88.7** | **71.8** |

### Key Findings
- The three components exhibit strong synergy: self-training provides global structural alignment, weak supervision corrects local topological errors, and regularization prevents forgetting.
- The advantage is more pronounced on corrupted data—δ₁ on NYU-C improves from 87.4 to 94.6 (+7.2%).
- SA-HDN significantly outperforms traditional HDN, as semantics-aware normalization avoids cross-object depth confusion.
- The weak supervision cost is minimal (only 5 pairwise comparisons per image), yet the performance gain is substantial.

## Highlights & Insights
- **The three-pronged synergistic design** is elegant: dense self-training + sparse weak supervision + regularization each address a distinct problem (structural alignment / topological correction / knowledge retention), with a clear and principled design logic. This "multi-layer defense" paradigm is transferable to other scenarios requiring safe adaptation of pretrained models.
- **Leveraging SAM2 for semantic segmentation to improve depth normalization**: A cross-task borrowing strategy where SAM2's general segmentation capability effectively serves the normalization requirements of depth estimation.
- **Extremely favorable cost-benefit ratio of weak supervision**: A small number of pairwise ordinal annotations suffice to break confirmation bias, making this approach highly practical for real-world deployment.

## Limitations & Future Work
- Weak annotation still requires human labeling of pairwise depth relationships; although low-cost, it cannot be fully automated.
- Validation is limited to relative depth estimation; absolute depth estimation tasks remain untested.
- SAM2 segmentation quality may degrade on severely corrupted images, potentially compromising SA-HDN effectiveness.
- Only Depth Anything V2 and MiDaS are used as backbones; generalizability to a broader range of foundation models remains to be verified.

## Related Work & Insights
- **vs. Depth Anything V2**: WeSTAR uses DAv2 as its backbone and further enhances its generalization through adaptation, serving as a standard downstream adaptation solution for DAv2.
- **vs. TTT++**: TTT++ performs test-time adaptation via contrastive learning and experiences performance degradation on some corrupted data; WeSTAR avoids this issue through regularization.
- **vs. SGRL**: SGRL relies solely on weak supervision without self-training and achieves lower performance than WeSTAR, demonstrating the value of dense self-training.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-component synergistic design is logically coherent, and SA-HDN represents a meaningful improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 9 datasets with corruption benchmarks and diverse baselines—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐ A practical and cost-effective adaptation solution for depth foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](../../CVPR2026/3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[AAAI 2026\] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation](cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)

</div>

<!-- RELATED:END -->
