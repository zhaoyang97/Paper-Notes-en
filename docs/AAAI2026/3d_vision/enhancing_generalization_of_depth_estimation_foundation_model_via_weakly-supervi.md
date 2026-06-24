---
title: >-
  [Paper Note] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization
description: >-
  [AAAI 2026][3D Vision][Monocular Depth Estimation] This work proposes the WeSTAR framework, which synergizes semantic-aware hierarchical normalized self-training, sparse pairwise ordinal weak supervision, and LoRA weight regularization. This parameter-efficient approach enhances the generalization capability of the depth estimation foundation model (Depth Anything V2) on unseen domains and corrupted data, achieving SOTA performance on multiple OOD benchmarks.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Domain Adaptation"
  - "Weakly-Supervised"
  - "LoRA"
  - "Self-Training"
date: 2026-05-08
content_hash: 53fb0ced66afeeca
---

# Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization

**Conference**: AAAI 2026  
**arXiv**: [2511.14238](https://arxiv.org/abs/2511.14238)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Estimation, Domain Adaptation, Weakly-Supervised, LoRA, Self-Training

## TL;DR
This work proposes the WeSTAR framework, which synergizes semantic-aware hierarchical normalized self-training, sparse pairwise ordinal weak supervision, and LoRA weight regularization. This parameter-efficient approach enhances the generalization capability of the depth estimation foundation model (Depth Anything V2) on unseen domains and corrupted data, achieving SOTA performance on multiple OOD benchmarks.

## Background & Motivation

**Background**: Foundation models like the Depth Anything series have achieved excellent zero-shot monocular depth estimation generalization through large-scale data training. However, their performance still has room for improvement when facing downstream task distribution shifts (e.g., adverse weather, sensor noise, out-of-domain scenarios).

**Limitations of Prior Work**:
   - Self-training faces confirmation bias in regression tasks, where inaccurate pseudo-labels reinforce model errors.
   - When the baseline model is already strong, self-training yields only marginal gains.
   - Aggressive adaptation processes can cause catastrophic forgetting, destroying pre-trained generalization knowledge.
   - Full fine-tuning is computationally expensive and prone to overfitting.

**Key Challenge**: How to preserve the generalization capability of the model while utilizing downstream data to improve performance?

**Goal**: Design a parameter-efficient and robust adaptation framework to enhance the generalization of depth foundation models under a small amount of target-domain data (unlabeled or weakly labeled).

**Key Insight**: A three-pronged approach: self-training provides dense structural supervision, weak supervision provides sparse yet independent ordinal constraints to break confirmation bias, and weight regularization anchors pre-trained knowledge to prevent forgetting.

**Core Idea**: Synergize semantic-aware hierarchical normalized self-training, low-cost pairwise ordinal weak supervision, and LoRA regularization to safely adapt depth foundation models to new domains.

## Method

### Overall Architecture
The input consists of a small set of target-domain RGB images (unlabeled or with only sparse pairwise depth ordinal labels), and the output is the adapted depth estimation model. A teacher-student architecture is adopted: the teacher is updated via EMA, and the student is fine-tuned with LoRA adapters. Weakly augmented images are input to the teacher to generate pseudo-labels, while strongly augmented images are input to the student to predict depth.

### Key Designs

1. **Semantic-Aware Hierarchical Depth Normalization (SA-HDN)**:

    - **Function**: Resolve scale and shift ambiguities between teacher pseudo-labels and student predictions in self-training.
    - **Mechanism**: Traditional Hierarchical Depth Normalization (HDN) divides image regions into fixed grids for normalization but ignores semantic information, potentially cutting the same object. This paper leverages the SAM2 segmentation model to automatically generate instance masks, constructing a two-level hierarchy: global context $\mathcal{C}_{global}$ (all pixels) and instance context $\mathcal{C}_{ins}^k$ (pixels of the $k$-th object).
    - Normalization formula: $\Phi(d_p, \mathcal{C}_p) = \frac{d_p - t(\mathcal{C}_p)}{s(\mathcal{C}_p) + \epsilon}$, where $t$ and $s$ are the median and MAD, respectively.
    - **Design Motivation**: Semantic-aware partitioning ensures that normalization statistics are calculated at the object level, preventing cross-object depth discontinuities from causing interference.

2. **Weakly Supervised Adaptation**:

    - **Function**: Break the confirmation bias of self-training using low-cost pairwise ordinal depth annotations.
    - **Mechanism**: Each weak label $w_j = \{p_{jn}^+, p_{jn}^-, l_{jn}\}$ represents the depth ordinal relation between two pixels (further/equal/closer). A margin ranking loss is employed to enforce that the model's predictions satisfy these constraints.
    - Sampling strategy: 5 structural samplings are conducted per image. Each sampling first selects an anchor point, then selects further and closer points, forming pairwise constraints that satisfy transitivity.
    - **Design Motivation**: Sparse but model-independent annotations provide additional supervision signals, correcting local topological errors that cannot be discovered by pseudo-labels.

3. **LoRA Weight Regularization**:

    - **Function**: Constrain the magnitude of model updates to prevent overfitting and catastrophic forgetting.
    - **Mechanism**: Inject low-rank adapters $\Theta_a + UV$ into the attention layers of the encoder, updating only $U$ and $V$. An additional regularization loss $\mathcal{L}_{reg} = \sum \|\frac{\alpha}{r} U_{tk} V_{tk}\|_2^2$ is added to penalize large deviations from initialization.
    - **Design Motivation**: While LoRA restricts the parameter space, it can still suffer from confirmation bias under severe domain shifts. Weight regularization ensures that parameters are updated only when new evidence from the target domain is sufficiently strong.

### Loss & Training
Total loss: $\mathcal{L} = \lambda_{st} \mathcal{L}_{st} + \lambda_w \mathcal{L}_{weak} + \lambda_r \mathcal{L}_{reg}$

Weight settings: $\lambda_{st}=1.0, \lambda_w=0.001, \lambda_r=1.0$. Optimizing with AdamW optimizer, cosine annealing learning rate scheduler, EMA decay factor 0.996, LoRA rank=8, alpha=16. Execution on a single RTX 3090, batch size=4.

## Key Experimental Results

### Main Results
Evaluation on 9 unseen real-world datasets (NYU, KITTI, Sintel, DIODE, NuScenes, DrivingStereo, etc.):

| Method | NYU $\delta_1$↑ | KITTI $\delta_1$↑ | Sintel $\delta_1$↑ | NuScenes $\delta_1$↑ | D-Rainy $\delta_1$↑ |
|------|---------|----------|-----------|-------------|-------------|
| Source (Zero-shot) | 97.7 | 93.4 | 74.8 | 74.4 | 84.8 |
| TTAC | 97.7 | 93.4 | 75.0 | 74.4 | 84.5 |
| SGRL | 97.6 | 94.1 | 76.9 | 75.8 | 85.3 |
| **WeSTAR** | **98.2** | **95.1** | **82.2** | **78.1** | **87.4** |

WeSTAR achieves the best performance across all datasets, with a $\delta_1$ improvement of 7.4% (74.8 $\to$ 82.2) on Sintel.

### Ablation Study (Corrupted Datasets)

| Method | NYU-C $\delta_1$↑ | KITTI-C $\delta_1$↑ | Sintel-C $\delta_1$↑ |
|------|-----------|------------|-------------|
| Source | 87.4 | 83.2 | 60.3 |
| iBOT* | 92.1 | 85.6 | 62.7 |
| SGRL | 92.4 | 87.4 | 66.5 |
| **WeSTAR** | **94.6** | **88.7** | **71.8** |

### Key Findings
- The synergy of the three components is highly effective: self-training provides global structural alignment, weak supervision corrects local topological errors, and weights regularization prevents forgetting.
- The advantage is even more pronounced on corrupted data—$\delta_1$ on NYU-C improves from 87.4 to 94.6 (+7.2%).
- SA-HDN performs significantly better than traditional HDN, as semantic-aware normalization avoids depth confusion across objects.
- The cost of weak supervision is extremely low (requiring only 5 pairs of pairwise comparisons per image), yet it yields substantial gains.

## Highlights & Insights
- **Clever three-pronged synergistic design**: Dense self-training + sparse weak supervision + regularization, with each component addressing a specific issue (structural alignment, topological correction, and knowledge preservation). The clear design logic makes this "multi-layered defense" approach transferable to other scenarios requiring safe adaptation of pre-trained models.
- **Boosting depth normalization with SAM2 semantic segmentation**: This cross-task leveraging approach allows SAM2's general segmentation performance to serve depth estimation normalization effectively.
- **Extremely high cost-benefit ratio of weak supervision**: Breaking confirmation bias requires only a few pairwise ordinal annotations, which is highly practical for real-world deployments.

## Limitations & Future Work
- Weak annotations still require human labeling of pairwise depth relationships. Although the cost is low, it is not fully automated.
- Validated only on relative depth estimation, absolute depth estimation remains untested.
- SAM2's segmentation quality might degrade on extremely corrupted images, affecting SA-HDN performance.
- Only two backbones (Depth Anything V2 and MiDaS) were evaluated in the experiments; generalization to more foundation models needs to be verified.

## Related Work & Insights
- **vs Depth Anything V2**: WeSTAR utilizes DAv2 as a backbone and further improves its generalization capability through adaptation, serving as a standard downstream adaptation solution for DAv2.
- **vs TTT++**: TTT++ performs test-time adaptation based on contrastive learning, but its performance actually degrades on some corrupted data. WeSTAR avoids this issue through weight regularization.
- **vs SGRL**: SGRL only uses weak supervision without self-training, and its performance is lower than WeSTAR, demonstrating the value of dense self-training.

## Rating
- Novelty: ⭐⭐⭐⭐ The clear logic behind the three-component synergistic design is noted, and SA-HDN is a meaningful improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 9 datasets, corrupted benchmarks, and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, and motivation is well-articulated.
- Value: ⭐⭐⭐⭐ A practical adaptation scheme for depth foundation models, offering low cost and high effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation](../../CVPR2026/3d_vision/depth_any_panoramas_a_foundation_model_for_panoramic_depth_estimation.md)
- [\[CVPR 2026\] RoSAMDepth: Robust Self-supervised Depth Estimation Leveraging Segment Anything Model](../../CVPR2026/3d_vision/rosamdepth_robust_self-supervised_depth_estimation_leveraging_segment_anything_m.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](../../ECCV2024/3d_vision/improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](../../CVPR2026/3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
