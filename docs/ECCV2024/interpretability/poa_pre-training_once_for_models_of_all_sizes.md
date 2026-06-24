---
title: >-
  [Paper Note] POA: Pre-training Once for Models of All Sizes
description: >-
  [ECCV 2024][Interpretability][Self-Supervised Pre-training] POA proposes introducing an **elastic student branch** into the self-supervised self-distillation framework. Through parameter sharing and random sub-network sampling, **hundreds of pre-trained models of different sizes can be produced simultaneously with a single pre-training run** (e.g., directly extracting ViT-S/B from ViT-L). Each sub-network achieves SOTA performance on k-NN, linear probing…
tags:
  - "ECCV 2024"
  - "Interpretability"
  - "Self-Supervised Pre-training"
  - "Elastic Networks"
  - "Self-Distillation"
  - "Pre-training Multi-Size Models Once"
  - "Once-for-All"
date: 2026-05-08
content_hash: a5b47861fe3618fa
---

# POA: Pre-training Once for Models of All Sizes

**Conference**: ECCV 2024  
**arXiv**: [2408.01031](https://arxiv.org/abs/2408.01031)  
**Code**: [https://github.com/Qichuzyy/POA](https://github.com/Qichuzyy/POA)  
**Area**: LLM Pre-training  
**Keywords**: Self-Supervised Pre-training, Elastic Networks, Self-Distillation, Pre-training Multi-Size Models Once, Once-for-All

## TL;DR

POA proposes introducing an **elastic student branch** into the self-supervised self-distillation framework. Through parameter sharing and random sub-network sampling, **hundreds of pre-trained models of different sizes can be produced simultaneously with a single pre-training run** (e.g., directly extracting ViT-S/B from ViT-L). Each sub-network achieves SOTA performance on k-NN, linear probing, and downstream tasks.

## Background & Motivation

Self-Supervised Learning (SSL) has achieved outstanding visual representation capabilities in large models. However, actual deployment requires a series of models of different sizes to match various computational and storage limits (such as Gemini Nano/Pro/Ultra). Current practices involve pre-training a large model first, then applying pruning, knowledge distillation, or retraining small models from scratch to adapt to different scenarios—approaches that incur high development costs.

**Key Challenge**: Current SSL methods (e.g., DINO, iBOT) only train one fixed-size model at a time. To obtain $N$ models of different dimensions, they require training $N$ times, leading to a linear growth in computational budgets.

**Key Insight**: Modern network architectures (ViT, Swin, ResNet) naturally possess the characteristic where "smaller models are sub-networks of larger models" (width reduction = fewer attention heads, depth reduction = skipping some blocks). Based on this observation, POA introduces an **elastic student branch** into the teacher-student self-distillation framework, randomly sampling a sub-network at each step for training. Consequently, high-quality sub-networks of any size can be directly extracted from the teacher once pre-training is complete.

**Core Idea**: Unify "pre-training" and "multi-size model generation" into a single training process using elastic sub-network sampling + same-view distillation.

## Method

### Overall Architecture

POA is a **three-branch self-distillation framework**: Teacher, Intact Student, and Elastic Student.

- The input image $x$ yields two augmented views, $x_a$ and $x_b$.
- The Teacher processes $x_a$, while both Students process $x_b$.
- The Intact Student performs cross-view distillation with the Teacher (standard SSL representation learning).
- The Elastic Student simultaneously receives cross-view distillation from the Teacher **and** same-view distillation from the Intact Student.
- The Teacher is updated via EMA, combining parameters of the Intact Student and the Elastic Student.
- In each training step, the Elastic Student randomly samples a sub-network from the intact student.

### Key Designs

1. **Elastic Student**:

    - **Function**: Randomly samples width and depth at each training iteration to construct a sub-network for training.
    - **Mechanism**: Realizes elasticity through parameter slicing. For ViT, elastic width is achieved by reducing the number of attention heads, while elastic depth is implemented by selecting blocks at equal intervals. Define $M+1$ types of elastic widths: $D_i = (N_h - i) \cdot D_h$, and $N+1$ types of elastic depths: $L_i = L_{max} - i$. In total, $(M+1) \times (N+1)$ sub-networks can be generated.
    - **Parameter Extraction**: For the elastic MSA, the input projection weights are formulated as $w_i^{a1} = w^{a1}[:, :D_i] \cdot \alpha_i$, where the scaling factor $\alpha_i = D_{max}/D_i$ compensates for scale changes caused by dimension reduction. The MLP and LN are processed similarly.
    - **Depth Elasticity**: For depth $L_i$, block IDs are selected at equal intervals: $BID_j^{L_i} = \lfloor (L_{max}-1) \cdot j / (L_i - 1) \rfloor$.
    - **Design Motivation**: Small models are naturally sub-networks of large models, and sharing parameters ensures that sub-networks are fully optimized during training. The elastic branch also functions as a **model ensemble** (different sub-networks contribute to the teacher's EMA at each step) and a **training regularizer** (stabilizing training and preventing loss divergence).

2. **Same-view Distillation**:

    - **Function**: Forces the Elastic Student to mimic the output of the Intact Student on the same view.
    - **Mechanism**: This is standard knowledge distillation, transferring the superior representations learned by the Intact Student to the elastic sub-network. The loss is computed as $\mathcal{L}_{ES2}^g = -p_{b1} \log(p_{b2})$.
    - **Design Motivation**: Ablation studies demonstrate that the contribution of same-view distillation to sub-network quality is **far greater** than that of cross-view distillation. This is because cross-view distillation focuses on representation learning, whereas same-view distillation directly extracts knowledge from an existing high-quality representation, making it much more efficient for smaller networks. Removing $\mathcal{L}_{ES2}$ drops the k-NN accuracy of ViT-S by 3.4%.

3. **Multiple Projection Heads (MPH)**:

    - **Function**: Appends multiple projection heads with identical structures but different numbers of prototypes behind the backbone.
    - **Mechanism**: Independently computes the distillation loss for each projection head and averages them: $\mathcal{L} = \frac{1}{H} \sum_{i=1}^H \mathcal{L}_{S_i}$.
    - **Design Motivation**: Since only one sub-network is randomly selected in each step, the training of individual sub-networks is insufficient. MPH introduces diverse semantic spaces (with varying prototype numbers), allowing the teacher to distill knowledge into the sub-networks from multiple perspectives, which yields particularly noticeable improvements for smaller networks.

### Loss & Training

The total loss is a weighted combination of the losses from the Intact Student and the Elastic Student:

$$\mathcal{L}_S = \lambda \mathcal{L}_{IS} + (1-\lambda)(\mathcal{L}_{ES1} + \mathcal{L}_{ES2})$$

where $\mathcal{L}_{IS}$ represents the cross-view distillation of the Intact Student, $\mathcal{L}_{ES1}$ is the cross-view distillation of the Elastic Student, and $\mathcal{L}_{ES2}$ signifies the same-view distillation of the Elastic Student. A multi-crop strategy is also adopted to generate multiple local views, enhancing local-to-global correspondence learning.

Training setup: Unsupervised pre-training on ImageNet-1K, AdamW optimizer, batch size 1600 (32 $\times$ A100), learning rate $lr = 0.004 \times \sqrt{batch\_size / 1024}$, linear warmup of 10 epochs followed by cosine decay. The teacher temperature warmups from 0.04 to 0.07.

## Key Experimental Results

### Main Results

| Dataset | Backbone | Metric | POA | Prev. SOTA | Gain |
|--------|----------|------|-----|-----------|------|
| ImageNet-1K | ViT-L/16 | k-NN | 82.3% | 82.0% (DINOv2) | +0.3% |
| ImageNet-1K | ViT-L/16 | LP | 83.6% | 83.3% (DINOv2) | +0.3% |
| ImageNet-1K | ViT-B/16 (Extracted) | k-NN | 80.9% | 77.1% (iBOT) | +3.8% |
| ImageNet-1K | ViT-S/16 (Extracted) | k-NN | 76.8% | 75.3% (ENT) | +1.5% |
| COCO | ViT-S/16 | AP^b | 50.6 | 49.4 (iBOT) | +1.2 |
| COCO | ViT-B/16 | AP^b | 52.4 | 51.2 (iBOT) | +1.2 |
| ADE20K | ViT-B/16 | mIoU(linear) | 40.3 | 38.3 (iBOT) | +2.0 |

**Key Highlights**: ViT-S/16 and ViT-B/16 require **zero additional pre-training** (directly extracted from the ViT-L teacher). Despite having 0 effective training epochs, they outperform DINO/iBOT models that were individually pre-trained for 3200 epochs.

### Ablation Study

| Configuration | ViT-S k-NN | ViT-B k-NN | ViT-L k-NN | Description |
|------|-----------|-----------|-----------|------|
| MPH + $\mathcal{L}_{ES1}$ + $\mathcal{L}_{ES2}$ | 76.8 | 80.9 | 82.3 | Complete POA |
| MPH + $\mathcal{L}_{ES1}$ (w/o same-view distillation) | 72.8 | 79.1 | 82.1 | ViT-S drops by 4.0% |
| MPH + $\mathcal{L}_{ES2}$ (w/o cross-view distillation) | 75.1 | 80.2 | 82.2 | Same-view distillation is more critical |
| W/o MPH + $\mathcal{L}_{ES1}$ + $\mathcal{L}_{ES2}$ | 76.2 | 80.7 | 82.2 | MPH significantly benefits small networks |

### Key Findings

- **Same-view distillation $\mathcal{L}_{ES2}$ is the most critical component**: It contributes the most to smaller networks, and removing it drops the ViT-S k-NN accuracy by 4.0%.
- **POA significantly outperforms the DINOv2+SEED two-stage paradigm**: ViT-S k-NN 76.8% vs 74.0% (at the same epoch), and POA eliminates the need for an extra distillation stage.
- **Dual role of the elastic branch**: (1) Stabilizing training (in ResNet experiments, the loss diverges to NaN without the elastic branch); (2) Serving as a model ensemble to enhance the teacher's representation quality.
- The performance of the three-branch variants POA-V1/V2 (which remove the Intact Student) drops heavily, validating the necessity of the intact student branch.

## Highlights & Insights

- **"Pre-train once, deploy many times" paradigm**: Obtains 143 high-quality models of different sizes simultaneously from a single ViT-L training session, significantly lowering deployment costs.
- **Elastic parameter extraction + scaling factor $\alpha_i$**: A simple and elegant design that directly slices the weight matrices and multiplies them by dimension ratio factors without introducing extra parameters.
- **Insight from same-view distillation**: Distilling from existing high-quality representations is more effective than directly performing cross-view representation learning, especially for smaller models.
- **High versatility**: Universally applicable across three mainstream architectures: ViT, Swin Transformer, and ResNet.

## Limitations & Future Work

- Currently only validated on ImageNet-1K, without testing on larger datasets (e.g., ImageNet-22K, LAION).
- The elastic design primarily covers width and depth, neglecting other elastic dimensions such as patch size or resolution.
- Not yet extended to multimodal large language models (noted by the authors as a future direction).
- POA-V3 (which adds an elastic teacher) shows slightly better performance but incurs higher computational costs, suggesting room for further optimization.

## Related Work & Insights

- **vs DINO/iBOT/DINOv2**: These methods only train a single fixed-size model at a time. In contrast, POA produces multiple models at once, and the sub-networks exhibit superior performance.
- **vs SEED (Self-supervised distillation)**: SEED is a two-stage approach (pre-training the teacher followed by distillation). POA unifies pre-training and multi-size generation into a single stage, and the sub-networks of POA outperform the distilled models from SEED at the same epoch.
- **vs NAS (e.g., AutoFormer)**: NAS has a massive search space ($>10^{16}$) and requires searching and retraining after the initial training. POA maintains a compact search space (143 configurations) and models are ready-to-use immediately after training.
- **vs OFA (Once-for-All)**: OFA operates under supervision, whereas POA achieves the Once-for-All paradigm under self-supervised learning for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce the Once-for-All concept into self-supervised learning; the three-branch design is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Verified with three backbones across multiple downstream tasks, featuring thorough ablation and comparison with KD results.
- Writing Quality: ⭐⭐⭐⭐ Overall clear structure with complete mathematical derivations, though some notations are heavy.
- Value: ⭐⭐⭐⭐⭐ Highly meaningful for practical deployment, drastically reducing costs by training once for multiple deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](../../CVPR2025/interpretability/scaling_vision_pre-training_to_4k_resolution.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICLR 2026\] Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training](../../ICLR2026/interpretability/learning_to_see_before_seeing_demystifying_llm_visual_priors_from_language_pre-t.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ECCV 2024\] Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)

</div>

<!-- RELATED:END -->
