---
title: >-
  [Paper Note] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning
description: >-
  [ICCV 2025][3D Vision][masked autoencoder] DAP-MAE is proposed to jointly learn multi-domain point cloud data via a Heterogeneous Domain Adapter (HDA) and a Domain Feature Generator (DFG), enabling a single pretraining run to adapt to diverse downstream tasks including object classification, expression recognition, part segmentation, and 3D detection.
tags:
  - ICCV 2025
  - 3D Vision
  - masked autoencoder
  - cross-domain learning
  - point cloud
  - domain adaptation
  - self-supervised learning
date: 2026-05-08
content_hash: d711e8197f026e37
---

# DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning

**Conference**: ICCV 2025
**arXiv**: [2510.21635](https://arxiv.org/abs/2510.21635)
**Code**: [https://github.com/CVI-SZU/DAP-MAE](https://github.com/CVI-SZU/DAP-MAE)
**Area**: 3D Vision / Point Cloud Analysis / Self-Supervised Learning
**Keywords**: masked autoencoder, cross-domain learning, point cloud, domain adaptation, self-supervised learning

## TL;DR

DAP-MAE is proposed to jointly learn multi-domain point cloud data via a Heterogeneous Domain Adapter (HDA) and a Domain Feature Generator (DFG), enabling a single pretraining run to adapt to diverse downstream tasks including object classification, expression recognition, part segmentation, and 3D detection.

## Background & Motivation

**Scarcity of point cloud data**: Compared to 2D data, point cloud datasets across different domains (object, face, and scene domains each with independent datasets) are extremely limited in scale, constraining the performance of supervised learning methods.

**Domain limitations of MAE pretraining**:
   - Existing point cloud MAEs (e.g., Point-MAE, 3DFaceMAE, PiMAE) require separate pretraining on the same domain as the downstream task.
   - This leads to redundant pretraining — each domain demands an independently trained model.
   - Point cloud data across different domains remains underutilized.

**Failure of naive domain mixing**:
   - Directly mixing point clouds from different domains for MAE pretraining causes downstream tasks to treat cross-domain information as noise.
   - Empirically validated: after mixed pretraining on $\mathbb{O}+\mathbb{F}+\mathbb{S}$, ReCon-SMC's expression recognition drops from 87.69% to 87.23%, and object detection drops from 42.7% to 42.5%.

**Core Problem**: How to effectively leverage multi-domain point cloud data for joint pretraining such that the model benefits on downstream tasks across all domains?

## Method

### Overall Architecture

DAP-MAE builds on a Transformer-based MAE architecture and introduces two key components:

- **Heterogeneous Domain Adapter (HDA)**: Operates in *adaptation mode* during pretraining to process each domain separately, and in *fusion mode* during fine-tuning to consolidate multi-domain knowledge.
- **Domain Feature Generator (DFG)**: Extracts domain features via contrastive learning to guide feature adaptation for downstream tasks.

**Pretraining data**: ShapeNet (object domain $\mathbb{O}$), FRGCv2 (face domain $\mathbb{F}$), S3DIS (scene domain $\mathbb{S}$); each point cloud is uniformly sampled to 4096 points.

**Pretraining pipeline**:
1. FPS + KNN partitions point clouds into patches, which are randomly split into visible and masked subsets.
2. PointNet tokenization followed by HDA in adaptation mode.
3. Transformer encoder encodes visible tokens.
4. Decoder reconstructs masked patches (Chamfer Distance loss).
5. DFG extracts domain features (contrastive loss).

### Key Designs

#### 1. Heterogeneous Domain Adapter (HDA)

HDA consists of three parallel MLPs corresponding to domains $\mathbb{O}$, $\mathbb{F}$, and $\mathbb{S}$, where each MLP comprises two FC layers with BN and ReLU.

**Adaptation mode (pretraining)**:
- Given an input point cloud from domain $d$, the corresponding $\text{MLP}_d$ processes the tokens.
- Different domain data passes through different MLP pathways, enabling the encoder to learn domain-specific geometric information under shared parameters.
- Formulation: $\mathcal{T}_{vis} = \text{MLP}_d(\mathcal{T}'_{vis})$

**Fusion mode (fine-tuning)**:
- Parameters of the three MLPs are frozen; two additional MLPs are introduced to generate fusion coefficients.
- All three MLPs process the input tokens simultaneously, with the MLP corresponding to the downstream task domain as the primary branch and the other two as auxiliary branches.
- The outputs are fused via learned coefficients through linear weighting: $\text{To}^{(1)} = \text{To}_d^{(1)} + \sum_{d' \neq d} \alpha_{d'} \text{To}_{d'}^{(1)}$
- Two-stage fusion is applied to the outputs of each FC layer.

#### 2. Domain Feature Generator (DFG)

DFG employs a cross-attention mechanism to extract domain and class features from the encoder output:

- A class token $\hat{c}$ and three domain tokens $\{\hat{o}, \hat{f}, \hat{s}\}$ are maintained.
- During pretraining: the domain token corresponding to the input domain is selected and concatenated with the class token as Q, while encoder features serve as K/V.
- Trained via contrastive loss: same-domain features are encouraged to be similar, while cross-domain features are pushed apart.
- $l(d_i, d_j) = 1 - \cos(d_i, d_j)$ (same domain); $\max(0, \cos(d_i, d_j) - a)$ (different domains).

During fine-tuning, the class feature $c$, domain feature $d$, and point cloud feature $\mathcal{F}$ output by DFG are jointly fed into the downstream task head.

### Loss & Training

**Pretraining loss**:
$$\mathcal{L} = w_1 \mathcal{L}_{rec} + w_2 \mathcal{L}_{con}$$

- Reconstruction loss $\mathcal{L}_{rec}$: Chamfer Distance.
- Contrastive loss $\mathcal{L}_{con}$: domain feature contrastive learning.
- Optimal weights: $w_1 = 100, w_2 = 0.001$ (upweighting reconstruction loss to prevent contrastive loss overfitting).

**Training configuration**: batch size 512, AdamW optimizer, lr=0.0005, cosine schedule, 300 epochs, NVIDIA V100 GPUs.

**Fine-tuning strategy**:
- The three MLP parameters of HDA are frozen; only the fusion coefficient MLPs are trained.
- Different downstream tasks use distinct configurations (lr, epochs, number of points, etc.); details are provided in the experimental section.

## Key Experimental Results

### Main Results

DAP-MAE is evaluated on five downstream tasks: object classification, few-shot learning, part segmentation, expression recognition, and 3D object detection.

#### Object Classification (ScanObjectNN)

| Method | PM | OBJ_BG | OBJ_ONLY | PB_T50_RS |
|------|-----|--------|----------|-----------|
| Point-MAE | PC | 90.02 | 88.29 | 85.18 |
| Point-FEMAE | PC | 95.18 | 93.29 | 90.22 |
| ReCon-SMC (baseline) | PC | 94.15 | 93.12 | 89.73 |
| $\mathbb{O}+\mathbb{F}+\mathbb{S}$ ReCon-SMC | PC | 94.32 | 93.12 | 89.90 |
| **DAP-MAE (Ours)** | **PC** | **95.18** | **93.45** | **90.25** |
| I2P-MAE | PC+I | 94.14 | 91.57 | 90.11 |
| ReCon-full | PC+I+T | 95.18 | 93.63 | 90.63 |

Using only a single modality (PC), DAP-MAE surpasses most cross-modal methods and improves over baseline ReCon-SMC by 1.03% on OBJ_BG.

#### Cross-Domain Pretraining Comparison

| Method | PM | ScanObjectNN | BU3DFE | BOS | AP50 | AP25 |
|------|-----|------|--------|------|------|------|
| ReCon-SMC (same-domain) | PC | 94.15 | 89.13 | 87.69 | 42.7 | 63.8 |
| $\mathbb{O}+\mathbb{F}+\mathbb{S}$ ReCon-SMC | PC | 94.32 | 88.52 | 87.23 | 42.5 | 63.5 |
| **DAP-MAE** | **PC** | **95.18** | **89.83** | **88.45** | **43.2** | **64.0** |

Naive domain mixing in ReCon-SMC leads to degraded performance on expression recognition and detection, whereas DAP-MAE achieves consistent gains across all tasks.

#### 3D Object Detection (ScanNetV2)

| Method | PM | AP50 | AP25 |
|------|-----|------|------|
| VoteNet | PC | 33.5 | 58.6 |
| 3DETR | PC | 37.9 | 62.1 |
| MaskPoint | PC | 40.6 | 63.4 |
| PiMAE | PC+I | 39.4 | 62.6 |
| ACT | PC+I | 42.1 | 63.8 |
| **DAP-MAE** | **PC** | **43.2** | **64.0** |

DAP-MAE with a single modality surpasses all cross-modal methods, outperforming ACT by 1.1% AP50.

### Ablation Study

#### Component Ablation (ScanObjectNN OBJ_BG)

| CD | HDA | DFG | Accuracy |
|----|-----|-----|----------|
| ✗ | ✗ | ✗ | 94.15 |
| ✓ | ✗ | ✗ | 94.32 |
| ✓ | ✓ | ✗ | 94.66 |
| ✓ | ✗ | ✓ | 94.66 |
| ✓ | ✓ | ✓ | **95.18** |

HDA and DFG each independently contribute +0.34%, and their combination yields an additional +0.52% improvement.

#### HDA Fusion Mode Ablation

| Fusion Mode | $\mathbb{O}+\mathbb{F}$ | $\mathbb{O}+\mathbb{F}+\mathbb{S}$ |
|------------|------|--------|
| Adding (direct summation) | 92.59 | 92.94 |
| FC (FC-predicted coefficients) | 94.84 | 93.80 |
| MLP (MLP-predicted coefficients) | 93.80 | **95.18** |

An improper fusion strategy (e.g., direct summation) causes accuracy to plummet from 94.66% to 92.59%, demonstrating that cross-domain data can degrade performance if not handled appropriately.

#### Feature Combination Ablation

| $c$ | $d$ | $\mathcal{F}$ | OBJ_BG |
|---|---|---|--------|
| ✓ | ✗ | ✗ | 93.12 |
| ✗ | ✓ | ✗ | 93.80 |
| ✗ | ✗ | ✓ | 94.49 |
| ✓ | ✓ | ✓ | **95.18** |

The combination of all three features achieves the best result. The point cloud feature $\mathcal{F}$ alone performs best among individual features (94.49%), as the reconstruction loss preserves complete point cloud information.

### Key Findings

1. **Naive domain mixing is unhelpful or even harmful**: Directly pretraining ReCon-SMC on mixed domains degrades performance on expression recognition and detection tasks.
2. **Freezing HDA parameters is critical**: Without freezing the MLP parameters of HDA during fine-tuning, the model tends to overfit in later training stages.
3. **Fusion coefficients evolve dynamically**: The fusion coefficients learned by the MLP follow a rise-then-fall pattern during fine-tuning — the model initially draws more heavily on knowledge from other domains and gradually reduces this reliance over time.
4. **Model efficiency**: DAP-MAE introduces only 0.2M additional parameters and 0.1G additional FLOPs compared to baseline ReCon.
5. **Sensitivity to contrastive loss weight**: $w_1=100, w_2=0.001$ is optimal; an overly large contrastive loss weight leads to overfitting.

## Highlights & Insights

1. **Precise problem formulation**: This work is the first to systematically investigate cross-domain pretraining for point cloud MAEs, clearly demonstrating the limitations of naive domain mixing through controlled experiments.
2. **Elegant dual-mode HDA design**: Domain-separated processing during pretraining avoids interference, while multi-domain fusion during fine-tuning leverages diverse knowledge — a single module addresses two distinct requirements across two training stages.
3. **Guidance role of DFG**: Domain features trained via contrastive learning not only differentiate domains but also guide feature adaptation during fine-tuning.
4. **Single pretraining for multi-task adaptation**: One pretraining run adapts to four different tasks across three distinct domains, substantially reducing pretraining overhead.
5. **Single modality surpasses cross-modal methods**: Without cross-modal distillation from images or text, a pure point cloud model matches or even outperforms cross-modal methods such as ACT and ReCon-full.

## Limitations & Future Work

1. **Domain extensibility**: The current model cannot incorporate new domains without retraining; incremental or continual learning strategies warrant exploration.
2. **Fixed three-domain structure**: The three parallel MLPs in HDA are hard-coded for three domains; extending to $N$ domains requires architectural redesign.
3. **Domain definition granularity**: The current coarse partition into object/face/scene domains may be suboptimal; finer-grained domain definitions could yield further improvements.
4. **Imbalanced pretraining data**: Differences in dataset size (ShapeNet 50k vs. FRGCv2 120k vs. S3DIS) may affect model performance.
5. **Absence of outdoor scenes**: Only indoor scene domains are covered; applicability to outdoor scenarios such as autonomous driving remains unvalidated.

## Related Work & Insights

- **Point-MAE / ReCon**: DAP-MAE is directly built upon these works, using ReCon as the baseline.
- **ACT / I2P-MAE**: Cross-modal distillation methods demonstrate the value of external knowledge; DAP-MAE achieves analogous effects by substituting cross-domain learning for cross-modal learning.
- **PointNet / PointNet++**: Classic point cloud processing backbones are adopted for tokenization.
- Insight: Rather than introducing additional modalities such as images or text, effectively mining and exploiting same-modality data across different domains may represent a more efficient alternative.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of cross-domain pretraining for point cloud MAEs; innovative dual-mode HDA design.
- **Technical Quality**: ⭐⭐⭐⭐ — Method design is well-motivated with thorough ablation experiments.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five downstream tasks, multiple datasets, comprehensive ablations, and visualization analyses.
- **Value**: ⭐⭐⭐⭐ — Single pretraining for multi-task adaptation substantially reduces training costs.
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)

</div>

<!-- RELATED:END -->
