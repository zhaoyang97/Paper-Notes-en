---
title: >-
  [Paper Note] Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models
description: >-
  [ICML 2026][Segmentation][Vision Foundation Models] GPUA treats VLMs (e.g., CLIP), which possess strong semantics but lack local precision, and VFMs (e.g., DINOv3), which have fine-grained details but lack semantics…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Vision Foundation Models"
  - "VFM-VLM Fusion"
  - "Cross-lingual Alignment"
  - "Orthogonal Procrustes"
  - "Sinkhorn"
  - "Hubness"
date: 2026-05-08
content_hash: 3bec163144b5801a
---

# Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2606.04385](https://arxiv.org/abs/2606.04385)  
**Code**: https://github.com/Yuteam14/GPUA (Available)  
**Area**: Self-Supervised Learning / Representation Learning / Multi-modal Alignment  
**Keywords**: Vision Foundation Models, VFM-VLM Fusion, Cross-lingual Alignment, Orthogonal Procrustes, Sinkhorn, Hubness

## TL;DR
GPUA treats VLMs (e.g., CLIP), which possess strong semantics but lack local precision, and VFMs (e.g., DINOv3), which have fine-grained details but lack semantics, as two different "visual languages." It uses Optimal Transport to mine soft correspondences and solves an Orthogonal Procrustes problem to learn a geometry-preserving linear mapping that translates VFMs into the VLM space. The process is entirely unsupervised, requires no updates to pre-trained parameters, and achieves an average gain of 11.8% in zero-shot classification.

## Background & Motivation
**Background**: The two main camps of computer vision foundation models have distinct strengths: **VLMs** like CLIP use large-scale image-text contrastive pre-training to provide a language-aligned semantic space, naturally supporting open-vocabulary recognition; **VFMs** like DINOv3 follow a self-supervised path, producing patch-level features with clear structures and strong local discriminative power, but they lack language anchors. Utilizing both is a consensus direction in the community, typical in open-vocabulary segmentation pipelines where CLIP's semantics are combined with DINO's fine-grained features.

**Limitations of Prior Work**: Existing fusion schemes generally suffer from two major drawbacks: (1) **Dependence on deep access**—they require extracting intermediate features or issuing dense mask queries, which is infeasible for closed-source models, APIs, or restricted deployments; (2) **Task/structure coupling**—fusion mechanisms are designed around pixel-level prediction, mask generation, or spatial post-processing, making them difficult to transfer to global semantic decision tasks like image-level zero-shot classification.

**Key Challenge**: To make heterogeneous foundation models "directly compatible at the representation layer," one must find a **task-agnostic, feature-only, parameter-frozen** alignment mechanism. However, representation spaces across models differ in dimensionality, scale, and geometry. Conventional alignment either relies on supervision to learn a projection or uses alternating optimization, which is highly sensitive to initialization and prone to collapsing into trivial solutions.

**Goal**: (1) Formally define "translating VFM features into the VLM semantic space"; (2) Solve this mapping in an entirely unsupervised and parameter-frozen manner; (3) Suppress modality gap and hubness issues common in VLM spaces to ensure the translated features are effective for zero-shot classification and segmentation.

**Key Insight**: The authors analogize this problem to **cross-lingual word embedding alignment** in NLP—word vector spaces of different languages can be aligned using **Orthogonal Procrustes** to solve a geometry-preserving linear mapping (Lample et al., 2018; Artetxe et al., 2018), based on the validated "isomorphism hypothesis." In vision, VFM features are treated as word vectors of a "visual language," while the VLM text side provides the "target language dictionary." As long as reliable "pseudo-dictionary" correspondences can be mined, the optimal mapping can be solved directly via SVD.

**Core Idea**: The VFM-VLM alignment is split into two stages: first, use **Sinkhorn-style Optimal Transport** under the dual constraints of VLM semantic structure and VFM geometric structure to mine a soft correspondence matrix $P$; second, feed $P$ into Orthogonal Procrustes to solve for a closed-form mapping $W$, followed by fine-tuning $W$ with a hubness-aware ranking loss to eliminate hyper-central prototypes.

## Method

### Overall Architecture
Input: VFM visual features $Z \in \mathbb{R}^{N \times d_v}$, VLM visual features $X \in \mathbb{R}^{N \times d_t}$, and VLM text prototypes $Y \in \mathbb{R}^{K \times d_t}$ (obtained by passing prompts like "a photo of {class}" through the text encoder, where $K$ is the number of classes).

Phase 1 (**UCM**, Unsupervised Correspondence Mining): Alternately update the soft correspondence matrix $P \in \mathbb{R}^{N \times K}_+$ and latent VFM centers $C \in \mathbb{R}^{K \times d_v}$ such that $P$ reflects both "VLM semantic scores" and "VFM geometric clusters." This is solved as an entropy-regularized transport problem using Sinkhorn.

Phase 2 (**GPA**, Geometry-Preserving Alignment): Use $P$ from Phase 1 to solve for an orthogonal mapping $W_0=UV^\top$ via a closed-form SVD solution, ensuring $ZW$ is close to the prototype mixture $PY$. Then, fine-tune $W_0$ into $W^*$ using a topology-aware hubness suppression loss $\mathcal{L}_{\text{THS}}$.

Inference: Image $\to$ VFM produces CLS / patch tokens $\to$ Apply $W^*$ to map to the VLM semantic space $\to$ Calculate cosine similarity with text prototypes for classification / segmentation. **Pre-trained VFMs and VLMs remain frozen; the pipeline only learns a lightweight linear transformation.**

### Key Designs

1. **UCM: Dual-Source Soft Correspondence Mining (Geometry + Semantics)**:
    - **Function**: Mines a reliable soft assignment matrix $P$ for "VFM samples $\to$ VLM text prototypes" in an unlabeled setting, serving as the "pseudo-dictionary" for Procrustes.
    - **Mechanism**: The authors observed that assigning each image to the nearest text prototype in the VLM space $\min_P \|X-PY\|_F^2$ is equivalent to a K-means assignment step with fixed text centers, which is noisy under domain shift. Thus, latent centers $C$ are introduced on the VFM side to share $P$: $\min_{P,C} (1-\lambda)\|Z-PC\|_F^2 + \lambda\|X-PY\|_F^2$. Both sides are **locked by $P$**, requiring geometric clustering and semantic assignment to coincide. Relaxing $P$ to a non-negative matrix $\Pi(r,c)$ with marginal constraints and adding entropy regularization $-\varepsilon H(P)$ transforms the $P$ sub-problem into $\max_{P\in\Pi(r,c)}\langle P,(1-\lambda)ZC^\top+\lambda XY^\top\rangle+\varepsilon H(P)$, solvable via **Sinkhorn-Knopp iterations**. The $C$ sub-problem has a closed-form solution $C_k=\sum_i P_{ik}Z_i / \sum_i P_{ik}$.
    - **Design Motivation**: (a) Dual sources are complementary—VLM provides semantic priors on "which class," while VFM provides geometric priors on "which samples belong to the same cluster"; (b) The Sinkhorn formulation naturally produces dense soft assignments, avoiding oscillations on boundary samples common in hard assignments; (c) It is more stable than "alternating $P$ and $W$" by fully solving for $P$ first, **decoupling the initialization sensitivity** of alternating optimization.

2. **GPA: Orthogonal Procrustes + Topology-aware Hubness Suppression**:
    - **Function**: Uses $P$ from UCM to learn a **geometry-preserving** linear translation $W$ from VFM to VLM while eliminating hubness.
    - **Mechanism**: Solving $\min_W \|ZW - PY\|_F^2$ s.t. $W^\top W=I$ uses $P$ as a "soft dictionary." This is the classic Orthogonal Procrustes problem with a closed-form solution $W_0=UV^\top$, where $U\Sigma V^\top=\text{SVD}(Z^\top PY)$. The orthogonal constraint ensures $W$ is an approximate isometry, preventing collapse or shear——**preserving VFM neighborhood geometry in the VLM space**. However, the VLM space inherently suffers from hubness: a few prototypes become nearest neighbors to many samples, distorting local discriminability. Thus, $W_0$ is fine-tuned via gradient optimization over a topology-aware ranking loss $\mathcal{L}_{\text{THS}}=\frac{1}{NK}\sum_i\sum_{c\in\mathcal{N}_i^K}(d_i^++m_{i,c}^{\text{base}}+h_c-d_{i,c})_+$, where $d_i^+$ is the distance to the correct prototype, $d_{i,c}$ to a competing prototype, $m_{i,c}^{\text{base}}=(1-y_{\ell_i}^\top y_c)/s$ is the semantic margin, and $h_c=\frac{1}{N}\sum_i \mathbb{I}(c\in\mathcal{N}_i^K)$ is the hubness penalty.
    - **Design Motivation**: The orthogonal constraint + closed-form solution provides a high-quality starting point (**the core of the method's stability**). The hubness loss addresses "hyper-central prototypes" that persist after orthogonal mapping. Using $K$-nearest neighbors in a ranking format provides denser and stronger signals than direct hubness scalar reduction.

3. **Task-Agnostic Interface: Feature-level Plug-and-Play**:
    - **Function**: Applies the same alignment framework to both image-level zero-shot classification and patch-level open-vocabulary segmentation.
    - **Mechanism**: For zero-shot classification, the VFM CLS token is passed through the UCM+GPA pipeline. For open-vocabulary segmentation, it operates at the patch level, translating DINOv3 patch features into the semantic spaces of MaskCLIP/SCLIP/SC-CLIP as a **plugin**, without modifying their heads, losses, or training. GPUA enhances both the discriminative power of VLM global representations and the fine-grained boundaries of patch-level segmentation.
    - **Design Motivation**: Requires only "feature access"—compatible with closed-source models, APIs, and restricted deployments. Decoupling alignment from the task head allows GPUA to be orthogonally combined with any downstream framework.

### Loss & Training
$$\mathcal{L}=\underbrace{(1-\lambda)\|Z-PC\|_F^2+\lambda\|X-PY\|_F^2-\varepsilon H(P)}_{\text{Stage 1: UCM}}+\underbrace{\|ZW-PY\|_F^2 \text{ s.t. } W^\top W=I + \eta\mathcal{L}_{\text{THS}}}_{\text{Stage 2: GPA}}$$
Stage 1 alternates between Sinkhorn and closed-form barycenters. Stage 2 solves for $W_0$ via SVD and refines it with a small learning rate. GPUA uses the full training set; GPUA* uses only 16 samples per class.

## Key Experimental Results

### Main Results
Zero-shot image classification (11 datasets, CLIP protocol, DINOv3 as VFM):

| Method | Flowers | Pets | Caltech | FGVC | EuroSAT | UCF101 | DTD | Food | Cars | SUN | ImageNet | Avg |
|------|---------|------|---------|------|---------|--------|-----|------|------|-----|----------|-----|
| CLIP | 70.7 | 89.1 | 93.2 | 24.7 | 48.3 | 67.5 | 43.5 | 85.9 | 65.6 | 62.5 | 66.6 | 65.2 |
| ZLaP | 73.5 | 87.1 | 93.1 | 25.4 | 55.6 | 71.5 | 48.6 | 86.9 | 65.6 | 67.4 | 70.0 | 67.7 |
| DPE | 75.1 | 91.1 | 94.8 | 29.0 | 55.8 | 70.4 | 54.2 | 86.2 | 67.3 | 70.1 | 71.9 | 69.6 |
| StatA | 75.2 | 92.4 | 94.2 | 24.7 | 67.3 | 73.5 | 48.4 | 87.1 | 68.0 | 68.7 | 69.9 | 69.9 |
| COSMIC | 82.1 | 94.2 | 96.8 | 31.4 | 58.8 | 76.2 | 58.2 | 86.6 | 71.3 | 72.3 | 78.2 | 73.3 |
| GPUA* (16-shot) | 86.6 | 94.5 | 98.1 | 34.7 | **80.3** | 78.4 | 56.7 | 87.9 | 77.4 | 72.6 | 74.3 | 76.5 |
| **GPUA (full)** | **83.8** | **95.0** | **95.3** | **33.8** | **88.2** | **80.4** | **58.5** | **89.5** | **77.7** | **74.2** | **75.4** | **77.4** |
| Gain vs CLIP | +14.0 | +6.0 | +3.8 | +5.5 | **+34.9** | +13.2 | +14.7 | +3.0 | +11.7 | +11.7 | +10.5 | **+11.8** |

GPUA improves the average by 11.8 points. Gains are most significant in datasets where VLMs typically underperform, such as EuroSAT (Remote Sensing, +34.9) and Flowers (Fine-grained, +14.0), indicating that **VFM geometric details are effectively integrated into the VLM semantic space**. GPUA* with only 16-shot also outperforms all baselines, showing excellent data efficiency.

### Ablation Study
| Configuration | Key Finding | Description |
|------|---------|------|
| Full GPUA | Complete version | UCM + GPA + THS |
| w/o VFM term ($\lambda=1$) | Degenerates to LFA-style | Validates necessity of geometric priors |
| w/o VLM term ($\lambda=0$) | Loses semantic alignment | Validates necessity of semantic priors |
| Direct SVD w/o THS | Severe hubness | $\mathcal{L}_{\text{THS}}$ is effective |
| w/o Orthogonal constraint | Training collapse / Geometric distortion | Orthogonal constraint preserves geometry |
| Replacing VFM | DINOv3 is stronger | Stronger VFMs lead to higher alignment gains |

### Key Findings
- Geometric signals from the VFM side (the $Z-PC$ term) are crucial; relying solely on VLM self-scoring for correspondence mining leads to high noise under domain shift.
- The orthogonal constraint provides stability—without it, the SVD path degenerates into ordinary least squares, over-fitting to pseudo-label noise.
- t-SNE visualizations (Pets dataset) show a clear modality gap in the original CLIP space; after GPUA, visual clusters are accurately pulled toward corresponding semantic anchors while preserving intra-class structures.

## Highlights & Insights
- The analogy of **cross-lingual alignment $\to$ cross-model alignment** is elegant: treating "VFM features as visual language" is a concise and powerful inductive bias, allowing mature NLP tools like Orthogonal Procrustes, Sinkhorn, and hubness losses to be directly applied.
- The **two-stage decoupling (P then W)** vs the classic alternating approach—most unsupervised alignment works (like LFA or MUSE) use alternating optimization, which is sensitive to initialization. Solving $P$ thoroughly before $W$ is a practical engineering upgrade for stability.
- The combination of **task-agnostic, frozen parameters, and learning a single matrix** makes GPUA a "zero-cost plugin." It works even with closed-source APIs (CLIP / Commercial VLMs) as long as features are accessible, which is much more practical for industry deployment than end-to-end fine-tuning of 1B+ models.
- Incorporating the hubness frequency $h_c$ directly into the margin in the **THS loss** is a simple yet effective trick that could be generalized to any prototype-based classification task.

## Limitations & Future Work
- Inference relies on a **single linear mapping** $W$, which might be insufficient for highly non-linear modality gaps; however, the authors demonstrate its efficacy for the CLIP+DINOv3 combination.
- UCM currently uses datasets like ImageNet as a "training pool" (effectively an unlabeled calibration set) and is not strictly zero-shot, as it requires sampling unlabeled images from the target domain. This introduces a calibration cost compared to pure CLIP zero-shot.
- The orthogonal constraint implies information loss when $d_v$ and $d_t$ are unequal (e.g., 4096-dim DINOv3 projected into a lower-dim VLM text space).
- The experiments focus primarily on CLIP+DINOv3; the extent to which this extends to heterogeneous models like SigLIP, EVA, or SAM-style models requires further validation.

## Related Work & Insights
- **vs LFA / Ouali et al. 2023**: LFA also uses unsupervised cross-lingual alignment ideas but alternates $P/W$ optimization; this work uses a two-stage approach and incorporates VFM geometric priors in UCM for better stability and gains.
- **vs Multi-model Fusion Segmentation Pipelines**: Those methods embed VFMs as patch refiners, creating a deep coupling with the segmentation task; GPUA is a task-agnostic feature-level translator that acts as a plugin.
- **vs Cross-lingual word embedding alignment**: Direct adaptation of the NLP framework to vision, with the core innovation being the explicit inclusion of VFM structural information in correspondence mining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[ICML 2026\] Unsupervised Hierarchical Skill Discovery](unsupervised_hierarchical_skill_discovery.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](../../CVPR2026/segmentation/occsam_bench_occlusion_robustness_segmentation.md)

</div>

<!-- RELATED:END -->
