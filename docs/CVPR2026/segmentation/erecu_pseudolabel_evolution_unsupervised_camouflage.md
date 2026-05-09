---
title: >-
  [Paper Note] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection
description: >-
  [CVPR 2026][Segmentation][unsupervised camouflaged object detection] EReCu is a unified framework built upon a DINO teacher-student architecture that employs Multi-cue Native Perception (MNP) to extract texture and semantic priors from raw images, guiding Pseudo-label Evolution Fusion (PEF) for global pseudo-label evolution, and Local Pseudo-label Refinement (LPR) for boundary detail recovery. It is the first framework to unify the two dominant UCOD paradigms—pseudo-label guidance and feature learning—achieving state-of-the-art performance across four COD benchmarks.
tags:
  - CVPR 2026
  - Segmentation
  - unsupervised camouflaged object detection
  - pseudo-label evolution
  - multi-cue perception
  - teacher-student
  - spectral attention fusion
date: 2026-05-08
content_hash: 087453f5503e2126
---

# EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection

**Conference**: CVPR 2026
**arXiv**: [2603.11521](https://arxiv.org/abs/2603.11521)
**Code**: [GitHub](https://github.com/JSLiam94/EReCu)
**Area**: Unsupervised Camouflaged Object Detection / Image Segmentation
**Keywords**: unsupervised camouflaged object detection, pseudo-label evolution, multi-cue perception, teacher-student, spectral attention fusion

## TL;DR

EReCu is a unified framework built upon a DINO teacher-student architecture that employs Multi-cue Native Perception (MNP) to extract texture and semantic priors from raw images, guiding Pseudo-label Evolution Fusion (PEF) for global pseudo-label evolution, and Local Pseudo-label Refinement (LPR) for boundary detail recovery. It is the first framework to unify the two dominant UCOD paradigms—pseudo-label guidance and feature learning—achieving state-of-the-art performance across four COD benchmarks.

## Background & Motivation

**Background**: Camouflaged Object Detection (COD) is highly challenging due to the visual similarity between targets and backgrounds. Fully supervised methods rely on expensive pixel-level annotations, limiting dataset scale and ecological diversity. Unsupervised COD (UCOD) currently comprises two paradigms: pseudo-label guidance and feature learning.

**Limitations of Prior Work**:

1. Pseudo-label guidance methods (e.g., UCOS-DA, UCOD-DPL) over-rely on high-dimensional embeddings while neglecting native image cues, leading to boundary overflow and semantic drift.
2. Feature learning methods (e.g., SdalsNet, EASE) lack explicit pseudo-label supervision, resulting in blurry boundaries and loss of fine details.
3. Both paradigms suffer from critical shortcomings and have not been unified—semantic reliability and texture fidelity are optimized in isolation.

**Key Challenge**: Pseudo-label guidance addresses "where" but yields imprecise boundaries; feature learning addresses "what it looks like" but suffers from localization ambiguity. The two are complementary, yet no existing method exploits both simultaneously.

**Goal**: To construct a unified UCOD framework in which pseudo-label reliability and feature fidelity co-evolve through a mutual feedback loop.

**Key Insight**: Extract Multi-cue Native Perception (texture + semantics) from raw images to jointly constrain both the semantic evolution and local detail refinement of pseudo-labels.

**Core Idea**: Drive both global pseudo-label evolution and local refinement using native image cues, achieving semantic–perceptual co-evolution.

## Method

### Overall Architecture

The framework adopts a DINO-based teacher-student architecture. The teacher is updated via EMA (momentum 0.99), while the student iteratively learns to refine segmentation masks. The pipeline proceeds as follows: input image → MNP extracts multi-cue features $F_{\text{MNP}}$ and a quality metric $S_{\text{mc}}$ from raw images → PEF leverages multi-cue signals to guide global pseudo-label evolution (EPL for teacher-student interaction denoising; STAF for multi-layer spectral attention fusion) → LPR selects high-confidence regions from teacher attention heads to generate local pseudo-labels for detail recovery → output segmentation mask. MNP simultaneously provides constraint signals to both PEF and LPR.

### Key Designs

1. **Multi-cue Native Perception (MNP)**

    - **Function**: Extracts low-level texture and mid-level semantic features from raw images to construct a multi-cue quality metric.
    - **Mechanism**: LBP and DoG are used to extract texture features $F_{\text{text}}$; a frozen ResNet-18 extracts semantic features $F_{\text{sem}}$; these are concatenated as $F_{\text{MNP}} = \mathcal{C}(F_{\text{text}}, F_{\text{sem}})$. The image is divided into three regions according to the mask—interior $R_i$, boundary $R_s$, and exterior $R_o$—and three groups of modified cosine similarity scores are computed (via random $K \times K$ patch sampling over $N$ rounds): $S_{\text{mc}} = (D_{\text{io}} + D_{\text{is}} + S_{\text{so}}) / 3$, with loss $\mathcal{L}_{\text{MNP}} = 1 - S_{\text{mc}}$.
    - **Design Motivation**: Even under heavy camouflage, subtle yet discriminative texture variations remain in the raw image. Random patch sampling handles the irregular geometry of segmentation regions.

2. **Pseudo-label Evolution Fusion (PEF = EPL + STAF)**

    - **EPL (Evolutionary Pseudo-label Learning)**: Student shallow features are enhanced via depthwise separable convolution (DSC) to improve spatial detail, yielding $M_s^{\text{dsc}}$. Pseudo-masks $M_s^p / M_t^p$ are obtained from student and teacher branches via semantic pooling. Iterative optimization: $M_s^{\text{dsc}(r+1)} = \arg\min[\mathcal{L}_D(M_s^{\text{dsc}}, M_s^p) + \mathcal{L}_D(M_s^{\text{dsc}}, M_t^p) + \mathcal{L}_{\text{MNP}}]$, jointly driven by Dice loss and multi-cue constraints.
    - **STAF (Spectral Tensor Attention Fusion)**: Attention maps from three student layer levels (1/3, 2/3, final layer) are stacked into a third-order tensor $\mathcal{T}_s \in \mathbb{R}^{3 \times C \times HW}$. Tucker decomposition and truncated SVD extract the top $t$ spectral components, yielding a low-rank approximation $A_s^{\text{fu}} = P_t \Sigma_t Q_t^\top$, which is then linearly projected and passed through Sigmoid to produce the fused prediction $M_s^{\text{fu}}$. Complexity: $\mathcal{O}(r^2 d)$.
    - **Design Motivation**: EPL enables interaction and denoising between shallow detail and deep semantics; STAF suppresses attention noise while preserving semantic and structural information.

3. **Local Pseudo-label Refinement (LPR = TAS + LPG)**

    - **TAS (Target-Aware Attention Selection)**: The focus entropy $E_k$ of each teacher attention head is computed; heads satisfying $E_k < \tau_e$ and $S_{\text{mc}}(\hat{A}_k, F_{\text{MNP}}) > \tau_s$ are selected (both thresholds are learnable, initialized at 0.5).
    - **LPG (Local Pseudo-label Generation)**: For selected heads, an adaptive threshold $\tau_k = \mu_{A_k} + \alpha \cdot \sigma_{A_k}$ ($\alpha > 1$, learnable) extracts high-confidence regions to generate local pseudo-labels $P_k$. Dice + CE losses guide $M_s^{\text{fu}}$ toward refined boundaries.
    - **Design Motivation**: Global pseudo-labels capture central regions but miss boundary and texture details; the spatial diversity across attention heads, each focusing on different regions, can be exploited for local correction.

### Loss & Training

The total loss comprises: EPL Dice loss (aligning student DSC masks with student/teacher pseudo-masks) + $\mathcal{L}_{\text{MNP}}$ (multi-cue constraint) + LPR Dice+CE loss (aligning fused predictions with local pseudo-labels). Training is conducted for 25 epochs with batch size 32, AdamW optimizer with cosine annealing, and AMP mixed precision. Backbone: DINO-ViT-S/8. Training set: CAMO-Train (1,000) + COD10K-Train (3,040), without annotations. Hardware: V100-SXM2 32 GB.

## Key Experimental Results

### Main Results

**Comparison with UCOD Methods (4 COD Benchmarks)**

| Method | Type | CHAMELEON $S_m$↑ | CAMO $S_m$↑ | COD10K $S_m$↑ | NC4K $S_m$↑ |
|--------|------|-----------|----------|-----------|----------|
| FOUND | UOS | .7161 | .6913 | .6783 | .7459 |
| UCOS-DA | UCOD | .6715 | .6581 | .6334 | .7189 |
| UCOD-DPL | UCOD | .7287 | .7013 | .7090 | .7538 |
| SdalsNet | UCOD | .7236 | .6971 | .6967 | .7386 |
| **EReCu** | **UCOD** | **.7321** | **.7027** | **.7221** | **.7583** |

### Ablation Study

**Module Combination Ablation (CAMO / COD10K $S_m$↑)**

| MNP | EPL | STAF | LPR | CAMO | COD10K |
|-----|-----|------|-----|------|--------|
| ✓ | ✓ | ✓ | ✓ | **.7027** | **.7221** |
| ✗ | ✓ | ✓ | ✓ | .6887 | .7111 |
| ✓ | ✗ | ✗ | ✓ | .6758 | .7038 |
| ✓ | ✓ | ✓ | ✗ | .6895 | .7109 |
| ✗ | ✗ | ✗ | ✗ | .6376 | .6400 |

### Key Findings

- The full model achieves UCOD state-of-the-art across all primary metrics on four benchmarks.
- PEF (EPL + STAF) contributes most significantly: removing it causes a 2.69% drop in CAMO $S_m$ (.7027 → .6758).
- The MNP + EPL combination yields the largest complementary gain, validating the critical role of native cues in guiding pseudo-label evolution.
- Single- or dual-module configurations perform substantially below three- or four-module combinations, confirming strong inter-module complementarity.
- DINO baseline (no modules): CAMO $S_m = .6376$; full model improves by +.0651.

## Highlights & Insights

- Unifying the two UCOD paradigms—pseudo-label guidance and feature learning—into a co-evolutionary framework is conceptually elegant and technically compelling.
- The three-region (interior/boundary/exterior) patch-sampling cosine metric $S_{\text{mc}}$ in MNP is a well-designed contribution that is transferable to mask quality estimation in other unsupervised segmentation tasks.
- STAF employs Tucker decomposition and SVD for spectral fusion of multi-layer attention maps in a lightweight and elegant manner ($\mathcal{O}(r^2 d)$), offering a new approach to multi-scale feature aggregation.
- The dual-condition selection mechanism in TAS—combining attention entropy and multi-cue consistency—demonstrates strong generalizability.

## Limitations & Future Work

- Performance gains on certain datasets and metrics are marginal (e.g., CAMO $S_m$ improves by only +.0014); on the MAE metric, COD10K is on par with UCOD-DPL.
- Validation is limited to DINO-ViT-S/8; larger-scale backbones such as DINOv2 have not been explored.
- Texture descriptors in MNP (LBP, DoG) are hand-crafted; learnable alternatives warrant investigation.
- The multi-branch loss, Tucker/SVD operations, and EMA introduce non-trivial training overhead.
- The handling of multi-instance camouflage scenarios is not discussed.

## Related Work & Insights

- **vs. UCOD-DPL**: Both adopt teacher-student dynamic pseudo-labeling; however, UCOD-DPL neglects native image cues, leading to boundary overflow. EReCu introduces MNP for native perceptual guidance and replaces simple weighted aggregation with STAF.
- **vs. SdalsNet**: Self-distillation with attention shift enables foreground-background separation but lacks pseudo-label supervision, causing blurry details. EReCu benefits from both supervision sources simultaneously.
- **vs. FOUND**: FOUND employs a background-first paradigm to infer foreground, but its coarse boundaries are ill-suited for highly similar camouflaged scenes.
- **Insights**: The $S_{\text{mc}}$ metric can be adapted for estimating mask quality of unlabeled samples in active learning; the pseudo-label evolution + native cue guidance paradigm is transferable to unsupervised salient object detection and medical image segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of unifying two UCOD paradigms is well-motivated and each module is thoughtfully designed, though the overall contribution feels somewhat compositional.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four benchmarks, comprehensive ablations, visualizations, and open-source code; some gains are modest.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear, formulations are complete, and the presentation is logically coherent.
- Value: ⭐⭐⭐⭐ — Achieves UCOD state-of-the-art with open-source release; $S_{\text{mc}}$ metric and STAF fusion scheme are broadly reusable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [\[CVPR 2026\] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)

<!-- RELATED:END -->
