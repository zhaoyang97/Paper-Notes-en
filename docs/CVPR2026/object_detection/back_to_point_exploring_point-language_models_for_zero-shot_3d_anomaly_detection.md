---
title: >-
  [Paper Note] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection
description: >-
  [CVPR 2026][Object Detection][Zero-Shot 3D Anomaly Detection] BTP is the first work to apply pretrained point-language models (PLMs, e.g., ULIP) to zero-shot 3D anomaly detection. It proposes a Multi-Granularity Feature…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Zero-Shot 3D Anomaly Detection"
  - "Point-Language Model"
  - "ULIP"
  - "Multi-Granularity"
  - "Geometric Feature"
date: 2026-05-08
content_hash: 8e094d225c6326e1
---

# Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection

**Conference**: CVPR 2026
**arXiv**: [2603.21511](https://arxiv.org/abs/2603.21511)  
**Code**: [https://github.com/wistful-8029/BTP-3DAD](https://github.com/wistful-8029/BTP-3DAD)  
**Area**: 3D Vision / Anomaly Detection
**Keywords**: Zero-Shot 3D Anomaly Detection, Point-Language Model, ULIP, Multi-Granularity, Geometric Feature

## TL;DR
BTP is the first work to apply pretrained point-language models (PLMs, e.g., ULIP) to zero-shot 3D anomaly detection. It proposes a Multi-Granularity Feature Embedding Module (MGFEM) that fuses patch-level semantics, geometric descriptors, and global CLS tokens, coupled with a joint representation learning strategy. BTP achieves 84.5% point-level AUROC on Real3D-AD, substantially outperforming the VLM-rendering-based method PointAD (73.5%).

## Background & Motivation

**Background**: 3D anomaly detection is critical for industrial quality inspection. Zero-shot (ZS) approaches are highly attractive as they require no training data from target categories, yet the field remains in its early stages.

**Limitations of Prior Work (VLM-based)**:
   - Dominant methods (PointAD, MVP) render 3D point clouds into multi-view 2D images and apply VLMs such as CLIP for anomaly detection.
   - **Key Challenge**: The rendering process **discards geometric details**, resulting in insensitivity to locally structural anomalies.
   - Performance is heavily dependent on the number and angles of rendered views, introducing view-selection bias.
   - Repeated projection–back-projection incurs additional computational overhead.

**Opportunity with PLMs**: Point-language models such as ULIP directly encode 3D point clouds, preserving intrinsic geometric and structural properties. This raises the question: can anomaly detection be performed **directly in 3D space** rather than via a 2D detour?

**Core Challenge**: ULIP was originally designed for point cloud classification (global embeddings) and is ill-suited for fine-grained anomaly **localization**, which demands patch-level features and geometry-aware representations.

**Core Idea**: "Back to Point" — perform anomaly detection directly on point clouds, enhancing PLM sensitivity to local anomalies through multi-granularity features and geometric descriptors.

## Method

### Overall Architecture
Input point cloud → ULIP encoder extracts semantic features + GFCM extracts geometric descriptors → MGFEM fuses multi-granularity features → comparison with text embeddings → anomaly scores.

### Key Designs

1. **Patch-Level Feature Exploitation**:

    - ULIP originally uses only the final-layer global embedding. BTP **additionally extracts patch-level representations from multiple intermediate layers**.
    - Different layers capture geometric and semantic information at varying levels of abstraction.
    - Combining these patch representations significantly improves sensitivity to local structural changes.

2. **Geometric Feature Creation Module (GFCM)**:

    - **Design Motivation**: Classical geometric descriptors such as FPFH effectively characterize local geometric relationships but, as hand-crafted features, cannot be optimized end-to-end.
    - **Mechanism**: A learnable PointNet-style network replaces FPFH:
        - A shared MLP is applied to neighboring points within each patch.
        - Max-pooling aggregates them into a patch-level geometric descriptor.
        - An FC layer projects the descriptor to the dimension aligned with text embeddings.
    $\mathbf{f}_i = \phi\left(\max_{j=1,...,M} \text{MLP}(\mathbf{p}_{ij})\right)$
    - Geometric priors are explicitly injected via a contrastive loss against FPFH.

3. **Multi-Granularity Feature Embedding Module (MGFEM)**:

    - Fuses three types of information:
        - Multi-layer intermediate semantic features $\{\mathbf{H}^{(l)}\}_{l=1}^L$ (weighted sum with learnable weights)
        - Geometric features $\mathbf{F}_{geo}$
        - Global CLS token $\mathbf{h}_{CLS}$
    - Each is projected to a unified space and then concatenated:
    $\mathbf{Z} = \phi_f\left([\sum_l \alpha_l \mathbf{S}^{(l)} \| \mathbf{G} \| \mathbf{C}]\right)$
    - The resulting $\mathbf{Z} \in \mathbb{R}^{N \times D}$ serves as structure-aware patch representations.

4. **Hybrid Learnable Prompts**:

    - Combines a small number of learnable context tokens with fixed templates ("normal object" / "defective object").
    - ULIP encodes these to produce normal/anomalous text embeddings.
    - Anomaly scores are derived by computing similarity with point cloud features.

### Loss & Training
Joint representation learning with three levels of supervision:
$$\mathcal{L} = \mathcal{L}_{local} + \lambda_1 \mathcal{L}_{global} + \lambda_2 \mathcal{L}_{geo}$$

- **$\mathcal{L}_{local}$ = Focal Loss + Dice Loss**: Point-level supervision, mitigating positive/negative sample imbalance.
- **$\mathcal{L}_{global}$ = BCE**: Global object-level discrimination (combining point-level and patch-level predictions).
- **$\mathcal{L}_{geo}$ = Contrastive Loss**: Aligns learned geometric features with FPFH (InfoNCE).
- $\lambda_1 = 0.5,\ \lambda_2 = 0.1$
- GFCM and MGFEM are trained on auxiliary point cloud data; no target-category data is required during zero-shot inference.

## Key Experimental Results

### Main Results (Real3D-AD, Zero-Shot)

| Method | Type | Object AUROC↑ | Point AUROC↑ |
|--------|------|---------------|--------------|
| CPMF | Non-ZS | 58.6 | 75.9 |
| PatchCore-FPFH | Non-ZS | 53.1 | 62.5 |
| PointCLIPV2 | ZS-VLM | 53.1 | 52.9 |
| AnomalyCLIP | ZS-VLM | 55.2 | 50.3 |
| PointAD | ZS-VLM | 74.8 | 73.5 |
| **BTP (Ours)** | **ZS-PLM** | 61.4 | **84.5** |

### Ablation Study (Inferred from contributions of individual MGFEM components)

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Global ULIP embedding only | Lower point-level AUROC | Vanilla ULIP unsuitable for fine-grained localization |
| + Patch-level features | Improved point-level AUROC | Intermediate-layer features enhance local awareness |
| + GFCM | Further improvement | Learnable geometric descriptors strengthen structural perception |
| + Joint learning (full BTP) | **84.5% point-level AUROC** | Three-level supervision is complementary; best overall |

### Key Findings
- **Point-level localization**: BTP's 84.5% substantially surpasses PointAD's 73.5% (+11.0%), confirming the advantage of direct 3D-space detection.
- **Object-level detection**: BTP's 61.4% falls below PointAD's 74.8%, indicating room for improvement in global discrimination.
- Exceptionally high point-level AUROC on categories such as diamond (97.9%), car (91.6%), and duck (90.9%).
- Operating directly in 3D eliminates the view-selection bias inherent in VLM-based approaches.

## Highlights & Insights
- The "Back to Point" proposition is meaningful — when 3D data is natively available, a 2D detour is unnecessary.
- The multi-granularity fusion strategy (semantics + geometry + global) exhibits strong complementarity; joint learning outperforms any individual component.
- The GFCM design — replacing FPFH with a learnable network while aligning with it via contrastive loss — retains the physical intuition of hand-crafted features while enabling end-to-end optimization.
- Strong zero-shot performance demonstrates that PLMs have learned transferable 3D structural priors.

## Limitations & Future Work
- Object-level AUROC (61.4%) is notably weaker than PointAD (74.8%), indicating insufficient global discriminative capability.
- The ULIP encoder is kept frozen; unfreezing and fine-tuning it could yield further gains.
- Training still requires auxiliary point cloud data (though not target-category data), leaving a gap from "truly zero-shot" inference.
- Generalization to a broader range of industrial defect types (e.g., micro-cracks, color deviations) remains to be verified.

## Related Work & Insights
- PointAD is the most direct competitor (VLM rendering approach); BTP substantially leads in localization performance.
- PLANE also leverages PLMs but requires target-category data for category-specific adaptation — BTP follows a cleaner zero-shot paradigm.
- The progression ULIP → ULIP2 → future PLMs may further enhance 3D understanding capabilities.
- The multi-granularity fusion strategy is potentially generalizable to 3D semantic segmentation, point cloud registration, and related tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to apply PLMs to ZS 3D anomaly detection, opening a new research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets with comprehensive metrics, though ablation details lack sufficient quantification.
- **Writing Quality**: ⭐⭐⭐⭐ Well-motivated; the "VLM vs. PLM" comparative framing is clear and compelling.
- **Value**: ⭐⭐⭐⭐ Practically valuable for industrial 3D inspection; the PLM-based direction shows strong potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)

</div>

<!-- RELATED:END -->
