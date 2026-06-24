---
title: >-
  [Paper Note] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields
description: >-
  [CVPR 2025][3D Vision][face editing] FFaceNeRF is proposed, a NeRF-based face editing method that adapts to any custom segmentation mask layout using only 10 annotated samples through a geometry adapter, tri-plane feature injection, and latent mixing for triplane augmentation (LMTA), achieving flexible 3D-aware face editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "face editing"
  - "NeRF"
  - "few-shot"
  - "geometry adapter"
  - "tri-plane augmentation"
  - "mask layout adaptation"
date: 2026-05-08
content_hash: 188ce4c9a8646b87
---

# FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields

**Conference**: CVPR 2025  
**arXiv**: [2503.17095](https://arxiv.org/abs/2503.17095)  
**Code**: See project page  
**Area**: Medical Images  
**Keywords**: face editing, NeRF, few-shot, geometry adapter, tri-plane augmentation, mask layout adaptation

## TL;DR

FFaceNeRF is proposed, a NeRF-based face editing method that adapts to any custom segmentation mask layout using only 10 annotated samples through a geometry adapter, tri-plane feature injection, and latent mixing for triplane augmentation (LMTA), achieving flexible 3D-aware face editing.

## Background & Motivation

**Background**: NeRF-based 3D-aware face editing methods (e.g., NeRFFaceEditing, IDE-3D) have achieved high-quality results by leveraging pretrained segmentation networks to generate semantic masks for editing guidance.

**Limitations of Prior Work**:
- The mask layouts of pretrained segmentation networks are fixed (e.g., 19 classes in BiSeNet), making it impossible to edit areas not included in the labels.
- Different editing requirements demand different mask layouts (e.g., makeup artists require eyelid control, while plastic surgeons need nasal wing editing).
- Supporting new mask layouts requires either massive labeled data or luckily finding a matching pretrained segmentation network.
- Existing mask editing methods perform poorly for small area editing (cross-entropy loss biases toward large regions).

**Key Challenge**: High-quality 3D face editing relies on precise semantic segmentation guidance, but the rigidity of predefined segmentation layouts severely limits user control and application scenarios.

**Goal**: Adapt NeRF face editing models to any custom mask layout using extremely few annotated samples (10 images).

**Key Insight**: Instead of retraining the entire geometry decoder, a lightweight geometry adapter (MLP) is appended after it. This adapter compensates for fine-grained semantic details lost during pretraining by injecting tri-plane features and view directions, utilizing latent mixing for data augmentation to prevent overfitting.

**Core Idea**: Fast adaptation of mask layouts at a 10-sample scale is achieved using a tripartite suite: an adapter, feature injection, and latent augmentation.

## Method

### Overall Architecture

1. **Pretraining Stage**: Trains decoupled appearance decoder $\Psi_{app}$ and geometry decoder $\Psi_{geo}$ based on EG3D and NeRFFaceEditing using a fixed-layout pretrained segmentation network as supervision.
2. **Adaptation Stage**: Appends the geometry adapter $\Phi_{geo}$ after $\Psi_{geo}$, injects tri-plane features and view directions, and is trained using augmented data from LMTA for only 40 minutes, updating only $\Phi_{geo}$.
3. **Inference Stage**: Inverts real images to the latent space via PTI, and optimizes the editing vector $\delta w^+$ after modifying the mask.

### Key Designs

**1. Geometry Adapter with Feature Injection**
- **Function**: A lightweight MLP $\Phi_{geo}$ is added after the frozen $\Psi_{geo}$ to map the segmentation output from the fixed layout to the custom layout. Additionally, normalized tri-plane features $\hat{F}'_{tri}$ and view directions $v_d$ are injected directly.
- **Mechanism**: Since $\Psi_{geo}$ only attends to fixed-layout geometry details during pretraining, other fine-grained semantic details (e.g., pupil boundaries, nasal contours) are discarded. Tri-plane features contain complete facial generation details, which can replenish these missing nuances when injected. The view direction, related to the data preprocessing (facial alignment) in EG3D, also carries semantic clues.
- **Design Motivation**: Ablation studies demonstrate that without feature injection, the accuracy using 30 training samples is still inferior to using only 10 samples with feature injection.

**2. Latent Mixing for Triplane Augmentation (LMTA)**
- **Function**: During the training of $\Phi_{geo}$, the last 5 layers (layers 10-14) of the ground-truth latent code $w^+$ are mixed with a random latent code at $\alpha = 0.5$ to generate augmented tri-plane features as training inputs.
- **Mechanism**: In style generators, early layers control geometry/coarse structures, while latter layers dictate details like color tones and brightness. Mixing the latter layers preserves semantic information (mIoU remains nearly constant) while boosting input diversity (higher L1 distance), effectively preventing overfitting in 10-sample training.
- **Design Motivation**: Experiments analyze the impact of mixing each of the 14 layers on semantics (mIoU) and diversity (L1), showing that the top-5 mIoU layers (10-14) provide the optimal trade-off between semantic preservation and augmentative diversity. Mixing all layers destroys geometric patterns, leading to catastrophic failure.

**3. Overlap-Based Inference Optimization**
- **Function**: During inference, the editing vector $\delta w^+$ is optimized to match the generated mask with the edited target mask. Besides cross-entropy, an overlap loss based on the DICE coefficient is incorporated: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{ovlp}$.
- **Mechanism**: LPIPS loss $\mathcal{L}_{LPIPS}(I' \otimes (1-r), I \otimes (1-r))$ is preserved for unedited areas, while $\mathcal{L}_{CE} + \mathcal{L}_{ovlp}$ is applied to edited regions.
- **Design Motivation**: Conventional optimization using only CE neglects small-region edits (e.g., pupil dilation). The DICE overlap loss calculates category-wise overlap rates, and is thus invariant to the class area size.

### Loss & Training

- Training: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{ovlp}$, where $\lambda$ is 0 for the first 4000 steps and 0.1 for the remaining 1000 steps.
- Inference Optimization: $\mathcal{L}_{edit} = \mathcal{L}_{LPIPS}(I' \otimes (1-r), I \otimes (1-r)) + \lambda_{CE} \mathcal{L}_{CE} + \lambda_{ovlp} \mathcal{L}_{ovlp}$
- Mixing ratio $\alpha = 0.5$, OnecycleLR scheduler (peak of 0.03)
- 5000 training steps, batch size of 4, total duration of roughly 40 minutes
- Only 10 annotated samples (sharing the same source identity)

## Key Experimental Results

### Main Results (Perceptual Evaluation, A/B Testing, 21 participants)

| Comparison Method | Faithfulness(%)↑ | Retention(%)↑ | Quality(%)↑ |
|---|---|---|---|
| vs NeRFFaceEditing | **72.29** | **67.83** | **68.68** |
| vs IDE-3D | **79.65** | **80.17** | **81.22** |

### Mask Generation Accuracy (22 test sets, mIoU %)

| Method | Average mIoU [min, max] |
|---|---|
| **FFaceNeRF** | **85.33** [84.8, 85.7] |
| NeRFFaceEditing | 81.37 [81.2, 81.5] |

### Ablation Study (Training Data Size vs mIoU)

| Dataset Size | Ours | w/o injection | w/o LMTA | Mixing all |
|---|---|---|---|---|
| 1 | 0.711 | **0.741** | 0.695 | 0.603 |
| 5 | **0.832** | 0.806 | 0.829 | 0.654 |
| 10 | **0.850** | 0.835 | 0.845 | 0.743 |
| 20 | **0.855** | 0.844 | 0.855 | 0.785 |
| 30 | **0.860** | 0.847 | 0.859 | 0.780 |

### Key Findings

1. **10 samples are sufficient**: Increasing from 5 to 10 samples improves the mIoU from 0.832 to 0.850, after which the marginal utility of more data diminishes.
2. **Feature injection is the most critical component**: Without injection, the accuracy with 30 samples (0.847) is still inferior to using only 10 samples with injection (0.850), and training suffers from color shifts.
3. **LMTA is crucial under extremely few-shot settings**: With 5 samples, the difference between with LMTA (0.832) and without LMTA (0.829) is small; however, with only 1 sample, the gap between with LMTA (0.711) and without (0.695) becomes highly significant.
4. **Mixing all layers is catastrophic**: Mixing all layers destroys the geometry, leading to an mIoU of only 0.743 with 10 samples (compared to 0.850 for the full model) and changes to the source identity.
5. **Overlap optimization is key for small region editing**: In the eye enlargement experiment, overlap-based optimization follows the target dimensions more faithfully than percentage-based optimization.

## Highlights & Insights

- The design philosophy of the "adapter + injection + augmentation" three-piece few-shot adaptation framework is highly coherent.
- Systematic analysis of hierarchical semantics in style generators (14-layer mIoU/L1 experiments) provides sound justification for LMTA.
- The application of DICE overlap loss to handle small region editing is elegant and effective.
- Not limited to EG3D/NeRFFaceEditing, experiments with FFaceGAN demonstrate the generalizability of the proposed method.

## Limitations & Future Work

- Inference requires iterative optimization (~31 seconds per edit), making real-time interaction unfeasible.
- 1-shot performance is limited (0.711 mIoU) since the geometry adapter still requires diverse training data.
- Dependence on the quality of PTI inversion—imprecise inversion results in inaccurate editing.
- Exclusively evaluated on the EG3D architecture, leaving new representations like 3D Gaussian Splatting unexplored.
- Users still need to manually annotate 10 masks, maintaining a minor labeling overhead.

## Related Work & Insights

- NeRFFaceEditing and IDE-3D depend on fixed segmentation networks, binding their editing power to predetermined mask layouts; FFaceNeRF bypasses this constraint via the adapter.
- DatasetGAN generates segmentations from StyleGAN using small datasets; FFaceGAN demonstrates that the adapter + LMTA combination can improve its quality.
- Insight: The few-shot adaptation strategy consisting of an adapter and feature injection can be extended to other generative tasks requiring custom labels.

## Rating

⭐⭐⭐⭐ — The logic behind few-shot mask adaptation is highly practical, and the experimental validation is solid, although the inference speed and 1-shot performance remain bottlenecks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)
- [\[CVPR 2025\] RelationField: Relate Anything in Radiance Fields](relationfield_relate_anything_in_radiance_fields.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)

</div>

<!-- RELATED:END -->
