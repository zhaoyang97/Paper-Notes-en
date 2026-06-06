---
title: >-
  [Paper Note] LayerD: Decomposing Raster Graphic Designs into Layers
description: >-
  [ICCV 2025][Layer decomposition] This paper proposes LayerD, a method that decomposes raster graphic designs into editable layers by iteratively extracting the unoccluded top layer and completing the background. It lever…
tags:
  - "ICCV 2025"
  - "Layer decomposition"
  - "graphic design"
  - "iterative matting"
  - "palette optimization"
  - "evaluation protocol"
date: 2026-05-08
content_hash: d5081727975e199e
---

# LayerD: Decomposing Raster Graphic Designs into Layers

**Conference**: ICCV 2025
**arXiv**: [2509.25134](https://arxiv.org/abs/2509.25134)  
**Code**: [https://cyberagentailab.github.io/LayerD/](https://cyberagentailab.github.io/LayerD/)  
**Area**: Other (Graphic Design / Image Decomposition)
**Keywords**: Layer decomposition, graphic design, iterative matting, palette optimization, evaluation protocol

## TL;DR

This paper proposes LayerD, a method that decomposes raster graphic designs into editable layers by iteratively extracting the unoccluded top layer and completing the background. It leverages domain priors of graphic design (texture-flat regions) for refinement, and introduces a DTW-based hierarchical evaluation protocol.

## Background & Motivation

Designers create and edit graphic designs using layers as the fundamental unit in tools such as Photoshop and PowerPoint. Once flattened into a raster image, layer information is lost, making editing and reuse difficult. Reversely decomposing a raster image into a layer sequence — i.e., solving the inverse of the compositing process — can re-enable layer-based editing workflows.

However, layer decomposition of graphic designs poses unique challenges:

**Element diversity**: Graphic designs contain a mixture of typography, decorative elements, vector graphics, illustrations, and even natural image assets.

**Difference from natural images**: Directly applying natural scene decomposition methods leads to undesirable decompositions (e.g., erroneous separation of objects within photographic assets) or artifacts (e.g., background lighting affecting solid-color vector graphics).

**Inherent ill-posedness**: Multiple valid solutions exist — a single layer can be arbitrarily split into multiple layers — making evaluation difficult.

Existing methods (e.g., MULAN, Accordion) adopt stacked pipelines (detection → segmentation → ordering → completion), which leverage pretrained models at each stage but inevitably accumulate errors.

## Method

### Overall Architecture

LayerD formulates the decomposition task as an **iterative top-layer matting + background completion** process. Layers are extracted front-to-back (from $m=M$ to $m=1$):
1. A matting model $F_\theta$ predicts the alpha map of the current top layer.
2. A background completion model $G_\phi$ (LaMa) inpaints the extracted region.
3. The foreground RGB values are computed via inverse alpha blending.
4. The process repeats until the predicted alpha map contains no significant foreground.

This design **unifies detection, segmentation, and layer ordering into a single end-to-end task**, avoiding error accumulation inherent in stacked pipelines.

### Key Designs

1. **Iterative Top-Layer Matting**:

   A trimap-free matting model (based on BiRefNet + Swin-L) is trained on the Crello dataset with supervised learning. The training target is the unambiguous "unoccluded top layer" — obtained by examining inter-layer occlusion relationships and merging the alpha maps of all unoccluded layers into a single target. This clear target definition eliminates training ambiguity.

   A key improvement: training data includes samples processed by the background completion model (i.e., the inpainted image after removing the top layer is used as input), making the matting model robust to completion artifacts and bridging the train-inference distribution gap.

   Loss function: $\mathcal{L} = \lambda_{BCE}\mathcal{L}_{BCE} + \lambda_{IoU}\mathcal{L}_{IoU} + \lambda_{SSIM}\mathcal{L}_{SSIM}$. Training uses all losses initially, with only the SSIM loss in later stages to improve boundary quality.

2. **Palette-based Refinement**:

   This component exploits a domain prior of graphic design — the prevalence of **texture-flat regions** (e.g., solid-color backgrounds, vector shapes, text):

   - **Background refinement**: The inpainting target region is segmented into connected components, and the color gradient of surrounding areas is computed. If zero-gradient regions dominate, a color palette is extracted and the completion result is mapped to the nearest palette color in Lab space, eliminating artifacts introduced by the completion model in flat regions.

   - **Foreground refinement**: Connected components of the foreground are similarly analyzed. For regions classified as flat: areas matching palette colors are extracted from the original image or intermediate completed background. If the overlap with the predicted alpha exceeds a threshold, a new alpha mask is generated from this region. This significantly improves boundary quality and detection of fine decorative layers (lines, borders).

3. **Foreground Color Estimation**:

   Rather than naively replacing the original image alpha with a segmentation mask, LayerD precisely estimates the foreground RGB values via inverse alpha blending using the known alpha map and completed background:

   $\hat{l}_m^C = \frac{\hat{x}_m^C - \hat{x}_{m-1}^C \odot (1 - \hat{l}_m^A)}{\hat{l}_m^A}$

   For semi-transparent pixels (alpha < 1), this better handles foreground-background mixing than direct replacement.

### Loss & Training

- The matting model is based on BiRefNet + Swin-L, trained on the Crello training set for 60 epochs with batch size 12.
- Background completion uses an off-the-shelf LaMa model without fine-tuning.
- Maximum number of iterations is set to 3.
- Maximum palette size: 10 colors for foreground, 2 for background.

## Key Experimental Results

### Main Results

Evaluated on the Crello test set using a custom evaluation protocol (DTW alignment + hierarchical edit distance):

| Method | RGB L1↓ (0 edits) | Alpha IoU↑ (0 edits) | RGB L1↓ (3 edits) | Alpha IoU↑ (3 edits) |
|------|-------------------|---------------------|-------------------|---------------------|
| YOLO baseline | ~0.055 | ~0.42 | ~0.045 | ~0.52 |
| VLM baseline | ~0.050 | ~0.45 | ~0.040 | ~0.55 |
| BiRefNet (no additional training) | ~0.045 | ~0.50 | ~0.038 | ~0.58 |
| LayerD (no text training) | ~0.032 | ~0.60 | ~0.025 | ~0.68 |
| LayerD + Hi-SAM | ~0.035 | ~0.57 | ~0.028 | ~0.65 |
| **LayerD** | **~0.030** | **~0.62** | **~0.023** | **~0.70** |

Note: Values are approximations read from Figure 5 of the paper. LayerD achieves the best performance across all metrics and edit distances.

### Ablation Study

| Configuration | RGB L1↓ | Alpha IoU↑ |
|------|---------|-----------|
| Naive (mask replacement) | baseline | baseline |
| + Inverse blending color estimation | ↓ significant improvement | — |
| + Background refinement | ↓↓ large improvement | ↑ downstream layers also improve |
| + Foreground refinement | ↓↓ | ↑ boundary quality improves |

Additional findings:
- LayerD alone outperforms LayerD + Hi-SAM, indicating that the domain-specifically trained matting model surpasses general-purpose text segmentation.
- LayerD trained with text layers slightly outperforms the version without text training even in the "text-excluded" evaluation, as text elements are essentially variants of vector shapes.

### Key Findings

- **Unified pipeline outperforms component stacking**: LayerD unifies detection, segmentation, and ordering into a single iterative matting task, systematically outperforming YOLO and VLM baselines on all metrics.
- **Domain priors are critical**: Palette-based refinement leverages the prevalence of flat regions in graphic designs, substantially eliminating artifacts.
- **Train-inference alignment is effective**: Including inpainted images as training inputs improves robustness to completion artifacts.
- **Novel evaluation protocol**: DTW-based layer alignment combined with edit distance provides a more principled metric than per-layer pixel comparison.
- **Foreground refinement improves fine details**: Particularly for thin lines and decorative borders, where plain matting models frequently fail.
- LayerD generalizes to graphic design images generated by FLUX.1.

## Highlights & Insights

1. **Well-posed problem definition**: Layer decomposition of graphic designs is a practically important problem that has received little attention in prior work.
2. **Elegant simplicity**: Unifying multiple sub-tasks via iterative matting is far more elegant than multi-stage pipelines.
3. **Evaluation protocol contribution**: Ill-posed tasks require appropriate evaluation; DTW alignment combined with edit distance is an insightful design choice.
4. **Domain-knowledge-driven refinement**: Palette-based refinement is simple yet highly effective, reflecting a deep understanding of the graphic design domain.
5. **Clear application value**: The decomposed layers directly support layer-level editing operations such as color transfer, translation, and scaling.

## Limitations & Future Work

- The maximum iteration count is fixed at 3, which may be insufficient for complex designs with more layers.
- Estimation of transparent layers is not the focus and is explicitly excluded.
- Palette-based refinement has limited effectiveness for designs with gradients or complex textures.
- Training and evaluation are conducted solely on the Crello dataset; generalization to other design styles (e.g., print media, UI design) remains to be verified.
- Semantic grouping of layers (e.g., whether a logo composed of an icon and text should constitute a single layer) is not addressed.

## Related Work & Insights

- MULAN: Natural image decomposition using a stacked pipeline of open-vocabulary detection and zero-shot segmentation.
- Accordion: A contemporaneous VLM-based graphic design decomposition work, though without publicly released code or models.
- LaMa: A high-quality image inpainting model and a key component of LayerD.
- BiRefNet: A trimap-free matting model serving as the backbone of LayerD.
- Color segmentation methods: Related but distinct in objective (semi-transparent color layers vs. object layers).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified iterative top-layer matting framework is elegant, and palette-based refinement cleverly exploits domain priors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines, ablations, and qualitative analyses are provided, though evaluation is limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The problem is clearly defined, the method is described in detail, and the figures are rich and intuitive.
- **Value**: ⭐⭐⭐⭐ The method has practical value for creative workflows, and the evaluation protocol contributes to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[ICCV 2025\] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)

</div>

<!-- RELATED:END -->
