---
title: >-
  [Paper Note] Text–Image Conditioned 3D Generation
description: >-
  [CVPR 2026][3D Vision][text-image joint conditioning] This paper identifies that image and text conditions provide complementary information for 3D generation—images supply precise appearance but are limited by viewpoint, while text provides global semantics but lacks visual detail—and proposes TIGON, a minimalist dual-branch DiT baseline that achieves native text-image jointly conditioned 3D generation via zero-initialized cross-modal bridges (early fusion) and step-wise prediction averaging (late fusion).
tags:
  - CVPR 2026
  - 3D Vision
  - text-image joint conditioning
  - 3D generation
  - dual-branch DiT
  - cross-modal fusion
  - rectified flow
date: 2026-05-08
content_hash: d2a94f6fb64f36c4
---

# Text–Image Conditioned 3D Generation

**Conference**: CVPR 2026
**arXiv**: [2603.21295](https://arxiv.org/abs/2603.21295)
**Code**: [https://jumpat.github.io/tigon-page](https://jumpat.github.io/tigon-page)
**Area**: 3D Vision / 3D Generation
**Keywords**: text-image joint conditioning, 3D generation, dual-branch DiT, cross-modal fusion, rectified flow

## TL;DR
This paper identifies that image and text conditions provide complementary information for 3D generation—images supply precise appearance but are limited by viewpoint, while text provides global semantics but lacks visual detail—and proposes TIGON, a minimalist dual-branch DiT baseline that achieves native text-image jointly conditioned 3D generation via zero-initialized cross-modal bridges (early fusion) and step-wise prediction averaging (late fusion).

## Background & Motivation
1. **State of the Field**: Native 3D generation models (e.g., TRELLIS, UniLat3D) can already produce high-quality 3D assets from a single condition (image or text). These methods perform well within their respective modalities but rely on a single conditioning signal.
2. **Limitations of Prior Work**: (a) **Image-conditioned** 3D generation is highly sensitive to the input viewpoint—when the reference view is uninformative (e.g., looking upward or heavily occluded), the model must "hallucinate" invisible regions, causing outputs to deviate from user intent; (b) **Text-conditioned** generation provides comprehensive semantics but lacks low-level visual constraints, often resulting in outputs of limited visual quality.
3. **Root Cause**: Images provide locally precise geometric and appearance cues but offer incomplete coverage; text provides global semantics but at insufficient granularity—the two modalities are precisely complementary.
4. **Paper Goals**: (a) Diagnose and quantify the limitations of single-modality 3D generation; (b) formalize the novel task of "text-image jointly conditioned 3D generation"; (c) design a concise and effective dual-modality baseline.
5. **Starting Point**: The authors conduct a diagnostic experiment in which the velocity fields of two pretrained rectified-flow models—one image-conditioned and one text-conditioned—are directly averaged at inference time (termed SimFusion). This naive fusion already substantially outperforms either single-modality method (FD_DINOv2: 82.40 vs. 125.93/154.88), revealing strong cross-modal complementarity.
6. **Core Idea**: Retain two modality-specific DiT backbones, exchange features via lightweight cross-modal linear bridges, and then progressively average predictions along the denoising trajectory to achieve joint text-image 3D generation.

## Method

### Overall Architecture
TIGON adopts a dual-branch architecture built on the rectified-flow framework of UniLat3D. Given a reference image $\mathbf{I}$ and a text description $\mathbf{T}$, two independent DiT branches predict velocity fields conditioned on image and text respectively. The branches exchange features through cross-modal bridges (early fusion), and their final predictions are merged via step-wise averaging (late fusion), yielding a decoded 3D asset (mesh or 3DGS).

### Key Designs

1. **Dual-Branch DiT Backbone**:

    - **Function**: Maintains separate diffusion Transformers for image and text conditions, avoiding the mixing of heterogeneous token semantics within a single backbone.
    - **Mechanism**: Image-condition tokens are dense, view-anchored, and locally rich; text-condition tokens are sparse, abstract semantic representations. For instance, "tiger" may require only one token in text but a large number of tokens in an image. This granularity mismatch makes joint processing prone to degradation. The two branches independently predict velocity fields $\mathbf{v}_{\text{img}} = \mathcal{F}_{\text{img}}(\tilde{\mathbf{z}}, t, \mathbf{I})$ and $\mathbf{v}_{\text{txt}} = \mathcal{F}_{\text{txt}}(\tilde{\mathbf{z}}, t, \mathbf{T})$.
    - **Design Motivation**: Preserving the original single-modality capacity of each backbone avoids the performance degradation that can arise from forced entanglement under limited data.

2. **Cross-Modal Linear Bridge (Early Fusion)**:

    - **Function**: Enables bidirectional feature exchange between each pair of DiT blocks.
    - **Mechanism**: After the $i$-th block output, learned linear projections $\mathcal{P}^{(i)}_{\text{txt}\rightarrow\text{img}}$ and $\mathcal{P}^{(i)}_{\text{img}\rightarrow\text{txt}}$ inject cross-modal information: $\mathbf{f}^{(i),\prime}_{\text{img}} = \mathbf{f}^{(i)}_{\text{img}} + \mathcal{P}^{(i)}_{\text{txt}\rightarrow\text{img}}(\mathbf{f}^{(i)}_{\text{txt}})$. Following the zero-initialization strategy from ControlNet, all bridge parameters are initialized to zero, so branch behavior at the start of training is identical to that of the pretrained models, with gradients gradually "opening" these gates.
    - **Design Motivation**: Without explicit interaction, the two branches may diverge during denoising, causing their averaged predictions to mutually corrupt fine details. Ablation experiments confirm that FD_DINOv2 is 66.78 without bridges and drops to 61.59 with bridges.

3. **Step-wise Prediction Averaging (Late Fusion)**:

    - **Function**: Fuses the final predictions of the two branches at each denoising step.
    - **Mechanism**: $\mathbf{v} = \frac{1}{2}(\mathbf{v}_{\text{txt}} + \mathbf{v}_{\text{img}})$, a simple equal-weight average. The authors verify that more complex fusion strategies (adaptive weighting AW, attention-based fusion AT) yield only marginal improvements (60.90 vs. 61.59).
    - **Design Motivation**: Since early fusion already implicitly conditions each branch on both modalities, branch parameters can absorb any potential benefit of dynamic fusion via reparameterization. Simple averaging avoids additional parameters and training variance.

### Loss & Training
Training proceeds in two stages: (1) The two branches are pretrained separately—the image branch uses the original UniLat3D checkpoint, while the text branch is trained from scratch on the same backbone for 1M iterations; (2) Joint fine-tuning for 50K iterations, updating both the cross-modal bridges and all parameters. During training, image and text conditions are independently dropped out with probability 0.5, producing a uniform mixture of 25% unconditional / 25% text-only / 25% image-only / 25% text+image, enabling the model to handle free-form conditioning inputs.

## Key Experimental Results

### Main Results (Toys4K Dataset)

| Model | Condition | Repr. | CLIP↑ | FD_DINOv2↓ |
|-------|-----------|-------|-------|------------|
| UniLat3D | Image | GS | 91.20 | 85.30 |
| UniLat3D | Text | GS | 86.14 | 154.88 |
| SimFusion (naive fusion) | Image+Text | GS | 91.95 | 66.78 |
| **TIGON** | **Image+Text** | **GS** | **92.33** | **61.59** |
| TRELLIS | Image (View-1) | GS | 88.16 | 143.58 |
| TIGON | Image | GS | 91.40 | 84.62 |
| TIGON | Text | GS | 86.77 | 152.34 |

### Ablation Study (Toys4K)

| Cross-Modal Bridge | Fusion Strategy | Joint Fine-tuning | CLIP↑ | FD_DINOv2↓ |
|--------------------|-----------------|-------------------|-------|------------|
| ✗ | Sim | ✗ | 91.95 | 66.78 |
| ✗ | Sim | ✓ | 92.05 | 66.04 |
| ✓ | Sim | ✓ | **92.33** | **61.59** |
| ✓ | AW | ✓ | 92.31 | 60.90 |
| ✓ | AT | ✓ | 92.26 | 62.00 |

### Key Findings
- **Cross-modal complementarity is genuine**: SimFusion alone reduces FD_DINOv2 from 85.30 (image-only) / 154.88 (text-only) to 66.78, confirming that the two modalities supply complementary information.
- **Cross-modal bridges are the core contribution**: Without bridges, joint fine-tuning yields only marginal improvement (66.78→66.04); adding bridges produces a substantial gain (→61.59). Qualitatively, without bridges the two branches diverge during denoising and produce structurally inconsistent outputs.
- **Complex fusion strategies are unnecessary**: AW and AT fusion show only negligible differences compared to simple averaging, demonstrating that early fusion is sufficient for mutual branch awareness.
- **TIGON preserves single-modality capability**: Under image-only or text-only conditioning, TIGON performs comparably to the corresponding UniLat3D single-modality models.

## Highlights & Insights
- **Diagnosis-driven task formulation** is particularly rigorous: the paper first quantitatively demonstrates single-modality limitations and cross-modal complementarity before defining the new task, avoiding the trap of "a solution in search of a problem."
- **Minimalist design philosophy**: Effective cross-modal fusion is achieved using only linear projections with zero initialization, without attention mechanisms or complex gating. This design simultaneously preserves single-modality generation capability and supports free-form conditioning.
- **Controllable generation** is notable: fixing the image while varying the text yields attribute-adjusted 3D objects; text dominates when image information is weak, and the image dominates when it is strong—this adaptive trade-off is learned implicitly.

## Limitations & Future Work
- Validation is limited to the UniLat3D framework; generalizability to other native 3D generators (e.g., direct 3DGS generation) has not been tested.
- When image and text conditions explicitly conflict, TIGON tends to follow the image—no explicit conflict resolution mechanism is provided.
- Training data is TRELLIS-500K, and evaluation is restricted to Toys4K and UniLat1K, both synthetic 3D asset benchmarks; real-world generalization remains to be verified.
- The dual-branch architecture incurs approximately 2× parameter count and inference cost; more lightweight conditioning injection strategies merit exploration.
- ULIP/Uni3D metrics are reported only for mesh outputs; point-cloud-level evaluation for 3DGS outputs is absent.

## Related Work & Insights
- **vs. TRELLIS/UniLat3D**: Single-modality baselines upon which TIGON adds cross-modal fusion capability, preserving single-modality performance while gaining the benefits of joint conditioning.
- **vs. TICD**: TICD integrates text-image conditions via modified SDS, relying on 2D diffusion priors; TIGON operates directly within a native 3D generation framework.
- **vs. FlexGen**: FlexGen focuses on 2D multi-view generation rather than native 3D synthesis; TIGON directly generates 3D representations.

## Supplementary Details

### Datasets and Evaluation Metrics
- Training set: TRELLIS-500K
- Test sets: Toys4K (approximately 4K objects across 105 categories) and UniLat1K (a more challenging 1K-object benchmark)
- Key metrics: CLIP (semantic alignment of rendered images), FD_DINOv2 (visual fidelity of rendered images, lower is better), ULIP/Uni3D (3D point cloud–image alignment, available for mesh outputs only)
- Each object is conditioned on three reference viewpoints (front, top, bottom) rather than ideal views, to evaluate viewpoint robustness

### Condition Conflict Behavior
When image and text conditions explicitly conflict, TIGON tends to follow the image, as images are generally more specific and less ambiguous than text. This suggests that future work could design explicit conflict trade-off mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The task formulation is novel and experimentally grounded, though the method itself is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are comprehensive and qualitative results are rich, but the test sets are small and consist of synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain is clear, progressing systematically from diagnosis → task definition → method → validation.
- **Value**: ⭐⭐⭐⭐ Opens a meaningful new direction; the minimalist baseline provides a clear foundation for future improvement.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [\[CVPR 2026\] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](forgedreamer_industrial_text-to-3d_generation_with_multi-expert_lora_and_cross-v.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] NI-Tex: Non-isometric Image-based Garment Texture Generation](ni-tex_non-isometric_image-based_garment_texture_generation.md)

<!-- RELATED:END -->
