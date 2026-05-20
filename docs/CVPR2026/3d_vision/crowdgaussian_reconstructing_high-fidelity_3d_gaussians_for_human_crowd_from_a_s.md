---
title: >-
  [Paper Note] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image
description: >-
  [CVPR 2026][3D Vision][Human body reconstruction] CrowdGaussian proposes a unified framework for reconstructing multi-person 3D Gaussian splatting representations from a single image. It recovers complete geometry of occ…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Human body reconstruction"
  - "3D Gaussian splatting"
  - "occlusion recovery"
  - "diffusion model refinement"
  - "crowd scenes"
date: 2026-05-08
content_hash: e9473c5ac8f02695
---

# CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image

**Conference**: CVPR 2026
**arXiv**: [2603.17779](https://arxiv.org/abs/2603.17779)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: Human body reconstruction, 3D Gaussian splatting, occlusion recovery, diffusion model refinement, crowd scenes

## TL;DR

CrowdGaussian proposes a unified framework for reconstructing multi-person 3D Gaussian splatting representations from a single image. It recovers complete geometry of occluded regions via a self-supervised-adapted Large Occluded Human Reconstruction Model (LORM), and enhances texture detail quality through a single-step diffusion refiner (CrowdRefiner) trained with Self-Calibrated Learning (SCL).

## Background & Motivation

**Background**: Single-image 3D human reconstruction has seen significant progress in recent years, with large reconstruction models (LRMs) leveraging Transformers and large-scale datasets to enable fast feed-forward reconstruction from a single image. However, existing methods almost exclusively handle clear, close-range single-person images.

**Limitations of Prior Work**:
   - **Severe occlusion**: In crowd scenes, frequent inter-person and person-object occlusions result in incomplete body parts. Existing methods applied directly to such inputs produce transparent holes and incomplete geometry.
   - **Low resolution**: Cropped images of individuals in crowds are of very low resolution, leading to blurry appearance and lack of high-frequency detail.
   - **Efficiency demands for multi-person scenes**: Simultaneously reconstructing a large number of people is required, yet sequential per-person processing is too inefficient.

**Key Challenge**: Existing large human reconstruction models possess strong 2D-to-3D generative priors but lack occlusion-aware training. When fed occluded inputs, the Transformer fails to integrate incomplete visual features, producing fragmented outputs. Fine-tuning with limited 3D supervision tends to amplify geometric bias from monocular ambiguity, thereby degrading the pre-trained prior.

**Goal**: (a) How to recover complete 3D human bodies from severely occluded crops? (b) How to recover high-frequency texture details from low-resolution inputs? (c) How to efficiently handle multi-person scenes simultaneously?

**Key Insight**: Rather than supervised fine-tuning with 3D annotations, the paper adopts a *self-supervised distillation* approach — a frozen teacher model generates pseudo ground truth on complete images, while a student model learns to recover complete geometry from occluded inputs.

**Core Idea**: A two-stage framework — Stage 1 uses a self-supervised-adapted LORM to generate coarse but complete multi-person 3DGS; Stage 2 uses a CrowdRefiner trained with SCL to refine the rendered results, which are then distilled back into the 3DGS.

## Method

### Overall Architecture

The input is a single image containing $N$ persons. **Stage 1**: Multi-person HMR estimates SMPL-X parameters and 3D positions for each individual → SAM segments each person → LORM recovers complete 3DGS from occluded crops → initial coarse multi-person 3DGS scene is assembled. **Stage 2**: The coarse scene is rendered → CrowdRefiner refines the rendering → refined results are distilled back into 3DGS via differentiable rendering as pseudo ground truth.

### Key Designs

1. **LORM (Large Occluded Human Reconstruction Model)**

    - **Function**: Recovers a complete 3D Gaussian representation — including geometry and texture — from an occluded single-person crop.
    - **Mechanism**: Built upon the pre-trained large human reconstruction model LHM-500M, which uses a Sapiens encoder (MAE architecture) + multimodal body-head Transformer (MBHT) + Gaussian decoder. To adapt to occluded inputs, the Sapiens encoder and Gaussian decoder are **frozen**, and **trainable LoRA modules** are injected only into the MBHT Transformer.
    - **Self-supervised adaptation framework**:
        - Teacher stream: The frozen pre-trained model processes the complete image $I_{\text{full}}$, generates complete 3D Gaussians $\mathcal{G}_{\text{full}}$, and renders clean pseudo GT $R_{\text{clean}}^{(v)}$ from $V$ novel viewpoints.
        - Student stream: A random occlusion mask (Bézier curves + keypoint ellipses) is applied to $I_{\text{full}}$ to obtain $I_{\text{occ}}$; LORM predicts 3DGS and renders coarse views $R_{\text{coarse}}^{(v)}$.
        - Self-distillation loss: $\mathcal{L}_{\text{self-distill}} = \sum_v (\lambda_{\text{rgb}} \|R_{\text{clean}}^{(v)} - R_{\text{coarse}}^{(v)}\|_2 + \lambda_{\text{ssim}} (1 - \text{SSIM}(R_{\text{clean}}^{(v)}, R_{\text{coarse}}^{(v)})))$
    - **Design Motivation**: External 3D annotations are avoided (as they amplify monocular ambiguity); the pre-trained model itself serves as teacher, imparting occlusion-handling capability through 2D consistency alone. LoRA fine-tunes only the Transformer attention weights, preserving the pre-trained visual feature extraction and Gaussian generation capacity.

2. **CrowdRefiner (Single-Step Diffusion Refiner)**

    - **Function**: Refines coarse multi-person 3DGS renderings into high-fidelity images, which serve as pseudo GT for 3DGS optimization.
    - **Mechanism**: Built on SD-Turbo as a single-step diffusion model. Input consists of the coarse RGB rendering $R_{\text{coarse}}$ and the corresponding SMPL normal map $N$ (geometric prior). The normal map is encoded by a lightweight PoseNet; RGB is encoded by a frozen VAE encoder. Both feature streams are injected into the UNet to guide generation. The VAE decoder is fine-tuned with LoRA adaptation.
    - **Design Motivation**: Although LORM outputs are geometrically complete, textures are over-smoothed (constrained by reconstruction model resolution). Diffusion models can leverage 2D generative priors to supplement high-frequency details. Single-step inference (rather than iterative sampling) is chosen to ensure efficiency.

3. **Self-Calibrated Learning (SCL)**

    - **Function**: Prevents facial distortion and artifacts caused by over-refinement during CrowdRefiner training.
    - **Mechanism**: During training, two types of sample pairs are randomly mixed — (a) standard degraded pairs $(R_{\text{coarse}}, R_{\text{gt}})$: recovering from coarse to high quality; (b) identity-preserving pairs $(R_{\text{gt}}, R_{\text{gt}})$: both input and target are GT, teaching the model not to modify already high-quality regions.
    - **Design Motivation**: Training exclusively on degraded pairs causes the model to aggressively "enhance" all regions, distorting well-reconstructed areas such as the face. Mixing in identity-preserving samples teaches the model to **adaptively** determine which regions require enhancement and which should be preserved.

4. **Cluster-based Refinement Strategy for Multi-Person Scenes**

    - **Function**: Avoids the high computational cost of per-person refinement.
    - **Mechanism**: DBSCAN clusters individuals into spatially coherent groups based on root positions. Each group is rendered and refined simultaneously rather than individually. Refined results are distilled back into 3DGS via L1 + SSIM loss.
    - **Design Motivation**: Crowd scenes contain many individuals, making sequential processing too slow. The clustering strategy ensures spatially adjacent persons are processed together while maintaining global scene consistency.

### Loss & Training

- **LORM self-distillation loss**: $\mathcal{L}_{\text{self-distill}} = \sum_v (\lambda_{\text{rgb}} \| \cdot \|_2 + \lambda_{\text{ssim}} (1 - \text{SSIM}))$, rendered from 24 fixed viewpoints.
- **CrowdRefiner training loss**: $\mathcal{L}_{\text{diff}} = \lambda_{L2}\mathcal{L}_{\text{L2}} + \lambda_{\text{lpips}}\mathcal{L}_{\text{LPIPS}} + \lambda_{\text{ssim}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{gram}}\mathcal{L}_{\text{Gram}}$
- **3DGS optimization loss**: $\mathcal{L}_{\text{optim}} = \|R_{\text{refined}} - R_{\text{coarse}}\|_1 + \lambda_{\text{ssim}}(1 - \text{SSIM})$
- LORM training data: 1,002 frontal images from HuGe100K.
- CrowdRefiner training data: 114 synthetic multi-person scenes from THuman2.1 (91 train / 23 test), with 126 viewpoints per scene.

## Key Experimental Results

### Main Results

Quantitative comparison on occluded human reconstruction (THuman2.1, random occlusion masks):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| IDOL | 18.063 | 0.919 | 0.994 |
| LHM | 18.171 | 0.918 | 1.012 |
| LORM (Ours) | 18.566 | 0.923 | 0.956 |
| LORM + CrowdRefiner | **18.619** | **0.931** | **0.914** |

Robustness under varying occlusion rates (THuman2.1):

| Method | Occlusion Rate | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|---------------|--------|--------|---------|
| IDOL | 20% | 18.196 | 0.921 | 0.978 |
| IDOL | 60% | 16.667 | 0.909 | 1.063 |
| LHM | 20% | 17.945 | 0.919 | 1.006 |
| LHM | 60% | 17.551 | 0.915 | 1.037 |
| **LORM** | **20%** | **18.428** | **0.923** | **0.947** |
| **LORM** | **60%** | **18.116** | **0.919** | **0.972** |

### Ablation Study

Ablation of SCL strategy and geometric conditioning input in CrowdRefiner:

| SCL | Normal Map | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----|-----------|--------|--------|---------|
| ✗ | ✗ | 20.013 | 0.888 | 0.141 |
| ✗ | ✓ | 20.130 | 0.892 | 0.138 |
| ✓ | ✗ | 20.382 | 0.896 | 0.129 |
| ✓ | ✓ | **20.790** | **0.901** | **0.122** |

### Key Findings

- **LORM degrades minimally under high occlusion rates**: As occlusion increases from 20% to 60%, LORM's PSNR drops by only 0.31 (18.43→18.12), compared to 1.53 for IDOL (18.20→16.67) and 0.39 for LHM. The self-supervised adaptation effectively instills occlusion-handling capability.
- **SCL is critical for preventing over-refinement**: Without SCL, PSNR drops by 0.77 (20.79→20.01), and facial distortion is observed qualitatively. Identity-preserving samples in SCL teach the model not to over-modify reconstructed regions.
- **Normal map conditioning improves geometric consistency**: Adding SMPL normal map input reduces LPIPS from 0.129 to 0.122, providing an explicit geometric constraint for the refiner.
- **Mesh-based methods fail comprehensively under occlusion**: PSHuman and SyncHuman cannot recover geometry for occluded parts, while 3DGS-based IDOL and LHM produce some results but with transparent artifacts and distorted textures.

## Highlights & Insights

- **The self-supervised adaptation strategy is elegant**: The pre-trained model itself serves as teacher; occlusion recovery is learned through synthetic occlusion combined with self-distillation, requiring no external 3D annotations. This paradigm is transferable to any scenario requiring a pre-trained model to adapt to a new degradation type — the generative prior is preserved while the model is trained to handle a new input distribution.
- **The intuition behind SCL is elegant**: Mixing identity-preserving samples (input = output) into training essentially tells the model "if the input is already good, do not alter it." This simple trick effectively resolves the over-modification problem in generative refinement.
- **A complete path from single-person models to multi-person scenes**: LORM + CrowdRefiner + DBSCAN clustering constitutes a complete and scalable multi-person 3D reconstruction pipeline, demonstrating how to build a multi-person system upon single-person models.

## Limitations & Future Work

- The pipeline relies on off-the-shelf pose estimation and segmentation (Multi-HMR, SAM); severe initialization errors propagate to the final results, with hand reconstruction being particularly challenging.
- At very low resolutions, refinement may hallucinate details inconsistent with reality (e.g., specific logos).
- Training data is limited to 114 synthetic scenes from THuman2.1, offering limited diversity.
- SMPL-X parameter estimation is required, which may not generalize to non-standard body shapes or extreme clothing.
- DBSCAN clustering may group too many individuals together in extremely dense crowds, resulting in insufficient refinement resolution.

## Related Work & Insights

- **vs. LHM**: This work directly adapts LHM-500M. LHM produces transparent artifacts under occluded inputs; LORM resolves this via LoRA + self-distillation using only 1,002 images.
- **vs. CHROME**: CHROME uses multi-view diffusion to generate occlusion-free images, but inconsistencies among synthesized views cause texture corruption (especially on faces). LORM performs recovery directly in 3DGS space, avoiding multi-view inconsistency.
- **vs. DIFIX/GSFix3D**: General-purpose 3DGS refinement methods. CrowdRefiner focuses on human-centric scenes and better preserves identity and facial detail through SMPL normal conditioning and the SCL strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The self-supervised adaptation and SCL strategy are innovative, though the overall framework is a modular combination of existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative and qualitative coverage is comprehensive; occlusion-rate gradient experiments are convincing, though a larger-scale real-world benchmark is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive pipeline diagrams.
- **Value**: ⭐⭐⭐⭐ Fills a gap in multi-person 3D reconstruction with direct applicability to VR, telepresence, and related domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)

</div>

<!-- RELATED:END -->
