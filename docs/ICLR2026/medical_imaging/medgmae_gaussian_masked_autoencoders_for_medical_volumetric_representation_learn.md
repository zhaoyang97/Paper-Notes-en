---
title: >-
  [Paper Note] MedGMAE: Gaussian Masked Autoencoders for Medical Volumetric Representation Learning
description: >-
  [ICLR 2026][Medical Imaging][Paper Note] MedGMAE shifts the MIM pre-training target for 3D medical imaging from "reconstructing discrete voxel intensities" to "predicting a set of continuous 3D Gaussian primitives followed by volume rendering." This learns encoder representations that better align with anatomical continuity and transforms the decoder into a t
tags:
  - ICLR 2026
  - Medical Imaging
date: 2026-05-08
content_hash: 098422ec06a2722a
---
# MedGMAE: Gaussian Masked Autoencoders for Medical Volumetric Representation Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Z2XIRLv535](https://openreview.net/forum?id=Z2XIRLv535)  
**Code**: [https://github.com/windrise/MedGMAE](https://github.com/windrise/MedGMAE)  
**Area**: Medical Imaging / Self-supervised Representation Learning / 3D Gaussian Splatting  
**Keywords**: Masked Autoencoder, 3D Gaussian Primitives, Volumetric Pre-training, Zero-shot Initialization, CT Reconstruction  

## TL;DR
MedGMAE shifts the MIM pre-training target for 3D medical imaging from "reconstructing discrete voxel intensities" to "predicting a set of continuous 3D Gaussian primitives followed by volume rendering." This learns encoder representations that better align with anatomical continuity and transforms the decoder into a transferable "geometric prior" capable of providing zero-shot initialization for 3DGS-CT reconstruction.

## Background & Motivation
- **Background**: The scarcity of annotations forces 3D medical imaging to rely on self-supervised pre-training. Masked Image Modeling (MIM) has become the mainstream due to high anatomical similarity, where the model regresses voxel intensities of masked regions from visible patches.
- **Limitations of Prior Work**: The authors identify three overlooked fundamental flaws in voxel-level reconstruction: (i) **Conflict between discrete reconstruction and anatomical continuity**: Voxel-wise regression is "local context-based filling," which excels at texture but fails to capture geometric abstraction and shape consistency; (ii) **Non-transferable decoders**: Decoders are designed only to reconstruct low-level pixel intensities and are typically discarded after pre-training, limiting zero-shot capabilities; (iii) **Parameter waste due to sparse anatomical distribution**: Anatomical organs occupy only about 11.8% of the space in medical volumes, making dense voxel representations naturally redundant.
- **Key Challenge**: MIM aims to learn structured, geometry-aware anatomical representations, but the proxy task of "voxel-wise regression" essentially encourages local interpolation rather than global structural understanding, making the two objectives contradictory.
- **Goal**: Upgrade the pre-training objective from "local reconstruction" to "geometric reasoning" using an intermediate representation that is continuous, parameter-efficient, and has a reusable decoder.
- **Core Idea**: **Use sparse 3D Gaussian primitives as the intermediate representation** — The model predicts a set of 3D Gaussian parameters (position, scale, rotation, intensity) describing the entire volume from sparse visible patches. Continuous and differentiable Gaussian primitives naturally encode the geometry and shape consistency of anatomical boundaries. The pre-trained Gaussian decoder serves directly as a zero-shot initializer for 3DGS-CT reconstruction.

## Method

### Overall Architecture
MedGMAE follows the asymmetric encoder-decoder skeleton of MAE: a $96^3$ volume is split into 512 $12^3$ patches and masked at a 75% ratio. The ViT encoder processes only visible patches. The decoder introduces $k$ learnable "Gaussian query tokens," which attend to the semantics of visible patches to output 11-dimensional Gaussian parameters each. Finally, differentiable volume rendering is used to render these $k$ Gaussians back into a volume, and the MSE reconstruction loss is calculated only in the masked regions. An enhanced version, MedGMAE*, adds multi-level residual blocks for coarse-to-fine Gaussian densification for reconstruction tasks.

```mermaid
flowchart LR
    A[96³ Volume<br/>Split into 512 12³ patches] --> B[75% Masking]
    B --> C[ViT Encoder<br/>Visible patches only]
    C --> D[Concatenation: cls token +<br/>k Gaussian query tokens + visible tokens]
    D --> E[Transformer Decoder]
    E --> F[4 Parameter Heads<br/>μ/s/φ/I → k 3D Gaussians]
    F --> G[Differentiable Volume Rendering]
    G --> H[Masked Region MSE Loss]
    F -. Zero-shot Initialization .-> I[3DGR-CT Reconstruction]
```

### Key Designs

**1. Replacing voxels with 3D Gaussian primitives as the reconstruction target: Transforming "filling" into "geometric reasoning."** Each 3D Gaussian consists of a center position $\mu\in\mathbb{R}^3$, covariance $\Sigma=RSS^TR^T$ (decomposed into a scale vector $s\in\mathbb{R}^3$ and a rotation quaternion $\phi\in\mathbb{R}^4$), and intensity $I$, totaling 11 dimensions $g=\{\mu,s,phi,I\}$. The intensity at any spatial point $X$ is obtained by the weighted summation of neighboring Gaussians based on Mahalanobis distance decay: $V(X|g_i)=\sum_{i:\|X-\mu_i\|\le d_i} I_i\cdot e^{-\frac{1}{2}(X-\mu_i)^T\Sigma_i^{-1}(X-\mu_i)}$. This continuous, differentiable ellipsoidal representation naturally encodes the orientation, scale, and boundary continuity of anatomical structures, shifting the pre-training goal from local interpolation to global geometric abstraction. It uses only about 3e5 parameters to represent a volume originally containing 4e7 voxels (a 99% parameter reduction), fitting the sparse distribution of organs.

**2. Decoupled Gaussian query tokens: Complete separation of Gaussian quantity and mask count.** The decoder input is formed by concatenating three parts: $X_{dec}=\{\hat x_1\}\cup\{q_j\}_{j=1}^{k}\cup\{\hat x_i\}_{i=2}^{n}$ — the encoder class token, $k$ learnable Gaussian query tokens, and the remaining visible tokens. Crucially, $k$ (the number of predicted Gaussians) can be set independently of the number of masked patches, allowing flexible control over reconstruction granularity. Query tokens aggregate spatial-semantic information from visible patches via multi-head self-attention. Parameters are then output through four specialized linear heads (center, scale, rotation, intensity) with specific activations: position/intensity are constrained to $[0,1]$ via sigmoid, and rotation undergoes L2 normalization to ensure unit quaternions. Custom bias initializations are used to stabilize training (scale head bias = -1.386 $\rightarrow$ ~0.2 after sigmoid; intensity head bias = -0.405 $\rightarrow$ ~0.5), ensuring consistent scale distribution across spatial dimensions.

**3. Differentiable volume rendering loss for masked regions only: Connecting geometric parameters back to supervision signals.** After obtaining $k$ predicted Gaussians, a differentiable renderer accumulates the contributions of each Gaussian on the target volume grid to reconstruct the volume. The MSE is calculated only between the originally masked regions and the ground truth. This ensures that supervision comes only from occluded areas (forcing the model to "reason" rather than "copy" visible areas) and makes differentiable rendering of large-scale medical volumes computationally feasible through local aggregation (calculating only Gaussians within the influence radius $d_i$).

**4. MedGMAE* multi-level residuals: Coarse-to-fine densification for high-precision reconstruction.** Across three levels $l\in\{0,1,2\}$, Level 0 consists of $N_0$ base Gaussians, while Levels 1/2 expand to $m_1N_0$ and $m_2N_0$ Gaussians, respectively, with parameter dependencies between adjacent layers. Scale is forced to shrink monotonically: $s_l=s_0+\hat s_l\cdot\sigma_{scale}-\Delta s_l$ ($\sigma_{scale}=0.1$, $\Delta s_1=0.02, \Delta s_2=0.05$). Position, intensity, and rotation are refined as residuals (e.g., $\mu_l=\mu_0+\hat\mu_l\cdot\sigma_\mu$, with rotation re-normalized), and all residual heads use tanh clipping. This hierarchical densification allows coarse layers to handle overall shape while fine layers capture texture details, significantly improving the depiction of fine-grained structures in CT reconstruction.

**5. Decoder as a zero-shot geometric prior for 3DGR-CT reconstruction.** Since pre-training learns Gaussian parameters describing real anatomy, the trained Gaussian decoder can directly perform zero-shot inference on FBP initial reconstructions. The output Gaussian point cloud serves as the initialization for 3DGR-CT reconstruction, replacing random or heuristic initializations with those carrying anatomical priors, thus bridging self-supervised pre-training and downstream CT reconstruction.

## Key Experimental Results

Pre-training used AbdomenAtlas1.0Mini (5,195 CT cases). Downstream evaluation used UNETR as the backbone across four task categories: segmentation, classification, registration, and reconstruction.

### Main Results

Segmentation (DSC%, 4 datasets × 1%/10%/100% data, excerpt of 1% low-data scenario):

| Method | AMOS 1% | FLARE'22 1% | BTCV 1% | SegTHOR 1% |
|------|---------|-------------|---------|------------|
| SwinUNETR (Scratch) | 28.94 | 35.89 | 27.71 | 44.82 |
| MAE | 54.67 | 62.35 | — | 66.72 |
| HySparK | 34.50 | 37.54 | 35.81 | 58.81 |
| VoCo (Prev. SOTA) | 55.81 | 57.66 | **73.20** | 67.12 |
| **MedGMAE** | **58.79** | **62.72** | 66.19 | **70.92** |

Under 1% data, MedGMAE outperformed VoCo by 2.98% / 5.06% on AMOS / FLARE'22, and achieved a 20–35% Gain over the scratch baseline.

Classification (CT-RATE, AUC%) and Registration (DSC%):

| Task | Metric | Prev. Best | MedGMAE |
|------|------|--------|---------|
| Classification CT-RATE | AUC | SUP 76.04 | **76.40** |
| Registration IXI | DSC | VoCo 73.6 | **73.7** |
| Registration OASIS | DSC | VoCo 84.4 | **85.7** |

IXI/OASIS use MRI modalities not seen during pre-training, demonstrating cross-modal generalization.

Zero-shot initialization for CT reconstruction (AAPM-Mayo, excerpt of 120 projections):

| Method | Time(min) | iter(P=35) | PSNR(full) | SSIM(full) |
|------|-----------|-----------|------------|------------|
| 3DGR (Original) | 507±47.8 | 1660 | 45.2 | 98.7 |
| MedGMAE Init | 357±22.0 | 1040 | **46.2** | 98.5 |
| MedGMAE* Init | 335±20.4 | **920** | 45.8 | **98.7** |

Overall training time was reduced by 31–37%. Iterations required to reach PSNR=35 / SSIM=90% decreased by an average of 39.4% / 28.1%, achieving a 1.39× acceleration (t-test p<0.001).

### Ablation Study

Proxy task ablation (DSC%, comparing Voxel SSL vs. Gaussian SSL):

| Proxy Task | AMOS | FLARE'22 | SegTHOR |
|----------|------|----------|---------|
| None (Scratch) | 77.02 | 70.81 | 85.82 |
| Voxel SSL | 83.61 | 82.56 | 88.52 |
| **Gaussian SSL** | **84.90** | **83.77** | **89.15** |

### Key Findings
- Voxel SSL improves outcomes by 6–12% relative to training from scratch, while switching the proxy task to Gaussian reconstruction provides an additional 1–2% Gain, directly verifying the core hypothesis that "Gaussian representation is superior to voxel reconstruction."
- The gains are most significant in low-data (1%) scenarios, indicating that geometric priors are most valuable when annotations are scarce.
- The Gaussian decoder is no longer "use-and-discard"; as a zero-shot initializer, it substantially accelerates downstream CT reconstruction without sacrificing final quality.

## Highlights & Insights
- **Intermediate representation as a first-order problem**: Rather than making incremental changes to masking strategies or architectures, this work replaces the reconstruction target itself—using continuous Gaussian primitives to align with anatomical continuity, representing a paradigm-level rethinking of MIM proxy tasks.
- **Dual-purpose decoder**: The same pre-trained decoder serves both the encoder transfer and zero-shot initialization for 3DGS-CT reconstruction, solving the waste of "discarded decoders" in voxel MIM.
- **Sparsity as a prior**: By exploiting the characteristic that organs occupy only 11.8% of medical volumetric space, the natural sparsity of Gaussians is turned into a 99% parameter efficiency advantage.
- **Decoupling k from masking**: This design detail is crucial, as it allows for adjustable reconstruction granularity and leaves room for coarse-to-fine expansion.

## Limitations & Future Work
- CT reconstruction results are affected by noise in the FBP initial reconstruction; the authors suggest using multi-view 3D Gaussian foundation models to mitigate this.
- Experiments focused on CT (pre-training) + a small amount of MRI registration; the generalizability to more modalities/tasks (e.g., PET, Ultrasound) remains to be verified.
- Systematic analysis of hyperparameters like Gaussian count $k$ and influence radius $d_i$ regarding the tradeoff between rendering quality and computational cost is missing.
- Zero-shot initialization was only validated on the 3DGR-CT reconstruction pipeline; transferability to other 3DGS medical reconstruction pipelines is unclear.

## Related Work & Insights
- **MIM Lineage**: MAE pioneered efficient pre-training using 25% visible patches and a lightweight decoder. In the medical field, Models Genesis, HySparK, and VoCo have introduced variants in masking and architecture, but all remain locked into the "voxel-level reconstruction" objective. This paper provides an alternative path: "change the target instead of the architecture."
- **3DGS in Medicine**: 3D Gaussian Splatting has been used for CT, coronary, and 4D-CT reconstruction. While 2D GMAE uses Gaussian z-axis inference for 2.5D slices to achieve spatial understanding, this work uses true 3D Gaussians to represent real anatomical volumes, differing in both motivation and object.
- **Inspiration**: When a self-supervised proxy task's "reconstruction target" conflicts with domain structural priors, instead of stacking masking/architecture tricks, it may be better to switch to a continuous intermediate representation aligned with domain geometry—while simultaneously turning the decoder from "disposable" to a "reusable prior."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing 3D Gaussian primitives as a reconstruction target for medical MIM and reusing the decoder for zero-shot reconstruction initialization is a clear idea that hits the pain points of voxel MIM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four task types (segmentation, classification, registration, reconstruction), compares against over ten SSL baselines, uses multiple data scales, and includes cross-modal generalization and statistical significance; the ablation is concise but addresses the core.
- **Writing Quality**: ⭐⭐⭐⭐ The three motivations correspond well to the three advantages of the method. The structure is organized, charts are clear, though there are minor typos in some formulas and expressions.
- **Value**: ⭐⭐⭐⭐ Provides a new "geometric intermediate representation + transferable decoder" framework for medical image pre-training, achieving both practicality (accelerated CT reconstruction) and representation quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Masked-Diffusion Autoencoders for 3D Medical Vision Representation Learning](../../CVPR2026/medical_imaging/masked-diffusion_autoencoders_for_3d_medical_vision_representation_learning.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](../../AAAI2026/medical_imaging/multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[ICLR 2026\] Frequency-Balanced Retinal Representation Learning with Mutual Information Regularization](frequency-balanced_retinal_representation_learning_with_mutual_information_regul.md)
- [\[ICLR 2026\] Anatomy-aware Representation Learning for Medical Ultrasound](anatomy-aware_representation_learning_for_medical_ultrasound.md)
- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](../../CVPR2026/medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)

</div>

<!-- RELATED:END -->
