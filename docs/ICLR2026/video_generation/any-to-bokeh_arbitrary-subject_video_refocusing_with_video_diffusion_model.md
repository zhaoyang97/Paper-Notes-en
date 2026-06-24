---
title: >-
  [Paper Note] Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model
description: >-
  [ICLR2026][Video Generation][Video Refocusing] Any-to-Bokeh models video refocusing/bokeh rendering as a **single-step** video diffusion process guided by a focal-plane adaptive MPI geometric prior. It allows users to freely specify the focal plane and blur intensity for any input video, addressing temporal flickering via three-stage progressive training and weighted overlapping inference, outperforming previous image/MPI bokeh methods on both synthetic and real-world data.
tags:
  - "ICLR2026"
  - "Video Generation"
  - "Video Refocusing"
  - "Multi-Plane Image (MPI)"
  - "Single-step Diffusion"
  - "Focal Plane Control"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: 51bc301dd0fc24a4
---

# Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=h05AulYT7g](https://openreview.net/forum?id=h05AulYT7g)  
**Code**: TBD (Project page provided in paper)  
**Area**: Video Generation / Diffusion Models / Computational Photography  
**Keywords**: Video Refocusing, Multi-Plane Image (MPI), Single-step Diffusion, Focal Plane Control, Temporal Consistency

## TL;DR
Any-to-Bokeh models video refocusing/bokeh rendering as a **single-step** video diffusion process guided by a focal-plane adaptive MPI geometric prior. It allows users to freely specify the focal plane and blur intensity for any input video, addressing temporal flickering via three-stage progressive training and weighted overlapping inference, outperforming previous image/MPI bokeh methods on both synthetic and real-world data.

## Background & Motivation
**Background**: Diffusion models have successfully simulated camera geometric transformations (pan, zoom, tilt) and have recently been extended to **optical effect** simulation. Image-level bokeh rendering, leveraging generative priors, has produced visually realistic blur transitions.

**Limitations of Prior Work**: Existing works are largely restricted to **single images**. Applying image-bokeh methods frame-by-frame to video reveals two major flaws: first, the lack of explicit temporal modeling combined with imperfect frame-wise depth estimation leads to **temporal flickering** and unstable blur boundaries due to accumulated random noise in multi-step diffusion. Second, even when using large video generation models, their "bokeh-like" effects emerge implicitly, making it **impossible to explicitly control focal plane position and blur intensity**.

**Key Challenge**: Bokeh is essentially a process where the blur radius varies **non-linearly** with depth (especially at object boundaries). Existing MPI-based methods slice the scene at **fixed equal-interval depths**. This discretization does not match optical bokeh laws, resulting in inaccurate blur transitions and artifacts at boundaries. To achieve both controllability and temporal coherence, one must explicitly inject scene geometry while leveraging the strong 3D priors of pre-trained video models.

**Goal**: Construct the first diffusion framework for video bokeh that simultaneously satisfies: (i) temporal coherence, (ii) geometric/depth accuracy, and (iii) explicit control over the focal plane and blur intensity.

**Key Insight**: The authors observe that the bokeh blur radius depends only on the "disparity difference between the pixel and the focal plane" (Circle of Confusion formula). Instead of uniform slicing by absolute depth, disparity should be sampled **adaptively around the focal plane**—finely near the focal plane and coarsely further away—to match the non-linearity of the CoC.

**Core Idea**: Use a **focal-plane adaptive MPI representation** as a geometric prior to condition a **single-step** video diffusion model, complemented by progressive training and weighted overlapping inference for temporal stability.

## Method

### Overall Architecture
The pipeline processes an all-in-focus input video along with a user-specified focal plane and blur intensity $K$ to output a **depth-aware, temporally coherent, and controllable** bokeh video. It consists of three parts: first, a pre-trained depth estimator generates disparity maps per frame to construct a **focal-plane adaptive MPI mask** $M$ and a normalized disparity difference map $V_D$ based on the "near-focal refinement" threshold function. These, along with $K$, form the geometric priors. These priors are injected via **MPI spatial attention blocks** into a Stable Video Diffusion (SVD)-based **single-step** U-Net to predict bokeh frames. Training follows a three-stage progressive strategy (Geometry Guidance → Temporal Refinement → Detail Enhancement), while inference utilizes a **Weighted Overlapping Inference Strategy (WOIS)** to seamlessly process long videos.

```mermaid
mermaid
flowchart TD
    A["Input: All-in-focus Video<br/>+ Focal Plane + Blur Intensity K"] --> B["Focal-plane Adaptive MPI Representation<br/>Non-linear Slicing via CoC<br/>Yields Mask M, Disparity Map V_D"]
    B --> C["Single-step Video Bokeh Diffusion<br/>SVD U-Net Direct Prediction"]
    C --> D["MPI Spatial Block<br/>Gated Attention Injects Geometric Prior"]
    D --> E["Three-stage Progressive Training<br/>Geometry→Temporal→Detail"]
    E -->|Long Video Segmenting| F["Weighted Overlapping Inference (WOIS)<br/>Cosine Weight Fusion at Boundaries"]
    F --> G["Output: Controllable Bokeh Video"]
```

### Key Designs

**1. Focal-plane Adaptive MPI Representation: Matching Non-linear Optics**
Traditional MPI slices scenes into fixed layers, but the blur radius $r$ is not linear with depth. Following optical principles:

$$r = K\left|\tfrac{1}{z}-\tfrac{1}{z_f}\right| = K\,|d - d_f|,$$

where $K$ is the blur intensity, $d$ is disparity, and $d_f$ corresponds to the focal plane. Since $r$ changes rapidly near the shallow focal plane, the authors define disparity sampling thresholds as:

$$h_i = \left(\tfrac{i}{N}\right)\tfrac{1}{d_f},\quad i=1,\dots,N-1,$$

where $1/d_f \in (0,1]$ acts as a scaling factor for finer sampling at shallow focus. This yields the mask $M=\{m_i \mid |d(m_i)-d_f| < h_i\}$. Compared to fixed slicing, this representation is **relative to the focal plane**, naturally achieving focus-aware sampling that ensures accurate boundary transitions.

**2. Single-step Diffusion + MPI Spatial Block: Gated Injection**
Multi-step diffusion is slow and introduces inter-frame instability. The authors formulate bokeh generation as **single-step** diffusion, removing random sampling from SVD and using a single-step U-Net to predict output frames directly. The condition signals are: disparity difference $V_D$, intensity $K$, and mask $M$. 

**MPI Attention**, a gated attention mechanism inspired by GLIGEN, is integrated into the spatial blocks:

$$\hat{Q} = Q + \tanh(\gamma)\cdot \mathrm{TS}\!\left(\mathrm{Attn}\big([Q+\Phi_M(E(K)),\ \Phi_A(V_A)],\ \bar{M}\big)\right),$$

where $Q$ represents query tokens, $V_A$ visual tokens, and $\gamma$ is a learnable gate initialized to 0 to preserve pre-trained priors. The injection is **hierarchical**: shallow blocks refine local transitions with narrow masks, while deep blocks maintain global structure with wider masks.

**3. Three-stage Progressive Training**
Direct end-to-end training is difficult due to interference between temporal consistency and depth robustness. **Stage 1: Geometry Guidance** finetunes MPI spatial blocks and temporal modules on clean data to learn depth-aware blur. **Stage 2: Temporal Refinement** freezes spatial blocks and trains on longer sequences with **active disparity perturbations** (elastic transforms and Perlin noise) to teach the model to tolerate depth inaccuracies. **Stage 3: Detail Enhancement** finetunes the VAE decoder with a skip connection and a gradient-based texture loss:

$$L_t = \sum_{x,y}\big(\nabla_x\hat{V}_B - \nabla_x V_B\big)^2 + \big(\nabla_y\hat{V}_B - \nabla_y V_B\big)^2,$$

restoring high-frequency details lost during compression.

**4. Weighted Overlapping Inference Strategy (WOIS)**
To handle arbitrary lengths, WOIS segments video into $2L$-frame chunks with $L$-frame overlaps. The $j$-th frame in the overlap zone is fused using cosine weights:

$$\tilde{V}^i_B[j] = \gamma_j \hat{V}^i_B[j] + (1-\gamma_j)\hat{V}^{i+1}_B[j+L],\qquad \gamma_j = \tfrac{1}{2}\big(1+\cos(\tfrac{\pi j}{L})\big).$$

This suppresses boundary artifacts and ensures seamless transitions.

## Key Experimental Results

### Main Results
Evaluation used a synthetic set of 200 videos (ray-traced) and real-world sequences.

| Method | FD↓ | RM↓ | VFID-I↓ | FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ | VEPI↑ | Time↓ |
|------|-----|-----|---------|------|-------|-------|--------|-------|-------|
| DeepLens | 1.162 | 0.030 | 16.042 | 125.338 | 0.819 | 24.574 | 0.183 | 0.715 | 0.226 |
| BokehDiff | 0.660 | 0.021 | 7.395 | 65.678 | 0.834 | 27.525 | 0.127 | 0.859 | 0.799 |
| BokehMe | 0.536 | 0.013 | 8.633 | 39.102 | 0.936 | 27.992 | 0.060 | 0.937 | 0.103 |
| Dr.Bokeh | 0.522 | 0.011 | 6.097 | 32.710 | 0.950 | 31.273 | 0.046 | 0.863 | 2.729 |
| MPIB | 0.481 | 0.011 | 5.444 | 35.766 | 0.950 | 31.390 | 0.040 | 0.921 | 0.521 |
| **Ours** | **0.431** | **0.007** | **1.479** | **9.005** | **0.974** | **38.899** | **0.019** | **0.944** | 0.363 |

Ours leads across all metrics. FVD dropped significantly from 32.7 to 9.0, and PSNR increased by over 7dB compared to traditional MPI methods, with faster inference times than Dr.Bokeh.

### Ablation Study
Analysis of MPI, Single-step (OS), WOIS, and Temporal Refinement (TR):

| Config | MPI | OS | WOIS | TR | FD↓ | VFID-I↓ | FVD↓ | PSNR↑ |
|------|-----|----|------|----|----|---------|------|-------|
| #1 Full | ✓ | ✓ | ✓ | ✓ | 0.517 | 3.865 | 18.922 | 32.250 |
| #2 w/o TR | ✓ | ✓ | ✓ | - | 0.540 | 4.209 | 20.743 | 32.035 |
| #3 w/o WOIS+TR | ✓ | ✓ | - | - | 0.551 | 4.521 | 21.941 | 31.936 |
| #4 w/o MPI | - | ✓ | - | - | 0.573 | 4.930 | 23.828 | 31.551 |
| #5 w/o OS | ✓ | - | - | - | 0.791 | 8.912 | 68.910 | 29.309 |

### Key Findings
- **Single-step is critical**: Switching from multi-step to single-step (#5 $\rightarrow$ #3) reduced FVD from 68.9 to 21.9, proving that accumulated random noise is the primary cause of flickering in video bokeh.
- **MPI Spatial blocks improve quality**: Removing MPI (#4 vs #3) degraded both VFID and FVD.
- **Robustness via Perturbed Training**: Stage 2 perturbation allowed the model to maintain performance even under noisy disparity maps, whereas baseline MPI methods failed under high-noise scenarios.

## Highlights & Insights
- **Encoding Physical Laws into Representation**: Slicing MPI non-linearly based on CoC formulas is an elegant way to inject physical priors into network conditions.
- **Deterministic Mapping**: For refocusing, the randomness of multi-step diffusion is a liability. Formulating it as a deterministic single-step mapping addresses both speed and stability.
- **Gated Fine-tuning**: Using $\tanh(\gamma)$ initialized at 0 protects pre-trained priors during early fine-tuning stages.

## Limitations & Future Work
- **Reliance on Depth Estimators**: Despite improved robustness, performance is capped by the quality of the upstream depth estimation.
- **Synthetic Training Data**: Lack of real paired video data means generalization to complex real-world transparent/semi-transparent objects requires further validation.
- **Model Size**: With 1880M parameters, the model is significantly heavier than mobile-centric solutions like BokehMe.

## Related Work & Insights
- **vs Image Bokeh**: Previous works were limited to static images. This framework bridges the gap by combining optical simulation with pre-trained video priors.
- **vs Traditional MPI**: Conventional MPI uses fixed slicing and CUDA rendering, often suffering from color bleeding and lack of flexibility. This adaptive MPI is faster and more precise.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First dedicated diffusion framework for video bokeh.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive baselines and ablation, though limited by synthetic training data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, though some notation requires the appendix for full clarity.
- **Value**: ⭐⭐⭐⭐⭐ High utility for content creation and mobile post-processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing](dreamswapv_mask-guided_subject_swapping_for_any_customized_video_editing.md)
- [\[ICLR 2026\] MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement](magref_masked_guidance_for_any-reference_video_generation_with_subject_disentang.md)
- [\[ICLR 2026\] Arbitrary Generative Video Interpolation](arbitrary_generative_video_interpolation.md)
- [\[ICLR 2026\] TPDiff: Temporal Pyramid Video Diffusion Model](tpdiff_temporal_pyramid_video_diffusion_model.md)
- [\[ICLR 2026\] MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation](mtvcraft_tokenizing_4d_motion_for_arbitrary_character_animation.md)

</div>

<!-- RELATED:END -->
