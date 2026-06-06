---
title: >-
  [Paper Note] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator
description: >-
  [ICLR 2026][3D Vision][Text-to-3D] This paper proposes VIST3A, a framework that seamlessly connects the latent space of a pretrained video generator to a feed-forward 3D reconstruction model (e.g.…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Text-to-3D"
  - "Model Stitching"
  - "Video Generator"
  - "3D Reconstruction"
  - "3DGS"
  - "Direct Reward Finetuning"
  - "Point Map"
date: 2026-05-08
content_hash: fa0708762269e876
---

# Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator

**Conference**: ICLR 2026
**arXiv**: [2510.13454](https://arxiv.org/abs/2510.13454)  
**Code**: [Project Page](https://gohyojun15.github.io/VIST3A/)  
**Area**: 3D Vision / Generation
**Keywords**: Text-to-3D, Model Stitching, Video Generator, 3D Reconstruction, 3DGS, Direct Reward Finetuning, Point Map

## TL;DR
This paper proposes VIST3A, a framework that seamlessly connects the latent space of a pretrained video generator to a feed-forward 3D reconstruction model (e.g., AnySplat/MVDUSt3R/VGGT) via model stitching, and employs direct reward finetuning to align the generative model with the stitched 3D decoder. The approach enables high-quality end-to-end text-to-3DGS and text-to-pointmap generation, achieving state-of-the-art results on T3Bench, SceneBench, and DPG-Bench.

## Background & Motivation

**Background**: Text-to-3D generation has become an active research frontier. Early SDS-based methods (e.g., DreamFusion) require slow per-scene optimization; multi-stage pipelines (image generation followed by 3D lifting) suffer from error accumulation and engineering complexity; the latest trend is end-to-end latent diffusion models (LDMs) that directly generate 3D representations.

**LDM Approach**: These methods repurpose pretrained 2D image/video model priors, fine-tune them into multi-view latent generators, and train VAE-style decoders to decode latents into 3D representations such as 3DGS.

**Limitations of Prior Work — Weak Decoder**: Existing methods naively adapt 2D VAEs into 3D output decoders, effectively learning 3D reconstruction from scratch. This demands large amounts of training data and yields results far inferior to dedicated 3D foundation models (e.g., DUSt3R/VGGT/AnySplat). As 3D foundation models continue to improve, the performance gap of such self-trained decoders will only widen.

**Limitations of Prior Work — Weak Alignment**: The generative model and VAE decoder are trained separately. Generation losses (diffusion loss / flow matching) only indirectly promote 3D consistency, causing generated latents to potentially deviate from the decoder's input distribution and resulting in poor decoding quality. Even when rendering losses are added, they are computed only from single-step samples without accounting for the full denoising trajectory.

**Core Idea**: Rather than training a 3D decoder from scratch, directly reuse the strongest existing 3D foundation models as decoders via model stitching, and use reward finetuning to ensure that generated latents remain within the decoder's valid input domain.

## Method

### Overall Architecture
VIST3A consists of two core components: (1) **Model Stitching** to construct a 3D VAE, and (2) **Direct Reward Finetuning** to align the generative model with the stitched decoder.

### 1. Constructing a 3D VAE via Model Stitching

**Objective**: Stitch the encoder $\mathcal{E}$ of a video LDM with the latter portion of a feed-forward 3D reconstruction model $F$ to form a new 3D VAE.

**Step 1: Finding the Optimal Stitching Layer**
- Pass $N$ samples through the encoder to obtain latents $\mathbf{B} \in \mathbb{R}^{N \times D_\mathcal{E}}$
- Scan each layer $k$ of the 3D model and extract activations $\mathbf{A}_k \in \mathbb{R}^{N \times D_F^k}$
- Fit a linear stitching layer per layer using least squares: $\mathbf{S}^*_k = (\mathbf{B}^\top \mathbf{B})^{-1} \mathbf{B}^\top \mathbf{A}_k$
- Select the layer $k^\star$ with minimum MSE as the stitching point
- Key finding: Despite the two models being trained independently on different data, there exists a layer at which a linear transformation aligns the video latents with the 3D model activations with high fidelity.

**Step 2: Stitching and Fine-tuning**
- Assemble the stitched 3D VAE: $\mathcal{M}_{\text{stitched}} = F_{k^\star+1:l} \circ \mathbf{S} \circ \mathcal{E}(\mathbf{x})$
- The stitching layer is implemented as a 3D convolution; subsequent layers are updated via LoRA
- Self-supervised fine-tuning is performed using the original 3D model's outputs $\mathbf{y}$ as pseudo-labels with an $\ell_1$ loss
- No 3D annotation data is required

### 2. Direct Reward Finetuning for Alignment

**Problem**: At inference time, latents are generated from noise via the denoising loop rather than from the encoder — it is therefore necessary to ensure that generated latents also fall within the stitched decoder's valid input domain.

**Total Loss**: $L_{\text{total}} = L_{\text{gen}} - r(z_0(\theta, c, z_T), c)$

**The reward function comprises three components**:
1. **Multi-view Image Quality**: Decode latents into multi-view images using the original video decoder $\mathcal{D}$, and evaluate text alignment and visual quality via CLIP and HPSv2 scores.
2. **3D Representation Quality**: Decode latents into a 3D scene using the stitched decoder $\mathcal{D}_{\text{stitched}}$, render back to 2D, and similarly evaluate with CLIP and HPSv2 scores.
3. **3D Consistency**: Compare 2D images from the video decoder with renderings of the 3D scene from the same viewpoints, computing $\ell_1$ + LPIPS loss.

**Optimization Strategy**:
- Gradients are computed by unrolling the full denoising trajectory, with stable backpropagation inspired by DRTune.
- Random sampling of denoising timesteps and stochastic selection of gradient propagation steps improve computational efficiency.
- No ground-truth images are required; only text prompts are needed.

## Experiments

### Experimental Setup
- **3D Models**: MVDUSt3R (point map + 3DGS), VGGT (point map + depth + pose), AnySplat (3DGS + pose)
- **Video Generator**: Primarily Wan 2.1 T2V large; also tested with CogVideoX, SVD, and HunyuanVideo
- **Training Data**: DL3DV-10K + ScanNet (no 3D labels); prompts from the HPSv2 training set

### Text-to-3DGS Main Results (Table 1)

| Method | T3Bench Imaging↑ | T3Bench CLIP↑ | SceneBench Imaging↑ | SceneBench CLIP↑ |
|------|:---:|:---:|:---:|:---:|
| Matrix3D-omni | 43.05 | 25.06 | 46.65 | 24.04 |
| Director3D | 54.32 | 30.94 | 47.79 | 29.31 |
| SplatFlow | 46.09 | 29.48 | 48.85 | 29.43 |
| VideoRFSplat | 46.52 | 30.13 | 58.19 | 29.76 |
| **VIST3A: Wan+MVDUSt3R** | **58.83** | **32.75** | **62.08** | **30.26** |
| **VIST3A: Wan+AnySplat** | 57.03 | 31.38 | **64.87** | 30.18 |

VIST3A consistently outperforms all baselines on both object-level (T3Bench) and scene-level (SceneBench) synthesis. The margin is even larger on DPG-Bench: VIST3A achieves a Global score of 81.82 versus the best baseline score of 69.70.

### DPG-Bench Detailed Text Alignment Evaluation (Table 2)

| Method | Global↑ | Entity↑ | Attribute↑ | Relation↑ | Other↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| SplatFlow | 69.70 | 68.43 | 65.55 | 50.49 | 40.91 |
| VideoRFSplat | 36.36 | 56.93 | 66.89 | 48.53 | 31.82 |
| **VIST3A: Wan+MVDUSt3R** | **81.82** | **84.31** | **86.13** | 68.93 | **54.55** |
| **VIST3A: Wan+AnySplat** | 78.79 | **85.58** | 84.12 | **76.70** | 45.45 |

VIST3A demonstrates overwhelming advantages in understanding and following long-form text prompts, exceeding 80% on most dimensions.

### NVS Evaluation of Model Stitching (Table 3)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| SplatFlow | 19.10 | 0.671 | 0.278 |
| VideoRFSplat | 19.05 | 0.674 | 0.281 |
| AnySplat (original) | 20.85 | 0.695 | 0.238 |
| **Wan+AnySplat (stitched)** | 21.29 | 0.718 | 0.232 |

The NVS capability of AnySplat is not degraded but actually improved after stitching, as the video VAE latents provide richer appearance representations.

### User Study (Table 4)
28 participants ranked 14 samples (lower is better): VIST3A ranked first in both text alignment (1.54) and visual quality (1.45), and was rated best in >68% and >87% of cases, respectively.

## Highlights & Insights

1. **Novel Application of Model Stitching**: This work is the first to elevate model stitching from a tool for analyzing network representations to a practical technique for constructing 3D VAEs. It demonstrates that independently trained video VAEs and 3D reconstruction models possess linearly alignable intermediate representations, and that stitching preserves nearly all capabilities of the original 3D model.

2. **Plug-and-Play Framework**: VIST3A flexibly combines different video generators (Wan/CogVideoX/SVD/Hunyuan) with different 3D models (AnySplat/MVDUSt3R/VGGT), consistently yielding significant improvements across all combinations.

3. **Elegant Design of Direct Reward Finetuning**: The three reward components respectively target 2D visual quality, 3D geometric quality, and cross-modal consistency, with no ground-truth annotation required throughout.

4. **Unlocking New Capabilities**: Beyond text-to-3DGS, the framework also enables the novel task of text-to-pointmap, and can generate coherent large-scale scenes without requiring training on long sequences.

## Limitations & Future Work

1. **Dependence on Base Model Quality**: The upper bound of generation quality is constrained by the individual capabilities of the chosen video and 3D models — stitching cannot compensate for fundamental deficiencies in either base model.
2. **Linearity Assumption in Stitching**: Finding the stitching point assumes that the optimal mapping between the two representations is linear. If the representational difference between the two models is nonlinear (e.g., due to highly disparate architectures), stitching quality may degrade.
3. **Computational Cost of Reward Alignment**: Direct reward finetuning requires unrolling the full denoising trajectory and rendering 3D scenes to compute rewards, incurring substantial memory and computational overhead.
4. **Dynamic Scenes**: The framework targets static scene generation and does not address dynamic 3D content such as 4D generation.

## Related Work & Insights

- **SDS Optimization**: DreamFusion, Magic3D, ProlificDreamer — per-scene optimization, slow convergence
- **Multi-stage Pipelines**: Zero-1-to-3, MVDream → SV3D/DreamView → 3D lifting — error accumulation
- **End-to-end LDM**: SplatFlow, Director3D, Prometheus3D, VideoRFSplat — weak self-trained decoders, poor alignment
- **Feed-forward 3D Reconstruction**: DUSt3R → MASt3R → MVDUSt3R → VGGT → AnySplat → Pi3 — increasingly powerful 3D foundation models
- **Model Stitching**: Lenc & Vedaldi (2015), Bansal et al. (2021), DeRy, SN-Net — this work is the first to apply stitching to 3D VAE construction

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐⭐: The idea of applying model stitching to construct a 3D VAE is highly elegant, avoiding training a decoder from scratch while leveraging state-of-the-art 3D models.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Three benchmarks, a user study, cross-model combination experiments, and detailed ablations — extremely comprehensive.
- **Writing Quality** ⭐⭐⭐⭐: Problem motivation is clear, the method is described completely, figures are well-designed, and the paper is well-structured.
- **Value** ⭐⭐⭐⭐: The plug-and-play framework can directly benefit from future advances in both video generation and 3D reconstruction models.
- **Reproducibility** ⭐⭐⭐: A project page is provided, but the extent of code release is yet to be confirmed; LoRA and reward finetuning details are given in the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](../../CVPR2026/3d_vision/gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)
- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](../../CVPR2026/3d_vision/coherent_humanscene_reconstruction_from_multiperso.md)
- [\[CVPR 2026\] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](../../CVPR2026/3d_vision/forgedreamer_industrial_text-to-3d_generation_with_multi-expert_lora_and_cross-v.md)
- [\[ICLR 2026\] StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams](streamsplat_towards_online_dynamic_3d_reconstruction_from_uncalibrated_video_str.md)

</div>

<!-- RELATED:END -->
