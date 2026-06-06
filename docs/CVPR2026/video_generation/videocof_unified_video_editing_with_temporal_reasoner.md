---
title: >-
  [Paper Note] VideoCoF: Unified Video Editing with Temporal Reasoner
description: >-
  [CVPR 2026][Video Generation][Video Editing] This paper proposes VideoCoF, a Chain-of-Thought-inspired "observe→reason→edit" video editing framework. By prompting a video diffusion model to first predict reasoning tokens…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Video Editing"
  - "Chain-of-Frames"
  - "Video Diffusion Model"
  - "Reasoning Frames"
  - "Length Extrapolation"
date: 2026-05-08
content_hash: 9b0b459b67e9db89
---

# VideoCoF: Unified Video Editing with Temporal Reasoner

**Conference**: CVPR 2026
**arXiv**: [2512.07469](https://arxiv.org/abs/2512.07469)  
**Code**: [https://github.com/knightyxp/VideoCoF](https://github.com/knightyxp/VideoCoF)  
**Area**: Diffusion Models / Video Editing
**Keywords**: Video Editing, Chain-of-Frames, Video Diffusion Model, Reasoning Frames, Length Extrapolation

## TL;DR

This paper proposes VideoCoF, a Chain-of-Thought-inspired "observe→reason→edit" video editing framework. By prompting a video diffusion model to first predict reasoning tokens (grayscale-highlighted region latents) before generating the target video tokens, VideoCoF achieves precise instruction-region alignment without requiring user-provided masks. Trained on only 50K video pairs, it achieves state-of-the-art performance and supports video extrapolation up to 16× the training length.

## Background & Motivation

1. **Background**: Existing video editing methods fall into two categories — expert models (adapter + external mask, precise but task-specific and dependent on additional inputs) and unified temporal in-context learning models (concatenating source video tokens with noisy edit tokens along the temporal axis, mask-free but lacking explicit spatial cues).

2. **Limitations of Prior Work**: Unified models suffer from weak instruction-region alignment due to the absence of explicit spatial guidance, resulting in poor accuracy in multi-instance recognition or spatial reasoning scenarios. Expert models, while precise, require user-provided masks or task-specific training and cannot handle diverse editing tasks in a unified manner.

3. **Key Challenge**: A fundamental trade-off between precision and unification — can a model simultaneously achieve the localization accuracy of expert models and the mask-free convenience of unified models?

4. **Goal**: (1) How to achieve precise editing region localization without mask inputs; (2) How to handle multi-instance editing tasks within a unified framework; (3) How to generalize at inference time to videos longer than the training length.

5. **Key Insight**: Drawing an analogy to Chain-of-Thought reasoning in LLMs, the paper advocates for "visual chain-of-reasoning" in video generation — predicting editing regions before executing edits. Prior work has demonstrated that video diffusion models (VDMs) possess inherent reasoning capabilities (e.g., solving visual puzzles), which can be elicited by explicitly modeling reasoning tokens.

6. **Core Idea**: By inserting "reasoning frames" (grayscale-highlighted editing region latents) between source and edited video sequences, VideoCoF forces the diffusion model to "observe, then reason, then act," enabling mask-free precise video editing.

## Method

### Overall Architecture

VideoCoF is built upon a VideoDiT backbone (e.g., WAN-14B) as a unified video editing framework. The input consists of a source video and a text editing instruction; the output is the edited video. The process is divided into three stages: the source video is first encoded into latents as the "observation" basis; the model then predicts reasoning latents (grayscale-highlighted frames marking editing regions) as the "reasoning" step; finally, the edited video latents are generated conditioned on the reasoning output. The three groups of latents are concatenated along the temporal dimension into a unified sequence $\mathbf{z}_{full}$, processed jointly by VideoDiT through self-attention (in-context learning) and cross-attention (language control). During training, noise is applied only to the reasoning frames and target frames, with supervision on the velocity field prediction.

### Key Designs

1. **Chain of Frames (CoF) Reasoning Mechanism**:

    - **Function**: Precisely localizes editing regions without mask inputs.
    - **Mechanism**: Given a source-reasoning-target video triplet $\{\mathbf{s}, \mathbf{r}, \mathbf{e}\}$, each component is encoded into latents $z_s, z_r, z_e$ and concatenated along the temporal dimension. During training, the source video latent remains clean (timestep=0), while the reasoning and target frames are jointly noised and treated as denoising targets. The ground truth for reasoning frames consists of grayscale semi-transparent highlights marking the editing regions. The model is thereby compelled to first learn the mapping from instruction to editing region before executing the edit. A progressive gray mask format (transparency gradually increasing from 0% to 75%) yields the best results, as it provides a smooth transition from source to edited video.
    - **Design Motivation**: Prior temporal in-context learning methods (e.g., ICVE, UNIC) directly concatenate source and noisy target without explicitly constraining instruction-region alignment, leading to poor editing precision. CoF addresses this by enforcing an intermediate reasoning step, enabling the model to actively learn the correspondence between editing instructions and target regions.

2. **RoPE Alignment Strategy (Length Extrapolation)**:

    - **Function**: Supports video lengths at inference time far exceeding the training length (up to 16× extrapolation) while maintaining motion alignment.
    - **Mechanism**: The original VideoDiT employs 3D decomposed RoPE for spatiotemporal positional encoding. A naive concatenation scheme assigns source video indices $[0, F-1]$ and target indices $[F, 2F-1]$, causing the model to overfit to fixed mappings and fail to extrapolate. Simply repeating indices causes index collisions (e.g., the 0th source frame, reasoning frame, and 0th target frame all share temporal index=0, producing visual artifacts). The proposed design assigns temporal indices $[1, F]$ to both the source and target videos, and index $0$ to the reasoning frames. This isolates reasoning tokens at a unique temporal position without colliding with any video frame, ensures consistent index ranges between source and target for motion alignment, and allows $F$ to be freely extended at inference for length extrapolation.
    - **Design Motivation**: This design resolves two issues — (1) naive sequential indexing $[0, 2F-1]$ causes positional encoding to overfit to the training length, preventing extrapolation; (2) index collisions cause reasoning tokens to interfere with the editing results of the first frame. Experiments demonstrate that the proposed design enables extrapolation from 33 training frames to 141 frames (4×) and even 513 frames (16×).

3. **Instance-Level Data Augmentation Pipeline**:

    - **Function**: Generates training triplets for complex multi-instance editing scenarios.
    - **Mechanism**: Diverse videos are collected from Pexels; Qwen-VL 72B performs multi-instance recognition; Grounding-SAM2 precisely segments each instance; Minimaxremover handles object removal/addition while VACE-14B in inpainting mode handles replacement/local style transfer. GPT-4o generates creative editing prompts. Training samples are filtered by Dover Score and VIE Score for quality, supplemented by a high-quality subset distilled from the Señorita 2M dataset, yielding 50K training samples in total.
    - **Design Motivation**: Existing video editing datasets predominantly cover single-instance simple operations and do not support complex spatial relationships (physical left/right, multi-instance interactions). Multi-instance data is essential for training the model's spatial reasoning capabilities.

### Loss & Training

Training adopts a Flow Matching objective with velocity field $\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{z}_{full}^{(0)}$, applying MSE loss only on reasoning and target frames: $\mathcal{L} = \frac{1}{L+F}\sum_{i=F}^{2F+L-1}\|\mathbf{v}_i - \hat{\mathbf{v}}_i\|_2^2$. At inference, an ODE solver evolves from Gaussian noise to clean latents while source latents remain fixed. Combined with DMD-LoRA, only 4 denoising steps are required, enabling editing of 33 frames in approximately 10 seconds on a single H100.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on VideoCoF-Bench (200 videos, 4 editing task categories, including instance-level editing):

| Method | Instruct Follow↑ | Preservation↑ | Quality↑ | Success Ratio↑ | CLIP-T↑ |
|------|-------------------|---------------|----------|----------------|---------|
| ICVE (1M pretrain+150K finetune) | 7.79 | 8.06 | 8.14 | 57.76% | 27.49 |
| VACE-14B | 7.47 | 5.82 | 7.61 | 26.60% | 27.02 |
| Lucy Edit | 5.24 | 6.50 | 6.37 | 29.64% | 26.98 |
| **VideoCoF (50K)** | **8.97** | **8.20** | **7.77** | **76.36%** | **28.00** |

Using only 50K training samples, VideoCoF surpasses ICVE (trained on 1M+ data) across all GPT-4o evaluation metrics, with a Success Ratio improvement of 18.6%.

### Ablation Study

| Configuration | Instruct Follow | Success Ratio | CLIP-T |
|------|----------------|---------------|--------|
| Naive temporal [0,2F-1] w/o CoF | 8.11 | 72.41% | 26.88 |
| Repeated index [0,F-1] w/o CoF | 8.06 | 65.52% | 27.09 |
| **VideoCoF [1-F,0,1-F] + CoF** | **8.97** | **76.36%** | **28.00** |

Reasoning frame format ablation:

| Format | Instruct Follow | Success Ratio |
|------|----------------|---------------|
| Black mask (0%) | 7.51 | 52.17% |
| Red mask (50%) | 7.81 | 60.33% |
| Gray mask (50%) | 8.15 | 68.45% |
| **Progressive gray (0–75%)** | **8.97** | **76.36%** |

### Key Findings

- Introducing CoF reasoning frames yields +10.65% in Instruct Follow and +5.46% in Success Ratio, demonstrating that explicit reasoning steps are critical for editing precision.
- The RoPE alignment design enables extrapolation from 33 training frames to 513 frames (16×); the naive scheme degrades severely at 81 frames (blurring, motion misalignment).
- Among reasoning frame formats, progressive gray masks substantially outperform black/red variants, as diffusion models are insensitive to pure black/white pixels and grayscale highlights are more suitable for latent space representation.
- Surpassing methods trained on 1M+ data with only 50K samples demonstrates that data quality and framework design are far more important than data volume.

## Highlights & Insights

- **Chain-of-Frames Reasoning Paradigm**: An elegant transfer of Chain-of-Thought reasoning from language to visual generation. The "observe→reason→edit" process naturally mirrors the human workflow of identifying editing regions before executing operations, and can be extended to image editing and 3D scene editing.
- **RoPE Index Isolation Strategy**: A single index offset (reasoning frame=0, video=$[1,F]$) simultaneously resolves both index collision and length extrapolation, demonstrating a remarkably clean and generalizable design applicable to any diffusion model that concatenates heterogeneous token sequences.
- **Data Efficiency**: The fact that 50K samples outperform 1M+ demonstrates that structured learning signals (editing region supervision via reasoning frames) are more effective than brute-force data scaling.

## Limitations & Future Work

- The ground truth for reasoning frames depends on the segmentation quality of Grounding-SAM2, which may introduce noise in failure cases.
- The current static grayscale highlighting of reasoning frames cannot adequately represent editing regions that change across frames (e.g., trajectory modifications).
- Although data-efficient, the 50K training set has limited diversity and may provide insufficient coverage of complex natural scenes.
- Attention visualization to verify whether reasoning frames genuinely drive region-specific attention in the model has not been explored.

## Related Work & Insights

- **vs. ICVE**: ICVE adopts naive temporal concatenation for unified video editing, requiring 1M pretraining + 150K fine-tuning but lacking explicit spatial guidance. VideoCoF addresses the spatial precision gap through CoF reasoning frames, surpassing ICVE with only 50K training samples.
- **vs. VACE**: VACE is a powerful video editing foundation model, but its inpainting mode requires mask inputs. VideoCoF improves editing precision over VACE's mask-free unified framework through the introduction of reasoning frames.
- **vs. EditVerse**: EditVerse also explores unified in-context learning but is based on a LLaMA-style DiT. VideoCoF achieves similar functionality on a standard video diffusion model, making it more broadly applicable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Chain-of-Frames represents the first exploration of CoT-style reasoning in video diffusion models, establishing a new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive (CoF, RoPE, reasoning frame format), though evaluation is primarily conducted on a self-constructed benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ The method is clearly articulated; the CoT analogy provides a compelling narrative, and the figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ The reasoning frame + RoPE alignment design is broadly transferable to other visual generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](../../ICML2026/video_generation/lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)

</div>

<!-- RELATED:END -->
