---
title: >-
  [Paper Note] Turbo3D: Ultra-Fast Text-to-3D Generation
description: >-
  [CVPR 2025][3D Vision][Text-to-3D Generation] Turbo3D compresses a multi-step multi-view diffusion model into a 4-step generator via dual-teacher distillation and introduces a latent space GS-LRM reconstructor. It generates high-quality 3D Gaussian Splatting assets from text in just 0.35 seconds on a single A100, while outperforming existing methods on CLIP Score and VQA Score.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Text-to-3D Generation"
  - "Diffusion Distillation"
  - "Multi-view Generation"
  - "Gaussian Splatting"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 3dee377a3142ec49
---

# Turbo3D: Ultra-Fast Text-to-3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.04470](https://arxiv.org/abs/2412.04470)  
**Code**: [https://turbo-3d.github.io/](https://turbo-3d.github.io/)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D Generation, Diffusion Distillation, Multi-view Generation, Gaussian Splatting, Efficient Inference

## TL;DR

Turbo3D compresses a multi-step multi-view diffusion model into a 4-step generator via dual-teacher distillation and introduces a latent space GS-LRM reconstructor. It generates high-quality 3D Gaussian Splatting assets from text in just 0.35 seconds on a single A100, while outperforming existing methods on CLIP Score and VQA Score.

## Background & Motivation

**Background**: The field of 2D image generation has achieved extremely fast inference speeds (e.g., single-step/few-step generation), but 3D generation remains slow. Current text-to-3D methods are primarily categorized into two types: optimization-driven methods (e.g., SDS) which require minutes to hours, and feed-forward generation methods (e.g., Instant3D, LGM) which, although faster, still require seconds to dozens of seconds for inference, with limited quality.

**Limitations of Prior Work**: Multi-view diffusion models fine-tuned on synthetic data (Objaverse) suffer from generation quality limited by the style bias of synthetic data, resulting in over-simplified and cartoonish appearances. Directly distilling the multi-view teacher model leads to "compounding mode collapse," where the dual quality losses from fine-tuning and distillation accumulate, causing the generated results to further deviate from photorealistic styles.

**Key Challenge**: There is a severe trade-off between inference efficiency and generation quality. Distillation can significantly boost speed but severely damages multi-view consistency and photorealism.

**Goal**: (1) Efficiently distill multi-step multi-view diffusion models into few-step generators while maintaining quality; (2) Further optimize 3D reconstruction efficiency by eliminating unnecessary decoding steps.

**Key Insight**: The authors observe that the fundamental cause of mode collapse is the reliance on a single multi-view teacher during distillation, which itself is already biased toward synthetic data styles. Introducing a single-view teacher trained on large-scale, high-quality real images can compensate for the lack of photorealism.

**Core Idea**: Address distillation quality degradation through dual-teacher distillation (a multi-view teacher to teach consistency + a single-view teacher to teach photorealism), and migrate the reconstructor from pixel space to latent space to eliminate VAE decoding overhead.

## Method

### Overall Architecture

Turbo3D is a two-stage pipeline: first, a 4-step multi-view latent-space generator produces latent representations of 4 viewpoints from a text prompt; then, a latent-space GS-LRM directly reconstructs 3D Gaussian Splatting representations from these multi-view latents. The entire pipeline takes only 0.35 seconds on a single A100 GPU.

### Key Designs

1. **Dual-Teacher Distillation**:

    - **Function**: Distill multi-step multi-view diffusion models into a 4-step fast generator while maintaining multi-view consistency and photorealism.
    - **Mechanism**: Introduce two teachers within the Distribution Matching Distillation (DMD) framework. The multi-view teacher (MV Teacher) teaches multi-view consistency to the student model by jointly computing DMD losses across all views. The single-view teacher (SV Teacher) independently computes DMD loss for each view, pulling the generation quality of each view toward the natural image distribution. The final loss is a weighted combination of both: $L_{\text{DMD}}^{\text{Dual}} = D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}}^{\text{MV}}) + \lambda \cdot \frac{1}{K}\sum_{i=1}^{K} D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}}^{\text{SV}})$, where $\lambda=1$, $K=4$.
    - **Design Motivation**: Distillation with only a multi-view teacher leads to severe compounding mode collapse—the MV teacher already loses some photorealism when fine-tuned on Objaverse, and distillation further amplifies this issue. The SV teacher, trained on large-scale, high-quality natural images, effectively "pulls" each view back to the natural image distribution.

2. **Latent-Space GS-LRM (Latent GS-LRM)**:

    - **Function**: Reconstruct 3D Gaussians directly from multi-view latent representations, skipping the VAE decoding step.
    - **Mechanism**: Change the input of GS-LRM from pixel space to latent space. Since the multi-view generator outputs latents (rather than pixels), directly passing latents to the reconstructor eliminates the computational overhead of VAE decoding, while also halving the sequence length of the transformer (as the latent resolution is 1/8 of the original image). During training, supervision is still performed using novel-view rendering loss in pixel space (L2 + perceptual loss).
    - **Design Motivation**: Conv2D operations in the VAE decoder are highly inefficient at high resolutions. Skipping the decoding step directly yields approximately 22% speedup without compromising reconstruction quality.

3. **Plücker Coordinate Embedding**:

    - **Function**: Inject explicit 3D camera-aware information into the student model.
    - **Mechanism**: Incorporate Plücker ray embeddings as an additional condition in the student multi-view generator, enabling the generator to better comprehend spatial relationships across different viewpoints.
    - **Design Motivation**: Enhance the 3D consistency perception of the distilled model, compensating for viewpoint understanding that might be lost during distillation.

### Loss & Training

Training consists of three stages: (1) Fine-tuning a DiT-based T2I model into a multi-step multi-view diffusion model on Objaverse (30K iterations, 32 A100); (2) Dual-teacher distillation to train the few-step generator (10K iterations, 32 A100); (3) Training the latent-space GS-LRM reconstructor from scratch (80K iterations, 32 A100). The dataset comprises approximately 400K Objaverse instances paired with Cap3D text annotations.

## Key Experimental Results

### Main Results

| Method | CLIP Score ↑ | VQA Score ↑ | Inference Time ↓ |
|------|-------------|------------|----------|
| TripoSR | 23.85 | 0.57 | 1.19s |
| SV3D | 24.92 | 0.64 | 12.52s |
| Instant3D | 26.23 | 0.65 | 15.02s |
| LGM | 24.73 | 0.58 | 6.56s |
| **Turbo3D** | **27.61** | **0.76** | **0.35s** |

### Ablation Study

| Configuration | CLIP Score ↑ | VQA Score ↑ | Description |
|------|-------------|------------|------|
| Multi-step MV Model (Teacher) | 28.04 | 0.77 | Full teacher model, slow speed |
| Few-step Model (MV Teacher Distill Only) | 26.60 | 0.69 | Single-teacher distillation, significant quality drop |
| Few-step Model (Dual-Teacher Distill) | 27.61 | 0.76 | Dual-teacher effectively restores quality |
| Pixel GS-LRM | 27.62 / 0.76 | - | 0.45s |
| Latent GS-LRM | 27.61 / 0.76 | - | 0.35s, 22% faster |

### Key Findings

- The effect of dual-teacher distillation is significant: compared to using MV teacher distillation alone, CLIP Score improves from 26.60 to 27.61, and VQA Score increases from 0.69 to 0.76, nearly matching the teacher model.
- Latent GS-LRM reduces inference time from 0.45s to 0.35s without sacrificing quality.
- In the user study, Turbo3D achieves a win rate of 89.8% against LGM, 74.9% against Instant3D, and 50.6% against the MV teacher model, indicating that distillation preserves the teacher's generation capability with virtually no loss.
- The distilled model is approximately 50x faster than the teacher model.

## Highlights & Insights

- The **dual-teacher distillation framework** is highly ingenious: by introducing a single-view teacher to compensate for the multi-view teacher's lack of realism, it addresses compounding mode collapse from a "complementary" perspective. This idea can be migrated to any scenario involving domain transfer distillation.
- The idea of **latent-space reconstruction** is very practical: since the generator output is already latent, there is no need to decode before encoding it back for the reconstructor. Transferring directly in the latent space saves time while preserving information. This "eliminating intermediate steps" mindset is highly worth adopting in other pipelines.
- The engineering optimization of the entire system is well-executed: 4-step generation + 1-step reconstruction completes text-to-3D generation end-to-end in 0.35 seconds.

## Limitations & Future Work

- The training data is limited to 400K instances of Objaverse, so the generation diversity and realism are constrained by this relatively limited 3D dataset.
- The generated 3D assets are represented as Gaussian Splatting, and the system does not directly output meshes or other more general 3D formats.
- It is worth exploring whether the 4-step multi-view generation can be further compressed into 1–2 steps, or whether the resolution can be improved while maintaining quality.
- Currently, only object-centric generation is supported, with limited support for complex scenes.

## Related Work & Insights

- **vs Instant3D**: Both adopt the multi-view generation + reconstruction paradigm, but Instant3D requires 15 seconds of inference, whereas Turbo3D is approximately 40x faster. Instant3D's text alignment capability is also weaker than Turbo3D's.
- **vs LGM**: LGM is prone to Janus problems and unstable quality, problems which Turbo3D avoids by utilizing a multi-view diffusion model.
- **vs GECO**: A concurrent work that also uses diffusion distillation for acceleration; however, GECO relies on cumbersome mesh reconstruction for 3D distillation, while Turbo3D's pipeline is more concise.

## Rating

- Novelty: ⭐⭐⭐⭐ Dual-teacher distillation is the core innovation, and latent-space reconstruction is a natural yet effective optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative + qualitative + user studies + ablations are all thorough.
- Writing Quality: ⭐⭐⭐⭐ Clean motivation, concise and clear method description.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value by pushing 3D generation speeds into the sub-second regime.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](../../ECCV2024/3d_vision/tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](../../ICCV2025/3d_vision/benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)

</div>

<!-- RELATED:END -->
