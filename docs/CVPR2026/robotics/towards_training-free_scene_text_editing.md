---
title: >-
  [Paper Note] Towards Training-Free Scene Text Editing
description: >-
  [CVPR 2026][Robotics][Scene Text Editing] This paper proposes TextFlow, a training-free scene text editing framework that employs Flow Manifold Steering (FMS) during the early denoising stage to preserve style consistenc…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Scene Text Editing"
  - "Training-Free"
  - "Diffusion Models"
  - "Attention Enhancement"
  - "Flow Matching"
date: 2026-05-08
content_hash: fb35f928d35e330f
---

# Towards Training-Free Scene Text Editing

**Conference**: CVPR 2026
**arXiv**: [2603.24571](https://arxiv.org/abs/2603.24571)  
**Code**: [https://github.com/lyb18758/TextFlow](https://github.com/lyb18758/TextFlow)  
**Area**: Robotics
**Keywords**: Scene Text Editing, Training-Free, Diffusion Models, Attention Enhancement, Flow Matching

## TL;DR

This paper proposes TextFlow, a training-free scene text editing framework that employs Flow Manifold Steering (FMS) during the early denoising stage to preserve style consistency and Attention Boost (AttnBoost) during the late stage to enhance text rendering accuracy, achieving editing quality comparable to or better than training-based methods without any task-specific training.

## Background & Motivation

1. **Background**: Scene Text Editing (STE) aims to modify or replace text content in natural images while preserving background and visual attributes of the original text (font, color, size, and geometric layout). Generative models have evolved from GANs to UNet-based diffusion models and Diffusion Transformers (DiT), driving advances in STE. Methods such as DiffSTE, AnyText, and textFlux have demonstrated strong text rendering performance.

2. **Limitations of Prior Work**: A fundamental trade-off exists between adaptability and editing quality. Training-based methods require large-scale, high-quality paired data—which is scarce in practice—and while synthetic data can supplement this, it limits generalization to diverse real-world scenes, with substantial computational costs. Training-free methods leverage pretrained models without fine-tuning, but most attention-manipulation-based approaches are designed for generic object editing and struggle to maintain precise typographic and structural detail in the STE setting, frequently producing character repetition, omission, or distortion.

3. **Key Challenge**: The fundamental difficulty of training-free methods lies in stage-dependent controllability discrepancy—the signal-to-noise ratio varies unevenly across diffusion timesteps. If structural and stylistic foundations are not established during the early denoising stage, the editing trajectory becomes unstable; if sufficient semantic and spatial guidance is absent in the late stage, text rendering becomes inaccurate.

4. **Goal**: How can both style preservation and text accuracy—the two core challenges in STE—be addressed simultaneously without any training?

5. **Key Insight**: Decouple the complex STE task into two complementary stages, each handled by a dedicated mechanism—style preservation in the early stage and text accuracy enhancement in the late stage.

6. **Core Idea**: Decompose STE into a two-stage process—FMS steers the trajectory in latent space via geometric correction in the early stage to maintain style consistency, while AttnBoost enhances text rendering accuracy through attention map guidance in the late stage, enabling end-to-end training-free editing.

## Method

### Overall Architecture

TextFlow is built upon the flow matching architecture of FLUX-Kontext. Given a source image and source/target text descriptions as input, the FMS module encodes the source image into latent space during the first half of the denoising process, constructs joint representations of source and target via noise injection and differential geometric transformation, and computes velocity field differentials to correct the editing trajectory and preserve style. In the second half, the AttnBoost mechanism extracts and enhances text-to-image attention maps from the dual-stream transformer blocks of DiT, generating fine-grained guidance signals to improve text rendering accuracy. The entire process is completed in 50 denoising steps without any fine-tuning.

### Key Designs

1. **Flow Manifold Steering (FMS)**:
    - **Function**: Maintains structural and style consistency between the edited image and the source image during the early denoising stage.
    - **Mechanism**: A noise-injected source latent representation is first constructed via linear interpolation: $\mathbf{z}_t^{src} = (1-t_i) \cdot \mathbf{z}_{src} + t_i \cdot \epsilon$. The target latent representation is then corrected through a differential geometric transformation: $\mathbf{z}_t^{tar} = \mathbf{z}_t + (\mathbf{z}_t^{src} - \mathbf{z}_{src})$, ensuring structural alignment between the target trajectory and the source. The source and target representations are each concatenated with the current state and fed into parallel DiT blocks. The velocity field differential is computed as $\mathbf{V}_\Delta = \Phi(z_t^{tar,cat}, e_p^{tar}) - \Phi(z_t^{src,cat}, e_p^{src})$, and the edited result is obtained via trajectory displacement: $\mathbf{z}_{edit} = \mathbf{z}_t + \mathbf{V}_\Delta \cdot (t_{i-1} - t_i)$.
    - **Design Motivation**: The differential term $(\mathbf{z}_t^{src} - \mathbf{z}_{src})$ precisely captures the geometric offset induced by noise injection, embedding structural constraints from the source into the generation trajectory. Ablation studies show that removing FMS reduces PSNR by 1.95 and increases MSE by 39.2%.

2. **Attention Boost (AttnBoost)**:
    - **Function**: Enhances the rendering accuracy of text content during the late denoising stage.
    - **Mechanism**: Text-to-image attention patterns are extracted from the self-attention of dual-stream transformer blocks. Target amplification is first applied to attention within the text region: $A_{enhanced}(b,h,q,k) = \mathcal{T}(A(b,h,q,k))$ (applied only within the index range of text tokens). Text-to-image attention maps $A_{t2i}$ are then extracted, aggregated along the query dimension, spatially pooled, and normalized to produce a guidance signal $\hat{A}$. The normalized attention guidance is finally integrated into the scheduler: $z_{t-1} = \mathcal{S}(z_t, \hat{A}, t)$.
    - **Design Motivation**: The late denoising stage is critical for rendering fine-grained text details. By amplifying attention weights in text-relevant regions, the model focuses more effectively on accurate character generation. Ablation studies show that removing AttnBoost causes text accuracy to drop sharply from 79.80% to 20.35%.

3. **Two-Stage Decoupling Strategy**:
    - **Function**: Leverages the distinct characteristics of different denoising stages to separately optimize style preservation and text accuracy.
    - **Mechanism**: The denoising process is naturally divided into two segments based on diffusion steps—the early segment, characterized by high perturbation, is suited for establishing global structure via FMS; the late segment, where fine details gradually emerge, is suited for refining text rendering with AttnBoost.
    - **Design Motivation**: The signal-to-noise ratio characteristics at different denoising stages dictate different optimization priorities; applying a uniform strategy throughout the entire process leads to suboptimal trade-offs. Experiments confirm that 50 denoising steps achieve the best balance between quality and efficiency.

### Loss & Training

- TextFlow is a fully training-free framework requiring no fine-tuning or loss functions.
- FLUX-Kontext serves as the core image editing generator.
- T5 and CLIP are used as text encoders to extract text embeddings.
- An Overshoot + Euler scheduler with 50 denoising steps is adopted.
- Generation resolution is 384×256 (aligned with the ScenePair dataset).

## Key Experimental Results

### Main Results

| Method | SSIM↑ | PSNR↑ | MSE↓ | FID↓ | ACC(%)↑ | NED↑ |
|--------|------|------|----------|------|------|------|
| DiffSTE (training) | 22.76 | 12.26 | 7.34 | 180.15 | 71.11 | 0.907 |
| AnyText (training) | 30.73 | 13.66 | 6.05 | 51.44 | 51.12 | 0.734 |
| TextFlux (training) | 86.57 | 17.96 | 1.83 | 54.64 | 80.40 | 0.911 |
| Flux-Kontext | 87.08 | 20.53 | 1.58 | 15.41 | 78.72 | 0.920 |
| FlowEdit (training-free) | 87.60 | 20.89 | 1.16 | 25.41 | 45.51 | 0.590 |
| **TextFlow (Ours)** | **89.03** | **22.47** | **0.91** | **13.53** | **79.98** | **0.914** |

### Ablation Study

| Configuration | SSIM↑ | PSNR↑ | MSE↓ | FID↓ | ACC(%)↑ |
|------|---------|------|------|------|------|
| FlowEdit | 87.60 | 20.89 | 1.16 | 25.41 | 45.51 |
| Ours w/o FMS | 87.09 | 20.47 | 1.35 | 16.69 | - |
| Ours w/ FMS | 89.04 | 22.42 | 0.97 | 13.52 | - |
| Ours w/o AttnBoost | - | - | - | - | 20.35 |
| Ours w/ AttnBoost | - | - | - | - | 79.80 |
| Euler scheduler | - | - | - | - | 78.73 |
| Overshoot scheduler | - | - | - | - | 79.90 |

### Key Findings

- TextFlow achieves state-of-the-art performance across all image quality metrics (SSIM, PSNR, FID); its MSE (0.91) is approximately 42% lower than the second-best method, Flux-Kontext (1.58).
- Text accuracy of 79.98% is close to the training-based TextFlux (80.40%), while significantly outperforming it on image quality metrics—FID 13.53 vs. 54.64.
- AttnBoost is critical for text accuracy: its removal causes ACC to plummet from 79.80% to 20.35%, a drop of approximately 75%.
- FMS is essential for structural preservation: its removal reduces PSNR by 1.95 and increases MSE by 39.2%.
- 50 denoising steps represent the optimal balance: 24 steps yield insufficient quality, while 70 steps offer diminishing returns at increased computational cost.
- The Overshoot scheduler consistently outperforms Euler: ACC 79.90% vs. 78.73%.

## Highlights & Insights

- The two-stage decoupling strategy addresses style preservation and text accuracy separately, leveraging the signal-to-noise ratio characteristics of different denoising stages for stage-aware guidance—an elegant and generalizable design philosophy.
- As a training-free method, comprehensively surpassing training-based methods on image quality metrics is a remarkable achievement—FID 13.53 is substantially lower than TextFlux's 54.64, demonstrating that the inherent capabilities of pretrained models are effectively unlocked.
- The velocity field differential formulation ($\mathbf{V}_\Delta$) cleverly exploits the differentiable trajectory properties of flow matching models, performing geometric operations in latent space to maintain structural consistency.
- The selective attention amplification strategy of AttnBoost for text regions is transferable to other tasks requiring fine-grained control over the accuracy of generated content.

## Limitations & Future Work

- As acknowledged by the authors, the computational overhead of diffusion models limits high-resolution real-time applications.
- Handling multi-line text and complex layouts remains difficult, with challenges in maintaining spatial and typographic consistency.
- On the ScenePair Random dataset, text accuracy (74.52%) falls below Flux-Kontext (76.63%), indicating slightly weaker adaptability to arbitrary target text.
- Evaluation is currently conducted only on cropped text regions; performance and practicality for full-image editing remain to be validated.
- The boundary between the two stages appears to be fixed; an adaptive stage-switching strategy may further improve performance.

## Related Work & Insights

- **vs. TextFlux**: TextFlux is a training-based method with marginally higher text accuracy (80.40% vs. 79.98%), but significantly inferior image quality metrics compared to TextFlow (FID 54.64 vs. 13.53)—suggesting that training on synthetic data may overfit and compromise visual naturalness.
- **vs. Flux-Kontext**: Flux-Kontext achieves reasonable style preservation but insufficient text accuracy; TextFlow further incorporates the dual enhancement of FMS and AttnBoost on this basis.
- **vs. FlowEdit**: FlowEdit, as a general training-free editing method, achieves only 45.51% text accuracy in the STE setting; TextFlow addresses STE-specific challenges through stage-aware guidance.
- **Insights**: The stage-aware training-free guidance strategy is transferable to other fine-grained controllable editing tasks, such as logo editing and handwriting generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of two-stage decoupling, flow manifold steering, and attention boosting represents a meaningful innovation in training-free STE.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on the ScenePair dataset with ablation studies covering each module and key hyperparameters.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are mathematically rigorous and clear, with intuitive architectural diagrams.
- **Value**: ⭐⭐⭐⭐ A milestone demonstrating that a training-free method can match training-based approaches, with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment](../../AAAI2026/robotics/gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated.md)
- [\[CVPR 2026\] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition](pixel-level_scene_understanding_in_one_token_visual_states_need_what-is-where_co.md)
- [\[ACL 2026\] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](../../ACL2026/robotics/groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICML 2026\] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](../../ICML2026/robotics/psg-nav_probabilistic_scene_graph_navigation_via_multiverse_decision_making.md)

</div>

<!-- RELATED:END -->
