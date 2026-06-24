---
title: >-
  [Paper Note] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models
description: >-
  [CVPR 2025][Video Generation][Interactive Dynamics] InterDyn proposes utilizing video diffusion models as implicit physics engines. By introducing an interactive control branch (ControlNet-like) on top of Stable Video Diffusion, the method generates physically plausible interactive dynamics videos from a single image and driving motion signals, outperforming the baseline CosHand by 77% in terms of the FVD metric on the Something-Something-v2 dataset.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Interactive Dynamics"
  - "Video Diffusion Models"
  - "Controllable Generation"
  - "Implicit Physical Simulation"
  - "Human-Object Interaction"
date: 2026-05-08
content_hash: 009ecf379615d58e
---

# InterDyn: Controllable Interactive Dynamics with Video Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.11785](https://arxiv.org/abs/2412.11785)  
**Code**: [https://interdyn.is.tue.mpg.de/](https://interdyn.is.tue.mpg.de/)  
**Area**: Diffusion Models  
**Keywords**: Interactive Dynamics, Video Diffusion Models, Controllable Generation, Implicit Physical Simulation, Human-Object Interaction

## TL;DR
InterDyn proposes utilizing video diffusion models as implicit physics engines. By introducing an interactive control branch (ControlNet-like) on top of Stable Video Diffusion, the method generates physically plausible interactive dynamics videos from a single image and driving motion signals, outperforming the baseline CosHand by 77% in terms of the FVD metric on the Something-Something-v2 dataset.

## Background & Motivation

1. **Background**: Predicting the dynamics of manipulated objects is a core capability of intelligent systems. Existing approaches can be categorized into: (1) explicit physics simulation-based methods, which require 3D reconstruction and physics engines, rendering them computationally expensive and limited in generalization; (2) keypoint/graph neural network-based methods, which are only validated in simplified synthetic environments.

2. **Limitations of Prior Work**:
    - Explicit physics simulations rely on accurate 3D reconstruction, suffer from error accumulation, and struggle in complex, real-world scenes.
    - Recent generative approaches (such as CosHand) can only predict a single future state (image-to-image) and cannot capture the subsequent continuous dynamics following the interaction.
    - Static state transitions cannot represent the continuous dynamics during interactions, such as water levels continuously rising after pouring.

3. **Key Challenge**: Interactive dynamics constitute a continuous temporal process, whereas existing methods either require complete physical simulation pipelines or rely on discrete state prediction. Neither approach successfully balances realism and practicality.

4. **Goal**: Generate physically plausible videos of interactive dynamics from a single image and control signals, without requiring 3D reconstruction or physics engines.

5. **Key Insight**: Large-scale video models pre-trained on massive video datasets have implicitly learned complex knowledge of physical interactions, and only an effective control mechanism is required to guide and extract this knowledge.

6. **Core Idea**: Freeze the weights of the pre-trained SVD and train only a ControlNet branch to inject driving motion signals, thereby utilizing the video diffusion model as an implicit physics engine.

## Method

The core of InterDyn is intuitive: treating Stable Video Diffusion (SVD) as a model that has already "learned physics," and precisely guiding it to generate object interactive dynamics by adding control signals (such as hand mask sequences). The key insight is that video models are not just renderers, but also implicit physics simulators.

### Overall Architecture

Input: An initial image $\boldsymbol{x} \in \mathbb{R}^{1 \times H \times W \times 3}$ + a control signal sequence $\boldsymbol{c} \in \mathbb{R}^{N \times H \times W \times 3}$ (e.g., hand binary mask sequence).
Output: An $N$-frame video $\boldsymbol{y} \in \mathbb{R}^{N \times H \times W \times 3}$ showing the hand movement and the resulting object dynamics.

The architecture is based on SVD (14-frame image-to-video) with a frozen backbone and an added trainable ControlNet encoder branch.

### Key Designs

1. **ControlNet-style Control Branch**:
    - **Function**: Inject driving motion control signals into the video generation process.
    - **Mechanism**: Replicate the SVD encoder $E$ as a trainable copy, connected to the frozen SVD decoder via zero-initialized convolutional skip connections. A small CNN $\mathcal{E}(\cdot)$ encodes the control signals into the latent space, which is then added to the input noise latent of the ControlNet encoder. The control branch also incorporates an interleaved structure of convolutional, spatial, and temporal blocks to process control signals in a temporally-aware manner.
    - **Design Motivation**: Freezing the SVD weights preserves the learned dynamic priors and avoids catastrophic forgetting. The ControlNet architecture allows precise control while maintaining generation quality. The temporally-aware design renders the model robust to noisy control signals, such as coarse hand masks output by SAM2.

2. **Binary Mask Driving Signals**:
    - **Function**: Encode the motion trajectory of the driving entity (e.g., a hand) in a highly simplified form.
    - **Mechanism**: Use SAM2 to generate frame-by-frame binary mask sequences from hand bounding boxes as control signals. This mask only encodes the motion of the "driver" and provides no signals regarding the manipulated object—the physical dynamics of the object are entirely inferred implicitly by the model.
    - **Design Motivation**: Binary masks are the easiest to obtain and are task-agnostic. Experiments indicate that the type of control signal has minimal impact on generation quality (see ablation in Appendix).

3. **Training & Inference Strategy**:
    - **Function**: Efficiently fine-tune and apply classifier-free guidance.
    - **Mechanism**: Use the EDM framework with a noise distribution of $\log\sigma \sim \mathcal{N}(0.7, 1.6^2)$, and optimize using the Adam optimizer at $lr=10^{-5}$. Videos are downsampled to 7FPS to balance short-range and long-range events. The input image is randomly dropped with a 5% probability for classifier-free guidance. During inference, an Euler scheduler with 50 denoising steps is used.
    - **Design Motivation**: Downsampling to 7FPS allows a 14-frame video to span approximately 2 seconds, which is sufficient to showcase most interactive dynamics. The motion ID is set to 40 to match SVD priors.

### Loss & Training

The standard diffusion training objective is used: denoising loss. Training is conducted on 2x 80GB H100 GPUs, with a batch size of 4 per GPU. Two versions are trained: 256×256 (matching CosHand) and 256×384 (matching the SVD prior aspect ratio).

## Key Experimental Results

### Main Results

**Something-Something-v2 (SSV2) Quantitative Comparison**

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | KVD↓ | Motion Fidelity↑ |
|------|-------|-------|--------|------|------|-------------------|
| Seer | 0.418 | 10.71 | 0.588 | 287.46 | 81.31 | — |
| DynamiCrafter | — | — | — | 204.11 | 31.81 | — |
| CosHand-Independent | 0.615 | 16.87 | 0.313 | 91.18 | 19.24 | 0.432 |
| CosHand-Autoregressive | 0.531 | 14.92 | 0.408 | 90.30 | 13.68 | 0.570 |
| **Ours 256×256** | **0.664** | **18.60** | **0.260** | **19.27** | **1.99** | **0.633** |
| **Ours 256×384** | **0.680** | **19.04** | **0.252** | **22.22** | **2.09** | **0.641** |

InterDyn outperforms CosHand by 37.5% in LPIPS and by 77% in FVD.

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| CLEVRER Force Propagation | Can generate multi-object collision chain reactions | Implicitly understands force propagation |
| CLEVRER Counterfactual Reasoning | Same image + different control signals → different plausible results | The model possesses counterfactual reasoning capabilities |
| Control Signal Type | Minimal difference between binary masks vs semantic masks | The model is insensitive to the type of control signals |
| Noise Mask Robustness | SAM2 coarse masks still generate fine hand details | Temporally-aware branch is effective |

### Key Findings
- InterDyn is capable of generating various complex physical phenomena: articulated object motion, water pouring (rising water levels), object dropping and bouncing, squishing/deforming and recovery, reflections, etc.
- Force propagation reasoning and counterfactual reasoning capabilities are validated on the CLEVRER synthetic dataset.
- CosHand's frame-independent method yields high image quality but poor temporal consistency; its autoregressive method generates better motion but suffers from image degradation.
- The model can even generate plausible hand details in frames that only contain motion blur.

## Highlights & Insights
- **Video Models as Physics Engines**: This is the most critical insight of the paper. Models trained on large-scale video datasets implicitly acquire physical interaction knowledge. This perspective can inspire the utilization of video generative models in a broader range of physical reasoning tasks.
- **Elegant Decoupling of Control and Generation**: By keeping SVD frozen and using a trainable ControlNet, the method achieves a "win-win" of "preserving physical knowledge + injecting precise control". This design pattern can be transferred to other conditional video generation tasks.
- **Counterfactual Reasoning Ability**: Experiments on CLEVRER demonstrate that the model can reason plausibly under different control signals for the same scene, hinting at the potential of video models as world models.

## Limitations & Future Work
- The generation of object dynamics is implicit and probabilistic, and physical accuracy (such as precise collision angles and velocities) cannot be guaranteed.
- Currently trained only on 14 frames; long-duration interactive dynamics generation remains unexplored.
- Primarily validated on hand-object interaction; full-body interaction and multi-person scenes have not been explored.
- Occasionally underperforms CosHand in image quality metrics (FID/KID), likely because SVD's multi-stage training degraded its spatial prior.
- Although the generated hand details are plausible, they are not always consistent, and finger movements can sometimes be unstable.

## Related Work & Insights
- **vs CosHand**: CosHand is an image-to-image state transition method that cannot capture the continuous dynamics after an interaction. InterDyn directly generates continuous video, showcasing the subsequent motion of objects under forces.
- **vs PhysGen**: PhysGen relies on explicit physics engines to compute motion and is limited to rigid bodies; InterDyn is entirely implicit and can handle complex physics like soft bodies and liquids.
- **vs Seer/DynamiCrafter**: These text-controlled methods lack fine-grained spatial control, resulting in FVD scores an order of magnitude worse than InterDyn's.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective of "video models as physics engines" is deeply inspiring, and the method design is clean.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Highly progressive verification from synthetic to real and simple to complex, with both quantitative and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative, with a logical flow from problem definition to experimental design.
- **Value**: ⭐⭐⭐⭐ Opens up a promising research direction for video models as implicit physical simulators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] VidTwin: Video VAE with Decoupled Structure and Dynamics](vidtwin_video_vae_with_decoupled_structure_and_dynamics.md)
- [\[ICLR 2026\] Vid2World: Crafting Video Diffusion Models to Interactive World Models](../../ICLR2026/video_generation/vid2world_crafting_video_diffusion_models_to_interactive_world_models.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)

</div>

<!-- RELATED:END -->
