---
title: >-
  [Paper Note] Taming Rectified Flow for Inversion and Editing
description: >-
  [ICML 2025][Image Generation][Rectified Flow] This work proposes RF-Solver and RF-Edit, two training-free methods that significantly improve inversion accuracy by accurately solving the Rectified Flow ODE via high-order Taylor expansion, and achieve high-quality image/video editing using self-attention feature sharing. They are compatible with mainstream models such as FLUX and OpenSora.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Rectified Flow"
  - "ODE Solver"
  - "Image Inversion"
  - "Image Editing"
  - "Taylor Expansion"
date: 2026-05-08
content_hash: 005bf27fdeec374c
---

# Taming Rectified Flow for Inversion and Editing

**Conference**: ICML 2025  
**arXiv**: [2411.04746](https://arxiv.org/abs/2411.04746)  
**Code**: [https://github.com/wangjiangshan0725/RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)  
**Area**: Diffusion Models / Image Editing / Video Editing  
**Keywords**: Rectified Flow, ODE Solver, Image Inversion, Image Editing, Taylor Expansion

## TL;DR

This work proposes RF-Solver and RF-Edit, two training-free methods that significantly improve inversion accuracy by accurately solving the Rectified Flow ODE via high-order Taylor expansion, and achieve high-quality image/video editing using self-attention feature sharing. They are compatible with mainstream models such as FLUX and OpenSora.

## Background & Motivation

**Background**: Diffusion Transformer models based on Rectified Flow (RF) (such as FLUX and OpenSora) have achieved outstanding performance in image and video generation. Compared to traditional Stable Diffusion, these models utilize the DiT architecture combined with a straight-line motion system, yielding higher generation quality.

**Limitations of Prior Work**: Despite their powerful generative capabilities, RF models perform poorly on inversion tasks. When directly using vanilla RF for inversion, the object positions and character appearances in the reconstructed images shift significantly; video inversion is even worse, exhibiting glaring distortions. The lack of inversion precision severely limits the performance of downstream tasks like editing.

**Key Challenge**: The inversion and generation processes of RF are essentially solving the RF ODE. Since the ODE contains a complex neural network term, it can only be roughly approximated by a sampler. By tracking the latent MSE at each timestep during the inversion-reconstruction process, the authors found that the errors introduced by existing samplers at each step accumulate continuously, ultimately leading to severe degradation in reconstruction quality.

**Goal**: (a) How to improve the ODE solving accuracy of RF without introducing additional training? (b) How to achieve high-fidelity image/video editing based on accurate inversion?

**Key Insight**: The authors derived the exact form of the RF ODE and found that the core issue lies in the approximation error of the non-linear component (the neural network velocity field) in the ODE. If this non-linear term can be estimated more accurately, the accumulation of error at each step can be significantly reduced.

**Core Idea**: Use high-order Taylor expansion instead of the first-order Euler approximation to estimate the non-linear component of the RF ODE, thereby achieving near-error-free inversion, and complete editing based on this through self-attention feature injection.

## Method

### Overall Architecture

The entire method comprises two core components:

- **RF-Solver**: A training-free ODE sampler designed to replace the default sampler in existing RF models. It takes noisy latents (or clean latents) as input and outputs more accurate denoising (or noising) results. It can be used simultaneously for both generation (denoising) and inversion (noising) directions.
- **RF-Edit**: An editing framework based on RF-Solver. It first uses RF-Solver to perform precise inversion on the source image/video to obtain the noise latent, and then injects self-attention features from the inversion stage during the editing sampling process, thereby maintaining the structures of the source image while accomplishing text-guided editing.

Pipeline: Source image → VAE encoding → RF-Solver inversion (noising to pure noise) → RF-Solver denoising with editing prompt + feature injection → VAE decoding → edited result.

### Key Designs

1. **Exact Derivation and Error Analysis of RF ODE**:

    - **Function**: Derives the closed-form representation of the Rectified Flow ODE and explicitly identifies the source of error.
    - **Mechanism**: Rectified Flow defines a straight-line interpolation path $x_t = (1-t)x_0 + t\epsilon$ from the data distribution $x_0$ to the noise distribution $x_1$, where the velocity field $v_\theta(x_t, t)$ is modeled by a neural network. The ODE is $\frac{dx_t}{dt} = v_\theta(x_t, t)$. Existing methods employ Euler's method $x_{t+h} = x_t + h \cdot v_\theta(x_t, t)$ to solve it, which is equivalent to assuming that the velocity field remains constant within $[t, t+h]$—whereas in reality, $v_\theta$ varies significantly with respect to $t$ and $x_t$. Consequently, the first-order approximation incurs severe errors when the step size is large.
    - **Design Motivation**: Only by clarifying the mathematical source of the error can a more precise solver be designed in a targeted manner.

2. **High-order Taylor Expansion Solver (RF-Solver)**:

    - **Function**: Approximates the change in the velocity field using a Taylor expansion to achieve high-precision ODE solving.
    - **Mechanism**: The velocity field $v_\theta(x_t, t)$ is expanded using a Taylor expansion at the current timestep $t_n$: $$v_\theta(x_{t_{n+1}}, t_{n+1}) \approx v_\theta(x_{t_n}, t_n) + \frac{dv_\theta}{dt}\bigg|_{t_n} \cdot h + \frac{1}{2}\frac{d^2v_\theta}{dt^2}\bigg|_{t_n} \cdot h^2 + ...$$ where the first derivative can be approximated via the difference between adjacent timesteps: $\frac{dv}{dt} \approx \frac{v_\theta(x_{t_n}, t_n) - v_\theta(x_{t_{n-1}}, t_{n-1})}{t_n - t_{n-1}}$. This method utilizes historical function evaluations, requiring no additional neural network forward passes.
    - **Design Motivation**: Compared with Euler's method (0-order), the Taylor expansion leverages information about the rate of change of the velocity field, drastically reducing truncation errors. The key lies in utilizing the historical sampling points "for free"—the $v_\theta(x_{t_{n-1}}, t_{n-1})$ calculated in the previous step can be reused to estimate the derivative without adding computational overhead.
    - **Comparison with Prior Work**: A similar idea is used in the DPM-Solver of traditional DDPM/DDIM, but the ODE structure of RF differs from DDPM (straight line vs. curve) and requires re-derivation. RF-Solver is specifically designed for Rectified Flow.

3. **Feature Sharing Editing Framework (RF-Edit)**:

    - **Function**: Injects self-attention features from the inversion stage during the editing process to maintain the structural information of the source image/video.
    - **Mechanism**: During the RF-Solver inversion process, self-attention keys/values (i.e., $K_{inv}^l, V_{inv}^l$) of specific layers in the DiT at each timestep are cached. During the editing denoising process, the self-attention of the corresponding timestep is replaced with the keys/values from the inversion stage: $\text{Attn}(Q_{edit}, K_{inv}, V_{inv})$. This ensures that the spatial layout of the edited image remains aligned with the source image, while the semantic content is driven by the new text prompt.
    - **Design Motivation**: The core challenge of image editing lies in "maintaining structure while changing content". The K/V of self-attention encodes spatial positioning and structural information, whereas Q determines "what to focus on". By injecting the K/V of the source image, the editing process can "perceive" the structural information of the original image. This concept is inspired by methods like Prompt-to-Prompt and PnP, but this work is the first to extend it to the RF+DiT architecture.
    - **Video Extension**: For video editing, the same feature injection strategy inherently supports temporal consistency. Since the attention features from the inversion stage already contain temporal relationships between frames, the feature injection can maintain the temporal coherence of the edited video.

### Loss & Training

This method is entirely training-free and does not require any additional training or fine-tuning. All operations are implemented during inference by modifying the sampler and the feature injection. The only extra overhead is caching the attention features from the inversion stage, which incurs a controllable increase in memory consumption.

## Key Experimental Results

### Main Results

| Method | Model | PSNR↑ | SSIM↑ | LPIPS↓ | MSE↓ |
|------|------|-------|-------|--------|------|
| Vanilla RF (Euler) | FLUX | ~25 | ~0.75 | ~0.15 | High |
| DDIM Inversion | FLUX | ~28 | ~0.82 | ~0.10 | Moderate |
| RF-Solver (Ours) | FLUX | **~35** | **~0.95** | **~0.02** | **Extremely Low** |
| Vanilla RF (Euler) | OpenSora | Poor | Poor | Poor | High |
| RF-Solver (Ours) | OpenSora | **Significant Improvement** | **Significant Improvement** | **Significant Improvement** | **Low** |

RF-Solver significantly outperforms the vanilla Euler sampler on both FLUX and OpenSora, with an improvement of about 10dB in PSNR and a reduction of about 85% in LPIPS, achieving near-perfect inversion reconstruction.

### Ablation Study

| Taylor Order | PSNR↑ | Extra NFE | Description |
|-------------|-------|---------|------|
| 0-order (Euler) | ~25 | 0 | Baseline, only using the current velocity at each step |
| 1st-order (Linear) | ~31 | 0 | Using the previous step's velocity to estimate the gradient |
| 2nd-order (Quadratic) | ~34 | 0 | Further utilizing second-order variation information |
| 3rd-order (Cubic) | ~35 | 0 | Nearing convergence, diminishing marginal returns |

**Key Findings**: The improvement from 1st-order to 2nd-order is the most significant, indicating that the rate of change of the velocity field is the primary source of error. Above the 2nd-order, the marginal returns diminish; therefore, the 2nd-order is recommended for practical use as a balance between accuracy and storage.

### Image Editing Comparison

| Method | Structure Preservation↑ | Editing Quality↑ | Time | Requires Training |
|------|----------|----------|------|--------|
| InstructPix2Pix | Moderate | Moderate | Fast | Yes |
| Prompt-to-Prompt | High | Moderate | Moderate | No |
| PnP-Diffusion | High | Moderate-High | Moderate | No |
| RF-Edit (Ours) | **Extremely High** | **High** | Moderate | **No** |

### Key Findings

- **ODE Accuracy is the Bottleneck**: The fundamental reason for inversion failure in RF models is the insufficient precision of the Euler sampler, rather than flaws in the capability of the model itself. A qualitative leap can be obtained solely by improving the sampler.
- **Free Use of Historical Information**: The high-order Taylor expansion of RF-Solver utilizes the velocity values already computed in previous steps, adding no extra NFE (Neural Function Evaluation), which is key to its high efficiency.
- **Image vs. Video**: Video inversion is more challenging than image inversion because the temporal dimension introduces additional error propagation paths. The improvement margin of RF-Solver in video scenarios is even greater than in images.
- **Selection of Injection Layers in Editing**: Not all DiT layers' attention features are suitable for injection. Shallow layers retain low-level textures, while deep layers encode high-level semantics. In practice, selecting the middle layers yields the best performance.

## Highlights & Insights

- **Elegant Training-free Design**: The entire method requires no training, achieving an order-of-magnitude improvement in inversion accuracy simply by modifying the ODE solving strategy and utilizing attention injection. This "inference-only, model-frozen" paradigm is highly elegant and practical, seamlessly plugging into any RF foundation model.
- **Error Analysis-Driven Design**: Rather than presenting an empirical trick, the authors start from the mathematical perspective of ODE solving, clearly locating the error source (Euler's truncation error), and then provide a systematic solution based on classical Taylor expansion. This "analysis $\rightarrow$ pinpointing $\rightarrow$ resolution" paradigm is highly commendable.
- **Transferable Attention Injection Technique**: The practice of injecting the self-attention K/V from the inversion stage into the editing process in RF-Edit can be transferred to other tasks requiring "structure preservation with semantic changing," such as style transfer, video translation, and virtual try-on. The core concept is "using attention to decouple structure and content."
- **Cross-Modal Universality**: The method is applicable to both images and videos, proving effective on both FLUX (T2I) and OpenSora (T2V), demonstrating the high universality of RF-Solver.

## Limitations & Future Work

- **Increased Cache Requirements**: RF-Solver needs to cache the velocity field values of historical timesteps for Taylor expansion, and RF-Edit additionally requires caching multi-layer attention K/V. For high-resolution images or long videos, memory overhead may become a bottleneck.
- **Dependency on Number of Steps**: The accuracy of Taylor expansion relies on an adequate number of sampling steps. When steps are extremely low (e.g., 4-8 steps), insufficient historical information may limit the efficacy of higher-order expansions. The performance of the method in extremely low-step scenarios remains to be validated.
- **Limited Editing Freedom**: RF-Edit maintains structure via feature injection, which inherently restricts the capability for large geometric changes (such as altering object sizes or significant posture changes). There is an inherent trade-off between structure preservation and editing flexibility.
- **Lack of Fine-Grained Text Condition Control**: The current feature injection strategy is global, failing to achieve region-level fine control. Combining it with attention masks or cross-attention manipulation might enable more precise local editing.
- **Validated Only on RF-Type Models**: The method is designed specifically for the straight-line ODE structure of Rectified Flow, making it theoretically inapplicable to traditional DDPM/DDIM models directly (though analogies can be drawn to DPM-Solver).

## Related Work & Insights

- **vs DPM-Solver (Lu et al., 2022)**: DPM-Solver is a high-order ODE solver designed for DDPM-type diffusion models, with a similar philosophy but different ODE formulations. RF-Solver is specifically derived for the straight-line interpolation structure of Rectified Flow, making them complementary rather than competing.
- **vs DDIM Inversion (Song et al., 2021)**: DDIM inversion is the most common inversion method in traditional diffusion models, but it performs poorly on RF models (due to different ODE structures). RF-Solver fills the gap of a specialized inverter for RF models.
- **vs Prompt-to-Prompt (Hertz et al., 2022)**: P2P achieves editing by manipulating cross-attention maps, whereas RF-Edit preserves structure by injecting self-attention K/V. They focus on different aspects: P2P controls "how text influences the image," while RF-Edit controls "how the source image structure is preserved."
- **vs Null-text Inversion (Mokady et al., 2023)**: Null-text improves inversion precision by optimizing unconditional embeddings, which requires dozens of optimization iterations. RF-Solver requires no optimization, directly boosting precision at the sampling level with higher efficiency.
- **Relation to Video Editing Methods**: Methods like TokenFlow and FateZero also utilize attention features to maintain temporal consistency, but they are based on traditional U-Net diffusion models. RF-Edit is the first solution tailored for the DiT + RF architecture.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea (improving ODE solvers with Taylor expansion) has a solid mathematical foundation, but high-order ODE solving is not pioneering in diffusion models (DPM-Solver came before it).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three tasks: generation, inversion, and editing across both image and video modalities, evaluated on multiple foundation models (FLUX, OpenSora), with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear and smooth from error analysis to methodological derivation, with intuitive and persuasive figures and tables.
- Value: ⭐⭐⭐⭐ High practical value as a training-free method, plug-and-play compatible with mainstream models, and provides a significant boost to the editing capabilities of RF models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICLR 2026\] UniEdit-Flow: Unleashing Inversion and Editing in the Era of Flow Models](../../ICLR2026/image_generation/uniedit-flow_unleashing_inversion_and_editing_in_the_era_of_flow_models.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](../../ICCV2025/image_generation/reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](../../CVPR2025/image_generation/unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)

</div>

<!-- RELATED:END -->
