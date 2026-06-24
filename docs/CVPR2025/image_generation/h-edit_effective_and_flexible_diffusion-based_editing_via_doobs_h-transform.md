---
title: >-
  [Paper Note] h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform
description: >-
  [CVPR 2025][Image Generation][Image Editing] h-Edit formalizes diffusion-based image editing as a backward-time bridge modeling problem based on Doob's h-transform. By decoupling editing updates into a "reconstruction term" and an "editing term," it achieves training-free joint editing guided by text and reward models for the first time, comprehensively outperforming existing SOTA methods on PIE-Bench.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "Doob's h-Transform"
  - "Diffusion Bridge"
  - "Training-Free Editing"
  - "Reward Model-Guided Editing"
date: 2026-05-08
content_hash: 821c0281fdce292d
---

# h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform

**Conference**: CVPR 2025  
**arXiv**: [2503.02187](https://arxiv.org/abs/2503.02187)  
**Code**: [https://github.com/nktoan/h-edit](https://github.com/nktoan/h-edit)  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Image Editing, Doob's h-Transform, Diffusion Bridge, Training-Free Editing, Reward Model-Guided Editing

## TL;DR
h-Edit formalizes diffusion-based image editing as a backward-time bridge modeling problem based on Doob's h-transform. By decoupling editing updates into a "reconstruction term" and an "editing term," it achieves training-free joint editing guided by text and reward models for the first time, comprehensively outperforming existing SOTA methods on PIE-Bench.

## Background & Motivation

**Background**: Diffusion-based image editing methods (e.g., DDIM inversion + P2P, Edit Friendly, PnP Inversion) have made significant progress. The core idea of these methods is to map the original image to the noise space through an inversion process, and then resample using target conditions to obtain the edited result.

**Limitations of Prior Work**: (1) Most methods are designed based on heuristics or intuition, lacking a clear theoretical foundation, which makes them difficult to generalize to complex scenarios. (2) There is a hard-to-reconcile trade-off between "edit effectiveness" and "content fidelity"—enhancing editing strength often comes at the expense of preserving unedited regions. (3) Almost all training-free methods are limited to text-guided editing and cannot incorporate external reward models (such as style transfer or face recognition) for compositional editing.

**Key Challenge**: Existing methods mix reconstruction and editing together during optimization, lacking a clear theoretical framework to decouple the two, which prevents flexible combinations of different types of target edits.

**Goal**: To establish a theoretically guaranteed diffusion editing framework that can decouple editing updates into independent reconstruction and editing terms, enabling the free composition of different editing objectives.

**Key Insight**: Doob's h-transform in probability theory can modify a Markov process into a bridge process that converges to a specified distribution. By treating the reverse diffusion process as the base process and encoding editing objectives into an h-function, we can naturally obtain a bridge process that converges to a distribution that is both realistic and possesses the target attributes.

**Core Idea**: Image editing is formalized as an h-transform of the reverse diffusion process, decomposing the editing sampling update into a reconstruction term $x_{t-1}^{base}$ and an editing gradient $\nabla \log h(x_{t-1}, t-1)$, where the h-function can represent text conditions, reward models, or their combinations.

## Method

### Overall Architecture
Given the original image $x_0^{orig}$ and editing conditions, the forward process (DDIM inversion or stochastic inversion) first maps the original image to $x_T^{orig}$. Letting $x_T^{edit} = x_T^{orig}$, sampling is performed from T to 0 along the reverse bridge process modified by the h-transform to obtain the edited image $x_0^{edit}$. The transition kernel of the bridge process is $p^h(x_{t-1}|x_t) \propto p(x_{t-1}|x_t) \cdot h(x_{t-1}, t-1)$, where $p$ is the reverse process of the original diffusion and $h$ encodes the editing objective.

### Key Designs

1. **Doob's h-transform Editing Framework**:

    - **Function**: Provides a unified theoretical foundation for image editing.
    - **Mechanism**: Defines the h-function as a positive function satisfying $h(x_0, 0) = p_{\mathcal{Y}}(x_0)$ (the probability of target attributes), which is recursively extended to any timestep via $h(x_t, t) = \mathbb{E}_{p(x_0|x_t)}[h(x_0, 0)]$. Guaranteed by Proposition 1, the constructed bridge process converges to $p^h(x_0) \propto p(x_0) \cdot p_{\mathcal Y}(x_0)$ at $t=0$, which is both realistic and possesses target attributes. Since $p^h(x_{t-1}|x_t)$ is generally non-Gaussian, Langevin Monte Carlo is used to approximate sampling, which naturally yields the decomposed form of $x_{t-1} = x_{t-1}^{base} + \gamma \nabla \log h(x_{t-1}, t-1)$.
    - **Design Motivation**: Most existing methods are ad-hoc modifications of the DDIM sampling process and lack theoretical guarantees. The h-transform framework provides a rigorous mathematical explanation of why editing can be decomposed into reconstruction and editing.

2. **Explicit and Implicit h-Edit Updates**:

    - **Function**: Provides two flexible implementation modes for editing.
    - **Mechanism**: **Explicit updates** (Eq.15) directly compute the editing gradient $\nabla \log h(x_t, t)$ on $x_t$, which is suitable when gradients are easily computed. **Implicit updates** (Eq.18) compute $\nabla \log h(x_{t-1}^{base}, t-1)$ on $x_{t-1}^{base}$, which can be viewed as optimizing $\log h$ starting from $x_{t-1}^{base}$ as the initial value, supporting multi-step gradient ascent (Eq.21) to enhance editing effects. For text-guided editing in Stable Diffusion, the editing term simplifies to $f(x_t, t) = w_{edit}\epsilon_\theta(x_t, t, c_{edit}) - \hat{w}_{orig}\epsilon_\theta(x_t, t, c_{orig}) + (\hat{w}_{orig} - w_{edit})\epsilon_\theta(x_t, t, \emptyset)$.
    - **Design Motivation**: Explicit updates are fast to compute but have limited editing strength; implicit updates can handle more challenging editing scenarios via multi-step optimization, with $x_{t-1}^{base}$ naturally serving as a fidelity anchor.

3. **Product of h-Experts Compositional Editing**:

    - **Function**: Enables free composition of multiple editing objectives.
    - **Mechanism**: Since $\log h$ can be interpreted as a negative energy function, multiple h-functions can be combined through simple multiplication $h = h_1 \cdot h_2 \cdots h_m$, which simplifies at the gradient level to the sum of gradients of each component $\nabla \log h = \sum_i \nabla \log h_i$. This allows seamless combination of different objectives such as text-guided editing ($h_1$ from classifier-free guidance), style transfer ($h_2$ from Gram matrix matching reward), and identity preservation ($h_3$ from ArcFace). Additionally, a dedicated reconstruction h-function $h_{rec} = \exp(-\lambda \|x_{t-1} - x_{t-1}^{base}\|^2)$ is designed to realize both non-optimization-based and optimization-based reconstruction.
    - **Design Motivation**: Existing methods struggle with composite tasks like simultaneous text editing, style transfer, and identity preservation. h-Edit solves this elegantly through the additivity of energy functions.

### Loss & Training
Entirely training-free. The core hyperparameters are the guidance weights $w_{edit}$, $\hat{w}_{orig}$ (governing the trade-off between editing strength and fidelity) and the number of optimization steps $K$ for implicit updates. The deterministic inversion version (h-Edit-D) and the stochastic inversion version (h-Edit-R) utilize different default parameters.

## Key Experimental Results

### Main Results: PIE-Bench Text-Guided Editing

| Method | Inversion | LPIPS↓ | DINO↓ | Local CLIP↑ | Whole CLIP↑ |
|------|-----------|--------|-------|-------------|-------------|
| h-Edit-D + P2P | Deterministic | **0.253** | 0.147 | **8.54** | **27.87** |
| PnP Inv + P2P | Deterministic | 0.250 | 0.095 | 8.48 | 27.22 |
| NT + P2P | Deterministic | 0.248 | 0.130 | 8.41 | 27.03 |
| NMG + P2P | Deterministic | 0.249 | 0.087 | 8.47 | 27.05 |
| h-Edit-R + P2P | Stochastic | **0.256** | **0.159** | **8.50** | **26.97** |
| EF + P2P | Stochastic | 0.255 | 0.126 | 8.40 | 26.30 |
| LEDITS++ | Stochastic | 0.254 | 0.113 | 8.11 | 23.36 |

### Face Swapping Experiment (CelebA-HQ)

| Method | ID↑ | Expr.↓ | Pose↓ | LPIPS↓ | FID↓ |
|------|-----|--------|-------|--------|------|
| h-Edit-R (1 step) | **0.80** | **2.76** | **3.78** | **0.04** | 17.68 |
| h-Edit-R (3 steps) | **0.84** | 3.10 | 4.29 | 0.05 | 19.12 |
| DiffFace | 0.61 | 3.04 | 4.35 | 0.10 | **11.89** |
| FaceShifter | 0.70 | 2.39 | 2.81 | 0.08 | 10.16 |
| EF | 0.74 | 3.10 | 4.12 | 0.06 | 20.78 |

### Key Findings
- h-Edit-D + P2P is comprehensively optimal among deterministic inversion methods, achieving a Local CLIP gain of 0.06 with competitive LPIPS, showing that the theoretical framework indeed improves editing effectiveness.
- PnP Inv and NMG often preserve the original image too strictly ("pretend not to edit") in hard editing scenarios, yielding deceptively good fidelity metrics while failing the actual edit. h-Edit does not suffer from this issue.
- Multi-step optimization in implicit updates (3 steps vs. 1 step) improves the ID similarity from 0.80 to 0.84 on the face swapping task, albeit at the cost of a slight drop in fidelity.
- For compositional editing (text + style), h-Edit-R + P2P significantly outperforms EF + P2P, whereas EF tends to introduce artifacts or distort unedited regions in compositional tasks.

## Highlights & Insights
- **The Elegance of the Theoretical Framework**: Formalizing diffusion-based editing as a Doob's h-transform bridge process is a very elegant theoretical contribution. The decomposition of reconstruction and edit terms is no longer heuristic but backed by rigorous probability theory. This provides a unified theoretical language for subsequent methods.
- **Product of h-Experts as a Flexible Composition Mechanism**: Under the energy function perspective, different editing goals are combined by simple addition, which is vastly simpler than the complex multi-stage pipelines in existing methods. Notably, this achieves training-free joint editing guided by both text and reward models for the first time.
- **Optimization Perspective of Implicit Updates**: Formulating editing at each step as a gradient ascent optimization problem starting from $x_{t-1}^{base}$ with adjustable steps provides practitioners with an intuitive "edit strength knob."

## Limitations & Future Work
- Reliance on Stable Diffusion v1.4, without validation on newer architectures like SDXL or SD3.
- Text-guided editing still requires combination with P2P's attention maps to preserve structure; the framework itself cannot entirely replace P2P.
- The multi-step optimization of implicit updates linearly increases inference time.
- Gradient computation of the h-function may be non-differentiable or unstable for certain reward models.
- Currently only handles image editing; extending it to video editing requires addressing temporal consistency issues.

## Related Work & Insights
- **vs PnP Inversion**: PnP directly injects inversion residuals into editing updates, which is equivalent to a special case in the h-Edit framework with only the reconstruction term and no editing term, resulting in weak editing strength.
- **vs Edit Friendly (EF)**: EF performs editing utilizing stochastic inversion + residual injection but lacks a theoretical foundation; h-Edit-R can be viewed as a theoretically enhanced version of EF that incorporates an explicit editing term.
- **vs FreeDoM/Universal Guidance**: These methods also leverage gradients of external rewards to guide diffusion sampling, but they are not unified under the h-transform framework and do not support seamless combination with text-guided editing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Doob's h-transform to diffusion editing is a brand-new theoretical contribution; the framework is both elegant and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage across three tasks: text editing, face swapping, and compositional editing, but lacks user studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivation is clear and complete, and the block diagram of the framework is intuitive.
- Value: ⭐⭐⭐⭐⭐ Unifies the theoretical foundation of diffusion editing; Product of h-Experts holds significant potential for compositional editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[ICML 2026\] AesFormer: Transform Everyday Photos into Beautiful Memories](../../ICML2026/image_generation/aesformer_transform_everyday_photos_into_beautiful_memories.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](../../CVPR2026/image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)

</div>

<!-- RELATED:END -->
