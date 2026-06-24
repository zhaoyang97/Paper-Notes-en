---
title: >-
  [Paper Note] Rectified Diffusion Guidance for Conditional Generation
description: >-
  [CVPR 2025][Image Generation][Classifier-Free Guidance] ReCFG theoretically reveals that the sum-to-one constraint of the two coefficients in standard Classifier-Free Guidance (CFG) leads to an expectation shift in the generated distribution. By relaxing the coefficient constraint and deriving a closed-form solution for $\gamma_0$, it provides a training-free post-processing scheme with virtually no extra inference overhead to rectify the guidance effect of CFG.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "expectation shift"
  - "rectified guidance"
  - "look-up table"
  - "post-processing scheme"
date: 2026-05-08
content_hash: e1bef21f86c538aa
---

# Rectified Diffusion Guidance for Conditional Generation

**Conference**: CVPR 2025  
**arXiv**: [2410.18737](https://arxiv.org/abs/2410.18737)  
**Code**: [https://github.com/thuxmf/recfg](https://github.com/thuxmf/recfg)  
**Area**: Diffusion Models / Conditional Generation  
**Keywords**: Classifier-Free Guidance, expectation shift, rectified guidance, look-up table, post-processing scheme

## TL;DR

ReCFG theoretically reveals that the sum-to-one constraint of the two coefficients in standard Classifier-Free Guidance (CFG) leads to an expectation shift in the generated distribution. By relaxing the coefficient constraint and deriving a closed-form solution for $\gamma_0$, it provides a training-free post-processing scheme with virtually no extra inference overhead to rectify the guidance effect of CFG.

## Background & Motivation

1. **Background**: Classifier-Free Guidance (CFG) is a core technology for conditional sampling in diffusion models, enhancing conditional fidelity by interpolating between the conditional and unconditional score functions $s_{t,\gamma}(x,c) = \gamma \nabla \log q_t(x|c) + (1-\gamma) \nabla \log q_t(x)$. CFG is widely used in almost all mainstream diffusion models (including DALL-E, Stable Diffusion, ImageGen, etc.).
2. **Limitations of Prior Work**: Despite its massive empirical success, CFG possesses a fundamental flaw from a theoretical perspective: the denoising process with CFG cannot be represented as the reverse process of any standard forward diffusion. Specifically, the score function of the gamma-powered distribution $q_{t,\gamma}(x|c) = q_t(x|c)^\gamma q_t(x)^{1-\gamma}$ has a non-zero expectation, violating the basic theoretical assumptions of diffusion models.
3. **Key Challenge**: Although the constraint that the two coefficients of CFG sum to 1 ($\gamma + (1-\gamma)=1$) appears natural, it actually causes an expectation shift of the generated distribution relative to the true conditional distribution $q_0(x_0|c)$, which becomes more severe as the guidance strength $\gamma$ increases.
4. **Goal**: Quantify the expectation shift phenomenon of CFG and design a correction scheme to make the guidance process strictly conform to diffusion theory.
5. **Key Insight**: Using the DDIM theoretical framework and a toy distribution ($q_0(x_0|c) \sim \mathcal{N}(c,1)$, $q(c) \sim \mathcal{N}(0,1)$), the precise closed-form expression of the expectation shift is derived, revealing that the source of the shift is the ratio of the expectations of the two score functions.
6. **Core Idea**: Relax the constraint $\gamma_1 + \gamma_0 = 1$ and allow $\gamma_0$ to take free values. Deriving from the zero-expectation condition $\mathbb{E}[\gamma_1 \epsilon_\theta(x_t,c,t) + \gamma_0 \epsilon_\theta(x_t,t)] = 0$, a closed-form solution for $\gamma_0$ is obtained, which can be directly retrieved from a precomputed look-up table.

## Method

### Overall Architecture

ReCFG is a post-processing correction of CFG. Standard CFG uses $\hat{\epsilon} = \gamma \epsilon_\theta(x_t,c,t) + (1-\gamma) \epsilon_\theta(x_t,t)$, which ReCFG replaces with $\hat{\epsilon} = \gamma_1 \epsilon_\theta(x_t,c,t) + \gamma_0 \epsilon_\theta(x_t,t)$, where $\gamma_0$ is no longer $1-\gamma_1$, but a dynamic value retrieved from a look-up table based on condition $c$ and timestep $t$. The look-up table is precomputed by traversing the training data, and is directly queried during inference, incurring virtually no impact on speed.

### Key Designs

1. **Theoretically proving the expectation shift (Theorem 1 & 2)**:
    - **Function**: Theoretically describes the shift in the generated distribution caused by CFG with high precision.
    - **Mechanism**: Theorem 1 proves that although CFG is compatible with the standard diffusion training objective (up to constant), its denoising process is not the reverse of any forward diffusion. This is because the expectation of the unconditional score function under the conditional distribution $\mathbb{E}_{q_t(x_t|c)}[\nabla \log q_t(x_t)]$ is non-zero. Theorem 2 quantifies the exact shift under the toy distribution: when $T \to \infty$, the expectation of the CFG generated distribution is $c \cdot \phi(\gamma)$, where $\phi(1)=1, \phi(3)=2, \phi(\gamma) \geq 2$ for $\gamma>3$. This implies that at $\gamma=3$, the expectation is already twice the true value.
    - **Design Motivation**: Elevate the theoretical flaw of CFG from "known but overlooked" to "precisely quantified and rectifiable."

2. **Rectifying the guidance coefficients with ReCFG (Theorem 3 & Closed-form Solution)**:
    - **Function**: Relaxes the coefficient constraints to eliminate the expectation shift.
    - **Mechanism**: Generalizes the guidance formula to $s_{t,\gamma_1,\gamma_0}(x,c) = \gamma_1 \otimes \nabla \log q_t(x|c) + \gamma_0 \otimes \nabla \log q_t(x)$, where $\gamma_1, \gamma_0 \in \mathbb{R}^D$ are pixel-wise functions related to the condition and timestep, and $\otimes$ denotes element-wise multiplication. Designed constraints: (1) $\gamma_{1,i} > 1$ ensures conditional fidelity enhancement; (2) the expectation shift is zero: $\mathbb{E}[\gamma_1 \epsilon(x_t,c,t) + \gamma_0 \epsilon(x_t,t)] = 0$; (3) the variance is not larger than the true distribution (more concentrated is better). From condition (2), the closed-form solution yields: $\gamma_0 = (1-\gamma_1) \cdot \mathbb{E}[\epsilon_\theta(x_t,c,t)] / \mathbb{E}[\epsilon_\theta(x_t,t)]$. Theorem 4 verifies that the variance indeed decreases under the toy distribution (when $\gamma_{0,i} \leq 0$ and $\gamma_{1,i}+\gamma_{0,i} \geq 1$).
    - **Design Motivation**: Instead of changing $\gamma_1$ (retaining control over the original guidance strength), adjust $\gamma_0$ to compensate for the shift. This allows ReCFG to replace CFG seamlessly, where users still only need to tune one hyperparameter $\gamma_1$.

3. **Precomputing Look-Up Tables**:
    - **Function**: Achieves high-efficiency post-processing correction at inference time.
    - **Mechanism**: Given condition $c$, samples from $q_0(x_0|c)$ in the training data are traversed. For each timestep $t$, $(\epsilon_\theta(x_t,c,t), \epsilon_\theta(x_t,t))$ are collected to compute the expectation ratio $\mathbb{E}[\epsilon_\theta(x_t,c,t)] / \mathbb{E}[\epsilon_\theta(x_t,t)]$, which is stored in a look-up table. During inference, $\gamma_0 = (1-\gamma_1) \times \text{ratio}$ is directly calculated based on $\gamma_1$. The look-up table is pixel-wise, meaning each pixel can have different correction coefficients across different timesteps.
    - **Design Motivation**: The closed-form solution allows $\gamma_0$ to be precisely precomputed, eliminating the need for any additional forward passes during inference. The pixel-wise $\gamma_0$ enables a more flexible and precise guidance than global CFG.

### Loss & Training

ReCFG does not require any training or fine-tuning. The core operation is precomputing the look-up table:
1. For each class/condition $c$, sample several $x_0 \sim q_0(x_0|c)$.
2. For each timestep $t$, apply forward noise to get $x_t$, and compute $\epsilon_\theta(x_t,c,t)$ and $\epsilon_\theta(x_t,t)$.
3. Compute and store the expectation ratio.
4. Query the table during inference and multiply by $(1-\gamma_1)$ to obtain $\gamma_0$.

## Key Experimental Results

### Main Results

ImageNet 512×512 (EDM2):

| Model | Method | NFE | FD_DINOv2↓ | FID↓ | Precision↑ | Recall↑ |
|------|------|-----|-----------|------|-----------|---------|
| EDM2-S | CFG | 63 | 52.32 | 2.29 | 0.83 | 0.59 |
| EDM2-S | **ReCFG** | 63 | **50.56** | **2.23** | 0.83 | 0.59 |
| EDM2-M | CFG | 63 | 41.98 | 2.12 | 0.81 | 0.60 |
| EDM2-M | **ReCFG** | 63 | **41.55** | **2.06** | 0.81 | 0.61 |
| EDM2-L | CFG | 63 | 38.20 | 1.96 | 0.81 | 0.62 |
| EDM2-L | **ReCFG** | 63 | **36.75** | **1.89** | 0.81 | 0.62 |

CC12M 512×512 (SD3):

| Method | γ₁ | NFE | CLIP-S↑ | FD_DINOv2↓ | MPS↑ |
|------|-----|-----|---------|-----------|------|
| CFG | 7.5 | 10 | 0.262 | 1105.51 | 9.828 |
| **ReCFG** | 7.5 | 10 | **0.263** | **1010.14** | **10.250** |
| RescaleCFG + ReCFG | 7.5 | 10 | **0.268** | **979.87** | **11.336** |
| CFG | 5.0 | 10 | 0.268 | 1053.44 | 10.883 |
| **ReCFG** | 5.0 | 10 | **0.269** | **999.48** | **11.031** |

ImageNet 256×256 (DiT-XL/2 & LDM):

| Model | Method | γ₁ | NFE | FID↓ |
|------|------|-----|-----|------|
| DiT-XL/2 | CFG | 1.50 | 250 | 2.27 |
| DiT-XL/2 | **ReCFG** | 1.50 | 250 | **2.13** |
| LDM | CFG | 5.0 | 20 | 18.87 |
| LDM | **ReCFG** | 5.0 | 20 | **16.95** |
| LDM | CFG | 2.0 | 20 | 5.32 |
| LDM | **ReCFG** | 2.0 | 20 | **4.40** |
| LDM | CFG | 5.0 | 10 | 16.78 |
| LDM | **ReCFG** | 5.0 | 10 | **14.46** |

### Ablation Study

| γ₁ Setting | CFG FID | ReCFG FID | Improvement | Note |
|---------|---------|-----------|------|------|
| 5.0 (LDM, 20 steps) | 18.87 | 16.95 | -1.92 | Most significant improvement under large γ |
| 3.0 (LDM, 20 steps) | 11.46 | 9.78 | -1.68 | Obvious improvement under moderate γ |
| 2.0 (LDM, 20 steps) | 5.32 | 4.40 | -0.92 | Smaller improvement under small γ but still consistent |
| 1.5 (LDM, 20 steps) | 5.36 | 4.78 | -0.58 | Minimal improvement under γ close to 1 |

### Key Findings

- ReCFG consistently improves FID/FD_DINOv2 across all tested diffusion models (EDM2, SD3, LDM, DiT).
- The larger the guidance strength $\gamma_1$, the greater the improvement of ReCFG—consistent with the theoretical prediction (large γ generates a more severe expectation shift).
- ReCFG is compatible with RescaleCFG, and combining them yields even better results (MPS on SD3 increases from 9.828 to 11.336).
- Visualization of the look-up table shows that the expectation ratio varies dramatically across different pixels and timesteps with no uniform pattern, proving the necessity of pixel-wise correction.
- With fewer NFEs (fewer sampling steps), the improvement of ReCFG is more pronounced (greater improvement at 10 steps than at 20 steps).
- Both Precision and Recall are improved or maintained, showing that ReCFG is not a simple precision-recall trade-off.

## Highlights & Insights

- **Rigorous theoretical foundation**: Not only points out the flaws of CFG, but also precisely describes the essence of the problem and solutions through 5 theorems from multiple perspectives. The conclusion of Theorem 2 showing that "expectation shift is 2× when $\gamma=3$" is highly intuitive and impactful.
- **Strong practicality**: Complete post-processing, zero training cost, zero extra inference overhead (the table lookup is negligible). It can be immediately applied to all existing diffusion models using CFG. This "free lunch" type of improvement is highly appealing.
- **Compatibility with RescaleCFG**: ReCFG resolves the expectation shift, while RescaleCFG resolves the variance issue. The two are orthogonal and complementary, working even better when combined.
- **Insight from pixel-wise dynamic coefficients**: Look-up table visualization shows that different spatial locations require different guidance strengths at different denoising steps. This challenges the usage of globally uniform coefficients in standard CFG, suggesting that future guidance methods should be more spatially adaptive.

## Limitations & Future Work

- The look-up table requires traversing a subset of the training data to precompute. For open-vocabulary text-to-image models, it cannot cover all possible conditions.
- The closed-form solution is based on first-order approximations (assuming $\Delta_t=0$ and then recursively generalizing), which may accumulate multi-step errors.
- The variance guarantee of Theorem 4 only holds under the toy distribution; the behavior of variance in general situations is not fully controllable yet.
- Possible improvements: online calculation of expectation ratios (to avoid look-up table coverage limits), extension to other guidance forms (e.g., classifier guidance), theoretical analysis of more general distribution families.

## Related Work & Insights

- **vs Standard CFG**: CFG uses the constraint $\gamma + (1-\gamma) = 1$, which leads to expectation shift. ReCFG relaxes this constraint and provides a closed-form correction solution, achieving comprehensive improvements theoretically and experimentally.
- **vs RescaleCFG**: RescaleCFG (Lin et al.) mitigates the over-saturation issue of CFG by scaling the outputs, mainly addressing the variance level. ReCFG tackles the shift at the expectation level, making the two approaches orthogonal and complementary.
- **vs APG/CFG++**: Other works improving CFG have also noticed the theoretical flaws of guidance, but ReCFG is the first to present a precise closed-form correction scheme.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Solid theoretical contribution. Extracts a closed-form solution from a known but overlooked problem; elegant mathematical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 mainstream diffusion models, multiple resolutions, and NFE settings, class/text conditioning, as well as compatibility with other methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theorem-proof structure, intuitive toy example, and insightful look-up table visualization.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play improvement for all CFG models, zero-cost FID enhancement, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2026\] Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling](../../CVPR2026/image_generation/accelerating_diffusion_via_hybrid_data-pipeline_parallelism_based_on_conditional.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)
- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)

</div>

<!-- RELATED:END -->
