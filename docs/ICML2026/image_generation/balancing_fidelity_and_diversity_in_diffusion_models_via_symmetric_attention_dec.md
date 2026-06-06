---
title: >-
  [Paper Note] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective
description: >-
  [ICML 2026][Image Generation][Attention Decomposition] This work decomposes the $\mathbf{QK}^\top$ attention matrix in diffusion models into symmetric components (representing the energy landscape) and anti-symmetric com…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Attention Decomposition"
  - "Hopfield Networks"
  - "Fidelity-Diversity Trade-off"
  - "Skew-symmetric Perturbation"
  - "Associative Memory"
date: 2026-05-08
content_hash: 59e98fed69fefbb6
---

# Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.27476](https://arxiv.org/abs/2605.27476)  
**Code**: https://github.com (Available)  
**Area**: Diffusion Models  
**Keywords**: Attention Decomposition, Hopfield Networks, Fidelity-Diversity Trade-off, Skew-symmetric Perturbation, Associative Memory  

## TL;DR
This work decomposes the $\mathbf{QK}^\top$ attention matrix in diffusion models into symmetric components (representing the energy landscape) and anti-symmetric components (representing circulation dynamics). Based on this, Hopfield-style stability metrics are derived to diagnose metastable mixing, and a training-free controllable trade-off between fidelity and diversity is achieved by regulating the anti-symmetric components.

## Background & Motivation

**Background**: Diffusion models (such as DDPM and SDXL) have become the mainstream paradigm for image generation. Their success relies heavily on the attention mechanism to establish global context and long-range dependencies during the denoising process. Attention allows the model to build rich compositional associations between spatial positions, enhancing generation diversity and novelty.

**Limitations of Prior Work**: However, global connectivity also prone to semantic leakage—where materials and textures of different objects are improperly mixed (e.g., material blending between two distinct objects), resulting in structurally incoherent artifacts. Crucially, this beneficial contextual integration and harmful semantic leakage share the same underlying mechanism, making them difficult to distinguish.

**Key Challenge**: A fundamental trade-off exists between generation fidelity and diversity. High-stability retrieval tends to converge to repetitive perspectives and features, sacrificing diversity; whereas low-stability retrieval brings diversity but is accompanied by structural fragmentation and artifacts. Existing methods lack theoretical tools to (1) identify when attention falls into metastable mixing and (2) controllably adjust this trade-off.

**Goal**: To establish a principled framework for analyzing the internal structure of the attention matrix, quantitatively diagnosing metastable mixing, and providing a training-free adjustable knob to control the fidelity-diversity trade-off.

**Key Insight**: The authors observe that $\mathbf{QK}^\top$ is formally equivalent to the associative memory matrix of classical Hopfield networks. After decomposing it into symmetric and anti-symmetric parts, the symmetric part defines the energy landscape (determining retrieval stability), while the anti-symmetric part drives circulation dynamics (which can break metastability). This perfectly aligns with the theory in classical asymmetric Hopfield networks that "increasing asymmetry leads to an exponential decrease in the number of attractors."

**Core Idea**: Use the symmetric-antisymmetric decomposition of the attention matrix to diagnose generation quality, and utilize the scaling of the anti-symmetric component as a "circulation knob" to regulate the fidelity-diversity trade-off during inference.

## Method

### Overall Architecture
The input consists of feature maps $\mathbf{X} \in \mathbb{R}^{L \times d_{\text{in}}}$ from the self-attention layers in the diffusion model's UNet, where $L$ is the number of spatial positions. The method follows three steps: (1) Decomposing $\mathbf{QK}^\top = \mathbf{XWX}^\top$ into a symmetric component $\mathbf{M}_{\text{sym}}$ and an anti-symmetric component $\mathbf{M}_{\text{skew}}$; (2) Utilizing the symmetric component to derive three Hopfield-style stability metrics to diagnose the retrieval state; (3) Injecting circulation perturbations by scaling the anti-symmetric component and blending it with the baseline retrieval for output. The entire process is training-free and only modifies the attention matrix during inference.

### Key Designs

1.  **Associative Memory Decomposition and Energy Landscape**:
    - **Function**: Decomposes the attention matrix into two analyzable parts: symmetric and anti-symmetric.
    - **Mechanism**: Defines the interaction weight matrix $\mathbf{W} = \mathbf{W}_Q \mathbf{W}_K^\top$, decomposing it into a symmetric part $\mathbf{S} = (\mathbf{W} + \mathbf{W}^\top)/2$ and an anti-symmetric part $\mathbf{N} = (\mathbf{W} - \mathbf{W}^\top)/2$, such that $\mathbf{QK}^\top = \mathbf{XSX}^\top + \mathbf{XNX}^\top$. The symmetric component defines the Hopfield energy $E_\mathbf{X}(\xi) = -\frac{1}{2}\xi^\top \mathbf{M}_{\text{sym}}(\mathbf{X})\xi$, while the anti-symmetric component contributes nothing to the quadratic energy ($\xi^\top \mathbf{M}_{\text{skew}} \xi = 0$) and only drives circulation. Based on this, three stability metrics are derived: energy $E_\mathbf{X}$, instability ratio $r_\mathbf{X}$, and alignment score $\mathbf{Align}_\mathbf{X}$.
    - **Design Motivation**: Classical Hopfield theory only handles symmetric matrices, whereas $\mathbf{QK}^\top$ is generally asymmetric. Decomposition allows for the separate analysis of energy stability and circulation dynamics, providing a theoretical foundation for subsequent controllable adjustments.

2.  **Skew Perturbation**:
    - **Function**: Breaks metastable mixing and regulates the fidelity-diversity trade-off by scaling the anti-symmetric component.
    - **Mechanism**: Applies a scaling factor $\alpha$ to the anti-symmetric component to obtain the perturbed retrieval $\Xi_\alpha = \Phi(\mathbf{XSX}^\top + \alpha \cdot \mathbf{XNX}^\top) \mathbf{X}$. The difference vector $\Delta = \Xi_\alpha - \Xi$ is calculated, and the injection intensity is controlled via a blending coefficient $\beta$: $\Xi_{\text{blended}} = \Xi + \beta \Delta$. Here, $\alpha$ controls the intensity of the circulation perturbation, and $\beta$ controls the injection ratio.
    - **Design Motivation**: Drawing from the classical conclusion that increasing asymmetry leads to an exponential decrease in the number of attractors. Moderate circulation injection can break metastability (fixing artifacts), while excessive injection destroys existing sound structures.

3.  **Adaptive Circulation Control**:
    - **Function**: Adaptively adjusts perturbation intensity based on the current state of each sample to avoid excessive interference with high-quality samples.
    - **Mechanism**: Defines a functional symmetry index $\eta_\mathbf{M}(\mathbf{X}) = (\|\mathbf{M}_{\text{sym}}\|_F^2 - \|\mathbf{M}_{\text{skew}}\|_F^2) / (\|\mathbf{M}_{\text{sym}}\|_F^2 + \|\mathbf{M}_{\text{skew}}\|_F^2)$ to measure the degree of symmetry dominance in the current retrieval. Correspondingly, effective scaling $\alpha_{\text{eff}} = (\alpha - 1)\bar{\eta}_\mathbf{M}$ and effective blending $\beta_{\text{eff}} = \beta(1 - \bar{\eta}_\mathbf{M})$ are adjusted, ensuring low-performance samples receive stronger perturbations while high-performance samples are less disturbed.
    - **Design Motivation**: Applying fixed $(\alpha, \beta)$ to all samples is suboptimal. Low-quality samples require stronger circulation correction, whereas high-quality samples are already at a good operating point and should not be over-perturbed.

## Key Experimental Results

### Main Results
On SDXL, 10K samples were generated using 1K COCO2014 prompts to evaluate the correlation between stability metrics and external metrics, as well as the effect of circulation perturbations.

| Metric | Baseline | $\alpha{=}1.05, \beta{=}5$ | $\alpha{=}1.10, \beta{=}5$ | $\alpha{=}1.15, \beta{=}4$ |
|------|----------|---------------------------|---------------------------|---------------------------|
| Aesthetic Score ↑ | 5.644 | 5.670 | 5.717 | 5.704 |
| ImageReward ↑ | 0.546 | 0.558 | 0.442 | 0.445 |
| CLIPScore ↑ | 0.264 | 0.263 | 0.259 | 0.260 |
| $\mathbf{Align}_\mathbf{X}$ | 0.669 | 0.651 | 0.650 | 0.637 |

### Ablation Study: Repair Effects on Low-Quality Subsets
Paired changes $\Delta$ after applying perturbation $(\alpha{=}1.05)$ for the worst 20% baseline samples of each metric:

| Target Subset | $\Delta$ Aesthetic | $\Delta$ ImageReward | $\Delta$ CLIPScore |
|----------|-------------------|---------------------|-------------------|
| Worst 20% Aesthetic | +0.166 | +0.043 | +0.004 |
| Worst 20% ImageReward | +0.022 | +0.453 | +0.004 |
| Worst 20% CLIPScore | +0.019 | +0.116 | +0.0065 |

### Adaptive Control vs. Static Control (350 COCO samples)

| Method | IR ↑ | CLIP ↑ | HPS ↑ | AES ↑ |
|------|------|--------|-------|-------|
| Baseline | 0.487 | 0.264 | 0.270 | 5.64 |
| Static Moderate $(\alpha{=}1.05, \beta{=}3)$ | 0.546 | 0.262 | 0.273 | 5.66 |
| Adaptive Moderate | 0.522 | 0.264 | 0.272 | 5.64 |
| Static Excessive $(\alpha{=}1.20, \beta{=}5)$ | -1.486 | 0.207 | 0.157 | 5.23 |
| **Adaptive Excessive** | **0.568** | **0.264** | **0.274** | **5.65** |

### Key Findings
- Significant Spearman correlations exist between stability metrics and external quality metrics: $\mathbf{Align}_\mathbf{X}$ correlates positively with Aesthetic Score ($\rho = +0.296$) and negatively with LPIPS diversity ($\rho = -0.297$), validating the fidelity-diversity trade-off.
- For low-quality subsets (worst 20%), circulation perturbation brings consistent improvements; for high-quality subsets (best 20%), excessive perturbation reduces quality, demonstrating state-dependent repair characteristics.
- Adaptive control excels in excessive perturbation settings: the ImageReward of the static method plummets to -1.486, while the adaptive method recovers to 0.568, even exceeding the baseline.
- Compared to global attention temperature scaling $\mathbf{QK}^\top / \tau$, circulation perturbation can more selectively suppress weakly supported mixing artifacts without generating improper structural replications (such as extra limbs).

## Highlights & Insights
- **Insight from Symmetric-Antisymmetric Decomposition**: By treating $\mathbf{QK}^\top$ as an associative memory matrix and performing symmetric decomposition, a bridge is built between attention and Hopfield networks. The symmetric component captures global object structure while the anti-symmetric component captures fine-grained irregular details; this finding is transferable to the analysis of any Transformer-based generative model.
- **Training-free Inference-time Controllable Generation**: Generation quality is regulated simply by modifying the attention matrix during inference through two scalar parameters $(\alpha, \beta)$, without requiring additional training or fine-tuning. This lightweight intervention approach can be generalized to LLMs and other Transformer architectures.
- **Adaptive Control to Avoid Over-correction**: Utilizing the functional symmetry index $\eta_\mathbf{M}$ to achieve sample-level adaptive perturbation solves the "one-size-fits-all" problem of fixed hyperparameters, demonstrating significant robustness advantages in excessive perturbation settings.

## Limitations & Future Work
- Experiments were primarily validated on the SDXL UNet architecture and have not yet been extended to modern Transformer-based diffusion models such as DiT (e.g., FLUX).
- The aggregation strategy for adaptive control (mean across batch and heads) is relatively simple; more optimal head-level or layer-level adaptive strategies may exist.
- Currently, only self-attention is considered; the $\mathbf{QK}^\top$ in cross-attention similarly possesses a symmetric-antisymmetric structure worth further exploration.
- This theoretical framework can naturally be extended to the attention analysis of LLMs to detect and regulate "metastable" behaviors in text generation.

## Related Work & Insights
- Ramsauer et al. (2021) formalized self-attention as the retrieval step of modern Hopfield networks; this work introduces a feature-level (rather than token-level) associative memory perspective based on that.
- Singh et al. (1995) found that increasing asymmetry in asymmetric Hopfield networks leads to an exponential decrease in attractors, directly inspiring the circulation perturbation mechanism of this study.
- Hwang et al. (2019) studied the impact of connection matrix symmetry on attractor structures, providing a theoretical basis for the design of $\eta_\mathbf{M}$ in adaptive control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](../../NeurIPS2025/image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[ICML 2026\] Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->
