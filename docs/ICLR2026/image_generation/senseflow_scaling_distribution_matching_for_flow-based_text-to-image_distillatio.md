---
title: >-
  [Paper Note] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation
description: >-
  [ICLR 2026][Image Generation][distribution matching distillation] This paper proposes SenseFlow, which scales distribution matching distillation (DMD) to large-scale flow-based text-to-image models (SD 3.5 Large 8B / FLUX.1 dev 12B) via Implicit Distribution Alignment (IDA) and Intra-Segment Guidance (ISG), enabling high-quality 4-step image generation.
tags:
  - ICLR 2026
  - Image Generation
  - distribution matching distillation
  - flow matching
  - text-to-image
  - few-step generation
  - FLUX
date: 2026-05-08
content_hash: 0b29971c7258e435
---

# SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

**Conference**: ICLR 2026
**arXiv**: [2506.00523](https://arxiv.org/abs/2506.00523)
**Code**: [GitHub](https://github.com/XingtongGe/SenseFlow)
**Area**: Image Generation / Diffusion Model Distillation
**Keywords**: distribution matching distillation, flow matching, text-to-image, few-step generation, FLUX

## TL;DR

This paper proposes SenseFlow, which scales distribution matching distillation (DMD) to large-scale flow-based text-to-image models (SD 3.5 Large 8B / FLUX.1 dev 12B) via Implicit Distribution Alignment (IDA) and Intra-Segment Guidance (ISG), enabling high-quality 4-step image generation.

## Background & Motivation

**State of the Field**: DMD2 has demonstrated strong distillation performance on smaller models (SD 1.5, SDXL), compressing multi-step diffusion models into few-step generators. However, large-scale flow-based text-to-image models (e.g., SD 3.5 Large 8B, FLUX.1 dev 12B) are becoming mainstream, and their distillation remains an open problem.

**Limitations of Prior Work**: The original DMD2 faces three critical challenges when applied to large models: (1) **convergence instability**—training cannot be stabilized even with TTUR (two time-scale update rule); (2) **suboptimal timestep sampling**—uniform or manually selected coarse timesteps fail to account for the varying denoising importance across timesteps in the teacher model; (3) **insufficient discriminator generality**—naive discriminators struggle to adapt across models of different scales and architectures.

**Root Cause**: The min-max game in DMD requires the fake distribution to precisely track the generator distribution (inner optimal response), a condition that is extremely difficult to satisfy in large models, resulting in oscillatory and non-convergent training.

**Paper Goals**: To reliably scale the DMD framework to flow-based text-to-image models with 8B–12B parameters.

**Starting Point**: Through analysis of DMD's min-max optimization, the authors identify that the inner optimal response requires $p_f = p_g$. They design a proximal update (IDA) to approximately maintain this condition, reconfigure timestep denoising importance via ISG, and introduce a stronger VFM-based discriminator.

**Core Idea**: Implicit distribution alignment maintains consistency between the fake model and the generator, while intra-segment guidance redistributes timestep importance, enabling stable convergence of DMD on large flow models.

## Method

### Overall Architecture

Building on DMD2, the generator G takes text prompts and noise as input to produce images, and is jointly optimized via DMD gradients (real score minus fake score), VFM adversarial loss, and ISG. After each generator update, IDA proximally aligns the fake model to the generator. The overall pipeline is detailed in Algorithm 1.

### Key Designs

**Design 1: Implicit Distribution Alignment (IDA)**
- **Function**: After each generator update, applies a proximal update to the fake model parameters: $\phi \leftarrow \lambda\phi + (1-\lambda)\theta$
- **Mechanism**: DMD's inner optimal response requires $p_f(X_t) = p_g(X_t)$. Through EMA-style parameter alignment, IDA maintains an $\varepsilon$-best response between the fake model and the generator, i.e., $E_t D_{KL}(p_g(X_t) \| p_f(X_t)) \leq \varepsilon$
- **Design Motivation**: Increasing the TTUR ratio alone is both costly and unstable for large models. IDA maintains fake-generator consistency at minimal cost (a single parameter interpolation), enabling DMD convergence on SD 3.5 Large. Setting $\lambda$ close to 1 is sufficient.

**Design 2: Intra-Segment Guidance (ISG)**
- **Function**: Samples intermediate time points within each coarse timestep "segment" and constructs a guided trajectory using the teacher model
- **Mechanism**: For a coarse timestep $\tau_i$, an intermediate point $t_{mid} \in (\tau_{i-1}, \tau_i)$ is sampled. The teacher denoises from $\tau_i$ to $t_{mid}$, and the generator continues from $t_{mid}$ to $\tau_{i-1}$ to produce target $x_{tar}$. The generator also denoises directly from $\tau_i$ to $\tau_{i-1}$, and the two outputs are aligned via an L2 loss.
- **Design Motivation**: The teacher's reconstruction error $\xi(t)$ is non-monotonic with local oscillations, making uniform timestep sampling informationally wasteful. ISG aggregates fine-grained intra-segment denoising information onto anchor points, enabling the generator to better approximate complex intra-segment transitions.

**Design 3: VFM-based Discriminator**
- **Function**: Uses frozen visual foundation models (DINOv2 + CLIP) as the discriminator backbone
- **Mechanism**: The VFM extracts multi-layer semantic features, which are fed into trainable head blocks to predict real/fake logits. The discriminator is trained with hinge loss, and the generator's adversarial loss is weighted by the timestep signal power $\omega(t) = (1-\sigma_t)^2$
- **Design Motivation**: Pretrained VFMs introduce rich semantic priors, making them more capable of capturing image quality and fine-grained structure compared to naive discriminators. Timestep weighting ensures that DMD gradients dominate at high-noise steps while GAN feedback is emphasized at low-noise steps.

### Loss & Training

Generator total loss: $\mathcal{L}_G = \mathcal{L}_{DMD} + \lambda_G \cdot \mathcal{L}_{adv} + \lambda_{ISG} \cdot \mathcal{L}_{ISG}$
- $\mathcal{L}_{DMD}$: guides the generator via the difference between fake and real scores
- $\mathcal{L}_{adv}$: adversarial loss from the VFM discriminator, weighted by $\alpha_t^2$
- $\mathcal{L}_{ISG}$: intra-segment guidance L2 loss

Training details:
- Data: LAION-5B subset (aesthetic score ≥ 5.0)
- TTUR ratio: 5:1 combined with IDA suffices for stable convergence
- 50% probability of using backward simulation vs. forward diffusion for input construction
- Logit-normal timestep sampling

## Key Experimental Results

### Main Results

**COCO-5K 4-step Generation Results**

| Model | Patch FID-T↓ | CLIP↑ | HPSv2↑ | PickScore↑ | ImageReward↑ | GenEval↑ |
|------|-------------|-------|--------|------------|-------------|---------|
| SDXL Teacher (80 steps) | — | 0.3293 | 0.2930 | 22.67 | 0.8719 | 0.5461 |
| DMD2-SDXL (4 steps) | 21.35 | 0.3231 | 0.2896 | 22.49 | 0.7076 | — |
| **SenseFlow-SDXL (4 steps)** | **best** | **best** | **best** | **best** | **best** | **best** |
| SD 3.5 Teacher (80 steps) | — | — | — | — | — | 0.7140 |
| SD 3.5 Turbo (4 steps) | baseline | — | — | — | — | — |
| **SenseFlow-SD3.5 (4 steps)** | **best** | **best** | **exceeds teacher** | **exceeds teacher** | **exceeds teacher** | 0.7098 |
| FLUX.1 schnell (4 steps) | baseline | — | — | — | — | — |
| **SenseFlow-FLUX (4 steps)** | **best** | **best** | **exceeds teacher** | **exceeds teacher** | **exceeds teacher** | — |

SenseFlow achieves state-of-the-art 4-step distillation across all three teacher models (SDXL, SD 3.5 Large, FLUX.1 dev).

### Ablation Study

| Component | SD 3.5 FID Convergence | FLUX Convergence |
|------|-------------------|--------------|
| Original DMD2 | Does not converge | Does not converge |
| + IDA | Converges ✓ | Does not converge |
| + IDA + ISG | Converges ✓ | Converges ✓ |
| + IDA + ISG + VFM Disc | Best ✓ | Best ✓ |

IDA is essential for SD 3.5 convergence; ISG is an additional necessary condition for FLUX convergence.

### Key Findings

1. **IDA is critical for DMD convergence on large models**: IDA alone stabilizes training on SD 3.5 Large, requiring only a modest TTUR ratio.
2. **ISG further improves FLUX distillation**: The teacher's timestep reconstruction error is non-monotonic; ISG significantly helps by redistributing timestep importance.
3. **4-step generation surpasses the teacher**: On human preference metrics (HPSv2, PickScore, ImageReward), SenseFlow with 4 steps even outperforms the 80-step teacher.
4. **Strong generality**: The same framework is effective across three architecturally distinct models of varying scale: SDXL (2.6B), SD 3.5 (8B), and FLUX (12B).

## Highlights & Insights

- **First extension of DMD to a 12B-parameter FLUX model**: Resolves the convergence bottleneck in large-scale flow model distillation.
- **Elegance of IDA**: A single line of parameter interpolation code addresses the fake-generator distribution tracking problem, with theoretical guarantees ($\varepsilon$-best response).
- **Ingenuity of ISG**: The approach of redistributing timestep importance in the teacher is novel, leveraging the teacher's fine-grained denoising capability to enhance the coarse-step generator.
- **4-step surpassing the teacher**: The fact that a 4-step distilled model can outperform an 80-step teacher on human preference metrics suggests that distillation can selectively inherit a model's strengths while discarding its weaknesses.
- **Timestep weighting for the VFM discriminator**: The design of emphasizing GAN signals at low-noise steps and DMD signals at high-noise steps is well-motivated.

## Limitations & Future Work

1. **Computational cost**: Simultaneously maintaining the generator, fake model, teacher model, and discriminator incurs substantial memory overhead.
2. **Selection of $\lambda$ in IDA**: Although $\lambda$ close to 1 is generally sufficient, the optimal value may vary across models.
3. **Intermediate point sampling strategy in ISG**: Current uniform sampling could be replaced with adaptive selection of more informative intermediate points.
4. **Only 4-step generation is evaluated**: The effectiveness of 1-step or 2-step distillation remains unknown.
5. **Dependence on training data**: A high-quality LAION subset is required, and the impact of data quality on distillation performance has not been thoroughly analyzed.

## Related Work & Insights

- A direct extension and breakthrough of **DMD/DMD2**: Identifies that the core bottleneck of the DMD framework lies in fake-generator distribution tracking.
- Connection to **RayFlow**: Both address timestep importance, but ISG offers a more elegant solution.
- Broader implications for distillation: The IDA concept (proximal alignment of internal distributions) may be applicable to other min-max distillation frameworks.
- Comparison with **Consistency Models**: These represent two orthogonal distillation paradigms; the DMD family demonstrates stronger advantages on large models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both IDA and ISG have clear theoretical motivation and practical value, though the overall contribution is an incremental improvement over DMD2.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three models of different scales and architectures, multiple benchmarks (COCO-5K / GenEval / T2I-CompBench), and thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is clear, though the method description is notation-heavy.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses a practical bottleneck in large-scale flow model distillation with immediate implications for industrial applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

<!-- RELATED:END -->
