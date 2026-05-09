---
title: >-
  [Paper Note] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning
description: >-
  [ICCV 2025][Image Generation][Diffusion model alignment] ShortFT is proposed to construct denoising shortcuts using trajectory-preserving few-step diffusion models, substantially compressing the original lengthy denoising chain to enable complete end-to-end reward gradient backpropagation, achieving efficient and effective alignment of diffusion models with reward functions.
tags:
  - ICCV 2025
  - Image Generation
  - Diffusion model alignment
  - reward fine-tuning
  - denoising shortcut
  - trajectory-preserving distillation
  - timestep-aware LoRA
  - backpropagation
date: 2026-05-08
content_hash: 62fb31f72d23d749
---

# ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning

**Conference**: ICCV 2025
**arXiv**: [2507.22604](https://arxiv.org/abs/2507.22604)
**Code**: [https://xiefan-guo.github.io/shortft](https://xiefan-guo.github.io/shortft)
**Area**: Image Generation
**Keywords**: Diffusion model alignment, reward fine-tuning, denoising shortcut, trajectory-preserving distillation, timestep-aware LoRA, backpropagation

## TL;DR

ShortFT is proposed to construct denoising shortcuts using trajectory-preserving few-step diffusion models, substantially compressing the original lengthy denoising chain to enable complete end-to-end reward gradient backpropagation, achieving efficient and effective alignment of diffusion models with reward functions.

## Background & Motivation

Diffusion models achieve remarkable performance in text-to-image generation, yet a fundamental conflict exists between their maximum likelihood training objective and downstream goals such as aesthetics, safety, and text-image consistency, necessitating alignment via reward functions.

Existing alignment methods fall into three categories:

**Reinforcement learning methods** (RLHF): Approaches such as DDPO and DPOK model the denoising process as an MDP and optimize it with RL. However, these methods suffer from high gradient variance, low efficiency, and poor adaptability to diverse prompts.

**Backpropagation methods (truncated to the latter half)**: Methods such as DRaFT-K and ReFL perform backpropagation only over the latter portion of the denoising chain. The drawback is that direct supervision of early denoising stages is omitted, leading to suboptimal text-image alignment.

**Backpropagation methods (partial gradient truncation)**: Methods such as AlignProp and DRTune disable gradients for certain terms within the denoising chain (retaining gradients through $\alpha_t \mathbf{x}_t$ while truncating those through $\beta_t \epsilon_\theta(\mathbf{x}_t, t)$), and use gradient checkpointing to propagate signals to early steps. Nevertheless, these approaches introduce gradient bias, cause optimization instability, and incur high computational cost.

**Root Cause**: Standard DDIM requires approximately 50 denoising steps, corresponding to a backpropagation chain of roughly 50 layers, leading to memory explosion and gradient explosion. Existing solutions either truncate the chain (losing information) or truncate the gradients (introducing bias), both of which are compromises.

**Paper Goals**: Since the fundamental problem lies in the excessive length of the denoising chain, the paper addresses this at its root by leveraging few-step diffusion models obtained via trajectory-preserving distillation (e.g., Hyper-SD 4-step) to construct "denoising shortcuts," compressing the 50-step chain into a handful of steps and thereby enabling complete gradient backpropagation.

## Method

### Overall Architecture

The ShortFT framework comprises three core components:

1. **Denoising Shortcut**: Utilizes Hyper-SD (4-step) to skip a large number of intermediate steps within the denoising chain.
2. **Timestep-aware LoRA**: Assigns independent LoRA parameters to different temporal segments of the denoising process.
3. **Progressive Training Strategy**: Eliminates training-inference discrepancy through staged training.

The optimization objective is consistent with DRaFT, AlignProp, and related methods:

$$J(\theta) = \mathbb{E}_{\mathbf{c}, \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{1})} \left[ \mathcal{R}\left(\text{Sample}(\theta, \mathbf{c}, \mathbf{x}_T), \mathbf{c}\right) \right]$$

Reward function $\mathcal{R}$ is maximized via gradient ascent. The key distinction is that the Sample procedure employs the compressed denoising chain.

### Key Design 1: Denoising Shortcut

Trajectory-preserving few-step models (e.g., Hyper-SD) learn through distillation to skip multiple steps while maintaining the original denoising trajectory. Compared to naive single-step DDIM denoising (which produces blurry, poorly structured outputs), shortcut outputs are highly consistent with the original SD 1.5 (with smaller HPS v2 deviation).

In practice, the 50-step denoising chain is divided into $k=4$ segments, with LoRA-corresponding timesteps set to {761, 501, 261, 1}. The denoising shortcuts operate over the following intervals:

- Timestep 741 → 501
- Timestep 481 → 261
- Timestep 241 → 1

The original ~50-step chain is compressed to approximately 7–8 steps, substantially shortening the backpropagation graph.

### Key Design 2: Timestep-aware LoRA

Prior work (eDiff-I) has revealed temporal dynamics in the diffusion denoising process: early stages primarily rely on text prompt guidance for sampling, while later stages progressively depend on visual features for denoising. Sharing LoRA parameters uniformly across all timesteps is therefore suboptimal.

ShortFT's approach:

- The denoising chain is divided into $k$ segments.
- Each segment except the first is assigned an independent LoRA at its final timestep.
- The first segment retains shared LoRA across all timesteps (consistent with DRaFT).
- Consecutive timestep intervals without LoRA in subsequent segments are skipped via denoising shortcuts.
- An incremental stacking strategy is adopted: LoRA $i$ adds a new LoRA branch on top of LoRA $i-1$.
- LoRA rank is set to 128, applied to feed-forward and attention layers of the UNet.

**Key advantage**: No additional computational cost is introduced at inference (only the corresponding LoRA is activated at the corresponding timestep), while model capacity is effectively increased during training, accelerating convergence.

### Key Design 3: Progressive Training Strategy

Shortcuts introduced by few-step models inevitably carry approximation errors (particularly in fine-grained details relative to the original model), and direct use leads to training-inference discrepancy.

The solution is a $k$-stage progressive training procedure:

- **Stage $i$**: Optimize the weights of LoRA $i$ through LoRA $k$.
- Denoising in segment $i$ and earlier uses the original denoising chain.
- Denoising shortcuts are introduced from segment $i$ onward.
- Truncated backpropagation is also applied concurrently.

**At inference**: No shortcuts are used; the original denoising chain is restored for final image generation.

### Loss & Training

The training objective is to maximize the reward function via gradient ascent, with no additional reconstruction loss. Reward functions employed in experiments include:

- **HPS v2**: Measures human preference for generated images.
- **PickScore**: A preference model based on user selections.
- **Symmetry**: Encourages horizontal symmetry in generated images.
- **Combined reward**: PickScore × 10 + HPS v2 × 2 + Aesthetic × 0.05.
- **Compressibility** and other reward functions.

For regularization, rather than using CLIPScore as in DRTune, a joint regularization term is constructed by mixing HPS v2 and PickScore at a 1:10 ratio.

## Key Experimental Results

### Main Results (Table 1: Objective Evaluation under Equal Compute Budget)

| Method | HPS v2 ↑ | PickScore ↑ | Symmetry ↓ |
|--------|----------|-------------|------------|
| SD 1.5 | 26.91 | 20.46 | 0.853 |
| DRaFT-LV | 33.13 | 23.35 | 0.418 |
| DRTune | 32.79 | 23.22 | 0.207 |
| **ShortFT** | **33.88** | **24.16** | **0.138** |

All methods are trained for 6 hours on 2×A800 GPUs. ShortFT achieves the best performance across all three metrics.

### Results at 10k Training Steps

Full training (10k steps with HPS v2 reward) achieves HPS v2 = **35.97** on HPDv2, surpassing the reported score of DRaFT-LV.

### Fine-tuning SD vs. Hyper-SD (Table 2)

| Method | HPS v2 ↑ |
|--------|----------|
| Fine-tune Hyper-SD | 32.92 |
| Fine-tune SD 1.5 (ShortFT) | **35.97** |

This validates that fine-tuning the base model is superior to fine-tuning the distilled few-step model — performance degradation and capacity loss introduced by distillation render direct fine-tuning of the few-step model suboptimal.

### Ablation Study (Table 3)

| Configuration | HPS v2 ↑ | PickScore ↑ | Symmetry ↓ |
|---------------|----------|-------------|------------|
| w/o T-LoRA | 33.46 | 23.82 | 0.187 |
| w/o P-Training | 33.27 | 23.97 | 0.146 |
| **ShortFT** | **33.88** | **24.16** | **0.138** |

- Removing timestep-aware LoRA: HPS v2 drops by 0.42, PickScore drops by 0.34.
- Removing progressive training: HPS v2 drops by 0.61, and generated images exhibit locally incoherent details (e.g., unsmooth hair).

### User Study

Eleven volunteers (5 image processing experts + 6 non-experts) supplemented by GPT-4V evaluation; ShortFT receives majority preference votes over both DRaFT-LV and DRTune in pairwise comparisons.

### Key Findings

1. ShortFT does not require gradient checkpointing, resulting in superior memory efficiency.
2. Owing to the shortened denoising chain, the learning speed is faster than DRTune (the current most efficient method).
3. The method is architecture-agnostic: it is effective on both UNet-based (SD 1.5) and Transformer-based (SD 3) architectures.
4. Good generalizability: models fine-tuned on HPDv2 can effectively handle complex wild prompts from Sora.
5. Applicable to a wide variety of reward functions: HPS v2, PickScore, Symmetry, Compressibility, and Combined reward.

## Highlights & Insights

1. **Novel perspective**: Rather than making incremental modifications to gradient truncation strategies, the paper addresses the root cause — an overly long denoising chain — by leveraging few-step distilled models as shortcuts, which is a clean and effective idea.
2. **Complete gradient propagation**: Unlike the partial gradient truncation in AlignProp and DRTune, ShortFT achieves genuine end-to-end backpropagation, eliminating gradient bias.
3. **Elegant timestep-aware LoRA design**: Exploiting the semantic differences across denoising stages, independent LoRA modules are allocated to increase model capacity without adding inference cost.
4. **Training-inference consistency**: The progressive training strategy systematically addresses the approximation error introduced by shortcuts rather than ignoring it.
5. **Strong practical utility**: No gradient checkpointing required, memory-friendly, architecture-agnostic, and compatible with general reward functions.

## Limitations & Future Work

1. **Dependency on trajectory-preserving distilled models**: The method presupposes the availability of high-quality trajectory-preserving few-step models (e.g., Hyper-SD); it cannot be directly applied when no such distilled counterpart exists for a given base model.
2. **Validation limited to medium-scale models**: Main experiments are conducted on SD 1.5; SD 3 is presented only with qualitative results, and systematic validation on SDXL or larger models is lacking.
3. **Approximation error from shortcuts**: Although mitigated by progressive training, the deviation between shortcut outputs and the original model persists at the fine-grained level and may affect certain detail-sensitive tasks.
4. **Limitations of the reward function itself**: The upper bound of the method's effectiveness is constrained by the quality of the reward model; biased reward models will lead to misaligned optimization directions.
5. **Absence of direct comparison with DPO-based methods**: Comparisons are made only against backpropagation-based methods, without quantitative evaluation against reward-model-free approaches such as Diffusion-DPO.

## Related Work & Insights

- **DRaFT** (Clark et al., ICLR 2024): Truncates backpropagation to the latter portion of the denoising chain. ShortFT avoids information loss by using shortcuts instead.
- **DRTune** (Wu et al., ECCV 2024): Combines partial gradient truncation with gradient checkpointing to propagate signals to early steps, but at the cost of gradient bias. ShortFT requires no gradient checkpointing and achieves complete gradients.
- **AlignProp** (Prabhudesai et al., 2023): Also truncates partial gradients; ShortFT provides a more fundamental solution.
- **Hyper-SD** (Ren et al., 2024): A trajectory-preserving distillation method that serves as a critical infrastructure component for ShortFT.
- **eDiff-I** (Balaji et al., 2022): Reveals temporal dynamics in the denoising process, inspiring the timestep-aware LoRA design.
- **Broader inspiration**: The paradigm of "leveraging distilled models to simplify the training pipeline" can be transferred to video diffusion model alignment, 3D generation alignment, and related settings.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | ⭐⭐⭐⭐ | The perspective of using distilled models as training shortcuts is novel and practical |
| Technical Quality | ⭐⭐⭐⭐ | The three components are complementarily designed with thorough ablation |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Multi-reward, multi-architecture validation with user study |
| Writing Quality | ⭐⭐⭐⭐⭐ | Excellent figures; motivation and method are presented with exceptional clarity |
| Value | ⭐⭐⭐⭐ | No gradient checkpointing required; clear efficiency advantage |
| **Overall** | **⭐⭐⭐⭐** | Solid work that addresses a core efficiency bottleneck in diffusion model alignment |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation](taxadiffusion_progressively_trained_diffusion_model_for_fine-grained_species_gen.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](../../AAAI2026/image_generation/self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](../../NeurIPS2025/image_generation/towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)

<!-- RELATED:END -->
