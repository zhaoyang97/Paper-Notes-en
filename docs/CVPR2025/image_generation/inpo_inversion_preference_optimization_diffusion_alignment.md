---
title: >-
  [Paper Note] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment
description: >-
  [CVPR 2025][Image Generation][Diffusion Models] This paper proposes DDIM-InPO, which treats the diffusion model as a single-step generative model and utilizes DDIM inversion to identify latent variables highly correlated with preference data, achieving state-of-the-art (SOTA) efficient diffusion preference alignment in only 400 fine-tuning steps.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Models"
  - "DPO"
  - "DDIM Inversion"
  - "Preference Optimization"
  - "Human Preference Alignment"
date: 2026-05-08
content_hash: cb68029b2479ab5f
---

# InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment

**Conference**: CVPR 2025  
**arXiv**: [2503.18454](https://arxiv.org/abs/2503.18454)  
**Code**: [GitHub](https://github.com/JaydenLyh/InPO)  
**Area**: Image Generation / Diffusion Model Alignment  
**Keywords**: Diffusion Models, DPO, DDIM Inversion, Preference Optimization, Human Preference Alignment

## TL;DR

This paper proposes DDIM-InPO, which treats the diffusion model as a single-step generative model and utilizes DDIM inversion to identify latent variables highly correlated with preference data, achieving state-of-the-art (SOTA) efficient diffusion preference alignment in only 400 fine-tuning steps.

## Background & Motivation

**Background**: Text-to-image diffusion models (e.g., SDXL) have possessed powerful generation capabilities, but aligning the generation results with human preferences remains an important challenge. DPO methods in the LLM field have been successfully applied to preference alignment, but diffusion models are difficult to apply directly due to the long Markov chains and the intractable reverse processes.

**Limitations of Prior Work**: (1) Reward-model-based methods (DRaFT, AlignProp) require backpropagation through the entire sampling process, resulting in high memory overhead and susceptibility to reward hacking. (2) RL methods (DDPO, DPOK) are limited by the length of the Markov chain, leading to low efficiency. (3) Direct preference optimization methods like Diffusion-DPO distribute rewards across all denoising steps, causing sparse reward issues, especially performing poorly on out-of-distribution inputs.

**Key Challenge**: Allocating rewards across the entire denoising chain in current methods leads to sparse training signals, whereas in reality, only a few latent variables are highly correlated with the final image quality.

**Goal**: Fine-tune only the outputs of a few latent variables that are highly correlated with the target image to achieve fast and high-quality preference optimization.

**Key Insight**: Re-parametrize the diffusion model into a single-step generation framework—at any timestep $t$, the model can estimate $x_0$ directly from $x_t$ in one step, allowing direct allocation of implicit rewards based on this.

**Core Idea**: Establish a one-step mapping between latent variables and the target image through DDIM re-parametrization, use DDIM inversion to find the latent variables highly correlated with preference data in the target image space, and optimize only the outputs of these latent variables.

## Method

### Overall Architecture

Given a preference dataset $\{(x_0^w, x_0^l, c)\}$ (winning/losing image pairs and corresponding prompts), the goal is to train the model $p_\theta$ to align with human preferences. The method consists of three steps: (1) Establish a single-step mapping from latent variables at any timestep to the $x_0$ space via DDIM re-parametrization; (2) Estimate highly correlated latent variables from preference images using inversion techniques; (3) Optimize only the outputs corresponding to these latent variables.

### Key Designs

1. **DDIM Re-parametrized DPO Reward Allocation**:

    - **Function**: Conceptualizes the diffusion model as a timestep-aware single-step generator, directly allocating implicit rewards at any timestep.
    - **Mechanism**: Utilizes DDIM re-parametrization $x_0(t) = \bar{x}_t - \sigma_t \epsilon_\theta^t(x_t, c)$ to associate $x_t$ with the $x_0$ space. Define a joint reward $r_t^c(x_0, x_t)$ satisfying $r(x_0, c) = \mathbb{E}_{p_\theta^c(x_t|x_0)}[r_t^c(x_0, x_t)]$. By minimizing the joint KL divergence $D_{KL}[p_\theta^c(x_0, x_t) \| p_{ref}^c(x_0, x_t)]$ as an upper bound of the standard KL, a DPO objective applicable to any single step is derived.
    - **Design Motivation**: Compared to Diffusion-DPO, which allocates rewards along the entire $x_{0:T}$ trajectory, the single-step mapping can precisely focus rewards on the variables most correlated with the target image.

2. **Latent Variable Selection via DDIM Inversion**:

    - **Function**: Identifies latent variables highly correlated with preference data and optimizes only these variables.
    - **Mechanism**: Given preference images $x_0^w$ and $x_0^l$, DDIM inversion is used to map back from the $x_0$ space to $x_t$ at each timestep. These inverted $x_t$ are highly correlated with the original preference images (since the deterministic DDIM process preserves image structure). The DPO loss is computed only on these specific $x_t$.
    - **Design Motivation**: Avoids optimizing on randomly sampled $x_t$ (which leads to sparse rewards), instead precisely targeting the latent variables that "have the greatest impact on generation quality".

3. **Efficient Single-Step Optimization Objective**:

    - **Function**: Achieves preference alignment with extremely low computational cost.
    - **Mechanism**: The final loss is simplified to: for a randomly sampled timestep $t$, obtain $x_t^w, x_t^l$ through inversion with the reference model, and compute the log-probability differences of the single-step predictions $x_0(t)$ of the current model and reference model at those $x_t$. Each training step requires only one forward pass and one inversion.
    - **Design Motivation**: Compared to methods requiring multi-step denoising, single-step optimization dramatically reduces memory and computational overhead, allowing fine-tuning to be completed in just 400 steps.

### Loss & Training

The core loss is based on re-parametrized DPO: for a randomly sampled timestep $t$, obtain $x_t^w, x_t^l$ via DDIM inversion, and compute $\log\sigma(\beta[\log p_\theta^c(x_0^w, x_t^w) / p_{ref}^c(x_0^w, x_t^w) - \log p_\theta^c(x_0^l, x_t^l) / p_{ref}^c(x_0^l, x_t^l)])$. Fine-tuning SDXL-base-1.0 requires only 400 steps.

## Key Experimental Results

### Main Results

| Method | PickScore ↑ | HPSv2 ↑ | ImageReward ↑ | GenEval ↑ | Training Steps |
|------|-----------|---------|-------------|----------|---------|
| SDXL (baseline) | 22.00 | 28.48 | 0.88 | 0.55 | - |
| Diffusion-DPO | 22.10 | 28.89 | 1.01 | 0.58 | 2000 |
| D3PO | 22.05 | 28.71 | 0.95 | 0.56 | 2000 |
| DenseReward | 22.12 | 28.93 | 1.05 | 0.59 | 2000 |
| **DDIM-InPO** | **22.25** | **29.12** | **1.18** | **0.62** | **400** |

### Ablation Study

| Configuration | PickScore | HPSv2 | Description |
|------|---------|-------|------|
| Full InPO (400 steps) | 22.25 | 29.12 | Full model |
| w/o Inversion (random $x_t$) | 22.08 | 28.82 | Removing inversion severely degrades performance |
| Full-chain DPO | 22.10 | 28.89 | Degenerates to Diffusion-DPO |
| InPO (200 steps) | 22.18 | 29.01 | Significant improvement already at 200 steps |

### Key Findings

- DDIM inversion is the key to performance improvement—randomly sampling $x_t$ without inversion degrades to the level of standard Diffusion-DPO.
- Fine-tuning for 400 steps already outperforms 2000 steps of Diffusion-DPO, achieving a training efficiency speedup of approximately 5x.
- Generated images show significant improvements in both visual aesthetics and prompt consistency.
- The method is relatively robust to the $\beta$ hyperparameter.

## Highlights & Insights

- **Conceptual Breakthrough: Diffusion Models as Single-step Generators**: Through DDIM re-parametrization, the complex multi-step denoising process is simplified into a single-step mapping problem, allowing DPO to be directly and efficiently applied. This perspective is highly elegant.
- **Inversion as Data Augmentation**: DDIM inversion essentially finds the most relevant latent space representation for preference data, which is equivalent to an "alignment-aware" data augmentation strategy.
- **Efficiency Revolution**: Fine-tuning SDXL in only 400 steps outperforms the state-of-the-art, which is highly valuable for practical deployment. It can be transferred to other scenarios requiring rapid adaptation (e.g., style customization).

## Limitations & Future Work

- Relies on the deterministic sampling assumption of DDIM; applicability to stochastic samplers remains to be verified.
- The inversion process introduces extra computational overhead (although it is one-off).
- Only validated on SDXL; applicability to other architectures (such as DiT, Flux) remains unknown.
- Online versions combining online preference data generation could be explored.

## Related Work & Insights

- **vs Diffusion-DPO**: Uniformly allocating rewards across the entire denoising chain leads to sparse signals; InPO precisely locates critical latent variables through inversion, speeding up training by 5x.
- **vs DRaFT/AlignProp**: Backpropagating gradients through the entire sampling process requires high memory; InPO only requires one-step forward propagation.
- **vs DDPO**: RL policy gradient methods require online generation of a large number of samples; InPO can be directly trained on offline preference data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Conceptualizing the diffusion model as a single-step generator and combining it with inversion is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple metrics, but lacks user studies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivation and clear logic.
- Value: ⭐⭐⭐⭐⭐ The massive boost in training efficiency is highly valuable for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[CVPR 2025\] ReNeg: Learning Negative Embedding with Reward Guidance](reneg_learning_negative_embedding_with_reward_guidance.md)

</div>

<!-- RELATED:END -->
