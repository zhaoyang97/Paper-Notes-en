---
title: >-
  [Paper Note] Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model
description: >-
  [AAAI 2026][Image Generation][Latent Diffusion Model] This paper proposes SOLD, a framework that integrates a latent diffusion model (LDM) with reinforcement learning (RL) via a step-wise single-step sampling optimizatio…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Latent Diffusion Model"
  - "RNA Inverse Folding"
  - "Reinforcement Learning"
  - "PPO"
  - "Step-wise Optimization"
  - "RNA-FM"
date: 2026-05-08
content_hash: 630974c313730861
---

# Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model

**Conference**: AAAI 2026
**arXiv**: [2601.19232](https://arxiv.org/abs/2601.19232)
**Code**: [Available](https://github.com/darkflash03/SOLD)
**Area**: Diffusion Models / RNA Inverse Folding / Reinforcement Learning
**Keywords**: Latent Diffusion Model, RNA Inverse Folding, Reinforcement Learning, PPO, Step-wise Optimization, RNA-FM

## TL;DR

This paper proposes SOLD, a framework that integrates a latent diffusion model (LDM) with reinforcement learning (RL) via a step-wise single-step sampling optimization strategy to directly optimize non-differentiable structural metrics in RNA inverse folding — including secondary structure similarity (SS), minimum free energy (MFE), and LDDT — achieving comprehensive improvements over existing methods across multiple metrics.

## Background & Motivation

RNA inverse folding aims to design RNA sequences that fold into specified 3D structures, with important applications in RNA therapeutics, gene regulation, and synthetic biology. Existing methods suffer from the following limitations:

**Physics-based methods (e.g., Rosetta)**: Rely on Monte Carlo optimization, incurring high computational costs and difficulty handling polymorphic conformations.

**2D structure-based methods (e.g., ViennaRNA)**: Neglect critical 3D geometric information.

**Deep learning methods (e.g., RhoDesign, RDesign)**: VAE-based approaches struggle to model complex distributions and long-range dependencies.

**Existing diffusion methods (e.g., RiboDiffusion)**: Perform SDE-based diffusion directly in sequence space, failing to leverage RNA co-evolutionary information.

**Core limitation**: Existing methods optimize only sequence recovery rate and cannot directly optimize non-differentiable structural objectives such as SS, MFE, and LDDT.

## Method

### Overall Architecture

SOLD operates in two stages: (1) pre-training an LDM to learn sequence generation capabilities; and (2) fine-tuning the LDM with RL to directly optimize structural metrics. The overall architecture consists of an MLP encoder, a GVP-GNN + DiT denoising network, and an MLP decoder.

### Key Design 1: RNA-FM-based Latent Diffusion Model

Unlike RiboDiffusion, which performs diffusion directly in the one-hot sequence space $(L, 4)$, SOLD leverages a pre-trained RNA-FM model to extract embeddings $(L, 640)$, which are then compressed to a latent space $(L, 32)$ via an MLP encoder before diffusion. RNA-FM embeddings encode co-evolutionary patterns and structural information of RNA, substantially improving sequence recovery. Ablation experiments show that $D=32$ achieves the optimal trade-off between generation quality and efficiency (performance peaks at $D=32$ and degrades for $D>32$).

The denoising network comprises 4 GVP-GNN layers (processing backbone geometric features) and 8 DiT Transformer layers (capturing sequence dependencies), and predicts the clean latent embedding $\hat{z}_0$.

### Key Design 2: Step-wise RL Optimization

The core innovation of SOLD is that it eliminates the need to sample complete denoising trajectories as required by DDPO/DPOK. Instead:
- A timestep $t$ is sampled randomly; a single DDIM step predicts $z_0'$ directly from noisy $z_t$.
- $z_0'$ is decoded into sequence $s_0'$ to compute a **long-term reward** $r_0(t) = R_i(s_0')$.
- Simultaneously, a single DDPM step produces $z_{t-1}'$, which is decoded to compute a **short-term reward** $r_t(t) = R_i(s_{t-1}')$.
- A piecewise reward function combines both: large $t$ (early denoising) uses the short-term reward; small $t$ (late denoising) uses the long-term reward.

This design yields substantial speedups, reducing per-epoch training time to only 4% of that required by DDPO.

### Key Design 3: Piecewise Reward Strategy

$$r_{\text{total}}(t) = w(t) \cdot r_t(t) + u(t) \cdot r_0(t)$$

where threshold $\tau$ controls the switching point. Ablation experiments show that different objectives favor different $\tau$ values: MFE and LDDT perform best at $\tau=90$ (long-term reward dominates for 90 steps), while SS performs best at $\tau=60$. The intuition behind this piecewise strategy is that early denoising steps involve high noise levels, making direct prediction of $z_0'$ unreliable, so short-term rewards are more informative; at later steps with low noise, long-term rewards are more effective.

### Loss & Training

During LDM pre-training, a joint MSE and cross-entropy loss is used:

$$\mathcal{L} = \mathbb{E}_{z_0,t}[\|z_0 - \hat{z}_0\|^2] - \mathbb{E}_s[\sum_{i=1}^L \log p(s_i | \text{Dec}(\hat{z}_0)_i)]$$

During RL fine-tuning, PPO with KL regularization is employed, with clip range $\epsilon=0.0001$ and KL weight $\lambda_{\text{ref}}$ controlling deviation from the reference policy.

## Key Experimental Results

### Main Results: Multi-objective Optimization Performance (Table 4)

| Method | Seq Recovery↑ | MFE↓ | SS↑ | RMSD↓ | LDDT↑ |
|--------|-------------|------|-----|-------|-------|
| RhoDesign | 0.2734 | -11.92 | 0.650 | 16.36 | 0.503 |
| RDesign | 0.4457 | -10.70 | 0.614 | 16.13 | 0.524 |
| gRNAde | 0.5108 | -10.54 | 0.562 | 18.00 | 0.485 |
| RiboDiffusion | 0.5125 | -15.21 | 0.763 | 12.32 | 0.610 |
| DRAKES | 0.4400 | -14.24 | 0.769 | 11.91 | 0.619 |
| LDM (baseline) | 0.5728 | -13.33 | 0.727 | 12.57 | 0.618 |
| **SOLD** | **0.5732** | **-16.86** | **0.760** | **11.86** | **0.636** |

While maintaining sequence recovery rate, SOLD reduces MFE by 26.5%, improves LDDT by 2.9%, and reduces RMSD by 5.7%.

### Ablation Study: Training Efficiency Comparison (Table 3)

| Method | MFE Training Time (s) | SS Training Time (s) | LDDT Training Time (s) |
|--------|-----------------------|----------------------|------------------------|
| DDPO | 5953 | 6190 | 14000 |
| DPOK | 7677 | 7330 | 14200 |
| **SOLD** | **256** | **263** | **6900** |

SOLD is **23×** faster than DDPO and **30×** faster than DPOK on MFE and SS optimization.

### Key Findings

1. **Advantage of RNA-FM embeddings**: Performing diffusion in latent space improves sequence recovery from 0.5125 to 0.5728 (+11.8%) compared to RiboDiffusion's sequence-space diffusion.
2. **Efficiency of single-step sampling**: Step-wise optimization avoids full trajectory sampling, achieving over 20× training speedup.
3. **Necessity of piecewise rewards**: Pure long-term or pure short-term rewards are both inferior to the hybrid strategy (MFE: -19.74 vs. -17.24/-17.73).
4. **Case study validation**: On TPP riboswitch design (PDB: 3D2V), only SOLD successfully designed sequences that fold into the target structure; all other methods failed.

## Highlights & Insights

1. **First application of RL to latent diffusion models for RNA inverse folding**: Addresses the gap in using diffusion models for RNA design when structural objectives are non-differentiable.
2. **Highly innovative step-wise optimization strategy**: Independently optimizes each denoising step rather than sampling complete trajectories; theoretically equivalent to the DDPO objective but over 20× more efficient.
3. **Simultaneous multi-objective optimization**: Directly invokes ViennaRNA for evaluation without requiring a differentiable reward model, avoiding the additional reward model training required by DRAKES.
4. **Rigorous experimental design**: Comprehensive ablation studies (latent dimensionality, piecewise reward strategy, varying sequence lengths) and statistical significance tests.

## Limitations & Future Work

1. **Data scarcity**: The limited availability of high-quality RNA 3D structure data (only 7,067 training samples) constrains model generalization.
2. **Multi-objective synergy underexplored**: The interactions among 1D/2D/3D metrics and optimal weighting schemes remain unclear.
3. **Approximation errors in reward tools**: Prediction errors inherent in ViennaRNA and RhoFold may compromise optimization accuracy.
4. **Performance degradation on long sequences**: All metrics degrade significantly for long sequences (128–512 nt), indicating that complex RNA folding remains a challenge.
5. **Inference still requires full trajectories**: Although single-step sampling accelerates training, inference still requires 100 complete denoising steps.

## Related Work & Insights

| Method | Diffusion Space | RL Fine-tuning | Reward Model | Optimization Target |
|--------|----------------|----------------|--------------|---------------------|
| RiboDiffusion | Sequence space (SDE) | ✗ | — | Sequence recovery |
| DRAKES | Discrete diffusion | ✓ (single objective) | Requires training | MFE |
| RNAdiffusion | Latent space | ✗ | Requires training | Translation efficiency |
| DDPO/DPOK | Continuous space | ✓ (full trajectory) | None | Image quality |
| **SOLD** | **Latent space (LDM)** | **✓ (step-wise single-step)** | **None (direct evaluation)** | **SS + MFE + LDDT** |

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Strong originality in combining step-wise RL optimization with latent diffusion)
- **Technical Contribution**: ⭐⭐⭐⭐⭐ (Complete framework design with theoretical derivation and thorough ablation)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Dual benchmarks on SOLD TEST + CASP15, multiple metrics, statistical testing)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and complete mathematical derivations)
- **Practical Impact**: ⭐⭐⭐⭐ (Promising application potential in RNA drug design)
- **Overall Recommendation**: ⭐⭐⭐⭐ (4.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](../../ICLR2026/image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[AAAI 2026\] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](../../ICLR2026/image_generation/latent_diffusion_model_without_variational_autoencoder.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)

</div>

<!-- RELATED:END -->
