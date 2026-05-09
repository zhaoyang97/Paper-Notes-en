---
title: >-
  [Paper Note] Heterogeneous Decentralized Diffusion Models
description: >-
  [CVPR2026][Image Generation][Decentralized diffusion models] This paper proposes a heterogeneous decentralized diffusion framework that allows different experts to train completely independently using distinct diffusion objectives (DDPM ε-prediction and Flow Matching velocity-prediction). At inference time, a deterministic schedule-aware conversion unifies all expert outputs into velocity space for fusion. Compared to homogeneous baselines, the framework simultaneously improves FID and generation diversity while reducing computation by 16×.
tags:
  - CVPR2026
  - Image Generation
  - Decentralized diffusion models
  - heterogeneous training objectives
  - DDPM
  - Flow Matching
  - mixture of experts
  - DiT
  - PixArt-α
date: 2026-05-08
content_hash: 2f098f0246adb4e3
---

# Heterogeneous Decentralized Diffusion Models

**Conference**: CVPR2026
**arXiv**: [2603.06741](https://arxiv.org/abs/2603.06741)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Decentralized diffusion models, heterogeneous training objectives, DDPM, Flow Matching, mixture of experts, DiT, PixArt-α

## TL;DR

This paper proposes a heterogeneous decentralized diffusion framework that allows different experts to train completely independently using distinct diffusion objectives (DDPM ε-prediction and Flow Matching velocity-prediction). At inference time, a deterministic schedule-aware conversion unifies all expert outputs into velocity space for fusion. Compared to homogeneous baselines, the framework simultaneously improves FID and generation diversity while reducing computation by 16×.

## Background & Motivation

1. **High computational barrier**: Training frontier diffusion models requires large-scale tightly-coupled clusters (e.g., hundreds of GPU-days), restricting participation to well-resourced institutions.
2. **Limitations of prior decentralized approaches**: DDM (McAllister et al.) demonstrated the feasibility of independently training experts and combining them, but requires all experts to share a homogeneous training objective and demands 1,176 GPU-days and 158M images.
3. **Inherent heterogeneity in real decentralized settings**: Different contributors possess different resources, preferences, and technical constraints, making enforced uniformity of training objectives impractical.
4. **Complementary properties of different objectives**: ε-prediction implicitly assigns stronger weighting at low-noise timesteps (favoring detail preservation), while velocity-prediction assigns stronger weighting at high-noise timesteps (favoring global structure), making the two naturally complementary.
5. **Underutilization of pretrained weights**: A large number of existing DDPM pretrained checkpoints cannot be directly reused in Flow Matching training pipelines.
6. **Architectural redundancy**: The per-layer AdaLN in standard DiT introduces substantial parameters; PixArt-α's AdaLN-Single can reduce parameters by 30% while maintaining generation quality.

## Method

### Overall Architecture

The dataset is partitioned into $K=8$ semantic clusters (e.g., portraits, landscapes, architecture) via DINOv2 feature extraction followed by hierarchical k-means clustering. Each expert is trained entirely independently on its assigned cluster, with no gradient, parameter, or activation synchronization. At inference time, a router network $p_\phi(k|x_t,t)$ dynamically selects and fuses expert predictions.

### Heterogeneous Objective Design

- **DDPM experts** (2 experts): predict noise $\epsilon$ using a cosine noise schedule, with loss $\mathcal{L}_{\text{DDPM}}^{(k)} = \mathbb{E}[\|\epsilon_{\theta_k}(\alpha_t x_0 + \sigma_t \epsilon, t) - \epsilon\|^2]$
- **Flow Matching experts** (6 experts): predict velocity field $v$ using linear interpolation $x_t = (1-t)x_0 + t\epsilon$, with loss $\mathcal{L}_{\text{FM}}^{(k)} = \mathbb{E}[\|v_{\theta_k}(x_t, t) - (\epsilon - x_0)\|^2]$
- **Objective assignment strategy**: DDPM is assigned to clusters 0 and 3, which contain high-fidelity subjects (e.g., cars, flowers).

### Deterministic Conversion at Inference Time

DDPM expert output $\epsilon_\theta$ must be converted to velocity $v$ to be unified with FM experts:

1. Estimate the clean sample from $\epsilon_\theta$: $\hat{x}_0 = (x_t - \sigma_t \epsilon_\theta) / \alpha_t$
2. Under the linear interpolation schedule ($\alpha_t=1-t,\ \sigma_t=t$), the conversion simplifies to: $v(x_t,t) = \epsilon_\theta(x_t,t) - \hat{x}_0$
3. Numerical stability measures: $\hat{x}_0$ is clamped to $[-20, 20]$, $\alpha_{\text{safe}} = \max(\alpha_t, 0.01)$, and adaptive velocity scaling is applied at high noise levels $t>0.85$.

This conversion is a purely algebraic operation and **requires no retraining**.

### Implicit Timestep Weighting Complementarity (Theoretical Analysis)

Both objectives are expressed as weighted forms of the clean sample estimation error:

- ε-prediction weight: $w_\epsilon(t) = \alpha_t^2 / \sigma_t^2$
- v-prediction weight: $w_v(t) = 1 / \sigma_t^2$
- Ratio $w_v / w_\epsilon = 1/\alpha_t^2 \geq 1$, diverging to infinity at high-noise timesteps.

This implies that velocity-prediction receives stronger gradients at high-noise timesteps (attending to global structure), while ε-prediction is relatively stronger at low-noise timesteps (attending to local detail), forming a natural complementarity.

### Efficient Architecture and Checkpoint Conversion

- **AdaLN-Single**: A global MLP computes modulation parameters $\mathbf{c} \in \mathbb{R}^{6Ld}$ for all layers in a single pass, combined with per-block learnable embeddings $\mathbf{E}_b$, reducing parameters from 891M to 605M (DiT-XL/2).
- **Pretrained checkpoint conversion**: Starting from ImageNet DDPM DiT weights, patch embeddings, positional embeddings, and transformer blocks are retained; the final layer and text projection are reinitialized. At runtime, the FM continuous timestep $t \in [0,1]$ is mapped to $t_{\text{DiT}} = \text{round}(999t)$, yielding a 1.2× convergence speedup.

### Router

- **Architecture**: DiT-B/2 (129M parameters), 12-layer Transformer.
- **Input**: Noisy latent $x_t$ and timestep $t$ (no text conditioning).
- **Training**: Full dataset with ground-truth cluster labels, cross-entropy loss, 25 epochs.
- **Inference modes**: Top-1 / Top-K / Full Ensemble.

## Key Experimental Results

### Decentralized vs. Monolithic Training (DiT-B/2, LAION-Art 3.9M)

| Inference Strategy | FID-50K ↓ |
|-------------------|-----------|
| Monolithic model | 29.64 |
| Top-1 | 30.60 |
| **Top-2** | **22.60** |
| Full Ensemble | 47.89 |

Top-2 expert selection improves FID by 23.7% over the monolithic model; Full Ensemble degrades performance.

### Resource Efficiency Comparison (DiT-XL/2)

| Method | Data | Compute | FID-50K ↓ |
|--------|------|---------|-----------|
| DDM (prior work) | 158M | 1176 A100-days | 5.5–10.5 |
| Ours homogeneous (8FM) | 11M | 72 A100-days | 12.45 |
| **Ours heterogeneous (2DDPM:6FM)** | **11M** | **72 A100-days** | **11.88** |

Computation reduced by **16×**, data by **14×**.

### Homogeneous vs. Heterogeneous (Aligned inference: CFG=7.5, 50 steps)

| Model | FID-50K ↓ | Intra-prompt LPIPS ↑ |
|-------|-----------|----------------------|
| Homogeneous 8FM | 12.45 | 0.617 (±0.074) |
| **Heterogeneous 2DDPM:6FM** | **11.88** | **0.631 (±0.078)** |

The heterogeneous approach improves both quality (FID) and diversity (LPIPS) over the homogeneous baseline.

### Ablation Study

#### DDPM→FM Conversion and Mixed Sampling

| Sampling Strategy | LPIPS ↑ | FID ↓ | CLIP ↑ |
|------------------|---------|-------|--------|
| Native DDPM | 0.787 | 27.04 | 0.316 |
| Native FM | 0.752 | 20.23 | 0.324 |
| DDPM→FM conversion | 0.761 | 25.61 | 0.319 |
| Mixed (same schedule) | 0.782 | 32.67 | 0.312 |

The DDPM→FM conversion effectively improves DDPM quality (FID 27.04→25.61) without retraining; mixed sampling substantially increases diversity at the cost of FID.

#### Routing Threshold

A threshold of 0.2 achieves optimal FID (38.28), while a threshold of 0.5 yields the highest diversity (LPIPS), revealing a quality–diversity trade-off.

## Highlights & Insights

1. **Truly heterogeneous decentralization**: This is the first framework to support independent training of different experts under distinct diffusion objectives, breaking the homogeneity constraint of prior DDM approaches.
2. **Elegant inference-time unification**: A schedule-aware algebraic conversion deterministically maps ε-prediction to velocity space without any retraining.
3. **Solid theoretical foundation**: Proposition 1 rigorously proves the complementarity of ε/v-prediction in timestep weighting, providing theoretical grounding for the heterogeneous design.
4. **Drastically reduced resource requirements**: 16× computation compression and 14× data compression; a single expert requires only 20–48 GB VRAM.
5. **Simultaneous improvement in quality and diversity**: The heterogeneous approach outperforms the homogeneous baseline on both FID and LPIPS.

## Limitations & Future Work

1. **Objective ratio insufficiently explored**: Only a limited number of DDPM:FM ratios (e.g., 2:6) are evaluated; optimal allocation depends on data distribution and downstream requirements.
2. **Manual tuning required for numerical stability**: Clamping, safe denominators, and adaptive scaling at high-noise timesteps are all manually designed heuristics.
3. **Restricted to two objective families**: Other parameterizations such as $x_0$-prediction and consistency objectives are not considered.
4. **Router does not support dynamic expert addition/removal**: Adding or removing experts requires retraining the router.
5. **Resolution limitation**: Experiments are conducted only at 256×256; high-resolution settings remain unvalidated.
6. **Absolute FID not directly comparable to prior work**: DDM achieves 5.5–10.5 FID under training scales more than 10× larger.

## Related Work & Insights

| Method | Key Difference |
|--------|---------------|
| DDM (McAllister 2025) | Requires homogeneous objectives + 1,176 GPU-days; this work supports heterogeneous objectives + 72 GPU-days. |
| Diff2Flow (Schusterbauer 2025) | Single-model DDPM→FM fine-tuning conversion; this work performs training-free multi-expert inference-time conversion. |
| PixArt-α (Chen 2024) | Proposes AdaLN-Single for efficient single-model training; this work applies it to decentralized multi-expert settings. |
| DiT (Peebles 2023) | Foundational Transformer diffusion architecture; this work extends it with heterogeneous objectives and checkpoint conversion. |
| DistriFusion (Li 2024) | Distributed parallel inference via patch parallelism; this work focuses on decentralized training. |
| VDM (Kingma 2021) | Unified variational framework analyzing implicit weighting of different prediction objectives; this work leverages its theory to support heterogeneous complementarity. |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Heterogeneous-objective decentralized diffusion training is a novel direction; the inference-time algebraic conversion is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes multi-scale model comparisons, ablation analyses, routing threshold analysis, and extensive qualitative results; lacks exploration of high-resolution settings and a broader range of objective ratios.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with complete theoretical derivations and consistent notation.
- **Value**: ⭐⭐⭐⭐ — Substantially lowers the barrier to decentralized diffusion training, providing a viable path toward community-driven model development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](../../ICCV2025/image_generation/feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[NeurIPS 2025\] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism](../../NeurIPS2025/image_generation/pragmatic_heterogeneous_collaborative_perception_via_generative_communication_me.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)

</div>

<!-- RELATED:END -->
