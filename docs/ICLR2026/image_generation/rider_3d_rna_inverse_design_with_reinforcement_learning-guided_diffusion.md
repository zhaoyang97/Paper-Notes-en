---
title: >-
  [Paper Note] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion
description: >-
  [ICLR 2026][Image Generation][RNA inverse design] This paper proposes RIDER, the first framework to incorporate reinforcement learning into 3D RNA inverse design. It first pretrains a conditional diffusion model (RIDE) t…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "RNA inverse design"
  - "3D structural similarity"
  - "diffusion model"
  - "RL fine-tuning"
  - "DDPO"
date: 2026-05-08
content_hash: d4d7f2dfbd221055
---

# RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion

**Conference**: ICLR 2026
**arXiv**: [2602.16548](https://arxiv.org/abs/2602.16548)  
**Code**: —  
**Area**: Biomolecular Design / Diffusion Models / Reinforcement Learning
**Keywords**: RNA inverse design, 3D structural similarity, diffusion model, RL fine-tuning, DDPO

## TL;DR

This paper proposes RIDER, the first framework to incorporate reinforcement learning into 3D RNA inverse design. It first pretrains a conditional diffusion model (RIDE) to learn sequence–structure relationships, then applies RL fine-tuning to directly optimize 3D structural similarity rather than native sequence recovery rate, achieving over 100% improvement across all 3D self-consistency metrics.

## Background & Motivation

RNA inverse design—given a target 3D structure, finding nucleotide sequences that fold into that structure—is a critical problem in therapeutic drug development and synthetic biology.

**Root Cause of existing methods**: Nearly all SOTA methods (gRNAde, RiboDiffusion, RDesign, etc.) optimize **native sequence recovery rate (NSR)** as a proxy objective. However, RNA exhibits high degeneracy—multiple distinct sequences can fold into similar structures, and similar sequences do not necessarily yield similar structures. Consequently:

1. NSR shows no clear correlation with structural similarity (at NSR ≈ 50%, GDT_TS can range from 0 to 0.9).
2. Over-optimizing NSR limits exploration of non-native sequences.

## Method

### Overall Architecture

RIDER = RIDE (pretrained diffusion model) + RL fine-tuning

### Stage 1: Conditional Diffusion Model RIDE

**Structural representation**: The RNA 3D backbone is represented as a geometric graph, where nodes correspond to nucleotides and edges encode spatial proximity. A GVP-GNN encoder processes this graph to produce equivariant node embeddings $\mathbf{h}_c$.

**Diffusion model**: Learns the conditional distribution $p(\mathbf{x}_0 | \mathbf{h}_c)$, where $\mathbf{x}_0 \in \{0,1\}^{N \times 4}$ is the one-hot encoded sequence.

Forward process: $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \varepsilon$

Training objective:

$$\mathcal{L}_{\text{pretrain}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon, \mathbf{h}_c}\left[\|\varepsilon - \epsilon_\theta(\alpha_t \mathbf{x}_0 + \sigma_t \varepsilon, t, \mathbf{h}_c)\|^2\right]$$

The noise prediction network consists of 5 GVP-GNN layers; inference uses a DDIM sampler (50 steps).

### Stage 2: RL Fine-Tuning

The denoising sampling process is formulated as an MDP:
- **State** $s_t = (\mathbf{x}_t, t, \mathbf{h}_c)$
- **Action** $a_t$: transition from $\mathbf{x}_t$ to $\mathbf{x}_{t-\Delta t}$
- **Policy** $\pi_\theta(a_t|s_t)$: parameterized by the diffusion model
- **Reward**: received only at the end of each trajectory

**Advantage estimation improvements**:
1. Batch-mean baseline: $b = \mathbb{E}_\tau[R_{\text{traj}}]$
2. **Exponential moving average baseline** for training stability: $b^{(i)} = \beta_{\text{baseline}} \cdot b^{(i-1)} + (1-\beta_{\text{baseline}}) \cdot \bar{R}^{(i)}_{\text{batch}}$

Policy gradient objective (with PPO clipping):

$$\mathcal{L}^{RL}(\theta) = \mathbb{E}\left[\sum_{k=0}^{N_{\text{steps}}-1}\min(r_k(\theta)A, \text{clip}(r_k(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}})A)\right]$$

### Reward Functions

Four reward functions are designed based on three 3D structural similarity metrics:
- $R^{\text{gdt}} = (\text{GDT\_TS} \times w)^2$
- $R^{\text{tm}} = (\text{TM-score} \times w)^2$
- $R^{\text{rmsd}} = -(\text{RMSD} \times w)^2$
- $R^{\text{gdt\_rmsd}}$: combined reward (best overall performance)

An additional bonus reward $R_{\text{bonus}}$ is applied when GDT_TS > 0.5 or RMSD < 2.0Å.

## Key Experimental Results

### Main Results (Pretraining)

| Method | NSR ↑ |
|--------|-------|
| gRNAde | 50% |
| RiboDiffusion | 52% |
| **RIDE (Ours)** | **61%** |

### Main Results (RL Fine-Tuning)

| Method | GDT_TS ↑ | RMSD ↓ | TM-score ↑ |
|--------|----------|--------|-----------|
| gRNAde | 0.28 (27%) | 10.89 (3%) | 0.30 (28%) |
| RIDE (pretrained) | 0.33 (31%) | 10.36 (8%) | 0.33 (36%) |
| **RIDER** ($R^{\text{tm}}$) | **0.62 (72%)** | 4.31 (31%) | **0.61 (72%)** |
| **RIDER** ($R^{\text{gdt\_rmsd}}$) | **0.62 (72%)** | **3.35 (33%)** | 0.56 (68%) |

Percentages indicate the proportion of designs exceeding the designated threshold. RIDER achieves 100%+ improvement across all metrics.

### Cross-Predictor Validation

Replacing RhoFold with AlphaFold3 as the folding oracle to assess generalizability: RIDER achieves GDT_TS = 0.57, a 119% improvement over gRNAde (0.26), demonstrating that the framework captures generalizable RNA design principles.

### Key Findings

- NSR shows no clear correlation with 3D structural similarity.
- After RL fine-tuning, NSR typically decreases while GDT_TS improves, indicating that the model discovers novel sequences that fold correctly but differ from native sequences.
- GDT_TS and TM-score are highly correlated (Pearson 0.885) but each captures distinct aspects.
- The combined reward $R^{\text{gdt\_rmsd}}$ yields the most balanced performance.

## Highlights & Insights

- First RL framework for 3D RNA inverse design, directly optimizing structural similarity.
- Demonstrates the inadequacy of NSR as a proxy objective from both empirical and theoretical perspectives.
- RL fine-tuning strategy (exponential moving average baseline + PPO clipping) is stable and effective.
- A lightweight model (only 10.2M parameters) achieves substantial gains.

## Limitations & Future Work

- Relies on structure prediction models such as RhoFold as folding oracles; prediction errors propagate into the reward signal.
- RL training requires extensive sampling (60 trajectories per epoch × 80 epochs).
- Training and evaluation are conducted on only 12,011 RNA structures, limiting data scale.
- No experimental validation has been performed (wet-lab verification of designed sequences).

## Related Work & Insights

- **RNA inverse design**: gRNAde, RiboDiffusion, RDesign, and others based on supervised learning.
- **RNA structure prediction**: RhoFold, AlphaFold3, and related tools.
- **RL fine-tuning of generative models**: DDPO, RLHF, Constitutional AI, and related approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First RL-driven 3D RNA inverse design framework.
- Motivation: ⭐⭐⭐⭐⭐ — Clear and compelling analysis of NSR's deficiencies.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple reward functions and cross-oracle validation.
- Value: ⭐⭐⭐⭐ — Significant implications for RNA-based drug design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model](../../AAAI2026/image_generation/structure-based_rna_design_by_step-wise_optimization_of_latent_diffusion_model.md)
- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [\[NeurIPS 2025\] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media](../../NeurIPS2025/image_generation/diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)

</div>

<!-- RELATED:END -->
