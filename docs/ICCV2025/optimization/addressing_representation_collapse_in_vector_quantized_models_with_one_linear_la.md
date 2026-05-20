---
title: >-
  [Paper Note] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer
description: >-
  [ICCV 2025][Optimization][Vector Quantization] This paper proposes SimVQ, a method that reparameterizes codebook vectors via a single learnable linear transformation layer ($\bm{C}\bm{W}$)…
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Vector Quantization"
  - "Representation Collapse"
  - "Codebook Utilization"
  - "Linear Transformation"
  - "Multimodal"
date: 2026-05-08
content_hash: 39d5e40ff4e9aec1
---

# Addressing Representation Collapse in Vector Quantized Models with One Linear Layer

**Conference**: ICCV 2025
**arXiv**: [2411.02038](https://arxiv.org/abs/2411.02038)  
**Code**: [https://github.com/youngsheen/SimVQ](https://github.com/youngsheen/SimVQ)  
**Area**: Optimization
**Keywords**: Vector Quantization, Representation Collapse, Codebook Utilization, Linear Transformation, Multimodal

## TL;DR

This paper proposes SimVQ, a method that reparameterizes codebook vectors via a single learnable linear transformation layer ($\bm{C}\bm{W}$), converting the disjoint optimization of the codebook into a joint spatial optimization, thereby fundamentally resolving representation collapse in VQ models and achieving near-100% codebook utilization.

## Background & Motivation

Vector quantization (VQ) is a foundational technique for discretizing continuous representations, widely adopted in image generation (VQGAN) and audio synthesis (EnCodec). However, VQ models suffer from severe "representation collapse": as codebook size grows, most code vectors become "dead codes" that are never selected or updated, resulting in extremely low codebook utilization (e.g., VQGAN achieves only 1.4% utilization on a 65k codebook).

This issue directly limits the potential of VQ models as multimodal tokenizers in combination with large language models (LLMs)—for example, the Chameleon model restricts its codebook size to 8k, far below the 128k vocabulary of typical LLMs.

Limitations of existing solutions:
- Complex optimization strategies (stochastic quantization, codebook reset): high engineering complexity
- Reducing latent space dimensionality (FSQ, LFQ, VQGAN-FC): improves utilization but sacrifices model capacity (dimension reduced from 128 to 8)
- VQGAN-LC-CLIP: relies on external pretrained models, limiting generalization and imposing a performance ceiling

## Method

### Overall Architecture

The core modification of SimVQ is remarkably simple: a learnable linear layer $\bm{W} \in \mathbb{R}^{d \times d}$ is appended after the standard VQ codebook $\bm{C} \in \mathbb{R}^{K \times d}$, such that the effective codebook becomes $\bm{CW}$. During training, $\bm{C}$ is frozen (initialized from a Gaussian distribution), and only $\bm{W}$ is optimized.

### Key Designs

1. **Theoretical Analysis of Disjoint Optimization (Root Cause Analysis)**: By analyzing the gradient of the VQ commitment loss, $\bm{C}^{(t+1)} = \bm{C}^{(t)} - \eta\mathbb{E}[\delta_k^T\delta_k\bm{C}^{(t)}] + \eta\mathbb{E}[\delta_k^T z_e]$, the paper identifies the root cause of collapse: $\delta_k^T\delta_k$ is a Kronecker delta matrix with a nonzero entry only at row $k$, column $k$, meaning only the selected code vector is updated via gradient descent. Ideally, $\mathbb{E}[\delta_k^T\delta_k]$ should converge to the identity matrix, but the nearest-neighbor selection mechanism in VQ causes only a small subset of codes to be repeatedly selected and updated, while the rest remain frozen—forming a vicious cycle akin to a "cocoon effect."

2. **Linear Reparameterization and Asymmetric Optimization**: After reparameterizing the codebook as $\bm{CW}$, the commitment loss becomes $\mathcal{L}_{commit} = \|z_e - \delta_k\bm{CW}\|_2^2$. The update equation for $\bm{W}$ is $\bm{W}^{(t+1)} = (\bm{I} - \eta\mathbb{E}[\bm{C}^T\delta_k^T\delta_k\bm{C}])\bm{W}^{(t)} + \eta\mathbb{E}[\bm{C}^T\delta_k^T z_e]$. Since $\bm{C}$ is sampled from a Gaussian distribution and frozen, $\mathbb{E}[\bm{q}_k^T\bm{q}_k] = \bm{I}$, ensuring that all elements of $\bm{W}$ are updated uniformly. Optimizing $\bm{W}$ jointly updates the entire codebook $\bm{CW}$ through rotations and scalings in the embedding space. A key insight is that although multiplying two linear matrices is algebraically equivalent to a single linear layer, the asymmetric optimization dynamics in VQ make this seemingly redundant decomposition critical—$\bm{C}$ must be frozen and only $\bm{W}$ optimized; otherwise, $\bm{C}$ dominates the optimization and collapse reoccurs.

3. **Efficiency Advantage**: The memory complexity of standard VQ codebook optimization is $O(Kd)$ (where $K$ is the codebook size), whereas SimVQ requires only $O(d^2)$, since $\bm{C}$ is frozen. When $K \gg d$ (e.g., $K=65536, d=128$), this yields substantial memory savings independent of vocabulary size.

### Loss & Training

The training objective follows the standard VQ-VAE formulation: $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta\|z_e - \text{sg}(q_k\bm{W})\|_2^2 + \|\text{sg}(z_e) - q_k\bm{W}\|_2^2$. The codebook $\bm{C}$ is initialized with a Gaussian distribution and frozen throughout training; only the encoder $f_\theta$, decoder $g_\phi$, and linear layer $\bm{W}_\psi$ participate in gradient-based optimization.

## Key Experimental Results

### Main Results

| Method | Latent Dim | Codebook Size | Utilization↑ | rFID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|--------|-----------|--------------|-------------|-------|--------|-------|-------|
| VQGAN | 128 | 65,536 | 1.4% | 3.74 | 0.17 | 22.20 | 70.6 |
| VQGAN-FC | 8 | 65,536 | 100% | 2.63 | 0.13 | 23.79 | 77.5 |
| LFQ | 16 | 65,536 | 100% | 2.88 | 0.13 | 23.60 | 77.2 |
| VQGAN-LC-CLIP | 768 | 65,536 | 100% | 2.40 | 0.13 | 23.98 | 77.3 |
| **SimVQ** | **128** | **65,536** | **100%** | **2.24** | **0.12** | **24.15** | **78.4** |
| **SimVQ** | **128** | **262,144** | **100%** | **1.99** | **0.11** | **24.68** | **80.3** |

On the audio modality (LibriTTS, 0.9 kbps bandwidth), SimVQ achieves UTMOS scores of 4.00/3.51, outperforming WavTokenizer (3.74/3.43), while maintaining 100% codebook utilization.

### Ablation Study

| Configuration | Utilization | rFID | Notes |
|---------------|------------|------|-------|
| SimVQ codebook 1k | 100% | 3.67 | No collapse even at small scale |
| SimVQ codebook 8k | 100% | 2.98 | Consistent improvement with scale |
| SimVQ codebook 65k | 100% | 2.24 | SOTA |
| SimVQ codebook 262k | 100% | 1.99 | Further scaling still beneficial |
| VQGAN-LC 100k | 99.9% | 2.62 | |
| VQGAN-LC 200k | 99.8% | 2.66 | Performance saturates and reverses |
| $\bm{C}$ Gaussian init + frozen | 100% | 2.24 | Default setting |
| $\bm{C}$ uniform init + frozen | 100% | 2.31 | Insensitive to initialization |
| $\bm{C}$ Gaussian init + trainable | 100% | 2.31 | Slight performance degradation |

### Key Findings

- As codebook size scales from 1k to 262k, SimVQ consistently maintains 100% utilization with continuously improving performance (rFID from 3.67 to 1.99), whereas VQGAN-LC saturates beyond 100k.
- Jointly training $\bm{C}$ and $\bm{W}$ causes $\bm{C}$ to dominate optimization while $\bm{W}$ remains nearly unchanged, causing collapse to reappear—visualized clearly via a 2D toy experiment.
- The rank of the linear basis matrix $\bm{W}$ adaptively decreases during training, with larger codebooks converging to lower rank values, indicating that larger codebooks alleviate the dimensionality pressure in the latent space.
- SimVQ demonstrates consistent effectiveness across both image and audio modalities, confirming the generality of the approach.

## Highlights & Insights

- The solution is remarkably simple (a single linear layer), yet supported by rigorous and convincing theoretical analysis.
- The insight that "multiplying two linear matrices is algebraically equivalent to a single linear layer, yet asymmetric optimization dynamics change everything" is both surprising and enlightening.
- The 2D toy experiment visualization is highly intuitive, clearly demonstrating the behavioral differences in codebook optimization under different training modes.
- The method requires no external pretrained models, imposes no domain restrictions, and is truly plug-and-play.

## Limitations & Future Work

- Improvements in VQ reconstruction quality do not necessarily translate directly into downstream generative model performance—a limitation the authors honestly acknowledge in the Discussion.
- The theoretical analysis assumes codebook vectors are sampled from a Gaussian distribution; real-world distributions may deviate from this assumption.
- A linear transformation may be insufficient to capture nonlinear latent space structure; lightweight nonlinear extensions merit further exploration.

## Related Work & Insights

- The paper directly addresses the classical representation collapse problem in the VQ-VAE/VQGAN line of work, with a well-positioned methodological contribution.
- The ability to scale codebooks to 262k directly aligns with the vocabulary demands of LLM tokenizers, offering a viable path toward unified multimodal tokenizers.
- The "freeze-and-reparameterize" paradigm may generalize to other optimization problems involving discrete selection mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A minimalist method addressing a fundamental problem, with deep theoretical insight
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-modality evaluation (image + audio), large-scale codebook testing, comprehensive ablation
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations, excellent toy experiment visualizations
- Value: ⭐⭐⭐⭐⭐ Plug-and-play applicability with significant implications for the VQ ecosystem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](../../ICLR2026/optimization/cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](../../ICLR2026/optimization/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)
- [\[NeurIPS 2025\] Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning](../../NeurIPS2025/optimization/neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)

</div>

<!-- RELATED:END -->
