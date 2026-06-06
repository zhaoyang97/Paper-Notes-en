---
title: >-
  [Paper Note] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss
description: >-
  [NeurIPS 2025][Image Generation][Activation Steering] This paper proposes LinEAS (Linear End-to-end Activation Steering), which jointly optimizes cross-layer affine transformations in an end-to-end manner using a 1D Wass…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Activation Steering"
  - "optimal transport"
  - "Toxicity Mitigation"
  - "text-to-image"
  - "Sparse Regularization"
date: 2026-05-08
content_hash: 896396641fa07d80
---

# LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss

**Conference**: NeurIPS 2025  
**arXiv**: [2503.10679](https://arxiv.org/abs/2503.10679)  
**Code**: [github.com/apple/ml-lineas](https://github.com/apple/ml-lineas)  
**Area**: Image Generation  
**Keywords**: Activation Steering, optimal transport, Toxicity Mitigation, text-to-image, Sparse Regularization

## TL;DR

This paper proposes LinEAS (Linear End-to-end Activation Steering), which jointly optimizes cross-layer affine transformations in an end-to-end manner using a 1D Wasserstein distributional loss for global activation alignment. With only 32 unpaired samples, LinEAS efficiently steers LLM toxicity and controls concept generation in T2I models.

## Background & Motivation

**Background**: Deployed generative models (LLMs and T2I) require efficient behavioral control mechanisms—toxicity reduction, style transfer, concept erasure, etc. Existing alignment methods (RLHF, LoRA) are computationally expensive and demand large annotated datasets.

**Limitations of Prior Work**:
   - Paired-data methods (CAA, ReFT) require counterfactual data, which is unavailable in many settings.
   - Layer-wise independent optimization methods (ITI-c, Lin-AcT) ignore cross-layer interactions, leading to causal inconsistency—interventions at one layer cause unexpected shifts in downstream layers.
   - There is no automatic mechanism to select which layers and neurons to intervene upon.

**Key Challenge**: A fundamental trade-off exists between steering effectiveness and model utility—excessive intervention degrades general capabilities, while insufficient intervention yields weak effects.

**Goal**: Achieve precise, low-cost activation steering using a minimal number of unpaired samples (32 source + 32 target).

**Key Insight**: Activation steering is framed as an optimal transport problem—all affine transformations across layers are jointly optimized using a global distributional loss.

**Core Idea**: Jointly optimize coordinate-wise affine mappings across all layers so that source-distribution activations are aligned to the target distribution at every layer, with sparse regularization automatically selecting critical neurons.

## Method

### Overall Architecture

Given a pretrained model $f_1, \ldots, f_{L+1}$, affine transformations $T_\ell$ are inserted between $L$ intermediate layers:
$$\mathbf{o} = f_{L+1} \circ T_L \circ f_L \circ \cdots \circ T_1 \circ f_1(\mathbf{x})$$

Each $T_\ell(z) = \omega_\ell \odot z + b_\ell$ has only element-wise scale $\omega_\ell$ and shift $b_\ell$ parameters.

### Key Designs

#### 1. Distributional Loss Function
- **Function**: Measures the discrepancy between transformed source-distribution activations and target-distribution activations at each layer.
- **Mechanism**: Uses the 1D Wasserstein distance computed independently along each activation dimension:
  $$\Delta(U, V) = \sum_{j=1}^d W_2^2(U_{\cdot j}, V_{\cdot j}) = \frac{1}{n}\sum_{j=1}^d \|\tilde{U}_{\cdot j} - \tilde{V}_{\cdot j}\|^2$$
  where $\tilde{U}, \tilde{V}$ are sorted activation matrices.
- **Total Loss**: $\mathcal{C} = \sum_{\ell \leq L} \Delta((\xi_\ell^i)_i, (\eta_\ell^j)_j)$.
- **Design Motivation**: In high-dimensional, low-sample regimes ($d_\ell \gg N$), multi-dimensional Wasserstein estimates are unstable; 1D marginal Wasserstein distances are more robust.

#### 2. End-to-End Joint Optimization
- **Function**: All $T_\ell$ are optimized simultaneously, accounting for causal dependencies across layers.
- **Novelty over Lin-AcT**: Lin-AcT solves each layer independently in closed form, freezing all other layers and ignoring how an intervention propagates through downstream layers. LinEAS updates all layer parameters jointly via backpropagation.
- **Optimization**: Proximal SGD with cosine learning rate decay, 1K steps.

#### 3. Sparse Regularization
- **Function**: Automatically selects the subset of layers and neurons requiring intervention.
- **Sparse Group Lasso**:
  $$\mathcal{R} = \lambda_1 \sum_\ell (\|\omega_\ell - \mathbf{1}\|_1 + \|b_\ell\|_1) + \lambda_G \sum_\ell \sqrt{d_\ell}(\|\omega_\ell - \mathbf{1}\|_2 + \|b_\ell\|_2)$$
  - The $\ell_1$ norm promotes intra-layer neuron sparsity; the $\ell_2$ norm promotes inter-layer selection.
  - Storage overhead is minimal: the identity transformation ($\omega = \mathbf{1}, b = \mathbf{0}$) corresponds to no intervention.
- **Effect**: Reduces the intervention support from 100% to approximately 1% while maintaining toxicity mitigation effectiveness and improving utility.

#### 4. Continuously Controllable Steering Strength
- A scaling factor $\lambda \in [0, 1]$ is introduced: $T_\ell^\lambda(z) = (\mathbf{1} + \lambda(\omega_\ell - \mathbf{1})) \odot z + \lambda b_\ell$.
- $\lambda = 0$ applies no intervention, $\lambda = 1$ applies full intervention, and intermediate values provide smooth interpolation.

### Loss & Training

$$\mathcal{L}(\mathbf{w}, \mathbf{b}) = \mathbb{E}_{(\mathbf{x}^i) \sim p, (\mathbf{y}^j) \sim q}[\mathcal{C}(\mathbf{w}, \mathbf{b}; (\mathbf{x}^i)_i, (\mathbf{y}^j)_j)] + \gamma \mathcal{R}(\mathbf{w}, \mathbf{b})$$

- Target-distribution $q$ activations can be precomputed (bypassing $T_\ell$), reducing computational cost.
- Source-distribution $p$ activations must be computed online (propagated through $T_\ell$).
- Training: SGD, lr=0.1, 1K steps, batch size = N (32).

## Key Experimental Results

### Main Results: LLM Toxicity Mitigation

Toxicity rates (%, lower is better) on 3 models using N=32 unpaired samples:

| Model | Method | Tox(RTP) ↓ | Tox(TET) ↓ | PPL(WIK) ↓ | MMLU ↑ |
|-------|--------|------------|------------|------------|--------|
| Qwen2.5-1.5B | No intervention | 3.00 | 23.09 | 13.67 | 60.95 |
| | Lin-AcT | 1.50 | 13.88 | 13.88 | 60.09 |
| | **LinEAS** | **1.07** | **12.70** | 14.10 | 59.97 |
| Gemma2-2B | No intervention | 4.00 | 13.39 | 14.79 | 53.03 |
| | Lin-AcT | 1.60 | 7.76 | 14.78 | 52.43 |
| | **LinEAS** | **0.73** | **4.02** | 15.46 | 52.22 |
| Qwen2.5-7B | No intervention | 3.92 | 25.16 | 10.67 | 74.26 |
| | Lin-AcT | 2.72 | 21.64 | 11.42 | 72.18 |
| | **LinEAS** | **1.95** | **14.95** | 10.91 | 73.67 |

LinEAS achieves a **5.5×** toxicity reduction on Gemma2-2B, approaching LoFIT-RL which uses oracle labels.

### T2I Concept Erasure (DMD2 Model)

| Method | User Preference ↑ | IMGScore ↑ | CLIPScore ↓ |
|--------|------------------|------------|-------------|
| ITI-c | 12.4% | 0.24 | 0.19 |
| Lin-AcT | 24.4% | 0.45 | 0.18 |
| **LinEAS** | **63.3%** | **0.66** | 0.18 |

### Ablation Study

| Ablation Dimension | Results |
|-------------------|---------|
| Data size (N=1→1024) | N=32 achieves near-saturated toxicity mitigation |
| Sparsity γ (0→0.1) | Support reduced to 1%; toxicity unchanged, PPL↓ MMLU↑ |
| Training steps (100→10K) | 1K steps is optimal; 100 steps under-mitigates; 10K steps degrades utility |
| Intervention layer selection | LayerNorm layers perform best; robust to layer type choice |

### Key Findings

- End-to-end optimization generalizes better than layer-wise independent optimization (Lin-AcT) and is more robust to layer selection.
- CAA and ReFT, despite stronger supervision signals, severely degrade utility in the unpaired setting (MMLU drops by 20+).
- The **inverse mapping** $T_\ell^{-1}$ can convert concept erasure into concept injection, demonstrating strong structural regularity in activation space.

## Highlights & Insights

- **Minimal data requirement**: Effective interventions are learned from only 32 unpaired samples, far fewer than fine-tuning approaches.
- **Theoretical grounding**: Distributional alignment based on optimal transport theory, with a theoretically guaranteed continuous and bounded $\lambda$-controlled steering strength.
- **Automatic sparse selection**: Sparse Group Lasso achieves simultaneous layer and neuron selection with a 100× reduction in support.
- **Modality-agnostic**: The same framework applies to LLMs (toxicity mitigation) and T2I models (concept control).

## Limitations & Future Work

- **Composability**: Simultaneously applying multiple interventions (e.g., erasing two concepts at once) yields limited effectiveness—only 19% joint success rate.
- **Intervention selectivity**: Interventions are currently applied uniformly across all tokens, without token-level selectivity.
- **Inference overhead**: Although affine transformations are computationally cheap, storing parameters for each control target incurs additional memory cost.
- Access to internal model activations is required, making the method inapplicable to API-only models.

## Related Work & Insights

- **Lin-AcT**: The direct predecessor of LinEAS, which solves affine mappings layer-by-layer in closed form. LinEAS's end-to-end optimization addresses the cumulative error introduced by independent per-layer solutions.
- **ITI-c**: Identifies steering vectors via linear classifiers but does not account for cross-layer effects.
- **ReFT**: A low-rank representation fine-tuning method requiring paired data, which degrades in the unpaired setting.
- **AurA**: Attenuates activations according to classification ability, less flexible than affine transformations.
- **Insight**: Distributional alignment combined with sparse regularization constitutes a powerful lightweight paradigm for model control, generalizable to broader generative model steering scenarios.

## Rating

⭐⭐⭐⭐⭐ (5/5)

The method is elegant and concise, with solid theoretical foundations (OT + Sparse Lasso), and experiments covering both LLM and T2I modalities with state-of-the-art results in both. The extremely low data requirement (32 samples) and parameter count (<0.25M) make it highly practical. Composability remains the primary open challenge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](../../ICCV2025/image_generation/repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](energy_loss_functions_for_physical_systems.md)

</div>

<!-- RELATED:END -->
