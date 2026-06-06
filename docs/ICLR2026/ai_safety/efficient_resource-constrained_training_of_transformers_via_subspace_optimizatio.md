---
title: >-
  [Paper Note] Efficient Resource-Constrained Training of Transformers via Subspace Optimization
description: >-
  [ICLR 2026][AI Safety][subspace optimization] This paper proposes WASI (Weight-Activation Subspace Iteration), which leverages the observation that parameter subspaces remain stable during fine-tuning to simultaneously c…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "subspace optimization"
  - "transformer compression"
  - "SVD"
  - "activation compression"
  - "edge deployment"
date: 2026-05-08
content_hash: f3b35636c0acb6ff
---

# Efficient Resource-Constrained Training of Transformers via Subspace Optimization

**Conference**: ICLR 2026
**arXiv**: [2510.09160](https://arxiv.org/abs/2510.09160)  
**Code**: [https://github.com/Le-TrungNguyen/ICLR2026-WASI.git](https://github.com/Le-TrungNguyen/ICLR2026-WASI.git)  
**Area**: AI Safety
**Keywords**: subspace optimization, transformer compression, SVD, activation compression, edge deployment

## TL;DR

This paper proposes WASI (Weight-Activation Subspace Iteration), which leverages the observation that parameter subspaces remain stable during fine-tuning to simultaneously compress both the weights (via SVD + Gram-Schmidt subspace iteration) and activations (via Tucker decomposition) of Transformers. Both training and inference are performed entirely within low-rank representations, achieving 62× training memory compression and 1.4× speedup on Raspberry Pi 5 with negligible accuracy loss.

## Background & Motivation

**Background**: Deploying Transformers on edge devices poses severe memory and computational challenges. While methods such as LoRA reduce the number of trainable parameters, inference still operates in the full-rank space; activation maps during the forward pass constitute the primary memory bottleneck.

**Limitations of Prior Work**:
- **LoRA and variants**: Reduce training parameters but require merging back to full rank at inference, leaving inference overhead unchanged; storing both frozen weights and adapters during training can actually increase memory usage.
- **ASVD / FWSVD**: Compress models via truncated SVD but lack a theoretical connection between truncation error and model performance.
- **SVD-LLM**: Addresses the theoretical gap but is limited to LLMs and does not support 4D or higher activation tensors in vision Transformers.
- **AMC**: Compresses activations with HOSVD, but recomputing HOSVD at every iteration incurs substantial overhead and rank fluctuations lead to unstable memory usage.
- **ASI**: Replaces HOSVD with subspace iteration at a fixed activation rank, reducing computation, but does not compress weights.

**Core Insight**: The intrinsic parameter subspace remains stable during fine-tuning (small learning rate → minuscule per-step updates → negligible change in SVD bases), so after an initial SVD, inexpensive subspace iteration suffices to track basis changes without recomputation at every step.

**Core Idea**: Jointly compress both weights (WSI) and activations (ASI) so that training and inference are executed entirely within the low-rank space.

## Method

### Overall Architecture

WASI is a unified framework combining WSI (Weight Subspace Iteration) and ASI (Activation Subspace Iteration):
- Forward pass: $\mathcal{A}_{i+1} = \mathcal{A}_i R_i^T L_i^T$ (computed in the low-rank space)
- Backward pass: gradients are computed directly in the low-rank space; weight update $L_i R_i = L_i R_i + \eta \cdot \widetilde{\nabla_{\mathcal{W}_i}\mathcal{L}}$
- Inference runs directly on the compressed representations $(L_i, R_i)$ without reconstructing full-rank weights

### Key Designs

1. **WSI (Weight Subspace Iteration)**:

    - **Initialization ($t=0$)**: A full SVD is performed on weight $\mathcal{W}_i$; the optimal rank $K_i$ is determined by a variance-explained threshold $\varepsilon$, yielding $\mathcal{W}_i \approx L_i R_i$.
    - **Subsequent iterations ($t>0$)**: Gram-Schmidt orthogonalization tracks subspace changes at a computational cost far lower than recomputing SVD.
    - **Error control**: The threshold $\varepsilon$ ensures that the retained variance satisfies $\sum_{j=1}^{K_i} \sigma_{i,j}^2 \geq \varepsilon$.

2. **ASI (Activation Subspace Iteration) enhancements**:

    - **Dynamic-programming rank selection**: Replaces the brute-force search in ASI with a DP formulation that minimizes memory usage subject to a target perplexity constraint, reducing search complexity from exponential to linear.
    - **3D activation support**: Extends Tucker decomposition to support 3D activation tensors $\mathcal{A}_i \in \mathbb{R}^{B \times N_i \times I_i}$ arising in Transformers.

3. **Unified forward-backward computation**: Both forward and backward passes are executed directly in the compressed representation, eliminating decompression/recompression round-trips.

### Loss & Training

Standard cross-entropy loss is used; the key contribution lies in performing all gradient computations within the low-rank space:
- Weight gradient: $\widetilde{\nabla_{\mathcal{W}_i}\mathcal{L}} = f_{LR}(\tilde{\mathcal{A}_i}, \widetilde{\nabla_{\mathcal{A}_{i+1}}\mathcal{L}})$
- Activation gradient: $\widetilde{\nabla_{\mathcal{A}_i}\mathcal{L}} = \widetilde{\nabla_{\mathcal{A}_{i+1}}\mathcal{L}} \cdot L_i R_i$

## Key Experimental Results

### Main Results: Multiple Models and Datasets

| Model | Dataset | Train Mem. Compression | Inference Mem. Compression | Train FLOPs Reduction | Accuracy Change |
|-------|---------|----------------------|--------------------------|----------------------|----------------|
| ViT | CIFAR-10 | **62×** | **62×** | **2×** | −0.5% |
| ViT | Pets | **62×** | **62×** | **2×** | 0% |
| SwinT | CUB | ~50× | ~50× | 1.5× | +2% (surpasses baseline) |
| SwinT | Flowers | ~50× | ~50× | 1.5× | −1% |
| SwinT | CIFAR-100 | ~50× | ~50× | 1.5× | 0% |
| TinyLlama | BoolQ | **953×** (activation) / **30×** (weight) | **30×** | **13×** | 0% |

### Ablation Study: WSI vs. Full SVD

| Method | ε=0.4 | ε=0.6 | ε=0.8 | ε=0.9 | Compute Cost Ratio |
|--------|-------|-------|-------|-------|--------------------|
| Full SVD | Low acc. | Medium | High | Near-full | 1.0× |
| WSI | Low acc. | Medium | High | Near-full | **0.74× (1.36× savings)** |
| Accuracy gap at equal FLOPs | — | — | — | — | WSI higher by **35%** |

### On-Device Evaluation: Raspberry Pi 5

| Setting | Train Time/Step | Inference Time/Step | Speedup |
|---------|----------------|--------------------|---------| 
| Vanilla | baseline | baseline | 1.0× |
| WASI (ε=0.9) | faster | faster | **~1.4×** |
| WASI (ε=0.4) | fastest | fastest | **>2×** |

### Key Findings

- Layer rank $K_i$ remains **constant** over 50 epochs, validating the subspace stability assumption.
- WSI requires 1.36× fewer FLOPs than full SVD recomputation, yielding 35% higher accuracy at the same budget.
- The leading principal components of activations capture >90% of the variance, indicating high compressibility.
- On CUB, WASI with SwinT **surpasses** vanilla fine-tuning accuracy, suggesting a regularization effect from the low-rank constraint.
- Activation compression reaches **953×** on TinyLlama, demonstrating significant compression potential for LLMs.

## Highlights & Insights

- **Both training and inference in the compressed space** — fundamentally distinct from LoRA, which must merge adapters back to full rank at inference, making WASI naturally suited for edge deployment.
- **Empirical validation of the subspace stability assumption**: Figure 3(a) directly visualizes the stability of singular values throughout fine-tuning, providing strong alignment between theory and experiment.
- **DP-based rank selection over brute-force search**: Reducing exponential to linear search complexity substantially improves practical usability.
- **Compression can surpass the baseline**: The accuracy gain on CUB indicates that the low-rank constraint acts as an implicit regularizer.
- **62× memory compression** implies that a model originally requiring 62 GB can be trained on a 1 GB device.

## Limitations & Future Work

- LLM validation is limited: experiments are conducted only on the last 5 layers of TinyLlama; effectiveness on larger-scale LLMs remains unknown.
- The threshold $\varepsilon$ requires task- and model-specific tuning; the optimal value may vary across settings.
- Gram-Schmidt orthogonalization may exhibit numerical instability at extremely small ranks.
- The LoRA adapters in SVD-LLM confer a FLOPs advantage; WASI's FLOPs benefit is less pronounced than its memory benefit.
- Combining WASI with orthogonal compression techniques such as quantization and knowledge distillation remains unexplored.

## Related Work & Insights

- **vs. LoRA**: LoRA reduces only training parameters without compressing inference; WASI compresses both, offering a clear edge-deployment advantage.
- **vs. SVD-LLM**: SVD-LLM is restricted to LLMs and memory can actually increase at low compression ratios due to LoRA adapter overhead; WASI is general-purpose with no additional overhead.
- **vs. ASI**: ASI compresses activations only, leaving the inference space unchanged; WASI jointly compresses both.
- **vs. AMC**: AMC incurs heavy per-step HOSVD recomputation; WASI uses an initial SVD followed by subspace iteration, achieving computational efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ — A unified framework that simultaneously compresses weights and activations, with theoretical grounding for the subspace stability assumption.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Convincing RPi5 deployment validation across ViT, SwinT, and TinyLlama.
- Writing Quality: ⭐⭐⭐⭐ — Complete mathematical derivations with detailed computational complexity analysis.
- Value: ⭐⭐⭐⭐ — A practical solution for edge deployment of Transformers; the 62× compression ratio carries significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometrically Constrained Outlier Synthesis](../../ICML2026/ai_safety/geometrically_constrained_outlier_synthesis.md)
- [\[AAAI 2026\] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks](../../AAAI2026/ai_safety/improving_the_convergence_rate_of_ray_search_optimization_for_query-efficient_ha.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[ICML 2026\] Training-Free Coverless Multi-Image Steganography with Access Control](../../ICML2026/ai_safety/training-free_coverless_multi-image_steganography_with_access_control.md)

</div>

<!-- RELATED:END -->
