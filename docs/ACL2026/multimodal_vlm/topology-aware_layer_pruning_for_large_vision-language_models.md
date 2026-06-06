---
title: >-
  [Paper Note] Topology-Aware Layer Pruning for Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Layer Pruning] Ours proposes TopoVLM, a layer pruning framework based on Topological Data Analysis (TDA). It models hidden states across layers as point clouds and quantifies inter-layer topolo…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Layer Pruning"
  - "Topological Data Analysis"
  - "Persistent Homology"
  - "Vision-Language Models"
  - "Model Compression"
date: 2026-05-08
content_hash: 44339e1c37939d8a
---

# Topology-Aware Layer Pruning for Large Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.16502](https://arxiv.org/abs/2604.16502)  
**Code**: [GitHub](https://github.com/zpc456/TopoVLM)  
**Area**: Multimodal VLM / Model Compression  
**Keywords**: Layer Pruning, Topological Data Analysis, Persistent Homology, Vision-Language Models, Model Compression

## TL;DR

Ours proposes TopoVLM, a layer pruning framework based on Topological Data Analysis (TDA). It models hidden states across layers as point clouds and quantifies inter-layer topological consistency via zigzag persistent homology. This approach adaptively preserves critical representation transition layers and removes structurally redundant ones, significantly outperforming existing pruning methods at 50-60% sparsity.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) such as LLaVA-NeXT and VideoLLaMA2 exhibit excellent performance in multimodal understanding, but the computational and memory overhead of deep Transformer decoder architectures limits practical deployment. Layer pruning has gained attention as an effective structural compression strategy.

**Limitations of Prior Work**: Existing layer pruning methods are divided into two categories: (1) Similarity-based methods (e.g., LLM-Pruner, LLM-Streamline) rely on local metrics like cosine similarity between adjacent layers; (2) Signal-driven methods (e.g., SparseGPT, Wanda) rely on static proxy signals like weight magnitude and activation statistics. Both categories only provide local snapshot views and fail to capture the global dynamic evolution of representations along the model depth.

**Key Challenge**: Representations in LVLMs undergo non-monotonic structural changes along the depth—from fine-grained visual encoding to vision-language alignment and then to instruction-conditioned reasoning. Layers that appear locally redundant may actually serve as critical bridges between different semantic stages; pruning these "transition-critical layers" leads to non-linear performance degradation.

**Goal**: Design a pruning criterion capable of capturing the global evolution of representations to distinguish between true structural redundancy and transition-critical layers.

**Key Insight**: Topological Data Analysis (TDA) focuses on the global geometry and structural organization of data. Persistent homology can track the birth and death of topological features (connected components, loops, voids) across different scales, making it ideally suited for analyzing the dynamic evolution of representations along depth.

**Core Idea**: Treat hidden states of each layer as point clouds to construct simplicial complexes via k-nearest neighbor graphs. Track birth and death patterns of topological features across layers using zigzag persistent homology and define inter-layer topological consistency to quantify structural redundancy. High consistency implies that a layer introduces no new topological structures and can be safely pruned.

## Method

### Overall Architecture

Input image-instruction pairs pass through the LVLM to obtain hidden states for each layer, with a special [RET] token inserted to aggregate multimodal information. Hidden states are transformed into point clouds to construct k-nearest neighbor graphs and simplicial complexes. Persistent homology is computed through zigzag filtration to generate Effective Persistence Images (EPI). Inter-layer topological consistency scores are extracted from the EPI, and layers exceeding a threshold are marked for pruning.

### Key Designs

1. **Zigzag Filtration Construction**:

    - **Function**: Capture the topological evolution of representations along the model depth.
    - **Mechanism**: For hidden states $\mathbf{H}_{L_\ell} \in \mathbb{R}^{N \times d}$ of each layer $L_\ell$, construct a k-nearest neighbor graph and obtain the simplicial complex $\mathcal{K}_{L_\ell}$ via clique expansion. An intersection complex $\mathcal{K}_{L_\ell, L_{\ell+1}} = \mathcal{K}_{L_\ell} \cap \mathcal{K}_{L_{\ell+1}}$ is defined between adjacent layers, forming a zigzag filtration sequence. 0-dimensional and 1-dimensional persistent homology are computed for this sequence to obtain birth-death intervals of topological features.
    - **Design Motivation**: Classical persistent homology requires monotonic filtration and cannot handle non-monotonic changes in inter-layer representations. Zigzag persistent homology allows forward and backward inclusion maps, enabling the tracking of topological feature appearance, persistence, and disappearance.

2. **Effective Persistence Image (EPI)**:

    - **Function**: Transform discrete persistence diagrams into continuous layer-persistence plane representations.
    - **Mechanism**: Project each birth-death interval $[b_j, d_j]$ onto the nearest model layer indices to obtain effective intervals $[\tilde{b}_j, \tilde{d}_j]$. Then, generate a continuous image using a weighted sum of Gaussian kernels: $\text{EPI}_p(u,v) = \sum_j \omega(\tau_j) \exp(-\frac{(u-\tilde{b}_j)^2 + (v-\tau_j)^2}{2\sigma^2})$, where $\tau_j$ is the persistence length.
    - **Design Motivation**: Persistence diagrams are discrete multisets not conducive to hierarchical analysis and comparison. EPI provides a differentiable and stable representation while emphasizing long-lived features and suppressing noise through the weight function $\omega(\tau_j)$.

3. **Inter-layer Topological Consistency and Adaptive Pruning**:

    - **Function**: Quantify structural redundancy for each layer and guide pruning decisions.
    - **Mechanism**: First calculate layer-wise topological activity $A(\ell)$ (aggregating EPI along the persistence dimension), then calculate the inter-layer consistency score $\bar{S_p}(\ell)$—a weighted probability measuring the persistence of topological features generated by layer $\ell$ in other layers, using a distance weight $\omega(\ell, \ell') = |\ell - \ell'|^\alpha$. Layers with consistency higher than threshold $\epsilon \cdot \bar{S_p}^{max}$ are pruned.
    - **Design Motivation**: High consistency implies that the topological contribution of a layer is already covered by other layers; removing it does not disrupt global topological continuity. The fundamental difference from local similarity measures is that it considers redundancy within the global structural evolution rather than local similarity between adjacent layers.

### Loss & Training

A maintenance-free, pure inference-time pruning method. It requires only one calibration forward pass (512 samples), with zigzag filtration performed offline, introducing no inference-time overhead. Hyperparameters include the k value for k-NN and the distance weight index α.

## Key Experimental Results

### Main Results

LLaVA-NeXT (8B) at 50% sparsity:

| Method | MME-cognition | MMMU | MathVista | MMBench | Relative Score |
|------|-------------|------|-----------|---------|---------|
| Full Model | 376.8 | 40.1 | 36.2 | 72.2 | 100% |
| TAMP | 341.0 | 35.7 | 31.9 | 66.3 | 90.9% |
| **Ours** | **353.1** | **38.2** | **34.6** | **69.8** | **94.6%** |

VideoLLaMA2 (7B) at 60% sparsity:

| Method | Clotho-AQA | MuchoMusic | VideoMME | NextQA-MC | Relative Score |
|------|-----------|-----------|---------|----------|---------|
| Full Model | 85.6 | 58.9 | 48.7 | 73.3 | 100% |
| TAMP | 84.2 | 55.9 | 42.5 | 70.9 | 95.0% |
| **Ours** | **84.9** | **58.1** | **48.0** | **72.5** | **96.7%** |

### Ablation Study

| Configuration | Description | Relative Score Change |
|------|------|------------|
| Remove zigzag (Standard PH only) | Cannot handle non-monotonic evolution | -2.1% |
| Remove EPI (Raw PD) | Unstable layer-wise analysis | -1.5% |
| k=5 vs k=15 vs k=25 | k=15 is optimal; too small/large degrades | k=15 Best |
| α=0.5 vs α=1.0 vs α=2.0 | α=1.0 is optimal | α=1.0 Best |

### Key Findings

- High topological activity in shallow layers (forming low-level multimodal structures) and high topological consistency in middle/deep layers (structural redundancy) aligns with intuition.
- Advantages are more pronounced at high sparsity rates (>60%), indicating that topology-aware pruning more accurately identifies truly important layers.
- The search phase takes only 5.7 minutes (single calibration), much faster than SparseGPT/Wanda which require multiple forward passes.
- VRAM reduced by 43% at 50% sparsity, with inference latency dropping from 105.4ms to 60.3ms (1.75x speedup).

## Highlights & Insights

- **Innovative connection between TDA and Model Compression** is elegant—transforming persistent homology from a pure mathematical tool into a practical pruning criterion, providing a new perspective for understanding the representation structure of deep networks.
- **The "Transition-Critical Layer" concept** is insightful—layers that appear redundant locally but are indispensable globally are difficult for traditional methods to identify; topological analysis is naturally suited for this type of global structure reasoning.
- **Generality of the method** is noteworthy—it does not depend on specific model architectures and is effective for both image and video LVLMs, with direct transferability to pure LLMs or other modalities.

## Limitations & Future Work

- Only 0D and 1D persistent homology are considered; higher dimensions might contain valuable structural information but incur higher computational overhead.
- Selection of calibration data may affect topological analysis results; robustness to out-of-distribution data remains to be verified.
- Currently a one-shot pruning method; progressive pruning or recovery post-fine-tuning has not been explored.
- Although the computational complexity of zigzag filtration is linear with the number of layers, the efficiency of practical implementation is still limited by the size of the point cloud.

## Related Work & Insights

- **vs LLM-Pruner / LLM-Streamline**: Local metrics based on cosine similarity of adjacent layers cannot capture global representation evolution; Ours provides a global perspective through zigzag PH.
- **vs TAMP**: TAMP is the strongest baseline but still relies on local signals; Ours shows more significant advantages at high sparsity rates.
- **vs other TDA applications in LLMs**: Existing TDA work primarily focuses on hallucination detection and reasoning analysis; Ours is the first to apply it to structural pruning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of zigzag persistent homology to LVLM layer pruning, theoretically novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of two architectures and multiple benchmarks, but verification is missing for larger-scale models.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical formalization, but the barrier to entry is high for readers without a TDA background.
- Value: ⭐⭐⭐⭐ Provides a new theoretical tool for model compression, though practical deployment requires TDA expertise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/multimodal_vlm/comp_collaborative_multi-mode_pruning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
