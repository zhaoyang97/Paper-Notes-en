---
title: >-
  [Paper Note] Topology-Aware Layer Pruning for Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Layer Pruning] This paper proposes TopoVLM, a topology-aware layer pruning framework that models hidden states at each layer as point clouds and quantifies inter-layer topological consistency v…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Layer Pruning"
  - "Topological Data Analysis"
  - "Persistent Homology"
  - "Vision-Language Models"
  - "Model Compression"
date: 2026-05-08
content_hash: 28d6429d646af40d
---

# Topology-Aware Layer Pruning for Large Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.16502](https://arxiv.org/abs/2604.16502)  
**Code**: [GitHub](https://github.com/zpc456/TopoVLM)  
**Area**: Multimodal VLM / Model Compression
**Keywords**: Layer Pruning, Topological Data Analysis, Persistent Homology, Vision-Language Models, Model Compression

## TL;DR

This paper proposes TopoVLM, a topology-aware layer pruning framework that models hidden states at each layer as point clouds and quantifies inter-layer topological consistency via zigzag persistent homology. The method adaptively retains critical representation-transition layers while removing structurally redundant ones, achieving significant improvements over existing pruning methods at 50–60% sparsity.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) such as LLaVA-NeXT and VideoLLaMA2 achieve strong performance on multimodal understanding tasks, yet their deep Transformer decoder architectures impose substantial computational and memory overhead that hinders practical deployment. Layer pruning has emerged as an effective structured compression strategy.

**Limitations of Prior Work**: Existing layer pruning methods fall into two categories: (1) similarity-based methods (e.g., LLM-Pruner, LLM-Streamline) that rely on local metrics such as cosine similarity between adjacent layers; and (2) signal-driven methods (e.g., SparseGPT, Wanda) that depend on static proxy signals such as weight magnitudes and activation statistics. Both categories offer only a local snapshot perspective and fail to capture the global dynamic evolution of representations across model depth.

**Key Challenge**: Representations in LVLMs undergo non-monotonic structural changes along depth—from fine-grained visual encoding to visual-language alignment and then to instruction-conditioned reasoning. Layers that appear locally redundant may in fact serve as critical bridges between distinct semantic stages, and pruning these transition-critical layers leads to nonlinear performance degradation.

**Goal**: To design a pruning criterion that captures the global evolution of representations and distinguishes genuinely redundant layers from transition-critical ones.

**Key Insight**: Topological data analysis (TDA) focuses on the global geometry and structural organization of data. Persistent homology can track the birth and death of topological features (connected components, loops, voids) across scales, making it well suited for analyzing the dynamic evolution of representations along model depth.

**Core Idea**: Hidden states at each layer are treated as point clouds; simplicial complexes are constructed via $k$-nearest-neighbor graphs; and zigzag persistent homology is applied to track birth–death patterns of topological features across layers. Inter-layer topological consistency is defined to quantify structural redundancy—high consistency indicates that a layer introduces no new topological structure and can be safely pruned.

## Method

### Overall Architecture

Given image–instruction pairs, hidden states are extracted from all layers of the LVLM, with a special token [RET] inserted to aggregate multimodal information. The hidden states of each layer are converted into point clouds, from which $k$-nearest-neighbor graphs and simplicial complexes are constructed. Zigzag filtration is applied to compute persistent homology, yielding the Effective Persistence Image (EPI). Inter-layer topological consistency scores are derived from the EPI, and layers whose scores exceed a threshold are marked as prunable.

### Key Designs

1. **Zigzag Filtration Construction**

    - **Function**: Captures the topological evolution of representations along model depth.
    - **Mechanism**: For the hidden states $\mathbf{H}_{L_\ell} \in \mathbb{R}^{N \times d}$ at layer $L_\ell$, a $k$-nearest-neighbor graph is constructed and expanded via clique complexes to obtain a simplicial complex $\mathcal{K}_{L_\ell}$. The intersection complex $\mathcal{K}_{L_\ell, L_{\ell+1}} = \mathcal{K}_{L_\ell} \cap \mathcal{K}_{L_{\ell+1}}$ is defined between adjacent layers, forming a zigzag filtration sequence. Zero- and one-dimensional persistent homology are computed over this sequence to obtain birth–death intervals of topological features.
    - **Design Motivation**: Classical persistent homology requires monotone filtrations and cannot handle the non-monotonic variation of representations across layers. Zigzag persistent homology admits both forward and backward inclusion maps, enabling tracking of the appearance, persistence, and disappearance of topological features.

2. **Effective Persistence Image (EPI)**

    - **Function**: Transforms discrete persistence diagrams into a continuous layer–persistence planar representation.
    - **Mechanism**: Each birth–death interval $[b_j, d_j]$ is projected onto the nearest model layer indices to obtain an effective interval $[\tilde{b}_j, \tilde{d}_j]$. A continuous image is then generated via Gaussian-kernel-weighted summation: $\text{EPI}_p(u,v) = \sum_j \omega(\tau_j) \exp\!\left(-\frac{(u-\tilde{b}_j)^2 + (v-\tau_j)^2}{2\sigma^2}\right)$, where $\tau_j$ denotes persistence length.
    - **Design Motivation**: Persistence diagrams are discrete multisets that are inconvenient for layer-wise analysis and comparison. The EPI provides a differentiable and stable representation; the weight function $\omega(\tau_j)$ emphasizes long-lived features while suppressing noise.

3. **Inter-Layer Topological Consistency and Adaptive Pruning**

    - **Function**: Quantifies structural redundancy of each layer and guides pruning decisions.
    - **Mechanism**: Layer-wise topological activity $A(\ell)$ is first computed by aggregating the EPI along the persistence dimension. The inter-layer consistency score $\bar{S_p}(\ell)$ is then computed as the distance-weighted probability that topological features generated at layer $\ell$ persist in other layers, using the distance weight $\omega(\ell, \ell') = |\ell - \ell'|^\alpha$. Layers whose consistency exceeds the threshold $\epsilon \cdot \bar{S_p}^{max}$ are pruned.
    - **Design Motivation**: High consistency indicates that a layer's topological contribution is already covered by other layers, so its removal does not disrupt global topological continuity. The fundamental distinction from local similarity metrics lies in its consideration of redundancy within the global structural evolution rather than local similarity between adjacent layers.

### Loss & Training

No training is required; the method operates purely at inference time. Only a single calibration forward pass over 512 samples is needed, and zigzag filtration is performed offline without incurring inference-time overhead. Hyperparameters include the number of nearest neighbors $k$ and the distance weight exponent $\alpha$.

## Key Experimental Results

### Main Results

LLaVA-NeXT (8B) at 50% sparsity:

| Method | MME-cognition | MMMU | MathVista | MMBench | Relative Score |
|--------|--------------|------|-----------|---------|----------------|
| Full Model | 376.8 | 40.1 | 36.2 | 72.2 | 100% |
| TAMP | 341.0 | 35.7 | 31.9 | 66.3 | 90.9% |
| **Ours** | **353.1** | **38.2** | **34.6** | **69.8** | **91.6%** |

VideoLLaMA2 (7B) at 60% sparsity:

| Method | Clotho-AQA | MuchoMusic | VideoMME | NextQA-MC | Relative Score |
|--------|-----------|-----------|---------|----------|----------------|
| Full Model | 85.6 | 58.9 | 48.7 | 73.3 | 100% |
| TAMP | 84.2 | 55.9 | 42.5 | 70.9 | 95.0% |
| **Ours** | **84.9** | **58.1** | **48.0** | **72.5** | **96.7%** |

### Ablation Study

| Configuration | Description | Relative Score Change |
|---------------|-------------|----------------------|
| Remove zigzag (standard PH only) | Cannot handle non-monotonic evolution | −2.1% |
| Remove EPI (use raw PD) | Unstable layer-wise analysis | −1.5% |
| $k=5$ vs $k=15$ vs $k=25$ | $k=15$ is optimal; too small or too large degrades performance | $k=15$ best |
| $\alpha=0.5$ vs $\alpha=1.0$ vs $\alpha=2.0$ | $\alpha=1.0$ is optimal | $\alpha=1.0$ best |

### Key Findings

- Shallow layers exhibit high topological activity (forming low-level multimodal structures), while mid-to-deep layers show high topological consistency (structural redundancy), consistent with intuition.
- The advantage is more pronounced at higher sparsity (>60%), indicating that topology-aware pruning more accurately identifies truly important layers.
- The search phase requires only 5.7 minutes (single calibration pass), substantially faster than SparseGPT/Wanda, which require multiple forward passes.
- At 50% sparsity, VRAM is reduced by 43% and inference latency decreases from 105.4 ms to 60.3 ms (1.75× speedup).

## Highlights & Insights

- The innovative connection **TDA → model compression** is highly elegant—transforming persistent homology from a purely mathematical tool into a practical pruning criterion and offering a new perspective for understanding representational structure in deep networks.
- The concept of **"transition-critical layers"** is insightful: layers that appear locally redundant yet are globally indispensable are difficult to identify with conventional methods, whereas topological analysis is naturally suited to such global structural reasoning.
- The **generality of the method** is noteworthy—it is architecture-agnostic and effective on both image and video LVLMs, and is directly transferable to pure LLMs or other modalities.

## Limitations & Future Work

- Only zero- and one-dimensional persistent homology are considered; higher dimensions may contain valuable structural information but incur greater computational cost.
- The choice of calibration data may affect topological analysis results, and robustness to out-of-distribution data remains to be validated.
- The current approach is one-shot pruning; progressive pruning or post-pruning fine-tuning recovery has not been explored.
- Although the computational complexity of zigzag filtration is linear in the number of layers, practical implementation efficiency is still affected by the scale of the point clouds.

## Related Work & Insights

- **vs. LLM-Pruner / LLM-Streamline**: These methods rely on local metrics such as cosine similarity between adjacent layers and cannot capture global representation evolution; this work provides a global perspective via zigzag persistent homology.
- **vs. TAMP**: TAMP is the strongest baseline but still depends on local signals; the proposed method demonstrates a larger advantage at higher sparsity.
- **vs. Other TDA Applications in LLMs**: Existing TDA work has primarily targeted hallucination detection and reasoning analysis; this is the first application of TDA to structured pruning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of zigzag persistent homology to LVLM layer pruning; theoretically novel and practically effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers two architectures and multiple benchmarks, but validation is limited to two models; larger-scale models are absent.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical formalization is clear, but the accessibility barrier is high for readers without a TDA background.
- Value: ⭐⭐⭐⭐ — Introduces a new theoretical tool for model compression, though practical deployment requires TDA expertise.

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
