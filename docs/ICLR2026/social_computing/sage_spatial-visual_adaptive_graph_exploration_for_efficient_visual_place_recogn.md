---
title: >-
  [Paper Note] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition
description: >-
  [ICLR 2026][Social Computing][Visual Place Recognition] This paper proposes SAGE, a unified VPR training framework that introduces a lightweight Soft Probing module to enhance local feature discriminability, reconstructs an affinity graph fusing geographic distance and visual similarity online at each epoch, and focuses on the hardest samples via greedy weighted clique expansion. With the DINOv2 backbone frozen and only 1.96M parameters trained, SAGE achieves state-of-the-art results across 8 benchmarks.
tags:
  - ICLR 2026
  - Social Computing
  - Visual Place Recognition
  - DINOv2
  - Graph-based Sampling
  - Hard Sample Mining
  - parameter-efficient fine-tuning
date: 2026-05-08
content_hash: d715e40b9888aa59
---

# SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition

**Conference**: ICLR 2026
**arXiv**: [2509.25723](https://arxiv.org/abs/2509.25723)
**Code**: [https://github.com/chenshunpeng/SAGE](https://github.com/chenshunpeng/SAGE)
**Area**: Social Computing
**Keywords**: Visual Place Recognition, DINOv2, Graph-based Sampling, Hard Sample Mining, parameter-efficient fine-tuning

## TL;DR

This paper proposes SAGE, a unified VPR training framework that introduces a lightweight Soft Probing module to enhance local feature discriminability, reconstructs an affinity graph fusing geographic distance and visual similarity online at each epoch, and focuses on the hardest samples via greedy weighted clique expansion. With the DINOv2 backbone frozen and only 1.96M parameters trained, SAGE achieves state-of-the-art results across 8 benchmarks.

## Background & Motivation

1. **Core challenges in Visual Place Recognition (VPR)**: Matching query images to correct locations in large-scale geo-tagged databases requires robust retrieval under extreme viewpoint changes, illumination variation, weather/seasonal drift, and dynamic occlusion.
2. **Bottleneck of static sampling**: Existing methods (SALAD-CM, Cliquemining, etc.) adopt an offline "compute-once, use-throughout" strategy—pre-clustering based on initial features with no updates during training. As the embedding space evolves, previously hard samples become easy, while new decision-boundary hard samples remain unmined, degrading learning efficiency.
3. **Disconnection between geographic and visual information**: Most methods independently use geographic proximity or visual similarity to construct training batches, ignoring their dynamic interaction—truly hard samples are determined by the coupled relationship of "geographically close but visually dissimilar."
4. **Uniform treatment of local features**: Aggregation methods such as CFP (Centroid-Free Probing) fuse all patch features with equal weights into a global descriptor, failing to highlight subtle yet discriminative local cues.
5. **Parameter efficiency requirements**: Full fine-tuning of VFM backbones is parameter-heavy and computationally expensive; practical deployment demands parameter-efficient adaptation strategies.

## Method

### Overall Architecture

SAGE comprises four stages: (1) a frozen DINOv2 backbone extracts patch tokens; (2) Soft Probing adaptively enhances local features and aggregates them into a global descriptor; (3) InteractHead models cross-image associations and reconstructs the geo-visual affinity graph online; (4) greedy weighted clique expansion constructs training batches.

### Feature Extraction and PEFT

A pretrained DINOv2 (ViT-B or ViT-L) serves as the frozen backbone. Learnable Dynamic Power Normalization (DPN) layers are inserted into the last $N$ encoder blocks for parameter-efficient fine-tuning. The output consists of one class token and $L$ patch tokens, concatenated as $\mathbf{f} \in \mathbb{R}^{(L+1) \times M}$.

### Soft Probing (SoftP)

**Design Motivation**: CFP aggregates all patch descriptors with equal weights, obscuring subtle but critical discriminative cues. SoftP amplifies salient patches through data-driven residual reweighting.

For each patch descriptor $X_i$:

1. Compute the $\ell_2$ response: $s_i = \|X_i\|_2 + \varepsilon$
2. A two-layer MLP $\phi$ predicts a scalar, which is sigmoid-compressed and scaled: $\beta_i = \alpha \cdot \sigma(\phi(s_i))$, with range $[0, \alpha]$
3. Residual modulation: $\widetilde{X}_i = (1 + \beta_i) X_i$

This residual formulation preserves the original feature's channel structure while selectively amplifying the variance contribution of high-response locations. The modulated descriptor set $\{\widetilde{X}_i\}$ is projected to $D=128$ and $K=64$ via Feature Compression and Feature Probing MLP branches, then aggregated into a global descriptor. SoftP adds only negligible parameters.

### InteractHead

Each image descriptor $\mathbf{f}_i \in \mathbb{R}^{D \times K}$ is deterministically partitioned into $S$ equal-length segments, which are rearranged across the batch and fed into a two-layer Transformer encoder (768-dim, 16-head attention, 1024-dim FFN). Cross-image attention is performed on each segment type, capturing inter-view consistency and enhancing descriptor robustness.

### Online Graph Creation (OGC)

At each epoch, the following procedure is executed:

1. **Sampling representative features**: For each city, images are grouped by unique cluster labels and one image is randomly sampled per cluster to obtain cluster-level descriptors via the model.
2. **Building the candidate set**: Cities are sampled proportionally by cluster count; one place is selected, and $P=15$ similar places are sampled probabilistically based on cosine descriptor distance.
3. **Geographic graph filtering**: Euclidean geographic distances $d_{\text{geo}}(i,j)$ are computed for all node pairs; edges below threshold $\tau_1=25$m are retained.
4. **Clique extraction**: Complete subgraphs (cliques) are extracted from the geographic graph; the first clique satisfying $|V_C| \geq 10$ is selected.
5. **Multiplicative affinity computation**: $W_{ij} = -(d_{\text{geo}}(i,j) \cdot d_{\text{vis}}(i,j))$, where $d_{\text{vis}} = \|\mathbf{F}_i - \mathbf{F}_j\|_2$
6. **Sparse affinity graph construction**: Only edges with $W_{ij}$ exceeding threshold $\tau_2$ are retained.

This multiplicative distance design assigns the highest affinity to node pairs that are geographically close and visually similar, precisely capturing the most confusable places. Since the graph is reconstructed every epoch, it remains aligned with the current embedding space.

### Greedy Weighted Sampling (GWS)

1. **Anchor selection**: A seed score $S(i) = \frac{1}{N-1}\sum_{j \neq i} W_{ij}$ is computed for each node; the highest-scoring node is selected as the initial clique member.
2. **Greedy expansion**: Nodes are iteratively added based on highest average affinity to current clique members, until the clique reaches size $k=4$.
3. **Effect**: This automatically drills into the densest subgraph regions of the affinity graph—i.e., the clusters of mutually confusable samples where the model struggles most with fine-grained discrimination.

## Key Experimental Results

### Main Results: Comprehensive Comparison on 8 Benchmarks (Table 2 & 3)

| Method | Dim | SPED R@1 | Pitts30k R@1 | MSLS-val R@1 | Nordland R@1 | AmsterTime R@1 | Tokyo24/7 R@1 |
|--------|-----|----------|-------------|-------------|-------------|---------------|-------------|
| CosPlace | 512 | 75.5 | 88.4 | 82.8 | 58.5 | - | - |
| MixVPR | 4096 | 84.7 | 91.5 | 88.0 | 76.2 | - | - |
| BoQ | 12288 | 92.5 | 93.7 | 93.8 | 90.6 | - | - |
| EMVP | 8448 | 94.6 | 94.0 | 93.9 | 88.7 | 65.6 | 96.8 |
| FoL | 8448 | 92.1 | 93.9 | 93.1 | 87.8 | 64.6 | 96.2 |
| **SAGE** | **8448** | **98.9** | **95.8** | **94.5** | **96.0** | **83.5** | **97.5** |

SAGE (8448-d) achieves R@10 of 100% and R@1 of 98.9% on SPED (+4.3pt over EMVP); R@1 of 96.0% on Nordland (+7.3pt over EMVP); and R@1 of 83.5% on AmsterTime (+17.9pt over EMVP), demonstrating a substantial advantage in cross-era historical image retrieval.

### Compact Descriptors Remain Competitive

Even with PCA compression to 4096 dimensions, SAGE still achieves 97.7% R@1 and R@10 of 100% on SPED, and 98.2% R@1 on Pitts250k, surpassing most 8448-dim methods.

### Parameter Efficiency (Table 4)

| Method | Total Params | Trainable Params | Requires Adapter |
|--------|-------------|-----------------|-----------------|
| SALAD | 88.0M | 29.8M | No |
| SelaVPR | 102.8M | 16.2M | Yes (14.2M) |
| CricaVPR | 95.7M | 9.15M | Yes (9.2M) |
| EMVP | 88.5M | **1.96M** | No |
| **SAGE** | 88.5M(+7.88M) | **1.96M**(+7.88M) | No |

SAGE shares the same number of backbone trainable parameters as EMVP (1.96M), with only an additional 7.88M for InteractHead (required only during training, optional at inference). Compared to SALAD's 29.8M or SelaVPR's 16.2M trainable parameters, the efficiency advantage is substantial.

### Ablation Study (Table 5)

| Configuration | SPED R@1 | Pitts30k R@1 | MSLS-val R@1 | Nordland R@1 |
|--------------|----------|-------------|-------------|-------------|
| EMVP-B (CFP, baseline) | 91.8 | 93.1 | 93.2 | 80.8 |
| +SoftP+OGC | 96.8 | 94.6 | 93.6 | 95.2 |
| +SoftP+GWS (w/o OGC) | 96.5 | 93.8 | 92.5 | 94.2 |
| +CFP+OGC+GWS | 97.5 | 94.9 | 93.9 | 95.4 |
| **+SoftP+OGC+GWS (full SAGE)** | **98.0** | **95.4** | **94.3** | **95.8** |

- SoftP vs. CFP: Under identical graph sampling, SoftP improves SPED R@1 by ~0.5pt and Pitts30k by ~0.5pt.
- OGC contributes the most: Nordland jumps from 80.8% to 95.2% (+14.4pt), demonstrating that dynamic graph reconstruction is critical for seasonal variation scenarios.
- GWS requires OGC to be effective: GWS alone (w/o OGC) yields unstable performance, but the OGC+GWS combination produces synergistic gains.

### Online vs. Offline Graph Construction (Table 6)

| Strategy | Mining Time per Epoch | SPED R@1 | MSLS-val R@1 |
|----------|-----------------------|----------|-------------|
| Offline SAGE | 30.9 min (one-time) | 98.5 | 94.2 |
| **Online SAGE** | **6.2 min** | **98.9** | **94.5** |

The online strategy adds only 17.7% training time per epoch while improving SPED R@1 by 0.4pt and MSLS-val by 0.3pt, confirming that dynamic adaptation to embedding evolution is worthwhile. No additional overhead is incurred at inference.

### Convergence Analysis

On MSLS-val, SAGE reaches 93.4% R@1 by epoch 4 versus Cliquemining's 92.7%, with the advantage consistently widening in early training, indicating that dynamic sampling accelerates effective learning.

## Highlights & Insights

- **"Slow-thinking" paradigm**: Departs from the static "mine-once, use-throughout" framework by reconstructing the graph every epoch, allowing the definition of hard samples to evolve dynamically with the model.
- **Multiplicative geo-visual coupling**: $W_{ij} = -(d_{\text{geo}} \cdot d_{\text{vis}})$ precisely captures truly hard samples that are geographically close yet visually confusable.
- **SoftP residual reweighting**: Extremely lightweight (two-layer MLP); $\ell_2$-response-driven residual scaling significantly outperforms uniform CFP aggregation.
- **Extreme parameter efficiency**: With the DINOv2 backbone frozen and only 1.96M backbone trainable parameters, SAGE surpasses methods requiring 16–30M trainable parameters across 8 benchmarks.
- **Remarkable gain on AmsterTime**: R@1 for historical grayscale vs. contemporary color image retrieval improves from 65.6% to 83.5% (+17.9pt), demonstrating that dynamic sampling combined with feature enhancement is particularly effective for cross-era scenarios.

## Limitations & Future Work

- InteractHead adds 7.88M parameters and cross-image attention computation during training, which may become a bottleneck at large scale.
- Although online graph reconstruction takes only 6 minutes per epoch, it relies on geographic annotations and is inapplicable to scenarios without GPS data.
- The clique size $k=4$ in greedy clique expansion is a manually set hyperparameter; whether different datasets require different settings is not thoroughly discussed.
- Training uses only GSV-Cities + MSLS; the potential gains from incorporating additional training data are unexplored.
- At inference, the framework reduces to a standard single-stage method with InteractHead's cross-image attention unused; the cost-effectiveness of this training overhead warrants further analysis.

## Related Work & Insights

- **Global descriptor aggregation**: NetVLAD (learnable VLAD) → MixVPR (feature mixing) → CFP/EMVP (centroid-free probing + second-order statistics) → SoftP (residual-weighted probing)
- **Training sampling strategies**: Static hard sample mining → Cliquemining (offline graph sampling) → SALAD-CM (offline clustering) → SAGE (online dynamic graph + greedy clique expansion)
- **Parameter-efficient fine-tuning**: Adapter (SelaVPR) → partial encoder fine-tuning (SALAD) → DPN (EMVP/SAGE, frozen backbone + lightweight normalization layers)
- **Cross-image association**: CricaVPR / EMVP cross-image attention → SAGE InteractHead (deterministic segmentation + Transformer encoding)

## Rating

- ⭐⭐⭐⭐ **Novelty**: Dynamic geo-visual graph + greedy clique expansion constitutes a novel sampling paradigm; SoftP is concise and effective.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Comprehensive SOTA across 8 benchmarks; ablations cover every module, including online/offline comparisons and convergence analysis.
- ⭐⭐⭐⭐ **Value**: Frozen backbone with minimal trainable parameters; no additional inference overhead; code is open-sourced.
- ⭐⭐⭐⭐ **Writing Quality**: Intuitive figures, clear motivation, and concise mathematical derivations.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[NeurIPS 2025\] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](../../NeurIPS2025/social_computing/deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)
- [\[ICCV 2025\] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination](../../ICCV2025/social_computing/propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)

<!-- RELATED:END -->
