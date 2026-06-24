---
title: >-
  [Paper Note] When Do Graph Foundation Models Transfer? A Data-Centric Theory
description: >-
  [ICML2026][Graph Learning][Graph Foundation Models] This paper utilizes graphons to represent graphs of varying sizes and domains within a unified continuous space. It proves that cross-domain output discrepancies in graph foundation models (GFMs) can be decomposed into two finite sampling errors and an intrinsic graphon domain discrepancy. Synthetic and real-world experiments demonstrate that graph size, structural shift, and spectral positional encoding stability collective…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "graphon"
  - "transfer theory"
  - "positional encoding"
  - "graph data curation"
date: 2026-05-08
content_hash: d61013bf4838a2b7
---

# When Do Graph Foundation Models Transfer? A Data-Centric Theory

**Conference**: ICML2026  
**arXiv**: [2605.29828](https://arxiv.org/abs/2605.29828)  
**Code**: https://github.com/zhuconv/GraphFM  
**Area**: Graph Learning  
**Keywords**: Graph Foundation Models, graphon, transfer theory, positional encoding, graph data curation

## TL;DR
This paper utilizes graphons to represent graphs of varying sizes and domains within a unified continuous space. It proves that cross-domain output discrepancies in graph foundation models (GFMs) can be decomposed into two finite sampling errors and an intrinsic graphon domain discrepancy. Synthetic and real-world experiments demonstrate that graph size, structural shift, and spectral positional encoding stability collectively determine transfer success.

## Background & Motivation
**Background**: Similar to language foundation models, GFMs aim to transfer a pre-trained backbone across multiple graph domains, such as social networks, recommendations, biological, and security graphs. Recent advancements focus on the model side, including larger-scale pre-training, complex tokenization, MoE routing, graph prompts, and adapters.

**Limitations of Prior Work**: Graph data exhibits significantly higher heterogeneity than text. Graphs from different domains vary not only in feature and label semantics but also in node scale, connection patterns, spectral structures, and suitable inductive biases. Consequently, a GFM may achieve positive transfer in some domains while suffering from negative transfer in others. Existing methods often provide architectures or training tricks but lack a data-centric theory to measure the transferability between two graph domains.

**Key Challenge**: If transfer difficulty is judged solely by the distance between finite graphs, the metric becomes confounded by sampling noise and scale differences. If measured by model representation distance, it is biased by the information loss of specific backbones. The paper seeks to decouple whether finite graphs sufficiently represent their underlying generation mechanisms from whether the latent graph domains themselves are similar.

**Goal**: The authors aim to establish a size-independent comparison for graph domains. Under a fixed Lipschitz backbone, they decompose cross-graph output variations into interpretable terms to provide suggestions for GFM training and data curation.

**Key Insight**: Graphons are continuous limit objects for dense graphs that represent latent graph generation mechanisms. By embedding finite graphs as step-graphons, graphs of different sizes can be compared in the same operator space, allowing domain discrepancies to be expressed as relabeling-invariant discrepancies of graphon operators.

**Core Idea**: The approach first isolates the error between finite graphs and latent graphons, then uses graphon mismatch to describe true domain discrepancies. For both set backbones with spectral positional encodings (PE) and message-passing backbones, the same "sampling error + graphon domain discrepancy + sampling error" upper bound for transfer is derived.

## Method
This work combines theory and controlled experiments to answer how difficult it is to transfer between two graph domains. It does not propose a new architecture but provides an interpretable transfer upper bound. By embedding finite graphs of any size into a continuous space, it proves that the output discrepancy between two graphs is determined by three factors: their respective sampling sufficiency and the gap between their latent generation mechanisms.

### Overall Architecture
The core approach involves substituting finite graphs with continuous objects for comparison. Given a finite graph $G$, the normalized graph operator $\Delta=A/n$ is converted into a step-graphon $W_\Delta$. Thus, graphs of sizes 128, 512, or 2048 are no longer matrices of different dimensions but piecewise functions on $[0,1]^2$, allowing for direct calculation of operator distances. In this space, the paper decomposes output discrepancies along the "sampling—domain discrepancy—sampling" chain: the more a graph deviates from its latent graphon, the larger the sampling error; the more two graphons differ, the larger the domain discrepancy. This conclusion holds for two mainstream GFM interfaces: set backbones that encode structure via PE and perform set readout (e.g., DeepSets, Graph Transformer), and message-passing backbones (e.g., GCN, GIN). Sensitivities are uniformly characterized by the Lipschitz constant $L_\theta$.

### Key Designs

**1. Step-graphon Embedding: Eliminating size-based comparability issues**
GFM transfer naturally spans different graph sizes. Since graphs of different dimensions cannot have their distances directly calculated, the difficulties of "poor approximation due to small size" and "inherent domain differences" become entangled. The solution maps a graph operator $\Delta$ of size $n$ to a step-graphon $W_\Delta(u,v)=\sum_{i,j}n\Delta_{ij}\mathbf{1}_{P_i}(u)\mathbf{1}_{P_j}(v)$, corresponding discrete neighborhood averaging to integration in continuous space. This embeds all graphs into the same function space, making cross-scale comparisons size-independent and providing a common metric for decomposing sampling error and domain discrepancy.

**2. Three-term Error Decomposition: Decomposing transfer difficulty into diagnosable terms**
With a unified space, the paper proves that under Lipschitz backbone and stable PE assumptions, the cross-graph output discrepancy for a set backbone is bounded by:
$$\|f_\theta(t_{G^{(1)}})-f_\theta(t_{G^{(2)}})\|\le L_\theta C_{PE}(\epsilon_1+\epsilon_{gra}+\epsilon_2)$$
where $\epsilon_1, \epsilon_2$ are the sampling errors of the two finite graphs relative to their latent graphons, and $\epsilon_{gra}$ is the intrinsic domain discrepancy (relabeling-invariant discrepancy). For message-passing, the constant tightens to $L_\theta(1+C_{PE})$ due to graph operator perturbations. This decomposition allows "improving transfer" to be addressed via three specific bottlenecks: increasing graph scale only reduces $\epsilon_1, \epsilon_2$, and cannot fix $\epsilon_{gra}$ when latent graphons differ significantly.

**3. Spectral PE Stability in the Upper Bound: Explaining PE dimensionality as a double-edged sword**
The constant $C_{PE}$ characteristically depicts spectral PE stability. For Eig-PE, the stability constant grows with $\sqrt{k}\,\max_{\ell\le k}1/\gamma_\ell$. If an eigengap $\gamma_\ell$ is small, the corresponding eigenvector rotates drastically under domain shift, amplifying error. Proj-PE uses top-$k$ subspace projection, which is less sensitive to basis rotation but limited by the subspace gap. The conclusion is that larger PE dimension $k$ increases expressivity but risks encountering small eigengaps, implying an inherent trade-off between expressivity and cross-domain stability.

### A Complete Example
Using synthetic experiments: take $G^{(1)}$ ($n=128$) and $G^{(2)}$ ($n=2048$, OOD size) with a DeepSets backbone + top-32 Eig-PE. If both come from the same graphon, $\epsilon_{gra}=0$ and the discrepancy is $\epsilon_1+\epsilon_2$. The small graph $G^{(1)}$ has a large $\epsilon_1$, while the large graph $G^{(2)}$ has a small $\epsilon_2$, aligning with observations that larger graphs closer to the latent graphon show monotonically decreasing token discrepancy. However, if $G^{(2)}$ comes from a perturbed graphon, $\epsilon_{gra}$ dominates, raising the upper bound. Increasing $k$ further inflates $C_{PE}$ due to small eigengaps, worsening OOD performance.

### Loss & Training
The theoretical part introduces no new loss. For experiments, a low-rank Fourier graphon classification task is constructed with a 100k total node budget, using a mixture parameter $\lambda$ to adjust size distributions across $n\in\{128,256,512,1024\}$. The test set covers $n\in\{128,256,512,1024,2048\}$. The default backbone is DeepSets + top-32 Eig-PE, with GIN used for message-passing. In real-world experiments, class-specific graphons are estimated, and graph merging augmentation (sampling larger synthetic graphs) is used to supplement missing size/graphon regions.

## Key Experimental Results

### Main Results
Real-world data experiments show the effect of graphon-based augmentation on COLLAB, IMDB-BINARY, and REDDIT-BINARY. Metrics reflect test error (lower is better).

| Augmentation Ratio | COLLAB | IMDB-BINARY | REDDIT-BINARY | Observation |
|--------------------|--------|-------------|---------------|-------------|
| Vanilla | 0.4384 | 0.4631 | 0.4108 | No augmentation |
| 1% | 0.4069 | 0.4631 | 0.4367 | Best for COLLAB, worse for REDDIT |
| 2% | 0.4428 | 0.4631 | 0.4293 | Generally unstable |
| 3% | 0.4256 | 0.4508 | 0.4084 | Best for IMDB and REDDIT |
| 4% | 0.4355 | 0.5369 | 0.4182 | Significant IMDB degradation |
| 5% | 0.4753 | 0.4631 | 0.4330 | Excessive augmentation hurts |

### Ablation Study
Experiments verify the roles of different error terms in the theoretical decomposition.

| Analysis Item | Experimental Setup | Key Phenomenon | Theoretical Support |
|---------------|-------------------|----------------|---------------------|
| Size shift | Fix node budget, increase $\lambda$ to larger graphs | Token discrepancy decreases, but test error is U-shaped | Reducing sampling error is insufficient; poor training size coverage raises ID error |
| Graph merging | Sample 1%-5% larger synthetic graphs per class | Most helpful for large size gaps; middle regions show low/volatile gains | Augmentation should target missing graphon/size regions |
| Graphon shift | Fix $\lambda=0.2$, replace 50% test graphs with perturbed graphons | In-graphon error stable; out-of-graphon error dominates | $\epsilon_{gra}$ becomes dominant during true domain shift |
| Eig-PE Dim | Scan top-$k$ eigenvector PE | Small $k$ under-expresses; medium $k$ is best; large $k$ is unstable | Matches $C_{eig}\propto\sqrt{k}/\min eigengap$ |
| Proj-PE Dim | Fix readout dim $m=32$, scan spectral rank $k$ | Large $k$ also worsens performance; OOD error sometimes better than Eig-PE | Proj-PE advantages involve learnable readouts and spectral structure |

### Key Findings
- Larger graphs are generally closer to their latent graphons, but biasing training toward large graphs reduces coverage of small graphs, increasing ID error. Size generalization entails a trade-off between coverage and approximation.
- Graph merging augmentation on real-world graphs is not monotonically effective. COLLAB requires only 1% synthetic data, while IMDB-BINARY and REDDIT-BINARY peak at 3%.
- Combining graphon shift and size shift is highly problematic. Intra-graphon size changes are manageable, but out-of-graphon error rapidly dominates total error.
- Spectral PE dimension follows a U-shape. Increasing $k$ provides structure but introduces smaller eigengaps and token instability.

## Highlights & Insights
- The paper shifts the focus of GFM transferability from model tricks to data generation mechanisms and finite sampling approximations, providing a guide for data curation.
- The three-term decomposition explains why simply scaling data or models fails if the latent graphon discrepancy is large.
- PE stability is explicitly linked to the transfer bound, offering a theoretical basis for the empirical trade-off between expressivity and stability in Graph Transformers.
- Real-world augmentation results suggest that graphon estimation and sampling must be carefully controlled, as too much synthetic data can introduce unwanted bias.

## Limitations & Future Work
- The theory primarily focuses on dense graph/graphon settings; extensions to sparse large graphs, heterogeneous graphs, and dynamic graphs are needed.
- Lipschitz backbone and spectral gap assumptions are useful for proofs but might be loose or difficult to estimate for deep GNNs/Graph Transformers.
- Synthetic graphon tasks verify the theory but remain distant from massive GFM pre-training corpora.
- Proj-PE experiments do not fully isolate theoretical predictions as learnable readouts and optimization factors are intertwined.

## Related Work & Insights
- **vs. Graphon Generalization**: Previous works analyze GNN convergence in the graphon limit; this work focuses on cross-domain output shifts in GFM scenarios for both set and message-passing interfaces.
- **vs. MMD / Gromov-Wasserstein**: MMD depends on the representation space, and GW for finite graphs mixes sampling noise. This work's graphon discrepancy compares latent mechanisms directly and separates sampling error.
- **vs. Graph Transformer PE**: Methods like PEG, SignNet, and BasisNet improve PE stability; this paper explains *how* PE stability affects the cross-domain transfer bound.
- **Insight**: When building GFM training sets, rather than just pursuing scale, one should estimate domain coverage, graphon mismatch, and size distributions to perform targeted augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Highly clear theoretical perspective on GFM transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Strong synthetic verification and real-world augmentation.
- Writing Quality: ⭐⭐⭐⭐☆ Complete theoretical chain, though notationally dense.
- Value: ⭐⭐⭐⭐☆ Significant implications for GFM data selection, size augmentation, and PE tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[ICLR 2026\] Bridging Input Feature Spaces Towards Graph Foundation Models](../../ICLR2026/graph_learning/bridging_input_feature_spaces_towards_graph_foundation_models.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[ICLR 2026\] Graphon Cross-Validation: Assessing Models on Network Data](../../ICLR2026/graph_learning/graphon_cross-validation_assessing_models_on_network_data.md)
- [\[ICLR 2026\] Multi-Domain Riemannian Graph Gluing for Building Graph Foundation Models](../../ICLR2026/graph_learning/multi-domain_riemannian_graph_gluing_for_building_graph_foundation_models.md)

</div>

<!-- RELATED:END -->
