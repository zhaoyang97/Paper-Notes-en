---
title: >-
  [Paper Note] When Do Graph Foundation Models Transfer? A Data-Centric Theory
description: >-
  [ICML2026][Graph Learning][graph foundation models] This paper utilizes graphons to place graphs of different sizes and domains into a unified continuous space. It proves that the cross-domain output discrepancy of graph…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "graph foundation models"
  - "graphon"
  - "transfer theory"
  - "positional encoding"
  - "graph data curation"
date: 2026-05-08
content_hash: 9954aca79c5ea3a5
---

# When Do Graph Foundation Models Transfer? A Data-Centric Theory

**Conference**: ICML2026  
**arXiv**: [2605.29828](https://arxiv.org/abs/2605.29828)  
**Code**: https://github.com/zhuconv/GraphFM  
**Area**: graph_learning  
**Keywords**: graph foundation models, graphon, transfer theory, positional encoding, graph data curation

## TL;DR
This paper utilizes graphons to place graphs of different sizes and domains into a unified continuous space. It proves that the cross-domain output discrepancy of graph foundation models can be decomposed into two finite sampling errors and an intrinsic graphon domain discrepancy. Experiments on synthetic and real graphs demonstrate that graph size, structural shift, and spectral positional encoding stability collectively determine transfer success.

## Background & Motivation
**Background**: Graph Foundation Models (GFMs) aim to transfer a pre-trained backbone to diverse graph domains (e.g., social networks, recommendation, biology, and security) similarly to language foundation models. Recent research primarily advances from the model side, including larger-scale graph pre-training, complex tokenization, MoE routing, graph prompts, and adapters.

**Limitations of Prior Work**: Graph data exhibits significantly stronger heterogeneity than text. Graphs from different domains differ not only in feature and label semantics but also in node scale, connection patterns, spectral structures, and suitable inductive biases. Consequently, a single GFM may exhibit positive transfer in some domains while suffering from negative transfer in others. Existing methods often provide architectures or training tricks but lack a data-centric theory to measure the intrinsic transfer difficulty between two graph domains.

**Key Challenge**: Judging transfer difficulty based solely on the distance between finite graphs results in a distance confounded by sampling noise and scale differences. Relying on model representation distances is influenced by the information loss of specific backbones. The paper seeks to decouple two issues: whether finite graphs sufficiently approximate their latent generation mechanisms, and whether the two latent graph domains themselves are similar.

**Goal**: The authors aim to establish a size-independent method for comparing graph domains. Under a fixed Lipschitz backbone, cross-graph output variations are decomposed into interpretable terms, which are then translated into practical recommendations for GFM training and data curation.

**Key Insight**: A graphon is the continuous limit object of dense graphs and represents the latent graph generation mechanism. By embedding finite graphs as step-graphons, graphs of different sizes can be compared within the same operator space, and the discrepancy between graph domains can be expressed as the relabeling-invariant discrepancy of graphon operators.

**Core Idea**: The error between finite graphs and the latent graphon is first isolated, followed by using graphon mismatch to describe the true domain discrepancy. For both set-based backbones with spectral positional encoding and message-passing backbones, the same "sampling error + graphon domain discrepancy + sampling error" transfer upper bound is derived.

## Method
The paper combines theoretical analysis with controlled experiments. The theoretical portion defines graphs, graphons, two types of backbones, and two spectral positional encodings, proving that: finite graphs can be equivalently embedded as step-graphons; graphs sampled from the same graphon approach the original graphon as size increases; and the difference in model output between two graphs is bounded by their respective sampling errors and the intrinsic discrepancy between their graphons.

### Overall Architecture
Given a finite graph $G$, the paper converts the normalized graph operator $\Delta=A/n$ into a step-graphon $W_\Delta$. Thus, graphs of size 128, 512, or 2048 are no longer matrices of different dimensions but piecewise functions on $[0,1]^2$. For two graphs $G^{(1)}$ and $G^{(2)}$ sampled from graphons $W^{(1)}$ and $W^{(2)}$, the model output discrepancy is decomposed along a path: the sampling error from $G^{(1)}$ to $W^{(1)}$, the graphon domain discrepancy between $W^{(1)}$ and $W^{(2)}$, and the sampling error from $W^{(2)}$ to $G^{(2)}$.

This decomposition holds for two common GFM interfaces. The first is the set-based backbone (e.g., DeepSets or Graph Transformer), which encodes graph structure into positional embeddings (PE) followed by set readout. The second is the message-passing backbone (e.g., GCN/GIN), which relies on both graph operators and node tokens. The paper uses Lipschitz conditions to encapsulate backbone sensitivity to input tokens and graph operator perturbations into a constant $L_\theta$.

### Key Designs
1. **Unifying Different Graph Sizes via step-graphons**:
    - **Function**: Transforms cross-size graph comparison from a problem of inconsistent matrix dimensions into an operator distance problem within the same function space.
    - **Mechanism**: Maps a graph operator $\Delta$ of size $n$ to $W_\Delta(u,v)=\sum_{i,j}n\Delta_{ij}1_{P_i}(u)1_{P_j}(v)$. Since discrete neighborhood averaging on a graph corresponds to integration on a step-graphon, model operations can be aligned in continuous space.
    - **Design Motivation**: GFM transfer frequently occurs across different sizes. Without size-independent representations, it is difficult to distinguish whether "poor approximation is due to small graph size" or "graph domains themselves are fundamentally different."

2. **Three-term Error Decomposition of Output Discrepancy**:
    - **Function**: Decomposes transfer difficulty into finite sampling errors and intrinsic domain discrepancy.
    - **Mechanism**: Under Lipschitz backbone and stable PE assumptions, the set backbone satisfies $\|f_\theta(t_{G^{(1)}})-f_\theta(t_{G^{(2)}})\|\le L_\theta C_{PE}(\epsilon_1+\epsilon_{gra}+\epsilon_2)$. The message-passing version includes an additional graph operator perturbation term, with the constant becoming $L_\theta(1+C_{PE})$.
    - **Design Motivation**: This formula implies that improving transfer cannot rely solely on scaling graph size or domain mixing; both sampling sufficiency and latent graphon mismatch can become bottlenecks.

3. **Incorporating Spectral Positional Encoding Stability into Transfer Bounds**:
    - **Function**: Explains why PE dimensions and eigengaps affect cross-domain generalization in GFMs.
    - **Mechanism**: The stability constant of Eig-PE increases with $\sqrt{k}\max_{\ell\le k}1/\gamma_\ell$, where small eigengaps amplify eigenvector perturbations. Proj-PE utilizes top-$k$ subspace projection and is theoretically less sensitive to basis rotation, though it still depends on the subspace gap.
    - **Design Motivation**: Graph Transformers and many GNNs rely on spectral PE. The paper transforms PE instability from an empirical tuning issue into a theoretical term $C_{PE}$, demonstrating that excessive PE dimensions may enhance expressivity while simultaneously amplifying domain shift.

### Loss & Training
The theoretical part does not propose new training losses. In experiments, the authors construct a low-rank Fourier graphon classification task, fixing a total node budget of 100k and adjusting the size distribution via a mixing parameter $\lambda$ across training sizes $n\in\{128,256,512,1024\}$. Tests include $n\in\{128,256,512,1024,2048\}$, with 2048 being the OOD size. The default model is DeepSets + top-32 Eig-PE, with GIN used to verify message-passing scenarios. For real graphs, the authors estimate class-specific graphons and sample small amounts of synthetic graphs for graph merging augmentation.

## Key Experimental Results

### Main Results
Real-world data experiments demonstrate the effect of graphon-based augmentation on COLLAB, IMDB-BINARY, and REDDIT-BINARY. The metric is test error (lower is better).

| Synthetic Graph Ratio | COLLAB | IMDB-BINARY | REDDIT-BINARY | Observation |
|-----------------------|--------|-------------|---------------|-------------|
| Vanilla | 0.4384 | 0.4631 | 0.4108 | No augmentation |
| 1% | 0.4069 | 0.4631 | 0.4367 | Best for COLLAB, degrades REDDIT |
| 2% | 0.4428 | 0.4631 | 0.4293 | Overall instability |
| 3% | 0.4256 | 0.4508 | 0.4084 | Best for IMDB & REDDIT |
| 4% | 0.4355 | 0.5369 | 0.4182 | IMDB degrades significantly |
| 5% | 0.4753 | 0.4631 | 0.4330 | Excessive augmentation harms |

### Ablation Study
The experiments validate the roles of different error terms around the theoretical decomposition rather than traditional module ablations.

| Analysis Item | Experimental Setup | Main Phenomena | Theoretical Support |
|---------------|-------------------|----------------|---------------------|
| Size shift | Fix total budget, increase $\lambda$ to shift training towards larger graphs | train-test token discrepancy decreases monotonically, but test error is U-shaped | Reducing sampling error is insufficient; inadequate coverage of smaller sizes raises ID error |
| Graph merging | Sample 1%-5% larger synthetic graphs after estimating class graphons | Most helpful during large train-test size gaps; gains are small/volatile in middle regions | Augmentation should target missing graphon/size regions rather than just increasing quantity |
| Graphon shift | Fix $\lambda=0.2$ and baseline test error <0.05, replace 50% test graphs with perturbed graphons | in-graphon error remains stable, while out-of-graphon error rises rapidly and dominates | $\epsilon_{gra}$ becomes the dominant term during true domain shifts |
| Eig-PE Dimension | Scan top-$k$ eigenvector PE | Under-expressive at small $k$, best at mid $k$, unstable at large $k$ due to small eigengaps | $C_{eig}\propto\sqrt{k}/\min eigengap$ trend matches experiments |
| Proj-PE Dimension | Fix readout dimension $m=32$, scan spectral rank $k$ | Also degrades at large $k$; OOD error is sometimes better than Eig-PE at low ranks | Theoretical advantages of Proj-PE are influenced by learnable readouts and spectral structure |

### Key Findings
- Larger graphs are generally closer to their latent graphons, but a training set biased solely toward large graphs reduces coverage of small graphs, leading to increased ID error; thus, a trade-off exists between coverage and approximation in size generalization.
- Graph merging augmentation on real graphs is not monotonically effective. COLLAB requires only 1% synthetic graphs, while IMDB-BINARY and REDDIT-BINARY perform better at approximately 3%; 5% can lead to degradation.
- The combination of graphon shift and size shift is particularly hazardous. Size variations within a known graphon can be managed by the model, but once test graphs originate from a perturbed graphon, OOD graphon error quickly dominates.
- The dimension of spectral PE should not be maximized arbitrarily. Increasing $k$ provides more structural information but introduces smaller eigengaps and greater token instability.

## Highlights & Insights
- The paper reframes "when GFMs transfer" from a model trick issue into a data generation mechanism and finite sampling approximation issue, offering a perspective suitable for guiding data curation.
- The three-term error decomposition is highly explanatory: if the latent graphon discrepancy between two domains is large, simply scaling graph size or increasing model capacity may not resolve negative transfer.
- PE stability is explicitly integrated into the transfer upper bound, addressing a realistic problem in Graph Transformers where PE dimension tuning is a compromise between expressivity and stability.
- Real-world graph augmentation results serve as a reminder that graphon estimation and sampling augmentation should be carefully controlled; excessive synthetic graphs may push the training distribution toward a different bias.

## Limitations & Future Work
- The theory primarily addresses dense graph/graphon settings; extensions are needed for sparse graphs, heterogeneous graphs, dynamic graphs, and real-world graphs with rich node attributes.
- Lipschitz backbones and spectral gap assumptions aid in derivation, but constants for deep GNNs/Graph Transformers are difficult to estimate and may be loose.
- The synthetic graphon tasks facilitate theory verification but remain distant from real-world GFM pre-training corpora.
- Proj-PE experiments do not fully isolate theoretical predictions because readouts are learnable, mixing optimization factors with spectral structures; future work could use fixed projectors and specific spectral gap structures for cleaner validation.

## Related Work & Insights
- **vs. graphon generalization theory**: Previous works analyzed the convergence or generalization of specific GNNs in the graphon limit; this paper focuses on cross-domain output shifts in GFM scenarios and covers both set tokenization and message-passing interfaces.
- **vs. MMD / Gromov-Wasserstein**: MMD depends on a selected representation space, while GW comparison of finite graphs is confounded by sampling noise; the graphon discrepancy in this paper directly compares latent generation mechanisms and explicitly separates sampling error.
- **vs. Graph Transformer PE works**: Methods like PEG, SignNet, BasisNet, and SPE improve PE stability or invariance; this paper does not propose a new PE but explains how PE stability affects the cross-domain transfer upper bound.
- **Insight**: When constructing GFM training sets, one should pursuit domain coverage, graphon mismatch, and size distributions rather than just maximum scale or dataset quantity, performing targeted data augmentation for missing regions.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Decomposing GFM transfer difficulty into graphon mismatch and finite sampling error provides a clear theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Synthetic experiments validate multiple theoretical predictions, supplemented by real graph augmentation; integration with large-scale pre-trained GFMs would be stronger.
- Writing Quality: ⭐⭐⭐⭐☆ The theoretical chain is complete, but high notation density presents a barrier for readers unfamiliar with graphons and spectral perturbations.
- Value: ⭐⭐⭐⭐☆ Highly instructive for GFM data selection, size augmentation, and PE tuning; particularly suitable for GFM data curation and evaluation design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models](finding_the_minimal_parameter_budget_for_implicit_reasoning_a_data_complexity_dr.md)
- [\[NeurIPS 2025\] Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models](../../NeurIPS2025/graph_learning/reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)

</div>

<!-- RELATED:END -->
