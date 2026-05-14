---
title: >-
  [Paper Note] Generating Directed Graphs with Dual Attention and Asymmetric Encoding
description: >-
  [ICLR 2026][Image Generation][directed graph generation] This paper proposes Directo, the first directed graph generation model based on Discrete Flow Matching (DFM)…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "directed graph generation"
  - "discrete flow matching"
  - "dual attention mechanism"
  - "asymmetric positional encoding"
  - "graph generation benchmark"
date: 2026-05-08
content_hash: b149b16c5d764add
---

# Generating Directed Graphs with Dual Attention and Asymmetric Encoding

**Conference**: ICLR 2026
**arXiv**: [2506.16404](https://arxiv.org/abs/2506.16404)
**Code**: [GitHub](https://github.com/acarballocastro/DIRECTO)
**Area**: Graph Generation
**Keywords**: directed graph generation, discrete flow matching, dual attention mechanism, asymmetric positional encoding, graph generation benchmark

## TL;DR

This paper proposes Directo, the first directed graph generation model based on Discrete Flow Matching (DFM), which captures directional dependencies of directed edges via a direction-aware dual attention mechanism and asymmetric positional encoding, while establishing a standardized evaluation benchmark for directed graph generation.

## Background & Motivation

- **Background**: Graph generative models have achieved notable progress in areas such as drug discovery and social network modeling, yet the vast majority of methods focus on undirected graph generation, leaving directed graph generation substantially underexplored.
- **Limitations of Prior Work**: Directed graphs (digraphs) face two major bottlenecks: (1) at the modeling level, edge directionality causes a combinatorial explosion in the learnable space (218 directed graphs vs. only 11 undirected graphs for $n=4$ nodes), making naive extensions of undirected graph architectures inadequate; (2) at the evaluation level, standardized benchmarks and metrics for directed graph generation are lacking.
- **Key Challenge**: The few existing DAG generation methods (D-VAE, LayerDAG) require topological sorting as preprocessing and are restricted to acyclic graphs, while general directed graph generation methods are nearly absent.
- **Goal**: To construct a general directed graph generation framework applicable to both DAGs and general directed graphs.
- **Key Insight**: Jointly enhancing directed graph modeling capability along three dimensions: architecture (direction-aware attention), generative framework (discrete flow matching), and input features (asymmetric positional encoding).
- **Core Idea**: By using dual attention to distinguish information flow along incoming and outgoing edges, combined with directed positional encodings and the DFM framework, the model achieves precise generation of directed graph structures.

## Method

### Overall Architecture

Directo is built upon the Discrete Flow Matching (DFM) framework. The generation process starts from a noise distribution $\bm{p}_0$ and progressively denoises via a Continuous-Time Markov Chain (CTMC) to recover the target graph distribution $\bm{p}_1$. The denoising network predicts the node/edge distribution of clean graphs, driving the rate matrix $\bm{R}_t^\theta$ to control state transitions.

### Key Designs

#### 1. Discrete Flow Matching for Directed Graphs

- **Function**: Extends DFM to the directed graph setting.
- **Mechanism**: Unlike undirected graphs, edges $(i,j)$ and $(j,i)$ in a directed graph are independent categorical variables that may belong to different classes. The noise process performs linear interpolation independently on each node $x^{(n)}$ and each directed edge $e^{(i,j)}$:
  $p_{t|1}^X(x_t^{(n)} | x_1^{(n)}) = t \cdot \delta(x_t^{(n)}, x_1^{(n)}) + (1-t) \cdot p_{\text{noise}}^X(x_t^{(n)})$
- **Training Loss**: Cross-entropy loss $\mathcal{L} = \mathbb{E}[-\sum_n \log p_{1|t}^{\theta,(n)} - \lambda \sum_{i \neq j} \log p_{1|t}^{\theta,(i,j)}]$
- **Design Motivation**: The training–sampling decoupling property of DFM enables post-training optimization (e.g., adaptive time scheduling), which is particularly beneficial for handling the additional complexity of directed graphs.

#### 2. Asymmetric Positional Encoding

- **Function**: Designs direction-aware positional encodings for directed graphs, injecting structural information beyond local neighborhoods.
- **Mechanism**: Three schemes are investigated: (1) Magnetic Laplacian (MagLap): preserves edge direction via complex-valued phase shifts; (2) Multi-$q$ MagLap: stacks multiple MagLap matrices with different potential functions for richer representations; (3) Directed RRWP: combines forward and reverse transition probabilities to capture asymmetric flow along incoming and outgoing directions.
- **Design Motivation**: Standard Laplacian encodings are based on symmetric adjacency matrices and cannot distinguish asymmetric structural roles.

#### 3. Dual Attention Mechanism

- **Function**: Designs an attention module that distinguishes information flow from source→target and target→source directions.
- **Mechanism**: Two sets of direction-specific attention maps are constructed:
  - Source→Target: $\bm{Y}_{\text{ST}}[i,j] = \frac{\bm{Q}_S[i] \cdot \bm{K}_T[j]}{\sqrt{d_q}}$
  - Target→Source: $\bm{Y}_{\text{TS}}[i,j] = \frac{\bm{Q}_T[i] \cdot \bm{K}_S[j]}{\sqrt{d_q}}$

  After modulation by edge features via FiLM layers, the attention maps from both directions are concatenated and jointly normalized: $\bm{A}_{\text{aggr}} = \text{softmax}(\text{concat}(\bm{Y}'_{\text{ST}}, \bm{Y}'_{\text{TS}}))$, and node features are updated by aggregation: $\bm{X}' = \bm{A}_{\text{aggr}} \bm{V}_{\text{aggr}}$
- **Design Motivation**: Standard attention treats edge relationships symmetrically and fails to capture the distinct semantics of source and target nodes in directed graphs.

#### 4. Multi-Scale Information Modulation

- **FiLM Layer**: $\text{FiLM}(\bm{E}, \bm{E}_{\text{attn}}) = \bm{E} \bm{W}_1 + (\bm{E} \bm{W}_2) \odot \bm{E}_{\text{attn}} + \bm{E}_{\text{attn}}$, achieving three-level feature fusion across nodes, edges, and graphs.
- **PNA Layer**: Aggregates local information via four pooling operations (max/min/mean/std) and updates global graph features.

### Loss & Training

A standard cross-entropy loss is applied independently to node and edge predictions, with hyperparameter $\lambda$ controlling the weight of the edge reconstruction loss.

## Key Experimental Results

### Main Results

| Model | ER-DAG Ratio↓ | ER-DAG V.U.N.↑ | SBM Ratio↓ | SBM V.U.N.↑ | TPU V.U.N.↑ | VG RBF↓ |
|---|---|---|---|---|---|---|
| MLE | 15.1 | 0.0 | 11.6 | 0.0 | 24.7 | 0.618 |
| LayerDAG | 4.2 | 21.5 | - | - | 98.5 | - |
| DeFoG | 1.6 | 75.0 | 4.3 | 37.0 | 72.0 | 0.085 |
| Directo RRWP | **1.7** | **94.0** | 1.8 | **99.5** | 77.0 | **0.038** |
| Directo MagLap | **1.3** | 92.0 | **2.0** | 96.5 | **80.5** | 0.051 |

### Ablation Study

| Configuration | ER-DAG V.U.N.↑ | SBM V.U.N.↑ |
|---|---|---|
| RRWP + Double depth (more params, no dual attention) | 72.0 | 0.0 |
| RRWP + Dual attention | **94.0** | **99.5** |
| MagLap + Double depth | 80.0 | 8.0 |
| MagLap + Dual attention | **91.0** | **77.0** |

### Key Findings

1. **Dual attention is essential**: Even without positional encoding (No PE), dual attention achieves non-zero V.U.N., whereas simply increasing network depth provides no benefit whatsoever.
2. **Direction-aware PE outperforms generic PE**: Directed RRWP and MagLap significantly outperform symmetric Laplacian encodings.
3. **Implicit learning of structural constraints**: Directo generates high-quality DAGs on DAG datasets without requiring explicit acyclicity constraints.
4. Although LayerDAG enforces acyclicity, it substantially underperforms Directo on distribution-matching metrics (Ratio).

## Highlights & Insights

1. **Systematic solution**: This work simultaneously addresses the modeling challenge and the evaluation gap in directed graph generation, constituting a foundational contribution to the field.
2. **Elegant architectural design**: The dual attention mechanism concatenates attention maps from both directions before applying a unified softmax, allowing the model to adaptively allocate attention weights to incoming and outgoing directions, which is more effective than processing them independently.
3. **Good extensibility**: The framework can be directly extended to conditional generation via classifier-free guidance.
4. **Convincing ablation study**: Table 2 clearly demonstrates the large performance gap between dual attention and simply deepening the network.

## Limitations & Future Work

1. **Limited scalability**: Experiments are conducted only on medium-scale graphs of up to ~200 nodes; large-scale graphs would require acceleration strategies such as sparse attention.
2. **No explicit structural constraints**: Properties such as acyclicity are learned only implicitly; scenarios requiring hard constraints may need integration with methods such as PRODIGY.
3. **Computational cost of positional encodings**: Multi-$q$ MagLap incurs significant computational overhead on large graphs.
4. **Evaluation limited to generation tasks**: The dual attention architecture could be generalized to discriminative tasks (link prediction, node classification), but this remains unverified.

## Related Work & Insights

- **Undirected graph generation**: DiGress (Vignac et al., 2023) and DeFoG (Qin et al., 2025) serve as the strongest baselines; naively removing symmetrization operations from these models leads to a substantial performance drop.
- **DAG generation**: D-VAE and LayerDAG require topological sorting, limiting their generality.
- **Directed GNNs**: The positional encoding ideas from MagNet (Zhang et al., 2021) and Dir-GNN are adapted in this work for generative tasks.
- **Insight**: The successful application of flow matching in discrete spaces merits attention; its training–sampling decoupling property offers clear advantages for complex structure generation.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ First flow matching-based directed graph generation model with a clearly defined problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on both synthetic and real-world datasets with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-designed benchmark setup.
- **Value**: ⭐⭐⭐⭐ Provides a complete evaluation framework and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](../../CVPR2026/image_generation/trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](../../CVPR2026/image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)

</div>

<!-- RELATED:END -->
