---
title: >-
  [Paper Note] QGShap: Quantum Acceleration for Faithful GNN Explanations
description: >-
  [AAAI 2026 (QC+AI Workshop)][Medical Imaging][GNN explanation] This paper proposes QGShap, a GNN explainability framework that leverages quantum amplitude amplification to accelerate exact Shapley value computation, achieving a quadratic speedup over classical Monte Carlo methods while maintaining exact (non-approximate) computation.
tags:
  - AAAI 2026 (QC+AI Workshop)
  - Medical Imaging
  - GNN explanation
  - Shapley values
  - quantum amplitude estimation
  - exact attribution
  - graph explainability
date: 2026-05-08
content_hash: cbe32e1c78a2eef9
---

# QGShap: Quantum Acceleration for Faithful GNN Explanations

**Conference**: AAAI 2026 (QC+AI Workshop)  
**arXiv**: [2512.03099](https://arxiv.org/abs/2512.03099)  
**Code**: [GitHub](https://github.com/smlab-niser/qgshap)  
**Area**: Medical Imaging  
**Keywords**: GNN explanation, Shapley values, quantum amplitude estimation, exact attribution, graph explainability

## TL;DR

This paper proposes QGShap, a GNN explainability framework that leverages quantum amplitude amplification to accelerate exact Shapley value computation, achieving a quadratic speedup over classical Monte Carlo methods while maintaining exact (non-approximate) computation.

## Background & Motivation

Graph Neural Networks (GNNs) have achieved remarkable success in critical domains such as drug discovery, social network analysis, and recommendation systems. However, their "black-box" nature hinders deployment in scenarios requiring transparency and accountability.

**Shapley values**, as the unique attribution scheme in cooperative game theory satisfying the four axioms of efficiency, symmetry, dummy player, and additivity, are considered the mathematically most rigorous approach for quantifying each component's contribution to model predictions. However, exact Shapley value computation requires enumerating all $2^n$ coalitions, which grows exponentially for a graph with $n$ nodes—a classically #P-complete problem.

Limitations of existing approximation strategies:

**SubgraphX**: Uses Monte Carlo Tree Search (MCTS) and sampling to approximate Shapley values; infeasible for large-scale dense graphs.

**GraphSVX**: Constructs a surrogate model to sample coalitions on perturbed datasets; undersamples medium-sized coalitions, reducing explanation fidelity.

**GNNShap**: Accelerates estimation via GPU parallelism and batching, but remains fundamentally a sampling-based approximation.

**GraphSHAP-IQ**: Exploits structural properties of message-passing GNNs to compute exact arbitrary-order Shapley interactions, but is inapplicable to deep architectures, dense graphs, or GNNs with nonlinear readout functions.

The root cause lies in an inherent tension: **fidelity and efficiency are mutually exclusive**—exact methods are too slow, while fast methods sacrifice accuracy. Quantum amplitude amplification offers a new path: achieving $\mathcal{O}(1/\epsilon)$ query complexity, a quadratic speedup over classical MC's $\mathcal{O}(1/\epsilon^2)$, while preserving exact computation.

## Method

### Overall Architecture

The QGShap pipeline proceeds as follows:

1. **Exhaustive coalition generation**: For input graph $G=(V,E)$, enumerate all non-empty node subsets $\mathcal{C} = \{S \subseteq V : S \neq \emptyset\}$.
2. **Coalition scoring**: Construct a masked graph $G_S$ for each subset via zero-fill encoding, replacing excluded nodes with zero vectors, then evaluate the cooperative game value $v(S) = f_\theta(G_S)$ using the trained GNN $f_\theta$.
3. **Quantum Shapley value estimation**: Normalize cooperative game values and encode them into quantum states; apply Quantum Amplitude Estimation (QAE) to compute exact Shapley values for each node.
4. **Normalization and ranking**: Normalize Shapley values across all nodes to produce a global importance ranking.

### Key Designs

1. **Three-register quantum circuit**:

    - **Partition register $Q_{pt}$** ($\ell$ qubits): Encodes the amplitude distribution of Shapley weighting coefficients $w_{|S|,|V|}$ via beta-function rotations, where $\ell = \mathcal{O}(\log \frac{(U_{max} - U_{min})n}{\varepsilon})$.
    - **Player register $Q_{pl}$** ($|V|$ qubits): Stores the superposition of all coalitions $S \subseteq V \setminus \{p_j\}$ excluding node $p_j$.
    - **Utility register $Q_{ut}$** (1 qubit): Stores the normalized cooperative game value for each coalition.
   **Design Motivation**: Controlled rotations make the amplitude of each basis state $|S\rangle$ proportional to $\sqrt{w_{|S|,|V|}}$, exactly encoding Shapley weights.

2. **Normalized cooperative game values**: Raw values are mapped to $[0,1]$: $\hat{v}(S) = \frac{v(S) - \min_{S'} v(S')}{\max_{S'} v(S') - \min_{S'} v(S')}$. This is a necessary step for quantum state encoding, ensuring utility values can be represented via qubit amplitude rotations.

3. **Quantum amplitude estimation speedup**: Two quantum oracles $U_{val}^{(+)}$ and $U_{val}^{(-)}$ implement the normalized utility function with and without node $p_j$, respectively. QAE extracts the weighted expected contributions $\phi^{(+)}(p_j)$ and $\phi^{(-)}(p_j)$, yielding the final Shapley value: $\phi(p_j) = (\max_S v(S) - \min_S v(S)) \cdot (\phi^{(+)}(p_j) - \phi^{(-)}(p_j))$. The total query complexity is $\mathcal{O}(\frac{U_{max} - U_{min}}{\varepsilon})$, achieving a near-quadratic speedup over classical MC's $\mathcal{O}(1/\varepsilon^2)$.

4. **Node-centric hierarchical approach**: QGShap computes each node's Shapley contribution within each coalition, treating nodes as "game players." This fundamentally differs from SubgraphX (which treats sampled subgraphs and remaining nodes jointly as players) and GraphSHAP-IQ (which computes exact interactions within the receptive field without explicitly enumerating coalitions).

### Loss & Training

QGShap is a **post hoc explanation framework** requiring no additional training. The underlying GNN is trained in the standard fashion: a GIN (Graph Isomorphism Network) architecture with hidden dimension 128, 3 GIN layers, trained for 100 epochs using the Adam optimizer with learning rate $10^{-3}$ and binary cross-entropy loss. QGShap operates solely on top of the pre-trained GNN to compute attribution explanations.

## Key Experimental Results

### Main Results

Comparison with classical explanation methods on two synthetic benchmark datasets, Bridge and BA2-Motif:

| Dataset | Method | Fid+ | Fid- | Sparsity | GEA | Top-2 Acc |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| Bridge | GNNExplainer | 0.60 | 1.00 | 0.75 | 0.07 | 0.10 |
| Bridge | PGExplainer | 0.99 | 1.00 | 0.75 | 0.00 | 0.00 |
| Bridge | SubgraphX | **1.00** | 1.00 | 0.75 | **1.00** | **1.00** |
| Bridge | **QGShap** | **1.00** | 1.00 | 0.75 | **1.00** | **1.00** |
| BA2-Motif | GNNExplainer | **1.00** | 1.00 | 0.75 | 0.32 | 0.50 |
| BA2-Motif | PGExplainer | 0.01 | 1.00 | 0.75 | 0.00 | 0.00 |
| BA2-Motif | SubgraphX | **1.00** | 1.00 | 0.75 | **0.40** | 0.79 |
| BA2-Motif | **QGShap** | **1.00** | 1.00 | 0.75 | **0.40** | **1.00** |

### Ablation Study

| Configuration | Key Result | Notes |
|------|---------|------|
| Classical exhaustive vs. quantum-accelerated | Query complexity $\mathcal{O}(1/\varepsilon)$ vs. $\mathcal{O}(1/\varepsilon^2)$ | Quantum method achieves near-quadratic speedup |
| Node count ≤ 8 | Feasible | Constrained by current quantum simulation resources |
| Bridge runtime | ~31 hours | 48-core AMD CPU + A100 GPU |
| BA2-Motif runtime | ~42 hours | Classical simulation on the same configuration |

### Key Findings

1. **Exact Shapley values yield perfect explanations**: On the Bridge dataset, both QGShap and SubgraphX achieve perfect GEA and Top-2 accuracy (1.00), while GNNExplainer and PGExplainer perform significantly worse. This validates the superiority of Shapley value methods in capturing critical graph structures.
2. **Critical distinction in Top-2 accuracy**: On BA2-Motif, QGShap achieves a Top-2 accuracy of 1.00 versus 0.79 for SubgraphX. This indicates that QGShap consistently identifies the two most important nodes correctly, whereas SubgraphX occasionally misses them due to sampling approximation.
3. **Failure of PGExplainer**: PGExplainer achieves GEA and Top-2 Acc of 0.00 on both datasets, demonstrating that its parameterized probabilistic model completely fails on these structured tasks.

## Highlights & Insights

- **Unifying exactness and efficiency**: QGShap is the first framework to leverage quantum computing for exact (non-approximate) Shapley value computation in GNN explainability, obtaining a quadratic speedup via amplitude amplification. This breaks the traditional dilemma of "exact but slow vs. fast but approximate."
- **Axiomatic guarantees**: Since QGShap computes exact Shapley values, its explanations automatically satisfy the four axioms of efficiency, symmetry, dummy player, and additivity—guarantees that approximation methods cannot provide.
- **Benchmark baseline for evaluation**: Exact Shapley values can serve as ground truth for evaluating other approximate explanation methods.
- **Cross-domain bridge**: This work connects quantum computing and GNN explainability, opening a new direction for quantum-classical hybrid approaches in AI interpretability.

## Limitations & Future Work

1. **Limited graph scale**: The current framework supports only graphs with ≤8 nodes. The number of coalitions $2^n$ grows exponentially with node count, rapidly increasing qubit requirements and circuit depth.
2. **High classical simulation overhead**: Running quantum simulation on an A100 GPU requires 31–42 hours; practical quantum hardware remains scarce.
3. **Quantum noise effects**: Noise and decoherence in real quantum hardware degrade amplitude estimation accuracy; the paper validates only on simulators.
4. **Validation on synthetic datasets only**: Both Bridge and BA2-Motif are synthetic datasets; the method has not been tested on real-world large-scale graph data.
5. **No direct comparison with GraphSHAP-IQ**: GraphSHAP-IQ can also compute exact Shapley interactions under certain conditions; a systematic comparison is absent.

## Related Work & Insights

The paper systematically traces the development of GNN explainability: from gradient/perturbation methods (GNNExplainer, PGExplainer) → surrogate models (GraphLIME) → Shapley value methods (SubgraphX, GNNShap, GraphSVX, GraphSHAP-IQ) → quantum acceleration (QGShap). The key insight is that quantum computing can accelerate not only model training/inference but also model **explanation**—a direction that has been largely unexplored. As quantum hardware matures, such methods may transition from theoretical validation to practical deployment. The paper's summary of the fidelity–efficiency tradeoff among existing Shapley approximation methods is particularly instructive for understanding the applicable scenarios of each approach.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of quantum computing to exact Shapley value explanation for GNNs
- Experimental Thoroughness: ⭐⭐⭐ — Only two small-scale synthetic datasets; lacks real-world validation
- Value: ⭐⭐ — Currently limited practical value due to constraints in quantum hardware and graph scale
- Writing Quality: ⭐⭐⭐⭐ — Theoretical exposition is clear; quantum algorithm details are complete

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)

<!-- RELATED:END -->
