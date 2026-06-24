---
title: >-
  [Paper Note] Nonparametric Teaching for Graph Property Learners
description: >-
  [ICML2025 Spotlight][Optimization][Nonparametric teaching] This paper proposes the GraNT paradigm to extend nonparametric teaching theory to graph property learning scenarios. By greedily choosing a subset of graph samples with the "largest prediction deviation," GraNT accelerates GCN training, reducing training time by 30%–47% while maintaining generalization performance.
tags:
  - "ICML2025 Spotlight"
  - "Optimization"
  - "Nonparametric teaching"
  - "graph convolutional networks"
  - "training efficiency"
  - "graph neural tangent kernel"
  - "greedy sample selection"
date: 2026-05-08
content_hash: f469e308801b4cc5
---

# Nonparametric Teaching for Graph Property Learners

**Conference**: ICML2025 Spotlight  
**arXiv**: [2505.14170](https://arxiv.org/abs/2505.14170)  
**Code**: [Project Homepage](https://chen2hang.github.io/_publications/nonparametric_teaching_for_graph_proerty_learners/grant.html)  
**Area**: Optimization / Graph Learning  
**Keywords**: Nonparametric teaching, graph convolutional networks, training efficiency, graph neural tangent kernel, greedy sample selection

## TL;DR
This paper proposes the GraNT paradigm to extend nonparametric teaching theory to graph property learning scenarios. By greedily choosing a subset of graph samples with the "largest prediction deviation," GraNT accelerates GCN training, reducing training time by 30%–47% while maintaining generalization performance.

## Background & Motivation
- **Core Problem**: Learning the implicit mapping of graphs to properties $f^*: \mathbb{G} \mapsto \mathcal{Y}$ using Graph Convolutional Networks (GCNs) is highly expensive, especially on large-scale graphs (with millions of nodes or hundreds of thousands of molecular graphs) where the training cost is massive.
- **Limitations of Prior Work**: Existing training acceleration methods (such as normalization, graph decomposition, and lazy updates) mostly stem from network architecture or engineering perspectives, lacking a systematic theoretical framework. Meanwhile, existing nonparametric teaching theories are only applicable to regular feature data (e.g., MLP + coordinates $\rightarrow$ signals) and do not account for the **graph structural information** of the input.
- **Key Challenge** (Theoretical Gap): GCNs perform gradient descent in the parameter space, whereas nonparametric teaching performs functional gradient descent in the function space. A gap exists between the two: how feature aggregation of the adjacency matrix $\mathbf{A}$ affects parameter gradients, and whether the dynamic GNTK can converge to the structure-aware kernel in functional gradients, remain unanswered.
- **Design Motivation**: Systematically analyze the impact of graph structures on parameter gradients and prove that the parameter gradient descent evolution of GCN matches functional gradient descent. This theoretically justifies introducing nonparametric teaching to graph property learning and enhances training efficiency via a greedy teaching algorithm.

## Method

### Overall Architecture: Graph Neural Teaching (GraNT)
GraNT introduces a **teacher-learner** paradigm:
1. The **target mapping** $f^*$ is realized by a dense set of graph-property pairs;
2. The **Teacher** greedily selects a subset $\{G_i\}_m^*$ ($m \leq N$) from the training set $\{(\mathbf{G}_i, \mathbf{y}_i)\}_N$ in each round;
3. The **Learner** (GCN) is updated via parameter gradient descent only on the selected subset.

### Key Designs 1: Structure-Aware Parameter Gradient Analysis
The paper adopts a flexible GCN formulation that weights aggregated features of different convolution orders:

$$\mathbf{X}^{(\ell)} = \sigma\!\left(\mathbf{A}^{[\kappa_\ell]} \operatorname{diag}(\mathbf{X}^{(\ell-1)}; \kappa_\ell) \cdot \mathbf{W}^{(\ell)}\right)$$

where $\mathbf{A}^{[\kappa]} = [\mathbf{I}, \mathbf{A}, \dots, \mathbf{A}^{\kappa-1}]$ concatenates neighbor aggregations of different orders. The paper analytically derives gradient expressions of output with respect to weights of each layer in a two-layer GCN (Eq. 13, 17), revealing:
- The parameter gradients **do not depend on the number of nodes** $n$ in the graph, but only depend on feature dimensions and convolution orders;
- When the convolution order is $\kappa=1$, the formulation degenerates to MLP gradients, demonstrating that this work generalizes the nonparametric teaching of MLPs.

### Key Designs 2: Consistency between GCN Evolution and Functional Gradients
Parameter updates are formulated as GCN evolution in the function space:

$$\frac{\partial f_{\theta^t}}{\partial t} = -\frac{\eta}{N} \sum_i \frac{\partial \mathcal{L}}{\partial f_{\theta^t}(\mathbf{G}_i)} \cdot K_{\theta^t}(\mathbf{G}_i, \cdot) + o(\cdot)$$

where $K_{\theta^t}(\mathbf{G}, \mathbf{G}') = \langle \frac{\partial f_{\theta^t}(\mathbf{G})}{\partial \theta^t}, \frac{\partial f_{\theta^t}(\mathbf{G}')}{\partial \theta^t} \rangle$ is the **dynamic Graph Neural Tangent Kernel (GNTK)**.

**Theorem 5 (Core Theoretical Result)**: For a convex loss $\mathcal{L}$, the dynamic GNTK pointwise converges to the structure-aware canonical kernel:
$$\lim_{t\to\infty} K_{\theta^t}(\mathbf{G}_i, \cdot) = K(\mathbf{G}_i, \cdot)$$

This provides the first proof that the evolution of GCN parameter gradient descent is consistent with functional gradient descent in nonparametric teaching.

### Key Designs 3: Greedy Teaching Selection Strategy
Based on Proposition 6 (sufficient loss descent guarantee), GraNT selects graphs with the largest GCN prediction deviation:

$$\{\mathbf{G}_i\}_m^* = \arg\max_{\{\mathbf{G}_i\}_m \subseteq \{\mathbf{G}_i\}_N} \| [f_\theta(\mathbf{G}_i) - f^*(\mathbf{G}_i)]_m \|_2$$

Two implementation variants:
- **GraNT (B)**: Selects the **batch** with the largest average deviation;
- **GraNT (S)**: Proportionally selects **individual graphs** with the largest deviation from each batch and reorganizes them into a new batch.

Node-level tasks require normalization by the number of nodes: selecting graphs with the largest $\|f_\theta(\mathbf{G}_i) - f^*(\mathbf{G}_i)\|_2 / n_i$.

## Key Experimental Results

### Main Results: Training Time and Test Performance (Table 1)

| Dataset | Task | Time w/o GraNT (s) | GraNT (B) Time (s) | Speedup | Performance Retained |
|:---|:---|:---|:---|:---|:---|
| QM9 | Graph-level Regression | 9654.81 | 6392.26 | **-33.79%** | MAE 0.0051 Unchanged |
| ZINC | Graph-level Regression | 33033.82 | 20935.24 | **-36.62%** | MAE 0.0048 Unchanged |
| ogbg-molhiv | Graph-level Classification | 2163.50 | 1457.39 | **-32.64%** | ROC-AUC 0.7572 $\rightarrow$ 0.7676 $\uparrow$ |
| ogbg-molpcba | Graph-level Classification | 130191.26 | 80465.06 | **-38.19%** | AP 0.3270 $\rightarrow$ 0.3358 $\uparrow$ |
| gen-reg | Node-level Regression | 3344.78 | 2308.97 | **-30.97%** | MAE 0.0007 Unchanged |
| gen-cls | Node-level Classification | 11662.25 | 6145.72 | **-47.30%** | ROC-AUC 0.9150 $\rightarrow$ 0.9157 $\uparrow$ |

### Comparison with SOTA Methods

**QM9 Comparison with Active Learning Methods (Table 2)**:

| Method | Time (s) | MAE |
|:---|:---|:---|
| AL-3DGraph (Default settings) | 12601.77 | 0.1682 |
| GraNT (B) | **6392.26** | **0.0051** |

GraNT cuts training time in half and reduces MAE by two orders of magnitude.

**ogbg-molhiv Comparison with Efficient GNN Methods (Table 3)**:

| Method | Time (s) | ROC-AUC |
|:---|:---|:---|
| GCN | 2888.80 | 0.7385 |
| GDeR-PNA | 5088.88 | 0.7616 |
| GraNT (B) | **1457.39** | **0.7676** |
| GraNT (S) | 1597.69 | **0.7705** |

### Ablation Study
- **GraNT (B) vs GraNT (S)**: (B) is faster (saving the overhead of batch decomposition and reconstruction), while the two achieve comparable performance; (S) yields slightly higher ROC-AUC on some classification tasks.
- **Convergence Speed**: Validation loss/MAE curves show that GraNT spans roughly 2/3 of the baseline's duration, presenting a faster rate of descent.
- **Impact of Imbalanced Labels**: The ROC-AUC curves for ogbg-molhiv and gen-cls exhibit sawtooth shapes due to label imbalance, yet GraNT consistently outperforms the baseline.
- **Infinite Width vs. Finite Width**: The paper considers the dynamic GNTK under realistic finite-width GCNs, rather than ideal infinite-width assumptions.

## Highlights & Insights
1. **Outstanding Theoretical Contribution**: This is the first work to prove the consistency between parameter gradient descent and functional gradient descent in GCNs (Theorem 5), generalizing nonparametric teaching theory from regular data to graph-structured data.
2. **Extremely Simple and Efficient Selection Strategy**: The greedy strategy only requires comparing the magnitude of prediction deviations without calculating kernel or functional norms, incurring minimal extra overhead.
3. **Comprehensive Performance**: Successfully covers four scenarios (graph-level/node-level $\times$ regression/classification) across six datasets, stably delivering 30%–47% acceleration.
4. **Rigorous Theory**: Establishes a complete theoretical chain from structure-aware gradient analysis to GNTK convergence and then to the guarantee of sufficient loss reduction.
5. **Elegant Unification with MLP Nonparametric Teaching**: When the convolution order is $\kappa=1$, the model perfectly degenerates into MLP nonparametric teaching.

## Limitations & Future Work
1. **Only Evaluated on GCN**: The method has not been extended to other graph neural networks like GAT and GIN, which the authors acknowledge as future work in the conclusion.
2. **Sensitivity of Selection Ratio $m/N$**: The paper does not thoroughly analyze how the selection ratio influences the trade-off between acceleration and performance.
3. **Node-Level Experiments Restricted to Synthetic Data**: The gen-reg/gen-cls datasets are generated using graphon, lacking real-world large-scale node-level benchmarks (such as ogbn-*).
4. **Theoretical Reliance on Convex Loss Assumption**: Both Theorem 5 and Proposition 6 require convex losses. In practice, the convexity of losses like cross-entropy with respect to $f_\theta$ must be carefully handled.
5. **Convergence Rate of Dynamic GNTK**: Although the paper proves limit convergence, it does not provide the rate of convergence, leaving the approximation accuracy under finite training steps unknown.

## Related Work & Insights
- **Nonparametric Teaching**: Zhang et al. (2023b, 2024a) established nonparametric teaching theories in the MLP context. This work generalizes them to the graph domain.
- **Graph Neural Tangent Kernel (GNTK)**: Du et al. (2019) and Krishnagopal & Ruiz (2023) studied static GNTKs under infinite-width GCNs, whereas this work investigates dynamic GNTK under finite-width constraints.
- **Curriculum Learning / Active Learning**: The "hard sample selection" strategy of GraNT shares similarities with anti-curriculum learning (Bengio et al., 2009) and active learning, but this work provides a solid theoretical foundation based on functional gradients.
- **Graph Training Acceleration**: FastGCN (Chen et al., 2018), GraphNorm (Cai et al., 2021), and GDeR (Zhang et al., 2024b) achieve speedup through sampling, normalization, and distillation, respectively. This work approaches the problem from teaching theory, offering orthogonal and complementary perspectives.

## Rating
- Novelty: ⭐⭐⭐⭐ (First work to extend nonparametric teaching to graph property learning, with a solid theoretical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluated across four tasks on six datasets, but lacks real benchmarks for the node-level task)
- Writing Quality: ⭐⭐⭐⭐ (Complete theoretical derivations and clear motivation, though notation is slightly heavy)
- Value: ⭐⭐⭐⭐ (Achieves 30%–47% speedup under a solid theoretical framework, inspiring for graph learning training efficiency)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICML 2025\] GCAL: Adapting Graph Models to Evolving Domain Shifts](gcal_adapting_graph_models_to_evolving_domain_shifts.md)
- [\[ICLR 2026\] Compositional Generalization through Gradient Search in Nonparametric Latent Space](../../ICLR2026/optimization/compositional_generalization_through_gradient_search_in_nonparametric_latent_spa.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](../../ICLR2026/optimization/rapid_training_of_hamiltonian_graph_networks_using_random_features.md)

</div>

<!-- RELATED:END -->
