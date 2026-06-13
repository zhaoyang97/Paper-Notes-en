---
title: >-
  [Paper Note] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis
description: >-
  [NeurIPS 2025][Model Compression][Neural Network Pruning] This paper proposes the "Graphon Limit Hypothesis": as network width tends to infinity, the binary mask sequences produced by different pruning methods converge…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Neural Network Pruning"
  - "Graphon"
  - "Graph Limit Theory"
  - "Neural Tangent Kernel"
  - "Sparse Networks"
  - "Infinite Width"
date: 2026-05-08
content_hash: 240e7f8eb4325a26
---

# The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.17515](https://arxiv.org/abs/2510.17515)  
**Code**: To be confirmed  
**Area**: Model Compression
**Keywords**: Neural Network Pruning, Graphon, Graph Limit Theory, Neural Tangent Kernel, Sparse Networks, Infinite Width

## TL;DR

This paper proposes the "Graphon Limit Hypothesis": as network width tends to infinity, the binary mask sequences produced by different pruning methods converge, under the cut distance, to their respective unique graphon limits. Building on this foundation, the paper derives a Graphon NTK to analyze the training dynamics of sparse networks, providing a theoretical explanation for why different pruning methods yield markedly different performance at the same sparsity level.

## Background & Motivation

Neural network pruning is a core technique in model compression, yet a systematic theoretical explanation for the fundamental question—"why are certain sparse structures easier to train than others at the same sparsity?"—remains absent. This paper introduces a novel analytical framework grounded in graph limit theory:

1. **Weak theoretical foundations for pruning**: The Lottery Ticket Hypothesis (LTH) establishes the existence of effective sparse subnetworks but fails to explain why different pruning strategies lead to drastically different training outcomes—two networks at 90% sparsity can differ by several percentage points in performance, and existing theory offers no unified explanation.
2. **Inapplicability of the NTK framework to sparse settings**: Standard NTK theory assumes fully connected architectures. After pruning, the non-uniform connectivity patterns introduced by masks preclude a direct definition of the NTK for sparse networks in the infinite-width limit—a fundamental mathematical obstacle.
3. **Lack of a unified language for cross-method comparison**: Methods such as Random Pruning, SNIP, GraSP, and Synflow each embody distinct design philosophies, yet no mathematical framework exists to quantify their structural differences—a universal tool analogous to frequency-domain analysis is needed.
4. **Missing bridge from discrete to continuous**: Graph-theoretic tools (e.g., Ramanujan graphs) have been applied to analyze connectivity in sparse networks, but finite-graph analysis methods do not readily generalize to arbitrary widths; a continuous framework capable of handling "limit behavior" is required.
5. **Structural origins of training dynamics unknown**: It is empirically known that SNIP/Synflow converge faster than Random Pruning in early training, but whether this difference stems from weight selection, gradient flow, or network topology remains unclear—theoretical tools to isolate the contribution of topology to training speed are lacking.
6. **Theoretical gap for infinitely wide sparse networks**: Yang et al. (2023) derived the NTK limit only for Random Pruning (a special case of the constant graphon), while the infinite-width behavior of more general structured pruning methods remains entirely unknown.

## Method

### Core Idea

Pruning masks are interpreted as adjacency matrices of bipartite graphs. As network width increases, these graph sequences converge under the cut distance to graphons (graph limit objects). Different pruning methods correspond to different graphons, and the structural prior encoded in each graphon determines the training dynamics of the resulting sparse network.

### Graphon Limit Hypothesis

**Hypothesis statement**: Given a fixed architecture class $\mathcal{A}$, sparsity level $p > 0$, and pruning method $\mathcal{P}$, as width tends to infinity, the pruning masks $(M_n^{(l)})$ converge layer-wise to deterministic graphons $\mathcal{W}^{(l)}$:

$$\lim_{n \to \infty} \delta_\square(\mathcal{W}_{M_n^{(l)}}, \mathcal{W}^{(l)}) = 0$$

where $\delta_\square$ denotes the cut distance. Different pruning methods converge to distinct graphon patterns:
- **Random Pruning** → constant graphon (Erdős-Rényi random graph), with uniform connection probability across all positions
- **SNIP/GraSP** → non-uniform graphon, with connection density forming a gradient distribution around high-centrality nodes
- **Synflow** → block graphon, exhibiting sharp connected/disconnected transition boundaries

### Empirical Validation of the Hypothesis

On 4-layer and 5-layer MLPs, 100 independent experiments are conducted for each of four PaI methods (Random/SNIP/GraSP/Synflow) across widths $n \in \{100, 500, 1000, 2000\}$ and sparsity levels $\{70\%, 80\%, 90\%\}$. Graphons are estimated using the SAS method (sorting by degree centrality → grid partitioning → density computation). Results show: (1) the graphon pattern of each method converges monotonically as width increases; (2) different methods converge to visibly distinct structural patterns.

### Graphon NTK Derivation

In graphon-structured networks, the variance of weights is modulated by the graphon: $W_{ij}^{(l)} \sim \mathcal{N}(0, \mathcal{W}^{(l)}(i/n_l, j/n_{l-1}))$. Via the CLT (with Lindeberg-Feller conditions ensuring validity in the non-i.i.d. setting), pre-activations converge to a position-dependent Gaussian process. The Graphon NTK is ultimately shown to converge in the infinite-width limit to a deterministic kernel:

$$\Theta(x, x') = \sum_{l=1}^{L} \int_0^1 \left( \dot{\Sigma}^{(l)} \int_{[0,1]^{L-l+1}} \prod_{m=l+1}^{L+1} \mathcal{W}^{(m)} \dot{\Sigma}^{(m)} d\mathbf{u}_{l+1} \right) d u_l$$

**Key distinction from the standard NTK**: In the standard NTK, pre-activation covariance equals the activation covariance of the preceding layer directly. In the Graphon NTK, the graphon function $\mathcal{W}^{(l)}$ at each layer acts as a weighting factor modulating covariance propagation, resulting in non-uniform signal transmission through the network.

### Special Case: Random Pruning

When $\mathcal{W}(u,v) = c$ (constant graphon), $\Theta(x,x') = c^L \Theta_{\text{std}}(x,x')$, i.e., a scaled version of the standard NTK. This implies that Random Pruning uniformly scales all eigenvalues as $\lambda_k \to c^L \lambda_k$, leaving the relative training dynamics unchanged while reducing the overall learning rate—theoretically explaining the empirical observation that randomly sparse networks converge slowly but exhibit unchanged training patterns.

## Key Experimental Results

| Pruning Method | Sparsity | Eigenvalue Decay Rate α | Effective Rank | Spectral Gap λ₁/λ₂ | Top-5 Energy Concentration |
|----------------|----------|------------------------|----------------|---------------------|---------------------------|
| Random | 80% | Stable | Highest | Lowest | Lowest (diffuse spectrum) |
| SNIP | 80% | Increases with sparsity | Medium | Medium | Medium |
| Synflow | 80% | Largest | Lowest | Highest | Highest (>80% in top-5 eigenvalues) |

| Experimental Setting | Observation |
|----------------------|-------------|
| 4-layer MLP, n=1024, MNIST, sparsity 50%–95% | SNIP/Synflow exhibit significantly faster loss decrease than Random within the first 200 training steps |
| Graphon NTK spectral analysis vs. actual training dynamics | Top-5 energy concentration is positively correlated with early-stage convergence speed |
| Graphon convergence, width 100→2000 | Euclidean distance of density matrices decreases monotonically for all methods |
| NTK scaling under Random Pruning | Experimentally validates $\Theta = c^L \Theta_{\text{std}}$ |

## Highlights & Insights

- **A genuinely novel perspective via graph limit theory**: Connecting pruning mask sequences to graphon limits provides, for the first time in pruning theory, a mathematical bridge from "finite graphs" to "continuous limits" for analyzing sparse networks.
- **Generality of the Graphon NTK framework**: Beyond covering Random Pruning (a previously known result), the framework extends to structured pruning methods such as SNIP, GraSP, and Synflow—generalizing from a special case to a unified framework with theoretical coherence.
- **Empirical correspondence between spectral analysis and training dynamics**: Methods with higher top-5 energy concentration in the Graphon NTK (Synflow/SNIP) converge faster early in training, providing a theoretically grounded indicator for predicting pruning quality without training.
- **Effective integration of intuition and mathematics**: The graphon visualizations (Fig. 1) are highly intuitive—a single figure conveys that Random Pruning yields uniform gray, Synflow yields sharp black-and-white blocks, and SNIP yields a gradient, communicating structural differences more clearly than any numerical metric.
- **Guidance for pruning algorithm design**: The theory formally indicates that "maintaining favorable spectral properties of the Graphon NTK" should be a design objective for pruning methods, consistent with the empirical findings of works such as NTK-SAP.

## Limitations & Future Work

- **The Graphon Limit Hypothesis lacks a formal proof**: The core hypothesis is supported only empirically (finite-width experiments); a rigorous mathematical proof of cut distance convergence is absent—the most significant theoretical gap.
- **Only PaI methods are analyzed**: The framework is restricted to pruning at initialization (static masks) and does not cover dynamic sparse training (DST) or post-training pruning, partially limiting its practical scope.
- **Experimental validation limited to MLP + MNIST**: Validation on modern architectures (CNN/Transformer/ResNet) and more complex datasets (CIFAR-10/ImageNet) is absent; the generalizability of the framework to non-MLP architectures remains unclear.
- **Spectral properties and training dynamics are only correlated**: The paper demonstrates a correspondence between Graphon NTK spectra and convergence speed, but does not establish a causal theoretical guarantee (e.g., rigorous bounds on convergence rates).
- **Graphon estimation method affects results**: The SAS method is used to estimate graphons; alternative estimation approaches (e.g., GWB/IGNR) may affect visualizations and metrics, but the sensitivity to estimation method is not discussed.

## Related Work & Insights

Compared to **Yang et al. (2023)**: the closest predecessor, which analyzed the NTK limit of Random Pruning only (corresponding to the constant graphon $\mathcal{W}(u,v)=c$). The present work extends this to arbitrary graphons covering all PaI methods—a qualitative leap.

Compared to the **Ramanujan graph perspective** (Vooturi et al. 2023, Hoang et al. 2023): the Ramanujan graph approach focuses on connectivity indicators (spectral gap) of finite graphs. The graphon framework continuously analyzes these spectral properties in the infinite-width limit, constituting a higher-level abstraction.

Compared to **gradient flow analysis** (Lubana et al., Evci et al.): gradient flow analysis examines how gradients traverse sparse networks during training—a dynamic perspective. The Graphon NTK provides a static structural analysis at initialization. The two are complementary: the topology encoded in the graphon determines the initial conditions for gradient flow.

Compared to **Graphon Neural Networks** (Ruiz et al. 2020): graphon analysis in the GNN literature studies the limit behavior of input graph data, whereas this work studies the limit behavior of the network weight graph—the same mathematical tool applied to an entirely different object.

**Inspirations and connections**:
- **Graphon-guided pruning algorithm design**: Since different graphons correspond to different training dynamics, the process can be inverted—designing a target graphon with favorable spectral properties first and then sampling pruning masks from it, closing the loop from theory to algorithm.
- **Dynamic graphon evolution**: Modeling iterative pruning/dynamic sparse training as gradient flow of graphons in some function space may reveal optimal paths for dynamic sparsification.
- **Cross-architecture generalization**: Weight sharing/attention structures in CNNs/Transformers may correspond to more complex graphon families (e.g., periodic graphons for CNNs, sparse attention graphons for Transformers), warranting further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of graph limit theory and neural network pruning is an entirely new perspective; the Graphon NTK concept is pioneering in sparse network theory.
- Experimental Thoroughness: ⭐⭐⭐ — Limited to MLP + MNIST; lacks validation on modern architectures and at scale; the core hypothesis is not rigorously proved.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous, and the logic from hypothesis to derivation to validation is clear, though the formula density is high.
- Value: ⭐⭐⭐⭐ — Opens a new analytical paradigm for sparse network theory; long-term impact may well exceed the scope of the current experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)
- [\[NeurIPS 2025\] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[NeurIPS 2025\] Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency](mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)

</div>

<!-- RELATED:END -->
