---
title: >-
  [Paper Note] Sheaf Cohomology of Linear Predictive Coding Networks
description: >-
  [NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)][Predictive Coding] This paper formalizes linear predictive coding (PC) networks as cellular sheaves…
tags:
  - "NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)"
  - "Predictive Coding"
  - "Cellular Sheaf"
  - "Cohomology"
  - "Hodge Theory"
  - "Sheaf Laplacian"
date: 2026-05-08
content_hash: c0a5fe216d62adc5
---

# Sheaf Cohomology of Linear Predictive Coding Networks

**Conference**: NeurIPS 2025 (Workshop: Symmetry and Geometry in Neural Representations)  
**arXiv**: [2511.11092](https://arxiv.org/abs/2511.11092)  
**Code**: None  
**Area**: Theoretical Deep Learning / Algebraic Topology / Predictive Coding  
**Keywords**: Predictive Coding, Cellular Sheaf, Cohomology, Hodge Theory, Sheaf Laplacian

## TL;DR

This paper formalizes linear predictive coding (PC) networks as cellular sheaves, proves that PC inference is equivalent to diffusion under the sheaf Laplacian, and employs the Hodge decomposition to factorize supervisory signals into eliminable errors (removed via inference) and irreducible errors (characterized by the cohomology of cyclic topology). This framework precisely explains why certain cyclic weight initializations lead to learning stagnation.

## Background & Motivation

**Background**: Predictive coding (PC) is a biologically inspired neural network training paradigm that replaces global backpropagation with local prediction error minimization. A key advantage of PC is its ability to handle networks with arbitrary cyclic topologies without unrolling computation graphs or applying BPTT. While PC has attracted growing attention in the deep learning community, a systematic theoretical analysis of its behavior in recurrent networks remains lacking.

**Limitations of Prior Work**: PC in recurrent networks introduces a fundamental yet overlooked problem: deeply recurrent nodes receive error signals from all their connections—some originating from supervision, others from contradictory feedback loops—yet the nodes cannot distinguish between the two. As a result, a node may spend a large portion of its "inference budget" reconciling internal contradictions rather than learning useful representations. Existing improvements to PC (e.g., depth scaling) primarily target feedforward architectures and do not address this structural problem arising from cyclic topology.

**Key Challenge**: The learning capacity of a recurrent PC network depends on the weight initialization of its feedback loops, yet no theoretical tools currently exist to predict which initializations cause learning failure. This is neither a question of weight magnitude (all initializations are orthonormal) nor a simple conditioning issue—it depends on how the weights are "knotted" around the cycles.

**Goal**: To provide a mathematical framework that (1) precisely characterizes which error patterns in a PC network can be eliminated by inference and which cannot; (2) explains how internal contradictions in cyclic topology affect learning; and (3) derives principled weight initialization guidelines for recurrent PC networks.

**Key Insight**: The central question in cellular sheaf theory—"when does local consistency glue into global consistency?"—maps perfectly onto the core pursuit of PC: whether locally informed layers can collectively resolve a global task.

**Core Idea**: Interpret PC networks as cellular sheaves over their computation graphs, and unify inference-irreducible errors, the global geometry of weight initialization, and learning dynamics within a single algebraic-topological framework via sheaf cohomology.

## Method

### Overall Architecture

The central correspondence is: PC network ↔ cellular sheaf. Activations (nodes) → 0-cochains; prediction errors (edges) → 1-cochains; the coboundary operator $\delta^0$ computes all prediction errors simultaneously; the PC energy $E_{PC} = \frac{1}{2}\|\delta^0 s\|^2$; and PC inference (energy minimization) = diffusion under the sheaf Laplacian $L = (\delta^0)^\top \delta^0$.

### Key Designs

1. **Predictive Coding Sheaf**:

    - **Function**: Formalizes the computational structure of a PC network as an algebraic-topological object.
    - **Mechanism**: For a graph $G = (V, E)$, each vertex $v$ is assigned a vector space $\mathcal{F}(v) = \mathbb{R}^{n_v}$ (neural activations), and each edge $e = (u \to v)$ is assigned a vector space $\mathcal{F}(e) = \mathbb{R}^{m_e}$ (prediction errors). The restriction maps are $\rho_{e \leftarrow u} = W_e$ (weight matrix) and $\rho_{e \leftarrow v} = I$ (identity). The coboundary operator $\delta^0$ maps activations to prediction errors: $(\delta^0 s)_e = s_v - W_e s_u$, encoding the entire network's forward computation as a single linear operator.
    - **Design Motivation**: The language of sheaves enables the use of mature mathematical tools—cohomology, Hodge decomposition—to analyze the global properties of PC networks, moving beyond purely local, layer-by-layer analysis.

2. **Relative Systems and Hodge Decomposition**:

    - **Function**: Precisely characterizes how supervisory signals are distributed across the network after clamping data.
    - **Mechanism**: Vertices are partitioned into free vertices (hidden layers) and clamped vertices (inputs $x$, targets $y$). The relative coboundary $D$ extracts the columns of $\delta^0$ corresponding to free vertices; the clamping effect produces a "target prediction error" vector $b$. The Hodge decomposition orthogonally decomposes $b$ as $b = (-Dz^*) + r^*$, where $-Dz^*$ is the component eliminable by inference (lying in $\text{im}\, D$), and $r^* = \mathcal{H}b$ is the irreducible error that inference cannot remove (lying in $\ker D^\top$, i.e., the relative cohomology $H^1_{\mathrm{rel}}$).
    - **Design Motivation**: This reveals the essence of PC inference as a discrete Dirichlet problem—harmonically extending boundary values (supervisory targets) into the interior—where $H^1_{\mathrm{rel}}$ precisely measures the obstruction to such extension.

3. **Harmonic–Diffusion Separation and Learning Conditions**:

    - **Function**: Precisely determines which edges in the network can learn and which will stagnate.
    - **Mechanism**: Define the harmonic projection $\mathcal{H} = I - DD^\dagger$ and the diffusion operator $\mathcal{G} = D^\dagger$. At the optimal inference point, the weight gradient for edge $e = (u \to v)$ is $\frac{\partial E}{\partial W_e} = (\mathcal{H}b)_e \cdot (\mathcal{G}b)_u^\top$—the outer product of the "harmonic component" (residual on the edge) and the "diffusion component" (activation at the source node). **Learning requires both to be simultaneously nonzero**: if the harmonic load on an edge is zero (no residual) or the diffusion activation at the source node is zero (no signal reaches it), the edge weight will not be updated.
    - **Design Motivation**: This yields a precise diagnostic for learning failure in PC: inspect whether the harmonic load and diffusion activation spatially overlap. If the feedback loops concentrate the harmonic load on edges unreachable by diffusion, learning stagnates.

### Monodromy Experiment

The paper constructs a 10-layer linear network with feedback connections at each layer. Forward weights $W_i$ are randomly orthogonally initialized, and feedback weights $W_i^{FB}$ are controlled by a single monodromy parameter $\theta$: $\Phi_i(\theta) = W_i^{FB} W_i$ produces a rotation by angle $\theta$. $\theta = 0$ corresponds to "resonance" ($W_i^{FB} = W_i^{-1}$, feedback reinforces itself), while $\theta = \pi$ corresponds to "internal tension" (feedback negates itself).

## Key Experimental Results

### Main Results: 10-Node Network, Identity Mapping Task

| Initialization angle $\theta$ | Validation MSE after 1000 steps | Harmonic load distribution | Diffusion activation distribution |
|-------------------------------|----------------------------------|----------------------------|------------------------------------|
| $\theta = 0$ (resonance) | $\leq 0.001$ | Uniform across all edges | Uniform across all nodes |
| $\theta = 0.33$ (intermediate) | $\leq 0.001$ (slower) | Gradually "unknots" during training | Progressively extends to all nodes |
| $\theta > 0.4$ | $\gg 0.001$ (stagnation) | Concentrated on interior edges | Nodes such as $h_2, h_9$ are "starved" |
| $\theta = \pi$ (contradiction) | Extremely slow convergence | Concentrated on edges unreachable by diffusion | Diffusion from $h_1, h_{10}$ is blocked |

### Ablation Study

| Variant | Effect | Remarks |
|---------|--------|---------|
| Different learning rates $\eta$ | Larger $\eta$ allows convergence at larger $\theta$ | The influence of $\theta$ still spans orders of magnitude |
| Orthogonal vs. non-orthogonal initialization | Orthogonal initialization isolates global wiring effects | All layer weight magnitudes are identical; differences are purely geometric |
| Fully connected PC network (appendix) | Similar patterns observed | Conclusions hold across different topological structures |

### Key Findings

- **Learnability is determined by the global wiring pattern, not weight magnitude**: All initializations are orthonormal (identical norms), yet convergence speed differs by orders of magnitude. The key factor is the cumulative effect (monodromy) of feedback weights around the cycle.
- **Precise distinction between resonance and tension**: At $\theta = 0$ (resonance), harmonic load and diffusion activation fully overlap spatially, and all edges can learn. At $\theta = \pi$ (tension), the two are spatially separated—harmonic load concentrates where diffusion cannot reach, causing a "signal open circuit."
- **"Unknotting" dynamics**: Training at an intermediate angle $\theta = 0.33$ demonstrates how the network gradually "unknots"—as weights update, the harmonic load transitions from concentrated to uniform, reflecting the progressive resolution of internal contradictions.

## Highlights & Insights

- **Elegance of the theoretical framework**: The paper unifies multiple PC concepts—inference, learning, and error propagation—within a single algebraic-topological framework. Sheaf cohomology captures irreducible errors, the Hodge decomposition governs signal routing, and the diffusion–harmonic overlap determines learnability; the unification of these three elements is remarkably clean.
- **Implications of the local-to-global perspective**: The central concern of sheaf theory is precisely "when local information can be glued into global consistency." This has deep connections to consistency problems in distributed and federated learning. The proposed framework may generalize beyond PC to provide analytical tools for any learning paradigm based on local optimization.
- **Practical value as a diagnostic tool**: The paper provides a concrete procedure for diagnosing learning problems in recurrent networks—compute $\mathcal{H}$ and $\mathcal{G}$, inspect their spatial overlap, and predict which network configurations will fail. This is considerably more efficient than empirically testing whether a network trains.

## Limitations & Future Work

- **Linear network assumption**: All analyses are restricted to linear networks. In the nonlinear setting, sheaf cohomology and the Hodge decomposition do not directly apply. The paper mentions using the Jacobian as a local linear approximation, but this has not been rigorously validated.
- **Scope limitations of a workshop paper**: Many interesting directions—nonlinear extensions, precision weighting, hypergraph sheaves—are briefly mentioned but not developed.
- **Extremely small experimental scale**: Validation is performed only on 10-layer networks with 2×2 dimensions, far from the scale of practical deep learning. It remains to be shown whether the framework's predictions hold at larger scales.
- **Edge-wise PC energy assumption**: The paper adopts an edge-wise decomposition of PC energy, where each edge independently contributes an error term. Classical PC literature more commonly employs neuron-wise (aggregated) formulations. The two may behave differently in the nonlinear regime; the paper acknowledges this but does not pursue a detailed comparison.
- **Lack of a bridge to standard deep learning**: The practical utility of the framework depends on whether it can yield new insights for non-PC networks trained with standard backpropagation. The current analysis is entirely confined to the PC paradigm.

## Related Work & Insights

- **vs. Sheaf Neural Networks (Hansen et al. 2020, Bodnar et al. 2022)**: Sheaf neural networks use sheaf diffusion as a message-passing primitive to enhance GNNs, operating on *data graphs* (social networks, molecules, etc.) and leveraging the sheaf Laplacian to mitigate oversmoothing and heterophily. The present work applies sheaf theory to the *computation graph* itself, analyzing network architecture rather than data structure. The two directions may intersect: applying PC networks to graph-structured data would require simultaneously reasoning about both the data sheaf and the computational sheaf.
- **vs. Deep Linear Network Theory (Saxe et al. 2013)**: The learning dynamics of deep linear networks have been extensively studied. This paper provides a new analytical perspective for recurrent deep linear networks—the critical quantity is not the singular values of the weight matrices (as in Saxe et al.) but the cumulative effect (monodromy) of the weights around the cycle.
- **vs. Studies on PC Learning Difficulties (Qi et al. 2025, Innocenti et al. 2024)**: These works address PC learning through techniques such as depth scaling. The contribution of the present paper is a *diagnostic framework* for understanding the root cause of such difficulties, rather than a direct remedy. The two approaches are complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing algebraic-topological tools into PC analysis is a genuinely novel perspective; the framework is elegant and insightful.
- **Experimental Thoroughness**: ⭐⭐⭐ — Appropriate for a workshop paper, but experiments are limited to extremely small 2D toy networks.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical exposition is precise and intuitive explanations are clear, though the entry barrier is high for readers unfamiliar with sheaf theory.
- **Value**: ⭐⭐⭐⭐ — Significant value for the PC theory community; the framework's potential applicability extends well beyond PC to any learning paradigm grounded in local-to-global consistency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet](the_geometry_of_cortical_computation_manifold_disentanglement_and_predictive_dyn.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] On a Geometry of Interbrain Networks](on_a_geometry_of_interbrain_networks.md)

</div>

<!-- RELATED:END -->
