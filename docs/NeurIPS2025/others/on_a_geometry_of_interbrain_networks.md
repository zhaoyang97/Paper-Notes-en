---
title: >-
  [Paper Note] On a Geometry of Interbrain Networks
description: >-
  [NeurIPS 2025 (Symmetry and Geometry in Neural Representations Workshop)][Discrete curvature] This opinion piece proposes introducing discrete graph curvature (Forman-Ricci and Ollivier-Ricci curvature) into interbrain n…
tags:
  - "NeurIPS 2025 (Symmetry and Geometry in Neural Representations Workshop)"
  - "Discrete curvature"
  - "Forman-Ricci curvature"
  - "Ollivier-Ricci curvature"
  - "hyperscanning"
  - "interbrain synchrony"
date: 2026-05-08
content_hash: d6e721d4503a757c
---

# On a Geometry of Interbrain Networks

**Conference**: NeurIPS 2025 (Symmetry and Geometry in Neural Representations Workshop)
**arXiv**: [2509.10650](https://arxiv.org/abs/2509.10650)  
**Code**: None  
**Area**: Computational Neuroscience / Network Science
**Keywords**: Discrete curvature, Forman-Ricci curvature, Ollivier-Ricci curvature, hyperscanning, interbrain synchrony

## TL;DR

This opinion piece proposes introducing discrete graph curvature (Forman-Ricci and Ollivier-Ricci curvature) into interbrain network analysis within hyperscanning research. It leverages the entropy of curvature distributions to detect network phase transitions and uses curvature values to infer interbrain information routing strategies, moving beyond the descriptive limitations of conventional correlation-based metrics.

## Background & Motivation

**Background**: Hyperscanning simultaneously records neural signals (EEG, fNIRS, fMRI) from interacting individuals and constructs interbrain networks to investigate the neural underpinnings of social interaction. Mainstream analytical approaches rely on interbrain synchrony (IBS) metrics such as the phase-locking value (PLV) to quantify neural coupling between brain regions.

**Limitations of Prior Work**: Correlation-based metrics such as PLV are inherently descriptive—they indicate that synchrony exists between two brain regions but cannot reveal why the network reorganizes in a particular way, how information is routed through the network, or at which critical moments during social interaction structural transitions occur.

**Key Challenge**: Social interaction is a dynamic and complex process (cooperation, conflict, understanding, misunderstanding) involving rapid network reconfiguration. Purely correlational methods overlook the dynamic evolution of network topology, rendering them incapable of capturing these critical transition points or providing mechanistic explanations.

**Goal**: To propose a discrete-geometry-based analytical framework capable of (1) detecting phase transitions in interbrain networks and (2) inferring information routing strategies within those networks.

**Key Insight**: Discrete graph curvature has been established in geometric machine learning as a powerful tool for characterizing the structure and dynamics of complex networks. The authors propose applying Forman-Ricci and Ollivier-Ricci curvature to time-varying interbrain networks, using changes in curvature distributions to detect network reorganization events.

**Core Idea**: Use entropy divergence of discrete curvature distributions to detect interbrain network phase transitions, and use curvature values to infer information routing strategies—advancing from descriptive analysis toward mechanistic understanding.

## Method

### Overall Architecture

The paper proposes an analytical pipeline: (1) construct time-varying weighted interbrain graphs from hyperscanning data; (2) compute discrete curvature (FRC or ORC) for each edge; (3) detect network phase transitions by tracking divergence of the differential entropy $H_{RC}(G_t)$ of the curvature distribution over time; (4) infer information routing strategies from the spatial distribution of ORC values.

### Key Designs

1. **Forman-Ricci Curvature (FRC) for Topological Characterization**:

    - Function: Quantifies the local geometric properties of network edges—positive curvature indicates that an edge resides in a densely connected region, while negative curvature indicates that an edge bridges highly connected modules.
    - Mechanism: For edge $e$ connecting nodes $i$ and $j$, $F(e) = w_e(\frac{z_i}{w_e} + \frac{z_j}{w_e} - \sum_{e_i \sim i} \frac{z_i}{\sqrt{w_e w_{e_i}}} - \sum_{e_j \sim j} \frac{z_j}{\sqrt{w_e w_{e_j}}})$, where $z_i, z_j$ are node weights and $w_e$ is the edge weight. FRC identifies information bottlenecks and has been shown in GNNs to reveal over-squashing.
    - Design Motivation: FRC is computationally efficient and suitable for real-time analysis of large-scale interbrain networks.

2. **Ollivier-Ricci Curvature (ORC) for Information Routing Inference**:

    - Function: Infers the "strategy" of information flow along each edge via its curvature value.
    - Mechanism: ORC is defined via optimal transport—comparing the Wasserstein-1 distance between the probability distributions of neighborhoods at the two endpoints of an edge: $\kappa(u,v) = 1 - W_1(m_u, m_v)/d_G(u,v)$. Negatively curved edges tend to attract information flow (shortest-path routing), while positively curved edges promote diffusion.
    - Design Motivation: Information routing in brain networks lies on a continuum between shortest-path traversal and random diffusion; ORC provides a natural means to quantify this spectrum.

3. **Phase Transition Detection via Curvature Distribution Entropy**:

    - Function: Automatically identifies time points at which the network topology undergoes significant reorganization.
    - Mechanism: The differential entropy of the FRC distribution at time $t$, $H_{RC}(G_t) = -\int f_{RC}^t(x) \log[f_{RC}^t(x)] dx$, is computed and tracked over time. Sharp changes (divergences) in this entropy signal structural reorganization of the network topology.
    - Design Motivation: If behavioral transitions (e.g., from cooperation to conflict) coincide temporally with divergences in curvature entropy, one can more confidently attribute them to the neural mechanisms underlying social interaction.

### Loss & Training

This is an opinion/framework paper and involves no model training. A small-world network simulation is used to demonstrate the feasibility of the pipeline.

## Key Experimental Results

### Main Results

The paper uses simulations of small-world networks (Watts-Strogatz model) to illustrate the proposed method.

| Simulation Parameters | Setting | Observation |
|---------|------|---------|
| $N=1000$, $K=50$ | Rewiring probability $p$ varied from 0 to 1 | FRC distribution entropy exhibits a sharp jump near $p \approx 10^{-2}$ |
| $p < 10^{-3}$ | Regular lattice (low rewiring) | Narrow curvature distribution, low entropy, highly segregated network |
| $p > 10^{-1}$ | Near-random network | Broad curvature distribution, high entropy, network trending toward integration |

### Ablation Study

| Modality/Condition | Edge Weight Range | Spatial/Temporal Properties |
|---------|----------|-------------|
| EEG – task | PLV ≈ 0.2–0.6 | Fast; captures rapid behavioral dynamics |
| EEG – resting state | PLV ≈ 0.1–0.4 | Fast; spontaneous activity |
| fNIRS – task | Corr. ≈ 0.1–0.3 | 0.1–1s; suitable for slow tasks |
| fMRI – task | Cohe. ≈ 0.2–0.5 | 1–2s; restricted to block designs |

### Key Findings

- FRC distribution entropy exhibits a sharp transition at the small-world network phase transition point, demonstrating the method's effectiveness in detecting topological transitions.
- The applicability of the framework to different hyperscanning modalities (EEG, fNIRS, fMRI) depends on whether the spatiotemporal sampling rate can resolve the target behavior.
- FRC identifies information bottlenecks (bridging edges) while ORC infers information routing strategies (shortest-path vs. diffusion); the two measures are complementary.

## Highlights & Insights

- The conceptual shift **from description to mechanism** is noteworthy. Conventional IBS can only state that "two brain regions are synchronized," whereas the curvature-based approach can characterize structural reorganization events and infer switches in routing strategy—approaching a mechanistic explanation.
- The **complementarity of FRC and ORC** is methodologically elegant. FRC addresses topological structure (where are the bottlenecks?), while ORC addresses dynamics (how does information flow?); together they form a naturally unified analytical toolkit.
- The framework is directly transferable to other network dynamics settings, including real-time monitoring in brain-computer interfaces and social decision-making research on collective neural activity.

## Limitations & Future Work

- As an opinion piece, the work lacks validation on real hyperscanning data; only small-world network simulations are provided.
- The computational efficiency of curvature calculation—particularly ORC, which requires solving an optimal transport problem—on large-scale networks warrants evaluation.
- The paper assumes alignment between curvature entropy divergence and behavioral transitions, but this assumption has not been empirically validated.
- The impact of artifacts and noise common in hyperscanning data on curvature computation is not discussed.
- The choice of edge weight definition (PLV vs. correlation coefficient vs. coherence) affects curvature values, and no clear selection criterion is provided.

## Related Work & Insights

- **vs. conventional IBS analysis (Hakim et al. 2023)**: Traditional methods compute correlation-based metrics such as PLV between brain regions and are inherently descriptive. The proposed geometric approach provides network-level topological and dynamical information, serving as a complement rather than a replacement.
- **vs. graph curvature in GNNs (Topping et al. 2022)**: In deep learning, FRC is used to identify information bottlenecks (over-squashing) in message passing. The paper transfers the same intuition to brain networks—negatively curved regions may constitute bottlenecks for interbrain information transmission.
- **vs. curvature analysis of intrabrain networks (Chatterjee et al. 2021)**: Prior work has applied curvature to intrabrain networks; the contribution of this paper lies in extending the approach to interbrain networks in the context of social neuroscience.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing discrete curvature into hyperscanning analysis represents a novel cross-disciplinary synthesis.
- Experimental Thoroughness: ⭐⭐⭐ Limited to simulations without real-data validation; acceptable for a workshop paper.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are clearly articulated and the argument is coherent, making the paper an effective directional position piece.
- Value: ⭐⭐⭐⭐ A promising research direction is proposed, though real-data validation is needed to assess its practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry](johnson-lindenstrauss_lemma_beyond_euclidean_geometry.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[NeurIPS 2025\] The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet](the_geometry_of_cortical_computation_manifold_disentanglement_and_predictive_dyn.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)

</div>

<!-- RELATED:END -->
