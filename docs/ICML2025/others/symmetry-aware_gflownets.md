---
title: >-
  [Paper Note] Symmetry-Aware GFlowNets
description: >-
  [ICML 2025][GFlowNets] Uncovers systematic sampling bias in GFlowNets for graph generation caused by equivalent actions (different actions yielding isomorphic graphs), where node-by-node generation favors low-symmetry graphs and fragment-based generation favors highly symmetric components. This work proposes SA-GFN, a simple correction method that scales rewards by the size of the final state's automorphism group, achieving unbiased sampling with only a single automorphism gr…
tags:
  - "ICML 2025"
  - "GFlowNets"
  - "graph symmetry"
  - "automorphism group"
  - "equivalent actions"
  - "molecule generation"
date: 2026-05-08
content_hash: b944b7ab00827967
---

# Symmetry-Aware GFlowNets

**Conference**: ICML 2025  
**arXiv**: [2506.02685](https://arxiv.org/abs/2506.02685)  
**Code**: [https://github.com/hohyun312/sagfn](https://github.com/hohyun312/sagfn)  
**Area**: Other  
**Keywords**: GFlowNets, graph symmetry, automorphism group, equivalent actions, molecule generation

## TL;DR

Uncovers systematic sampling bias in GFlowNets for graph generation caused by equivalent actions (different actions yielding isomorphic graphs), where node-by-node generation favors low-symmetry graphs and fragment-based generation favors highly symmetric components. This work proposes SA-GFN, a simple correction method that scales rewards by the size of the final state's automorphism group, achieving unbiased sampling with only a single automorphism group computation.

## Background & Motivation

**Background**: GFlowNets are generative frameworks trained to sample combinatorial objects (such as molecular graphs) with probabilities proportional to a given reward. They construct graphs through a sequence of actions, and their training objectives (TB/DB) rely on the precise calculation of state transition probabilities.

**Limitations of Prior Work**: Graph construction processes suffer from the "equivalent actions problem": different actions can produce isomorphic graphs (e.g., in $G_1$, nodes 2 and 3 are equivalent, and adding a new node to either results in isomorphic graphs $G_2 \cong G_3$). In this case, transition probabilities must sum up the likelihoods of all equivalent actions. However, detecting equivalent actions requires computationally expensive graph isomorphism testing, which existing methods either ignore (introducing bias) or approximate inexactly.

**Key Challenge**: The bias introduced by ignoring equivalent actions is **systemic**: under the node-by-node generation scheme, the model systematically under-samples highly symmetric graphs by a factor of $1/|\text{Aut}(G)|$, biasing towards low-symmetry graphs. Conversely, in the fragment-based assembly scheme, highly symmetric fragments are over-sampled because they offer more equivalent forward actions. Over 50% of molecules in ZINC250k have more than one symmetry, posing severe impacts on molecule discovery.

**Goal**: Identify an exact, efficient, and easy-to-implement method to eliminate the sampling bias caused by graph symmetry in GFlowNets.

**Key Insight**: Observe that the ratio of equivalent actions can be formulated via the ratio of automorphism group sizes (Lemma 4.5). Consequently, the cumulative corrections along a trajectory telescope, simplifying to a single correction that depends solely on the size of the final state's automorphism group.

**Core Idea**: Eliminate equivalent action bias by multiplying the reward by the size of the final automorphism group $|\text{Aut}(G_n)|$, without requiring step-by-step detection.

## Method

### Overall Architecture

The core modification of SA-GFN is extremely concise: in existing GFlowNet pipelines, the reward function $R(G)$ is replaced with $\tilde{R}(G) = |\text{Aut}(G)| \cdot R(G)$ (for node-by-node generation) or $\tilde{R}(G) = \frac{|\text{Aut}(G)|}{\prod_i |\text{Aut}(C_i)|} \cdot R(G)$ (for fragment-based generation). This requires no modifications to the network architecture, training pipeline, or inference process, and can be implemented in only 17 lines of code.

### Key Designs

1. **Theoretical Characterization of Equivalent Actions (Theorems 4.3, 4.4)**:

    - Function: Establishes the relationship between orbit-equivalence and transition-equivalence, proving that orbit-equivalence is sufficient for GFlowNets.
    - Mechanism: Defines orbit-equivalence: actions $(G_1, t_1, u_1)$ and $(G_2, t_2, u_2)$ are orbit-equivalent if and only if $t_1=t_2$ and there exists an automorphism $\pi$ such that $\pi(G_1)=G_2$ and $\pi(u_1)=u_2$. **Theorem 4.3** proves that orbit-equivalence $\Rightarrow$ transition-equivalence (actions in the same orbit produce isomorphic graphs). **Theorem 4.4** proves that satisfying the flow constraints for orbit-equivalence implies satisfying them for transition-equivalence, meaning substituting transition-equivalence with orbit-equivalence is sufficient for GFlowNets.
    - Design Motivation: Transition-equivalence requires step-by-step graph isomorphism testing ($O(K \times H)$ times), whereas orbit-equivalence can be directly computed from the graph structure, laying the foundation for subsequent simplifications.

2. **Automorphism Group Correction Theorem (Theorem 4.6 + Lemma 4.5)**:

    - Function: Simplifies step-by-step orbit counting into the ratio of automorphism group sizes.
    - Mechanism: **Lemma 4.5** proves $\frac{|\text{Orb}(G,u,v)|}{|\text{Orb}(G',u,v)|} = \frac{|\text{Aut}(G)|}{|\text{Aut}(G')|}$ (for AddEdge; AddNode is similar). **Theorem 4.6** decomposes the ratio of forward and backward policies as: $\frac{p_{\bar{\mathcal{A}}}(a|s)}{q_{\bar{\mathcal{A}}}(a|s')} = \frac{|\text{Aut}(G)|}{|\text{Aut}(G')|} \cdot \frac{p_\mathcal{E}(G'|G)}{q_\mathcal{E}(G|G')}$. Telescoping this product along a complete trajectory cancels out the intermediate terms, leaving only $|\text{Aut}(G_0)|/|\text{Aut}(G_n)| = 1/|\text{Aut}(G_n)|$ (since the initial state's automorphism group size is 1).
    - Design Motivation: This mathematical simplification reduces the $O(K \times H)$ complexity of "step-by-step equivalent action detection" to $O(1)$ (computing the automorphism group only once for the final state) exactly and without approximations.

3. **Reward Scaling Correction (Corollary 5.1 + Theorem 5.2)**:

    - Function: Translates the theoretical correction into a minimal implementation via reward modification.
    - Mechanism: **TB Correction** (Corollary 5.1): Replaces $R(G_n)$ in the TB loss with $|\text{Aut}(G_n)| R(G_n)$. Without correction, the model samples according to $p \propto R(G)/|\text{Aut}(G)|$, systematically under-sampling high-symmetry graphs. **DB Correction** (Theorem 5.2): Defines a graph-level flow function $\tilde{F}$, also corrected by scaling the reward. **Fragment Correction** (Theorem 5.3): $\tilde{R}(G) = |\text{Aut}(G)| R(G) / \prod_{i=1}^k |\text{Aut}(C_i)|$. High-symmetry fragments are over-sampled due to providing more equivalent forward actions, and are thus penalized by dividing by the automorphism group sizes of individual fragments.
    - Design Motivation: The reward scaling approach is compatible with both TB and DB objectives, is extremely simple to implement, and the automorphism groups can be computed efficiently using the bliss algorithm.

### Loss & Training

The corrected TB loss is formulated as $\mathcal{L}_{\text{TB}}(\tau) = (\log \frac{Z \prod p_\mathcal{E}(G_{t+1}|G_t)}{|\text{Aut}(G_n)| R(G_n) \prod q_\mathcal{E}(G_t|G_{t+1})})^2$. For DB objectives, one can either multiply by the symmetry ratio at each step (Flow Scaling) or equivalently scale only the final reward (Reward Scaling). The paper also introduces an unbiased likelihood estimator: $\bar{p}_\mathcal{A}(x) \approx \frac{1}{M|\text{Aut}(G_n)|} \sum_{i=1}^M \frac{p_\mathcal{E}(\tau_i)}{q_\mathcal{E}(\tau_i|G_n)}$.

## Key Experimental Results

### Synthetic Graph Experiments (72,296 final states, TB objective)

| Method | L1 Error ↓ | Description |
|------|----------|------|
| Vanilla | High (does not converge) | Does not handle equivalent actions |
| PE (Ma et al. 2024) | Moderate | Approximate orbit detection, computed at every step |
| Transition Correction | Extremely Low | Exact graph isomorphism testing, cost is $K \times H$ times higher |
| **Reward Scaling** | **Extremely Low** | Single final-state computation, equivalent to the exact method |

### Molecule Generation Experiments

| Generation Method | Method | Diversity ↑ | Top-K div ↑ | Top-K reward ↑ | Unique Frac |
|---------|------|------------|-------------|----------------|-------------|
| Atom | Vanilla | 0.929 | 0.077 | 1.09 | 0.93 |
| Atom | **Reward Scaling** | **0.959** | **0.046** | **1.091** | **1.0** |
| Fragment | Vanilla | 0.877 | 0.153 | 0.941 | 1.0 |
| Fragment | **Reward Scaling (Exact)** | **0.879** | 0.151 | **0.952** | 1.0 |

### Key Findings

- Illustrative experiments (6-node connected graphs, 112 final states, uniform reward) perfectly validate the theory: the final state probabilities of the Vanilla model cluster according to the isomorphism class size $|x|$, whereas they become uniform after Reward Scaling.
- Vanilla strictly over-favors high-symmetry components in fragment-based generation: cyclohexane ($|\text{Aut}|=12$) was generated 5,220 times vs. only 1,042 times after correction.
- Under the DB objective, Reward Scaling converges slightly slower than Flow Scaling because the correction signal is provided only at the end of the trajectory (analogous to sparse reward vs. dense reward), but both reach the same final accuracy.
- The likelihood estimator provides accurate estimates with only a small number of samples ($M \sim 10$); trained models estimate more easily than random models.

## Highlights & Insights

- **Accurate Problem Identification**: Equivalent action bias is not only present but systemic: node-by-node generation under-samples highly symmetric graphs, while fragment-based generation over-samples highly symmetric components. These two opposite biases were not clearly characterized before.
- **Simple and Elegant Solution**: Simplifies complex "step-by-step equivalent action detection" into "a single final-state automorphism group calculation + reward scaling". This theoretical derivation is clean and logical (Lemma 4.5 → Theorem 4.6 → telescope → Corollary 5.1), requiring only about 17 lines of code to implement.
- **Broad Applicability**: Works for both TB and DB objectives, as well as both node-by-node and fragment-based generation schemes. Highly generalizable, and recommended for all future GFlowNet graph generation studies.

## Limitations & Future Work

- Theoretical guarantees rely on the specific structure of pre-defined graph action sets (AddNode/AddEdge/AddFragment). New action types must be validated separately.
- The limited expressive power of GNNs might cause nodes in different orbits to obtain identical representations, hindering the policy's ability to distinguish non-equivalent actions (this is a GNN expressiveness limitation rather than a limitation of the proposed method).
- Primarily validated on molecular generation; performance on other types of graph generation (e.g., social networks, program graphs) remains to be tested.
- Correction performance is limited when the reward is negatively correlated with symmetry (e.g., in the atom-based task on the QM9 dataset, where rewards are negatively correlated with symmetry).

## Related Work & Insights

- **vs. Ma et al. (2024)**: Employs positional encodings to approximate equivalent action detection, which is inexact and requires computation at every step. SA-GFN is exact and requires only a single computation at the final state, speeding up the process by a factor of $K \times H$.
- **vs. Chen et al. (2021)**: Discusses the relationship between orbits and equivalent transitions in autoregressive graph generation, which inspired this work. However, no concrete correction scheme was provided for GFlowNets.
- **vs. MaxEnt RL (Tiapkin et al. 2024)**: GFlowNets have been shown to be equivalent to MaxEnt RL, but remain biased when equivalent actions are not handled. The correction proposed in SA-GFN aligns the two frameworks in graph DAG environments.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Identifies systematic symmetry biases in GFlowNets for graph generation; the proposed solution is theoretically elegant and simple to implement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic graphs perfectly validate the theory, and molecular tasks demonstrate practical value, although the domain coverage is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem definition to theoretical derivation and experimental validation is highly polished and clear.
- Value: ⭐⭐⭐⭐ An essential correction for all GFlowNet graph generation works, recommended as the default configuration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/others/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[ICML 2025\] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability](constrained_hamiltonian_systems_on_observation-induced_fiber_bundles_theory_of_s.md)

</div>

<!-- RELATED:END -->
