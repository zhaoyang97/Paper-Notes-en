---
title: >-
  [Paper Note] Characterization and Learning of Causal Graphs from Hard Interventions
description: >-
  [NeurIPS 2025][Causal Inference][hard intervention] This work systematically analyzes the theoretical advantages of hard interventions in causal discovery with latent variables for the first time. It proposes a generalized do-calculus (4 rules) and a twin augmented MAG graph representation, establishes the necessary and sufficient graphical conditions for $\mathcal{I}$-Markov equivalence classes, and designs a provably sound FCI-variant learning algorithm. Experiments show th…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "hard intervention"
  - "causal discovery"
  - "Markov equivalence class"
  - "twin augmented MAG"
  - "do-calculus"
  - "latent variables"
date: 2026-05-08
content_hash: 180114b1e2dfe3fa
---

# Characterization and Learning of Causal Graphs from Hard Interventions

**Conference**: NeurIPS 2025  
**arXiv**: [2505.01037](https://arxiv.org/abs/2505.01037)  
**Author**: Zihan Zhou, Muhammad Qasim Elahi, Murat Kocaoglu (Johns Hopkins / Purdue)  
**Code**: To be confirmed  
**Area**: Audio and Speech  
**Keywords**: hard intervention, causal discovery, Markov equivalence class, twin augmented MAG, do-calculus, latent variables

## TL;DR

This work systematically analyzes the theoretical advantages of hard interventions in causal discovery with latent variables for the first time. It proposes a generalized do-calculus (4 rules) and a twin augmented MAG graph representation, establishes the necessary and sufficient graphical conditions for $\mathcal{I}$-Markov equivalence classes, and designs a provably sound FCI-variant learning algorithm. Experiments show that hard interventions reduce the equivalence class size by 37-57% compared to soft interventions.

## Background & Motivation

**Background**: Causal discovery aims to infer causal structures from observational and experimental data, with conditional independence (CI) constraints and the d-separation criterion serving as core tools. Existing interventional causal discovery methods primarily handle soft interventions, which alter the conditional distributions of variables while preserving the causal graph structure.

**Limitations of Prior Work**: Soft interventions do not remove the incoming edges of interventional variables, and thus fail to break inducing paths. In the presence of latent variables, many distinct causal graphs yield identical family distributions under soft interventions, resulting in excessively large Markov equivalence classes (MECs) that cannot be further narrowed down.

**Key Challenge**: Hard interventions ($\text{do}(X=x)$) directly set variables to constants and remove all incoming edges, which can non-locally alter d-separation relations. For instance, performing a hard intervention $\text{do}(Z)$ on the graph $\{X \to Z \to Y, Z \leftrightarrow Y\}$ breaks the inducing path $\langle X,Z,Y\rangle$ and renders $X$ and $Y$ independent, whereas soft interventions cannot achieve this. However, a theoretical framework and learning algorithms for hard interventions have not been previously established.

**Goal**:
   - What additional distribution constraints do hard interventions provide compared to soft interventions?
   - How can the $\mathcal{I}$-Markov equivalence classes of causal graphs with latent variables under hard interventions be characterized?
   - How can algorithms be designed to learn causal structures from multiple hard interventional datasets?

**Key Insight**: The authors start from the converse of Pearl’s do-calculus—inferring the necessary structural constraints in the graph when two interventional distributions are unequal. An F-node augmented graph is utilized to explicitly encode interventional effects into the graph structure for a unified analysis.

**Core Idea**: Hard interventions remove incoming edges to make the causal graph sparser, generating more do-invariance constraints. The proposed Twin Augmented MAG fully captures these new constraints, significantly shrinking the equivalence classes.

## Method

### Overall Architecture

Three progressive theoretical contributions:

1. **Generalized do-calculus**: Extends Pearl's original three rules to compare distributions between any two hard intervention sets (Proposition 3.2, 4 rules).
2. **$\mathcal{I}$-Markov Equivalence Class Characterization**: Provides necessary and sufficient graphical conditions via Twin Augmented MAGs and $\mathcal{I}$-augmented MAGs (Theorem 4.7, Proposition 5.2).
3. **Learning Algorithm**: A 3-stage algorithm based on the FCI framework, containing 4 new orientation rules (Algorithm 1), with proven soundness.

### Key Designs

#### 1. Generalized do-calculus (Proposition 3.2)

- **Function**: Generalizes Pearl's 3 do-rules to 4 rules capable of comparing distributions between any two hard intervention sets $\mathbf{I}, \mathbf{J}$.
- **Mechanism**: Defines the symmetric difference $\mathbf{K} = \mathbf{I} \Delta \mathbf{J}$ and decomposes it into multiple subsets ($\mathbf{K_I}, \mathbf{K_J}, \mathbf{R_I}, \mathbf{R_J}, \mathbf{W_I}, \mathbf{W_J}$). Each rule dictates that $P_\mathbf{I}(\mathbf{y}|\mathbf{w})$ equals $P_\mathbf{J}(\mathbf{y}|\mathbf{w})$ when a d-separation condition is met in a specifically edited graph (removing incoming/outgoing edges):
    - Rule 1 (CI): Conditional independence within a single interventional distribution.
    - Rule 2 (do-see): Equality of two interventional distributions after conditioning on symmetric difference variables.
    - Rule 3 (do-do): Equality of marginals of two interventional distributions.
    - Rule 4 (mixed): A general form that unifies do-see and do-do.
- **Key Insight**: Hard interventions eliminate incoming edges to interventional variables, making the graph sparser and producing more d-separation relationships, which induces more constraints than soft interventions.

#### 2. Twin Augmented MAG (Definition 4.5, Theorem 4.7)

- **Function**: Constructs a graph structure such that verifying whether two causal graphs are $\mathcal{I}$-Markov equivalent reduces to checking standard graphical properties (skeleton + unshielded colliders + discriminating paths).
- **Construction Process**:
  1. For an interventional pair $(\mathbf{I}, \mathbf{J})$, build two copies of the variables, $\mathbf{V}^{(\mathbf{I})}$ and $\mathbf{V}^{(\mathbf{J})}$.
  2. Retain the edges of the corresponding edited causal graphs $\mathcal{D}_{\overline{\mathbf{I}}}$ and $\mathcal{D}_{\overline{\mathbf{J}}}$ within each copy.
  3. Add auxiliary nodes $F$ pointing to all variables in the symmetric difference $\mathbf{K}$.
  4. Form the MAG and add symmetrizing edges to obtain the Twin Augmented MAG.
- **Core Theorem (Theorem 4.7)**: Two graphs are $\mathcal{I}$-Markov equivalent $\iff$ for all interventional pairs $(\mathbf{I}, \mathbf{J})$, their corresponding Twin Augmented MAGs share the same skeleton, same unshielded colliders, and same discriminating path collider properties.
- **Lemma 4.6**: Proves that the Twin Augmented MAG is a valid MAG (preserving ancestry and maximality).

#### 3. $\mathcal{I}$-augmented MAG (Definition 5.1)

- **Design Motivation**: Since $k$ interventional targets yield $\binom{k}{2}$ Twin Augmented MAGs, a more compact representation is required.
- **Mechanism**: Each intervention $\mathbf{I}$ is associated with an $\mathcal{I}$-augmented MAG that consolidates information from the $\mathbf{V}^{(\mathbf{I})}$ domains of all Twin MAGs containing $\mathbf{I}$. This yields a tuple of $k$ graphs rather than $\binom{k}{2}$ graphs.
- **Proposition 5.2**: $\mathcal{I}$-augmented MAGs and Twin Augmented MAGs are fully equivalent with respect to equivalence class characterization.

#### 4. Learning Algorithm (Algorithm 1)

A 3-stage FCI variant:

- **Phase I (Initialization)**: For each $\mathbf{I} \in \mathcal{I}$, create copy $\mathbf{V}^{(\mathbf{I})}$ of variables and initialize it as a fully connected graph (with circle edges), and establish $F$-nodes connected to all variables.
- **Phase II (Skeleton Learning)**: Algorithm 3 determines separating sets via conditional independence tests. It tests $P_\mathbf{I}(y|\mathbf{w},x) = P_\mathbf{I}(y|\mathbf{w})$ between non-$F$ nodes, and tests $P_\mathbf{I}(y|\mathbf{w}) = P_\mathbf{J}(y|\mathbf{w})$ between $F$-nodes and variables.
- **Phase III (Orientation)**: First orient unshielded colliders using Rule 0, then repeatedly apply the 7 standard FCI rules plus 4 new rules:
    - **Rule 8** ($F$-node): Orient all edges of $F$-nodes outwards.
    - **Rule 9** (Interventional node): If $X \in \mathbf{I}$ and $X, Y$ are adjacent $\to$ $X \to Y$ (since $Y$ must be a descendant of $X$).
    - **Rule 10** (Consistency across skeletons): If $X \to Y$ is oriented in one domain, mark the $Y$ end of the edge as an arrowhead in other domains (ancestral relationships cannot be reversed by hard interventions).
    - **Rule 11** (inducing path): If $\mathbf{J} = \mathbf{I} \cup \{X\}$ and $F$ is adjacent to $Y \to X \to Y$.

### Theoretical Guarantees

- **Theorem 6.3 (Soundness)**: Under the h-faithfulness assumption, in the $\mathcal{I}$-augmented graph output by Algorithm 1, every adjacency relation and orientation mark consistently holds across all $\mathcal{I}$-Markov equivalent graphs.
- Completeness is left as an open problem.

## Experimental Results

### Experiment 1: Exhaustive Evaluation of All ADMGs

Exhaustively evaluate all ADMGs for small-scale variables ($n = 2, 3, 4$) to compare the size of $\mathcal{I}$-MECs under hard versus soft interventions:

| $n$ | Hard Interventions MEC | Soft Interventions MEC | Ratio | Total ADMGs |
|-----|-----------|-----------|------|----------|
| 2 (Random) | 2.03 | 2.93 | 0.69 | 6 |
| 3 (Random) | 19.50 | 30.57 | 0.64 | 200 |
| 4 (Random) | 677.13 | 1218.83 | 0.56 | 34,752 |
| 2 (Complete) | 2.37 | 3.67 | 0.65 | 6 |
| 3 (Complete) | 14.03 | 24.70 | 0.57 | 200 |
| 4 (Complete) | 721.37 | 1529.57 | 0.47 | 34,752 |

**Key Findings**: The MEC size for hard interventions is approximately 47-69% of that for soft interventions. As the number of variables increases, the ratio continues to decrease, which demonstrates that hard interventions offer greater advantages in more complex graphs.

### Experiment 2: Sampling ADMGs ($n=5$)

At $n=5$, the total number of ADMGs reaches 29,983,744, making exhaustive evaluation intractable. Using Hoeffding's inequality to bound the estimation error ($\epsilon=0.01$ with 99% confidence), 23,025 ADMGs were sampled for each setting:

- The expected MEC size under hard interventions $\mathbb{E}_\mathcal{S}^{hard}$ is significantly lower than that under soft interventions $\mathbb{E}_\mathcal{S}^{soft}$.
- The trend aligns with the exhaustive search experiments: hard interventions shrink the equivalence classes much more effectively.

## Analysis of Strengths and Limitations

### Strengths

1. **Solid Theoretical Contributions**: It characterizes the $\mathcal{I}$-Markov equivalence classes of causal graphs with latent variables under hard interventions for the first time, providing necessary and sufficient conditions in Theorem 4.7.
2. **Unified Framework**: The 4 rules of the generalized do-calculus cover all possible distribution comparisons between hard interventions, treating soft interventions as a special case.
3. **Compact Representation**: $\mathcal{I}$-augmented MAGs compress $\binom{k}{2}$ pairwise graphs into $k$ graphs, significantly reducing complexity.
4. **Four New Orientation Rules**: Beautifully designed to exploit the unique properties of hard interventions (removal of incoming edges, invariance of ancestral relations, and breaking of inducing paths).
5. **Provably Sound Algorithm**: Soundness of the algorithm is theoretically guaranteed.

### Limitations

1. **Unproven Completeness**: Algorithm 1 is only proven to be sound; whether it can recover all identifiable edges remains an open question.
2. **High Computational Complexity**: Skeleton learning requires testing all possible conditioning sets, leading to exponential complexity in the worst case.
3. **Limited to Hard Interventions**: In practical applications (especially in biology), precise hard interventions can be difficult to implement; however, they are easier to conduct in computer systems (like microservice architectures).
4. **Small Experimental Scale**: Due to the super-exponential growth of the number of ADMGs, experiments are restricted to $n \leq 5$, and performance on large-scale graphs remains to be verified.
5. **Lack of Finite-Sample Guarantees**: The theoretical results assume oracle-level independence and do not analyze the impact of estimation error on the algorithm.

## Related Work & Insights

| Method | Intervention Type | Latent Variables | Equivalence Class Characterization | Learning Algorithm |
|------|---------|--------|-----------|---------|
| Hauser & Bühlmann (2012) | Hard/Soft | ✗ | ✓ | ✓ |
| Yang et al. (2018) | Hard/Soft | ✗ | ✓ (Identical Characterization) | ✓ |
| Kocaoglu et al. (2019) | Soft | ✓ | ✓ (Augmented MAG) | ✓ |
| Jaber et al. (2020) | Unknown Soft | ✓ | ✓ ($\Psi$-MEC) | ✓ |
| **Ours** | **Hard** | **✓** | **✓ (Twin Augmented MAG)** | **✓** |

This work fills the theoretical gap in the combination of "hard interventions + latent variables". Compared with the closest work by Kocaoglu et al. (2019), this work exploits the fact that hard interventions remove incoming edges to yield stronger constraints, reducing the equivalence class size by over 30-50%.

## Personal Reflections

This paper represents an important advancement in causal discovery theory. The core insight is intuitive—hard interventions are more informative than soft interventions—but formalizing this intuition into a rigorous equivalence class characterization requires substantial technical effort. The construction of the Twin Augmented MAG elegantly encodes the "non-local effects" of hard interventions into the adjacencies of auxiliary $F$-nodes, allowing standard MAG comparison methods to be directly reused.

From a practical standpoint, this work is highly valuable for causal discovery in computer systems (such as microservice causal analysis and A/B testing) where hard interventions are highly feasible. Its applicability in biology is relatively limited. Soundness/completeness analyses under finite-sample regimes are obvious future directions.

## Key Experimental Results

### Main Results (Exhaustive Evaluation of All ADMGs)

| Number of Variables | Hard Intervention MEC | Soft Intervention MEC | Ratio (Hard/Soft) |
|-------|---------|---------|----------|
| n=2 | 2.03 | 2.93 | 0.69 |
| n=3 | 19.50 | 30.57 | 0.64 |
| n=4 (Complete DAG) | 721.37 | 1529.57 | **0.47** |
| n=5 (Sampling) | 0.028 | 0.061 | 0.46 |

### Ablation Study: Impact of Confounder Density ($n=5$)

| Bidirectional Edge Density $\rho$ | Hard/Soft Ratio |
|----------|---------|
| $\rho=0.1$ (Low Confounding) | 0.804 |
| $\rho=0.5$ (Medium Confounding) | 0.459 |
| $\rho=0.9$ (High Confounding) | **0.365** |

### Key Findings
- **Hard intervention advantages grow with graph scale**: 31% reduction for $n=2$ versus a 53% reduction for $n=4$.
- **More confounding enhances the benefit of hard interventions**: Under high confounding ($\rho=0.9$), the MEC is reduced by 63.5%.
- **Non-local effects are key**: Hard interventions disrupt inducing paths between distant variables by removing incoming edges.

## Highlights & Insights
- **Non-local information of hard interventions quantified for the first time**: Beyond theoretical superiority, experiments provide concrete reduction ratios.
- **Rule 4 of the generalized do-calculus unifies extreme cases**: Laying the foundation for future work on hybrid hard/soft interventions.
- **Symmetrization strategy of Twin Augmented MAGs**: Expresses unobservable constraints using observables, enabling learning based on pairwise distributions.

## Limitations & Future Work
- Only the soundness of the algorithm is proven, leaving completeness unproven.
- The h-faithfulness assumption is difficult to verify in real-world data.
- Computational complexity explodes under multiple interventional targets (due to the $\binom{k}{2}$ pairwise graph construction).
- Lack of experiments on large-scale graphs ($n > 6$).

## Related Work & Insights
- **vs Causal Discovery via Soft Interventions (e.g., GIES)**: Hard interventions provide more information, leading to smaller MECs.
- **vs FCI Algorithm**: This work extends FCI to handle constraints unique to hard interventions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic theoretical framework for causal discovery under hard interventions.
- Experimental Thoroughness: ⭐⭐⭐ Exhaustive and sampling-based experiments, but lacks real-world data and finite-sample analysis.
- Writing Quality: ⭐⭐⭐⭐ Theory is rigorous, with complete defining concepts and theorem systems.
- Value: ⭐⭐⭐⭐ Provides new theoretical tools and algorithmic foundations for causal discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data](../../ICLR2026/causal_inference/characterization_and_learning_of_causal_graphs_with_latent_confounders_and_post-.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[NeurIPS 2025\] It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation](its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)

</div>

<!-- RELATED:END -->
