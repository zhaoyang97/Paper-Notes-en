---
title: >-
  [Paper Note] The Human Brain as a Combinatorial Complex
description: >-
  [NeurIPS 2025 (Workshop: NeurReps)][Medical Imaging][Combinatorial Complexes] This paper proposes a data-driven framework that constructs Combinatorial Complexes (CCs) directly from fMRI time series using information-theoretic measures—namely S-information and O-information—encoding higher-order synergistic interactions among brain regions into topological structures, thereby laying the groundwork for applying topological deep learning to brain network analysis.
tags:
  - "NeurIPS 2025 (Workshop: NeurReps)"
  - "Medical Imaging"
  - "Combinatorial Complexes"
  - "Higher-Order Networks"
  - "Information Theory"
  - "S-Information"
  - "O-Information"
  - "fMRI"
date: 2026-05-08
content_hash: 3b358f95b481dfa9
---

# The Human Brain as a Combinatorial Complex

**Conference**: NeurIPS 2025 (Workshop: NeurReps)
**arXiv**: [2511.20692](https://arxiv.org/abs/2511.20692)  
**Code**: [TopoBrainX](https://github.com/)  
**Area**: Topological Deep Learning / Brain Network Analysis / Medical Imaging
**Keywords**: Combinatorial Complexes, Higher-Order Networks, Information Theory, S-Information, O-Information, fMRI

## TL;DR

This paper proposes a data-driven framework that constructs Combinatorial Complexes (CCs) directly from fMRI time series using information-theoretic measures—namely S-information and O-information—encoding higher-order synergistic interactions among brain regions into topological structures, thereby laying the groundwork for applying topological deep learning to brain network analysis.

## Background & Motivation

**Background**: Current brain network analysis relies almost exclusively on graph representations, modeling brain regions as nodes and constructing edges via pairwise statistical measures such as Pearson correlation, partial correlation, or mutual information. This framework has achieved remarkable success in network neuroscience, advancing the understanding of functional connectivity, network topology, and organizational principles of the brain.

**Limitations of Prior Work**: Graph representations suffer from a fundamental limitation: they can only capture pairwise relationships and are entirely incapable of expressing collective dependencies among three or more brain regions (higher-order interactions). For a system with $N$ brain regions, a graph captures at most $\binom{N}{2}$ pairwise relationships, yet real neural information processing frequently involves synergistic interactions—information that is only accessible through the joint observation of multiple regions and cannot be decomposed into pairwise relationships.

**Key Challenge**: Neuroscientific research increasingly suggests that higher-order interactions may constitute a crucial organizational principle of brain networks, yet existing representational methods are fundamentally incapable of capturing such structures. Although topological data analysis (TDA) provides some tools (e.g., Betti numbers, persistent homology), these primarily rely on geometric or distance-based filtrations and depend on assumptions about embedding spaces.

**Goal**: How can higher-order interactions be captured from neural data and encoded into a mathematical structure amenable to topological deep learning architectures?

**Key Insight**: The authors observe that Combinatorial Complexes (CCs) are more flexible than simplicial complexes—requiring neither closure nor inclusion constraints—making them well-suited for directly integrating continuous information-theoretic measures. Concurrently, recent advances in multivariate information theory (S-information, O-information) enable systematic quantification of higher-order dependencies.

**Core Idea**: S-information quantifies the strength of higher-order statistical dependencies, while O-information discriminates between synergistic and redundant character. A dual-threshold strategy is employed to construct CCs directly from fMRI data, bypassing conventional topological lifting pipelines.

## Method

### Overall Architecture

The input is an fMRI time series matrix $\mathbf{X} \in \mathbb{R}^{N \times T}$ ($N$ brain regions, $T$ time points), and the output is a Combinatorial Complex $(S, \mathcal{X}, \mathrm{rk})$. The overall pipeline proceeds in three steps: (1) compute pairwise statistical dependencies to construct rank-1 edges; (2) enumerate candidate higher-order subsets and compute information-theoretic measures; (3) apply a dual-threshold strategy to select synergy-dominated subsets as higher-order cells.

### Key Designs

1. **Mathematical Definition of Combinatorial Complexes (CCs)**:

    - Function: Provides a unified mathematical structure capable of simultaneously accommodating pairwise and higher-order relationships.
    - Mechanism: A CC is a triple $(S, \mathcal{X}, \mathrm{rk})$, where $S$ is a finite set (the set of brain regions), $\mathcal{X} \subseteq \mathcal{P}(S) \setminus \{\emptyset\}$ is a subset of the power set of $S$, and $\mathrm{rk}: \mathcal{X} \to \mathbb{Z}_{\geq 0}$ is a rank function. The rank function requires that every singleton $\{s\}$ belongs to $\mathcal{X}$, and that subset relations imply rank monotonicity ($x \subseteq y \Rightarrow \mathrm{rk}(x) \leq \mathrm{rk}(y)$). Rank-0 corresponds to nodes, rank-1 to edges, and rank-$k$ to $(k+1)$-tuples.
    - Design Motivation: Compared to simplicial complexes (which require closure: if a triplet exists, all its sub-edges must also exist), CCs relax these constraints, allowing higher-order cells to exist independently of their lower-order subsets. This is critical for construction based on statistical measures—a triplet may exhibit strong synergy even when its pairwise connections are weak.

2. **S-Information (Total Statistical Interdependence Measure)**:

    - Function: Quantifies the overall strength of statistical dependencies among a set of variables.
    - Mechanism: $\Sigma(X) = TC(X) + DTC(X)$, where $TC$ (total correlation) and $DTC$ (dual total correlation) each quantify multivariate dependence from distinct perspectives. Higher $\Sigma$ values indicate stronger multivariate dependencies within a subset, potentially harboring important higher-order synergistic interactions.
    - Design Motivation: Serves as a first-pass filter to exclude candidate sets with weak statistical dependence, ensuring that retained higher-order cells carry sufficient information-theoretic significance.

3. **O-Information (Redundancy–Synergy Bias Measure) and the Dual-Threshold Strategy**:

    - Function: Distinguishes the nature of higher-order dependencies—whether redundant (information appearing repeatedly across multiple sources) or synergistic (information accessible only through joint observation).
    - Mechanism: $\Omega(X) = TC(X) - DTC(X)$; $\Omega < 0$ indicates net synergy, while $\Omega > 0$ indicates redundancy. The construction rule applies dual thresholds: $\Sigma(x) > \tau$ and $\Omega(x) \lesssim 0$, where the former ensures dependency strength and the latter ensures synergy dominance. The use of $\lesssim 0$ rather than strict $< 0$ accommodates structures that are weakly synergy-dominated but exhibit slightly positive $\Omega$.
    - Design Motivation: The higher-order interactions of primary interest in neuroscience are synergistic—collective information processing patterns that cannot be decomposed into pairwise relationships. S-information alone cannot distinguish redundancy from synergy; the introduction of O-information enables targeted identification of synergistic structures.

### Loss & Training

This work focuses on representation construction and does not involve downstream learning tasks. The resulting CCs can serve as inputs to Combinatorial Complex Neural Networks (CCNNs), which perform downstream tasks (e.g., brain state classification, cognitive task decoding) via message passing across multiple topological ranks; however, this is deferred to future work. Information-theoretic measures are computed using the JIDT (Java Information Dynamics Toolkit) library, accessed via a Java–Python bridge through JPype.

## Key Experimental Results

### Main Results (NetSim Synthetic Data Validation)

Experiments use NetSim synthetic BOLD time series data (sim1.mat, 50 subjects, 5 brain regions, 200 time points per subject). NetSim generates neural processes via Dynamic Causal Modeling (DCM) coupled with a nonlinear balloon–Windkessel hemodynamic forward model to simulate realistic fMRI characteristics.

| Measure | Triplet (2,3,4) | Triplet (1,2,3) | Threshold Criterion |
|---------|----------------|----------------|---------------------|
| S-information Σ | 0.51 | 0.49 | > 0.45 ✓ |
| O-information Ω | 0.06 | 0.04 | ≲ 0 ✓ (weakly positive) |
| Rank | 2 | 2 | — |

Rank-1 edges are constructed based on pairwise mutual information with threshold MI ≥ 0.02, ensuring sparsity while retaining meaningful connections.

### Ablation Study (Method Characteristics Comparison)

| Property | Traditional Graph | Simplicial Complex | Ours (CC) |
|----------|------------------|-------------------|-----------|
| Maximum relation order | Pairwise | Arbitrary | Arbitrary |
| Closure constraint | None | Required | Not required |
| Information type | Pairwise correlation | Geometric/distance | Synergy/redundancy (data-driven) |
| Construction basis | Statistical testing | Distance filtration | Information-theoretic dual threshold |
| Scalability | $O(N^2)$ | $O(N^k)$ | $O(N^k)$, requires optimization |

### Key Findings

- Even in a minimal 5-region setting, CCs reveal triplet structures entirely invisible to traditional graph methods, validating the proposed approach.
- The O-information values of the two selected triplets are weakly positive (0.06 and 0.04), indicating that the boundary between synergy and redundancy in real neural signals is not sharply defined; the design choice of $\lesssim 0$ rather than strict $< 0$ is thus well-justified.
- Computational complexity is a critical bottleneck: the number of candidate triplets grows rapidly from $\binom{5}{3} = 10$ to $\binom{100}{3} = 161{,}700$, exceeding desktop computing capacity for $N > 50$.

## Highlights & Insights

- **Data-Driven vs. Topological Lifting**: Existing TDL work typically maps from graph structures to higher-order domains via topological lifting; this paper bypasses that step entirely, constructing CCs directly from statistical dependencies in time series. This avoids potential information loss inherent in the "build graph then lift" pipeline.
- **Complementary Information-Theoretic Measures**: S-information addresses "whether dependencies exist," while O-information addresses "what kind of dependencies they are"—the two are complementary and together form a complete selection criterion. This dual-criterion strategy is directly transferable to any multivariate analysis task requiring discrimination between redundant and synergistic interactions.
- **Generality of the Framework**: Although fMRI serves as the entry point, the framework is applicable to any multivariate time series data and can be extended to domains such as finance and climate science.

## Limitations & Future Work

- **Proof-of-Concept Stage**: Validation is limited to 5-region NetSim synthetic data; no real fMRI data are tested, no downstream learning tasks (e.g., classification, decoding) are executed, and it remains unclear whether the constructed CCs are genuinely useful in practice.
- **Fixed Threshold Strategy**: The thresholds for S-information and O-information (0.45 and $\lesssim 0$) are determined through exploratory experiments and may not generalize to other datasets. Adaptive criteria, statistical testing, and robust estimators beyond the Gaussian assumption are needed.
- **Combinatorial Explosion**: The number of candidate rank-$k$ cells is $\binom{N}{k+1}$, rendering the approach infeasible for standard brain atlases with $N \approx 100$. The authors mention mitigation strategies such as locality-sensitive hashing, Kalman filtering, and similarity-based pre-selection, but none are implemented.
- **Dependence on Brain Region Definition**: Results may be strongly influenced by the choice of brain parcellation scheme (ICA vs. anatomical parcellation), a challenge common to higher-order network modeling.
- **Gaussian Estimator Limitations**: JIDT employs Gaussian estimators for information-theoretic measures, which may be inaccurate for the non-Gaussian distributions characteristic of real fMRI data.

## Related Work & Insights

- **vs. Persistent Homology / Betti Number Methods**: These TDA methods focus on global topological invariants derived from geometric/distance filtrations, whereas the proposed method is grounded in statistical dependencies. The two are complementary—persistent homology captures global shape features, while CCs capture local higher-order interaction patterns.
- **vs. Hypergraph Methods**: Hypergraphs can also represent higher-order relationships but lack the hierarchical structure provided by a rank function and generally do not distinguish between synergistic and redundant interactions.
- **vs. TopoBench (Telyatnikov 2025)**: TopoBench provides a standardized benchmarking platform for TDL architectures. The CC construction pipeline proposed here can be directly integrated with TopoBench, establishing a foundation for evaluating CCNNs on brain network data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First direct integration of information theory and combinatorial complexes for brain network analysis; the approach of bypassing topological lifting is conceptually novel.
- Experimental Thoroughness: ⭐⭐⭐ — Proof-of-concept only; no real data, no downstream tasks, no quantitative comparison against alternative methods.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical framework is clearly presented, motivation is well-articulated, and the pipeline is described in detail.
- Value: ⭐⭐⭐⭐ — Promising direction, but considerable follow-up work is required to validate practical feasibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SapiensID: Foundation for Human Recognition](../../CVPR2025/medical_imaging/sapiensid_foundation_for_human_recognition.md)
- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[ICML 2025\] Bayesian Inference for Correlated Human Experts and Classifiers](../../ICML2025/medical_imaging/bayesian_inference_for_correlated_human_experts_and_classifiers.md)
- [\[CVPR 2026\] Dynamic Stream Network for Combinatorial Explosion Problem in Deformable Medical Image Registration](../../CVPR2026/medical_imaging/dynamic_stream_network_for_combinatorial_explosion_problem_in_deformable_medical.md)
- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2025/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)

</div>

<!-- RELATED:END -->
