---
title: >-
  [Paper Note] Logical Characterizations of GNNs with Mean Aggregation
description: >-
  [AAAI 2026][Graph Learning][GNN expressiveness] This paper provides a systematic logical characterization of GNNs using mean aggregation: under the non-uniform setting, mean-GNNs are equivalent to Ratio Modal Logic (RML); under the uniform setting (relative to MSO), they are equivalent to Modal Logic (ML); when the combination function is additionally required to be continuous and the classification function is a threshold, the expressiveness drops significantly to Alternation-Free Modal Logic (AFML).
tags:
  - AAAI 2026
  - Graph Learning
  - GNN expressiveness
  - mean aggregation
  - modal logic
  - ratio modal logic
  - alternation-free modal logic
  - non-uniform/uniform setting
date: 2026-05-08
content_hash: 8ddd82766fafcdea
---

# Logical Characterizations of GNNs with Mean Aggregation

**Conference**: AAAI 2026
**arXiv**: [2507.18145](https://arxiv.org/abs/2507.18145)
**Area**: Graph Learning / Graph Neural Network Theory
**Keywords**: GNN expressiveness, mean aggregation, modal logic, ratio modal logic, alternation-free modal logic, non-uniform/uniform setting

## TL;DR

This paper provides a systematic logical characterization of GNNs using mean aggregation: under the non-uniform setting, mean-GNNs are equivalent to Ratio Modal Logic (RML); under the uniform setting (relative to MSO), they are equivalent to Modal Logic (ML); when the combination function is additionally required to be continuous and the classification function is a threshold, the expressiveness drops significantly to Alternation-Free Modal Logic (AFML).

## Background & Motivation

The choice of aggregation function (sum/max/mean) in GNNs directly determines which graph properties the model can express. Prior work has established correspondences between sum-GNNs and Graded Modal Logic (GML), and between max-GNNs and Modal Logic (ML); however, **mean aggregation—the core operation in mainstream architectures such as GraphSAGE and GCN—lacks a complete logical characterization**.

Mean aggregation is widely used in practice for several reasons:
1. Systems such as GraphSAGE and PinSAGE directly employ mean aggregation.
2. GCN uses a weighted variant of mean aggregation.
3. Mean aggregation is compatible with stochastic sampling, making it suitable for high-degree nodes.

The core problem is: which graph properties can mean-GNNs precisely express? And what logic do they correspond to under different theoretical settings (uniform/non-uniform, with or without continuity constraints)?

## Method

### Overall Architecture

The paper characterizes the expressiveness of mean-GNNs across two major settings (non-uniform vs. uniform) and two classes of technical constraints (general vs. continuous+threshold), establishing bidirectional translations (GNN→logic and logic→GNN), and systematically comparing the results with sum and max aggregation.

| Setting | Mean (continuous+threshold) | Mean (general) | Sum | Max |
|---|---|---|---|---|
| Non-uniform | RML | RML | GML | ML |
| Uniform (relative to MSO) | **AFML** | ML | GML | ML |
| Uniform (absolute) | >> AFML | >> ML | >> GML | ML |

### Key Design 1: Ratio Modal Logic (RML) and the Non-Uniform Setting

- A new logic RML is defined: the modal operator $\Diamond^{\geq r}\varphi$ denotes "at least a fraction $r$ of successors satisfy $\varphi$."
- **Mean-GNN → RML**: Since mean aggregation depends only on the ratio distribution of neighbor feature vectors, an RML formula can be constructed for each feature vector at each layer.
- **RML → Mean-GNN**: Sub-formulas of an RML formula are encoded layer by layer as components of GNN feature vectors, using an $n^2$ amplification factor to distinguish different ratios.
- Corollary: Under the non-uniform setting, Max-GNN $\subsetneq$ Mean-GNN $\subsetneq$ Sum-GNN (strict hierarchy).

### Key Design 2: Mean-GNN $\cap$ MSO = ML Under the Uniform Setting

- The modal operators of RML cannot be expressed in MSO, so RML is truncated to ML in the uniform setting.
- The key tool is **graph scaling invariance**: scaling a graph by a factor $c$ (i.e., $c \cdot G$) does not change the result of mean aggregation.
- Using Ehrenfeucht-Fraïssé games, the paper proves that a Duplicator strategy for the scaling + GML game can be derived from a strategy for the ML game.
- Result: Under the uniform setting, mean-GNN (relative to MSO) has the same expressiveness as max-GNN, both being equivalent to ML.

### Key Design 3: AFML Characterization Under Continuous+Threshold Constraints

- In practice, GNN combination functions are typically continuous (to support backpropagation) and classification functions are typically thresholds.
- **Mean$^{c,t}$-GNN → AFML**: Using the AFML game (where Spoiler can only choose one direction) together with a graph extension lemma, the paper proves that mean-GNNs with continuous combination and threshold classification cannot mix $\Diamond$ and $\Box$.
- **AFML → Mean$^{c,t}$-GNN**: Truth values in AFML formulas are encoded as positive values in $(0,1]$ and false values as $0$, realized via truncated ReLU/sigmoid.
- Core finding: The continuous+threshold constraint reduces the expressiveness of mean-GNN from ML to AFML, while leaving sum and max unaffected.

### Loss & Training

This is a purely theoretical work and involves no training loss functions. The central objective is to establish bidirectional equivalence translations between logics and GNNs.

## Experiments

This is a theoretical paper with no experimental section; however, a complete theorem-proof system is provided.

### Main Results

| Theorem | Content | Setting |
|---|---|---|
| Thm 1 | Mean-GNN $\subseteq$ RML $\subseteq$ simple Mean-GNN | Non-uniform |
| Thm 7 | Mean-GNN $\cap$ MSO $\subseteq$ ML $\subseteq$ simple Mean-GNN | Uniform |
| Thm 12 | Mean$^{c,t}$-GNN $\cap$ MSO $\subseteq$ AFML $\subseteq$ simple Mean$^{c,t}$-GNN | Uniform + continuous threshold |
| Cor 1 | Max-GNN $\subsetneq$ Mean-GNN $\subsetneq$ Sum-GNN | Non-uniform |
| Cor 2 | Max-GNN = Mean-GNN $\cap$ MSO $\subsetneq$ Sum-GNN $\cap$ MSO | Uniform |

### Ablation Study / Comparisons

| Property | Mean | Sum | Max |
|---|---|---|---|
| Continuous+threshold constraints reduce expressiveness? | ✅ From ML to AFML | ❌ | ❌ |
| Graph scaling invariance | ✅ | ❌ | ❌ |
| Generality of activation functions (non-uniform setting) | Any continuous non-polynomial | Any continuous non-polynomial | ReLU/truncated ReLU only |

### Key Findings

1. **The practical expressiveness of mean aggregation is lower than intuition suggests**: under realistic constraints (continuous combination + threshold classification), it is equivalent to AFML and cannot mix existential ($\Diamond$) and universal ($\Box$) modalities.
2. **The three aggregation functions form a strict hierarchy**: Sum > Mean > Max (non-uniform); Sum > Mean = Max (uniform + MSO).
3. Continuity/threshold constraints affect only mean aggregation and have no impact on sum and max.

## Highlights & Insights

- **The first complete logical characterization of mean-GNNs**, filling a significant gap in GNN expressiveness theory.
- Introduces two new logical tools: **Ratio Modal Logic (RML)** and **Alternation-Free Modal Logic (AFML)**.
- Reveals the **unique effect of practical constraints (continuous+threshold) on mean aggregation**—a phenomenon not observed for sum or max.
- Graph scaling invariance serves as a key technical device that elegantly reduces the analysis from RML to ML.
- Theoretical results cover mainstream activation functions including ReLU, truncated ReLU, and sigmoid.

## Limitations & Future Work

- **Purely theoretical work** with no experimental validation of whether the theoretical predictions translate to guidance for practical tasks.
- Under the uniform setting, the translation from ML to mean-GNN relies on discontinuous activation functions or unnatural classification functions (classifying irrational numbers as 1), limiting practical utility.
- Extension to heterogeneous aggregation (different aggregation functions at different layers) is not explored.
- The logical characterization of weighted mean (e.g., the normalization scheme in GCN) is not discussed.
- The question of whether the generality of activation functions for max-GNN can be extended to all continuous non-polynomial functions remains open.

## Related Work & Insights

- **Logical characterization of sum-GNNs**: Barceló et al. (2020) prove that Sum-GNN $\leftrightarrow$ GML (relative to FO).
- **Logical characterization of max-GNNs**: Tena Cucala et al. (2024) translate max-GNNs into non-recursive datalog.
- **Non-logical approaches to GNN expressiveness**: Xu et al. (ICLR 2019) study the expressiveness of different aggregation functions without using logical characterizations.
- **Modal logic characterization of Transformer layers**: closely related to mean aggregation.
- **Logical interpretation of monotone mean-GNNs**: Morris & Horrocks (2025).

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contributions are rigorous and comprehensive, systematically closing the gap in the logical characterization of mean-GNNs. The introduction of RML and AFML provides precise analytical tools for future research. However, the purely theoretical nature of the work limits its direct applicability to practical GNN design, and the utility of some translation constructions is insufficient.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[AAAI 2026\] Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)

<!-- RELATED:END -->
