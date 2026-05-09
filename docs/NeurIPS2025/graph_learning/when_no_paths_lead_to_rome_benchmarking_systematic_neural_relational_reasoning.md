---
title: >-
  [Paper Note] When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning
description: >-
  [NeurIPS 2025][Graph Learning][relational reasoning] This paper introduces the NoRA benchmark, which systematically breaks the assumption underlying existing relational reasoning benchmarks that "reasoning can be reduced to path composition." By introducing off-path reasoning, ambiguous facts, and multi-relational settings, it reveals fundamental deficiencies in all existing models—including o3—on off-path reasoning.
tags:
  - NeurIPS 2025
  - Graph Learning
  - relational reasoning
  - benchmark
  - off-path reasoning
  - systematic generalization
  - graph neural networks
date: 2026-05-08
content_hash: 59ca117ce07d46b0
---

# When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2510.23532](https://arxiv.org/abs/2510.23532)
**Code**: [GitHub](https://github.com/axd353/WhenNoPathsLeadToRome/)
**Area**: Graph Learning
**Keywords**: relational reasoning, benchmark, off-path reasoning, systematic generalization, graph neural networks

## TL;DR

This paper introduces the NoRA benchmark, which systematically breaks the assumption underlying existing relational reasoning benchmarks that "reasoning can be reduced to path composition." By introducing off-path reasoning, ambiguous facts, and multi-relational settings, it reveals fundamental deficiencies in all existing models—including o3—on off-path reasoning.

## Background & Motivation

**Background**: Systematic relational reasoning is a central challenge in designing models that can apply learned rules in a compositional manner. CLUTRR is the most widely used benchmark, focusing on family relationship reasoning. State-of-the-art methods include CTP (differentiable logic programming), R5/NCRL (path composition), Edge Transformer, and EpiGNN.

**Limitations of Prior Work**:
- All reasoning in CLUTRR can be reduced to **path composition**: answers are derived by composing relations along a single path connecting the source and target.
- Many state-of-the-art model architectures hard-code a path bias (e.g., R5 and NCRL can only reason along paths).
- STaR requires composing multiple paths but remains primarily path-based in its reasoning.

**Key Challenge**: Real-world relational reasoning often requires "detours"—using information not on the source-to-target path to draw conclusions. For example, knowing that Wes has "no daughters" (off-path information) is necessary to infer that Ann is Todd's maternal aunt. Existing benchmarks cannot test this capability.

**Goal**: To construct a benchmark that systematically challenges path bias, precisely quantify the difficulty of off-path reasoning, and drive the development of more general reasoning models.

**Key Insight**: Design a rich rule set incorporating gender-specific family relations, everyday relations (classmates, neighbors, etc.), and ambiguous facts, such that reasoning must involve off-path nodes and constraint disambiguation.

**Core Idea**: NoRA breaks the path composition assumption by introducing off-path reasoning, ambiguity resolution, and multi-relational settings, thereby exposing the true bottleneck of systematic reasoning.

## Method

### NoRA Benchmark Design

#### Three Key Innovations

1. **Off-Path Reasoning**: Inference requires "detours" using edges not on the source-to-target path. As shown in Figure 1, determining that Ann is Todd's maternal aunt requires accessing the off-path node Wes's "no daughters" attribute.

2. **Ambiguous Facts**: Facts are encoded using ASP syntax as $l\{r_1(x_1,y_1),\ldots,r_n(x_n,y_n)\}u$, indicating that between $l$ and $u$ of the listed facts are true. Models must disambiguate via constraints to determine which relations hold.

3. **Multi-Relational**: Multiple relations can exist simultaneously between entities—either hierarchically (aunt and maternal_aunt) or independently (brother and classmate).

#### Difficulty Metrics

- **Reasoning Depth**: The minimum number of rule application steps required for derivation.
- **Reasoning Width**: The number of unique derivation paths, quantifying the degree of ambiguity.
- **Backtrack Load (BL)**: The ratio of reasoning steps to entities involved, measuring whether inference must traverse back and forth along a path.
- **Off-Path Edge Count (OPEC)**: The number of edges used in derivation that lie on no direct path between the source and target.

#### Dataset Splits

| Split | Depth | Width | BL | OPEC |
|-------|-------|-------|----|------|
| Train-a | ≤6 | ≤5 | ≤1.5 | ≤2 |
| Test-D | **>6** | ≤5 | ≤1.5 | ≤2 |
| Test-W | ≤6 | **>5** | ≤1.5 | ≤2 |
| Test-BL | ≤6 | ≤5 | **>1.5** | - |
| Test-OPEC | - | - | - | **≥3** |

### Reasoning Framework

Based on Answer Set Programming (ASP):
- **World rules**: Deterministic rules (e.g., transitivity of living_in_same_place) and constraint rules (e.g., minors cannot be parents).
- **Answer set**: All non-contradictory interpretations of a given story.
- **Query**: Given a story $\mathcal{S}$ and an entity pair $(x,y)$, predict all derivable relations $\mathcal{R} = \bigcap\{\text{rels}(x,y,\mathcal{A}) | \mathcal{A}\text{ is answer set}\}$.

### NoRA v1.1: Extended Version

A recursive subgraph expansion technique is used to generate test instances with higher OPEC, depth, and BL. Each test instance is guaranteed to be constructible by "splicing" training instances, ensuring the feasibility of compositional generalization.

## Key Experimental Results

### Main Results: Exact-Match Accuracy

| Model | In-dist | Test-D | Test-W | Test-BL | Test-OPEC |
|-------|---------|--------|--------|---------|-----------|
| ET (single-edge) | 0.885 | 0.741 | 0.703 | 0.245 | **0.060** |
| ET (multi-edge) | **0.900** | 0.493 | **0.790** | **0.785** | 0.037 |
| RAT (multi-edge) | 0.900 | 0.676 | 0.668 | 0.540 | 0.028 |
| EpiGNN-mul (BCE) | 0.520 | 0.604 | 0.491 | 0.156 | 0.009 |
| NBFNet (BCE) | 0.576 | 0.531 | 0.460 | 0.153 | 0.009 |
| R-GCN | 0.347 | 0.672 | 0.283 | 0.051 | 0.032 |

### Unambiguous Test Sets

| Model | In-dist-na | D-na | BL-na | OPEC-na |
|-------|-----------|------|-------|---------|
| ET (single-edge) | 0.800 | 0.822 | 0.104 | 0.110 |
| ET (multi-edge) | 0.800 | 0.494 | 0.056 | 0.077 |
| EpiGNN-mul (BCE) | 0.539 | 0.716 | 0.027 | 0.045 |

### LRM Evaluation (o3, with world rules provided)

- At OPEC=0: o3 performs near-perfectly.
- At OPEC=3, depth=7: o3 accuracy **drops to 0**.
- o4-mini performs slightly below o3.

### Ablation: ET Performance Under Controlled Variables

- Controlling for BL and width, performance drops sharply from depth 4 (>0.8) to depth 6 (0.2–0.6).
- This confirms that "easy" test instances along a single dimension mask true difficulty.

### Key Findings

1. **OPEC is the most challenging dimension**: All models approach complete failure (<10%) on Test-OPEC, including o3 with explicit rules provided.
2. **BL is equally challenging**: Under the unambiguous setting, all models achieve <11% on BL-na.
3. **Ambiguity is surprisingly manageable** (but shortcuts exist): Performance on Test-W is reasonable, though further analysis reveals models exploit shortcuts.
4. **GNN-based methods perform poorly overall**: Even on in-distribution sets, they fall far behind ET due to strong path bias.
5. **ET is the best-performing model but still far from adequate**: The best OPEC accuracy is only 6%.

## Highlights & Insights

1. **Precise quantification of reasoning difficulty**: The proposed four-dimensional framework (Depth/Width/BL/OPEC) is systematic and generalizable to other reasoning tasks.
2. **Empirical force of breaking the path assumption**: The failure of all models—including the LRM o3—on off-path reasoning is striking.
3. **Rigor of benchmark construction**: NoRA v1.1 guarantees compositional generalization feasibility through recursive splicing.
4. **Independent control of multiple dimensions**: Each test split extrapolates along only one difficulty axis, avoiding confounding effects.

## Limitations & Future Work

1. Random sampling makes it difficult to obtain instances with simultaneously high values across multiple difficulty dimensions.
2. Metrics such as Depth and BL are sensitive to how knowledge bases are encoded.
3. Ambiguity does not pose a genuine challenge due to shortcuts; the generation mechanism for ambiguous facts requires improvement.
4. Additional GNN variants (e.g., higher-order GNNs, subgraph GNNs) were not evaluated.
5. Reasoning rules are fixed—a more general setting should allow the rule set to vary.

## Related Work & Insights

- **CLUTRR** [Sinha et al., 2019]: The classic family relationship reasoning benchmark, in which all reasoning is path-based.
- **STaR** [Khalid & Schockaert, 2025]: A spatiotemporal reasoning benchmark requiring multi-path composition but remaining primarily path-based.
- **Edge Transformer** [Bergen et al., 2021]: A Transformer with triangular attention, representing the most general architecture for systematic reasoning.
- **EpiGNN** [Khalid & Schockaert, 2025]: A GNN supporting multi-path composition that outperforms ET on HetioNet.
- **Dziri et al. (2023)**: Argues for fundamental limitations of LLMs in compositional reasoning.

## Rating

⭐⭐⭐⭐⭐

An outstanding benchmark contribution. The four-dimensional difficulty quantification is precise and insightful. The most significant contribution is the empirical demonstration that all existing models—including LRMs—fundamentally fail at off-path reasoning, providing a clear direction for the development of novel reasoning architectures. NoRA has the potential to become the new standard for evaluating relational reasoning, succeeding CLUTRR.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)

<!-- RELATED:END -->
