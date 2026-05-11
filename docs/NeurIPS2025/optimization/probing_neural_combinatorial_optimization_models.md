---
title: >-
  [Paper Note] Probing Neural Combinatorial Optimization Models
description: >-
  [NeurIPS 2025][Optimization][Neural Combinatorial Optimization] This work is the first to systematically introduce probing methodology into the study of neural combinatorial optimization (NCO) models. It proposes a CS-Pr…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Neural Combinatorial Optimization"
  - "Interpretability"
  - "Probing Analysis"
  - "Embedding Representations"
  - "Generalization"
date: 2026-05-08
content_hash: 09a7b57cb4d60e90
---

# Probing Neural Combinatorial Optimization Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.22131](https://arxiv.org/abs/2510.22131)
**Code**: [GitHub](https://github.com/123zhangzq/NeurIPS2025_probing/)
**Area**: Optimization
**Keywords**: Neural Combinatorial Optimization, Interpretability, Probing Analysis, Embedding Representations, Generalization

## TL;DR

This work is the first to systematically introduce probing methodology into the study of neural combinatorial optimization (NCO) models. It proposes a CS-Probing framework to analyze the decision-relevant knowledge, inductive biases, and generalization mechanisms encoded in model representations, and identifies critical embedding dimensions that can be leveraged to improve model generalization.

## Background & Motivation

NCO methods have achieved near-optimal or even superior performance compared to specialized heuristics on classical combinatorial optimization problems such as TSP and CVRP. However, what knowledge these Transformer-based deep models internalize and how they make decisions remains largely opaque.

**Limitations of Prior Work**: The lack of interpretability in NCO models impedes both academic research and practical deployment. Researchers and stakeholders need a deeper understanding of internal mechanisms, including: (1) what decision-relevant knowledge is learned by the model, and (2) how such knowledge is acquired and utilized.

**Limitations of Existing Approaches**: Traditional interpretability methods such as gradient attribution, visualization, and neuron ablation each have significant shortcomings. Gradient attribution only reveals output sensitivity to inputs and cannot expose how knowledge is encoded in hidden representations; neuron ablation lacks statistical rigor.

**Key Insight**: This paper draws inspiration from the probing methodology that has proven successful in NLP — if a simple linear model can accurately predict certain knowledge from model embeddings, that knowledge is considered to be encoded in the representations. However, unlike NLP, combinatorial optimization lacks naturally defined sub-tasks, necessitating the deliberate design of probe tasks. The paper further proposes CS-Probing, which obtains deeper insights by analyzing the absolute values and statistical significance of linear probe coefficients.

## Method

### Overall Architecture

1. Design hierarchical probe tasks for combinatorial optimization problems (low-level + high-level).
2. Apply linear probe models to test whether NCO embeddings encode the corresponding knowledge.
3. Apply CS-Probing to identify which specific embedding dimensions contribute most significantly.

Three representative Transformer-based NCO models are studied: AM (Attention Model), POMO, and LEHD.

### Key Designs

1. **Probe Task Design**: Two levels of tasks are designed for TSP —

   - **Task 1 (Low-Level)**: Can the model perceive Euclidean distances? A linear probe regresses pairwise Euclidean distances from embeddings, with $R^2$ as the evaluation metric.
   - **Task 2 (High-Level)**: Can the model avoid greedy myopia? This task assesses whether the model has learned to avoid always selecting the nearest node, measured by AUC.
   - **Tasks 3/4 (CVRP)**: Understanding constraint relationships (demand additivity) and route membership information.

   Key findings: Initial embeddings (after linear projection) fail to linearly capture Euclidean distances ($R^2 \approx 0$), but $R^2$ increases substantially after attention layers, reaching 0.94 for LEHD. All models achieve AUC > 0.8 on Task 2, indicating that globally optimal, non-myopic knowledge has been learned.

2. **CS-Probing (Coefficient Significance Probing)**: The core methodological contribution. This approach analyzes the absolute value and statistical significance (p-value) of each embedding dimension's coefficient in the linear probe model, with false discovery rate (FDR) controlled at 0.05 via the Benjamini–Hochberg procedure. This deepens the analysis from "does the embedding encode a given type of knowledge?" to "which dimensions encode what knowledge?"

   Key findings: LEHD (the best-performing model) exhibits a clearly sparse activation pattern — fewer than 20 dimensions show strong activations (absolute values reaching tens), while AM and POMO exhibit more diffuse activations (absolute values < 4). LEHD's candidate-node embeddings contain significantly more statistically significant dimensions compared to its current-node embeddings.

3. **Generalization Mechanism Analysis**: CS-Probing reveals a direct relationship between generalization ability and embedding dimension reuse. When models generalize from 20 nodes to 100 nodes, LEHD consistently reuses the same top-2 dimensions (dimensions 31 and 69 for distance perception; dimensions 31 and 97 for avoiding myopia), whereas the critical dimensions for AM and POMO shift during generalization. This provides direct structural evidence that "generalization ability stems from consistent knowledge encoding."

### Loss & Training

Probe models use linear regression (Task 1) or linear classification (Task 2), with NCO model parameters frozen and only probe parameters trained. The NCO models are trained according to their original training procedures (RL or SL). Motivated by CS-Probing insights, L1 regularization $\lambda \|\mathbf{h}\|_1$ is added to LEHD to encourage embedding sparsity and improve generalization.

## Key Experimental Results

### Main Results (Probe Performance)

| Probe Input | Task 1 ($R^2$) | Task 2 (AUC) | Notes |
|-------------|----------------|--------------|-------|
| AM-Init | -0.0003 | 0.49 | Initial embeddings encode no information |
| AM-Enc-l3 (w/ ints) | 0.9282 | 0.83 | Encoder output contains rich knowledge |
| POMO-Enc-l6 (w/ ints) | 0.7917 | 0.86 | Similar pattern |
| LEHD-Dec-l6 (w/o ints) | 0.9418 | 0.86 | High $R^2$ without interaction terms |
| LEHD-Dec-l6 (w/ ints) | 0.9415 | 0.86 | LEHD representations are richest |

### Ablation Study (Critical Dimension Verification)

| Configuration | TSP100 Optimality Gap | Notes |
|---------------|-----------------------|-------|
| LEHD original (all 128 dims) | 0.57% | Baseline |
| Retain only dims 31, 97 | 0.65% | Two dimensions nearly replicate full performance |
| Retain only dims 32, 97 | 0.75% | Replacing one dimension degrades performance |
| Retain only dims 21, 123 | 183.80% | Random dimensions completely fail |
| Zero out dims 31, 97 | 60.43% | Removing critical dimensions causes collapse |
| Zero out dim 98 | 0.59% | Removing non-critical dimension has no effect |
| Zero out dims 98, 126 | 0.62% | High-activation but non-critical dimensions are also unimportant |

| Regularization Strength λ | TSP100 | TSP200 (Generalization) | TSP1000 (Generalization) |
|---------------------------|--------|-------------------------|--------------------------|
| 0 (original) | 0.57% | 0.86% | 3.17% |
| 1e-6 | 0.58% | 0.88% | **2.87%** |
| 1e-4 | 0.57% | **0.73%** | 2.97% |

### Key Findings

- **Hierarchical Knowledge Encoding**: Shallow layers learn spatial relationships (distance perception); deeper layers develop strategic reasoning (avoiding myopia).
- **Training Dynamics**: AM and POMO rapidly develop myopia-avoidance ability in early training; LEHD exhibits this capability from the very beginning of training.
- **Only 2 dimensions suffice to nearly reproduce full model performance** — suggesting that the effective representational space of NCO models is extremely compressed.
- **Simple regularization inspired by probing insights improves generalization**: the optimality gap on TSP1000 decreases from 3.17% to 2.87%.

## Highlights & Insights

- **Pioneering Contribution**: This is the first systematic application of probing methodology to NCO, filling a gap in NCO interpretability research.
- **Generalizability of CS-Probing**: The framework is not limited to Transformer-based models; it extends to GNNs (on JSSP) and diffusion models (DIFUSCO).
- **The "2 dimensions suffice" finding is striking**: only 2 out of 128 embedding dimensions contain nearly all decision-relevant information, suggesting substantial representational redundancy in NCO models and providing a theoretical basis for model compression and distillation.
- **Direct evidence for generalization mechanisms**: Unlike prior work that indirectly analyzes performance differences, CS-Probing provides a structural explanation for generalization success or failure through dimension reuse rates.

## Limitations & Future Work

- The study primarily focuses on TSP and CVRP; probe task design for more complex problems (scheduling, bin packing, etc.) remains to be explored.
- The regularization experiments, while effective, are preliminary; more systematic model improvements guided by probing insights warrant further investigation.
- CS-Probing relies on the linear separability assumption and may miss knowledge encoded in non-linear ways.
- Only three NCO models are analyzed; applicability to more recent architectures (e.g., Mamba, state space models) has yet to be verified.

## Related Work & Insights

- Methodologically inspired by probing research in NLP, with combinatorial-optimization-specific innovations in task design.
- Complements NCO generalization research (e.g., LEHD's original claim of a "dynamic learning strategy"), with CS-Probing providing quantitative validation of that claim.
- Suggested direction: probing analysis could become a standard evaluation tool for the NCO community, analogous to established benchmarks in NLP.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of probing into NCO; CS-Probing is an original methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Experiments are extremely comprehensive, covering multiple models, tasks, ablations, and cross-distribution generalization.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though some content requires navigation to the appendix.
- Value: ⭐⭐⭐⭐⭐ A paradigm-level contribution to the NCO community, inaugurating a new direction in NCO interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)
- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)

</div>

<!-- RELATED:END -->
