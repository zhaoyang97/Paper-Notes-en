---
title: >-
  [Paper Note] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification
description: >-
  [ICML2025][Graph Learning][GNN] Analyzes the internal mechanisms of the "LLM Enhancer + GNN" paradigm from the perspective of causal mechanism identification, finding that LLM enhancers primarily provide node-level/raw-data-level information, and subsequently proposes an Attention Transmission (AT) module to optimize the information transfer between them.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "GNN"
  - "LLM Enhancer"
  - "Causal Mechanism Identification"
  - "Interchange Intervention"
  - "Attention Transmission Module"
date: 2026-05-08
content_hash: 695bb49af6600211
---

# LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification

**Conference**: ICML2025  
**arXiv**: [2505.08265](https://arxiv.org/abs/2505.08265)  
**Code**: [GitHub](https://github.com/WX4code/LLMEnhCausalMechanism)  
**Area**: Graph Learning / LLM-Enhanced GNN  
**Keywords**: GNN, LLM Enhancer, Causal Mechanism Identification, Interchange Intervention, Attention Transmission Module

## TL;DR

Analyzes the internal mechanisms of the "LLM Enhancer + GNN" paradigm from the perspective of causal mechanism identification, finding that LLM enhancers primarily provide node-level/raw-data-level information, and subsequently proposes an Attention Transmission (AT) module to optimize the information transfer between them.

## Background & Motivation

### Background

**Background**: Rise of the LLM+GNN Paradigm: In recent years, methods that utilize LLMs as feature enhancers to optimize node representations and then input them into GNNs for graph representation learning have achieved significant success. However, the essential properties and internal mechanisms of this paradigm have not yet been deeply studied.

### Limitations of Prior Work

**Limitations of Prior Work**: High Difficulty of Analysis: Both LLM enhancers and GNNs are neural networks. Modeling either in isolation is already challenging, and a unified analysis of their combination is even more difficult.

### Approach

**Core Problem**: What role does the LLM enhancer play in this framework? What level of information do its output features transmit? How can the information transmission from the LLM to GNN be optimized?

### Key Challenge

**Key Challenge**: Methodology Gap: Prior works heavily adopt this paradigm, but lack dedicated deep analytical studies to reveal its underlying working principles.

## Method

### 1. CCSG Synthetic Dataset

The authors construct the **Controlled Causal-Semantic Graph (CCSG)** dataset based on Wikipedia entries, which features the following properties:

- Controllable node features (semantic + manually generated)
- Controllable edge connections and topological structures
- Injectable predefined causal relations
- Covers 3 macro-classes and 15 sub-classes with a total of 5,660 Wikipedia entries
- The total number of combinations reaches up to 226,400, vastly exceeding existing synthetic datasets (e.g., Spurious-Motif has only 36)

### 2. Interchange Intervention Analysis Framework

Based on the **Interchange Intervention** method in causal inference, the core mechanism works as follows:

1. Model the causal relations of the dataset as a high-level causal model $h(\cdot)$
2. Treat the LLM+GNN model as a low-level neural network model $f(\cdot)$
3. Compare the consistency of outputs between the two models by replacing internal variable values

**Interchange Intervention Loss**:

$$\mathcal{L}_{\text{II}} = \frac{1}{|\mathcal{G}|^2} \sum_{G^{\text{orig}}} \sum_{G^{\text{diff}}} \mathcal{D}\big(\text{INTINV}(h, G^{\text{orig}}, G^{\text{diff}}, Z^h),\ \text{INTINV}(f, G^{\text{orig}}, G^{\text{diff}}, Z^f)\big)$$

where $\mathcal{D}(\cdot)$ denotes the cross-entropy loss. Minimizing $\mathcal{L}_{\text{II}}$ identifies a hidden variable $Z^f$ in the neural network that aligns with the causal variable $Z^h$, thereby uncovering the internal logical structure of the model.

**Theoretical Guarantee** (Theorem 3.2): When a bijective mapping exists between $Z^f$ and $Z^h$, minimizing $\mathcal{L}_{\text{II}}$ ensures that the Total Effect of both is consistent. Even if the bijection does not exist, the conclusion still holds as long as the intervention outputs are equal (Corollary 3.3).

### 3. Key Findings

- **Finding 1**: The features output by the frozen-parameter LLM enhancer primarily serve **node-level and raw-data-level** information representation, rather than high-order relation modeling.
- **Finding 2**: After the GNN receives input from the LLM enhancer, its internal logical structure exhibits a **relatively consistent pattern**, which does not change significantly with model scale.
- **Finding 3**: The optimal value of $\mathcal{L}_{\text{II}}$ can **partially reflect model capability**, where a lower optimal value typically indicates a stronger model.

### 4. Attention Transmission (AT) Module

Based on these findings, the authors design a plug-and-play **Attention-based Transmission (AT)** module:

1. Generate $q$ sets of different prompts using the LLM to obtain $q$ groups of feature sets $X^1, X^2, \ldots, X^q$
2. Uniformly sample $m$ token features $S^i = \{s^i_j\}_{j=1}^m$ from each group
3. Compute the attention matrix via a Transformer Encoder: $A^i = Q^i (K^i)^\top$
4. Calculate the average attention score and apply global softmax normalization: $\alpha^i_j = \frac{1}{m}\sum_{l=1}^m A^i_{jl}$
5. Perform weighted aggregation to obtain the final node features: $\mathbf{z} = \frac{1}{qm}\sum_{i=1}^q \sum_{j=1}^m \bar{\alpha}^i_j \mathbf{s}^i_j$

The first $\delta$ epochs of training are used for prompt selection, after which the best prompt is fixed.

## Key Experimental Results

Evaluated on three datasets (Cora, Pubmed, Instagram) using three LLMs (Llama2/Qwen2/Llama3) $\times$ three GNNs (GCN/GAT/GraphSAGE):

### Main Results

| Dataset | LLM | GCN Gain | GAT Gain | GraphSAGE Gain |
|--------|------|---------|---------|---------------|
| Cora | Llama2 | +0.96 | +0.80 | +0.67 |
| Cora | Qwen2 | +2.03 | +1.58 | +1.23 |
| Cora | Llama3 | +1.76 | +1.42 | +1.30 |
| Pubmed | Llama2 | +2.64 | +2.00 | +2.95 |
| Pubmed | Qwen2 | +2.95 | +1.66 | +2.47 |
| Pubmed | Llama3 | +2.37 | +2.38 | +3.09 |
| Instagram | Llama2 | +1.32 | +1.70 | +2.23 |
| Instagram | Llama3 | +1.31 | +1.87 | +1.77 |

- The AT module brings improvements across **all LLM $\times$ GNN combinations**, typically ranging from +0.67 to +3.09.
- The improvement is most significant on Pubmed, with the GraphSAGE + Llama3 combination reaching **+3.09**.
- The impact of the feature selection location on performance $>$ the impact of the LLM backbone $>$ the impact of GNN layer depth.

## Highlights & Insights

1. **Novel Analytical Perspective**: Systems analysis of the LLM+GNN paradigm is conducted for the first time from the perspective of causal mechanism identification, rather than simply stacking experiments.
2. **Dual Support from Theory and Experiment**: Theorem 3.2 and Corollary 3.3 provide rigorous theoretical guarantees for the analytical methodology.
3. **Controllable Synthetic Dataset**: The CCSG dataset is elegantly designed with up to 226K combinations, which far exceeds existing datasets of its kind.
4. **Plug-and-Play**: The AT module requires no modifications to the LLM or GNN architectures, taking effect directly when inserted between them.
5. **Key Insight**: The value of LLM enhancers lies in providing raw node semantics rather than modeling high-order graph relations, clarifying the community's understanding of this paradigm.

## Limitations & Future Work

1. **Analysis Restricted to Frozen-Parameter LLM**: The case of fine-tunable LLMs is not explored, while fine-tuning might alter the LLM's role in the framework.
2. **Synthetic vs. Real-World Data**: The core analysis relies heavily on the synthetic CCSG dataset, whereas causal relations in real-world scenarios are more complex and uncontrollable.
3. **Simple AT Module Design**: Feature selection employs only a standard Transformer attention mechanism, leaving more sophisticated information fusion strategies unexplored.
4. **Limited Dataset Scale**: Experiments are only validated on Cora, Pubmed, and Instagram, lacking evaluations on large-scale heterogeneous graph scenarios.
5. **Modest Performance Gain**: Most improvements lie within 1-3 percentage points, of which the practical value in already high-accuracy tasks warrants further discussion.

## Related Work & Insights

- **Causal Mechanism Identification**: Leverages analytical methods for neural network causal structures from the NLP domain (Geiger et al., 2020/2021).
- **LLM+GNN Paradigm**: Representative works such as TAPE (Chen et al., 2023) and OFA (Liu et al., 2024).
- **Insights**: The proposed analytical framework can be generalized to other "Large Model Enhancing Small Model" scenarios, such as LLM+CNN, LLM+Transformer, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ — The analysis of LLM+GNN from the perspective of causal mechanism identification is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic data analysis is exhaustive, but real-world data coverage is slightly narrow.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and rigorous theoretical derivations, although the abundance of notation may pose a reading barrier.
- Value: ⭐⭐⭐⭐ — Makes important theoretical contributions to understanding the LLM+GNN paradigm, with the AT module showing reasonable practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.md)
- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](../../NeurIPS2025/graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[ICML 2025\] Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes](machines_and_mathematical_mutations_using_gnns_to_characterize_quiver_mutation_c.md)
- [\[ICML 2025\] A Recipe for Causal Graph Regression: Confounding Effects Revisited](a_recipe_for_causal_graph_regression_confounding_effects_revisited.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
