---
title: >-
  [Paper Note] A Recipe for Causal Graph Regression: Confounding Effects Revisited
description: >-
  [ICML 2025][Graph Learning][Causal Graph Learning] This paper systematically extends causal graph learning from classification to regression tasks for the first time. By acknowledging the predictive capacity of confounding subgraphs via an Enhanced Graph Information Bottleneck (Enhanced GIB) and substituting discrete-label-dependent causal intervention methods with contrastive learning, the proposed method significantly outperforms existing approaches on graph-level OOD regre…
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Causal Graph Learning"
  - "Graph Regression"
  - "Confounding Effects"
  - "Graph Information Bottleneck"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 916f037fc0726017
---

# A Recipe for Causal Graph Regression: Confounding Effects Revisited

**Conference**: ICML 2025  
**arXiv**: [2507.00440](https://arxiv.org/abs/2507.00440)  
**Code**: [causal-graph/CGR](https://github.com/causal-graph/CGR)  
**Area**: Graph Learning  
**Keywords**: Causal Graph Learning, Graph Regression, Confounding Effects, Graph Information Bottleneck, Contrastive Learning

## TL;DR

This paper systematically extends causal graph learning from classification to regression tasks for the first time. By acknowledging the predictive capacity of confounding subgraphs via an Enhanced Graph Information Bottleneck (Enhanced GIB) and substituting discrete-label-dependent causal intervention methods with contrastive learning, the proposed method significantly outperforms existing approaches on graph-level OOD regression benchmarks.

## Background & Motivation

Causal Graph Learning (CGL) enhances GNN generalization in out-of-distribution (OOD) scenarios by identifying causal subgraphs, which has achieved success in classification tasks. However, **graph-level regression tasks**—a more challenging setting—have been almost entirely neglected. Existing CGL methods cannot be directly transferred to regression for two core reasons:

**Unavailable discrete labels**: The transition from finite support to infinite support prevents graphs from being grouped by category. Backdoor adjustment in CAL and counterfactual reasoning in DisC both rely on discrete label information.

**Overly strong assumption on confounding subgraphs**: Existing methods (e.g., CAL, DisC) assume that the confounding subgraph $S$ **completely lacks predictive information**. However, in practical scenarios (e.g., in molecular property prediction, where molecular weight is strongly correlated with toxicity although not causally related), this assumption is clearly unreasonable.

The authors verify the failure of "direct adaptation" in their experiments: simply adapting classification-based CGL methods to regression results in OOD performance that is even worse than the most basic ERM (Empirical Risk Minimization).

## Method

### Overall Architecture

The framework follows the Structural Causal Model (SCM) $G \to C, G \to S, C \to Y, S \to C$. The overall process consists of four steps:

1. **Graph Encoding**: The GNN encoder computes the graph embedding $H_g$ for the input graph $G = (\mathbf{A}, \mathbf{X})$.
2. **Subgraph Separation**: An attention module generates soft masks $\mathbf{M}_{\text{edge}}, \mathbf{M}_{\text{node}}$ to extract the causal subgraph $C$ and confounding subgraph $S = G - C$.
3. **Dual-path Encoding**: Two parameter-sharing GNN modules $\mathcal{G}_c, \mathcal{G}_s$ process $C$ and $S$ respectively, and output regression predictions through readout layers.
4. **Contrastive Intervention**: Randomly combining $H_{c,i} + H_{s,j}$ generates counterfactual mixed representations $H_{\text{mix}}$ to perform causal intervention via contrastive loss.

The total loss is: $L = L_{\text{GIB}} + \lambda L_{\text{CI}}$

### Key Designs

#### Design 1: Enhanced Graph Information Bottleneck (Enhanced GIB)

The standard GIB objective is $-I(C;Y) + \alpha I(C;G)$, which only compresses redundant information in the causal subgraph $C$. This paper argues that this ignores the predictive capability of the confounding subgraph $S$, causing the model to cram all $Y$-related information into $C$, which results in incomplete separation.

The **enhanced objective** introduces a mutual information term for the confounding subgraph:

$$L_{\text{GIB}} = -I(C;Y) + \alpha I(C;G) - \beta I(S;Y)$$

where $\beta I(S;Y)$ explicitly encourages $S$ to retain certain predictive information to avoid overloading $C$. Note that the $I(S;G)$ term is intentionally excluded because $S$ mainly introduces shortcut correlations, and imposing excessive structural regularization on it would undermine the separation effect.

**Variational Approximation Upper Bound**:

- $I(C;G)$: Assuming $p(C|G) = \mathcal{N}(\mu_\phi(G), I)$, $q(C) = \mathcal{N}(0, I)$, the KL divergence simplifies to $\frac{1}{2}\|\mu_\phi(G)\|^2$.
- $I(C;Y)$: Modeling $p(Y|H_c) = \mathcal{N}(Y; \mu_{(c)}, \sigma^2_{(c)})$, which degenerates to the least squares loss after assuming a constant variance $\sigma^2 = 1$.
- $I(S;Y)$: Similarly degenerates to the least squares loss of the confounding subgraph.

The actual computed loss is:

$$L_{\text{GIB}} = \underbrace{\frac{1}{N}\sum_i(Y_i - \mu_{(c),i})^2}_{L_c: \text{Causal Regression Loss}} + \alpha \cdot \underbrace{\frac{1}{2}\mathbb{E}[\|\mu_\phi(G)\|^2]}_{\text{Compression Regularize}} + \beta \cdot \underbrace{\frac{1}{N}\sum_i(Y_i - \mu_{(s),i})^2}_{L_s: \text{Confounding Regression Loss}}$$

#### Design 2: Contrastive Learning-based Causal Intervention

Existing causal interventions (backdoor adjustments) rely on stratifying $P(Y|C,S)$ by category, which is infeasible under continuous $Y$. This paper replaces **class separation** with **instance discrimination**, utilizing contrastive learning to achieve label-independent causal intervention.

**Counterfactual Sample Construction**: Randomly pair the representation of the $i$-th causal subgraph with that of the $j$-th confounding subgraph:

$$H_{\text{mix},ij} = H_{c,i} + H_{s,j}$$

**Contrastive Loss (InfoNCE style)**: The original graph $H_{g,i}$ and its mixed graph $H_{\text{mix},ij}$ serve as a positive pair, while other graphs $H_{g,k}$ serve as negative pairs:

$$L_{\text{CI}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\text{sim}(H_{g,i}, H_{\text{mix},ij}))}{\sum_{k \neq i}\exp(\text{sim}(H_{g,i}, H_{g,k}))}$$

**Core Intuition**: If the causal subgraph $C_i$ is correctly extracted, then regardless of which confounding subgraph $S_j$ it is paired with, the mixed representation should align with the original graph representation—because the prediction is determined by the causal part. This forces the model to learn causal representations that are **invariant to confounding variations**.

#### Design 3: Mask Generation Mechanism

The causal/confounding subgraphs are separated via learnable soft masks:

$$C = (\mathbf{M}_{\text{edge}} \odot \mathbf{A},\ \mathbf{M}_{\text{node}} \cdot \mathbf{X})$$

$\mathbf{M}_{\text{edge}} \in [0,1]^{n \times n}$ and $\mathbf{M}_{\text{node}}$ (diagonal elements $\in [0,1]$) are generated by an MLP conditioned on the representation of $G$, and optimized end-to-end.

### Loss & Training

The final training objective is a weighted combination of three parts:

$$L = L_c + \alpha \cdot L_{\text{reg}} + \beta \cdot L_s + \lambda \cdot L_{\text{CI}}$$

- $L_c$: MSE regression loss for the causal subgraph (main task)
- $L_{\text{reg}}$: $\ell_2$ regularization for causal embeddings (information compression)
- $L_s$: MSE regression loss for the confounding subgraph (acknowledging the predictive power of confounders)
- $L_{\text{CI}}$: InfoNCE contrastive loss (causal intervention)
- Hyperparameters $\alpha, \beta, \lambda$ control the weights of each term.

## Key Experimental Results

### Main Results

OOD MAE on the GOOD-ZINC dataset (lower is better):

| Method | Scaffold-OOD | Size-OOD | Type |
|------|-------------|----------|------|
| ERM | 0.1660 | 0.1248 | Baseline |
| IRM | 0.2313 | 0.1245 | Invariant Learning |
| VREx | 0.1561 | 0.1271 | Invariant Learning |
| DANN | 0.1734 | 0.1289 | Domain Adaptation |
| Coral | 0.1734 | 0.1260 | Domain Adaptation |
| CIGA | 0.2986 | 0.2415 | Causal Classification Method |
| DIR | 0.3650 | 0.2619 | Causal Classification Method |
| **CGR (Ours)** | **Best** | **Best** | Causal Regression |

**Key Findings**: When causal methods designed specifically for classification (CIGA, DIR) are directly applied to regression, their OOD performance is **far worse than even the ERM baseline**, validating the motivation of this study.

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| W/o $\beta I(S;Y)$ | OOD performance drops significantly | Ignoring confounder predictive power leads to insufficient separation |
| W/o $L_{\text{CI}}$ | OOD performance drops | Lack of causal intervention reduces generalization capability |
| Replacing contrastive loss with label supervision | Performs worse than the contrastive version | Label supervision has limited effectiveness under continuous labels |
| W/o $I(C;G)$ regularization | Redundant information in causal subgraph | Compression regularization is necessary for precise separation |
| Full model | Best OOD | The three components are complementary |

### Key Findings

1. **Classification causal methods are unsuitable for regression**: CIGA and DIR perform far worse than simple ERM on regression, proving that regression requires dedicated designs.
2. **Confounding subgraphs indeed possess predictive power**: Performance improves after adding $\beta I(S;Y)$, indicating that the "confounder = pure noise" assumption does not hold in regression.
3. **Contrastive learning outperforms label supervision**: In continuous label scenarios, unsupervised contrastive loss of the InfoNCE style is more suitable for causal intervention than supervised methods.

## Highlights & Insights

- **Key Insight**: Overturning the strong assumption of the classification era that "confounding subgraphs have no predictive power" to "confounding subgraphs have predictive power but are non-causal" is better suited for real regression scenarios.
- **Technical Bridge**: Replacing label-stratified backdoor adjustment in classification with the instance discrimination principle of contrastive learning serves as a natural generalization of causal inference from classification to regression.
- **Theoretical Elegance**: Simplifying the enhanced GIB objective into two MSE losses + an $\ell_2$ regularization via variational approximation yields a concise implementation backed by solid theory.
- **Generality**: The framework is independent of specific GNN backbones and can be embedded into any graph neural network architecture.

## Limitations & Future Work

1. **Multiple hyperparameters**: The three weights $\alpha, \beta, \lambda$ need tuning, which makes sensitivity analysis relatively costly.
2. **Constant variance assumption**: The assumption $\sigma^2 = 1$ for $p(Y|H_c)$ is made for simplification; modeling heteroscedasticity might yield further improvements.
3. **Simple mixing strategy**: The additive combination of $H_{\text{mix}} = H_c + H_s$ might not be optimal; more complex fusion strategies are worth exploring.
4. **Prior on causal subgraphs**: Currently, data-driven soft masks are used; incorporating domain knowledge (e.g., molecular functional groups) could improve separation quality.
5. **Limited to graph-level regression**: Node-level or edge-level regression tasks are not addressed.

## Related Work & Insights

- **CAL/CAL+** (Sui et al., 2022/2024): A backdoor adjustment framework; this paper inherits its subgraph separation architecture but improves the loss design.
- **DisC** (Fan et al., 2022): Counterfactual reasoning that relies on discrete labels, which this paper replaces with contrastive learning.
- **GSAT** (Miao et al., 2022): An explainable GNN with stochastic attention sampling, sharing a common GIB theoretical foundation.
- **CIGA** (Chen et al., 2022): Invariant subgraph learning; this paper's experiments show it is not applicable to regression.
- **GIB** (Wu et al., 2020): Graph Information Bottleneck principle; this paper introduces the confounding mutual information term on top of it.
- Insight: The transfer of causal learning from classification to regression is not merely a replacement of loss functions but requires revisiting the modeling assumptions of confounding effects.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first to systematically address causal graph regression with two precise, core improvements (Enhanced GIB + Contrastive Intervention).
- Theoretical Thoroughness: ⭐⭐⭐⭐ — Complete variational approximation derivation; self-consistent motivation and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple OOD benchmarks with sufficient ablations and comprehensive baseline coverage.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, consistent notation, and highly informative tables/figures.
- Value: ⭐⭐⭐⭐ — Generic framework with open-source code, applicable to real-world scenarios like molecular property prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICLR 2026\] Diverse and Sparse Mixture-of-Experts for Causal Subgraph–Based Out-of-Distribution Graph Learning](../../ICLR2026/graph_learning/diverse_and_sparse_mixture-of-experts_for_causal_subgraphbased_out-of-distributi.md)
- [\[ICLR 2026\] Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs](../../ICLR2026/graph_learning/beyond_entity_correlations_disentangling_event_causal_puzzles_in_temporal_knowle.md)
- [\[ICML 2025\] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [\[ICML 2025\] Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via Contextual Stochastic Block Models](graph_attention_is_not_always_beneficial_a_theoretical_analysis_of_graph_attenti.md)

</div>

<!-- RELATED:END -->
