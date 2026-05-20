---
title: >-
  [Paper Note] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference
description: >-
  [NeurIPS 2025][Optimization][Entity Alignment] This paper proposes NeuSymEA, a neuro-symbolic reasoning framework based on a variational EM algorithm that unifies symbolic rule reasoning and neural network embeddings wit…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Entity Alignment"
  - "Neuro-symbolic Reasoning"
  - "Variational Inference"
  - "Knowledge Graph"
  - "Markov Random Field"
date: 2026-05-08
content_hash: f1f094977a1b19d3
---

# NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference

**Conference**: NeurIPS 2025
**arXiv**: [2410.04153](https://arxiv.org/abs/2410.04153)  
**Code**: [GitHub](https://github.com/chensyCN/NeuSymEA-NeurIPS25)  
**Area**: Optimization
**Keywords**: Entity Alignment, Neuro-symbolic Reasoning, Variational Inference, Knowledge Graph, Markov Random Field

## TL;DR

This paper proposes NeuSymEA, a neuro-symbolic reasoning framework based on a variational EM algorithm that unifies symbolic rule reasoning and neural network embeddings within a Markov Random Field for entity alignment, achieving significant performance gains and low-resource robustness on DBP15K.

## Background & Motivation

Entity Alignment (EA) aims to merge two knowledge graphs (KGs) by identifying equivalent entity pairs. Existing methods fall into two categories:

- **Symbolic models** (e.g., PARIS): Rule-based reasoning that is precise and interpretable, but performs poorly on low-degree nodes and structurally heterogeneous subgraphs, leading to low recall.
- **Neural models** (e.g., GCN-based): Retrieve similar entities via embedding spaces, but struggle to distinguish similar representations as the entity pool grows, resulting in degraded precision and lack of interpretability.

Existing neuro-symbolic methods (PRASE, EMEA) simply concatenate the two model types without a unified optimization objective. Furthermore, the search space for cross-KG rules grows exponentially with rule length, making efficient inference a major challenge.

## Method

### Overall Architecture

NeuSymEA models the truth scores of all candidate entity pairs as a joint probability distribution in a Markov Random Field, constrained by a set of weighted rules, and optimizes iteratively via a variational EM algorithm. The E-step parameterizes truth scores with a neural model and infers missing alignments; the M-step updates rule weights based on observed and inferred alignments.

### Key Designs

1. **Variational EM Framework Unifying Symbolic and Neural Models**: Entity alignment is formalized as a probabilistic inference problem. Each entity pair $(e, e')$ is associated with a binary variable $v_{(e,e')}$, and the objective is to maximize the log-likelihood of observed alignments $\log p_w(v_O)$. Since direct optimization is intractable, the ELBO lower bound is optimized instead. In the E-step, rule weights $w$ are fixed and the neural model $q_\theta$ approximates the posterior; in the M-step, $q_\theta$ is fixed and rule weights are updated. The core innovation lies in incorporating both model types into a single shared optimization objective.

2. **Efficient Optimization via Logical Deduction**: The search space for rules of length $L$ grows exponentially. NeuSymEA exploits logical deduction to decompose long rules into a sequence of unit-length sub-rules. Each inference step then only requires aggregating alignment probabilities from neighbors, weighted by relation pattern $\eta(r)$ (measuring relation uniqueness) and sub-relation probability $p_{sub}(r \subseteq r')$. Parameter complexity is linear in dataset size, and computational complexity is quadratic.

3. **Interpretable Explainer**: Through a reverse rule decomposition process, the Explainer extracts long rule paths as explicit evidence for each alignment prediction and recovers rule weights as quantified confidence scores. Two modes are supported: hard-anchor mode (using only pre-aligned anchor pairs) and soft-anchor mode (including inferred anchor pairs), with the latter providing richer explanations.

### Loss & Training

- **Pseudo-label Filtering in the E-step**: After the neural model computes matching scores for all latent pairs, a greedy one-to-one matching strategy is applied—positive samples are labeled in descending order of confidence, skipping any entity already assigned, effectively reducing false positives.
- **Threshold $\delta$ Control**: Pairs for which the symbolic model predicts a probability exceeding $\delta$ are treated as positive samples; the remainder serve as negative sampling candidates.
- **Hyperparameter Search**: $\delta \in \{0.6, \ldots, 0.99\}$; EM iterations range from 1 to 9.

## Key Experimental Results

### Main Results: DBP15K Full Version

| Category | Model | JA-EN Hit@1 | FR-EN Hit@1 | ZH-EN Hit@1 | ZH-EN MRR |
|----------|-------|------------|------------|------------|-----------|
| Neural | GCNAlign | 0.221 | 0.205 | 0.189 | 0.271 |
| Neural | BootEA | 0.454 | 0.443 | 0.486 | 0.600 |
| Neural | Dual-AMN | 0.627 | 0.652 | 0.650 | 0.732 |
| Neural | LightEA | 0.736 | 0.782 | 0.725 | 0.779 |
| Symbolic | PARIS | 0.589 | 0.618 | 0.603 | — |
| Neuro-symbolic | PRASE | 0.611 | 0.647 | 0.652 | — |
| **Ours** | **NeuSymEA-D** | **0.806** | **0.827** | **0.801** | **0.843** |
| **Ours** | **NeuSymEA-L** | 0.781 | **0.834** | 0.785 | 0.825 |

NeuSymEA-D achieves a Hit@1 improvement of **7.6%** over the strongest baseline LightEA on ZH-EN.

### Low-Resource Experiments (JA-EN Condensed Version)

| Model | 1% Hit@1 | 5% Hit@1 | 10% Hit@1 | 20% Hit@1 |
|-------|---------|---------|----------|----------|
| AlignE | 0.007 | 0.080 | 0.244 | 0.433 |
| PARIS | 0.145 | 0.340 | 0.450 | 0.565 |
| Dual-AMN | 0.239 | 0.509 | 0.652 | 0.750 |
| EMEA | 0.411 | 0.630 | 0.688 | 0.736 |
| **NeuSymEA-D** | 0.481 | 0.692 | 0.742 | 0.835 |
| **NeuSymEA-L** | **0.632** | **0.733** | **0.773** | **0.858** |

With only 1% seed alignments, NeuSymEA-L achieves a Hit@1 of 0.632, substantially outperforming all baselines. On FR-EN, 73.7% Hit@1 is attained with only 1% seed alignments.

### Key Findings

- **Advantage of a Unified Optimization Objective**: NeuSymEA jointly optimizes symbolic and neural models within a single probabilistic framework rather than simply concatenating them, consistently surpassing PRASE and EMEA.
- **Complementary Properties of Full vs. Condensed**: Neural models suffer substantial performance drops on the Full version (more low-degree entities), while symbolic models improve (more long-tail entities with increased connectivity); both NeuSymEA variants remain robust across settings.
- **Fast Convergence**: During EM iterations, the number of rule-inferred pairs grows steadily with high precision, and neural model MRR converges within a few rounds.
- **Scalability**: NeuSymEA remains operative on DBP1M with millions of entities and outperforms LightEA at that scale.

## Highlights & Insights

- This work is the first to extend variational EM from knowledge graph completion to cross-KG entity alignment, designing cross-KG weighted rules and Markov Random Field joint probability modeling.
- Logical deduction decomposition reduces the complexity of long-rule inference from exponential to linear, representing a key innovation in reasoning efficiency.
- The Explainer design makes entity alignment transparent rather than a black box, providing traceable rule paths and confidence scores for each prediction.
- Performance in low-resource settings is particularly impressive—a substantial lead over all baselines with only 1% seed alignments—demonstrating the effectiveness of symbolic reasoning in alleviating cold-start problems.

## Limitations & Future Work

- The computational complexity of the symbolic reasoning component is quadratic; although mitigated via parallelization and batching, this remains a bottleneck for very large-scale KGs.
- Only structural information is currently utilized; entity names, attributes, and other side information are not incorporated, leaving a gap relative to recent methods that exploit multi-modal information.
- Rule confidence is computed as a product, which naturally yields lower confidence for longer rules and may cause valuable long-range alignment evidence to be overlooked.

## Related Work & Insights

- Applications of variational EM to KG completion (e.g., pLogicNet) provide the theoretical foundation for this work, though cross-KG rule design and dual-graph structural inference represent non-trivial extensions.
- Rule mining ideas from symbolic methods such as PARIS are integrated into the unified framework, exemplifying a complementary neuro-symbolic fusion strategy.
- Compared to the pseudo-label iterative strategy of EMEA, the unified objective function offers stronger convergence guarantees.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of unifying neuro-symbolic reasoning via variational EM is original, and the logical deduction decomposition makes a theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers Full/Condensed datasets, low-resource settings, large-scale KGs, interpretability analysis, and convergence analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and experimental design is well-structured.
- **Value**: ⭐⭐⭐⭐ — Offers practical advances for entity alignment, with strong applicability in low-resource scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] Brain-like Variational Inference](brain-like_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

</div>

<!-- RELATED:END -->
