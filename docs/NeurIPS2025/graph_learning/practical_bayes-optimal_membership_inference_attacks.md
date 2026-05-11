---
title: >-
  [Paper Note] Practical Bayes-Optimal Membership Inference Attacks
description: >-
  [NeurIPS 2025][Graph Learning][Membership Inference Attack] This paper proposes BASE and G-BASE, two practical Bayes-optimal membership inference attack methods targeting i.i.d. data and graph-structured data…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Membership Inference Attack"
  - "Bayes-Optimal"
  - "Graph Neural Networks"
  - "Privacy"
  - "MIA"
date: 2026-05-08
content_hash: 3a83ff2bd12d6d55
---

# Practical Bayes-Optimal Membership Inference Attacks

**Conference**: NeurIPS 2025

**arXiv**: [2505.24089](https://arxiv.org/abs/2505.24089)

**Code**: None

**Area**: Graph Learning / Privacy & Security

**Keywords**: Membership Inference Attack, Bayes-Optimal, Graph Neural Networks, Privacy, MIA

## TL;DR

This paper proposes BASE and G-BASE, two practical Bayes-optimal membership inference attack methods targeting i.i.d. data and graph-structured data, respectively, achieving theoretical optimality while substantially reducing computational cost.

## Background & Motivation

Membership Inference Attacks (MIA) aim to determine whether a given data sample was used to train a target model, serving as an important tool for evaluating privacy leakage risks in machine learning. Existing MIA methods (e.g., LiRA, RMIA) have achieved notable performance but suffer from the following limitations:

**High computational cost**: LiRA requires training a large number of shadow models to estimate the loss distributions of members and non-members, incurring substantial computational overhead.

**Lack of theoretical foundation for graph data**: Node-level MIA against Graph Neural Networks (GNNs) lacks theoretical analysis of optimal query strategies.

**Gap between theory and practice**: The existing Bayesian decision-theoretic framework (Sablayrolles et al.) applies only to i.i.d. data and has not been extended to graph-structured data.

The core motivation of this paper is to derive optimal inference rules for graph data within the Bayesian decision-theoretic framework and to design computationally efficient practical approximations.

## Method

### Overall Architecture

This paper formulates MIA as a hypothesis testing problem under the Bayesian decision-theoretic framework. Given a target model $\theta$ and a query sample $x$, the goal is to determine whether $x$ belongs to the training set in the sense of minimizing the expected error rate.

### Key Designs

**1. BASE (Bayes-optimal Approximation for Statistical Estimation)**

- Uses the loss value of the target model on the query sample as the test statistic
- Estimates the loss distributions of members and non-members via Gaussian approximation
- Requires only a small number of reference models (no need to train shadow models per query point)
- Makes decisions by comparing the likelihood ratio against a threshold

**2. G-BASE (Graph-aware BASE)**

- Extends the Bayes-optimal inference rule to graph-structured data
- Accounts for dependency relationships among nodes in the graph and derives the optimal query strategy for node-level MIA
- Demonstrates that the optimal strategy is to query the $k$-hop neighborhood of the target node, rather than the target node alone
- Exploits graph structure to estimate the joint distribution of node features

**3. Equivalence between BASE and RMIA**

- Proves that BASE is equivalent to RMIA under specific hyperparameter settings
- Provides a theoretical justification for RMIA from a Bayes-optimal perspective

### Loss & Training

- The attacker does not train models; instead, loss statistics from reference models are utilized
- Gaussian approximation is employed to estimate $P(\ell | \text{member})$ and $P(\ell | \text{non-member})$
- The decision rule is based on the log-likelihood ratio: $\Lambda(x) = \log \frac{P(\ell | \text{member})}{P(\ell | \text{non-member})}$

## Key Experimental Results

### Main Results

Attack performance comparison on i.i.d. datasets (TPR@1%FPR):

| Method | CIFAR-10 | CIFAR-100 | Purchase | Texas |
|--------|----------|-----------|----------|-------|
| LiRA | 3.2% | 8.5% | 5.1% | 7.3% |
| RMIA | 3.4% | 8.7% | 5.3% | 7.5% |
| BASE | **3.5%** | **8.9%** | **5.4%** | **7.6%** |

Node-level MIA performance on graph datasets (AUC):

| Method | Cora | CiteSeer | PubMed | Flickr |
|--------|------|----------|--------|--------|
| Classifier-based MIA | 0.62 | 0.58 | 0.55 | 0.61 |
| LiRA (node-level) | 0.68 | 0.64 | 0.60 | 0.66 |
| G-BASE | **0.73** | **0.69** | **0.65** | **0.71** |

### Ablation Study

Effect of the number of reference models on BASE performance (CIFAR-100, TPR@1%FPR):

| # Reference Models | 2 | 4 | 8 | 16 | 64 |
|--------------------|---|---|---|----|----|
| BASE | 7.8% | 8.3% | 8.7% | 8.8% | 8.9% |
| LiRA | 5.2% | 6.8% | 7.9% | 8.3% | 8.5% |

### Key Findings

1. BASE with only 4 reference models matches the performance of LiRA with 64 models, yielding approximately a 16× improvement in computational efficiency.
2. G-BASE significantly outperforms classifier-based node-level MIA across all graph datasets.
3. The optimal graph query strategy does involve the neighborhood of the target node, validating the theoretical derivation.

## Highlights & Insights

- **Theoretical contribution**: The paper is the first to derive Bayes-optimal MIA rules for graph-structured data, filling a notable theoretical gap.
- **Strong practicality**: BASE matches or surpasses the state of the art while reducing computational cost by an order of magnitude.
- **Unified perspective**: The established equivalence between BASE and RMIA provides a theoretical grounding for existing methods.

## Limitations & Future Work

1. The Gaussian approximation assumption may be insufficiently accurate under certain distributions.
2. G-BASE is primarily validated on small-scale graph datasets; scalability to large-scale graphs remains to be verified.
3. The work focuses solely on node classification tasks and has not been extended to other GNN tasks such as link prediction.

## Related Work & Insights

- **LiRA** (Carlini et al.): A likelihood-ratio-based MIA requiring a large number of shadow models.
- **RMIA** (Zarifzadeh et al.): A reference-model-based MIA, proven in this paper to be equivalent to BASE.
- **Sablayrolles et al.**: Proposed the Bayesian decision-theoretic framework for MIA, which this paper extends to graph-structured data.

## Rating

- ⭐ Novelty: 8/10 — Extending the Bayes-optimal framework to graph data represents a significant theoretical contribution.
- ⭐ Value: 9/10 — Substantially reduces computational cost while maintaining competitive performance.
- ⭐ Writing Quality: 8/10 — Theoretical derivations are clear and experiments are thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[NeurIPS 2025\] GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection](graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[NeurIPS 2025\] FastJAM: a Fast Joint Alignment Model for Images](fastjam_a_fast_joint_alignment_model_for_images.md)

</div>

<!-- RELATED:END -->
