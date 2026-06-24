---
title: >-
  [Paper Note] GCAL: Adapting Graph Models to Evolving Domain Shifts
description: >-
  [ICML 2025][Optimization][Graph Continual Learning] This work proposes Graph Continual Adaptive Learning (GCAL). Through an "adapt and generate memory" bi-level optimization strategy, GCAL utilizes unsupervised domain adaptation via information maximization when GNN models face continuously evolving OOD graph sequences. At the same time, a variational memory graph generation module based on the information bottleneck theory is designed to compress historical graph knowledge…
tags:
  - "ICML 2025"
  - "Optimization"
  - "Graph Continual Learning"
  - "Domain Adaptation"
  - "Catastrophic Forgetting"
  - "Memory Replay"
  - "Information Bottleneck"
date: 2026-05-08
content_hash: 954704f66549ead8
---

# GCAL: Adapting Graph Models to Evolving Domain Shifts

**Conference**: ICML 2025  
**arXiv**: [2505.16860](https://arxiv.org/abs/2505.16860)  
**Code**: [https://github.com/joe817/GCAL](https://github.com/joe817/GCAL)  
**Area**: Optimization  
**Keywords**: Graph Continual Learning, Domain Adaptation, Catastrophic Forgetting, Memory Replay, Information Bottleneck

## TL;DR

This work proposes Graph Continual Adaptive Learning (GCAL). Through an "adapt and generate memory" bi-level optimization strategy, GCAL utilizes unsupervised domain adaptation via information maximization when GNN models face continuously evolving OOD graph sequences. At the same time, a variational memory graph generation module based on the information bottleneck theory is designed to compress historical graph knowledge, effectively mitigating catastrophic forgetting.

## Background & Motivation

While Graph Neural Networks (GNNs) have achieved significant success in social network analysis, bioinformatics, recommendation systems, and other fields, they face critical challenges when dealing with continuously arriving out-of-distribution (OOD) graph data:

**Limitations of Single-Step Adaptation**: Existing graph domain adaptation methods (e.g., MMD, adversarial learning) only handle single-step adaptation scenarios and cannot cope with continuous domain shifts.

**Catastrophic Forgetting**: When the model adapts to a new graph domain, it loses its ability to handle previous graph domains. The authors experimentally verify that the SOTA method EERM suffers from a continuous and severe performance decline across four OOD graph datasets.

**Unlabeled Constraints**: Most existing continual learning methods rely on labeled data for memory selection and replay, but labels are typically unavailable in graph domain adaptation scenarios.

**Research Gap**: There is a lack of continuous adaptation methods with unsupervised memory generation and replay tailored for graph models.

The core motivation is: How to enable graph models to both adapt to new domains and preserve historical domain knowledge without any labels when facing sequentially arriving graph data with different distributions?

## Method

### Overall Architecture

GCAL adopts a bi-level optimization strategy of "Adapt and Generate Memory". When the $t$-th target graph $G_t$ arrives:

1. **Inner Loop (Adapt)**: Use information maximization to adapt the model to the new graph $G_t$, while replaying historical memory graphs from the memory pool to prevent forgetting, updating the model parameters from $\Theta_{t-1} \to \Theta_t$.
2. **Outer Loop (Generate Memory)**: Based on the adapted model $f(\Theta_t)$, leverage a variational memory graph generator to learn a compressed memory graph $\hat{G}_t$, which is added to the memory pool for future replay.

### Key Designs

#### 1. Adaptation with Memory Replay

Since the target domain lacks labels, Information Maximization (IM) is adopted as the self-supervised adaptation objective. The core idea is that an effective model should make highly confident predictions on target data (outputs close to one-hot) while maintaining class diversity. The adaptation loss consists of two terms:

- **Entropy Minimization**: $-\mathbb{E}_{v \sim \mathcal{V}}[\sum_k p_{v,k} \log p_{v,k}]$, which makes model predictions for each node more confident.
- **Diversity Regularization**: $\sum_k \hat{p}_k \log \hat{p}_k$, which prevents all nodes from converging to the same pseudo-label.

During adaptation, the IM loss is applied simultaneously to the new graph $G_t$ and the memory pool $\mathcal{G} = \{\hat{G}_1, ..., \hat{G}_{t-1}\}$:

$$\mathcal{L}_{AMR} = \mathcal{L}_{Adp}(G_t; \Theta_{t-1}) + \sum_{i=1}^{t-1} \mathcal{L}_{Adp}(\hat{G}_i; \Theta_{t-1})$$

#### 2. Variational Memory Graph Generation

Memory graphs must satisfy three criteria: (1) much smaller size than the original graph ($K \ll N_t$); (2) retaining sufficient task-related information; and (3) remaining generalizable across different distributions.

**Information Bottleneck Derivation**: Based on the Graph Information Bottleneck (GIB) principle, the optimization goal is to maximize the mutual information between the memory graph and task signals, while minimizing that between the memory graph and the original graph. By introducing a variational latent variable $Z_t$ and the chain rule, a tractable lower bound is derived (Theorem 3.1):

$$\mathcal{L}(\Phi) \geq \mathbb{E}[\log P_f(\hat{Y}_t|\hat{G}_t)] - \beta \mathbb{E}[\text{KL}(P_g(\hat{G}_t|G_t,Z_t) \| Q(\hat{G}_t))] + \beta \mathbb{E}[\log P_g(\hat{G}_t|G_t,Z_t)]$$

**Variational Graph Generator Architecture**:

- **Variational GNN Encoder**: Maps the input graph $G_t$ to a latent variable distribution $[\mu; \log\sigma] = \text{TopKSelector}(\text{GNN}_{\mu,\sigma}(A_t, X_t))$.
- **TopK Selector**: Scores the node distribution using a trainable parameter $\mathbf{p}$ and selects the top-K most important nodes, achieving graph compression.
- **Node Feature Generation**: Samples from a Gaussian distribution using the reparameterization trick: $\hat{z}_i = \mu_i + \sigma_i^2 \odot \varepsilon$.
- **Edge Generation**: Assuming edges follow independent Bernoulli distributions, an MLP is used to compute edge weights, and differentiable sampling is achieved via Gumbel-Max reparameterization:
    - $w_{i,j} = (\text{MLP}([\hat{z}_i; \hat{z}_j]) + \text{MLP}([\hat{z}_j; \hat{z}_i])) / 2$
    - $a_{i,j} = \text{Sigmoid}((w_{i,j} + \log\frac{\delta}{1-\delta}) / \tau)$

This symmetric design ensures that edges are undirected, while Gumbel-Sigmoid provides a continuous relaxation to enable gradient backpropagation.

### Loss & Training

The overall objective is a bi-level optimization, where the outer loop optimizes the memory graph generator parameters $\Phi$ and the inner loop optimizes the model parameters $\Theta$. The total loss for memory graph generation comprises three terms:

#### (1) Memory Graph Learning Loss $\mathcal{L}_{MGL}$ (Gradient Matching)

Utilizes a gradient matching strategy for graph distillation, aligning the gradient directions of model parameters between the memory graph and the original graph:

$$\mathcal{L}_{MGL} = D\left(\frac{\partial \mathcal{L}_{Adp}(\hat{G}_t; f(\Theta_t))}{\partial \Theta_t}, \frac{\partial \mathcal{L}_{Adp}(G_t; f(\Theta_t))}{\partial \Theta_t}\right)$$

where $D$ represents the sum of cosine distances of layer-wise gradients: $D(\hat{\mathbf{g}}, \mathbf{g}) = \sum_i (1 - \frac{\hat{\mathbf{g}}_i \cdot \mathbf{g}_i}{\|\hat{\mathbf{g}}_i\| \|\mathbf{g}_i\|})$.

#### (2) Regularization Loss $\mathcal{L}_{Reg}$ (KL Divergence)

Constrains the generated distribution to prior distributions. The prior for node features is a standard normal distribution $\mathcal{N}(0, I)$, and the prior for edges is a Bernoulli distribution $\text{Bernoulli}(q)$:

$$\mathcal{L}_{Reg} = \frac{1}{2}\sum_{i,j}(\mu_{i,j}^2 + \sigma_{i,j}^2 - \log\sigma_{i,j}^2 - 1) + \sum_{i,j}(w_{i,j}\log\frac{w_{i,j}}{q} + (1-w_{i,j})\log\frac{1-w_{i,j}}{1-q})$$

#### (3) Generation Loss $\mathcal{L}_{Gen}$ (Distribution Alignment)

Measures the distribution difference between the memory graph and the original graph in the latent representation space of the model using L2 distance:

$$\mathcal{L}_{Gen} = \|\sum_{i=1}^{K} \hat{u}_i(\Theta) - \sum_{i=1}^{N_t} u_i(\Theta)\|_2$$

**Overall Optimization Objective**:

$$\min_{\hat{G}_t, \Phi} \mathcal{L}_{MGL} + \lambda_1 \mathcal{L}_{Reg} + \lambda_2 \mathcal{L}_{Gen} \quad \text{s.t.} \quad \Theta_t = \arg\min_\Theta \mathcal{L}_{AMR}$$

Additionally, an Exponential Moving Average (EMA) strategy is used to smooth model parameter updates. A new generator is used at each time step, and only the generated memory graphs are retained.

## Key Experimental Results

### Main Results

Comparisons with eight baseline methods across four graph datasets (AP = Average Performance, AF = Average Forgetting, ↑ higher is better):

| Dataset | Metric | GCAL | CoTTA (Strongest Baseline) | Gain |
|--------|------|------|-----------------|------|
| Twitch-explicit | AP-AUC/% | **55.65±0.09** | 53.94±0.36 | +1.71 |
| Facebook-100 | AP-ACC/% | **52.72±0.36** | 50.12±0.16 | +2.60 |
| Elliptic | AP-F1/% | **56.57±0.14** | 54.08±0.05 | +2.49 |
| OGB-Arxiv | AP-ACC/% | **45.22±0.17** | 40.28±0.01 | +4.94 |

Regarding the forgetting metric, GCAL achieves positive values (AF > 0) on all four datasets, demonstrating that it not only avoids forgetting but also brings some improvement to old domains after adapting to new ones.

### Ablation Study

| Configuration | Twitch | Facebook | Elliptic | OGB-Arxiv |
|------|--------|----------|----------|-----------|
| w/o $\mathcal{L}_{Reg}$ & $\mathcal{L}_{Gen}$ | 54.03±2.63 | 52.05±0.31 | 46.53±0.01 | 44.70±0.06 |
| w/o $\mathcal{L}_{Reg}$ | 55.34±0.41 | 52.37±0.56 | 55.23±0.32 | 44.76±0.48 |
| w/o $\mathcal{L}_{Gen}$ | 55.37±0.33 | 52.14±0.32 | 55.64±0.57 | 44.91±0.11 |
| w/o EMA | 54.79±0.04 | 47.66±0.06 | 53.83±0.20 | 43.19±0.08 |
| **GCAL (Full)** | **55.65±0.09** | **52.72±0.36** | **56.57±0.14** | **45.22±0.17** |

### Key Findings

1. **Consistently Outperforming SOTA**: GCAL significantly outperforms all baseline methods across all four datasets, with the most pronounced improvement on OGB-Arxiv (+4.94% AP).
2. **Robust Anti-Forgetting Capability**: The AF metrics are all positive, indicating that the model not only avoids forgetting during continuous adaptation but also retroactively improves performance on previous domains.
3. **EMA is a Critical Component**: Removing EMA in the ablation study leads to the largest performance drop (a decrease of 5.06% on Facebook-100), demonstrating that smoothing parameter updates is vital for continuous adaptation.
4. **Simultaneous Removal of $\mathcal{L}_{Reg}$ and $\mathcal{L}_{Gen}$ Causes Severe Degradation**: On the Elliptic dataset, performance drops from 56.57 to 46.53, indicating that the joint force of the three losses derived from information bottleneck theory is indispensable.
5. **Effective with Extremely Low Memory Compression Ratios**: Generating memory graphs with only 1% to 9% of the nodes of the original graph yielded strong results.

## Highlights & Insights

1. **Novel Problem Formulation**: For the first time, this work systematically defines the unsupervised continual adaptation problem for graph models under continuously evolving OOD graph sequences, bridging the gap between graph domain adaptation and continual learning.
2. **Theory-Driven Design**: An optimizable lower bound (Theorem 3.1) is derived from the Information Bottleneck theory, providing a solid theoretical foundation for memory graph generation rather than depending on purely heuristic designs.
3. **Unsupervised Memory Generation**: The method does not rely on any label information to generate memory graphs, instead utilizing information maximization to provide pseudo-supervisory signals, significantly broadening the application scope.
4. **Highly Compressed Experience Pool**: The memory graphs only require 1%–9% of the nodes of the original graphs, resulting in minimal storage and computational overhead.
5. **End-to-End Differentiable Generation**: Nodes are sampled via VAE reparameterization and edges via Gumbel-Sigmoid reparameterization, rendering the entire memory graph generation pipeline fully differentiable.

## Limitations & Future Work

1. **Task Type Restrictions**: Currently, only node classification has been validated, leaving other graph tasks like link prediction and graph classification unexplored.
2. **Constant Class Count Assumption**: It is assumed that the source domain and all target domains share the same set of classes, which is inapplicable to open-world or class-incremental scenarios.
3. **Linear Memory Pool Growth**: The size of the memory pool grows linearly with the number of time steps, leading to cumulative replay overhead over long sequences.
4. **New Generator per Step Required**: A new generator is trained at each time step, failing to reuse generator knowledge across steps.
5. **Limited Dataset Scale**: Experiments were primarily validated on social networks and citation network datasets, lacking verification on larger-scale or structurally more complex graph datasets.

## Related Work & Insights

- **Graph Domain Adaptation**: Representative methods like EERM (Wu et al., 2022b) and GTrans (Jin et al., 2022) train each step independently and cannot prevent forgetting.
- **Continual Test-Time Adaptation**: CoTTA (Wang et al., 2022) uses EMA updates but lacks a memory mechanism; EATA (Niu et al., 2022) employs sample selection but overlooks the graph structure.
- **Graph Distillation/Compression**: The gradient matching strategy originates from Dataset Condensation (Jin et al., 2021). GCAL innovatively applies this to memory graph learning.
- **Information Bottleneck**: Graph Information Bottleneck (Wu et al., 2020b; Sun et al., 2022) inspires the theoretical derivation of memory graphs.
- **Insights**: Combining variational memory generation with gradient-matching distillation is a powerful paradigm worthy of general adoption in other continual learning scenarios.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Deeply integrates continual learning with graph domain adaptation for the first time, backed by complete theoretical derivation. |
| Technical Depth | 4.5 | Rigorous derivation of the information bottleneck lower bound, with a well-designed bi-level optimization and triple-loss system. |
| Experimental Thoroughness | 4 | 4 datasets, 8 baselines, and complete ablation studies, though lacking verification on giant-scale graphs. |
| Writing Quality | 4 | Well-structured, presenting a coherent logic chain from problem definition to methodology and experiments. |
| Value | 3.5 | Code is open-sourced, but the application scenario (unsupervised graph sequence adaptation) is relatively specialized. |
| **Overall** | **4** | A solid theory-driven work that establishes a strong baseline for the new field of graph continual adaptation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICML 2025\] Nonparametric Teaching for Graph Property Learners](nonparametric_teaching_for_graph_property_learners.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](../../CVPR2025/optimization/federated_learning_with_domain_shift_eraser.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](../../NeurIPS2025/optimization/mergebench_a_benchmark_for_merging_domain-specialized_llms.md)

</div>

<!-- RELATED:END -->
