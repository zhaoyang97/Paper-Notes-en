---
title: >-
  [Paper Note] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation
description: >-
  [Optimization] This paper proposes the SLOGAN framework, which decouples causal and spurious features through sparse causal graph construction and information bottlenecks. It incorporates a generative intervention mechanism utilizing cross-domain spurious feature swapping alongside class-adaptive dynamic pseudo-label calibration to achieve stable causal feature transfer in unsupervised graph domain adaptation.
tags:
  - "Optimization"
date: 2026-05-08
content_hash: 0ab8920b9fd76905
---

# Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation

- **Conference**: ICML 2025
- **arXiv**: [2507.07621](https://arxiv.org/abs/2507.07621)
- **Code**: Not publicly available
- **Area**: Graph domain adaptation / Causal discovery / Stable learning
- **Keywords**: Unsupervised graph domain adaptation, sparse causal modeling, generative intervention, pseudo-label calibration, mutual information bottleneck

## TL;DR

This paper proposes the SLOGAN framework, which decouples causal and spurious features through sparse causal graph construction and information bottlenecks. It incorporates a generative intervention mechanism utilizing cross-domain spurious feature swapping alongside class-adaptive dynamic pseudo-label calibration to achieve stable causal feature transfer in unsupervised graph domain adaptation.

## Background & Motivation

Unsupervised Graph Domain Adaptation (UGDA) aims to leverage labeled source-domain graph data to achieve superior performance in an unlabeled target domain. Existing methods face two core challenges:

**Entanglement of Causal and Spurious Features**: Graph data naturally contains both causal relationships and statistical correlations. Traditional methods train models relying solely on semantic labels, failing to differentiate between the two. For instance, in the PTC toxicity prediction dataset, the molecular skeleton is the causal factor for carcinogenicity, whereas experimental variables like gender and species are merely statistical correlations. Without explicit decoupling, spurious factors disturb cross-domain generalization through residual correlations.

**Failure of Global Alignment Strategies**: Existing adversarial learning schemes employ global domain distribution alignment strategies, which easily cause information collapse by discarding crucial but rare substructures, while failing to effectively suppress spurious factors. High-dimensional sparsity and structural complexity of graph data further exacerbate these issues.

These limitations motivate the authors to design explicit causal-spurious decoupling and local stability preservation mechanisms.

## Method

### Overall Architecture

SLOGAN consists of three complementary stability-enhancing components:
1. Feature decoupling based on sparse causal discovery
2. Progressive stable alignment discriminative learning with confidence calibration
3. Generative intervention mechanism with covariance constraints

### Key Design 1: Sparse Causal Discovery and Feature Decoupling

**Structural Causal Model (SCM) Construction**: A sparse causal graph is constructed, where $L$ represents labels, $C$ represents causal features, $S$ represents spurious features, and $PL$ represents pseudo-labels. Stability is achieved through three key causal pathways:
- Sparse feature generation: $C \rightarrow \mathcal{G} \leftarrow S$
- Label stability: $L \rightarrow C \leftarrow PL$
- Domain sparsity: $D^{so} \rightarrow S \leftarrow D^{ta}$

**Stability-Aware Decoupling Learning**: Based on the principle of Sparse Variable Independence (SVI), the optimization objective is formulated as:

$$\underbrace{\max I(Y; Z^c)}_{\text{stable prediction}} - \underbrace{\beta I(Z^s; Z)}_{\text{spurious control}} + \underbrace{\min I(Z^s; Y)}_{\text{spurious suppression}}$$

**Causal Feature Extraction**: InfoNCE is utilized to maximize the mutual information between causal features $\mathbf{z}^c$ and labels $\mathbf{y}$:

$$\min \mathcal{L}_{MI}^c = \mathbb{E}_{p(\mathbf{z}^c, \mathbf{y})}[\xi] - \log \mathbb{E}_{p(\mathbf{z}^c)p(\mathbf{y})}[e^\xi]$$

where $\xi = F^c(\mathbf{z}^c, \mathbf{y}) = {\mathbf{z}^c}^T W \mathbf{y}$ is a bilinear mapping.

**Spurious Feature Suppression**: A Variational Information Bottleneck (VIB) is employed to minimize the mutual information between spurious features and labels while controlling residual information:

$$\min \mathcal{L}_{MI}^s = I(\mathbf{z}^s, \mathbf{y}) - \beta I(\mathbf{z}^s, \mathbf{z})$$

### Key Design 2: Unbiased Discriminative Learning

To mitigate the issue of overconfident pseudo-labels, a class-adaptive dynamic calibration strategy is designed:

1. Compute target-domain predictive distributions and confidence scores based on causal features: $s_i^{ta} = \max_c \mathbf{p}_i^{ta}[c]$
2. Calculate class-adaptive coefficients $\mathcal{M}_c$ and thresholds $\tau_c = \mathcal{M}_c \cdot \tau$ ($\tau = 0.95$)
3. Construct a confident sample set $\mathcal{C}$ for joint cross-domain optimization

### Key Design 3: Generative Intervention and Covariance Constraints

A two-layer MLP is adopted as the generative model $G(\cdot, \cdot)$, optimized via $L_2$ distance-based reconstruction:

$$\mathcal{L}_{ge} = \mathbb{E} \|\mathbf{z} - G(\mathbf{z}^c, \mathbf{z}^s)\|_2^2$$

**Cross-Domain Spurious Feature Swapping**: Spurious features from different domain samples are swapped within mini-batches to generate combined samples $\mathbf{z}_i^{+k} = G(\mathbf{z}_i^c, \mathbf{z}_k^s)$, forcing the model to reconstruct relying solely on causal features.

**Intervention Invariance Constraint**:

$$\mathcal{L}_{inv} = \mathcal{L}_{re} + \mathbb{E}_{\mathbf{z}_i \in \mathcal{B}^{so}, \mathbf{z}_k \in \mathcal{B}^{ta}} \|\mathbf{z}_i^{+k} - \mathbf{z}_i\|^2$$

### Loss & Training

Overall optimization objective:

$$\mathcal{L} = \mathcal{L}_{sup} + \gamma \mathcal{L}_{dis} + \eta \mathcal{L}_{inv}$$

where $\mathcal{L}_{sup} = \mathcal{L}_{so} + \mathcal{L}_{ta}$, $\gamma = 0.003$, and $\eta = 0.1$. The model is first warmed up on the source domain before integrating target-domain optimization.

## Theoretical Guarantees

**Theorem 3.1**: Under the stable causal graph construction, when the three conditions of causal sufficiency, spurious suppression, and generative intervention are satisfied, the target domain error is bounded with high probability by:

$$\epsilon_T(h) \leq \hat{\epsilon}_S(h) + C\sqrt{\epsilon_1} + L\sqrt{\epsilon_2} + C(n_S, \delta)$$

That is, the target-domain error is jointly bounded by the source-domain empirical error, the degree of spurious correlation suppression, and reconstruction accuracy.

## Key Experimental Results

### Main Results

| Dataset | SLOGAN Avg. | MTDF (SOTA) Avg. | Gain |
|--------|------------|------------------|------|
| PTC | **67.8%** | 65.5% | +2.3% |
| NCI1 | **70.6%** | 69.5% | +1.1% |
| Twitter | **64.7%** | 64.2% | +0.5% |
| Letter-Med | **73.5%** | 71.3% | +2.2% |

Across a total of 48 domain transfer scenarios on 4 benchmark datasets, SLOGAN consistently achieves state-of-the-art or competitive performance.

### Ablation Study

| Variant | Avg. | Decrease |
|------|------|------|
| Full Model | 70.6% | - |
| w/o $\mathcal{L}_{sup}$ | 67.5% | -3.1% |
| w/o $\mathcal{L}_{inv}$ | 69.5% | -1.1% |
| w/o $\mathcal{L}_{dis}$ | 69.6% | -1.0% |
| Baseline (GCN) | 63.3% | -7.3% |

Each component contributes to the final performance, with $\mathcal{L}_{sup}$ exhibiting the most significant impact.

### Key Findings

1. **Visualization**: t-SNE visualizations demonstrate that causal features align consistently across domains (domain-agnostic) and are clearly separated by semantic labels.
2. **Scalability**: SLOGAN achieves superior performance while maintaining minimal parameter overhead and latency.
3. **Hyperparameter Robustness**: The framework is robust to variations in $\gamma$ and $\eta$, demonstrating stable performance.

## Highlights & Insights

1. **Causal Perspective on the UGDA Problem**: In contrast to global alignment strategies, decomposing features from a causal inference formulation is theoretically more sound.
2. **Innovative Generative Intervention**: Breaking local spurious couplings by swapping cross-domain spurious features is more fine-grained than conventional adversarial learning.
3. **Class-Adaptive Pseudo-Label Calibration**: Effectively mitigates error propagation issues in long-tailed or imbalanced scenarios.
4. **Rigorous Theoretical Guarantees**: Provides a probabilistic upper bound on the target domain error.

## Limitations & Future Work

1. **Limited Experimental Scale**: The benchmark datasets (e.g., PTC, NCI1) are relatively small-scale, lacking evaluation on ultra-large-scale graph data.
2. **Strong Assumptions**: The structure of the causal graph needs to be pre-determined, whereas real-world causal mechanisms can be substantially more complex.
3. **Limited to Graph-level Classification**: The proposed method is not yet extended to other graph mining tasks such as node classification and link prediction.
4. **Insufficient Computational Efficiency Analysis**: Although claims of scalability are made, detailed execution-time comparisons against baselines are missing.

## Related Work & Insights

- **Graph Domain Adaptation**: Global alignment methods like CoCo and MTDF, and domain-discrepancy minimization methods like DUA and DARE-GRAM.
- **Causal Discovery and Decoupling**: Causal representation learning schemes such as IRM (Arjovsky et al., 2019) and IDEA (Wang et al., 2024a).
- **Graph Neural Networks Classification**: Message-passing architectures including GCN, GIN, and GAT.

## Rating

⭐⭐⭐⭐ (4/5)

The methodology design is comprehensive, integrating causal decoupling, generative intervention, and pseudo-label calibration as a cohesive triad. The theoretical analysis are sound and the experiments are extensively covered. However, the benchmark datasets are relatively small, showing modest improvements (0.5%–2.3%), and the strong structural priors on the causal graph may restrict the model's generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](../../NeurIPS2025/optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[ICML 2025\] GCAL: Adapting Graph Models to Evolving Domain Shifts](gcal_adapting_graph_models_to_evolving_domain_shifts.md)

</div>

<!-- RELATED:END -->
