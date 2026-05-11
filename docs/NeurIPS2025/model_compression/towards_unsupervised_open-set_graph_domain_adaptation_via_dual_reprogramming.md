---
title: >-
  [Paper Note] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming
description: >-
  [NeurIPS 2025][Model Compression][Graph Domain Adaptation] This paper proposes GraphRTA, a framework that addresses the challenges of known-class classification and unknown-class detection in unsupervised open-set graph…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Graph Domain Adaptation"
  - "Open-Set Recognition"
  - "Model Reprogramming"
  - "Graph Reprogramming"
  - "Adversarial Learning"
date: 2026-05-08
content_hash: a61dce322d8efeb7
---

# Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming

**Conference**: NeurIPS 2025
**arXiv**: [2510.18363](https://arxiv.org/abs/2510.18363)
**Code**: [Available](https://github.com/cszhangzhen/GraphRTA)
**Area**: Model Compression
**Keywords**: Graph Domain Adaptation, Open-Set Recognition, Model Reprogramming, Graph Reprogramming, Adversarial Learning

## TL;DR

This paper proposes GraphRTA, a framework that addresses the challenges of known-class classification and unknown-class detection in unsupervised open-set graph domain adaptation through two complementary mechanisms: model reprogramming (gradient-guided weight pruning) and graph reprogramming (target graph structure and feature optimization), without requiring manually specified thresholds.

## Background & Motivation

Graph neural networks suffer from performance degradation due to distribution shift in cross-domain scenarios. Unsupervised graph domain adaptation aims to transfer knowledge from label-rich source graphs to unlabeled target graphs. Existing methods predominantly focus on the **closed-set** setting, where source and target domains share the same label space — an assumption that rarely holds in practice, as target domains often contain novel categories absent from the source. For example, a fraud detection model trained on known fraud patterns may encounter entirely new fraud types in the target domain.

Existing open-set methods exhibit two fundamental limitations: (1) they rely on manually specified entropy thresholds to distinguish known from unknown instances, and a single threshold cannot adapt to varying distributions; (2) they primarily focus on aligning the source domain with the target known group while neglecting explicit separation of the target unknown group, resulting in ambiguous decision boundaries.

GraphRTA's innovation lies in simultaneously reprogramming from both the **model side** and the **data side**, while eliminating threshold dependency.

## Method

### Overall Architecture

GraphRTA consists of three core modules: (1) domain-agnostic model reprogramming — reducing source-domain bias via gradient-guided weight pruning; (2) distribution-aware graph reprogramming — modifying the structure and node features of the target graph to reduce domain shift; (3) three-group domain adversarial learning — explicitly aligning and separating instances grouped as source, target-known, and target-unknown.

### Key Designs

1. **Domain-Agnostic Model Reprogramming**: Inspired by the lottery ticket hypothesis, only a subset of parameters is critical for cross-domain generalization. A differentiable mask $\mathbf{M}^l$ is introduced to prune the weights $\mathbf{W}^l$ at each layer: $\mathbf{Z}^l = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{Z}^{l-1}(\mathbf{W}^l \odot \mathbf{M}^l))$. The absolute value of gradients serves as an importance score, and the lowest $\rho$% of weights are zeroed out. The design motivation is that weights with small gradients capture domain-specific patterns; pruning them allows the model to focus on transferable features.

2. **Threshold-Free Classifier Extension**: An additional output dimension is appended to the classifier for the unknown class: $g_\phi(\mathbf{z}) = [\boldsymbol{\phi}^\top \mathbf{z}, \hat{\boldsymbol{w}}^\top \mathbf{z}]$. The extended logits are passed through a softmax layer to produce posterior probabilities, and the class with the highest probability is the prediction. Unlike conventional methods relying on fixed thresholds, this mechanism makes dynamic decisions based on the input node representation $\mathbf{z}$.

3. **Distribution-Aware Graph Reprogramming**: A transformation function modifies the node features and structure of the target graph: $\hat{\mathbf{X}}_t = \mathbf{X}_t + \Delta\mathbf{X}_t$ (learnable feature perturbation), $\hat{\mathbf{A}}_t = \mathbf{A}_t \oplus \Delta\mathbf{A}_t$ (XOR operation for edge addition/deletion, constrained by budget $\mathcal{B}$). This directly reduces domain shift at the data level, compensating for structural discrepancies that model reprogramming cannot resolve.

4. **Three-Group Domain Adversarial Learning**: Conventional adversarial learning distinguishes only source from target, but in the open-set setting, aligning all target samples causes negative transfer. GraphRTA first estimates the probability of each target node belonging to the known or unknown group using a Beta mixture model with the EM algorithm (no threshold required), then performs adversarial training with three-class domain labels to explicitly push away target-unknown features.

### Loss & Training

The overall loss is $\mathcal{L} = \mathcal{L}_{adv} + \mathcal{L}_{cls} + \mathcal{L}_{ent}$:

- **Adversarial loss** $\mathcal{L}_{adv}$: Cross-entropy for three-class domain classification, implementing a minimax game via the gradient reversal layer (GRL).
- **Classification loss** $\mathcal{L}_{cls}$: Supervised classification on source graphs, plus alignment of non-ground-truth labels to the unknown class (simplified mixup), training the model to explicitly assign mismatched patterns to the unknown category.
- **Entropy minimization loss** $\mathcal{L}_{ent}$: Encourages confident predictions on target instances and discriminative feature generation for the unknown group.

## Key Experimental Results

### Main Results

**Node Classification on Citation Datasets (Table 2, Selected Transfer Directions)**:

| Method | A→C Acc | A→C HS | C→D Acc | C→D HS |
|--------|---------|--------|---------|--------|
| GCN | 40.64 | 41.02 | 51.48 | 56.13 |
| GRADE | 57.23 | 59.49 | 61.94 | 64.21 |
| DANCE | 57.77 | 60.94 | 62.97 | 65.42 |
| G2Pxy | 59.75 | 54.47 | 61.42 | 59.13 |
| SDA | 58.23 | 59.97 | **63.55** | **65.53** |
| **GraphRTA** | **66.26** | **66.33** | 63.87 | 65.99 |

**Large-Scale ogbn-arxiv Dataset (Table 3)**:

| Method | I→II Acc | I→II HS | I→III Acc | I→III HS |
|--------|----------|---------|-----------|----------|
| SAGE | 44.95 | 37.83 | 42.75 | 38.63 |
| A2GNN | 42.07 | 45.00 | 38.92 | 43.14 |
| DANCE | OOM | OOM | OOM | OOM |
| **GraphRTA** | - | - | - | - |

### Ablation Study

As described in the paper, Appendix B provides an ablation comparing different transformation functions in the graph reprogramming module. Key findings:

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| w/o Model Reprogramming | Performance drop | Source-domain bias not removed |
| w/o Graph Reprogramming | Performance drop | Data-level domain shift unresolved |
| w/o Extended Classifier | Requires manual threshold | Degenerates to conventional approach |
| GraphRTA (Full) | Best | Dual reprogramming synergistically complementary |

### Key Findings

- GraphRTA achieves state-of-the-art results on 4 out of 6 citation network transfer directions (Acc) and 5 out of 6 (HS).
- Strong performance on the heterophilic WebKB dataset demonstrates applicability to heterophilic graph structures.
- GraphRTA is architecture-agnostic and can be combined with various GNN backbones including GCN, GAT, and GraphSAGE.
- DANCE runs out of memory on the large-scale ogbn-arxiv dataset, whereas GraphRTA handles it successfully.

## Highlights & Insights

- **Novel dual reprogramming perspective**: Introducing the "reprogramming" concept into graph domain adaptation, simultaneously optimizing from both model and data sides, with a clear and coherent design rationale.
- **Practical threshold-free design**: The Beta mixture model combined with the extended classifier eliminates the dependency on manual thresholds found in conventional open-set methods.
- **Strong architecture agnosticism**: Can be plug-and-play integrated into different GNN architectures.
- The probabilistic treatment of known/unknown group membership via the Beta mixture model is more principled than direct entropy thresholding.

## Limitations & Future Work

- The edge modification budget $\mathcal{B}$ in graph reprogramming must be set manually, remaining a hyperparameter that requires tuning.
- Graph reprogramming requires learning $\Delta\mathbf{X}_t \in \mathbb{R}^{n_t \times f}$, which may incur substantial memory overhead for large-scale target graphs.
- Evaluation is limited to node classification; edge-level and graph-level tasks are not explored.
- No comparison with graph foundation models or pre-trained GNN methods is provided.

## Related Work & Insights

- The model reprogramming approach could be combined with parameter-efficient fine-tuning methods such as LoRA.
- Graph reprogramming overlaps with the graph structure learning literature (SLAPS, GSL), from which more efficient structure optimization methods could be borrowed.
- The three-group domain adversarial learning framework is generalizable to multi-domain scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Dual reprogramming proposed for the first time in graph DA)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers citation networks, ogbn-arxiv, and WebKB)
- Writing Quality: ⭐⭐⭐⭐ (Well-organized, though some ablations are deferred to the appendix)
- Value: ⭐⭐⭐⭐ (Open-set graph DA addresses a practically important problem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](graph_your_own_prompt.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[AAAI 2026\] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](../../AAAI2026/model_compression/earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)

</div>

<!-- RELATED:END -->
