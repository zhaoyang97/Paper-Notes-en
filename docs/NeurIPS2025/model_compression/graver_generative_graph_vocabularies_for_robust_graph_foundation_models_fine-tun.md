---
title: >-
  [Paper Note] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning
description: >-
  [NeurIPS 2025][Model Compression][graph foundation model] This paper proposes Graver, a framework that decouples ego-graphs to extract transferable subgraph vocabularies, models their distributions via graphon experts, and routes relevant vocabularies to augment support samples through MoE-CoE, addressing the instability caused by structural mismatch in few-shot fine-tuning of graph foundation models (GFMs).
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "graph foundation model"
  - "few-shot fine-tuning"
  - "graphon"
  - "vocabulary generation"
  - "MoE"
date: 2026-05-08
content_hash: 8591cf68677d2477
---

# Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning

**Conference**: NeurIPS 2025
**arXiv**: [2511.05592](https://arxiv.org/abs/2511.05592)  
**Code**: [GitHub](https://github.com/RingBDStack/GRAVER)  
**Area**: Model Compression
**Keywords**: graph foundation model, few-shot fine-tuning, graphon, vocabulary generation, MoE

## TL;DR

This paper proposes Graver, a framework that decouples ego-graphs to extract transferable subgraph vocabularies, models their distributions via graphon experts, and routes relevant vocabularies to augment support samples through MoE-CoE, addressing the instability caused by structural mismatch in few-shot fine-tuning of graph foundation models (GFMs).

## Background & Motivation

**Background**: Graph foundation models (GFMs) aim to achieve universal graph learning across domains and tasks via a pre-train-then-fine-tune paradigm. Prior methods such as GCOPE, MDGPT, and SAMGPT have made progress in multi-domain pre-training and cross-domain transfer.

**Limitations of Prior Work**: Few-shot prompt fine-tuning of GFMs suffers from severe instability — performance and adaptation efficiency are highly sensitive to the random selection of support samples. When selected support nodes exhibit structural patterns consistent with the pre-training graphs (e.g., triangles), performance is strong; when patterns diverge (e.g., ladder structures), performance degrades significantly, resulting in high variance.

**Key Challenge**: Under few-shot settings with extremely limited labeled samples (e.g., one-shot), the model cannot adequately capture structural patterns of the target domain, while the transfer of pre-trained knowledge is further constrained by domain discrepancy.

**Goal**: How to achieve robust and efficient GFM fine-tuning under randomly sampled support sets?

**Key Insight**: Generative augmentation — leveraging "graph vocabularies" (transferable subgraph patterns) learned during pre-training to augment support samples, thereby reducing dependence on specific support selections.

**Core Idea**: Extract transferable subgraph vocabularies from pre-training graphs, model their distributions via graphon generators, and use MoE-CoE routing at fine-tuning time to embed relevant vocabularies into support samples for contextual augmentation.

## Method

### Overall Architecture

A three-stage framework:
1. **Pre-training stage**: Multi-domain alignment → ego-graph decoupling for vocabulary extraction → self-supervised contrastive pre-training
2. **Vocabulary modeling stage**: Graphon experts separately model structural tokens and feature tokens
3. **Fine-tuning stage**: MoE-CoE routing generates vocabularies → augments support samples → prompt fine-tuning

### Key Designs

**1. Factor-aware Ego-Graph Decoupling**

The ego-graph of node $u$ is decomposed into $K$ factor-aware subgraphs (vocabularies):
$$\alpha_{v \to k}^{(t)} \propto \text{Softmax}_k(\langle \mathbf{h}_{u,k}^{\mathcal{S}(t)}, \mathbf{h}_{v,k}^{\mathcal{S}(t)} \rangle / \tau)$$

- Neighbors are assigned to $K$ channels via soft routing
- Mutual information regularization ensures semantic independence across channels: $\mathcal{R}_{\text{MI}}^u = \sum_{i \neq j} I(\mathbf{h}_{u,i}^{\mathcal{S}}; \mathbf{h}_{u,j}^{\mathcal{S}})$
- **Proposition 1** (theoretical transferability guarantee): The upper bound on semantic discrepancy is determined by the optimal matching distance over vocabulary combinations

**2. Graphon Generative Experts**

- **Structural tokens**: Per-class adjacency matrices $\{\mathcal{A}_i^{(c)}\}$ are collected to estimate a nonparametric graphon $W_c^{\mathcal{A}}: [0,1]^2 \mapsto [0,1]$
- **Feature tokens**: The feature distribution conditioned on latent positions is estimated as $W_c^{\mathcal{X}}: [0,1] \mapsto \mathbb{R}^d$
- **Conditional generation**: Latent positions $\mathbf{u}_i \sim \mathcal{U}[0,1]$ are sampled, then $\tilde{\mathcal{A}}[i,j] \sim \text{Bern}(W_c^{\mathcal{A}}(\mathbf{u}_i, \mathbf{u}_j))$
- **Proposition 2** (distributional convergence): $\|\mathbf{g}_c^{\text{gen}} - \mathbf{g}_c^{\text{emp}}\|_{\text{TV}} \to 0$ as $N_c \to \infty$

**3. MoE-CoE Routing Network**

A two-level hierarchical routing scheme:
- **MoE layer (which domain)**: $\mathbf{S}_{\text{M}} = \text{Softmax}(\mathbf{W}_{\text{M}}^\top \cdot \phi(\hat{\mathbf{X}}_i^{\mathcal{T}}))$, selects relevant source domains
- **CoE layer (which class)**: $\mathbf{S}_{\text{C}} = \text{Softmax}(\mathbf{W}_{\text{C}}^\top \cdot \phi(\hat{\mathbf{X}}_i^{\mathcal{T}} \| \tilde{\mathcal{X}}))$, combines class-level tokens
- Vocabulary-augmented support samples: $\tilde{G}_i^{\mathcal{T}} = G_i^{\mathcal{T}} \oplus \tilde{\mathbf{g}}_i^{\text{gen}}$

### Loss & Training

- **Pre-training**: Contrastive link prediction + MI regularization: $\mathcal{L}_{\text{pre}} = \mathcal{L}_{\text{contrastive}} + \lambda \sum_u \mathcal{R}_{\text{MI}}^u$
- **Fine-tuning**: Class prototype matching + MoE-CoE sparse activation regularization: $\mathcal{L}_{\text{ftn}} = \mathcal{L}_{\text{cls}} + \mu \cdot \mathcal{L}_{\text{MoE-CoE}}$
- Pre-trained parameters $\Theta^*$ are frozen; only graph prompts $\mathcal{P}_\Omega$ and MoE-CoE weights are updated

## Key Experimental Results

### Main Results: One-shot Node Classification (Cross-Dataset)

| Method | Cora | CiteSeer | PubMed | arXiv | Tech | Home | Wiki-CS |
|--------|------|----------|--------|-------|------|------|---------|
| GCN | 28.40±4.62 | 29.25±3.39 | 40.33±6.90 | 61.59±5.13 | 53.89±3.35 | 36.74±2.53 | 28.58±5.39 |
| SAMGPT | 46.79±6.54 | 38.65±6.35 | 51.92±9.50 | 73.60±— | — | — | — |
| **Graver** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

- Average gain: +2.8% on node classification, +3.2% on graph classification (over runner-up)
- Average relative reduction in standard deviation: **54.0%** (node) and **54.6%** (graph)

### Ablation Study

| Removed Component | Effect |
|-------------------|--------|
| w/o ego-graph decoupling | Performance drops; insufficient vocabulary discriminability |
| w/o graphon generation | Cannot augment support; degenerates to standard prompt tuning |
| w/o MoE-CoE | No selective routing; may introduce negative transfer |
| w/o MI regularization | Channel semantic overlap; reduced transferability |

### Key Findings

- Graver consistently outperforms 15 state-of-the-art baselines in both one-shot and five-shot settings
- Graver's advantage is more pronounced in the harder cross-domain setting, owing to the domain adaptation capability of vocabulary augmentation
- The most significant improvement is in robustness — standard deviation decreases by over 54%, indicating insensitivity to support sample selection
- LLM-enhanced semantic alignment further enables zero-shot transfer

## Highlights & Insights

- **First work to apply generative vocabularies for GFM fine-tuning augmentation**: Unlike approaches that augment training data or graph structure, Graver augments the support set itself
- **Elegant application of graphon theory**: Continuous graphon functions serve as generative models for graph vocabularies, combining theoretical elegance (distributional convergence guarantees) with practical utility
- **Sophisticated MoE-CoE design**: Two-level routing handles "which domain" and "which class" separately, effectively mitigating negative transfer
- **Vivid "root + affix" analogy**: Structural tokens = roots (determine semantic type); feature tokens = affixes (determine domain characteristics)

## Limitations & Future Work

- The vocabulary count $K$ and graphon resolution are critical hyperparameters with limited sensitivity analysis
- Graphon estimation may be unstable on large-scale sparse graphs
- Multi-domain pre-training incurs high computational cost (simultaneously processing 7 datasets with LLM augmentation)
- Ego-graph decoupling requires multiple iterations, adding pre-training overhead
- Evaluation is limited to citation, shopping, and web page domains; scientific domains such as molecular and protein graphs are not assessed

## Related Work & Insights

- GFT's computation-tree vocabularies are restricted to tree structures, whereas Graver's ego-graph vocabularies are more general
- $\mathcal{G}$-Mixup also employs graphons but for data augmentation; Graver uses them for vocabulary modeling
- The approach conceptually aligns with in-context learning in NLP: guiding adaptation by embedding contextual vocabularies
- Insight: Prompt tuning for graphs may require more structural prior injection than text or image counterparts

## Rating

⭐⭐⭐⭐ (4/5)

The method is elegantly designed with solid theoretical support (two Propositions), comprehensive experiments, and significant robustness improvements. However, the overall system complexity is high, which may limit practical applicability; validation across a broader range of domains is also lacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](graph_your_own_prompt.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)

</div>

<!-- RELATED:END -->
