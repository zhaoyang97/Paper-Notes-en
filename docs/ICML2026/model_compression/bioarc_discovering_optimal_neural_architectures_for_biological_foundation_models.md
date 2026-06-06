---
title: >-
  [Paper Note] BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models
description: >-
  [ICML 2026][Model Compression][Biological Foundation Models] BioArc proposes a heterogeneous neural architecture search framework for biological foundation models. By automatically discovering optimal hybrid architecture…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Biological Foundation Models"
  - "Neural Architecture Search"
  - "Heterogeneous Search Space"
  - "DNA/Protein Modeling"
  - "Hybrid Architecture"
date: 2026-05-08
content_hash: f86785bb892dc9a3
---

# BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2512.00283](https://arxiv.org/abs/2512.00283)  
**Code**: None  
**Area**: Scientific Computing / Neural Architecture Search  
**Keywords**: Biological Foundation Models, Neural Architecture Search, Heterogeneous Search Space, DNA/Protein Modeling, Hybrid Architecture  

## TL;DR

BioArc proposes a heterogeneous neural architecture search framework for biological foundation models. By automatically discovering optimal hybrid architectures within a search space comprising five basic modules (CNN, LSTM, Transformer, Mamba, and Hyena), it surpasses existing SOTA biological foundation models with less than 1/25 of the parameters.

## Background & Motivation

**Background**: Current biological foundation models (e.g., ESM, Nucleotide Transformer, DNABERT-2) almost exclusively adopt the Transformer architecture. However, Transformers were originally designed for human language and are not naturally adapted to the unique "grammar" of biological sequences.

**Limitations of Prior Work**: Biological sequences present dual challenges: they require processing extremely long contexts (e.g., whole genomes), where the quadratic complexity of standard Transformers is prohibitive; and they require precise capture of local structural motifs, which global attention mechanisms do not naturally prioritize. Furthermore, the biological domain lacks a universally accepted optimal architectural paradigm like NLP, leading to a heavy reliance on manual intuition for architecture design.

**Key Challenge**: Biological information arises from natural evolution, and its underlying physicochemical laws are not yet fully revealed. Consequently, architecture design cannot be guided by a prior understanding of language as in NLP. Moreover, architecture selection is deeply coupled with tokenization strategies and training objectives, preventing isolated optimization.

**Goal**: (1) Automatically discover optimal biological sequence architectures within a heterogeneous search space; (2) systematically decouple the interaction between architecture, tokenization, and training strategies; (3) verify whether the discovered architectures can hierarchically capture biological grammar.

**Key Insight**: Existing NAS methods are limited to homogeneous spaces (e.g., only tuning CNN layers/channels) or hybrids of at most two types of modules. This work expands the search space to open-ended combinations of five heterogeneous modules, allowing data-driven discovery of optimal topologies.

**Core Idea**: Utilize heterogeneous NAS to search within a combinatorial space of CNN, LSTM, Transformer, Mamba, and Hyena modules to discover optimal hybrid architectures for DNA and proteins.

## Method

### Overall Architecture

The BioArc pipeline consists of four stages: (1) Search space design—defining five basic module types and their depth/width combinations; (2) Supernet construction—encoding all candidate architectures into a weight-shared supernet, where each path represents a candidate; (3) Supernet pre-training—employing a one-shot strategy where a path is randomly sampled for self-supervised pre-training during each forward pass; (4) Evaluation and ranking—independently fine-tuning sampled architectures and aggregating rankings via Z-score normalization to select the optimal architecture for full pre-training as a foundation model.

### Key Designs

1.  **Heterogeneous Search Space and Representative Sampling**:
    - **Function**: Construct a search space containing CNN, LSTM, Transformer, Mamba, and Hyena modules, supporting open-ended combinations.
    - **Mechanism**: Each path $a = (l_1, l_2, \dots, l_d)$ is defined by depth $d \in \mathcal{D}$, a module type tuple $\mathbf{m}$, and a hidden dimension tuple $\mathbf{h}$. Due to the uneven distribution of topological families in heterogeneous spaces, naive sampling overfits to redundant patterns. BioArc selects 360 representative architectures using a three-step strategy: distance filtering based on log-transformed dimensions to remove topological redundancy; imposing monotonic width constraints (fixing the widest layer at the end) to increase the sampling frequency of parameter-intensive modules; and finally using K-Means clustering to select cluster centers.
    - **Design Motivation**: The combinatorial explosion of heterogeneous spaces makes exhaustive search impossible; representative sampling maintains structural diversity while keeping the search scale computationally feasible.

2.  **One-Shot Supernet Pre-training**:
    - **Function**: Simultaneously train all candidate architectures within a single weight-shared supernet.
    - **Mechanism**: During each forward pass, a path $a \sim \mathcal{A}$ is sampled uniformly at random, and the shared weights corresponding to that path are updated to minimize the self-supervised loss: $\min_W \mathbb{E}_{a \sim \mathcal{A}}[\mathcal{L}(\mathcal{A}(X; w(a)))]$. A single supernet supports three pre-training objectives: Masked Modeling (MM), Contrastive Learning (CL), and Next Token Prediction (NTP). Random path sampling acts as a natural regularizer, preventing excessive co-adaptation between modules.
    - **Design Motivation**: Compared to training each architecture independently, the one-shot strategy allows the evaluation of all candidates within a single pre-training run, significantly reducing computational costs.

3.  **Z-score Normalized Cross-task Architecture Ranking**:
    - **Function**: Fairly aggregate rankings across multiple heterogeneous downstream tasks to select the optimal architecture.
    - **Mechanism**: Raw performance $P_t(a)$ for each task $t$ is standardized using Z-score and averaged: $\text{Score}(a) = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} s_t \cdot \frac{P_t(a) - \mu_t}{\sigma_t}$, where $s_t \in \{1, -1\}$ aligns the optimization direction. During ranking, individual paths are fine-tuned independently rather than using the supernet weights to eliminate interference from shared parameters.
    - **Design Motivation**: Different tasks use varying metrics (e.g., Accuracy vs. RMSE); raw summation would bias towards tasks with wider numerical distributions. Z-score normalization ensures each task contributes equally to the final ranking.

## Key Experimental Results

### Main Results (DNA Experiment on GUE benchmark, 12 tasks)

| Method | Parameters | TFP-0 | TFP-1 | TFP-2 | TFP-3 | TFP-4 | CPD-all | CPD-notata | CPD-tata |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NT-2500M | 2500M | 66.31 | 68.30 | 58.70 | 49.08 | 67.59 | 67.39 | 67.46 | 69.66 |
| DNABERT-2 | 117M | 71.99 | 76.06 | 66.52 | 58.54 | 77.43 | 69.37 | 68.04 | 74.17 |
| VQDNA | 103M | 72.48 | 76.43 | 66.85 | 58.92 | 78.10 | 71.02 | 70.58 | 78.50 |
| **Ours (mask-ft)** | **4.89M** | **84.80** | **86.00** | **85.80** | **77.10** | **89.20** | **83.60** | **85.43** | **89.40** |
| **Ours (con-ft)** | **3.28M** | **84.80** | **86.10** | **86.50** | **77.50** | **89.30** | **83.53** | **84.66** | **90.05** |

BioArc outperforms all baselines across all DNA tasks, with parameter counts between 1/24 and 1/36 of DNABERT-2, achieving performance gains of 12-19 percentage points.

### Main Results (Protein Experiment on PEER benchmark, controlled comparison)

| Method | Parameters | Solubility | HumanPPI | PPIAffinity↓ | Fold | Subcellular | Binary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ESM-2 8M (Official) | 8M | 73.48 | 80.16 | 3.098 | 22.14 | 71.47 | 91.25 |
| ESM-2 8M (Reprod.) | 8M | 71.84 | 74.68 | 3.567 | 18.25 | 70.36 | 90.68 |
| **Ours 8M** | **8M** | **73.29** | **76.79** | **2.756** | **20.75** | **72.77** | **91.82** |

Under identical pre-training conditions (full UniRef50, 50K steps), BioArc 8M outperforms ESM-2 8M on all 6 protein tasks, demonstrating architectural superiority beyond mere pre-training scale.

### Key Findings

| Finding | Details |
| :--- | :--- |
| Optimal DNA Architecture Pattern | Hyena (Long-range) → Transformer (Contextual) → CNN (Local feature extraction) |
| Shared Architectures for Similar Tasks | Top 10% architectures for TFP-3 and TFP-4 show 98.0% similarity |
| Foundation Model Performance | BioArc-F surpasses DNABERT-2 with 1/20 of the parameters and 1/10 of the training steps |
| Tokenization & Architecture Coupling | Transformers prefer 6-mer, CNNs prefer 1-mer; joint optimization is necessary |
| Pre-training Strategy No Universal Winner | Masked modeling is overall best, but training from scratch is superior for some tasks |
| Interpretability Validation | Hybrid architectures hierarchically capture promoter grammar: Hyena establishes global context → Transformer anchors TSS → CNN detects Inr+DPE coordination |

## Highlights & Insights

- First to construct a NAS search space containing five heterogeneous module types in the biological sequence domain, breaking the previous limitation of mixing only two types.
- Discovered architectures significantly outperform baselines 100x their size with minimal parameters (3-8M), demonstrating the importance of architectural inductive bias.
- Systematically decoupled the interaction between architecture, tokenization, and training strategies, providing actionable design principles for biological models.
- Interpretability analysis shows that hybrid architectures spontaneously learn hierarchical representations corresponding to known biological mechanisms.

## Limitations & Future Work

- Validated only on DNA and protein modalities; the generalization to RNA and single-cell data remains unknown.
- The search space is fixed to 360 representative architectures, potentially missing promising topological combinations.
- Absolute performance on protein structure prediction (e.g., Fold task) still lags behind large-scale pre-trained models like ESM-2, indicating that architectural advantages cannot fully substitute for data scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)
- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)
- [\[ICML 2026\] Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice](auditing_and_fixing_economic_validity_in_tabular_foundation_models_for_discrete_.md)
- [\[ICML 2026\] UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models](ub-smoe_universally_balanced_sparse_mixture-of-experts_for_resource-adaptive_fed.md)

</div>

<!-- RELATED:END -->
