---
title: >-
  [Paper Note] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients
description: >-
  [ICLR2026][AI Safety][personalized federated learning] This paper proposes FedMosaic, a framework addressing dual heterogeneity in personalized federated learning (PFL): RELA measures task relevance via gradient similarity to enable customized aggregation (addressing data heterogeneity), while Co-LoRA enables cross-architecture knowledge sharing (e.g., Llama vs. Qwen) through dimension-invariant modules $P \in \mathbb{R}^{r \times r}, Q \in \mathbb{R}^r$ (addressing model heterogeneity). The framework achieves substantial improvements over SOTA on DRAKE, a newly proposed 40-task multimodal PFL benchmark.
tags:
  - ICLR2026
  - AI Safety
  - personalized federated learning
  - LoRA
  - model heterogeneity
  - multimodal
  - knowledge sharing
date: 2026-05-08
content_hash: 9a8e50d4d7f56978
---

# Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients

**Conference**: ICLR2026  
**arXiv**: [2506.11024](https://arxiv.org/abs/2506.11024)  
**Code**: [https://github.com/snumprlab/fedmosaic](https://github.com/snumprlab/fedmosaic)  
**Area**: AI Safety  
**Keywords**: personalized federated learning, LoRA, model heterogeneity, multimodal, knowledge sharing

## TL;DR
This paper proposes FedMosaic, a framework addressing dual heterogeneity in personalized federated learning (PFL): RELA measures task relevance via gradient similarity to enable customized aggregation (addressing data heterogeneity), while Co-LoRA enables cross-architecture knowledge sharing (e.g., Llama vs. Qwen) through dimension-invariant modules $P \in \mathbb{R}^{r \times r}, Q \in \mathbb{R}^r$ (addressing model heterogeneity). The framework achieves substantial improvements over SOTA on DRAKE, a newly proposed 40-task multimodal PFL benchmark.

## Background & Motivation

**Background**: Personalized federated learning (PFL) enables collaborative learning across clients while preserving privacy. Existing PFL methods such as DITTO and FedDAT address data heterogeneity through dual-adapter designs (local + global), but assume all clients share the same model architecture.

**Limitations of Prior Work**: (a) **Data heterogeneity** — existing benchmarks simulate heterogeneity via non-IID splits of a single dataset, which is unrealistic (real-world clients perform distinct tasks); (b) **Model heterogeneity** — clients operate on different hardware and use different model families (Llama vs. Qwen) and scales (1B vs. 3B), making direct averaging of LoRA matrices infeasible due to differing dimensions; (c) existing methods for model heterogeneity (HETLoRA, FlexLoRA) assume a shared base architecture and differ only in rank, thus failing to support truly heterogeneous architectures.

**Key Challenge**: The LoRA matrices $A \in \mathbb{R}^{r \times d_I}$ and $B \in \mathbb{R}^{d_O \times r}$ depend on model-specific hidden dimensions $d_I, d_O$, which vary across architectures, precluding aggregation. Meanwhile, naively averaging models trained on different tasks causes parameter interference.

**Goal**: Simultaneously address data heterogeneity (clients performing different tasks) and model heterogeneity (clients using different architectures/scales) to enable effective collaboration in realistic multimodal PFL settings.

**Key Insight**: Insert dimension-invariant modules $P$ and $Q$ into the intermediate layer of LoRA — aggregating only these modules enables cross-architecture knowledge transfer.

**Core Idea**: Leverage dimension-invariant Co-LoRA modules for cross-architecture knowledge sharing, combined with gradient-similarity-driven relevance aggregation to reduce task interference.

## Method

### Overall Architecture
Each communication round proceeds as follows: each client locally fine-tunes with Co-LoRA → uploads modules $(P_i, Q_i)$ and sanitized gradients $\tilde{g}_i$ → the server uses RELA to compute inter-client task relevance → a customized global Co-LoRA $G_i$ is constructed for each client → $G_i$ is sent back and frozen during inference. At inference time, the local LoRA and global Co-LoRA are adaptively fused via a learnable gate $\beta$.

### Key Designs

1. **RELA (Relevance-Guided Aggregation)**:

    - **Function**: Constructs a customized global model for each client based on task relevance, rather than naive averaging.
    - **Mechanism**: Each client extracts the last-layer gradient $g_i$ from a small pre-trained model; EMA updates capture distribution shift → Gaussian noise and dimension subsampling are applied for sanitization → the server computes a cosine similarity matrix $S_{ij} = \cos(\tilde{g}_i, \tilde{g}_j)$ → softmax-weighted aggregation: $G_i = \sum_j w_{ij} L_j$.
    - **Design Motivation**: Naively averaging models trained on unrelated tasks causes parameter interference. Sharing knowledge only with relevant clients reduces conflict. EMA gradients reflect the current knowledge state (accounting for forgetting) and are more informative than cumulative gradients.

2. **Co-LoRA (Collaborative LoRA)**:

    - **Function**: Inserts rank-dependent modules between LoRA matrices $A$ and $B$, enabling cross-architecture knowledge sharing.
    - **Mechanism**: $h_O = W_p h_I + B(PA h_I + Q)$, where $P \in \mathbb{R}^{r \times r}$ and $Q \in \mathbb{R}^r$ depend only on rank $r$, allowing direct aggregation of $P$ and $Q$ across architectures. During training, $A$ and $B$ are frozen (preserving alignment); only $P$ and $Q$ are updated.
    - **Block-wise Aggregation**: CKA analysis reveals that layers of models with different depths align at corresponding relative positions. Models are divided into $N_B$ blocks by relative depth, with Co-LoRA attached to the last layer of each block; aggregation is performed between corresponding blocks.
    - **Weight Alignment**: Matrix $A$ is aligned in the $r$-dimensional representation space via L2 loss on public data; matrix $B$ employs CCA to find maximally correlated projection spaces → representations are projected to a shared space and back-projected. An orthogonality constraint maximizes expressiveness (Theorem 1: the weight update space spans $r^2$ dimensions).
    - **Design Motivation**: More lightweight and privacy-preserving than federated distillation (which requires public logits). More general than HETLoRA (which only handles rank heterogeneity).

3. **Gated Fusion**:

    - **Function**: Adaptively balances local personalized knowledge and global shared knowledge.
    - **Formula**: $h_O = W_p h_I + (1-\tilde{\beta})h_L + \tilde{\beta}h_G$, where $\tilde{\beta} = \sigma(\beta)$ is learnable.
    - **Design Motivation**: Different layers and tasks require varying degrees of global knowledge; the learnable gate automatically adjusts this balance.

### Loss & Training
- $A$/$B$ alignment is performed once before federated training (one-time overhead).
- Communication cost is low: only $P \in \mathbb{R}^{r \times r}$ and $Q \in \mathbb{R}^r$ are transmitted (much smaller than full LoRA matrices).
- Privacy protection: gradient EMA + Gaussian noise + random dimension subsampling.

## Key Experimental Results

### Main Results (DRAKE Benchmark, 40 Tasks)

| Method | Homogeneous Setting (Avg Acc) | Heterogeneous Setting (Avg Acc) | Notes |
|--------|-------------------------------|----------------------------------|-------|
| Local only | Baseline | Baseline | No federated collaboration |
| FedAvg | Below Local | N/A | Naive averaging is harmful |
| DITTO | Moderate | N/A | Dual adapter, no heterogeneity support |
| FedDAT | Moderate–High | N/A | Same as above |
| HETLoRA | — | Moderate | Handles rank heterogeneity only |
| **FedMosaic** | **Best** | **Best** | Significantly outperforms all methods |

### Ablation Study

| Configuration | Acc Change | Notes |
|---------------|-----------|-------|
| Full FedMosaic | Best | Complete method |
| w/o RELA (FedAvg) | Decrease | Task interference |
| w/o Co-LoRA (HETLoRA) | Decrease | Insufficient handling of architecture heterogeneity |
| w/o block alignment | Decrease | Incorrect layer correspondence |
| w/o weight alignment | Decrease | Inconsistent optimization trajectories |
| w/o gate $\beta$ | Decrease | Fixed global/local ratio is suboptimal |

### Key Findings
- **FedAvg underperforms non-collaborative baselines in realistic heterogeneous settings**: naively averaging models from unrelated tasks causes severe parameter interference.
- **Selective aggregation via RELA is critical**: collaborating only with relevant clients substantially outperforms global averaging.
- **Co-LoRA successfully transfers knowledge across architectures**: effective collaboration is demonstrated for both Llama-1B ↔ Llama-3B and Llama-1B ↔ Qwen-3B.
- **CKA validates the layer alignment hypothesis**: layers at corresponding relative depths exhibit the highest representational similarity, supporting the block alignment strategy.
- **Communication overhead is minimal**: only $P$ ($r^2$ parameters) and $Q$ ($r$ parameters) are transmitted alongside sanitized gradients.

## Highlights & Insights
- **Dimension-invariant modules offer an elegant solution for cross-architecture federated learning**: $P \in \mathbb{R}^{r \times r}$ depends only on rank, not on hidden dimensions — this design principle generalizes to any scenario requiring cross-architecture knowledge transfer.
- **Three-component gradient sanitization**: EMA mixing + Gaussian noise + random dimension subsampling — each step is backed by privacy theory, yielding a practical and secure overall scheme.
- **DRAKE benchmark fills a gap in multimodal PFL evaluation**: 40 diverse tasks + distribution shift + multi-image inputs make it substantially more realistic than prior non-IID MNIST-style evaluations.
- **Orthogonality constraint in Theorem 1**: under frozen orthogonal $A$/$B$, the weight update space of Co-LoRA spans $r^2$ dimensions (the maximum possible), providing theoretical guarantees on expressiveness.

## Limitations & Future Work
- **DRAKE covers only 40 tasks**: real-world agentic AI scenarios may involve hundreds of task types; scalability remains to be validated.
- **Public data requirement**: $A$/$B$ alignment requires a small amount of public data, which may be unacceptable in extreme privacy settings.
- **Only 1B/3B scales are evaluated**: effectiveness at 7B/13B and above, and whether communication costs remain manageable, are open questions.
- **Uniform rank $r$ across clients**: Co-LoRA currently does not support clients requiring LoRA adapters of different ranks.

## Related Work & Insights
- **vs. HETLoRA/FlexLoRA**: These methods handle heterogeneity only in rank within the same base architecture, implicitly assuming identical $d_I, d_O$. Co-LoRA addresses true architecture heterogeneity (different model families, depths, and dimensions).
- **vs. FedMD/FedMKT**: Federated distillation transfers knowledge via logits on public data, but logit extraction is computationally expensive for large models and poses gradient inversion privacy risks. Co-LoRA transmits only small matrices, offering greater security and efficiency.
- **vs. DITTO**: DITTO maintains local and global dual adapters but aggregates the global adapter via naive averaging. FedMosaic replaces this with task-aware aggregation via RELA and supports heterogeneous architectures through Co-LoRA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of dimension-invariant modules, gradient-based relevance aggregation, and block/weight alignment is systematic and highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The 40-task benchmark is comprehensive and ablations are thorough, though experiments on larger-scale models are absent.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and methodological derivations are rigorous, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and viable solution for heterogeneous federated learning; the DRAKE benchmark offers lasting value to the community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)

<!-- RELATED:END -->
