---
title: >-
  [Paper Note] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows
description: >-
  [ICLR 2026][Model Compression][Performance Prediction] This paper proposes Agentic Predictor, a multi-view workflow encoding framework that jointly models graph structure, code semantics, and prompt information to predict the performance of LLM-based agentic workflows, substantially reducing costly trial-and-error evaluations.
tags:
  - ICLR 2026
  - Model Compression
  - Performance Prediction
  - Multi-View Encoding
  - Agentic Workflows
  - Graph Neural Networks
  - Unsupervised Pretraining
date: 2026-05-08
content_hash: 4ada70451430281d
---

# Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows

**Conference**: ICLR 2026
**arXiv**: [2505.19764](https://arxiv.org/abs/2505.19764)
**Code**: [GitHub](https://github.com/deepauto-ai/agentic-predictor)
**Area**: Model Compression
**Keywords**: Performance Prediction, Multi-View Encoding, Agentic Workflows, Graph Neural Networks, Unsupervised Pretraining

## TL;DR

This paper proposes Agentic Predictor, a multi-view workflow encoding framework that jointly models graph structure, code semantics, and prompt information to predict the performance of LLM-based agentic workflows, substantially reducing costly trial-and-error evaluations.

## Background & Motivation

LLM-based agentic systems have advanced rapidly in recent years, yet optimizing their workflow configurations poses significant challenges due to a vast search space. Existing automated design methods (e.g., ADAS, AFlow) rely on large numbers of LLM API calls for evaluation, incurring prohibitive computational costs. This paper proposes replacing full-execution evaluation with a **performance predictor**, drawing inspiration from predictor-based approaches in neural architecture search (NAS).

Two core challenges are identified:

**Workflow Heterogeneity**: Different workflows vary substantially in communication structure, prompting strategies, and tool-calling patterns, making unified modeling difficult.

**Label Scarcity**: Obtaining performance labels via full execution is expensive, leaving insufficient data for supervised learning.

## Method

### Overall Architecture

Agentic Predictor comprises three stages: (a) a multi-view workflow encoder that maps agentic workflows into a unified representation; (b) a cross-domain unsupervised pretraining stage that learns generalizable representations; and (c) a predictor-guided search stage that trains the predictor with a small number of labeled samples.

### Key Designs

1. **Multi-View Workflow Encoding**:

   - **Graph View $\mathcal{G}$**: Models the workflow as a DAG, encoding inter-agent communication dependencies via a GNN.
   - **Code View $\mathcal{C}$**: Encodes the complete workflow code using an MLP to capture logical structure and tool-usage patterns.
   - **Prompt View $\mathcal{P}$**: Encodes role descriptions and behavioral specifications from system prompts using an MLP.
   - The three views are fused through an aggregation layer: $\mathbf{Z} = \text{MLP}([\mathbf{Z}_\mathcal{G}, \mathbf{Z}_\mathcal{C}, \mathbf{Z}_\mathcal{P}])$

2. **Cross-Graph Attention Mechanism**:

   - Three graph types are constructed: prompt graph $\mathcal{G}_\text{prompt}$, code graph $\mathcal{G}_\text{code}$, and operator graph $\mathcal{G}_\text{operator}$.
   - Node-level information exchange is performed via cross-view self-attention.
   - ViewAttnPool adaptively learns importance weights for each view.

3. **Cross-Domain Unsupervised Pretraining**:

   - Reconstruction loss: $\mathcal{L}_{rec} = \frac{1}{M}\sum_{i=1}^{M}\|\mathcal{G}_i - \hat{\mathcal{G}}_i\|^2 + \|\mathcal{C}_i - \hat{\mathcal{C}}_i\|^2 + \|\mathcal{P}_i - \hat{\mathcal{P}}_i\|^2$
   - Cross-modal contrastive loss: InfoNCE loss applied across three view pairs — $(\mathcal{G}, \mathcal{C})$, $(\mathcal{G}, \mathcal{P})$, and $(\mathcal{C}, \mathcal{P})$.
   - Pretraining uses no performance labels, avoiding label leakage.

### Loss & Training

- Pretraining stage: $\mathcal{L}_{enc} = \mathcal{L}_{rec} + \mathcal{L}_{con}$ (reconstruction + contrastive)
- Predictor stage: cross-entropy loss for binary classification; MSE loss for regression.
- The text encoder uses all-MiniLM-L6-v2 (384-dim); the code encoder uses CodeRankEmbed (768-dim); both are projected to a unified 512-dim space.

## Key Experimental Results

### Main Results

| Domain | Metric | Agentic Predictor | Prev. SOTA | Gain |
|--------|--------|-------------------|------------|------|
| Code Generation (GD) | Accuracy | 85.33% | 85.24% (Graph Trans.) | +0.09% |
| Code Generation (AF) | Accuracy | 85.62% | 84.71% (Graph Trans.) | +0.91% |
| Math (GD) | Accuracy | 66.20% | 64.84% (GAT) | +1.36% |
| Math (AF) | Accuracy | 79.56% | 76.44% (GAT) | +3.12% |
| Average | Accuracy | **79.97%** | 78.36% (GAT) | +2.05% |
| Average | Utility | **76.33%** | 73.54% (Dir-GNN) | +3.79% |

### Ablation Study

| Configuration | Avg. Accuracy | Avg. Utility | Note |
|---------------|--------------|--------------|------|
| Code + Graph + Text (Full) | **84.38%** | **81.88%** | Full model |
| w/o Code | Decreased | Decreased | Code view is critical for logical understanding |
| w/o Graph | Decreased | Decreased | Graph structure is critical for interaction modeling |
| w/o Text | Decreased | Decreased | Prompt semantics are indispensable |

### Key Findings

- The three views are strongly complementary; removing any single view leads to performance degradation.
- Cross-domain unsupervised pretraining is particularly effective under label-scarce conditions (the pretrained variant Agentic Predictor+ yields larger gains in low-label regimes).
- The method is search-agnostic and can be combined with arbitrary search strategies.

## Highlights & Insights

- Transferring the performance prediction paradigm from NAS to agentic workflow optimization represents a novel research direction.
- Multi-view encoding effectively exploits the heterogeneous information present in agentic workflows (structure, code, and semantics).
- The cross-domain pretraining strategy mitigates label scarcity, demonstrating the potential of self-supervised learning in emerging domains.

## Limitations & Future Work

- Evaluation is conducted solely on FLORA-Bench, limiting the breadth of dataset coverage.
- The generalization of the predictor to unseen workflow variations requires further investigation.
- Larger-scale agentic systems and more complex multimodal workflows remain unexplored.
- The code and prompt encoders rely on fixed pretrained models without end-to-end fine-tuning.

## Related Work & Insights

- Compared to FLORA-Bench: this work introduces multi-view encoding and unsupervised pretraining rather than relying on a single graph view.
- Compared to MAS-GPT: this work employs a lightweight predictor rather than fine-tuning an LLM to generate workflows.
- Predictor-based approaches from NAS (e.g., CAP, FlowerFormer) can be directly adapted to the agentic domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing performance prediction into agentic workflow design is a new direction.
- Experimental Thoroughness: ⭐⭐⭐ Limited to a single benchmark, though ablations are detailed.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the framework is well described.
- Value: ⭐⭐⭐⭐ Practically meaningful for reducing the development cost of agentic systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](../../ACL2026/model_compression/supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[ICLR 2026\] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [\[ICLR 2026\] Parallel Token Prediction for Language Models](parallel_token_prediction_for_language_models.md)
- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)
- [\[ICLR 2026\] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)

</div>

<!-- RELATED:END -->
