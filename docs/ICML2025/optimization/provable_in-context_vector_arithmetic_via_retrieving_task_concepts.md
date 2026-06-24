---
title: >-
  [Paper Note] Provable In-Context Vector Arithmetic via Retrieving Task Concepts
description: >-
  [ICML2025][Optimization][in-context learning] This paper proves from an optimization theory perspective that a non-linear Transformer with residual connections and Layer Normalization, trained via gradient descent on QA data, can perform factual recall ICL tasks through vector addition (task vector + query). Additionally, training on ICL data leads to harmful memorization of low-level features.
tags:
  - "ICML2025"
  - "Optimization"
  - "in-context learning"
  - "task vector"
  - "vector arithmetic"
  - "optimization theory"
  - "transformer"
  - "factual recall"
  - "OOD generalization"
date: 2026-05-08
content_hash: abb920ac8f241943
---

# Provable In-Context Vector Arithmetic via Retrieving Task Concepts

**Conference**: ICML2025  
**arXiv**: [2508.09820](https://arxiv.org/abs/2508.09820)  
**Code**: None  
**Area**: Optimization / ICL Theory  
**Keywords**: in-context learning, task vector, vector arithmetic, optimization theory, transformer, factual recall, OOD generalization

## TL;DR
This paper proves from an optimization theory perspective that a non-linear Transformer with residual connections and Layer Normalization, trained via gradient descent on QA data, can perform factual recall ICL tasks through vector addition (task vector + query). Additionally, training on ICL data leads to harmful memorization of low-level features.

## Background & Motivation

**Core Observation**: Recent empirical studies (Merullo et al., 2024; Hendel et al., 2023) find that during ICL inference, LLMs produce a "task vector" $\mathbf{a}_\theta^f(\mathbf{T})$ in intermediate layers and complete predictions through simple vector addition:

$$f(\mathbf{x}_{\text{query}}) = \mathbf{a}_\theta^f(\mathbf{T}) + \mathbf{b}_\theta^{\text{query}}(\mathbf{x}_{\text{query}})$$

This behavior resembles vector arithmetic in Word2Vec (e.g., "France - Paris + Poland = Warsaw"), but still lacks theoretical explanation:

**Why** does task vector arithmetic naturally emerge in non-linear residual Transformers trained via gradient descent?

**QA Data vs. ICL Data**: Empirical evidence indicates that QA data is crucial for factual retrieval capability (Allen-Zhu & Li, 2024), but theoretical support is lacking.

**Transformer vs. Word2Vec**: What is the advantage of Transformers in the context of vector arithmetic?

**Limitations of Prior Work**:
- Existing ICL theories ignore the key role of the residual stream or treat it unnaturally.
- Common simplistic assumptions are made, such as linearized attention and square loss.
- Both training and testing utilize ICL data, which is inconsistent with practice.

## Method

### Hierarchical Data Modeling

Based on geometric observations of intermediate layers in GPT (Figure 1), the paper proposes hierarchical concept modeling:

**High-level Task Concept Vectors** $\mathbf{a}_k \in \mathbb{R}^d$: There are $K$ high-level binary concepts $z_k \in \{0,1\}$. These vectors are pairwise orthogonal, representing independent task concepts like "capital" or "national flower".

**Low-level Task-Specific Vectors** $\mathbf{b}_k \in \mathbb{R}^d$: Each high-level concept is associated with a pair of semantic antonym vectors $\pm\mathbf{b}_k$, with $\mathbf{a}_{k_1} \perp \mathbf{b}_{k_2}$ (orthogonal between high and low levels), corresponding to entity-level information such as "specific country" or "specific flower".

**Word-Label ICL Prompt**: $\mathbf{T} = [\mathbf{x}_1, \mathbf{y}_1, \cdots, \mathbf{x}_J, \mathbf{y}_J, \mathbf{x}_{J+1}]$, where each pair shares a co-task concept $k_\mathbf{T}$:

$$\mathbf{x}_l = \sum_{k \in \mathcal{X}_{\mathbf{T},l}} (x_a \cdot \mathbf{a}_k + y_{k,l} \cdot \mathbf{b}_k) + \boldsymbol{\xi}_{l,\mathbf{x}}$$

$$\mathbf{y}_l = \sum_{k \in \mathcal{Y}_{\mathbf{T},l}} (\mathbf{a}_k + y_{k,l} \cdot \mathbf{b}_k) + \boldsymbol{\xi}_{l,\mathbf{y}}$$

Intuitive example: in the prompt "Japan Sakura France Rooster China", the co-task is "national symbol", and the expected output is "Panda".

**QA Sentence Distribution**: $\mathbf{S} = [\mathbf{x}^{\text{QA}}, \mathbf{y}]$, where the prefix consists of a common token $\boldsymbol{\nu}_{n}$ + task vector $\mathbf{a}_{k_\mathbf{S}}$ (e.g., "What is the capital of"). The key difference is that **the QA prefix does not contain lower-level features $\mathbf{b}_k$**.

### Residual-LayerNorm Transformer Model

Unlike prior theoretical works that use structured embeddings (e.g., words on top, labels on bottom), this paper directly processes mixed sequences:

$$\mathbf{h}_{\theta,0}(\mathbf{T}) = \sum_{l=1}^{L-1} \mathbf{W}_V \mathbf{T}_l \cdot \sigma_S((\mathbf{W}_K \mathbf{T}_l)^\top (\mathbf{W}_Q \mathbf{T}_L))$$

$$\mathbf{h}_\theta = \mathbf{W}_O \text{LN}(\mathbf{h}_{\theta,0}(\mathbf{T})) + \mathbf{T}_L$$

where $\text{LN}(\mathbf{z}) = \mathbf{z}/\|\mathbf{z}\|_2$ is the $\ell_2$ Layer Normalization, $\mathbf{W}_O = \mathbb{I}$, and the final output = normalized attention output + residual.

**Connection to Word2Vec**: When $\|\mathbf{a}_k\| = \|\mathbf{b}_{k'}\|$, we approximately have $\mathbf{y}_{J+1} \approx \mathbf{a}_{k_\mathbf{T}} + \mathbf{x}_{J+1}$, indicating that the model only needs to retrieve the high-level task vector from the demonstrations and add it to the query.

**Training Objective**: Cross-entropy loss with $L_2$ regularization + gradient descent, with the vocabulary containing $7K + K'$ tokens.

### Core Theoretical Results

**Theorem 3.2 (Task Vector Retrieval)**:
- **Training on ICL/QA-ICL data** $\rightarrow$ The model generates a mixed vector containing both high-level $\mathbf{a}_k$ and low-level $\mathbf{b}_k$, failing to retrieve a clean task vector.
- **Training on QA data** $\rightarrow$ The model approximately retrieves a pure high-level task vector: $\cos\langle \mathbf{h}_{\theta,0}, \mathbf{a}_{k^\star}\rangle = \Theta(1)$, and $o(1)$ in other directions.

**Proposition 3.3 (Test Loss Discrepancy)**:
- ICL training $\rightarrow$ Test loss is $\Theta(1)$ (constant-level error, around 20%).
- QA training $\rightarrow$ Test loss is $\leq \varepsilon$ (arbitrarily small), and supports:
    - Direct regression of the task vector from demo pairs (no query needed).
    - The task vector can be added to any query of the same concept to obtain the correct answer.

**Proposition 3.4 (OOD Generalization)**: Models trained on QA can generalize to:
1. **Vocabulary Drift**: New high-level concepts only need to lie in the conical combination of training concepts; low-level and irrelevant concepts can be entirely unseen.
2. **Distribution Drift**: When the prompt contains multiple co-tasks, the model forms a Bayesian Model Averaging-style mixed task vector: $\mathbf{h}_{\theta,0} \approx \sum_{k \in \mathcal{K}} w_{\theta,k} \mathbf{a}_k$.

## Key Experimental Results

### Main Results: Training Dynamics Comparison (Figure 2 vs. Figure 3)

| Training Data | Test Error | Projection of $\mathbf{W}_V$ on $\mathbf{b}_k$ | Task Vector Quality |
|---------|---------|--------------------------------------|-----------|
| ICL Prompt $\mathcal{P}_\mathbf{T}$ | ~20% (constant level) | Significant growth (harmful memorization) | Mixed with low-level features |
| QA Sentence $\mathcal{P}_{\text{QA}}$ | →0 (converges) | Remains negligible | Pure high-level vector |

### Key Findings

- **Figure 2(d)**: Under ICL training, $\mathbf{W}_V$ develops a non-negligible projection in the direction of $\mathbf{b}_k$, causing the test error in Figure 2(b) to plateau at ~0.2.
- **Figure 3(d)**: Under QA training, the projection of $\mathbf{W}_V$ onto $\mathbf{b}_k$ remains near zero, and the test error in Figure 3(b) continues to decrease to 0.
- **Attention Matrix**: The projection of $\mathbf{W}_K^\top \mathbf{W}_Q$ in the direction of $\mathbf{a}_k$ grows under both training schemes (slowing down and then accelerating), but only QA training successfully translates this into effective task retrieval.

### Ablation Study (Theoretical Level)

| Condition Change | Impact on QA Training |
|---------|---------------|
| Prompt length $J^\star$ increases | More accurate task identification, $\varepsilon$ can be smaller |
| Noise $\sigma_p^\star$ increases | Requires longer prompt to compensate |
| Number of co-tasks $|\mathcal{K}|$ increases ($\leq 3$) | Model forms weighted mix, still generalizes |
| Low-level features completely unseen | Correct prediction is still possible (relying on high-level task vector) |

## Highlights & Insights

1. **First theoretical explanation of the role of residual stream and LayerNorm in ICL**: The residual stream provides query information, and LayerNorm ensures the normalized additive structure of the task vector.
2. **Uncovering the "harmful memorization" mechanism in ICL training**: Unlike noise memorization in computer vision, here the co-occurrence asymmetry of low-level features causes $\mathbf{W}_V$ to incorrectly learn $\mathbf{b}_k$.
3. **Theoretical validation of the advantages of QA data**: The QA prefix naturally does not contain low-level concept vectors, forcing the model to only learn the high-level task vector.
4. **Composability of OOD generalization**: Conical combinations of task vectors support vocabulary drift and distribution drift, gesturing to the "celebrity helps minority" effect.
5. **Connection to the BMA perspective**: The mixed vector $\sum w_k \mathbf{a}_k$ under a multi-co-task prompt naturally corresponds to Bayesian Model Averaging.

## Limitations & Future Work

1. **Limited to single-token factual recall**: Explicitly states that it does not cover multi-token or complex factual tasks, offering limited scope.
2. **Strong assumptions in data modeling**: Simplifying assumptions such as orthogonal high/low-level concepts, single-layer Transformer, and $\ell_2$ LayerNorm (non-standard LayerNorm) deviate from actual LLMs.
3. **Lack of multi-layer analysis**: In practical LLMs, task vectors emerge in layers 15-19, whereas this paper only analyzes a single layer.
4. **Vocabulary size constraints**: The constraint $K' \geq C \max\{M, K\}$ might be too restrictive.
5. **Lack of real LLM experiments**: All experiments are theoretical validations on synthetic data, lacking comparisons on real models like GPT-J.
6. **Practical significance of QA data**: It remains unclear whether the advantages of QA training are equally significant in large-scale pre-training.

## Related Work & Insights

- **Empirical Task Vectors**: Hendel et al. (2023), Merullo et al. (2024), Todd et al. (2024) — This work provides the theoretical foundation for these observations.
- **Factual Knowledge Storage**: Allen-Zhu & Li (2024, 2025) — Theorizing how QA data enhances factual retrieval.
- **ICL Theory**: Zhang et al. (2024), Kim & Suzuki (2024), Chen et al. (2024) — This work overcomes limitations such as "no residuals" and "linearized attention".
- **Linear Concept Geometry**: Park et al. (2025) — Direct inspiration for the data modeling in this paper.
- **Word2Vec vs. Transformer**: Wibisono & Wang (2023) — This work explicitly demonstrates the theoretical advantages of Transformers over Word2Vec.
- **Application Prospects**: Task vector arithmetic can be extended to downstream tasks such as concept erasure, model editing, and model merging.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical framework to explain the vector arithmetic of ICL in residual Transformers; the comparative analysis of QA vs. ICL training is novel.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments align with theory, but lacks validation on real LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, modeling starts from empirical observations, and the proof sketch is well-structured.
- Value: ⭐⭐⭐⭐ Significantly advances the understanding of ICL mechanisms, bridging multiple perspectives such as task vector, QA training, and BMA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](../../ICLR2026/optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[ICML 2025\] On Understanding Attention-Based In-Context Learning for Categorical Data](on_understanding_attention-based_in-context_learning_for_categorical_data.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](../../ICML2026/optimization/distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/optimization/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)

</div>

<!-- RELATED:END -->
