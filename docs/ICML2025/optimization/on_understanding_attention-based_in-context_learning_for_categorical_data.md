---
title: >-
  [Paper Note] On Understanding Attention-Based In-Context Learning for Categorical Data
description: >-
  [ICML2025][Optimization][in-context learning] This work generalizes the in-context learning (ICL) of Transformers from real-valued outputs to **categorical data** (categorical outcomes). It demonstrates that an architecture alternating between self-attention and cross-attention can **exactly implement** multi-step functional gradient descent (functional GD). Furthermore, it theoretically proves that this GD parameter configuration constitutes a stationary point of the attenti…
tags:
  - "ICML2025"
  - "Optimization"
  - "in-context learning"
  - "functional gradient descent"
  - "categorical data"
  - "attention mechanism"
  - "cross-attention"
  - "softmax"
  - "reproducing kernel Hilbert space"
date: 2026-05-08
content_hash: 0d546187c9223ef3
---

# On Understanding Attention-Based In-Context Learning for Categorical Data

**Conference**: ICML2025  
**arXiv**: [2405.17248](https://arxiv.org/abs/2405.17248)  
**Code**: Not open-sourced  
**Area**: Optimization / ICL Theory  
**Keywords**: in-context learning, functional gradient descent, categorical data, attention mechanism, cross-attention, softmax, reproducing kernel Hilbert space

## TL;DR

This work generalizes the in-context learning (ICL) of Transformers from real-valued outputs to **categorical data** (categorical outcomes). It demonstrates that an architecture alternating between self-attention and cross-attention can **exactly implement** multi-step functional gradient descent (functional GD). Furthermore, it theoretically proves that this GD parameter configuration constitutes a stationary point of the attention model's loss function.

## Background & Motivation

- Prior works (von Oswald et al. 2023; Ahn et al. 2023; Cheng et al. 2024) have interpreted the ICL process of Transformers as **functional gradient descent**, but these analyses are restricted to **real-valued outputs** (e.g., linear regression, kernel regression) under the assumption $p(Y|X) = \mathcal{N}(f(x), \sigma^2 I)$.
- The outputs of language models are **discrete tokens (categorical variables)**, which do not satisfy the Gaussian assumption. In the real-valued case, $\mathbb{E}(Y)|_{f_{i,k}} = f_{i,k}$ is linear, allowing multi-step GD to be implemented solely via stacked self-attention. In contrast, in the categorical case, $\mathbb{E}(w_e)|_{f_{i,k}}$ is a **nonlinear function** of $f_{i,k}$ (involving softmax), which cannot be computed by self-attention alone.
- **Core Motivation**: To bridge the gap between theoretical ICL analysis and actual language models by extending the functional GD framework to categorical observations, making the theory more aligned with how Transformers function in real-world NLP.

## Method

### Overall Architecture

The model is viewed from two perspectives:
1. **Transformer** $T_\theta(z)$: An attention-based forward inference model.
2. **Softmax Classification Model** $p_\phi(Y=y|X=x) = \frac{\exp(w_{e,y}^T f_\phi(x))}{\sum_{c=1}^C \exp(w_{e,c}^T f_\phi(x))}$

Both share the embedding matrix $W_e \in \mathbb{R}^{d' \times C}$. The forward propagation of the Transformer essentially infers the latent function $f_\phi(x)$, which is then fed into the softmax layer to yield token probabilities.

### Functional Gradient Descent (Functional GD)

Assuming the latent function $f_\phi(x) = A\psi(x) + b$ lies in a Reproducing Kernel Hilbert Space (RKHS), performing GD on the cross-entropy loss yields the following update rule:

$$f_{j,k+1} = f_{j,k} + \frac{\alpha}{N} \sum_{i=1}^N \left[ w_{e,y_i} - \mathbb{E}(w_e)|_{f_{i,k}} \right] \kappa(x_i, x_j)$$

where $\mathbb{E}(w_e)|_{f_{i,k}} = \sum_{c=1}^C w_{e,c} \cdot p_{\phi_k}(Y=c|X=x_i)$ is the weighted expectation of the embedding vectors, and $\kappa(x_i, x_j) = \psi(x_i)^T \psi(x_j)$ represents the kernel function (corresponding to attention weights).

### Key Designs: Alternating Self-Attention and Cross-Attention Architecture

Each attention block consists of **two layers**:

**Self-Attention Layer** (Two Heads):

- **Head 1 (Function Update)**: Uses Key/Query to extract $x_i$, and Value to extract $w_{e,y_i} - \mathbb{E}(w_e)|_{f_{i,k}}$. It computes $\Delta f_{i,k}$ via kernel attention to update $f_{i,k} \to f_{i,k+1}$.
- **Head 2 (Expectation Erasure)**: Employs a large $\lambda$ to collapse the attention matrix to a Kronecker delta $\delta_{i,j}$ (self-matching). This erases the old $\mathbb{E}(w_e)|_{f_{i,k}}$ from the "scratch space", clearing room for the next computation step.

**Cross-Attention Layer** (Single Head, the Core Innovation):

- Queries are derived from the updated $f_{i,k+1}$, while Keys and Values are column vectors $\{w_{e,c}\}_{c=1}^C$ of the embedding matrix $W_e$.
- It utilizes softmax attention to exactly compute $\mathbb{E}(w_e)|_{f_{i,k+1}} = \sum_{c=1}^C w_{e,c} \frac{\exp(w_{e,c}^T f_{i,k+1})}{\sum_{c'} \exp(w_{e,c'}^T f_{i,k+1})}$.
- The newly computed expectation is written into the previously erased scratch space.

### Naturalness of Token Embeddings

When parameters are initialized to zero, $\mathbb{E}(w_e)|_{f_{i,0}} = \frac{1}{C}\sum_c w_{e,c} = \bar{w}_e$. If $\bar{w}_e = 0$, the input for the first GD step is precisely the token embedding vector $w_{e,y_i}$. This demonstrates that the widely used "learned embedding" encoding scheme in language models is **naturally aligned** with the GD perspective.

### Single-Step GD Simplification

If only a single GD step is performed, expectation updates are unnecessary. This allows removing the erasure head and the cross-attention layer, simplifying the block to a single self-attention layer with the input encoded as $e_{i,0} = (x_i, w_{e,y_i} - \bar{w}_e, 0_{d'})^T$.

## Key Experimental Results

### Synthetic Data (C=25 classes, N=10 in-context samples)

| Model | Attention Type | Top-1 Accuracy Trend |
|------|-----------|-----------------|
| GD (1 layer) | RBF / Softmax | Converges with small training set (L<5000) |
| Trained TF (1 layer) | RBF / Softmax | Requires larger training set (L>10000) to match GD |
| GD (2 layers) | Softmax | Converges with L<5000, outperforming 1 layer |
| Trained TF (2 layers) | Softmax | Requires L>25000 to match GD |

- Increasing the number of attention blocks (from 2 to 6) consistently improves the Top-1 accuracy and NLL of GD models.
- The parameter matrices of the Trained TF after convergence highly align with the stationary points predicted by the GD theory.

### ImageNet In-Context Classification (900 train classes / 100 test classes, N=50, VGG features d=512)

| Model | Top-1 Accuracy |
|------|-------------|
| Linear Probing | Baseline (requires retraining for each test context) |
| GD 1-layer | Slightly lower than linear probing |
| GD 2-3 layers | **Almost identical to linear probing** (without fine-tuning) |

### Language Generation (Tiny Stories + Children Stories, C=50257 tokens, d'=512, 8 heads)

| Model | Parameter Count | GPT-4o Score (Grammar/Consistency/Plot/Creativity) |
|------|-------|--------------------------------------------------|
| Softmax GD | **8K** attention parameters | Competes closely with Transformer |
| Softmax GD + FF | **8K** attention parameters | **Almost equal to Transformer** |
| Single-layer Transformer | **6M** attention parameters | Baseline |

- The GD model uses only about **0.13%** of the Transformer's parameters, but its generation quality practically matches the Transformer when supplemented with Feed-Forward (FF) layers.
- Typical failure modes for both models involve **repetitive generation**.

## Highlights & Insights

1. **Theoretical Breakthrough**: First to extend the functional GD $\leftrightarrow$ Transformer ICL correspondence from real-valued to categorical data. It proves that the GD parameter configuration is a **stationary point** of the attention model's loss function (Theorem 1/2), applicable to softmax attention.
2. **Architectural Insight**: Provides a theoretical explanation for the alternating self-attention and cross-attention structure (already present in the original Transformer decoder)—where self-attention performs function updates and cross-attention computes nonlinear expectations.
3. **Explanation for Embeddings**: Naturally derives the necessity of token embeddings from a GD perspective, rather than viewing them solely as an empirical design choice.
4. **Efficiency Implications**: The GD model matches the language generation quality of a 6M-parameter Transformer using only 8K parameters + FF, suggesting substantial parameter redundancy in standard Transformers.
5. **Importance of FF**: Experiments reveal the critical contribution of feed-forward layers to Transformer performance—while the GD model itself lacks FF, adding FF leads to a dramatic performance leap.

## Limitations & Future Work

1. **Limited Language Experiments**: Only tested on simple corpora like Tiny Stories using single-layer models; scalability to large-scale, real-world language modeling remains unverified.
2. **Structural Constraints of GD Models**: The query can only use positional embeddings (excluding token information), which differs from actual Transformers.
3. **Cross-Attention Requires All C Embeddings**: For vocabularies with a massive size ($C=50\text{K}+$ ), calculating $\mathbb{E}(w_e)$ in cross-attention requires a softmax over the entire vocabulary, which restricts efficiency.
4. **Strong Theoretical Assumptions**: Theorem 1 assumes rotational invariance (where the distribution of $x_i$ is rotationally symmetric) and exact calculation of expectations via cross-attention, which may not hold in practice.
5. **Lack of Theoretical Explanation for FF Layers**: Although experiments show FF is extremely important, the paper does not provide a theoretical explanation for FF from a GD perspective.

## Related Work & Insights

- **von Oswald et al. (2023)** / **Ahn et al. (2023)**: Explored the ICL $\leftrightarrow$ GD correspondence for linear regression; this work directly extends their findings to the categorical setting.
- **Cheng et al. (2024)**: Proposed ICL theory for kernel regression in RKHS; this work extends that theory to softmax attention and categorical loss.
- **Akyurek et al. (2022)**: First introduced the concept of "scratch space".
- **Vaswani et al. (2017)**: The original Transformer decoder inherently includes alternating self- and cross-attention; this paper offers a rigorous theoretical perspective for this structure.
- **Insight**: The essential role of FF layers in Transformers warrants deeper investigation; the GD perspective may guide the design of more efficient attention architectures.

## Rating

- Novelty: ⭐⭐⭐⭐ — Although extending from real-valued to categorical is a natural continuation, the design of using cross-attention to compute nonlinear expectations and its theoretical proof offer substantial novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic data, ImageNet, and language generation across three levels. GPT-4o auto-evaluation is persuasive, though the language experiments are relatively small-scale.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, progressing gradually from simple to complex, though the density of notation requires careful reading.
- Value: ⭐⭐⭐⭐ — Highly significant theoretical contribution to understanding the mechanism of Transformer ICL, with practical insights gained from the experiments on FF layers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICML 2025\] Provable In-Context Vector Arithmetic via Retrieving Task Concepts](provable_in-context_vector_arithmetic_via_retrieving_task_concepts.md)
- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](understanding_mode_connectivity_via_parameter_space_symmetry.md)

</div>

<!-- RELATED:END -->
