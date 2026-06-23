---
title: >-
  [Paper Note] Conditioned Initialization for Attention
description: >-
  [ICLR 2026][Pretraining][Jacobian] This paper theoretically attributes the optimization stability of attention layers to the condition number of their Jacobian. It proposes "Conditioned Initialization"—initializing the value matrix as a rectangular identity matrix and the query/key matrices as semi-orthogonal matrices (both having a condition number of
tags:
  - ICLR 2026
  - Pretraining
  - Jacobian
date: 2026-05-08
content_hash: 7c1da7b1f7e9e086
---
# Conditioned Initialization for Attention

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=cKNOCYPo2W](https://openreview.net/forum?id=cKNOCYPo2W)  
**Code**: TBD  
**Area**: Optimization / Initialization / Transformer  
**Keywords**: Attention initialization, condition number, Jacobian, semi-orthogonal matrix, optimization stability

## TL;DR
This paper theoretically attributes the optimization stability of attention layers to the condition number of their Jacobian. It proposes "Conditioned Initialization"—initializing the value matrix as a rectangular identity matrix and the query/key matrices as semi-orthogonal matrices (both having a condition number of 1). This tightens the upper bound of the Jacobian condition number at the start of training, consistently accelerating convergence (by 20–30%) and improving generalization across various Transformer tasks including image classification, detection/segmentation, language modeling, and long sequences.

## Background & Motivation
**Background**: The success of Transformers centers on the attention layer, where the three projection matrices (query, key, and value) determine how tokens interact. While extensive work has focused on the efficiency, scalability, and expressivity of attention, few have rigorously investigated the fundamental question: how these three matrices should be **initialized**. Current mainstream approaches rely on either simple random initialization (truncated normal/uniform) or heuristic alternatives—mimetic initialization (mimicking weight statistics of converged models) and weight selection (transferring weights from larger teacher models).

**Limitations of Prior Work**: In the CNN era, Xavier/Kaiming initialization proved that proper weight scaling at the start of training significantly improves gradient optimization and stability, which is crucial for training deep networks like ResNet. However, Transformers have not received equivalent theoretical scrutiny in this regard. Existing mimetic/weight selection methods recognize that "initialization matters" but remain **heuristic**, lacking a principled connection to the conditioning of the attention mechanism itself—they do not explain why they work or what specific quantity they aim to optimize.

**Key Challenge**: The optimization stability of self-attention essentially depends on the conditioning of its Jacobian (a well-conditioned Jacobian converges faster and more stably), which in turn depends on the singular value spectrum of the $Q/K/V$ projections. Random initialization ignores this, starting training from a point with poor spectral properties and high condition numbers.

**Goal**: To design an initialization **specifically for attention structures** that yields better-conditioned attention layers from the very first step of training.

**Key Insight**: The authors do not modify the training objective (loss/regularization) but instead treat **initialization** as a way to inject an inductive bias. Theoretically, they prove that the condition number of the attention Jacobian is controlled by an upper bound involving $\kappa(W_Q), \kappa(W_K), \kappa(W_V)$. Consequently, the upper bound can be tightened by minimizing the condition numbers of these three matrices (setting them to 1) at initialization.

**Core Idea**: Replace random initialization of $Q/K/V$ with "matrices with a condition number of 1" (rectangular identity / semi-orthogonal matrices) to reduce the upper bound of the attention Jacobian condition number at the source, thereby achieving more stable optimization and better generalization.

## Method

### Overall Architecture
The method follows a two-step approach: first, it establishes a **theoretical framework** linking the "condition number of the attention Jacobian" to the "individual condition numbers of the $Q/K/V$ weight matrices" via an inequality; second, it provides a **minimalist initialization scheme** based on this theory—it only modifies the initial values of the $Q/K/V$ matrices without changing the network architecture or adding any training-time regularization.

Let self-attention be denoted as $A(X) = \mathrm{softmax}(XW_QW_K^TX^T)\,XW_V$, where $X\in\mathbb{R}^{N\times D}$ is the input sequence and $W_Q, W_K, W_V\in\mathbb{R}^{D\times d}$. The paper focuses on the Jacobian $J(A(X))$ with respect to parameters $W_Q, W_K, W_V$. The condition number of a matrix $Z$ is defined as $\kappa(Z)=\sigma_{\max}(Z)/\sigma_{\min}(Z)$. The aim is to minimize $\kappa(J(A(X)))$ at initialization.

### Key Designs

**1. Reducing Optimization Stability to an Upper Bound of the Jacobian Condition Number**

The pain point is: everyone knows initialization is important, but how can "importance" be quantified? The authors frame this as a controllable quantity. By deriving $\partial A/\partial W_Q, \partial A/\partial W_K, \partial A/\partial W_V$ (Proposition 3.1, using the softmax derivative $\partial\,\mathrm{softmax}/\partial z = \Lambda(\mathrm{softmax}(z))$, where $\Lambda(z)=\mathrm{Diag}(z)-zz^T$), they prove the core Theorem 3.1:

$$\kappa(J(A(X))) \le \kappa(X)^3\,\kappa\!\big(\Lambda(\mathrm{softmax}(XW_QW_K^TX^T))\big)\,\kappa(W_V)\big(\kappa(W_Q)+\kappa(W_K)\big) + \kappa(X)\,\kappa\!\big(\mathrm{softmax}(XW_QW_K^TX^T)\big)$$

This upper bound is critical because it decomposes a difficult-to-optimize quantity (Jacobian condition number) into two terms, and the **first term explicitly contains $\kappa(W_Q), \kappa(W_K), \kappa(W_V)$**—three variables that can be directly controlled. The authors denote this upper bound as a proxy objective $B(J(A))$. Minimizing $\kappa(W_Q), \kappa(W_K), \kappa(W_V)$ to 1 at initialization minimizes $B(J(A))$.

**2. Conditioned Initialization: Replacing Random Initialization with Condition-1 Matrices**

To ensure $W_Q, W_K, W_V$ have condition numbers of 1, the authors identify two "well-conditioned" families for $D\times d$ matrices: ① scalar multiples of the identity $\lambda I_{D\times d}$ ($\lambda\neq0$); ② semi-orthogonal matrices $O_{D\times d}$ (orthonormal rows or columns, near-isometric). Standard Gaussian/Uniform initializations do not belong to these families and have large, random condition numbers. Proposition 3.2 proves that using either family to initialize $W_Q, W_K, W_V$ results in a proxy bound $B(J(A))$ that is **strictly no larger** than that of Gaussian/Uniform initialization.

**3. Differentiated Treatment for Q/K vs. V: Identity for Value, Semi-orthogonal for Query/Key**

The authors differentiate the initialization based on the **algebraic roles** of $Q/K/V$. The value matrix $W_V$ enters the output **linearly** as $(\mathrm{softmax}(\cdots))(XW_V)$; initializing it as a rectangular identity matrix allows $XW_V=X$, preserving the scale of the input representation and ensuring $\kappa(W_V)=1$ without distorting the Jacobian. Query and Key interact **bilinearly** via $S=XW_QW_K^TX^T$. If they were also initialized as identity matrices, projections would be biased toward coordinate subspaces, leading to anisotropic logits and unstable softmax dynamics. Therefore, $W_Q$ and $W_K$ use semi-orthogonal initialization to provide near-isometric embeddings, ensuring a balanced representation and diverse attention patterns. Specifically, $W_Q^{(i)}$ and $W_K^{(i)}$ for each head are initialized independently as semi-orthogonal projections ($(W_Q^{(i)})^TW_Q^{(i)}=I_d, (W_K^{(i)})^TW_K^{(i)}=I_d$).

## Key Experimental Results

Experiments cover image classification (ImageNet-1k), object detection and instance segmentation (COCO), long sequence modeling (LRA), and language modeling (Crammed BERT + GLUE).

### Main Results

Top-1 accuracy for five modern Vision Transformers on ImageNet-1k, where Conditioned Initialization outperforms other methods:

| Model | Default Init | Mimetic | Conditioned (Ours) | Gain |
|------|-----------|---------|--------------------|------|
| ViT-B | 80.3 | 80.5 | **81.5** | +1.2 |
| DeiT-B | 81.6 | 81.6 | **82.7** | +1.1 |
| Swin-B | 83.4 | 83.5 | **84.6** | +1.2 |
| XCiT-M | 82.6 | 82.6 | **83.5** | +0.9 |
| DaViT-B | 84.3 | 84.4 | **85.3** | +1.0 |

On language modeling, Crammed BERT GLUE average scores: Default 78.6 / Mimetic 78.9 / Conditioned **79.6**. Improvements were also consistent across LRA tasks (e.g., Text 63.8→64.9) and COCO detection/segmentation.

### Convergence Efficiency Analysis

Number of epochs required to reach the final accuracy of the default initialization (lower is better). Conditioned Initialization is consistently 20–30% faster:

| Model | Default (Baseline Ep.) | Mimetic | Conditioned (Ours) | Gain (Speedup) |
|------|-------------------|---------|--------------------|------|
| ViT-B | 300 | 288 | **211** | ~30% |
| DeiT-B | 300 | 279 | **206** | ~31% |
| Swin-B | 400 | 394 | **321** | ~20% |
| XCiT-M | 400 | 391 | **318** | ~21% |

### Key Findings
- **Theoretical Validation**: Figures show that the average condition number of the attention Jacobian remains lower for Conditioned Initialization throughout training, closely following the theoretical upper bound from Theorem 3.1.
- **Benefits for Small Datasets**: For ViT-T on small datasets like Pets/CIFAR, where standard ViTs lack inductive bias, Conditioned Initialization significantly improves performance (Pets: 26.7 → 47.7).
- **Architecture Agnostic**: Gains are observed across standard self-attention (ViT), cross-covariance attention (XCiT), and linear attention (Nyströmformer).
- **Zero Cost**: Only the initial values of the three matrices are changed; no parameters are added, and there is no change to training objectives.

## Highlights & Insights
- **Quantifying Initialization Quality**: Translating the vague concept of "good initialization" into a controllable scalar (Jacobian condition number upper bound) is the paper's strongest contribution.
- **Algebraic Role Customization**: Using identity for linear roles (Value) and semi-orthogonal for bilinear roles (Q/K) demonstrates deep understanding of attention dynamics.
- **Principled vs. Heuristic**: Unlike Mimetic or Weight Selection, this method requires no external models and relies purely on spectral properties.
- **Initialization as Inductive Bias**: Injecting bias only at the "zeroth" step of training without affecting inference or training logic is a compelling paradigm for large-scale efficient training.

## Limitations & Future Work
- **Upper Bound vs. Exact Value**: The method optimizes an upper bound ($B(J(A))$), not the exact condition number. While they correlate in experiments, it remains an indirect proxy.
- **Transient Effect**: The conditioning is only guaranteed at $t=0$; as weights evolve during training, the "well-conditioned" property may degrade.
- **Architectural Scope**: The theory is derived for standard self-attention; the tightest bounds for variants with relative position encoding or complex normalization are not explicitly proven.
- **Modest Absolute Gains**: The value lies primarily in convergence speed and cost-effectiveness rather than massive jumps in absolute SOTA accuracy.

## Related Work & Insights
- **vs. Xavier / Kaiming**: While classical schemes focus on variance to prevent vanishing/exploding gradients, this work specializes in attention by controlling the Jacobian condition number.
- **vs. Mimetic Initialization**: Mimetic initialization is heuristic; this paper provides a theoretical optimization target and often yields better results without needing extra data.
- **vs. Weight Selection**: This method is autonomous and does not rely on a pre-trained teacher model.
- **vs. Training-time Conditioning**: Unlike methods that pre-condition during training, this approach asks if the initialization itself can produce a well-conditioned layer from the start.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[CVPR 2026\] Unlocking Pre-trained Weights: Parameter Inheritance for Zero-Shot Initialization](../../CVPR2026/llm_pretraining/unlocking_pre-trained_weights_parameter_inheritance_for_zero-shot_initialization.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](../../ICML2026/llm_pretraining/focus_and_dilution_the_multi-stage_learning_process_of_attention.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](../../CVPR2025/llm_pretraining/robust_message_embedding_via_attention_flow-based_steganography.md)
- [\[ICML 2025\] Benign Overfitting in Token Selection of Attention Mechanism](../../ICML2025/llm_pretraining/benign_overfitting_in_token_selection_of_attention_mechanism.md)

</div>

<!-- RELATED:END -->
