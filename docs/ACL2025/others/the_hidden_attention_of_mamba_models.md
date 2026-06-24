---
title: >-
  [Paper Note] The Hidden Attention of Mamba Models
description: >-
  [ACL 2025][Mamba model] Reveals that Mamba (selective state space model S6) can be reformulated as an implicit causal self-attention mechanism, and based on this, proposes attention visualization and interpretability methods (Attention Rollout and Mamba-Attribution) applicable to Mamba models, proving that its interpretability metrics are comparable to those of Transformers.
tags:
  - "ACL 2025"
  - "Mamba model"
  - "state space model"
  - "implicit attention"
  - "interpretability"
  - "selective SSM"
date: 2026-05-08
content_hash: cdf1e996c7b78929
---

# The Hidden Attention of Mamba Models

**Conference**: ACL 2025  
**arXiv**: [2403.01590](https://arxiv.org/abs/2403.01590)  
**Code**: [https://github.com/AmeenAli/HiddenMambaAttn](https://github.com/AmeenAli/HiddenMambaAttn)  
**Area**: Other  
**Keywords**: Mamba model, state space model, implicit attention, interpretability, selective SSM

## TL;DR

Reveals that Mamba (selective state space model S6) can be reformulated as an implicit causal self-attention mechanism, and based on this, proposes attention visualization and interpretability methods (Attention Rollout and Mamba-Attribution) applicable to Mamba models, proving that its interpretability metrics are comparable to those of Transformers.

## Background & Motivation

**Background**: The Mamba model (based on the Selective SSM/S6 layer) has demonstrated outstanding performance across multiple fields such as NLP, computer vision, and long-sequence modeling. It achieves performance comparable to Transformers while maintaining linear complexity, and can be switched to an efficient RNN mode during inference. However, understanding the internal working mechanisms of Mamba models—specifically how information flows and dependency relationships are captured between tokens—remains very limited.

**Limitations of Prior Work**: The attention matrices of Transformers have been widely used for model interpretability analysis (e.g., Attention Rollout, Transformer Attribution, etc.). However, SSM models traditionally only have convolutional and recurrent views, lacking interpretability tools similar to attention matrices. This severely restricts the application of Mamba models in domains with strict interpretability requirements, such as healthcare and finance.

**Key Challenge**: While Mamba models perform exceptionally in practice, their information flow mechanism is opaque—neither explainable via fixed convolutional kernels like traditional SSMs (since S6 is time-varying/data-dependent), nor possessing explicit attention weights for analysis like Transformers.

**Goal**: (1) Theoretically reveal the implicit attention structure of Mamba; (2) build interpretability tools for Mamba models based on this; (3) compare the similarities and differences in attention mechanisms between Mamba and Transformers.

**Key Insight**: The authors observe that the S6 layer is a "data-controlled linear operator". By expanding the time-varying recurrence formula into a matrix form, they obtain an input-dependent lower triangular matrix whose structure is highly analogous to a causal self-attention matrix.

**Core Idea**: Reformulate the S6 layer into a matrix multiplication form of $y = \tilde{\alpha} x$, where $\tilde{\alpha}$ is a data-dependent lower triangular matrix that can be regarded as the "implicit attention matrix" of Mamba.

## Method

### Overall Architecture

The paper provides a third perspective for the Mamba model (in addition to the existing parallel scan view and RNN recurrent view): the attention view. Through mathematical derivation, the output of each S6 channel is expressed as the product of a data-dependent lower triangular matrix and the input vector. Based on the extracted implicit attention matrix, the Attention Rollout and Transformer Attribution methods from Transformers are further adapted to Mamba models.

### Key Designs

1. **Derivation of the implicit attention matrix**:

    - **Function**: Derive the equivalent attention matrix form from the recurrence formula of the S6 layer
    - **Mechanism**: Given the time-varying system matrices $\bar{A}_t$, $\bar{B}_t$, and $C_t$, expanding the recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$, $y_t = C_t h_t$ yields $y_t = C_t \sum_{j=1}^{t} (\prod_{k=j+1}^{t} \bar{A}_k) \bar{B}_j x_j$. Written in matrix form as $y = \tilde{\alpha} x$, where $\tilde{\alpha}_{i,j} = C_i (\prod_{k=j+1}^{i} \bar{A}_k) \bar{B}_j$, which is the implicit attention matrix
    - **Design Motivation**: Since $\bar{A}_t$ is a diagonal matrix, it can be further decomposed into the sum of $N$ independent internal attention matrices. One S6 channel produces $N$ internal attention matrices, resulting in $D \times N$ matrices across $D$ channels, which is significantly more than the $H$ attention heads in Transformers

2. **Query-Key-History decomposition**:

    - **Function**: Decompose the implicit attention matrix into an intuitive form similar to Q/K/V
    - **Mechanism**: After approximating softplus as ReLU, one can obtain $\tilde{\alpha}_{i,j} \approx \tilde{Q}_i \tilde{H}_{i,j} \tilde{K}_j$, where $\tilde{Q}_i = S_C(\hat{x}_i)$ corresponds to the query, $\tilde{K}_j = \text{ReLU}(S_\Delta(\hat{x}_j)) S_B(\hat{x}_j)$ corresponds to the key, and $\tilde{H}_{i,j} = \exp(\sum_{k=j+1}^{i} S_\Delta(\hat{x}_k)) A$ corresponds to the "historical context" term
    - **Design Motivation**: This decomposition reveals a key difference between Mamba and Transformer attention—Mamba additionally introduces $\tilde{H}_{i,j}$ to control the decay of historical token importance, which might explain why Mamba is better at modeling continuous historical contexts

3. **Mamba Attention Rollout**:

    - **Function**: Provide a class-agnostic interpretability method
    - **Mechanism**: Extract the implicit attention matrix $\tilde{\alpha}^{\lambda,d}$ for each layer and channel, average over the channel dimension, add the identity matrix (skip connection), and then multiply across layers to obtain the global attention $\rho = \prod_{\lambda=1}^{\Lambda} (\mathbb{I} + \mathbb{E}_{d}[\tilde{\alpha}^{\lambda,d}])$. For bidirectional Mamba, the attention matrices of both directions are summed
    - **Design Motivation**: Directly reuse the framework of Transformer Attention Rollout, with modifications only needed for the source of the attention matrix

4. **Mamba Attribution**:

    - **Function**: Provide a class-specific interpretability method
    - **Mechanism**: Combine the implicit attention matrix with gradient information: $\tilde{\beta}^\lambda = \mathbb{I} + (\mathbb{E}_{d}[\nabla \hat{y}'^{\lambda,d}] \odot \mathbb{E}_{d}[\tilde{\alpha}^{\lambda,d}])^+$. Unlike Transformer Attribution, gradients are not taken with respect to the attention matrix, but with respect to $\hat{y}'$ (the product of the S6 output and the gate) to capture the class-specific signals of both the S6 mixer and the gating mechanism simultaneously
    - **Design Motivation**: Directly replacing LRP scores with the attention matrix yields better results, and utilizing the gradient of the gating mechanism allows for more powerful class-specific attribution

### Loss & Training

This work does not involve training new models but analyzes existing pre-trained models. Vision experiments use pre-trained Vision Mamba (ViM) and DeiT, while NLP experiments use Mamba-130M and Pythia-160M.

## Key Experimental Results

### Main Results

| Method | Positive Perturbation AUC↓ (Mamba) | Positive Perturbation AUC↓ (Trans.) | Negative Perturbation AUC↑ (Mamba) | Negative Perturbation AUC↑ (Trans.) |
|--------|------|------|------|------|
| Raw-Attention | 17.27 | 20.69 | 34.03 | 40.77 |
| Attn-Rollout | 18.81 | 20.59 | 41.86 | 43.53 |
| Attribution | 16.62 | 15.35 | 39.63 | 48.09 |

### Segmentation Test (ImageNet-Segmentation)

| Model | Method | Pixel Accuracy↑ | mAP↑ | mIoU↑ |
|------|---------|------|------|------|
| Mamba | Raw-Attention | 67.64 | 74.88 | 45.09 |
| Transformer | Raw-Attention | 59.69 | 77.25 | 36.94 |
| Mamba | Attn-Rollout | 71.01 | 80.78 | 51.51 |
| Transformer | Attn-Rollout | 66.84 | 80.34 | 47.85 |
| Mamba | Attribution | 74.72 | 81.70 | 54.24 |
| Transformer | Trans.-Attribution | 79.26 | 84.85 | 60.63 |

### Key Findings

- Mamba's Raw Attention significantly outperforms Transformer's Raw Attention in terms of pixel accuracy and mIoU, indicating that the implicit attention matrix itself already possesses good interpretability.
- Under the Attn-Rollout method, Mamba fully outperforms the Transformer, whereas under the Attribution method, the Transformer performs better, suggesting that Mamba-Attribution may require further targeted design.
- The position of the CLS token significantly affects the attention distribution of Vision Mamba—patches closer to CLS have greater influence, implying that a non-spatial global CLS token might be a better choice.
- The Mamba attention matrix exhibits a highly similar structure to the Transformer: shallow layers focus on local diagonal patterns, and deep layers capture long-range dependencies.

## Highlights & Insights

- **Theoretical contribution is highly compelling**: It proves that a single-channel S6 layer can express all functions of a single-head Transformer, but not vice versa (Theorem 5.2). This theoretically explains why Mamba is at least comparable to Transformers in practice.
- **Staggering number of implicit attention matrices**: The number of attention matrices generated by Mamba is approximately $DN/H \approx 100N$ times that of a Transformer, but they share the Q matrix and are distinguished only by the K and H terms. This structure of "many lightweight attention matrices" could be the key to Mamba's efficiency.
- The paper also provides a theoretical analysis of the evolution of the attention mechanism in SSM models (Theorem 5.1): from S4 (fixed mixing) $\rightarrow$ GSS/Hyena (fixed mixing + diagonal data control) $\rightarrow$ Selective SSM (data-controlled non-diagonal mixing), revealing that the "data-controlled non-diagonal mixer" is the crucial capability shared by both Mamba and Transformers.

## Limitations & Future Work

- The segmentation performance of Mamba-Attribution is still lower than that of Transformer-Attribution, possibly because it was adapted directly from Transformer methods without fully utilizing Mamba-specific architectures.
- In perturbation experiments, Mamba consistently scores lower than Transformer under negative perturbation, likely because Mamba is more sensitive to patch masking; blurring rather than removing perturbations should be explored.
- The study only analyzes Mamba-1 (S6) and does not cover subsequent architectures like Mamba-2.
- In practical applications, extracting all $D \times N$ attention matrices could incur a high computational overhead, necessitating efficient approximation methods.

## Related Work & Insights

- **vs Transformer Attention**: Mamba's implicit attention is lower triangular (causal), does not use softmax (thus avoiding over-smoothing), and naturally encodes positional relationships and history decay through the $\tilde{H}_{i,j}$ term.
- **vs Traditional SSM (S4/DSS)**: The attention matrix of traditional SSMs is fixed (independent of input), whereas the attention of Selective SSM (Mamba) is data-dependent, which is the fundamental reason its expressive power significantly surpasses the former.
- **vs Attention Rollout / Chefer et al.**: This work adapts these two classic Transformer interpretability methods to Mamba, laying a foundation for interpretability research within the SSM model family.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically reveal Mamba's attention nature, with elegant theoretical derivations and highly insightful perspective shifts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers interpretability validation in both Vision and NLP fields, though quantitative comparisons could be deeper.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematically rigorous with a clear structure, though the high density of formulas might impact readability for some.
- **Value**: ⭐⭐⭐⭐⭐ Provides an important theoretical foundation for understanding and improving SSM models, offering extensive value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[ACL 2025\] Inferring Functionality of Attention Heads from their Parameters](inferring_functionality_of_attention_heads_from_their_parameters.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)

</div>

<!-- RELATED:END -->
