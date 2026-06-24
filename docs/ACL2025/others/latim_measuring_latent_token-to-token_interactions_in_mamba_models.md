---
title: >-
  [Paper Note] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models
description: >-
  [ACL 2025][State Space Models] Proposed LaTIM, a token-level decomposition method tailored for Mamba-1 and Mamba-2 that reconstructs the implicit computations of SSM into a Transformer-like token-to-token contribution matrix, enabling fine-grained interpretability analysis for Mamba models.
tags:
  - "ACL 2025"
  - "State Space Models"
  - "Mamba"
  - "Interpretability"
  - "Token Interaction Decomposition"
  - "Attention Attribution"
date: 2026-05-08
content_hash: 67d16c16521b8534
---

# LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models

**Conference**: ACL 2025  
**arXiv**: [2502.15612](https://arxiv.org/abs/2502.15612)  
**Code**: [https://github.com/deep-spin/latim](https://github.com/deep-spin/latim)  
**Area**: Other  
**Keywords**: State Space Models, Mamba, Interpretability, Token Interaction Decomposition, Attention Attribution

## TL;DR

Proposed LaTIM, a token-level decomposition method tailored for Mamba-1 and Mamba-2 that reconstructs the implicit computations of SSM into a Transformer-like token-to-token contribution matrix, enabling fine-grained interpretability analysis for Mamba models.

## Background & Motivation

State Space Models (SSMs) such as Mamba have emerged as efficient alternatives to Transformers, capable of processing long sequences with linear complexity. However, while Transformers possess attention matrices as a natural interpretability tool to intuitively display interactions between tokens, Mamba lacks a similar explicit mechanism.

Existing work on Mamba interpretability has several limitations:
- **MambaAttention** (Ali et al., 2024), though reformulating Mamba's computation as an implicit attention matrix, struggles because the channel dimension in Mamba-1 is often very large (e.g., $D=1024$ channels for a 370M model), making it impossible to produce a single attention map per layer.
- **MambaLRP** (Jafari et al., 2024) utilizes layer-wise relevance propagation to analyze gradient flow, but only supports Mamba-1 and cannot explicitly decompose the contributions of individual tokens.
- None of these methods **achieve fine-grained token-level contribution decomposition** similar to that in Transformers.

This work bridges this interpretability gap by introducing LaTIM, enabling researchers to apply mature attribution methods like ALTI to Mamba models.

## Method

### Overall Architecture

The core idea of LaTIM is to rearrange the forward computation of Mamba so that the output $\boldsymbol{y}_i$ can be expressed as the sum of contributions $T_i(\boldsymbol{x}_j)$ from all preceding tokens, i.e., $\boldsymbol{y}_i = \sum_{j=1}^{i} T_i(\boldsymbol{x}_j)$. This directly corresponds to the attention decomposition form in Transformers, thereby allowing the reuse of existing attribution techniques.

### Key Designs

1. **Mamba-1 Decomposition**:

    - First, the SSM recurrence is unrolled to obtain the implicit attention tensor $\boldsymbol{M}_{i,j}$, representing the implicit contribution of token $j$ to token $i$.
    - The key challenge is the non-additivity of the SiLU activation function—making it impossible to directly split the output of the convolutional layer by tokens.
    - Solution: Assume there exists an additive function $f$ that approximates SiLU, decomposing the post-convolution activation into the independent contribution of each token.
    - Experimental verification shows that directly setting $f := \text{SiLU}$ unexpectedly yields the lowest approximation error across all layers.
    - Finally, combining the gating mechanism and the output projection, the $(i,j)$ contribution vector is obtained: $T_i(\boldsymbol{x}_j) = \boldsymbol{W}_o^\top (\boldsymbol{Z}_i \odot \boldsymbol{\upsilon}_{i \leftarrow j})$

2. **Mamba-2 Decomposition**:

    - The $\boldsymbol{A}$ matrix in Mamba-2 is simplified to a scalar multiplied by the identity matrix, making the decomposition more concise.
    - The newly added GroupNorm layer can be viewed as an affine map with respect to $\boldsymbol{u}_i$ during inference, allowing the contribution of each token to pass through linearly.
    - The final decomposition is: $T_i(\boldsymbol{x}_j) = \boldsymbol{W}_o^\top [\gamma_i(\boldsymbol{u}_i) \boldsymbol{u}_{i \leftarrow j}]$

3. **Multiple Aggregation Approaches**:

    - **LaTIM($\ell_p$)**: Uses vector norms to measure the magnitude of contributions.
    - **LaTIM(ALTI)**: Adopts a context-mixing approach, calculating the change in the $\ell_1$ norm after removing the contribution of a specific token.
    - **LaTIM(ALTI-Logit)**: Tracks the contribution of tokens to the final prediction through the residual stream.

4. **Exact Decomposition Strategy**: An alternative Mamba variant without SiLU activation is proposed (setting $f$ to the identity function), which requires retraining but achieves zero approximation error. Experiments show that this variant achieves fully exact decomposition while preserving task performance.

### Loss & Training

- The exact strategy requires retraining the model (removing SiLU), while the approximation strategy ($f := \text{SiLU}$) can be directly applied to pre-trained models.
- Models for the copy task are trained from scratch using the mimetic initialization scheme.
- Machine translation models are fine-tuned on the IWSLT17 dataset.
- Approximation error experiments are conducted via continued pre-training on FineWeb-Edu.

## Key Experimental Results

### Main Results

**Copy Task (Synthetic Benchmark)**:

| Method | AUC | AP | R@K |
|------|-----|-----|-----|
| Mamba-Attention (M1) | 0.84 | 0.36 | 0.22 |
| MambaLRP (M1) | 0.40 | 0.22 | 0.20 |
| **LaTIM(ALTI) (M1)** | **0.86** | **0.47** | **0.36** |
| Mamba-Attention (M2) | 0.79 | 0.49 | 0.39 |
| **LaTIM($\ell_2$) (M2)** | **0.98** | **0.86** | **0.74** |

**Machine Translation AER (IWSLT17 de→en, GoldAlign)**:

| Method | M1-Small | M1-Large | M2-Small | M2-Large |
|------|----------|----------|----------|----------|
| Mamba-Attention | 0.84 | 0.85 | 0.84 | 0.85 |
| LaTIM($\ell_2$) | **0.46** | **0.44** | **0.49** | **0.52** |
| LaTIM(ALTI-Logit) | 0.68 | 0.69 | 0.63 | 0.69 |

### Ablation Study

**Approximation Error Analysis (Different Activation Functions)**:

| Activation Function | Layer 0-16 Error | Layer 16-32 Error | AER | COMET |
|---------|-----------|-----------|-----|-------|
| SiLU (Default) | 0.21 | 0.45 | 0.47 | 83.4 |
| SiLU + Continued Pre-training | 0.21 | 0.43 | 0.46 | 83.6 |
| ReLU | 0.35 | 0.83 | 0.51 | 82.8 |
| Identity (Exact) | **0.00** | **0.00** | **0.46** | 83.3 |

### Key Findings

- LaTIM achieves an R@K of 0.74 on the Mamba-2 copy task, nearly doubling the 0.39 achieved by Mamba-Attention.
- Layer-wise analysis performs better than global aggregation—AER is lower when layer-wise methods are used for translation alignment.
- The exact strategy of removing SiLU achieves zero approximation error without compromising performance.
- Mamba has a pronounced limitation in multi-key retrieval tasks: accuracy drops sharply as the number of keys increases.
- Mamba's attention to repeated words decays over time, which explains its failure in word frequency extraction tasks.

## Highlights & Insights

- **Methodological Elegance**: Ingeniously unrolls the recurrence of SSMs into an attention-like matrix, allowing a wealth of attribution methods developed for Transformers to be seamlessly migrated to Mamba.
- **Counter-intuitive Finding in SiLU Approximation**: Directly using SiLU as the additive approximation function unexpectedly yields lower error than more "formal" methods such as Taylor expansion.
- **High Scalability**: LaTIM is not only applicable to Mamba-1/2 but can in principle be extended to other linear recurrence architectures such as DeltaNet and mLSTM.
- **Mechanistic Explanation for Mamba's Limitations**: Reveals the attention dispersion issue of Mamba in multi-key retrieval through visualization.

## Limitations & Future Work

- Approximation decomposition still carries errors, while the exact version requires removing SiLU and retraining.
- Evaluations are mainly focused on tasks with clear token interaction patterns, such as copy and translation; the quality of interpretability in more complex tasks remains to be verified by human evaluation.
- Extra adaptation is required for hybrid architectures (Attention + SSM).
- At present, it only showcases "what is seen"; its practical guiding value on "how to improve the model" requires further exploration.

## Related Work & Insights

- Forms a perfect correspondence with attention decomposition methods for Transformers (Kobayashi et al., 2021; Ferrando et al., 2022, 2023).
- Complements theoretical analyses of Mamba (Vo et al., 2025 on the asymptotic behavior of token states; Trockman et al., 2024 on mimetic initialization).
- Finds that the linear Mamba variant without SiLU is both interpretable and performs without degradation, echoing the related work of Bick et al., 2024.

## Rating

- Novelty: ⭐⭐⭐⭐ Reconstructing SSM recurrence into a token-to-token decomposition is a natural yet valuable contribution, with the exact strategy further enhancing the completeness of the method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three different tasks (copy, translation, retrieval-augmented generation), multiple model scales, and rich quantitative and qualitative analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, progressive background introduction, and excellent figure/table designs.
- Value: ⭐⭐⭐⭐ Provides a much-needed interpretability tool for the increasingly popular Mamba architecture, offering broad practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)

</div>

<!-- RELATED:END -->
