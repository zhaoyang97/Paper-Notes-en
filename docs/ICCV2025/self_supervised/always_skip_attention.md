---
title: >-
  [Paper Note] Always Skip Attention
description: >-
  [ICCV 2025][Self-Supervised Learning][self-attention] This paper theoretically demonstrates that the self-attention mechanism in Vision Transformers is inherently ill-conditioned, leading to training collapse in the absence of skip connections. It further proposes Token Graying (TG), a method that improves the condition number of input tokens to enhance ViT training stability and performance.
tags:
  - ICCV 2025
  - Self-Supervised Learning
  - self-attention
  - skip connection
  - condition number
  - ill-conditioning
  - token graying
  - Vision Transformer
date: 2026-05-08
content_hash: 48460315b2e08e43
---

# Always Skip Attention

**Conference**: ICCV 2025
**arXiv**: [2505.01996](https://arxiv.org/abs/2505.01996)
**Code**: Not yet publicly available
**Area**: Self-Supervised Learning / Vision Transformer / Theoretical Analysis
**Keywords**: self-attention, skip connection, condition number, ill-conditioning, token graying, Vision Transformer

## TL;DR

This paper theoretically demonstrates that the self-attention mechanism in Vision Transformers is inherently ill-conditioned, leading to training collapse in the absence of skip connections. It further proposes Token Graying (TG), a method that improves the condition number of input tokens to enhance ViT training stability and performance.

## Background & Motivation

**Background**: Vision Transformers (ViTs) have achieved remarkable success in computer vision tasks. Their core architecture consists of Self-Attention Blocks (SABs), feed-forward networks (FFNs), and skip connections. While skip connections are universally adopted as a standard component, a rigorous theoretical explanation for their role in ViTs has remained elusive.

**Limitations of Prior Work**:
   - The authors identify an intriguing, previously unreported empirical phenomenon: removing the skip connection from the SAB causes **catastrophic** performance collapse in ViTs (a 22% drop on CIFAR-10), whereas removing the FFN skip connection results in only a mild degradation (a 2% drop).
   - This asymmetric dependency does not exist in CNNs (e.g., ConvMixer)—removing skip connections has virtually no impact on CNN performance (±0.2%).
   - The only prior work attempting to train Transformers without skip connections required 5× more training time, and the underlying reason was left unexplained.

**Key Challenge**: Why is the self-attention mechanism so critically dependent on skip connections, while other components (FFN, convolution) are not? What is the fundamental mechanism at play?

**Goal**:
   - Provide a theoretical explanation for SAB's extreme reliance on skip connections.
   - Characterize the true role of skip connections within SABs.
   - Propose a novel method to improve ViT training based on these theoretical insights.

**Key Insight**: The analysis is conducted through the lens of the matrix condition number, which measures the degree of ill-conditioning. A larger condition number implies a more ill-conditioned Jacobian matrix and less stable gradient-based training.

**Core Idea**: The triple matrix multiplication structure inherent to self-attention causes the condition number of the output embeddings to grow as the **cube** of the input condition number, leading to intrinsic ill-conditioning. Skip connections serve precisely to regularize this condition number.

## Method

### Overall Architecture

The paper's contributions are divided into two parts:
1. **Theoretical Analysis**: Proving that SAB output embeddings are intrinsically ill-conditioned, and characterizing how skip connections ameliorate this.
2. **Token Graying (TG)**: A preprocessing method that improves the condition number of input tokens as a complementary enhancement to skip connections.

### Key Designs

1. **Ill-Conditioning Analysis of Self-Attention (Proposition 4.1)**:

    - Function: Theoretically derives an upper bound on the condition number of SAB output embeddings (without skip connections).
    - Mechanism: For linear attention (without softmax), the condition number of the SAB output $\mathbf{XW_QW_K^TX^TXW_V}$ satisfies $\kappa(\text{output}) \leq C \cdot (\sigma_{max}/\sigma_{min})^3$, i.e., the **cube** of the input matrix's condition number.
    - Design Motivation: This explains why the condition number of SAB output embeddings reaches $e^6$ experimentally, compared to $e^3$ for FFN outputs. In contrast, the FFN's multiplicative structure $\mathbf{W_{down}W_{up}X}$ yields a condition number upper bound that scales only **linearly** with the input condition number.
    - Although the theoretical proof is based on linear attention, experiments confirm that softmax attention exhibits the same ill-conditioning behavior.

2. **Regularizing Role of Skip Connections (Proposition 4.2)**:

    - Function: Theoretically proves that skip connections substantially improve the conditioning of SAB outputs.
    - Mechanism: $\kappa(\mathbf{XM + X}) \ll \kappa(\mathbf{XM})$; adding the identity mapping dramatically reduces the condition number.
    - This provides a rigorous mathematical explanation for why skip connections are indispensable to SABs: they function not merely as gradient highways, but as condition number regularizers.

3. **Token Graying (TG) — SVD Variant**:

    - Function: Reconstructs input tokens via SVD decomposition, amplifying small singular values to improve the condition number.
    - Mechanism: Applies SVD to the token matrix $\mathbf{X = U\Sigma V^T}$, normalizes the singular values, and amplifies them with an exponent $\epsilon \in (0,1]$ (i.e., $\tilde{\Sigma} = (\Sigma/\max(\Sigma))^\epsilon$), then reconstructs $\tilde{\mathbf{X}} = \mathbf{U\tilde{\Sigma}V^T}$.
    - Limitation: SVD computation is prohibitively expensive (approximately 6× slower training), making it impractical.

4. **Token Graying (TG) — DCT Variant**:

    - Function: Approximates the effect of SVD using the Discrete Cosine Transform (DCT).
    - Mechanism: In natural images, the dominant singular vectors of SVD typically correspond to low-frequency content, and DCT is inherently a frequency-domain transform. The formulation is $\hat{\mathbf{X}} = D\mathbf{X}D^T$, with amplification applied in the frequency domain followed by IDCT reconstruction.
    - Computational complexity is $O(nd\log(nd))$ versus $O(nd\min(n,d))$ for SVD, introducing negligible training overhead (0.732 days vs. 0.723 days baseline vs. 4.552 days for SVD).

### Loss & Training

The method introduces no new loss functions. Standard cross-entropy (supervised) or MSE (MAE self-supervised pretraining) is used throughout. TG is applied as a preprocessing step prior to patch embedding, with a single hyperparameter $\epsilon$ controlling the amplification magnitude (default: 0.95).

## Key Experimental Results

### Main Results

ImageNet-1K classification results across multiple ViT variants:

| Model | Top-1 Acc (%) | Top-5 Acc (%) | With DCTTG |
|------|---------|---------|------|
| ViT-S | 80.2 | 95.1 | — |
| ViT-S + DCTTG | 80.4 | 95.2 | ✓ |
| ViT-B | 81.0 | 95.3 | — |
| ViT-B + DCTTG | 81.3 | 95.4 | ✓ |
| Swin-S | 81.3 | 95.6 | — |
| Swin-S + DCTTG | 81.6 | 95.6 | ✓ |
| CaiT-S | 82.6 | 96.1 | — |
| CaiT-S + DCTTG | 82.7 | 96.3 | ✓ |
| PVT V2 b3 | 82.9 | 96.0 | — |
| PVT V2 b3 + DCTTG | 83.0 | 96.1 | ✓ |

Self-supervised learning (MAE pretraining + fine-tuning):

| Method | Top-1 Acc (%) | Top-5 Acc (%) |
|------|---------|---------|
| MAE | 83.0 | 96.4 |
| MAE + DCTTG | 83.2 | 96.6 |

### Ablation Study

Effect of different $\epsilon$ values on ViT-B (SVD variant):

| $\epsilon$ | Top-1 Acc | $\kappa_{in}$ (log) | $\kappa_{out}$ (log) |
|------|---------|------|------|
| — (baseline) | 81.0 | 6.72 | 6.74 |
| 0.9 | 81.2 | 6.64 | 6.66 |
| 0.7 | 81.4 | 6.15 | 6.17 |
| 0.6 | 81.4 | 5.73 | 5.71 |
| 0.5 | 81.0 | 5.29 | 5.25 |

Skip connection ablation (ViT-Tiny, CIFAR-10):

| Configuration | CIFAR-10 Acc | Notes |
|------|---------|------|
| Standard (SAB+FFN skip) | ~92% | Baseline |
| w/o FFN skip | ~90% | Mild degradation (−2%) |
| w/o SAB skip | ~70% | Catastrophic collapse (−22%) |

Training time comparison:

| Method | ViT-B Training Time (days) |
|------|---------|
| Baseline | 0.723 |
| + SVDTG | 4.552 |
| + DCTTG | 0.732 |

### Key Findings

- **SAB vs. FFN Asymmetry**: Removing the SAB skip connection causes the condition number to spike from $e^3$ to $e^6$, with training diverging after 30 epochs; removing the FFN skip has minimal impact.
- **CNNs Are Unaffected**: ConvMixer exhibits performance changes of <±0.2% upon skip connection removal, confirming that this is a pathology specific to self-attention.
- **Condition Number Improvement Correlates with Performance**: Performance peaks when $\epsilon$ is reduced to 0.6–0.7, yielding the best condition numbers; excessively small $\epsilon$ (e.g., 0.5) achieves good conditioning but may discard useful information.
- **DCT Is an Efficient Approximation of SVD**: Performance is comparable, but with virtually zero additional training overhead.

## Highlights & Insights

- **Profound Theoretical Insight**: This work is the first to reveal, through the lens of condition numbers, the fundamental reason for self-attention's dependence on skip connections. The cubic growth of the condition number arising from SAB's triple matrix multiplication structure constitutes an elegant and empirically verifiable theoretical explanation.
- **Simple and Practical Improvement**: DCT Token Graying requires no architectural modification, introduces no additional parameters, and incurs negligible training overhead (+1.2%). Its implementation is straightforward—a DCT-based frequency-domain amplification step applied before patch embedding.
- **Implications for ViT Design**: This finding suggests a new optimization direction for ViTs: rather than designing more complex attention mechanisms, one may instead focus on improving the condition number within the self-attention computation. This may inspire novel attention normalization or initialization strategies.

## Limitations & Future Work

- **Modest Performance Gains**: Improvements on ImageNet-1K are limited to 0.2–0.4%. While consistent across architectures, the absolute gains are small.
- **Potential Issues with Low-Precision Training**: DCT involves extensive multiplications and summations, which may be sensitive to quantization errors under low-precision arithmetic (FP16/BF16); this is not validated in the paper.
- **Theory Covers Only Linear Attention**: Proposition 4.1 is formally proved only for linear attention; the extension to softmax attention is supported solely by empirical evidence.
- **Future Directions**: Adaptive $\epsilon$ scheduling strategies (varying across layers or training stages) could be explored. Condition number regularization could also be incorporated directly as part of the training objective.

## Related Work & Insights

- **vs. He et al. 2023 (Deep Transformers without Shortcuts)**: That work achieved skip-free training by introducing inductive biases into self-attention, but at the cost of 5× training time. The present analysis suggests that its success may be fundamentally attributed to implicit condition number improvement.
- **vs. ResNet Skip Connections**: In CNNs, skip connections primarily alleviate vanishing gradients; in ViTs, they play the more critical role of condition number regularization. This explains why VGG (a skip-free CNN) remains trainable, whereas skip-free ViTs do not.
- **vs. cosFormer, Sigmoid Attention, and Other Attention Variants**: These works replace softmax with alternative activation functions. The condition number perspective introduced here offers a new theoretical criterion for evaluating such alternatives.
- This finding carries significant implications for large-scale ViT design: as model depth increases, the cumulative degradation of condition numbers may be a primary source of training instability in deep Transformers.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to explain SAB's skip connection dependency through condition numbers; theoretically incisive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple architectures and tasks, though absolute performance gains are modest.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; experimental presentation is systematic.
- Value: ⭐⭐⭐⭐ Theoretical contribution outweighs empirical gains, but insights for ViT design are of high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[ICCV 2025\] From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations](from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)
- [\[ICCV 2025\] To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)

<!-- RELATED:END -->
