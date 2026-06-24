---
title: >-
  [Paper Note] Value Residual Learning
description: >-
  [ACL 2025][Value Residual Connections] The authors propose ResFormer and SVFormer, which introduce a residual connection from the first-layer Value vector to subsequent layers in the attention mechanism. This dynamic enhances the propagation of initial token-level information in deep networks. Consequently, these models achieve comparable performance to standard Transformers with 16.11% fewer parameters and 20.3% less training data, while SVFormer also reduces KV cache by nea…
tags:
  - "ACL 2025"
  - "Value Residual Connections"
  - "Transformer Architecture"
  - "Information Propagation"
  - "KV Cache Compression"
  - "Over-smoothing"
date: 2026-05-08
content_hash: 6b8cfa57bba29e53
---

# Value Residual Learning

**Conference**: ACL 2025  
**arXiv**: [2410.17897](https://arxiv.org/abs/2410.17897)  
**Code**: None  
**Area**: Others  
**Keywords**: Value Residual Connections, Transformer Architecture, Information Propagation, KV Cache Compression, Over-smoothing

## TL;DR

The authors propose ResFormer and SVFormer, which introduce a residual connection from the first-layer Value vector to subsequent layers in the attention mechanism. This dynamic enhances the propagation of initial token-level information in deep networks. Consequently, these models achieve comparable performance to standard Transformers with 16.11% fewer parameters and 20.3% less training data, while SVFormer also reduces KV cache by nearly half.

## Background & Motivation

Standard Transformers propagate information through hidden state residual connections across deep networks, but suffer from a core problem: **over-smoothing**. The smoothing effect of self-attention causes token representations to become increasingly similar as networks deepen, resulting in sequence-level features dominating while token-level features are diluted.

Limitations of Prior Work:
- **DenseFormer**: Uses learnable dense connections (similar to DenseNet). The learned connection coefficients indicate that deeper layers indeed require more attention to initial embeddings. However, directly summing initial embeddings and deep hidden states can significantly affect the modeling of attention distributions in higher layers.
- **NeuTRENO**: Ameliorates over-smoothing by adding the difference between the first-layer and current-layer Value vectors, but handles this from a regularization perspective, leading to limited effectiveness.

Key Insight: Although the initial token embedding ($\mathbf{H}_0$) and the first-layer Value state ($\mathbf{V}_1$) both contain local token information (differing only by a linear transformation), transmitting initial information through a Value residual rather than a hidden state residual minimizes interference with the attention distribution. Because the Value residual connection is introduced prior to the attention matrix computation and shared with the existing attention matrix, it does not alter the modeling of attention patterns.

## Method

### Overall Architecture

Building on the standard Transformer, ResFormer introduces only a simple Value residual connection: it performs a weighted fusion of the first-layer Value vector $\mathbf{V}_1$ and the current-layer Value vector $\mathbf{V}_n$, which then share the current layer's attention matrix. SVFormer goes a step further by sharing the first-layer Value state across all layers.

### Key Designs

1. **ResFormer Core Formula**: $\mathbf{V}_n = \lambda_{n,1}\mathbf{V}_1 + \lambda_{n,2}\mathbf{H}_{n-1}\mathbf{W}_n^V$. Here, $\lambda$ can be a fixed constant (Constant-ResFormer), identity/uniform weight (Identity-ResFormer), learnable parameters (Learnable-ResFormer), or sparsely applied (Sparse-ResFormer). The design motivation is to allow deep networks direct access to original token-level information that has not been diluted by attention smoothing.

2. **Variant Designs**:

    - **Identity-ResFormer**: $\lambda_{n,1} = \lambda_{n,2} = 0.5$, representing the simplest variant.
    - **Constant-ResFormer**: Manually tuned constant, with the optimal being $\lambda = 2$.
    - **Sparse-ResFormer**: Applies the Value residual only in the final few layers. Experiments show that the last 3 layers (layers 6–8 in an 8-layer model) benefit the most.
    - **Learnable-ResFormer Plus**: Utilizes adaptive initialization that assigns larger weights to $\mathbf{V}_1$ in deeper layers.

3. **SVFormer**: Decouples the Value from attention operations, where all layers share the first layer's Value: $\mathbf{U}_n = \mathbf{A}_n\mathbf{V}_1$. The main advantage is that it only requires computing and storing the first layer's Value vector, **reducing the KV cache by nearly half**. Experiments show that the negative impact of sharing Values is much smaller than sharing Keys.

4. **Dense-ResFormer**: The most general form, $\mathbf{V}_n = \lambda_{n,n}\mathbf{H}_{n-1}\mathbf{W}_n^V + \sum_{i=1}^{n-1}\lambda_{n,i}\mathbf{V}_i$, which allows Value connections across all preceding layers.

### Loss & Training

- Uses the AdamW optimizer with a weight decay of 0.1, $\beta_1=0.9$, and $\beta_2=0.95$.
- Batch size is approximately 2M tokens, sequence length is 2048, trained for 10,000 steps.
- Linear learning rate warmup for 1200 steps with a peak of 6e-4, decaying cosinely to 10% of the peak.
- Training data: Sub-sampled 20B SlimPajama.
- Standard language modeling objective.

## Key Experimental Results

### Main Results

| Model | Params | Wiki. PPL | Downstream Avg. ACC |
|------|------|-----------|----------------|
| Transformer | 468M | 24.8 | 40.6 |
| NeuTRENO | 468M | 24.3 | 41.4 |
| DenseFormer | 468M | 24.0 | 40.8 |
| Identity ResFormer | 468M | 23.8 | 41.3 |
| Learnable ResFormer | 468M | 23.7 | **42.3** |
| Learnable ResFormer Plus | 468M | **23.2** | 42.0 |

### Scaling Experiments

| Metric | ResFormer vs Transformer |
|------|------------------------|
| Parameter Efficiency | Requires 16.11% fewer parameters to achieve equivalent valid loss |
| Data Efficiency | Requires 20.3% less training data to achieve equivalent valid loss |
| 1.6B Scale | Trained on 200B tokens, Value residuals consistently improve performance |

### Ablation Study

| Configuration | Valid Loss | Description |
|------|----------|------|
| Vanilla Transformer | 2.739 | Baseline |
| Identity-ResFormer ($\lambda=0.5$) | 2.712 | Significant improvement with the simplest variant |
| Constant-ResFormer ($\lambda=2$) | 2.700 | Manual tuning performs better |
| Sparse-ResFormer (Layers 6-8, $\lambda=5$) | **2.687** | Best performance when applied only to the final few layers |
| ResFormer-Plus (Learnable) | 2.681 | Adaptive initialization is optimal |
| Additional Hidden Residual to $\mathbf{H}_0$ | 2.781 | Detrimental |
| Query Residual | 2.742 | No benefit |
| Key Residual | 2.746 | No benefit |
| Attention Residual | 2.757 | Detrimental |

### Key Findings

1. **Why $\mathbf{V}_1$ and not $\mathbf{V}_2$**: Only skip connections from the first-layer Value significantly improve performance. This occurs because the default hidden state residual already propagates information from $\mathbf{H}_1$ to subsequent layers ($\mathbf{V}_2 = \mathbf{H}_1\mathbf{W}_2^V$), whereas information from $\mathbf{H}_0$ is lost due to dilution by subsequent information. When the residual connection from $\mathbf{H}_0$ to $\mathbf{H}_1$ is removed (preventing $\mathbf{V}_2$ from obtaining $\mathbf{H}_0$'s information via the hidden residual), the skip connection from $\mathbf{V}_2$ begins to provide a significant gain.

2. **Deeper layers benefit the most**: For an 8-layer model, applying the Value residual solely on layer 7 yields the best single-layer performance; expanding it to layers 6–8 achieves the optimal overall effect, while benefits diminish when extending from layer 5 onwards.

3. **Value residual outperforms hidden residual**: An extra hidden state residual is equivalent to simultaneously applying residuals to Q, K, and V, which interferes with the attention distribution.

4. **Learned $\lambda$ patterns**: The learnable ResFormer automatically discovers that deeper layers require more information from $\mathbf{V}_1$, aligning with the manually configured pattern of Sparse-ResFormer.

## Highlights & Insights

1. **Minimalist design, profound insights**: Adding only a single Value residual connection (with almost zero extra parameters) significantly enhances performance, demonstrating impressive simplicity in design.
2. **Thorough ablation analysis**: Validates design choices from multiple angles, answering questions such as "Why Value instead of Q/K/A?", "Why $\mathbf{V}_1$ instead of $\mathbf{V}_2$?", and "Why not use extra hidden residuals?".
3. **Practical value of SVFormer**: Reducing the KV cache by nearly half is extremely valuable for long-sequence inference deployment, and the approach can be combined with other methods like GQA.
4. **Beyond training acceleration**: Experiments with different learning rates demonstrate that the performance gain is not merely due to shortcut connections accelerating training convergence, but because of learning fundamentally better representations.
5. **Information propagation perspective**: Reveals a critical issue in standard Transformers where initial token-level information is diluted in deeper layers.

## Limitations & Future Work

1. SVFormer requires 12.2% more parameters to achieve an equivalent valid loss to standard Transformers, though it performs better at high sequence lengths.
2. Evaluated only at 82M–1.6B parameter scales; effectiveness on larger models remains to be verified.
3. The optimal layer configuration for Sparse-ResFormer requires manual search; while the learnable version is automatic, it may not be optimal.
4. The combination of Value residuals with other efficient methods (e.g., MoE, sparse attention) has not been explored.
5. Theoretical explanations are primarily intuitive and lack rigorous mathematical proofs.

## Related Work & Insights

- **Shortcut Connections**: Evolutionary trajectory from ResNet and DenseNet to DenseFormer.
- **KV Cache Compression**: Methods such as MQA, GQA, and CLA; this study is the first to propose solely decoupling Value.
- **Over-smoothing**: Zhou et al. 2021 (32-layer ViT underperforming 24-layer), Shi et al. 2022 (over-smoothing in BERT).
- **Insight**: Value residuals can be applied during fine-tuning or continual pre-training of existing large models as a low-cost architectural enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of Value residual connections is remarkably simple yet effective, and SVFormer's decouple of Value is also a pioneering attempt.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The ablation studies are textbook quality, rigorously verifying the design decisions across various dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic where every design choice is supported by empirical results, with precise and informative charts.
- Value: ⭐⭐⭐⭐⭐ The method is simple, highly generalizable, plug-and-play, and exerts a profound impact on standard Transformer architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](../../ICML2025/others/residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)
- [\[ICML 2025\] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions](../../ICML2025/others/learning_safe_strategies_for_value_maximizing_buyers_in_uniform_price_auctions.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)

</div>

<!-- RELATED:END -->
