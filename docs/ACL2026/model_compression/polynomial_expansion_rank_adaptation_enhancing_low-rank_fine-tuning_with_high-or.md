---
title: >-
  [Paper Note] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions
description: >-
  [ACL 2026][Model Compression][Low-Rank Adaptation] This paper proposes PERA (Polynomial Expansion Rank Adaptation), which extends the linear adaptation space of LoRA into a polynomial manifold by introducing structured p…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Low-Rank Adaptation"
  - "Polynomial Expansion"
  - "High-Order Feature Interaction"
  - "PEFT"
  - "LoRA Improvement"
date: 2026-05-08
content_hash: 671e8751a9aa214d
---

# Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions

**Conference**: ACL 2026  
**arXiv**: [2604.11841](https://arxiv.org/abs/2604.11841)  
**Code**: [https://github.com/zhangwenhao6/PERA](https://github.com/zhangwenhao6/PERA)  
**Area**: Parameter-Efficient Fine-Tuning / Model Compression  
**Keywords**: Low-Rank Adaptation, Polynomial Expansion, High-Order Feature Interaction, PEFT, LoRA Improvement

## TL;DR

This paper proposes PERA (Polynomial Expansion Rank Adaptation), which extends the linear adaptation space of LoRA into a polynomial manifold by introducing structured polynomial expansions (square and cross terms) in the parameter space of low-rank factors. This significantly enhances the expressive power of weight updates without increasing rank or inference overhead, consistently outperforming methods like LoRA, DoRA, and HiRA on commonsense reasoning and NLU tasks.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) has become the standard paradigm for adapting Large Language Models. Among these, LoRA achieves efficiency by constraining weight updates to a low-rank subspace $\Delta W = BA$. However, its strict bilinear structure only captures first-order linear dependencies between low-rank factors, limiting the model's ability to model non-linear and high-order parameter interactions.

**Limitations of Prior Work**: (1) The weight update in LoRA, $\Delta W = \sum_{i=1}^{r} \mathbf{b}_i \mathbf{a}_i^T$, is a linear combination of rank-one matrices, restricting expressive power by the rank $r$; (2) DoRA improves upon LoRA via magnitude-directional decomposition but remains a linear transformation; (3) MoRA achieves high-rank adaptation through compression-transformation-decompression but introduces additional overhead; (4) HiRA enriches representations via Hadamard modulation with pre-trained weights, but the update mechanism remains linear with respect to trainable parameters and depends on external weight coupling.

**Key Challenge**: From the perspective of function approximation, there is a fundamental difference in expressive power between a first-order linear function $f(x) = c + c_1 x$ and a polynomial function containing high-order terms $f(x) = c + c_1 x + c_2 x^2 + \cdots$. If LoRA is viewed as a first-order linear approximation of weight updates, its limitations in expressive power are fundamental.

**Goal**: To enhance the expressive power of low-rank adaptation by introducing high-order feature interactions without increasing rank or inference costs.

**Key Insight**: Inspiration is drawn from polynomial feature expansion techniques in classical feature engineering, identifying their potential when applied to the parameter space of low-rank factors rather than the input feature space.

**Core Idea**: Polynomial expansion and Hadamard-based polynomial expansion are applied to low-rank matrices $B$ and $A$, respectively, to generate square terms ($\mathbf{b}_i \odot \mathbf{b}_i$) and cross terms ($\mathbf{b}_i \odot \mathbf{b}_j$). By using matrix concatenation (rather than addition), additional inference overhead is avoided while expanding the adaptation space from a linear subspace to a polynomial manifold.

## Method

### Overall Architecture

PERA follows the decomposition framework of LoRA, decomposing the weight update into $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$. The core improvement lies in performing polynomial expansion on both factors before combination: a standard second-order polynomial expansion $\text{Poly}^2(B)$ for $B$, and a Hadamard-based polynomial expansion $\text{Poly}_H^2(A)$ for $A$ (with learnable coefficients $\mathbf{h}$ to ensure stability). The final update is $\Delta W = \text{Poly}^2(B) \cdot \text{Poly}_H^2(A)$.

### Key Designs

1. **Parameter Space Polynomial Expansion**:
    - **Function**: Expands the low-rank factors from $r$ dimensions to $2r + C(r,2)$ dimensions, introducing high-order non-linear interactions.
    - **Mechanism**: For $B = [\mathbf{b}_1, \ldots, \mathbf{b}_r]$, a second-order polynomial expansion generates $\hat{B} = [B; B_{square}; B_{cross}]$, where $B_{square} = \{\mathbf{b}_i \odot \mathbf{b}_i\}$ contains $r$ square terms and $B_{cross} = \{\mathbf{b}_i \odot \mathbf{b}_j | i < j\}$ contains $C(r,2)$ cross terms. Matrix $A$ is expanded similarly but includes learnable coefficients $h_{ij}$ (initialized to zero) to maintain training stability. The final weight update is decomposed as: $\Delta W = \sum_{i} \mathbf{b}_i \mathbf{a}_i^T + \sum_{i=j} h_{ij}(\mathbf{b}_i \odot \mathbf{b}_j)(\mathbf{a}_i^T \odot \mathbf{a}_j^T) + \sum_{i<j} h_{ij}(\mathbf{b}_i \odot \mathbf{b}_j)(\mathbf{a}_i^T \odot \mathbf{a}_j^T)$.
    - **Design Motivation**: Since polynomial expansion is a classic feature enhancement technique, migrating it from feature space to parameter space is a natural generalization. The column vectors of low-rank factors are themselves "features" of the adaptation direction; high-order interaction terms capture the non-linear coupling between these directions.

2. **Zero-initialization Strategy for Hadamard Coefficients**:
    - **Function**: Ensures the training starting point is consistent with LoRA, with high-order terms gradually participating in optimization.
    - **Mechanism**: Learnable coefficients $\mathbf{h} = \{h_{ij}\}$ are initialized to zero, making PERA degenerate into standard LoRA at the beginning of training (where square and cross terms contribute zero when $\mathbf{h}=0$). As training progresses, the model autonomously learns which high-order interactions are beneficial. LoRA is thus a special case of PERA.
    - **Design Motivation**: Zero initialization prevents high-order terms from introducing unstable gradients during early training while preserving the full upper bound of expressive power. This "progressive" introduction of non-linearity ensures optimization smoothness.

3. **Zero Inference Overhead via Matrix Concatenation**:
    - **Function**: High-order terms are implemented via column/row concatenation, allowing them to be pre-computed and merged into weights during inference.
    - **Mechanism**: The product of expanded $\hat{B} \in \mathbb{R}^{m \times (2r+C(r,2))}$ and $\hat{A} \in \mathbb{R}^{(2r+C(r,2)) \times n}$ remains an $\mathbb{R}^{m \times n}$ matrix. During inference, $\Delta W = \hat{B}\hat{A}$ can be pre-computed and merged into $W_0$, thus introducing no inference latency.
    - **Design Motivation**: Inference efficiency is critical in real-world deployment. PERA achieves high-order interaction through concatenation rather than sequential addition, maintaining the zero-inference-overhead property of LoRA.

### Loss & Training

The standard next-token prediction loss is utilized. During training, only the low-rank matrices $A, B$ and Hadamard coefficients $\mathbf{h}$ are optimized, while the pre-trained weights $W_0$ remain frozen. The learning rate is set to $1 \times 10^{-4}$, and other hyperparameters are kept consistent with the HiRA baseline. $A$ is initialized to zero, and $B$ is initialized with a Gaussian distribution.

## Key Experimental Results

### Main Results

| Model | Method | Params(%) | Commonsense Reasoning Avg Acc |
|------|------|----------|---------------|
| LLaMA2-7B | LoRA (r=32) | 0.83% | 77.61 |
| LLaMA2-7B | DoRA (r=32) | 0.83% | 79.69 |
| LLaMA2-7B | HiRA (r=32) | 0.83% | 81.42 |
| LLaMA2-7B | **PERA (r=16)** | **0.41%** | **82.61** |
| LLaMA3-8B | LoRA (r=16) | 0.35% | 82.80 |
| LLaMA3-8B | HiRA (r=16) | 0.35% | 86.08 |
| LLaMA3-8B | **PERA (r=16)** | **0.35%** | **87.38** |
| Qwen2.5-7B | LoRA (r=16) | 0.35% | 73.80 |
| Qwen2.5-7B | HiRA (r=16) | 0.35% | 85.40 |
| Qwen2.5-7B | **PERA (r=16)** | **0.35%** | **88.29** |

| Model | Method | Params | GLUE Avg |
|------|------|-------|----------|
| RoBERTa-base | LoRA | 0.3M | 83.40 |
| RoBERTa-base | DeLoRA | 0.3M | 84.60 |
| RoBERTa-base | **PERA** | 0.3M | **85.10** |
| RoBERTa-large | LoRA | 0.8M | 87.30 |
| RoBERTa-large | **PERA** | 0.8M | **88.13** |

### Ablation Study

| Config | Weight Update Formula | Avg Acc |
|------|-----------|----------|
| LoRA (First-order only) | Eq.8 | 82.80 |
| LoRA + Square terms only | Eq.10 | 87.48 |
| LoRA + Cross terms only | Eq.11 | 86.83 |
| PERA (Square + Cross) | Eq.9 | 87.38 |

### Key Findings

- **Significant Gains from High-Order Terms**: PERA outperforms LoRA by 5 percentage points on LLaMA2-7B (82.61% vs 77.61%) and by 14.5 percentage points on Qwen2.5-7B (88.29% vs 73.80%).
- **Square Terms are Crucial**: Adding only square terms (87.48%) yields a larger gain than adding only cross terms (86.83%) and is close to the full PERA (87.38%), indicating that non-linear interactions within the same dimension are more important than those across dimensions.
- **Superior Performance at Extremely Low Rank**: PERA achieves 86.91% even at $r=2$ and 87.01% at $r=4$, nearing the peak result of 87.38% at $r=16$. This is attributed to the expansion, which increases the theoretical upper bound of the effective rank from $r$ to $2r + C(r,2)$.
- **Training and Inference Overhead Comparable to LoRA**: Training memory is 19.12GB vs LoRA 18.70GB, and inference memory is 19.70GB vs 19.50GB, performing significantly better than DoRA (22h07m training time vs PERA 13h30m).
- **Surpassing LoRA Full Data with 10% Data**: PERA achieved 83.07% on 10% of commonsense170K, exceeding LoRA's 82.80% trained on 100% of the data, demonstrating exceptional data efficiency.

## Highlights & Insights

- **Elegant Migration from Feature to Parameter Engineering**: Migrating polynomial feature expansion from input feature space in traditional ML to the parameter space of low-rank adaptation is a conceptually simple yet highly effective and novel design approach.
- **LoRA as a Special Case of PERA**: When $\mathbf{h}=0$, PERA degenerates into LoRA. This provides a beautiful theoretical unification and justifies the progressive introduction strategy of zero initialization.
- **Theoretical Increase in Rank Bound**: While the rank upper bound for LoRA's adaptation weight is $r_0 + r$, PERA increases this to $r_0 + 2r + C(r,2)$. For $r=16$, this expands from $r_0+16$ to $r_0+152$, nearly a ten-fold increase in theoretical expressive power.
- **Hessian Interaction Strength Analysis**: By calculating the interaction strength matrix of second-order partial derivatives, it is visually demonstrated that PERA possesses a stronger capability for modeling global feature interactions compared to LoRA.

## Limitations & Future Work

- Evaluations were restricted to commonsense reasoning and GLUE, without covering tasks like arithmetic reasoning, code generation, or multi-modal generation.
- Only second-order polynomial expansion was used; the efficacy of higher-order expansions ($k>2$) has not been explored.
- The contribution of cross terms was limited (87.38% vs 87.48% for square terms only), suggesting potential redundancy and a need for finer-grained term selection strategies.
- Comparisons with the latest Mixture-of-Experts LoRA (e.g., MELoRA) or adaptive rank methods (e.g., AdaLoRA) were not conducted.
- Polynomial expansion might lead to dimensional explosion at high ranks ($C(r,2)$ grows quadratically with $r$), necessitating research into scalability for high-rank scenarios.

## Related Work & Insights

- **vs LoRA**: PERA is a strict generalization of LoRA that introduces high-order terms through polynomial expansion. LoRA corresponds to the special case where $\mathbf{h}=0$.
- **vs HiRA**: HiRA introduces non-linearity through Hadamard products with pre-trained weights, relying on external weight coupling. PERA introduces high-order interactions entirely within the trainable parameters, independent of external modules.
- **vs DoRA**: DoRA decomposes magnitude and direction but remains a linear transformation, whereas PERA introduces true non-linear parameter interactions.
- **vs MoRA**: MoRA increases expressive power via high-rank transformations but introduces extra inference overhead; PERA maintains zero inference overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of applying polynomial expansion to the parameter space of low-rank factors is novel with clear theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models and tasks with comprehensive ablations on rank, modules, data volume, and components.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological description, rigorous mathematical derivation, and an elegant demonstration of the relationship with LoRA.
- Value: ⭐⭐⭐⭐ Provides a simple and effective enhancement for LoRA with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](../../ICML2026/model_compression/scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[ACL 2026\] Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation](not_all_directions_matter_towards_structured_and_task-aware_low-rank_model_adapt.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](../../NeurIPS2025/model_compression/beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)

</div>

<!-- RELATED:END -->
