---
title: >-
  [Paper Note] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions
description: >-
  [ACL 2026][Model Compression][Low-rank adaptation] This paper proposes PERA (Polynomial Expansion Rank Adaptation), which introduces structured polynomial expansions (square and cross terms) into the parameter space of l…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Low-rank adaptation"
  - "polynomial expansion"
  - "high-order feature interaction"
  - "parameter-efficient fine-tuning"
  - "LoRA improvement"
date: 2026-05-08
content_hash: ac6ea0460b3269e7
---

# Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions

**Conference**: ACL 2026
**arXiv**: [2604.11841](https://arxiv.org/abs/2604.11841)  
**Code**: [https://github.com/zhangwenhao6/PERA](https://github.com/zhangwenhao6/PERA)  
**Area**: Parameter-Efficient Fine-Tuning / Model Compression
**Keywords**: Low-rank adaptation, polynomial expansion, high-order feature interaction, parameter-efficient fine-tuning, LoRA improvement

## TL;DR

This paper proposes PERA (Polynomial Expansion Rank Adaptation), which introduces structured polynomial expansions (square and cross terms) into the parameter space of low-rank factors, extending LoRA's linear adaptation space into a polynomial manifold. Without increasing rank or inference overhead, PERA significantly enhances the expressiveness of weight updates and consistently outperforms LoRA, DoRA, and HiRA on commonsense reasoning and NLU tasks.

## Background & Motivation

**Background**: Parameter-efficient fine-tuning (PEFT) has become the standard paradigm for adapting large language models. LoRA achieves efficient adaptation by constraining weight updates to a low-rank subspace $\Delta W = BA$, but its strictly bilinear structure captures only first-order linear dependencies between low-rank factors, limiting the model's ability to capture nonlinear and high-order parameter interactions.

**Limitations of Prior Work**: (1) LoRA's weight update $\Delta W = \sum_{i=1}^{r} \mathbf{b}_i \mathbf{a}_i^T$ is a linear combination of rank-one matrices, with expressiveness bounded by rank $r$; (2) DoRA improves via magnitude-direction decomposition but remains a linear transformation; (3) MoRA achieves high-rank adaptation via compression-transformation-decompression but introduces additional overhead; (4) HiRA enriches representations via Hadamard modulation with pretrained weights, yet its update mechanism remains linear in trainable parameters and relies on external weight coupling.

**Key Challenge**: From a function approximation perspective, there is a fundamental expressive gap between a first-order linear function $f(x) = c + c_1 x$ and a polynomial function $f(x) = c + c_1 x + c_2 x^2 + \cdots$. Viewing LoRA as a first-order linear approximation of weight updates reveals that its expressive limitations are fundamental in nature.

**Goal**: To enhance the expressiveness of low-rank adaptation by introducing high-order feature interactions without increasing rank or inference cost.

**Key Insight**: Drawing inspiration from polynomial feature expansion in classical feature engineering—applying it to the parameter space of low-rank factors rather than the input feature space.

**Core Idea**: Apply polynomial expansion and Hadamard-based polynomial expansion to low-rank matrices $B$ and $A$ respectively, generating square terms ($\mathbf{b}_i \odot \mathbf{b}_i$) and cross terms ($\mathbf{b}_i \odot \mathbf{b}_j$). Matrix concatenation (rather than addition) is used to avoid additional inference overhead, expanding the adaptation space from a linear subspace to a polynomial manifold.

## Method

### Overall Architecture

PERA follows LoRA's factorization framework, decomposing weight updates into $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$. The core improvement applies polynomial expansions to both factors before combining them: a standard second-order polynomial expansion $\text{Poly}^2(B)$ is applied to $B$, and a Hadamard-based polynomial expansion $\text{Poly}_H^2(A)$ with learnable coefficients $\mathbf{h}$ (for stability) is applied to $A$. The final update is $\Delta W = \text{Poly}^2(B) \cdot \text{Poly}_H^2(A)$.

### Key Designs

1. **Polynomial Expansion in Parameter Space**:
    - **Function**: Expands the low-rank factors from $r$ dimensions to $2r + C(r,2)$ dimensions, introducing high-order nonlinear interactions.
    - **Mechanism**: A second-order polynomial expansion is applied to $B = [\mathbf{b}_1, \ldots, \mathbf{b}_r]$, producing $\hat{B} = [B; B_{square}; B_{cross}]$, where $B_{square} = \{\mathbf{b}_i \odot \mathbf{b}_i\}$ contains $r$ square terms and $B_{cross} = \{\mathbf{b}_i \odot \mathbf{b}_j \mid i < j\}$ contains $C(r,2)$ cross terms. An analogous expansion is applied to $A$, with learnable coefficients $h_{ij}$ (initialized to zero) for training stability. The final weight update decomposes as: $\Delta W = \sum_{i} \mathbf{b}_i \mathbf{a}_i^T + \sum_{i=j} h_{ij}(\mathbf{b}_i \odot \mathbf{b}_j)(\mathbf{a}_i^T \odot \mathbf{a}_j^T) + \sum_{i<j} h_{ij}(\mathbf{b}_i \odot \mathbf{b}_j)(\mathbf{a}_i^T \odot \mathbf{a}_j^T)$.
    - **Design Motivation**: Polynomial expansion is a classical feature augmentation technique. Transferring it from the feature space to the parameter space is a natural generalization—the column vectors of low-rank factors serve as "features" of the adaptation directions, and high-order interaction terms can capture nonlinear coupling among these directions.

2. **Zero-Initialization Strategy for Hadamard Coefficients**:
    - **Function**: Ensures the training starting point is consistent with LoRA, allowing high-order terms to participate in optimization gradually.
    - **Mechanism**: The learnable coefficients $\mathbf{h} = \{h_{ij}\}$ are initialized to zero, causing PERA to reduce to standard LoRA at the beginning of training (when $\mathbf{h}=0$, square and cross terms contribute nothing). As training proceeds, the model autonomously learns which high-order interactions are beneficial for the task. LoRA is thus a special case of PERA.
    - **Design Motivation**: Zero initialization prevents high-order terms from introducing unstable gradients in early training while preserving the full upper bound of expressiveness. This progressive nonlinearity introduction strategy ensures smooth optimization.

3. **Zero Inference Overhead via Matrix Concatenation**:
    - **Function**: High-order terms are realized through column/row concatenation and can be precomputed and merged into weights at inference time.
    - **Mechanism**: The product of expanded $\hat{B} \in \mathbb{R}^{m \times (2r+C(r,2))}$ and $\hat{A} \in \mathbb{R}^{(2r+C(r,2)) \times n}$ remains an $\mathbb{R}^{m \times n}$ matrix. At inference time, $\Delta W = \hat{B}\hat{A}$ can be precomputed and merged into $W_0$, introducing no inference latency.
    - **Design Motivation**: Inference efficiency is critical for deployment. By using concatenation rather than sequential addition to realize high-order interactions, PERA preserves LoRA's zero inference overhead property.

### Loss & Training

The same next-token prediction loss as LoRA is used. Only the low-rank matrices $A$, $B$, and the Hadamard coefficients $\mathbf{h}$ are optimized during training; the pretrained weights $W_0$ remain frozen. The learning rate is $1 \times 10^{-4}$, and other hyperparameters follow the HiRA baseline. $A$ is initialized to zero and $B$ is initialized with a Gaussian distribution.

## Key Experimental Results

### Main Results

| Model | Method | Params (%) | Commonsense Avg. Acc. |
|-------|--------|-----------|----------------------|
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

| Model | Method | Params | GLUE Avg. |
|-------|--------|--------|-----------|
| RoBERTa-base | LoRA | 0.3M | 83.40 |
| RoBERTa-base | DeLoRA | 0.3M | 84.60 |
| RoBERTa-base | **PERA** | 0.3M | **85.10** |
| RoBERTa-large | LoRA | 0.8M | 87.30 |
| RoBERTa-large | **PERA** | 0.8M | **88.13** |

### Ablation Study

| Configuration | Weight Update Formula | Avg. Acc. |
|---------------|-----------------------|-----------|
| LoRA (first-order only) | Eq. 8 | 82.80 |
| LoRA + square terms only | Eq. 10 | 87.48 |
| LoRA + cross terms only | Eq. 11 | 86.83 |
| PERA (square + cross) | Eq. 9 | 87.38 |

### Key Findings

- **High-order terms yield substantial gains**: PERA outperforms LoRA by 5 percentage points on LLaMA2-7B (82.61% vs. 77.61%) and by 14.5 percentage points on Qwen2.5-7B (88.29% vs. 73.80%).
- **Square terms are the most critical high-order component**: Adding only square terms (87.48%) yields a larger improvement than adding only cross terms (86.83%), and approaches the full PERA result (87.38%), indicating that same-dimension nonlinear interactions are more important than cross-dimension interactions.
- **Strong performance at extremely low ranks**: PERA achieves 86.91% at $r=2$ and 87.01% at $r=4$, close to its best result of 87.38% at $r=16$. This is attributed to polynomial expansion raising the effective rank upper bound from $r$ to $2r + C(r,2)$.
- **Training and inference overhead close to LoRA**: Training memory is 19.12 GB vs. LoRA's 18.70 GB; inference memory is 19.70 GB vs. 19.50 GB—far more efficient than DoRA (22h07m training time vs. PERA's 13h30m).
- **10% data surpasses LoRA at full data**: PERA achieves 83.07% on 10% of commonsense170K, exceeding LoRA's 82.80% on 100% of the data, demonstrating superior data efficiency.

## Highlights & Insights

- **Elegant transfer from feature engineering to parameter engineering**: Migrating polynomial feature expansion from input feature spaces in classical ML to the parameter space of low-rank adaptation is conceptually clean yet empirically effective—a novel and insightful design choice.
- **LoRA as a special case of PERA**: When $\mathbf{h}=0$, PERA reduces to LoRA, providing not only theoretical unification but also justifying the progressive introduction strategy via zero initialization.
- **Theoretical rank upper bound improvement**: LoRA's adapted weight rank upper bound is $r_0 + r$; PERA raises this to $r_0 + 2r + C(r,2)$. For $r=16$, this increases from $r_0+16$ to $r_0+152$—nearly a 10× improvement in theoretical expressiveness.
- **Hessian interaction strength analysis**: By computing an interaction strength matrix of second-order partial derivatives, the paper intuitively demonstrates that PERA possesses stronger global feature interaction modeling capability than LoRA.

## Limitations & Future Work

- Evaluation is limited to commonsense reasoning and GLUE; arithmetic reasoning, code generation, and multimodal generation tasks are not covered.
- Only second-order polynomial expansion is adopted; the effect of higher-order expansions ($k>2$) remains unexplored.
- The contribution of cross terms is marginal (87.38% vs. 87.48% with square terms only), suggesting potential redundancy; finer-grained term selection strategies are needed.
- Comparisons with recent mixture-of-experts LoRA methods (e.g., MELoRA) or adaptive rank methods (e.g., AdaLoRA) are absent.
- Polynomial expansion may cause dimensionality explosion at large ranks ($C(r,2)$ grows quadratically with $r$), and scalability in high-rank settings warrants investigation.

## Related Work & Insights

- **vs. LoRA**: PERA is a strict generalization of LoRA, introducing high-order terms via polynomial expansion. LoRA corresponds to the special case of PERA where $\mathbf{h}=0$.
- **vs. HiRA**: HiRA introduces nonlinearity via Hadamard products with pretrained weights, relying on external weight coupling; PERA introduces high-order interactions entirely within trainable parameters, requiring no external modules.
- **vs. DoRA**: DoRA decomposes magnitude and direction but remains a linear transformation; PERA introduces genuinely nonlinear parameter interactions.
- **vs. MoRA**: MoRA increases expressiveness through high-rank transformations at the cost of additional inference overhead; PERA maintains zero inference overhead.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of applying polynomial expansion to the parameter space of low-rank factors is novel, with clear theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple models and tasks, with comprehensive ablations on rank, modules, data scale, and components.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, mathematical derivations are rigorous, and the relationship with LoRA is elegantly established.
- **Value**: ⭐⭐⭐⭐ Provides a simple yet effective enhancement of LoRA with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)

</div>

<!-- RELATED:END -->
