---
title: >-
  [Paper Note] Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence
description: >-
  [ICML 2025][LLM (Other)][Interchangeable token embeddings] Proposes a dual-part token embedding strategy (a shared learnable part + a random distinguishing part), enabling language models to generalize to larger vocabularies post-training and possess inherent robustness to alpha-equivalent transformations.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Interchangeable token embeddings"
  - "vocabulary expansion"
  - "Alpha-equivalence"
  - "formal reasoning"
  - "Transformer"
date: 2026-05-08
content_hash: 25efb6905efe243e
---

# Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence

**Conference**: ICML 2025  
**arXiv**: [2410.17161](https://arxiv.org/abs/2410.17161)  
**Code**: [necrashter.github.io/interchangeable-token-embeddings](https://necrashter.github.io/interchangeable-token-embeddings)  
**Area**: LLM/NLP  
**Keywords**: Interchangeable token embeddings, vocabulary expansion, Alpha-equivalence, formal reasoning, Transformer

## TL;DR

Proposes a dual-part token embedding strategy (a shared learnable part + a random distinguishing part), enabling language models to generalize to larger vocabularies post-training and possess inherent robustness to alpha-equivalent transformations.

## Background & Motivation

In tasks such as formal logic (e.g., Linear Temporal Logic (LTL), propositional logic), the concept of **interchangeable tokens** exists: tokens that are semantically equivalent but symbolically different (e.g., bound variables a, b, c). Traditional language models learn independent embedding vectors for each token, posing two fundamental issues:

**Inability to generalize to new tokens**: Having only seen {a, b} as atomic propositions (APs) during training, the model cannot handle formulas containing {c, d} at inference, even if c and d are semantically identical to a and b.

**Lack of alpha-equivalence awareness**: Renaming variables within a formula (alpha-conversion) should preserve semantics, but traditional models yield inconsistent predictions.

This problem is prevalent in formal reasoning domains such as LTL solving, theorem proving, and lambda calculus. As the number of variables in formulas increases, the dataset generation time grows exponentially (generating 100k samples for formulas with 10 APs and length 50 takes 2 hours and 21 minutes). Thus, the capability to generalize from training on few APs to more APs holds substantial practical value.

## Method

### Overall Architecture

Core Idea: The embeddings of interchangeable tokens are divided into two components:
- **Shared learnable part $\alpha$** ($d_\alpha$ dimensions): All interchangeable tokens share a common set of learnable parameters, conveying the semantic meaning that "these tokens belong to the same semantic category."
- **Random distinguishing part $\beta$** ($d_\beta$ dimensions): A unique vector is randomly generated for each interchangeable token, allowing the model to distinguish between different tokens.

The total embedding dimension is $d_{\text{model}} = d_\alpha + d_\beta$. For non-interchangeable tokens, the $d_\alpha$ dimensions contain their independent learnable parameters, while the $d_\beta$ dimensions are set to 0.

### Key Designs

1. **Embedding Matrix Construction**: Let the vocabulary have $n$ non-interchangeable tokens and $m$ interchangeable tokens. Let $\bm{L} \in \mathbb{R}^{n \times d_\alpha}$ represent the learnable embeddings of non-interchangeable tokens, $\bm{\alpha} \in \mathbb{R}^{1 \times d_\alpha}$ represent the learnable embedding shared by all interchangeable tokens, and $\bm{\beta}_i \in \mathbb{R}^{1 \times d_\beta}$ represent the random embedding of the $i$-th interchangeable token. The complete embedding matrix $\bm{U}$ is constructed through concatenation and normalization: the rows for non-interchangeable tokens are $[f_{bn}(\bm{L}), \bm{0}]$, and the rows for interchangeable tokens are $[f_{bn}(\bm{\alpha}), f_{bn}(\bm{\beta}_i)]$, followed by applying $f_{fn}$ normalization to the entire row. **During training, $\bm{\beta}_i$ is resampled in every forward pass** to prevent the model from adapting to specific random values; at inference time, it is sampled only once and fixed.

2. **Three Random Vector Generation Methods**: The paper proposes three strategies to generate $\bm{\beta}_i$:

    - **Normal Distribution**: Each dimension is independently sampled from $\mathcal{N}(0,1)$, with an infinite candidate pool.
    - **Neighboring Points**: Each dimension takes a value in $\{-1, 0, 1\}$, with a candidate set size of $3^{d_\beta} - 1$ (excluding the zero vector).
    - **Hypercube Vertices**: Each dimension takes a value in $\{-1, 1\}$, with a candidate set size of $2^{d_\beta}$.
   
   The latter two discrete methods efficiently generate unique vectors using integer-to-vector mapping combined with reservoir sampling, bypassing the need to physically materialize the entire candidate set. When $d_\beta > 32$, since the candidate set grows exponentially, the collision probability becomes negligible, and unique checks are bypassed.

3. **Normalization Strategies**: L2 normalization is heavily used in building the embedding matrix, serving three purposes:

    - $f_{bn}$ (block-wise normalization): Separately normalizes the $\alpha$ and $\beta$ parts to prevent the magnitude of one part from dominating the other.
    - $f_{fn}$ (final normalization): Normalizes the concatenated entire row to maintain a consistent embedding magnitude.
    - Applying $f_{fn}$ normalization to the final layer output (feature vector) of the decoder as well, making logits solely determined by cosine similarity.

4. **Three-way Weight Tying**: Binds the encoder embedding matrix, decoder embedding matrix, and the final projection matrix. Since the embedding matrix contains manually constructed random components, weight tying is **necessary** (rather than optional) in this method.

### Loss & Training

- **AdaCos Loss**: Since both embeddings and feature vectors are normalized, logits are determined solely by the cosine angle. Directly applying softmax loss corresponds to cosine loss, which is sensitive to hyperparameters. The paper adopts AdaCos for adaptive logit scaling.
- **Sequence Length Adaptation**: AdaCos was originally designed for classification tasks (without a sequence dimension). The paper merges the batch and length dimensions (ignoring padding), effectively increasing the batch size. To avoid numerical instability, the AdaCos scale value is clipped to a maximum of 100.
- **Alpha-renaming Data Augmentation (Baseline)**: For comparison, the paper also proposes a simpler method: performing random alpha-renaming on the fly during training forward passes to expose the fixed embedding model to a wider variety of tokens.

## Key Experimental Results

### Main Results

**LTL Solving (LTLRandom35 Dataset)**:

| Model | Accuracy | Exact Match | Alpha-Cov (3AP) | Alpha-Cov (5AP) |
|------|--------|---------|-----------------|-----------------|
| Baseline (Clean Data) | 98.23% | 83.23% | 96.87% | 91.80% |
| Baseline (Perturbed Data) | 34.13% | 12.12% | 64.93% | 40.91% |
| Alpha-Renaming (Perturbed) | 97.96% | 77.66% | 99.55% | 98.86% |
| **Ours (Perturbed)** | **95.94%** | **76.45%** | **97.66%** | **98.29%** |
| Llama 3.2 3B | 24.33% | 0.34% | 68.17% | 62.34% |

**Vocabulary Generalization (Train 5AP $\rightarrow$ Test 10AP)**:

| Method | LTL Accuracy | Prop. Logic Accuracy | Description |
|------|-----------|--------------|------|
| Full-Vocabulary Baseline (Train 10AP) | Best | Best | Upper Bound Reference |
| **Ours (Train 5AP)** | 90.76% | 77.70% | Slightly lower than Full-Vocabulary |
| Alpha-Renaming (Train 5AP) | Lower | Lower | Some generalization ability |
| Vanilla Baseline (Train 5AP) | Worst | Worst | Cannot handle new APs |

### Ablation Study

| Configuration | LTL Accuracy | Prop. Logic Accuracy | Description |
|------|-----------|--------------|------|
| Full Method | 90.76% | 77.70% | Baseline |
| W/o $f_{bn}$ | 29.53% | 14.12% | **Catastrophic performance drop**, imbalanced $\alpha/\beta$ magnitude |
| W/o AdaCos + $f_{fn}$ | 81.45% | ~77% | Significant impact on LTL, minor impact on propositional logic |
| Computational Overhead | +13% training time | — | Embedding preparation at inference takes only 0.0003s |

### Key Findings

1. **$f_{bn}$ block-wise normalization is critical**: Removing it causes LTL accuracy to plunge from 90.76% to 29.53%, showing that the magnitude balance between the shared and random components is the core mechanism of success.
2. **Ours is immune to perturbation**: When the dataset is perturbed by the AP appearance order bias, the traditional baseline accuracy drops sharply from 98% to 34%, whereas our method and the alpha-renaming baseline remain almost unaffected.
3. **Outperforming general LLMs**: Llama 3.2 3B (with parameters vastly exceeding our small model) only achieves 24.33% accuracy on LTL, far lower than ours (95.94%).
4. **Perfect generalization in copy task**: In the vocabulary-extendable copy task, our method achieves perfect performance under out-of-distribution scenarios (larger vocabulary + longer sequences).
5. **Parameter Efficiency**: Whereas traditional embeddings grow linearly in parameters with the number of interchangeable tokens, our method keeps the parameter count constant.

## Highlights & Insights

- **Precise problem formulation**: Clearly formalizes the overlooked concept of "interchangeable tokens", connects it to alpha-equivalence, and defines the experimental protocol for vocabulary generalization.
- **Alpha-covariance metric**: The proposed normalization metric $1 - \frac{|\mathbb{U}|-1}{|\mathbb{P}|-1}$ is highly intuitive (1 = completely unaffected by alpha-conversion) and can generalize to any domain involving alpha-equivalence.
- **Elegant design philosophy**: The shared part carries the semantic meaning "I am a variable", and the random part carries the identity "which variable I am"—perfectly matching the essence of bound variables in formal logic.
- **Resampling during training as a key insight**: Re-sampling the $\beta$ vector in every forward pass forces the model to learn correct reasoning on top of arbitrary random embeddings, instead of memorizing specific ones.
- **Clear practical value**: Generating LTL datasets grows exponentially with the number of APs. Our method can achieve performance close to 10AP training with only 5AP training, saving massive costs of data generation.

## Limitations & Future Work

1. **Inapplicable to natural language**: Natural language tokens carry semantic information (e.g., electricity_bill vs water_bill) and do not satisfy the interchangeability condition.
2. **Requires manual definition of the interchangeable token set**: This may not be feasible in certain scenarios.
3. **Requires training from scratch**: Modification of embedding architectures makes it incompatible with direct integration with pre-trained models.
4. **Slight decay in in-distribution performance**: On LTLRandom35, the accuracy is 95.94% vs baseline 98.23%, manifesting a bias-variance trade-off.
5. **Potential directions to explore**: New randomization and normalization methods, combination with pre-trained models, and supporting multiple sets of interchangeable tokens from different semantic categories.

## Related Work & Insights

- **DeepLTL (Hahn et al., 2021)**: The first to solve LTL end-to-end using Transformers; this paper addresses vocabulary generalization based on it.
- **Tree-positional encoding (Shiv & Quirk, 2019)**: Handles positional encoding for tree-structured formulas, orthogonal and complementary to this method.
- **AdaCos (Zhang et al., 2019)**: Adaptive cosine loss; this paper extends it to sequence modeling.
- **Weight Tying (Press & Wolf, 2016)**: Shared embedding and projection matrices; it transitions from an "optional optimization" to a "structural necessity" in this method.
- **Insights**: This idea can be extended to variable name handling in code generation, node label generalization in graph neural networks, and any symbolic reasoning tasks exhibiting symmetries.

## Rating

- Novelty: ⭐⭐⭐⭐ — The problem definition is novel and overlooked, and the method design is elegant and simple. However, the core conceptual mechanism (shared + random) is not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three tasks covering different difficulties, comprehensive ablation studies, comparison with LLMs, and complete computational efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem formalization, excellent figure/table designs (heatmaps intuitively demonstrate generalization capabilities), and smooth narrative logic.
- Value: ⭐⭐⭐⭐ — Clear value in formal reasoning, though applicability is restricted to scenarios containing interchangeable tokens.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](../../ACL2025/llm_nlp/token_prepending_training_free.md)
- [\[ICML 2025\] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)
- [\[NeurIPS 2025\] Reparameterized LLM Training via Orthogonal Equivalence Transformation](../../NeurIPS2025/llm_nlp/reparameterized_llm_training_via_orthogonal_equivalence_transformation.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](../../ACL2025/llm_nlp/better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](../../ACL2025/llm_nlp/contrastive_prompting_embeddings.md)

</div>

<!-- RELATED:END -->
