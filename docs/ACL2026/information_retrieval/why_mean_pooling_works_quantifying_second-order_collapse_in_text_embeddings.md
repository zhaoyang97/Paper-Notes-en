---
title: >-
  [Paper Note] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings
description: >-
  [ACL 2026][Information Retrieval & RAG][Mean Pooling] This paper argues that mean pooling theoretically loses the second-order structure of token embeddings and proposes the SOCM metric to quantify this second-order collapse; experiments demonstrate that modern contrastively fine-tuned text encoders produce more concentrated token embeddings, making them less prone to col
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - Mean Pooling
  - SOCM
date: 2026-05-08
content_hash: 197bbf86e23b0ec5
---
# Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings

**Conference**: ACL2026  
**arXiv**: [2604.27398](https://arxiv.org/abs/2604.27398)  
**Code**: Not mentioned  
**Area**: Information Retrieval / Text Embeddings  
**Keywords**: Mean Pooling, Text Embeddings, Second-Order Statistics, SOCM, Contrastive Learning

## TL;DR
This paper argues that mean pooling theoretically loses the second-order structure of token embeddings and proposes the SOCM metric to quantify this second-order collapse; experiments demonstrate that modern contrastively fine-tuned text encoders produce more concentrated token embeddings, making them less prone to collapse than base models, and lower SOCM correlates with higher MTEB performance.

## Background & Motivation
**Background**: Modern text embedding models typically utilize Transformer encoders to output token embeddings, which are then aggregated via mean pooling to obtain sentence, paragraph, or document vectors. This representation is widely used in retrieval, RAG, semantic search, and automatic evaluation due to its simplicity, low cost, and suitability for approximate nearest neighbor search.

**Limitations of Prior Work**: Mean pooling only preserves the first-order statistics of the token embedding distribution, specifically the mean. Two texts might have entirely different token distribution shapes, but if their means are similar, the final text vectors may become nearly identical. This implies that second-order information, such as spatial structure, variance, and covariance, is flattened.

**Key Challenge**: From an information preservation perspective, mean pooling is coarse; however, in practice, modern text encoders like GTE, E5, and MPNet demonstrate strong performance. The question thus arises: are real-world models actually harmed by this second-order information loss? If not, what occurs internally within the models to make mean pooling effective?

**Goal**: The paper aims to provide a measurable and verifiable explanation for the effectiveness of mean pooling. This involves defining a metric to measure the risk of collapse (where means are similar but second-order structures differ), measuring this risk using real models and texts, and analyzing how fine-tuned encoders avoid such collapse.

**Key Insight**: Each text's token embeddings are treated as an empirical distribution, with the mean representing first-order statistics and the covariance representing second-order statistics. Whether mean pooling collapses depends on whether the first-order distance between two texts is small while the second-order distance is large.

**Core Idea**: The $SOCM=(1-d_\mu)d_\Sigma$ metric is used to quantify second-order collapse. It is proven that modern contrastively fine-tuned encoders reduce the risk of second-order collapse by causing token embeddings within the same text to become more concentrated.

## Method

### Overall Architecture
Ours does not propose a new encoder but builds an analytical framework to answer "what mean pooling loses and why it remains effective." Given two texts $t_1, t_2$, the encoder outputs two sets of token embeddings $X_1, X_2$. Mean pooling considers only $\mu(X_i)$ as the text vector, while covariance $\Sigma(X_i)$ describes the spatial structure of the tokens. The framework first measures mean differences using a first-order distance $d_\mu$ and covariance differences using a second-order distance $d_\Sigma$, then combines them into the SOCM metric to quantify the collapse risk of "similar means but different second-order structures." SOCM is subsequently measured across a large number of real text pairs, and the reason why fine-tuned models suffer less from collapse is explained via the attention mechanism of Transformer layers. Experiments involve 499,500 pairs sampled from 1,000 Wikipedia texts, comparing base models like BERT, MiniLM, MPNet, and nomic-bert with their contrastive fine-tuned versions. Validations using MS MARCO passages and hard negatives, as well as correlation analyses between SOCM and MTEB English v2 scores, are also provided.

### Key Designs

**1. SOCM Second-Order Collapse Metric: Measuring collapse risk through the interaction of "proximity in first-order and divergence in second-order"**

Since mean pooling only preserves first-order means, two texts with vastly different token distribution shapes will be pulled together after pooling if their means are close—this is precisely second-order collapse. The challenge lies in the fact that mean distance alone cannot determine if a critical structure was lost, nor can covariance distance alone reveal if such a structure was obscured by the mean. Thus, the metric must constrain both ends. Ours defines a normalized first-order distance $d_\mu = \lVert \mu(X_1) - \mu(X_2) \rVert_2^2 / 4$ and a scaled Bures-Wasserstein covariance distance $d_\Sigma$, both falling within $[0, 1]$. These are combined into $SOCM(d_\mu, d_\Sigma) = (1 - d_\mu)d_\Sigma$: the value reaches 1 only when means are nearly identical ($d_\mu \to 0$) and covariance difference is maximal ($d_\Sigma \to 1$), while returning to 0 if means are sufficiently far apart or covariances are identical, aligning with the intuitive definition of collapse.

**2. Paired Measurement of Base vs. Contrastively Fine-tuned Models: Explicitly visualizing token geometric changes before and after fine-tuning**

If mean pooling itself is coarse but fine-tuned encoders like GTE/E5 perform strongly, the difference likely lies not in the pooling operator but in the token geometry learned by the encoder. To verify this, ours calculates the average SOCM for all text pairs across various backbones and their text embedding versions: BERT vs. Unsup-SimCSE/E5/GTE, MiniLM vs. all-MiniLM/E5small/GTEsmall, MPNet vs. all-mpnet-base-v2, and nomic-bert vs. nomic-embed-text-v1.5. This paired design allows for a direct reading of whether and to what extent fine-tuning reduces collapse, while also exposing outliers like all-MiniLM.

**3. Mechanism explanation via token concentration: Deriving why collapse diminishes with fine-tuning from single-layer attention**

To explain the lower SOCM in fine-tuned models, ours analyzes how token embeddings cluster toward the mean within a text using a simplified single-head self-attention layer. If the attention projection branch satisfies the contraction condition $\lambda < 1$, the relative influence $r$ of the input spread in the residual output is small, and the per-token transformation does not significantly amplify the spread, then the normalized spread $S(X) / \lVert \mu(X) \rVert_2^2$ will decrease. It is further proven that when this normalized spread is smaller than $\epsilon$, $SOCM = O(\epsilon)$. Intuitively, the more tokens of the same text cluster around their mean, the smaller the covariance becomes, which naturally limits the second-order information lost by mean pooling—explaining why a seemingly coarse averaging operation remains sufficient for modern encoders.

## Loss & Training
Ours does not train new models, but processing existing models is a prerequisite for comparable conclusions. No task prefixes such as "query" or "passage" are added to ensure comparability across models and datasets. Token embeddings for each text are normalized such that the pooled mean satisfies the unit norm assumption $\lVert \mu(X) \rVert_2 = 1$ required by the SOCM definition. Theoretically, the Transformer layer is abstracted into three parts: self-attention, residual connection, and per-token transformation. Experimentally, $\lambda$, $r$, $C$, and $S(X) / \lVert \mu(X) \rVert_2^2$ are calculated layer-by-layer for real BERT and GTEbase to verify that fine-tuned encoders are more likely to form token concentration in deeper layers.

## Key Experimental Results

### Main Results
The average SOCM on Wikipedia text pairs shows that most contrastively fine-tuned models have lower scores than their base models, with the BERT series showing particularly significant changes.

| Backbone / Model | Avg. SOCM ↓ | Relative Change from Base | Conclusion |
|-----------------|-------------|---------------------------|------------|
| BERT | 0.396 | - | Base model has high collapse risk |
| Unsup-SimCSE-mean | 0.193 | -0.203 | Contrastive fine-tuning significantly reduces SOCM |
| E5base | 0.029 | -0.367 | Large reduction |
| GTEbase | 0.018 | -0.378 | Lowest in the BERT series |
| MiniLM | 0.242 | - | Moderate collapse risk |
| all-MiniLM-L12-v2 | 0.313 | +0.071 | Exceptional case of degradation |
| E5small | 0.099 | -0.143 | Significant reduction |
| GTEsmall | 0.055 | -0.187 | Significant reduction |
| MPNet | 0.117 | - | Base is already low |
| all-mpnet-base-v2 | 0.100 | -0.017 | Slight reduction |
| nomic-bert-2048 | 0.139 | - | Base is relatively low |
| nomic-embed-text-v1.5 | 0.122 | -0.017 | Slight reduction |

### Ablation Study
The authors replicated experiments on MS MARCO and compared the correlation between SOCM and downstream MTEB performance. The trends on MS MARCO passages and hard negatives were largely consistent: the E5/GTE series scored significantly lower than the BERT/MiniLM backbones.

| Analysis Item | Metric / Data | Key Result | Description |
|---------------|---------------|------------|-------------|
| Wikipedia SOCM vs MTEB | Spearman $\rho$ | -0.678, $p=0.015$ | Lower SOCM correlates with higher MTEB |
| Token concentration vs MTEB | Spearman $\rho$ | -0.622 | Also correlated, but weaker than SOCM |
| MS MARCO passages | BERT $\to$ GTEbase | 0.491 $\to$ 0.025 | Conclusions hold across datasets |
| MS MARCO passages | MiniLM $\to$ GTEsmall | 0.289 $\to$ 0.055 | Reduction also occurs in small models |
| MS MARCO hard negatives | BERT $\to$ GTEbase | 0.480 $\to$ 0.017 | Robust on query-negative pairs |
| MS MARCO hard negatives | MiniLM $\to$ GTEsmall | 0.340 $\to$ 0.048 | Significant drop in collapse risk |

### Key Findings
- Contrastive fine-tuning often does not make mean pooling more complex, but rather makes the geometric structure of token embeddings more suitable for mean pooling: tokens within the same text become more concentrated, and means of different texts become more separable.
- SOCM explains MTEB performance better than token concentration alone because it simultaneously considers mean separation between texts; a model can make tokens highly concentrated, but if means of different texts remain close, retrieval performance still suffers.
- all-MiniLM-L12-v2 exhibited a higher SOCM than MiniLM on Wikipedia, indicating that "fine-tuning necessarily reduces collapse" is not an absolute law; specific training objectives and model architectures remain important.
- In visualization cases, BERT produced an SOCM of 0.618 for two semantically unrelated texts due to similar means but different token distribution spreads; GTEbase produced an SOCM of 0.024 for the same pair as the means had been separated, matching the metric's intuition.

## Highlights & Insights
- Ours transforms a commonplace engineering choice into a quantifiable scientific question. The effectiveness of mean pooling, previously attributed to empirical results or efficiency, can now be explained through first- and second-order statistical collapse.
- The form of SOCM is simple yet captures the key: collapse is not "second-order difference" per se, but "second-order difference masked by similar means." This interaction makes the metric more aligned with the risks of mean pooling than looking at covariance distance in isolation.
- The explanation of token concentration is enlightening. Since contrastive learning supervises the pooled embedding, models may naturally pull tokens of the same text toward the mean to ensure the mean more stably carries discriminative information.
- This paper also suggests that retrieval model training can explicitly constrain pre-pooling geometry. If SOCM can be made differentiable or sampled approximately, it could serve as a training regularizer to reduce information collapse in mean pooling.

## Limitations & Future Work
- SOCM only considers second-order statistics, approximating the token embedding list as a Gaussian distribution; third-order and higher-order structures may still be lost by mean pooling, which was not analyzed.
- The metric relies on assumptions such as the unit norm of the mean and trace bounds of the covariance. While authors prove these can be satisfied under normalization and certain LayerNorm conditions, broader architectures or unnormalized scenarios might require modifications.
- The theoretical section explains how token concentration reduces SOCM, but it does not fully explain why contrastive fine-tuning inevitably induces concentration, which remains an open question.
- Experiments focus on English data and mean pooling. Additional validation is needed for other languages, long documents, multi-vector retrieval, LLM context compression, or aggregation methods like SIF and max pooling.

## Related Work & Insights
- **vs ColBERT / BERTScore / OT token-list methods**: These methods preserve token-level high-order structures at a higher computational cost; ours suggests that modern mean pooling encoders may reduce the necessity of preserving such structures via internal geometry.
- **vs GaussCSE / distributional embeddings**: GaussCSE directly predicts first- and second-order statistics; ours explains why using only the first-order mean suffices in certain models because second-order differences are actively suppressed by the encoder.
- **vs Text Embedding Geometry Research**: Prior work focuses on anisotropy, dimensional collapse, or embedding space structure; ours extends geometric analysis to the first- and second-order statistics of token distributions before pooling.
- **Inspiration**: When training retrieval models, "intra-text token concentration" and "inter-text mean separation" can be used simultaneously as diagnostic metrics to determine if a model is effective only by chance through its post-pooling vectors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The topic is niche but the approach is elegant, re-interpreting mean pooling efficacy through second-order statistical collapse with a concise metric.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple backbones/fine-tuned models and Wikipedia/MS MARCO/MTEB; however, language and pooling type coverage remains limited.
- Writing Quality: ⭐⭐⭐⭐☆ Theory, metrics, experiments, and mechanism explanations are tightly connected; the mathematical section has a slight barrier for readers without a geometric background.
- Value: ⭐⭐⭐⭐☆ Insightful for retrieval model diagnostics and pooling regularization design in RAG, representing an explanatory yet practical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](../../ACL2025/information_retrieval/sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](../../ACL2025/information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](../../ACL2025/information_retrieval/redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)

</div>

<!-- RELATED:END -->
