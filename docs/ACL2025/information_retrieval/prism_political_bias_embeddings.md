---
title: >-
  [Paper Note] PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder
description: >-
  [ACL 2025][Information Retrieval & RAG][Political Bias Embedding] This work proposes the PRISM framework, which models political bias embedding as an interpretable task for the first time. It automatically extracts controversial topics and left/right bias indicators from a weakly labeled news corpus as embedding dimensions. Then, a political-aware cross-encoder is employed to score articles along each topic dimension, generating sparse and semantically transparent political b…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Political Bias Embedding"
  - "Interpretable Embedding"
  - "Cross-Encoder"
  - "Bias Classification"
  - "Diverse Retrieval"
date: 2026-05-08
content_hash: d0499d2dc0249539
---

# PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder

**Conference**: ACL 2025  
**arXiv**: [2505.24646](https://arxiv.org/abs/2505.24646)  
**Code**: [https://github.com/dukesun99/ACL-PRISM](https://github.com/dukesun99/ACL-PRISM)  
**Area**: Information Retrieval  
**Keywords**: Political Bias Embedding, Interpretable Embedding, Cross-Encoder, Bias Classification, Diverse Retrieval

## TL;DR

This work proposes the PRISM framework, which models political bias embedding as an interpretable task for the first time. It automatically extracts controversial topics and left/right bias indicators from a weakly labeled news corpus as embedding dimensions. Then, a political-aware cross-encoder is employed to score articles along each topic dimension, generating sparse and semantically transparent political bias embeddings. The proposed method achieves $86.1\%$ accuracy on NewsSpectrum (outperforming POLITICS by $34.8\%$) while supporting diverse retrieval.

## Background & Motivation

**Background**: Semantic text embedding is a fundamental task in NLP, which encodes text into vectors and measures semantic similarity through vector distance. Current mainstream embedding models (e.g., SimCSE, AnglE) perform exceptionally well in general retrieval and clustering tasks. However, they only encode "what the content says" and ignore "what the stance is"—two articles reporting on the same event may have highly similar semantics but completely opposite political stances.

**Limitations of Prior Work**: Existing methods for political bias analysis fall into three categories, each with distinct drawbacks. Domain-specific models like POLITICS capture bias signals but produce black-box embeddings, failing to explain what information each dimension encodes. Instruction-following models (e.g., InstructOR, InBedder) guide embedding directions via prompts but show limited effectiveness on highly subjective tasks like political bias. Interpretable embedding methods (e.g., CQG-MBQA) use yes/no questions as dimensions but cannot handle the complexity and multi-dimensional nature of political views.

**Key Challenge**: Political bias analysis faces three major difficulties: (1) Dimensional complexity: political issues are not simple left/right binaries but involve multiple dimensions such as economy, society, and foreign affairs, which are difficult to enumerate manually; (2) Scarcity of annotations: fine-grained political stance annotation is highly expensive and subjective, making large-scale, high-quality labeling practically unfeasible; (3) Black-box nature: existing bias models cannot explain why a certain article is classified as right-leaning, lacking transparency.

**Goal**: Design a political bias embedding framework that requires no fine-grained manual annotations, automatically discovers political topics, and assigns clear semantic meanings to each embedding dimension.

**Key Insight**: The authors observe that news corpora naturally contain weak annotation information—overall media bias ratings (such as AllSides ratings) can serve as distant supervision signals. By automatically locating controversial topics through clustering and variance analysis, and using LLMs to summarize the left/right stance descriptions for each topic as bias indicators, semantically clear embedding dimensions can be constructed.

**Core Idea**: Use "controversial political topics + left/right bias indicators" as interpretable embedding dimensions, and generate sparse political bias embeddings of articles through a cross-encoder that calculates the alignment score between the article and each bias indicator.

## Method

### Overall Architecture

PRISM consists of two phases: The first phase automatically extracts controversial topics and their left/right bias indicators from a weakly labeled news corpus (Controversial Topic Bias Indicator Mining). The second phase uses the extracted topics as embedding dimensions and trains a political-aware cross-encoder to produce structured bias vectors for each article (Cross-Encoder Political Bias Embedding). The input is the original news article, and the output is a sparse, interpretable bias vector where each non-zero dimension corresponds to a specific political topic, positive values represent right-leaning stances, and negative values represent left-leaning stances.

### Key Designs

1. **Controversial Topic Mining**:

    - **Function**: Automatically discovers controversial political topics from a large-scale, weakly labeled news corpus and generates left/right bias indicator descriptions for each topic.
    - **Mechanism**: First, a pre-trained semantic encoder is used to encode all articles into vectors, and then K-means clustering is applied to group articles reporting on similar topics. For each cluster, the Bias Dispersion ($\text{BD}(\bm{R}) = \frac{1}{n}\sum_{i=1}^{n}(r_i - \bar{r})^2$) is calculated. Clusters with a variance higher than a threshold $\tau$ and containing more than $p$ articles are identified as controversial topics. Finally, an LLM is used to generate a neutral topic summary and left/right bias indicators based on sample articles within the cluster (e.g., for the topic "healthcare reform", the left indicator is "support expanding public health funding", and the right indicator is "prioritize cutting government spending").
    - **Design Motivation**: Utilizing media-level bias ratings (AllSides) as weak supervision avoids manual annotation, and utilizing variance measurement automatically filters for genuinely controversial topics rather than redundant neutral ones.

2. **Important Topic Retrieval**:

    - **Function**: Selects the top-$m$ most relevant and bias-discriminative topics for each article from all available topics, ensuring that the embeddings are sparse and focused.
    - **Mechanism**: For each topic, an importance score is calculated as $\text{Score}(i) = \lambda(\bm{x} \cdot \bm{t}_i) + (1-\lambda)|\bm{x} \cdot \bm{r}_i - \bm{x} \cdot \bm{l}_i|$, where the first term measures the semantic relevance between the article and the topic, and the second term measures the disagreement between the left and right bias scores of the article on that topic. The top-$m$ topics with the highest scores are selected as the active dimensions for the article.
    - **Design Motivation**: Not all political topics are relevant to every article, and activating all dimensions would introduce noise. By using a scoring function that balances relevance and bias distinctiveness, only truly informative dimensions are preserved, improving embedding quality and interpretability.

3. **Political-Aware Cross-Encoder Embedding**:

    - **Function**: Learns the alignment relationship between articles and bias indicators, producing the final interpretable political bias vector for each article.
    - **Mechanism**: Train a cross-encoder $f_\theta(\bm{a}, \bm{b}) \in (0,1)$ that takes an article and a bias indicator pair as input and outputs an alignment score. It is trained using weak labels: article-indicator pairs of matching stances in the same cluster are labeled 1, mismatching stances are labeled 0, and random topics across clusters are treated as negative sample labels 0. During inference, for each active topic $i$, the final embedding value is calculated as $e_i = s_i^r - s_i^l$ (the right-bias score minus the left-bias score), where a positive value indicates right-leaning, a negative value indicates left-leaning, and zero indicates neutral or irrelevant.
    - **Design Motivation**: Compared to bi-encoders, cross-encoders capture fine-grained interactions between articles and bias indicators. This difference design gives the embedding dimensions intuitive semantic meanings, allowing users to understand the stance distribution of an article directly through its non-zero dimensions.

### Loss & Training

The cross-encoder is trained using MSE loss, with supervision signals derived from weak labels: matching stances within the same cluster are positive samples (label 1), while mismatching stances within the same cluster and random topics across clusters serve as negative samples (label 0). This weak supervision strategy completely bypasses the need for fine-grained manual annotation, relying solely on media-level bias ratings for training.

## Key Experimental Results

### Main Results

Evaluate political bias classification performance on two large-scale news datasets, training an SVM classifier with the embedding vectors:

| Model | Type | NewsSpectrum Acc↑ | NewsSpectrum F1-Ma↑ | BigNews Acc↑ | BigNews F1-Ma↑ |
|------|------|------------------|--------------------|--------------|--------------------|
| AnglE | General | 48.4 | 48.2 | 69.0 | 69.0 |
| InstructOR | Instruction | 47.9 | 47.7 | 63.7 | 63.7 |
| InBedder | Instruction | 50.2 | 49.8 | 64.6 | 64.6 |
| CQG-MBQA | Interpretable | 45.1 | 44.9 | 61.0 | 61.0 |
| POLITICS | Domain-Specific | 51.3 | 51.1 | **85.7** | **85.7** |
| **PRISM** | **Interpretable + Political** | **86.1** | **86.2** | 73.5 | 74.0 |

### Parameter Sensitivity Analysis

| Parameter | Experimental Setup | Findings |
|------|---------|------|
| Number of clusters $k$ | 10 → 5000 | Performance improves steadily as $k$ increases, peaks around $k=1000$, and then slightly declines. |
| Number of top-m topics | $m=1$ → $m=9+$ | F1-macro rises steadily from $m=1$ to $m=9$, then stabilizes. |
| Topic granularity | Too few vs. Too many | Too few limits topic diversity, while too many introduces noise and redundancy. |
| Sparse vs. Full activation | top-m vs. All topics | Top-$m$ retrieval significantly outperforms using all topics, validating the necessity of selective activation. |

### Key Findings

- **General embeddings fail completely**: General models like AnglE and InstructOR achieve only $\sim 48\text{-}50\%$ accuracy on NewsSpectrum, which is close to random guessing. This demonstrates that political bias signals are almost orthogonal to general semantics and require dedicated modeling.
- **PRISM generalizes far better than POLITICS**: On NewsSpectrum (not seen during training), PRISM outperforms POLITICS by a large margin (86.1% vs 51.3%, +34.8%). POLITICS performs better on BigNews because the test set overlaps with its training data.
- **Interpretability does not come at the cost of performance**: While maintaining semantic transparency for each dimension, PRISM surpasses black-box models in both classification and retrieval, breaking the stereotype of "interpretable = weaker performance."
- **Win-win in diverse retrieval**: In DiversiNews retrieval experiments, PRISM maintains higher content relevance under the same level of political diversity and provides greater ideological diversity under the same level of relevance.

## Highlights & Insights

- **Ingenious design of weakly-supervised topic mining**: Using Bias Dispersion (variance) to automatically select controversial topics is a simple and effective strategy. High variance naturally implies that multiple stances co-exist on the same topic, needing no manual definitions. This idea can be transferred to any scenario that requires discovering opposite-opinion topics from weakly-labeled data.
- **Intuitive design of difference embedding**: The simple difference operation $e_i = s_i^r - s_i^l$ encodes the article's stance on each topic into a real value, where positive/negative directly correspond to right/left, and zero means irrelevant. This design makes the embedding intuitive even for non-technical users.
- **Selective activation ensuring sparsity**: Through the top-$m$ topic retrieval mechanism, only dimensions that are highly relevant to the article and possess bias discriminativeness are activated, and the rest remain zero. This improves both computational efficiency and interpretability, allowing users to focus only on a few non-zero dimensions.

## Limitations & Future Work

- **Efficiency bottleneck**: The two-stage design (topic mining + pairwise cross-encoder scoring) is more computationally expensive than standard embedding models. Specifically, the cross-encoder needs to calculate alignment scores for each active topic separately, making it difficult to deploy in ultra-large-scale real-time scenarios.
- **Topic independence assumption**: Each topic dimension is treated as an independent axis, ignoring potential dependencies between topics (e.g., "healthcare reform" and "government spending" are highly correlated), which may lead to redundancy in the embedding space.
- **English/US-centric focus**: The framework was only evaluated on English news and the US political spectrum. Political dimensions vary dramatically across different cultural contexts (e.g., European multi-party systems), so the cross-cultural transferability of the framework needs to be verified.
- **Entanglement of topics and stances**: Current clustering may blend topic content and ideological framing. Future work could explicitly decouple them using multi-view representation learning or factorized embeddings.

## Related Work & Insights

- **vs POLITICS**: POLITICS learns political bias features by fine-tuning RoBERTa on BigNews, but its embeddings are black-box and heavily overfitted to the training data. PRISM addresses both interpretability and generalization by explicitly constructing topic dimensions.
- **vs CQG-MBQA**: CQG-MBQA uses yes/no questions to construct interpretable embedding dimensions, but pre-defined questions cannot capture the dynamic complexity of political bias. PRISM replaces manually designed questions with data-driven topic mining, making it more suitable for open-domain scenarios.
- **vs Media Framing Analysis**: Works like SLAP4SLIP focus on concept discovery and framing analysis, modeling ideological polarization but without producing embeddings. PRISM translates framing analysis concepts (identifying controversial topics + characterizing stance differences) into an actionable embedding generation framework.

## Rating

- Novelty: ⭐⭐⭐⭐ Models political bias embedding as an interpretable task for the first time. The combined design of topic mining and cross-encoder embedding is original, though individual components (clustering, cross-encoders, weak supervision) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against 5 classes of baselines on two large-scale datasets, including classification, retrieval, case studies, and parameter sensitivity analysis, but lacks ablation experiments dissecting the contribution of each individual component.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, rigorous motivational derivations, intuitive and powerful case studies, with highly logical narrative.
- Value: ⭐⭐⭐⭐ Highly practical for news bias analysis and diverse recommendation systems, but computational efficiency and cross-cultural transfer restrict large-scale deployment.

## Highlights & Insights

- For the first time, defines the "political bias embedding" task and provides a complete technical framework where each dimension has a clear, interpretable semantic meaning.
- Requires absolutely no fine-grained annotations—capable of learning fine-grained topic bias using only media-level weak labels (AllSides ratings).
- Improves performance on NewsSpectrum from POLITICS' 51.3% to 86.1%, demonstrating a massive advantage.
- Embeddings meet two key interpretability properties: selective activation (only relevant topics are non-zero) and explicit bias representation (positive/negative values directly correspond to right/left leanings).
- Topic mining is fully automated, allowing new topics to be continuously discovered as the news corpus updates.

## Limitations & Future Work

- The method relies on existing media bias ratings (AllSides), which themselves may contain bias.
- Currently targeted only at the US left-right political spectrum; multi-party or non-Western political systems would require redesign.
- The number of clusters $k$ and the bias dispersion threshold $\tau$ in the topic mining phase must be set manually.
- The inference cost of the cross-encoder is higher than that of bi-encoders, potentially limiting real-time encoding of large-scale news streams.
- Evaluated only on English news corpora; cross-lingual political bias embeddings remain to be explored.

## Related Work & Insights

- **Political Bias Analysis**: Ranges from binary classification (Iyyer et al., 2014) to multi-dimensional approaches (Kim & Johnson, 2022; Liu et al., 2023), but lacks interpretable embedding solutions.
- **Domain-Specific Embeddings**: Specialized models have been developed for domains like biomedicine (Lee et al., 2020), finance, and science. POLITICS (Liu et al., 2022) fine-tunes RoBERTa for political texts.
- **Interpretable Embeddings**: Projects like CQG-MBQA (Sun et al., 2025) use Q&A pairs as dimensions, but are poorly suited to the complex subjectivity of political bias.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneeringly defines the political bias embedding task with a novel and systematic framework design.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to news bias detection and diverse recommendation systems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across two large-scale datasets on both classification and retrieval tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Highly rigorous logic in problem motivation, technical solutions, and interpretability discussions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Cross-Lingual Relevance Transfer for Document Retrieval](cross-lingual_relevance_transfer_for_document_retrieval.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)

</div>

<!-- RELATED:END -->
