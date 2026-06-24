---
title: >-
  [Paper Note] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations
description: >-
  [ACL 2025][Information Retrieval & RAG][text embedding] This paper proposes LDIR, a method that selects anchor texts using farthest point sampling (FPS) and computes the semantic similarity between target texts and each anchor text. This constructs low-dimensional ($\le 500$ dimensions), dense, and interpretable text embeddings, yielding performance close to black-box models and significantly outperforming existing interpretable embedding approaches.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "text embedding"
  - "interpretable representation"
  - "relative representation"
  - "farthest point sampling"
  - "low-dimensional"
date: 2026-05-08
content_hash: 4208cd707e454b4a
---

# LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations

**Conference**: ACL 2025  
**arXiv**: [2505.10354](https://arxiv.org/abs/2505.10354)  
**Code**: [szu-tera/LDIR](https://github.com/szu-tera/LDIR)  
**Area**: Information Retrieval  
**Keywords**: text embedding, interpretable representation, relative representation, farthest point sampling, low-dimensional

## TL;DR

This paper proposes LDIR, a method that selects anchor texts using farthest point sampling (FPS) and computes the semantic similarity between target texts and each anchor text. This constructs low-dimensional ($\le 500$ dimensions), dense, and interpretable text embeddings, yielding performance close to black-box models and significantly outperforming existing interpretable embedding approaches.

## Background & Motivation

Text embedding is a foundational technology in NLP, but current methods face a practical trade-off between performance and interpretability:

**Black-box dense embeddings** (e.g., SimCSE, LLM2Vec): High performance, but the physical meaning of each individual dimension cannot be traced, typically spanning $768\text{--}4096$ dimensions.

**Bag-of-Words (BoW)**: High interpretability but poor semantic performance, spanning approximately $30\text{K}$ dimensions.

**QA Embeddings** (e.g., QAEmb-MBQA, CQG-MBQA): Generate $0/1$ binary embeddings by using LLMs to answer yes/no questions. They are interpretable but require $\sim 10\text{K}$ dimensions and rely heavily on GPT-4 to generate questions.

The **Key Challenge** is: **binary $0/1$ representations are limited in expressiveness, requiring extremely high dimensions to cover the semantic space, while dense floating-point representations have strong expressiveness but lack explicit semantic interpretability for each dimension.**

The **Key Insight** of LDIR is: if each dimension represents the "semantic similarity to a known, concrete anchor text", the floating-point values themselves carry clear, interpretable semantic meanings. Since continuous values are far more expressive than $0/1$ binary values, the required dimensionality can be dramatically reduced.

## Method

### Overall Architecture

The workflow of LDIR consists of: Anchor Text Selection $\rightarrow$ Similarity Computation $\rightarrow$ Embedding Generation.

Given a text $t$, the interpretable embedding of LDIR is defined as:

$$e_{\text{dense}}^{\text{interp}}(t) = [\text{Rel}(a_1, t), \text{Rel}(a_2, t), \ldots, \text{Rel}(a_n, t)]$$

where $a_1, \ldots, a_n$ are the designated anchor texts, and $\text{Rel}$ is the similarity function.

### Key Designs

1. **From Binary $0/1$ Embeddings to Dense Embeddings**

    - While QA embeddings use binary yes/no answers as $0/1$ values, LDIR leverages continuous similarity scores.
    - Similarity is computed via cosine similarity: $\text{Rel}(a_j, t) = \frac{\text{Enc}(a_j) \cdot \text{Enc}(t)}{\|\text{Enc}(a_j)\| \cdot \|\text{Enc}(t)\|}$
    - The encoder $\text{Enc}$ can be any pre-trained model (e.g., SimCSE, ModernBERT, AngIE) without requiring additional fine-tuning.
    - **Design Motivation**: Continuous floating-point values carry substantially more information than binary values, thus requiring significantly fewer dimensions to represent the semantic space.

2. **Anchor Text Selection via Farthest Point Sampling (FPS)**

    - First, use the base encoder to generate embeddings for all candidate texts in a reference corpus.
    - Then, apply the FPS algorithm to iteratively select the $n$ most dispersed texts in the semantic space as anchor points.
    - FPS guarantees that the selected anchor texts have the maximum mutual distance, effectively covering diverse regions of the semantic space.
    - This process does not require GPT-4 to generate questions, nor does it require manual filtering.
    - **Design Motivation**: If anchor texts are too close to each other, the values across different embedding dimensions will converge, leading to a loss of discriminative power.

3. **Interpretability Guarantee**

    - Each dimension corresponds to a specific anchor text, where the continuous value represents the semantic similarity between the input text and that anchor.
    - Users can inspect the anchor texts corresponding to dimensions with the highest values to interpret "what this text is primarily about."
    - Although continuous similarity is less direct than binary yes/no answers, it provides a traceable semantic reference.

### Loss & Training

LDIR **does not require any training or fine-tuning**:
- Anchor texts are automatically and deterministically selected from the corpus via FPS.
- Embedding computation relies solely on the cosine similarity computed by existing frozen encoders.
- The entire process is deterministic and involves no parameter learning.

## Key Experimental Results

### Main Results (STS Semantic Textual Similarity, Spearman's Correlation Coefficient)

| Model | Dim | Type | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|------|------|------|-------|-------|-------|-------|-------|-------|--------|-----|
| SimCSE_sup | 768 | Black-box | 75.30 | 84.67 | 80.19 | 85.40 | 80.82 | 84.25 | 68.38 | 79.86 |
| AngIE | 1024 | Black-box | 79.09 | 89.62 | 85.02 | 89.51 | 86.61 | 89.06 | 82.62 | 85.93 |
| QAEmb-MBQA | 10654 | Interpretable | 59.40 | 63.19 | 57.68 | 69.29 | 63.18 | 71.33 | 72.33 | 65.20 |
| CQG-MBQA | 9614 | Interpretable | 69.21 | 80.19 | 73.91 | 80.66 | 78.30 | 82.69 | 78.21 | 77.60 |
| **LDIR (AngIE, 500)** | **500** | **Interpretable** | **78.85** | **84.35** | **80.93** | **84.79** | **83.61** | **86.31** | **80.85** | **82.82** |

LDIR achieves an average score of $82.82$ with only $500$ dimensions, outperforming all interpretable baselines (e.g., CQG-MBQA with $77.60$) and closely approaching the performance of the black-box SimCSE_sup ($79.86$).

### Ablation Study (Comparison of Anchor Selection Methods, AngIE Encoder, 500 Dim)

| Sampling Method | STS Avg | Retrieval Avg | Clustering Avg |
|----------|---------|---------------|----------------|
| Uniform Sampling | ~78 | ~47 | ~28 |
| K-Means | ~80 | ~48 | ~30 |
| **FPS** | **82.82** | **50.31** | **31.39** |

FPS consistently outperforms uniform sampling and K-Means cluster centers across all tasks, validating the core effectiveness of farthest point sampling.

### Key Findings

1. **Extremely high dimensionality efficiency**: Using only $500$ dimensions, LDIR outperforms $9614\text{--}10654$-dimensional binary $0/1$ interpretable embeddings.
2. **Encoder choice is critical**: Stronger base encoders lead to better LDIR performance (AngIE > ModernBERT > SBERT).
3. **Competitive at 200 dimensions**: LDIR with $200$ dimensions performs on par with CQG-MBQA using $9614$ dimensions.
4. **Gap in retrieval tasks**: On information retrieval tasks, there remains a noticeable gap in performance compared to black-box models ($41\text{--}50$ vs $56\text{--}58$ nDCG@10), as low-dimensional representations inevitably compress and lose fine-grained discriminative details.
5. **Zero external overhead**: It eliminates the need for expensive GPT-4 API calls or additional module training; it requires only a single pass of offline FPS sampling.

## Highlights & Insights

- **Clever application of Relative Representations**: Drawing inspiration from the cross-model invariance discovered by Moschella et al. (2023), LDIR successfully translates this concept into the domain of interpretable text embeddings.
- **Ultra-simplistic pipeline**: The entire method contains no learnable parameters, requires no fine-tuning, and relies purely on sampling and cosine similarity, offering outstanding reproducibility.
- **Elegant dimension-interpretability trade-off**: While binary $0/1$ embeddings require tens of thousands of dimensions to achieve reasonable fidelity, LDIR compresses this representation down to the hundreds using continuous values.
- **"Spectrum of Interpretability" perspective**: Instead of pursuing absolute, rigid interpretability (such as yes/no answers), LDIR provides key "relative traceability," which stands as a highly pragmatic design choice.

## Limitations & Future Work

1. The interpretability is slightly weaker than QA-based embeddings: continuous similarity scores are inherently less intuitive than direct binary yes/no answers, requiring users to look up the actual anchor texts.
2. Performance on retrieval tasks is significantly lower than black-box models, since low-dimensional relative representations compress and lose fine-grained discriminative info.
3. Anchor text selection is heavily dependent on the corpus distribution, which might necessitate resampling when switching to a different domain.
4. The potential for learnable anchor text selection methods (such as end-to-end optimization of anchor positions) remains unexplored.
5. There is a lack of validation regarding its utility in large-scale, real-world industrial scenarios (e.g., search engines or recommendation systems).

## Related Work & Insights

- QAEmb-MBQA and CQG-MBQA pioneered the paradigm of "defining embedding dimensions via questions." LDIR extends this to continuous values based on reference corpora.
- The relative representation framework proposed by Moschella et al. (2023) was originally intended for cross-model alignment; LDIR demonstrates its novel utility for enhancing representations' interpretability.
- **Insight**: Can the core idea of LDIR be applied to multimodal embeddings? For instance, using image anchors to define interpretable dimensions for visual representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying relative representations to interpretable text embeddings is a highly creative approach, and selecting anchors via FPS is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across three major task categories (STS, Retrieval, and Clustering), with extensive comparisons against strong baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear comparison tables, intuitive explanations of the methodology, and highly distinct comparisons against the state of the art.
- **Value**: ⭐⭐⭐⭐ — Offers practical value for real-world scenarios requiring interpretable text representations (e.g., trustworthy AI, model inspection/auditing).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder](prism_political_bias_embeddings.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)
- [\[ACL 2025\] InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](inspiredebate_multidim_evaluation_debating.md)
- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)

</div>

<!-- RELATED:END -->
