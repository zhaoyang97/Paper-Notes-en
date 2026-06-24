---
title: >-
  [Paper Note] Investigating Language Preference of Multilingual RAG Systems
description: >-
  [ACL 2025][Information Retrieval & RAG][multilingual RAG] This work systematically investigates the language preference issue in both the retrieval and generation stages of multilingual RAG (mRAG) systems. It proposes the MLRS metric to quantify the degree of retriever preference for specific languages, revealing that retrievers favor high-resource and query languages, while generators prefer the query language and Latin-script languages. Finally…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "multilingual RAG"
  - "language preference"
  - "MLRS"
  - "DKM-RAG"
  - "cross-lingual retrieval"
date: 2026-05-08
content_hash: ec38113f08757b56
---

# Investigating Language Preference of Multilingual RAG Systems

**Conference**: ACL 2025  
**arXiv**: [2502.11175](https://arxiv.org/abs/2502.11175)  
**Code**: [GitHub](https://github.com/jeonghyunpark2002/LanguagePreference.git)  
**Area**: NLP / Multilingual Retrieval-Augmented Generation  
**Keywords**: multilingual RAG, language preference, MLRS, DKM-RAG, cross-lingual retrieval

## TL;DR

This work systematically investigates the language preference issue in both the retrieval and generation stages of multilingual RAG (mRAG) systems. It proposes the MLRS metric to quantify the degree of retriever preference for specific languages, revealing that retrievers favor high-resource and query languages, while generators prefer the query language and Latin-script languages. Finally, it designs the DKM-RAG framework, which effectively mitigates the preference issue by fusing translated passages with the model's internal knowledge.

## Background & Motivation

Multilingual retrieval-augmented generation (mRAG) systems enhance LLM responses by integrating multilingual external knowledge. However, they suffer from severe **language preference** issues, which result in suboptimal retrieved documents and inaccurate or inconsistent generated answers.

**Two Core Problems**:

1. **Retriever Language Preference**: Retrievers tend to prioritize documents in high-resource languages (e.g., English) or the same language as the query, even when lower-resource languages contain more relevant information. For instance, when a Korean query searches a multilingual knowledge base, English documents might rank higher due to language dominance, burying the truly relevant Korean documents. This causes the generator to produce incorrect answers or refuse to answer due to a lack of relevant input.
2. **Generator Language Preference**: Even when relevant multilingual documents are retrieved, the generator may favor passages in the query language or Latin-script languages, neglecting crucial evidence in other languages, leading to inconsistent cross-lingual answers.

**Limitations of Prior Work**:

- Mainly focus on limited language combinations, failing to reflect the true ranking dynamics of documents across different languages.
- Methods like Language-Preference-Based Re-ranking only focus on a single mRAG stage.
- Lack standardized metrics for quantifying retriever language preference.

This work explores three research questions: Which languages do retrievers prefer? Which languages do generators prefer and how does this affect performance? How can language preference be mitigated?

## Method

### Overall Architecture

This work is a systematic empirical study combined with a methodological proposal. The structure is as follows:
1. Proposes the MLRS metric to quantify retriever language preference (RQ1).
2. Evaluates generator language preference through multilingual answer consistency (RQ2).
3. Analyzes the correlation between language preference and mRAG performance.
4. Proposes the DKM-RAG framework to mitigate language preference (RQ3).

### Key Designs

1. **MLRS (MultiLingualRankShift) Metric**:

    - Function: Quantifies the degree of retriever preference for a specific language, answering the question: "How much would the ranking improve if language disparities were eliminated?"
    - Mechanism: A three-stage calculation: (i) Retrieve documents for a query $q$ from a multilingual knowledge base to obtain initial rank $r_d^{init}$; (ii) Translate non-query language documents into the query language; (iii) Re-rank the translated documents to measure the rank shift $\Delta r_d = \max(r_d^{init} - r_d^{re\text{-}rank}, 0)$.
    - Normalization: $MLRS_q = \frac{\Delta r_q}{\Delta r_q^{max}} \times 100$, where $\Delta r_q^{max}$ is the maximum possible rank shift (all target documents shifting to rank 1), averaged over all queries.
    - Key Insight: If the rankings improve substantially after translation to a target language, it indicates that language differences (rather than content differences) caused the lower original rankings—demonstrating a preference against that target language.
    - Design Motivation: Existing methods rely only on statistical equivalence tests or simple fairness metrics, failing to accurately capture the language preferences in ranking dynamics.

2. **Generator Language Preference Evaluation**:

    - Function: Evaluates LLM answer consistency across different languages to reveal the generator's language preferences.
    - Mechanism: For the same query and retrieved document set, the generator is prompted to answer in 8 different languages (en, ko, zh, fr, ja, it, pt, es). Embedding similarities between all language pairs of answers are calculated using LaBSE, resulting in an 8x8 similarity matrix. The preference for a specific language is computed as the average similarity of its answers to the answers in other languages.
    - Design Motivation: If the generator is unaffected by language preferences, the answers in different languages for the same input should be semantically consistent.

3. **DKM-RAG (Dual Knowledge Multilingual RAG)**:

    - Function: Mitigates language preferences in mRAG and improves cross-lingual answer quality.
    - Four-step process:
        - Step 1 (Retrieval & Re-ranking): Retrieve documents from the multilingual knowledge base and re-rank them using BGE-m3.
        - Step 2 (Translation): Translate all retrieved passages into the query language, yielding $P_{translated}$.
        - Step 3 (Refinement): Use a Rewriter LLM to rewrite the translated passages combining internal knowledge—removing redundancy, filtering irrelevant information, and supplementing reliable content, yielding $P_{refined}$.
        - Step 4 (Fusion): Concatenate $P_{translated}$ and $P_{refined}$ as the final input for the generator.
    - Design Motivation: Simply translating passages can fix language mismatch but fails to filter irrelevant content from high-resource languages; refinement utilizing the model's internal knowledge can further enhance information quality. The dual knowledge sources (external translation + internal refinement) are complementary.

### Loss & Training

This work does not involve model training—MLRS is an evaluation metric and DKM-RAG is an inference-time framework. All components (retriever, translation model, rewriter, generator) directly adopt off-the-shelf models:

- **Retriever**: BGE-m3 (primary), p-mMiniLM, p-mMpNet
- **Translation Model**: NLLB-200-distilled-600M (primary), GPT-4o-mini (for translation quality experiments)
- **Generator**: aya-expanse-8B, Qwen2.5-7B-Instruct, Phi-4 14B, Llama-3.1-8B-Instruct
- **Semantic Similarity**: LaBSE (multilingual sentence embeddings)
- **Evaluation Metrics**: character 3-gram recall
- **Dataset**: MKQA (10k samples, 25 languages, based on English Wikipedia), taking the 2.7k overlap subset with KILT NQ.

## Key Experimental Results

### Main Results

**Retriever Language Preference (MLRS Scores, BGE-m3)**:

| Query Lang | Same-Lang Match | → en | → ko | → zh | → fr |
|---|---|---|---|---|---|
| en | 56.03 | — | 33.02 (-23.0) | 33.10 (-22.9) | 36.61 (-19.4) |
| ko | 43.49 | 41.15 (-2.3) | — | 34.42 (-9.1) | 36.42 (-7.1) |
| zh | 45.26 | 44.98 (-0.3) | 34.52 (-10.7) | — | 36.34 (-8.9) |
| fr | 43.18 | 47.23 (+4.1) | 33.29 (-9.9) | 33.58 (-9.6) | — |

**DKM-RAG Performance Comparison (character 3-gram recall)**:

| Query Lang | Model | All | Best Single-Lang | DKM-RAG |
|---|---|---|---|---|
| en | aya-expanse-8B | 80.09 | 79.34 (en) | **82.60** |
| zh | aya-expanse-8B | 32.55 | 38.31 (zh) | **44.57** |
| ko | aya-expanse-8B | 40.60 | 49.66 (ko) | **55.01** |
| en | Phi-4 | 79.69 | 78.89 (en) | **82.59** |
| zh | Phi-4 | 16.75 | 36.76 (zh) | **44.56** |
| ko | Phi-4 | 26.80 | 49.25 (ko) | **54.82** |

### Ablation Study

DKM-RAG Ablation (aya-expanse-8B):

| Configuration | Lq=en | Lq=zh | Lq=ko |
|---|---|---|---|
| DKM-RAG (Full) | **82.60** | **44.57** | **55.01** |
| w/o $P_{refined}$ | 79.34 (-3.26) | 38.31 (-6.26) | 49.66 (-5.35) |
| w/o $P_{translated}$ | 81.10 (-1.50) | 39.44 (-5.13) | 46.15 (-8.86) |

Both components are critical for performance: removing the refined passages $P_{refined}$ leads to a substantial performance drop for non-English queries (-5 to -6 points), and removing the translated passages $P_{translated}$ has an even larger impact (-5 to -9 points). This demonstrates that the reciprocal complementarity of external translated knowledge and internal refined knowledge is key to the success of DKM-RAG.

### Key Findings

1. **Retrievers strongly prefer same-language and high-resource languages**: MLRS is highest during same-language matching (56.03 for en-en). English, as a document language, almost always receives the highest preference—even surpassing same-language matches in non-English queries (e.g., for a French query, English document preference is 47.23 > French same-language 43.18).
2. **Languages of the same family exhibit smaller preference gaps**: Cross-lingual preference drops within the Romance language family (French, Italian, Portuguese, Spanish) are only 1-6 points, while the drop reaches 7-23 points for East Asian languages (Chinese, Japanese, Korean).
3. **Document language resources are more critical than query language resources**: The resource level of the document language has a significant impact on MLRS (High-Resource > Medium-Resource > Low-Resource), whereas the resource level of the query language has limited impact.
4. **Generators prefer Latin-script languages and the query language**: Answer consistency among Latin-script languages (en, fr, it, pt, es) is much higher than that of non-Latin languages (ko, zh, ja); query language answer consistency is slightly higher, but the improvement is limited.
5. **High preference $\neq$ high performance**: A retriever's preference for English does not imply that answering all queries with English documents yields the best performance—for non-English queries, answering with documents in the query language is more effective.
6. **DKM-RAG comprehensively and significantly improves performance**: It achieves optimal performance across all query languages and generators, with particularly remarkable gains for non-English queries (Chinese +6 to 28 points, Korean +5 to 28 points).

## Highlights & Insights

- **Ingenious MLRS Metric Design**: Through the counterfactual experimental concept of "post-translation re-ranking," it cleverly decouples language factors from content factors, providing a standardized tool to quantify language preferences.
- **Comprehensive Coverage of Both Ends of the mRAG Pipeline**: It analyzes both retriever and generator preferences, as well as their interactions, presenting a more systematic approach than prior works that only focus on a single stage.
- **Counter-intuitive Finding of "High Preference $\neq$ High Performance"**: It challenges the simplistic assumption that dominating mRAG with high-resource languages is sufficient to boost performance.
- **Simple yet Effective DKM-RAG Method**: Requiring no training of new models, it significantly mitigates language preferences merely through an inference-time workflow of translation, refinement, and concatenation, making it highly plug-and-play.

## Limitations & Future Work

- **Dependence on Translation Quality**: Both the MLRS metric and the DKM-RAG framework are highly dependent on the accuracy of the translation model; translation errors can distort the original semantics.
- **Increased Computational Overhead**: MLRS requires translation and re-ranking, while DKM-RAG requires translation, rewriting, and concatenation. Latency and cost could become bottlenecks in large-scale real-time systems.
- **Limited Language Coverage**: Experiments were conducted only with 8 languages (primarily high- and medium-resource), lacking validation on low-resource languages (e.g., Arabic, Hindi, Swahili).
- **Dataset Limitations**: MKQA is based on English Wikipedia, biasing the knowledge base towards English, which might not fully reflect preference patterns in genuine multilingual knowledge-sharing scenarios.
- **Black-box Refinement in DKM-RAG**: How the Rewriter LLM decides which information to filter out or supplement lacks controllability and interpretability.
- **Unexplored Trainable Fusion Mechanisms**: Currently, translated and refined passages are simply concatenated; dynamic weighting or attention-based integration might be superior.

## Related Work & Insights

- **vs Bergen** (mRAG Baseline): Bergen explores component choices for building effective mRAG pipelines. Building on top of it, this work delves deeply into the language preference issue and proposes mitigation strategies.
- **vs Language-Preference-Based Re-ranking**: It only performs re-ranking in the retrieval stage without considering generator preferences; in contrast, this work comprehensively covers both stages.
- **vs Cross-lingual IR Research** (Yang et al., Telemala & Suleman): These primarily focus on fairness metrics or limited language pairs, whereas MLRS provides a more accurate measurement of ranking dynamics.
- **Insights**: (1) mRAG system designs should explicitly optimize for language preferences rather than ignoring them; (2) Although translation is simple, its effect in mRAG is outstanding, acting as a highly cost-effective cross-lingual bridge; (3) Fusing internal knowledge (LLM parametric knowledge) with external knowledge (retrieved passages) is an important direction for improving RAG quality.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the MLRS metric is novel, presenting the first systematic study of language preferences across the entire mRAG pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 languages, 3 re-ranking encoders, and 4 generators across both retrieval and generation stages with a complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear RQ-driven structure with progressive analysis, offering a logically complete progression from identifying phenomena to proposing solutions.
- Value: ⭐⭐⭐⭐ Offers practical guidelines for designing and deploying multilingual RAG systems; both MLRS and DKM-RAG can be directly applied.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ICML 2026\] Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG](../../ICML2026/information_retrieval/linguistic_nepotism_trading-off_quality_for_language_preference_in_multilingual_.md)
- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)

</div>

<!-- RELATED:END -->
