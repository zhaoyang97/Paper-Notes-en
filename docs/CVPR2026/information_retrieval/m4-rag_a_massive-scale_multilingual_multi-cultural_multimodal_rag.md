---
title: >-
  [Paper Note] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG
description: >-
  [CVPR 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper proposes M4-RAG, the first large-scale multilingual, multicultural, multimodal RAG evaluation framework…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Multilingual"
  - "Multicultural"
  - "Visual Question Answering"
  - "Multimodal Retrieval"
date: 2026-05-08
content_hash: c0787257ab86d675
---

# M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG

**Conference**: CVPR 2026
**arXiv**: [2512.05959](https://arxiv.org/abs/2512.05959)
**Code**: [https://github.com/davidanugraha/M4-RAG](https://github.com/davidanugraha/M4-RAG)
**Area**: Information Retrieval
**Keywords**: Retrieval-Augmented Generation, Multilingual, Multicultural, Visual Question Answering, Multimodal Retrieval

## TL;DR

This paper proposes M4-RAG, the first large-scale multilingual, multicultural, multimodal RAG evaluation framework, covering 42 languages and 189 countries with 80K+ cultural VQA instances. It systematically reveals two key findings: RAG is effective for smaller models but does not scale positively with model size, and cross-lingual retrieval suffers from severe performance degradation.

## Background & Motivation

1. **Background**: RAG has been widely adopted in LLMs and VLMs to enhance generation quality through external knowledge retrieval. Progress has been made in multilingual RAG and multimodal RAG separately, but their intersection—multilingual multimodal RAG—remains largely unexplored.
2. **Limitations of Prior Work**: Existing RAG evaluation benchmarks either cover only the text modality or support only English, lacking a large-scale framework that simultaneously addresses multilingual and multimodal settings. Cultural knowledge is inherently long-tailed and region-specific, making it difficult for even large models to encode reliably.
3. **Key Challenge**: In the real world, knowledge access is inherently both multilingual and multimodal, yet existing RAG evaluations fail to reflect this complexity.
4. **Goal**: (1) Construct a multimodal RAG evaluation benchmark covering 42 languages and 56 dialects; (2) Systematically study the effect of different retrieval strategies on VLMs of varying scales; (3) Quantify RAG performance degradation under cross-lingual conditions.
5. **Key Insight**: Cultural knowledge is selected as the test scenario—being naturally long-tailed and region-specific, it is well-suited for assessing RAG effectiveness.
6. **Core Idea**: Construct the first multilingual multimodal RAG benchmark and reveal an inverse relationship between RAG utility and model scale.

## Method

### Overall Architecture

The M4-RAG evaluation framework comprises four configurations: (a) No-RAG baseline: VLM directly processes the question and image; (b) No-RAG with Oracle context: an upper bound providing perfectly relevant knowledge; (c) Text RAG: retrieves text documents via a text encoder; (d) Multimodal RAG: jointly leverages textual and visual signals for retrieval. The retrieval system employs a top-5 strategy over a million-scale multilingual document corpus.

### Key Designs

1. **Multilingual Multicultural VQA Benchmark Construction**:

    - Function: Provides 80K+ culturally diverse image–question–answer pairs spanning 42 languages and 56 dialects.
    - Mechanism: Integrates two complementary datasets—CVQA (30 countries, 31 languages, 10 cultural categories) and WorldCuisines (30 languages, 60K global cuisine VQA instances)—to achieve comprehensive linguistic and cultural coverage. WorldCuisines provides cross-lingual parallelism; CVQA provides domain diversity.
    - Design Motivation: Cultural knowledge is long-tailed and region-specific, making reliable encoding by large models difficult and thus serving as a natural testbed for RAG.

2. **Controlled Retrieval Environment**:

    - Function: Provides reproducible retrieval conditions that balance authenticity and controllability.
    - Mechanism: Constructs a large-scale multilingual knowledge corpus from a April 2025 Wikipedia snapshot, using multiple query types (question-only, answer-only, culturally augmented queries) to maximize coverage. Top-25 articles are retrieved independently in both English and the target language, then cleaned and deduplicated, yielding 307K articles for CVQA and 223K for WorldCuisines.
    - Design Motivation: Ensures non-English passages reflect culturally accurate terminology rather than direct translations, improving retrieval authenticity.

3. **Cross-Lingual Evaluation Design**:

    - Function: Quantifies the impact of language switching on VLM performance.
    - Mechanism: Instruction prompts and Oracle contexts are each translated into target languages using Gemini-2.5-Flash, with human annotation for quality verification. The effects of "multilingual prompts" and "multilingual context" on performance are measured independently.
    - Design Motivation: Isolates the model's ability to perform cultural reasoning under different language conditions, distinguishing instruction comprehension from evidence integration.

### Loss & Training

This paper presents an evaluation framework and does not involve model training. Evaluation uses macro-averaged accuracy over multiple-choice answers. Annotation quality is assessed via a VLM-as-a-judge approach, with retrieval relevance scored against a reasoning-based rubric.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best No-RAG | Best RAG | Best Oracle |
|---------|--------|-------------|----------|-------------|
| CVQA | Accuracy | Gemma3-27B: 74.34% | mmE5 multimodal RAG shows the largest gain | Gemma3-27B achieves the highest score |
| WorldCuisines | Accuracy | Gemma3-27B: 66.20% | Qwen2.5-VL-72B (Oracle) | Significantly outperforms baseline |

**Retrieval Strategy Comparison**:

| Retrieval Method | Effect |
|-----------------|--------|
| Text RAG (Caption-Query) | Worst; even underperforms the no-RAG baseline |
| Multimodal RAG (mmE5) | Best; consistently outperforms text RAG |
| Multimodal RAG (B3) | Second best; smaller gains than mmE5 |
| Oracle-Query RAG | Moderate; limited by text-based querying |

### Ablation Study

| Configuration | Key Finding | Explanation |
|--------------|-------------|-------------|
| Small model + RAG vs. large model without RAG | Small model + RAG can match or exceed large model without RAG | External knowledge is more effective than parameter scaling |
| High retrieval quality (>4) | Correct-answer retention 95–100%, correction rate 80–90% | High-quality retrieval reliably enhances performance |
| Low retrieval quality (<2) | Correct-answer retention drops to 40–60% | Irrelevant context actively misleads the model |
| Large model correction rate | Lower than small models | Strong parametric knowledge inertia resists adopting external evidence |

### Key Findings

- **Inverse Relationship Between RAG and Model Scale**: RAG consistently benefits smaller VLMs, but returns diminish as model scale increases. In larger models, parametric knowledge competes with rather than complements retrieved evidence. Reasoning-oriented VLMs (e.g., Qwen3-VL) exhibit greater robustness under RAG settings compared to non-reasoning models.
- **Severe Cross-Lingual Degradation**: Switching prompts from English to target languages incurs only a 1–2% drop, but switching Oracle context to target languages causes a sharp performance collapse, with low-resource languages dropping by up to −32.4% (Qwen2.5-VL-32B on CVQA). Pangea, despite being specifically trained on multilingual data, is still severely affected.
- **Text RAG Worse Than No RAG**: Naive text RAG—converting images to captions before retrieval—introduces noise and can underperform the no-RAG baseline. Multimodal RAG is more reliable but not universally effective.

## Highlights & Insights

- **Asymmetry Between Correction Rate and Retention Rate**: Under high-quality retrieval, retaining correct answers is easy (95–100%), but correcting wrong answers is harder (80–90% with large inter-model variance). This reveals a fundamental bottleneck in how current VLMs integrate external evidence—convincing a model it is wrong is far harder than confirming it is right.
- **Larger Models Exhibit Stronger Prior Inertia**: Large models are less susceptible to misleading by low-quality retrieval (high retention) but also less receptive to correction from high-quality retrieval (low correction rate), exhibiting a double-edged effect. This is an important finding regarding diminishing returns on RAG investment.
- **Code-Switching in Small Models**: Small models tend to code-switch to English when given non-English prompts, resulting in paradoxically smaller multilingual performance drops. Large models attempt to respond entirely in the target language and fail more severely.

## Limitations & Future Work

- Evaluation is limited to cultural VQA scenarios and may not fully generalize to RAG performance on other knowledge-intensive tasks.
- Only open-source VLMs are evaluated; the latest closed-source models (e.g., GPT-4o's multimodal RAG capability) are excluded.
- The knowledge base is drawn from Wikipedia, introducing coverage bias—Wikipedia content for certain cultures and languages may be incomplete.
- **Future Directions**: (1) Model-aware retrieval strategies that dynamically adjust retrieval depth and modality based on model capability; (2) Joint retriever–VLM post-training; (3) Test-time adaptation enabling models to autonomously determine whether and how to leverage retrieved results.

## Related Work & Insights

- **vs. MRAG-Bench**: MRAG-Bench contains only 1,353 English samples; M4-RAG covers 42 languages with 80K+ samples, far surpassing it in scale and multilingual coverage.
- **vs. MIRACL**: MIRACL is a text-only multilingual retrieval benchmark lacking multimodal evaluation. M4-RAG covers both text and image modalities.
- **vs. ICQ (multimodal composed retrieval)**: ICQ focuses on retrieval effectiveness itself, whereas M4-RAG evaluates end-to-end RAG impact on generation quality, more closely reflecting real-world application scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The first large-scale multilingual multimodal RAG evaluation framework, filling an important gap; however, the core contribution is evaluation rather than methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic evaluation across 11 models, 6 retrieval configurations, and 42 languages is comprehensive, with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, precise articulation of findings, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ The revealed inverse relationship between RAG and model scale, and the identified bottleneck in cross-lingual evidence integration, offer important guidance for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](../../ACL2026/information_retrieval/all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](../../ACL2026/information_retrieval/is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)

</div>

<!-- RELATED:END -->
