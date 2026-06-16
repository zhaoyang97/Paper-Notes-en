---
title: >-
  [Paper Note] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG
description: >-
  [CVPR 2026][Information Retrieval & RAG][Paper Note] Ours proposes M4-RAG, the first large-scale multilingual, multicultural, and multimodal RAG evaluation framework. Covering 42 languages and 80K+ cultural VQA instances from 189 countries, it systematically reveals that RAG is effective for small models but fails to scale positively with model size, while showing severe
tags:
  - CVPR 2026
  - Information Retrieval & RAG
date: 2026-05-08
content_hash: e332501eeec8ac0f
---
# M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG

**Conference**: CVPR 2026  
**arXiv**: [2512.05959](https://arxiv.org/abs/2512.05959)  
**Code**: [https://github.com/davidanugraha/M4-RAG](https://github.com/davidanugraha/M4-RAG)  
**Area**: Information Retrieval  
**Keywords**: Retrieval-Augmented Generation, Multilingual, Multicultural, Visual Question Answering, Multimodal Retrieval

## TL;DR

Ours proposes M4-RAG, the first large-scale multilingual, multicultural, and multimodal RAG evaluation framework. Covering 42 languages and 80K+ cultural VQA instances from 189 countries, it systematically reveals that RAG is effective for small models but fails to scale positively with model size, while showing severe performance degradation in cross-lingual retrieval.

## Background & Motivation

1. **Background**: RAG technology is widely used in LLMs/VLMs to enhance generation quality via external knowledge. While multilingual RAG and multimodal RAG have progressed independently, their intersection—multilingual multimodal RAG—remains largely unexplored.
2. **Limitations of Prior Work**: Existing RAG benchmarks either cover only the text modality or support only English, lacking a large-scale evaluation framework that simultaneously addresses multilingual and multimodal requirements. Cultural knowledge is inherently long-tail and region-specific, making it difficult for models to encode reliably.
3. **Key Challenge**: In the real world, knowledge access is inherently both multilingual and multimodal, yet existing RAG evaluations fail to reflect this complexity.
4. **Goal**: (1) Construct a multimodal RAG benchmark covering 42 languages and 56 dialects; (2) Systematically study the impact of different retrieval strategies on VLMs of varying scales; (3) Quantify RAG performance degradation under cross-lingual conditions.
5. **Key Insight**: Cultural knowledge is selected as the test scenario because it is long-tail and region-specific, making it ideal for detecting RAG effectiveness.
6. **Core Idea**: Build the first multilingual multimodal RAG benchmark to reveal the inverse relationship between RAG utility and model scale.

## Method

### Overall Architecture

M4-RAG does not train new models but establishes a "controlled experimental platform" to disassemble the utility of RAG in multilingual multimodal cultural VQA. For a given cultural VQA instance (one image + one question), the framework requires the same VLM to respond under four supply conditions: (a) No RAG—VLM relies solely on parametric cultural knowledge; (b) Oracle Context—human-verified relevant standard knowledge is provided as a performance upper bound for "perfect retrieval"; (c) Text RAG—images are converted to captions, and text encoders retrieve documents; (d) Multimodal RAG—multimodal retrievers (e.g., mmE5, B3) use both image and text signals. Retrieval consistently takes the top-5 results from a million-scale multilingual Wikipedia corpus. Comparing scores across these configurations quantifies the real gain of RAG, the gap between text vs. multimodal retrieval, and the distance to the Oracle bound.

### Key Designs

**1. Multilingual Multicultural VQA Benchmark: Cultural Long-tail Knowledge as a Touchstone**

To test RAG effectiveness, a set of questions likely absent from model parameters is required. M4-RAG merges two complementary datasets to cover 42 languages, 56 dialects, and 80K+ VQA pairs: CVQA provides domain diversity (30 countries, 31 languages, 10 cultural categories), while WorldCuisines provides cross-lingual parallelism (30 languages, 60K global food VQA with aligned multilingual versions). The former ensures cultural complexity, while the latter allows for clean "language-switching" control experiments.

**2. Controllable Retrieval Environment: Reproducible yet Realistic**

To ensure RAG gains are attributable, M4-RAG freezes a large-scale multilingual knowledge corpus from a April 2025 Wikipedia snapshot. Multiple query methods (question-only, answer-only, culture-enhanced) are used to maximize recall. Crucially, **English and target languages retrieve top-25 results independently** rather than translating English results, ensuring non-English passages retain authentic terminology rather than machine-translated artifacts. After cleaning, CVQA yields 307,000 articles and WorldCuisines yields 223,000, forming a fixed retrieval pool.

**3. Cross-lingual Evaluation Design: Decoupling Instruction Understanding from Evidence Utilization**

Performance drops in non-English RAG could stem from either a failure to understand target-language instructions or an inability to integrate target-language evidence. M4-RAG translates instruction prompts and Oracle contexts **separately** into target languages (Gemini-2.5-Flash translation + manual validation). This allows independent observation of "multilingual prompt" vs. "multilingual context" variables. This disassembly reveals that switching prompt language causes only a 1–2% drop, while switching evidence language causes drops up to -32.4% in low-resource settings, suggesting that instruction understanding is rarely the bottleneck compared to cross-lingual evidence integration.

### Loss & Training

As a pure evaluation framework, no model training is involved. Scoring utilizes macro-average accuracy for multiple-choice answers. Retrieval relevance quality is assessed via VLM-as-a-judge, where a VLM scores each result based on reasoning criteria to analyze the impact of high vs. low-quality retrieval.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best No-RAG | Best RAG | Best Oracle |
|--------|------|-----------|---------|------------|
| CVQA | Accuracy | Gemma3-27B: 74.34% | mmE5 Multimodal RAG significant gain | Gemma3-27B Max |
| WorldCuisines | Accuracy | Gemma3-27B: 66.20% | Qwen2.5-VL-72B (Oracle) | Significantly superior to baseline |

**Comparison of RAG Strategies**:

| Retrieval Method | Effect |
|---------|------|
| Text RAG (Caption-Query) | Worst, often lower than No-RAG baseline |
| Multimodal RAG (mmE5) | Best, consistently superior to Text RAG |
| Multimodal RAG (B3) | Second best, smaller gain than mmE5 |
| Oracle-Query RAG | Moderate, limited by text-only queries |

### Ablation Study

| Configuration | Key Finding | Description |
|------|---------|------|
| Small Model + RAG vs. Large Model No-RAG | Small Model + RAG can match or exceed Large Models | External knowledge is more effective than parameter scaling |
| High Retrieval Quality (>4 points) | Correctness persistence 95-100%, Correction rate 80-90% | High-quality retrieval reliably enhances performance |
| Low Retrieval Quality (<2 points) | Correctness persistence drops to 40-60% | Irrelevant context actively misleads the model |
| Large Model Correction Rate | Lower than small models | Large models have strong parametric knowledge inertia |

### Key Findings

- **Inverse Relationship between RAG and Model Scale**: RAG is consistently effective for small VLMs, but returns diminish as model size increases. Parametric knowledge in large models competes with rather than complements retrieved evidence. Reasoning-focused VLMs (e.g., Qwen3-VL) are more robust under RAG settings than non-reasoning counterparts.
- **Severe Cross-lingual Degradation**: Switching prompts from English to target languages results in only a 1-2% drop, but switching Oracle context to target languages causes performance to plummet, with drops up to -32.4% for low-resource languages (Qwen2.5-VL-32B on CVQA). Even models trained on multilingual data, like Pangea, are severely affected.
- **Text RAG is Counterproductive**: Naive text RAG (converting images to captions before retrieval) introduces noise and often performs worse than the No-RAG baseline. Multimodal RAG is more reliable but not a universal solution.

## Highlights & Insights

- **Asymmetry in Correction vs. Persistence**: It is easy to maintain a correct answer with high-quality retrieval (95-100%), but difficult to correct a wrong one (80-90% with large inter-model variance). This reveals a fundamental bottleneck in VLM evidence integration: persuading a model "you are wrong" is much harder than confirming "you are right."
- **Model Scale Increases Inertial Priors**: Large models are less likely to be misled by low-quality retrieval (high persistence) but are also less likely to be corrected by valid retrieval (low correction), showing a "double-edged sword" effect. This points to diminishing returns on RAG investment as models grow.
- **Code-Switching in Small Models**: Smaller models tend to code-switch to English answers under non-English prompts, resulting in smaller perceived multilingual performance drops. Large models attempt to answer strictly in the target language, leading to more frequent failures.

## Limitations & Future Work

- Evaluation is limited to cultural VQA, which may not represent RAG performance in other knowledge-intensive tasks.
- Only open-source VLMs were evaluated; latest closed-source models (e.g., GPT-4o) were not included.
- The knowledge base from Wikipedia contains coverage bias; some cultures/languages may be underrepresented.
- **Future Directions**: (1) Model-aware retrieval strategies—dynamically adjusting retrieval depth based on model capability; (2) Joint retriever-VLM post-training; (3) Test-time adaptation—enabling models to autonomously decide whether to retrieve and how to utilize results.

## Related Work & Insights

- **vs. MRAG-Bench**: MRAG-Bench contains only 1,353 English samples; M4-RAG covers 42 languages and 80K samples, vastly exceeding it in scale and diversity.
- **vs. MIRACL**: MIRACL is a text-only multilingual retrieval benchmark lacking multimodal evaluation. M4-RAG covers both text and image modalities.
- **vs. ICQ (multimodal composed retrieval)**: While ICQ focuses on retrieval performance itself, M4-RAG focuses on the end-to-end impact of RAG on generation quality, closer to real-world applications.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale multilingual multimodal RAG evaluation framework, filling a significant gap, though focused on evaluation rather than method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive systematic evaluation of 11 models, 6 retrieval configurations, and 42 languages with deep analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, precise findings, and highly informative visualizations.
- Value: ⭐⭐⭐⭐⭐ The revealed inverse relationship between RAG and model scale and the cross-lingual integration bottleneck provide important guidance for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](../../ACL2026/information_retrieval/coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] 生物医学 RAG 中检索何时无效：大规模实证研究](../../ACL2026/information_retrieval/when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/information_retrieval/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)

</div>

<!-- RELATED:END -->
