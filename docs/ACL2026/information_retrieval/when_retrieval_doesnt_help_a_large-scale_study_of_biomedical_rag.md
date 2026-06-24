---
title: >-
  [Paper Note] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study
description: >-
  [ACL 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This large-scale empirical study spanning 5 models, 10 datasets, 4 retrieval methods, and 4 retrieval corpora finds that biomedical RAG only provides marginal and unstable improvements of 1-2 points. The true bottleneck is the model's capacity to effectively utilize retrieved evidence rather than the quality of retrieval itself.
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Biomedical QA"
  - "RAG"
  - "Large Language Models"
  - "Evidence Utilization"
date: 2026-05-08
content_hash: c0f61b4e6edb0de8
---

# When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study

**Conference**: ACL 2026  
**arXiv**: [2606.04127](https://arxiv.org/abs/2606.04127)  
**Code**: https://github.com/erfan-nourbakhsh/BioMedicalRAG  
**Area**: Information Retrieval / LLM / Biomedical NLP  
**Keywords**: Retrieval-Augmented Generation, Biomedical QA, RAG, Large Language Models, Evidence Utilization

## TL;DR
This large-scale empirical study spanning 5 models, 10 datasets, 4 retrieval methods, and 4 retrieval corpora finds that biomedical RAG only provides marginal and unstable improvements of 1-2 points. The true bottleneck is the model's capacity to effectively utilize retrieved evidence rather than the quality of retrieval itself.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become a mainstream solution in the medical QA field, with works such as MedRAG reporting accuracy gains of up to 18% on large models. Medical QA is a high-risk scenario where any factual error can lead to severe consequences, making accuracy improvements particularly crucial.

**Limitations of Prior Work**: Existing research primarily evaluates large-scale proprietary models (GPT-4, GPT-3.5, Mixtral-8×7B, Llama2-70B) or 70B-class models, mostly using zero-shot multiple-choice evaluations. There is a lack of systematic evaluation for more practical and resource-constrained 7-8B open-source models. Furthermore, most evaluations focus on professional-grade biomedical questions, ignoring the vast amount of real-world consumer-level health queries.

**Key Challenge**: Is the gain from RAG truly as significant as shown in large model studies? This question remains unanswered for small and medium-sized models. If retrieval does not help, what is the true limiting factor—retrieval algorithms, retrieval corpora, or the model itself?

**Goal**: To re-examine the effectiveness of biomedical RAG under a more comprehensive setting. The study seeks to answer: (1) Is the retrieval gain consistent across multiple model scales? (2) What is the relative importance of different retrieval methods and corpora? (3) What factors most limit biomedical QA performance?

**Key Insight**: This study focuses on small and medium-scale open-source instruction-tuned models, covering a range from 7B to 72B. It evaluates 10 datasets across expert-level and consumer-level questions and introduces a no-retrieval baseline for comparative analysis.

**Core Idea**: Through a large-scale systematic comparative experiment, the study reveals that the true scale of retrieval improvement is much smaller than previously reported. The key bottleneck lies in the model's ability to utilize retrieved evidence rather than the retrieval process itself.

## Method

### Overall Architecture

This is a purely empirical study aimed at measuring the true gains of biomedical RAG on small and medium-scale models. The overall pipeline is straightforward: given a biomedical question, multiple retrievers are used to fetch $top-k$ documents from various knowledge bases. These documents are then concatenated into the prompt for the generative model. In parallel, a no-retrieval baseline is established where no context is provided. Finally, performance is scored using ROUGE-L (for open-ended questions) or accuracy (for multiple-choice questions) to isolate the marginal contribution of retrieval. The experiments consist of a full combinatorial scan of "Model $\times$ Retrieval Method $\times$ Knowledge Base $\times$ Dataset."

### Key Designs

**1. Multi-dimensional full combinatorial comparative experiments: Isolating the marginal contribution of retrieval using a no-retrieval baseline**

The study decomposes influencing factors into three orthogonal dimensions: retrieval methods (BM25, TF-IDF, MedCPT, Hybrid RRF, and a no-retrieval baseline), knowledge bases (PubMed, Medical Textbooks, Yahoo Answers, HealthCareMagic), and 10 datasets (5 consumer-level + 5 expert-level). All combinations are evaluated independently using FP16 precision, greedy decoding, and a maximum of 300 new tokens. The core mechanism is the no-retrieval condition: by using the model's intrinsic knowledge as a reference point, the study prevents model capability from masking the true contribution of retrieval, exposing fluctuations as small as 1-2 points.

**2. Clean/Noise dual diagnosis: Separating failure causes between "failed to retrieve" and "misutilization"**

To locate the root cause of weak gains, the study constructs two scenarios. **Clean Retrieval** uses LLM-as-judge to filter 100 questions where all retrieval methods successfully hit relevant content to test if the model can use high-quality evidence. **Noise Retrieval** mixes 20 irrelevant documents with 5 relevant documents to test if the model is distracted. Design Motivation: if performance does not improve in the clean setting, the issue is evidence utilization; if it drops significantly in the noise setting, the issue is a lack of robustness to irrelevant context.

**3. Sensitivity scanning of $top-k$ and few-shot: Exposing intrinsic differences across model scales**

The study scans parameters for the number of retrieved documents ($top-k \in \{1, 3, 5, 10, 25, 50\}$) and the number of examples ($few-shot \in \{1, 3, 5, 10\}$). This scan reveals a scale-dependent phenomenon: 7-8B models experience a catastrophic performance collapse when the few-shot count is high (5 or 10), whereas 70B-grade models remain stable. This indicates that long-context capacity is a primary constraint for RAG gains in smaller models.

## Key Experimental Results

### Main Results: Open-ended QA Results (ROUGE-L)

| Model | No-Retrieval Baseline | BioASQ Retrieval | HealthCareMagic | Medical Textbooks | Yahoo Answers | Max Gain |
|------|----------|-----------|-----------------|-----------|---------|--------|
| Llama-3.1-8B | 13.06 | 14.24 | 12.90 | 13.22 | 12.81 | 1.18 |
| Llama-3.1-70B | 14.22 | 14.66 | 14.44 | 14.14 | 14.19 | 0.44 |
| Mistral-7B | 13.64 | 14.44 | 14.26 | 13.80 | 14.32 | 0.80 |
| Qwen2.5-7B | 12.91 | 13.56 | 13.00 | 13.25 | 13.20 | 0.65 |
| Qwen2.5-72B | 13.56 | 13.91 | 13.72 | 13.86 | 13.90 | 0.35 |

Key observations: (1) Retrieval gains are generally marginal (max 1.18 points) and unstable; (2) Model selection has a much larger impact than retrieval configuration; (3) The performance difference between expert-level and consumer-level retrieval corpora is $< 1$ point.

### Main Results: Multiple-Choice Accuracy Results

| Model | No-Retrieval | BioASQ | HealthCareMagic | Medical Textbooks | Yahoo Answers |
|------|------|--------|-----------------|-----------|---------|
| Llama-3.1-8B | 82.8 | 80.9 | 77.3 | 80.5 | 79.9 |
| Llama-3.1-70B | 86.4 | 86.9 | 82.3 | 83.6 | 86.5 |
| Mistral-7B | 75.7 | 68.6 | 69.5 | 72.3 | 71.2 |
| Qwen2.5-7B | 83.3 | 79.7 | 79.7 | 81.1 | 81.4 |
| Qwen2.5-72B | 85.6 | 84.3 | 84.6 | 84.9 | 84.0 |

Key finding: Small models (7-8B) are often harmed by retrieval; Mistral-7B even dropped by 6 points. Large models are more stable but show no significant gains.

### Key Findings

- **Evidence quality is not the bottleneck**: In the clean retrieval setting, performance improvements remain small and inconsistent. For example, Llama-3.1-70B improved from 0.410 to 0.660 under BM25, while Qwen2.5-72B showed almost no improvement.
- **Models are extremely sensitive to interference**: In the noise retrieval setting (mixing 20 irrelevant documents), all models dropped significantly. Llama-3.1-70B fell from 0.660 to 0.260, and Mistral-7B fell from 0.530 to 0.340.
- **Model scale dominates performance**: The no-retrieval accuracy of Qwen2.5-72B (85.6) exceeds the best retrieval configuration of any 7B model ($\le 83.3$).
- **Small models are sensitive to few-shot counts**: 7-8B models crash when the few-shot count reaches 5 or 10, while 70B models remain stable.

## Highlights & Insights

- **Direct Challenge to RAG Assumptions**: The paper uses large-scale systematic evaluation to overturn the common conclusion that "retrieval significantly improves medical QA," providing an important negative result for the RAG community.
- **Ingenious Dual Diagnosis Framework**: The comparison between clean and noise retrieval clearly separates two problems—"whether retrieval can fetch relevant documents" and "whether the model can effectively use those documents."
- **Balanced Coverage of Data**: The combination of 10 datasets and 4 diverse knowledge bases ensures the generalizability of the findings.
- **Transferable Observations**: The observation regarding small model collapse in long contexts ($few-shot > 3$) is a practical finding that can guide model and prompt configuration in industrial applications.

## Limitations & Future Work

**Limitations acknowledged by the authors**:
- Evaluation relies solely on reference-based metrics; faithfulness or factual consistency was not directly measured. Models might generate correct answers from parametric knowledge rather than retrieved evidence.
- Closed-source models at the level of GPT-4 were not included.
- Retrieval settings were standardized; advanced strategies like adaptive retrieval, re-ranking, or iterative retrieval were not explored.

**Own identified limitations**:
- The paper does not distinguish between different reasons for "no retrieval gain"—is it due to the model's inability to understand medical terminology, attention being distracted, or context length limits?
- The quality of the retrieval corpora was not quantified.

**Goal for improvement**:
- Potential directions include: (1) Integrating evidence alignment objectives during fine-tuning or distillation; (2) Implementing post-retrieval re-ranking or confidence filtering; (3) Adopting faithfulness evaluation frameworks to replace or supplement reference-based metrics.
- Exploring question-type-specific retrieval strategies and fusion mechanisms that combine parametric knowledge with retrieved evidence.

## Related Work & Insights

**vs. MedRAG (Xiong et al., 2024)**: MedRAG reported an 18% improvement on large models, but primarily evaluated multiple-choice questions and very large models. This paper finds marginal improvements on small/medium models and diverse question types, revealing the conditional nature of MedRAG's gains.

**vs. Self-RAG (Asai et al., 2024)**: Self-RAG improves retrieval and generation through self-reflection. This paper implicitly confirms the necessity of Self-RAG—if standard retrieval has weak effects, models must learn to selectively utilize retrieval.

**vs. General RAG Surveys (Gao et al., 2023)**: Surveys summarize RAG variants but lack large-scale empirical comparisons. This systematic evaluation fills that gap.

**Insights**:
- RAG is not a silver bullet; its effectiveness is highly context-dependent. RAG strategies should be designed for specific application scenarios rather than applied blindly.
- Model improvements (scaling, better instruction tuning) may be more cost-effective than optimizing retrieval methods alone.

## Rating

- **Novelty**: ⭐⭐⭐⭐ High-quality data-driven questioning of common RAG assumptions with broad coverage (5 models $\times$ 10 datasets $\times$ 4 methods $\times$ 4 corpora).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Massive scale (full combination of 200+ experimental conditions), including diagnostic clean/noise settings, ablation studies, and multi-metric evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic with well-supported tables and figures.
- **Value**: ⭐⭐⭐⭐⭐ Direct implications for the effectiveness of RAG in real-world applications, potentially shifting decisions regarding RAG investments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ICLR 2026\] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora](../../ICLR2026/information_retrieval/linearrag_linear_graph_retrieval_augmented_generation_on_large-scale_corpora.md)
- [\[ICLR 2026\] When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/when_to_use_graphs_in_rag_a_comprehensive_analysis_for_graph_retrieval-augmented.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
