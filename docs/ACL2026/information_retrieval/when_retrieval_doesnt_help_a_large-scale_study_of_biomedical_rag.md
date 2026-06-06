---
title: >-
  [Paper Note] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study
description: >-
  [ACL 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This large-scale empirical study across 5 models, 10 datasets, 4 retrieval methods…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Biomedical Question Answering"
  - "RAG"
  - "Large Language Models"
  - "Evidence Utilization"
date: 2026-05-08
content_hash: 298dbc7d9dfbaa4c
---

# When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study

**Conference**: ACL 2026  
**arXiv**: [2606.04127](https://arxiv.org/abs/2606.04127)  
**Code**: https://github.com/erfan-nourbakhsh/BioMedicalRAG  
**Area**: Information Retrieval / LLM / Biomedical NLP  
**Keywords**: Retrieval-Augmented Generation, Biomedical Question Answering, RAG, Large Language Models, Evidence Utilization

## TL;DR
This large-scale empirical study across 5 models, 10 datasets, 4 retrieval methods, and 4 retrieval corpora finds that biomedical RAG yields only small, unstable improvements of 1-2 points. The true bottleneck is identified as the models' ability to effectively utilize retrieved evidence rather than retrieval quality itself.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become a mainstream solution in medical question answering (QA). Prior work like MedRAG reported accuracy gains of up to 18% on large-scale models. Medical QA is a high-stakes scenario where factual errors can have serious consequences, making accuracy improvements particularly critical.

**Limitations of Prior Work**: Existing research primarily evaluates large proprietary models (GPT-4, GPT-3.5, Mixtral-8×7B, Llama2-70B) or 70B-class models, often using zero-shot multiple-choice evaluation. There is a lack of systematic evaluation for more practical, resource-constrained 7-8B open-source models. Furthermore, most evaluations focus on professional-grade biomedical questions, neglecting common real-world consumer health queries.

**Key Challenge**: Are the gains from RAG truly as significant as studies on large models suggest? This remains unanswered for small-to-medium scale models. If retrieval does not help, what is the actual limiting factor—the retrieval algorithm, the corpora, or the model itself?

**Goal**: To revisit the effectiveness of biomedical RAG under a more comprehensive setup. The study aims to answer: (1) Are retrieval gains consistent across model scales? (2) What is the relative importance of different retrieval methods and corpora? (3) What factors most limit biomedical QA performance?

**Key Insight**: Centering the study on small-to-medium scale open-source instruction-tuned models (7B to 72B), covering 10 datasets spanning expert and consumer questions, and introducing no-retrieval baselines for comparative analysis.

**Core Idea**: Through large-scale systematic comparative experiments, this work reveals that the actual scale of retrieval improvement is far smaller than previously reported, and the critical bottleneck lies in the models' capacity to utilize evidence rather than the retrieval process itself.

## Method

### Overall Architecture

The paper designs a complete experimental pipeline to evaluate the effectiveness of biomedical RAG. The core framework includes:

- **Input Phase**: Questions from biomedical QA datasets.
- **Retrieval Phase**: Querying 4 knowledge bases using 4 retrievers to obtain top-k documents.
- **Fusion Phase**: Concatenating retrieved documents into the prompt for the generative model.
- **Output Phase**: Model-generated answers.
- **Evaluation Phase**: Measuring performance using ROUGE-L (open-ended) or Accuracy (multiple-choice).

In the no-retrieval condition, the model receives only the question without any retrieved context.

### Key Designs

1.  **Multi-dimensional Comparative Experimental Design**:
    - **Function**: To systematically isolate the impact of different components on performance.
    - **Mechanism**: Three independent dimensions are defined—retrieval methods (BM25, TF-IDF, MedCPT, Hybrid RRF, and No-Retrieval baseline), retrieval corpora (PubMed, Medical Textbooks, Yahoo Answers, HealthCareMagic), and evaluation datasets (5 consumer-grade + 5 expert-grade, 10 total). Independent evaluations are conducted for all combinations. All models use FP16 precision, greedy decoding, and a maximum of 300 new tokens.
    - **Design Motivation**: To avoid confounding factors. The standalone baseline allows researchers to accurately measure the marginal contribution of retrieval without it being masked by model capability or other variables.

2.  **Dual Settings for Quality Diagnosis**:
    - **Function**: To identify the real cause of insufficient retrieval effectiveness.
    - **Mechanism**: Two special evaluation scenarios are designed: (1) Clean Retrieval: Using LLM-as-judge to determine if retrieval results contain sufficient information, selecting 100 questions where all methods succeeded; (2) Noisy Retrieval: Mixing 20 irrelevant documents with 5 relevant ones to test model robustness.
    - **Design Motivation**: To pinpoint the root of the problem. If performance does not improve under clean retrieval, the issue lies with the model; if it drops significantly under noisy retrieval, the model lacks robustness.

3.  **Ablation and Sensitivity Analysis**:
    - **Function**: To identify key hyperparameters and configurations affecting performance.
    - **Mechanism**: Parameter sweeps are performed on the number of retrieved documents (top-k: 1/3/5/10/25/50) and the number of few-shot examples (1/3/5/10).
    - **Design Motivation**: To understand differences in model behavior. Small models (7-8B) show sharp performance declines at high few-shot counts, while large models remain relatively stable, reflecting intrinsic limitations of different scales.

## Key Experimental Results

### Main Results: Open-Ended QA (ROUGE-L)

| Model | No-Retrieval | BioASQ | HealthCareMagic | Textbooks | Yahoo | Max Gain |
|-------|--------------|--------|-----------------|-----------|-------|----------|
| Llama-3.1-8B | 13.06 | 14.24 | 12.90 | 13.22 | 12.81 | 1.18 |
| Llama-3.1-70B | 14.22 | 14.66 | 14.44 | 14.14 | 14.19 | 0.44 |
| Mistral-7B | 13.64 | 14.44 | 14.26 | 13.80 | 14.32 | 0.80 |
| Qwen2.5-7B | 12.91 | 13.56 | 13.00 | 13.25 | 13.20 | 0.65 |
| Qwen2.5-72B | 13.56 | 13.91 | 13.72 | 13.86 | 13.90 | 0.35 |

Key Observations: (1) Retrieval gains are generally small (max 1.18 points) and unstable; (2) Model selection has a much larger impact than retrieval configuration; (3) The difference between expert and consumer corpora is <1 point.

### Multiple-Choice Accuracy Results

| Model | No-Retrieval | BioASQ | HealthCareMagic | Textbooks | Yahoo |
|-------|--------------|--------|-----------------|-----------|-------|
| Llama-3.1-8B | 82.8 | 80.9 | 77.3 | 80.5 | 79.9 |
| Llama-3.1-70B | 86.4 | 86.9 | 82.3 | 83.6 | 86.5 |
| Mistral-7B | 75.7 | 68.6 | 69.5 | 72.3 | 71.2 |
| Qwen2.5-7B | 83.3 | 79.7 | 79.7 | 81.1 | 81.4 |
| Qwen2.5-72B | 85.6 | 84.3 | 84.6 | 84.9 | 84.0 |

Surprising Finding: Small models (7-8B) are often harmed by retrieval—Mistral-7B dropped by 6 points; larger models are more stable but show no significant gain.

### Key Findings

- **Evidence quality is not the bottleneck**: Performance improvements remain small and inconsistent even in clean retrieval settings. For instance, Llama-3.1-70B improved from 0.410 to 0.660 under BM25, while Qwen2.5-72B showed almost no gain.
- **Models are extremely sensitive to distraction**: In noisy retrieval settings (20 irrelevant documents), all models declined sharply. Llama-3.1-70B dropped from 0.660 to 0.260.
- **Model scale dominates performance**: The no-retrieval accuracy of Qwen2.5-72B (85.6) exceeds the best retrieval configuration of any 7B model ($\le 83.3$).
- **Small models are sensitive to few-shot counts**: 7-8B models collapse when few-shot counts reach 5 or 10, whereas 70B models remain stable.

## Highlights & Insights

- **Challenge to RAG Assumptions**: The paper uses large-scale systematic evaluation to overturn the common conclusion that "retrieval significantly improves medical QA," providing an important negative result for the RAG community.
- **Dual-Layer Diagnostic Framework**: The comparison between clean and noisy retrieval clearly separates two issues: "whether retrieval can find relevant documents" and "whether models can effectively use them."
- **Balanced Coverage**: The combination of 10 datasets and 4 diverse corpora ensures the generalizability of findings across consumer and expert domains.
- **Transferable Observations**: The discovery regarding small model collapse in long contexts (few-shot > 3) is a practical finding for configuring model+prompting in industrial applications.

## Limitations & Future Work

**Limitations acknowledged by the authors**:
- Use of reference-based metrics only, without direct measurement of faithfulness or factual consistency. Models might generate correct answers from parametric knowledge rather than retrieved evidence.
- Exclusion of closed-source models at the GPT-4 level.
- Relatively standardized retrieval setups, without exploring advanced strategies like adaptive retrieval, re-ranking, or iterative retrieval.

**Self-identified limitations**:
- The paper does not distinguish between different reasons for "no retrieval gain"—is it due to medical terminology comprehension, attention distraction, or context length limits?
- The quality of retrieval corpora is not quantified.

**Future Work**:
- Directions for improvement include: (1) Incorporating evidence alignment objectives during fine-tuning or distillation; (2) Implementing post-retrieval re-ranking or confidence filtering; (3) Adopting faithfulness evaluation frameworks.
- Potential areas: Question-type specific retrieval strategies and fusion mechanisms that combine parametric knowledge with external evidence.

## Related Work & Insights

**vs. MedRAG (Xiong et al., 2024)**: MedRAG reported 18% improvement but focused on large models and multiple-choice questions. This work finds minimal improvement in small models and diverse query types, revealing the conditionality of MedRAG's gains.

**vs. Self-RAG (Asai et al., 2024)**: Self-RAG improves retrieval and generation through self-reflection. This work implies the necessity of Self-RAG—if standard retrieval is weak, models must learn to selectively utilize information.

**vs. General RAG Surveys (Gao et al., 2023)**: While surveys summarize variants, they lack large-scale empirical comparisons. This systematic evaluation fills that gap for the biomedical domain.

**Insights**:
- RAG is not a silver bullet; its effectiveness is highly context-dependent. RAG strategies should be designed for specific scenarios rather than applied blindly.
- Model improvements (scaling up, better instruction tuning) may be more cost-effective than optimizing retrieval methods alone.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Data-driven questioning of common RAG assumptions with extensive coverage (5 models × 10 datasets × 4 methods × 4 corpora).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large scale (200+ full experimental combinations) including diagnostic settings and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic with well-supported tables and figures.
- **Value**: ⭐⭐⭐⭐⭐ Direct implications for the feasibility of RAG in real-world applications, potentially shifting decisions on RAG investment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](../../ICLR2026/information_retrieval/efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
