---
title: >-
  [Paper Note] On Synthesizing Data for Context Attribution in Question Answering
description: >-
  [ACL 2025][NLP Understanding][Context Attribution] This paper proposes SynQA, a synthetic data strategy based on the "given context sentences $\rightarrow$ generate QA pairs" paradigm, designed to train small models for context attribution tasks (i.e., identifying supporting evidence sentences for QA system answers). SynQA significantly outperforms zero-shot inference and LLM ensemble methods across multiple QA tasks and cross-domain scenarios.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Context Attribution"
  - "Question Answering"
  - "Synthetic Data"
  - "LLM Fine-tuning"
  - "Hallucination Detection"
date: 2026-05-08
content_hash: b6a766034673cbbe
---

# On Synthesizing Data for Context Attribution in Question Answering

**Conference**: ACL 2025  
**arXiv**: [2504.05317](https://arxiv.org/abs/2504.05317)  
**Code**: None  
**Area**: NLP Understanding / Information Retrieval  
**Keywords**: Context Attribution, Question Answering, Synthetic Data, LLM Fine-tuning, Hallucination Detection

## TL;DR

This paper proposes SynQA, a synthetic data strategy based on the "given context sentences $\rightarrow$ generate QA pairs" paradigm, designed to train small models for context attribution tasks (i.e., identifying supporting evidence sentences for QA system answers). SynQA significantly outperforms zero-shot inference and LLM ensemble methods across multiple QA tasks and cross-domain scenarios.

## Background & Motivation

**Background**: Question Answering (QA) is one of the most prominent application scenarios for LLMs, but LLMs sometimes generate false or misleading answers ("hallucinations"). Therefore, anchoring generated answers in the provided context information—i.e., providing sources of evidence for the generated text—is crucial for the reliability of LLMs.

**Limitations of Prior Work**: Context attribution tasks require models to label which context sentences correspond to each part of the answer. The key challenges of this task include: (1) High-quality attribution annotation data is extremely scarce, and human annotation costs are prohibitive; (2) Directly using LLMs for zero-shot inference yields unstable attribution performance; (3) Existing methods either require exceptionally strong LLM capabilities (high cost) or focus only on simple binary judgments rather than precise sentence-level attribution.

**Key Challenge**: A large amount of training data is required to train reliable small models for attribution, yet human-annotated data is too expensive. The key question is: how to leverage the generation capabilities of LLMs to automatically synthesize high-quality attribution training data?

**Goal**: To systematically study LLM-based context attribution methods (zero-shot, ensemble, fine-tuning) and propose an effective synthetic data strategy to train cost-efficient small models that achieve or even surpass the attribution performance of large models.

**Key Insight**: The traditional paradigm lets the LLM generate the answer first and then annotate the attribution, which tends to introduce errors. SynQA reverses this process—first selecting the context sentences, and then letting the LLM generate the QA pairs based on these sentences. Consequently, the attribution relationship is naturally built into the synthetic data.

**Core Idea**: Utilizing a reverse synthesis strategy of "first select evidence sentences $\rightarrow$ then generate QA pairs," leveraging the text generation strengths of LLMs while ensuring that the synthetic data inherently possesses clear attribution paths.

## Method

### Overall Architecture

The pipeline of SynQA is: (1) Sample a set of context sentences from a document to serve as evidence; (2) Use a large LLM (such as GPT-4) to generate a question and its corresponding answer based on these sentences; (3) Treat the (question, answer, evidence sentences) triples as training data; (4) Use this data to fine-tune small models (such as Flan-T5, Mistral-7B) for attribution classification. During inference, given a question, context, and the generated answer, the small model determines whether each context sentence is supporting evidence for the answer.

### Key Designs

1. **Reverse Synthetic Data Strategy (SynQA)**:

    - **Function**: Automatically generate a large amount of high-quality context attribution training data.
    - **Mechanism**: While the traditional approach is "given a question $\rightarrow$ generate an answer $\rightarrow$ annotate which sentences support the answer", SynQA reverses this to "randomly select $k$ context sentences $\rightarrow$ let the LLM generate questions and answers based on these sentences." Correspondingly, the generated QA pairs are naturally supported by the selected sentences, and the attribution labels are automatically acquired. The number of sampled sentences $k$ varies from one to many, generating training samples of different difficulty levels.
    - **Design Motivation**: LLMs excel at text generation but struggle with precise annotation. The reverse strategy converts attribution annotation into input constraints, transforming a difficult labeling task into a generation task that LLMs are better suited for.

2. **Systematic Multi-Method Comparison Framework**:

    - **Function**: Comprehensively evaluate the pros and cons of different LLM attribution methods.
    - **Mechanism**: Three classes of methods are studied: (i) Zero-shot inference—directly prompting the LLM to identify supporting sentences and testing different prompting strategies; (ii) LLM ensembles—aggregating judgments from multiple LLMs through majority voting or weighted fusion; (iii) Fine-tuning small LMs—fine-tuning small models such as Flan-T5-XL (3B) and Mistral-7B-Instruct on SynQA synthetic data.
    - **Design Motivation**: It is necessary to identify the optimal choice under different cost-performance trade-offs, and whether fine-tuning small models can indeed approach or even surpass large models.

3. **Cross-Domain Generalization Evaluation**:

    - **Function**: Validate the transferability of SynQA synthetic data across different QA tasks and domains.
    - **Mechanism**: Evaluate on multiple QA datasets, including Natural Questions, ExpertQA, and HAGRID, which cover various scenarios like Wikipedia QA and expert domain QA. Training and test data originate from different domains to assess zero-shot transfer performance.
    - **Design Motivation**: Practical applications of context attribution span diverse domains; methods that work only within the training domain offer limited value. Therefore, cross-domain generalization capability is key to practical utility.

### Loss & Training

Standard sequence-to-sequence loss is used when fine-tuning small models. The input format is "(question, answer, context_sentence) $\rightarrow$ {attributable, not_attributable}". Training uses SynQA synthetic data, with ablation experiments conducted on different synthetic data scales (1k-10k samples).

## Key Experimental Results

### Main Results

| Method | Natural Questions (F1) | ExpertQA (F1) | HAGRID (F1) | Cost |
|------|----------------------|--------------|------------|------|
| GPT-4 Zero-shot | 68.2 | 62.5 | 65.4 | High |
| LLM Ensemble (3 models) | 71.3 | 64.8 | 68.1 | Very High |
| Flan-T5-XL + SynQA | 74.5 | 67.3 | 71.2 | Low |
| Mistral-7B + SynQA | **76.8** | **69.1** | **73.5** | Medium |

### Ablation Study

| Configuration | Natural Questions (F1) | Description |
|------|----------------------|------|
| Mistral-7B + SynQA (10k) | 76.8 | Full model |
| Mistral-7B + SynQA (5k) | 75.2 | Data halved with minor impact |
| Mistral-7B + SynQA (1k) | 71.4 | Significant degradation with insufficient data |
| Traditional Forward Synthesis (question-answering-annotation) | 70.1 | Reverse strategy advantage of approx. +6.7% |
| Direct Zero-shot (no synthetic data) | 68.2 | Fine-tuning brings significant improvement |

### Key Findings

- **Small Models + SynQA Fine-tuning > Large Models Zero-shot**: Fine-tuned Mistral-7B outperforms GPT-4 zero-shot inference across all test sets while being significantly cheaper.
- **Reverse Synthesis Strategy is Effective**: SynQA's "evidence selection $\rightarrow$ QA generation" approach yields substantially higher data quality than the traditional "generation $\rightarrow$ annotation" approach, translating to a 5-7% F1 improvement in downstream fine-tuning.
- **Good Cross-Domain Transfer**: Models trained on synthetic data from one domain can effectively transfer to other QA domains, indicating that the attribution patterns learned by SynQA are generic.
- **User Study Validates Utility**: Human evaluation confirms that attributions provided by the fine-tuned small model indeed help users verify answer correctness more rapidly.

## Highlights & Insights

- **Ingenious Concept of "Reverse Synthesis"**: Transforming the annotation problem into a generation problem is the core contribution. This paradigm can be generalized to any NLP task requiring paired annotation where human labeling is expensive—such as evidence attribution in summarization, knowledge lineage in dialogue systems, etc.
- **Highly Practical Cost-Performance Trade-off**: It proves that on the deployment side, training small models on synthetic data is more economical and efficient than calling large LLM APIs. This serves as a direct reference for production systems that require large-scale deployment of attribution functionalities.
- **Limited Marginal Returns of LLM Ensembles**: The improvement from multi-LLM voting is far less than using a single LLM to generate training data for fine-tuning small models. This finding challenges the intuition that "ensembles are always better."

## Limitations & Future Work

- **Synthetic Data Quality Hooked to LLM Capability**: The performance of SynQA is constrained by the quality of the LLM generating the QA pairs; if a weaker LLM is used to generate synthetic data, the performance will degrade.
- **Validated Only in English**: Cross-lingual attribution scenarios (e.g., multilingual RAG systems) have not been explored.
- **Attribution Granularity Fixed at Sentence Level**: Practical applications might require finer-grained (clause-level) or coarser-grained (paragraph-level) attribution.
- Future work can explore extending SynQA to multimodal settings, such as visual evidence attribution in mixed text-and-image QA.

## Related Work & Insights

- **vs ALCE (Gao et al., 2023)**: ALCE focuses on making LLMs include citations during generation ("attribution-during-generation"); SynQA focuses on post-hoc attribution, and the two are complementary.
- **vs AttriBench**: AttriBench provides benchmarks for evaluating attribution but does not offer solutions; SynQA provides both evaluation and training methods.
- **Connection to RAG Systems**: The attribution model trained by SynQA can be directly integrated as a post-processing module in RAG systems, providing evidence tracing for retrieval-augmented answers.

## Rating

- Novelty: ⭐⭐⭐⭐ The "reverse synthesis" idea is novel and effective, though synthetic data fine-tuning itself is already a common paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, comparison of various methods, ablations, and a user study are all provided.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic description of method.
- Value: ⭐⭐⭐⭐ Direct practical value for trustworthy AI and RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

</div>

<!-- RELATED:END -->
