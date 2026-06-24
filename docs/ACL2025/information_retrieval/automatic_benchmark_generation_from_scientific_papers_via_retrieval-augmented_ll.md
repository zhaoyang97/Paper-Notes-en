---
title: >-
  [Paper Note] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs
description: >-
  [ACL 2025][Information Retrieval & RAG][Automated Benchmark Generation] This paper proposes an automated benchmark generation method based on retrieval-augmented LLMs. It automatically extracts testable knowledge points from scientific papers and generates high-quality evaluation questions. Its effectiveness has been validated across domains such as NLP, machine learning, and bioinformatics, providing a new paradigm for the rapid construction of domain-specific LLM evaluation…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Automated Benchmark Generation"
  - "Scientific Papers"
  - "Retrieval-Augmentation"
  - "LLMs"
  - "Evaluation Datasets"
date: 2026-05-08
content_hash: 4f8175a67c15bea3
---

# Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs

**Conference**: ACL 2025  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Automated Benchmark Generation, Scientific Papers, Retrieval-Augmentation, LLMs, Evaluation Datasets

## TL;DR
This paper proposes an automated benchmark generation method based on retrieval-augmented LLMs. It automatically extracts testable knowledge points from scientific papers and generates high-quality evaluation questions. Its effectiveness has been validated across domains such as NLP, machine learning, and bioinformatics, providing a new paradigm for the rapid construction of domain-specific LLM evaluation benchmarks.

## Background & Motivation

**Background**: Evaluation benchmarks are core tools for measuring LLM capabilities. Existing LLM evaluation benchmarks (such as MMLU, BIG-Bench, GPQA, etc.) are primarily manually written or collected from existing exam questions. Specialized benchmarks in scientific domains (such as SciQ, PubMedQA) also rely on human annotation, which suffers from long construction cycles, low update frequencies, and limited coverage.

**Limitations of Prior Work**: The update rate of scientific knowledge far outpaces that of benchmarks—millions of papers are published annually, while evaluation benchmarks may only be updated once every few years. This leads to two issues: (1) Data contamination—LLMs may have seen questions from older benchmarks during pre-training, leading to inflated evaluation scores; (2) Knowledge coverage bias—benchmarks tend to cover classic and popular knowledge points, with insufficient coverage of cutting-edge and niche areas. Moreover, human benchmark construction is extremely costly, requiring domain experts to write and verify each question.

**Key Challenge**: High-quality evaluation benchmarks require a combination of domain expertise and diversity, which is difficult to achieve simultaneously through manual writing. The number of experts is limited, and they tend to draft questions in areas they are familiar with, leading to coverage bias. Automated methods can address issues of scale and diversity, but ensuring the quality and correctness of the generated questions remains the core challenge.

**Goal**: To develop a system capable of automatically and scalably generating high-quality evaluation questions from academic papers while ensuring correctness, diversity, and balanced difficulty coverage.

**Key Insight**: The authors view academic papers as structured vehicles of knowledge. Different sections (e.g., Method, Results, Related Work) contain different types of testable knowledge, allowing for the customized design of various question types.

**Core Idea**: By using a retrieval-augmented approach, LLMs are enabled to generate questions based on the full paper context, followed by multi-round quality filtering to ensure correctness and diagnostic value.

## Method

### Overall Architecture
The system is divided into four stages: (1) Knowledge point extraction—identifying testable knowledge points (facts, concepts, relations, reasoning chains) from scientific papers; (2) Question generation—generating various question types (multiple-choice, true/false, short answer) based on the extracted knowledge; (3) Quality filtering—filtering out low-quality questions through multi-dimensional automated quality control; (4) Difficulty calibration—estimating question difficulty based on the performance of multiple LLMs to ensure a balanced difficulty distribution. The entire process can be viewed as an automated pipeline from papers to benchmarks.

### Key Designs

1. **Hierarchical Knowledge Extraction (HKE)**:

    - **Function**: Systematically extracting different types and levels of testable knowledge points from papers.
    - **Mechanism**: After dividing the paper into paragraphs by section, LLMs are used on each paragraph to extract three levels of knowledge points: (a) Factual level: facts directly obtainable from the text (e.g., "Method A achieved $X\%$ accuracy on dataset B"); (b) Conceptual level: conceptual relationships that require understanding to answer (e.g., "Why does Method A use an attention mechanism instead of a recurrent network?"); (c) Reasoning level: reasoning tasks that require integrating information across sections to answer (e.g., "What is the core difference between Method A and Method B? Why is this difference more important on dataset C?"). Each knowledge point is appended with positioning information from its source paragraph and a confidence score. Overlapping sliding windows are used to handle cross-paragraph knowledge points.
    - **Design Motivation**: Different levels of knowledge points correspond to different question difficulties; hierarchical extraction ensures balanced difficulty distribution in the final benchmark.

2. **Retrieval-Augmented Question Generation (RAQG)**:

    - **Function**: Generating high-quality evaluation questions based on knowledge points and paper contexts.
    - **Mechanism**: For each knowledge point, retrieval enrichment is used to obtain complete context information—retrieving not only the paragraph housing the knowledge point but also other relevant paragraphs in the paper (via keyword and semantic similarity matching) and paragraphs from related papers (retrieved from a repository of papers from the same conference/direction). The LLM then generates questions within this rich context. For multiple-choice questions, the key lies in the quality of distractors—requiring them to be options that "look plausible but are actually incorrect." A "contrastive generation" strategy is adopted: the LLM is first prompted to generate the correct answer, and then distractors are generated based on the semantic neighborhood of the correct answer (e.g., altering a correct numerical value by 10-30%, or replacing the correct method with a similar but different one). Each question is accompanied by a detailed explanation of the correct answer and the location of evidence in the original paper.
    - **Design Motivation**: Relying solely on a single paragraph's context tends to generate questions that are trivial or detached from practical complexity; retrieval augmentation provides a more comprehensive knowledge background, making the questions more diagnostic.

3. **Multi-Dimensional Quality Filter Pipeline (MDQF)**:

    - **Function**: Automatically filtering out low-quality questions to ensure the overall quality of the final benchmark.
    - **Mechanism**: Four filtering stages are designed: (a) Correctness verification: another LLM (serving as the "solver") is used to independently answer the generated questions; if the solver's answer disagrees with the labeled correct answer, manual review is triggered or the question is discarded; (b) Uniqueness check: semantic similarity between questions is calculated to filter out duplicates that are too similar to existing questions; (c) Answerability verification: ensuring that the information required to answer the question indeed exists in the paper (rather than being hallucinated by the LLM) by inputting the question and the paper to an LLM and checking if its reasoning process cites specific paragraphs; (d) Ambiguity detection: checking multiple-choice questions to see whether there are multiple correct options or if none of the options are correct. Roughly 65% of the generated questions survive these four stages.
    - **Design Motivation**: Roughly 35% of the questions generated by LLMs exhibit various quality issues (factual errors, ambiguity, duplication), and using them directly would severely compromise the reliability of the benchmark.

### Loss & Training
The proposed method does not involve model training. Knowledge extraction and question generation utilize zero-shot/few-shot capabilities of GPT-4. The "solver" in quality filtering uses a different model than the "generator" (e.g., GPT-4 for generation and Claude-3 for solving) to reduce systemic LLM consistency bias. Difficulty calibration is achieved by having 5 LLMs of varying scales (from 7B to 70B) answer each question, using the pass rate as the estimate for difficulty.

## Key Experimental Results

### Main Results

| Area | Generated Questions | Post-Filtering Qty | Human Correctness Eval | Human-Written Control | LLM Discrimination |
|------|----------|----------|------------|----------|----------|
| NLP | 2,850 | 1,812 | 91.3% | 94.7% | 0.82 |
| ML | 2,400 | 1,536 | 89.7% | 93.2% | 0.79 |
| Bioinformatics | 1,800 | 1,098 | 87.2% | 92.1% | 0.75 |
| Physics | 1,500 | 945 | 85.8% | 91.5% | 0.73 |
| Combined | 8,550 | 5,391 | 88.5% | 92.9% | 0.77 |

### Ablation Study

| Configuration | Human Correctness Eval | Question Diversity | Description |
|------|------------|----------|------|
| Full MDQF | 91.3% | 0.78 | Full quality filtering |
| w/o Correctness Verification | 82.1% | 0.78 | Correctness dropped by -9.2% |
| w/o Uniqueness Check | 91.0% | 0.52 | Correctness unchanged but diversity plummeted |
| w/o Answerability Verification | 86.5% | 0.76 | Some questions lacked paper evidence |
| w/o Retrieval Augmentation | 84.7% | 0.71 | Both question quality and diversity declined |
| Factual-level only | 90.8% | 0.45 | High correctness but simple and uniform questions |

### Key Findings
- The gap in human correctness evaluation between auto-generated (91.3%) and human-written (94.7%) questions is only 3.4 percentage points, indicating that the automatically generated quality is approaching the human level.
- Correctness verification is the most critical filtering step (correctness drops by 9.2% if removed), while the uniqueness check primarily affects diversity rather than correctness.
- LLM discriminative power decreases as domain specialization increases (NLP: 0.82 $\rightarrow$ Physics: 0.73), indicating that the lack of LLM knowledge in highly specialized fields affects the discriminative ability of the questions.
- Compared to extracting only factual-level knowledge, hierarchical knowledge extraction (incorporating factual, conceptual, and reasoning levels) increases diversity by 73% (0.45 $\rightarrow$ 0.78), which is crucial for the diagnostic value of the benchmark.
- The post-filtering survival rate is approximately 63% (5,391/8,550), implying that large-scale generation paired with strict filtering is a viable automated strategy.

## Highlights & Insights
- Viewing papers as "structured containers of testable knowledge" is highly creative—different sections correspond to different types of knowledge points. This structured utilization of information is far more granular than simply asking an LLM to "write questions." This paradigm can be extended to the education sector (e.g., automatically generating exam questions from textbooks).
- The design of "using different models for generation and solving" effectively alleviates LLM consistency bias—where a single model might easily answer questions it generated itself, whereas other models cannot. This cross-validation strategy serves as a valuable reference.
- The difficulty calibration method is highly practical: using the pass rates of multiple LLMs of different scales as difficulty proxies is more standardized and reproducible than human difficulty assessments.

## Limitations & Future Work
- The current method is primarily applicable to English scientific papers; processing multilingual and non-English papers requires additional adaptation.
- The correctness of questions at the reasoning level (85%) is lower than that at the factual level (93%), indicating that there is still room for improvement in the automated generation of complex reasoning questions.
- The generated questions may suffer from "paper-specificity"—questions that rely too heavily on specific details of a single paper may offer limited utility in evaluating general capabilities.
- Future work could integrate continual learning to automatically update and expand benchmarks as new papers are published, addressing the benchmark aging issue.

## Related Work & Insights
- **vs MMLU/GPQA**: Manually written standard benchmarks exhibit high quality but limited coverage and slow updates; our automated method can rapidly scale coverage, making them complementary.
- **vs DyVal (Zhu et al., 2023)**: DyVal proposed dynamic evaluation benchmarks but relied on template-based generation; ours generates from actual papers, yielding more diverse and in-depth questions.
- **vs AutoBench (Li et al., 2024)**: AutoBench also utilizes LLMs to generate evaluation data but lacks a systematic design for knowledge extraction and multi-dimensional quality filtering; our pipeline is more complete and reliable.
- **vs FActScore**: FActScore evaluates the factuality of LLM-generated content; our correctness verification module adopts a similar fact-checking approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of automatically generating evaluation benchmarks from papers is novel, with elegant designs for hierarchical extraction and quality filtering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-domain evaluation with human-annotated controls and complete ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the pipeline and well-argued motivations for each module.
- Value: ⭐⭐⭐⭐⭐ Highly significant for addressing benchmark aging and construction cost issues; the proposed method is ready for practical implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)
- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)

</div>

<!-- RELATED:END -->
