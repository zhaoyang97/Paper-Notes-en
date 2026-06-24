---
title: >-
  [Paper Note] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification
description: >-
  [ACL 2025][Multimodal VLM][Scientific Claim Verification] SciVer is the first benchmark dataset for multimodal scientific claim verification, containing 3,000 expert-annotated samples across 1,113 Computer Science (CS) papers. It designs four reasoning subtasks: Direct, Parallel, Sequential, and Analytical. Evaluation of 21 foundation models shows that the strongest model, o4-mini (77.7%), still exhibits a significant gap of 16% compared to human experts (93.8%).
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Scientific Claim Verification"
  - "Multimodal Reasoning"
  - "Benchmark Evaluation"
  - "Foundation Models"
  - "Evidence Annotation"
date: 2026-05-08
content_hash: ad92e0c27d509e7f
---

# SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification

**Conference**: ACL 2025  
**arXiv**: [2506.15569](https://arxiv.org/abs/2506.15569)  
**Code**: [QDRhhhh/SciVer](https://github.com/QDRhhhh/SciVer)  
**Area**: Multimodal VLM / Scientific Literature Understanding  
**Keywords**: Scientific Claim Verification, Multimodal Reasoning, Benchmark Evaluation, Foundation Models, Evidence Annotation

## TL;DR

SciVer is the first benchmark dataset for multimodal scientific claim verification, containing 3,000 expert-annotated samples across 1,113 Computer Science (CS) papers. It designs four reasoning subtasks: Direct, Parallel, Sequential, and Analytical. Evaluation of 21 foundation models shows that the strongest model, o4-mini (77.7%), still exhibits a significant gap of 16% compared to human experts (93.8%).

## Background & Motivation

**Background**: Scientific claim verification requires models to determine whether a scientific statement is supported or refuted by given evidence. Existing benchmarks are primarily limited to a single modality: SciFact uses only abstract text, SciTAB is based solely on a single table, and TabFact/ChartCheck are based on Wikipedia tables or charts. In contrast, verifying real-world scientific papers requires the simultaneous understanding of multimodal information, including text paragraphs, data tables, and statistical charts. Although multimodal scientific QA benchmarks like ArXivQA and CharXiv have emerged recently, they are still limited to single-image or single-table QA, lacking cross-modal joint reasoning.

**Limitations of Prior Work**: (1) No existing benchmark tests the multimodal claim verification capabilities of models within the context of entire scientific papers. (2) Most existing datasets rely on crowdsourced annotations where annotators lack domain expertise, leading to unreliable claim quality and evidence annotation. (3) There is a lack of fine-grained evaluation across different reasoning types—direct retrieval, multi-source parallel integration, multi-hop sequential reasoning, and analytical reasoning requiring domain knowledge differ vastly in difficulty, making it impossible to diagnose model bottlenecks when tests are lumped together.

**Key Challenge**: Understanding real-world scientific literature requires complex cross-modal reasoning, but existing benchmarks fail to differentiate and evaluate this capability. Consequently, there is a lack of accurate understanding regarding the true performance of foundation models on this critical task.

**Goal**: Build the first multimodal scientific claim verification benchmark that covers various reasoning types and multimodal evidence, ensure quality through expert annotation, and systematically evaluate the capabilities of current foundation models.

**Key Insight**: The authors design four subtasks based on reasoning types (Direct, Parallel, Sequential, and Analytical). Each sample contains not only entailed/refuted labels but also expert-annotated supporting evidence (text paragraphs + tables + charts) to support fine-grained error diagnosis. The source data consists of arXiv papers published between September and November 2024, ensuring that no models have seen these data during pre-training.

**Core Idea**: Build the first multimodal scientific claim verification benchmark designed from the perspective of reasoning types. By ensuring quality through expert annotation by 18 Computer Science graduate students, the study systematically reveals the capability bottlenecks of foundation models in multimodal reasoning on scientific literature.

## Method

### Overall Architecture

SciVer is an evaluation benchmark rather than a model. The construction pipeline is as follows: (1) HTML versions of CS papers containing tables and charts are collected from arXiv, filtering for papers with peer-review acceptance records. (2) Eighteen CS graduate student annotators draft entailed claims across four reasoning types after undergoing a 2-hour training session, followed by semi-automatic generation of refuted claims. (3) A second annotator independently annotates the supporting evidence and verifies the labels; in case of disagreement, a third arbitrator is introduced, achieving a 94.0% inter-annotator agreement rate. (4) Evaluation is conducted on 21 models, where the input is the multimodal context of the scientific paper (text paragraphs + table screenshots + chart screenshots) + the claim, and the output is entailed/refuted.

### Key Designs

1. **Subtask Division by Four Reasoning Types**:

    - **Function**: Fine-grained evaluation of model capabilities across the dimension of reasoning complexity.
    - **Mechanism**: Direct Reasoning—verifiable by directly extracting from a single information source; Parallel Reasoning—requires integrating multiple independent search sources simultaneously; Sequential Reasoning—requires building a multi-step reasoning chain, where the conclusion of the previous step serves as the premise for the next; Analytical Reasoning—requires in-depth analysis combining domain knowledge and methodological understanding. Each type contains 750 samples (with 500 in the test set).
    - **Design Motivation**: Experiments verify that different reasoning types indeed correspond to varying difficulty gradients—o4-mini achieves 85.0% on Direct but only 67.6% on Analytical, exhibiting a gap of 17.4%. This layered evaluation can pinpoint the exact reasoning bottlenecks of models.

2. **Expert-Annotated Supporting Evidence**:

    - **Function**: Provide the minimum set of evidence (text paragraphs, tables, charts) required to verify each claim, enabling the evaluation of evidence retrieval capabilities.
    - **Mechanism**: Claim annotators initially write claims based on three randomly selected multimodal elements (ensuring claims must reference visual elements), and then a second independent annotator marks all supporting evidence. On average, each sample requires 2.62 pieces of evidence.
    - **Design Motivation**: With annotated evidence, the performance difference between RAG methods and full-text inputs can be compared. Under oracle evidence, Qwen2.5-VL-72B improves from 70.2% to 75.3% (+5.1%), illustrating that evidence localization is a major bottleneck for current models.

3. **Semi-Automatic Generation Strategy for Refuted Claims**:

    - **Function**: Generate high-quality refuted claims, avoiding formulaic negative expressions.
    - **Mechanism**: Annotators introduce factual errors (modifying numbers, replacing relationships, swapping comparison directions, etc.) based on annotated entailed claims to make the claims contradict the evidence. Thus, refuted claims are lexically highly similar to their corresponding entailed claims, preventing models from relying on superficial cues.
    - **Design Motivation**: Drafting refuted claims from scratch is extremely difficult for annotators. Perturbing entailed claims ensures that positive and negative samples share the same paper context and evidence scope, reducing bias.

### Loss & Training

SciVer is an evaluation benchmark and does not involve model training. The evaluation uses a zero-shot setting with optional Chain-of-Thought prompting. The models take the multimodal context of the scientific paper + the claim as input, and output binary classification labels.

## Key Experimental Results

### Main Results

Accuracies (%) of 21 models on the SciVer test set:

| Model | Direct | Parallel | Sequential | Analytical | Avg |
|------|--------|----------|------------|------------|-----|
| Human Expert | 100.0 | 95.0 | 90.0 | 90.0 | **93.8** |
| o4-mini | 85.0 | 80.6 | 77.6 | 67.6 | **77.7** |
| Gemini-2.5-Flash | 79.8 | 76.0 | 73.2 | 71.4 | 75.1 |
| GPT-4o | 77.0 | 71.2 | 73.6 | 73.8 | 73.9 |
| GPT-4.1 | 77.6 | 73.2 | 71.2 | 70.8 | 73.2 |
| Mistral-Small-3.1-24B | 74.8 | 66.0 | 68.6 | 75.6 | 71.3 |
| Qwen2.5-VL-72B | 70.8 | 69.2 | 68.2 | 69.2 | 69.4 |
| Llama-3.2-11B-Vision | — | — | — | — | ~52 |
| Random Guess | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |

### Ablation Study

RAG Experiments (Qwen2.5-VL-72B):

| Retrieval Strategy | Accuracy | Gain |
|---------|-------|------|
| Full-Text Input | 70.2 | Baseline |
| Random Evidence | 66.3 | -3.9 |
| OpenAI Embedding RAG | 72.9 | +2.7 |
| Oracle Evidence | **75.3** | +5.1 |

Error Analysis (Qwen2.5-VL-72B, 100 error samples):

| Error Type | Proportion |
|---------|------|
| Relevant information not retrieved | 32% |
| Misinterpretation of visual elements | 21% |
| Multi-step reasoning failure | 17% |
| Over-reliance on textual modality | 12% |
| Domain knowledge error | 10% |
| Others | 8% |

### Key Findings

- **Obvious Reasoning Complexity Gradient**: Model performance decreases linearly from Direct (85%) to Analytical (67.6%), verifying that the four reasoning types indeed capture different difficulty levels. Interestingly, Mistral-Small outperforms GPT-4o on Analytical (75.6% vs. 73.8%) while lagging significantly behind on Parallel (66.0% vs. 71.2%).
- **Retrieval is the Major Bottleneck**: 32% of errors stem from the failure to retrieve relevant evidence. Utilizing oracle evidence provides a 5.1% gain, indicating that long-document localization capability is a key limiting factor.
- **Visual Understanding Remains Unreliable**: 21% of errors arise from misreading tables/charts. Models tend to over-rely on text while ignoring visual information (12% of errors), even when visual information is necessary to verify the claim.
- **Large Gap Between Open-Source and Closed-Source**: The strongest open-source model, Mistral-Small-3.1-24B (71.3%), lags behind o4-mini (77.7%) by 6.4%. Smaller open-source models, such as Llama-3.2-11B, perform close to random guess level.

## Highlights & Insights

- **Layered Design of Reasoning Types as the Key Highlight**: Dividing reasoning into four types allows the benchmark to not only measure overall scores but also accurately diagnose specific model weaknesses across dimensions like evidence retrieval, multi-source integration, multi-hop reasoning, and domain knowledge. This design concept can be transferred to the construction of benchmarks in other domains.
- **High Cost but High Return of Expert Annotations**: 18 CS graduate student annotators, 2-hour individual training, double annotation + arbitration, achieving a 94.0% agreement rate. Although expensive, this ensures that the quality of claims and evidence far surpasses crowdsourced alternatives, rendering the benchmark conclusions more trustworthy.
- **RAG Analysis Reveals Practical Insights**: Oracle evidence only yields a 5.1% gain (with 75.3% still far below the human performance of 93.8%), indicating that even perfect retrieval cannot compensate for insufficient reasoning capabilities. Future work needs to improve both retrieval and reasoning simultaneously.

## Limitations & Future Work

- **Limited to CS Papers**: All papers are sourced from the CS domain of arXiv, which may not represent the verification difficulty in other scientific fields such as biomedicine or physics.
- **Limited Modality Coverage**: The dataset only includes text, tables, and charts, leaving out modalities critical to certain fields, such as equations, experimental setup diagrams, or schematics.
- **Scale Bound by Expert Annotation Costs**: Although high in quality, the 3,000 samples are limited in scale, making it difficult to support the evaluation of long-tail reasoning patterns.
- **Overly Simplified Binary Format**: Real-world scientific claim verification often involves "partially supported/partially refuted" or "insufficient evidence" cases. The binary classification format may fail to capture such fine-grained judgments.

## Related Work & Insights

- **vs. SciFact (Wadden et al. 2020)**: SciFact uses only the paper abstract text for claim verification, whereas SciVer utilizes the multimodal context of the entire paper, which is closer to real-world scenarios.
- **vs. CharXiv / ArXivQA**: These benchmarks focus on single-image/chart QA tasks, whereas SciVer targets claim verification across cross-modal sources, requiring higher reasoning complexity.
- **vs. SciTAB (Lu et al. 2023)**: SciTAB performs claim verification based on a single scientific table, while SciVer simultaneously integrates text, tables, and charts.

## Rating
- Novelty: ⭐⭐⭐⭐ The first claim verification benchmark for multimodal scientific literature, presenting a creative layered design for reasoning types.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 21 models (both open-source and closed-source), featuring multi-dimensional experiments including CoT, RAG, and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with detailed descriptions of the data construction pipeline and complete statistical analysis.
- Value: ⭐⭐⭐⭐ Fills a gap in multimodal verification benchmarks for scientific literature, with error analysis pointing toward future improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
