---
title: >-
  [Paper Note] Basic Reading Distillation
description: >-
  [ACL 2025][Model Compression][Knowledge Distillation] This paper proposes Basic Reading Distillation (BRD). By having a teacher LLM generate basic reading behavior data (including NER and QA) on general corpora, a small student model is trained to mimic these behaviors. This allows a 564M-parameter small model to reach or exceed the performance of a teacher model 20 times its size across various NLP tasks, without being exposed to downstream task data.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Basic Reading Education"
  - "Named Entity Recognition"
  - "Question Answering"
  - "Small Model Enhancement"
date: 2026-05-08
content_hash: a88154cd8b2d8354
---

# Basic Reading Distillation

**Conference**: ACL 2025  
**arXiv**: [2507.19741](https://arxiv.org/abs/2507.19741)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Basic Reading Education, Named Entity Recognition, Question Answering, Small Model Enhancement

## TL;DR
This paper proposes Basic Reading Distillation (BRD). By having a teacher LLM generate basic reading behavior data (including NER and QA) on general corpora, a small student model is trained to mimic these behaviors. This allows a 564M-parameter small model to reach or exceed the performance of a teacher model 20 times its size across various NLP tasks, without being exposed to downstream task data.

## Background & Motivation
Existing distillation methods are mainly divided into two categories: knowledge distillation (mimicking the implicit internal features of teacher models, such as logits, hidden states, and attention maps) and task distillation (mimicking the output behavior of teacher models on specific tasks). Both approaches overlook a fundamental issue: **the student model lacks basic reading education on general text**.

This paper draws an analogy to human education: an LLM should first receive "high school/university" level reading education to develop basic text comprehension abilities before taking "exams". In contrast, traditional training merely consumes tokens for next token prediction and then directly undergoes testing. The core idea is: **a student model with basic reading education is more effective than one without education**.

Two advantages of BRD: (1) It can extend any general text into basic reading training data, overcoming the lack of data scale and diversity in task distillation; (2) It avoids the lack of interpretability caused by mimicking implicit features in knowledge distillation.

## Method

### Overall Architecture
Two-stage pipeline: (1) Utilize a teacher LLM (Vicuna-13B) to generate basic reading behavior data for each sentence in a general corpus (CC-100); (2) Mix the generated data with the original corpus to train the student model (XGLM-564M). The entire process does not involve any downstream task data.

### Key Designs
1. **Definition of Basic Reading Behaviors**:

    - Two core capabilities: Named Entity Recognition (NER) and Question-Response Association (QRA)
    - NER: Identifies entities and their types (person, organization, location, etc.) in a sentence, with the teacher model also generating additional descriptions.
    - QRA: Formulates questions regarding the content/structure/attitude of the sentence and finds answers within the original text.
    - Few-shot prompting is used to prompt the teacher model to generate these behaviors, where the prompt includes task descriptions, exemplars, and the input sentence.

2. **Training Data Construction**:

    - Training is organized at the paragraph level: sentences are arranged in their original order, with each sentence followed by its NER or QRA annotations.
    - NER paragraph format: $s_1$ \<sep\> NER($s_1$) \<sep\> $s_2$ \<sep\> NER($s_2$) ...
    - QRA paragraph format: $s_1$ \<sep\> QRA($s_1$) \<sep\> $s_2$ \<sep\> QRA($s_2$) ...
    - Original paragraphs are also retained for mixed training to avoid catastrophic forgetting.
    - Three-way data mixing: $D_{ORI}$ (original paragraphs) + $D_{NER}$ (NER paragraphs) + $D_{QRA}$ (QRA paragraphs).

3. **Testing Strategy**:

    - Uses the average per-token log-probability as a scoring function for candidate answers.
    - $\bar{P} = \frac{1}{n} \sum_{i=1}^{n} \log P_i(y_i | x_{\text{prompt}})$
    - Selects the candidate answer with the highest score as the final prediction.
    - This method is suitable for tasks where candidate answers have different lengths.

4. **Orthogonality Design of BRD**:

    - BRD focuses on basic reading capabilities on general texts, without involving any implicit features or specific tasks.
    - Therefore, it is orthogonal to knowledge distillation and task distillation and can be combined with them.

### Loss & Training
Standard auto-regressive language model loss: $L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(y_t | y_{<t})$

Using 5 million sentences from the CC-100 corpus, paragraph-level training is conducted, mixed with the original corpus. The student model is initialized based on XGLM-564M.

## Key Experimental Results

### Main Results - Without Downstream Task Supervision (Blind Test)

| Model | XNLI | RTE | CB | PAWS-X | BOOLQ | SST-2 | BIG-bench | Avg. |
|------|------|-----|-----|--------|-------|-------|-----------|------|
| Vicuna-13B (Teacher) | 59.1 | 78.3 | 71.4 | 62.9 | 84.3 | 81.5 | 35.6 | 67.6 |
| XGLM-7.5B | 36.6 | 50.9 | 60.7 | 56.8 | 57.2 | 69.5 | 34.3 | 52.3 |
| XGLM-564M | 35.5 | 46.2 | 53.6 | 51.3 | 51.2 | 63.9 | 34.0 | 48.0 |
| MiniLLM (KD SOTA) | 34.2 | 58.1 | 73.2 | 44.1 | 55.9 | 62.4 | 34.6 | 51.8 |
| **XGLM-BRD** | **36.2** | 53.8 | 58.9 | **56.7** | **61.0** | **78.1** | **34.8** | **54.2** |

### With Task Input (Relaxed Test)

| Model | XNLI | RTE | CB | PAWS-X | BOOLQ | SST-2 | BIG-bench | Avg. |
|------|------|-----|-----|--------|-------|-------|-----------|------|
| TaskDistillation | 57.1 | 58.1 | 60.7 | 64.8 | 74.8 | 77.2 | 41.6 | 62.0 |
| **XGLM-BRD2** | **59.2** | **62.5** | **82.1** | 64.8 | **75.0** | **81.9** | **44.1** | **67.1** |
| Vicuna-13B | 59.1 | 78.3 | 71.4 | 62.9 | 84.3 | 81.5 | 35.6 | 67.6 |

### Ablation Study

| Configuration | Avg. | Description |
|------|------|------|
| XGLM-BRD2 Full | 70.9 | Baseline (Relaxed Test) |
| -NER | 68.4 | Without NER, performance drops by 2.5% |
| -QRA | 67.8 | Without QRA, performance drops by 3.1%, QRA is more important |
| Paragraph-level training vs. Sentence-level | 57.5 vs 55.6 | Paragraph-level training outperforms sentence-level |
| XGLM-564M-FURTHER | 47.1 | Continuing training only with the original corpus drops performance instead |

### Orthogonality Verification (BRD + Other Distillation Methods)

| Method | Original Avg. | Avg. after +BRD | Gain |
|------|---------|-----------|------|
| SKD (GPT-2 120M) | 51.0 | 58.9 | +7.9 |
| MiniLLM (GPT-2 120M) | 52.6 | 57.8 | +5.2 |
| SKD (GPT-2 760M) | 48.7 | 56.7 | +8.0 |
| TaskDistillation (XGLM-564M) | 65.5 | 68.2 | +2.7 |

### Key Findings
- The 564M XGLM-BRD is close to or even exceeds the 15-times larger XGLM-7.5B in blind tests (54.2 vs. 52.3).
- XGLM-BRD2 outperforms the 26-times larger Vicuna-13B teacher model on some tasks in relaxed tests.
- Cross-entropy analysis confirms that BRD effectively pushes the student model's probability distribution towards the teacher model (cross-entropy decreases for all tasks except PIQA).
- Continuing training solely on the original corpus (XGLM-564M-FURTHER) leads to a performance drop, indicating that the key lies in reading behavior data rather than more training.
- BRD is orthogonal to KD and TD: it brings significant improvements in all combinations.
- QRA contributes more to reading comprehension than NER.
- Performance improves steadily as the volume of BRD data increases, saturating at around 1 million paragraphs.

## Highlights & Insights
- The pedagogical analogy is highly intuitive: models should learn "reading" first before doing "exercises", making the motivation very compelling.
- Orthogonality with knowledge distillation and task distillation is a highly desirable property, allowing BRD to serve as a general enhancement technique.
- Significantly improving the performance of small models on extensive downstream tasks through only two basic tasks, NER and QRA, indicates that "basic reading comprehension" is indeed a fundamental underlying capability.
- The finding that paragraph-level training outperforms sentence-level training is logical, as downstream tasks usually require processing multi-sentence inputs.
- Using a pre-training corpus like CC-100 for BRD implies that the data sources are abundant and diverse.

## Limitations & Future Work
- Only Vicuna-13B is used as the teacher (for efficiency considerations); stronger teachers (e.g., GPT-4) might perform better but at higher costs.
- The student model is only validated on XGLM-564M, lacking validation on other architectures (e.g., small models in the LLaMA family).
- Are NER and QRA the optimal combination of reading behaviors? More behavior types can be explored (such as summarization, sentiment analysis, logical reasoning, etc.).
- Using per-token probability scoring during inference has limitations and cannot be applied to open-ended generation tasks.
- Research idea: BRD can be extended to multi-turn conversational reading (rather than single-turn QRA), which might learn deeper comprehension capabilities.
- The data scale of 5M sentences might be insufficient for larger models, requiring study on the scaling behavior of BRD across different model sizes.

## Related Work & Insights
- Related to but distinct from PICL (Pre-training on In-context Learning): PICL requires 37 downstream tasks to train the retriever, whereas BRD does not rely on downstream tasks at all.
- Comparison with self-supervised learning: BRD can be viewed as a teacher-guided self-supervised enhancement method.
- The approach of mimicking human reading education can be extended to other modalities, such as "basic observation training" for vision models.

## Rating
- Novelty: ⭐⭐⭐⭐ The philosophy of "basic reading education distillation" is novel, though the concrete implementation (NER+QRA) is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete experimental settings (blind/relaxed/supervised tests), orthogonality verification, and ablation studies, though limited to a single student model.
- Writing Quality: ⭐⭐⭐⭐ The educational analogy makes the paper flow naturally with clean logic, although the method description is somewhat verbose.
- Value: ⭐⭐⭐⭐ Provides a low-cost general approach for enhancing small models, with its orthogonality ensuring broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](../../ECCV2024/model_compression/basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)

</div>

<!-- RELATED:END -->
