---
title: >-
  [Paper Note] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following
description: >-
  [ACL 2025][LLM (Other)][multi-document] This paper proposes the MDCure framework, which automatically constructs high-quality multi-document instruction data via a two-stage pipeline (generation and filtering). It trains MDCureRM, a multi-objective reward model, for data filtering. Fine-tuning LLMs (up to 70B) with this data yields performance improvements of up to 75.1% over baselines on multi-document and long-context tasks, demonstrating strong cross-task and cross-domain…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "multi-document"
  - "instruction tuning"
  - "synthetic data"
  - "reward model"
  - "data filtering"
  - "long-context"
date: 2026-05-08
content_hash: 46726f178d35dc79
---

# MDCure: A Scalable Pipeline for Multi-Document Instruction-Following

## Basic Information

**Conference**: ACL 2025  
**Code**: [yale-nlp/MDCure](https://github.com/yale-nlp/MDCure)  
**Institution**: Yale University / Google Research  
**Area**: Multi-Document Processing / Instruction Tuning / Data Synthesis  
**Keywords**: multi-document, instruction tuning, synthetic data, reward model, data filtering, long-context  

## TL;DR

This paper proposes the MDCure framework, which automatically constructs high-quality multi-document instruction data via a two-stage pipeline (generation and filtering). It trains MDCureRM, a multi-objective reward model, for data filtering. Fine-tuning LLMs (up to 70B) with this data yields performance improvements of up to 75.1% over baselines on multi-document and long-context tasks, demonstrating strong cross-task and cross-domain generalization capabilities.

## Background & Motivation

- **Importance of Multi-Document Processing**: Scientific, financial, educational, and news domains require summarization, question answering, and reasoning capabilities across multiple documents.
- **Limitations of LLMs**: Although modern LLMs can handle inputs of hundreds of thousands of tokens, they still face unique challenges in multi-document understanding and reasoning:
    - Cross-document information aggregation
    - Resolving conflicting information
    - Filtering redundant information
    - Bridging information gaps
    - Synthesizing coherent narratives
- **Limitations of Prior Work**:
    - Pre-training methods (e.g., PRIMERA, QAMDen) require massive pre-training data and cannot scale to broader tasks.
    - Human annotation is costly and limited in scope.
    - Existing synthetic data generation methods focus primarily on single documents or support only QA tasks.
- **Goal**: To build the first systematic multi-document instruction data generation framework to enhance LLMs' multi-document capabilities without pre-training.

## Method

### Overall Architecture: The Two-stage MDCure Pipeline

#### Phase 1: Generation

- **Input**: A cluster of related documents
- **Method**: Employs carefully designed zero-shot prompt templates to generate cross-document instruction-response pairs.
- **Template Design Principles**:
    - Requires the response to synthesize information across multiple documents.
    - Diversifics templates to cover various task formats (from single-word answers to detailed summaries).
    - Encourages cross-document reasoning to strengthen cross-document understanding.
- **Document Source**: Thematically related news clusters based on the NewSHead dataset.
- **Generator Model**: GPT-3.5-Turbo (balancing quality and cost), also compatible with open-source LLaMA3.1-70B.

#### Phase 2: Filtering

Trains **MDCureRM**—a multi-objective, multi-document-specific reward model to evaluate and filter the generated instruction data.

**MDCureRM's Six-Dimensional Scoring Aspect**:
1. Instruction quality
2. Response quality
3. Factuality
4. Multi-document relevance
5. Cross-document reasoning requirement
6. Sample diversity

**Training Data**:
- Uses GPT-4o-mini and Mistral-7B to generate multi-document instruction data of varying quality (approx. 20,000 samples).
- Uses GPT-4o to score each sample based on the six-dimensional criteria.
- Target scores are normalized to $[0, 1]$.

**Model Architecture**:
- Based on Llama3-8B, initialized from a Bradley-Terry reward model.
- Replaces the output layer with a 6-dimensional linear regression layer.
- Trained using MSE loss while freezing the backbone model.
- During inference, it outputs a 6-element score, computes a weighted average, and selects the top-N samples.

### MDCureRM + PPO

MDCureRM can be seamlessly integrated into PPO policy optimization:
- Trains a customized multi-document instruction generator using the reward signals from MDCureRM.
- Enables small open-source models (e.g., LLaMA3.1-8B-Instruct) to generate multi-document instruction data of a quality exceeding that of GPT-level models.
- Eliminates the need for subsequent data filtering.

## Experiments

### Experimental Setup

**Fine-tuning Models**:
- FlanT5-Base (250M) & Large (750M)
- Qwen2-Instruct 1.5B & 7B
- LLaMA3.1-Instruct 8B & 70B

**Data Scale**: 12K, 36K, 72K (optimal is 72K)

**Baselines**:
- Pre-training methods: PRIMERA, QAMDen
- Long-context LLMs: LongAlign-7B, ProLong-8B-64k
- General LLMs: GPT-4o, Gemini 1.5 Pro

**Evaluation Benchmarks (6)**:
- Multi-document: SEAM (including MultiNews, OpenAsp, MuSiQue, ECB+, SciCo), WikiHop, HotpotQA, Multi-XScience, QMDSCNN
- Long-context: ZeroScrolls

### Main Results (Selected from Table 1)

| Model | HQA | WikiHop | Multi-XSci | QMDSCNN | SEAM | ZeroScrolls | Avg |
|------|-----|---------|-----------|---------|------|-------------|-----|
| **FlanT5-Base** | | | | | | | |
| Untuned | 4.4 | 45.1 | 38.7 | 48.0 | 1.7 | 13.1 | 14.5 |
| +MDCure | **47.3** | **48.3** | **93.8** | **57.3** | **2.1** | **22.6** | **25.4** |
| **Qwen2-7B** | | | | | | | |
| Untuned | 30.5 | 39.6 | 95.6 | 79.3 | 7.4 | 23.9 | 27.4 |
| +MDCure | **44.7** | **46.0** | **95.1** | **87.3** | **10.3** | **29.8** | **32.7** |
| **LLaMA3.1-8B** | | | | | | | |
| Untuned | 35.5 | 27.1 | 95.1 | 65.3 | 10.2 | 18.7 | 24.3 |
| +MDCure | **44.7** | **43.7** | **95.3** | **93.8** | **11.9** | **30.9** | **34.0** |
| **LLaMA3.1-70B** | | | | | | | |
| Untuned | 53.9 | 38.1 | 95.1 | 88.2 | 13.0 | 36.4 | 37.1 |
| +MDCure | **58.4** | **45.5** | **95.1** | **88.7** | **13.3** | **37.7** | **38.5** |

### Key Findings

1. **Consistent Effectiveness Across Models**: MDCure yields significant improvements across all model families and sizes.
2. **Astonishing Performance Gain**: FlanT5-Base improves by an average of **75.1%** and LLaMA3.1-8B by **40.2%**.
3. **Diminishing Gains with Larger Models**: The 70B model improves by only 3.8%, indicating that larger models already possess strong inherent capabilities.

### Importance of MDCureRM Filtering

| Filtering Method | FlanT5-Base Avg | Qwen2-7B Avg | LLaMA3.1-8B Avg |
|---------|----------------|-------------|----------------|
| No Filtering | 23.2 | 29.9 | 31.1 |
| GPT-3.5 Filtering | 24.1 | 31.4 | 32.1 |
| **MDCureRM** | **25.4** | **32.7** | **34.0** |

MDCureRM outperforms GPT-3.5-as-a-judge filtering across all settings.

### Cross-Task and Cross-Domain Generalization

- MDCure not only enhances training-domain multi-document tasks but also improves out-of-distribution (OOD) tasks such as multi-document coreference resolution, multi-document classification, and text reranking.
- It generalizes across domains to scientific, literary, and media domains that are absent from the training data.
- Performance on single-document long-context tasks (ZeroScrolls) is also improved.

### Compatibility Experiments

- **Combination with ProLong**: Further improves performance on an already strong long-context model (Avg $32.1 \rightarrow 34.9$).
- **Open-source Generator**: LLaMA3.1-70B used as a generator achieves performance comparable to GPT-3.5.
- **PPO Training**: Training LLaMA3.1-8B-Instruct as a generator using MDCureRM reward signals produces synthetic data quality that surpasses closed-source models.

## Highlights & Insights

1. **First Multi-Document Instruction Data Generation Framework**: Fills the gap in multi-document post-training data, presenting a significant methodological contribution.
2. **Dual Value of MDCureRM**: Serves both as a data filter and as a reward signal for PPO, enabling open-source models to generate high-quality data autonomously.
3. **High Practicality**: Compatible with both open-source and closed-source models, with a simple and scalable generation workflow.
4. **Strong Generalization**: Generalizes from news training data to diverse domains such as science and literature, outperforming domain-specific pre-training methods.
5. **Complementarity**: MDCure data is complementary to general instruction data such as FLAN and can be used in combination.

## Limitations & Future Work

1. **Single-Domain Training Data**: Primarily uses news-domain documents, which may limit capabilities in certain highly specific domains.
2. **Reliance on Document Clusters**: Requires pre-assembled clusters of related documents; applicability to arbitrary document sets needs further verification.
3. **Evaluation Limitations**: Some evaluations rely on LLM-as-a-judge, which may introduce bias.
4. **Generation Cost**: Although cheaper than pre-training, generating and filtering 72K samples still incurs considerable API costs.
5. **Diminishing Returns at Scale**: The improvement on 70B models is only 3.8%, indicating limited marginal returns for extremely large models.

## Related Work & Insights

- **Multi-Document Modeling**: PRIMERA (Xiao et al., 2022), QAMDen (Caciularu et al., 2023), Longformer (Beltagy et al., 2020)
- **Synthetic Data Generation**: Self-Instruct (Wang et al., 2023), Alpaca (Taori et al., 2023)
- **Reward Models**: Bradley-Terry RM, multi-objective RM (Wu et al., 2023; Wang et al., 2024)
- **Long-Context LLMs**: LongAlign (Bai et al., 2024), ProLong (Gao et al., 2024)

## Rating

⭐⭐⭐⭐⭐ (4.5/5)

- **Novelty**: First systematic multi-document instruction data generation framework, filling an important gap (+1)
- **Experimental Thoroughness**: $6 \text{ benchmarks} \times 6 \text{ model families} \times 3 \text{ data scales} \times \text{multiple filtering strategies}$ (+1)
- **Practical Value**: Open-sourced framework and datasets, compatible with various models, ready for immediate use (+0.5)
- **Methodology Design**: Clear two-stage process; the multi-objective design of MDCureRM is rational and effective (+0.5)
- **Deductions**: Single training domain, limited improvements on ultra-large models (-0.5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [\[ACL 2025\] T5Score: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets](t5score_a_methodology_for_automatically_assessing_the_quality_of_llm_generated_m.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](../../ACL2026/llm_nlp/muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2025\] ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)
- [\[ACL 2025\] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)

</div>

<!-- RELATED:END -->
