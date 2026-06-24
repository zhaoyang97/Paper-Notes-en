---
title: >-
  [Paper Note] DeepRTL2: A Versatile Model for RTL-Related Tasks
description: >-
  [ACL 2025][RTL code] DeepRTL2 is the first LLM to unify the processing of both RTL (Register-Transfer Level) generation and embedding tasks. Through a meticulously constructed dataset and the GRIT training strategy, it achieves SOTA performance across five major tasks: code generation, code understanding, natural language code search, functional equivalence checking, and performance prediction.
tags:
  - "ACL 2025"
  - "RTL code"
  - "LLM"
  - "embedding tasks"
  - "code generation"
  - "GRIT training"
date: 2026-05-08
content_hash: f55e917af3ac6889
---

# DeepRTL2: A Versatile Model for RTL-Related Tasks

**Conference**: ACL 2025  
**arXiv**: [2506.15697](https://arxiv.org/abs/2506.15697)  
**Code**: None (dataset and benchmark may be open-sourced)  
**Area**: Electronic Design Automation (EDA) / Code Generation & Understanding  
**Keywords**: RTL code, LLM, embedding tasks, code generation, GRIT training

## TL;DR

DeepRTL2 is the first LLM to unify the processing of both RTL (Register-Transfer Level) generation and embedding tasks. Through a meticulously constructed dataset and the GRIT training strategy, it achieves SOTA performance across five major tasks: code generation, code understanding, natural language code search, functional equivalence checking, and performance prediction.

## Background & Motivation

In the EDA (Electronic Design Automation) domain, LLMs have demonstrated breakthrough capabilities in generative tasks such as RTL code generation and understanding. However, equally critical **embedding tasks** have been largely neglected. These tasks include:

**Natural language code search**: Designers query large RTL codebases using natural language to quickly identify reusable modules.

**Functional equivalence checking**: Rapidly evaluating whether two designs are functionally equivalent, which significantly reduces verification time.

**Performance prediction**: Estimating Power, Performance, and Area (PPA) at the early RTL stage to guide optimization.

Previous methodologies either focused on generative tasks (such as CodeV and DeepRTL) or employed design-specific machine learning approaches for verification and prediction, lacking general RTL representation capabilities. The core motivation of DeepRTL2 is: **a single model solving both generative and embedding tasks simultaneously** to provide a comprehensive solution in the EDA domain.

## Method

### Overall Architecture

DeepRTL2 adopts a decoder-only architecture (based on Llama-3.1 8B and DeepSeek-Coder 6.7B), utilizing a two-stage training strategy to learn generative and embedding capabilities concurrently. The generative side handles code generation and understanding, while the embedding side manages search, equivalence checking, and performance prediction.

### Key Designs

1. **Comprehensive Dataset Construction**:

    - **Code Generation/Understanding**: Verilog files were collected from GitHub $\rightarrow$ split into modules $\rightarrow$ deduplicated using MinHash $\rightarrow$ syntax-checked $\rightarrow$ annotated with GPT-4o Chain-of-Thought (line-level comments $\rightarrow$ module-level specifications $\rightarrow$ high-level functional descriptions). Open-source datasets such as RTLCoder, MG-Verilog, and DeepCircuitX were also integrated.
    - **Natural Language Code Search**: Reusing the understanding dataset, GPT-4o was employed to rewrite functional descriptions into user query formats (removing identifiers, retaining core logic, ensuring clarity and conciseness).
    - **Functional Equivalence Checking**: An innovative feedback-driven CoT strategy was designed—GPT-4o modified the internal logic of Verilog modules $\rightarrow$ Yosys was used for logical equivalence checking $\rightarrow$ iteration was performed for 2-3 rounds based on feedback $\rightarrow$ equivalent/non-equivalent design pairs were generated. This expanded 50 designs from RTLLM v2.0 into 400 pairs.
    - **Performance Prediction**: High-level synthesis was performed using Yosys and the SkyWater 130nm process library, and the ABC tool was used to extract delay and area metrics.

2. **Two-Stage Training Strategy**:

    - **Stage 1**: Curriculum learning (line-level data $\rightarrow$ module-level specifications $\rightarrow$ high-level descriptions $\rightarrow$ diverse prompts), training only generation/understanding tasks.
    - **Stage 2**: Joint training of generative and embedding tasks using the GRIT framework. Generative tasks utilized high-quality data from the fourth sub-stage of Stage 1, while embedding tasks utilized contrastive learning, starting without hard negatives and subsequently incorporating hard negatives.

3. **GRIT Training Framework Adaptation**: The GRIT approach, originally designed for general NLP, was adapted to the RTL domain. The core idea is to enable a decoder-only model to possess both generative capabilities (autoregressive prediction) and encoding capabilities (fixed-length representation vectors) via a multi-task learning objective function.

4. **Annotation Quality Assurance for Code Understanding**: A benchmark expanded to 500 cases was annotated by professional hardware designers. The CoT annotation strategy of GPT-4o proved more accurate than direct annotation, allowing DeepRTL2 trained on CoT data to surpass GPT-4o itself in understanding tasks.

### Loss & Training

- Generative Tasks: Standard autoregressive language modeling loss.
- Embedding Tasks: Contrastive learning loss, including hard negative mining.
- Embedding Extraction: Cosine similarity is computed between representation vectors.
- Performance Prediction: Inference via a regression model (XGBoost) trained on the extracted embeddings.

## Key Experimental Results

### RTL Code Generation (Table 1, pass@k)

| Model | syntax pass@1 | syntax pass@5 | function pass@1 | function pass@5 |
|------|-------------|-------------|----------------|----------------|
| GPT-4o | 72.00% | 77.31% | 49.70% | 56.80% |
| o1-preview | 76.20% | 83.71% | 50.00% | 60.86% |
| DeepRTL2 (Llama) | 68.30% | **81.31%** | 33.70% | 49.57% |
| DeepRTL2 (DeepSeek) | **71.60%** | 80.58% | 38.50% | **52.62%** |
| Llama-3.1 base | 32.40% | 57.01% | 14.60% | 26.04% |

### Embedding Task Performance (Tables 2, 4, 5)

**Natural Language Code Search (F1):**

| Model | F1 |
|------|-----|
| text-embedding-3-large | 0.290 |
| GritLM-7B | 0.269 |
| DeepRTL2 (Llama) no-hard | **0.476** |
| DeepRTL2 (DeepSeek) | 0.453 |

**Functional Equivalence Checking (Average Precision):**

| Model | AP |
|------|------|
| text-embedding-3-small | 0.565 |
| GritLM-7B | 0.541 |
| DeepRTL2 (Llama) | **0.667** |

**Performance Prediction (Area, r2_score):**

| Model | r2_score | MAPE |
|------|---------|------|
| text-embedding-3-large | 0.699 | 4.446 |
| DeepRTL2 (DeepSeek) | **0.773** | **1.598** |

### Key Findings

1. **Generative and Embedding Unity is Mutually Beneficial**: DeepRTL2 achieves open-source SOTA on generative tasks (surpassing CodeV and the original DeepRTL) while significantly outperforming general-purpose embedding models on embedding tasks.
2. **Domain-Specific Embeddings Greatly Outperform General Models**: On the code search task, the F1 score of DeepRTL2 (0.476) is 64% higher than that of OpenAI's text-embedding-3-large (0.290).
3. **CoT-Annotated Student Surpasses the Teacher**: In code understanding tasks, DeepRTL2 outperforms GPT-4o (the source of its annotation data), due to the use of more granular CoT annotations in the training data.
4. **Contributions of Curriculum Learning and Data Diversity**: The drastic improvement from the base model to DeepRTL2 validates the effectiveness of the data construction and training strategies.

## Highlights & Insights

- **Pioneering Nature**: It is the first model to unify RTL generation and embedding tasks, addressing a critical gap in embedding tasks within the EDA domain.
- **Emphasis on Data Engineering**: The feedback-driven CoT strategy for generating equivalent/non-equivalent code pairs is an elegant design, utilizing Yosys as an in-the-loop verifier to ensure data quality.
- **Practical Value**: Natural language search accelerates code reuse, equivalence checking reduces verification overhead, and performance prediction guides early-stage optimization—these three tasks directly address core pain points in hardware design.
- **Surpassing the Annotator**: The CoT-enhanced training data allows the model to outperform GPT-4o on understanding tasks, a notable instance of the "student surpassing the teacher".

## Limitations & Future Work

- Support is limited to the Verilog language, excluding other HDLs such as VHDL or SystemVerilog.
- Potential overlap may exist between the generation benchmarks and the training data of OpenAI models, making fair evaluation a challenge.
- The AP for functional equivalence checking is only 0.667, whereas practical deployment demands higher precision.
- Performance prediction only estimates area and delay, neglecting power metrics (which require workload information).
- Deployment efficiency may be constrained by embedding dimensions and model size.

## Related Work & Insights

- **DeepRTL (Liu et al., 2025)**: The predecessor to this work, which only handled generative tasks. DeepRTL2 expands this to the embedding domain.
- **GRIT (Muennighoff et al., 2025)**: A training strategy unifying generative and representation learning, adapted in this paper from general NLP to the RTL domain.
- **CodeV (Zhao et al., 2024)**: A competing model for RTL code generation, which DeepRTL2 outperforms on most metrics.
- **Key Difference Between RTL and General Code Generation**: Verilog possesses more structured semantics, enabling automated functional verification using EDA tools.

## Rating

- **Novelty**: 8/10 — The first model to unify RTL generation and embedding, with benchmark construction for embedding tasks serving as an additional contribution.
- **Experimental Thoroughness**: 8/10 — Covers five major tasks, multiple baselines, and diverse evaluation metrics, though it lacks ablation studies on the specific training strategies.
- **Writing Quality**: 7/10 — Structured clearly but highly dense; certain details require referencing the appendix.
- **Value**: 8/10 — Directly advances the EDA domain, providing a unified framework for RTL-related AI applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] Your Model is Overconfident, and Other Lies We Tell Ourselves](your_model_is_overconfident_and_other_lies_we_tell_ourselves.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/others/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)

</div>

<!-- RELATED:END -->
