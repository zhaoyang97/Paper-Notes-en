---
title: >-
  [Paper Note] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Evaluation] Proposes the CIA (Cross Lingual Auto Evaluation) Suite, a cross-lingual LLM evaluation framework including the evaluator model Hercule and the human-annotated test set Recon. By leveraging English reference answers to score non-English LLM responses, the 8B model outperforms closed-source large models like GPT-4o in multilingual evaluation.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Evaluation"
  - "Multilingual LLM"
  - "Evaluator LLM"
  - "Reference-Based"
  - "Weight Merging"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: a51b7d8a4df5a442
---

# Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs

**Conference**: ACL 2025  
**arXiv**: [2410.13394](https://arxiv.org/abs/2410.13394)  
**Code**: [github.com/CIA](https://github.com/CIA) / [huggingface.co/CIA-Suite](https://huggingface.co/CIA-Suite)  
**Area**: LLM NLP / Multilingual Evaluation  
**Keywords**: Cross-Lingual Evaluation, Multilingual LLM, Evaluator LLM, Reference-Based, Weight Merging, Low-Resource Languages

## TL;DR

Proposes the CIA (Cross Lingual Auto Evaluation) Suite, a cross-lingual LLM evaluation framework including the evaluator model Hercule and the human-annotated test set Recon. By leveraging English reference answers to score non-English LLM responses, the 8B model outperforms closed-source large models like GPT-4o in multilingual evaluation.

## Background & Motivation

**Background**: Evaluating machine-generated text in NLP remains a core challenge. Existing evaluation methods—automatic metrics (BLEU/ROUGE), human evaluation, and LLM evaluation—mainly focus on English, leaving a severe lack of multilingual evaluation frameworks.

**Limitations of Prior Work**:
   - **Missing Multilingual Benchmarks**: Lack of multilingual evaluation benchmarks covering complex open-ended tasks; existing multilingual benchmarks are mostly limited to classification and short sentence generation.
   - **Unreliable Human Evaluation**: Non-expert evaluators rely on "vibe checks," which are influenced by personal biases.
   - **Scarcity of Reference Answers**: Reference answers in non-English languages are extremely scarce, whereas English reference answers are abundant.
   - **Inconsistent GPT-4 Multilingual Evaluation**: Prior work shows that GPT-4 is inconsistent as a multilingual evaluator.

**Core Observations**:
   - **𝒜** There is an urgent need to develop robust multilingual benchmarks.
   - **ℬ** Trained evaluators significantly outperform untrained ones, rivaling closed-source models.
   - **𝒞** Reference-based methods are more accurate and reliable than reference-free approaches.
   - **D** English reference answers can be utilized to cross-lingually evaluate non-English responses.

**Core Idea**: Constructing a cross-lingual evaluation LLM—where instructions and responses are in the target language, while reference answers, evaluation criteria, and rubrics remain in English, and the model generates feedback and scores in English.

## Method

### Overall Architecture

The CIA Suite consists of three core components: (1) Recon test set—a human-annotated multilingual evaluation benchmark; (2) Intel training set—automatically translated cross-lingual training data; (3) Hercule evaluator model—a cross-lingual evaluation LLM fine-tuned based on Llama-3.1-8B.

### Recon Test Set

- **Scale and Languages**: 500 human-generated instructions, covering 6 languages—Bengali (bn), German (de), French (fr), Hindi (hi), Telugu (te), and Urdu (ur).
- **Data Sources**: 250 instructions from BigGenBench (planning, instruction-following, reasoning, etc.), and 250 instructions from UltraEval, WizardLM, LIMA, MT-Bench, and FBI (long-form writing, creative generation, factual QA, etc.).
- **Data Format**: Each instance is a 5-tuple $(P^X, C^{En}, R_{eval}^X, R_{ref}^{En}, s)$, containing instructions and evaluated responses in the target language, evaluation criteria and reference answers in English, and human-annotated ground-truth scores.
- **Construction Workflow**:
    - Evaluation Criteria: GPT-4o generates question-specific criteria and a 1-5 scoring rubric, with 3 hand-written examples as in-context demos.
    - Reference Answers: GPT-4o generates 5-point reference answers based on the criteria.
    - Evaluated Responses: GPT-4o generates responses of corresponding quality for different score levels to ensure a uniform distribution of scores.
    - Translation: Instructions are manually translated by professional translators for each language; evaluated responses are translated by GPT-4o and then human-verified.

### Intel Training Set

- Based on the Feedback-Collection dataset, translating instructions and responses into target languages while keeping the rest in English.
- GPT-4o is used for automatic translation, obtaining approximately 100k training and 1000 validation samples per language.
- Translation Quality Check: 100 samples per language are evaluated by humans, showing that invalid translations are below 5%.

### Hercule Evaluator Model

- **Base Model**: Llama-3.1-8B-Instruct.
- **Training Objective**: Absolute scoring—given instructions $P^X$ and responses $R_{eval}^X$ in the target language, English evaluation criteria $C^{En}$ and reference answers $R_{ref}^{En}$, the model generates English feedback $F^{En}$ and score $s$ (1-5 points).
- **Training Strategy**: Generate evaluation details first, then output the score (Chain-of-Thought style evaluation).
- **Training Details**: Sequence length 4096, FlashAttention 2, AdamW optimizer, learning rate 1e-5, 3 epochs, 8 $\times$ H100 GPUs.

### Evaluation Metrics

Linearly weighted Cohen's Kappa ($\kappa$) is adopted to measure the consistency between the evaluator LLM and human ground-truth scores, where a $\kappa$ close to 1 indicates a strong correlation.

## Experiments

### Main Results (Table 1)

| Model | Type | bn | de | fr | hi | te | ur | avg |
|------|------|------|------|------|------|------|------|------|
| GPT-4o | Zero-Shot | 0.64 | 0.66 | 0.65 | 0.64 | 0.61 | 0.64 | 0.64 |
| Gemini-1.5-Pro | Zero-Shot | 0.54 | 0.58 | 0.59 | 0.57 | 0.53 | 0.57 | 0.56 |
| Llama-3.1-405B | Zero-Shot | 0.60 | 0.66 | 0.66 | 0.62 | 0.51 | 0.65 | 0.62 |
| Hercule 8B | FFT | **0.74** | **0.75** | **0.75** | **0.74** | **0.69** | **0.74** | **0.73** |
| Hercule 8B | LoRA | 0.72 | 0.74 | 0.72 | 0.72 | 0.70 | 0.70 | 0.72 |

**Key Findings**: Hercule 8B significantly outperforms GPT-4o (0.64) and Llama-3.1-405B (0.62) with an average $\kappa$ of 0.73. Even when the base model exhibits poor tokenizer fertility for some languages, fine-tuning remains highly effective.

### Human Evaluation Comparison (Table 2)

In evaluations of real LLM outputs across 100 samples in 4 low-resource languages (bn/hi/te/ur), Hercule 8B achieves the highest Pearson correlation with human evaluation, significantly outperforming GPT-4o, especially on te (0.74) and ur (0.78).

### Ablation Study

1. **Cross-Lingual Zero-Shot Transfer (Table 3)**: Models trained on language X can effectively evaluate other languages, performing significantly better than direct zero-shot evaluation after training strictly on English.
2. **Importance of Reference Answers (Table 4)**: Removing reference answers leads to a performance drop of approximately 7 percentage points ($0.73 \rightarrow 0.66$), confirming that reference answers are crucial for accurate evaluation.
3. **LoRA vs FFT**: LoRA training achieves comparable performance to Full Fine-Tuning (0.72 vs 0.73), serving as a viable alternative in resource-constrained scenarios.
4. **Weight Merging (Table 6)**: Linearly merging 6 language-specific models into a unified model yields comparable performance on high-resource languages to individual training, outperforming joint training.

### Qualitative Analysis

LLM evaluators tend to show generosity bias (giving higher scores). In cases with a score discrepancy $\ge 2$, the model relies on parametric knowledge rather than reference answers—reasoning correctly in high-resource languages (de/fr) but failing in low-resource languages. Approximately 5% of Bengali and 20% of Telugu samples exceed the 4096 token limit.

## Highlights & Insights

1. **Innovative Cross-Lingual Evaluation Paradigm**: For the first time, an "English Reference + Target Language Response" cross-lingual evaluation scheme is systematically proposed and validated, circumventing the high cost of creating reference answers for every language.
2. **Small Models Outperforming Large Models**: The fine-tuned 8B model surpasses 405B and closed-source models in multilingual evaluation, indicating that task-specific training is more important than model scale in evaluation tasks.
3. **Zero-Shot Cross-Lingual Transfer**: Evaluators trained on one language can transfer to unseen languages, providing a viable path for the evaluation of low-resource languages.
4. **Engineering Standardization**: The code, datasets, and models are fully open-sourced, forming the CIA Suite evaluation toolchain.

## Limitations & Future Work

1. Restricted by translation costs, the framework only covers 6 languages, so its generalizability remains to be verified.
2. The multilingual models available for testing are limited.
3. Weight merging techniques have not been fully explored regarding different configurations (e.g., balancing language contribution weights).
4. Tokenizer fertility issues lead to sample truncation in some languages.

## Related Work & Insights

- **LLM Evaluators**: Training evaluator LLMs like Prometheus and AlpacaFarm; multi-agent evaluation like ChatEval.
- **Multilingual Evaluation**: Multilingual benchmarks such as XTREME/XNLI, which are limited to classification tasks; FBI reveals the inconsistencies of GPT-4 in multilingual evaluation.
- **Model Merging**: Linear merging, TIES, and other techniques used to create unified multi-task models.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The first systematic cross-lingual evaluation framework, providing end-to-end integration from data to model to evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rich ablation studies—including zero-shot transfer, reference answer ablation, LoRA/FFT, weight merging, and human evaluation.
- **Value**: ⭐⭐⭐⭐ The entire suite of artifacts is publicly available and can be directly used for multilingual development iterations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, presenting a complete chain of derivation from observations to conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)

</div>

<!-- RELATED:END -->
