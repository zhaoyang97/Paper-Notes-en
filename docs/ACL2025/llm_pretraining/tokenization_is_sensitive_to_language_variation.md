---
title: >-
  [Paper Note] Tokenization is Sensitive to Language Variation
description: >-
  [ACL2025][LLM Pretraining][tokenization] This paper systematically investigates the contrasting impacts of three key design choices of BPE tokenizers (fitting corpus, pre-tokenizer, and vocabulary size) on downstream performance across tasks requiring language variation robustness versus sensitivity. It proposes a task-aware tokenizer evaluation metric based on logistic regression, which significantly outperforms task-agnostic metrics such as Rényi efficiency.
tags:
  - "ACL2025"
  - "LLM Pretraining"
  - "tokenization"
  - "BPE"
  - "language variation"
  - "pre-tokenizer"
  - "vocabulary size"
  - "robustness"
  - "sensitivity"
date: 2026-05-08
content_hash: 443af20dc076e93f
---

# Tokenization is Sensitive to Language Variation

**Conference**: ACL2025  
**arXiv**: [2502.15343](https://arxiv.org/abs/2502.15343)  
**Code**: [nlpsoc/Tokenization-Language-Variation](https://github.com/nlpsoc/Tokenization-Language-Variation)  
**Area**: LLM Pre-training  
**Keywords**: tokenization, BPE, language variation, pre-tokenizer, vocabulary size, robustness, sensitivity

## TL;DR

This paper systematically investigates the contrasting impacts of three key design choices of BPE tokenizers (fitting corpus, pre-tokenizer, and vocabulary size) on downstream performance across tasks requiring language variation robustness versus sensitivity. It proposes a task-aware tokenizer evaluation metric based on logistic regression, which significantly outperforms task-agnostic metrics such as Rényi efficiency.

## Background & Motivation

**Background**: Language variations (e.g., spelling variations, dialects, syntactic changes) are ubiquitous in natural language and are systematically associated with geographic, social, and contextual factors. BPE is currently the most widely adopted tokenization algorithm in mainstream LLMs (such as LLaMA 3, GPT-4, and DeepSeek-v3).

**Limitations of Prior Work**: Tokenizers process uncommon linguistic forms inconsistently (e.g., spelling variations like "doin" versus "doing" where the standard form may be a single token while the variant is split into multiple subwords). Such inconsistencies can details downstream LLM performance. However, prior research often operates under the assumption that "there exists a single best tokenizer for all tasks in a given language".

**Key Challenge**: Different types of downstream tasks have contrasting demands regarding language variation. Semantic tasks (e.g., NLI) require **robustness** to variations (British and American spelling should yield the same label), whereas formal tasks (e.g., authorship verification) require **sensitivity** to variations (distinguishing spelling styles provides crucial signals). Whether a single tokenizer configuration can satisfy both sets of demands remains a core conflict.

**Goal**: (RQ1) Can a single tokenizer configuration perform well on both robustness and sensitivity tasks? (RQ2) Can a simple task-aware metric replace task-agnostic metrics to predict downstream tokenizer performance?

**Key Insight**: Multiple BERT-base models are pre-trained systematically by varying three BPE parameters (source corpus, pre-tokenizer, and vocabulary size) to conduct controlled experiments across both task categories, while logistic regression is proposed as a novel evaluation metric.

**Core Idea**: The optimal tokenizer configuration differs depending on whether the downstream task requires robustness or sensitivity to language variation, with the pre-tokenizer identified as the most critical design choice.

## Method

### Overall Architecture

BERT-base (110M parameters) is pre-trained for each tokenizer configuration on 750M tokens, followed by fine-tuning and evaluation on two classes of downstream tasks. Each configuration is trained with three different seeds to ensure statistical significance.

### Key Designs

#### Key Design 1: Three-Dimensional Variations

**Function**: Systematically varies three parameters of the BPE tokenizer.  
**Design Motivation**: To isolate the independent influence of each parameter.  
**Mechanism**:  

- **Source Corpus**: PubMed (biomedical), Wikipedia (encyclopedic), Twitter (social media), and Miscellaneous (mixed multi-source), with approximately 1.5B words each.  
- **Pre-tokenizer** (5 types): None (NO), whitespace-only (WS), keep leading whitespace (_WS), LLaMA 3 style (LLAMA3), and GPT-2 style (GPT2).  
- **Vocabulary Size**: 500, 4k, 32k, 64k, and 128k.  

The default configuration consists of Miscellaneous corpus + GPT2 pre-tokenizer + 32k vocabulary size, with only one parameter varied at a time.

#### Key Design 2: Two Classes of Evaluation Tasks

**Function**: Designs and compiles two categories of task suites.  
**Design Motivation**: To independently evaluate robustness and sensitivity to language variation.  
**Mechanism**:  

- **Robustness Tasks**: GLUE (SST-2, QQP, MNLI, QNLI) + GLUE+typo (typos injected via TextFlint) + GLUE+dialect (dialectal transformations injected via Multi-VALUE, covering 5 English dialects).  
- **Sensitivity Tasks**: AV (Authorship Verification, 40.8k training samples), PAN (multi-author style analysis), CORE (register classification), NUCLE (grammatical error classification), and Dialect (dialect classification).

#### Key Design 3: Logistic Regression Evaluation Metric

**Function**: Proposes a task-aware tokenizer evaluation method.  
**Design Motivation**: Traditional metrics (such as Rényi efficiency and Corpus Token Count) are task-agnostic, yielding identical predictions for different tasks on the same corpus, which diverges from empirical behavior.  
**Mechanism**: A logistic regression model is fitted using the bag-of-tokens from the tokenizer's vocabulary as the feature set and task labels as target variables. For dual-text input tasks, combinations of token pairs between the sentences are utilized as features.

### Loss & Training

- **Pre-training**: BERT-base with 110M parameters, sequence length of 512, batch size of 32, and 45k steps.  
- **Fine-tuning**: 3 epochs, max_seq_len=128, batch_size=32, lr=2e-5.  
- **Significance Testing**: McNemar's test with Bonferroni correction.

## Key Experimental Results

### Main Results: Robustness Task Performance (Table 2)

| Setting | GLUE | +typo | +dialect | AVG |
|------|------|-------|----------|-----|
| **Fitting Corpus** | | | | |
| Twitter | **81.1** | 69.1 | 78.8 | **76.4** |
| PubMed | 80.8 | **69.1** | 78.6 | 76.2 |
| Wikipedia | 80.7 | 68.6 | **79.3** | 76.2 |
| **Pre-tokenizer** | | | | |
| GPT2 | **81.3** | 68.2 | 79.2 | 76.2 |
| _WS | 80.8 | **68.9** | 79.0 | 76.2 |
| NO | 72.1 | 61.6 | 70.1 | 67.9 |
| **Vocabulary Size** | | | | |
| 500 | 77.2 | **70.3** | 75.6 | 74.4 |
| 32k | **81.3** | 68.2 | **79.2** | **76.2** |
| 128k | 78.7 | 64.6 | 76.1 | 73.1 |

### Main Results: Sensitivity Task Performance (Table 3)

| Setting | AV | PAN | CORE | NUCLE | Dialect | AVG |
|------|-----|-----|------|-------|---------|-----|
| **Fitting Corpus** | | | | | | |
| Twitter | **82.9**| **66.7**| **56.5**| 21.4  | 88.3    | **63.2** |
| Wikipedia | 81.9| 65.5| 55.5| **23.5**| **88.9** | 63.1 |
| **Pre-tokenizer** | | | | | | |
| _WS | 82.5| 66.3| **56.6**| 22.6  | 88.4    | **63.3** |
| GPT2 | **82.6**| 66.6| 56.3| 21.8  | 88.4    | 63.1 |
| NO | 81.8| 59.9| 51.7| 16.3  | 77.3    | 57.4 |
| **Vocabulary Size** | | | | | | |
| 32k | 82.6| 66.6| **56.3**| 21.8  | **88.4** | **63.1** |
| 64k | **82.7**| **67.2**| 54.9| **22.0**| 88.1    | 63.0 |
| 500 | 78.2| 62.6| 51.1| 13.1  | 85.6    | 58.1 |

### Ablation Study: Comparison of Tokenizer Evaluation Metrics (Table 5)

| Metric | Correlation with Robustness Tasks | Correlation with Sensitivity Tasks |
|------|-------------------|-------------------|
| Rényi Efficiency | -0.22 | -0.03 |
| Corpus Token Count | -0.45 | +0.37 |
| **Logistic Regression** | **0.85** | **0.84** |

### Key Findings

1. **Pre-tokenizer has the largest impact**: Across both task categories, pre-tokenizer choice exhibits the widest performance variance, and omitting the pre-tokenizer (NO) yields significantly poorer performance than other options.
2. **Optimal configuration varies by task type**: Sensitivity tasks favor larger vocabularies (32k–64k), whereas small vocabularies (500) exhibit the highest robustness to spelling variations in robustness tasks.
3. **Twitter corpus is unexpectedly versatile**: It performs consistently well across both categories of tasks, rather than being confined only to sensitivity tasks.
4. **Reversal of Corpus Token Count correlation direction**: It is negatively correlated with robustness tasks ($-0.45$) but positively correlated with sensitivity tasks ($+0.37$), demonstrating the limitation of task-agnostic metrics.
5. **Logistic regression is highly correlated with downstream performance** ($r = 0.85/0.84$), substantially outperforming traditional intrinsic metrics.

## Highlights & Insights

- **Novel Perspective**: For the first time, downstream tasks are systematically categorized by "robustness versus sensitivity to language variation" to evaluate tokenizers, exposing the limitations of the traditional "one-size-fits-all" assumption.
- **Three Practical Recommendations**: (1) focus primarily on the pre-tokenizer; (2) use larger vocabularies for sensitivity tasks; and (3) employ small-scale logistic regression for rapid task-aware tokenizer evaluation.
- **Low-Cost, High-Value Experimental Design**: Insightful conclusions are drawn using a 110M parameter BERT-base ($<15$ GPU hours per model), which is far more economical than training 350M–2.5B parameter models.
- **Methodological Contribution**: Proposing a task-aware tokenizer evaluation paradigm, breaking away from the traditional practice of evaluating tokenizers solely on compression efficiency.

## Limitations & Future Work

1. **Model scale constraints**: Using only a 110M BERT-base makes it difficult to guarantee that the conclusions generalize to larger decoder-only models (e.g., LLaMA 3, GPT-4).
2. **Confounded language variation types**: The impacts of spelling variations, lexical variations, and syntactic variations are mixed, which warrants disentangled investigations in the future.
3. **Unexplored parameter interactions**: Parameters were varied in isolation without exploring the interaction effects of source corpus $\times$ pre-tokenizer $\times$ vocabulary size.
4. **Linguistic limitations**: This study is restricted to English; different writing systems (e.g., Chinese, Arabic) might exhibit distinct patterns.
5. **Limited scope of downstream tasks**: Robustness evaluations rely solely on GLUE, and the dichotomy of "semantic versus formal" for sensitivity tasks may be overly simplified.
6. **Limitations of logistic regression**: The proposed metric is only applicable to classification tasks and cannot be easily adapted to complex generative tasks.

## Related Work & Insights

### vs. Ali et al. (2024) — Tokenizer Choice for LLM Training

Ali et al. evaluated different tokenizers on 350M–2.5B parameter models and found that under a constant model scale, varying vocabulary sizes has a negligible impact on performance. While this study validates similar findings on a smaller model, the key innovation is the distinction between the two types of tasks—revealing that the optimal vocabulary size diverges for robustness versus sensitivity tasks, thus challenging the assumption of a "universally optimal tokenizer".

### vs. Schmidt et al. (2024) — NLU Evaluation of BPE Tokenizers

Schmidt et al. also identified that pre-tokenizers affect downstream performance but did not consider tasks sensitive to language variations. This paper extends the evaluation scope, revealing inconsistent behaviors of the same pre-tokenizer across both task categories (e.g., GPT2 is optimal on standard GLUE, but _WS behaves more robustly on variation tasks).

### vs. Zouhar et al. (2023) — Rényi Efficiency

Zouhar et al. proposed Rényi efficiency as an intrinsic evaluation metric for tokenizers. This study demonstrates its extremely low correlation with downstream performance ($-0.22/-0.03$) and introduces logistic regression (yielding correlations of $0.85/0.84$) as a far more reliable alternative.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The perspective of partitioning downstream tasks into robustness versus sensitivity to language variation is highly novel, and the logistic regression-based evaluation metric is highly practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic control of variables, statistical significance testing, and extensive comparisons between two task categories are present, though parameter interactions remain unexplored.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The structure is clear, research questions are inquiry-driven, and tables are informative and highly structured.
- **Value**: ⭐⭐⭐⭐ — Offers direct, practical guidance for tokenizer design, serving as an excellent reference for tokenizer selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Adversarial Tokenization](adversarial_tokenization.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)

</div>

<!-- RELATED:END -->
