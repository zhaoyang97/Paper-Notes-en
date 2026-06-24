---
title: >-
  [Paper Note] RePanda: Pandas-powered Tabular Verification and Reasoning
description: >-
  [ACL 2025][Tabular Fact Verification] RePanda is proposed to translate natural language claims into executable pandas queries for tabular fact verification, achieving an accuracy of 84.09% on TabFact and 84.72% on OOD WikiFact without additional fine-tuning. Meanwhile, with only a 7B parameter model, it approaches the zero-shot performance of the 671B DeepSeek-Chat, and scales to tabular question-answering tasks, achieving 75.1% accuracy.
tags:
  - "ACL 2025"
  - "Tabular Fact Verification"
  - "pandas query"
  - "interpretable reasoning"
  - "knowledge distillation"
  - "execution-based reasoning"
  - "OOD generalization"
date: 2026-05-08
content_hash: d6f027dc0517a775
---

# RePanda: Pandas-powered Tabular Verification and Reasoning

**Conference**: ACL 2025  
**arXiv**: [2503.11921](https://arxiv.org/abs/2503.11921)  
**Code**: [datasets/AtoosaChegini/PanTabFact](https://huggingface.co/datasets/AtoosaChegini/PanTabFact)  
**Authors**: Atoosa Malemir Chegini, Keivan Rezaei, Hamid Eghbalzadeh, Soheil Feizi  
**Institutions**: University of Maryland, AI at Meta  
**Area**: Others  
**Keywords**: Tabular Fact Verification, pandas query, interpretable reasoning, knowledge distillation, execution-based reasoning, OOD generalization

## TL;DR

RePanda is proposed to translate natural language claims into executable pandas queries for tabular fact verification, achieving an accuracy of 84.09% on TabFact and 84.72% on OOD WikiFact without additional fine-tuning. Meanwhile, with only a 7B parameter model, it approaches the zero-shot performance of the 671B DeepSeek-Chat, and scales to tabular question-answering tasks, achieving 75.1% accuracy.

## Background & Motivation

**Background**: Tabular fact verification requires models to perform numerical reasoning, filtering, and comparison on structured data. Existing methods such as TAPAS, TAPEX, and PASTA enhance tabular understanding through table-aware pre-training, but suffer from two core problems.

**Core Problem**:
   - **Insufficient Structural Understanding**: LLMs are pre-trained sequentially. Flattening tables into sequences loses structural relationships between rows and columns. While TAPAS/TAPEX introduce tabular positional encodings, they still struggle with complex operations like aggregation and multi-row comparison.
   - **Lack of Interpretability**: Existing methods act as black-box classifiers, outputting only True/False without displaying the reasoning process. This makes it impossible to verify the basis of decisions in high-risk scenarios such as legal auditing and finance.

**Key Insight**: pandas queries are inherently designed for tabular operations (filtering, counting, aggregation, etc.). Translating claims into pandas queries not only provides transparent reasoning steps but also yields verifiable results through execution.

## Method

### Overall Architecture

The core idea of RePanda is to reformulate fact verification as a **structured representation learning task**—the model learns to generate executable pandas queries instead of direct classification.

Formal definition: Given a table $\mathcal{T}$ and a claim $s$, the model $f_\theta$ outputs a pandas query $q_s = f_\theta(s, \mathcal{T})$. Executing $q_s$ on $\mathcal{T}$ yields the verification result.

### Dataset Construction

**PanTabFact (Fact Verification Dataset)**:
- Constructed based on the TabFact training set.
- Uses DeepSeek-Chat (671B) to generate corresponding pandas queries for each claim-table pair.
- Each query encodes logical operation steps (filtering, aggregation, comparison, etc.).

**PanWiki (Question Answering Dataset)**:
- Constructed based on WikiTableQuestions.
- Translates each question into a pandas query, which retrieves the answer upon execution.
- Contains only **1200** training samples.

### Error Correction Pipeline

A three-stage automatic correction pipeline ensures data quality:

1. **Logical Correction**: Verifies if the pandas query execution result matches the expected answer. If not, the original query and the expected result are fed back to the model for regeneration (used only during training data creation).
2. **Syntactic Correction**: Iteratively fixes runtime errors by feeding error messages back to the model for correction (used during both training and inference).
3. **Filtering**: Removes samples where execution fails or results do not match.

### Loss & Training

- **Base Model**: DeepSeek-coder-7B-instruct-v1.5
- **Training Method**: Autoregressive generation of pandas queries
- **Loss Function**: Negative log-likelihood
  $$\mathcal{L} = -\sum_{t=1}^{T} \log P(q_t | q_{<t}, s, \mathcal{T}; \theta)$$
- **Training Details**: AdamW optimizer, learning rate 2e-4, cosine scheduler, 4 epochs, batch size 4.
- Fact verification queries output boolean values, while QA queries output specific answers.

### OOD Generalization Design

- Constructs WikiFact: converts QA pairs from WikiTableQuestions into fact verification claims.
- Evaluates the model trained on PanTabFact directly without additional fine-tuning.
- Tests the ability to transfer across tabular structures and domains.

## Experiments

### Main Results

**In-Distribution (TabFact Test Set, Table 1)**:

| Method | Accuracy |
|------|--------|
| RePanda (Fact Verification) | **84.09%** |
| Finetuned-Direct | 67.85% |
| ZeroShot-Pandas | 51.82% |
| ZeroShot-Direct | 50.76% |

RePanda outperforms direct classification fine-tuning by **16.24%** and zero-shot by **33.33%**.

**OOD Generalization (WikiFact, Table 2)**:

| Method | Accuracy |
|------|--------|
| RePanda (Fact Verification) | **84.72%** |
| Finetuned-Direct | 74.10% |
| ZeroShot-Pandas | 59.92% |
| ZeroShot-Direct | 53.20% |

Without additional fine-tuning, the OOD accuracy (84.72%) is even slightly higher than the in-distribution accuracy (84.09%).

### Comparison with SOTA Methods (WikiFact Balanced Dataset, Table 3)

On a balanced dataset of 300 original (all positive cases) + 300 modified (all negative cases):

| Method | All False | All True | Overall |
|------|-----------|----------|---------|
| RePanda | **88.33%** | **85.67%** | **87.00%** |
| TAPEX | 41.00% | 59.33% | 50.16% |
| TAPAS | 55.00% | 65.33% | 60.16% |
| PASTA | 47.67% | 51.67% | 49.67% |

RePanda overall outperforms TAPAS by approximately **27 percentage points**, demonstrating that it has learned true discriminative capabilities rather than simply favoring the positive class.

### Zero-shot Comparison with DeepSeek-Chat (671B)

| Dataset | RePanda (7B) | DeepSeek-Chat (671B) |
|--------|-------------|---------------------|
| TabFact | **84.09%** | 82.62% |
| WikiFact | 84.72% | 85.39% |

The 7B model even outperforms the 671B model on TabFact, successfully achieving knowledge distillation of structured reasoning capabilities.

### Tabular QA Results (WikiTableQuestions, Table 4)

| Method | Accuracy | Training Data Size |
|------|--------|-----------|
| TabLaP | 76.6% | Large Scale |
| SynTQA (GPT) | 74.4% | Large Scale |
| **RePanda** | **75.1%** | **1,200** |
| Mix SC | 73.6% | Large Scale |
| Chain-of-Table | 67.31% | Large Scale |

Accurate results competitive with SOTA methods are achieved using only 1200 training samples.

### Ablation Study: Impact of Error Correction (Table 5)

| Dataset | Without Correction | With Correction | Gain |
|--------|--------|--------|------|
| TabFact | 78.02% | 84.09% | +6.07% |
| WikiFact | 74.43% | 84.72% | +10.29% |
| WQA | 67.59% | 75.1% | +7.51% |

Error correction brings significant improvements across all tasks, with the largest gain in OOD scenarios (+10.29%).

## Highlights & Insights

1. **Interpretability Advantages of Execution-based Reasoning**: The generated pandas queries natively represent the reasoning process, allowing users to verify the logic step-by-step—which is impossible with black-box models.
2. **Impressive OOD Generalization Capability**: Achieving 84.72% on completely unseen WikiFact tables proves that the model learns general tabular reasoning patterns rather than specific data distributions.
3. **Efficient Knowledge Distillation**: The 7B model successfully approaches or even exceeds the 671B model through structured reasoning learning, indicating that intermediate representations (pandas queries) are a more efficient medium for knowledge transfer than direct classification.
4. **Extremely Low Data Requirements**: Tabular QA requires only 1200 samples to reach SOTA performance, demonstrating the extreme efficiency of the execution framework in using training data.
5. **Importance of Automatic Error Correction**: Syntactic correction during inference (4 iterative repair rounds) contributes to a performance gain of about 6–10 percentage points.

## Limitations & Future Work

1. **Single-Table Reasoning Only**: All experiments are based on single-table scenarios, leaving more complex structured reasoning, such as multi-table joins and cross-table reasoning, unverified.
2. **Dependence on DeepSeek-Chat for Training Data**: The queries for PanTabFact and PanWiki are generated by a 671B model, making data construction highly costly.
3. **Syntactic Correction Increases Inference Latency**: Reasoning requires up to 4 iterations to repair syntactic errors, which may not be suitable for latency-sensitive online scenarios.
4. **Expressive Boundaries of pandas Queries**: Extremely complex reasoning (involving statistical modeling, pattern discovery, etc.) may exceed the expressive power of pandas queries.
5. **Limited Evaluation Scope**: Validated only on two benchmarks, TabFact and WikiTableQuestions, while broader tabular reasoning benchmarks (e.g., HybridQA, SQA) remain untested.

## Related Work

- **Tabular Fact Verification**: Table-BERT (Chen et al., 2019), TAPAS (Herzig et al., 2020), TAPEX (Liu et al., 2021), PASTA (Gu et al., 2022)
- **Structured Representation Learning**: ProgVGAT (Yang et al., 2020), ReasTAP (Zhao et al., 2022), StructGPT (Jiang et al., 2023)
- **Tabular Question Answering**: Chain-of-Table (Wang et al., 2024b), TabLaP (Wang et al., 2024a), SynTQA (Zhang et al., 2024)
- **Tool-Augmented Reasoning**: ReAct (Yao et al., 2023), Chain-of-Thought (Wei et al., 2022)

## Rating

⭐⭐⭐⭐ — The method is clear and practical, with strong interpretability, excellent OOD generalization, and high data efficiency. However, supporting only single-table reasoning is a notable limitation, and the multi-round iterative syntactic correction increases inference costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](../../ICML2025/others/prediction-powered_adaptive_shrinkage_estimation.md)

</div>

<!-- RELATED:END -->
