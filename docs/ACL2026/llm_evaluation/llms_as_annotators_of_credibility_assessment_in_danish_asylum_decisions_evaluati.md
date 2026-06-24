---
title: >-
  [Paper Note] LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics
description: >-
  [ACL 2026][LLM Evaluation][LLM-as-annotator] The RAB-Cred expert-annotated dataset for three-class ("Absent / Positive / Negative") credibility assessment was constructed using 273 asylum decision documents from the Danish Refugee Appeals Board (RAB). A systematic evaluation of 21 open-source LLMs across 30 system×user prompt combinations reveals that prompt design is more significant than model selection. While Phi-4 (14B) achieved a 94.7% F1 in a zero-shot setting…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "LLM-as-annotator"
  - "Refugee Asylum"
  - "Credibility Assessment"
  - "Prompt Engineering"
  - "Error Analysis"
  - "Ensemble Voting"
date: 2026-05-08
content_hash: 3791a17e7cb10ff3
---

# LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics

**Conference**: ACL 2026  
**arXiv**: [2605.13412](https://arxiv.org/abs/2605.13412)  
**Code**: https://github.com/glhr/RAB-Cred (Available)  
**Area**: LLM Annotation / Legal NLP  
**Keywords**: LLM-as-annotator, Refugee Asylum, Credibility Assessment, Prompt Engineering, Error Analysis, Ensemble Voting

## TL;DR
The RAB-Cred expert-annotated dataset for three-class ("Absent / Positive / Negative") credibility assessment was constructed using 273 asylum decision documents from the Danish Refugee Appeals Board (RAB). A systematic evaluation of 21 open-source LLMs across 30 system×user prompt combinations reveals that prompt design is more significant than model selection. While Phi-4 (14B) achieved a 94.7% F1 in a zero-shot setting, individual models consistently committed "unacceptable" errors. Consequently, a majority-voting ensemble utilizing the 15 optimal model-prompt combinations is recommended, which increased accuracy by 1.5 pp to 96%.

## Background & Motivation

**Background**: The use of LLMs as "off-the-shelf text annotators" has expanded across sociology and the humanities. Legal NLP has seen preliminary attempts (e.g., argument mining, legal interpretation), but most studies evaluate only a few prompt-model combinations (e.g., GPT-4), leading to unreliable conclusions regarding model capability or prompt impact.

**Limitations of Prior Work**: The task of identifying whether a credibility assessment was made in an asylum decision, and whether it was positive or negative, presents three concurrent challenges: (1) The concept itself is ambiguous—legal scholars lack a consensus on credibility assessment; (2) High overlap with "risk assessment" in phrasing and proximity—a single document can contain both; (3) The language is Danish (medium-resource) combined with professional legal terminology. Most open-source LLMs have limited training data in Danish (e.g., Phi-4 reports only 8% multilingual data, and Llama 3 does not officially support Danish).

**Key Challenge**: Domain experts require both high automated annotation accuracy and an understanding of how and where LLMs fail. Aggregate metrics like Macro-F1 fail to address (a) cross-model error consistency, (b) correlation with human annotation confidence, and (c) the severity of errors.

**Goal**: (1) Establish a high-quality legal classification benchmark, RAB-Cred, in an under-represented language/domain with metadata (annotator confidence, case outcome); (2) Systematically benchmark 21 open-source models × 30 prompt combinations to quantify the importance of model vs. prompt; (3) Analyze error types, cross-model consistency, and severity rather than focusing solely on F1.

**Key Insight**: LLM annotators should be treated as "prone-to-error professional interns." This requires both aggregate accuracy and root-cause error analysis. The instance-level PromptSensiScore (PSS) is introduced to quantify prediction flips across prompts, distinguishing "pseudo-robust" models (e.g., Qwen3-30B) that are stable at the dataset level but unstable at the instance level.

**Core Idea**: Replace single LLM annotators with a majority vote of the 15 most optimal model-prompt combinations and utilize post-hoc expert assessments of error severity (acceptable/understandable/unacceptable) to evaluate the feasibility of model-expert substitution.

## Method

### Overall Architecture
This study investigates the extent to which open-source LLMs can replace legal experts in annotating credibility assessments. A four-step pipeline was developed spanning database construction to error attribution. Initially, the RAB-Cred benchmark was built using Danish RAB documents. Then, a grid of 21 open-source LLMs and 30 prompt combinations was evaluated on a validation set. The 15 best combinations were selected for ensemble voting on the test set. Diagnostic analysis focused on per-case cross-model consistency, prompt sensitivity, and error severity. The pipeline was executed on a single H100 (80GB) GPU.

### Key Designs

**1. 6×5 Orthogonal Prompt Grid: Decoupling Domain Knowledge and Reasoning Structure**

To quantify the "task-specific" nature of prompts, an orthogonal grid of system × user prompts was designed. System prompts (SP) increased domain knowledge injection hierarchically: SP0 (empty) → SP1 (expert persona) → SP2 (SP1 + codebook) → SP3 (SP2 + typical Danish phrases) → SP4 (SP3 + boundary cases) → SP5 (SP1 + expert disambiguation). User prompts (UP) increased reasoning complexity: UP1 (direct choice) → UP1-FS (few-shot) → UP2 (two-step: presence then polarity) → UP3 (zero-shot CoT) → UP4 (metacognitive prompting). Results showed that SP3/SP4, co-written by CS and domain experts, outperformed those written solely by domain experts (SP2/SP5). UP gains were highly model-dependent.

**2. Top-3 Prompt Averaging + 15-Annotator Ensemble: Simulating Expert Consensus**

Single LLM annotators are risky; the best test set performer (Phi-4+SP4+UP4) achieved 94.7% F1 but still committed "unacceptable" errors. The ensemble selected the top 5 models based on validation performance (Phi-4, Gemma-3-27B, Ministral-3-14B, Mistral-Small-24B, Qwen3-30B) and their respective top-3 prompts. On the test set, majority voting reached 96% accuracy (+1.5pp over single SOTA). More importantly, for the 8 remaining error cases, experts judged 4 as "acceptable," 2 as "understandable," and only 2 as "unacceptable," indicating that ensembles effectively mitigate severe errors.

**3. Instance-level PromptSensiScore (PSS): Distinguishing Robustness Types**

Aggregate F1 variance across prompts can hide "pseudo-robustness" where specific predictions flip between cases. Borrowing from Zhuo et al. (2024), PSS calculates the stability of predictions across a set of prompts $\mathcal{P}$ for each case $x_i$, denoted as PSS$(x_i)$. Findings suggest prompts are more critical than models: Phi-4 was most stable (PSS=0.043), while Qwen3-30B was unstable (PSS=0.110). Qwen3-30B exhibited a plateaued F1 across prompts, but failed on different cases each time—a risk exposed by PSS.

### Loss & Training
This study is a pure zero/few-shot benchmark without model training. "Training" involved few-shot selection of "hard cases" where experts had high confidence but LLMs failed. Constrained decoding was implemented via the `outlines` library for specific schemas: $\text{Literal}[\text{"NO/POSITIVE/NEGATIVE CREDIBILITY ASSESSMENT"}]$. Models not supporting `outlines` used regex post-processing. All runs used greedy decoding.

## Key Experimental Results

### Main Results
The evaluation involved 21 models × 30 prompts on 70 validation samples, with the top 15 combinations tested on 200 samples. Test set F1 ranged from 84.4% to 94.7%.

| Model-Prompt | val Macro-F1 | test Macro-F1 | test Accuracy | Note |
|------------|--------------|---------------|---------------|------|
| Phi-4 + SP4 + UP4 | 86.34 | **94.69** | **94.50** | Single Model SOTA |
| Phi-4 + SP4 + UP2 | **90.51** | 93.66 | 94.00 | Highest val |
| Mistral-Small-24B + SP3 + UP1-FS | 85.63 | 92.52 | 92.50 | Strongest few-shot |
| Ministral-3-14B + SP4 + UP3 | 86.59 | 91.63 | 92.00 | Strongest CoT |
| Gemma-3-27b + SP4 + UP1-FS | 79.91 | 84.39 | 86.50 | Weakest in selection |
| Qwen3-30B + SP5 + UP2 | 83.61 | 87.95 | 89.00 | Larger model not necessarily better |
| **Ensemble (15 LLMs, majority vote)** | – | – | **96.00** | +1.5 pp vs single SOTA |
| Outcome-as-classifier baseline | – | – | 53.00 | Naive baseline |
| Human H1 vs H2 | – | – | 98.4 ($\kappa=0.967$) | Upper Bound |

Performance did not correlate with model size: Qwen3-30B and others plateaued ≤85% F1, significantly trailing the 14B Phi-4. Despite its model card stating only 8% multilingual data, Phi-4 performed best, suggesting high-quality data outweighs multilingual breadth.

### Ablation Study
Error structure and component contributions (Test set, 200 cases):

| Configuration / Analysis | Key Number | Explanation |
|------------|---------|------|
| All 15 LLMs correct | 144 / 200 = 72% | "Easy" cases |
| ≥ 8 LLMs correct (Majority Vote) | 190 / 200 = 95% | Majority vote ceiling |
| Every LLM made ≥ 1 "unacceptable error" | 15 / 15 | Individual model failure inevitable |
| Ensemble majority vote errors | 8 / 200 = 4% | 1.5pp lower than single SOTA |
| Ensemble error severity | 4 acc / 2 und / 2 unacc | 50% are boundary cases |
| Phi-4 instance-level PSS | 0.043 | Most stable |
| Qwen3-30B instance-level PSS | 0.110 | Aggregate stable, instance-level unstable |
| Average prompt sensitivity | ~0.08 | > model sensitivity (0.05) |

### Key Findings
- **Prompts are more critical than models**: The PSS for fixing prompts and changing models (0.05) was lower than fixing models and changing prompts (0.06-0.11). Multi-prompt evaluation should be standard.
- **Large Model $\neq$ Good Model**: Phi-4 (14B) outperformed 30B+ models. Quality of data is more important than parameter count for medium-resource languages.
- **CoT/metacognition gains depend on SP**: Reasoning prompts primarily helped when the system prompt lacked a comprehensive codebook.
- **Task decomposition (UP2) is model-specific**: Breaking down the decision improved Phi-4 but degraded Gemma-3.
- **LLM errors align with human uncertainty**: Most cases where <75% of LLMs were correct also saw "Low/Medium" confidence from human experts.

## Highlights & Insights
- **Expert-graded error severity**: Moving beyond binary accuracy to a three-tier severity scale provides a high-density evaluation paradigm essential for high-stakes deployments (legal, medical).
- **PSS reveals hidden risks**: PSS distinguished models like Qwen3-30B that appeared robust at the aggregate level but were unstable per case.
- **Interdisciplinary prompt design is superior**: Prompts co-authored by domain and technical experts outperformed those written by domain experts alone.
- **Cost-effective ensemble strategy**: A 15-model majority vote suppressed "unacceptable" errors to 1%, offering expert-level reliability at a fraction of the cost of hiring multiple legal experts for large-scale annotation.

## Limitations & Future Work
- **Small Dataset (273 cases)**: Limited statistical representation for the entire corpus of 10,000+ RAB cases.
- **Exclusion of closed-source models**: GPT-4o and Claude were not tested due to data sensitivity requirements for offline processing.
- **Single run with greedy decoding**: Multi-seed variance and the potential negative impact of constrained decoding were not fully explored.
- **English Prompt/Danish Input**: The cross-lingual effect was not compared against pure Danish or translated English prompts.
- **Reasoning Trace Alignment**: The study did not analyze whether LLM reasoning aligned with expert logic, only focusing on final labels.

## Related Work & Insights
- **vs AsyLex (Barale et al. 2023)**: AsyLex is in English; RAB-Cred fills the gap for medium-resource languages and includes expert confidence metadata.
- **vs General LLM-as-annotator literature**: This study extends standard aggregate metrics by incorporating 21 models × 30 prompts with instance-level PSS and severity analysis.
- **Insight**: High-stakes fields should normalize "severity-classified error analysis." Small, high-quality models like Phi-4 may offer better paths for low-reasoning/medium-resource tasks than massive multilingual models.

## Rating
- Novelty: ⭐⭐⭐⭐ New dataset and combined innovation of ensemble voting with severity classification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive grid search across 630 combinations with post-hoc diagnostic metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear legal context, intuitive examples, and high reproducibility via provided code.
- Value: ⭐⭐⭐⭐ Provides a robust methodological framework for LLM-based annotation in social sciences and law.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA](beyond_case_law_evaluating_structure-aware_retrieval_and_safety_in_statute-centr.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[ACL 2025\] Where Are We? Evaluating LLM Performance on African Languages](../../ACL2025/llm_evaluation/where_are_we_evaluating_llm_performance_on_african_languages.md)

</div>

<!-- RELATED:END -->
