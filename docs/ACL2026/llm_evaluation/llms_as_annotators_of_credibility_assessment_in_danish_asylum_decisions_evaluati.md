---
title: >-
  [Paper Note] LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics
description: >-
  [ACL 2026][LLM Evaluation][LLM-as-annotator] The RAB-Cred dataset, a three-class ("Absent / Positive / Negative") credibility assessment expert-annotated benchmark…
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
content_hash: 298f60fb17dc329a
---

# LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics

**Conference**: ACL 2026  
**arXiv**: [2605.13412](https://arxiv.org/abs/2605.13412)  
**Code**: https://github.com/glhr/RAB-Cred (Available)  
**Area**: LLM Annotation / Legal NLP  
**Keywords**: LLM-as-annotator, Refugee Asylum, Credibility Assessment, Prompt Engineering, Error Analysis, Ensemble Voting

## TL;DR
The RAB-Cred dataset, a three-class ("Absent / Positive / Negative") credibility assessment expert-annotated benchmark, was constructed using 273 asylum decision documents from the Danish Refugee Appeals Board (RAB). By systematically evaluating 21 open-source LLMs across 30 system-user prompt combinations, the study found that prompt design is more critical than model selection. While Phi-4 (14B) achieved a 94.7% F1 in a zero-shot setting, individual models consistently made "unacceptable" errors. Consequently, a majority voting ensemble of the "top 15 model-prompt combinations" is proposed, which improved accuracy by 1.5 percentage points to 96%.

## Background & Motivation

**Background**: Utilizing LLMs as "out-of-the-box text annotators" has become widespread in the social sciences and humanities. While there have been sporadic attempts in Legal NLP (e.g., argument mining, legal interpretation), most studies evaluate only one or two prompts combined with a few closed-source models (such as GPT-4), leading to unreliable conclusions regarding model capabilities or prompt impacts.

**Limitations of Prior Work**: The task of "identifying whether a credibility assessment was made in an asylum decision and, if so, whether it was positive or negative" presents three simultaneous challenges: (1) Conceptual ambiguity—there is no consensus on credibility assessment even within refugee law scholarship; (2) High overlap with "risk assessment"—terms are often adjacent and easily confused, as a single document can contain both judgments; (3) Language—the documents are in Danish (a medium-resource language) with specialized legal terminology. Most mainstream open-source LLMs have very low proportions of Danish in their training data (e.g., Phi-4 admits to only 8% multilingual data, and Llama 3 does not officially support Danish).

**Key Challenge**: Domain experts require not only "high automated annotation accuracy" but also an understanding of "how and on which cases the LLM fails." Aggregate metrics like Macro-F1 fail to address the latter. Previous LLM-as-annotator literature relies almost exclusively on dataset-level F1, obscuring: (a) error consistency across models, (b) correlation with human annotation confidence, and (c) the severity of specific errors.

**Goal**: (1) Construct a high-quality legal classification benchmark (RAB-Cred) for an under-represented language and domain, including metadata (annotator confidence, case outcomes); (2) Quantify the relative importance of models versus prompts via a systematic benchmark of 21 models and 30 prompt combinations; (3) Perform a granular error analysis focusing on error types, cross-model consistency, correlation with human confidence, and severity.

**Key Insight**: LLM annotators should be treated as "error-prone professional interns." This necessitates both aggregate accuracy and root-cause analysis of errors. An instance-level PromptSensiScore (PSS) is introduced to quantify whether single-sample predictions flip when prompts change, distinguishing "pseudo-robust" models (e.g., Qwen3-30B) that appear stable at the dataset level but are unstable at the instance level.

**Core Idea**: Replace any single LLM annotator with a "majority vote of the top 15 model-prompt combinations." Use post-hoc expert assessments of error severity (categorized as acceptable/understandable/unacceptable) to evaluate the extent to which LLMs can realistically replace human experts.

## Method

### Overall Architecture
The process involves four steps: (1) **Dataset Construction**—10,817 public decision documents were crawled from the RAB website. Stratified sampling yielded 73 validation and 200 test samples. Two Danish refugee law experts (H1, H2) independently annotated labels and confidence levels (Low/Medium/High), with disagreements resolved by H3 (Cohen's $\kappa = 0.97$). (2) **Model x Prompt Grid**—21 open-source LLMs (3B-35B parameters, 9 model families) were tested with 30 combinations (6 system prompts x 5 user prompts). Each combination underwent greedy decoding on 70 validation samples, using the `outlines` library for constrained decoding to ensure outputs matched the set $\{\text{NO}, \text{POSITIVE}, \text{NEGATIVE}\} \times \text{CREDIBILITY ASSESSMENT}$. (3) **Top-15 Annotator Selection**—Models were ranked by the average Macro-F1 of their top 3 prompts. The top 5 models, each with its top 3 prompts, formed the 15 LLM annotators for the test set. (4) **Granular Error Analysis**—Analysis included case-by-case ensemble agreement, PSS quantification of sensitivity, and expert-rated error severity. The pipeline was executed on a single H100 (80GB).

### Key Designs

1.  **6x5 Orthogonal Grid of Nested System Prompts and User Prompts**:
    - **Function**: Decouples "domain knowledge injection" from "task reasoning structure" to avoid confounded prompt effects.
    - **Mechanism**: System prompts (SP) increase in nested complexity: SP0 (Empty) → SP1 (Expert persona) → SP2 (SP1 + original codebook) → SP3 (SP2 rewrite + typical Danish phrases per class) → SP4 (SP3 + boundary cases like hypothetical legal constructs) → SP5 (SP1 + manual disambiguation for "credibility vs. risk assessment"). User prompts (UP) increase in reasoning complexity: UP1 (Direct selection) → UP1-FS (UP1 + 3 few-shot samples) → UP2 (Two-step: presence detection then valence classification) → UP3 (Zero-shot CoT) → UP4 (Zero-shot metacognitive prompting). All prompts are in English for Danish inputs.
    - **Design Motivation**: Quantitative results showed that SP3/SP4 (co-written by CS and domain experts) outperformed SP2/SP5 (domain experts only), proving the value of interdisciplinary prompt design. Conversely, UP impact was **highly model-dependent**; for instance, Phi-4 achieved its peak with UP2, while Gemma-3 degraded significantly with the same prompt.

2.  **Top-3 Prompt Average + 15 Annotator Ensemble Voting**:
    - **Function**: Uses the mode of a diverse yet high-performing set of LLM annotators to simulate multi-expert consensus.
    - **Mechanism**: The top 5 models (Phi-4, Gemma-3-27B-it, Ministral-3-14B, Mistral-Small-24B, Qwen3-30B) were selected based on validation performance. Each contributed its top 3 prompts, totaling 15 combinations. Majority voting determined the final prediction for each test case.
    - **Design Motivation**: While the highest single-model test F1 was 94.7% (Phi-4+SP4+UP4), every individual annotator committed at least one "unacceptable error." Ensemble voting increased accuracy to 96% (+1.5pp). In the 8 remaining error cases, expert evaluation found 4 were "acceptable," 2 were "understandable," and only 2 remained "unacceptable," significantly reducing the rate of severe errors.

3.  **Instance-level PromptSensiScore (PSS)**:
    - **Function**: Traditional prompt sensitivity often ignores "prediction flipping" when aggregate F1 remains stable.
    - **Mechanism**: PSS calculates the variance of prediction accuracy across a set of prompts $\mathcal{P}$ for each case $x_i$. Two slices were analyzed: (a) **Fixed Model, Variable Prompt** (Prompt sensitivity: Phi-4 PSS=0.043 [lowest], Qwen3-30B PSS=0.110 [highest]); (b) **Fixed Prompt, Variable Model** (Model sensitivity: PSS=0.05).
    - **Design Motivation**: The comparison revealed that **instability from prompt changes > instability from model changes**. PSS serves as a diagnostic tool to identify models like Qwen3-30B, which appear robust at an aggregate level (F1 ≈ 87.9% across prompts) but actually flip labels on different cases, canceling out errors by chance.

### Loss & Training
This work is a **pure zero/few-shot benchmark** without model training. The only "tuning" involved few-shot example selection: choosing one high-information-density case per category where domain expert confidence was high but LLM zero-shot performance failed. Constrained decoding used the `outlines` library to enforce the schema `Literal["NO/POSITIVE/NEGATIVE CREDIBILITY ASSESSMENT"]`. For models not supporting `outlines`, regex post-processing was used.

## Key Experimental Results

### Main Results
Evaluation involved 21 models across 30 prompts. The top 15 combinations were evaluated on the test set (200 samples). Test F1 scores ranged from 84.4% to 94.7%, while the outcome-as-classifier baseline was 53%.

| Model-Prompt | val Macro-F1 | test Macro-F1 | test Accuracy | Remarks |
|------------|--------------|---------------|---------------|------|
| Phi-4 + SP4 + UP4 | 86.34 | **94.69** | **94.50** | Single Model SOTA |
| Phi-4 + SP4 + UP2 | **90.51** | 93.66 | 94.00 | Highest val performance |
| Mistral-Small-24B + SP3 + UP1-FS | 85.63 | 92.52 | 92.50 | Best Few-shot |
| Ministral-3-14B + SP4 + UP3 | 86.59 | 91.63 | 92.00 | Best CoT |
| Gemma-3-27b + SP4 + UP1-FS | 79.91 | 84.39 | 86.50 | Weakest in top selection |
| Qwen3-30B + SP5 + UP2 | 83.61 | 87.95 | 89.00 | Size doesn't guarantee strength |
| **Ensemble (15 LLMs, majority vote)** | – | – | **96.00** | +1.5 pp vs Single SOTA |
| Outcome-as-classifier baseline | – | – | 53.00 | Naive Baseline |
| Human H1 vs H2 | – | – | 98.4 ($\kappa=0.967$) | Upper Bound |

Model size did not correspond to performance: Qwen2.5-32B, Qwen3-30B, and Aya-Expanse-32B all plateaued at ≤85% F1, falling behind the 14B Phi-4. Despite Phi-4's model card stating it was not designed for multilingual tasks (8% multilingual data), it performed best, suggesting data quality outweighs multilingual coverage. Inter-LLM Cohen's $\kappa$ peaked at 0.950, lower than the human expert $\kappa$ of 0.967.

### Ablation Study
Error structure and component contribution (Test set, $n=200$):

| Configuration / Analysis | Value | Description |
|------------|---------|------|
| All 15 LLMs correct | 144 / 200 = 72% | "Simple" cases |
| ≥ 8 LLMs correct (Majority Vote) | 190 / 200 = 95% | Majority vote ceiling |
| Every LLM made ≥ 1 "unacceptable" error | 15 / 15 | Single model failure is inevitable |
| Ensemble majority vote errors | 8 / 200 = 4% | 1.5pp fewer than single SOTA |
| Ensemble error severity (Expert) | 4 acceptable / 2 understandable / 2 unacceptable | 50% are ambiguous boundary cases |
| Phi-4 instance-level PSS | 0.043 | Most stable |
| Qwen3-30B instance-level PSS | 0.110 | Least instance-stable |
| Average Prompt Sensitivity | ~0.08 | Higher than Model Sensitivity (0.05) |
| Gemma-3-27B F1 var across prompts | 54% | Least robust |

### Key Findings
- **Prompt > Model**: Fixed-prompt model sensitivity (PSS 0.05) was lower than fixed-model prompt sensitivity (PSS 0.06-0.11), making multi-prompt evaluation essential for LLM annotation studies.
- **Bigger is Not Better**: Phi-4 (14B) outperformed >30B models like Qwen2.5, Qwen3, and Aya-Expanse, demonstrating that high-quality data trumps parameter count and broad multilingual coverage.
- **CoT/Metacognition depends on SP**: Meta-prompts were only effective over UP1 when system prompts were simple (SP0/SP1). With a complete codebook in the SP, CoT provided no gains.
- **Binary Decomposition (UP2) is Model-Specific**: Breaking the task into "detect then classify" helped Phi-4 but harmed Gemma-3; task decomposition is not a universal benefit.
- **Error overlap with human uncertainty**: Cases where <75% of LLMs were correct typically involved cases where at least one human expert had low/medium confidence, suggesting inherent ambiguity in the data rather than model failure.

## Highlights & Insights
- **Expert post-hoc severity grading**: This paradigm shifts evaluation from "correctness" to "utility," identifying an error as "acceptable" if it occurs in a fuzzy boundary case.
- **PSS as a diagnostic for "pseudo-robustness"**: Exposes models that maintain aggregate F1 through lucky error cancellation rather than true prompt invariance.
- **Value of interdisciplinary prompt design**: Proved that prompts co-authored by domain and technical experts outperform those by domain experts alone.
- **15-LLM Ensemble as a Human Proxy**: Majority voting significantly reduces unacceptable errors to 1%, providing expert-level reliability at a fraction of the cost of manual legal expert annotation.

## Limitations & Future Work
- **Small sample size**: 273 cases may have limited statistical representative power for the full 10,000+ RAB corpus.
- **Open-source constraints**: Closed-source models (GPT-4o) were not tested due to data sensitivity, leaving the total upper bound unknown.
- **Greedy decoding limitations**: The study did not explore multi-seed variance or the potential negative impact of constrained decoding on generation quality.
- **Language configuration**: The use of English prompts for Danish inputs was not controlled against a full-Danish or translated-English pipeline.
- **Reasoning interpretability**: The analysis focused on final labels rather than whether the LLM's CoT reasoning aligned with legal logic.

## Related Work & Insights
- **Comparison with AsyLex (Barale et al. 2023)**: While AsyLex is a refugee law NLP dataset, it is in English. RAB-Cred fills the gap for medium-resource languages and includes expert confidence metadata.
- **LLM-as-annotator literature**: Moves beyond simple model comparisons to a three-dimensional evaluation focusing on prompts, model stability (PSS), and error severity.
- **Codebook LLMs**: Supports the finding that merely providing a codebook is insufficient; effective annotation requires rewriting and the inclusion of edge cases within the prompt.

## Rating
- Novelty: ⭐⭐⭐⭐ (New dataset for Danish RAB; innovative ensemble + severity evaluation framework).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive grid search across models and prompts; detailed instance-level sensitivity and error analysis).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear legal context, intuitive examples, and high reproducibility).
- Value: ⭐⭐⭐⭐ (Provides a robust methodological framework for LLM-based annotation in high-stakes social/legal domains).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)

</div>

<!-- RELATED:END -->
