---
title: >-
  [Paper Note] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions
description: >-
  [ACL2026][Code Intelligence][Synthetic Code Instructions] QAQ starts from the concept of "reverse semantic coherence"—whether an answer can reverse-predict its question. By utilizing stratified Reverse Mutual Information…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Synthetic Code Instructions"
  - "Data Selection"
  - "Reverse Mutual Information"
  - "Hard Samples"
  - "Model Disagreement"
date: 2026-05-08
content_hash: 6ed8669c493906f7
---

# QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions

**Conference**: ACL2026  
**arXiv**: [2603.12165](https://arxiv.org/abs/2603.12165)  
**Code**: Not open sourced  
**Area**: Code Intelligence / Synthetic Data Selection  
**Keywords**: Synthetic Code Instructions, Data Selection, Reverse Mutual Information, Hard Samples, Model Disagreement

## TL;DR
QAQ starts from the concept of "reverse semantic coherence"—whether an answer can reverse-predict its question. By utilizing stratified Reverse Mutual Information (RMI) and disagreement between strong and weak models, QAQ filters synthetic code instructions. Using only 25% of the WarriorCoder data, it achieves performance close to full training and significantly outperforms traditional metrics like IFD.

## Background & Motivation
**Background**: Code generation models increasingly rely on large-scale synthetic instruction-response data. Seedless synthesis pipelines like Magpie and WarriorCoder can generate massive code tasks directly from alignment model interactions. However, they also introduce noise such as hallucinated terms, nonsensical queries, mismatched Q&A pairs, and templated answers.

**Limitations of Prior Work**: Many data selection methods focus solely on the answer direction. For example, IFD measures the difficulty of generating an answer given a query ($A|Q$). On noisy synthetic code data, this perspective confuses two scenarios: the task itself is difficult, or the query/answer pair is inherently mismatched.

**Key Challenge**: Quality issues in synthetic code data are often hidden on the query side. An answer might be syntactically correct code while the query contains fake terminology or non-code requests; evaluating only the answer quality would misidentify such samples as high quality.

**Goal**: Design a scalable data filtering method that filters semantic mismatches and surface-level repetitions while retaining hard samples with real training value for code models, thereby reducing fine-tuning costs.

**Key Insight**: The authors reverse the direction, asking "can the model better predict the original question after seeing the code answer?" A code answer that truly fits a question should contain enough information for a model to infer the task type; conversely, a mismatched answer provides almost no explanatory power for the query.

**Core Idea**: Use Reverse Mutual Information (RMI) to evaluate the explanatory power of an answer for its query, and retain samples where "the strong model recognizes its validity while the weak model still finds it difficult" via model disagreement.

## Method
The key to QAQ is not simply selecting samples with the highest RMI. The authors define RMI and observe that both extremes can be problematic: too low indicates query-answer mismatch, while too high may indicate the answer merely repeats the query or contains flawed patterns easily recognized by the model. Therefore, QAQ incorporates question complexity stratification and strong-weak model disagreement, shifting the selection logic from "high-score priority" to "moderate quality within complexity tiers + cognitive gap."

### Overall Architecture
The input is a large-scale synthetic code instruction-tuning dataset $D=\{(Q_i,A_i)\}$. QAQ first uses language models to calculate the query perplexity $PPL(Q)$ and the query perplexity given the answer $PPL(Q|A)$ to derive RMI. Then, samples are grouped into 10 complexity bins based on $PPL(Q)$, and RMI is ranked within each bin. Finally, stratified RMI ranks are calculated using both a strong model and a weak model. Samples categorized as "Diff-High" (ranked high by the strong model but low by the weak model) are retained, forming a subset of approximately 25%.

### Key Designs
1.  **Reverse Mutual Information (RMI)**:
    - **Function**: Measures the explanatory power of the answer regarding the question, rather than how much the question helps generate the answer.
    - **Mechanism**: Defined as $RMI(Q,A)=\log PPL(Q)-\log PPL(Q|A)$. If the query becomes easier to predict after seeing the answer, the pair is semantically coherent; if the change is minimal, it may be a mismatch or irrelevant.
    - **Design Motivation**: Code answers typically contain function names, inputs/outputs, algorithmic structures, and boundary conditions that should reflect problem specifications. Reverse modeling directly checks "whether the code truly corresponds to this problem."

2.  **Stratified RMI rank by question complexity**:
    - **Function**: Prevents simple questions from being penalized by global RMI rankings due to their naturally low $PPL(Q)$.
    - **Mechanism**: Samples are divided into $K=10$ deciles based on $PPL(Q)$. RMI ranks are calculated independently within each bin and normalized to $[0,1]$. This ensures simple questions are only compared with other simple questions.
    - **Design Motivation**: The paper finds a heteroscedastic relationship between RMI and $\log PPL(Q)$, where lower complexity questions have lower RMI ceilings. Global sorting would filter out simple but effective samples.

3.  **Strong-Weak Model Disagreement (Diff)**:
    - **Function**: Extracts samples that are valid yet possess significant learning value from RMI signals.
    - **Mechanism**: DeepSeek-Coder-6.7B-Base is used as the strong model and Qwen3-0.6B as the weak model to obtain stratified RMI ranks $r_s$ and $r_w$ respectively. Disagreement is defined as $Diff=r_s-r_w$. High Diff indicates the strong model recognizes the query-answer relationship while the weak model cannot. Samples with $Diff>0.1$ are retained.
    - **Design Motivation**: Samples where both models rank highly might be repetitions or paraphrases; samples where both rank low might be bad data or overly difficult. "Strong-high and weak-low" represents a more effective and learnable training signal.

### Loss & Training
QAQ is a data selection strategy and does not change the loss function of the downstream code model. In experiments, the authors fine-tuned DeepSeek-Coder-6.7B-Base using LlamaFactory for 3 epochs. Batch size and learning rate were adjusted based on data scale: full data used a batch size of 512 and LR of 1.2e-4; 50% data used a batch size of 256 and LR of 0.8e-4; 25% data used a batch size of 256 and LR of 0.4e-4, with a 0.2 warmup ratio and cosine decay. Evaluation utilized greedy decoding, reporting pass@1 on HumanEval, HumanEval+, MBPP, and MBPP+.

## Key Experimental Results

### Main Results

| Method | Data Ratio | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|----------|-----------|------------|------|-------|
| Full Data | 100% | 78.05 | 72.56 | 71.69 | 59.52 |
| RMI Top 50% | 50% | 78.05 | 73.17 | 72.22 | 58.20 |
| RMI 50-75% | 25% | 77.44 | 72.56 | 71.43 | 58.47 |
| Random | 25% | 73.78 | 69.51 | 68.52 | 57.67 |
| IFD | 25% | 71.95 | 66.46 | 64.81 | 54.76 |
| RDS+ | 25% | 76.83 | 71.34 | 71.69 | 58.99 |
| SCAR | 25% | 75.00 | 70.73 | 70.63 | 57.67 |

### Ablation Study

| Selection Strategy | HumanEval | HumanEval+ | MBPP | MBPP+ | Note |
|----------|-----------|------------|------|-------|------|
| Diff-High | 77.44 | 71.95 | 71.43 | 58.73 | Strong-high, Weak-low; core strategy |
| Sum-High | 74.39 | 68.90 | 71.16 | 59.52 | High consensus; may include patterns |
| Sum-Low | 71.34 | 65.85 | 66.14 | 55.56 | Low consensus; worst quality |
| Diff-Low | 71.34 | 67.07 | 74.87 | 62.43 | Bias toward simple patterns; aids MBPP |

### Key Findings
- The correlation between RMI and IFD is very low (Spearman correlation of 0.252), indicating that $Q|A$ and $A|Q$ capture different quality dimensions.
- The 25% subset from RMI 50-75% nearly matches full training performance, with HumanEval+ at 72.56 (same as Full Data).
- The overlap between Diff-High and Sum-High selection sets is only 13.85%, showing that model disagreement selects a very different type of sample.
- The RMI rank correlation before and after fine-tuning the same model is 0.9539, indicating that RMI is stable and suitable for one-time static filtering.
- On Magpie-Qwen2.5-Coder-Pro-300K, QAQ with 25% data also achieved balanced results: HumanEval 71.95, HumanEval+ 65.24, MBPP 68.25, MBPP+ 56.35.

## Highlights & Insights
- The most clever aspect is using "whether the code can explain the problem" as a quality signal. Code generation tasks naturally feature structured answers and specified queries, making reverse prediction more semantically explanatory than in general text tasks.
- The observation that both RMI extremes can be detrimental is crucial. Low RMI indicates mismatches, while high RMI may indicate keyword echoes or paraphrasing shortcuts. Thus, simple top-k selection is insufficient; stratification and disagreement are necessary.
- Strong-weak model disagreement shifts data selection from "finding the easiest high-quality samples to recognize" to "finding samples that the strong model understands but the weak model still needs to learn," which shares intuition with curriculum learning or the zone of proximal development.
- The method is cost-friendly for training: RMI calculations can be performed offline, and fine-tuning with 25% of the data after filtering approaches full-scale performance.

## Limitations & Future Work
- Primary experiments focused on WarriorCoder, with supplemental validation on only one other seedless synthetic code dataset; generalization to seed-based data, general instruction data, and domain-specific code requires more evidence.
- RMI calculation requires running teacher-forcing perplexity on large-scale data. While cheaper than full fine-tuning, it still depends on multi-GPU offline scoring.
- The choice of strong and weak models affects the Diff signal. While the paper shows different pairs are effective, it does not provide principles for automatically selecting model pairs.
- For very short questions or very long answers, RMI may still be influenced by length, templates, and linguistic style, requiring further calibration.
- The implementation is not open sourced; care is needed regarding chat templates, per-token normalization, and filtering threshold details for replication.

## Related Work & Insights
- **vs IFD**: IFD looks at the $A|Q$ direction, focusing on instruction-following difficulty; QAQ looks at the $Q|A$ direction, focusing on the explanatory power of the answer, making it more sensitive to query quality.
- **vs Superfiltering**: Superfiltering uses weak models to approximate strong model scores for cost savings; QAQ leverages strong-weak inconsistency, treating the disagreement itself as a signal.
- **vs SCAR**: SCAR focuses on instruction-response stylistic consistency; QAQ directly models semantic coherence, leading to more balanced performance across benchmarks.
- **vs RDS+ / coreset**: RDS+ requires target task seeds, while QAQ does not depend on downstream test set seeds, making it more suitable for general synthetic data cleaning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Reverse Mutual Information and model disagreement is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers main results, disagreement ablation, cross-dataset performance, and model pair sensitivity.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive failure mode examples, and high information density in tables.
- Value: ⭐⭐⭐⭐ Practical significance for code instruction data cleaning, synthetic data denoising, and low-cost fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)

</div>

<!-- RELATED:END -->
