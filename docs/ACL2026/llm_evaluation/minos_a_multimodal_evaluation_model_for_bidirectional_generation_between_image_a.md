---
title: >-
  [Paper Note] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text
description: >-
  [ACL 2026][LLM Evaluation][Multimodal Evaluation] The authors developed Minos, an 8B evaluation model capable of scoring bidirectional multimodal generation tasks (I2T and T2I)…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Multimodal Evaluation"
  - "Data Quality Control"
  - "Bidirectional I2T/T2I Evaluation"
  - "DPO Alignment"
  - "Reference-free Scoring"
date: 2026-05-08
content_hash: 1f0a8643a15f7469
---

# Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text

**Conference**: ACL 2026  
**arXiv**: [2506.02494](https://arxiv.org/abs/2506.02494)  
**Code**: https://github.com/reroze/MINOS  
**Area**: Multimodal Evaluation / MLLM-as-a-Judge / Preference Alignment  
**Keywords**: Multimodal Evaluation, Data Quality Control, Bidirectional I2T/T2I Evaluation, DPO Alignment, Reference-free Scoring

## TL;DR
The authors developed Minos, an 8B evaluation model capable of scoring bidirectional multimodal generation tasks (I2T and T2I), through a three-step pipeline of "strict data quality control + SFT + DPO alignment." Using a high-quality evaluation dataset of 57K samples—less than half the scale of existing works—Minos outperforms all open-source MLLM-evaluators across 16 out-of-domain tasks and approaches the performance of GPT-4o.

## Background & Motivation

**Background**: As MLLM capabilities advance, "MLLM-as-a-Judge" has become the mainstream automatic evaluation paradigm for multimodal generation (image captioning, VQA, text-to-image, etc.). Representative works like Prometheus-Vision, LLaVA-Critic, and UnifiedReward train MLLMs as pointwise or pairwise scorers.

**Limitations of Prior Work**: (1) Existing evaluation models rely on massive data volume—LLaVA-Critic uses 113K and UnifiedReward uses 236K samples—with minimal quality filtering, often directly training on raw GPT-4o outputs. (2) Current models struggle to maintain high performance in both I2T and T2I simultaneously; LLaVA-Critic focuses almost exclusively on I2T, while UnifiedReward is relatively weak in I2T. (3) Preference alignment is rarely performed after SFT, leaving the potential gains of the alignment phase untapped.

**Key Challenge**: Is the bottleneck of evaluation capability "data quantity" or "data accuracy"? The authors argue for the latter—if an evaluation sample's GPT score is inconsistent with human judgment or if the overall score distribution is severely skewed (e.g., an excess of perfect scores), additional data serves only to pollute the model.

**Goal**: (1) Construct a high-quality, low-volume evaluation dataset spanning I2T and T2I; (2) Implement simultaneous SFT and DPO on this basis; (3) Validate the "Quality > Quantity" hypothesis in multimodal evaluation.

**Key Insight**: Treat human-annotated high-quality evaluation data (Polaris/LAVE/ImageReward) as the "gold standard." Generate 10 candidate evaluations per sample via GPT-4o and filter for reliability based on "human-score consistency" or "GPT internal mode." Further filter DPO preference pairs using the criterion "chosen/rejected score difference $\ge 2$."

**Core Idea**: Use strict instance-level + dataset-level quality control combined with delta-score-filtered DPO to enable 57K samples to outperform 236K raw samples.

## Method

### Overall Architecture
The Minos pipeline follows two tracks: **Data Construction** (Minos-57K + Minos-DPO-5.8K) and **Two-stage Training** (SFT $\to$ DPO). On the data side, each evaluation instance is formalized as $(q, d, g, k, [r], a, s)$—containing task input $q$, task description $d$, model output $g$, evaluation criteria $k$, optional reference answer $r$, output evaluation analysis $a$, and 1–5 Likert score $s$. This schema accommodates both I2T (image+question $\to$ text) and T2I (text $\to$ image), which is key to achieving "bidirectional unification." The backbone is Qwen3-VL-8B. SFT uses Minos-57K (2 epochs, lr 1e-5), and DPO uses Minos-DPO-5.8K (1 epoch, lr 2e-6, $\beta=0.03$).

### Key Designs

1.  **Pre-defined Guideline**:
    - **Function**: Provide a "task description + evaluation criteria" for six categories of multimodal tasks (captioning, VQA, T2I, text reading, reasoning, instruction following).
    - **Mechanism**: Just as human annotators are trained before evaluating different tasks, every evaluation input is concatenated with its corresponding task guideline. This forces the model to understand the specific context and standards for its current evaluation.
    - **Design Motivation**: Eliminates confusion regarding evaluation criteria across tasks, allowing the 8B model to reason based on guidelines for unseen OOD tasks rather than relying on intuition.

2.  **Consistency Filter**:
    - **Function**: Select the most "reliable" evaluation candidate from 10 GPT-4o outputs or discard the sample.
    - **Mechanism**: For data with **human scores** (Polaris/LAVE/ImageReward), only candidates where GPT score = human score are retained; otherwise, the sample is discarded. For data **without human scores**, the mode of the 10 candidate scores $\hat{s}=\mathrm{mode}(s_1,\ldots,s_{10})$ is used as a pseudo-label, and one analysis corresponding to $\hat{s}$ is randomly selected.
    - **Design Motivation**: Forces "model-generated evaluation data" to converge toward high-confidence modes, avoiding high-variance scoring issues inherent in GPT-4o. This filtered 124K raw samples down to 102K.

3.  **Score Balance + Delta Score DPO Filter**:
    - **Function**: Address dataset-level skew and noise in DPO preference pairs.
    - **Mechanism**: Post-consistency filtering, the distribution was still heavily biased toward score 5 (56% of samples). Random down-sampling balanced the distribution to approximately 16/17/21/23/23%. For DPO, the consistency-filtered sample serves as the chosen candidate, while the candidate with the largest score difference serves as the rejected one. Pairs are further filtered by $|s_\text{chosen} - s_\text{rejected}| \ge 2$, reducing 38K pairs to 5.8K.
    - **Design Motivation**: Authors found that "DPO without score-difference filtering dropped performance from 40.9 to 40.1." Sparse, high-quality DPO signals are required to prevent noise from damaging SFT-acquired capabilities.

### Loss & Training
The SFT phase utilizes standard next-token prediction, supervising the entire evaluation analysis $a$ followed by the score $s$ (experiments show SFT with analysis yields a 2.1 point higher average Pearson-r than learning scores only). DPO follows the standard formula from Rafailov et al., with $\beta=0.03$ and $\gamma=0$. Training was conducted on 4 H100 GPUs in BF16; SFT took approximately 10 hours and DPO 2 hours.

## Key Experimental Results

### Main Results
Following the LLaVA-Critic protocol: Pearson-r was used to measure correlation with human scores across 14 I2T tasks and 2 T2I tasks (RichHF-18K + GenAI-Bench), totaling 16 OOD datasets.

| Model | Scale | Avg. Pearson-r (16 tasks) | Notes |
| :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | / | 41.5 | Closed-source |
| GPT-4o | / | 44.2 | Closed-source ceiling |
| Qwen3-VL (base) | 8B | 38.4 | Same backbone baseline |
| LLaVA-Critic | 7B | 30.7 | I2T training only |
| LLaVA-Critic | 72B | 39.8 | Prev. SOTA (Open-source) |
| UnifiedReward_Q | 8B | 37.2 | Previous same-scale SOTA |
| **Minos** | **8B** | **42.3** | 2.5 points above 72B LLaVA-Critic |

### Ablation Study

| Configuration | Avg. Pearson-r | Description |
| :--- | :--- | :--- |
| RAW (124K, no QC) | 36.3 | Lower than base model 38.4 $\to$ "Dirty data causes negative training" |
| + Guideline | 37.1 | +0.8, task guidelines are significantly useful |
| + Consistency Filter (102K) | 39.0 | +1.9, filters GPT evaluation noise |
| + Score Balance $\to$ Minos-57K (SFT only) | 40.9 | +1.9, redistributes score balance |
| + Naive DPO (38K pairs) | 40.1 | **-0.8**, validates naive DPO harm |
| + Delta-Score DPO (5.8K) | **42.3** | +1.4, only high-quality pairs are effective |
| T2I training only (10K) $\to$ I2T Avg. | 25.4 | 11.3 lower than base 36.7; severe negative transfer |
| I2T training only (47K) $\to$ T2I Avg. | 46.1 | 4.8 lower than joint training 50.9 |
| **Joint I2T+T2I (57K)** | I2T 39.5 / T2I 50.9 | Bidirectional mutual gain |

### Key Findings
- **Quality is far more important than scale**: 57K samples (approx. 1/2 of LLaVA-Critic and 1/4 of UnifiedReward) achieved results 4.6 points higher than the 124K raw data; this is the most significant "anti-consensus" conclusion of the paper.
- **Naive DPO causes negative transfer**: 38K preference pairs dropped Pearson-r from 40.9 to 40.1, indicating evaluation models are highly sensitive to noise in preference pairs; the $|s_\text{chosen}-s_\text{rejected}| \ge 2$ heuristic corrected this by focusing on the top 1/6 of data.
- **I2T and T2I are complementary**: Training only on T2I severely degrades I2T capability (36.7 $\to$ 25.4), but joint training provides bidirectional gains, suggesting that evaluating "image-text consistency" shares underlying capabilities in both directions.
- **SFT with analysis is more accurate**: Requiring the model to output a rationale before the score is 2.1 Pearson-r points higher than direct scoring; the analysis anchors the scoring behavior.

## Highlights & Insights
- **"Quality > Scale" empirically proven in multimodal evaluation**: This work provides clear evidence that "dirty evaluation data can make the model worse than the base model" (36.3 < 38.4), a negative result that serves as a warning for future reward/judge model development.
- **Delta-Score Filtered DPO as a clean heuristic**: Since evaluation tasks utilize explicit scores $s$, they naturally allow for preference strength quantification. The authors use $|s_\text{chosen}-s_\text{rejected}| \ge 2$ as a "high confidence" filter, avoiding the need for additional reward models typically used to prune preference pairs. This trick is reusable in any alignment task with scorable outputs.
- **Unified schema $(q,d,g,k,[r],a,s)$ for joint I2T/T2I training**: By abstracting both "image+question" and "prompt" into "task input $q$ + task description $d$," the model supports bidirectional evaluation and can easily extend to other modalities like video-to-text or audio-to-text.

## Limitations & Future Work
- Authors acknowledge that due to compute constraints, scaling trends were not verified on 70B backbones. Some early human evaluation datasets were unavailable due to expired links.
- Observation: Tasks focus on "general" dimensions like generation quality and text alignment, missing specialized dimensions like safety, factuality, or hallucination. Guidelines are currently manual; scaling to dozens of tasks will require automated guideline generation.
- Pruning DPO data down to 5.8K suggests a scarcity of effective preference pairs. Future work could explore "soft score differences" or online preference mining.
- Improvement idea: Extend the Delta-Score Filter to process-level differences (e.g., independent scores for each evaluation criterion) or perform multi-task joint training of "analysis + score" during SFT.

## Related Work & Insights
- **vs LLaVA-Critic (7B/72B)**: LLaVA-Critic relies on 113K GPT-4o samples for I2T only. Minos uses 1/2 the data with strict quality control, DPO, and bidirectional coverage to outperform the 72B version by 2.5 Pearson-r points.
- **vs UnifiedReward**: UnifiedReward uses 236K samples but lacks quality control and is weak in I2T (37.2). Minos is 5.1 points higher at the same 8B scale due to filtering and DPO.
- **vs Prometheus-V**: Prometheus-V uses synthetic data without consistency filtering (Avg 20.3). Minos effectively pairs "GPT-as-annotator" with rigorous screening.
- **Insight**: Any project using "LLM-as-a-judge" should prioritize consistency checks over scaling data volume; this applies equally to current reward model training.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of quality control, bidirectional training, and delta-score DPO filter. The empirical "less is more" result is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 OOD datasets and 4 ablation tables with detailed comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and high table density. Strong motivation, though the method section is brief on DPO details.
- Value: ⭐⭐⭐⭐⭐ Provides both a SOTA open-source evaluator and a crucial negative result regarding data quality. Directly applicable to those working on reward/judge models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)

</div>

<!-- RELATED:END -->
