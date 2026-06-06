---
title: >-
  [Paper Note] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology
description: >-
  [ACL 2026][LLM Evaluation][K-MetBench] The authors constructed K-MetBench, comprising 1,774 questions based on 25 years of the South Korean National Meteorological Engineer certification exams. Evaluating 55 LLMs/MLLMs a…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "K-MetBench"
  - "Skew-T diagram"
  - "Meteorological reasoning"
  - "Geo-cultural alignment"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: b8dea6e949f520a3
---

# K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology

**Conference**: ACL 2026  
**arXiv**: [2604.24645](https://arxiv.org/abs/2604.24645)  
**Code**: https://github.com/kmetbench/kmetbench-release  
**Area**: Meteorology LLM Evaluation / Multimodal Benchmark / Cultural Localization  
**Keywords**: K-MetBench, Skew-T diagram, Meteorological reasoning, Geo-cultural alignment, LLM-as-Judge

## TL;DR
The authors constructed K-MetBench, comprising 1,774 questions based on 25 years of the South Korean National Meteorological Engineer certification exams. Evaluating 55 LLMs/MLLMs across four orthogonal dimensions—"Multimodal Vision / Expert Reasoning / Geo-culture / Sub-domain Granularity"—revealed a widespread modality gap (average 18.6% accuracy drop on visual charts vs. text), a reasoning gap (correct answers with hallucinated rationales), and a geo-cultural gap (the small Korean model A.X-4.0 scored 78.9 on Korea-specific questions, surpassing the 235B Qwen3-VL's 72.6). This demonstrates that parameter scale alone cannot resolve cultural localization issues.

## Background & Motivation
**Background**: LLMs/MLLMs have reached passing thresholds for professional licenses in medicine (MedQA) and law (BarExam). In meteorology, benchmarks like ClimaQA, ClimateIQA, and WeatherQA have emerged, but they typically rely on single aggregate scores to measure model performance.

**Limitations of Prior Work**: The authors categorize issues in existing meteorological evaluations into four gaps: (1) **modality gap**—most benchmarks are text-only, whereas forecasting is inherently multimodal (surface maps, 500/200 hPa high-altitude charts, Skew-T Log-P diagrams, radar, etc.); (2) **reasoning gap**—evaluating only accuracy ignores whether the rationale is sound, meaning models may guess correctly via shortcut learning; (3) **geo-cultural gap**—meteorological rules, terrain, and KMA (Korea Meteorological Administration) standards are localized, which global models tend to abstract away; (4) **granularity gap**—single scores hide performance variances across sub-domains like "factual recall" vs. "quantitative reasoning" vs. "atmospheric dynamics."

**Key Challenge**: Meteorology is a composite task involving high stakes, multimodality, strong localization, and complex physical reasoning. Any single-dimensional benchmark offers only a partial view. To provide a credible diagnosis of whether a model can serve as an assistant to a Korean forecaster, all four dimensions must be evaluated simultaneously using authoritative certification exams as the source-of-truth.

**Goal**: (1) Create a diagnostic benchmark based on 25 years of official Korean National Meteorological Engineer exam questions; (2) Label every question across four dimensions (multimodal / reasoning rationale / Korean-specific / sub-domain); (3) Conduct experiments on 55 models to map diagnostic results for each capability; (4) Use LLM-as-a-Judge for rationale evaluation, validated against human experts.

**Key Insight**: Elevate professional LLM evaluation from simple pass/fail metrics to a diagnostic assessment of specific capability gaps, using national certification exams as the "gold standard" to ensure authority, objectivity, and localization.

**Core Idea**: Utilize multi-dimensional annotations, four orthogonal diagnostic subsets, and LLM-as-a-Judge with meta-validation to reshape meteorological LLM evaluation from an accuracy leaderboard into a "four-dimensional capability radar."

## Method

### Overall Architecture
The K-MetBench construction and evaluation pipeline consists of three stages: (1) **Data Construction**: 2,500 questions were extracted from official Korean National Meteorological Engineer exam PDFs (2003–2022). After deduplication (0.6 similarity threshold, preserving logic-reversal cases), 1,774 questions remained. Questions were rewritten and options randomized using Gemini-2.5-Pro to mitigate contamination, followed by a 14.88% manual audit. (2) **Four-Dimensional Labelling**: Questions were tagged across Modality (82 with images), Reasoning (141 with expert rationales), Geo-Cultural (73 Korea-specific), and Granularity (5 sub-disciplines P1-P5). (3) **Evaluation**: 55 models were tested zero-shot, recording hard accuracy and LLM-as-a-Judge rationale scores (Factuality/Logicality/Depth/Clarity axes).

### Key Designs

1. **Data Construction based on Authoritative National Certification + Multi-Stage Decontamination**:
    - **Function**: Ensures authoritative questions, full coverage, and resistance to data contamination.
    - **Mechanism**: (a) Source = Official Korean HRDK meteorological exam questions (25 years, 2003-2022), split into P1 Forecast Theory, P2 Observation, P3 Dynamics, P4 Climatology, and P5 Physics; (b) Deduplication via `difflib.SequenceMatcher` followed by manual audit; (c) Option randomization and stem rewriting via Gemini-2.5-Pro; (d) Converting formula images to LaTeX while preserving professional charts (Skew-T, isobar maps) to test true visual capabilities.
    - **Design Motivation**: National exams provide a clear pass threshold (60%) as a human baseline. Rewriting prevents scores from being artificially inflated by training set memorization.

2. **Orthogonal Diagnostic Subsets**:
    - **Function**: Decomposes professional capability into four independent axes to isolate model weaknesses.
    - **Mechanism**: (a) **Modality Subset** (82 questions) covers high-altitude charts and Skew-T diagrams to test extraction of pressure gradients and thermal indices; (b) **Reasoning Subset** (141 questions) provides expert-verified rationales as references; (c) **Geo-Cultural Subset** (73 questions) identifies Korea-specific concepts like Yeongdong terrain and KMA regulations; (d) **Granularity Subset** covers 5 official disciplines. These axes are orthogonal, allowing a question to belong to multiple subsets.
    - **Design Motivation**: Unlike coarse benchmarks, K-MetBench diagnostics can directly inform developers which specific capability (e.g., modality vs. reasoning) requires more data.

3. **LLM-as-a-Judge for Rationale Evaluation + Meta-evaluation Validation**:
    - **Function**: Provides a credible way to evaluate rationale quality without massive human expert labeling.
    - **Mechanism**: Gemini-2.5-Pro acts as a judge, scoring rationales on Factuality, Logicality, Depth, and Clarity (4-20 scale). For validation, 100 rationales were sampled and scored by two professors and Gemini. The result showed high agreement (Kendall's $\tau_b > 0.8$, Reasoning Total $\alpha = 0.838$).
    - **Design Motivation**: Expert time is expensive; a meta-validated LLM judge allows for cost-effective evaluation of thousands of rationales across dozens of models.

### Loss & Training
This work focuses on evaluation and does not involve training. Key settings: all models evaluated zero-shot with Korean prompts; final answers extracted via regex. Temperature was set to 0.1 (1.0 for reasoning models) with seed=42. Geo-Cultural evaluation used a matrix of Implicit/Explicit × Standard/Advanced prompts to ensure fairness for global models.

## Key Experimental Results

### Main Results: Comprehensive Ranking of 55 Models on K-MetBench

| Model | Acc. | Reas. (4-20) | Geo-Cult. (Korea) | Modality (Multi) | P1 / P2 / P3 / P4 / P5 |
|------|------|-------------|-----------------|------------------|------------------------|
| **Gemini-3-Pro-Preview (Thinking)** | **93.7** | **18.01** | **90.4** | **75.6** | 92.5/97.9/94.2/92.8/91.6 |
| GPT-5.2 (Thinking) | 87.8 | 17.33 | 80.8 | 29.3 | 86.3/93.4/88.0/86.2/85.3 |
| Qwen3-VL-235B-A22B-Thinking | 84.4 | 17.22 | 72.6 | 48.8 | 81.5/88.6/87.2/83.2/82.0 |
| Qwen3-VL-32B-Thinking | 78.6 | 16.19 | 60.3 | 51.2 | 74.3/85.2/78.8/78.7/76.3 |
| **A.X-4.0 (72B, Korean)** | 76.1 | 15.46 | **78.9** | – | 76.6/77.7/68.2/**81.3**/76.5 |
| GPT-OSS-120B | 77.3 | 16.12 | 62.0 | – | 72.5/85.8/76.5/77.4/74.9 |
| InternVL3.5-38B | 57.3 | 11.38 | 47.9 | 40.2 | 56.0/64.8/48.7/61.4/55.7 |
| Llama-3.2-90B-Vision | 56.9 | 9.72 | 52.1 | 30.5 | 57.1/59.3/52.4/62.2/53.3 |
| EXAONE-4.0-32B (Korean) | 59.9 | 13.57 | 59.2 | – | 58.2/64.8/52.4/63.1/61.2 |

### Ablation Study: Quantifying the Gaps

| Gap | Metric | Key Figure |
|------|---------|--------|
| Modality gap | Multimodal vs. Text-Only Acc Gain | Avg. drop **−18.55%**; Gemini-3-Pro only 75.6% vs. 93.7% overall |
| Reasoning gap | Answer Acc vs. Rationale Score | Kendall’s $\tau_b = 0.78$; rationales show term hallucination |
| Geo-Cultural gap | Qwen3-VL-235B vs. A.X-4.0 (72B) on Korea-specific | 72.6 vs. **78.9** (72B Korean model beats 235B global model) |
| Granularity gap | A.X-4.0 on P3 vs. P4 | P3 (Dynamics) 68.2 vs. P4 (Climatology) 81.3, gap of 13 pts |
| Benchmark Orthogonality | K-MetBench Multimodal vs. KMMLU-Redux | $\tau_b = 0.29$ (weak correlation, proving new signal) |
| Meta-evaluation | Human-LLM Reasoning Total $\alpha$ | **0.838** (above 0.7 threshold, valid proxy) |

### Key Findings
- **Modality is the biggest weakness for top models**: Radar charts for all models show an indentation on the Multimodal axis. Even Gemini-3-Pro drops to 75.6%, indicating that general vision training fails for technical meteorological charts.
- **Reasoning shows "Correct Answer, Wrong Process"**: Models often generate correct answers alongside rationales with hallucinated terminology, revealing shortcut learning.
- **Scale does not solve geo-cultural issues**: The 72B Korean model A.X-4.0 outperformed the 235B Qwen3-VL on localized questions, confirming that massive parameter scale does not compensate for lack of region-aware training.
- **Granularity is imbalanced**: Models excel in P2 Observation (factual recall) but struggle in P3/P5 Dynamics and Physics (quantitative reasoning).
- **Weak correlation with existing benchmarks**: A low correlation with ClimaQA/WeatherQA ($\tau < 0.14$) proves K-MetBench captures unique professional logic and visual interpretation signals.

## Highlights & Insights
- **Diagnostic Radar Paradigm**: Moving from a simple leaderboard to a four-dimensional gap analysis provides a template for other high-stakes domains (medicine, law).
- **Localization Rigor**: Separating "linguistic ambiguity" from "knowledge deficit" via implicit/explicit prompting ensures a fair evaluation of geo-cultural capabilities.
- **Pragmatic Data Engineering**: Preserving original charts while converting formulas to LaTeX effectively tests modality without introducing OCR noise into text reasoning.
- **Validated LLM-as-Judge**: Proving that meta-validated LLM judges can match experts in professional domains provides a low-cost pathway for large-scale evaluation of reasoning traces.

## Limitations & Future Work
- **Static Vision**: Only single charts are tested; time-series reasoning (e.g., radar loops) is not yet included.
- **Regional Focus**: While the design is generalizable, the instances are specific to South Korea. Benchmarks for other regions need to follow this pattern.
- **Human Ceiling**: A 60% passing score is used as a baseline, but the actual performance limit of top-tier professional forecasters has not been quantified.
- **Contamination Control**: While rewriting is used, the exact reduction in contamination-driven score inflation was not explicitly measured via comparison experiments.

## Related Work & Insights
- **vs. KMMLU**: KMMLU treats meteorology as one of 45 general disciplines; K-MetBench provides much deeper multimodal and reasoning analysis.
- **vs. ClimaQA**: While ClimaQA uses textbook questions, K-MetBench uses official professional certification exams, making it more representative of real-world forecasting tasks.
- **vs. ClimateIQA/WeatherQA**: These rely on template-generated data in English; K-MetBench introduces expert-verified rationales and localized Korean context which these benchmarks overlook.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[ACL 2026\] Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement](aggregate_vs_personalized_judges_in_business_idea_evaluation_evidence_from_exper.md)

</div>

<!-- RELATED:END -->
