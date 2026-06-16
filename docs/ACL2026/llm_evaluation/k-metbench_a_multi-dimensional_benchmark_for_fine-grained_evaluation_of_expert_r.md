---
title: >-
  [Paper Note] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology
description: >-
  [ACL 2026][LLM Evaluation][K-MetBench] The authors constructed K-MetBench, containing 1,774 questions based on 25 editions of the South Korean National Meteorological Engineer certification exams. Evaluating 55 LLMs/MLLMs across four orthogonal dimensions—"Multimodal Vision / Expert Reasoning / Geo-cultural / Sub-domain Granularity"—the study reveals a univ
tags:
  - ACL 2026
  - LLM Evaluation
  - K-MetBench
  - LLM-as-Judge
date: 2026-05-08
content_hash: 3a725a36408c5474
---
# K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.24645](https://arxiv.org/abs/2604.24645)  
**Code**: https://github.com/kmetbench/kmetbench-release  
**Area**: Meteorological LLM Evaluation / Multimodal Benchmark / Cultural Localization  
**Keywords**: K-MetBench, Skew-T Diagram, Meteorological Reasoning, Geo-cultural Alignment, LLM-as-Judge  

## TL;DR
The authors constructed K-MetBench, containing 1,774 questions based on 25 editions of the South Korean National Meteorological Engineer certification exams. Evaluating 55 LLMs/MLLMs across four orthogonal dimensions—"Multimodal Vision / Expert Reasoning / Geo-cultural / Sub-domain Granularity"—the study reveals a universal **modality gap** (an average 18.6% drop in accuracy for visual meteorological charts compared to text), a **reasoning gap** (correct answers with hallucinated rationales), and a **geo-cultural gap** (the smaller local model A.X-4.0 outperformed the 235B Qwen3-VL 78.9 to 72.6 on Korean-specific questions). This demonstrates that parameter scale alone cannot resolve cultural localization issues.

## Background & Motivation
**Background**: LLMs/MLLMs have reached passing thresholds for licensing exams in professional fields (e.g., MedQA for medicine, BarExam for law). In the meteorological domain, benchmarks like ClimaQA, ClimateIQA, and WeatherQA have emerged, but they typically use a single aggregate score to measure model capability.

**Limitations of Prior Work**: The authors categorize issues in existing meteorological evaluations into four gaps: (1) **modality gap**—most benchmarks are text-only, whereas forecasting is inherently multimodal (surface maps, 500/200 hPa upper-air charts, Skew-T Log-P diagrams, radar, etc.), leading to severe underestimation of visual weaknesses; (2) **reasoning gap**—evaluating only answer accuracy without assessing rationale allows models to guess correctly through shortcut learning; (3) **geo-cultural gap**—meteorological rules, topography, and KMA (Korea Meteorological Administration) standards are localized, and global models often abstract this information away; (4) **granularity gap**—a single total score masks differences across sub-domains like "fact recall" vs. "quantitative reasoning" vs. "atmospheric dynamics."

**Key Challenge**: Meteorological work is a complex task involving high stakes, multimodality, strong localization, and physical reasoning. Any single-dimensional benchmark captures only one facet. To provide a truly credible diagnosis of whether a model can assist a Korean forecaster, all four dimensions must be evaluated together using official certification exams as the source-of-truth.

**Goal**: (1) Create a diagnostic benchmark based on 25 editions of the Korean National Meteorological Engineer exam across 5 major sub-disciplines; (2) Tag each question with four-dimensional labels (multimodal / reasoning rationale / Korean-specific / sub-domain); (3) Conduct experiments on 55 models to provide a diagnostic map of model failures; (4) Use LLM-as-a-Judge to evaluate rationales and prove its consistency with human experts through meta-evaluation.

**Key Insight**: Elevate professional LLM evaluation from "passing a threshold" to a diagnostic assessment of "which sub-capabilities failed and why," using official certification questions as a "gold standard" to ensure authority, objectivity, and localization.

**Core Idea**: By using multi-dimensional annotation, four orthogonal diagnostic subsets, and LLM-as-a-Judge with meta-validation, the authors reshape meteorological LLM evaluation from an accuracy ranking into a "4D capability radar" that exposes gaps in modality, reasoning, geo-cultural knowledge, and granularity.

## Method

### Overall Architecture
K-MetBench transforms meteorological evaluation into a "4D capability radar" via a pipeline consisting of data construction, 4D annotation, and evaluation. 2,500 questions were extracted from official Korean National Meteorological Engineer exam PDFs (2003–2022). After deduplication using `difflib.SequenceMatcher` (threshold 0.6), 1,774 questions remained. To prevent contamination, stems were rewritten and options randomized using Gemini-2.5-Pro (with 14.88% manual verification). For multimodal items, OCR artifacts were fixed and formula images converted to LaTeX, while professional meteorological charts were preserved as original images. Each question was then tagged with orthogonal labels: Modality (82 image-based), Reasoning (141 with expert rationales), Geo-Cultural (73 Korea-specific), and Granularity (5 sub-disciplines P1–P5). 55 models were evaluated via zero-shot prompting in the original Korean to avoid translation artifacts.

### Key Designs

**1. Data Construction via Authoritative Certification + Multi-stage Decontamination**  
Selecting official Korean HRDK exams (P1 Forecasting Theory / P2 Observation / P3 Atmospheric Dynamics / P4 Climatology / P5 Atmospheric Physics) provides three benefits: a 60% passing line as a human anchor, professional authority, and inherent localization. To prevent data leakage, stems were rewritten under strict constraints to preserve technical terms while breaking memory-based patterns. A hybrid approach was used for visual content: formula images were converted to LaTeX to bypass OCR bottlenecks, while Skew-T and isobaric charts were kept as images to test true visual capabilities.

**2. Orthogonal Diagnostic Subsets: Exposing Specific Weaknesses**  
Professional capability is decomposed into four axes. The **Modality** subset (82 items) covers surface and upper-air charts and Skew-T diagrams to test extraction of pressure gradients and thermodynamic indices. The **Reasoning** subset (141 items) includes reference rationales drafted by GPT-5 and verified by meteorology professors. The **Geo-Cultural** subset (73 items) identifies "Korea-specific" concepts (e.g., Yeongdong region, KMA regulations, Changma front). The **Granularity** subset reflects the official 5 disciplines. This diagnostic approach tells developers specifically what data needs improvement (e.g., "modality is 10 points lower than reasoning").

**3. LLM-as-a-Judge for Rationales + Meta-evaluation**  
To evaluate 7,755 rationales (55 models × 141 questions), Gemini-2.5-Pro was used as a judge, scoring across Factuality, Logicality, Depth, and Clarity (Total 4–20). Meta-evaluation was performed by comparing the judge’s scores with those of two human experts on 100 samples. Results showed Kendall's $\tau_b > 0.8$ and Reasoning Total $\alpha=0.838$, proving that LLM-as-a-Judge can reliably proxy human experts in professional domains.

## Key Experimental Results

### Main Results: Comprehensive Ranking Across 55 Models

| Model | Acc. | Reas. (4-20) | Geo-Cult. (KR) | Modality (Multi) | P1 / P2 / P3 / P4 / P5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemini-3-Pro-Preview (Thinking)** | **93.7** | **18.01** | **90.4** | **75.6** | 92.5/97.9/94.2/92.8/91.6 |
| GPT-5.2 (Thinking) | 87.8 | 17.33 | 80.8 | 29.3 | 86.3/93.4/88.0/86.2/85.3 |
| Qwen3-VL-235B-A22B-Thinking | 84.4 | 17.22 | 72.6 | 48.8 | 81.5/88.6/87.2/83.2/82.0 |
| Qwen3-VL-32B-Thinking | 78.6 | 16.19 | 60.3 | 51.2 | 74.3/85.2/78.8/78.7/76.3 |
| **A.X-4.0 (72B, Korean)** | 76.1 | 15.46 | **78.9** | – | 76.6/77.7/68.2/**81.3**/76.5 |
| GPT-OSS-120B | 77.3 | 16.12 | 62.0 | – | 72.5/85.8/76.5/77.4/74.9 |
| InternVL3.5-38B | 57.3 | 11.38 | 47.9 | 40.2 | 56.0/64.8/48.7/61.4/55.7 |

### Ablation Study: Quantifying the Gaps

| Gap | Metric | Key Figure |
| :--- | :--- | :--- |
| Modality gap | Multimodal vs. Text-Only Acc | Avg. drop of **−18.55%** |
| Reasoning gap | Accuracy vs. Rationale score | $\tau_b = 0.78$; rationales show hallucinations |
| Geo-Cultural gap | Qwen3-VL-235B vs. A.X-4.0 (72B) | 72.6 vs. **78.9** (72B local model wins) |
| Granularity gap | A.X-4.0: P3 vs. P4 | P3 (Dynamics) 68.2 vs. P4 (Climatology) 81.3 |
| Orthogonality | K-MetBench vs. KMMLU-Redux | $\tau_b = 0.29$ (Weakly correlated) |

### Key Findings
- **Modality is the biggest bottleneck**: Even top models like Gemini-3-Pro Thinking drop to 75.6% on multimodal tasks compared to 90%+ on text, showing that general visual training is insufficient for specialized meteorological charts.
- **Right Answer, Wrong Process**: Models often provide correct choices with rationales containing hallucinated technical terms, suggesting shortcut learning.
- **Scale does not solve locality**: A.X-4.0 (72B local model) significantly outperformed Qwen3-VL-235B on Korean-specific questions, proving that 3x parameters cannot compensate for missing localized knowledge.
- **Sub-discipline Variance**: Models perform best on P2 (Fact Recall) and worst on P3/P5 (Quantitative Reasoning).
- **Metric Validity**: Higher $\alpha=0.838$ confirms Gemini-2.5-Pro as a credible surrogate for human experts in professional domains.

## Highlights & Insights
- **The "4D Diagnostic Radar" Paradigm**: Shifting from a single leaderboard to a multi-dimensional map is a paradigm shift in benchmark design, applicable to other high-stakes fields like medicine or finance.
- **Geo-Cultural Isolation**: Using implicit/explicit prompt protocols separates "linguistic ambiguity" from "knowledge deficiency," contributing to evaluation fairness.
- **Data Engineering Best Practices**: Converting formulas to LaTeX while keeping charts as images effectively balances OCR reliability with multimodal testing.
- **Local Model Superiority**: Empirical evidence that 72B local models can beat 235B global models on regional tasks provides a strong case for region-aware fine-tuning.

## Limitations & Future Work
- **Static Vision**: Only single charts are tested, omitting time-series reasoning like radar loops or satellite animations.
- **Korean Specificity**: While the paradigm is robust, the current data is focused on South Korea; benchmarks for other regions need equivalent construction.
- **Human Ceiling**: A passing score of 60% is used as a human anchor, but the performance of top-tier professional forecasters remains unquantified.
- **Future Directions**: Expansion to sequential multimodal data, establishing similar benchmarks in other languages (Chinese/English), and using K-MetBench rationales for reasoning-trace fine-tuning.

## Related Work & Insights
- **vs. KMMLU**: KMMLU is broader; K-MetBench is deeper in meteorology with added multimodal and reasoning dimensions.
- **vs. ClimaQA**: ClimaQA is text-only and textbook-based; K-MetBench uses professional exams and multimodal data.
- **vs. MedQA/BarExam**: Adopted the "authority-driven" benchmark strategy while introducing the 4D diagnostic framework.

## Rating
- Novelty: ⭐⭐⭐⭐ (First 4D diagnostic approach for meteorology)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (55 models, extensive meta-evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear organization and intuitive visualizations)
- Value: ⭐⭐⭐⭐⭐ (A replicable paradigm for professional domain evaluation)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ACL 2026\] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control](locar_localization-aware_evaluation_of_in-vehicle_assistants_through_fine-graine.md)
- [\[ACL 2026\] Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement](aggregate_vs_personalized_judges_in_business_idea_evaluation_evidence_from_exper.md)

</div>

<!-- RELATED:END -->
