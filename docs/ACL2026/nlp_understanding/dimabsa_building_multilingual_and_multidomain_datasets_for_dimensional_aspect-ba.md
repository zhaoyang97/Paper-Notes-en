---
title: >-
  [Paper Note] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis
description: >-
  [ACL 2026][NLP Understanding][ABSA] The authors constructed DimABSA, the first multilingual (6 languages) and multi-domain (4 domains) dataset for dimensional aspect-based sentiment analysis (76,958 aspect instances / 42…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "ABSA"
  - "Dimensional Sentiment"
  - "Valence-Arousal"
  - "Multilingual"
  - "cF1"
date: 2026-05-08
content_hash: 61bf7b40e442333f
---

# DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis

**Conference**: ACL 2026  
**arXiv**: [2601.23022](https://arxiv.org/abs/2601.23022)  
**Code**: https://github.com/DimABSA/DimABSA2026 (Available)  
**Area**: Multilingual / Sentiment Analysis / Evaluation Benchmark  
**Keywords**: ABSA, Dimensional Sentiment, Valence-Arousal, Multilingual, cF1

## TL;DR
The authors constructed DimABSA, the first multilingual (6 languages) and multi-domain (4 domains) dataset for dimensional aspect-based sentiment analysis (76,958 aspect instances / 42,590 sentences). It replaces traditional "positive/negative/neutral" tri-classification with continuous valence–arousal scores, designs three new subtasks and a unified metric $cF1$, and provides a systematic evaluation of six open- and closed-source LLMs.

## Background & Motivation

**Background**: Traditional ABSA (Aspect-Based Sentiment Analysis) has followed a standard paradigm since SemEval-2014 consisting of (aspect term, aspect category, opinion term, polarity) quadruplets. The mainstream approach involves extraction + classification pipelines where labels are coarse-grained: positive, negative, or neutral.

**Limitations of Prior Work**: Coarse-grained labels cannot capture subtle nuances in sentiment intensity. For instance, "good" and "excellent" are both labeled as positive, while "a little slow" and "extremely slow" are both negative, despite significant differences in semantic intensity. Information regarding lexical intensity and affective modifiers (slightly, very, tremendously) is lost in polarity labels.

**Key Challenge**: Sentiment is inherently continuous. In affective science, Russell's circumplex model describes sentiment in a continuous two-dimensional space of valence $\times$ arousal. However, ABSA labels are discrete, resulting in (1) zero discriminative power for fine-grained differences within the same polarity and (2) difficulty in transferring across tasks like mood dynamics or mental health tagging.

**Goal**: Upgrade ABSA from "categorical prediction" to a hybrid "continuous dimensional regression + categorical extraction" task, ensuring (i) multilinguality (including low-resource languages), (ii) multi-domain coverage, and (iii) unified evaluation metrics.

**Key Insight**: By adopting the Self-Assessment Manikin (SAM) annotation protocol and emoji assistance from psychology, valence (1–9) and arousal (1–9) are treated as continuous labels. Each tuple is scored by five annotators, with outliers beyond $\pm 1.5\sigma$ removed before averaging, effectively reducing noise.

**Core Idea**: Replace the (A, C, O, polarity) quadruplet with an (A, C, O, V#A) quintuplet. A continuous F1 metric ($cF1$) is designed for the hybrid "extraction + regression" task, where categorical TP is only counted if the categories are entirely correct, after which the VA distance is converted into a soft score in the $[0,1]$ range.

## Method

DimABSA is not just a model, but a complete suite consisting of a **dataset, subtasks, evaluation metrics, and an LLM benchmark**. 

### Overall Architecture

The input consists of raw texts crawled from 12 real-world sources including Yelp, Amazon, Rakuten Travel, EDINET, SemEval-2016, SIGHAN-2024, Mobile01, and MOPS. It covers 6 languages (English, Japanese, Russian, Tatar, Ukrainian, Chinese) across 4 domains (restaurant, laptop, hotel, finance), totaling 10 sub-datasets. The pipeline involves two stages:

1.  **Triplet Extraction Stage**: Two annotators independently label (A, C, O). A third person adjudicates inconsistencies; otherwise, the instance is discarded.
2.  **VA Scoring Stage**: Using the SAM scale and VA emojis, five annotators provide V (1–9) and A (1–9) scores for each confirmed tuple, with the final score being the mean after $\pm 1.5\sigma$ filtering.

Tatar and Ukrainian are low-resource languages. Their data were obtained by machine translating Russian data via Yandex Translate followed by native speaker verification (45.5% of Tatar and 35.6% of Ukrainian data were manually revised).

### Key Designs

1.  **Dimensional Sentiment Annotation Protocol**:
    - **Function**: Upgrades each aspect tuple from $\text{polarity} \in \{\text{pos, neg, neu}\}$ to $(V, A) \in [1,9]^2$.
    - **Mechanism**: Valence measures positiveness (1 = extremely negative, 9 = extremely positive, 5 = neutral), while arousal measures activation (1 = calm, 9 = excited). The annotation interface uses SAM pictorial scales and emoji anchors. Final scores $\hat{r} = \mathrm{mean}(\{r_i : |r_i - \mu| \le 1.5\sigma\})$ automatically exclude outliers.
    - **Design Motivation**: Arousal is more difficult to label than valence. Prior research confirmed that multi-annotation and outlier removal can keep arousal RMSE within the 0.76–2.29 range. The data exhibits a "U-shaped distribution," where arousal is higher at valence extremes and lower at neutral points, consistent with affective science laws and validating annotation quality.

2.  **Three Progressive Subtasks (DimASR → DimASTE → DimASQP)**:
    - **Function**: Increases complexity from pure regression to hybrid "extraction + classification + regression" tasks.
    - **Mechanism**: (i) DimASR: Given text + aspect, predict V#A (pure regression, RMSE evaluation). (ii) DimASTE: Given text, extract (A, O) and predict VA (extraction + regression, $cF1$ evaluation). (iii) DimASQP: Extends DimASTE by adding aspect category (C) classification.
    - **Design Motivation**: Hierarchical design allows researchers to tackle specific capabilities. For example, DimASR targets numerical regression, while DimASTE/DimASQP targets structural induction. Differences and correlations between tasks reveal different LLM learning curves for regression vs. extraction.

3.  **Continuous F1 ($cF1$) Unified Metric**:
    - **Function**: Simultaneously evaluates "categorical exact matching" and "VA numerical error" within a standard F1 framework.
    - **Mechanism**: For a predicted tuple $t$, categorical TP is first determined (exact match of (A, O) or (A, C, O)). If it is a categorical TP, it is softened into a continuous TP: $\mathrm{cTP}^{(t)} = 1 - \mathrm{dist}(\mathrm{VA}_p, \mathrm{VA}_g)$. Here, normalized Euclidean distance $\mathrm{dist} = \sqrt{(V_p-V_g)^2 + (A_p-A_g)^2} / \sqrt{128}$, where $\sqrt{128}$ is the maximum distance in the $[1,9]^2$ space. $cPrecision = \sum \mathrm{cTP} / |P|$, $cRecall = \sum \mathrm{cTP} / |G|$, and $cF1$ is their harmonic mean.
    - **Design Motivation**: Standard F1 wastes continuous VA information through binarization, while reporting F1 and RMSE separately prevents single-number comparisons. $cF1$ converges to standard F1 when VA is perfect ($\mathrm{dist} = 0$) and penalizes results as VA error increases.

### Loss & Training
The study does not train a custom model but benchmarks existing ones:
-   **Zero/few-shot**: API access to GPT-5 mini and Kimi K2 Thinking. For few-shot, the first $k$ training samples are used as in-context examples.
-   **Supervised fine-tuning**: Qwen3-14B, Ministral-3-14B, Llama-3.3-70B, and GPT-OSS-120B utilize 4-bit QLoRA with AdamW, linear scheduler, $lr = 2e-5$, $batch = 4$, for 5 epochs on H200 hardware.

## Key Experimental Results

### Main Results: Comprehensive LLM Comparison across Languages and Subtasks

DimASR is evaluated using RMSE (lower is better), while DimASTE/DimASQP use $cF1$ (higher is better).

| Subtask | Dataset | GPT-5 mini (0-shot) | Kimi K2 (0-shot) | Llama-3.3 70B (FT) | GPT-OSS 120B (FT) |
|---------|---------|---------------------|------------------|---------------------|--------------------|
| DimASR (RMSE↓) | eng-rest | 2.949 | 2.343 | 2.524 | **1.461** |
| DimASR (RMSE↓) | jpn-hot | 3.141 | 2.329 | 2.626 | **0.719** |
| DimASR (RMSE↓) | zho-fin | 2.655 | 2.966 | 2.563 | **0.651** |
| DimASR (RMSE↓) | AVG (10 langs) | 2.760 | 2.344 | 2.567 | **1.192** |
| DimASTE (cF1↑) | eng-rest | 0.499 | 0.510 | 0.542 | **0.544** |
| DimASTE (cF1↑) | jpn-hot | 0.173 | 0.315 | 0.469 | **0.540** |
| DimASTE (cF1↑) | AVG | 0.353 | 0.379 | **0.464** | 0.457 |
| DimASQP (cF1↑) | eng-rest | 0.404 | 0.374 | **0.505** | 0.501 |
| DimASQP (cF1↑) | AVG | 0.225 | 0.254 | **0.386** | 0.373 |

Observations: (i) In DimASR, 120B SFT halves the RMSE, whereas 14B/70B models often underperform compared to prompting baselines. (ii) In DimASTE/DimASQP, 70B and 120B models perform similarly, while 14B is insufficient. (iii) Tatar remains the weakest, while Chinese/Japanese close the gap with English after fine-tuning.

### Ablation Study: Few-shot Prompts vs. $cF1$ (GPT-5 mini)

| Configuration | DimASR (avg RMSE) | DimASTE (avg cF1) | DimASQP (avg cF1) | Description |
|------|-------------------|--------------------|--------------------|------|
| 0-shot | 2.760 | 0.353 | 0.225 | No examples |
| 1-shot | 2.155 | 0.348 | 0.234 | DimASR improves significantly; structural tasks remain stable |
| 32-shot | ~1.9 (plateau) | ~0.40 | ~0.26 | All tasks reach a plateau |
| 256-shot | ~1.9 | ~0.41 | ~0.27 | Still weaker than 70B/120B FT baseline |

### Key Findings
- **Regression is sensitive to examples**: A single example can calibrate the VA numerical scale, but gains saturate after 32-shot.
- **Structural extraction requires scale and fine-tuning**: On DimASTE/DimASQP, 14B models show little improvement or even degradation after FT; qualitative shifts only occur at the 70B scale.
- **Performance drops with more categories**: DimASQP drops 0.07–0.1 $cF1$ compared to DimASTE on average, with larger drops in the laptop domain (148 categories) than restaurant (18 categories).
- **Arousal is harder than valence**: RMSE for arousal is higher than valence across all languages.

## Highlights & Insights
- **$cF1$ is an elegant single-value design for hybrid tasks**: By normalizing Euclidean distance and integrating it into TP counts, it retains the rigor of F1 while allowing for soft decay based on numerical error.
- **U-shaped VA distribution as a sanity check**: The consistent U-shape across 10 datasets validates annotation quality; the idea that distribution shapes can verify subjective labels is a valuable takeaway for annotation projects.
- **Regression vs. Extraction are distinct LLM capabilities**: DimASR requires numerical calibration (1-shot), while DimASTE requires structural induction (scale + FT).
- **Transparency in low-resource data**: Explicitly reporting revision ratios (Tatar 45.5%, Ukrainian 35.6%) provides a clear upper bound for translation noise.

## Limitations & Future Work
- **Incomparable cross-cultural sentiment interpretation**: Valence/arousal scales shift across cultures (e.g., more centralized in East Asia vs. polarized in European languages).
- **Structural signal compression in low-resource target languages**: Low $cF1$ in Tatar partly reflects inconsistencies in projected annotations from Russian rather than purely model capability.
- **Missing structural tasks in Finance**: The finance domain only includes DimASR, preventing a full comparison of task difficulty against the review domain.

## Related Work & Insights
- **vs. M-ABSA (Wu2025)**: M-ABSA is also multilingual but uses categorical polarity. DimABSA is broader in dimension (continuous VA) and domain.
- **vs. SIGHAN-2024 Chinese DimABSA**: This work is the 6-language, 4-domain expansion of that single-domain predecessor.
- **vs. NRC-VAD lexicon**: While NRC-VAD provides word-level VA, this work focuses on the aspect-level and integrates VA into the ABSA pipeline.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First multilingual multi-domain dimensional ABSA dataset + ingenious $cF1$ metric.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive benchmark across 10 datasets, 6 LLMs, and 3 subtasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with high information density in formulas and tables.
- **Value**: ⭐⭐⭐⭐⭐ Already attracts significant community interest as a SemEval-2026 track; data and code are open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[ACL 2026\] A Computational Method for Measuring "Open Codes" in Qualitative Analysis](a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)

</div>

<!-- RELATED:END -->
