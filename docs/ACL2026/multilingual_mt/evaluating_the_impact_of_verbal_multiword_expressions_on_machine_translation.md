---
title: >-
  [Paper Note] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][VMWE] This paper presents the first systematic evaluation of the impact of Verbal Multiword Expressions (VMWEs: Verbal Idioms VID, Verb-Particle Constructions VPC…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "VMWE"
  - "Verbal Idioms"
  - "Verb-particle"
  - "Light Verb"
  - "xCOMET"
  - "MetricX"
date: 2026-05-08
content_hash: a1f894c50af44ff2
---

# Evaluating the Impact of Verbal Multiword Expressions on Machine Translation

**Conference**: ACL 2026  
**arXiv**: [2508.17458](https://arxiv.org/abs/2508.17458)  
**Code**: <https://github.com/cincynlp/vmwe-mt-eval>  
**Area**: Machine Translation / Multilingual / Evaluation  
**Keywords**: VMWE, Verbal Idioms, Verb-particle, Light Verb, xCOMET, MetricX

## TL;DR
This paper presents the first systematic evaluation of the impact of Verbal Multiword Expressions (VMWEs: Verbal Idioms VID, Verb-Particle Constructions VPC, Light Verb Constructions LVC) on machine translation quality. Testing across 8 MT systems × 7 language pairs using two types of QE models and human DA scores, the study proves that VMWEs universally cause performance degradation. This degradation is strictly positively correlated with "non-compositionality" (VID > VPC > LVC), a regression that even GPT-4.1/GPT-5.1 cannot eliminate.

## Background & Motivation

**Background**: Although MT quality has surged in the last five years due to LLMs, the translation community has long recognized that certain linguistic phenomena—such as structural differences, morphologically complex words, and MWEs—remain challenging. Prior works either studied idiom translation in the era of statistical MT or performed case studies on Chinese-English pairs, lacking systematic evaluation in the modern neural MT and LLM era.

**Limitations of Prior Work**: (i) VMWEs are inherently highly non-compositional (e.g., "spill the beans" is "leak a secret," not "spill legumes"), and models often translate word-for-word, losing the semantics. (ii) Existing MWE evaluations either cover a single type or rely on metrics like BLEU, which are insensitive to phrase-level semantics. (iii) Confounding variables are not controlled; it remained unclear whether degradation was caused by VMWEs themselves or the inherent complexity/length of the sentences containing them.

**Key Challenge**: To prove that "VMWEs themselves degrade MT," one must simultaneously: cover three typical types of VMWEs, span multiple MT systems, validate with both reference-free QE and human DA, and use regression to control for confounders such as sentence length, polysemy, and structural complexity. No prior work has addressed all four requirements.

**Goal**: (i) Quantify degradation across three VMWE types (VID / VPC / LVC) × 7 language pairs × 8 MT systems. (ii) Validate using both specialized VMWE datasets and real-world WMT evaluation data. (iii) Use xCOMET error spans to locate whether errors actually fall on VMWE tokens. (iv) Use regression to prove VMWE is a significant negative predictor even after controlling for sentence difficulty.

**Key Insight**: The authors observe that the three classes of VMWEs have a natural gradient of non-compositionality (VID: completely non-derivable > VPC: semi-derivable > LVC: semantics hosted mainly by the noun). This serves as a natural control variable for the "translation degradation vs. compositionality" hypothesis.

**Core Idea**: A "five-fold chain" evaluation framework—dual datasets (VMWE-specific + WMT) × dual evaluation (QE + DA) × dual analysis (error span + regression control)—transforms the impact of VMWEs from an industrial intuition into a statistically rigorous attribution.

## Method

### Overall Architecture
The evaluation follows two parallel paths. Path A uses "Specialized VMWE Datasets": idioms from EPIE/MAGPIE, VPCs from Tu (2012), LVCs from Tu-Roth (2011), and a control group from BNC. 8 MT systems translate these into 7 target languages, and scores are provided by MetricX-24-QE and xCOMET-QE, calculating $\delta = \text{score}_{\text{VMWE}} - \text{score}_{\text{non-VMWE}}$. Path B uses "WMT Evaluation Data": VMWE sentences are extracted from WMT2017-2024 source sentences using heuristics and GPT-4o, comparing VMWE vs. non-VMWE using historical human DA scores. Both paths report $\delta$ heatmaps.

### Key Designs

1.  **Linguistic Classification & Non-compositionality Gradient Hypothesis**:
    *   Function: Decomposes the broad concept of VMWE into three quantitatively comparable difficulty Tiers.
    *   Mechanism: VID (Verbal Idiom) like "spill the beans" is non-derivable; VPC (Verb-Particle) like "give up" is semi-derivable (particle modifies verb meaning); LVC (Light Verb) like "take a walk" carries semantics in the noun. Each class is paired with high-quality datasets, and control groups are constructed using spaCy dependency parsing and idiom dictionaries.
    *   Design Motivation: If MT degradation stems from non-compositionality, a gradient of VID > VPC > LVC should be observed. Experimental results (Table 2 error overlap %) confirm this: Opus shows 78.64% error overlap for VID, 65.51% for VPC, and 62.21% for LVC.

2.  **Two-step VMWE Extraction on WMT (Heuristics + GPT-4o disambiguation)**:
    *   Function: High-precision identification of VMWE sentences in WMT data lacking gold labels.
    *   Mechanism: Step 1 uses heuristic recall (EPIE/MAGPIE dictionaries for idioms, spaCy `prt` relations for VPC/LVC). Step 2 uses GPT-4o with Chain-of-Thought prompts following PARSEME guidelines for disambiguation. F1 scores for VID/VPC/LVC reached 81.8, 80.0, and 81.6 respectively (Table 1).
    *   Design Motivation: WMT data provides human DA scores and ecological validity, but lacks VMWE labels. This pipeline enables large-scale real-world evaluation and is open-sourced for reuse.

3.  **Error Span Localization + Regression Control**:
    *   Function: Strictly attributes "poor translation of VMWE sentences" to VMWE tokens rather than length or polysemy.
    *   Mechanism: xCOMET token-level error spans are used alongside `simalign` for bilingual alignment to count how many error spans fall on target tokens corresponding to source VMWE phrases (Table 2). A linear regression $$\text{score}_i = \beta_0 + \beta_1 I_{vmwe} + \beta_2 S_{len} + \beta_3 P_{deg} + \beta_4 T_{cmp} + \varepsilon_i$$ is performed on 300,000 segments, controlling for VMWE indicators, sentence length, word sense polysemy (WordNet senses), and structural complexity (dependency arc length).
    *   Design Motivation: This addresses the most critical confounder—that VMWE sentences might just be inherently more difficult. Regression shows $\beta_{vmwe}$ is -0.0813 for xCOMET and +0.9954 for MetricX ($p<0.001$), proving VMWEs contribute significant additional degradation.

### Loss & Training
This is a pure evaluation paper; no models are trained. QE evaluation uses MetricX-24-QE (mT5-based, lower is better, 0-25) and xCOMET-QE (XLM-RoBERTa-XL-based, higher is better, 0-1). MT systems include SeamlessM4T, Madlad400, M2M100, Opus-MT, LLaMAX3 Alpaca, Phi-4-multimodal, GemmaX2, and Google Translate API. Target languages are de/cs/ru/zh/es/ja/tr. A controlled subset of 100 sentences × 4 categories × 4 languages was further evaluated with GPT-4.1 and GPT-5.1.

## Key Experimental Results

### Main Results

| MT System | VID error overlap (%) | VPC (%) | LVC (%) | xCOMET Avg |
|-----------|-----------------------|---------|---------|------------|
| Opus      | 78.64                 | 65.51   | 62.21   | 66.89      |
| M2M       | 82.15                 | 67.67   | 62.19   | 73.02      |
| Phi-4-m   | 66.50                 | 54.43   | 56.36   | 74.68      |
| Madlad    | 69.88                 | 52.79   | 50.14   | 78.22      |
| Seamless  | 67.91                 | 56.30   | 55.75   | 76.43      |
| LLaMAX    | 66.82                 | 53.11   | 55.89   | 78.08      |
| GemmaX2   | 55.41                 | 46.77   | 43.90   | 85.69      |
| Google API| 52.98                 | 40.77   | 39.25   | 87.19      |

General Trend: As MT system strength increases (higher xCOMET), the proportion of errors falling on VMWEs decreases; however, all systems exhibit the VID > VPC > LVC degradation gradient.

### Ablation Study (Regression Control)

| Predictor | xCOMET β (SE) | MetricX β (SE) | Interpretation |
|-----------|---------------|----------------|----------------|
| $\beta_0$ (intercept) | 0.7908 (0.0010)*** | 5.6889 (0.0183)*** | Expected score for non-VMWE |
| $I_{vmwe}$ | **-0.0813** (0.0012)*** | **+0.9954** (0.0240)*** | Additional degradation from VMWE |
| $S_{len}$ | -0.0359 (0.0012)*** | +0.6348 (0.0240)*** | Effect per 1 SD length increase |
| $P_{deg}$ | -0.0126 (0.0006)*** | +0.0244 (0.0110)* | Effect of polysemy |
| $T_{cmp}$ | -0.0120 (0.0009)*** | +0.0059 (0.0210) ns | Effect of structural complexity |

$N=305{,}428$ segment-level regression. The VMWE coefficient is significantly larger than other difficulty factors, proving degradation is primarily due to VMWEs themselves.

### Key Findings
- **Non-compositionality Gradient Holds**: Across all 8 MT systems and top-tier LLMs (GPT-4.1/5.1), VID degradation is the highest and LVC the lowest. Table 4 shows GPT-4.1's average $\delta=+0.10$ for VID vs. +0.01 for VPC/LVC.
- **Strong LLMs Cannot Resolve Idioms**: Even GPT-5.1 fails to eliminate VID degradation (e.g., en-cs $\delta=+0.22$), indicating that the "literal-by-default" tendency is deeply rooted and cannot be solved by scale alone.
- **Human Translation is Unaffected**: DA score differences between VMWE and non-VMWE in human WMT translations are negligible, proving degradation is a fundamental limitation of MT systems, not the task.
- **Error Span Correlates with QE**: Systems with errors more concentrated on VMWEs have lower overall QE scores, suggesting VMWE handling is a leading indicator of MT quality.
- **LLM-based MT Sideline Effects**: GemmaX2 showed >75% failure rates (wrong language or empty output) on ja/cs/tr, highlighting hidden brittleness in LLM-based MT.

## Highlights & Insights
- **"Five-fold Chain" Framework**: The design of dual sources, dual evaluations, and dual analyses ensures conclusions are supported by independent evidence, setting a methodological benchmark for idiom and metaphor research.
- **Error Span Overlap as a Leading Indicator**: Table 2 reveals that xCOMET scores are almost monotonically inversely correlated with VMWE error overlap. This suggesting that reward shaping or DPO targeting "error reduction on MWEs" could improve overall MT quality.
- **Regression + Simalign Paradigm**: Proving linguistic hypotheses through statistical causal analysis combined with token-level alignment is significantly more persuasive than reporting BLEU differences alone.

## Limitations & Future Work
- Evaluation covers only 7 languages (all translated from English); conclusions may not generalize to non-Indo-European or low-resource languages like Swahili.
- The specialized VMWE datasets lack human reference scores, relying only on QE models.
- No remediation method is proposed; the paper identifies the problem without providing a solution (e.g., sampling VMWE sentences or MWE-aware reranking).
- Other non-literal phenomena like metonymy or metaphor were not covered.

## Related Work & Insights
- **vs. Song & Xu (2024)**: While they analyzed errors in idioms and named entities for ZH-EN MT, this work scales to 8 systems, 3 VMWE types, and 7 language pairs with regression-based attribution.
- **vs. Baziotis et al. (2023) / Zaninello & Birch (2020)**: They focused on improvement methods, whereas this work provides the systematic framework to benchmark whether such improvements actually resolve the VMWE problem.
- **vs. PARSEME shared task**: Naturally extends standardized VMWE identification tasks into downstream MT evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The evaluation dimension is known, but the "five-fold chain" design and rigorous regression control are significant methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Massive engineering effort: 8 MT × 7 Languages × 3 VMWE types × 2 Sources × 2 QE + regression on 300k segments + GPT-4.1/5.1.
- Writing Quality: ⭐⭐⭐⭐ Clear takeaways per section, though dense tables require switching between the main text and appendix.
- Value: ⭐⭐⭐⭐ The open-source pipeline allows any future MT system to be benchmarked, establishing a standard for VMWE-related MT research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)

</div>

<!-- RELATED:END -->
