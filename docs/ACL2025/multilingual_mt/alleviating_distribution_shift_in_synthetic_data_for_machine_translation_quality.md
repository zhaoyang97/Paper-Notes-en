---
title: >-
  [Paper Note] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Quality Estimation] The DCSQE framework is proposed to effectively alleviate the distribution shift in synthetic QE data through constrained beam search for generating more realistic synthetic translations, an independent annotator model to correct label bias, and the SPCE algorithm to aggregate token-level labels into phrase-level labels. It outperforms state-of-the-art baselines like CometKiwi in both supervised and unsupervise…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Quality Estimation"
  - "Distribution Shift"
  - "Synthetic Data"
  - "MQM Annotation"
  - "Translation Quality Estimation"
date: 2026-05-08
content_hash: 29d7619fbe24063b
---

# Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation

**Conference**: ACL 2025  
**arXiv**: [2502.19941](https://arxiv.org/abs/2502.19941)  
**Code**: [https://github.com/NJUNLP/njuqe](https://github.com/NJUNLP/njuqe)  
**Area**: Multilingual Translation  
**Keywords**: Quality Estimation, Distribution Shift, Synthetic Data, MQM Annotation, Translation Quality Estimation

## TL;DR

The DCSQE framework is proposed to effectively alleviate the distribution shift in synthetic QE data through constrained beam search for generating more realistic synthetic translations, an independent annotator model to correct label bias, and the SPCE algorithm to aggregate token-level labels into phrase-level labels. It outperforms state-of-the-art baselines like CometKiwi in both supervised and unsupervised settings.

## Background & Motivation

Quality Estimation (QE) aims to evaluate machine translation quality without relying on reference translations, serving as a reward model for translation tasks. Multidimensional Quality Metrics (MQM) annotation is the current mainstream standard for QE, providing fine-grained error spans and severity information. However, human annotation is extremely costly, resulting in small dataset scales and limited language coverage.

To address data scarcity, previous works have attempted to generate synthetic MQM data from parallel corpora. However, existing methods face severe **distribution shift** issues:

**MQMQE** method: By randomly masking segments in reference translations and replacing them with negative samples from a translation model, the generated synthetic translations suffer from poor fluency. Moreover, using the same translation model to annotate its own output leads to overconfidence issues, causing misalignment between labels and human preferences.

**InstructScore** method: Utilizing GPT-4 prompts to generate errors yields fluent translations, but the generated errors are unnatural, and relying on closed-source LLMs is highly expensive.

These issues not only degrade QE performance but also impact downstream human preference optimization.

## Method

### Overall Architecture

The DCSQE (Distribution-Controlled Data Synthesis for QE) framework consists of two core stages: **generating more realistic synthetic translations** and **annotating more accurate synthetic labels**.

First, two independent translation models are trained: the Generator and the Annotator. The Generator employs constrained beam search to generate synthetic translations, while the Annotator performs fine-grained label correction based on generation probabilities.

### Key Designs

**1. Constrained Beam Search (CBS) for Synthetic Translation Generation**

Unlike standard beam search, CBS retains tokens from the reference translation whose generation probabilities exceed a threshold during decoding, avoiding synonymous substitutions and making errors more natural. CBS preserves the essential structure of the reference translation, yielding translation errors closer to those produced by actual translation models.

**2. Model Diversity Enhancement**

To improve the diversity of synthetic translations, the authors train multiple Generators (such as L and L') on different subsets of parallel corpora, enabling them to produce outputs with distinct styles but similar translation quality. In experiments, the BLEU score between the two generators on Flores-200 is only 80.06, indicating significant diversity.

**3. Coarse-grained Labeling: TER Alignment**

The Translation Edit Rate (TER) tool is utilized to align synthetic translations with reference translations at the word level. Matched parts are labeled as "OK", while mismatched parts are labeled as "BAD".

**4. Fine-grained Labeling: Annotator Correction**

For tokens labeled as "BAD" by TER, an independent Annotator model rehabilitees them based on their generation probabilities. By setting three ascending thresholds (tMINOR, tMAJOR, tCRITICAL), generation probabilities are mapped to four severity levels: MINOR, MAJOR, CRITICAL, and OK.

Key Insight: **A model cannot accurately annotate its own output**. Translation models exhibit overconfidence in their own generated text, leading to an extremely low ratio of error labels (only 0.11%–1.60%). Therefore, an independent Annotator must be employed.

**5. Enhancing Annotator with Supervised Signals**

Ensuring that the Annotator is trained on the parallel corpora used for synthetic data generation endows it with professional-grade annotation capabilities on these datasets.

**6. SPCE Algorithm (Shortest Phrase Covering Error)**

Human annotators tend to annotate complete phrases rather than fragmented tokens. The SPCE algorithm achieves token-to-phrase aggregation using dependency trees:
- Construct a candidate set for contiguous BAD tokens.
- Identify the Lowest Common Ancestor (LCA) in the dependency tree.
- Fill in the tokens along the path and intermediate tokens.
- Iterate until the candidate set stabilizes.
- Use the maximum severity of the candidate tokens as the severity level for the phrase.

### Loss & Training

The QE model is based on the XLM-R-Large backbone. Its training objective integrates a sentence-level MSE regression loss and a word-level cross-entropy classification loss. In the supervised setting, pre-training is first conducted on synthetic data, followed by fine-tuning on real data; the unsupervised setting only uses synthetic data for training.

During inference, the error severity is determined by comparing the "OK" probability against different thresholds. Contiguous "BAD" tokens are grouped into spans, with the severity assigned as the highest severity found within the span.

## Key Experimental Results

### Main Results

Evaluation is conducted on the WMT QE Shared Task datasets, covering three language directions: EN-DE, ZH-EN, and HE-EN:

**Supervised Setting:**
- DCSQE achieves a Spearman correlation of 43.17 (vs. CometKiwi's 40.47) and an MCC of 27.11 (vs. 21.50) on WMT23 EN-DE.
- On ZH-EN, DCSQE achieves Spearman of 46.41 vs. CometKiwi's 40.35.
- On average, it outperforms CometKiwi by 4.38 (Spearman) and 3.41 (MCC), despite having fewer parameters.
- Significantly outperforms the GPT-4-based GEMBA-MQM.

**Unsupervised Setting:**
- The performance of MQMQE and InstructScore drops by an average of 15.74 and 7.64, respectively.
- DCSQE drops by only 6.64, demonstrating the best robustness.
- Unsupervised DCSQE on HE-EN (56.46 Spearman) outperforms supervised CometKiwi (55.00).

### Ablation Study

In the WMT23 EN-DE unsupervised setting:
- Full DCSQE: Spearman 35.78, MCC 18.00
- Removing SPCE: Spearman 30.99, MCC 15.70 (significant decline)
- Removing both SPCE and Annotator: Spearman drops further

Model Self-Annotation vs. Independent Annotation experiments:
- M annotates M (Self-Annotation): Error rate is only 1.60%, Spearman 25.91
- M generates + L annotates (Independent Annotation): Error rate is 19.23%, Spearman 35.78

### Key Findings

1. **Distribution shift is the core issue of synthetic QE data**: DCSQE alleviates distribution shift from both the translation and label perspectives.
2. **A model cannot fairly annotate its own output**: Self-annotation leads to overconfidence and a large number of false negatives.
3. **Generator diversity is beneficial**: The L+L' dual-generator setup improves Spearman by about 1 compared to a single L generator.
4. **Generator capability needs to be balanced**: Models that are too strong (few errors) or too weak (unrealistic errors) are both sub-optimal; moderate capability M is optimal.
5. **Stronger Annotator performance is always better**: Both leveraging supervised signals and expanding the training corpus are effective.
6. **Generation efficiency far exceeds InstructScore**: DCSQE is 14.29 times faster than InstructScore.

## Highlights & Insights

- Positioning the QE model as a reward model for translation tasks provides a unique perspective on analyzing distribution shift.
- The decoupled design of Generator and Annotator can be generalized to other synthetic data scenarios.
- The SPCE algorithm cleverly utilizes dependency syntax to achieve token $\rightarrow$ phrase aggregation, aligning with human annotation habits.
- Systematic controlled-variable experiments (fixing Similarity while varying Error Rate / fixing Error Rate while varying Similarity) provide clear causal analysis.

## Limitations & Future Work

- The impact of using LLMs (such as GPT-4) as an Annotator was not explored due to computational constraints.
- Robustness under extreme data scarcity scenarios (without parallel corpora) remains to be validated.
- The scalability and transfer value of insights from synthetic QE data to general reward models merit further investigation.
- The optimal balance of Generator capability needs to be tuned for specific language pairs.

## Related Work & Insights

- Comparison with CometKiwi demonstrates that synthetic data approaches can surpass cross-lingual annotation transfer even with fewer parameters.
- The SPCE algorithm shares similarities in spirit with CRF structures used in structured prediction.
- The paradigm of Generator-Annotator separation has inspiring implications for reward model construction in RLHF.
- Diversity enhancement strategies can be generalized to other data augmentation scenarios.

## Rating

- **Novelty**: 7/10 — Systematically decomposes the distribution shift problem into two dimensions: translation distribution and label distribution
- **Technical Depth**: 8/10 — Elaborate experimental design and comprehensive controlled-variable analysis
- **Practicality**: 8/10 — Practical and highly efficient method, which has been open-sourced
- **Writing Quality**: 8/10 — Clear logic and comprehensive analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_speech_data_quality.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](../../ACL2026/multilingual_mt/fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2025\] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](llms_can_achieve_high-quality_simultaneous_machine_translation_as_efficiently_as.md)

</div>

<!-- RELATED:END -->
