---
title: >-
  [Paper Note] Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Quality Estimation] > This paper systematically exposes gender disparities in machine translation quality estimation (QE) metrics: masculine forms score higher than feminine forms when source genders are ambiguous; feminine forms have higher error rates in the presence of contextual cues; and the biases propagate to downstream MT systems through data filtering and quality-aware decoding (QAD).
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Quality Estimation"
  - "gender bias"
  - "machine translation"
  - "fairness"
  - "QE metrics"
date: 2026-05-08
content_hash: ad32e6d9f1009c56
---

# Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation

**Conference**: ACL 2025  
**arXiv**: [2410.10995](https://arxiv.org/abs/2410.10995)  
**Code**: [GitHub](https://github.com/deep-spin/gender-bias-qe-metrics)  
**Area**: Multilingual / MT  
**Keywords**: Quality Estimation, gender bias, machine translation, fairness, QE metrics  

## TL;DR

> This paper systematically exposes gender disparities in machine translation quality estimation (QE) metrics: masculine forms score higher than feminine forms when source genders are ambiguous; feminine forms have higher error rates in the presence of contextual cues; and the biases propagate to downstream MT systems through data filtering and quality-aware decoding (QAD).

## Background & Motivation

### Background

**Background**: QE metrics are used to automatically evaluate translation quality and have been widely applied in translation pipelines such as data filtering, training, and decoding. However, whether these metrics encode social biases—particularly gender bias—has not been systematically studied.

**Limitations of Prior Work**:

### Key Challenge

**Key Challenge**: Bias ignored in evaluation metric research: A large body of work evaluates the correlation of automatic metrics with human judgments, but does not examine gender fairness.

### Limitations of Prior Work

**Limitations of Prior Work**: MT bias research focuses solely on system outputs: Prior work predominantly detects gender bias in translation system outputs, without studying the evaluation metrics themselves.

### Proposed Solution

**Proposed Solution**: Downstream impact of QE bias is unknown: How biased QE metrics affect data filtering and quality-aware decoding was previously unexplored.

**Core Motivation:** Define gender bias in QE metrics, systematically measure it, and reveal its propagation effects in the MT pipeline.

## Method

### Overall Architecture

1. **Bias Definition:** Formally define two conditions for gender bias in QE metrics.
2. **Controlled Experimental Design:** Use minimal edit contrastive pairs to isolate the gender factor.
3. **Multi-dimensional Evaluation:** Ambigious/unambiguous gender scenarios $\times$ multiple metrics $\times$ multiple languages.
4. **Downstream Impact:** Data filtering and quality-aware decoding experiments.

### Key Designs

**1. Bias Definition:** A QE metric is gender-biased if and only if:
- (i) When the source language is gender-ambiguous, it systematically assigns higher scores to a specific gender form.
- (ii) When gender cues are present, it exhibits unequal error rates across genders.

**2. Experimental Setup:**
- **Gender-Ambiguous Scenario:** Using minimal edit contrastive pairs from MT-GenEval and GATE datasets, the ratio of QE scores for feminine and masculine forms $QE(s, h_F) / QE(s, h_M)$ is calculated, where the ideal value is 1.
- **Gender-Explicit Scenario (Intra-sentential Cues):** Using the counterfactual subset of MT-GenEval, the error rate (ER) and the ratio of error rates between genders $\Phi(S^F, S^M) = ER(S^F) / ER(S^M)$ are calculated, where the ideal value is 1.
- **Gender-Explicit Scenario (Extra-sentential Cues):** Using contextual disambiguation, testing the gender sensitivity of context-aware metrics.

**3. QE Metrics Coverage:** 11 SOTA metrics, including:
- Neural metrics: CometKiwi 22/23 XL/XXL, xCOMET XL/XXL, MetricX 23 L/XL
- GEMBA metrics: Mistral 7B, Gemma 2 9B, Llama 3.1 70B, GPT-4o

### Loss & Training

As this is an evaluation study, no model training is involved. The core evaluation metrics are the QE score ratio and the error rate ratio.

## Experiments

### Main Results — Gender-Ambiguous Scenario

Most QE metrics assign higher scores to masculine forms in gender-ambiguous source languages. The larger the CometKiwi model, the more pronounced the bias (22 < 23 XL < 23 XXL). MetricX is closest to fairness but lacks sensitivity.

### Main Results — Gender-Explicit Scenario (Intra-sentential Cues)

| Metric | Error Rate ER↓ | Error Rate Ratio $\Phi \to 1$ |
|------|---|---|
| CometKiwi 22 | 0.11 | 1.70 |
| CometKiwi 23 XL | 0.09 | 1.18 |
| CometKiwi 23 XXL | 0.07 | **0.87** |
| xCOMET XL | 0.10 | 1.81 |
| xCOMET XXL | 0.08 | 1.32 |
| MetricX 23 L | 0.31 | 1.25 |
| MetricX 23 XL | 0.12 | 1.19 |
| GPT 4o | 0.16 | 1.15 |

Most metrics exhibit higher error rates for feminine entities than for masculine entities, with CometKiwi 23 XXL being the only metric close to fairness.

### Ablation Study — Downstream Impact

| Scenario | Key Findings |
|------|------|
| **Data Filtering** | A threshold of 0.8 retains 75% of masculine but only 63% of feminine instances (CometKiwi 23 XXL). |
| **MT Quality Evaluation** | Bias persists when evaluating Google Translate outputs ($\Phi$ 1.17-1.96). |
| **Quality-aware Decoding (QAD)** | QAD + CometKiwi 22 exacerbates bias ($\delta_M$: -45.7 $\to$ -46.8); QAD + CometKiwi 23 XXL mitigates bias ($\delta_M$: -45.7 $\to$ -43.5). |

### Key Findings

1. **Masculine Preference is Pervasive:** In the absence of gender information, almost all QE metrics systematically award higher scores to masculine forms.
2. **Neutral Forms are Penalized:** Neutrally-gendered translations consistently receive lower QE scores than gendered translations.
3. **Translation Context Exacerbates Bias:** Utilizing translation context reduces the overall error rate but amplifies the gender error rate ratio by approximately 3-fold.
4. **GEMBA Metrics are Coarse-grained:** LLM-based GEMBA metrics tend to assign coarse scoring (85/90/95/100), failing to capture gender differences.
5. **Bias Propagation Effect:** Biased QE metrics unevenly filter out feminine data during data filtering, and amplify gender bias in MT systems during QAD.
6. **CometKiwi 23 XXL is Pareto-Optimal:** Achieves the best balance between accuracy and fairness.

## Highlights & Insights

- This work is the first to systematically define and measure gender bias in QE metrics, filling an important research gap.
- Elegant experimental design: utilizes minimal edit contrastive pairs to isolate the gender factor, covering 11 metrics $\times$ 3 datasets $\times$ 8 languages.
- Exposes the downstream propagation effects of bias: from evaluation metrics to data filtering to translation systems.
- Proposes a joint evaluation framework for fairness and accuracy (Pareto frontier analysis).

## Limitations & Future Work

- Only studies sentence-level translation, without covering document-level or dialogue-level MT scenarios.
- Gender analysis is primarily based on binary classification, with limited coverage of non-binary gender (only partially addressed through mGeNTE neutral experiments).
- GEMBA metrics are only evaluated under zero-shot configurations; few-shot strategies might improve results.
- Context-aware experiments utilize inference-time strategies to inject context, rather than fine-tuning at training time.

## Related Work & Insights

- **MT Metric Evaluation:** WMT shared tasks (Kocmi et al., 2021; Freitag et al., 2023) focus on correlation between metrics and human judgments, but do not address bias.
- **MT Gender Bias:** Stanovsky et al. (2019), Vanmassenhove et al. (2018) detect gender bias in translation system outputs; this work shifts focus to the evaluation metrics themselves.
- **NLG Metric Bias:** Qiu et al. (2023) study gender bias in image captioning metrics; Sun et al. (2022) quantify social bias in NLG metrics. However, no prior work focuses on QE metrics.
- **QE Models:** CometKiwi (Rei et al., 2022/2023), xCOMET (Guerreiro et al., 2024), GEMBA (Kocmi & Federmann, 2023) represent the current SOTA.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Evaluation | 8.8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](../../ACL2026/multilingual_mt/fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)

</div>

<!-- RELATED:END -->
