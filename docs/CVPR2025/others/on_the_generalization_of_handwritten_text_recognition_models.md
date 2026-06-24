---
title: >-
  [Paper Note] On the Generalization of Handwritten Text Recognition Models
description: >-
  [CVPR 2025][Handwritten Text Recognition] This paper presents the first systematic analysis of the out-of-distribution (OOD) generalization capability of HTR models. Through 336 OOD evaluations of 8 SOTA models across 7 datasets (5 languages), it is discovered that textual discrepancy is the most critical factor affecting generalization, and the OOD error can be reliably predicted in 70% of cases (with a deviation $< 10$ percentage points).
tags:
  - "CVPR 2025"
  - "Handwritten Text Recognition"
  - "Domain Generalization"
  - "Out-of-Distribution Generalization"
  - "Cross-Lingual"
  - "Factor Analysis"
date: 2026-05-08
content_hash: 99b18c153c28c09a
---

# On the Generalization of Handwritten Text Recognition Models

**Conference**: CVPR 2025  
**arXiv**: [2411.17332](https://arxiv.org/abs/2411.17332)  
**Code**: [https://github.com/carlos10garrido/HTR-OOD](https://github.com/carlos10garrido/HTR-OOD)  
**Area**: OCR/Text Recognition  
**Keywords**: Handwritten Text Recognition, Domain Generalization, Out-of-Distribution Generalization, Cross-Lingual, Factor Analysis

## TL;DR

This paper presents the first systematic analysis of the out-of-distribution (OOD) generalization capability of HTR models. Through 336 OOD evaluations of 8 SOTA models across 7 datasets (5 languages), it is discovered that textual discrepancy is the most critical factor affecting generalization, and the OOD error can be reliably predicted in 70% of cases (with a deviation $< 10$ percentage points).

## Background & Motivation

**Background**: Handwritten Text Recognition (HTR) has made significant progress on standard benchmarks in recent years. Mainstream methods include CTC decoding (CRNN, VAN), sequence-to-sequence (Transformer), and hybrid methods (CTC+CE). However, these advances are predicated on the assumption that training and testing data are identically and independently distributed (i.i.d.).

**Limitations of Prior Work**: (1) Existing "generalization" evaluations are limited to i.i.d. train-test splits (different lines from the same manuscript), which is not true cross-domain generalization; (2) when models encounter unseen manuscripts, different languages, or texts from different historical periods, their performance degrades drastically—preliminary experiments show that the Character Error Rate (CER) in OOD scenarios surges from an average of ~7% to ~35%; (3) transfer learning and domain adaptation require target-domain data, rendering them inapplicable to entirely unknown target domains.

**Key Challenge**: The definition of "generalization" in the HTR field is overly narrow, failing to explore true OOD generalization scenarios (zero-shot unseen manuscripts/unseen languages) and lacking a systematic understanding of the key factors that affect OOD performance.

**Goal**: (1) To analyze the OOD performance of HTR models within a domain generalization (DG) framework; (2) to identify the core factors affecting generalization; (3) to evaluate whether OOD error is predictable.

**Key Insight**: To construct two proxy metrics—visual divergence and textual divergence—to quantify the discrepancies between source and target domains, and to perform factor analysis to reveal the significant factors influencing OOD performance.

**Core Idea**: Through large-scale systematic experiments (8 models $\times$ 7 datasets $\times$ 6 OOD targets = 336 OOD evaluations), it is found that textual discrepancy (language/character inventory differences) has a greater impact on generalization than visual discrepancy (writing style), and the OOD error can be reliably predicted using these proxy metrics.

## Method

### Overall Architecture

The experimental framework consists of two parts: (1) **Practical Analysis**—evaluating the ID and OOD performance of 8 HTR models under standardized conditions, and investigating the impacts of model capacity, selection strategies, and synthetic data on generalization; (2) **Factor Analysis**—defining visual divergence and textual divergence metrics, performing multi-factor analysis of variance (ANOVA) to determine the significant factors affecting OOD performance, and constructing an OOD error prediction model based on these factors.

### Key Designs

1. **Standardized Cross-Domain Evaluation Framework**:

    - Function: To ensure fair and comparable evaluation of the OOD generalization capability of different HTR models.
    - Mechanism: Eight models (CRNN, VAN, C-SAN, HTR-VT, Kang Transformer, Michael, LT, VLT) covering three major categories (CTC, Seq2Seq, and Hybrid) are trained from scratch on 7 datasets (IAM, Rimes, Bentham, Saint-Gall, G.W., Rodrigo, ICFHR2016, spanning 5 languages: English, French, Latin, Spanish, and German) using a unified 94-character merged Unicode alphabet. Each model is trained on one source domain and tested in an OOD manner on all other domains, yielding a total of 336 evaluations.
    - Design Motivation: Previous studies utilized varied training configurations and differing alphabet handling methods, making their results incomparable. Standardizing these setups enables a fair comparison of generalization discrepancies among models, thereby revealing the impact of the architecture itself on generalization.

2. **Visual/Textual Divergence Metrics**:

    - Function: To quantify the visual and linguistic differences between the source and target domains.
    - Mechanism: **Visual divergence** uses FID (Fréchet Inception Distance) to measure the distance between the image feature distributions of the source and target domains. **Textual divergence** measures the linguistic distance of textual contents between the two domains based on alphabet overlap and the KL divergence of character frequency distributions. These two metrics are expected to correlate positively with OOD performance degradation.
    - Design Motivation: The degradation in OOD performance may stem from visual reasons (substantial handwriting style differences) or textual reasons (different languages/alphabets). Distinguishing their impacts helps in targeted model improvement.

3. **Factor Analysis and OOD Error Prediction**:

    - Function: To identify key factors affecting OOD generalization and predict OOD errors.
    - Mechanism: Taking OOD CER as the dependent variable, multi-factor analysis of variance (ANOVA) is conducted to investigate the significance of factors such as model architecture, source domain, target domain, visual divergence, and textual divergence. Regression models are then constructed based on the significant factors to predict OOD errors. The results demonstrate that textual divergence is the most significant factor, followed by visual divergence. In 70% of cases, the discrepancy between the predicted error and the actual error does not exceed 10 CER percentage points.
    - Design Motivation: If OOD errors are predictable, the reliability of HTR systems on new data can be pre-evaluated prior to deployment without needing actual ground-truth labels for the test data.

### Loss & Training

Depending on the model architecture, CTC models use the CTC loss, Seq2Seq models use the CE loss, and Hybrid models use $\mathcal{L} = \lambda \mathcal{L}_{\text{ctc}} + (1-\lambda) \mathcal{L}_{\text{ce}}$ (with $\lambda=0.5$). All models are trained from scratch for 500 epochs. The best model is selected based on the validation set CER, and early stopping is applied if no improvement is observed for 100 epochs.

## Key Experimental Results

### Main Results (ID vs OOD CER%, representative models selected)

| Dataset | CRNN ID | CRNN OOD | VAN ID | VAN OOD | HTR-VT ID | HTR-VT OOD |
|--------|---------|----------|--------|---------|-----------|------------|
| IAM (En) | 6.4 | 34.9 (+28.5) | 6.6 | 28.6 (+22.0) | 5.8 | 33.7 (+27.9) |
| Rimes (Fr) | 3.7 | 25.0 (+21.2) | 5.6 | 21.3 (+15.6) | 7.9 | 28.3 (+20.4) |
| Bentham (En) | 4.7 | 25.3 (+20.6) | 7.4 | 26.6 (+19.2) | 8.4 | 33.3 (+24.9) |
| S.G. (Lat) | 7.2 | 33.6 (+26.3) | 7.8 | 39.8 (+32.0) | 17.1 | 36.5 (+19.3) |
| Rodrigo (Sp) | 4.1 | 36.5 (+32.4) | 4.2 | 29.9 (+25.7) | 5.1 | 34.2 (+29.1) |

### Ablation Study (Factor Analysis - ANOVA Results)

| Factor | F-statistic | p-value | Significance |
|------|------------|---------|--------|
| Textual Divergence | Highest | <0.01 | Highly Significant |
| Visual Divergence | Medium | <0.01 | Significant |
| Source Domain | Medium | <0.01 | Significant |
| Model Architecture | Lower | <0.05 | Weakly Significant |

### Key Findings

- **Huge ID-OOD gap**: The average OOD CER of the 8 models is around 35%, which is approximately 28 percentage points higher than the ID average of ~7%. This indicates that current HTR models suffer from a severe lack of OOD generalization capability.
- **Textual divergence is the top factor**: OOD performance degrades most severely when there is a large language/alphabet difference between the source and target domains. Visual divergence (writing style) is the secondary factor.
- **Relative insignificance of model architecture**: No single architecture consistently outperforms others across all OOD scenarios. CTC-based models (CRNN, VAN) exhibit relative robustness in OOD settings, whereas large-parameter Transformer models tend to overfit.
- **Synthetic data is helpful but limited**: Utilizing synthetic data as the source training domain yields performance improvements in certain OOD scenarios but cannot completely bridge the domain gap.
- OOD error prediction achieves a deviation of $< 10$ CER percentage points in 70% of cases, providing a viable pre-evaluation scheme for real-world deployment.

## Highlights & Insights

- **First systematic study revealing the OOD generalization deficiencies of HTR**: The large-scale experiment consisting of 336 evaluations provides reliable statistical conclusions, dispelling the illusion that "ever-improving benchmarks equate to resolved problems."
- **The finding of "Textual Divergence > Visual Divergence" is surprising yet reasonable**: HTR models inherently learn implicit language priors (statistical regularities of character sequences). Consequently, when the target domain language is completely different, the learned language priors become entirely ineffective. This suggests that future research directions should focus on language-agnostic visual feature extraction.
- The predictability of OOD error is of high practical value: when deploying HTR systems, the reliable performance on unseen targets can be estimated based on the textual and visual divergences between the source and target domains.

## Limitations & Future Work

- Only line-level HTR was evaluated, without extending to end-to-end document recognition.
- The alphabet was unified to 94 characters, which may lack granularity in handling character inventories specific to certain languages.
- Multi-source domain training and domain generalization algorithms (e.g., DRO, IRM) were not explored on HTR.
- Future work could explore the design of language-agnostic HTR architectures, as well as the effects of large-scale pre-training and self-supervised learning on OOD generalization.

## Related Work & Insights

- **vs. Large-scale Pre-training Methods Like TrOCR**: TrOCR improves generalization through massive pre-training, but requires vast data and computing resources. The analysis in this paper can help such methods pinpoint the most valuable training data (e.g., covering more languages rather than more visual styles).
- **vs. Domain Adaptation Methods**: Domain adaptation (DA) relies on target-domain data, limiting its applicability. The domain generalization (DG) framework in this paper is more realistic, as the target domain is typically unknown beforehand during deployment.
- **vs. Prior HTR Evaluations**: Previous evaluations focused exclusively on ID performance, neglecting OOD scenarios. The massive ID-OOD gap uncovered in this work calls for the research community to reconsider the standard evaluation paradigms.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic analysis of OOD generalization in the HTR field, presenting a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous statistical analysis involving 336 evaluations across 8 models, 7 datasets, and 5 languages.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and highly convincing conclusions.
- Value: ⭐⭐⭐⭐ Uncovers critical blind spots in HTR and guides directions for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](gradient-guided_annealing_for_domain_generalization.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](../../ACL2025/others/principled_generalization_arithmetic.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](../../ACL2025/others/towards_text-image_interleaved_retrieval.md)

</div>

<!-- RELATED:END -->
