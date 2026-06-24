---
title: >-
  [Paper Note] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors
description: >-
  [ACL 2025][Text Generation][NMT Robustness] A systematic representation-level analysis of how NMT encoders process grammatical errors reveals that encoders first "detect" errors in shallow layers (indicated by rising GED probing $F1$), and then "correct" them in deep layers (indicated by falling CKA distance). It proposes the concept of "Robustness Heads" to identify the specific attention heads involved in error correction, validating this two-stage "detection-then-correctio…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "NMT Robustness"
  - "Grammatical Errors"
  - "Representation Analysis"
  - "Robustness Heads"
  - "CKA"
date: 2026-05-08
content_hash: f395367143586e7b
---

# A Representation Level Analysis of NMT Model Robustness to Grammatical Errors

**Conference**: ACL 2025  
**arXiv**: [2505.21224](https://arxiv.org/abs/2505.21224)  
**Code**: [GitHub](https://github.com/issam9/nmt-robustness-analysis)  
**Area**: Text Generation  
**Keywords**: NMT Robustness, Grammatical Errors, Representation Analysis, Robustness Heads, CKA

## TL;DR

A systematic representation-level analysis of how NMT encoders process grammatical errors reveals that encoders first "detect" errors in shallow layers (indicated by rising GED probing $F1$), and then "correct" them in deep layers (indicated by falling CKA distance). It proposes the concept of "Robustness Heads" to identify the specific attention heads involved in error correction, validating this two-stage "detection-then-correction" mechanism across 4 models $\times$ 5 language directions.

## Background & Motivation

**The vulnerability of NMT to noisy inputs is a well-recognized problem.** Grammatical errors by non-native speakers (article misuse, preposition substitution, noun number variation) lead to significant degradation in translation quality. Prior work primarily focuses on documenting robustness failures or improving robustness through data augmentation, leaving the underlying mechanisms of how models internally process these errors poorly understood.

**Interpretability analysis tools are mature but have not been applied to robustness.** Techniques such as probing, representation similarity (CKA), and attention analysis are widely used in Transformer understanding, yet they have never been systematically utilized to comprehend NMT robustness to grammatical errors. **The core hypothesis of this paper is that NMT encoders feature an implicit pipeline similar to grammatical error correction (GEC)**—first detecting grammatical errors, and then shifting the representation of the erroneous token toward its correct form. This hypothesis is validated through a three-layer corroborative analysis framework.

## Method

### Overall Architecture

Construct a controlled synthetic grammatical error dataset $\rightarrow$ Perform a three-layer analysis (GED probing, CKA distance, Robustness Heads) on 4 NMT models $\rightarrow$ Compare three model variants (Base / Clean-Finetuned / Noise-Finetuned) to isolate standard robustness effects.

### Key Designs

1. **Synthetic Grammatical Error System**:
    - **Function**: Precisely insert a single controlled grammatical error into a clean sentence.
    - **Mechanism**: Three types of errors—article substitution (Article), preposition substitution (Prep), and noun number variation (Nounnum)—are sampled for substitution based on the statistical distribution of the CoNLL-14 GEC shared task. Additionally, the MORPHEUS black-box adversarial attack is used to evaluate generalization (greedily introducing inflectional errors to minimize BLEU).
    - **Design Motivation**: Controlled errors allow the analysis to precisely pinpoint representation changes of specific words, rather than making vague attributions amidst global noise.

2. **Three-Layer Analysis Method**:
    - **Function**: Analyze encoder behavior across three dimensions: detection, correction, and attention mechanisms.
    - **Mechanism**:
        - **GED Probing**: Linearly train classifiers at each encoder layer to detect incorrect words, tracking $F1$ score variations across layers.
        - **CKA Distance**: Measure the distance between the representation of the erroneous token and its correct form in each layer using Centered Kernel Alignment: $1 - CKA(\widetilde{W}, W)$.
        - **Robustness Heads**: Mask attention heads individually and measure the change in the CKA distance between the erroneous and correct token representations: $1 - CKA(\widetilde{w_{h_i}}, w)$. The heads whose masking causes the largest increase in distance are identified as robustness heads.
    - **Design Motivation**: The three-layer analysis is mutually corroborative—probing measures detection, CKA measures correction, and Robustness Heads explain the correction mechanism.

3. **POS Tag Attention Analysis**:
    - **Function**: Analyze which parts-of-speech (POS) are targeted by robustness heads when processing errors.
    - **Mechanism**: Collect attention weights from the erroneous token to other tokens in robustness heads, aggregating them by POS tags.
    - **Design Motivation**: Verify whether robustness heads focus on interpretable linguistic cues (e.g., attending to nouns in the case of article errors).

### Loss & Training

Fine-tune only the encoder (validating that encoder-only fine-tuning achieves robustness on par with full-model fine-tuning, whereas fine-tuning the decoder or cross-attention is insufficient). The optimizer is Base AdamW with a learning rate of 5e-05, a batch size of 64, and the best model saved based on validation set BLEU up to 5000 steps.

## Key Experimental Results

### Robustness Fine-tuning Results (COMET, En-Es)

| Model | Error Type | Base $\Delta$ | Noise-FT $\Delta$ | Gain |
|------|---------|--------|------------|------|
| NLLB | Article | 0.74 | **0.01** | Eliminates 99% of performance drop |
| NLLB | Nounnum | 0.66 | **0.06** | Eliminates 91% of performance drop |
| NLLB | Prep | 0.83 | **0.29** | Eliminates 65% of performance drop |
| M2M100 | Article | 1.15 | **0.03** | Eliminates 97% of performance drop |
| MBART | Article | 0.81 | **0.04** | Eliminates 95% of performance drop |

### Core Findings of Representation Analysis

| Analysis Method | Finding | Implication |
|---------|------|------|
| GED Probing | $F1$ rises in the first half of the layers, plateaus/declines in the second half | Encoder detects errors first |
| CKA Distance | Monotonically decreases across layers (nearly reaches 0 for Noise-FT) | Encoder corrects the representation of erroneous words |
| Robustness Heads | Noise-FT relies more on robustness heads (especially in deeper layers) | Fine-tuning strengthens the correction mechanism |
| GED + Noise-FT | Probing $F1$ drops more in deep layers | Correction makes detection harder (representations are already fixed) |

### Key Findings

- **The two-stage "detection-then-correction" mechanism is consistent across all 4 models**: OPUS-MT, M2M100, MBART, and NLLB exhibit highly similar behaviors.
- **Preposition errors are the most difficult to handle**: There remains a significant performance drop after fine-tuning ($\Delta=0.20-0.29$), because preposition substitutions alter more semantic meaning.
- **Cross-linguistic variation exists**: The probing $F1$ for French Nounnum errors is significantly higher than that for English (0.84 vs 0.48), as noun number in French is morphologically marked by articles and adjectives.
- **Noise fine-tuning has a regularization effect**: Performance on clean data actually improves rather than degrades (e.g., M2M100 Prep: 77.51 vs Clean-FT 77.14).
- **Robustness heads focus on interpretable linguistic units**: They attend to the target noun for article errors, and to determiners for noun number errors.

## Highlights & Insights

- **The discovery of the "detection-to-correction" mechanism is the core contribution**: NMT encoders feature an implicit GEC-like pipeline, providing a novel understanding of how Transformers handle noisy inputs.
- **The concept of Robustness Heads is highly generalizable**: Identifying specific heads involved in dedicated functions from an attention-mechanism perspective offers a methodology transferable to other scenarios.
- **A three-layer mutually corroborative analysis paradigm**: GED probing, CKA distance, and attention head analysis support the same hypothesis from different angles, making the findings exceptionally convincing.
- **Highly thorough experimental scale**: 4 models $\times$ 5 language directions $\times$ 3 error types + MORPHEUS adversarial attacks, ruling out spurious factors.

## Limitations & Future Work

- **Limited to grammatical errors**: Other noise types such as spelling mistakes, word-order errors, and semantic errors are not covered.
- **Realism of synthetic errors**: Only one error per sentence is introduced, sampled based on statistical distribution, which might not fully mirror authentic non-native errors.
- **Expressiveness of linear probing**: Non-linear probing might uncover richer information-encoding patterns.
- **Lack of in-depth encoder-decoder interaction**: The causal relationship of how the effects of robustness heads cascade to the decoder and eventually impact translation quality remains unestablished.

## Related Work & Insights

- **vs Belinkov & Bisk 2018**: Only documented NMT robustness failures, whereas this study delves into explanation at the representation level.
- **vs CV Adversarial Training Analysis (Cianfarani 2022)**: Analyzed DNN representation changes under adversarial training in computer vision, whereas this work is the first to conduct a similar analysis in the context of NMT grammatical errors.
- **vs General Probing Literature**: Conventional probing studies standard linguistic feature encoding, whereas this work focuses specifically on error-processing behaviors.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "detection-then-correction" mechanism and the concept of Robustness Heads are novel findings; the analytical paradigm offers methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models $\times$ 5 languages $\times$ 3 error types + MORPHEUS + three-layer mutually corroborative verification, exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear research questions, rigorous analytical structure, and highly coherent three-layer analysis.
- Value: ⭐⭐⭐⭐ Significant contribution to understanding NMT robustness mechanisms, with highly generalizable methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)
- [\[ACL 2025\] gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)
- [\[ACL 2025\] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)

</div>

<!-- RELATED:END -->
