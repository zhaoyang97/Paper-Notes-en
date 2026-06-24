---
title: >-
  [Paper Note] Did Translation Models Get More Robust Without Anyone Even Noticing?
description: >-
  [Multilingual & Machine Translation] Through experiments with synthetic noise and social media texts, it is found that modern large-scale pre-trained translation models (such as TowerInstruct 13B and GPT-3.5) far outperform traditional NMT models (OPUS) in robustness to various character-level noises without using any specialized robustness training techniques. Furthermore, the combination of source-side correction and LLM translation can even surpass GPT-3.5.
tags:
  - "Multilingual & Machine Translation"
date: 2026-05-08
content_hash: 1048f14c5b4f26c1
---

# Did Translation Models Get More Robust Without Anyone Even Noticing?

- **Conference**: ACL 2025
- **arXiv**: [2403.03923](https://arxiv.org/abs/2403.03923)
- **Code**: [GitHub](https://github.com/utter-project/robust-mt)
- **Area**: Multilingual / Machine Translation / Robustness
- **Keywords**: Machine Translation Robustness, Character Noise, LLM Translation, Social Media Text, Source Correction

## TL;DR

Through experiments with synthetic noise and social media texts, it is found that modern large-scale pre-trained translation models (such as TowerInstruct 13B and GPT-3.5) far outperform traditional NMT models (OPUS) in robustness to various character-level noises without using any specialized robustness training techniques. Furthermore, the combination of source-side correction and LLM translation can even surpass GPT-3.5.

## Background & Motivation

- **Limitations of Prior Work**: The NLP community has long believed that neural machine translation is highly sensitive to source-side noise (typos, abbreviations, formatting anomalies). Consequently, numerous robustness training methods have been proposed (noisy data augmentation, character-level architectures, visual representations, etc.). However, these methods are costly and difficult to migrate to the LLM paradigm—LLMs cannot be easily trained with noise due to their massive parameter scale, and their architectures cannot be modified.

- **Key Challenge**: MT has shifted from the "training bilingual models from scratch" paradigm to the "instruction-tuned LLM translation" paradigm, but robustness research remains stuck under assumptions of the old paradigm. The core question is: are those specialized robustness techniques still necessary in the era of LLMs? Have larger models and scaling of training data naturally brought about sufficient robustness?

- **Goal**: (1) Systematically compare the robustness of translation models with different scales/architectures to synthetic noise; (2) verify whether synthetic robustness transfers to real social media text translation; and (3) evaluate the effectiveness of source-side correction and noise-aware training as mitigation strategies.

- **Key Insight**: The authors propose the COMET-slope robustness metric—using linear regression to fit the decay slope of translation quality with respect to the noise ratio, instead of just evaluating performance at a single noise level. This enables more precise and analysable robustness comparisons across different models and noise types.

## Method

### Overall Architecture

The input consists of the FLORES-200 test set combined with 4 types of synthetic noise (swap/dupe/drop/key) across 10 noise levels ($p=0.1\text{ to }1.0$), plus social media datasets (MTNT, MultiLexNorm). The output comprises the COMET scores and robustness metrics for 4 types of translation models under each configuration.

### Key Designs

1. **COMET-slope Robustness Metric**:

    - **Function**: Quantify the speed of quality degradation of translation models as noise increases.
    - **Mechanism**: For each noise type and language pair, COMET is measured at 10 noise levels. A linear regression $\text{COMET}(p) = a + b \cdot p$ is fitted using the least squares method, where a smaller absolute value of slope $b$ (COMET-slope) indicates stronger robustness. For example, the slope of GPT-3.5 under en→fr swap noise is only $-4.46$, compared to $-69.59$ for OPUS.
    - **Design Motivation**: Prior methods only report the COMET score drop at specific noise levels, which fails to capture the overall shape of the degradation curve.

2. **Synthetic Noise Experimental Design**:

    - **Function**: Controllably measure the impact of different noise types.
    - **Mechanism**: Four types of noise are used to simulate typing typos: swap (swapping adjacent characters), dupe (duplicating characters), drop (deleting characters), and key (replacing a character with an adjacent keyboard key). Each token is perturbed independently with probability $p$. Tests are conducted on 4 language pairs (de/fr/ko/pt ↔ en) with 4 models: OPUS (74M, bilingual), NLLB (3.3B, multilingual), TowerInstruct (13B, instruction-tuned LLM), and GPT-3.5.
    - **Design Motivation**: The four noise types cover common typing typo scenarios, and the 10 levels provide a complete robustness curve.

3. **Source-Side Correction Pipeline**:

    - **Function**: Improve robustness through a correct-then-translate pipeline.
    - **Mechanism**: Correct noisy text using GECToR (a grammatical error correction model), LLM zero-shot correction (Llama-3-8B), or BartLM, and then translate using the translation model. The correction + NLLB pipeline can surpass GPT-3.5's robustness in 3/4 of the noise types, while correction + TI surpasses GPT-3.5 across all types.
    - **Design Motivation**: Correction is a model-agnostic approach that requires no modification to the translation model itself.

### Social Media Evaluation

Using MTNT (Reddit text) and MultiLexNorm (a 12-language social media normalization dataset, used for MT evaluation here for the first time). Evaluation is conducted using reference-free COMET.

## Key Experimental Results

### Main Results — Clean Data Performance (COMET on FLORES)

| Model | Parameters | xx→en Avg | en→xx Avg |
|------|-----------|-----------|-----------|
| OPUS | 74M | 88.02 | 86.79 |
| NLLB | 3.3B | 89.00 | 88.61 |
| TowerInstruct | 13B | 89.60 | 89.47 |
| GPT-3.5 | Unknown | 89.22 | 89.05 |

### Robustness — COMET-slope (xx→en Average, smaller absolute value is better)

| Model | swap | dupe | drop | key |
|------|------|------|------|-----|
| OPUS | -57.52 | -26.91 | -46.39 | -58.29 |
| NLLB | -20.93 | -4.58 | -18.48 | -23.53 |
| TI (13B) | -25.90 | -3.38 | -18.17 | -28.82 |
| GPT-3.5 | **-9.47** | **-2.47** | **-9.28** | **-11.09** |

### Source-Side Correction Performance (en→xx, COMET-slope)

| Correction Method + Translation Model | swap | dupe | drop | key |
|-------------------|------|------|------|-----|
| No Correction + NLLB | -22.04 | -4.93 | -21.26 | -24.56 |
| GECToR + NLLB | -10.80 | -3.17 | -14.77 | -13.22 |
| No Correction + GPT-3.5 | -4.23 | -2.15 | -6.81 | -6.58 |
| LLM-correction + TI | **-2.14** | **-1.67** | **-4.02** | **-3.30** |

### Key Findings

- **LLM translation is inherently more robust**: The slope of GPT-3.5 is 5 to 6 times smaller than that of OPUS, even when clean data performances are similar.
- **Robustness is obtained 'for free'**: No open LLM uses specialized robustness techniques; instead, robustness naturally emerges from larger model capacity and diverse pre-training data.
- **dupe noise has the least impact**: All models exhibit the highest tolerance for character duplication (NLLB slope of only -4.58), because duplication has the minimal impact on subword tokenization.
- **Source-side correction is effective**: GECToR + NLLB outperforms GPT-3.5 on 3/4 of the noise types; LLM correction + TI outperforms GPT-3.5 on all types.
- **Consistent social media results**: Robustness on synthetic noise is positively correlated with translation performance on social media datasets.

## Highlights & Insights

- **Challenging traditional conventions**: The notion that "NMT is fragile to noise" no longer holds in the LLM era, suggesting that many robustness studies might be outdated.
- **COMET-slope method**: Using a single slope metric to quantify robustness serves as a simple and elegant tool, which can be adapted to other NLP robustness evaluations.
- **Synergy between correction and translation**: Even the combination of a 3B NLLB and GECToR can surpass GPT-3.5 in robustness, offering a practical solution for resource-constrained scenarios.

## Limitations & Future Work

- The training data of GPT-3.5 and TI are unknown, introducing potential risks of data leakage (having seen the test sets).
- There remains a gap between synthetic noise and real-world noise, and key noise dictates a dependency on the QWERTY keyboard layout.
- Only 4 language pairs were tested, and evaluation on low-resource languages is lacking.
- The source of robustness was not analyzed—whether it stems from model size, pre-training data volume, or BPE vocabulary size.
- MultiLexNorm uses reference-free COMET, which might be less reliable than reference-based evaluations.

## Related Work & Insights

- **vs Belinkov & Bisk (2018)**: First to report that NMT is sensitive to character perturbations; this work demonstrates that this finding is no longer universally applicable in the LLM era.
- **vs Noise-aware training methods (Karpukhin et al., 2019)**: This work finds that simply fine-tuning NLLB with noise can significantly improve robustness (with slope decreasing from $-22$ to $-6.5$).
- **vs Character-level models (Xue et al., 2022)**: Character-level models are theoretically more robust but have slow inference, whereas LLMs naturally achieve better robustness while utilizing subword tokenization.

## Rating

- **Novelty**: 8/10 — The finding of "getting robustness for free" holds paradigm-shifting significance.
- **Technical Depth**: 7/10 — The COMET-slope method is simple yet effective, and the experimental design is systematic.
- **Experimental Thoroughness**: 8/10 — Thorough testing across synthetic and real noise, 4 models, multilinguality, and error correction ablation.
- **Writing Quality**: 9/10 — Excellent writing, with logical and compelling arguments.
- **Overall Score**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)

</div>

<!-- RELATED:END -->
