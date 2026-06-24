---
title: >-
  [Paper Note] LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-lingual ABSA] The LACA framework is proposed to leverage LLMs to generate high-quality pseudo-labeled data in the target language (rather than relying on machine translation). This significantly improves cross-lingual ABSA performance across six languages, outperforming the previous SOTA by an average of 1.50% and 2.62% on mBERT and XLM-R, respectively.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-lingual ABSA"
  - "LLM Data Augmentation"
  - "Pseudo-labelled Data"
  - "Multilingual Sentiment"
  - "Zero-shot Transfer"
date: 2026-05-08
content_hash: 6a1b9c972bc483b7
---

# LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation

**Conference**: ACL 2025  
**arXiv**: [2508.09515](https://arxiv.org/abs/2508.09515)  
**Code**: [https://nlp.kiv.zcu.cz](https://nlp.kiv.zcu.cz)  
**Area**: Multilingual Translation  
**Keywords**: Cross-lingual ABSA, LLM Data Augmentation, Pseudo-labelled Data, Multilingual Sentiment, Zero-shot Transfer  

## TL;DR

The LACA framework is proposed to leverage LLMs to generate high-quality pseudo-labeled data in the target language (rather than relying on machine translation). This significantly improves cross-lingual ABSA performance across six languages, outperforming the previous SOTA by an average of 1.50% and 2.62% on mBERT and XLM-R, respectively.

## Background & Motivation

### Challenges of Cross-lingual ABSA

Aspect-Based Sentiment Analysis (ABSA) aims to identify the sentiment polarity associated with a specific aspect in a sentence. For instance, in "Great tea but terrible service", "tea" is positive while "service" is negative. Because most labeled datasets are concentrated in English, low-resource languages suffer from a severe lack of labeled data, making cross-lingual ABSA an important research direction.

### Limitations of Prior Work

**Limitations of Translation-based Methods**: Traditional methods rely on machine translation to translate source language data into the target language. However, aspect terms are prone to misalignment or loss during translation, leaving the model unable to correctly identify aspect terms in the target language.

**Language Gap in Direct Transfer**: Directly applying a model fine-tuned on the source language to the target language is hindered by language-specific aspect terms, slang, and abbreviations.

**Under-representation of Low-Resource Languages in mPLMs**: Certain low-resource languages comprise only a minor fraction of the pre-training corpora in multilingual pre-trained language models.

### Core Motivation

The data augmentation capabilities of LLMs offer a new paradigm for cross-lingual ABSA—directly generating diverse training samples in the target language to avoid the noise introduced by machine translation.

## Method

### Overall Architecture

LACA (LLM Augmented Cross-lingual ABSA) is a two-stage framework:

**Stage 1: ABSA Model Prediction**
1. Fine-tune the ABSA model on the labeled English source language data $\mathcal{D}_\mathcal{S}$.
2. Apply the fine-tuned model to the unlabeled target language data $\mathcal{D}_\mathcal{T}$ to obtain noisy predictive labels $\hat{y}^\mathcal{T}$.

**Stage 2: LLM Data Augmentation**
1. Input the predicted labels $\hat{y}^\mathcal{T}$ into the LLM, prompting it to generate natural sentences $\hat{x}^\mathcal{T}$ in the target language aligned with the labels.
2. Construct the pseudo-labeled dataset $\mathcal{D}_\mathcal{G} = \{(\hat{x}_i^\mathcal{T}, \hat{y}_i^\mathcal{T})\}$.
3. Merge $\mathcal{D}_\mathcal{G}$ with $\mathcal{D}_\mathcal{S}$ and further train the ABSA model on this mixed dataset.

### Key Designs

**Support for Multiple Architectures of ABSA Models**:

- **Encoder models** (mBERT, XLM-R): Sequence labeling approach, using BIO tagging + 3 sentiment polarities (POS/NEG/NEU) for token-level prediction.
- **Encoder-Decoder models** (mT5): Text generation approach, outputting in the format of "[A] aspect [P] polarity".
- **Decoder-only models** (LLaMA 3.1, Orca 2): Autoregressive generation approach.

**Quality Control for LLM Generation**:

1. **Preprocessing**: Ensure that the predicted labels contain at least one sentiment element.
2. **Prompt Design**: Specify the target language, instruct the LLM not to introduce additional sentiment elements, and provide 10 few-shot examples from the source language.
3. **Post-processing Filtering**:

    - Filter out generated instances that lack the predicted aspect terms.
    - Filter out instances where the ABSA model's re-prediction on the generated text is inconsistent with the original label.

**Handling Class Imbalance**: Modify 20% of the overrepresented positive sentiment samples to generate new instances with a 60% probability of neutral sentiment and a 40% probability of negative sentiment.

### Loss & Training

- Encoder models use token-level cross-entropy loss: $\mathcal{L} = \frac{1}{|\mathcal{D}|}\sum -\frac{1}{n}\sum y_i \log P_\Theta(y_i|x_i)$
- Encoder-Decoder models utilize sequence-level cross-entropy loss.
- Training proceeds in two steps: first fine-tuning on $\mathcal{D}_\mathcal{S}$, then continuing fine-tuning on $\mathcal{D}_\mathcal{S} \cup \mathcal{D}_\mathcal{G}$.
- Use the source language validation set for model selection, ensuring a truly unsupervised setting.

## Key Experimental Results

### Main Results

**Datasets**: SemEval-2016, containing restaurant reviews in six languages: English (en), Spanish (es), French (fr), Dutch (nl), Russian (ru), and Turkish (tr).

**Main Results** (micro-F1, average of 5 runs):

| Method | mBERT Avg | XLM-R Avg |
|------|-----------|-----------|
| Zero-shot | 45.68 | 60.35 |
| Equi-XABSA (Prev. SOTA) | 54.40 | 63.47 |
| LACA_LLaMA8 | 56.25 | 65.18 |
| LACA_Orca13 | 57.07 | 66.18 |
| **LACA_LLaMA70** | **57.29** | **66.35** |
| Supervised Upper Bound | 61.34 | 67.15 |

**Extension to More Backbone Models** (Avg F1):

| Backbone | Zero-shot | +LACA_LLaMA70 | Gain |
|----------|-----------|---------------|------|
| mBERT | 45.68 | 57.29 | +11.61 |
| XLM-R | 60.35 | 66.35 | +6.00 |
| mT5 | 59.77 | 65.90 | +6.13 |
| LLaMA 3.1 | 63.79 | 68.75 | +4.96 |

### Key Findings

1. **LACA comprehensively outperforms translation-based methods**: It outperforms Equi-XABSA by 1.50% on mBERT and by 2.62% on XLM-R.
2. **XLM-R + LACA approaches supervised performance**: It matches the supervised performance in Spanish (71.89 vs 71.93) and even outperforms it in Dutch (65.35 vs 64.28).
3. **English-centric Orca 2 13B performs exceptionally well**: Despite being primarily English-oriented, it nearly matches the performance of the multilingual LLaMA 3.1 70B, likely benefited by its advanced reasoning capabilities.
4. **Scaling effect of LLMs**: LLaMA 70B > Orca 13B ≈ LLaMA 8B, though larger models suffer from slower inference.
5. **Language similarity affects performance**: Spanish, which is closer to English, yields the best results, whereas Russian exhibits slightly lower gains due to larger linguistic family differences.
6. **Fine-tuned LLMs outperform small multilingual models**: LLaMA 3.1 as an ABSA model itself achieves the best overall performance.

## Highlights & Insights

1. **Core Innovation—Replacing Translation with Generation**: Unlike translation methods that generate semantically redundant data, LLM generation provides greater semantic diversity, thus enhancing model generalization.
2. **Elegant Handling of Noisy Labels**: By having the LLM generate aligned texts based on (potentially noisy) predicted labels, rather than directly training on raw noisy (sentence, prediction) pairs, the adverse effect of prediction noise is effectively mitigated.
3. **Eliminating Translation Tool Dependencies**: The framework is entirely independent of third-party translation resources, making it applicable to low-resource languages where translation systems perform poorly.
4. **Strong Generalization**: LACA supports three major model families: encoder, encoder-decoder, and decoder-only architectures.
5. **Practical Feasibility**: XLM-R + LACA approaches supervised performance without needing any labeled target-language data.

## Limitations & Future Work

1. **Dependence on LLM Generation Quality in Target Languages**: Generation quality can degrade for languages underrepresented in the LLM's pretraining corpus (such as Turkish).
2. **Computational Cost**: Employing LLaMA 70B for data generation incurs high computational costs.
3. **Error Propagation in the Two-Stage Pipeline**: The performance of the first-stage prediction directly determines the label accuracy of the LLM-generated data.
4. **Verification Limited to Restaurant Reviews**: The framework has not been verified in other domains (e.g., electronics, hotels, etc.).
5. **Unexplored Iterative Optimization**: Future work could examine multi-round generation-training-prediction iterations to further boost performance.

## Related Work & Insights

- **A New Paradigm for Cross-lingual Data Augmentation**: Replacing traditional translation tools with LLMs is readily generalized to other cross-lingual NLP tasks.
- **Improving Self-training**: Distinct from directly utilizing noisy predictions as pseudo-labels, LACA filters and purifies the pseudo-labeled data via LLM-conditional generation.
- **Leveraging LLMs for Low-Resource Scenarios**: This paper demonstrates the significant potential of LLMs for data augmentation in low-resource target languages.
- Zhang et al. (2021)'s ACS-Distill presents approaches for distillation on unlabeled target language data.
- Lin et al. (2024)'s Equi-XABSA handles class imbalance, which naturally complements LACA's imbalance mitigation strategy.

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 7 | The concept of replacing translation with LLM data augmentation is novel, though the overall framework remains fairly straightforward. |
| Experimental Thoroughness | 9 | Comprehensive comparison across 6 languages, 5 backbone architectures, and various LLMs. |
| Writing Quality | 8 | Well-structured with detailed experimental analysis. |
| Value | 8 | Achieves performance near supervised settings while bypassing the need for translation APIs. |
| Overall Score | 8 | A solid piece of work with extensive experimentation and high practical utility. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)
- [\[NeurIPS 2025\] Quantifying Climate Policy Action and Its Links to Development Outcomes: A Cross-National Data-Driven Analysis](../../NeurIPS2025/multilingual_mt/quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Comparative Analysis of Multilingual Hate Speech Detection](comparative_analysis_of_multilingual_hate_speech_detection.md)

</div>

<!-- RELATED:END -->
