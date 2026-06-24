---
title: >-
  [Paper Note] Probing Subphonemes in Morphology Models
description: >-
  [ACL 2025][Interpretability][Morphological Inflection] This paper proposes a language-agnostic probing method to investigate how Transformer models trained on morphological inflection tasks implicitly learn phonological features. It is found that local features (such as final devoicing) are well-encoded in phoneme embeddings, while long-distance dependencies (such as vowel harmony) are more prominent in the contextualized representations of encoder layers.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Morphological Inflection"
  - "Phonological Feature Probing"
  - "Transformer"
  - "Minimum Description Length"
  - "Phoneme Embeddings"
date: 2026-05-08
content_hash: 1f82fd2f03e09b18
---

# Probing Subphonemes in Morphology Models

**Conference**: ACL 2025  
**arXiv**: [2505.11297](https://arxiv.org/abs/2505.11297)  
**Code**: [Yes](https://github.com/MeLeLBGU/probing-subphonemes)  
**Area**: Interpretability  
**Keywords**: Morphological Inflection, Phonological Feature Probing, Transformer, Minimum Description Length, Phoneme Embeddings

## TL;DR

This paper proposes a language-agnostic probing method to investigate how Transformer models trained on morphological inflection tasks implicitly learn phonological features. It is found that local features (such as final devoicing) are well-encoded in phoneme embeddings, while long-distance dependencies (such as vowel harmony) are more prominent in the contextualized representations of encoder layers.

## Background & Motivation

### Background

The Transformer architecture has achieved SOTA performance on morphological inflection tasks, but its cross-lingual generalization capability is limited. A potential explanation is the extent to which the models capture implicit phonological and subphonemic phenomena. Prior work has explored information in neural phoneme embeddings but lacks systematic, cross-lingual quantitative analysis.

### Key Challenge

- Morphology and phonology interact meaningfully in many languages (e.g., vowel harmony, consonant assimilation). Investigating how this correspondence is reflected in model representations is crucial.
- Prior studies on phoneme embeddings mostly focused on a single language or feature, lacking a systematic cross-lingual and multi-feature evaluation.
- Traditional probing methods (e.g., accuracy/F1) have limitations: they can perform well even on random labels or randomly initialized representations.

### Research Hypothesis

When models are trained on reliable phonological representations, they acquire morphology-dependent subphonemic features (such as VOICE, ROUND). This acquisition depends on how heavily the language relies on these features.

## Method

### Overall Architecture

The experimental pipeline of this paper consists of three stages:
1. Train a phoneme-based Transformer model on a morphological task for a specific language.
2. Use a probing classifier to probe phonological features from the model embeddings.
3. Analyze probing results using the Minimum Description Length (MDL) method.

### Key Designs

1. **Phoneme-based Transformer**: A character-level encoder-decoder Transformer is used (SOTA architecture for the SIGMORPHON 2017 shared task), where orthographic forms are transcribed into IPA phonemes using Epitran. Two versions are trained: the inflection model (morphological inflection task) and the lemma copying model (where inflection properties are replaced with COPY, producing the same output as input). The encoder and decoder share the embedding table (weight tying).

2. **Phoneme Probe**: A probing classifier is trained for each phonological feature, with phoneme embeddings as input and PanPhon-extracted feature values (+/-/0) as labels. Due to the limited size of phoneme inventories in individual languages, diverse embeddings are generated via multi-seed training for data augmentation, and 3x oversampling is applied. t-SNE visualization confirms the lack of clustering across different seeds, validating the effectiveness of the augmentation strategy.

3. **Harmony Probe**: Probes are designed to study the models' capacity to encode long-distance phonological dependencies. Using nonce words, the contextualized phoneme vectors from the final encoder layer are used as inputs. The probe classifies three harmony types: all +, all -, and disharmonious (containing both + and -). Probes are trained separately for vowel harmony and consonant harmony.

4. **MDL Probing Method**: An information-theoretic approach replaces traditional accuracy metrics. Minimum Description Length is computed using online coding by splitting data into segments, training the probe on each prefix, and measuring the cross-entropy loss of the next segment. Comparison is normalized using compression score $\mathcal{C} = \frac{n \log_2 K}{L}$, where a higher score indicates stronger feature encoding.

### Loss & Training

- The probing classifier uses an MLP with two 100-neuron hidden layers.
- The loss function uses inverse weight of feature frequencies to address class imbalance.
- Control task: Labels are randomly shuffled as a baseline control to verify the validity of the compression score.

## Key Experimental Results

### Main Results — Phoneme Probing Compression Scores

| Feature | Turkish | Hungarian | Hebrew | Russian | Spanish | German | Georgian |
|------|---------|---------|---------|------|---------|------|-----------|
| VOICE | High | Medium | Medium | Medium | Medium | Medium | Medium |
| CONTINUANT | High | Medium | Medium | Medium | Medium | Medium | Medium |
| LONG | Medium | **Highest** | - | - | - | - | - |
| Control Task (Random Labels) | <1.0 | <1.0 | <1.0 | <1.0 | <1.0 | <1.0 | <1.0 |

### Harmony Probing Results

| Probe Type | Inflection Model | Copy Model |
|---------|---------|---------|
| ROUND (Vowel Harmony) | High compression score (Turkish, Hungarian) | Lower |
| BACK (Vowel Harmony) | High compression score (Turkish, Hungarian) | Lower |
| Consonant Harmony | Effective for some features | Lower |

### Key Findings

1. **Local vs. Long-distance Features**: Local phonological features (e.g., VOICE and CONTINUANT in final devoicing in Turkish) are well-encoded in phoneme embeddings. Long-distance features (e.g., ROUND and BACK in vowel harmony) are more pronounced in contextualized embeddings of the encoder.
2. **LONG Feature in Hungarian**: This achieves the highest compression score across all languages and features in phoneme probing, reflecting the morphological importance of gemination/degemination in Hungarian.
3. **Copy Model Performs Unexpectedly Well**: In phoneme probing, the performance of the copy model is comparable to or even better than the inflection model, which may be due to dataset noise.
4. **Control Task Validation**: All control task scores are below 1.0, validating the reliability of the MDL compression score as an indicator of phonological feature representation.

## Highlights & Insights

- **Methodological Contribution**: Combines language-agnostic probe design, information-theoretic MDL evaluation, and cross-lingual comparison to provide a new analytical tool for interpreting morphological models.
- **Practical Implication**: Explains why adding subphonemic features barely improves performance in morphological inflection — because the model already learns these features implicitly.
- **Transfer Learning Implication**: The success of cross-lingual transfer learning in morphological inflection may stem from the model acquiring approximately universal subphonemic features.
- **Support for Pre-training Strategies**: The strong performance of the copy model supports the common practice of first pre-training with a copy task before shifting to the inflection task.

## Limitations & Future Work

- The methodology depends on grapheme-to-phoneme tools like Epitran and the quality of character-level Transformers, which may introduce tool bias.
- Experiments were conducted on only 7 languages, which limits the morphological diversity covered.
- Data hallucination (augmentation) was not used, as it might generate phonologically invalid words.
- The reasons behind the performance discrepancy between the copy and inflection models are not fully explored and might be affected by data noise.
- Future work could investigate variance across different models and languages.

## Related Work & Insights

- Extends the research of Muradoglu & Hulden (2023) on the phonological capabilities of Transformer models, directly proving that models explicitly encode phonological features using interpretability methods.
- The MDL probing method is derived from Voita & Titov (2020), which is more robust than traditional probing.
- Complements the findings of Guriel et al. (2023) that adding subphonemic features barely improves performance.
- Provides a new explanation for transfer learning methods (McCarthy et al., 2019; Elsner, 2021).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The method itself is a combination of existing techniques (probing + MDL), but applying it to subphonemic feature analysis is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 7 languages and multiple features with control tasks, though the presentation of quantitative data could be more thorough.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure and rigorous argument with coherent logic across experiments.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for morphological modeling and transfer learning practices, although the scope of impact is limited to the subfield of computational linguistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](../../ACL2026/interpretability/experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Probing for Reading Times](../../ACL2026/interpretability/probing_for_reading_times.md)
- [\[ACL 2025\] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)
- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](../../ACL2026/interpretability/mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)

</div>

<!-- RELATED:END -->
