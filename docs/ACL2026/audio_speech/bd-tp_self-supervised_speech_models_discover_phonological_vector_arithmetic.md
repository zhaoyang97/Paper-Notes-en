---
title: >-
  [Paper Note] [b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
description: >-
  [ACL 2026][Audio & Speech][Self-supervised speech models] The paper systematically demonstrates that linear phonological feature vectors exist in the representation space of self-supervised speech models (S3Ms). These ve…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Self-supervised speech models"
  - "Phonological vector arithmetic"
  - "Speech representation structure"
  - "Acoustic controllable synthesis"
  - "Cross-lingual generalization"
date: 2026-05-08
content_hash: 48775b7f54a58e76
---

# [b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic

**Conference**: ACL 2026  
**arXiv**: [2602.18899](https://arxiv.org/abs/2602.18899)  
**Area**: Audio & Speech / Speech Representation Learning  
**Keywords**: Self-supervised speech models, Phonological vector arithmetic, Speech representation structure, Acoustic controllable synthesis, Cross-lingual generalization

## TL;DR

The paper systematically demonstrates that linear phonological feature vectors exist in the representation space of self-supervised speech models (S3Ms). These vectors satisfy word2vec-like vector arithmetic relationships, and their scaling factors exhibit a continuous correlation with acoustic measurements.

## Background & Motivation

**Background**: Self-supervised speech models (such as wav2vec 2.0, HuBERT, and WavLM) have demonstrated powerful performance in downstream tasks like speech recognition, synthesis, and spoken language understanding. Existing studies show that S3Ms encode rich phonetic information, where distances in the representation space reflect acoustic similarity and form clusters corresponding to phonemic units.

**Limitations of Prior Work**: Although it is known *what* information S3Ms encode, there is still a lack of deep understanding regarding *how* this information is structured. Analogous to the classic semantic vector arithmetic in word2vec (king - man + woman ≈ queen), whether a similar compositional structure exists for phonological features in the speech representation space remains unexplored.

**Key Challenge**: S3Ms perform excellently across various tasks, but the internal structure of their representation space—specifically whether phonological features are encoded in a compositional and controllable manner—remains unclear.

**Goal**: To verify two hypotheses: (1) linear phonological feature vectors exist in the S3M representation space (Direction Hypothesis), and (2) the scaling factors of these vectors are continuously correlated with the degree of realization of acoustic features (Scale Hypothesis).

**Key Insight**: Borrow the methodology of vector analogy tests from word2vec and extend it to phonological features in the audio domain.

**Core Idea**: $[b] - [p] + [t] \approx [d]$ (voicing vector), implying that compositional phonological vectors exist in the representation space of speech models, and scaling these vectors allows for continuous control over the corresponding acoustic features.

## Method

### Overall Architecture

The study consists of two core experiments: (1) Direction Experiment—to verify if phonological vector arithmetic holds (whether linear directions satisfying analogy relationships exist); (2) Scale Experiment—to train a vocoder to map S3M representations back to speech signals, verifying the continuous correlation between scaling and acoustic measurements through vector scaling and resynthesis. Two datasets are used: TIMIT (English) and VoxAngeles (95 languages), covering a total of 96 languages.

### Key Designs

1.  **Phonological Analogy Construction and Cosine Similarity Evaluation**:
    *   Function: Systematically test if linear directions satisfying phonological analogies exist in S3M representations.
    *   Mechanism: Use PanPhon to extract 21-dimensional phonological feature vectors for each phoneme. Filter phoneme quadruplets based on feature difference consistency. Calculate $cos(r_{p1}, r_{p2} + r_{p3} - r_{p4})$ and compare it against same-phoneme and different-phoneme baselines. Define the success rate $S(Q)$ as the proportion of quadruplets satisfying the $cos^- < cos < cos^+$ ordering.
    *   Design Motivation: Use bootstrapping to construct 99% confidence intervals to ensure statistical reliability and avoid bias from single random samplings.

2.  **Scaling Phonological Vectors and Vocoder Inversion**:
    *   Function: Verify whether the scaling factor $\lambda$ of phonological vectors is continuously correlated with the degree of acoustic feature realization.
    *   Mechanism: Define a phonological vector as the difference between the average representations of all phonemes with and without a specific feature. Modify S3M representations by adding the scaled vector to target frames. Train a Vocos-based vocoder to resynthesize the modified representations into speech. Extract acoustic measurements (F1, F2, HNR, COG, etc.) and calculate the Spearman rank correlation with $\lambda$.
    *   Design Motivation: The Vocos vocoder is robust to out-of-distribution inputs, making it particularly suitable for analyzing artificially modified S3M representations.

3.  **Layer-wise Analysis and Vowel/Consonant Separation**:
    *   Function: Reveal how phonological information is encoded across different layers of S3Ms.
    *   Mechanism: Calculate success rates across 25 layers separately. It was found that WavLM exhibits three peaks—vowels peak in early intermediate layers, while consonants peak in late intermediate layers, with the final layer integrating all information. Perform fine-grained analysis by grouping phonological analogies into vowels and consonants.
    *   Design Motivation: Vowels and consonants have different acoustic-temporal characteristics (vowel cues are more localized, while consonant cues span larger time windows); this explores whether they are prioritized in different layers.

### Loss & Training

Vocoder training utilizes the standard Vocos framework, trained on LibriTTS (English) and FLEURS-R (multilingual). The core analysis does not involve model training but rather post-hoc probing of the representation spaces of existing pre-trained S3Ms.

## Key Experimental Results

### Main Results

Phonological analogy success rates on TIMIT for different models (at their best layers):

| Model | Best Success Rate | Best Layer |
| :--- | :--- | :--- |
| MelSpec | 0% | - |
| MFCC | 19% | - |
| wav2vec 2.0 | 61% | Intermediate |
| HuBERT | 94% | Final |
| WavLM | 92% | Final |

Success rates on VoxAngeles (95 languages):

| Model | Best Success Rate |
| :--- | :--- |
| MelSpec | 0% |
| MFCC | 19% |
| wav2vec 2.0 | 39% |
| HuBERT | 45% |
| WavLM | 93% |

Cross-lingual generalization: Out of 468 analogies, 316 (68%) included at least one phoneme not present in English. WavLM still achieved a 93% success rate.

### Ablation Study

Spearman correlation $\rho$ between scaling factor $\lambda$ and acoustic measurements for 8 phonological features (TIMIT, WavLM):

| Phonological Feature | Acoustic Measurement | Correlation $\rho$ | Expected Sign |
| :--- | :--- | :--- | :--- |
| High | F1 | -0.801 | - ✓ |
| Low | F1 | +0.908 | + ✓ |
| Back | F2 | -0.759 | - ✓ |
| Round | F2 | -0.833 | - ✓ |
| Nasal | F1BW | -0.441 | - ✓ |
| Sonorant | HNR | +0.649 | + ✓ |
| Strident | COG | +0.819 | + ✓ |
| Voice | COG | -0.720 | - ✓ |

All signs of the 8 correlations align with theoretical expectations.

### Key Findings

*   Phonological analogies hold consistently across 19 phonological features in S3M representation spaces, far exceeding spectral feature baselines.
*   WavLM maintains a 93% success rate in cross-lingual settings (95 languages), demonstrating strong generalization capabilities.
*   Vowel-related analogies peak in shallower layers, while consonant analogies peak in deeper layers—consistent with their differing temporal characteristics.
*   The scaling factor $\lambda$ is effective not only within the interpolation range ($|\lambda| \le 1$) but also maintains continuous correlation in the extrapolation range ($|\lambda| > 1$).
*   S3Ms trained only on English can generalize to phonological arithmetic for phonemes not present in English.

## Highlights & Insights

*   **Elegant Analogy**: Extending word2vec’s semantic vector arithmetic to phonological features in the audio domain is a conceptually simple yet profound contribution.
*   **Surprising Cross-lingual Generalization**: Models pre-trained only on English can encode the phonological structures of 96 languages, suggesting that S3Ms learn truly universal phonetic knowledge rather than just language-specific patterns.
*   **Large-scale Experiments**: The study covers 96 languages, 19 phonological features, 3 S3M models, and a layer-wise analysis across 25 layers.
*   **Potential for Controllable Speech Synthesis**: Achieving continuous control of acoustic features by scaling phonological vectors provides a new direction for interpretable speech synthesis.

## Limitations & Future Work

*   Only three English pre-trained S3Ms (wav2vec 2.0, HuBERT, WavLM) were tested; multilingual pre-trained models were not included.
*   The quality of vocoder resynthesis may introduce noise, affecting the accuracy of acoustic measurements.
*   The current analysis is mainly at the phoneme level and does not explore higher-level compositionality (e.g., syllables, prosody).
*   Future work could explore applications of phonological vectors in controllable voice conversion or speech enhancement.

## Related Work & Insights

*   **vs word2vec Analogy Test**: Unlike the 3CosAdd/3CosMul methods of Mikolov et al. (2013b), this paper uses a robust evaluation based on statistical confidence intervals.
*   **vs Traditional Speech Probing**: While probing studies focus on *what* information S3Ms encode, this paper further reveals the *compositional structure* of that information.
*   **vs Choi et al. (2024)**: Previous cluster analysis found that S3Ms form phoneme clusters; this work builds on that by discovering linear relationships between those clusters.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic proof of phonological vector arithmetic in S3Ms; concept is highly novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive analysis across 96 languages, 19 features, and multiple layers/models.
*   Writing Quality: ⭐⭐⭐⭐⭐ Fluent writing, clear figures, and an elegant introduction of the analogy.
*   Value: ⭐⭐⭐⭐ Deepens the understanding of S3M representation structures with implications for speech synthesis and analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them](how_tokenization_limits_phonological_knowledge_representation_in_language_models.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)

</div>

<!-- RELATED:END -->
