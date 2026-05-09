---
title: >-
  [Paper Note] [b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
description: >-
  [ACL 2026][Self-Supervised Learning][Self-Supervised Speech Models] This paper systematically demonstrates that linear phonological feature vectors exist in the representation space of self-supervised speech models (S3M), satisfying word2vec-like vector arithmetic relations, with their scaling factors continuously correlating with acoustic measurements.
tags:
  - ACL 2026
  - Self-Supervised Learning
  - Self-Supervised Speech Models
  - Phonological Vector Arithmetic
  - Speech Representation Structure
  - Acoustic Controllable Synthesis
  - Cross-Lingual Generalization
date: 2025-05-08
content_hash: 97e943e907fa3d89
---

# [b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic

**Conference**: ACL 2026  
**arXiv**: [2602.18899](https://arxiv.org/abs/2602.18899)  
**Area**: Audio & Speech / Speech Representation Learning  
**Keywords**: Self-Supervised Speech Models, Phonological Vector Arithmetic, Speech Representation Structure, Acoustic Controllable Synthesis, Cross-Lingual Generalization

## TL;DR

This paper systematically demonstrates that linear phonological feature vectors exist in the representation space of self-supervised speech models (S3M), satisfying word2vec-like vector arithmetic relations, with their scaling factors continuously correlating with acoustic measurements.

## Background & Motivation

**State of the Field**: Self-supervised speech models (e.g., wav2vec 2.0, HuBERT, WavLM) have demonstrated strong performance on downstream tasks including speech recognition, synthesis, and spoken language understanding. Prior work has shown that S3Ms encode rich phonetic information, with distance relations in representation space reflecting acoustic similarity and forming clusters corresponding to phoneme units.

**Limitations of Prior Work**: While it is known what information S3Ms encode, there is a lack of deep understanding of how this information is structured. By analogy to the classic semantic vector arithmetic in word2vec (king - man + woman ≈ queen), whether similar compositional structure exists in speech representation spaces has not been explored.

**Root Cause**: S3Ms perform excellently across various tasks, but the internal structure of their representation spaces — particularly whether phonological features are encoded in a composable, manipulable manner — remains unclear.

**Paper Goals**: Verify two hypotheses — (1) linear phonological feature vectors exist in S3M representation spaces (direction hypothesis), (2) the scaling factors of these vectors continuously correlate with the realization degree of acoustic features (scale hypothesis).

**Starting Point**: Adapting the word2vec vector analogy testing methodology and extending it to phonological features in the speech domain.

**Core Idea**: [b] - [p] + [t] ≈ [d] (voicing vector) — composable phonological vectors exist in speech model representation spaces, and scaling these vectors can continuously control the degree of corresponding acoustic features.

## Method

### Overall Architecture

The study consists of two core experiments: (1) Direction experiment — verifying whether phonological vector arithmetic holds (whether linear directions satisfying analogy relations exist); (2) Scale experiment — training a vocoder to inverse-map S3M representations to speech signals, then scaling phonological vectors and resynthesizing to verify continuous correlation between scale and acoustic measurements. Two datasets are used: TIMIT (English) and VoxAngeles (95 languages), covering 96 languages.

### Key Designs

1. **Phonological Analogy Construction and Cosine Similarity Evaluation**:

    - Function: Systematically test whether linear directions satisfying phonological analogies exist in S3M representations
    - Mechanism: Uses PanPhon to extract 21-dimensional phonological feature vectors for each phoneme, filters phoneme quadruplets through feature difference consistency, computes cos(r_p1, r_p2 + r_p3 - r_p4) and compares with same-phoneme and different-phoneme baselines; defines success rate S(Q) as the proportion of quadruplets satisfying the cos⁻ < cos < cos⁺ ordering
    - Design Motivation: Bootstrap construction of 99% confidence intervals ensures statistical reliability, avoiding biases from single random samples

2. **Phonological Vector Scaling Modification and Vocoder Inverse Mapping**:

    - Function: Verify whether the scaling factor λ of phonological vectors continuously correlates with acoustic feature realization degree
    - Mechanism: Defines phonological vectors as the difference between average representations of all phonemes with/without a given feature; adds scaled vectors to target frames to modify S3M representations; trains a Vocos-based vocoder to resynthesize modified representations into speech; extracts acoustic measurements (F1, F2, HNR, COG, etc.) and computes Spearman rank correlation with λ
    - Design Motivation: The Vocos vocoder is robust to out-of-distribution inputs, making it particularly suitable for analyzing artificially modified S3M representations

3. **Layer-Wise Analysis and Vowel/Consonant Separation**:

    - Function: Reveal how different layers of S3M encode phonological information
    - Mechanism: Computes success rate separately across 25 layers, finding that WavLM exhibits three peaks — vowels peak early in middle layers, consonants peak later in middle layers, and the final layer integrates all information; phonological analogies are analyzed at finer granularity by grouping into vowel/consonant categories
    - Design Motivation: Vowels and consonants have different acoustic-temporal properties (vowel cues are more localized, consonant cues span larger temporal windows), exploring whether they are preferentially encoded in different layers

### Loss & Training

The vocoder is trained using the standard Vocos framework on LibriTTS (English) and FLEURS-R (multilingual). The core analysis involves no model training but rather post-hoc probing of existing pretrained S3M representation spaces.

## Key Experimental Results

### Main Results

Phonological analogy success rates on TIMIT across different models (best layer):

| Model | Best Success Rate | Best Layer |
|-------|------------------|------------|
| MelSpec | 0% | - |
| MFCC | 19% | - |
| wav2vec 2.0 | 61% | Middle layers |
| HuBERT | 94% | Last layer |
| WavLM | 92% | Last layer |

Success rates on VoxAngeles (95 languages):

| Model | Best Success Rate |
|-------|------------------|
| MelSpec | 0% |
| MFCC | 19% |
| wav2vec 2.0 | 39% |
| HuBERT | 45% |
| WavLM | 93% |

Cross-lingual generalization: Of 468 analogies, 316 (68%) involve at least one phoneme not present in English; WavLM still achieves 93% success rate.

### Ablation Study

Spearman correlation between scaling factor λ and acoustic measurements for 8 phonological features (TIMIT, WavLM):

| Phonological Feature | Acoustic Measurement | Correlation ρ | Expected Sign |
|---------------------|---------------------|---------------|---------------|
| High | F1 | -0.801 | - ✓ |
| Low | F1 | +0.908 | + ✓ |
| Back | F2 | -0.759 | - ✓ |
| Round | F2 | -0.833 | - ✓ |
| Nasal | F1BW | -0.441 | - ✓ |
| Sonorant | HNR | +0.649 | + ✓ |
| Strident | COG | +0.819 | + ✓ |
| Voice | COG | -0.720 | - ✓ |

All 8 features show correlation signs consistent with theoretical expectations.

### Key Findings

- Phonological analogies in S3M representation spaces consistently hold across 19 phonological features, far exceeding spectral feature baselines
- WavLM maintains 93% success rate in cross-lingual settings (95 languages), demonstrating strong generalization ability
- Vowel-related analogies peak at shallower layers while consonant analogies require deeper layers — consistent with the different temporal properties of these phoneme classes
- Scaling factor λ maintains continuous correlation not only in the interpolation range (|λ| ≤ 1) but also in the extrapolation range (|λ| > 1)
- S3Ms trained only on English generalize to phonological arithmetic involving phonemes that do not exist in English

## Highlights & Insights

- **Elegant analogy**: Extending word2vec's semantic vector arithmetic to phonological features in the speech domain — conceptually simple yet profound
- **Surprising cross-lingual generalization**: Models pretrained only on English encode phonological structure across 96 languages, suggesting S3Ms learn truly universal phonetic knowledge rather than language-specific patterns
- **Grand experimental scale**: Covering 96 languages, 19 phonological features, 3 S3M models, and 25-layer layer-wise analysis
- **Potential for controllable speech synthesis**: Continuous control of acoustic features through phonological vector scaling offers a new approach to interpretable speech synthesis

## Limitations & Future Work

- Only tested 3 English-pretrained S3Ms (wav2vec 2.0, HuBERT, WavLM); multilingual pretrained models not included
- Vocoder resynthesis quality may introduce noise, affecting acoustic measurement accuracy
- Current analysis focuses on the phoneme level; higher-level compositionality (e.g., syllable, prosody) is not explored
- Future work could explore using phonological vectors for controllable voice conversion or speech enhancement applications

## Related Work & Insights

- **vs word2vec analogy test**: This paper's analogy testing methodology differs from Mikolov et al. (2013b)'s 3CosAdd/3CosMul, using statistical confidence interval-based evaluation for greater robustness
- **vs Traditional speech probing**: Probing studies only focus on what information S3Ms encode; this paper further reveals the compositional structure of that information
- **vs Choi et al. (2024)**: Prior cluster analysis found that S3Ms form phoneme clusters; this paper discovers linear relationships between clusters

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic demonstration of phonological vector arithmetic in S3Ms — highly novel concept
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 96 languages, 19 features, multi-model multi-layer analysis — extremely comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent writing, clear figures, elegant analogy introduction
- Recommendation: ⭐⭐⭐⭐ Deepens understanding of S3M representation structure, inspiring for speech synthesis and analysis

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/self_supervised/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)

<!-- RELATED:END -->
