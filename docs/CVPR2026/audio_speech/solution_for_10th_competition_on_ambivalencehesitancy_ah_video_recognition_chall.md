---
title: >-
  [Paper Note] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion
description: >-
  [CVPR2026][Audio & Speech][Multimodal Fusion] For the Ambivalence/Hesitancy (A/H) recognition task of the 10th ABAW competition, this paper proposes a divergence-based multimodal fusion strategy that explicitly models cross-modal conflict by computing pairwise absolute differences among embeddings from three modalities — visual (AU), audio (Wav2Vec 2.0), and text (BERT) — achieving a Macro F1 of 0.6808 on the BAH dataset, substantially surpassing the baseline of 0.2827.
tags:
  - CVPR2026
  - "Audio & Speech"
  - Multimodal Fusion
  - Ambivalence/Hesitancy Recognition
  - Action Units
  - Cross-Modal Conflict
  - Affective Computing
date: 2026-05-08
content_hash: 2e380e21d4e38227
---

# Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion

**Conference**: CVPR2026
**arXiv**: [2603.16939](https://arxiv.org/abs/2603.16939)
**Authors**: Aislan Gabriel O. Souza, Agostinho Freire, Leandro Honorato Silva et al. (Universidade de Pernambuco)
**Area**: Audio & Speech
**Keywords**: Multimodal Fusion, Ambivalence/Hesitancy Recognition, Action Units, Cross-Modal Conflict, Affective Computing

## TL;DR

For the Ambivalence/Hesitancy (A/H) recognition task of the 10th ABAW competition, this paper proposes a divergence-based multimodal fusion strategy that explicitly models cross-modal conflict by computing pairwise absolute differences among embeddings from three modalities — visual (AU), audio (Wav2Vec 2.0), and text (BERT) — achieving a Macro F1 of 0.6808 on the BAH dataset, substantially surpassing the baseline of 0.2827.

## Background & Motivation

Ambivalence and Hesitancy represent a class of complex affective states whose defining characteristic is the coexistence of conflicting emotions or intentions regarding behavioral change. Unlike discrete emotions such as happiness or anger, A/H does not manifest as a fixed facial expression or vocal tone, but rather as **inconsistency** across communication channels — the contradiction among what a person says, how they say it, and their facial expression constitutes the essential signal of A/H.

Existing multimodal affective computing methods exhibit a fundamental assumption bias in their fusion strategies:

- **Concatenation Fusion**: Directly concatenates features from all modalities end-to-end, assuming complementary relationships among modalities and delegating cross-modal interaction learning to the classifier. This implicit learning struggles to capture subtle cross-modal conflict signals under limited data.
- **Late Blending**: Fuses independent predictions from each modality at the decision level, equally unable to explicitly model inter-modal divergence.
- **Co-attention Fusion**: Improves inter-modal interaction modeling but incurs higher computational cost, and the physical interpretation of attention weights is less transparent than direct difference measures.

The common limitation of these approaches is treating modalities as "complementary information sources" rather than "potentially conflicting signal sources." For A/H detection, the diagnostically valuable signal lies precisely in **inter-modal inconsistency** — for example, a facial smile paired with a hesitant vocal tone, or affirmative speech accompanied by a tense facial expression.

Furthermore, in the preceding ABAW-8 competition, Savchenko achieved competitive results using EmotiEffLib facial descriptors with Wav2Vec 2.0 and RoBERTa via late blending; Hallmen et al. found the text modality to provide the strongest signal. However, neither work explicitly modeled cross-modal conflict, which is the defining characteristic of A/H.

Based on this insight, the authors argue: since A/H is theoretically defined as "coexisting conflicting signals," the fusion module should directly compute inter-modal **divergence/discrepancy** rather than simply stacking representations. This rationale is both concise and theoretically well-motivated.

## Method

### Overall Architecture

The system employs a three-modality pipeline: visual (facial Action Units) → audio (Wav2Vec 2.0) → text (BERT). Each modality is independently encoded, then processed through BiLSTM + Attention Pooling to generate fixed-dimensional representations. These are projected into a shared embedding space via projection layers, and finally classified through a divergence fusion module.

### Key Designs

**Visual Modality**: Py-Feat is used to extract 20 Action Units (AUs) frame-by-frame from pre-cropped face images at a sampling rate of 1 frame per 3 frames (approximately 10 fps). To model temporal dynamics, sliding window statistics are computed — within a window of size $W=16$ and stride $S=8$, four statistics (mean, standard deviation, slope, range) are computed for each AU, yielding an 80-dimensional window descriptor. The key insight is that **A/H manifests as temporal fluctuations in facial actions rather than as a fixed expression**.

**Audio Modality**: Audio is extracted at 16 kHz mono and encoded using pretrained Wav2Vec 2.0 (wav2vec2-base-960h), producing 768-dimensional embeddings at a temporal resolution of approximately 50 Hz.

**Text Modality**: BERT-base encodes the transcribed text, taking the 768-dimensional [CLS] token output, with the last two layers fine-tuned at a reduced learning rate to adapt to the downstream task.

### Loss & Training

**Temporal Modeling and Attention Pooling**: Temporal features from the visual and audio modalities are each processed by 2-layer BiLSTMs (hidden dimension 64), followed by attention pooling to compress variable-length sequences into fixed-length vectors. Each modality's output is then projected via a linear layer into a shared $D=128$-dimensional embedding space, yielding $\mathbf{h}'_v$, $\mathbf{h}'_a$, and $\mathbf{h}'_t$.

**Three Fusion Strategies**: This is the core design of the paper — the authors systematically compare three fusion strategies:

**Fusion A (Implicit Fusion)**: Conventional concatenation, directly concatenating the three modality embeddings:
$$\mathbf{f}_A = [\mathbf{h}'_v;\, \mathbf{h}'_a;\, \mathbf{h}'_t]$$
The classifier must learn inter-modal relationships from a $3 \times 128 = 384$-dimensional vector.

**Fusion B (Divergence Fusion)**: Computes element-wise absolute differences between all modality pairs:
$$\mathbf{f}_B = [|\mathbf{h}'_v - \mathbf{h}'_a|;\, |\mathbf{h}'_v - \mathbf{h}'_t|;\, |\mathbf{h}'_a - \mathbf{h}'_t|]$$
This design directly captures cross-modal conflict — if the visual and audio representations are consistent, the difference approaches zero; if a contradiction exists, the difference produces significant responses in the relevant dimensions. The dimensionality is likewise $3 \times 128 = 384$.

**Fusion C (Combined Fusion)**: Retains both original embeddings and divergence information:
$$\mathbf{f}_C = [\mathbf{f}_A;\, \mathbf{f}_B]$$
With dimensionality 768, this is the most information-rich variant but requires more parameters.

**Classification and Training**: The fused vector is passed through a 3-layer MLP (with Dropout $p=0.3$) for binary classification. Training details:
- Loss: BCEWithLogitsLoss with class weights (to address class imbalance)
- Optimizer: AdamW, with learning rate $5 \times 10^{-5}$ for BERT parameters and $5 \times 10^{-4}$ for others
- Learning rate schedule: cosine annealing over 30 epochs
- Gradient clipping threshold 1.0, early stopping patience of 8

**Statistical Validation of AU Temporal Variability**: Mann-Whitney U tests are conducted on 1,132 videos to validate the discriminative power of AU features. The results confirm that the **temporal variability (standard deviation) of AUs** is the primary visual discriminant for A/H, rather than mean AU intensity — suggesting that A/H manifests as facial "instability," i.e., frequent fluctuations of facial muscles rather than sustained expressions.

## Key Experimental Results

### Table 1: AU Feature Discriminability Analysis (Mann-Whitney U, N=1132)

| Feature | Statistic | A/H vs. Non-A/H | Effect Size \|r\| |
|---|---|---|---|
| AU06 (Cheek Raiser) | std | 0.076 vs 0.059 | 0.186 |
| AU09 (Nose Wrinkler) | std | 0.095 vs 0.084 | 0.186 |
| AU12 (Lip Corner Puller) | std | 0.089 vs 0.068 | 0.172 |
| AU26 (Jaw Drop) | zcr | 0.421 vs 0.384 | 0.168 |
| AU02 (Outer Brow Raiser) | std | 0.110 vs 0.102 | 0.149 |

All features remain significant after Bonferroni correction. Effect sizes \|r\| < 0.2 indicate small effects, explaining why purely visual models have a performance ceiling (approximately 0.56 F1) and multimodal fusion is necessary.

### Table 2: BAH Dataset Results (Macro F1)

| Model | Val F1 | Test F1 |
|---|---|---|
| **Single-Modal** | | |
| Visual AUs (XGBoost) | 0.6194 | 0.5642 |
| Audio Wav2Vec (LSTM) | 0.5218 | **0.6141** |
| Text BERT | 0.5758 | 0.5904 |
| **Multimodal (Raw AU)** | | |
| Fusion A (Implicit Concat) | 0.6788 | 0.6604 |
| Fusion B (Divergence Fusion) | 0.6524 | **0.6808** |
| Fusion C (Combined Fusion) | 0.6700 | 0.6766 |
| **Multimodal (Window AU)** | | |
| Fusion B (Divergence Fusion) | **0.6912** | 0.6602 |
| **Competition Baseline** | — | 0.2827 |

Key findings:
- Audio is the strongest single modality (Test F1 = 0.6141), consistent with prior literature.
- Fusion B (divergence fusion) achieves the best test performance of 0.6808, a **140.8%** improvement over the baseline.
- Window AU Fusion B achieves the highest validation F1 (0.6912) but degrades on the test set (0.6602), reflecting overfitting due to the limited training set (only 598 videos).
- Fusion C, despite being the most information-rich, does not outperform pure divergence Fusion B, indicating that divergence signals are sufficient to capture A/H characteristics.

## Highlights & Insights

- **Theory-Driven Fusion Design**: The divergence fusion strategy is directly motivated by the theoretical definition of A/H — coexisting conflicting signals — rather than empirically stacking modalities. This paradigm of "deriving architecture from task definition" merits broader adoption.
- **Interpretable Visual Features**: Choosing Action Units over black-box deep features enables statistical analysis to reveal which facial behaviors correlate with A/H (e.g., fluctuations in cheek and nose raising), enhancing the interpretability of the approach.
- **Facial Instability Hypothesis**: Statistical analysis confirms that A/H manifests as temporal fluctuations in AUs rather than as a fixed expression, a finding that suggests complex affective states may be better characterized by dynamic rather than static features.
- **Simplicity and Effectiveness**: The overall architecture does not rely on complex attention mechanisms or Transformers; the BiLSTM + element-wise difference scheme is computationally efficient and particularly well-suited for small-data settings.

## Limitations & Future Work

- **Limited Data Scale**: The training set contains only 598 videos. Window AU features perform best on the validation set but degrade on the test set, clearly indicating overfitting; richer feature representations require larger datasets.
- **Overly Simple Divergence Measure**: Only element-wise absolute differences are used to measure cross-modal conflict, without considering higher-order distributional divergences (e.g., KL divergence, MMD), potentially missing nonlinear conflict patterns.
- **Absence of Temporal Alignment**: The three modalities operate at different temporal resolutions (visual 10 fps, audio 50 Hz, text global). The current approach compresses each modality to a single vector via attention pooling before computing divergence, discarding fine-grained temporal alignment information.
- **Limited Visual Representation**: Only AUs are used as visual features, and their effect sizes are small (\|r\| < 0.2). Alternative representations such as deep facial embeddings (e.g., EmotiEffLib) or MediaPipe blendshapes remain unexplored.
- **Generalizability Unverified**: Evaluation is performed solely on the BAH dataset; the generalizability of the proposed method to other affective conflict detection datasets has not been validated.

## Related Work & Insights

- **ABAW Competition Series**: From the 1st to the 10th edition, the competition has progressively expanded from discrete emotion and valence-arousal recognition on Aff-Wild2 to more complex affective tasks such as A/H recognition.
- **ABAW-8 Solutions**: Savchenko achieved top performance using EmotiEffLib + Wav2Vec 2.0 + RoBERTa with late blending; Hallmen et al. employed ViT-Huge + Wav2Vec 2.0 + GTE and found the text signal to be strongest. Neither work explicitly modeled cross-modal conflict.
- **Multimodal Fusion**: Fusion strategies have evolved from early concatenation and late blending to attention bottleneck approaches, yet dedicated designs for "modal conflict" remain scarce.
- **Action Unit Detection**: EAC-Net, JAA-Net, and related methods have advanced deep learning-based AU detection, but the temporal dynamics of AUs in A/H discrimination represent a novel direction explored in this work.
- **Paper Positioning**: Starting from the theoretical definition of A/H, this paper proposes a "divergence-as-feature" fusion paradigm, filling the gap in explicit cross-modal conflict modeling within existing fusion methods.

## Rating

- Novelty: ⭐⭐⭐ — The divergence fusion idea is conceptually clean with clear theoretical motivation, but technically reduces to element-wise differences, limiting the degree of innovation.
- Experimental Thoroughness: ⭐⭐⭐ — The comparison of three fusion strategies plus single-modal ablations and statistical analysis is relatively complete, but evaluation is confined to a single dataset with no cross-dataset validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with internally consistent logic between statistical analysis and method design; meets competition paper writing standards.
- Value: ⭐⭐⭐ — The divergence fusion paradigm offers reference value for tasks requiring detection of "modal conflict," such as sarcasm detection and deception detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valen.md)
- [\[ICLR 2026\] TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](../../ICLR2026/audio_speech/triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)

</div>

<!-- RELATED:END -->
