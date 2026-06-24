---
title: >-
  [Paper Note] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
description: >-
  [CVPR 2025 (ABAW Workshop)][Human Understanding][Ambivalence/Hesitancy Recognition] This paper proposes a multimodal approach for video-level ambivalence/hesitancy (A/H) recognition, integrating four modalities: scene (VideoMAE), face (EmotionEfficientNetB0), audio (EmotionWav2Vec2.0+Mamba), and text (EmotionDistilRoBERTa). Through a prototype-augmented Transformer fusion model, it achieves an average MF1 of 83.25%, with a five-model ensemble ultimately reaching 71.43% on the…
tags:
  - "CVPR 2025 (ABAW Workshop)"
  - "Human Understanding"
  - "Ambivalence/Hesitancy Recognition"
  - "Multimodal Fusion"
  - "Prototype-augmented"
  - "Mamba"
  - "Transformer Fusion"
date: 2026-05-08
content_hash: cdb06c7819a11867
---

# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**Conference**: CVPR 2025 (ABAW Workshop)  
**arXiv**: [2603.12848](https://arxiv.org/abs/2603.12848)  
**Code**: [GitHub](https://github.com/LEYA-HSE/ABAW10-BAH)  
**Area**: Affective Computing / Multimodal Fusion  
**Keywords**: Ambivalence/Hesitancy Recognition, Multimodal Fusion, Prototype-augmented, Mamba, Transformer Fusion

## TL;DR
This paper proposes a multimodal approach for video-level ambivalence/hesitancy (A/H) recognition, integrating four modalities: scene (VideoMAE), face (EmotionEfficientNetB0), audio (EmotionWav2Vec2.0+Mamba), and text (EmotionDistilRoBERTa). Through a prototype-augmented Transformer fusion model, it achieves an average MF1 of 83.25%, with a five-model ensemble ultimately reaching 71.43% on the test set.

## Background & Motivation
**Background**: Affective computing aims to empower intelligent systems with the capability to perceive human emotions. The 10th ABAW Competition introduces the video-level Ambivalence/Hesitancy (A/H) recognition task—determining whether a video contains ambivalent or hesitant behaviors.

**Limitations of Prior Work**: Unlike basic emotions, A/H is characterized by cross-modal inconsistencies (where verbal content, tone, and facial expressions may conflict), making it difficult to capture via a single modality. Prior works mostly rely on simple fusion strategies and fail to fully model cross-modal interactions.

**Key Challenge**: A/H signals are subtle and context-dependent. They require cooperative multimodal understanding to detect contradictory "actions mismatching words" cues, but cross-modal fusion easily fails when presented with conflicting evidence across modalities.

**Goal**: How to effectively fuse four complementary modalities—scene, face, audio, and text—to recognize A/H behaviors in videos?

**Key Insight**: Independent expert encoders are first trained for each modality to extract compact representations, followed by a Transformer fusion module that models interactions on modal tokens, supplemented by a prototype classification objective to enhance generalization.

**Core Idea**: Quad-modal expert encoders + Transformer cross-modal fusion + prototype-augmented classification + multi-seed ensemble.

## Method

### Overall Architecture
A four-stage pipeline: (1) independent encoders for each modality extract embeddings; (2) projection into a shared latent space; (3) a Transformer encoder fuses modality tokens; (4) a classification head + an optional prototype head predict A/H.

### Key Designs

1. **Scene Modality (VideoMAE)**

    - **Function**: Capture behavioral dynamics and uncertainty cues in videos.
    - **Mechanism**: Uniform sampling of 16 frames $\rightarrow$ tubelet embedding $\rightarrow$ Transformer encoder $\rightarrow$ global average pooling to obtain the scene embedding $h_s$.
    - **Design Motivation**: Pre-trained on Kinetics-400, VideoMAE can capture spatiotemporal dependencies, which is suitable for analyzing behavioral patterns.

2. **Facial Modality (EmotionEfficientNetB0 + Statistical Pooling)**

    - **Function**: Extract frame-level facial emotion embeddings and aggregate them into video-level representations.
    - **Mechanism**: YOLO face detection $\rightarrow$ EfficientNetB0 (AffectNet + fine-tuning) to extract frame-level emotional embeddings $\rightarrow$ statistical pooling $[\mu; \sigma]$ to obtain video-level representations.
    - **Design Motivation**: The mean captures the dominant emotional state, while the standard deviation captures emotional fluctuations—instability is indeed a key characteristic of A/H.

3. **Audio Modality (EmotionWav2Vec2.0 + Mamba Encoder)**

    - **Function**: Extract emotional prosody features from speech and model temporal dependencies.
    - **Mechanism**: EmotionWav2Vec2.0 extracts acoustic embedding sequences $\rightarrow$ Mamba encoder models temporal dependencies $\rightarrow$ mean pooling $\rightarrow$ linear layer.
    - **Design Motivation**: The linear complexity of Mamba is suitable for handling variable-length audio sequences, and the state-space model can capture temporal patterns such as hesitation and pauses in speech.

4. **Text Modality (EmotionDistilRoBERTa Fine-tuning)**

    - **Function**: Extract linguistic hesitation cues from transcribed text.
    - **Mechanism**: Direct fine-tuning of EmotionDistilRoBERTa $\rightarrow$ MLP classification head.
    - **Design Motivation**: Text is the strongest single-modal cue (70.02% MF1), as hesitation and ambivalence are frequently expressed through wording.

5. **Prototype-augmented Transformer Fusion Model**

    - **Function**: Fuse the four modal embeddings and enhance generalization through prototype classification.
    - **Mechanism**: Each modal embedding is projected onto a shared space $u_m = \phi_m(x_m)$, added with modal embeddings, and input into a 6-layer Transformer $\rightarrow$ masked mean pooling $\rightarrow$ main classification head + prototype classification head.
    - **Prototype Head**: 16 learnable prototypes per class, calculating class scores using $\ell_2$ normalized cosine similarity + log-sum-exp.
    - **Design Motivation**: The prototype auxiliary loss provides smoother gradient signals during training, preventing overfitting to hard labels.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$, where $\lambda_{\text{proto}}=0.2$ and $\lambda_{\text{div}}=0$ (diversity regularization term is disabled). The RMSprop optimizer is used, and training is averaged over 5 fixed seeds to reduce sensitivity to initialization.

## Key Experimental Results

### Main Results (BAH Corpus)

| Model Configuration | Modality | Average MF1 | Final Test MF1 |
|---------|------|---------|-------------|
| EmotionDistilRoBERTa | Text | 70.02% | — |
| EmotionWav2Vec2.0+Mamba | Audio | 69.03% | — |
| Four-modal Fusion | All | 82.66% | 68.32% |
| Four-modal Fusion + Prototype | All | 83.25% | 65.21% |
| 5-Model Ensemble | All | 81.29% | 70.17% |
| 5-Model Ensemble + Prototype | All | **81.89%** | **71.43%** |

### Ablation Study (Modality Combinations)

| Modality Combination | Average MF1 | Description |
|---------|---------|------|
| Scene + Text | 80.39% | Strongest bi-modal combination |
| Face + Scene + Text | 78.77% | Strongest tri-modal combination |
| Face + Audio | 67.40% | Poor performance without text |
| Full Quad-modal | 82.66% | Optimal quad-modal |

### Key Findings
- Text is the strongest single modality (~70% MF1), followed by audio (69%), while face and scene are individually weaker (~62%).
- Multimodal fusion significantly outperforms all single modalities—the best fusion exceeds the best single modality by 13+ percentage points.
- Prototype-augmentation improves validation performance but degrades individual model test generalization; the benefits of prototype-augmentation are only reflected on the test set after ensembling.
- Scene + Text is the strongest bi-modal combination (80.39%), indicating that behavioral dynamics and linguistic cues are highly complementary.

## Highlights & Insights
- **Text Modality Dominates A/H Recognition**: Verifies the core role of linguistic content in ambivalence/hesitancy detection—people's hesitation is often most directly expressed through wording.
- **Prototype Enhancement + Ensemble Strategy**: Prototype classification as an auxiliary loss provides a regularization effect during training, yet an ensemble is required to compensate for single-model instability.
- **Mamba for Audio Temporal Modeling**: The linear complexity of Mamba is suitable for variable-length audio, being more efficient than Transformer and exhibiting superior performance on the audio modality.
- **Statistical Pooling Captures Emotion Fluctuations**: The facial modality utilizes $[\mu; \sigma]$ rather than just $\mu$. The standard deviation encodes the degree of emotional fluctuation—a key signal in ambivalent behavior.

## Limitations & Future Work
- The BAH corpus is small (1427 videos), which limits the generalization ability of deep models.
- The text modality relies on automatic transcription quality; ASR errors in real-world scenarios may impact performance.
- Inconsistency across modalities is not modeled—while a core characteristic of A/H is the misalignment between "what is said" and facial expressions, the current fusion approach focuses on alignment rather than contrast.
- The scene modality only samples 16 frames, which may miss crucial moments of hesitation.

## Related Work & Insights
- **vs González-González et al. (baseline)**: The baseline uses simple concatenation fusion, whereas this work employs Transformer fusion + prototype-augmentation to better capture cross-modal interactions.
- **vs Savchenko & Savchenko**: Their best results came from a text + face combination, while this work demonstrates that quad-modal fusion and scene information can yield further improvements.
- **vs Hallmen et al.**: Their tri-modal fusion utilizes an MLP, whereas the Transformer fusion module used in this work is more powerful.

## Rating
- Novelty: ⭐⭐⭐ No major innovations at the component level (combinations of existing modules), but the prototype-augmented fusion and the quad-modal design for the A/H task offer some novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Exhaustive ablation experiments cover all modality combinations and model variants.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed experimental descriptions and good reproducibility.
- Value: ⭐⭐⭐ A competition solution paper; the generalizability of the method is limited, but it provides a valuable baseline for A/H recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)
- [\[CVPR 2025\] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing](unipose_a_unified_multimodal_framework_for_human_pose_comprehension_generation_a.md)

</div>

<!-- RELATED:END -->
