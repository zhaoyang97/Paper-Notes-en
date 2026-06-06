---
title: >-
  [Paper Note] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
description: >-
  [CVPR 2026 (ABAW Workshop)][Human Understanding][Ambivalence/Hesitancy Recognition] A four-modality fusion pipeline (scene VideoMAE + face EfficientNetB0 + audio Wav2Vec2.0/Mamba + text EmotionDistilRoBERTa) is proposed.…
tags:
  - "CVPR 2026 (ABAW Workshop)"
  - "Human Understanding"
  - "Ambivalence/Hesitancy Recognition"
  - "Multimodal Fusion"
  - "Prototype-Augmented Classification"
  - "Mamba"
  - "VideoMAE"
  - "ABAW Competition"
date: 2026-05-08
content_hash: 5ef17cb4b843d71e
---

# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**Conference**: CVPR 2026 (ABAW Workshop)  
**arXiv**: [2603.12848](https://arxiv.org/abs/2603.12848)  
**Code**: [LEYA-HSE/ABAW10-BAH](https://github.com/LEYA-HSE/ABAW10-BAH)  
**Area**: Affective Computing / Multimodal Understanding  
**Keywords**: Ambivalence/Hesitancy Recognition, Multimodal Fusion, Prototype-Augmented Classification, Mamba, VideoMAE, ABAW Competition

## TL;DR

A four-modality fusion pipeline (scene VideoMAE + face EfficientNetB0 + audio Wav2Vec2.0/Mamba + text EmotionDistilRoBERTa) is proposed. Each modality embedding is projected into a shared 128-dimensional space via a prototype-augmented Transformer fusion module and regularized with a prototype classification auxiliary loss. A 5-model ensemble achieves **71.43% Macro F1** on the final test set of the BAH corpus, substantially outperforming all unimodal baselines.

## Background & Motivation

- **Task Definition**: The Ambivalence/Hesitancy (A/H) video recognition challenge in the 10th ABAW Competition—given a video clip, determine whether it contains A/H behavior (video-level binary classification).
- **Practical Value**: A/H is closely associated with decision-making uncertainty and motivational fluctuation, and can be used in digital behavioral health interventions to assess whether a user is ready to change behavior or at risk of resistance or disengagement.
- **Core Challenge**: Unlike basic emotions (e.g., happiness, surprise), A/H is extremely subtle and often manifests through **cross-modal inconsistency**—for example, positive verbal content paired with a hesitant facial expression, or confident speech content paired with an unconfident prosody. This makes simple unimodal or shallow fusion approaches insufficient for capturing the key signals.
- **Limitations of Prior Work**: Prior work (González-González et al., Hallmen et al., Savchenko & Savchenko) primarily employed face + audio + text trimodal setups with relatively simple fusion strategies (concatenation / MLP / logistic blending), without incorporating scene modality or strongly regularized fusion.
- **Goal**: Incorporate a scene modality to provide global context, model cross-modal interactions via a Transformer fusion module, and introduce a prototype-augmented classification head as an auxiliary regularizer during training.

## Method

### Overall Architecture

A two-stage strategy is adopted:

1. **Stage 1** — Four dedicated unimodal models (scene / face / audio / text) are trained independently, each mapping a video to a fixed-dimensional embedding vector.
2. **Stage 2** — The four embeddings are projected into a shared 128-dimensional space, cross-modal interactions are modeled via a 6-layer Transformer encoder, and a final video-level A/H binary classification is performed.

### Key Designs

#### 1. Scene Visual Encoder (VideoMAE)

- **Function**: Captures global scene dynamics and behavioral context information in the video.
- **Mechanism**: VideoMAE (ViT-based, pretrained on Kinetics-400) is used as the video-level scene encoder. $T_v=16$ frames are uniformly sampled from each video and resized to $224 \times 224$. Tubelet embedding divides the video into $2 \times 16 \times 16$ spatiotemporal patches, which are projected into a $D=768$-dimensional space with positional encoding. The Transformer encoder performs spatiotemporal self-attention over all tokens, and global average pooling yields the scene embedding $h_s$.
- **Design Motivation**: The scene modality provides environmental context and overall behavioral patterns (e.g., body posture, gesture motion), serving as a complementary signal not exploited in prior A/H work.
- **Training**: 15 epochs, AdamW, lr=2e-5, weight decay=1e-2, batch size=4, cosine annealing, label smoothing=0.1.

#### 2. Face Emotion Encoder (EmotionEfficientNetB0)

- **Function**: Extracts facial micro-expression information as visual cues for A/H states.
- **Mechanism**: A YOLO face detector is applied per frame (the largest bounding box is selected when multiple faces are present; the full frame is used as a fallback when no detection occurs). Cropped faces are resized to $224 \times 224$ and fed into an AffectNet-finetuned EfficientNetB0 to extract per-frame emotion embeddings. **Statistical pooling** is applied over $F$ frame embeddings $\{e_f\}_{f=1}^F$—the mean $\mu$ and standard deviation $\sigma$ are computed and concatenated as $[\mu; \sigma]$, forming a compact video-level face representation that preserves distributional information.
- **Design Motivation**: A/H may manifest as temporal fluctuations in facial expression (e.g., alternation between smiling and frowning); the standard deviation in statistical pooling captures such variability.
- **Hyperparameters**: 30 uniformly sampled frames, 16 hidden states, 256 output features, lr=1e-3, AdamW.

#### 3. Audio Temporal Encoder (EmotionWav2Vec2.0 + Mamba)

- **Function**: Extracts acoustic cues of ambivalence/hesitancy from speech prosody (pitch variation, pauses, speech rate fluctuation, etc.).
- **Mechanism**: Audio is extracted from the video and resampled to 16 kHz, then fed into a Wav2Vec2.0 model finetuned on the MSP-Podcast corpus (output of **layer 10** is taken, dimension $T_a \times 1024$). A **Mamba encoder** then models temporal dependencies, followed by temporal average pooling to obtain the audio embedding $a$. Mamba parameters: state size=8, conv kernel=4, expansion factor=2, hidden=256, FFN=512, dropout=0.1.
- **Design Motivation**: Mamba provides linear-complexity sequence modeling, outperforming Transformers empirically and being particularly suitable for variable-length audio sequences. Layer 10 rather than the final layer is selected because intermediate layers better preserve emotion-relevant prosodic features.
- **Loss**: Standard cross-entropy; a linear layer is applied on top of Mamba output for classification.

#### 4. Text Semantic Encoder (EmotionDistilRoBERTa)

- **Function**: Extracts semantic-level A/H cues from linguistic content (hedging words, self-contradictory statements, etc.).
- **Mechanism**: Automatic speech recognition transcripts provided with the BAH corpus are used. The primary configuration is EmotionDistilRoBERTa (an emotion-pretrained DistilRoBERTa), directly finetuned on the A/H task with an MLP classification head. Alternative configurations include TF-IDF + Logistic Regression/CatBoost (MF1: 68–69%) and EmotionTextClassifier finetuning (70.00%).
- **Design Motivation**: Prior work consistently identifies text as the strongest unimodal cue for A/H recognition. Finetuning an emotion-pretrained model leverages both affective priors and task-specific knowledge.
- **Training**: Partial backbone freezing, MLP head with 1–3 layers (hidden=64–128), dropout 0–0.3, AdamW/SGD, lr 1e-5–0.1, batch=16, 3–20 epochs with early stopping.

#### 5. Transformer Multimodal Fusion Module

- **Function**: Models interactions among the four modality embeddings in a shared space to generate a fused representation for final classification.
- **Mechanism**: Each modality embedding $x_m \in \mathbb{R}^{d_m}$ is mapped to a shared $d=128$-dimensional space via a **modality-specific projector** (Linear + LayerNorm + GELU + Dropout) to obtain $u_m$. Learnable modality embeddings $E_{\text{mod}}$ are added before passing through a 6-layer Transformer encoder (4-head attention, FFN expansion factor 6, dropout=0.45). The output is aggregated via masked mean pooling to obtain the fused representation $z_{\text{fused}}$, which is fed into a linear classifier to produce logits.
- **Missing Modality Handling**: When a modality is unavailable, a binary mask $\mu_m \in \{0,1\}$ is provided to suppress the corresponding token during self-attention, enhancing robustness.
- **Design Motivation**: The Transformer encoder adaptively learns cross-modal attention weights, offering greater flexibility than hand-crafted fusion strategies. Projecting into a shared low-dimensional space (128-dim) reduces parameter count and promotes cross-modal alignment.

#### 6. Prototype-Augmented Classification Head

- **Function**: Provides an auxiliary regularization signal during training via prototype matching, encouraging more compact intra-class structure in the fused representations.
- **Mechanism**: $K=16$ learnable prototypes $p_{c,k}$ per class are maintained. The fused representation $z_{\text{fused}}$ and the prototypes are $\ell_2$-normalized, and cosine similarities are computed (temperature $\tau=0.3$). Class prototype scores $\hat{y}^{\text{proto}}_c$ are aggregated via log-sum-exp.
- **Loss & Training**: Total loss $\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.2 \cdot \mathcal{L}_{\text{proto}} + 0 \cdot \mathcal{L}_{\text{div}}$, where $\mathcal{L}_{\text{cls}}$ is the primary classification cross-entropy and $\mathcal{L}_{\text{proto}}$ is the prototype auxiliary classification loss. The diversity regularizer $\mathcal{L}_{\text{div}}$ is disabled during experiments ($\lambda_{\text{div}}=0$). Only the primary linear classifier head is used at inference.
- **Design Motivation**: Prototype matching acts as an implicit clustering constraint, promoting more compact intra-class and more separated inter-class representations, serving as a regularizer on small datasets.

### Training & Ensemble Strategy

- **Fusion Model Training**: RMSprop, lr=9.44e-5, weight decay=5.55e-4, label smoothing=0.02, gradient clipping=0.5, cosine LR scheduler.
- **Stability Engineering**: Optuna hyperparameter search + training with 5 fixed random seeds (42/2025/7777/12345/31415); each configuration is trained 5 times and selected by mean MF1.
- **Final Ensemble**: Class probabilities from the 5 seed models are averaged as the final prediction.

## Key Experimental Results

### Dataset

The BAH corpus: 1,427 video clips, 300 participants, 10.60 hours total. Collected via online avatar-guided interaction, including video-level and frame-level A/H annotations, face crops, and speech transcriptions. Splits are made by participant into train/valid/public test/private test. Evaluation metric: Macro F1 (MF1).

### Main Results

| ID | Configuration | Modality | Dev MF1(%) | Valid MF1(%) | Avg MF1(%) | Final Test(%) |
|----|--------------|----------|-----------|-------------|------------|--------------|
| 1 | EmotionEfficientNetB0 + statistical features + MLP | Face | 65.29 | 60.05 | 62.67 | — |
| 2 | VideoMAE + Linear | Scene | 61.71 | 62.21 | 61.96 | — |
| 3 | EmotionWav2Vec2.0 + Mamba + Linear | Audio | 67.20 | 70.87 | 69.03 | — |
| 4 | TF-IDF + Logistic Regression | Text | 68.30 | 67.75 | 68.03 | — |
| 5 | TF-IDF + CatBoost | Text | 65.56 | 72.02 | 68.79 | — |
| 6 | EmotionTextClassifier finetune + MLP | Text | 69.28 | 70.72 | 70.00 | — |
| 7 | EmotionDistilRoBERTa finetune + MLP | Text | 68.54 | 71.49 | **70.02** | — |
| 11 | Four-modality fusion (no prototype) | All | 85.38 | 79.94 | 82.66 | 68.32 |
| 12 | Four-modality fusion (prototype-augmented) | All | 83.79 | 82.72 | **83.25** | 65.21 |
| 13 | 5-model ensemble (no prototype) | All | 81.94 | 80.64 | 81.29 | 70.17 |
| **14** | **5-model ensemble (prototype-augmented)** | **All** | **83.00** | **80.77** | **81.89** | **71.43** |

### Ablation Study: Modality Combinations

| ID | Modality Combination | Dev MF1(%) | Valid MF1(%) | Avg MF1(%) |
|----|---------------------|-----------|-------------|------------|
| 15 | Face + Audio | 63.36 | 71.44 | 67.40 |
| 16 | Face + Text | 65.29 | 61.19 | 63.24 |
| 17 | Face + Scene | 78.07 | 77.09 | 77.58 |
| 18 | Audio + Text | 67.05 | 70.99 | 69.02 |
| 19 | Scene + Audio | 77.37 | 77.66 | 77.51 |
| **20** | **Scene + Text** | **81.77** | **79.00** | **80.39** |
| 21 | Scene + Audio + Text | 79.89 | 77.63 | 78.76 |
| 22 | Face + Scene + Text | 79.89 | 77.65 | 78.77 |
| 23 | Face + Scene + Audio | 76.10 | 79.15 | 77.62 |
| 24 | Face + Audio + Text | 68.08 | 70.41 | 69.25 |
| 11 | All four modalities | 85.38 | 79.94 | **82.66** |

### Key Findings

1. **Text is the strongest unimodal modality**: EmotionDistilRoBERTa finetuning achieves 70.02%, followed by audio Mamba (69.03%); face and scene are weaker (~62%).
2. **Scene + Text is the strongest bimodal combination** (80.39%): scene provides behavioral context and text provides semantic content, offering the highest complementarity.
3. **Four-modality fusion substantially outperforms all subsets**: 82.66% vs. best bimodal 80.39% (+2.27%), vs. best trimodal 78.77% (+3.89%).
4. **Prototype augmentation improves dev/valid performance** (83.25% vs. 82.66%), but single-model final test performance actually decreases (65.21% vs. 68.32%), suggesting that the prototype head introduces additional overfitting risk on the validation set.
5. **Ensembling is critical for generalization**: single model 68.32% → 5-model ensemble 71.43% (+3.11%), effectively mitigating initialization sensitivity.
6. **Mamba outperforms Transformer** as the audio temporal encoder (the paper explicitly reports that layer 10 + Mamba is the optimal audio configuration).

## Highlights & Insights

- **Full four-modality coverage**: Compared to prior work using only face/audio/text, this work introduces the scene modality as a critical global context signal; the scene + text combination even surpasses the face + audio + text trimodal setup (80.39% vs. 69.25%).
- **Regularization effect of prototype augmentation**: The prototype head guides fused representations toward cleaner intra-class clusters during training and is discarded at inference, constituting a zero-cost training augmentation technique.
- **Robustness-oriented engineering**: 5-seed training + Optuna hyperparameter search + probability averaging ensemble forms a complete engineering paradigm for stability in competition settings.
- **Missing modality handling**: The fusion module incorporates a built-in binary mask mechanism that gracefully handles cases where certain modalities are unavailable.

## Limitations & Future Work

- **Substantial gap between dev/valid and final test performance** (83.25% → 71.43%) reveals severe overfitting/generalization issues on the small dataset of only 1,427 video clips.
- **No explicit modeling of cross-modal inconsistency**: A core characteristic of A/H is the mismatch between what is said and what is expressed, yet the Transformer fusion performs generic attention aggregation without an explicit mechanism for detecting modal contradiction signals (e.g., contrastive learning or cross-modal discrepancy measures).
- **Face and scene modalities are relatively weak** (~62%), possibly because the pretraining tasks (AffectNet emotion classification, Kinetics action recognition) are insufficiently aligned with the A/H discrimination objective.
- **Prototype augmentation degrades single-model final test performance** (65.21% vs. 68.32%), and relies on ensembling for stability, suggesting the prototype head may introduce additional overfitting on small datasets.

## Related Work & Insights

- **Hallmen et al. (CVPRW 2025)**: Trimodal ViT + LSTM + BERT + MLP fusion. The present work adds a scene modality and substitutes Mamba + prototype augmentation, yielding a more systematic approach.
- **Savchenko & Savchenko (CVPRW 2025)**: Lightweight text + face fusion; best validation performance achieved by combining two modalities. The present work covers four modalities with a more flexible Transformer fusion.
- **González-González et al. (ICLR 2026)**: BAH dataset creators who established multiple baselines. This work builds upon their foundation by introducing VideoMAE scene modeling and prototype augmentation.
- **Future Directions**: ① The essence of A/H is cross-modal "contradiction detection"; future work could explicitly model modal consistency/contradiction via contrastive learning or discrepancy measures. ② Prototype-augmented classification is transferable to other fine-grained emotion recognition tasks with limited data. ③ The success of Mamba in audio temporal modeling provides a lightweight alternative for speech-based tasks.

## Rating

- **Novelty**: ⭐⭐⭐ — A competition technical report; individual components are drawn from existing work, though the introduction of the scene modality and prototype-augmented fusion offer some originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 7 unimodal configurations + 7 fusion/ensemble configurations + 10 bimodal/trimodal ablations provide comprehensive analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, information-rich experimental tables, and complete descriptions of each module.
- **Value**: ⭐⭐⭐ — A competition technical report with reference value for multimodal fusion in affective computing, though generalization issues remain to be addressed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)

</div>

<!-- RELATED:END -->
