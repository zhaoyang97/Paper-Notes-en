---
title: >-
  [Paper Note] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification
description: >-
  [CVPR 2025 (Workshop)][Human Understanding][Facial Expression Recognition] The HSEmotion team proposed a lightweight pipeline for the ABAW-10 competition: using pre-trained EfficientNet to extract facial embeddings, combined with MLP + GLA (Generalized Logit Adjustment) + sliding window smoothing. It significantly outperformed the official baselines on all four tasks (EXPR/VA/AU/VD), among which the violence detection task achieved a macro F1 of 0.783 using ConvNeXt-T + TCN.
tags:
  - "CVPR 2025 (Workshop)"
  - "Human Understanding"
  - "Facial Expression Recognition"
  - "Valence-Arousal Estimation"
  - "Action Unit Detection"
  - "Violence Detection"
  - "EfficientNet"
  - "MLP"
date: 2026-05-08
content_hash: ba5056a0c26cea5f
---

# HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification

**Conference**: CVPR 2025 (Workshop)  
**arXiv**: [2603.12693](https://arxiv.org/abs/2603.12693)  
**Code**: [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) / [VD](https://github.com/kdtsypliakova/ABAW-10-Violence-Detection)  
**Area**: Facial Expression Recognition / Affective Computing  
**Keywords**: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection, Violence Detection, EfficientNet, MLP

## TL;DR
The HSEmotion team proposed a lightweight pipeline for the ABAW-10 competition: using pre-trained EfficientNet to extract facial embeddings, combined with MLP + GLA (Generalized Logit Adjustment) + sliding window smoothing. It significantly outperformed the official baselines on all four tasks (EXPR/VA/AU/VD), among which the violence detection task achieved a macro F1 of 0.783 using ConvNeXt-T + TCN.

## Background & Motivation
**Background**: The ABAW (Affective Behavior Analysis in-the-wild) competition is a mainstream benchmark in affective computing. The 10th edition comprises four tasks: Facial Expression Recognition (EXPR), Valence-Arousal Estimation (VA), Action Unit Detection (AU), and Fine-Grained Violence Detection (VD).

**Key Challenge**: In-the-wild data faces issues such as occlusion, head pose/illumination variations, domain shift, label noise, and class imbalance. Existing methods typically require complex temporal modeling (Transformers/TCNs) and multimodal fusion, resulting in high computational costs.

**Key Insight**: Rather than pursuing architectural complexity, the authors deploy a high-quality pre-trained encoder + simple MLP + post-processing techniques (GLA, confidence filtering, sliding window smoothing) to construct a "simple but effective" pipeline.

**Mechanism**: Pre-trained models already possess powerful feature extraction capabilities; the key lies in how to efficiently utilize these features and address class imbalance and inter-frame noise issues.

## Method

### Facial Expression Recognition (EXPR)
1. **Feature Extraction**: Face embeddings are extracted using EmotiEffNet-B0 (EfficientNet pre-trained on AffectNet).
2. **MLP Classifier**: A single-hidden-layer MLP is trained using weighted softmax loss to handle class imbalance.
3. **GLA (Generalized Logit Adjustment)**: Per-class bias $b_y^*$ is searched on the validation set to maximize the F1 score, effectively correcting class prior bias.
4. **Confidence Filtering**: If the maximum prediction probability of the pre-trained model exceeds $p_0$ ($0.8$-$0.9$), its prediction is directly adopted; otherwise, the MLP is used for classification.
5. **Temporal Smoothing**: A sliding window is applied to average the probabilities of adjacent frames, eliminating frame-level noise.
6. **Optional Audio Fusion**: Features from wav2vec 2.0 are extracted and utilized to train a separate MLP, which is then fused with the visual branch via weighted summation.

### Valence-Arousal Estimation (VA)
- A pre-trained MT-DDAMFN model is used to extract embeddings, followed by a zero-hidden-layer MLP for regression.
- The loss function combines MSE and Concordance Correlation Coefficient (CCC).
- Sliding window smoothing is similarly applied.

### Action Unit Detection (AU)
- Multi-label classification for 12 AUs is performed using an MLP with a sigmoid output.
- Weighted BCEWithLogitsLoss is employed, where positive weights are computed based on class frequencies.
- Innovation: Blending predictions from two separate MLPs trained on embeddings and logits, respectively.
- Per-AU optimal thresholds are searched instead of using a uniform threshold of 0.5.

### Violence Detection (VD)
- Best single-stream model: ConvNeXt-T (pre-trained on ImageNet-1K) extracts 768-d frame features, followed by a 5-layer dilated TCN.
- Multimodal variant: MediaPipe Pose skeletal features (406-d compressed to 256-d) are incorporated, followed by cross-attention fusion and a BiLSTM.
- Training is conducted using AdamW + OneCycleLR + TrivialAugmentWide, with a positive class weight of 1.15.

## Key Experimental Results

### EXPR Classification (AffWild2 Validation Set)

| Method | F1-score | Accuracy |
|------|----------|----------|
| Baseline VGGFACE | 25.0 | - |
| EmotiEffNet, GLA, sliding window | 44.85 | 55.41 |
| EmotiEffNet, GLA, filtering + sliding window | 45.79 | 55.69 |
| EmotiEffNet + wav2vec, GLA, filtering + sliding window | **47.40** | **57.98** |
| Comparison: CLIP+TCN [68] | 46.51 | - |

### VA Estimation (AffWild2 Validation Set)

| Method | CCC_V | CCC_A | $P_{VA}$ |
|------|-------|-------|----------|
| Baseline ResNet-50 | 0.24 | 0.20 | 0.22 |
| MT-DDAMFN, MLP, sliding window | 0.510 | 0.615 | 0.562 |
| Comparison: CLIP+TCN [68] | 0.562 | 0.612 | 0.587 |

### AU Detection (AffWild2 Validation Set)

| Method | F1-score |
|------|----------|
| Baseline VGGFACE | 39.0 |
| EmotiEffNet, logits+embeddings, sliding window, optimal threshold | **54.7** |
| Comparison: CLIP+TCN [68] | 58.0 |

### Violence Detection (DVD Validation Set)

| Method | F1_V | F1_NV | Macro F1 |
|------|------|-------|----------|
| Baseline ResNet-50 + BiLSTM | 0.56 | 0.71 | 0.640 |
| ConvNeXt-T + TCN | 0.738 | 0.828 | **0.783** |
| ConvNeXt-T + Skel. attn + BiLSTM | 0.715 | 0.828 | 0.772 |

### Key Findings
- A 2D pre-trained encoder + simple temporal head consistently outperforms 3D video backbones (e.g., SlowFast, VideoMAE).
- Two-stream optical flow fusion performs worse than pure RGB ConvNeXt-T.
- GLA yields significant improvements in calibrating class imbalance (F1 increases from 38.68 to 41.40).
- Confidence filtering and sliding window smoothing each contribute roughly a 1-2% boost in F1.
- Per-AU threshold searching consistently yields a 0.2-0.5% improvement compared to using a uniform 0.5 threshold.

## Highlights & Insights
- **Extreme Engineering Simplicity**: The entire EXPR pipeline consists only of a pre-trained encoder + single-layer MLP + three post-processing techniques, yet achieves near-SOTA performance.
- **Effective Application of GLA**: Migrating post-hoc logit adjustment from general classification to affective recognition is straightforward and highly effective.
- **Intuitive Confidence Filtering**: Pre-trained models already exhibit accurate judgments on high-confidence samples, reserving the need for an extra classifier specifically for low-confidence ones.
- **Systematic Ablation for the VD Task**: Extensive combinations of backbones, temporal heads, and multimodal options are tested, yielding clear conclusions.

## Limitations & Future Work
- Methodology innovation is relatively limited, primarily representing an engineering assembly of existing techniques (EfficientNet + MLP + GLA + smoothing).
- Gaps still exist between their solutions and previous top works in EXPR/VA/AU tasks (specifically, AU detection lags behind CLIP+TCN by approximately 3.3%).
- VA estimation utilizes only single frames + simple smoothing, failing to fully leverage temporal dependencies.
- The integration of the audio modality is relatively primitive (simple weighted fusion), without exploring more sophisticated fusion mechanisms like cross-attention.
- Violence detection is evaluated only on the DVD dataset; its generalizability remains to be verified.

## Rating
- Novelty: ⭐⭐⭐ Limited methodological innovation; mainly a combination of mature techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detailed ablations across all four tasks, with extensive architectural comparisons for VD.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though tending towards a technical report style.
- Value: ⭐⭐⭐⭐ Holds practical engineering reference value as a competition solution, demonstrating the upper bound of "simple methods."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ECCV 2024\] Generalizable Facial Expression Recognition](../../ECCV2024/human_understanding/generalizable_facial_expression_recognition.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](../../NeurIPS2025/human_understanding/part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)
- [\[CVPR 2026\] Dynamic Label Noise Suppression with Optimal Teacher Pool for Facial Expression Recognition](../../CVPR2026/human_understanding/dynamic_label_noise_suppression_with_optimal_teacher_pool_for_facial_expression_.md)

</div>

<!-- RELATED:END -->
