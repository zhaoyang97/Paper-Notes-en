---
title: >-
  [Paper Note] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
description: >-
  [CVPR 2026][Human Understanding][Ambivalence/Hesitancy recognition] This paper presents a multimodal Ambivalence/Hesitancy (A/H) recognition approach for the 10th ABAW Competition, integrating four modalities—scene, facial, audio, and text—via a Transformer-based fusion module and a prototype-augmented classification strategy. The best single model achieves an MF1 of 83.25%, and a five-model ensemble reaches 71.43% on the final test set.
tags:
  - CVPR 2026
  - Human Understanding
  - Ambivalence/Hesitancy recognition
  - multimodal fusion
  - prototype learning
  - affective computing
  - ABAW competition
date: 2026-05-08
content_hash: b77b254c00f74657
---

# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**Conference**: CVPR 2026  
**arXiv**: [2603.12848](https://arxiv.org/abs/2603.12848)  
**Code**: [LEYA-HSE/ABAW10-BAH](https://github.com/LEYA-HSE/ABAW10-BAH)  
**Area**: Human Understanding  
**Keywords**: Ambivalence/Hesitancy recognition, multimodal fusion, prototype learning, affective computing, ABAW competition

## TL;DR
This paper presents a multimodal Ambivalence/Hesitancy (A/H) recognition approach for the 10th ABAW Competition, integrating four modalities—scene, facial, audio, and text—via a Transformer-based fusion module and a prototype-augmented classification strategy. The best single model achieves an MF1 of 83.25%, and a five-model ensemble reaches 71.43% on the final test set.

## Background & Motivation
Ambivalence/Hesitancy (A/H) recognition is a challenging task in affective computing, closely related to decision uncertainty, resistance, and fluctuating motivation for behavioral change. The core difficulties of A/H include:

**Cross-modal inconsistency**: A/H states often manifest as contradictions across modalities—what a person says, how they say it, and their facial expression may be inconsistent.

**Fine-grained behavioral signals**: Unlike basic emotions (e.g., happiness, surprise), A/H is more subtle and requires comprehensive multimodal modeling.

**Text dominance but insufficiency**: Prior work shows text is the strongest single-modal cue, yet text alone cannot capture the full manifestation of A/H.

**Starting Point**: Building upon prior work that primarily uses facial, audio, and text modalities, this paper **additionally incorporates scene information** and designs a Transformer-based fusion module combined with a **prototype-augmented classification objective**, performing fusion over modality-level embeddings rather than naive concatenation.

## Method

### Overall Architecture
A four-stage pipeline: (1) training dedicated encoders per modality independently; (2) extracting fixed-dimensional modality embeddings; (3) projecting into a shared latent space; (4) a Transformer fusion module modeling cross-modal dependencies to produce the final A/H prediction.

### Key Designs
1. **Scene Model (VideoMAE)**: Employs the VideoMAE architecture (ViT-based, pretrained on Kinetics-400), uniformly sampling 16 frames per video and partitioning them into $2 \times 16 \times 16$ spatiotemporal patches via tubelet embedding. A Transformer encoder models spatiotemporal dependencies, and the scene embedding $h_s = \frac{1}{N}\sum_{i=1}^N z_i$ is obtained via global average pooling. Trained for 15 epochs with LR=2e-5 and label smoothing 0.1.

2. **Facial Model (EmotionEfficientNetB0)**: YOLO face detection → largest-box selection → cropping to 224×224 → EmotionEfficientNetB0 (fine-tuned on AffectNet+) for frame-level emotion embeddings. The key design is **statistical pooling** for aggregation: $\mu = \frac{1}{F}\sum_f e_f$, $\sigma = \sqrt{\frac{1}{F}\sum_f (e_f - \mu)^2}$, with the final video-level facial representation formed by concatenating $[\mu; \sigma]$. This preserves inter-frame variability, which is valuable for capturing emotional fluctuations in A/H.

3. **Audio Model (EmotionWav2Vec2.0 + Mamba)**: Audio resampled to 16kHz → pretrained EmotionWav2Vec2.0 (fine-tuned on MSP-Podcast for emotion) extracts feature sequences of $T_a \times 1024$ → a **Mamba encoder** models temporal dependencies → mean pooling yields a compact embedding. Key choices: layer-10 features + Mamba (outperforms Transformer), hidden size 256, feedforward size 512, Mamba state size 8, convolution kernel 4.

4. **Text Model**: Multiple strategies evaluated—TF-IDF with traditional classifiers (Logistic Regression, CatBoost) and fine-tuned Transformers (EmotionDistilRoBERTa, EmotionTextClassifier). The best configuration is fine-tuned EmotionDistilRoBERTa with an MLP classification head, achieving 70.02% average MF1.

5. **Modality Fusion Model**: Each modality embedding $x_m$ is mapped to a shared space via a modality-specific projector (Linear + LayerNorm + GELU + Dropout): $u_m = \phi_m(x_m)$. These are stacked into matrix $U = [u_1; ...; u_M]$, augmented with learnable modality embeddings $E_{\text{mod}}$, processed by $L=6$ Transformer encoder layers, and finally mean-pooled with masking to obtain the fused representation $z_{\text{fused}}$. Missing modalities are handled via binary modality masks.

6. **Prototype-Augmented Variant**: Maintains $K=16$ learnable prototypes $\{p_{c,k}\}$ per class, computing log-sum-exp similarity between the fused representation and prototypes:
    $$\hat{y}_c^{\text{proto}} = \log \sum_{k=1}^K \exp\left(\frac{\tilde{z}_{\text{fused}}^\top \tilde{p}_{c,k}}{\tau}\right)$$
   The prototype head serves as an **auxiliary training loss** (not used for final prediction). The total loss is: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$, with $\lambda_{\text{proto}}=0.2$.

### Loss & Training
The fusion model uses RMSprop (LR=9.44e-5) with cosine learning rate scheduling, label smoothing 0.02, and gradient clipping at 0.5. Each configuration is trained with 5 fixed random seeds (42, 2025, 7777, 12345, 31415), and the configuration with the highest average MF1 is selected. Final ensemble averages class probabilities across the 5 seed models.

## Key Experimental Results

### Main Results

| Model | Modality | Avg MF1 | Final Test |
|------|------|---------|-----------|
| EmotionEfficientNetB0 | Face | 62.67% | - |
| VideoMAE | Scene | 61.96% | - |
| EmotionWav2Vec2.0+Mamba | Audio | 69.03% | - |
| EmotionDistilRoBERTa | Text | 70.02% | - |
| Four-modal fusion (w/o prototype) | All | 82.66% | 68.32% |
| Four-modal fusion (prototype-augmented) | All | **83.25%** | 65.21% |
| Five-model ensemble (w/o prototype) | All | 81.29% | 70.17% |
| **Five-model ensemble (prototype-augmented)** | **All** | 81.89% | **71.43%** |

### Ablation Study

| Modality Combination | Avg MF1 | Notes |
|---------|---------|------|
| Scene + Text | 80.39% | Strongest two-modal |
| Face + Scene + Text | 78.77% | Strongest three-modal |
| Audio + Text | 69.02% | Limited audio–text complementarity |
| Face + Audio | 67.40% | Visual+audio inferior to text |
| Face + Text | 63.24% | Weaker combination |
| **All four modalities** | **82.66%** | Best overall |

### Key Findings
- Text consistently serves as the strongest single modality (70.02%), yet scene—despite being individually weak (61.96%)—provides the strongest complementary contribution in fusion (Scene+Text=80.39%).
- Prototype augmentation yields a notable improvement on the validation set (83.25% vs. 82.66%), but the single model underperforms on the test set (65.21% vs. 68.32%), indicating overfitting risk.
- **Ensemble is critical for generalization**: the 5-model ensemble raises test set performance from 65–68% to 70–71%.
- Multimodal fusion (82.66%) substantially outperforms the best single modality (70.02%) by 12.64 percentage points.
- The Mamba temporal encoder outperforms Transformers for audio modeling.

## Highlights & Insights
- **Value of the scene modality**: Prior A/H work overlooks scene information; this paper demonstrates that scene is the most important complementary modality.
- **Prototype augmentation as regularization**: The prototype head serves as an auxiliary loss rather than the primary predictor, providing structured regularization while preserving the flexibility of the main classifier.
- **Stability-oriented hyperparameter search**: Each configuration is evaluated across 5 fixed random seeds to reduce selection bias.
- **Missing modality handling**: Binary modality masks allow the fusion model to gracefully handle partial modality absence.

## Limitations & Future Work
- A large gap exists between validation and test set performance (83.25% vs. 65.21%), indicating that generalization requires further improvement.
- Temporal interactions across modalities are not modeled (fusion is performed solely at the video-level embedding stage).
- The BAH corpus is relatively small (1,427 videos), limiting the adequacy of model training.

## Related Work & Insights
- **vs. González-González et al.**: The baseline work validates text superiority; this paper builds upon it by introducing scene information and improving fusion.
- **vs. Savchenko & Savchenko**: Uses a more lightweight feature pipeline; the Transformer fusion in this paper more thoroughly models cross-modal interactions.
- **vs. Hallmen et al.**: A three-modal fusion scheme; this paper adds the scene modality and adopts a more advanced fusion strategy.

## Rating
- Novelty: ⭐⭐⭐ Individual components (VideoMAE, Mamba, prototype learning) are not new; the contribution lies in their systematic combination and the introduction of the scene modality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation covering all modality combinations, with comparisons across multiple encoders and fusion strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed experimental setup descriptions, and high reproducibility.
- Value: ⭐⭐⭐ A competition solution with limited methodological novelty, though the experimental findings offer useful reference value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)

<!-- RELATED:END -->
