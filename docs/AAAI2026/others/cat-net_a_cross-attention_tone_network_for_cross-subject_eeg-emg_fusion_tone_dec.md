---
title: >-
  [Paper Note] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding
description: >-
  [AAAI 2026][Brain-Computer Interface] This paper proposes CAT-Net (Cross-Attention Tone Network), which achieves Mandarin four-tone classification using only 20 EEG channels and 5 EMG channels via spatial-temporal featur…
tags:
  - "AAAI 2026"
  - "Brain-Computer Interface"
  - "EEG-EMG Fusion"
  - "Cross-Attention"
  - "Mandarin Tone Classification"
  - "Cross-Subject Generalization"
date: 2026-05-08
content_hash: ce10004acdf7724f
---

# CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding

**Conference**: AAAI 2026
**arXiv**: [2511.10935](https://arxiv.org/abs/2511.10935)  
**Code**: [github.com/YifanZhuang/CAT-Net](https://github.com/YifanZhuang/CAT-Net)  
**Area**: Other
**Keywords**: Brain-Computer Interface, EEG-EMG Fusion, Cross-Attention, Mandarin Tone Classification, Cross-Subject Generalization

## TL;DR

This paper proposes CAT-Net (Cross-Attention Tone Network), which achieves Mandarin four-tone classification using only 20 EEG channels and 5 EMG channels via spatial-temporal feature extraction branches, a cross-attention fusion mechanism, and domain adversarial training. The model achieves 87.83%/88.08% accuracy under voiced/silent speech conditions and 83.27%/85.10% under cross-subject evaluation, outperforming all 8 baseline methods.

## Background & Motivation

### Demand for BCI Speech Decoding

BCI-based speech decoding is a transformative technology for patients with language disorders caused by stroke, ALS, or brainstem injury. Mandarin Chinese presents a unique challenge for BCI due to its four lexical tones (flat, rising, dipping, falling), each conveying entirely distinct semantics, making tone classification a distinctive problem in this domain.

### Limitations of EEG and the Need for Multimodal Fusion

EEG offers excellent temporal resolution and portability but has limited spatial resolution, resulting in high misclassification rates. In particular, Tone 2 (rising) and Tone 4 (falling) are frequently confused due to similar frequency contours. Prior best results include Li et al. achieving 42.9% using Riemannian manifold features, and Wang et al. achieving 68% with an end-to-end CNN.

### Three Key Problems in Existing EEG-EMG Fusion

**High-density electrode dependency**: Large numbers of EEG channels and multiple EMG sensors are required, making daily use impractical.

**Simplistic fusion strategies**: Most methods rely on concatenation or late fusion, failing to capture complex neural-muscular interactions.

**Poor cross-subject generalization**: Due to individual anatomical differences, electrode impedance variation, and cortical activation pattern differences, model performance degrades significantly on unseen subjects.

### Core Contributions of CAT-Net

1. A cross-attention mechanism enabling bidirectional dynamic interaction between EEG and EMG modalities
2. High-accuracy tone classification using only 20 EEG + 5 EMG channels
3. Domain adversarial training to enhance cross-subject generalization
4. Comprehensive evaluation under both voiced and silent speech conditions

## Method

### Overall Architecture

CAT-Net is a three-stage architecture (Figure 1):
1. **Spatial-Temporal Encoder**: Independently extracts spatial and temporal features from EEG and EMG
2. **Cross-Attention Fusion**: Bidirectional attention for dynamic inter-modal information exchange
3. **Dual-Head Output**: Tone classification head + domain discriminator head (via gradient reversal layer)

### Key Designs

#### 1. **Spatial and Temporal Encoders**

**Function**: Extract spatial and temporal features independently for EEG and EMG.

**Spatial Encoding**: Two layers of $1\times1$ pointwise Conv1D with kernel size $1 \times 2C_{EEG/EMG} \times F$ ($F$=64, 128), learning channel-wise spatial combinations independently at each time step:

$$\mathbf{H}_t = \text{ReLU}(X_t W_{conv}) \in \mathbb{R}^{1 \times F}$$

1D MaxPooling (kernel=2, stride=2) then reduces the sequence length from 499 to 249, preserving key signals such as neural spikes.

**Channel Attention** (inspired by CBAM): GlobalAvgPool and GlobalMaxPool respectively capture channel-scale and peak information; the outputs are passed through fully connected layers with sigmoid normalization, summed, and broadcast along the temporal dimension:

$$\tilde{\mathbf{H}} = s' \odot \mathbf{H} \in \mathbb{R}^{T \times F}$$

**Temporal Encoding**: BiLSTM captures long-range dependencies and bidirectional patterns, yielding $\mathbf{Z} \in \mathbb{R}^{T \times 2F}$.

**Design Motivation**:
- The input includes the raw signal and its first-order temporal difference $\Delta x_t = x_t - x_{t-1}$; the latter enhances rapid transients, improves stationarity, and has been shown to improve EEG/EMG BCI decoding accuracy.
- BiLSTM is chosen over a full Transformer encoder for greater training stability and higher data efficiency.

#### 2. **Cross-Modal Attention Fusion**

**Function**: Enable EEG and EMG to dynamically attend to each other's most informative features.

**Mechanism**: Given $\mathbf{Z}^{EEG}$ and $\mathbf{Z}^{EMG}$, each modality generates its own $Q, K, V$ matrices, which are then cross-queried:

$$\mathbf{C}^{(e)} = \text{MHA}(Q^{(e)}, K^{(m)}, V^{(m)})$$
$$\mathbf{C}^{(m)} = \text{MHA}(Q^{(m)}, K^{(e)}, V^{(e)})$$

That is, EEG queries the Key-Value pairs of EMG using its own Query, and vice versa. Four-head attention is used with $K=V=32$, producing outputs of size $T \times 128$.

After fusion, GlobalAvgPool and GlobalMaxPool outputs are concatenated into a 256-dimensional vector, which is then projected to 128 dimensions via a Dense layer as the final feature representation.

**Design Motivation**: Unlike simple concatenation, cross-attention captures the coordination patterns between neural activity (EEG) and muscular execution (EMG)—specifically the temporal correspondence between motor planning in the brain and facial muscle movements during speech.

#### 3. **Domain Discriminator + GRL**

**Function**: Adversarial training forces the feature extractor to learn subject-invariant representations.

**Mechanism**: A domain discriminator head is appended to the fused feature $\mathbf{f} \in \mathbb{R}^{128}$ via a gradient reversal layer (GRL):

$$\mathcal{R}_\lambda(\mathbf{f}) = \mathbf{f}, \quad \frac{\partial \mathcal{R}_\lambda}{\partial \mathbf{f}} = -\lambda \mathbf{I}$$

The forward pass is unchanged; during backpropagation, gradient signs are reversed—the discriminator learns to predict subject identity while the backbone is forced to learn subject-independent features.

### Loss & Training

The total loss is a weighted sum of three terms:

$$\mathcal{L} = \mathcal{L}_{focal} + 0.05 \cdot \mathcal{L}_{dom} + (0.2, 0.3, 0.2, 0.3) \cdot \mathcal{L}_{cent}$$

- **Focal Loss** ($\gamma=2$, $\alpha=(0.2, 0.3, 0.2, 0.3)$): Assigns higher weight to easily confused tones (Tones 2 and 4)
- **Domain adversarial loss**: Cross-entropy, propagated through GRL
- **Hyperparameters**: Adam, lr=1e-3, batch=64, epochs=50, dropout=0.4, ReduceLROnPlateau, early stopping patience=10

## Key Experimental Results

### Main Results

**Silent speech condition, 5-fold cross-validation (20 EEG + 5 EMG channels)**

| Method | Tone1↑ | Tone2↑ | Tone3↑ | Tone4↑ | Avg↑ | Kappa↑ |
|--------|--------|--------|--------|--------|------|--------|
| ETE-CNN | 71.15 | 69.16 | 46.59 | 48.18 | 57.33 | 0.373 |
| FBCSP+SVM | 87.30 | 41.01 | 39.04 | 37.24 | 51.65 | 0.355 |
| VLAAI | 94.21 | 39.34 | 69.84 | 49.98 | 61.12 | 0.552 |
| EEG-Transformer | 92.25 | 79.86 | 82.63 | 69.94 | 81.10 | 0.748 |
| DRDA | 99.71 | 83.27 | 80.22 | 61.01 | 81.23 | 0.780 |
| GAT | 97.72 | 73.18 | 88.56 | 74.33 | 83.62 | 0.789 |
| EEGNet | 98.10 | 83.15 | 84.75 | 75.62 | 85.29 | 0.804 |
| DeepConvNet | 99.57 | 82.69 | 83.75 | 81.79 | 86.56 | 0.813 |
| **CAT-Net** | **98.67** | **83.64** | **87.27** | **83.10** | **88.08** | **0.842** |

CAT-Net achieves the best average accuracy and Kappa score. A notable advantage is observed on Tone 4 (83.10% vs. the second-best 81.79%), one of the most difficult tones to classify.

**Cross-subject evaluation (Leave-One-Subject-Out, silent speech)**

| Method | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | Avg↑ |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| EEGNet | 25.83 | 79.17 | 75.00 | 78.96 | 79.37 | 81.25 | 92.92 | 82.92 | 95.42 | 87.08 | 77.79 |
| DeepConvNet | 41.67 | 73.75 | 76.46 | 82.50 | 84.58 | 86.88 | 90.42 | 85.83 | 95.63 | 87.08 | 80.48 |
| GAT | 43.75 | 78.75 | 71.04 | 79.79 | 82.29 | 90.42 | 89.58 | 87.29 | 92.08 | 90.42 | 80.54 |
| **CAT-Net** | **53.33** | 79.58 | **82.71** | **85.00** | 87.08 | 91.67 | **95.00** | 89.17 | 94.79 | **92.71** | **85.10** |

CAT-Net's accuracy drops only from 88.08% to 85.10% (−2.98%), compared to drops of −7.5% and −6.08% for EEGNet and DeepConvNet respectively. **Performance on the outlier subject S1 is particularly notable** (53.33% vs. second-best 43.75%, a margin of +9.58%).

### Ablation Study

**5-fold training scenario**

| Configuration | Precision↑ | Recall↑ | F1↑ | Note |
|---------------|-----------|---------|-----|------|
| w/o Cross-Attention | 77.63 | 77.62 | 77.00 | −10.45%; cross-attention is critical |
| w/o BiLSTM | 78.42 | 78.42 | 78.41 | Temporal modeling is important |
| EMG→EEG only | 76.60 | 76.60 | 76.50 | Both directions are necessary |
| EEG→EMG only | 76.46 | 76.46 | 76.35 | Both directions are necessary |
| **Full CAT-Net** | **88.08** | **88.08** | **88.06** | Best |

**Cross-subject scenario — domain discriminator ablation**

| Configuration | S1↑ | Avg↑ | Note |
|---------------|-----|------|------|
| w/o Domain Discriminator | 48.96 | 84.69 | −4.37% on S1 |
| **w/ Domain Discriminator** | **53.33** | **85.10** | Significant improvement on outlier subjects |

**Channel count ablation (silent speech)**

| EEG Channels | Avg Acc↑ | Kappa↑ | Note |
|-------------|---------|--------|------|
| 5 | 86.92 | 0.825 | Still high accuracy |
| 10 | 87.02 | 0.825 | Marginal improvement |
| **20** | **88.08** | **0.842** | Optimal accuracy-channel tradeoff |
| All 64 | 88.45 | 0.850 | Only +0.37% |

The gap between 20 channels and full channels is only 0.37%, validating the feasibility of the minimal-channel configuration.

### Key Findings

1. **Cross-attention is the critical performance factor**: Removing it causes accuracy to drop by over 10%, far exceeding the impact of any other component.
2. **Bidirectional interaction is essential**: Unidirectional attention (EEG→EMG or EMG→EEG) yields similar and substantially lower performance compared to bidirectional attention.
3. **Domain discriminator is most valuable for difficult subjects**: Overall improvement is +0.41%, but +4.37% for outlier subjects such as S1.
4. **20 channels are sufficient**: Channel attention weights indicate that frontal, central, and parietal regions contribute most, consistent with known neurophysiology of tone perception.
5. **Tones 2 and 4 are hardest for all models**: Temporal feature visualizations show that the EEG/EMG patterns for these two tones are highly similar.

## Highlights & Insights

1. **Biologically inspired design**: The cross-attention mechanism models the coordination between the neural system (EEG) and muscular system (EMG) during speech production, which is conceptually and empirically more justified than simple concatenation.
2. **Practical value of the minimal-channel configuration**: The 20 EEG + 5 EMG channel setup substantially lowers deployment barriers and user discomfort for BCI systems, representing an important step toward real-world applicability.
3. **Breakthrough in cross-subject generalization**: The −2.98% accuracy drop is far smaller than baseline methods (−6–8%), and the domain adversarial training shows especially notable improvements for outlier subjects.
4. **Strong performance under silent speech**: The 88.08% accuracy in silent speech conditions is marginally higher than in voiced speech (87.83%), suggesting that EMG signals may provide cleaner muscular information when vocalization is absent.
5. **SHAP-based interpretability**: SHAP values are used to quantify the relative contributions of EEG and EMG features, enhancing model credibility.

## Limitations & Future Work

1. **Limited subject pool**: 10 subjects may be insufficient to represent broader population diversity (age, gender, accent variation, etc.).
2. **Tone-level rather than phoneme/word-level decoding**: Four-tone classification remains far from the demands of practical speech BCIs (i.e., continuous speech decoding).
3. **Laboratory conditions**: Data were collected in controlled environments; robustness in real-world scenarios (noise, motion artifacts) has not been validated.
4. **Single-language validation**: Only Mandarin four-tone classification is evaluated; generalization to other tonal languages (e.g., Cantonese with six tones, Vietnamese with six tones) remains unknown.
5. **Efficiency limitations of BiLSTM**: The inference speed of sequential models may limit real-time BCI applications; more efficient temporal modeling approaches (e.g., Mamba/SSM) could be explored.

## Related Work & Insights

- **EEGNet/DeepConvNet**: The most widely used EEG decoding baselines; CAT-Net builds upon them by introducing multimodal fusion and domain adaptation.
- **Transformer applications in BCI**: EEG-Transformer applies self-attention directly, yet underperforms compared to CAT-Net's cross-attention + BiLSTM combination.
- **Domain adaptation methods**: DRDA, DANN, and related approaches provide the foundation for cross-subject generalization; CAT-Net's GRL design is directly inherited from Ganin et al. (2016).
- **Implications for practical BCI**: The combination of minimal channels, cross-subject generalization, and silent speech support brings BCI speech interfaces closer to everyday deployment scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Cross-attention fusion of EEG-EMG with minimal channel design has practical value)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 baselines + cross-subject evaluation + channel ablation + module ablation + temporal feature visualization; very comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous experimental design, well-executed interpretability analysis)
- Value: ⭐⭐⭐⭐ (Meaningfully advances Mandarin BCI speech decoding; open-sourced code enhances reproducibility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](../../ICCV2025/others/afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)
- [\[AAAI 2026\] Shrinking the Teacher: An Adaptive Teaching Paradigm for Asymmetric EEG-Vision Alignment](shrinking_the_teacher_an_adaptive_teaching_paradigm_for_asymmetric_eeg-vision_al.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/others/heegnet_hyperbolic_embeddings_for_eeg.md)
- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)

</div>

<!-- RELATED:END -->
