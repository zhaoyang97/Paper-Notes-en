---
title: >-
  [Paper Note] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][incomplete multimodal learning] This paper proposes the CyIN framework, which constructs an informative latent space via token-level and label-level information bottlenecks (IB)…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "incomplete multimodal learning"
  - "information bottleneck"
  - "cyclic translation"
  - "variational approximation"
  - "modality missing"
date: 2026-05-08
content_hash: 89bb160b1d19219a
---

# CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning

**Conference**: NeurIPS 2025
**arXiv**: [2602.04920](https://arxiv.org/abs/2602.04920)
**Code**: [GitHub](https://github.com/RH-Lin/CyIN)
**Area**: Multimodal VLM
**Keywords**: incomplete multimodal learning, information bottleneck, cyclic translation, variational approximation, modality missing

## TL;DR

This paper proposes the CyIN framework, which constructs an informative latent space via token-level and label-level information bottlenecks (IB), and employs cyclic cross-modal translation to reconstruct missing modality information, simultaneously optimizing complete and incomplete multimodal learning within a single unified model.

## Background & Motivation

### 1. State of the Field

Multimodal learning has advanced rapidly in integrating linguistic, visual, and audio information. Transformer-based models achieve strong performance on tasks such as sentiment analysis and emotion recognition through multimodal fusion. The vast majority of existing methods assume all modalities are present during both training and inference.

### 2. Limitations of Prior Work

- **Modality missing problem**: In real-world scenarios, sensor failures and incomplete data collection cause random modality absence, leading to severe performance degradation in pretrained multimodal models.
- **Insufficiency of existing methods**:
    - Alignment methods (contrastive learning, CCA): underutilize missing modality information.
    - Generative methods (VAE, diffusion models): tend to introduce task-irrelevant noise.
    - Require training separate models for each missing modality combination, resulting in poor generalization.
- **Complete vs. incomplete trade-off**: Most methods sacrifice complete-modality performance when improving robustness to missing modalities.

### 3. Root Cause

How can a **single unified model** simultaneously maintain performance on complete multimodal inputs while robustly handling diverse dynamic modality-missing scenarios?

### 4. Paper Goals

To construct a unified informative latent space that benefits both complete-modality fusion and missing-modality reconstruction.

### 5. Starting Point

The paper introduces Information Bottleneck (IB) theory into multimodal fusion, performing cyclic information compression at both the token and label levels to filter noise and retain task-relevant features, and conducting missing modality translation within the purified informative space.

### 6. Core Idea

A two-level IB cycle purifies multimodal representations into informative bottleneck latent variables, within which both efficient fusion and accurate translation of missing information can be achieved.

## Method

### Overall Architecture

CyIN comprises three core components:
1. **Token-level Information Bottleneck**: Performs intra-modal and inter-modal cyclic compression at the token embedding level.
2. **Label-level Information Bottleneck**: Injects high-level semantic supervision via label guidance.
3. **Cross-modal Cyclic Translation**: Reconstructs missing modalities via forward and backward propagation.

### Key Designs

#### Module 1: Token-level Information Bottleneck (Token-level IB)

**Function**: Filters redundant noise and extracts task-relevant features at the low-level perceptual stage.

**Mechanism**: The token embeddings of source modality $F_S$ and target modality $F_T$ are compressed into bottleneck latent variables $B_S$ via an IB encoder:

$$\mathcal{L}_{tib}^{S \to T} \approx \frac{1}{L} \sum_i^L \left\{ KL(\mathcal{N}(\mu_B^i, (\sigma_B^i)^2) \| \mathcal{N}(0, \mathbf{I})) + \beta \mathbb{E}_{b_S}[\|f_T - D_T(b_S)\|^2] \right\}$$

The first term compresses information (KL regularization), while the second preserves the ability to reconstruct the target modality.

**Cyclic interaction**:
- **Intra-modal** ($F_S = F_T$): Learns modality-specific features, $\mathcal{L}_{tib}^{S \to S}$
- **Inter-modal** ($F_S \neq F_T$): Learns modality-shared features, $\mathcal{L}_{tib}^{S \to T}$ + $\mathcal{L}_{tib}^{T \to S}$

Overall: $\mathcal{L}_{tib} = \mathbb{E}_{S \cup T}[\mathcal{L}_{tib}^{S \to S} + \frac{1}{2}(\mathcal{L}_{tib}^{S \to T} + \mathcal{L}_{tib}^{T \to S})]$

**Design Motivation**: Variational approximation combined with the reparameterization trick enables differentiable information compression. The cyclic interaction ensures that both intra-modal distinctiveness and inter-modal consistency are preserved.

#### Module 2: Label-level Information Bottleneck (Label-level IB)

**Function**: Injects high-level task semantic supervision signals.

**Mechanism**: Using ground truth labels $y_{gt}$ as targets, IB compression is applied to the representation of each modality:

Regression tasks:
$$\mathcal{L}_{lib} \approx \mathbb{E}_S \left\{ \frac{1}{N} \sum_i^N KL(\mathcal{N}(\mu_B^i, (\sigma_B^i)^2) \| \mathcal{N}(0, \mathbf{I})) + \beta \mathbb{E}_{B_S}[\|y_{gt} - P_S(B_S)\|] \right\}$$

Classification tasks:
$$\mathcal{L}_{lib} \approx \mathbb{E}_S \left\{ \frac{1}{N} \sum_i^N KL(\cdot \| \cdot) - \beta \mathbb{E}_{B_S}[\sum^V y_{gt} \log P_S(B_S)] \right\}$$

**Design Motivation**: While token-level IB focuses on low-level perception, label-level IB ensures that the bottleneck latent variables are strongly correlated with the downstream task.

#### Module 3: Cross-modal Cyclic Translation

**Function**: Reconstructs representations of missing modalities within the informative latent space.

**Forward propagation**: Translation is performed using a Cascaded Residual Autoencoder (CRA):
$$B_{S \to T}^{rec} = \Gamma_{S \to T}(B_S), \quad \mathcal{L}_{rec}^{S \to T} = \|B_T - B_{S \to T}^{rec}\|^2$$

**Backward propagation** (back-translation):
$$B_S^{cyc} = \Gamma_{T \to S}(B_{S \to T}^{rec}), \quad \mathcal{L}_{cyc}^{T \to S} = \|B_S - B_S^{cyc}\|^2$$

**Generalization to multiple missing modalities**: When multiple modalities are retained, the Gaussian mixture property allows direct summation of translator outputs:
$$B_i^{rec} = \sum_{j \neq i}^{|u|} \Gamma_{j \to i}(B_j)$$

This eliminates the need to train dedicated translators for each missing modality combination.

### Loss & Training

Total loss:
$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{1}{\beta}(\mathcal{L}_{tib} + \mathcal{L}_{lib}) + \gamma \mathcal{L}_{tran}$$

**Two-stage training**:
- Stage 1 ($\gamma = 0$): Constructs a stable informative latent space on complete data.
- Stage 2 ($\gamma > 0$): Introduces translation loss to progressively train the cross-modal translators.

## Key Experimental Results

### Main Results: Complete + Incomplete Learning on 4 Datasets

**MOSI dataset (sentiment regression)**:

| Setting | Model | Acc7↑ | F1↑ | MAE↓ | Corr↑ |
|------|------|------|------|------|------|
| Complete | MMIN | 43.2 | 85.0 | 0.744 | 0.782 |
| Complete | IMDer | 43.8 | 85.7 | 0.724 | 0.796 |
| Complete | **CyIN** | **48.0** | **86.3** | **0.712** | **0.801** |
| Fixed missing (avg) | MMIN | 31.3 | 68.4 | 1.093 | 0.433 |
| Fixed missing (avg) | IMDer | 31.4 | 70.6 | 1.043 | 0.533 |
| Fixed missing (avg) | **CyIN** | **32.8** | **72.2** | **1.037** | **0.599** |
| Random missing (avg) | MMIN | 33.3 | 70.9 | 1.014 | 0.584 |
| Random missing (avg) | **CyIN** | **35.0** | **75.7** | **0.943** | **0.650** |

**IEMOCAP dataset (emotion classification)**:

| Setting | GCNet | IMDer | **CyIN** |
|------|------|------|------|
| Complete Acc/wF1 | 63.0/63.0 | 64.4/64.8 | **66.1/66.0** |
| Fixed missing Acc/wF1 | 52.8/51.9 | 54.7/54.4 | **57.4/56.6** |
| Random missing Acc/wF1 | 55.3/55.3 | 55.8/56.1 | **57.5/57.5** |

**MELD dataset (multiparty emotion recognition)**:

| Setting | IMDer wF1 | LNLN wF1 | **CyIN** wF1 |
|------|------|------|------|
| Complete | 59.7 | 57.1 | **59.8** |
| Fixed missing | 49.8 | 44.7 | **49.4** |
| Random missing | 49.5 | 49.0 | **50.5** |

### Ablation Study

**Ablation of IB components** (inferred from supplementary material):
- Token-level IB improves inter-modal interaction quality.
- Label-level IB injects task supervision and significantly boosts performance in incomplete scenarios.
- Both forward and backward propagation of cyclic translation are indispensable.
- Two-stage training is more stable than end-to-end training.

### Key Findings

1. **Unified optimization of complete and incomplete learning**: CyIN is the only method that simultaneously achieves optimal or near-optimal performance across complete and all incomplete settings.
2. **Superior robustness**: At high missing rates (MR=0.7), CyIN exhibits smaller performance degradation compared to baselines.
3. **Translation quality visualization**: t-SNE analysis shows that latent variables produced by cross-modal translation cluster closely with those of the original modalities.
4. **Single-model generalization**: One model handles all missing combinations without requiring dedicated training per scenario.
5. **Effective on weak modalities**: CyIN can effectively leverage information even from low-information modalities (e.g., visual or audio).

## Highlights & Insights

1. **Deep integration of IB with multimodal learning**: The dual-level IB (token + label) purifies representations from both perceptual and semantic perspectives, filtering noise more effectively than simple contrastive learning.
2. **Elegant cyclic translation design**:
    - Forward translation reconstructs missing information.
    - Back-translation imposes cycle-consistency constraints to improve translation quality.
    - The Gaussian mixture property elegantly handles multi-modality missing scenarios.
3. **Translation in the informative space**: Reconstruction is performed in the purified bottleneck space rather than the raw feature space, reducing difficulty and improving quality.
4. **Two-stage training**: Establishing a stable informative space before training the translator avoids instability in early training from degrading overall quality.
5. **Strong practicality**: Handling incomplete scenarios requires no prior knowledge of which modalities are missing.

## Limitations & Future Work

1. Experiments are limited to three-modality (speech/text/video) sentiment analysis settings; validation on broader modality combinations (e.g., image-text, image-point cloud) is absent.
2. The CRA (Cascaded Residual Autoencoder) translator is relatively simple; stronger generative models (e.g., diffusion models) may serve as more powerful replacements.
3. The sensitivity of hyperparameters $\beta$ and $\gamma$ across different datasets is insufficiently discussed.
4. Token-level IB requires source and target token sequences to be of equal length or aligned, potentially limiting applicability to modalities with large sequence length discrepancies.
5. Comparisons with recent diffusion-model-based incomplete multimodal methods are lacking.
6. Theoretical guidance for optimally selecting the compression ratio of the information bottleneck (controlled by $\beta$) is absent.

## Related Work & Insights

- **Relation to MMIN**: MMIN employs modality-missing simulation with contrastive learning; CyIN provides more principled information control via IB.
- **Relation to GCNet**: GCNet performs graph-based missing reconstruction; CyIN conducts translation in a more compact bottleneck space.
- **Relation to IMDer**: IMDer implicitly models the effect of missing modalities; CyIN explicitly constructs cross-modal translators.
- **Inspiration from IB theory**: IB serves not only as a tool for learning compact representations but also as a mechanism for controlling cross-modal information flow.
- **Inspiration from cycle consistency**: The back-translation concept resembles CycleGAN's approach but is applied in the latent space rather than the pixel space.

## Rating

⭐⭐⭐⭐ (4/5)

The framework design is comprehensive, and the integration of information bottleneck with multimodal fusion and missing modality reconstruction is natural. Experiments span 4 datasets under both complete and diverse incomplete settings, providing thorough validation. The theoretical intuition is clear, though validation at larger scales and with more modality combinations is lacking. Performing cyclic translation within the informative latent space constitutes the core contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)

</div>

<!-- RELATED:END -->
