---
title: >-
  [Paper Note] How Video Meetings Change Your Expression
description: >-
  [ECCV 2024][Human Understanding][facial expression] Proposes FacET (Facial Explanations through Translations), an interpretable framework based on generative domain translation. By learning disentangled facial spatial features and interpretable spatiotemporal linear transformations, it automatically discovers subtle facial expression variation patterns between video conferencing (VC) and face-to-face (F2F) communication, while supporting "de-zooming" to translate VC videos in…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "facial expression"
  - "video conferencing"
  - "interpretability"
  - "generative domain translation"
  - "_beta_-VAE"
date: 2026-05-08
content_hash: 008324d31c2c3021
---

# How Video Meetings Change Your Expression

**Conference**: ECCV 2024  
**arXiv**: [2406.00955](https://arxiv.org/abs/2406.00955)  
**Code**: [https://facet.cs.columbia.edu](https://facet.cs.columbia.edu)  
**Area**: Human Understanding  
**Keywords**: facial expression, video conferencing, interpretability, generative domain translation, _beta_-VAE

## TL;DR

Proposes FacET (Facial Explanations through Translations), an interpretable framework based on generative domain translation. By learning disentangled facial spatial features and interpretable spatiotemporal linear transformations, it automatically discovers subtle facial expression variation patterns between video conferencing (VC) and face-to-face (F2F) communication, while supporting "de-zooming" to translate VC videos into F2F styles.

## Background & Motivation

**Research Question**: Does video conferencing (VC) alter our facial expressions? If so, what are the specific spatiotemporal variation patterns? Addressing this question is crucial for understanding the impact of VC on human behavior, improving AR/VR technologies, and cognitive psychology research. With VC becoming a primary communication mode post-COVID-19, studying phenomena like "Zoom fatigue" holds significant societal value.

**Why discriminative methods are insufficient?** Two core challenges exist:
   - **Dataset Bias**: VC videos online usually face the camera directly, while F2F videos have profile views. A simple linear classifier achieves 88% accuracy on disentangled features using only a single frame (without temporal information), primarily relying on obvious bias features such as Head Pitch and Head Tilt.
   - **Fundamental Difference in Task**: The objective is not classification (distinguishing two domains) but **discovering** variation patterns between domains. Discriminative methods only capture the most salient variation patterns (often biases), and post-hoc explainability methods (e.g., GradCAM) are only meaningful when humans are already adept at the task.

**Core Idea**: Employs a **generative domain translation** approach—translating samples from one domain to another—to discover all possible variation patterns. Synthesizing models must learn all differences (not just the most salient ones) to succeed in translation. By constraining the translation function to be an interpretable linear transformation (shift + scale), it is possible to precisely analyze how each dimension changes.

**Key Insight**: First utilizes a $\beta$-VAE to learn disentangled facial spatial features (12 dimensions, each corresponding to an interpretable facial attribute), and then learns an input-dependent piecewise linear transformation (shift-and-scale) within this feature space to achieve interpretable domain translation.

## Method

### Overall Architecture

Input: Facial landmark sequences (two domains $X$ and $Y$, unpaired) $\rightarrow$ $\beta$-VAE encoder extracts per-frame disentangled features $z \in \mathbb{R}^l$ $\rightarrow$ translation function $G_{XY}$ predicts per-chunk translator parameters $(\omega, \phi)$ $\rightarrow$ applies $f(z) = \omega \odot z + \phi$ to obtain the translated features $z'$ $\rightarrow$ $\beta$-VAE decoder reconstructs landmarks. An interpretable difference report is generated via the distribution changes of the disentangled features.

### Key Designs

#### 1. Spatial Feature Disentanglement ($\beta$-VAE) — Establishing an Interpretable Foundational Representation

**Function**: Learns disentangled low-dimensional representations from facial landmarks, where each dimension corresponds to an independent change in facial attributes.

**Mechanism**: Trains a $\beta$-VAE on all frame-level landmarks across domains $X \cup Y$:

$$\mathcal{L}(X \cup Y) = -\mathbb{E}_{q(z|d)}[\log p(d|z)] + \beta D_{KL}(q(z|d) || p(z))$$

The first term optimizes reconstruction quality, while the second enforces disentanglement via KL divergence. A larger $\beta$ enhances disentanglement at the expense of reconstruction. This yields a 12-dimensional latent space, where each dimension corresponds to interpretable attributes such as Head Pitch, Jaw Open, Smile, Eyebrow Raise, Head Steer, Head Tilt, etc.

**Design Motivation**: Directly learning disentangled spatiotemporal features is difficult for humans to interpret, so spatial features are disentangled first, upon which a translation function is applied to capture temporal variations. $\beta$-VAE has been proven to generate interpretable disentangled representations in various domains.

#### 2. Interpretable Translation Function (Core of FacET) — Discovering Cross-Domain Variations

**Function**: Learns a translation function $G_{XY}$ that maps domain $X$ samples to domain $Y$ while maintaining interpretability.

**Mechanism**: $G_{XY}$ does not directly predict translation results but rather **predicts a translator function** $f$, which then operates on the input. The translator is parameterized as a simple shift-and-scale operation:

$$f(z) = \omega \odot z + \phi, \quad \omega, \phi \in \mathbb{R}^l$$

Optimized via adversarial training:

$$G_{XY}^* = \arg\min_{G_{XY}} \max_D \mathcal{L}_{adv}(X, Y)$$

**Temporal Partitioning Design**: Expressions vary within a clip, so $G_{XY}$ is split into two sub-modules:
- $G_t$: predicts $c-1$ temporal transition points $\{\tau_1, \ldots, \tau_{c-1}\}$, segmenting the clip into $c$ chunks
- $G_f$: independently predicts translator parameters $(\omega_k, \phi_k)$ for each chunk

Temporal partitioning is approximated using a continuously differentiable sigmoid:

$$w_k = \begin{cases} \sigma(\tau_k - T, Q) & k=1 \\ \min(\sigma(T - \tau_{k-1}, Q), \sigma(\tau_k - T, Q)) & k \in [2, c-1] \\ \sigma(T - \tau_{k-1}, Q) & k=c \end{cases}$$

where $T$ is the temporal index vector, $Q$ is the temperature parameter, and $\bar{w}_k = w_k / \sum_k w_k$ is the normalized weight.

**Design Motivation**:
- The shift-and-scale constraint allows the translation to decompose into independent changes in each dimension, enabling direct analysis of how each disentangled dimension varies.
- Predicting the translator rather than directly predicting the translation results allows the model to learn consistent translation parameters for similar expressions (e.g., "smiling"), which can then be clustered and analyzed.
- Temporal partitioning enables the model to capture expression transitions within a clip, while learning similar translators for similar segments.

#### 3. Interpretable Report Generation — Extracting Insights from the Model

**Function**: Performs clustering analysis on the trained translator parameters to generate detailed reports on the differences between the two domains.

**Mechanism**:
1. Performs k-means clustering on translator parameters $(\omega, \tau)$ across all chunks to obtain semantically consistent expression clusters (e.g., "speaking with smile", "listening", "eyebrow raising").
2. For each cluster, compares the distribution changes in each disentangled dimension before and after translation.
3. Analyzes temporal expression transition patterns through the cluster transition matrix of adjacent chunks.

### Loss & Training

- **$\beta$-VAE**: Standard variational objective, where $\beta$ controls the disentanglement-reconstruction trade-off.
- **Translation Function**: GAN adversarial loss, alternately training $G_{XY}$ and the discriminator $D_Y$.
- No cycle consistency loss required—the shift-and-scale parameterization itself imposes sufficient constraints on the model, preventing degenerate solutions (such as memorizing one-to-one mappings).

## Key Experimental Results

### Main Results

**Translation Quality Evaluation (Discriminator Accuracy ↓, 50% is optimal)**

| Model | $G_f$ Type | $G_t$ Type | chunks | ZoomIn Avg | Presidents Avg |
|------|-----------|-----------|--------|------------|----------------|
| Fixed translator set | No Partitions | 1 | 87.58 | 81.40 |
| Fixed translator set | Var. Chunks | 7 | 97.67 | 97.17 |
| Predicted translator | No-partitions | 1 | 78.54 | 79.25 |
| Predicted translator | Fixed-size | 7 | 81.84 | 89.86 |
| **FacET** | **Var. Chunks** | **2** | **73.28** | **79.35** |
| **FacET** | **Var. Chunks** | **7** | **73.16** | **78.14** |

FacET (variable chunks + predicted translator) significantly outperforms all ablation variants. The fixed translator set approach is overly constrained (accuracy > 87%), and failing to partition the clip also yields suboptimal performance (78%).

### Ablation Study

**Impact of Key Designs**

| Ablation Configuration | ZoomIn Discriminator Accuracy ↓ | Description |
|----------|---------------------|------|
| No partitioning, fixed translator set | 87.58% | Most constrained, poor translation quality |
| No partitioning, predicted translator | 78.54% | Predicted translator outperforms fixed set |
| Equal-sized chunks (c=2) | 82.99% | Fixed partitioning, fails to adapt to expression changes |
| Equal-sized chunks (c=7) | 81.84% | More chunks help slightly but remain constrained |
| **FacET (c=2)** | **73.28%** | Optimal, variable partitioning + predicted translator |
| FacET (c=7) | 73.16% | Diminishing returns with more chunks |

Key finding: Increasing the number of chunks from 2 to 7 yields almost no gain in conversation data, as expressions in 7-second conversational clips typically do not transition more than once.

### Key Findings

1. **People laugh smaller in VC**: In the "smiling while speaking" cluster, the distribution of the laugh dimension shifts significantly towards smaller smiles. This is a distinct finding from "less laughter in VC"—since VC systems typically allow only one person to speak at a time, hearty laughter is more difficult than subtle expressions.
2. **People emote more in VC**: Eyebrow raise (#11) is more prominent in VC. It is hypothesized that because subtle reactions are less visible in VC, people unconsciously amplify their expressions.
3. **Head steer/tilt exhibit a bimodal distribution in F2F**: Head orientation is fixed in VC (looking at the screen), whereas in F2F, head orientation switches between two directions during two-person conversations.
4. **Trump speaks with a rounder mouth and listens with higher eyebrows**: The model successfully discovers subtle patterns of individual speech style variations under different settings.
5. **"De-zooming" application**: Translates VC videos into videos that look like F2F interactions, including micro-adjustments to eye blinks and smiles, making virtual conversations more natural.

## Highlights & Insights

1. **Core argument of Generative vs. Discriminative models**: Discriminative models focus only on the most salient variation patterns (often biases), whereas generative models must learn all differences to succeed in translation. This is a profound methodological insight.
2. **Interpretability by design**: Rather than providing post-hoc explanations of a black box, interpretability is guaranteed by architectural constraints (shift-and-scale) itself. Each translator parameter directly maps to changes in a disentangled feature dimension.
3. **Unsupervised temporal partitioning**: $G_t$ learns semantically consistent expression segmentation without any temporal annotations, which is a valuable byproduct.
4. **Novel research perspective**: Applying computer vision techniques to social science questions (changes in communication styles in the COVID era) offers significant interdisciplinary value.

## Limitations & Future Work

1. Relying on the quality of $\beta$-VAE disentanglement, which remains a hard problem without prior information.
2. Currently based only on facial landmarks; extending the method to raw image/video pixels requires non-trivial architectural modifications.
3. Data is sourced from public YouTube videos, where VC and F2F videos may exhibit systematic differences other than the communication medium (e.g., recording environments, participant demographics).
4. Linear shift-and-scale transformations may fail to capture complex, non-linear expression variations.

## Related Work & Insights

- **$\beta$-VAE [Higgins et al.]**: A classical method for disentangled representation learning, which serves as the foundation for the spatial features in FacET.
- **CycleGAN [Zhu et al.]**: A classical approach to adversarial domain translation; FacET adopts a similar adversarial objective but does not require cycle consistency.
- **Interpretable linear transformations [Rudin et al., series]**: Methodological foundation for interpretability-by-design.
- **Zoom Fatigue studies [Bailenson et al.]**: Research on the impact of VC in social psychology, for which FacET provides a computational approach to quantify these effects.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unique and socially meaningful problem formulation; the methodology of generative interpretable domain translation is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative evaluation (GAN discriminator accuracy), rich qualitative analysis and findings, validation on two datasets (ZoomIn + Presidents), but lacks comparison with more baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Compelling motivation, highly informative figures/tables, and clearly presented insights.
- **Value**: ⭐⭐⭐⭐ — High methodological value (generative interpretable analysis), interesting applications (de-zooming, behavior analysis), and strong interdisciplinary impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects](../../ICML2025/human_understanding/how_to_move_your_dragon_text-to-motion_synthesis_for_large-vocabulary_objects.md)
- [\[ECCV 2024\] Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)
- [\[ECCV 2024\] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)
- [\[CVPR 2026\] How to Take a Memorable Picture? Empowering Users with Actionable Feedback](../../CVPR2026/human_understanding/how_to_take_a_memorable_picture_empowering_users_with_actionable_feedback.md)
- [\[CVPR 2026\] Omni-Supervised Motion Editing: Balancing Change and Invariance through Positive-Negative Learning](../../CVPR2026/human_understanding/omni-supervised_motion_editing_balancing_change_and_invariance_through_positive-.md)

</div>

<!-- RELATED:END -->
