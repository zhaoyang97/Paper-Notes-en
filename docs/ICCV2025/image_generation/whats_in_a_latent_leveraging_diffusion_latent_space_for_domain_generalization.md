---
title: >-
  [Paper Note] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization
description: >-
  [ICCV 2025][Image Generation][Domain Generalization] This paper systematically analyzes the domain separation capacity of latent spaces from six pretrained models (CLIP, DiT, SD, MAE, DINOv2…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Domain Generalization"
  - "Diffusion Features"
  - "Pseudo-Domain Discovery"
  - "Latent Space Analysis"
  - "Label-Free Domain"
date: 2026-05-08
content_hash: f6e1c3d5fd22c290
---

# What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization

**Conference**: ICCV 2025
**arXiv**: [2503.06698](https://arxiv.org/abs/2503.06698)  
**Code**: [xthomasbu/GUIDE](https://xthomasbu.github.io/GUIDE)  
**Area**: Domain Generalization / Diffusion Model Representations
**Keywords**: Domain Generalization, Diffusion Features, Pseudo-Domain Discovery, Latent Space Analysis, Label-Free Domain

## TL;DR

This paper systematically analyzes the domain separation capacity of latent spaces from six pretrained models (CLIP, DiT, SD, MAE, DINOv2, ResNet) and demonstrates that diffusion model features are most effective at separating domain information in an unsupervised setting. Building on this insight, the authors propose GUIDE — a framework that leverages diffusion features to discover pseudo-domain representations and augment classifier features — achieving 66.3% average accuracy across five DomainBed datasets without domain labels (surpassing the ERM baseline by +2.6% and +4.3% on TerraIncognita), while outperforming most methods that require domain labels.

## Background & Motivation

### Problem Definition

Domain Generalization (DG) requires a model trained on multiple source domains to maintain strong classification performance on completely unseen target domains. The more challenging setting considered here is one where domain labels are also unavailable at training time (i.e., the domain identity of each sample is unknown).

### Limitations of Prior Work

**ERM baseline is deceptively strong**: Gulrajani & Lopez-Paz have shown that most sophisticated DG methods fail to outperform simple Empirical Risk Minimization (ERM) under rigorous evaluation.

**Methods requiring domain labels (CORAL, SagNet, DANN, etc.)**: These rely on explicit domain labels during training and are inapplicable when labels are unavailable or noisy.

**Data augmentation approaches**: Using text-to-image models for augmentation requires fine-tuning diffusion models, which is computationally expensive and often requires access to test data.

**Existing pseudo-domain methods**: DA-ERM requires training a dedicated domain prototype network; AdaClust operates in a feature space with limited domain separation capacity.

### Core Motivation

**Key insight**: Feature spaces induced by different pretraining objectives differ substantially in their tendency to encode domain information versus class information. Through T-SNE visualization and quantitative NMI analysis, the authors find that:
- **Diffusion models** (DiT, SD-2.1): Because the generative objective is entirely class-agnostic, domain-specific variations (style, texture, environment) naturally emerge in the latent space, yielding the highest domain NMI scores.
- **Discriminative models** (ResNet): Feature spaces are dominated by class-level aggregation, suppressing domain information.
- **Contrastive learning** (CLIP): Focuses on high-level semantic alignment, resulting in low domain and class NMI alike.

This motivates a complementary strategy: use diffusion features to uncover domain structure, then augment the features of a discriminative classifier.

## Method

### Overall Architecture

GUIDE operates in two stages: (1) extract features from a frozen diffusion model and apply K-Means++ clustering to discover pseudo-domains, computing per-cluster centroids as domain representations; (2) map these pseudo-domain representations into the classifier feature space via RBF kernel ridge regression and concatenate them to the classifier input to train a domain-adaptive classifier.

### Key Designs

#### 1. **Unsupervised Pseudo-Domain Discovery**

- **Function**: Extract sample features from a frozen pretrained diffusion model and discover the latent domain structure in the data through unsupervised clustering.
- **Mechanism**: A feature extractor $\Psi$ (the diffusion model) computes a representation for each training sample, after which K-Means++ clustering yields $K$ clusters (pseudo-domains). The centroid $\widehat{\Psi}_k$ of each cluster serves as a compact representation of that pseudo-domain. Each training sample is assigned to its nearest centroid:
  $$\widehat{\Psi}_x = \widehat{\Psi}_k, \quad k = \arg\min_j \|\Psi(x) - \widehat{\Psi}_j\|$$
  The number of clusters $K$ is determined by a simple heuristic: $K = \max(\{1,3,5\} \times n_c, 200)$, where $n_c$ is the number of classes.
- **Design Motivation**: Clustering serves two purposes — (1) smoothing sample-specific noise to produce more stable domain representations, and (2) providing more compact and domain-representative information compared to directly concatenating raw diffusion features. Experiments confirm that clustering yields substantially larger gains (PACS +3.3%) than direct concatenation without clustering (+1.3%).

#### 2. **Feature Space Transformation and Classifier Augmentation**

- **Function**: Map pseudo-domain representations from the $\Psi$ space into the classifier feature space $\Phi$ and concatenate them for training.
- **Mechanism**: A transformation $\mathcal{T}: \Psi \mapsto \Phi$ is implemented via RBF kernel ridge regression to capture nonlinear mappings. Specifically, $\mathcal{T}$ maps the centroid $\widehat{\Psi}_k$ of pseudo-domain $k$ to the mean of $\Phi(x)$ features over all samples assigned to that cluster. The augmented classifier input becomes:
  $$[\Phi(x); \mathcal{T}(\widehat{\Psi}_k)]$$
  A logarithmic update schedule periodically refreshes $\mathcal{T}$ to accommodate changes in $\Phi$ during training. At test time, features are extracted via $\Psi$, assigned to the nearest pseudo-domain cluster, transformed via $\mathcal{T}$, and concatenated.
- **Design Motivation**: The RBF kernel is well-suited for modeling nonlinear distance relationships and has been validated in domain adaptation settings. The logarithmic schedule applies frequent updates early in training and reduces them later, balancing timely adaptation with computational efficiency.

#### 3. **Analysis of Domain Separation Capacity of Diffusion Features**

- **Function**: Systematically analyze the domain separation capacity of six pretrained models across seven datasets using domain NMI versus class NMI.
- **Mechanism**: Normalized Mutual Information (NMI) is used to quantify the alignment between cluster assignments and ground-truth domain/class labels:
  $$\text{NMI}(U,V) = \frac{2 \cdot I(U,V)}{H(U) + H(V)}$$
  An ideal feature space for domain augmentation should exhibit **high domain NMI and low class NMI** — i.e., features cluster by domain rather than by class.

  Key findings:
  - DiT achieves the best performance on datasets with large global style shifts (PACS domain NMI = 0.85, Synth-Artists = 0.89).
  - SD-2.1 is superior on datasets requiring fine-grained spatial features (TerraIncognita domain NMI = 0.55).
  - Diffusion models consistently exhibit low class NMI (DiT on PACS: 0.08), confirming that generative objectives do not encourage class-level aggregation.
- **Design Motivation**: This analysis not only guides the selection of $\Psi$ but also provides a new perspective on the types of information captured by different pretraining paradigms.

### Loss & Training

- Standard cross-entropy classification loss.
- ResNet-50 (pretrained with AugMix) as the classifier $\Phi$.
- DomainBed default settings: batch size 32 per domain, lr = 5e-5, 5001 steps, no dropout, weight decay = 0.
- Leave-one-domain-out cross-validation averaged over 3 seeds.
- Diffusion feature extraction: DiT uses block 14 at $t = 50$; SD-2.1 uses the up_ft:1 layer at $t = 50$.

## Key Experimental Results

### Main Results

**DomainBed five-dataset generalization performance (test accuracy %)**:

| Method | Domain Labels | VLCS | PACS | OH | TI | DN | Avg |
|--------|--------------|------|------|------|------|------|------|
| ERM | ✗ | 76.6 | 83.8 | 67.2 | 47.0 | 44.1 | 63.7 |
| CORAL | ✓ | 78.8 | 86.2 | 68.7 | 47.6 | 41.5 | 64.5 |
| SagNet | ✓ | 77.8 | 86.3 | 68.1 | 48.6 | 40.3 | 64.2 |
| MIRO | ✗ | 79.0 | 85.4 | 70.5 | 50.4 | 44.3 | 65.9 |
| AdaClust | ✗ | 78.9 | 87.0 | 67.7 | 48.1 | 43.6 | 64.9 |
| **GUIDE-BEST** | **✗** | **78.5** | **87.1** | **68.6** | **51.3** | **45.9** | **66.3** |

### Ablation Study

**Effect of different $\Psi$ on domain generalization (test accuracy %)**:

| $\Psi$ Features | VLCS | PACS | OH | TI | Avg |
|----------------|------|------|------|------|------|
| DiT | 78.5 | **87.1** | 68.4 | 48.2 | 70.6 |
| SD-2.1 | 77.0 | 86.9 | **68.6** | **51.3** | **71.0** |
| CLIP | 76.8 | 84.7 | 64.6 | 47.4 | 68.4 |
| DINOv2 | 77.3 | 84.9 | 68.3 | 48.4 | 69.7 |
| MAE | 76.4 | 84.6 | 65.2 | 50.2 | 69.1 |
| ERM (no augmentation) | 76.6 | 83.8 | 67.2 | 47.0 | 68.7 |

**Effect of enhanced training strategies (PACS / TerraIncognita)**:

| Method | PACS | TI |
|--------|------|------|
| ERM | 83.8 | 47.0 |
| ERM++ | 88.0 | 50.7 |
| GUIDE + ERM++ | **89.2** | **53.6** |

### Key Findings

1. **Diffusion features are consistently superior**: DiT and SD-2.1 outperform all non-diffusion features across all datasets and offer complementary strengths.
2. **DiT vs. SD-2.1 complementarity**: DiT excels on datasets with large global style shifts (PACS +3.3%), while SD-2.1 excels on datasets with fine-grained spatial variation (TI +4.3%), consistent with their architectural characteristics.
3. **CLIP features are nearly ineffective**: CLIP features exhibit low domain NMI and low class NMI, yielding negligible gains when used for domain augmentation.
4. **Clustering is essential**: Direct concatenation without clustering yields only +1.3% improvement, whereas clustering achieves +3.3% on PACS, demonstrating the importance of noise smoothing.
5. **GUIDE is orthogonal to training optimizations**: GUIDE can be combined with SWAD, MIRO, ERM++, and other strategies for further gains.
6. **Label-free method surpasses label-dependent methods**: GUIDE-BEST achieves 66.3% average accuracy, exceeding CORAL (64.5%) and SagNet (64.2%) without requiring any domain labels.

## Highlights & Insights

1. **Insightful feature space analysis**: The paper systematically compares six pretraining paradigms along the domain separation vs. class separation axes, revealing that diffusion models possess a uniquely class-agnostic yet domain-rich structure.
2. **Extreme simplicity of the method**: The entire framework reduces to clustering and concatenation, with no complex losses, adversarial training, or meta-learning — demonstrating that correctly exploiting complementary features is more impactful than designing sophisticated algorithms.
3. **First use of frozen diffusion features for domain generalization**: Prior work either uses diffusion models for data augmentation (requiring fine-tuning) or leverages text-conditioned generation; GUIDE exploits diffusion features in a completely training-free manner.
4. **Construction of synthetic datasets**: The Synth-Artists and Synth-Photography datasets generated with SDXL provide controlled domain-shift benchmarks for future research.

## Limitations & Future Work

1. **Classifier backbone limited to ResNet-50**: Stronger backbones (e.g., ViT-L) may alter the magnitude of gains from diffusion feature augmentation.
2. **Inference cost of diffusion feature extraction**: Each sample requires a full forward pass through the diffusion model, adding preprocessing overhead compared to end-to-end training.
3. **$K$ selection relies on a simple heuristic**: More principled cluster count selection (e.g., via information-theoretic criteria) could further improve performance.
4. **Single-layer diffusion features only**: Combining features from multiple layers and/or multiple timesteps may provide richer domain information.
5. **Limited gains on OfficeHome**: Low domain NMI scores (0.25–0.28) on this dataset indicate that the approach weakens when domain boundaries are ambiguous.

## Related Work & Insights

- **vs. DA-ERM**: DA-ERM trains a dedicated domain prototype network and requires domain labels; GUIDE uses frozen pretrained diffusion features without domain labels.
- **vs. AdaClust**: AdaClust clusters using the classifier's own early convolutional layers; GUIDE uses a complementary external diffusion feature space.
- **Broader insight**: Features from different pretraining paradigms can be viewed as projections of images onto different axes — discriminative models project onto the class axis, while generative models project onto the style/domain axis. Intelligently combining complementary projections enhances robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First in-depth analysis of the domain separation properties of diffusion features and their application to domain generalization; insightful findings, though the resulting method is relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 7 datasets, 6 feature extractors, dual-axis analysis of domain NMI and class NMI, and ablations across multiple training strategies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Analysis is exceptionally thorough, with clear correspondence between the domain shift characteristics of each dataset and the strengths and weaknesses of each feature space.
- **Value**: ⭐⭐⭐⭐ — The method is simple and effective, but the greater contribution lies in the deeper understanding of pretrained feature spaces it provides.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[NeurIPS 2025\] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](../../NeurIPS2025/image_generation/vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [\[ICCV 2025\] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation](multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)
- [\[NeurIPS 2025\] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](../../NeurIPS2025/image_generation/pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)

</div>

<!-- RELATED:END -->
