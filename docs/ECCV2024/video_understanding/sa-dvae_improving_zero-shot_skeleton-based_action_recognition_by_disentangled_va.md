---
title: >-
  [Paper Note] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders
description: >-
  [ECCV 2024][Video Understanding][Skeleton-Based Action Recognition] SA-DVAE introduces feature disentanglement to zero-shot skeleton-based action recognition for the first time. Using a dual-head VAE, it splits skeleton features into a semantic-related branch and a semantic-unrelated branch, aligning only the semantic-related part with text. Coupled with an adversarial total correlation penalty to enhance disentanglement, it achieves SOTA performance on NTU RGB+D 60/120 and P…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Skeleton-Based Action Recognition"
  - "Zero-Shot Learning"
  - "Feature Disentanglement"
  - "Variational Autoencoder"
  - "Cross-Modal Alignment"
date: 2026-05-08
content_hash: 7d3420e97ffe4ce9
---

# SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders

**Conference**: ECCV 2024  
**arXiv**: [2407.13460](https://arxiv.org/abs/2407.13460)  
**Code**: [https://github.com/pha123661/SA-DVAE](https://github.com/pha123661/SA-DVAE)  
**Area**: Video Understanding  
**Keywords**: Skeleton-Based Action Recognition, Zero-Shot Learning, Feature Disentanglement, Variational Autoencoder, Cross-Modal Alignment

## TL;DR

SA-DVAE introduces feature disentanglement to zero-shot skeleton-based action recognition for the first time. Using a dual-head VAE, it splits skeleton features into a semantic-related branch and a semantic-unrelated branch, aligning only the semantic-related part with text. Coupled with an adversarial total correlation penalty to enhance disentanglement, it achieves SOTA performance on NTU RGB+D 60/120 and PKU-MMD benchmarks.

## Background & Motivation

Skeleton-based action recognition has attracted significant attention due to its robustness against appearance and background variations. However, annotating such data is expensive and time-consuming. Zero-Shot Learning (ZSL) provides an alternative by utilizing semantic information (class names, descriptions, etc.) to bridge seen and unseen classes.

**Limitations of Prior Work**: All existing methods (ReViSE, CADA-VAE, SynSE, JPoSE, SMIE) directly align skeleton features with text features in a shared space. However, they overlook a fundamental asymmetry problem:

- **Skeleton sequences within the same category vary significantly**: Different actors possess distinct body shapes and action magnitudes, and varied camera angles introduce prominent differences. For example, the action "wear a shoe" exhibits extremely high variance in skeleton sequences across different subjects.
- **Text labels, conversely, are static**: Each category is represented by only a single label (e.g., "wear a shoe").
- This many-to-one asymmetry renders directly forcing all skeleton features to align with a single textual representation highly difficult.

**Key Challenge**: Skeleton features contain both semantic-related information (what action is performed) and semantic-unrelated style information (who performs it, and from which angle). Directly aligning them inevitably pollutes the shared space with semantic-unrelated noise.

**Key Insight**: Inspired by the SDGZSL method in image-based ZSL—where visual embeddings are decomposed into semantic-consistent and semantic-unrelated parts—this work proposes to apply a similar disentanglement to skeleton features.

**Core Idea**: A dual-head encoder disentangles the latent skeleton representation into a semantic-related part $z_x^r$ and a semantic-unrelated part $z_x^v$. Only $z_x^r$ is aligned with the text feature $z_y$, while an adversarial total correlation penalty is employed to ensure that the two parts are statistically independent.

## Method

### Overall Architecture

The system consists of three main components: (1) two modality-specific feature extractors (Shift-GCN/ST-GCN for skeletons, and Sentence-BERT/CLIP for text); (2) a cross-modal alignment module (dual VAE + feature disentanglement + adversarial discriminator); (3) three classifiers used for ZSL/GZSL inference. During training, the alignment is learned on seen classes, and during inference, the disentangled semantic-related features are utilized to recognize unseen classes.

### Key Designs

1. **Dual-Head Skeleton Encoder and Feature Disentanglement**:

    - **Function**: Encodes skeleton features $f_x$ into two independent latent vectors—semantic-related $z_x^r$ and semantic-unrelated $z_x^v$.
    - **Mechanism**: The skeleton encoder $E_x$ is designed with a dual-head architecture, where one head outputs $z_x^r \sim \mathcal{N}(\mu_x^r, \Sigma_x^r)$ and the other outputs $z_x^v \sim \mathcal{N}(\mu_x^v, \Sigma_x^v)$. The complete skeleton latent representation is the concatenation $z_x = z_x^v \oplus z_x^r$. The text encoder $E_y$ outputs a single-head feature $z_y$. The loss functions for the two VAEs are formulated as:
    $\mathcal{L}_x = \mathbb{E}[\log p_\theta(f_x|z_x)] - \beta_x D_{KL}(q_\phi(z_x^r|f_x) \| p(z_x^r)) - \beta_x D_{KL}(q_\phi(z_x^v|f_x) \| p(z_x^v))$
    - **Design Motivation**: t-SNE visualizations validate the effectiveness of this design—$z_x^r$ exhibits distinct clustering by class, whereas $z_x^v$ shows highly mixed categories, indicating that semantic-unrelated information is successfully stripped away.

2. **Cross-Alignment Loss**:

    - **Function**: Establishes correspondences between the latent spaces of the two modalities via cross-reconstruction.
    - **Mechanism**:
    $$\mathcal{L}_C = \|D_y(z_x^r) - f_y\|_2^2 + \|D_x(z_x^v \oplus z_y) - f_x\|_2^2$$
    The first term requires that the text feature be reconstructed using only the semantic-related part $z_x^r$; the second term demands that the skeleton feature be reconstructed using a combination of the semantic-unrelated part $z_x^v$ and the text feature $z_y$.
    - **Design Motivation**: This cross-reconstruction elegantly enforces disentanglement—$z_x^r$ must encode sufficient semantic information to reconstruct the text, while $z_x^v$ must supplement style information (body shape, viewpoints) that text cannot provide, so as to fully reconstruct the skeleton.

3. **Adversarial Total Correlation Penalty**:

    - **Function**: Ensures statistical independence between $z_x^r$ and $z_x^v$, preventing information leakage.
    - **Mechanism**: A discriminator $D_T$ is trained to predict whether a concatenated vector $z_x^v \oplus z_x^r$ originates from the same skeleton feature:
    $$\mathcal{L}_T = \log D_T(z_x) + \log(1 - D_T(\tilde{z}_x))$$
    where $\tilde{z}_x$ is constructed by randomly shuffling the indices of $z_x^v$ within the batch and concatenating it with the original $z_x^r$. $D_T$ maximizes this loss while $E_x$ minimizes it—this adversarial game drives the two feature components toward independence.
    - **Design Motivation**: Simply relying on the KL-divergence regularization of VAEs is insufficient to guarantee independence between the two subspaces. The total correlation penalty imposes a stronger constraint, significantly reducing feature redundancy and preventing the domain classifier from biasing toward seen classes.

### Loss & Training

- Total Loss: $\mathcal{L} = \mathcal{L}_{VAE} + \lambda_1 \mathcal{L}_C + \lambda_2 \mathcal{L}_T$
- VAE and the discriminator are trained alternately: the VAE is updated $n_d$ times before each single update of $D_T$.
- A cyclical annealing strategy is employed to mitigate KL divergence vanishing: $\lambda_2'$ is set to 0 for the first 1/3 of samples in each epoch, then linearly scaled up to $\lambda_2$.
- $\lambda_1$ is set to 0 in the first epoch and 1 thereafter (ensuring proper reconstruction capability is learned before enforcing cross-modal alignment).
- GZSL inference adopts a dual-classifier strategy: a classifier for seen classes $C_s$ (utilizing the raw $f_x$), a classifier for unseen classes $C_u$ (utilizing $z_x^r$), and a domain classifier $C_d$ (using logistic regression to fuse their probabilities).

## Key Experimental Results

### Main Results (ZSL)

| Dataset | Split | Ours | Prev. SOTA (SMIE) | Gain |
|--------|------|------|-----------------|------|
| NTU-60 | 55/5 | 82.37% | 77.98% | +4.39% |
| NTU-60 | 48/12 | 41.38% | 40.18% | +1.20% |
| NTU-120 | 110/10 | 68.77% | 65.74% | +3.03% |
| NTU-120 | 96/24 | 46.12% | 45.30% | +0.82% |

### Main Results (GZSL Harmonic Mean)

| Dataset | Split | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| NTU-60 | 55/5 | 66.27% | 59.02% (SynSE) | +7.25% |
| NTU-60 | 48/12 | 42.56% | 36.33% (SynSE) | +6.23% |
| NTU-120 | 110/10 | 60.42% | 54.94% (SynSE) | +5.48% |
| NTU-120 | 96/24 | 44.50% | 41.04% (SynSE) | +3.46% |

### Ablation Study (Random Splits, NTU-60 ZSL)

| Configuration | Accuracy | Description |
|------|--------|------|
| Naive alignment | 69.26% | Baseline without disentanglement |
| FD (Feature Disentanglement Only) | 82.21% | +12.95%, the core contribution |
| SA-DVAE (FD+TC) | 84.20% | +1.99%, further improved by TC |

### Key Findings

- **Feature disentanglement is the core contribution**: Using only FD improves accuracy by +12.95% on the random split of NTU-60, yielding a significant boost.
- **TC penalty primarily improves GZSL**: TC slightly decreases seen-class accuracy but substantially boosts unseen-class accuracy (improving NTU-60 GZSL H from 70.71% to 75.27%), which reduces the bias of the domain classifier toward seen classes.
- Improvements on GZSL tasks are more pronounced than on ZSL (+7.25% vs +4.39%), because feature disentanglement assists the domain classifier in better distinguishing seen from unseen classes.
- The seen-class classifier achieves better accuracy using raw $f_x$ rather than $z_x^r$ (as style information occasionally assists classification).

## Highlights & Insights

- **Precise problem insights**: First to identify the fundamental issue of many-to-one asymmetry in skeleton-based ZSL, observing that massive variation in same-class skeleton sequences acts as the primary barrier to alignment.
- **Exquisite cross-reconstruction design**: Asymmetric reconstruction via $z_x^r → f_y$ and $(z_x^v, z_y) → f_x$ naturally bifurcates semantic-related and style information.
- **Simple and efficient**: All encoders, decoders, and classifiers are single-layer MLPs, and the discriminator has only two layers, yielding low training costs (approx. 4.6 hours on a single RTX 3090 for NTU-60).
- **No reliance on Part-of-Speech (PoS) tags**: Compared to SynSE/JPoSE which require PoS tagging, SA-DVAE directly utilizes simple class names.

## Limitations & Future Work

- The skeleton feature extractor (Shift-GCN/ST-GCN) is pre-trained separately and then frozen, preventing end-to-end joint training with the VAE, which may limit feature quality.
- The latent dimensions for $z_x^r$ and $z_x^v$ require manual tuning (e.g., 160 vs 8); adaptive dimension allocation could yield better results.
- Only class names are leveraged as text (without leveraging rich action descriptions); integrating descriptions generated by LLMs could yield further improvements.
- Multi-view skeleton data augmentation is not explored to increase the diversity of $z_x^v$.
- The random split experiments were averaged over only three runs, which slightly weakens statistical significance.

## Related Work & Insights

- **vs CADA-VAE**: SA-DVAE's direct predecessor, but CADA-VAE does not perform feature disentanglement, forcing all skeleton features to align with text. SA-DVAE outperforms it by +5.53% in ZSL on NTU-60 55/5.
- **vs SynSE/JPoSE**: These methods rely on Part-of-Speech (PoS) tags to align verbs/nouns separately, which increases preprocessing complexity; SA-DVAE is more straightforward and shows superior performance.
- **vs SDGZSL**: An image-based ZSL feature disentanglement method that relies on class-level attributes; SA-DVAE adapts this concept to skeleton-based ZSL and directly substitutes pre-defined attributes with textual descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces feature disentanglement to skeleton-based ZSL for the first time with transparent motivation, although the VAE + disentanglement technical line itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested across three datasets, under both ZSL/GZSL protocols, on both fixed and random splits, and supported with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivation, intuitive architecture scheme, and highly convincing t-SNE visualizations.
- Value: ⭐⭐⭐⭐ The disentanglement approach is generalizable and can be transferred to other cross-modal zero-shot learning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)

</div>

<!-- RELATED:END -->
