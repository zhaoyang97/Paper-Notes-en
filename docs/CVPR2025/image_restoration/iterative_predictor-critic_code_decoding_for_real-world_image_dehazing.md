---
title: >-
  [Paper Note] Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing
description: >-
  [CVPR 2025][Image Restoration][Real-World Image Dehazing] IPC-Dehaze proposes an iterative Predictor-Critic decoding framework based on a VQGAN codebook prior. By utilizing a Code-Critic to evaluate the inter-relations among codebook sequences to determine which codes should be retained or resampled, the framework achieves progressive, easy-to-hard dehazing from clear regions to dense haze regions, significantly surpassing state-of-the-art methods in real-world scenarios.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Real-World Image Dehazing"
  - "Codebook Prior"
  - "Iterative Decoding"
  - "VQGAN"
  - "Predictor-Critic"
date: 2026-05-08
content_hash: 7682343ebf2d6cd4
---

# Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing

**Conference**: CVPR 2025  
**arXiv**: [2503.13147](https://arxiv.org/abs/2503.13147)  
**Code**: Yes (marked as [Code] [Website] in the paper)  
**Area**: Image Restoration / Dehazing  
**Keywords**: Real-World Image Dehazing, Codebook Prior, Iterative Decoding, VQGAN, Predictor-Critic

## TL;DR

IPC-Dehaze proposes an iterative Predictor-Critic decoding framework based on a VQGAN codebook prior. By utilizing a Code-Critic to evaluate the inter-relations among codebook sequences to determine which codes should be retained or resampled, the framework achieves progressive, easy-to-hard dehazing from clear regions to dense haze regions, significantly surpassing state-of-the-art methods in real-world scenarios.

## Background & Motivation

**Background**: Real-world image dehazing is a classic image restoration problem. Traditional methods rely on hand-crafted priors (such as the dark channel prior) and exhibit poor generalization capabilities. Deep learning-based approaches perform well on synthetic datasets but yield suboptimal results when facing complex degradation in real-world scenes (e.g., non-uniform haze, color cast, low-light conditions). RIDCP pioneered the introduction of a high-quality codebook prior from a pre-trained VQGAN for dehazing, achieving significant progress.

**Limitations of Prior Work**: Existing codebook-based methods employ one-shot decoding, which matches hazy image tokens to a high-quality codebook in a single pass. This introduces two issues: (1) it overlooks the spatial variation of haze degradation, where thin haze regions retain more details and are easy to restore, while dense haze regions suffer from severe information loss and are difficult to recover in one step; (2) the nearest-neighbor matching processes each token independently, ignoring the interdependencies among codes, which can cause inconsistencies across adjacent regions.

**Key Challenge**: The one-shot mechanism fails to utilize the physical reality that "restored clear regions contain vital clues for recovering dense haze regions." Meanwhile, the independent code sampling strategy neglects the correlation among sequences.

**Goal**: (1) To progressively utilize restored high-quality codes to guide subsequent code predictions; (2) to determine which codes should be retained and which should be resampled in each iteration.

**Key Insight**: The idea of masked token modeling from MaskGIT is adapted for image restoration, but with two key adjustments: (1) low-quality features are used as conditioning inputs (rather than generating from scratch); (2) a Code-Critic is introduced to evaluate code-to-code correlations (rather than relying solely on the confidence of individual codes).

**Core Idea**: Code-Predictor is utilized to iteratively predict high-quality codes, and Code-Critic evaluates the inter-code relationships to decide on retention or resampling, enabling progressive easy-to-hard dehazing.

## Method

### Overall Architecture

The training consists of three stages: (1) Pre-training VQGAN to learn a high-quality codebook (codebook size $K=1024$, $\text{dim}=256$); (2) Stage I training of the Code-Predictor: mixing tokens from the hazy image and the clear image based on random masks, and training the Predictor to predict high-quality codes at all locations; (3) Stage II training of the Code-Critic: evaluating whether the codes sampled by the Predictor are correct. During inference, starting from the hazy image tokens, a $T=8$ iteration process is conducted: at each step, the Predictor predicts all codes $\rightarrow$ the Critic evaluates and retains the most reliable codes $\rightarrow$ the remaining positions are re-predicted in the next round, guided by the retained codes.

### Key Designs

1. **Code-Predictor (RSTB-based Code Predictor)**:

    - **Function**: Predicts a complete sequence of high-quality codes from an input that mixes low-quality and high-quality tokens.
    - **Mechanism**: The encoder $E_L$ encodes the hazy image into $Z_l$, which is then mixed with the restored high-quality tokens $Z_c$ according to a mask $M_t$ to obtain $Z_t = Z_l \odot M_t + Z_c \odot (1-M_t)$. The Code-Predictor $G_\theta$ (consisting of 4× RSTB blocks) takes $Z_t$ as input and outputs a probability distribution $p_\theta \in \mathbb{R}^{N \times K}$ over the codebook, trained using cross-entropy loss. Meanwhile, an SFT (Spatial Feature Transform) module is introduced in the encoder and decoder to align the feature distributions of the hazy and clear images.
    - **Design Motivation**: Through training on random mask mixtures, the Predictor learns "how to better predict the remaining positions when some positions already have high-quality codes," which serves as the foundation for iterative inference.

2. **Code-Critic (Code Evaluator)**:

    - **Function**: Evaluates whether each code in the sequence sampled by the Predictor is correct, deciding whether to retain or resample it.
    - **Mechanism**: Taking the code sequence $S$ sampled by the Predictor as input, 2× RSTB blocks are used to output a mask score $p_\phi$ for each position. The training label is $M = (S \neq S_h)$, indicating that if the sampled code differs from the ground truth code, it should be rejected. The network is trained with BCE loss. To increase training diversity, the sampling temperature of the Predictor is set to 2 to generate more erroneous codes for the Critic to learn and evaluate.
    - **Design Motivation**: The output confidence of the Predictor alone cannot effectively judge the quality of the codes because this judgment is independent and ignores individual relations. The Critic, taking the entire sequence as input, can capture global correlations (e.g., semantic consistency across adjacent regions) among codes, thereby making more rational retention/rejection decisions.

3. **Iterative Decoding Schedule**:

    - **Function**: Controls how many codes are retained in each iteration.
    - **Mechanism**: A cosine scheduling function $\gamma(t/T)$ is used to control the number of retained codes in each round. Initially, all positions need to be predicted ($M_1=1$). As the iteration progresses, more and more codes are retained. After the Critic's evaluation, $\lceil\gamma(t/T) \cdot N\rceil$ positions with the highest mask scores are selected for resampling, while the rest are retained. After $T=8$ iterations, all codes are replaced by high quality codes.
    - **Design Motivation**: The cosine schedule ensures that early iterations retain a small number of the most reliable codes (e.g., thin haze regions) and gradually increase them in later iterations, allowing the restored clear regions to naturally guide the recovery of dense haze regions.

### Loss & Training

VQGAN pre-training: $\mathcal{L}_{VQGAN} = \mathcal{L}_1 + \mathcal{L}_{code} + \mathcal{L}_{per} + 0.1 \mathcal{L}_{adv}$. Stage I: cross-entropy loss $\mathcal{L}_\theta$. Stage II: BCE loss $\mathcal{L}_\phi$. Optimized using Adam (learning rate = 1e-4), trained on 4 × RTX 3090 GPUs, with 400K, 100K, and 20K iterations for the three stages, respectively.

## Key Experimental Results

### Main Results

| Dataset | Metric | IPC-Dehaze | RIDCP | KA-Net | Gain (vs RIDCP) |
|--------|------|------------|-------|--------|----------------|
| RTTS | MUSIQ↑ | **59.60** | 55.23 | 54.64 | +4.37 |
| RTTS | Q-Align↑ | **3.49** | 3.24 | 3.09 | +0.25 |
| RTTS | CLIPIQA↑ | **0.44** | 0.30 | 0.28 | +0.14 |
| Fattal | MUSIQ↑ | **66.22** | 65.48 | 64.09 | +0.74 |
| Fattal | Q-Align↑ | **4.234** | 3.799 | 3.982 | +0.435 |
| URHI | MUSIQ↑ | **62.5** | 61.39 | 58.57 | +1.11 |

Across 6 no-reference IQA metrics (MUSIQ, PI, MANIQA, CLIPIQA, Q-Align, TOPIQ), IPC-Dehaze consistently ranks first or second on three real-world datasets: RTTS, Fattal, and URHI.

### Ablation Study

| Method | MUSIQ↑ | PI↓ | MANIQA↑ | Q-Align↑ | TOPIQ↑ |
|------|--------|-----|---------|----------|--------|
| NN Matching | 58.19 | 3.25 | 0.303 | 3.25 | 0.458 |
| w/o Code-Critic | 57.74 | 3.32 | 0.303 | 3.36 | 0.462 |
| Ours (Full) | **59.60** | **3.22** | **0.327** | **3.49** | **0.500** |

### Key Findings

- The Code-Predictor performs better than Nearest-Neighbor (NN) matching in iterative scenarios: increasing the number of iterations does not change the results for NN (due to independent matching), whereas the Predictor continuously improves by utilizing the restored codes.
- The Code-Critic is crucial for performance gains: without the Critic, the Predictor cannot effectively decide which codes to retain, resulting in limited iterative improvement; with the Critic, TOPIQ increases from 0.462 to 0.500.
- Visualizations display that the mask changes guided by the Critic follow an "easy-to-hard" pattern, progressing from near-to-far and from thin-to-thick haze, which validates the design motivation.
- The method is particularly prominent in challenging scenarios such as color casts (sandstorms, low light) and dense haze.

## Highlights & Insights

- **The adaptation of MaskGIT's concept to image restoration is elegant**: The "progressive unveiling" strategy in generative tasks is transformed into a "progressive restoration from clear to hazy regions" in restoration. This physical intuition is highly natural—real-world haze is indeed spatially variant, and results from thin haze regions can provide vital contextual clues for dense haze regions.
- **Evaluating code relationships via the Code-Critic rather than independent confidence values**: This addresses a critical drawback of MaskGIT when applied to restoration tasks. While the diversity of independent sampling is an advantage in image generation, code sequences in restoration must be globally consistent to produce natural images.
- **The iterative process can be visualized in real-time**: The intermediate results of each iteration clearly illustrate the dehazing progress, facilitating better understanding and debugging.

## Limitations & Future Work

- The 8 iterations increase inference time, which may not be fast enough for real-time applications such as autonomous driving dehazing.
- The codebook size is fixed at 1024, which might limit representation capabilities in extremely complex scenes.
- The training of the Code-Predictor relies on RIDCP's synthetic data generation method, meaning that a domain gap between synthetic and real-world data still exists.
- The training of the Code-Critic depends on the sampling distribution of the Code-Predictor, causing a coupling effect between the two; joint end-to-end training might be a superior alternative.

## Related Work & Insights

- **vs RIDCP**: RIDCP uses one-shot codebook matching, which limits its performance in dense and non-uniform haze; IPC-Dehaze's iterative mechanism can progressively handle regions of varying difficulty.
- **vs MaskGIT**: MaskGIT is designed for unconditional image generation ($HQ \rightarrow HQ$), making its direct application to $LQ \rightarrow HQ$ restoration impractical; IPC-Dehaze uses low-quality tokens as conditioning inputs and introduces a Critic to evaluate relationships.
- **vs KA-Net**: KA-Net also utilizes a codebook prior but remains a one-shot solution, overall performing worse than IPC-Dehaze on quantitative metrics.

## Rating

- Novelty: ⭐⭐⭐⭐ The iterative Predictor-Critic decoding paradigm in image restoration is innovative, though its base building blocks (VQGAN, RSTB, masked modeling) are already established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three real-world datasets using 6 metrics alongside detailed ablation studies, but lacks evaluation with reference-based metrics on synthetic datasets.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described, the algorithmic pseudo-code is well-structured, and visual comparisons are intuitive.
- Value: ⭐⭐⭐⭐ The concept of iterative codebook decoding can generalize to other low-level vision tasks (e.g., super-resolution, denoising).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](../../ICCV2025/image_restoration/self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](../../ICCV2025/image_restoration/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)

</div>

<!-- RELATED:END -->
