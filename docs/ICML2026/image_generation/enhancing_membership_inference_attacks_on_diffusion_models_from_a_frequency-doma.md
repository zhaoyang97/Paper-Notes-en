---
title: >-
  [Paper Note] Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective
description: >-
  [ICML 2026][Image Generation][Diffusion Model] This paper analyzes failure modes of Membership Inference Attacks (MIA) on diffusion models from a frequency-domain perspective. It identifies that high-frequency content amplifies the standard deviation of scores for both member and hold-out samples, thereby diluting membership advantage. The authors propose a "high-f
tags:
  - ICML 2026
  - Image Generation
  - Diffusion Model
date: 2026-05-08
content_hash: 8d268138720b2ceb
---
# Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective

**Conference**: ICML 2026  
**arXiv**: [2505.20955](https://arxiv.org/abs/2505.20955)  
**Code**: https://github.com/poetic2/FreMIA  
**Area**: AI Security / Privacy Attacks (Membership Inference / Diffusion Models)  
**Keywords**: Membership Inference Attack, Diffusion Models, Frequency Domain Analysis, High-frequency Defect, Plug-and-play Filter

## TL;DR
This paper analyzes failure modes of Membership Inference Attacks (MIA) on diffusion models from a frequency-domain perspective. It identifies that high-frequency content amplifies the standard deviation of scores for both member and hold-out samples, thereby diluting membership advantage. The authors propose a "high-frequency filter" module that requires no training and zero additional inference cost. By applying the same FFT low-pass processing to the predicted and target images before calculating reconstruction error, mainstream MIAs such as Naive/SecMI/PIA achieve performance gains of 4–11 percentage points in ASR/AUC/TPR@1%FPR on DDIM and Stable Diffusion (with TPR@1%FPR jumping from 6% to 41% in specific scenarios).

## Background & Motivation

**Background**: Diffusion models demonstrate stunning image generation results, but the risk of "memorizing" the training set is amplified accordingly. Evaluating training data privacy via Membership Inference Attack (MIA) has become a popular research direction. Mainstream MIAs for diffusion models (Naive loss, SecMI, PIA, PIAN, etc.) belong to the "reconstruction error school": given a test image $x_i$, the model predicts $x_{i,t}$ at a certain timestep $t$ versus a target $x_{i,t}^{target}$. The membership score is defined as the distance $\|x_{i,t}-x_{i,t}^{target}\|_q$, followed by thresholding.

**Limitations of Prior Work**: Empirical evidence shows these attacks systematically fail on certain "seemingly easy" samples—**member images with high-frequency content are often misclassified as non-members, while hold-out images with low-frequency content are misclassified as members**. Scatter plots (Fig. 1) and failure sample statistics (Table 1) across MS-COCO/Flickr/CIFAR-100/TINY-IN verify this pattern.

**Key Challenge**: Diffusion models possess "frequency hierarchy"—recovering low-frequency global structures first, then filling in high-frequency details. High-frequency restoration naturally entails higher variance and uncertainty (as proven via spectrum and SNR by Yang et al. 2023; Falck et al. 2025). However, existing MIAs use pixel-level errors, mixing "model-intrinsic randomness" from high frequencies with the "membership" signal, causing the score distributions of members and hold-outs to be simultaneously "widened" by high-frequency noise.

**Goal**: (1) Characterize how high-frequency content systematically damages existing MIAs; (2) Provide a universal enhancement module that requires no training or additional inference time; (3) Theoretically prove why this enhancement is effective.

**Key Insight**: Since high frequency acts as a "common noise source," it should be removed from the error calculation. Utilizing the membership advantage formula from Yeom et al. (2018), $Adv^M(\mathcal{A})\propto \sigma_H/\sigma_M$ (ratio of hold-out vs. member score standard deviations), it is observed that high-frequency variance $\sigma^{high}$ contributes almost equally to both, which lowers the $\sigma_H/\sigma_M$ ratio by inflating the denominator disproportionately.

**Core Idea**: Before inputting into the distance function, both the predicted $x_{i,t}$ and target $x_{i,t}^{target}$ are processed via FFT. High-frequency regions with radius greater than $r_t$ are multiplied by a decay factor $s$ (default 0), then transformed back via IFFT. Using the **low-frequency reconstruction error** instead of the original error provides a plug-and-play enhancement for any "reconstruction error school" MIA, termed FreMIA.

## Method

### Overall Architecture
This work addresses the systematic failure of reconstruction-error-based MIAs on high-frequency samples by inserting a symmetric low-pass filter before error calculation without modifying the attack itself. The authors unify Naive/SecMI/PIA into a general paradigm (Eq. 6): $\mathcal{A}(x_i,\theta)=\mathbb{1}[\|x_{i,t}-x_{i,t}^{target}\|_q \le \tau]$. The differences lie only in how the prediction $x_{i,t}$ and target $x_{i,t}^{target}$ are obtained (Naive uses one-step noise-denoise loss; SecMI uses multi-step DDIM inversion; PIA uses proximal initialization for deterministic noise prediction). FreMIA applies a frequency-domain low-pass filter $\mathcal{F}(\cdot)$ to both images, upgrading the criterion to (Eq. 11): $\mathcal{A}'(x_i,\theta)=\mathbb{1}[\|\mathcal{F}(x_{i,t})-\mathcal{F}(x_{i,t}^{target})\|_q \le \tau]$. This involves only one FFT/IFFT pair before distance calculation with zero learnable parameters.

### Key Designs

**1. MIA General Paradigm Formulation: Universal Filter Modification**

While different attacks appear distinct, patching them individually is tedious. The authors prove in Appendix B that the decision metrics for Naive/SecMI/PIA can be rewritten as the $\ell_q$ distance $\|x_{i,t}-x_{i,t}^{target}\|_q$. By converging all attacks into this expression, inserting a filter before the distance calculation becomes a single-point modification applicable to all methods. This ensures modular plug-and-play capability and allows the theoretical analysis of standard deviation to cover all paradigm-compliant attacks.

**2. Symmetric High-frequency Filter $\mathcal{F}$: Removing Model Randomness from Error**

To address high-frequency noise, $\mathcal{F}$ removes high-frequency components with radius $r>r_t$ before the distance function. Specifically, it computes $\mathbf{X}=FFT(x_{i,t})$, applies a mask $\beta_{i,t}(r)=s$ (if $r>r_t$) or $1$ (otherwise), and performs IFFT: $\mathcal{F}(x_{i,t})=IFFT(FFT(x_{i,t})\odot\beta_{i,t}(r))$. A value of $s=0$ indicates a hard-cutoff low-pass. Using the same mask for both $x_{i,t}$ and $x_{i,t}^{target}$ ensures that high-frequency signals cancel out during subtraction. The remaining distance reflects the "fitting difference in low-frequency parts," which is the most stable band for reflecting memorization. This is justified by the fact that high-frequency signals in failed member samples are significantly higher than in hold-outs (Table 1).

**3. Frequency Decomposition of Membership Advantage (Proposition 4.2): Theoretical Proof**

To provide a provable basis, the total score standard deviation for members/hold-outs is decomposed: $\sigma_M^2=\sigma_M^{low\,2}+\sigma_M^{high\,2}$ and $\sigma_H^2=\sigma_H^{low\,2}+\sigma_H^{high\,2}$. Assuming the low-frequency gap is $\sigma_H^{low}-\sigma_M^{low}=\Delta$ and high-frequency satisfies $\sigma_M^{high}=k\cdot\sigma_H^{high}$, the paper proves that if $k\ge 1$ (member high-frequency variance is no less than hold-out), then $\sigma_H'/\sigma_M' > \sigma_H/\sigma_M$. According to $Adv^M(\mathcal{A})\propto \sigma_H/\sigma_M$ (Eq. 8), the advantage strictly increases. Intuitively, while high frequency adds similar noise variance to both, the relative proportion added to the denominator (member) is higher; removing it improves the ratio.

### Loss & Training
**No training required**. The module adds an FFT→mask→IFFT step during inference for existing MIAs. The complexity is $O(N\log N)$, which is negligible compared to multi-step diffusion sampling. The only hyperparameter is the high-frequency radius $r_t$ (dataset resolution dependent: $r_t=2$ for CIFAR-100/TINY-IN; $r_t=5$ for MS-COCO/Flickr), with the decay factor $s$ defaulting to 0.

## Key Experimental Results

### Main Results

Comparison of three datasets × three baselines (Naive / SecMI / PIA) with and without +F (High-frequency filter) on DDIM (from Table 2):

| Dataset / Method | ASR (Baseline → +F) | AUC (Baseline → +F) | TPR@1%FPR (Baseline → +F) |
|---|---|---|---|
| STL10-U / SecMI | 81.14 → 86.51 | 87.39 → 91.39 | 11.11 → 14.63 |
| CIFAR-100 / SecMI | 80.56 → 88.09 | 87.21 → 93.74 | 16.50 → 24.32 |
| Tiny-IN / PIA | 80.87 → 89.12 | 86.30 → 93.23 | 14.66 → 32.91 |
| Avg. Gain across 3 Datasets | +5.4~+7.3 | +4.5~+6.4 | +4.5~+11.8 |

Results on fine-tuned Stable Diffusion (Table 3) are even more significant:

| Dataset / Method | ASR (Baseline → +F) | AUC (Baseline → +F) | TPR@1%FPR (Baseline → +F) |
|---|---|---|---|
| Pokémon / Naive | 79.50 → 87.88 | 86.97 → 94.14 | 6.49 → 41.25 |
| MS-COCO / Naive | 80.29 → 93.60 | 87.85 → 98.32 | 4.80 → 41.99 |
| Flickr / Naive | 79.29 → 90.90 | 86.14 → 96.82 | 16.59 → 67.60 |

TPR@1%FPR is a strict metric for MIA. The jump from 4.80% to 41.99% indicates the filter is particularly effective at high-confidence decision points.

### Ablation Study

| Configuration / Phenomenon | Observations / Description |
|---|---|
| Baseline (No filtering) | High-frequency content in failed member samples is ~0.07–0.18 higher than hold-outs (Table 1). |
| +F (Symmetric low-pass) | Eliminates aforementioned bias; ASR/AUC increases monotonically across all baselines and datasets. |
| Radius $r_t$ | Insensitive within reasonable ranges; too small retains noise, too large loses memorization signal. |
| Normality Assumption | Proposition 4.2 is proven under normality, but Appendix D.2 shows gains hold for non-normal distributions. |
| Time Overhead | Single FFT/IFFT cost is < 1% compared to multi-step DDIM sampling; "negligible." |

### Key Findings
- **High frequency is the primary source of "false signals"**: Failure statistics (Table 1), scatter plots (Fig. 1), and pixel-level distance visualizations (Appendix D.1) converge on the conclusion that reconstruction error spikes caused by high frequency are unrelated to memorization.
- **Filtering yields maximum gains in low FPR regions**: TPR@1%FPR improvement is significantly higher than ASR/AUC. Cleaner low-frequency signals allow easier thresholding in high-confidence zones, which is most critical in real-world privacy investigations.
- **Universal across architectures and datasets**: Gains are observed on both unconditional DDIM and fine-tuned Stable Diffusion, across low-res natural images (STL10), small samples (Pokémon), and high-res natural images (MS-COCO). The "high-frequency defect" is an intrinsic property of diffusion models.

## Highlights & Insights
- **Theoretical and Engineering Synergy**: A simple "low-pass filter" trick is validated by a formal proof based on the membership advantage formula ($Adv^M \propto \sigma_H / \sigma_M$). This research paradigm of "small engineering modification + provable guarantee" is a valuable model for empirical fields like MIA.
- **Unifying Attack and Defense via Frequency**: Discussions on the frequency domain of diffusion models usually focus on generation quality. This work transfers it to "privacy identifiability," suggesting that any distance-based privacy/security analysis (attribution, unlearning verification) could be re-evaluated in the frequency domain.
- **Zero-cost Plug-and-play**: No retraining, no shadow models, no internal attack tuning, and no speed penalty. This engineering friendliness makes it likely to be integrated into all future diffusion MIA benchmarks.
- **Symmetric Perturbation as a Transferable Idea**: Applying the same perturbation symmetrically to two compared images to cancel noise while preserving signal is a design pattern that might be applicable to OOD detection or adversarial sample detection.

## Limitations & Future Work
- **Scope limited to "Reconstruction Error" attacks**: White-box gradient attacks (e.g., GSA) or likelihood/token probability-based attacks (common in autoregressive models) are not covered by this paradigm.
- **Hyperparameter $r_t$ is dataset-dependent**: The paper suggests values based on experience (2 for CIFAR, 5 for COCO), but lacks an automated selection strategy for unknown datasets.
- **Strong Theoretical Assumptions**: Proposition 4.2 strictly holds under normality with $k \ge 1$. While empirical tests support this, extreme outlier datasets (e.g., high-contrast medical images) may require further verification.
- **Defense Perspectives**: This is an attack enhancement; naturally, the next question is whether models can resist this via frequency-balanced training or DP-noise injection specifically into high-frequency bands.
- **Multi-modal and Video Diffusion**: Stability remains to be tested on video or 3D diffusion where high-frequency patterns may differ.

## Related Work & Insights
- **vs SecMI (Duan et al., 2023)**: SecMI uses DDIM multi-step inversion as $x_{i,t}^{target}$. Applying $\mathcal{F}$ to SecMI increases all metrics, suggesting its failure mode stems from high-frequency instability rather than the inversion process itself.
- **vs PIA (Kong et al., 2023)**: PIA uses proximal initialization for deterministic noise to eliminate sampling randomness. This work shows that even without sampling randomness, **image intrinsic high frequency still causes attack uncertainty**. These two types of uncertainty are orthogonal.
- **vs LiRA (Carlini et al., 2022; 2023)**: LiRA estimates likelihood ratios via shadow models, which is computationally prohibitive for diffusion models. FreMIA provides a "zero-training" path, though not as tight as LiRA in an information-theoretic sense.
- **vs Frequency-aware Generation (Yang et al., 2023; Falck et al., 2025)**: These works characterize the frequency hierarchy for generation quality. This paper borrows those conclusions to explain attack failures, demonstrating a technical transfer from "generation mechanism research" to "privacy analysis."

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce frequency-domain perspective to diffusion MIA; unified paradigm is strong.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 baselines × 6 datasets × 2 architectures; extensive appendix coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization and intuitive explanations; minor missing definitions in body.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero cost, universal gains. Likely to become a standard component for diffusion MIA benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Black-box Membership Inference Attacks on the Pre-training Data of Image-generation Models](../../CVPR2026/image_generation/black-box_membership_inference_attacks_on_the_pre-training_data_of_image-generat.md)
- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](../../ECCV2024/image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)
- [\[CVPR 2026\] Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective](../../CVPR2026/image_generation/toward_diffusible_high-dimensional_latent_spaces_a_frequency_perspective.md)
- [\[AAAI 2026\] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective](../../AAAI2026/image_generation/unseen_enhancing_dataset_pruning_from_a_generalization_perspective.md)

</div>

<!-- RELATED:END -->
