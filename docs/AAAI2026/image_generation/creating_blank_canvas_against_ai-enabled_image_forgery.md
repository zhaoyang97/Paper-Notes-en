---
title: >-
  [Paper Note] Creating Blank Canvas Against AI-Enabled Image Forgery
description: >-
  [AAAI 2026][Image Generation][Image Tampering Localization] This paper proposes a "blank canvas" mechanism that applies adversarial perturbations to make SAM "see nothing" in protected images. When a protected image is tampered with, the tampered regions disrupt the perturbations and become automatically detectable by SAM, enabling proactive tampering localization without requiring any tampered training data.
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Image Tampering Localization"
  - "Adversarial Perturbation"
  - "SAM"
  - "Frequency-Aware Optimization"
  - "Proactive Protection"
date: 2026-05-08
content_hash: 355a4a069694a780
---

# Creating Blank Canvas Against AI-Enabled Image Forgery

**Conference**: AAAI 2026
**arXiv**: [2511.22237](https://arxiv.org/abs/2511.22237)  
**Code**: [GitHub](https://github.com/qsong2001/blank_canvas)  
**Area**: Image Generation
**Keywords**: Image Tampering Localization, Adversarial Perturbation, SAM, Frequency-Aware Optimization, Proactive Protection

## TL;DR

This paper proposes a "blank canvas" mechanism that applies adversarial perturbations to make SAM "see nothing" in protected images. When a protected image is tampered with, the tampered regions disrupt the perturbations and become automatically detectable by SAM, enabling proactive tampering localization without requiring any tampered training data.

## Background & Motivation

AIGC-based image editing tools (e.g., SD Inpaint, ControlNet, SDXL) have made highly realistic image forgery trivially easy, posing serious threats to public trust and social stability. Existing tampering localization methods predominantly adopt a **passive post-hoc analysis** paradigm, relying on forgery patterns learned during training, and suffer from the following limitations:

**Poor Generalization**: Effective only for forgery types seen during training; performance degrades sharply when encountering novel AIGC edits.

**Heavy Data Dependency**: Requires large quantities of annotated tampered samples for training.

**High Computational Overhead**: Training tampering detection foundation models demands substantial computational resources.

The authors propose a key insight: **alterations on a blank canvas are far easier to detect than modifications on a complex image**. Rather than detecting tampering traces after the fact, it is preferable to proactively transform the image into a "blank canvas" from the perspective of visual foundation models—making any subsequent tampering immediately apparent.

## Method

### Overall Architecture

The overall pipeline consists of two stages:

1. **Blank Canvas Creation**: Imperceptible adversarial perturbations are added to the original image so that SAM cannot segment anything ("Segment Nothing").
2. **Tampering Localization**: When the protected image is tampered with, the tampered regions disrupt the adversarial perturbations, allowing SAM to re-perceive and segment those areas.

Image owners apply the protection operation before publishing; the perturbations are invisible to the human eye but fully deceive SAM. SAM ViT-H is used as the backbone model, with single-point prompting at coordinate $(0,0)$ for inference.

### Key Designs

#### 1. Blank Canvas Creation — Basic Adversarial Attack

The core objective is to find a perturbation $\delta$ such that the output confidence of SAM on the protected image converges to a constant $C$:

$$\Phi' = \text{SAM}(x_{\text{clear}} + \delta, \mathcal{P}), \quad \Phi'[i,j] \approx C, \; \forall i,j$$

An MSE loss is used to optimize the perturbation:

$$\mathcal{L}_{\text{attack}} = \text{MSE}(\Phi, C)$$

where $C=15$ (consistent with the typical confidence value of background regions in SAM). Optimization is performed under the PGD framework with a maximum perturbation magnitude of $16/255$ and a step size of $2/255$.

#### 2. Frequency-Aware Optimization (Core Contribution)

Experiments reveal that naive adversarial attacks cannot fully deceive SAM—SAM retains perceptual sensitivity in high-frequency edge and texture regions. A frequency-aware optimization strategy is therefore proposed, comprising three synergistic components:

**(a) Wavelet-Domain High-Frequency Decomposition ($\mathcal{L}_{\text{hfc}}$)**: Discrete wavelet transform using the Daubechies-8 basis extracts high-frequency components, with perturbation constrained by a Canny edge mask:

$$\mathcal{L}_{\text{hfc}} = \sum_{k=1}^{K} \|\mathcal{W}_k(\tilde{x}) \odot M_{\text{edge}} - \mathcal{W}_k(x) \odot M_{\text{edge}}\|_F^2$$

**(b) Structure-Preserving Constraint ($\mathcal{L}_{\text{lfc}}$)**: Adaptive SSIM is employed to protect low-frequency components and maintain visual naturalness:

$$\mathcal{L}_{\text{lfc}} = \text{SSIM}(\phi_m, \tilde{\phi}_m)$$

**(c) Adaptive Spectral Optimization**: A spectral projection mask $\mathcal{M}$ is designed in the frequency domain to concentrate perturbation energy in high-frequency bands:

$$\mathcal{M}(u,v) = \begin{cases} 1, & \sqrt{u^2+v^2} \geq f_{\text{cutoff}} \\ 0, & \text{otherwise} \end{cases}$$

The final optimization objective integrates all loss terms, updated via momentum-based adaptive-step gradient methods.

#### 3. Tampering Localization

For a protected "blank canvas" image, the tampering operation $\Delta x$ disrupts the local adversarial perturbations, allowing SAM to re-perceive those regions:

$$\mathcal{M}_{\text{tamper}} = \mathbb{I}(\|\text{SAM}(\tilde{x} + \Delta x)\|_2 > \tau_{\text{detect}})$$

The detection threshold $\tau_{\text{detect}}$ is determined adaptively via Otsu's method.

### Loss & Training

The overall optimization objective is:

$$\delta^* = \arg\max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}_{\text{attack}} + \lambda \mathcal{L}_{\text{lfc}} - \beta \mathcal{L}_{\text{hfc}}$$

- **Training-Free**: The entire process requires no network training; only image-level perturbations are optimized.
- Optimization uses the PGD framework with momentum gradient updates and spectral projection.
- Learning rate follows an exponential warm-up schedule: $\alpha_t = \alpha_0(1 - e^{-5t/T})$

## Key Experimental Results

### Main Results

**Table 1: Comparison on Classic Tampering Localization Benchmarks**

| Method | CASIA1+ IoU/F1 | Columbia IoU/F1 | NIST IoU/F1 |
|--------|---------------|-----------------|-------------|
| MVSS-Net | 0.40/0.48 | 0.48/0.61 | 0.24/0.29 |
| FakeShield | 0.56/0.62 | 0.68/0.76 | 0.34/0.39 |
| EditGuard | 0.60/0.67 | 0.70/0.78 | 0.35/0.40 |
| **Ours** | **0.62/0.67** | **0.74/0.81** | **0.31/0.45** |

**Table 2: Comparison on AIGC Editing Methods (F1/IoU)**

| Method | SD Inpaint | ControlNet | SDXL | RePaint |
|--------|-----------|------------|------|---------|
| MVSS-Net† | 0.694/0.575 | 0.678/0.558 | 0.482/0.359 | 0.185/0.111 |
| EditGuard | 0.966/0.936 | 0.968/0.940 | 0.965/0.936 | 0.967/0.938 |
| **Ours** | **0.972/0.958** | **0.973/0.938** | **0.970/0.958** | **0.961/0.957** |

Under AIGC editing scenarios, the proposed method achieves F1 > 95% and IoU ≈ 95%, comprehensively outperforming passive methods and matching or slightly surpassing EditGuard.

### Ablation Study

| Configuration | F1 | IoU |
|---------------|----|-----|
| No protection (a) | 0.352 | 0.378 |
| $\mathcal{L}_{\text{mse}}$ + $\mathcal{L}_{\text{stealth}}$ only (b) | 0.934 | 0.928 |
| $\mathcal{L}_{\text{mse}}$ + adaptive optimization (c) | 0.931 | 0.921 |
| **Full method** | **0.964** | **0.955** |

### Key Findings

1. Naive adversarial attacks (MSE loss only) still fail to fully deceive SAM in high-frequency regions, producing false positives.
2. Frequency-aware optimization is critical—removing it degrades performance to near the unprotected baseline.
3. The method requires no tampered training data whatsoever, constituting genuine zero-shot tampering localization.
4. The proposed method significantly outperforms all passive methods in AIGC editing scenarios (passive methods achieve F1 < 0.7).

## Highlights & Insights

1. **Paradigm Innovation**: Shifts from passive detection to proactive protection, reformulating "detecting tampering" as "finding traces on a blank canvas"—a concept that is both elegant and effective.
2. **Training-Free Design**: Only image-level perturbations are optimized; no model training is required, leveraging off-the-shelf SAM directly.
3. **Deep Frequency-Domain Understanding**: Accurately diagnoses SAM's robustness in high-frequency regions and proposes targeted frequency-aware optimization.
4. **High Practicality**: Image owners can apply one-click protection before publishing, and anyone can verify tampering using standard SAM.

## Limitations & Future Work

1. **White-Box Assumption**: The current method requires full access to SAM's model weights; transferability to other visual foundation models has not been validated.
2. **Perturbation Visibility**: The perturbation bound of $16/255$ may affect image quality in certain scenarios.
3. **Robustness to Social Platform Compression**: Common social media operations such as JPEG compression and rescaling may destroy the adversarial perturbations.
4. **Adversarial Tampering**: If a forger is aware of the protection mechanism, they may design targeted "de-protection" attacks.
5. Extending the method to SAM2 or other visual foundation models is a promising direction.

## Related Work & Insights

- **EditGuard**: Also a proactive protection method, but relies on steganographic information embedding with limited interpretability.
- **SAM-Attack / Dark-SAM**: Study the adversarial robustness of SAM, but with the goal of attacking rather than protecting.
- This paper creatively applies adversarial attacks to image protection, offering a transferable conceptual framework to other domains.
- Inspiration: Could a similar approach protect video or 3D content from AI-based tampering?

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The blank canvas concept is highly original)
- Technical Depth: ⭐⭐⭐⭐ (Frequency-aware optimization is elegantly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on both classic and AIGC scenarios)
- Practical Value: ⭐⭐⭐⭐ (Training-free and immediately deployable)
- Overall Score: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](../../NeurIPS2025/image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](../../ICML2026/image_generation/order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[AAAI 2026\] MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](../../CVPR2026/image_generation/rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)

</div>

<!-- RELATED:END -->
