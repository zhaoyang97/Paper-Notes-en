---
title: >-
  [Paper Note] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images
description: >-
  [NeurIPS 2025][Image Generation][Watermark Forging Attack] This paper proposes WMCopier, the first diffusion-model-based no-box watermark forging attack that requires no prior knowledge of the target watermarking algorit…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Watermark Forging Attack"
  - "Diffusion Models"
  - "DDIM Inversion"
  - "Invisible Watermark"
  - "No-Box Attack"
date: 2026-05-08
content_hash: df6af8b7fb46d1a1
---

# WMCopier: Forging Invisible Image Watermarks on Arbitrary Images

**Conference**: NeurIPS 2025
**arXiv**: [2503.22330](https://arxiv.org/abs/2503.22330)
**Code**: [GitHub](https://github.com/holdrain/WMCopier)
**Area**: Image Generation
**Keywords**: Watermark Forging Attack, Diffusion Models, DDIM Inversion, Invisible Watermark, No-Box Attack

## TL;DR

This paper proposes WMCopier, the first diffusion-model-based no-box watermark forging attack that requires no prior knowledge of the target watermarking algorithm. By training an unconditional diffusion model to learn the watermark distribution, injecting watermark signals via shallow DDIM inversion, and refining results through iterative optimization, WMCopier achieves high forging success rates against both open-source and commercial watermarking systems, including Amazon.

## Background & Motivation

With the explosive growth of AI-generated content (AIGC), invisible image watermarking has become a critical technology for content provenance and accountability. Major AI service providers such as Google, Amazon, and OpenAI are actively integrating watermarking systems. However, **watermark forging attacks**—embedding traceable watermarks into unauthorized content—pose a serious threat to the credibility of these systems:
- Attackers can distribute harmful images bearing forged legitimate watermarks.
- Innocent AI service providers may be erroneously attributed as the content source.
- This leads to reputational damage and legal liability.

Existing forging attacks primarily operate under a black-box setting (assuming access to the embedding interface or detector), but in practice watermark embedding is typically integrated within the generation service and cannot be accessed independently. The **no-box setting** (only watermarked images are available, with no knowledge of the watermarking algorithm) is more realistic, yet existing methods (Yang et al.) perform poorly—they assume a constant watermark signal across all images and estimate the watermark by computing the average residual between watermarked and natural images, ignoring domain differences and the adaptive nature of watermarks.

The authors' key insight is that **diffusion models are natural distribution learners** capable of implicitly capturing the watermark distribution in training data. Training an unconditional diffusion model on watermarked images causes the denoising process to naturally steer outputs toward the watermarked distribution.

## Method

### Overall Architecture

WMCopier consists of three stages:
1. **Watermark Estimation**: Train an unconditional diffusion model on watermarked images.
2. **Watermark Injection**: Inject the watermark signal into target images via shallow DDIM inversion.
3. **Refinement**: Iteratively optimize to balance visual fidelity and watermark detection rate.

### Key Designs

1. **Diffusion-Model-Based Watermark Distribution Estimation**: An unconditional diffusion model $\mathcal{M}_\theta$ is trained on an auxiliary dataset $\mathcal{D}_{\text{aux}} = \{x^w | x^w \sim p_w(x)\}$ of 5,000 watermarked images. The core theoretical analysis shows that for a watermarked image $x^w = x + w$, after forward diffusion $x_t^w = x_t + \sqrt{\alpha_t} w$, the noise predictor outputs:
    $\epsilon_\theta(x_t^w, t) = \hat{\epsilon}(x_t + \sqrt{\alpha_t} w) \approx \hat{\epsilon}(x_t) + \delta_t(w)$
   The prediction bias $\delta_t(w)$ introduced by watermark signal $w$ accumulates over each denoising step, steering the model output distribution toward $p_w(x)$. This means the model implicitly learns the statistical characteristics of the watermark.

2. **Shallow Inversion Injection**: Using full inversion ($T_S = T$) leads to severe quality degradation, as inversion of out-of-distribution images accumulates large reconstruction errors in deeper steps. Experiments reveal that the watermark signal is primarily destroyed/recovered in shallow steps ($t \leq 400, T=1000$). Therefore, only shallow inversion up to $T_S < T$ is performed (default $T_S=40, T=100$):

    - Deep diffusion steps that contribute little to watermark injection but severely damage semantics are skipped.
    - The visual fidelity of the original image is preserved.
    - The watermark bias $\delta_t(w)$ during denoising still effectively guides generation.

3. **Iterative Refinement**: Slight artifacts may remain after shallow inversion. Gradient ascent is applied to simultaneously optimize the watermark distribution likelihood and semantic fidelity:
    $x^{f(i+1)} = x^{f(i)} + \eta \nabla_{x^{f(i)}} \left[\log p_w(x^{f(i)}) - \lambda \|x^{f(i)} - x\|^2 \right]$
   where $\log p_w(x^f)$ is approximated using the score function of the trained diffusion model:
    $\nabla_{x^f} \log p_w(x^f) \approx -\frac{1}{\sqrt{1-\alpha_{t_l}}} \epsilon_\theta(x_{t_l}^f, t_l)$
   $\lambda=100$ controls the trade-off between semantic preservation and watermark injection, with $L=100$ iterations.

### Loss & Training

- Diffusion model training: Standard denoising objective $\mathbb{E}[\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2]$
- Auxiliary dataset: Only 5,000 watermarked images required
- DDIM sampling steps $T=100$, shallow inversion up to $T_S=40$
- Refinement parameters: $\lambda=100$, $\eta=10^{-4}$, $L=100$, $t_l=1$

## Key Experimental Results

### Main Results

**Comparison of Attacks Against Four Open-Source Watermarking Schemes (Average Results)**

| Attack Method | Setting | PSNR↑ | Forged Bit Acc↑ | FPR@$10^{-6}$↑ |
|---------|------|:---:|:---:|:---:|
| Wang et al. | Black-box | 31.50 | 84.32% | 76.64% |
| Yang et al. | No-box | 30.62 | 54.52% | 0.08% |
| **WMCopier** | **No-box** | **32.94** | **94.58%** | **83.71%** |

**Attack Against Amazon Commercial Watermarking System**

| Method | Dataset | PSNR↑ | Success Rate↑ | Confidence↑ |
|------|--------|:---:|:---:|:---:|
| Yang et al. | DiffusionDB | 23.42 | 29.0% | 2 |
| **WMCopier** | DiffusionDB | **32.57** | **100.0%** | **2.94** |
| Yang et al. | MS-COCO | 24.18 | 32.0% | 2 |
| **WMCopier** | MS-COCO | **32.93** | **100.0%** | **2.97** |

### Ablation Study

| Configuration | PSNR | Bit Acc | Notes |
|------|:---:|:---:|------|
| Full-step inversion ($T_S=T$) | Low | High | Semantic content severely degraded |
| Shallow inversion ($T_S=40$) | Medium | Medium-High | Minor artifacts |
| Shallow inversion + Refinement | **High** | **Highest** | Artifacts eliminated; best quality and performance |
| Refinement iterations $L$: 0→100 | Gradual↑ | Gradual↑ | Saturates after $L=100$ |
| Trade-off coefficient $\lambda$: ↑ | PSNR↑ | Bit Acc slightly↓ | Excessive regularization reduces forging rate |

### Key Findings

- WMCopier in the no-box setting even **surpasses the black-box attack** (Wang et al.) in forging success rate: 83.71% vs. 76.64%.
- Near 100% attack success rate against the Amazon commercial system, with confidence approaching the maximum level (2.94/3).
- Forged bit accuracy reaches 99.34% with FPR of 95.9% on the HiddeN scheme.
- Forged watermarks exhibit slightly lower robustness than genuine ones (10–20% degradation in some scenarios), but genuine and forged watermarks cannot be effectively distinguished by bit accuracy alone.
- **Multi-message defense is effective**: When service providers randomly select from $K=50$ or $K=100$ watermark messages, WMCopier's FPR drops to 0%, and increasing training data cannot overcome this defense.

## Highlights & Insights

- The paper cleverly leverages the distribution learning capacity of diffusion models to "copy" watermark signals—an intuitive and effective approach.
- The shallow inversion strategy precisely exploits the property that watermark signals are primarily injected/recovered in shallow denoising steps.
- Using the score function of the diffusion model as an approximation of the watermark distribution during refinement is theoretically well-grounded.
- The authors attacked Amazon's real deployed system and responsibly disclosed findings (with an official Amazon statement), exemplifying responsible security research.
- The proposed multi-message defense strategy is simple yet effective, offering a practical countermeasure for industry deployment.

## Limitations & Future Work

- Training the diffusion model requires collecting 5,000 watermarked images, which may be non-trivial to obtain.
- The attack assumes a static watermarking scheme; dynamically updated schemes may be harder to circumvent.
- Forged watermarks exhibit lower robustness than genuine ones under strong perturbations and may be selectively detected.
- The multi-message defense completely neutralizes the current attack, indicating that the attack's applicability is limited.
- The attack's effectiveness against semantic watermarks (e.g., Tree-Ring) is not explored.

## Related Work & Insights

- WMCopier forms a "spear and shield" relationship with T2SMark (another paper in this batch): T2SMark designs more robust watermarks, while WMCopier investigates forging attacks.
- This work enriches the research line of "diffusion models as attack tools": prior work has used diffusion models to remove watermarks, and WMCopier is the first to use them for forging.
- The effectiveness of the multi-message defense inspires watermarking system designs to incorporate built-in anti-forging mechanisms.
- The general framework of using generative models to learn and replicate covert signals is extensible to broader security domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of diffusion models to no-box watermark forging; the shallow inversion + refinement design is novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 open-source schemes + 1 commercial system, 4 datasets, multiple ablation studies and robustness analyses, with a proposed defense.
- **Writing Quality**: ⭐⭐⭐⭐ Paper structure is clear and the threat model is rigorously defined, though some theoretical derivations could be elaborated further.
- **Value**: ⭐⭐⭐⭐⭐ Raises important security concerns for watermarking systems while responsibly proposing defense strategies, with significant academic and industrial impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)

</div>

<!-- RELATED:END -->
