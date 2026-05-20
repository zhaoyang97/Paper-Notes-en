---
title: >-
  [Paper Note] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks
description: >-
  [ICCV 2025][Multimodal VLM][VLM safety] This work identifies a pervasive vulnerability of mainstream VLMs to Gaussian noise, proposes the Robust-VLGuard safety dataset (covering both image-text aligned and misaligned sce…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VLM safety"
  - "Gaussian noise"
  - "adversarial defense"
  - "diffusion purification"
  - "safety fine-tuning"
date: 2026-05-08
content_hash: fec437232ebfbef3
---

# Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

**Conference**: ICCV 2025
**arXiv**: [2504.01308](https://arxiv.org/abs/2504.01308)  
**Code**: [https://github.com/JarvisUSTC/DiffPure-RobustVLM](https://github.com/JarvisUSTC/DiffPure-RobustVLM)  
**Area**: Multimodal VLM
**Keywords**: VLM safety, Gaussian noise, adversarial defense, diffusion purification, safety fine-tuning

## TL;DR
This work identifies a pervasive vulnerability of mainstream VLMs to Gaussian noise, proposes the Robust-VLGuard safety dataset (covering both image-text aligned and misaligned scenarios) with noise-augmented fine-tuning to improve Gaussian noise robustness, and combines it with DiffPure to convert adversarial noise into Gaussian-like noise, forming the DiffPure-VLM general defense framework that effectively resists adversarial attacks of varying strengths.

## Background & Motivation
VLMs extend the capabilities of LLMs by integrating visual and textual information, yet they face more severe safety challenges than text-only LLMs:

**Visual modality introduces new attack surfaces**: Traditional optimization-based adversarial attacks (e.g., PGD) can inject imperceptible perturbations to jailbreak VLMs.

**A critical blind spot — Gaussian noise vulnerability**: The authors identify a neglected fundamental issue — **mainstream VLMs (InternVL2, LLaVA, MiniGPT-4) lack noise augmentation during training**, causing even simple Gaussian noise to simultaneously degrade both utility and safety.

Concrete manifestations:
- **Utility degradation**: InternVL2's performance on MM-Vet drops from 59.9% to 54.4% after adding Gaussian noise.
- **Safety alignment collapse**: InternVL2's attack success rate on RealToxicityPrompts rises from 50.5% to 57.2% with only $\sigma=0.1$ Gaussian noise.

**Key Challenge**: Existing defense methods (e.g., VLGuard) focus solely on content-level safety data and overlook the fact that **noise perturbations alone can undermine safety alignment**. Although DiffPure can purify adversarial noise, it does not eliminate it entirely but instead converts it into a Gaussian distribution — which is naturally complementary to noise-augmented fine-tuning.

**Core Idea**: A **two-stage defense** — first immunize the VLM against Gaussian noise through noise-augmented safety fine-tuning, then leverage DiffPure to convert arbitrary adversarial perturbations into Gaussian-like noise, thereby defending against a broad spectrum of attacks.

## Method

### Overall Architecture
DiffPure-VLM consists of two cascaded components:
1. **Front-end**: DiffPure diffusion model preprocessing (converting adversarial noise → Gaussian-like noise).
2. **Back-end**: A robust VLM fine-tuned with noise-augmented training on Robust-VLGuard (immune to Gaussian noise).

### Key Designs
1. **Robust-VLGuard Dataset**:

    - **Function**: Constructs a multimodal safety fine-tuning dataset comprising three categories of data.
    - **Data composition**:
        - **General instruction data** (4,467 samples): Covers QA, knowledge, mathematics, OCR, spatial reasoning, etc., with GPT-4V-refined annotations (original annotations were too brief to be effective for learning).
        - **Image-text aligned safety data** (1,000 samples): Sourced from VLGuard, where image content aligns with safety-related text.
        - **Image-text misaligned safety data** (1,000 samples): Scenarios where images are irrelevant to the safety-related text — the key innovation.
    - **Design Motivation**: Fine-tuning VLMs for visual tasks inherently degrades the safety alignment of the pretrained LLM backbone, and adversarial perturbations are injected independently of the text prompt; thus, the model must learn to handle image-text mismatched scenarios.

2. **Noise-Augmented Safety Fine-Tuning**:

    - **Function**: Fine-tunes the visual encoder with LoRA on Robust-VLGuard, applying Gaussian noise with random standard deviation ($\sigma \in [0.01, 0.15]$) to images with 70% probability during training.
    - **Mechanism**: Only LoRA parameters of the visual encoder are fine-tuned; training runs for 3 epochs (~3 hours on a single A100 GPU).
    - **Design Motivation**: By training the model to produce correct safety-aligned responses on noisy images, the approach instills inherent robustness to noise perturbations.
    - **Key Ablation**: A general-to-safety data ratio of 4:2 is optimal; increasing epochs has negligible impact on utility but consistently reduces attack success rates.

3. **Distribution-Shifting Property of DiffPure**:

    - **Function**: Leverages the forward-reverse process of diffusion models to purify adversarial images.
    - **Core Finding**: At an appropriate timestep $t^* \in [50, 150]$, DiffPure does not fully denoise but shifts the distribution of adversarial noise from non-Gaussian to Gaussian-like.
    - **Quantitative Validation**:
        - **Kurtosis**: Gaussian distribution has kurtosis 3; residual noise after DiffPure falls in the interval $[3, 6]$.
        - **Q-Q deviation**: RMSE between the DiffPure-processed residuals and the theoretical Gaussian distribution is $\leq 0.01$.
    - **Design Motivation**: Applying DiffPure directly to unmodified VLMs fails to reduce attack success rates (since the models are inherently non-robust to Gaussian noise), but the effect is significant when combined with noise-augmented fine-tuned VLMs.

### Loss & Training
- LoRA fine-tunes the visual encoder; all other parameters are frozen.
- Standard instruction fine-tuning loss (next-token prediction).
- DiffPure uses an unconditional diffusion model with timestep $t^*$ set to 50 or 150.

## Key Experimental Results

### Main Results

| Model | Configuration | MM-Vet (↑) | Attack Success Rate (↓) | Notes |
|-------|--------------|-----------|------------------------|-------|
| InternVL2-8B | Original | 59.9% | 50.5% | Baseline |
| +VLGuard | Fine-tuned | 42.9% (−17.0) | 27.7% | Severe utility degradation |
| +RobustVLGuard | Fine-tuned | **56.2%** (−3.7) | **29.9%** | Comparable safety, far better utility |
| LLaVA-v1.5-7B | Original | 33.0% | 57.7% | — |
| +RobustVLGuard | Fine-tuned | 30.3% (−2.7) | **43.6%** | Substantially improved safety |
| MiniGPT-4-13B | Original | 26.7% | 34.8% | — |
| +RobustVLGuard | Fine-tuned | **26.9%** (+0.2) | **16.0%** | Utility maintained or improved |

### Ablation Study / DiffPure-VLM Defense Effectiveness

| Image Type ($\varepsilon=32/255$) | InternVL2 Attack Rate | LLaVA Attack Rate | MiniGPT-4 Attack Rate |
|-----------------------------------|-----------------------|-------------------|-----------------------|
| Clean images | 29.9% | 43.6% | 16.0% |
| Gaussian-noisy images | 34.5% | 42.3% | 16.5% |
| Adversarial images | **70.6%** | **62.5%** | **53.7%** |
| +DiffPure-VLM ($t^*=50$) | 33.4% | 43.9% | **13.6%** |
| +DiffPure-VLM ($t^*=150$) | **32.8%** | **42.5%** | 11.9% |

| Defense Method (LLaVA, $\varepsilon=32/255$) | Attack Success Rate | Notes |
|----------------------------------------------|---------------------|-------|
| No defense (VLGuard) | 70.4% | Baseline |
| JailGuard + VLGuard | 52.1% | Detection-based defense |
| DiffPure + VLGuard | 51.1% | Purification-based defense |
| JailGuard + RobustVLGuard | 48.9% | — |
| DiffPure + RobustVLGuard | **43.9%** | Best combination |

### Key Findings
- **Gaussian noise is a severely underestimated threat**: Random noise with only $\sigma=0.1$ raises InternVL2's attack success rate by 6.7 percentage points.
- **VLGuard's "over-defense" problem**: InternVL2-VLGuard's MM-Vet score plummets from 59.9% to 42.9%, indicating that indiscriminate addition of safety data substantially sacrifices utility.
- **Importance of image-text misaligned data**: VLGuard contains only aligned safety data and shows limited improvement on RealToxicityPrompts (which involves image-text misaligned attacks).
- **DiffPure-VLM nearly fully recovers the clean baseline**: Attack rates under adversarial inputs can be reduced to near clean-image levels (e.g., MiniGPT-4: 13.6% vs. 16.0%).
- The key role of DiffPure is not "denoising" but "noise redistribution" — converting non-Gaussian adversarial noise into Gaussian noise.

## Highlights & Insights
1. **Discovery and solution in tandem**: The paper is the first to systematically demonstrate Gaussian noise vulnerability in mainstream VLMs and provides a complete defense framework in response.
2. **Clever exploitation of distribution shifting**: DiffPure's incomplete denoising is an advantage rather than a limitation — it unifies adversarial noise of unknown distribution into a known Gaussian distribution, which the fine-tuned model can readily handle, forming a closed defense loop.
3. **Thoughtful data design**: Image-text misaligned safety data addresses the blind spot of VLGuard; a small amount of high-quality data suffices to significantly improve robustness.
4. **Strong practicality**: Training requires only 3 hours on a single A100 GPU, and DiffPure inference is less costly than JailGuard (no repeated model inference needed).

## Limitations & Future Work
1. Evaluation is limited to three VLMs; newer models (e.g., Qwen2-VL, LLaMA-3.2-Vision) are not tested — the appendix provides preliminary results but lacks depth.
2. Noise augmentation is applied only during fine-tuning; integrating it into pretraining may yield further gains.
3. The DiffPure timestep $t^*$ must be chosen according to attack strength; an adaptive timestep selection mechanism could be explored.
4. The safety dataset is relatively small (6,467 samples in total); extending to more tasks and safety categories may further improve performance.

## Related Work & Insights
- **Relationship to VLGuard**: Robust-VLGuard extends VLGuard by incorporating image-text misaligned scenarios and noise-augmented training, essentially integrating robustness as a dimension of safety.
- **Relationship to the original DiffPure**: The original DiffPure targets image classifiers; this work adapts it to the VLM setting and discovers its distinctive property of "distribution shifting" rather than "denoising."
- **Insight**: VLM safety is not solely a content-level issue — **signal-level noise can also break safety alignment** — suggesting that safety training should consider a broader attack surface.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of Gaussian noise vulnerability is novel, and the DiffPure-VLM compositional design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three base models, multiple attack strengths, and detailed ablations, though a wider variety of VLMs would strengthen the results.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, progressing coherently from problem identification to solution.
- Value: ⭐⭐⭐⭐ Reveals an important blind spot in VLM safety; the defense framework is practical and generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models](one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)
- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out_of_distribution_detection.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](../../AAAI2026/multimodal_vlm/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](../../CVPR2026/multimodal_vlm/llavashield_multimodal_multiturn_safety.md)
- [\[ICCV 2025\] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)

</div>

<!-- RELATED:END -->
