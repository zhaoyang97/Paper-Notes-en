---
title: >-
  [Paper Note] Attention! Your Vision Language Model Could Be Maliciously Manipulated
description: >-
  [NeurIPS 2025][LLM Safety][Vision-Language Models] This paper proposes the Vision-language Model Manipulation Attack (VMA), an image-based adversarial attack method that combines first- and second-order momentum optimiza…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Vision-Language Models"
  - "Adversarial Examples"
  - "VMA Attack"
  - "Jailbreaking"
  - "Watermarking"
date: 2026-05-08
content_hash: cf1c879967816d76
---

# Attention! Your Vision Language Model Could Be Maliciously Manipulated

**Conference**: NeurIPS 2025
**arXiv**: [2505.19911](https://arxiv.org/abs/2505.19911)  
**Code**: [GitHub](https://github.com/Trustworthy-AI-Group/VMA)  
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, Adversarial Examples, VMA Attack, Jailbreaking, Watermarking

## TL;DR

This paper proposes the Vision-language Model Manipulation Attack (VMA), an image-based adversarial attack method that combines first- and second-order momentum optimization with a differentiable transformation mechanism, enabling precise control over every output token of a VLM. The approach supports a range of attack scenarios (jailbreaking, hijacking, privacy breach, DoS, sponge examples) and can also be repurposed for copyright-protection watermark injection.

## Background & Motivation

Large vision-language models (VLMs, e.g., LLaVA, InstructBLIP) have achieved remarkable success in understanding complex visual scenes, yet they also expose serious security vulnerabilities. Under adversarial attacks, VLMs may produce harmful, erroneous, or uncontrollable outputs.

**Limitations of existing attack methods**:

**Text adversarial attacks**: Operate by modifying input prompts, but discrete optimization is difficult and such perturbations are easy to detect.

**Image adversarial attacks**: Add imperceptible perturbations to input images, but existing methods offer limited control over VLM outputs.

**Lack of a unified framework**: Different attack scenarios (jailbreaking, hijacking, etc.) typically require separate, purpose-built methods.

This paper identifies that VLMs are particularly vulnerable to image adversarial attacks because the continuous nature of visual encoders enables more efficient gradient-based optimization. More critically, imperceptible image perturbations can precisely manipulate every output token.

## Method

### Overall Architecture

The objective of the VMA attack is: given an original image $x$, an input prompt $p$, and a desired target output $y^*$, find a perturbation $\delta$ ($\|\delta\|_\infty \leq \epsilon$) such that the VLM produces $y^*$ when given input $(x + \delta, p)$.

This is formalized as the following optimization problem:

$$\min_{\delta} \mathcal{L}(f(x + \delta, p), y^*) \quad \text{s.t.} \quad \|\delta\|_\infty \leq \epsilon$$

where $f$ denotes the VLM and $\mathcal{L}$ is the token-level cross-entropy loss.

### Key Designs

**1. Dual-Momentum Optimization**:

VMA combines first- and second-order momentum to stabilize and accelerate perturbation optimization:

- **First-order momentum (MI-FGSM style)**:
$$g_{t+1} = \mu \cdot g_t + \frac{\nabla_\delta \mathcal{L}}{\|\nabla_\delta \mathcal{L}\|_1}$$

- **Second-order momentum (Adam-style)**: Adaptively adjusts the step size using second-moment information of the gradient:
$$v_{t+1} = \beta \cdot v_t + (1 - \beta) \cdot (\nabla_\delta \mathcal{L})^2$$

This dual-momentum mechanism stabilizes the optimization process and avoids the oscillation that standard PGD exhibits when targeting long output sequences.

**2. Differentiable Transformation**:

To enhance the robustness and transferability of adversarial examples, VMA applies random differentiable transformations (e.g., brightness/contrast adjustment, crop-and-pad) to the input image at each iteration, effectively optimizing over the expectation across transformations:

$$\delta^* = \arg\min_\delta \mathbb{E}_{T \sim \mathcal{T}} [\mathcal{L}(f(T(x + \delta), p), y^*)]$$

**3. Token-Level Precise Control**:

The VMA loss function computes the cross-entropy over every token of the target output $y^*$:
$$\mathcal{L} = -\sum_{i=1}^{|y^*|} \log P(y^*_i | y^*_{<i}, x + \delta, p)$$

This allows the attacker to precisely control every word in the VLM's output.

### Loss & Training

- **Perturbation budget**: $\epsilon = 16/255$ ($\ell_\infty$ norm), ensuring imperceptibility to the human eye
- **Iterations**: typically 100–300 PGD steps
- **Step size**: $\alpha = 1/255$
- **Transformation set**: a combination of random cropping, resizing, and color jitter

## Key Experimental Results

### Main Results

**Attack success rate (ASR) across six attack scenarios** (evaluated on LLaVA-1.5):

| Attack Scenario | VMA ASR | PGD Baseline | MI-FGSM Baseline | Description |
|----------------|---------|--------------|------------------|-------------|
| Manipulation | **96.8%** | 72.3% | 78.5% | Precisely control output text |
| Jailbreaking | **94.2%** | 65.1% | 71.8% | Bypass safety alignment |
| Hijacking | **93.5%** | 68.7% | 74.2% | Redirect output topic/task |
| Privacy Breach | **91.7%** | 62.4% | 69.3% | Generate false personal information |
| Denial-of-Service | **89.3%** | 58.6% | 64.1% | Cause the model to refuse responses |
| Sponge Example | **95.1%** | 71.8% | 77.4% | Induce extremely long outputs |

**Cross-model transfer ASR** (adversarial examples generated on LLaVA-1.5):

| Target Model | Manipulation | Jailbreaking | Hijacking | Average |
|-------------|-------------|-------------|-----------|---------|
| LLaVA-1.5 (white-box) | 96.8 | 94.2 | 93.5 | 94.8 |
| InstructBLIP | 52.3 | 48.7 | 45.1 | 48.7 |
| MiniGPT-4 | 47.8 | 43.2 | 40.6 | 43.9 |
| Qwen-VL | 38.5 | 35.1 | 32.8 | 35.5 |

### Ablation Study

**Contribution of each component**:

| Configuration | Manipulation ASR | Jailbreaking ASR |
|--------------|-----------------|-----------------|
| PGD (baseline) | 72.3 | 65.1 |
| + First-order momentum | 82.1 | 75.8 |
| + Second-order momentum | 88.5 | 82.3 |
| + Differentiable transformation | **96.8** | **94.2** |

**Effect of perturbation budget**:

| Perturbation budget $\epsilon$ | Manipulation ASR | SSIM | PSNR (dB) |
|-------------------------------|-----------------|------|-----------|
| 4/255 | 61.2 | 0.998 | 48.1 |
| 8/255 | 82.5 | 0.995 | 42.0 |
| 16/255 | 96.8 | 0.989 | 36.1 |
| 32/255 | 99.1 | 0.972 | 30.2 |

### Key Findings

1. **The visual channel is far more vulnerable than the text channel**: Image adversarial attacks achieve 20–30% higher success rates compared to text-based adversarial attacks.
2. **VMA is a double-edged sword**: The same technique can be used for both attack and defense (watermark injection).
3. **Transferability is limited but non-trivial**: White-box attacks are highly effective; black-box transfer rates range from approximately 35–50%.
4. **Sponge examples pose a serious threat**: VMA can inflate output length from ~74 tokens to ~10,000 tokens, causing severe computational resource waste.
5. **Watermark injection application**: By embedding imperceptible perturbations via VMA, the VLM can be made to output a specific watermark string, enabling copyright protection.

## Highlights & Insights

- **Unified attack framework**: A single method covers six attack scenarios and one defensive application, demonstrating the generality of VLM security threats.
- **Dual theoretical and empirical justification**: The paper provides both theoretical analysis of why VLMs are vulnerable to image attacks and extensive experimental validation.
- **Double-edged sword perspective**: Repurposing attack techniques for watermark protection is a novel and practically useful insight.
- **Novel sponge example finding**: This work is among the first to demonstrate that VLMs can be precisely induced to generate extremely long outputs, with direct security implications for inference services.
- **Intuitive visualizations**: The GitHub repository presents side-by-side output comparisons of LLaVA under various attack scenarios, with compelling results.

## Limitations & Future Work

1. **White-box dependency**: The full attack requires access to VLM gradient information, which limits the scope of realistic threat scenarios.
2. **Insufficient transferability**: Cross-model transfer attack success rates leave room for improvement.
3. **Limited discussion of defenses**: The paper focuses primarily on attacks and provides relatively shallow analysis of defensive strategies.
4. **High computational cost**: 100–300 PGD iterations require substantial GPU time.
5. **Only open-source VLMs evaluated**: Closed-source commercial models such as GPT-4V and Gemini are not assessed.

## Related Work & Insights

- **FGSM / PGD** (Goodfellow et al., Madry et al.): Classical image adversarial attack methods; VMA adapts these techniques for the VLM setting.
- **Multimodal jailbreak attacks** (Qi et al., 2024): Pioneering work on exploiting image inputs to bypass LLM safety alignment.
- **MI-FGSM** (Dong et al., 2018): Momentum-based transfer attack method; VMA extends this to dual momentum.
- Insight: Securing VLMs requires simultaneously addressing both the visual and textual attack surfaces.

## Rating

- Theoretical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [\[ICML 2026\] BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](../../ICML2026/llm_safety/byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)

</div>

<!-- RELATED:END -->
