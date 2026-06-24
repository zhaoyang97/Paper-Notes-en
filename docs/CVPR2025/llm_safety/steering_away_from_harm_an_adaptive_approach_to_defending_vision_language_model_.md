---
title: >-
  [Paper Note] Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks
description: >-
  [CVPR 2025][LLM Safety][VLM Jailbreak Defense] This paper proposes ASTRA, which locates jailbreak-related visual tokens in adversarial images via **image attribution**, constructs **steering vectors** representing harmful response directions, and performs **adaptive activation steering** during inference to steer the model away from harmful directions. It achieves SOTA defense performance, with a 12% lower toxicity score, 18% lower ASR, and 9x faster speed compared to JailGua…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "VLM Jailbreak Defense"
  - "Activation Steering"
  - "Image Attribution"
  - "Adversarial Attack"
  - "Inference-time Safety"
date: 2026-05-08
content_hash: 3f4fc0cdf7b5e789
---

# Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks

**Conference**: CVPR 2025  
**arXiv**: [2411.16721](https://arxiv.org/abs/2411.16721)  
**Code**: [https://github.com/ASTRAL-Group/ASTRA](https://github.com/ASTRAL-Group/ASTRA)  
**Area**: Multimodal VLM  
**Keywords**: VLM Jailbreak Defense, Activation Steering, Image Attribution, Adversarial Attack, Inference-time Safety

## TL;DR

This paper proposes ASTRA, which locates jailbreak-related visual tokens in adversarial images via **image attribution**, constructs **steering vectors** representing harmful response directions, and performs **adaptive activation steering** during inference to steer the model away from harmful directions. It achieves SOTA defense performance, with a 12% lower toxicity score, 18% lower ASR, and 9x faster speed compared to JailGuard.

## Background & Motivation

VLMs (e.g., MiniGPT-4, LLaVA, Qwen2-VL) face severe security threats when deployed: visual inputs provide a new attack surface for jailbreak attacks. Existing jailbreak attacks fall into two categories: (1) **perturbation-based attacks**—generating adversarial images via methods like PGD to bypass safety alignment; (2) **structural attacks**—embedding malicious text into images through typography.

Existing defense methods have critical limitations:
- **Input preprocessing/adversarial training**: requires substantial computational resources to purify images or fine-tune models, resulting in extremely high training costs.
- **Response-evaluation methods** (e.g., JailGuard, ECSO): require multiple inference steps to generate responses to judge whether they are harmful, incurring over 9x inference cost.
- **LLM activation steering methods**: directly transferring steering vectors from the textual domain to VLMs yields poor performance due to representation mismatches between textual and visual domains.

Key Challenge: **Trade-off between efficiency and effectiveness**—effective defenses either require high training costs or multiple generations during inference. Key Insight: Leverage activation steering, but address two critical problems: (1) how to construct effective steering vectors in the visual modality; (2) how to ensure steering has no impact on benign inputs while being highly effective against adversarial ones.

## Method

### Overall Architecture

ASTRA consists of two steps: (1) **Offline stage**—identifying jailbreak-related visual tokens in adversarial images via image attribution to construct steering vectors; (2) **Online inference stage**—adaptive activation steering, which conditionally modifies the activation direction based on the projection relationship between calibrated activations and steering vectors. The entire defense can be constructed with minimal inference and adds no inference time during deployment.

### Key Designs

1. **Image Attribution for Steering Vectors**
    - **Function**: Locates truly "toxic" visual tokens in adversarial images, extracting steering vectors that represent harmful response directions.
    - **Mechanism**: Randomly ablates visual tokens of PGD-generated adversarial images and observes the impact of each token's presence/absence on the jailbreak probability. Specific steps:
     - Randomly sample multiple ablation vectors $g \sim \{0,1\}^m$, and calculate the log-probability $f(g)$ of the model generating "Sure,..." (the jailbreak indicator) after each ablation.
     - Fit a linear surrogate model $\hat{f}$ using Lasso, where the weights represent the attribution scores of individual tokens.
     - Select the tokens with the top-$k$ attribution scores, pair them with empty query templates, and calculate the difference in activations with versus without these tokens as the steering vector $v^l$.
     - Average over multiple adversarial samples to obtain the final steering vector.
    - **Design Motivation**: Not all adversarial visual tokens are related to jailbreaks; precise localization of "toxic" tokens is required to build effective and transferable steering vectors. Directly using LLM steering vectors constructed from textual contrastive pairs yields poor results on VLMs, making a visual-domain approach essential.

2. **Adaptive Activation Steering**
    - **Function**: Adaptively decides whether and to what extent to perform steering during inference based on whether the input contains components in the harmful direction.
    - **Mechanism**: Traditional linear steering $h^l = h^l - \alpha \cdot v^l/\|v^l\|$ modifies all inputs indiscriminately, leading to severe performance degradation on benign inputs. ASTRA introduces conditional projection:
     $$h^l = h^l - \alpha \cdot \max\left(\frac{(h^l - h_0^l)^\top v^l}{\|h^l - h_0^l\| \|v^l\|} \cdot \|h^l\|, 0\right) \cdot \frac{v^l}{\|v^l\|}$$
     When the calibrated activation $h^l - h_0^l$ and the steering vector $v^l$ are in different directions, the $\max$ term is 0 and the activation remains unchanged; steering is performed according to the projection size only when their directions are aligned.
    - **Design Motivation**: The activations of benign inputs should not be modified. The adaptive effect of "strong steering for harmful inputs, zero impact for benign inputs" is realized through projection and the max operation.

3. **Activation Calibration**
    - **Function**: Solves the problem where activations of different inputs cluster in regions far from the origin, rendering angular differentiation ineffective.
    - **Mechanism**: A large number of test inputs are collected, and the mean activation of each layer $h_0^l$ is calculated as the calibration point. During projection, $(h^l - h_0^l)$ is used instead of $h^l$ to "center" the activations before calculating the angle with the steering vector.
    - **Design Motivation**: In the original activation space, vectors of different inputs may have similar angles (pointing in similar directions); calibration allows better differentiation between the directional differences of harmful and benign activations.

### Loss & Training

During the steering vector construction phase, Lasso regression (L1 regularization) is used to fit the surrogate model, encouraging sparse solutions to locate a few key tokens. The entire framework does not require fine-tuning the VLM itself, and the defense construction can be completed with a small amount of forward inference (dozens of ablations + forward passes).

## Key Experimental Results

### Main Results (Comparison of Perturbation-based Attack Defense on MiniGPT-4)

| Method | Toxicity Score ($\epsilon=16/255$) ↓ | Toxicity Score ($\epsilon=64/255$) ↓ | ASR ($\epsilon=16/255$) ↓ | ASR (unconstrained) ↓ |
|------|---------------------|---------------------|-----------------|---------------------|
| w/o defense | 39.73 | 54.70 | 44.55 | 53.64 |
| JailGuard | 16.51 | 20.93 | 30.00 | 28.18 |
| ECSO | 34.59 | 38.54 | 40.91 | 37.27 |
| **ASTRA** | **11.30** | **4.51** | **9.09** | **9.09** |

### Defense Effectiveness on Qwen2-VL

| Method | Toxicity Score ($\epsilon=64/255$) ↓ | ASR (unconstrained) ↓ |
|------|---------------------|---------------------|
| w/o defense | 55.59 | 76.36 |
| JailGuard | 28.74 | 15.45 |
| **ASTRA** | **2.39** | **15.45** |

### Comparison of Inference Efficiency (Inference Time per Token, MiniGPT-4)

| Method | Inference Time (ms) | Single Inference? |
|------|-------------|-------------|
| w/o defense | 173.19 | ✓ |
| JailGuard | 1557.98 (9x) | ✗ |
| ECSO | 457.55 (2.6x) | ✗ |
| **ASTRA** | **173.77** (~1x) | ✓ |

### OOD Transferability (MiniGPT-4, Using Steering Vector with $\epsilon=16/255$)

| Attack Type | w/o defense ASR | ASTRA ASR |
|---------|-----------------|-----------|
| SD (Structural) | 13.75 | 3.75 |
| TYPO (Typographic) | 43.75 | 11.25 |
| PGD $\epsilon=32/255$ | 78.18 | 12.73 |
| MI-FGSM $\epsilon=32/255$ | 79.09 | 13.64 |
| GCG (Text-only) | 58.18 | 9.09 |

### Resistance to Adaptive Attacks (Attacker Knows All Details of ASTRA)

| Model | Attack w/o Defense ASR ($\epsilon=16$) | Adaptive Attack w/ Defense ASR ($\epsilon=16$) |
|------|-------------------|------------------------|
| MiniGPT-4 | 44.55 | 13.64 |
| Qwen2-VL | 67.27 | 58.16 |

### Key Findings

1. ASTRA significantly outperforms all baselines under all perturbation strengths, displaying a more pronounced advantage under high perturbations (toxicity is only 4.51% when $\epsilon=64/255$).
2. Inference time barely increases (+0.3%), which is 1/9 of JailGuard's.
3. The steering vector exhibits good cross-attack transferability; vectors constructed using one type of attack can defend against structural, PGD variants, and even text-only attacks.
4. It remains remarkably effective even when facing adaptive attacks (where the attacker fully understands the defense).

## Highlights & Insights

1. **Image attribution is a key innovation**—precisely localizing "toxic" visual tokens via Lasso regression, instead of blindly using all adversarial signals, makes the steering vector more transferable.
2. **The adaptive steering design is elegant**—it transforms simple mathematical operations of projection and max into an adaptive mechanism of "zero impact on benign, strong intervention on malicious", which is far superior to traditional fixed-coefficient steering.
3. **High efficiency**—the defense construction requires only dozens of forward inferences (taking a few minutes), and adds no inference time post-deployment, making it highly suitable for real-world deployment.
4. **Excellent OOD transferability**—steering vectors constructed using PGD adversarial images can defend against typographic and text-only attacks, indicating that different attacks share harmful directions in the activation space.

## Limitations & Future Work

1. ASR remains relatively high (58%) against adaptive attacks on Qwen2-VL, indicating limited defense efficacy on stronger models.
2. Steering vectors need to be constructed specifically for each model and cannot be transferred across different models.
3. Activation calibration requires collecting an additional set of test data, which increases the preparation workload to some extent.
4. Although the defense against structural attacks (typography) is improved, it is less pronounced compared to perturbation-based attacks.

## Related Work & Insights

- **Activation Engineering (LLM)**: Rimsky et al. construct steering vectors through contrastive pairs, and Ball et al. study the representation mechanisms of different jailbreak templates. This work extends this approach to the visual domain of VLMs.
- **VLM Safety**: Different from JailGuard (multi-inference detection) and ECSO (generate-evaluate-regenerate), ASTRA is the first single-inference activation-layer defense method.
- **Insights**: The separability of harmful directions in the activation space provides a new perspective for VLM safety—ensuring safety does not necessarily require retraining models or performing multiple inferences; merely intervening in the representation space can suffice.

## Rating

⭐⭐⭐⭐ — The method is simple yet highly efficient with solid experiments (3 models $\times$ multiple attacks $\times$ transferability $\times$ adaptive attacks). Zero inference overhead is the primary selling point, though there is still room for improvement in robustness against strong adaptive attacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ICLR 2026\] VEAttack: Downstream-Agnostic Vision Encoder Attack Against Large Vision Language Models](../../ICLR2026/llm_safety/veattack_downstream-agnostic_vision_encoder_attack_against_large_vision_language.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](../../ACL2026/llm_safety/ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)

</div>

<!-- RELATED:END -->
