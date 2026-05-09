---
title: >-
  [Paper Note] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves
description: >-
  [ICCV 2025][Multimodal VLM][VLM safety] This paper proposes IDEATOR, a framework that leverages VLMs themselves as red-team models to autonomously generate multimodal jailbreak image-text pairs, achieving a 94% attack success rate against MiniGPT-4's safety mechanisms. Based on this framework, the authors construct VLJailbreakBench, a safety evaluation benchmark comprising 3,654 samples.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLM safety
  - jailbreak attacks
  - red-teaming
  - multimodal safety benchmark
  - adversarial image-text pairs
date: 2026-05-08
content_hash: ba98c1cab50c06f5
---

# IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves

**Conference**: ICCV 2025
**arXiv**: [2411.00827](https://arxiv.org/abs/2411.00827)
**Code**: [https://github.com/roywang021/IDEATOR](https://github.com/roywang021/IDEATOR)
**Area**: Multimodal VLM
**Keywords**: VLM safety, jailbreak attacks, red-teaming, multimodal safety benchmark, adversarial image-text pairs

## TL;DR

This paper proposes IDEATOR, a framework that leverages VLMs themselves as red-team models to autonomously generate multimodal jailbreak image-text pairs, achieving a 94% attack success rate against MiniGPT-4's safety mechanisms. Based on this framework, the authors construct VLJailbreakBench, a safety evaluation benchmark comprising 3,654 samples.

## Background & Motivation

As large vision-language models (VLMs) such as GPT-4o and Gemini become widely deployed, ensuring their safe operation is of critical importance. VLM safety is threatened by jailbreak attacks, in which adversaries craft inputs to elicit harmful outputs from the model.

Existing VLM jailbreak methods suffer from three notable limitations: (1) **Reliance on white-box access**: Methods such as GCG and VAJM require model gradient information to optimize adversarial perturbations, making them infeasible in real-world deployment scenarios. (2) **Lack of semantic naturalness**: Generated adversarial images are typically meaningless noise patterns that are readily detected by safety mechanisms. (3) **Dependence on manual design**: Methods such as MM-SafetyBench require hand-crafted attack pipelines (e.g., typographic attacks combined with contextually related images), limiting flexibility and scalability.

The root cause of these shortcomings lies in a fundamental tension: existing approaches either require white-box access but are unrealistic in practice, or operate in a black-box setting but offer limited effectiveness and diversity. The **core insight of IDEATOR** is that VLMs themselves are natural red-team models — equipped with the ability to understand both vision and text, they can autonomously generate semantically rich, contextually relevant multimodal attack samples, offering greater flexibility and effectiveness than manual design or pure gradient-based optimization.

## Method

### Overall Architecture

IDEATOR is a black-box, training-free, end-to-end jailbreak attack framework. It consists of an attacker VLM $\mathcal{M}_\mathcal{A}$ (e.g., MiniGPT-4 Vicuna-13B) and a diffusion model (Stable Diffusion 3). The attacker VLM analyzes the target VLM's responses and generates structured JSON outputs containing an analysis, an image prompt, and a text prompt. The diffusion model then synthesizes the corresponding jailbreak image from the image prompt.

### Key Designs

#### 1. Iterative Multi-Turn Attack

- **Function**: Simulates a multi-turn dialogue between an adversarial user and the target VLM, progressively refining the attack strategy.
- **Mechanism**: In the first round, the attacker VLM receives only the jailbreak goal $\mathcal{G}$ and generates an initial image-text prompt. In subsequent rounds, the attacker receives the target VLM's previous response $\mathcal{R}_{n-1}$ and the previously generated image $I_{n-1}$, analyzes the reason for failure, and produces an improved attack prompt: $\mathcal{O}_{\text{json}}^{(n)} = \mathcal{M}_\mathcal{A}(I_{n-1}, \mathcal{R}_{n-1}) = \{\mathcal{A}_n, P_t^{(n)}, P_i^{(n)}\}$
- **Design Motivation**: Attacking a target VLM is an iterative adversarial game — an initial attempt may be refused, but by analyzing the refusal (e.g., identifying which keywords triggered the safety filter), the attacker can iteratively adjust its strategy, such as embedding harmful content within images or bypassing safeguards through role-playing.

#### 2. Breadth-Depth Exploration Strategy

- **Function**: Balances exploration across multiple independent attack trajectories (breadth $N_b$) and iterative refinement within each trajectory (depth $N_d$).
- **Mechanism**: $N_b$ parallel attack streams are launched, each independently undergoing $N_d$ rounds of iterative optimization. This allows each stream to explore distinct attack strategies (e.g., emotional manipulation, role-playing, cartoonization), avoiding local optima caused by over-reliance on a single approach. Experiments use $N_b=7, N_d=3$, yielding 21 total attempts.
- **Design Motivation**: VLM safety mechanisms may exhibit different vulnerabilities to different attack types. Breadth exploration enables discovery of a broader range of vulnerabilities, while depth exploration optimizes each strategy to its full potential.

#### 3. Prompt Engineering and Chain-of-Thought

- **Function**: Constrains the behavior of the attacker VLM through system prompts and a structured JSON output template.
- **Mechanism**: The system prompt configures the attacker VLM as a red-team assistant and constrains its output to a JSON format containing `analysis`, `image_prompt`, and `text_prompt` fields. The `analysis` field implements chain-of-thought reasoning — explicitly analyzing the reason for the previous round's failure and proposing improvements. The attacker VLM's response is initialized with `{"analysis":"` to enforce format compliance.
- **Design Motivation**: Vicuna is selected over LLaMA as the attacker backbone due to its more permissive stance toward generating adversarial content. The open-source nature of MiniGPT-4 allows customization of system prompts, enabling fine-grained control over attack behavior.

### Loss & Training

IDEATOR is a fully training-free framework that requires no optimization process. Attack effectiveness relies entirely on prompt engineering and the intrinsic reasoning capabilities of the attacker VLM.

## Key Experimental Results

### Main Results — Attack Effectiveness

| Method | Black-Box | Training-Free | ASR (%) |
|--------|-----------|---------------|---------|
| GCG (white-box text) | ✗ | ✗ | 50.0 |
| GCG-V (white-box visual) | ✗ | ✗ | 85.0 |
| VAJM (white-box image) | ✗ | ✗ | 68.0 |
| UMK (white-box dual-modal) | ✗ | ✗ | 94.0 |
| MM-SafetyBench (black-box) | ✓ | ✓ | 66.0 |
| **IDEATOR (black-box)** | **✓** | **✓** | **94.0** |

As a black-box method, IDEATOR achieves an attack success rate of 94%, matching the best white-box method UMK, and substantially outperforming the black-box baseline MM-SafetyBench by +28%.

### Ablation Study — Exploration Strategy and Modality Analysis

| Configuration | ASR (%) | Avg. Queries |
|---------------|---------|--------------|
| $N_b=1, N_d=1$ | 45.0 | — |
| $N_b=7, N_d=1$ | 85.0 | — |
| $N_b=7, N_d=3$ | **94.0** | — |
| Adversarial image only (Adv Img) | 85.0 | 5.84 |
| Adversarial text only (Adv Text) | 86.0 | 7.46 |
| Combined image + text (Adv I+T) | **94.0** | **5.34** |

Cross-model transferability: jailbreak samples generated on MiniGPT-4 transfer directly to LLaVA (82%), InstructBLIP (88%), and Chameleon (75%), far exceeding the transfer performance of MM-SafetyBench (46% / 29% / 22%).

### Key Findings

- **Commercial models are also vulnerable**: Evaluation on the VLJailbreakBench challenge set reveals that GPT-4o Mini achieves an ASR of 72.21%, Gemini-2.0-Flash 66.84%, and even the most safety-conscious Claude-3.5-Sonnet exhibits a 19.65% ASR.
- **Combined image-text attacks are the most effective**: Simultaneously exploiting both text and image modalities requires fewer queries and achieves higher success rates than unimodal attacks.
- **Image attacks are more effective at eliciting crime-related content** (where text is more likely to be blocked), while **text attacks are more effective for hate speech and self-harm content** (where images perform more weakly), demonstrating strong complementarity between modalities.

## Highlights & Insights

- **Using the shield as the spear**: Employing VLMs to attack VLMs is an elegant approach that avoids the dependence of traditional adversarial attacks on gradients and white-box access.
- **Systematic design of VLJailbreakBench**: The benchmark covers 12 safety topics, 46 subcategories, 916 queries, and 3,654 jailbreak samples, with a two-tier difficulty design (base/challenge), making it the most comprehensive multimodal jailbreak benchmark to date.
- **Practical impact**: The findings expose the fragility of safety alignment in current mainstream VLMs, demonstrating that even commercially deployed models aligned with RLHF can be systematically compromised.

## Limitations & Future Work

- The choice of attacker model is constrained by its alignment level — an overly safe model cannot serve as an effective attacker, while an insufficiently capable model produces weak attacks, revealing a fundamental capability-alignment trade-off.
- VLJailbreakBench remains limited in scale (3,654 samples); expanding it would require additional computational resources and automated filtering methods.
- Generated jailbreak images depend on the capabilities of Stable Diffusion; safety filters built into image generation models may restrict the production of certain adversarial images.
- Attack success rate (ASR) is used as the primary evaluation metric, without a graded assessment of the severity of the generated harmful content.

## Related Work & Insights

- The framework shares conceptual lineage with LLM red-teaming work (e.g., Chao et al.'s 20-query jailbreak) but extends the paradigm to the multimodal domain.
- The findings directly motivate VLM safety alignment research — there is a clear need to develop defense mechanisms that are robust against multimodal attacks, not merely text-based ones.
- Defensive frameworks such as AdaShield can be evaluated for their effectiveness on this benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ICCV 2025\] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks](safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)
- [\[ICCV 2025\] Growing a Twig to Accelerate Large Vision-Language Models](growing_a_twig_to_accelerate_large_vision-language_models.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)

<!-- RELATED:END -->
