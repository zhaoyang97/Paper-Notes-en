---
title: >-
  [Paper Note] Towards Robust Multimodal Large Language Models Against Jailbreak Attacks
description: >-
  [CVPR 2026][LLM Safety][Paper Note] SAFEMLLM is the first adversarial training framework designed specifically for Multimodal Large Language Models (MLLMs). By injecting compact, learnable perturbation matrices into the token embedding layer to simulate cross-modal attacks (CoE-Attack) and iteratively updating model parameters to neutralize these perturb
tags:
  - CVPR 2026
  - LLM Safety
date: 2026-05-08
content_hash: 663b0ac807272ecc
---
# Towards Robust Multimodal Large Language Models Against Jailbreak Attacks

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yin_Towards_Robust_Multimodal_Large_Language_Models_Against_Jailbreak_Attacks_CVPR_2026_paper.html)  
**Code**: https://github.com/ericyinyzy/SafeMLLM (Claimed to be open-sourced)  
**Area**: AI Safety / Multimodal Large Language Model Jailbreak Defense  
**Keywords**: Jailbreak Defense, Adversarial Training, MLLM, Token Embedding Perturbation, Contrastive Loss

## TL;DR
SAFEMLLM is the first adversarial training framework designed specifically for Multimodal Large Language Models (MLLMs). By injecting compact, learnable perturbation matrices into the token embedding layer to simulate cross-modal attacks (CoE-Attack) and iteratively updating model parameters to neutralize these perturbations, SAFEMLLM reduces the Attack Success Rate (ASR) of six jailbreak methods to near 0 in white-box scenarios while maintaining standard multimodal task performance.

## Background & Motivation

**Background**: MLLMs excel in tasks like VQA and image-text understanding but inherit and amplify safety risks from LLMs. Jailbreak attacks can bypass safety guardrails, inducing models to generate harmful content. Current defenses follow two paths: **Inference-time modules** (e.g., using external LLMs as detectors, steering decoding distributions via reward models, or adding classifiers to hidden states) and **Safety Alignment Fine-tuning** (e.g., fine-tuning on "harmful query → refusal" data or using RLHF).

**Limitations of Prior Work**: External modules rely on being **confidential** to users; if an attacker acquires the detection mechanism, it can be bypassed. Furthermore, external modules only intercept at the output stage without inherently improving the model's safety. Safety fine-tuning methods (like VLGuard) are fragile in **white-box scenarios**. Evaluations show that while VLGuard defends against black-box FigStep on LLaVA-1.5, it fails against gradient-based white-box attacks like ImgJP and GCG (ASR reaching 79–88%).

**Key Challenge**: To truly enhance **intrinsic** safety under white-box conditions (where attackers have parameter and gradient access), Adversarial Training (AT) is a natural choice. However, existing AT methods cannot be directly ported to MLLMs. AT for closed-set classification does not apply to open-ended generation. Extending Latent Adversarial Training (LAT) from LLMs to MLLMs faces two hurdles: (1) Text-only perturbations cannot stop stronger continuous-valued image noise; (2) Directly perturbing every token embedding is computationally prohibitive—a single image in LLaVA-1.5-13B occupies 576 tokens, making optimization both slow and weak.

**Goal**: Design an adversarial training framework that defends against image, text, and multimodal jailbreak attacks in white-box settings without sacrificing utility, particularly addressing the efficiency bottleneck caused by the high volume of image tokens in MLLMs.

**Core Idea**: Instead of perturbing pixels or individual tokens, the framework injects **two compact, learnable perturbation matrices** into the text embedding layer (one before the query to simulate adversarial images and one after to simulate adversarial text suffixes). By optimizing these under a contrastive objective to find the "worst-case" attack and iteratively updating the model to neutralize them, the framework uses 8 perturbation tokens to replace 576 image tokens, making both attack and defense faster and more potent.

## Method

### Overall Architecture
SAFEMLLM is a **two-step alternating** adversarial training framework. Given a benign MLLM with parameters $\theta$, the goal is to learn robust parameters $\theta^*$ such that both $\theta^*$ and its gradients remain robust even when exposed to attackers. Trainable parameters $\Delta\theta^*$ are derived from the cross-modal adapter and LLM decoder (optimized via LoRA), while the vision encoder remains frozen.

In each iteration $i$: **Step I (Attack)** fixes model parameters and optimizes the strongest adversarial perturbations $\{P^h_M, P^t_M\}$ at the token embedding layer using CoE-Attack. **Step II (Defense)** fixes these perturbations and updates model parameters to neutralize the attack effect while using a utility loss to preserve standard VQA capabilities. The updated model proceeds to the next iteration's Step I for $T$ rounds until $\theta^* = \theta_T$ is reached.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Benign MLLM θ (Frozen Vision Encoder, <br/>Adapter+Decoder tuned via LoRA)"] --> B["Step I: CoE-Attack Injection<br/>Ph placed before query (Image sim)<br/>Pt placed after query (Suffix sim)"]
    B --> C["Contrastive Attack Objective<br/>L_adv = L_target + λ·L_contra<br/>Gradient ascent for worst-case perturbation"]
    C -->|Fixed perturbations Ph_M, Pt_M| D["Step II: Defense Update<br/>L_def neutralizes perturbations + L_utility preserves utility"]
    D -->|Update θ_{i-1}→θ_i| B
    D --> E["Robust MLLM θ* (Public parameters & gradients)"]
```

### Key Designs

**1. Dual-Token Embedding Perturbation in CoE-Attack: 8 Tokens vs. 576 Image Tokens**

Direct worst-case attacks optimize an adversarial image $I'$ and a text suffix $x'$, which is computationally heavy. CoE-Attack observes that since images are processed before text and suffixes after text in MLLMs, one can **bypass the modalities themselves by injecting perturbation matrices** $P^h_0 \in \mathbb{R}^{K\times C}$ and $P^t_0 \in \mathbb{R}^{K\times C}$ (where $K$ is the number of tokens and $C$ is embedding dimension) directly into the text embedding layer. $P^h_0$ acts as the adversarial image and $P^t_0$ as the adversarial suffix. This **removes real $I'$ and $x'$** from the input, allowing direct gradient optimization on these matrices.

These matrices are re-initialized for each iteration from sampled malicious queries. This design significantly improves efficiency: for LLaVA-1.5-13B, SAFEMLLM uses only 8 perturbation tokens instead of 576, reducing iteration time for the attack from 263.56s (direct image optimization) or 192.39s (LAT) to **38.70s**, while maintaining ASR and minimizing VRAM usage.

**2. Contrastive Attack Objective: Promoting Affirmation and Suppressing Refusal**

Potent jailbreaks must both maximize the probability of an "affirmatory response $c_n$" (e.g., "Sure, here are the steps...") and minimize the probability of a "refusal response $r_n$". The target loss handles the former:

$$L_{\rm adv}^{\rm target} = -\sum_{n=1}^{N}\log\big[p(c_n \mid P^h_0, x_n, P^t_0)\big]$$

To avoid nonsensical outputs caused by directly punishing $\log p(r_n\mid\cdot)$, SAFEMLLM uses a **contrastive loss** to ensure $c_n$ is preferred over $r_n$ relatively:

$$L_{\rm adv}^{\rm contra} = -\sum_{n=1}^{N}\log\sigma\Big[\log p(c_n\mid P^h_0,x_n,P^t_0) - \log p(r_n\mid P^h_0,x_n,P^t_0)\Big]$$

Where $\sigma$ is the Sigmoid function. The final objective is $L_{\rm adv} = L_{\rm adv}^{\rm target} + \lambda\cdot L_{\rm adv}^{\rm contra}$. $c_n$ and $r_n$ are generated by GPT-4 Turbo with instructions for semantic diversity. Ablations show ASR increases by 13.67% without the contrastive loss.

**3. Step II Defense Update: Balancing Neutralization and Utility**

With fixed perturbations, Step II updates parameters to "neutralize perturbations + preserve utility." The defense loss mirrors the attack loss but targets the safety response $r_n$ given malicious inputs:

$$L_{\rm def} = L_{\rm def}^{\rm target} + \lambda\cdot L_{\rm def}^{\rm contra}$$

To prevent "over-refusal" where the model rejects benign questions, a **utility loss** is added using $H$ benign samples:

$$L_{\rm utility} = -\sum_{j=1}^{H}\log\big[p(y_j\mid I_j, q_j)\big]$$

Final updates use $L_{\rm def} + L_{\rm utility}$ on LoRA parameters. Without $L_{\rm utility}$, the MM-Vet score for LLaVA-1.5 drops from 37.8 to 21.6.

### Loss & Training
- **Attack Side**: $L_{\rm adv} = L_{\rm adv}^{\rm target} + \lambda L_{\rm adv}^{\rm contra}$ via $M$-step gradient ascent on perturbation matrices.
- **Defense Side**: $L_{\rm def} + L_{\rm utility}$ updates LoRA parameters (adapter + decoder) with frozen vision encoder.
- **T iterations** of the outer loop; $c_n/r_n$ utilize diverse templates.

## Key Experimental Results

### Main Results
Evaluated against six attacks (ImgJP/VAA, GCG/AutoDAN, FigStep/MM-SafetyBench) across six MLLMs using **ASR (%)** as the metric (determined by GPT-4 Turbo).

| Attack (Modality) | Original | R2D2 | CAT | SAFEMLLM |
|--------------|----------|------|-----|----------|
| ImgJP (Image) | 51.33 | 27.33 | 11.33 | **5.17** |
| VAA (Image) | 32.92 | 8.75 | 4.75 | **1.25** |
| GCG (Text) | 33.83 | 15.83 | 4.83 | **0.00** |
| AutoDAN (Text) | 66.67 | 32.33 | 22.33 | **1.33** |
| FigStep (Multimodal) | 38.00 | 20.33 | 19.33 | **1.00** |
| MM-SafetyBench (Multimodal) | 26.62 | 13.37 | 10.39 | **2.27** |

SAFEMLLM suppresses average ASR to single digits or zero across all attacks. Baselines like R2D2/CAT (standard LLM-AT) fail against multimodal attacks like FigStep because they only address text-based harmfulness.

Comparison with VLGuard (LLaVA-1.5):

| Attack | Model | VLGuard | SAFEMLLM |
|------|------|---------|----------|
| ImgJP | 7B / 13B | 88.00 / 36.00 | **6.00 / 0.00** |
| GCG | 7B / 13B | 79.00 / 26.00 | **0.00 / 0.00** |
| AutoDAN | 7B / 13B | 81.00 / 61.00 | **1.00 / 0.00** |
| FigStep | 7B / 13B | 2.00 / 0.00 | 0.00 / 0.00 |

VLGuard's high ASR under white-box gradient attacks highlights the necessity of active adversarial training over simple safety fine-tuning.

### Ablation Study
On 13B models under ImgJP / AdvBench ("×" denotes removal):

| Removed Module | MiniGPT-4 ASR↓ | LLaVA-1.5 ASR↓ | Note |
|------------|----------------|----------------|------|
| w/o $P^h_0$ | 5.00 | 1.00 | Prefix perturbations are critical for image-based attacks |
| w/o $P^t_0$ | 2.00 | 0.00 | Suffix perturbations |
| w/o $L^{\rm contra}_{\rm adv}$ | 8.00 | 0.00 | Attack potency decreases |
| w/o $L^{\rm contra}_{\rm def}$ | 23.00 | 0.00 | Primary source of robustness; ASR +13.67% average |
| **SAFEMLLM (Full)** | **0.00** | **0.00** | — |
| w/o $L_{\rm utility}$ (MM-Vet Score↑) | 7.2 | 21.6 | Utility collapses due to over-refusal |
| SAFEMLLM (MM-Vet Score↑)| 22.8 | 37.8 | Utility preserved |

### Key Findings
- **Contrastive loss (especially for defense) is the core of robustness**: ASR jumps from 0 to 23 on MiniGPT-4 without it.
- **Prefix perturbation $P^h_0$ is more significant than suffix $P^t_0$** because most current multimodal attacks inject noise prior to the query.
- **Utility loss is essential**: Removing it leads to nearly halved MM-Vet scores, indicating safety and utility must be optimized jointly.
- **Efficiency gains**: Using 8 perturbation tokens enables a 38.7s iteration time on 13B models vs. 263.6s for full image optimization.

## Highlights & Insights
- **Modality-Agnostic Perturbation**: Bypassing heavy vision encoders to optimize compact matrices at the embedding level is a clever solution to MLLM efficiency bottlenecks.
- **Symmetric Contrastive Loss**: Using the same objective structure to generate attacks (Step I) and train defenses (Step II) is elegant and effective.
- **Preference vs. Absolute Probability**: Using contrastive signals to steer the model away from harmfulness rather than "hard-suppressing" refusal probabilities prevents gibberish outputs.
- **True White-Box Setting**: Proving robustness even when both parameters and gradients are public provides a more credible safety guarantee than external detectors.

## Limitations & Future Work
- Validated only on six **known** attack types; generalization to novel multimodal jailbreaks remains a question.
- Reliance on GPT-4 Turbo for response generation and safety judging introduces dependency on closed-source APIs and potential bias.
- Requires model re-training/fine-tuning; not applicable to API-only closed-source MLLMs.
- Future work: adaptive perturbation placement, regularization for unseen attacks, and exploring lightweight evaluators to replace GPT-4.

## Related Work & Insights
- **vs. VLGuard (Fine-tuning)**: VLGuard is passive imitation; SAFEMLLM is active adversarial training. White-box attacks easily bypass the former but fail against the latter.
- **vs. R2D2 / CAT (LLM-AT)**: Standard LLM-AT ignores image/multimodal modalities; SAFEMLLM unifies them in the embedding space.
- **vs. LAT (Latent AT)**: Scaling LAT to MLLMs involves too many tokens and layers; SAFEMLLM's 8-token design is faster and produces stronger adversarial signals.
- **vs. Inference Detectors**: Detectors are often easily bypassed if their mechanisms are leaked; SAFEMLLM enhances intrinsic robustness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First direct adversarial training for MLLMs with a clever efficiency-focused perturbation design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive attack/model coverage and efficiency/ablation studies, though missing sensitivity analysis on some hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-structured symmetric attack-defense logic.
- Value: ⭐⭐⭐⭐ Provides a practical, re-trainable solution for securing open-source MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](../../ACL2026/llm_safety/gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models](towards_reasoning-preserving_unlearning_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)

</div>

<!-- RELATED:END -->
