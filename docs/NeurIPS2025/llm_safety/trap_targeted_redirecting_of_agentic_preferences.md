---
title: >-
  [Paper Note] TRAP: Targeted Redirecting of Agentic Preferences
description: >-
  [NeurIPS 2025][LLM Safety][adversarial attack] TRAP introduces a diffusion-based semantic injection adversarial framework that optimizes image semantics in the CLIP embedding space. Under black-box conditions, it systematically misdirects the decision preferences of multiple mainstream VLM agents in a visually natural manner, achieving attack success rates of up to 100% across six models including LLaVA-34B and GPT-4o.
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "adversarial attack"
  - "vision-language models"
  - "semantic injection"
  - "agentic safety"
  - "diffusion models"
date: 2026-05-08
content_hash: ab92d572ef3155ec
---

# TRAP: Targeted Redirecting of Agentic Preferences

**Conference**: NeurIPS 2025
**arXiv**: [2505.23518](https://arxiv.org/abs/2505.23518)  
**Code**: [https://github.com/uiuc-focal-lab/TRAP](https://github.com/uiuc-focal-lab/TRAP)  
**Area**: AI Safety
**Keywords**: adversarial attack, vision-language models, semantic injection, agentic safety, diffusion models

## TL;DR

TRAP introduces a diffusion-based semantic injection adversarial framework that optimizes image semantics in the CLIP embedding space. Under black-box conditions, it systematically misdirects the decision preferences of multiple mainstream VLM agents in a visually natural manner, achieving attack success rates of up to 100% across six models including LLaVA-34B and GPT-4o.

## Background & Motivation

As autonomous agent systems built upon vision-language models (VLMs) move toward real-world deployment, their cross-modal reasoning capabilities introduce new attack surfaces. Existing adversarial attack methods primarily rely on visible pixel perturbations or require privileged access to model internals or the deployment environment—neither of which is stealthy nor practical in realistic settings. Traditional pixel-level attacks (e.g., FGSM, PGD) focus on low-level noise injection, but modern multimodal systems exhibit considerable robustness to pixel noise; the true vulnerability lies in cross-modal alignment at the semantic level.

The root cause lies in the fact that autonomous agents inherently trust their perceptual inputs, and this trust can be exploited through subtle semantic-level manipulation. The paper's starting point is to leverage diffusion models for semantic optimization in the CLIP shared embedding space, generating visually natural yet semantically manipulated adversarial images that redirect agent selection preferences under black-box conditions.

Core idea: combining negative-prompt degradation with positive semantic optimization, employing a twin semantic network and spatial layout mask to perform semantic-level adversarial manipulation within the embedding space.

## Method

### Overall Architecture

TRAP operates in four stages: (1) extracting CLIP embeddings of the target image and adversarial prompts; (2) iteratively optimizing the image embedding in the embedding space using a twin semantic network and prompt-alignment guidance, together with a spatial layout mask; (3) applying perceptual and semantic losses to preserve image identity and photorealism; and (4) decoding the optimized embedding into the final adversarial image via the Stable Diffusion decoder.

The entire pipeline operates under fully black-box conditions—the attacker requires no access to the target model's weights, parameters, or gradients, and can only substitute images under their control while observing the agent's final selection.

### Key Designs

1. **Semantic Alignment Loss ($\mathcal{L}_{sem}$)**: High-level semantic concepts are injected into the image representation by minimizing the cosine distance between the adversarial embedding $e_{adv}$ and the positive prompt embedding $e_{pos}$. This exploits CLIP's joint embedding space, where semantically similar content is mapped to closer embeddings. Attacker-chosen positive prompts (e.g., "luxury," "premium quality") serve as semantic proxies that generalize across a range of user queries.

2. **Distinctive Feature Preservation Loss ($\mathcal{L}_{dist}$) + Twin Semantic Network**: Optimizing for semantic alignment alone causes the image to lose its distinctive identity. The twin network decomposes embeddings into a "common component" branch and a "distinctive component" branch. By penalizing changes in the distinctive branch, the optimizer is constrained to concentrate semantic modifications in the common branch, thereby injecting semantics while preserving image identity. This push-pull dynamic constitutes one of the core innovations of the method.

3. **Perceptual Similarity Loss ($\mathcal{L}_{LPIPS}$) + Spatial Layout Mask**: To ensure visual plausibility of the decoded image, a differentiable decoding pipeline is employed: a lightweight MLP encoder-decoder first generates a semantic layout mask $A$ from prompt and image embeddings, which is then multiplied with the foreground mask $F_{seg}$ from a DeepLabv3 segmentation model to yield the refined mask $A_{final}$. This confines semantic editing to the subject region of the image, and LPIPS constraints bound the perceptual distance between the decoded adversarial image and the original.

### Loss & Training

The total loss is a weighted sum of three terms:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{sem} + \lambda_2 \mathcal{L}_{dist} + \lambda_3 \mathcal{L}_{LPIPS}$$

Optimization uses the Adam optimizer (learning rate 0.005), with each iteration comprising $K=20$ outer loops and $T=20$ inner gradient descent steps. A grid search is conducted over diffusion strength $[0.3, 0.8]$ and CFG $[2.0, 12.0]$. The sole optimization variable is $e_{adv}$; optimization in embedding space rather than pixel space is the core strategic choice.

## Key Experimental Results

### Main Results

Evaluated on 100 image-caption pairs from the COCO dataset, simulating a black-box $N$-way selection scenario.

| Method | LLaVA-34B | Gemma3-8B | Mistral-3.1-24B | Mistral-3.2-24B | GPT-4o | CogVLM |
|--------|-----------|-----------|-----------------|-----------------|--------|--------|
| Initial Bad Image | 21% | 17% | 14% | 6% | 0% | 8% |
| SPSA | 36% | 27% | 22% | 11% | 1% | 18% |
| Bandit | 6% | 2% | 1% | 0% | 0% | 0% |
| SSA_CWA | 65% | 42% | 28% | 18% | 8% | 4% |
| SA_AET | 85% | 67% | 61% | 55% | 12% | 42% |
| **TRAP** | **100%** | **100%** | **100%** | **99%** | **63%** | **94%** |

### Defense Robustness

| Method | LLaVA-34B | Gemma3 | Mistral-3.1 | Mistral-3.2 | Robust-LLaVA |
|--------|-----------|--------|-------------|-------------|--------------|
| TRAP | 100% | 100% | 100% | 97% | 92% |
| TRAP + Gaussian Noise | 100% | 100% | 100% | 96% | 92% |
| TRAP + CIDER | 100% | 100% | 96% | 90% | 85% |
| TRAP + MirrorCheck | 100% | 98% | 88% | 82% | 74% |

### Ablation Study

| Configuration Change | LLaVA-34B | Gemma3 | Mistral-3.1 | Mistral-3.2 |
|----------------------|-----------|--------|-------------|-------------|
| Distinctive Loss 0.3→0.8 | 88% | 70% | 72% | 65% |
| Semantic Loss 0.5→0.0 | 90% | 82% | 77% | 70% |
| Perceptual Loss 1.0→1.5 | 100% | 100% | 100% | 98% |

### Key Findings

- TRAP substantially outperforms all baselines across all six evaluated models, including both open-source and closed-source systems.
- The attack transfers to non-contrastive architectures (CogVLM) and fully closed-source GPT-4o.
- Strong robustness to system prompt variants is observed, with ASR deviation confined to the low single digits.
- Attack effectiveness remains stable across different sampling temperatures ($T=0.1$ and $T=0.7$).
- Overemphasizing the distinctive loss degrades cross-model transferability.
- Removing the semantic loss term causes the most significant performance drop (ASR reduced to 70–90%).

## Highlights & Insights

- This work is the first to systematically demonstrate the threat of semantic-level cross-modal manipulation against autonomous agents, transcending the traditional pixel-level attack paradigm.
- The embedding-space optimization strategy is elegant—rather than directly modifying pixels, it operates on high-level semantics in CLIP space, enabling model-agnostic attack transferability.
- The twin network's "common-distinctive" decomposition is a key innovation, resolving the tension between semantic injection and identity preservation.
- The layout-aware mask design precisely confines semantic edits to foreground regions, enhancing the stealthiness of the attack.
- The black-box threat model is realistic and meaningful—attackers control only their own images and require no access to the environment or model internals.

## Limitations & Future Work

- The method assumes agents rely on contrastive visual-language similarity scoring; future non-contrastive architectures may diminish attack effectiveness.
- Attack success depends on the quality of auxiliary components (layout masks, diffusion models), which may degrade on edge cases.
- Computational cost is high (approximately 520 seconds per sample), several times that of pixel-level attacks; scalability in real-time scenarios remains a challenge.
- Effective defense strategies are not deeply explored; the work only demonstrates the insufficiency of existing defenses.
- Success rates drop significantly in e-commerce webpage scenarios (51%), indicating greater challenges in real-world deployment settings.

## Related Work & Insights

- Compared to traditional pixel-level attacks (FGSM, PGD, C&W), TRAP operates at the semantic level, making it more stealthy and harder to detect.
- Unlike existing diffusion-based semantic manipulation works (AdvDiff, Instruct2Attack), TRAP uses only model embeddings and requires no access to diffusion model parameters.
- This work motivates the development of embedding-space defenses and semantic-level robustness benchmarks, rather than relying solely on pixel-space robustness.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](../../ICML2025/llm_safety/targeted_unlearning_with_single_layer_unlearning_gradient.md)
- [\[ICLR 2026\] Building a Foundational Guardrail for General Agentic Systems via Synthetic Data](../../ICLR2026/llm_safety/building_a_foundational_guardrail_for_general_agentic_systems_via_synthetic_data.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)

</div>

<!-- RELATED:END -->
