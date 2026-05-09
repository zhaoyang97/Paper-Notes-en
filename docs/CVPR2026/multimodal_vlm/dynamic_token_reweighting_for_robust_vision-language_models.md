---
title: >-
  [Paper Note] Dynamic Token Reweighting for Robust Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][VLM safety] This paper proposes DTR (Dynamic Token Reweighting), the first inference-time defense against multimodal jailbreak attacks that operates by optimizing the KV cache of VLMs. DTR introduces the concept of "Reversal Safety-Relevant Shift" (RSS) to identify visual tokens responsible for safety degradation, dynamically adjusts their weights to restore the model's safety alignment, and preserves benign task performance.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM safety
  - jailbreak defense
  - KV cache optimization
  - token reweighting
  - refusal direction
date: 2026-05-08
content_hash: ab7119b5c7d6132d
---

# Dynamic Token Reweighting for Robust Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2505.17132](https://arxiv.org/abs/2505.17132)
**Code**: [GitHub](https://github.com/TanqiuJiang/DTR)
**Area**: Multimodal VLM
**Keywords**: VLM safety, jailbreak defense, KV cache optimization, token reweighting, refusal direction

## TL;DR
This paper proposes DTR (Dynamic Token Reweighting), the first inference-time defense against multimodal jailbreak attacks that operates by optimizing the KV cache of VLMs. DTR introduces the concept of "Reversal Safety-Relevant Shift" (RSS) to identify visual tokens responsible for safety degradation, dynamically adjusts their weights to restore the model's safety alignment, and preserves benign task performance.

## Background & Motivation

**Background**: Large vision-language models (VLMs) achieve powerful multimodal reasoning by integrating visual and linguistic capabilities, yet the introduction of the visual modality creates new security vulnerabilities — multimodal jailbreak attacks exploit vision-text interactions to bypass safety guardrails.

**Limitations of Prior Work**: Fine-tuning-stage approaches (e.g., RLHF-based safety alignment) are computationally expensive and require annotated data. Inference-stage approaches either rely on iterative prompting (high overhead) or image-to-text conversion (severe information loss). Recent distribution-shift correction methods (ShiftDC, CoCA) require safety references, which are typically obtained via lossy image-to-text translation.

**Key Challenge**: Accurately quantifying the "safety shift" induced by the visual modality requires comparing states with and without an image, but obtaining an accurate text-only counterpart inherently involves information loss.

**Goal**: Design an inference-time jailbreak defense that requires no safety reference data, no image-to-text conversion, and incurs minimal computational overhead.

**Key Insight**: Rather than measuring "how much shift the image introduces," DTR measures "how much of the shift can be reversed by adjusting visual token weights" — this RSS directly distinguishes jailbreak queries from benign ones.

**Core Idea**: Jailbreak attacks optimize a query from "refused" to "accepted," and therefore the shift can be reversed — whereas benign queries lack this reversibility.

## Method

### Overall Architecture
Given a query $\mathbf{x} = \mathbf{x}_{txt} \| \mathbf{x}_{img}$, a scaling factor $\alpha_i \in [0,1]$ is assigned to each visual token. Gradient descent is used to optimize $\alpha$ so as to minimize the projection of the last-layer activation along the refusal direction (i.e., reversing the safety shift), while a distance constraint to the original activation is imposed to maintain benign performance. After optimization, low-weight tokens are evicted, and inference proceeds with the adjusted KV cache.

### Key Designs

1. **Reversal Safety-Relevant Shift (RSS)**

   - **Function**: Bypasses image-to-text conversion by directly quantifying the reversibility of the safety shift through optimization.
   - **Mechanism**: Defines $\Delta^*_{safe}(\mathbf{x}) = \max_{\alpha} \frac{(f(\mathbf{x}) - f(\mathbf{x}(\alpha))) \cdot \mathbf{d}_{ref}}{\|\mathbf{d}_{ref}\|}$, i.e., the maximum reversal shift achievable along the refusal direction by adjusting visual token weights. The RSS of jailbreak queries is substantially larger than that of benign queries, since attacks are inherently optimized along the refusal direction and are thus naturally reversible.
   - **Design Motivation**: Eliminates the information loss and additional VLM overhead associated with image-to-text conversion, while creating a fundamental dilemma for attackers — increasing adversarial token importance amplifies RSS and makes detection easier, whereas decreasing it renders the jailbreak ineffective.

2. **Dynamic Token Reweighting Optimization**

   - **Function**: Optimizes the scaling vector for visual tokens to simultaneously recover safety and preserve performance.
   - **Mechanism**: $\alpha^* = \arg\min_{\alpha} \left[\frac{f(\mathbf{x}(\alpha)) \cdot \mathbf{d}_{ref}}{\|\mathbf{d}_{ref}\|} + \lambda \|f(\mathbf{x}) - f(\mathbf{x}(\alpha))\|_2 \right]$, where the first term minimizes the projection along the refusal direction (safety recovery) and the second term constrains the distance from the original activation (performance preservation), with $\lambda$ balancing the two objectives.
   - **Design Motivation**: Minimizing the safety shift alone degrades benign performance; the distance constraint ensures minimal impact on benign queries.

3. **Early Stopping + Token Eviction**

   - **Function**: Only 3–4 optimization steps are required; evicting low-weight tokens further improves efficiency.
   - **Mechanism**: The loss for jailbreak queries decreases rapidly in the first few steps, making full convergence unnecessary. Tokens whose weights fall below threshold $\beta$ are evicted directly from the KV cache — visual tokens are inherently redundant, so eviction actually accelerates inference.
   - **Design Motivation**: Minimal optimization steps combined with token eviction yield inference latency comparable to or lower than the baseline.

4. **Robustness of the Refusal Direction**

   - **Function**: A stable refusal direction can be extracted from as few as 32 harmful/harmless text prompt pairs.
   - **Mechanism**: 32 harmful prompts are sampled from AdvBench and 32 benign prompts from AlpacaEval; the refusal direction is computed as the difference between the mean last-layer activations. Experiments demonstrate that this direction generalizes across languages, attack types, and datasets.
   - **Design Motivation**: The refusal direction captures an intrinsic model-level property rather than dataset-specific artifacts, making small-sample extraction sufficient.

### Loss & Training
DTR is a fully inference-time method requiring no training. Optimization uses AdamW with a learning rate of 0.01, $\lambda=0.1$, and a default of 3–4 gradient descent steps.

## Key Experimental Results

### Main Results

**LLaVA-LLaMA2-7B Attack Success Rate (ASR↓, lower is better)**

| Defense | HADES-S | HADES-S+A | HADES-S+T+A | MM-Safety-S | MM-Safety-T | JailBreak-Style |
|---------|---------|-----------|-------------|-------------|-------------|-----------------|
| No Defense | 31.4% | 44.9% | 56.9% | 70.0% | 72.7% | 34.0% |
| AdaShield | 7.5% | 5.5% | 17.6% | 8.2% | 4.5% | 8.5% |
| ShiftDC | 20.0% | 32.9% | 16.8% | 10.9% | 5.5% | 25.5% |
| CoCA | 23.6% | 20.8% | 35.7% | 24.3% | 26.3% | 8.5% |
| **DTR** | **8.9%** | **4.8%** | **15.9%** | **3.6%** | **3.6%** | **6.4%** |

### Ablation Study

| Configuration | ASR↓ | MM-Vet↑ | Inference Time |
|---------------|------|---------|----------------|
| DTR (full) | ~5% | ~35 | ~7s |
| w/o distance constraint ($\lambda=0$) | ~3% | ~28 | ~7s |
| w/o token eviction | ~5% | ~35 | ~9s |
| eviction only, no reweighting | ~15% | ~34 | ~5s |
| Baseline (no defense) | ~45% | ~35 | ~6s |

### Key Findings
- DTR achieves the lowest ASR across nearly all attack types, with the largest reduction on MM-Safety-S (70% → 3.6%).
- DTR maintains or improves inference efficiency, as token eviction reduces KV cache size.
- The distance constraint $\lambda$ is critical for benign performance; removing it causes MM-Vet to drop from 35 to 28.
- The refusal direction requires only 32 sample pairs and generalizes strongly across domains.
- Attackers face a fundamental dilemma: strengthening adversarial tokens increases RSS, making detection easier.

## Highlights & Insights
- **The attacker's dilemma** is the most profound contribution — rather than engaging in a specific attack-defense arms race, the paper establishes a fundamental trade-off between attack effectiveness and detectability.
- **First work to apply KV cache optimization for safety** — unifying efficiency optimization (token eviction) and safety defense (token reweighting) within a single optimization framework.
- **RSS as a substitute for image-to-text conversion** is an elegant design — instead of asking "how much shift does the image introduce," it asks "how much of the shift can be reversed by adjusting tokens," neatly circumventing the information loss problem.

## Limitations & Future Work
- Each inference requires 3–4 gradient optimization steps, which, while fast, still incurs overhead.
- The refusal direction assumes that safety concepts are linearly represented in activation space, which may be insufficient for highly complex safety scenarios.
- Validation is limited to image-text VLMs; generalization to video/audio multimodal settings remains unexplored.
- The eviction threshold $\beta$ requires tuning.

## Related Work & Insights
- **vs. AdaShield**: Relies on iterative prompting to check image safety, incurring high computational cost; DTR operates directly at the KV cache level.
- **vs. ShiftDC**: Requires image-to-text conversion to obtain safety references, resulting in information loss; DTR bypasses this requirement via RSS.
- **vs. CoCA**: Corrects the shift at the decoding logit level; DTR operates at the lower-level KV cache, yielding superior performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Both the RSS concept and KV cache-based safety optimization are pioneering contributions; the attacker dilemma analysis is particularly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 5 VLMs, 3 attack benchmarks, multiple attack types, adaptive attacks, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from problem formulation to theoretical analysis to experiments is complete and clear.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to safe VLM deployment; opens a new direction in KV cache-based safety optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DTR: Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_visionlanguag.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)
- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_safeguarding_multimodal_multi-turn_dialogues_in_vision-language_mode.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

</div>

<!-- RELATED:END -->
