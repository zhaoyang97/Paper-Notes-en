---
title: >-
  [Paper Note] DTR: Dynamic Token Reweighting for Robust Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][VLM jailbreak defense] DTR is proposed as the first method to defend against multimodal jailbreak attacks via KV cache optimization. It identifies adversarial visual tokens using a Reversal Safety-Relevant Shift (RSS) and suppresses their influence through dynamic reweighting. With only 4 optimization steps and without relying on image-to-text conversion, DTR substantially reduces attack success rates (HADES S+T+A: 56.9%→15.9%) while preserving VLM performance and inference efficiency.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM jailbreak defense
  - KV cache optimization
  - visual token reweighting
  - refusal direction
  - inference-time safety
date: 2026-05-08
content_hash: 2364f7709c778bdf
---

# DTR: Dynamic Token Reweighting for Robust Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2505.17132](https://arxiv.org/abs/2505.17132)
**Code**: [https://github.com/TanqiuJiang/DTR](https://github.com/TanqiuJiang/DTR)
**Area**: AI Safety / Multimodal Adversarial Defense
**Keywords**: VLM jailbreak defense, KV cache optimization, visual token reweighting, refusal direction, inference-time safety

## TL;DR
DTR is proposed as the first method to defend against multimodal jailbreak attacks via KV cache optimization. It identifies adversarial visual tokens using a Reversal Safety-Relevant Shift (RSS) and suppresses their influence through dynamic reweighting. With only 4 optimization steps and without relying on image-to-text conversion, DTR substantially reduces attack success rates (HADES S+T+A: 56.9%→15.9%) while preserving VLM performance and inference efficiency.

## Background & Motivation
VLMs (e.g., LLaVA, InternVL) are more vulnerable than pure LLMs due to the introduction of the visual modality. Attackers can bypass safety guardrails by embedding harmful content via adversarial perturbations, SD-generated images, or typographic manipulation. Limitations of existing defenses: fine-tuning approaches (e.g., RLHF) require annotated safety data and are computationally expensive; inference-time methods (e.g., AdaShield with iterative prompts, ECSO with image-to-text conversion) incur high overhead or significant information loss; shift calibration methods (e.g., ShiftDC, CoCA) rely on image-to-text references to estimate safety-relevant shifts accurately, limiting their effectiveness. The core challenge is how to efficiently and accurately quantify and eliminate the safety-relevant distributional shift introduced by the visual modality.

## Core Problem
How to effectively defend against multimodal jailbreak attacks through inference-time intervention, without requiring safety reference data, image-to-text conversion, or model fine-tuning?

## Method

### Overall Architecture
Pre-compute the refusal direction vector $\mathbf{d}_{ref}$ (requires only 32 harmful/benign prompt pairs, computed once) → At inference time, optimize a visual token scaling vector $\alpha \in [0,1]^n$ for each input (4 gradient descent steps) → Apply $\alpha$ to reweight the KV cache → Optionally prune low-weight tokens to accelerate inference.

### Key Designs
1. **Reversal Safety-Relevant Shift (RSS)**: Rather than directly measuring the safety-relevant shift (which requires a text reference $\tilde{x}$), RSS measures how far the representation can be moved along the *reversed* refusal direction by optimizing $\alpha$. The insight is that jailbreak queries are optimized to mislead safety judgments and can thus be reversed back into the safe region (large RSS), whereas benign queries are not shifted and offer little room for reversal (small RSS). Empirical validation confirms a clear separation between RSS distributions of 100 jailbreak vs. 100 benign queries.

2. **Dynamic Token Reweighting**: The optimization objective is
$$\mathcal{L}(\alpha) = f(x(\alpha)) \cdot \mathbf{d}_{ref}/\|\mathbf{d}_{ref}\| + \lambda\|f(x) - f(x(\alpha))\|_2$$
   - First term: minimizes safety-relevant shift along the refusal direction (actively corrects jailbreak queries; negligible effect on benign ones).
   - Second term: keeps the reweighted activations close to the original (preserves VLM capabilities).
   - $\lambda=0.1$ balances safety and utility.

3. **The Attacker's Fundamental Dilemma**: DTR creates an adversarial dilemma — increasing the weight of adversarial tokens (to bypass safety guardrails) disrupts visual semantic coherence, lowering ASR-G; preserving feature token weights allows DTR to reverse the safety-relevant shift, lowering ASR-R. The attacker cannot optimize both objectives simultaneously.

### Optimization Strategy
- Early stopping: 4 steps suffice, as the loss drops sharply in the first 4 steps for jailbreak queries.
- Token pruning: tokens with $\alpha < \beta$ are removed directly from the KV cache; a 20% pruning rate balances efficiency and performance.
- Inference time: 4.01s vs. Base 3.65s (only +10% overhead), far more efficient than ShiftDC at 10.66s.

## Key Experimental Results

| LLaVA-Llama2-7b | HADES S+T+A ASR↓ | MM-SafetyBench S+T ASR↓ | JailBreakV Style ASR↓ |
|---|---|---|---|
| Base | 56.9% | 74.5% | 34.0% |
| AdaShield | 17.6% | 13.6% | 8.5% |
| ShiftDC | 16.8% | 13.6% | 25.5% |
| CoCA | 35.7% | 53.6% | 8.5% |
| **DTR** | **15.9%** | **10.0%** | **6.4%** |

- MM-Vet performance retention: DTR matches or outperforms the base model on 5 of 6 capability dimensions; Recognition is fully preserved (50.3→50.3), whereas CoCA degrades substantially (50.3→28.7).
- Cross-model generalization: InternVL-2.5-26b HADES S+T+A 23.1%→3.5%; Llama-4-Scout-17B 11.2%→8.4%.
- Adaptive attack + PGD: ASR drops from 68% (undefended) to 18% with DTR (still effective).
- VLGuard text attack: safe-image + harmful-text ASR 66.5%→7.4%.

### Ablation Study
- $n_{ref}=32$ is sufficient (16 also approaches the optimum); the refusal direction generalizes stably across datasets and domains.
- $m=4$ optimization steps suffice; the loss decreases sharply in the first 4 steps.
- $\lambda=0.1$ achieves the optimal safety–utility trade-off.
- 20% pruning rate reduces inference time further with no change in ASR.
- Uniform scaling vs. DTR: uniform $\alpha=0.3$ causes severe hallucinations, whereas DTR precisely localizes adversarial tokens.

## Highlights & Insights
- DTR is the first to apply KV cache optimization to VLM safety, establishing an entirely new defense paradigm.
- The RSS formulation is elegant: it avoids the information loss associated with image-to-text conversion and operates directly in activation space.
- The analysis of the attacker's dilemma is thorough: the inverse relationship between ASR-R and ASR-G makes DTR robust even against adaptive attacks.
- Strong interpretability: the $\alpha$ heatmap directly reveals which visual tokens are adversarial.
- Minimal computational overhead (+10% inference time only), no training required, plug-and-play deployment.

## Limitations & Future Work
- Extension to natively multimodal VLMs (e.g., GPT-4o, where vision and text are not processed separately) remains to be addressed.
- The refusal direction requires a small number of harmful/benign prompt pairs for pre-computation — while 32 suffice, the method is not completely data-free.
- Strong adaptive attacks (PGD minimizing RSS) can still achieve 18% ASR; complete elimination remains challenging.
- The optimal layer selection (layer 14) may require adjustment for different architectures.

## Related Work & Insights
- **vs. ShiftDC**: ShiftDC requires an image-to-text reference and doubles inference time; DTR requires no reference and adds only 10% overhead.
- **vs. CoCA**: CoCA calibrates at the logit level but causes severe performance degradation (Recognition 50.3→28.7); DTR operates at the KV cache level and preserves performance.
- **vs. AdaShield**: AdaShield iteratively refines prompts with high overhead (5.24s); DTR directly optimizes $\alpha$ (4.01s).
- **vs. KV pruning methods (e.g., MADTP)**: these target efficiency; DTR is the first to apply KV optimization for safety purposes.

The cross-domain transferability of the refusal direction implies that safety mechanisms are intrinsic properties at the model level rather than data-specific artifacts. The paradigm of applying KV cache optimization for safety may extend to audio and video multimodal settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ KV cache-based safety optimization is an entirely new paradigm; the RSS formulation avoids the image-to-text bottleneck.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 VLMs, 3 attack benchmarks, adaptive attacks, cross-domain transfer, uniform vs. dynamic analysis, and 11 ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ A complete logical chain from motivation → observation → method → theory → experiments, with a rigorous attack–defense game analysis.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, minimal overhead, and preserved performance make DTR highly practical for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)
- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
