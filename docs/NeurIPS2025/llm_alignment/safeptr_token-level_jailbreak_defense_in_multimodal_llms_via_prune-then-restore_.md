---
title: >-
  [Paper Note] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism
description: >-
  [NeurIPS 2025][LLM Alignment][multimodal safety] By analyzing the propagation mechanism of harmful tokens in multimodal LLMs, this work finds that fewer than 1% of tokens trigger jailbreak behavior in early-to-middle layers. Based on this finding, the training-free SafePTR framework is proposed, which prunes harmful tokens at vulnerable layers and restores benign features in subsequent layers, significantly improving safety without sacrificing task performance.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - multimodal safety
  - jailbreak defense
  - token pruning
  - MLLM
  - training-free defense
date: 2026-05-08
content_hash: 5a15072a7e3b3bf4
---

# SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism

**Conference**: NeurIPS 2025
**arXiv**: [2507.01513](https://arxiv.org/abs/2507.01513)
**Code**: [GitHub](https://github.com/BT-C/SafePTR)
**Area**: LLM Alignment
**Keywords**: multimodal safety, jailbreak defense, token pruning, MLLM, training-free defense

## TL;DR
By analyzing the propagation mechanism of harmful tokens in multimodal LLMs, this work finds that fewer than 1% of tokens trigger jailbreak behavior in early-to-middle layers. Based on this finding, the training-free SafePTR framework is proposed, which prunes harmful tokens at vulnerable layers and restores benign features in subsequent layers, significantly improving safety without sacrificing task performance.

## Background & Motivation

**Multimodal Jailbreak Threats**: MLLMs extend LLM capabilities by integrating visual inputs, but also introduce new security vulnerabilities — multimodal jailbreak attacks (e.g., JailbreakV-28K, FigStep, MM-SafetyBench) can bypass model safety mechanisms.

**Limitations of Prior Work**:
   - **Image-to-text methods** (e.g., ECSO): Convert visual inputs into text descriptions, but remain vulnerable to text-driven jailbreaks.
   - **Safety prompt methods** (e.g., AdaShield): Statically inject safety constraints, lacking adaptability and prone to over-defense (e.g., misclassifying "toy water guns" as "real weapons").
   - **Multimodal safety fine-tuning** (e.g., TGA): Requires large-scale training (1,223K samples, 64×V100 GPUs) with limited generalization.

**Key Challenge**: Existing methods rely on LLMs' built-in safety mechanisms without deeply investigating the underlying mechanism by which harmful multimodal tokens bypass safety alignment.

## Method

### Overall Architecture

SafePTR is a **training-free** defense framework consisting of two core modules:

1. **Harmful Token Pruning (HTP)**: Identifies and prunes harmful tokens at vulnerable layers.
2. **Benign Feature Restoration (BFR)**: Restores benign features in subsequent layers to preserve task capability.

### Key Findings (Three Findings)

**Finding-1 (Where)**: Through Layer-wise Intervention Analysis (LIA), only a small number of early-to-middle layers are found to be particularly vulnerable to jailbreak attacks:
- LLaVA-1.5-7B: layers $[7, 9)$
- MiniGPT-4-7B: layers $[7, 9)$
- DeepSeek-VL2: layers $[4, 6)$

Pruning harmful tokens in these 2–4 consecutive layers reduces ASR from 67.3% to 4.2%.

**Finding-2 (How)**: Greater semantic deviation from safety-aligned instructions correlates with higher jailbreak success rates. Safe samples cluster near safety-aligned representations, while unsafe samples shift away from the safe region (average centroid distance: 0.11–0.14).

**Finding-3 (Which)**: Fewer than 1% of multimodal tokens cause significant semantic deviation:
- LLaVA-1.5 on MM-SafetyBench: 0.62%
- MiniGPT-4 on MM-SafetyBench: 0.93%
- DeepSeek-VL2 on MM-SafetyBench: 1.66%

### Harmful Token Pruning (HTP)

Within the vulnerable layers $[n, n+\Delta_n)$, cosine similarity is computed between visual/instruction tokens and the safety-aligned instruction representations. The Top-K tokens with the greatest deviation from the safety space are selected for pruning. The safety-aligned instructions follow a fixed template.

Visual and textual modalities are pruned independently, as the embedding distance distributions differ between the two modalities. $K$ defaults to 10% of total tokens.

### Benign Feature Restoration (BFR)

After HTP pruning, subsequent layers operate on incomplete visual representations. BFR maintains a **parallel branch** for standard inference, then selectively restores benign features at safe layers. Pruned positions receive features from the standard inference branch, while non-pruned positions retain features from the pruned branch; the two are re-concatenated to recover the complete sequence.

This dual-path design ensures that restored tokens are less susceptible to attack influence in subsequent layers, primarily serving cross-modal integration and language refinement.

### Loss & Training

- **Completely training-free**: Requires no additional safety datasets or fine-tuning.
- **Single-pass inference**: Defense is completed within a single forward pass (one-bypass inference).
- **Zero additional computational overhead**: No new parameters or auxiliary models are introduced.

## Key Experimental Results

### Main Results: ASR (%, lower is better) on MM-SafetyBench

| Model | Method | Avg. ASR↓ |
|-------|--------|-----------|
| LLaVA-1.5-7B | Original | 51.7 |
| | AdaShield | 14.3 |
| | Immune | 2.1 |
| | **SafePTR** | **1.3** |
| MiniGPT-4-7B | Original | 58.3 |
| | CoCA | 29.7 |
| | Immune | 18.3 |
| | **SafePTR** | **~15** |
| DeepSeek-VL2 | Original | 72.7 |
| | AdaShield | 14.4 |
| | **SafePTR** | **10.1** |

### Utility Preservation

SafePTR achieves performance close to the original model on MME and MM-Vet benchmarks, demonstrating that the BFR module effectively recovers task-relevant benign features.

### Ablation Study

| Configuration | Safety | Utility |
|---------------|--------|---------|
| HTP only | High | Notably degraded |
| BFR only | Insufficient | Good |
| HTP + BFR | High | Good |

### Key Findings

1. Top-K = 10% is optimal: too few tokens fail to prune effectively; too many degrade utility.
2. Layer selection is critical: intervention in only 2–4 vulnerable layers achieves the best safety-utility trade-off.
3. BFR significantly improves utility: restoring features at subsequent safe layers brings task performance close to the original model.
4. Attention Sink insight: harmful tokens concentrate at attention sink positions.

## Highlights & Insights

1. **Interpretable safety analysis**: The first work to systematically analyze MLLM jailbreak mechanisms along three dimensions — Where, How, and Which.
2. **Elegant training-free design**: Requires no safety data and introduces no inference overhead.
3. **Dual-modality defense**: Simultaneously defends against both vision-driven and text-driven jailbreak attacks.
4. **Semantic heatmaps**: Visualization of harmful tokens intuitively demonstrates high deviation in tokens associated with violent scenes such as "armed figures" and "smoke."

## Limitations & Future Work

1. Layer selection relies on prior LIA, requiring per-model analysis for each new architecture.
2. The fixed Top-K strategy lacks flexibility; adaptive $K$ selection warrants further exploration.
3. Safety-aligned instructions are fixed, which may be insufficient for complex attacks requiring dynamic reference points.
4. Validation is limited to three open-source 7B-scale MLLMs.

## Related Work & Insights

- **ECSO**: Image-to-text translation defense — bypassed by text-driven attacks.
- **AdaShield**: Safety prompt injection — prone to over-defense.
- **Immune**: Safety fine-tuning — high training cost and limited generalization.
- **FastV**: Source of inspiration for SafePTR's Top-K pruning strategy.
- **Insight**: The approach is generalizable to other multimodal settings, such as audio-language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to systematically analyze MLLM jailbreak mechanisms at token granularity.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 5 benchmarks with complete ablations, though validation on larger models is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The Where/How/Which analytical framework is clearly presented.
- Value: ⭐⭐⭐⭐⭐ A practical training-free defense solution with direct value for safe MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[NeurIPS 2025\] Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](../../ACL2026/llm_alignment/trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)

</div>

<!-- RELATED:END -->
