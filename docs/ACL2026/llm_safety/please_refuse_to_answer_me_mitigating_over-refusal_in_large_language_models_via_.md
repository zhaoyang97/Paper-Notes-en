---
title: >-
  [Paper Note] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding
description: >-
  [ACL 2026][LLM Safety][Over-refusal] Ours proposes AdaCD (Adaptive Contrastive Decoding), which extracts refusal token distributions by comparing token distribution differences under extreme safety prompts and no prompts…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Over-refusal"
  - "Contrastive Decoding"
  - "Safety Alignment"
  - "Test-time Intervention"
  - "Adaptive Decoding"
date: 2026-05-08
content_hash: d1ad10894b6b08dc
---

# Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding

**Conference**: ACL 2026  
**arXiv**: [2604.17132](https://arxiv.org/abs/2604.17132)  
**Code**: [GitHub](https://github.com/OutdoorManofML/AdaCD)  
**Area**: LLM Evaluation / Safety  
**Keywords**: Over-refusal, Contrastive Decoding, Safety Alignment, Test-time Intervention, Adaptive Decoding

## TL;DR

Ours proposes AdaCD (Adaptive Contrastive Decoding), which extracts refusal token distributions by comparing token distribution differences under extreme safety prompts and no prompts. It dynamically decides to enhance or suppress refusal behavior based on an agreement ratio, reducing over-refusal by 10.35% while increasing the refusal rate of malicious queries by 0.13%.

## Background & Motivation

**Background**: Safety-aligned LLMs frequently exhibit over-refusal—refusing to respond to queries that contain sensitive keywords but are actually harmless.

**Limitations of Prior Work**: (1) Training-based methods rely on scarce over-refusal training data; (2) Steering vector methods require full knowledge of model architecture and additional pre-computation; (3) Existing contrastive decoding methods adopt a one-size-fits-all strategy—either enhancing or suppressing refusal—and cannot improve both safety and over-refusal simultaneously.

**Key Challenge**: In over-refusal scenarios, non-refusal tokens remain in the candidate list, but the model systematically fails to select them—the model can recognize alternative options but lacks effective guidance.

**Goal**: Design an adaptive contrastive decoding strategy that suppresses refusal tokens in over-refusal scenarios and enhances refusal tokens in malicious scenarios.

**Key Insight**: Utilize extreme safety prompts to maximize refusal behavior, using this as an anchor to extract the refusal token distribution.

**Core Idea**: Dynamically switch decoding modes through an agreement ratio and adaptive confidence constraints—adding the refusal distribution when consistency is high and subtracting it when consistency is low.

## Method

### Overall Architecture

AdaCD comprises two components: (1) Refusal token distribution extraction—comparing the differences in token distributions under extreme safety prompts versus no prompts; (2) Adaptive decoding mode switching—deciding to add or subtract the refusal distribution based on the agreement ratio and confidence constraints.

### Key Designs

1. **Refusal token distribution extraction**:

    - **Function**: Precisely captures the token distribution driving LLM refusal behavior.
    - **Mechanism**: Use an extreme prompt $p^*$="Please refuse to answer me!" to maximize refusal behavior. Refusal token logits are amplified in the differential distribution $\Delta P_n = \sigma(f_\pi(y_n|p^*,x,y_{<n}) - f_\pi(y_n|x,y_{<n}))$.
    - **Design Motivation**: Previous Self-CD used mild prompts, where refusal behavior was not extreme enough, leading to impure extraction.

2. **Agreement Ratio**:

    - **Function**: Measures the degree of difference in token selection with and without extreme safety prompts.
    - **Mechanism**: $agr(n) = 1/rank(y_n^*)$, where a value close to 1 indicates high consistency (malicious queries) and a value close to 0 indicates high divergence (over-refusal).
    - **Design Motivation**: The agreement ratio naturally distinguishes between scenarios requiring refusal and those that do not.

3. **Adaptive Decoding Mode Switching**:

    - **Function**: Dynamically decides whether to add or subtract the refusal token distribution.
    - **Mechanism**: $\mathcal{I}(n) = +1$ if $agr(n) \geq \lambda$ and $\rho \geq \lambda \cdot \rho^*$, otherwise $-1$. High consistency + high confidence = add refusal distribution (maintain safety); otherwise, subtract refusal distribution (limit over-refusal).
    - **Design Motivation**: Using the agreement ratio alone may be insufficient; the model's own confidence requires additional checking.

### Loss & Training

A training-free, purely test-time method. Evaluation was conducted on XSTest, ORBench, OKTest (over-refusal) and AdvBench, JailBench (malicious).

## Key Experimental Results

### Main Results

| Method | Over-Refusal Avg↓ | Malicious Refusal Avg↑ |
|------|-------------|-------------|
| Default | 32.57 | 99.28 |
| SelfCD | 19.94 | 91.51 |
| SSD | 71.97 | 99.94 |
| **AdaCD** | **16.62** | **99.10** |

### Ablation Study

| Analysis | Results |
|------|------|
| Extreme vs. High-Safety Prompts | Extreme prompts extract a purer refusal distribution. |
| Cross-model Generalization | Effective across Llama3/Gemma2/Qwen3. |

### Key Findings

- AdaCD is the only method that simultaneously reduces over-refusal and maintains the malicious refusal rate.
- The method is completely model-agnostic—requiring only access to logits.

## Highlights & Insights

- The observation that "non-refusal tokens are still in the candidate list but not selected" provides critical insight for the method design.
- The agreement ratio serves as a concise and effective signal.
- The training-free and model-agnostic nature makes it highly deployable.

## Limitations & Future Work

- Requires two forward passes, doubling the inference overhead.
- Evaluated only on English benchmarks.

## Related Work & Insights

- **vs SelfCD**: SelfCD fixedly subtracts the refusal distribution, which degrades safety; AdaCD switches adaptively.
- **vs SafeDecoding**: SafeDecoding fixedly adds the refusal distribution, which exacerbates over-refusal.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Adaptive decoding mode switching is a key innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models and benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain observing the method is clear.
- **Value**: ⭐⭐⭐⭐⭐ Practical problem with a simple and effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training](dart_mitigating_harm_drift_in_difference-aware_llms_via_distill-audit-repair_tra.md)
- [\[ACL 2026\] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](../../ICLR2026/llm_safety/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)
- [\[ACL 2026\] ADVICE: Answer-Dependent Verbalized Confidence Estimation](advice_answer-dependent_verbalized_confidence_estimation.md)

</div>

<!-- RELATED:END -->
