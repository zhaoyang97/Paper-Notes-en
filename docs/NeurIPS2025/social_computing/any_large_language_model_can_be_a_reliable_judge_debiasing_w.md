---
title: >-
  [Paper Note] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector
description: >-
  [NeurIPS 2025][Social Computing][LLM-as-judge] This paper proposes the Reasoning-based Bias Detector (RBD), a plug-and-play debiasing module for LLM judges. By externally detecting four types of evaluation bias (verbosity, position, bandwagon, and sentiment), RBD generates structured feedback with reasoning chains to guide judges toward self-correction. RBD-8B achieves an average accuracy improvement of 18.5% and consistency improvement of 10.9% across 8 LLM judges.
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "LLM-as-judge"
  - "bias detection"
  - "reasoning-based debiasing"
  - "self-correction"
  - "evaluation reliability"
date: 2026-05-08
content_hash: d8fc923ff072923d
---

# Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector

**Conference**: NeurIPS 2025
**arXiv**: [2505.17100](https://arxiv.org/abs/2505.17100)  
**Code**: [GitHub](https://github.com/Joyyang158/Reasoning-Bias-Detector)  
**Area**: Social Computing
**Keywords**: LLM-as-judge, bias detection, reasoning-based debiasing, self-correction, evaluation reliability

## TL;DR
This paper proposes the Reasoning-based Bias Detector (RBD), a plug-and-play debiasing module for LLM judges. By externally detecting four types of evaluation bias (verbosity, position, bandwagon, and sentiment), RBD generates structured feedback with reasoning chains to guide judges toward self-correction. RBD-8B achieves an average accuracy improvement of 18.5% and consistency improvement of 10.9% across 8 LLM judges.

## Background & Motivation
**Background**: LLM-as-a-Judge has been widely adopted for automated evaluation (e.g., Chatbot Arena, model ranking), yet judges themselves exhibit systematic biases—favoring longer responses (verbosity bias), preferring the first option (position bias), being swayed by majority opinion (bandwagon bias), and being influenced by emotional tone (sentiment bias).

**Limitations of Prior Work**: (1) In-context learning (prompt engineering) fails to correct deep-seated biases, especially for weaker models; (2) fine-tuning-based debiasing is inapplicable to closed-source models (GPT-4o, Claude, etc.) and risks overfitting; (3) existing methods merely instruct judges to "avoid bias" without providing concrete bias diagnoses or corrective guidance.

**Key Challenge**: How can sufficiently specific bias feedback be delivered to enable judge self-correction, without modifying the judge itself (thus remaining compatible with closed-source models)?

**Goal**: Design an external module capable of detecting bias and providing reasoning-chain-based corrective suggestions, enabling any LLM—including weaker models—to serve as a reliable judge.

**Key Insight**: Drawing on the chain-of-thought capabilities of Large Reasoning Models (LRMs), the paper trains a dedicated "bias detection reasoner" whose output format is `<think>reasoning analysis</think>bias label`, with the reasoning analysis covering bias type identification, comparative analysis, and judge capability assessment.

**Core Idea**: Use a fine-tuned reasoning model as an external bias detector and debias through an iterative detect→feedback→re-judge loop.

## Method

### Overall Architecture
A four-stage pipeline: (1) construct datasets $D$ and $D_{bias}$ for four bias types (0.5K samples each for control and biased groups); (2) use a teacher LRM to generate reasoning traces for bias analysis; (3) distill the reasoning traces into RBD models (1.5B–14B); (4) at inference time, RBD and the LLM judge collaborate iteratively until no bias is detected or the maximum number of iterations is reached.

### Key Designs

1. **Bias Dataset Construction (4 Bias Types × Control/Biased Groups)**:

    - Function: Construct paired datasets for each bias type—$D$ (normal evaluation) and $D_{bias}$ (bias-injected evaluation)—labeling instances as "biased" when the judge answers correctly on $D$ but incorrectly on $D_{bias}$.
    - Mechanism:
        - Verbosity bias: The correct answer is shortened from a full reasoning chain + answer to the final answer only (correct but short vs. incorrect but long).
        - Position bias: The order of options is swapped.
        - Bandwagon bias: A fabricated majority opinion—"90% of people consider option X better"—is inserted, pointing toward the wrong answer.
        - Sentiment bias: GPT-4o rewrites option tones (correct option in negative tone, incorrect option in positive tone).
    - Design Motivation: Precisely controlling bias sources ensures reliable labels under $b_i = \mathbb{1}[\hat{y}_i = y_i \land \hat{y}_i^{bias} \neq y_i]$.

2. **Reasoning-based Bias Detection (RBD Training)**:

    - Function: Use a teacher model (DeepSeek-R1) to generate bias analysis reasoning traces, filter them, and distill into smaller models.
    - Mechanism: Each reasoning trace covers three components—(a) identification of potential bias types; (b) comparative analysis of options, assessing whether the judge's decision was influenced by bias based on bias definitions; (c) evaluation of judge capability, since different models exhibit different sensitivities to bias.
    - Design Motivation: Training on labels alone (bias-only fine-tuning) leads to overfitting on surface patterns (e.g., "short answer → Yes"), causing accuracy to drop to 0% on diagnostic sets; reasoning-based training preserves robustness.

3. **Iterative Collaborative Debiasing (Algorithm 1)**:

    - Function: RBD inspects the judge's decision → if bias is detected, generates reasoning-based feedback → the judge revises its decision using the feedback → RBD re-inspects → repeats until no bias is detected or the maximum iterations are reached.
    - Mechanism: $\hat{y}^{bias} \leftarrow \mathcal{M}_J(x^{bias}, \hat{y}^r)$, where the judge performs self-reflection using RBD's reasoning analysis as additional reference.
    - Design Motivation: A single detection pass may be insufficient (the judge may err in a different biased manner), so iterative cycling ensures convergence to an unbiased outcome.

### Training Details
- Four RBD model sizes: 1.5B, 7B, 8B, 14B (distilled from the DeepSeek-R1 series).
- 1.67K training samples; all bias types are trained jointly rather than separately.
- Output format: `<think>reasoning trace</think>bias label (Yes/No)`.

## Key Experimental Results

### RBD-8B Performance across 4 Bias Types × 8 Judges

| Bias Type | Avg. Accuracy Gain | Avg. Consistency Gain |
|-----------|-------------------|----------------------|
| Verbosity Bias | +22.1% | +14.3% |
| Position Bias | +15.8% | +9.2% |
| Bandwagon Bias | +16.4% | +8.7% |
| Sentiment Bias | +19.7% | +11.4% |
| **Overall Average** | **+18.5%** | **+10.9%** |

### Comparison with Baselines

| Method | Accuracy Gain |
|--------|--------------|
| Zero-shot prompting | +3.2% |
| 4-shot prompting + reasoning | +5.7% |
| Fine-tuned judge | +1.3% |
| DeepSeek-R1 (zero-shot) | +8.6% |
| **RBD-8B** | **+18.5%** |

### Scaling Behavior

| RBD Model Size | Bias Detection F1 | Judge Accuracy Gain |
|---------------|------------------|-------------------|
| 1.5B | 0.72 | +12.3% |
| 7B | 0.79 | +16.1% |
| 8B | 0.81 | +18.5% |
| 14B | 0.83 | +19.8% |

### Key Findings
- **Reasoning-based vs. label-only training**: Label-only training achieves acceptable accuracy on the standard test set but fails entirely on the diagnostic set (verbosity bias drops to 0%), indicating that it only learns surface patterns (e.g., "short answer = biased"); reasoning-based training remains robust across all settings.
- **All 8 judges exhibit bias**: Even GPT-4o and Claude-3.5-sonnet consistently display detectable bias (verbosity bias is the most severe, affecting 31.3% of samples).
- **RBD generalizes across domains**: The model remains effective on unseen domains and bias variants.
- **RBD-7B surpasses zero-shot DeepSeek-R1**: After distillation fine-tuning, a model far smaller than the teacher exceeds the teacher's zero-shot performance.

## Highlights & Insights
- **External modular design**: RBD does not modify the judge and can be used plug-and-play with any LLM, including closed-source models—a fundamental advancement over existing approaches.
- **Reasoning-based vs. instruction-based debiasing**: Rather than simply saying "please avoid bias," RBD provides concrete bias diagnoses and comparative analyses, enabling even weak judges to correct their decisions effectively.
- **Clever diagnostic set design**: Constructing "anti-bias" diagnostic sets (e.g., making the longer response correct for verbosity bias) precisely exposes the overfitting problem inherent in label-only training.
- **Joint training over 4 bias types**: A single model handling all bias types jointly is more efficient and generalizes better than training separate models.

## Limitations & Future Work
- **Only 4 bias types**: More bias categories may exist in practice (e.g., self-preference bias, knowledge bias), and the framework needs to be extended accordingly.
- **Iterative overhead**: Each iteration requires one call to both RBD and the judge, increasing latency.
- **Teacher model dependency**: Training data quality is bounded by DeepSeek-R1's reasoning capability.
- **Binary bias labels**: Real-world bias may be graded rather than binary; a Yes/No label may be overly coarse.
- **Small training data scale**: Only 0.5K training samples per bias type.

## Related Work & Insights
- **vs. Prompt-engineering debiasing (ICL-based)**: Surface-level instructions fail to correct deep-seated biases, especially for weaker models; RBD provides concrete reasoning-based feedback instead.
- **vs. Fine-tuned judges (JudgeLM, Prometheus)**: These methods require large preference datasets and are inapplicable to closed-source models; RBD operates externally without requiring access to judge parameters.
- **vs. Multi-agent debiasing**: Multi-agent approaches require additional model calls but lack specialized bias expertise; RBD is a purpose-trained "bias expert."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The external reasoning-based bias detection module represents an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 bias types × 8 judges × 4 model sizes, with diagnostic sets and cross-domain generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures and tables are clear; the method pipeline is described in detail; bias dataset construction is transparent.
- Value: ⭐⭐⭐⭐⭐ Directly and significantly advances practical LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](../../ACL2025/social_computing/translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[ACL 2025\] How does Misinformation Affect Large Language Model Behaviors and Preferences?](../../ACL2025/social_computing/how_does_misinformation_affect_large_language.md)

</div>

<!-- RELATED:END -->
