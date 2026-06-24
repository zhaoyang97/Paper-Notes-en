---
title: >-
  [Paper Note] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs
description: >-
  [ACL 2025][LLM Alignment][jailbreak] This paper proposes the M2S framework, which compresses multi-turn human jailbreak conversations into single-turn prompts using three simple format conversion methods (Hyphenize/Numberize/Pythonize). This approach not only maintains but even exceeds the original multi-turn attack effectiveness (achieving an ASR up to 95.9%, improving by up to 17.5% over multi-turn attacks) while reducing token usage by more than half.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "jailbreak"
  - "red teaming"
  - "multi-turn"
  - "single-turn conversion"
  - "contextual blindness"
date: 2026-05-08
content_hash: a513d092c7de859b
---

# M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.04856](https://arxiv.org/abs/2503.04856)  
**Code**: [https://github.com/Junuha/M2S_DATA](https://github.com/Junuha/M2S_DATA)  
**Area**: LLM Alignment  
**Keywords**: jailbreak, red teaming, multi-turn, single-turn conversion, contextual blindness

## TL;DR
This paper proposes the M2S framework, which compresses multi-turn human jailbreak conversations into single-turn prompts using three simple format conversion methods (Hyphenize/Numberize/Pythonize). This approach not only maintains but even exceeds the original multi-turn attack effectiveness (achieving an ASR up to 95.9%, improving by up to 17.5% over multi-turn attacks) while reducing token usage by more than half.

## Background & Motivation

**Background**: Multi-turn human jailbreak attacks are highly effective (ASR ~70%, bypassing SOTA defenses), whereas automated single-turn jailbreaks have an ASR close to 0% under strong defense. The Crescendo multi-turn strategy achieves a 98% ASR on GPT-4.

**Limitations of Prior Work**: Although multi-turn jailbreaking is highly effective, it requires substantial human effort (professional red team operation) and time, rendering it difficult to deploy at scale. Conversely, single-turn jailbreaks are efficient but perform poorly. A fundamental trade-off exists between effectiveness and efficiency.

**Key Challenge**: How to maintain the high effectiveness of multi-turn jailbreaking while achieving the high efficiency of single-turn jailbreaking?

**Goal**: To systematically convert multi-turn jailbreak conversations into single-turn prompts for the first time, balancing both effectiveness and efficiency.

**Key Insight**: It is observed that the core of multi-turn jailbreaking lies in the step-by-step evolution of prompt sequences rather than iterative feedback from intermediate responses. Therefore, multi-turn prompt sequences can be directly concatenated into a structured single-turn input.

**Core Idea**: Flatten multi-turn conversations into a single turn using list, numbered, or code formats, leveraging "contextual blindness" to bypass safety guardrails.

## Method

### Overall Architecture
Input: Multi-turn jailbreak conversation $(P_1, A_1, P_2, A_2, ..., P_n)$. M2S Conversion: Remove intermediate responses $A_i$, and concatenate $P_1, P_2, ..., P_n$ into a single-turn prompt using one of the three formats. Output: The LLM's response $A$ to the single-turn prompt.

### Key Designs

1. **Hyphenize (List Format)**:

    - Format each prompt into a list prefixed by hyphens ("-").
    - Advantage: Simple and clear; most LLMs can correctly comprehend list semantics.

2. **Numberize (Numbered Format)**:

    - Replace hyphens with numerical indices to explicitly reinforce sequential dependency.
    - Advantage: Ensures LLMs process each sub-prompt sequentially.

3. **Pythonize (Code Format)**:

    - Encapsulate the conversation into an iterable Python list structure: `prompts = ["...", "...", ...]` with a for-loop traversal and a print statement.
    - Advantage: Leverages the uniqueness of code formatting, as LLMs may relax safety filters under a "code execution mode". This method achieves the highest ASR in experiments.

4. **Ensemble Strategy**: Take the highest harmfulness score among the three methods to further improve the ASR.

### Evaluation Framework
- StrongREJECT evaluator (continuous 0-1 harmfulness score)
- ASR threshold of 0.25 (determined by F1 optimization on human annotations)
- Perfect-ASR (the ratio of instances achieving a score of 1.0)

## Key Experimental Results

### Main Results (MHJ Dataset)

| Model | Method | ASR (%) | Perfect-ASR (%) | Average Score |
|------|------|:---:|:---:|:---:|
| GPT-4o | Original (Multi-turn) | 71.5 | 39.3 | 0.62 |
| GPT-4o | Hyphenize (M2S) | 81.4 (+9.9) | 36.7 | 0.70 |
| GPT-4o | Pythonize (M2S) | 85.8 (+14.3) | 44.7 | 0.76 |
| GPT-4o | **Ensemble (M2S)** | **89.0 (+17.5)** | **57.5** | **0.82** |
| Llama-3-70b | Original | 67.0 | 16.0 | 0.51 |
| Llama-3-70b | Ensemble (M2S) | Significant Improvement | Significant Improvement | Significant Improvement |

### Ablation Study

| Finding | Key Metrics | Description |
|------|---------|------|
| Pythonize is the strongest | ASR of 85.8% on GPT-4o | Code format is most effective at inducing safety evasion |
| M2S token consumption halved+ | Average reduction of >50% | Removing intermediate responses drastically reduces costs |
| Bypassing LlamaGuard | M2S bypass rate is significantly higher than multi-turn | Safety guardrails are "blind" to structured formats |
| Tactical Retention Analysis | Specific attack tactics are more effective in M2S | Nested formats enhance adversarial effectiveness |

### Key Findings
- **Single-turn can outperform multi-turn**: Counter-intuitively, the ASR increases (up to +17.5%) after removing iterative feedback. This suggests that the core of multi-turn jailbreaks is not interactive adaptation but rather the adversarial design of the prompt sequences themselves.
- **Contextual blindness is a critical vulnerability**: Safety guardrail models (e.g., LlamaGuard) detect malicious content turn-by-turn. However, when malicious sequences are embedded into list or code structures, detection fails.
- **Code format is the most dangerous**: Pythonize achieves the highest ASR across almost all models, implying that LLMs employ more lenient safety filters when processing code-formatted inputs.
- **Substantial advantage in token efficiency**: API costs are directly cut by more than half while achieving higher ASR, which is highly practical for large-scale red teaming.

## Highlights & Insights
- **Counter-intuitive finding of "simpler is more effective"**: All three M2S methods are simple, rule-based format conversions requiring no LLM or optimization, yet they consistently outperform carefully crafted multi-turn attacks.
- **Security implications of contextual blindness**: Current safety guardrails are designed under the assumption that malicious content appears in natural conversations. They show significant deficiencies in detecting structured or code-based formats, underscoring the need for a new defense paradigm.
- **Provides a practical tool for large-scale red teaming**: By compressing attacks that previously required multiple expert interactions into a single API call, automated security auditing becomes feasible.

## Limitations & Future Work
- It relies on existing multi-turn jailbreak datasets (e.g., MHJ) and cannot automatically generate new attacks.
- M2S removes intermediate responses $A_i$, which may reduce effectiveness for attacks that depend on model feedback to adjust strategies.
- Only four LLMs and one safety guardrail were evaluated; coverage could be further expanded.
- Discussion on mitigation and defense strategies is limited.

## Related Work & Insights
- **vs Crescendo**: Crescendo achieves a 98% ASR through progressive multi-turn jailbreaks but requires iterative interaction; M2S reaches an 89% ASR in a single turn while dramatically reducing costs.
- **vs GCG/AutoDAN**: These automated single-turn methods achieve close to 0% ASR under strong defenses, whereas M2S maintains the high ASR of multi-turn attacks.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic multi-turn to single-turn conversion, with the counter-intuitive discovery of outperforming the original attacks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested across multiple models, strategies, safety guardrails, and token analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and rigorous evaluation (F1-optimized threshold, Perfect-ASR).
- Value: ⭐⭐⭐⭐⭐ Significant implications for both red-teaming practices and safety defense designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)
- [\[ACL 2025\] Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming](constitutional_classifiers_defending_against_universal_jailbreaks_across_thousan.md)

</div>

<!-- RELATED:END -->
