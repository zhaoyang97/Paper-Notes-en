---
title: >-
  [Paper Note] ExtendAttack: Attacking Servers of LRMs via Extending Reasoning
description: >-
  [AAAI 2026][Reasoning][LRM Security] This paper proposes ExtendAttack, a resource exhaustion attack targeting Large Reasoning Models (LRMs): by randomly converting characters in the prompt into multi-base ASCII encodings, the attack forces models to perform extensive character-by-character decoding before answering, increasing o3's response length by more than 2.7× and doubling latency, while keeping answer accuracy largely intact.
tags:
  - "AAAI 2026"
  - "Reasoning"
  - "LRM Security"
  - "Resource Exhaustion Attack"
  - "Reasoning Extension"
  - "Adversarial Attack"
  - "DDoS"
date: 2026-05-08
content_hash: 81b0ca9f3dc722f2
---

# ExtendAttack: Attacking Servers of LRMs via Extending Reasoning

**Conference**: AAAI 2026
**arXiv**: [2506.13737](https://arxiv.org/abs/2506.13737)  
**Code**: [GitHub](https://github.com/zzh-thu-22/ExtendAttack)  
**Area**: LLM Reasoning
**Keywords**: LRM Security, Resource Exhaustion Attack, Reasoning Extension, Adversarial Attack, DDoS

## TL;DR
This paper proposes ExtendAttack, a resource exhaustion attack targeting Large Reasoning Models (LRMs): by randomly converting characters in the prompt into multi-base ASCII encodings, the attack forces models to perform extensive character-by-character decoding before answering, increasing o3's response length by more than 2.7× and doubling latency, while keeping answer accuracy largely intact.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) represented by OpenAI o1 and DeepSeek-R1 have achieved breakthrough performance on complex tasks such as mathematics and coding through extended chain-of-thought reasoning, yet these lengthy reasoning processes inherently consume substantial computational resources.

**Emerging Threat**: Traditional adversarial attacks focus on content manipulation (e.g., jailbreaking), but an emerging threat class targets the *computational process itself*—maliciously extending a model's reasoning chain to exhaust server resources, analogous to DDoS attacks in network security. For services offering free APIs (e.g., Google AI Studio), such attacks pose serious economic threats.

**Limitations of Prior Work**: The most representative prior method, OverThinking, injects a context-irrelevant "decoy task" (e.g., solving an MDP), but suffers from a dual failure mode: (a) stronger models such as o3 recognize and ignore the fixed-pattern decoy, rendering the attack ineffective; (b) weaker models are distracted by the off-topic instruction, causing accuracy to collapse (QwQ-32B drops from 63.3% to 15.3% on BCB-C)—the attack succeeds but is immediately detectable.

**Key Challenge**: Resource exhaustion attacks must simultaneously satisfy two conflicting objectives—maximizing computational overhead (effectiveness) and preserving answer correctness (stealthiness)—a balance OverThinking cannot achieve.

**Key Insight**: Rather than injecting an external decoy, this work embeds a computationally intensive task directly within the semantic structure of the user query, forcing models to expend substantial reasoning at the *comprehension* stage itself through character-level obfuscation.

**Core Idea**: Characters in the prompt are randomly converted into multi-base ASCII encodings (e.g., the letter `a` → `<(7)141>`). The model must identify the base, convert to decimal, map to ASCII, and recover the original character for each symbol—a decoding process that is semantically inseparable from "understanding the question" and therefore cannot be skipped.

## Method

### Overall Architecture
The input is a normal user prompt $Q$; the output is an obfuscated adversarial prompt $Q'$ such that the LRM response $Y' = R' \oplus A'$ satisfies: (1) $L(Y') \gg L(Y)$ (substantially increased response length); (2) $\text{Acc}(A') \approx \text{Acc}(A)$ (accuracy largely preserved). The attack is purely black-box, requiring only API access.

The attack follows a four-step pipeline: character segmentation → probabilistic selection → multi-base encoding → concatenation into the adversarial prompt.

### Key Designs

1. **Probabilistic Character Selection**:

    - **Function**: Given an obfuscation rate $\rho \in [0,1]$, randomly samples $k = \lceil |\mathcal{S}_{valid}| \cdot \rho \rceil$ characters from the transformable character set $\mathcal{S}_{valid}$ (alphanumeric, excluding special symbols) for obfuscation.
    - **Design Motivation**: (a) Partial obfuscation retains sufficient readable context for the model to grasp the question's intent; (b) random selection increases the unpredictability of the attack pattern, countering rule-based filtering defenses. $\rho \in [0.4, 0.6]$ achieves the best trade-off between effectiveness and stealthiness.

2. **Poly-Base ASCII Transformation**:

    - **Function**: For each selected character $c_j$, its decimal ASCII value $d_j$ is computed; a base $n_j$ is then randomly drawn from $\mathcal{B} = \{2,\ldots,9,11,\ldots,36\}$ (excluding base 10), $d_j$ is converted to base $n_j$, and formatted as `<(n_j)val>`.
    - **Mechanism**: Each character is encoded in a different random base, preventing the model from learning a single repeated decoding pattern; the model must independently execute the three-step computation—identify base → convert → map ASCII—for every character.
    - **Design Motivation**: The decoding task is semantically equivalent to "understanding the question," making it impossible for the model to bypass it, unlike an external decoy that can simply be ignored.

3. **Decoding Instruction Note ($\mathcal{N}_{note}$)**:

    - **Function**: A brief explanatory note appended to the obfuscated prompt, informing the model that content within `<>` is a number in a given base, the base value appears in `()`, and the result corresponds to an ASCII character code.
    - **Design Motivation**: This is the key trigger for full decoding reasoning chains. Without the note, models exploit contextual shortcuts (e.g., seeing `import p<(13)76>ndas` and directly guessing `pandas`), shortening the decoding reasoning chain by approximately 30%. With the note, models are compelled to perform complete character-by-character mathematical conversion.
    - **Notable Trade-off**: $\mathcal{N}_{note}$ simultaneously increases attack effectiveness (longer reasoning) and improves accuracy (more correct decoding)—a rare case where attack effectiveness and accuracy are positively correlated.

### Attack Scenarios
Two scenarios are supported: (1) direct submission of the obfuscated prompt; (2) indirect injection—planting obfuscated text in public documents or wikis so that RAG systems retrieve and feed it to the LRM.

## Key Experimental Results

### Main Results
Evaluated on 4 models (o3, o3-mini, QwQ-32B, Qwen3-32B) × 4 benchmarks (AIME24/25, HumanEval, BCB-C):

| Benchmark | Model | Original Length | ExtendAttack Length | Ratio | Original Acc | Attack Acc | OverThinking Acc |
|-----------|-------|----------------|--------------------|----|------------|-----------|----------------|
| HumanEval | o3 | 769 | 2,153 | **2.8×** | 97.6% | **97.6%** | 97.0% |
| HumanEval | QwQ-32B | 2,823 | 5,266 | 1.9× | 97.0% | **97.0%** | 73.8% ↓23.2 |
| HumanEval | Qwen3-32B | 3,413 | 5,535 | 1.6× | 97.6% | **97.6%** | 65.9% ↓31.7 |
| AIME24 | o3 | 8,571 | 11,798 | 1.4× | 90.8% | **86.7%** | 85.0% |
| BCB-C | QwQ-32B | 4,535 | 8,891 | 2.0× | 63.3% | **64.0%** | 15.3% ↓48.0 |

### Ablation Study

| Ablation | Model | Response Length | Accuracy |
|----------|-------|----------------|---------|
| Full ($\rho$=0.5, with $\mathcal{N}_{note}$) | QwQ-32B | 8,891 | 64.0% |
| Without $\mathcal{N}_{note}$ | QwQ-32B | 5,122 | 62.7% |
| Full | Qwen3-32B | 7,739 | 63.3% |
| Without $\mathcal{N}_{note}$ | Qwen3-32B | 5,347 | 58.7% |

### Key Findings
- **Dual failure of OverThinking**: Strong models (o3) recognize and ignore the decoy, rendering the attack ineffective; weaker models (QwQ, Qwen3) are distracted, causing accuracy to collapse (QwQ drops from 63.3% to 15.3% on BCB-C). ExtendAttack exhibits neither failure mode.
- **$\mathcal{N}_{note}$ is essential**: Without the decoding note, models tend to exploit context to guess the original words (shortcut) rather than performing full base conversion. Adding the note increases response length by ~40% and also improves accuracy.
- **Optimal range for $\rho$ is 0.4–0.6**: Higher obfuscation rates no longer increase response length (models may abandon decoding) and further reduce accuracy.
- **o3 is most vulnerable to ExtendAttack**: On HumanEval, latency increases from 17 s to 36 s, length grows by 2.8×, while accuracy is perfectly preserved at 97.6%.

## Highlights & Insights
- The core insight of the attack design is elegant: by embedding the decoding task on the unavoidable path of "understanding the question," the model cannot distinguish "decoding obfuscated characters" from "comprehending the problem" at the semantic level, making it fundamentally harder to defend against than the OverThinking paradigm.
- The finding that stronger models are more vulnerable is counterintuitive and significant—o3 is exploited precisely because it more faithfully follows decoding instructions. This challenges the common assumption that "stronger models are safer."
- The attack is fully black-box, gradient-free, applicable to arbitrary LRMs, and supports indirect injection via RAG pipelines, yielding a broad practical threat surface.

## Limitations & Future Work
- The presence of $\mathcal{N}_{note}$ constitutes a conspicuous attack fingerprint detectable by simple pattern matching. Although the authors suggest that future stronger models may not require the note, this remains the primary weakness of the current approach.
- The defense discussion is relatively shallow, analyzing only the limitations of pattern matching, perplexity filtering, and guardrail models, without proposing effective countermeasures.
- Experiments are limited to code and mathematics tasks; effectiveness on natural language tasks (e.g., writing, translation) is unverified—character-level obfuscation in such tasks may be more easily "guessed around" by models.
- The impact of the attack on token costs is not quantified—the obfuscated prompt is itself longer, increasing the attacker's input cost as well.
- Only four LRMs are tested (o3, o3-mini, QwQ-32B, Qwen3-32B); results are not validated on other reasoning models such as DeepSeek-R1 and Claude.
- The composability of the attack is unexplored—whether it can be combined with jailbreak attacks to simultaneously achieve content manipulation and resource exhaustion is an open question.
- Systematic validation of attack effectiveness in realistic multi-user concurrent scenarios (e.g., whether it can actually cause service degradation) is absent.

## Related Work & Insights
- **vs. OverThinking**: Injects an external decoy task; strong models recognize and ignore it while weaker models suffer accuracy collapse. ExtendAttack embeds the computational burden within the semantic structure, making it effectively unavoidable for both strong and weak models while preserving accuracy.
- **vs. CatAttack**: Extends reasoning by appending irrelevant facts, but the primary effect is reducing accuracy rather than increasing computation. ExtendAttack is specifically designed to preserve accuracy.
- **vs. Jailbreak Attacks**: Traditional jailbreaks target content safety; this work opens a new dimension of *computational safety*—the attack objective is not what the model says, but how long it computes.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Embedding resource exhaustion attacks within the semantic structure is a highly original idea that reveals a fundamental vulnerability in LRM reasoning mechanisms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four models × four benchmarks with comprehensive ablations, but natural language tasks and defense experiments are missing.
- **Writing Quality**: ⭐⭐⭐⭐ The threat model is formally and clearly defined; the method is presented with step-by-step derivation; figures effectively illustrate the three-scenario comparison.
- **Value**: ⭐⭐⭐⭐⭐ Provides direct and practical security warnings for LRM service providers, particularly platforms offering free APIs.

## Supplementary Notes
- **Defense direction**: Input normalization before inference (restoring ASCII-encoded characters to their original form) may be the lowest-cost mitigation measure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](../../NeurIPS2025/llm_reasoning/let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)
- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](text-to-scene_with_large_reasoning_models.md)
- [\[AAAI 2026\] A Reasoning Paradigm for Named Entity Recognition](a_reasoning_paradigm_for_named_entity_recognition.md)
- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)

</div>

<!-- RELATED:END -->
