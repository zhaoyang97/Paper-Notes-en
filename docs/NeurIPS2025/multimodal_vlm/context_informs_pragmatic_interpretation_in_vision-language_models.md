---
title: >-
  [Paper Note] Context Informs Pragmatic Interpretation in Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Pragmatic reasoning] This work systematically evaluates the pragmatic reasoning capabilities of VLMs using iterated reference games. Models perform substantially worse than humans in the absence of context, but can rapidly leverage relevant dialogue history to achieve approximately 80% accuracy, revealing a strong dependence on contextual information.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Pragmatic reasoning
  - reference games
  - context sensitivity
  - cognitive evaluation of VLMs
  - abstract visual reasoning
date: 2026-05-08
content_hash: ac33065ab88126be
---

# Context Informs Pragmatic Interpretation in Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.03908](https://arxiv.org/abs/2511.03908)
**Code**: [GitHub](https://github.com/benpry/vlm-tg-context/tree/coginterp)
**Area**: Multimodal VLM
**Keywords**: Pragmatic reasoning, reference games, context sensitivity, cognitive evaluation of VLMs, abstract visual reasoning

## TL;DR

This work systematically evaluates the pragmatic reasoning capabilities of VLMs using iterated reference games. Models perform substantially worse than humans in the absence of context, but can rapidly leverage relevant dialogue history to achieve approximately 80% accuracy, revealing a strong dependence on contextual information.

## Background & Motivation

Multi-turn dialogue is a central feature of human communication—shared conversational history underpins the formation of semantic conventions. **Iterated reference games** are a classical paradigm for studying this phenomenon: a director must describe a target figure in language so that a matcher can correctly select it from among multiple options. As the game progresses, participants develop concise, conventionalized expressions.

This capability is critical for AI dialogue systems and requires two behavioral properties:

**Pragmatic interpretation**: Understanding the intended meaning of an utterance in context

**Context sensitivity**: Leveraging prior interaction history to guide current comprehension

Nevertheless, reference games involving abstract figures (tangrams) remain highly challenging for AI systems, particularly in few-shot settings. This study is the first systematic comparison of human and state-of-the-art open-weight VLM performance on pragmatic reasoning in iterated reference games.

## Method

### Overall Architecture

**Data**: The study uses the iterated reference game dataset from Boyce et al. In each game, players view a grid of 12 tangram figures; the director describes a highlighted target, and the matcher selects it. Each game involves 2–6 players across 6 rounds totaling 72 trials.

**Evaluation setup**: Ten games are selected, and four open-weight VLMs are evaluated:

- Qwen 2.5 VL 32B
- Gemma 3 27B
- Llama 3.2 11B
- Kimi VL A3B

Each model receives a system prompt, a concatenated image of 12 labeled figures, and the dialogue history of prior trials (as chat history), and outputs log probabilities over letters A–L. Accuracy is measured as the normalized probability assigned to the correct target.

### Key Designs

**Eight control conditions** (systematically manipulating the quantity, order, and relevance of context):

| Condition | Context Source | Same Game | Trial Order | Target Visible in History |
|---|---|---|---|---|
| Yoked | Same game | ✓ | Original order | ✓ |
| Shuffled | Same game | ✓ | Randomly shuffled | ✓ |
| Backward | Same game | ✓ | Reversed | ✓ |
| Ablated | Same game | ✓ | Original order | ✗ |
| Other-within | Different game (single) | ✓ | Original order | ✓ |
| Other-across | Different game (multiple) | ✗ | Original order | ✓ |
| Random | Different game (multiple) | ✗ | Randomly shuffled | ✓ |
| No context | None | ✗ | N/A | ✗ |

### Human Baselines

- **Original players**: Interactive participants from the original games
- **Naïve matchers**: Human participants who read only the dialogue transcripts without having participated in the original game
    - Yoked ($N=99$), Shuffled ($N=97$), Backward ($N=89$), Random ($N=107$)

## Key Experimental Results

### Main Results

**Model vs. human accuracy across conditions** (steady-state estimates):

| Condition | Human (Original) | Human (Naïve) | Gemma 3 | Qwen 2.5 | Llama 3.2 | Kimi VL |
|---|---|---|---|---|---|---|
| No context | ~0.75 | — | ~0.15 | ~0.15 | ~0.12 | ~0.12 |
| Yoked (late rounds) | ~0.95 | ~0.75 | ~0.80 | ~0.40 | ~0.75 | ~0.75 |
| Other-within | — | — | ~0.40 | ~0.35 | ~0.30 | ~0.35 |

**Model–human correlation** (trial-level):

| Condition | Human Split-Half | Qwen 2.5 | Gemma 3 | Llama 3.2 | Kimi VL |
|---|---|---|---|---|---|
| Yoked | .42 [.32, .50] | .10 | .20 | .25 | .27 |
| Backward | .48 [.40, .56] | .23 | .31 | .40 | .35 |
| Random | — | .61 | .58 | .55 | .57 |

### Ablation Study

**Context relevance analysis** (core systematic manipulation):

| Comparison | Finding |
|---|---|
| Yoked vs. Shuffled | Shuffling order degrades performance, mirroring human behavior |
| Yoked vs. Backward | Models outperform their shuffled baseline under reversed order, contrary to humans |
| Yoked vs. Other-within | Cross-game context substantially reduces accuracy (0.3–0.5) |
| Yoked vs. Ablated | Removing prior descriptions of the target figure causes a large performance drop |

### Key Findings

1. **Models substantially underperform humans without context**: Human accuracy is approximately 0.75 upon first exposure to a description; models perform only marginally above chance (0.083).
2. **Models can rapidly exploit relevant context**: Under the Yoked condition, most models reach approximately 0.80 accuracy in later rounds.
3. **Context must originate from the same game**: Cross-game context provides little benefit (0.3–0.5), indicating that conventions are game-specific.
4. **Model and human error patterns diverge**: Trial-level correlations are weak ($r = .10$–$.40$), far below inter-human consistency.

## Highlights & Insights

- **Elegant experimental design**: Eight control conditions systematically isolate the effects of context quantity, order, and relevance.
- **Cognitive science perspective**: Pragmatic reasoning is situated within a classical paradigm from human communication research rather than treated as a purely model-comparison exercise.
- **Revealing fundamental model differences**: Humans exhibit an intuitive zero-shot understanding (~0.75), whereas VLMs are heavily reliant on explicit context.
- **Intriguing finding in the Backward condition**: Models are better than humans at back-inferring early-game expressions from conventionalized ones, suggesting qualitatively different reasoning mechanisms.
- **Methodological contribution**: The study provides a reproducible framework for evaluating context sensitivity and pragmatic reasoning in VLMs.

## Limitations & Future Work

- Only abstract tangram figures are tested; generalization to naturalistic images remains unverified.
- Models receive richer feedback than humans (models observe the correct answer, whereas human participants receive only binary correctness feedback).
- Only comprehension is evaluated; generating appropriate descriptions constitutes a harder and unaddressed task.
- Only four open-weight models are assessed.
- As a workshop paper, the scope is limited and certain analyses lack depth.

## Related Work & Insights

- **Clark (1996)**: Common ground theory provides the cognitive foundation for this work.
- **Hawkins et al. (2021)**: A hierarchical model of convention formation from partner-specific to community-level representations.
- **Gul et al. (2024) CoGen**: Attempts to train models on the production side of reference games, with still-limited success.
- Implication: VLM "comprehension" may reflect pattern matching rather than genuine pragmatic reasoning; future work should distinguish in-context learning from true pragmatic adaptation.

## Rating

- ⭐ Novelty: 4/5 — First systematic evaluation of VLMs on pragmatic reasoning in iterated reference games, with a well-designed experimental framework.
- ⭐ Experimental Thoroughness: 3/5 — Rich control conditions, but limited in number of models and stimulus types.
- ⭐ Writing Quality: 4/5 — Clear logic, intuitive figures, and adequate exposition of the cognitive science background.
- ⭐ Value: 3/5 — Moderate depth as a workshop paper, but opens a meaningful direction for evaluating pragmatic capabilities in VLMs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](../../CVPR2026/multimodal_vlm/phantasia_context-adaptive_backdoors_in_vision_language_models.md)

<!-- RELATED:END -->
