---
title: >-
  [Paper Note] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark
description: >-
  [ICLR 2026][LLM Safety][privacy awareness] This paper proposes EAPrivacy — the first 4-tier benchmark for evaluating LLM physical-world privacy awareness (400+ procedurally generated scenarios, 60+ physical scenes). It finds that all frontier models exhibit "asymmetric conservatism" (over-cautious on task execution yet insufficient on privacy protection), that enabling reasoning/thinking mode actually degrades privacy performance, and that the best model (Gemini 2.5 Pro) achieves only 59% accuracy in dynamic environments.
tags:
  - ICLR 2026
  - LLM Safety
  - privacy awareness
  - embodied agent
  - physical privacy
  - contextual integrity
  - benchmark
  - PDDL
date: 2026-05-08
content_hash: 4c21e7cb30c30be8
---

# Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark

**Conference**: ICLR 2026
**arXiv**: [2510.02356](https://arxiv.org/abs/2510.02356)
**Code**: [GitHub](https://github.com/Graph-COM/EAPrivacy)
**Area**: AI Safety / Privacy / Embodied Intelligence
**Keywords**: privacy awareness, embodied agent, physical privacy, contextual integrity, benchmark, PDDL

## TL;DR
This paper proposes EAPrivacy — the first 4-tier benchmark for evaluating LLM physical-world privacy awareness (400+ procedurally generated scenarios, 60+ physical scenes). It finds that all frontier models exhibit "asymmetric conservatism" (over-cautious on task execution yet insufficient on privacy protection), that enabling reasoning/thinking mode actually degrades privacy performance, and that the best model (Gemini 2.5 Pro) achieves only 59% accuracy in dynamic environments.

## Background & Motivation
**Background**: LLMs are increasingly deployed as embodied agents (home robots, medical assistants, office robots) operating in physical spaces. Existing privacy benchmarks (e.g., Mireshghallah 2023) only evaluate text-level privacy leakage.

**Limitations of Prior Work**:
   - **Physical privacy ≠ textual privacy**: Physical-world privacy requires spatial reasoning (e.g., "a diary is on the desk"), contextual integrity judgment (e.g., "should not start cleaning while a meeting is in progress"), and multimodal perception (e.g., "hearing faint conversation")
   - **Task–privacy conflicts are unevaluated**: An agent instructed to "clear the desk" encounters a hidden surprise gift on it — how should it balance the two?
   - **Social norms vs. privacy**: A neighbor's apartment emits screams — should the agent report it (sacrificing privacy) or ignore it (respecting privacy)?
   - Aligned LLMs perform well on textual privacy benchmarks (Gemini/GPT-4's secret disclosure rate can reach 0%), yet physical privacy poses an entirely different challenge

**Key Challenge**: In the physical world, privacy is not a static rule but a dynamic social contract that depends on context and requires reasoning — the question is whether LLMs possess this reasoning capacity.

**Core Idea**: Construct a 4-tier progressive evaluation using procedurally generated physical scenes in PDDL format (encoding spatial relationships and multimodal perception cues), ranging from simple sensitive-object identification to complex ethical dilemmas.

## Method

### Overall Architecture
A 4-tier progressive design covering different levels of cognitive complexity in physical privacy:

### 4-Tier Design

1. **Tier 1: Sensitive Object Identification**
    - Function: Identify sensitive objects (e.g., social security cards, passports) among 3–30 distractors on desks or in containers
    - Input: Spatial relationships among objects in PDDL format (not natural-language descriptions)
    - Evaluation: True positive rate, false positive rate, and spatial localization accuracy
    - Clutter levels (3/5/10/30 distractors) are varied to assess the effect of environmental complexity on privacy awareness
    - Design Motivation: The most fundamental physical-privacy capability — can the agent "see" what is privacy-sensitive in a real scene?

2. **Tier 2: Privacy Reasoning in Dynamic Environments**
    - Function: Judge the appropriateness (scored 1–5) of a given action under different contexts
    - Input: Multimodal perception cues (Visual: "5 people at table"; Audio: "continuous speech"), simulating the perception of a physical agent
    - Evaluation modes: (i) Rating Mode (MAD against human scores); (ii) Selection Mode (choose the most appropriate action from three options)
    - Coverage: park / library / private residence × cleaning / security patrol / food delivery × normal→emergency / empty room→private meeting
    - Design Motivation: Privacy is context-dependent — "start cleaning" is appropriate in an empty room but not during a private meeting

3. **Tier 3: Inferential Privacy and Task Conflicts**
    - Function: The agent must infer implicit privacy constraints from multimodal cues (Theory of Mind) and respect them while executing tasks
    - Example scenario: Observing someone hide a gift under a desk → instructed to "move everything off the desk" → should skip the gift
    - Evaluation: (i) privacy violation rate; (ii) task completion rate (how much of the task is completed without violating privacy); (iii) three-way selection (one privacy-violating action vs. two non-violating alternatives)
    - Design Motivation: Closest to real deployment — the agent must simultaneously satisfy explicit instructions and implicit privacy constraints

4. **Tier 4: Social Norms vs. Privacy Ethical Dilemmas**
    - Function: In high-stakes scenarios, critical social norms (public safety, child protection) should override individual privacy
    - Example scenario: Hearing "a cry for help" from a neighboring apartment + observing "erratic silhouettes" → should call the police (sacrificing the neighbor's privacy)
    - Evaluation: Binary judgment accuracy
    - Grounded in U.S. legal and social normative frameworks, with acknowledgment of cross-cultural variation
    - Design Motivation: Test the agent's ethical judgment under extreme circumstances

### Technical Characteristics
- **Procedural generation**: 400+ scenarios across 60+ unique physical scenes (offices, laboratories, homes, etc.)
- **PDDL format**: Structured representation of physical spatial relationships, beyond pure text narration
- **Multimodal perception simulation**: Visual / Audio / Action cues simulating the real perception of an embodied agent
- **Human annotation validation**: Five PhD-level annotators to establish ground truth

## Key Experimental Results

### Main Results (16 Models × 4 Tiers)

| Tier | Best Model | Performance | Key Challenge |
|------|-----------|-------------|---------------|
| Tier 1 | Gemini 2.5 Pro | 96%→26% (3→30 distractors) | Collapses in cluttered environments |
| Tier 2 | Gemini 2.5 Pro | 59% Selection Acc, MAD=1.32 | Insufficient dynamic context understanding |
| Tier 3 | Gemini 2.5 Flash | 71% privacy violation rate (best) | All models severely insufficient |
| Tier 4 | Multiple models | 81–95% accuracy | Relatively easier but still gaps remain |

### Core Finding: Asymmetric Conservatism

| Dimension | Performance | Explanation |
|-----------|------------|-------------|
| Task execution | Over-cautious (Tier 3 task completion rate near 0%) | Models "prefer refusing tasks over making mistakes" |
| Privacy protection | Severely insufficient (violation rate 71–98%) | Models simultaneously fail to protect privacy |
| Overall outcome | Neither tasks nor privacy handled well | Over-safe and under-safe coexist |

### Thinking Mode Degradation (Counter-intuitive Finding)

| Model | Standard Mode | Thinking Mode | Change |
|-------|-------------|---------------|--------|
| Gemini 2.5 Pro | Baseline | Declines across Tier 1–3 | Reasoning introduces over-interpretation |
| Claude 3.5 | Baseline | Similar degradation | — |

### Key Findings
- **Asymmetric conservatism is the most important finding**: Models are over-cautious about "doing things" (Tier 3 completion rate near 0% — almost all potentially privacy-related tasks are refused) yet insufficiently cautious about "protecting privacy" (violation rate 71–98%) — both types of errors occur simultaneously
- **Thinking/Reasoning mode degradation** (Tier 1–3): Enabling reasoning mode leads to worse performance — likely because longer reasoning chains increase false positives (labeling irrelevant objects as sensitive) and over-interpretation (judging normal actions as inappropriate)
- **Sensitivity to environmental complexity**: Accuracy is 96% with 3 distractors but drops to 26% with 30 — physical scene complexity is a key bottleneck
- Textual privacy ≠ physical privacy: Models with 0% disclosure rates on text benchmarks exhibit severe deficiencies in physical privacy
- GPT-4o and Claude-3.5-haiku ignore social norms in more than 15% of Tier 4 cases

## Highlights & Insights
- **Deeper implication of "asymmetric conservatism"**: Current alignment training creates a distorted safety posture — models have learned "refusal" as a safety strategy but have not learned to "actively protect" privacy. This is a systemic bias of RLHF
- **Pioneering nature of physical privacy evaluation**: Extending privacy assessment from text to the physical world, and using PDDL combined with multimodal cues to simulate embodied perception, represents an important paradigm shift in evaluation
- **Thinking mode degradation as a warning for scaling reasoning**: More reasoning is not always better — in privacy scenarios that require "common sense" rather than "deep analysis," reasoning may over-complicate straightforward judgments

## Limitations & Future Work
- The framework is grounded solely in U.S. legal and social normative frameworks; cross-cultural applicability requires further exploration
- The PDDL-based physical descriptions differ from real visual perception — no actual images or videos are used
- The scale of 400+ scenarios remains limited relative to the complexity of the physical world
- The "correct answers" in Tier 4 ethical dilemmas may be contested across cultures and personal value systems
- Truly embodied systems (robots) are not tested; only the text-based reasoning of LLMs is evaluated

## Related Work & Insights
- **vs. Mireshghallah 2023 (textual privacy)**: That work only tests contextual integrity of information flows; EAPrivacy extends evaluation to spatial reasoning and multimodal perception in the physical world
- **vs. robot safety evaluation (Robey 2024 et al.)**: Prior work primarily focuses on jailbreaks and adversarial attacks; EAPrivacy addresses privacy-awareness deficiencies under normal usage
- **Implications for embodied AI deployment**: Current LLMs lack the physical-privacy reasoning capability required for deployment in private spaces — dedicated physical-privacy alignment training is needed

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First physical-world privacy evaluation; the 4-tier design is systematic and theoretically grounded (contextual integrity)
- Experimental Thoroughness: ⭐⭐⭐⭐ 16 models × 400+ scenarios × human annotation validation
- Writing Quality: ⭐⭐⭐⭐ Failure modes are clearly categorized; findings are insightful
- Value: ⭐⭐⭐⭐⭐ Important implications for the safe deployment of embodied AI; reveals fundamental deficiencies in alignment

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_models.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](../../AAAI2026/llm_safety/sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)

<!-- RELATED:END -->
