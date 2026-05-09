---
title: >-
  [Paper Note] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models
description: >-
  [ACL 2026][Multimodal VLM][Embodied Safety] This paper proposes the SafetyALFRED benchmark, which introduces six categories of kitchen safety hazards into the ALFRED embodied task setting. It reveals a critical alignment gap: multimodal large language models can identify hazards in static QA (up to 92%) but fail to proactively mitigate them during embodied planning (<60%), advocating a paradigm shift from QA-based to embodied safety evaluation.
tags:
  - ACL 2026
  - Multimodal VLM
  - Embodied Safety
  - Hazard Mitigation
  - Multimodal Evaluation
  - Safety Planning
  - ALFRED
date: 2026-05-08
content_hash: c9e105c360238317
---

# SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.19638](https://arxiv.org/abs/2604.19638)
**Code**: [https://github.com/sled-group/SafetyALFRED](https://github.com/sled-group/SafetyALFRED)
**Area**: Multimodal VLM
**Keywords**: Embodied Safety, Hazard Mitigation, Multimodal Evaluation, Safety Planning, ALFRED

## TL;DR
This paper proposes the SafetyALFRED benchmark, which introduces six categories of kitchen safety hazards into the ALFRED embodied task setting. It reveals a critical alignment gap: multimodal large language models can identify hazards in static QA (up to 92%) but fail to proactively mitigate them during embodied planning (<60%), advocating a paradigm shift from QA-based to embodied safety evaluation.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) are increasingly deployed as autonomous agents in embodied environments, translating high-level natural language instructions into executable plans. Existing safety benchmarks such as ASIMOV, Multimodal Situational Safety, and MM-SafetyBench primarily evaluate hazard recognition through static image- or video-based question answering.

**Limitations of Prior Work**: Existing evaluations suffer a fundamental flaw — they test only whether models *recognize* hazards, not whether models can generate hazard-mitigating plans in dynamic embodied environments. A model that correctly identifies "a phone in the sink" as dangerous may completely ignore the need to remove the phone before executing a "wash the knife" task. This knowledge–action gap has never been systematically quantified.

**Key Challenge**: High accuracy in static QA creates a false sense of safety: models *know* what is dangerous, yet when required to simultaneously complete a task and mitigate hazards, they systematically prioritize task completion over safety. QA performance is a poor proxy for embodied safety.

**Goal**: (1) Construct an embodied benchmark that jointly evaluates hazard recognition and proactive mitigation; (2) quantify the alignment gap between QA recognition and embodied mitigation; (3) investigate whether a multi-agent framework can reduce this gap.

**Key Insight**: Extend the ALFRED benchmark (an embodied instruction-following task built on AI2-THOR) by introducing six real-world safety hazard categories across 30 kitchen environments. Pre-rendered trajectories provide ground-truth history, isolating safety reasoning ability from task execution ability.

**Core Idea**: Evaluate the same scene simultaneously under two protocols — QA evaluation (can the model identify hazards?) and embodied evaluation (can the model mitigate hazards while executing a task?) — and quantify the gap between the two via an alignment rate.

## Method

### Overall Architecture
SafetyALFRED models safety-constrained planning as a tuple $\mathcal{P} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{H}, \mathcal{R}_{\text{safe}} \rangle$, requiring a safety-aware policy $\pi^*$ to prioritize corrective actions $\mathcal{R}_{\text{safe}}(h_i, s_t)$ when hazards are present, and advance task goals only under hazard-free states. The evaluation pipeline consists of: (1) environmental perturbations that introduce hazards; (2) a QA task in which models act as safety judges to identify hazards; and (3) an embodied task in which models generate plans that incorporate mitigation.

### Key Designs

1. **Six Kitchen Hazard Categories**:

    - **Function**: Cover the major types of real-world kitchen accidents.
    - **Mechanism**: Six categories are defined based on kitchen accident statistics: appliance misuse (metal or flammable objects in a microwave), food spoilage (refrigerator door left open), falls/trips (cabinet door left open), fire hazards (stove left on), property damage (water-sensitive objects in the sink), and unsanitary conditions (target object on a dirty floor). Environmental condition predicates and corrective actions are defined for each category.
    - **Design Motivation**: These six categories span a complete risk spectrum from high-frequency events (falls/trips are the most common injury source) to high-consequence events (fires are the most destructive accident type).

2. **Dual-Setting Evaluation (QA + Embodied)**:

    - **Function**: Quantify the transfer gap from abstract safety knowledge to concrete action.
    - **Mechanism**: The same model is evaluated on the same scene under two independent instances — a QA instance in which the model acts as an external safety judge determining whether a hazard is present (validated through a two-stage structure + NLI pipeline), and an embodied instance in which the model generates the next action and subgoal frame-by-frame while performing a household task. The alignment rate $\mathcal{A} = \frac{1}{K}\sum_{k=1}^{K}\mathbb{I}(v_{ik} = a_{ik})$ measures the consistency between QA recognition and embodied mitigation.
    - **Design Motivation**: This design directly exposes the "knows but does not act" problem and constitutes a fundamental complement to existing QA-only evaluations.

3. **Multi-Agent Framework**:

    - **Function**: Attempt to improve safety mitigation through role separation.
    - **Mechanism**: Hazard recognition and mitigation are decoupled — a dedicated safety-judge agent identifies hazards and communicates safety information to the embodied agent. This tests the hypothesis of "if the model is informed of a hazard, can it mitigate it?"
    - **Design Motivation**: If single-agent failure stems from task interference (task execution distracting attention from safety), then multi-agent division of labor should improve performance.

### Loss & Training
This is an evaluation study; no model training is involved. All models are evaluated with temperature 0 and a maximum of 512 tokens.

## Key Experimental Results

### Main Results
Performance comparison of 11 MLLMs on QA recognition versus embodied mitigation.

| Model | QA Recognition (w/ metadata) | Embodied Mitigation (w/ metadata) | Gap |
|---|---|---|---|
| Qwen 2.5 VL 72B | 60.8% | 12.3% | −48.5% |
| Qwen 3 VL 32B | 57.2% | 19.7% | −37.5% |
| Gemini 1.5 ER | 77.9% | 45.7% | −32.2% |
| Gemini 2.5 | 92.5% | 60.1% | −32.4% |

### Multi-Agent Improvement

| Model | Single-Agent | Multi-Agent | Gain |
|---|---|---|---|
| Gemma 3 27B | 7.0% | 25.1% | +18.1% |
| Qwen 3 VL 32B | 19.7% | 32.5% | +12.8% |
| Qwen 2.5 VL 72B | 12.3% | 28.5% | +16.2% |

### Key Findings
- **Striking alignment gap**: Even the strongest model, Gemini 2.5, translates its 92.5% QA recognition rate into only a 60.1% embodied mitigation rate.
- Models systematically **prioritize task completion over hazard mitigation**: Qwen 3 VL-32B achieves 80.7% action prediction accuracy on hazard-free frames but only 19.7% hazard mitigation success.
- **Fire hazards** are the only category with consistently strong performance across both settings (stove on/off states are straightforward to perceive and manipulate); other categories show large gaps.
- The multi-agent framework helps but does not fully resolve the problem: even when the safety-judge agent correctly identifies a hazard, the embodied agent may still fail to execute the mitigation action.
- Models **frequently hallucinate hazards** in safe scenes (>50% false-positive rate), exhibiting an over-cautious bias.
- Scaling model size generally **decreases** safety alignment — larger models recognize more hazards in QA but mitigate them at a disproportionately lower rate in embodied settings.

## Highlights & Insights
- The **"knows but does not act" finding** is highly impactful: it fundamentally challenges the validity of current MLLM safety evaluation practice. A large body of work assesses safety via QA or multiple-choice tasks, and this paper demonstrates that such assessment is insufficient.
- The **controlled variable design** is instructive: providing ground-truth history isolates safety reasoning, while visual-only and metadata-augmented modes disentangle perception from reasoning deficiencies.
- The multi-agent results reveal a deeper issue: the failure is not merely one of attention allocation. Models face a fundamental planning difficulty when required to *interrupt* a task flow to insert safety actions.
- The findings transfer to domains such as autonomous driving, where evaluation of planning under safety constraints is a general need.

## Limitations & Future Work
- The use of pre-rendered trajectories rather than real-time interaction does not fully represent real robotic scenarios.
- Only three model families (Qwen, Gemma, Gemini) are evaluated, limiting the generalizability of the conclusions.
- Kitchen hazards in the AI2-THOR simulator are simplified and cannot fully capture the complexity and unpredictability of real-world environments.
- Automatic evaluation of QA responses using NLI models may introduce bias.
- The paper does not explore improving embodied safety capabilities through training data augmentation.

## Related Work & Insights
- **vs. ASIMOV / MM-SafetyBench**: These benchmarks evaluate hazard recognition in static QA only; SafetyALFRED adds the embodied mitigation dimension and quantifies the gap between the two.
- **vs. Son et al. / Chen et al.**: The former is limited to text-based PDDL environments; the latter is limited to static AI-generated images. SafetyALFRED evaluates models in a multimodal simulated environment with navigation.
- **Insight**: Future safety evaluations should require models to *enact* safety rather than merely *describe* it; training data must include examples that balance safety and task objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of the alignment gap between QA hazard recognition and embodied hazard mitigation; the problem formulation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 11 models, 6 hazard categories, and multiple evaluation metrics, though the use of pre-rendered trajectories is a simplification.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, but the paper is lengthy and portions of the analysis are dispersed across the appendix.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)

<!-- RELATED:END -->
