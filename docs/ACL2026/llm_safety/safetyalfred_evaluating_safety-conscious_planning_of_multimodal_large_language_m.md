---
title: >-
  [Paper Note] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models
description: >-
  [ACL 2026][LLM Safety][Embodied Safety] This paper introduces the SafetyALFRED benchmark, incorporating six categories of kitchen safety hazards into ALFRED embodied tasks. It reveals a significant alignment gap in Multi…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Embodied Safety"
  - "Hazard Mitigation"
  - "Multimodal Evaluation"
  - "Safety Planning"
  - "ALFRED"
date: 2026-05-08
content_hash: 748272e066ff30d8
---

# SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.19638](https://arxiv.org/abs/2604.19638)  
**Code**: [https://github.com/sled-group/SafetyALFRED](https://github.com/sled-group/SafetyALFRED)  
**Area**: Multimodal VLM  
**Keywords**: Embodied Safety, Hazard Mitigation, Multimodal Evaluation, Safety Planning, ALFRED

## TL;DR
This paper introduces the SafetyALFRED benchmark, incorporating six categories of kitchen safety hazards into ALFRED embodied tasks. It reveals a significant alignment gap in Multimodal Large Language Models: while they can identify hazards in static QA (up to 92%), they struggle to actively mitigate them in embodied planning (<60%), advocating for a shift from QA-based to embodied safety evaluation paradigms.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) are increasingly deployed as autonomous agents in embodied environments, translating high-level natural language instructions into executable plans. Existing safety benchmarks like ASIMOV, Multimodal Situational Safety, and MM-SafetyBench primarily evaluate hazard recognition capabilities through static image/video-based question answering (QA) tasks.

**Limitations of Prior Work**: Existing evaluations suffer from a fundamental flaw—they only test whether a model "knows" a hazard, not whether it can generate plans to mitigate hazards in dynamic embodied environments. A model that identifies "a phone in the sink" as dangerous might still completely ignore removing the phone before executing a "wash the knife" task. This "knowledge-action" gap has never been systematically quantified.

**Key Challenge**: The high accuracy in static QA evaluations provides a false sense of security—models "know" what is dangerous, but when required to perform a task while simultaneously mitigating hazards, they systematically prioritize task completion over safety. QA performance serves as a poor proxy for embodied safety.

**Goal**: (1) Construct an embodied benchmark that evaluates hazard recognition and proactive mitigation concurrently; (2) Quantify the alignment gap between QA recognition and embodied mitigation; (3) Explore whether multi-agent frameworks can reduce this gap.

**Key Insight**: Extend the ALFRED benchmark (embodied instruction following based on AI2-THOR) by introducing six categories of real-world safety hazards across 30 kitchen environments. Utilize pre-rendered trajectories to provide ground-truth history, isolating "safety reasoning capability" from "task execution capability."

**Core Idea**: Run both QA evaluation (recognizing the hazard) and embodied evaluation (mitigating the hazard during task execution) on the same scenario, quantifying the gap via an alignment rate.

## Method

### Overall Architecture
SafetyALFRED models safety-conscious planning as a tuple $\mathcal{P} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{H}, \mathcal{R}_{\text{safe}} \rangle$, requiring a safety-conscious policy $\pi^*$ to prioritize corrective actions $\mathcal{R}_{\text{safe}}(h_i, s_t)$ when hazards exist, and only proceed with task goals in hazard-free states. The evaluation pipeline includes: (1) environment perturbations to introduce hazards; (2) models acting as safety judges in QA tasks to identify hazards; (3) models generating plans including mitigation in embodied tasks.

### Key Designs

1. **Six Kitchen Safety Hazard Categories**:
    - **Function**: Covers primary types of real-world kitchen accidents.
    - **Mechanism**: Defines six categories based on kitchen accident statistics: appliance misuse (metal/flammables in microwave), food spoilage (fridge door open), trips/falls (cabinet doors open), fire hazards (stove on), property damage (water-sensitive items in sink), and hygiene (target object on dirty floor). Each category defines environmental condition predicates and corrective actions.
    - **Design Motivation**: These categories cover the full risk spectrum from high frequency (trips/falls are the most common injury source) to high destructiveness (fires are the most destructive accident type).

2. **Dual-Setting Evaluation (QA + Embodied)**:
    - **Function**: Quantifies the translation gap from abstract safety knowledge to concrete behavior.
    - **Mechanism**: The same model evaluates the same scenario in two independent instances—the QA instance acts as an external safety judge (verified via a two-stage structure+NLI process), while the embodied instance generates next-step actions and subgoals frame-by-frame during task execution. The alignment rate $\mathcal{A} = \frac{1}{K}\sum_{k=1}^{K}\mathbb{I}(v_{ik} = a_{ik})$ measures consistency between QA recognition and embodied mitigation.
    - **Design Motivation**: This design directly exposes the "know but don't do" problem, serving as a fundamental supplement to existing pure QA evaluations.

3. **Multi-Agent Framework**:
    - **Function**: Attempts to improve safety mitigation through role separation.
    - **Mechanism**: Decouples hazard recognition from mitigation—a dedicated safety judge agent identifies hazards and passes safety information to the embodied agent. This tests the hypothesis: "If the model is told a hazard exists, can it mitigate it?"
    - **Design Motivation**: If single-agent failure stems from task interference (task execution distracting from safety), then multi-agent division of labor should improve performance.

### Loss & Training
This is an evaluative work and does not involve model training. All models use a temperature of 0 and a maximum of 512 tokens.

## Key Experimental Results

### Main Results
Comparison of 11 MLLMs in QA recognition vs. embodied mitigation.

| Model | QA Recognition (with Metadata) | Embodied Mitigation (with Metadata) | Gap |
|------|------|------|------|
| Qwen 2.5 VL 72B | 60.8% | 12.3% | -48.5% |
| Qwen 3 VL 32B | 57.2% | 19.7% | -37.5% |
| Gemini 1.5 ER | 77.9% | 45.7% | -32.2% |
| Gemini 2.5 | 92.5% | 60.1% | -32.4% |

### Multi-Agent Improvement

| Model | Single-Agent | Multi-Agent | Gain |
|------|--------|--------|------|
| Gemma 3 27b | 7.0% | 25.1% | +18.1% |
| Qwen 3 VL 32b | 19.7% | 32.5% | +12.8% |
| Qwen 2.5 VL 72b | 12.3% | 28.5% | +16.2% |

### Key Findings
- **Alignment gap is significant**: Even for the strongest model, Gemini 2.5, a 92.5% recognition rate in QA translates to only 60.1% mitigation in embodied tasks.
- Models systematically **prioritize task completion over hazard mitigation**: Qwen 3 VL-32B achieves 80.7% action prediction accuracy in hazard-free frames, but only 19.7% success in hazard mitigation.
- **Fire hazards** are the only category performing well in both settings (stove status is easily perceived and operated), while gaps in other categories are massive.
- Multi-agent frameworks help but do not fully solve the issue: even when the safety judge agent correctly identifies a hazard, the embodied agent may still fail to execute mitigation.
- Models **frequently hallucinate hazards** in safe scenarios (>50% false positive rate), showing an over-conservative bias.
- Scaling model size generally **decreases** safety alignment—larger models recognize more in QA but mitigate disproportionately less in embodied tasks.

## Highlights & Insights
- The **"know but don't do" finding** is highly impactful: it fundamentally challenges the validity of current MLLM safety evaluations. While much work relies on QA/multiple-choice for safety, this paper proves it is insufficient.
- The **controlled variable approach** in experimental design is exemplary: providing ground-truth history to isolate safety reasoning and using both vision-only and metadata-augmented modes to separate perception from reasoning deficits.
- Multi-agent results reveal a deeper issue: it is not just an attention allocation problem; models face fundamental planning difficulties when required to "interrupt" a task flow to insert safety actions.
- Transferable to domains like autonomous driving: evaluation of planning under safety constraints is a universal requirement.

## Limitations & Future Work
- Use of pre-rendered trajectories rather than real-time interaction does not fully represent real-world robotic scenarios.
- Only three model families (Qwen, Gemma, Gemini) were evaluated, limiting the generalizability of conclusions.
- Kitchen hazards in the AI2-THOR simulator are simplified and do not fully capture real-world complexity and unpredictability.
- Automated evaluation of QA responses using NLI models may introduce bias.
- Methods to enhance embodied safety via training data augmentation were not explored.

## Related Work & Insights
- **vs ASIMOV/MM-SafetyBench**: These benchmarks only evaluate hazard recognition in static QA. SafetyALFRED adds the embodied mitigation dimension and quantifies the gap between the two.
- **vs Son et al./Chen et al.**: The former is limited to text-based PDDL environments, and the latter to static AI-generated images. SafetyALFRED evaluates in multimodal simulated environments with navigation.
- **Insights**: Future safety evaluations should require models to "do" safety rather than just "say" safety; training data needs to include examples of safety-task balancing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of the alignment gap between QA safety recognition and embodied safety mitigation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 models, 6 hazard categories, multiple metrics, though pre-rendered trajectories are a simplification.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, though the paper is long and some analyses are dispersed in the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)
- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
