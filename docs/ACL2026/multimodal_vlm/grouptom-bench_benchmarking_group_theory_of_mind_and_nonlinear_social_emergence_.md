---
title: >-
  [Paper Note] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs
description: >-
  [ACL 2026][Multimodal VLM][Paper Note] This paper introduces GroupToM-Bench, utilizing 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether MLLMs can reason from individual beliefs/desires/intentions to group tensions, structural constraints, and nonlinear collective outcomes. Results indicat
tags:
  - ACL 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 04467bb3a53693b0
---
# GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs

**Conference**: ACL2026  
**arXiv**: [2606.04184](https://arxiv.org/abs/2606.04184)  
**Code**: Not found in cache  
**Area**: Multimodal VLM / Social Cognition Evaluation / Theory of Mind  
**Keywords**: Group Theory of Mind, Multimodal Evaluation, Social World Model, Nonlinear Emergence, Structural Constraints  

## TL;DR
This paper introduces GroupToM-Bench, utilizing 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether MLLMs can reason from individual beliefs/desires/intentions to group tensions, structural constraints, and nonlinear collective outcomes. Results indicate that current models exhibit a significant "group cognitive gap."

## Background & Motivation
**Background**: Theory of Mind (ToM) evaluation has expanded from static individual stories to multimodal, interactive, and multi-agent scenarios. Many benchmarks focus on whether models can identify the beliefs, desires, and intentions of a single character or track information asymmetry in local interactions.

**Limitations of Prior Work**: Real-world social behavior is not a linear summation of individual intentions. Power structures, cultural norms, information asymmetry, and conformity pressure can distort true individual intentions, leading groups toward outcomes that no individual actually desired, such as the Abilene Paradox or groupthink. Existing benchmarks often fail to measure this nonlinear emergence.

**Key Challenge**: Models may perform well at the individual level yet still predict group outcomes as a "rational consensus reached by everyone." This suggests that individual ToM capability does not equate to a social world model, particularly in understanding how structural constraints translate private mental states into public behavior.

**Goal**: Construct a benchmark where models must simultaneously process private mental states, public dialogues, and visual cues (expressions, postures, and spatial relationships) to reason along the causal chain of "individual state - group tension - structural constraint - collective outcome."

**Key Insight**: The authors model group interaction as a Constrained Dynamic Field, where the micro-level consists of each character's BDI states, the meso-level involves group tension and structural constraints, and the macro-level covers collective outcome prediction and mechanistic attribution.

**Core Idea**: Instead of asking "what a specific person believes," the benchmark asks "how multiple individuals form nonlinear collective failures under the pressure of power, conformity, information flow, and public stances."

## Method
GroupToM-Bench is essentially a diagnostic benchmark. Its value lies in decomposing group social reasoning into measurable cognitive hierarchies and exposing the "linear superposition bias" of models in multimodal scenarios.

### Overall Architecture

The benchmark consists of 240 expert-curated scenarios covering 8 overlapping socio-psychological domains, generating over 3,000 reasoning tasks. Each scenario includes private states of characters, public dialogues, structural constraints, and a global scene image. The images provide critical social cues like expressions, body language, and spatial positioning.

The evaluation utilizes a 7-layer cognitive audit framework. L1-L3 focus on the individual level: Belief, Desire, and Intention. L4-L7 focus on the group level: Group Tension, Structural Constraint, Collective Outcome Prediction, and Mechanistic Attribution. L1, L2, L3, L4, and L6 use exact-match multiple-choice questions, while L5 and L7 use open-ended responses scored by GPT-5 based on expert references.

### Key Designs

**1. Three-level Social Causal Chain Modeling: Upgrading "Individual Identification" to "Structural Distortion Explanations"**

Traditional ToM evaluations only ask "what an individual believes," which models can bypass via text pattern matching. GroupToM models group interaction as a Constrained Dynamic Field across three layers: the micro layer (individual BDI states), the meso layer (latent tension and structural constraints involving power, communication, and culture), and the macro layer (collective outcomes and structural attribution). This forces models to explain why private intentions distort into undesired outcomes, capturing phenomena like the Abilene Paradox.

**2. Seven-layer Cognitive Audit Framework: Pinpointing Social Reasoning Failures**

To distinguish between failing to read individual psychology versus failing to infer group structure, the framework splits reasoning into L1–L7. L1-L3 form the individual cognitive foundation, while L4-L7 represent emergent group dynamics. This layering allows for the visualization of a "cognitive cliff," where model performance often plummets when transitioning from L3 (individual) to L4/L6 (group), defined as the Cognitive Gap.

**3. Human-in-the-loop Data Construction and Validation: Expert-driven Structural Design**

Group social reasoning lacks the clear mechanical ground truth of physical tasks. GroupToM employs an expert-led three-stage process: experts design seeds (intentions, constraints, decision points); frontier MLLMs and diffusion models expand these into multimodal interactions; and two-stage human audits verify logical consistency and ensure visual cues are essential for reasoning.

### Loss & Training

As a benchmark paper, no new models were trained. For evaluation, L1-L4 and L6 involve multi-choice tasks where any non-empty combination of four options is possible, resulting in a random guess baseline of $1/15$ (approx. 6.7%). L5 and L7 are open-ended, scored 0-100 by GPT-5. A judge meta-evaluation showed GPT-5's Pearson correlation with human experts at $r=0.76, p<0.001$.

## Key Experimental Results

### Main Results

| Model | L1 Belief | L2 Desire | L3 Intention | L4 Tension | L5 Constraint | L6 Outcome | L7 Attribution | Cognitive Gap |
|------|-----------|-----------|--------------|------------|---------------|------------|----------------|---------------|
| Human | 91.7 | 90.5 | 88.4 | 89.4 | 90.1 | 89.2 | 88.1 | 1.0 |
| GPT-5 | 76.7 | 74.1 | 72.3 | 50.5 | 56.9 | 45.0 | 61.0 | 21.0 |
| GPT-4o | 79.8 | 75.3 | 72.7 | 50.3 | 47.2 | 48.6 | 53.4 | 26.1 |
| Gemini 3-pro | 78.9 | 77.1 | 73.9 | 53.1 | 59.7 | 48.3 | 64.2 | 20.3 |
| Qwen3 VL-8B | 73.3 | 68.8 | 69.6 | 37.3 | 47.8 | 34.3 | 53.6 | 27.3 |
| InternVL 3.5-8B | 66.5 | 60.7 | 64.2 | 33.1 | 41.4 | 26.2 | 47.5 | 26.8 |

The most significant drop occurs between L3 and L4/L6. While models maintain relatively high scores in individual intention, performance collapses in group tension and collective outcome prediction.

### Ablation Study

| Evaluated Object | Base Average Trend | Change after Text-only | L6 Drop | Explanation |
|------|---------------|------------------|---------|------|
| Human | L1-L7 approx. 88-92 | Decrease of 3.7-4.3 | 3.9 | Humans utilize visual social cues |
| GPT-4o | Base L1-L7: 79.8-53.4 | Decrease of 1.8-2.1 | 2.0 | Images help but dependency is low |
| Qwen3 VL-8B | Base L1-L7: 73.3-53.6 | Decrease of 0.3-0.7 | 0.5 | Relies heavily on text heuristics; visual blind spot |

| Failure Mode | GPT-4o L6 Error % | Qwen3 VL-8B L6 Error % | Meaning |
|------|------------------|------------------------|------|
| Optimistic consensus prediction | 48% | 61% | Tendency to predict rational consensus, missing groupthink/Abilene Paradox |
| Misattributed non-optimality | N/A | N/A | Recognizes poor results but attributes them to individuals rather than structures |
| Random/incoherent selection | <8% | <8% | Errors are systematic linear superposition biases, not random |

### Key Findings

- The Cognitive Transition Gap across 11 models ranges from 18.8% to 27.3% (median ~24.5%), indicating a systemic shortcoming in current architectures.
- L5 (Structural Constraint) is a key bottleneck. Gemini 3-pro outperformed GPT-4o in L5, suggesting structural mechanism representation differs from general individual ToM.
- Text-only ablations reveal insufficient image utilization by models; Qwen3 VL-8B drops only ~0.5 points without images, compared to ~4.0 for humans.

## Highlights & Insights
- **Advancing ToM Evaluation to Group Emergence**: The paper shifts from testing individual false beliefs to measuring how structural pressures alter group outcomes, providing a more realistic definition of "social world models."
- **7-layer Framework for Diagnostic Precision**: The hierarchy clearly identifies where models fail (from individual understanding to group misjudgment) rather than providing a single aggregated score.
- **Linear Superposition Bias as a Key Failure Mode**: Models tend to "average" public stances into a rational consensus, failing to simulate conformity, silence, and power dynamics.
- **Need for Stronger Visual Causal Control**: Text-only results suggest models still rely on shortcuts; future benchmarks must ensure correct answers depend on visual cues that cannot be recovered from text.

## Limitations & Future Work

- **Uneven Multimodal Dependency**: Some samples can still be partially solved via text.
- **Alignment-induced Conservatism**: Models may be reluctant to predict destructive or irrational group collapses due to safety tuning. Comparing base vs. instruction-tuned models is needed.
- **Limited Cultural Scope**: Scenarios primarily reflect Western social norms. High-context cultures or hierarchical orders may not generalize.
- **Judge Bias**: LLM-as-a-judge (GPT-5) may have stylistic preferences in scoring open-ended social reasoning.

## Related Work & Insights
- **vs. Individual ToM Benchmarks**: Unlike prior work (Sap, Wu, etc.) focusing on individual states, GroupToM-Bench targets structural group outcomes.
- **vs. Multi-agent Reasoning**: While multi-agent tasks often focus on tasks or hidden info, GroupToM emphasizes social psychology (conformity, groupthink).
- **Implications for Model Design**: Future social intelligence might require explicit structural modeling (power, communication topology, norm pressure) rather than simple context concatenation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Distinct shift to Group ToM and nonlinear social emergence.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive model coverage and diagnostic layering, though multimodal dependency could be stronger.
- Writing Quality: ⭐⭐⭐⭐ Inspiring framework and failure mode analysis.
- Value: ⭐⭐⭐⭐⭐ Critical for social intelligence and agent safety, highlighting that individual ToM is insufficient for social world modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting](../../CVPR2026/multimodal_vlm/if-bench_benchmarking_and_enhancing_mllms_for_infrared_images_with_generative_vi.md)
- [\[ICML 2025\] Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner](../../ICML2025/multimodal_vlm/overcoming_multi-step_complexity_in_multimodal_theory-of-mind_reasoning_a_scalab.md)

</div>

<!-- RELATED:END -->
