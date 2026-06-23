---
title: >-
  [Paper Note] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs
description: >-
  [ACL 2026][Multimodal VLM][Paper Note] This paper introduces GroupToM-Bench, which utilizes 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether MLLMs can reason from individual beliefs/desires/intentions to group tensions, structural constraints, and nonlinear collective outcomes. Results in
tags:
  - ACL 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 2b6a5d0bf464d057
---
# GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs

**Conference**: ACL2026  
**arXiv**: [2606.04184](https://arxiv.org/abs/2606.04184)  
**Code**: Not found in cache  
**Area**: Multimodal VLM / Social Cognition Evaluation / Theory of Mind  
**Keywords**: Group Theory of Mind, Multimodal Evaluation, Social World Models, Nonlinear Emergence, Structural Constraints  

## TL;DR
This paper introduces GroupToM-Bench, which utilizes 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether MLLMs can reason from individual beliefs/desires/intentions to group tensions, structural constraints, and nonlinear collective outcomes. Results indicate a significant "group cognitive gap" in current models.

## Background & Motivation
**Background**: Theory of Mind (ToM) evaluation has expanded from static individual stories to multimodal, interactive, and multi-agent scenarios. Many benchmarks focus on whether models can identify the beliefs, desires, and intentions (BDI) of a single character or track information asymmetry in local interactions.

**Limitations of Prior Work**: Real-world social behavior is not a linear summation of individual intentions. Power structures, cultural norms, information asymmetry, and conformity pressure can distort true individual intentions, leading groups toward outcomes that no individual actually desired (e.g., the Abilene Paradox or groupthink). Existing benchmarks often fail to measure this nonlinear emergence.

**Key Challenge**: Models may perform well at the individual level yet still predict group outcomes as a "rational consensus reached by everyone." This suggests that individual ToM capability does not equate to a social world model, specifically the understanding of how structural constraints transform private mental states into public behaviors.

**Goal**: Construct a benchmark where models must simultaneously process private mental states, public dialogues, and visual cues (expressions, gestures, spatial relationships) to reason along the causal chain of "Individual State - Group Tension - Structural Constraint - Collective Outcome."

**Key Insight**: The authors model group interaction as a Constrained Dynamic Field: the micro-layer consists of the BDI states of each character, the meso-layer consists of group tension and structural constraints, and the macro-layer consists of collective outcome prediction and mechanistic attribution.

**Core Idea**: Instead of asking "what does a specific person believe," the benchmark asks how multiple individuals form nonlinear collective failures under the pressure of power, conformity, information flow, and public positioning.

## Method
GroupToM-Bench is essentially a diagnostic benchmark. Its value lies not in proposing a new model, but in decomposing group social reasoning into measurable cognitive hierarchies and using multimodal scenarios to expose the linear superposition bias of models.

### Overall Architecture

The benchmark contains 240 expert-curated scenarios covering 8 overlapping socio-psychological fields, generating over 3,000 reasoning tasks. Each scenario includes private states of multiple characters, public dialogues, structural constraints, and a global scene image. The images are not decorative; they provide social cues such as facial expressions, body language, and spatial positioning.

The evaluation employs a 7-layer cognitive audit framework. L1-L3 are at the individual level: Belief, Desire, and Intention. L4-L7 are at the group level: Group Tension, Structural Constraint, Collective Outcome Prediction, and Mechanistic Attribution. L1/L2/L3/L4/L6 use exact-match multiple-choice questions, while L5/L7 utilize open-ended responses scored by GPT-5 against expert reference answers.

### Key Designs

**1. Three-tier social causal chain modeling: Upgrading "individual psychological recognition" to "explaining how individual psychology is distorted by group structure"**

Traditional ToM evaluations only ask "what someone believes," which models can often bypass using text pattern matching. GroupToM models group interaction as a Constrained Dynamic Field with three layers: the micro layer (BDI states), the meso layer (latent tension and structural constraints like power, communication, and culture), and the macro layer (collective outcomes and structural attribution). Explicitly embedding structural pressures ensures models must explain why private intentions warp into undesired collective outcomes, testing phenomena like the Abilene Paradox or groupthink.

**2. Seven-layer cognitive audit framework: Gradually locating where social reasoning fails**

A single aggregate score cannot distinguish between a model failing to understand individual psychology versus failing to derive group structure. The framework splits reasoning into L1–L7. L1–L3 form the individual cognitive foundation, while L4–L7 represent group emergent dynamics. This allows the plotting of a "cognitive cliff": if a model performs well up to L3 but drops significantly at L4/L6, this disparity is defined as the Cognitive Gap, serving as a diagnostic signal.

**3. Human-involved data construction and verification: Expert-defined social causal structures with manual validation**

Social reasoning lacks the clear mechanical ground truth of physical tasks. GroupToM uses an expert-led three-stage process: experts design seeds with fixed intentions and constraints; frontier MLLMs and diffusion models expand these into multimodal interactions; finally, human reviewers verify logic, factual consistency, and the necessity of visual cues. This ensures that images are integral to the task and that the 3,000+ tasks are grounded in evaluable social causal logic.

### Loss & Training

As a benchmarking paper, no new model was trained. For evaluation, L1, L2, L3, L4, and L6 are multiple-choice tasks where the answer can be any non-empty combination of four options, resulting in a random guess baseline of 6.7% ($1/15$). L5 and L7 are open-ended, scored 0-100 by GPT-5. A judge meta-evaluation showed that GPT-5 has a Pearson correlation of $r=0.76, p<0.001$ with human experts, outperforming Gemini-3-pro and Qwen3-Max.

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

The most significant break occurs between L3 and L4/L6: models maintain relatively high scores at the individual intention level but drop sharply when predicting group tension and collective outcomes.

### Ablation Study

| Subject | Base Trend | Text-only Change | L6 Drop | Explanation |
|------|---------------|------------------|---------|------|
| Human | L1-L7 approx. 88-92 | Decrease 3.7-4.3 | 3.9 | Humans utilize visual social cues |
| GPT-4o | 79.8/75.3/72.7/50.3/47.2/48.6/53.4 | Decrease 1.8-2.1 | 2.0 | Images help but dependency is low |
| Qwen3 VL-8B | 73.3/68.8/69.6/37.3/47.8/34.3/53.6 | Decrease 0.3-0.7 | 0.5 | Relies on text heuristics; visual blind spot |

| Failure Mode | GPT-4o L6 Error % | Qwen3 VL-8B L6 Error % | Meaning |
|------|------------------|------------------------|------|
| Optimistic consensus prediction | 48% | 61% | Predicts rational consensus; misses groupthink |
| Misattributed non-optimality | N/A | N/A | Correctly identifies bad outcome but blames individuals, not structure |
| Random/incoherent selection | <8% | <8% | Systematic linear superposition bias, not randomness |

### Key Findings

- The Cognitive Transition Gap across 11 models ranges from 18.8% to 27.3% (median ~24.5%), indicating a systemic bottleneck in current architectures for group-level cognition.
- L5 Structural Constraint is a major bottleneck. Gemini 3-pro outperformed GPT-4o in L5, suggesting that structural mechanism representation is distinct from general ToM.
- Text-only ablations reveal insufficient image utilization: removing images caused a negligible drop (~0.5) for Qwen3 VL-8B compared to ~4.0 for humans, indicating visual social cues do not deeply enter the causal reasoning process.

## Highlights & Insights
- **Advancing ToM Evaluation to Group Emergence**: Rather than repeating false belief tests, this benchmark measures how structural pressures alter group outcomes, aligning "social world models" with real social complexity.
- **Diagnostic Utility of the Seven-layer Framework**: The L1-L7 stratification identifies exactly where the model's reasoning transitions from individual understanding to group misjudgment.
- **Linear Superposition Bias**: This failure mode explains why models default to predicting a "rational consensus" derived from public statements, ignoring conformity, silence, and power dynamics.
- **Need for Stronger Visual Causal Control**: The text-only ablation proves that even with visual-centric design, models often exploit text shortcuts. Future benchmarks should enforce a dependency on visual cues that cannot be recovered from text.

## Limitations & Future Work

- **Uneven Multimodal Dependency**: Some samples can still be partially solved via text, as evidenced by the minimal performance drop in GPT-4o and Qwen3 when images are removed.
- **Alignment-induced Conservatism**: Models may recognize negative individual intentions but avoid predicting destructive or irrational collective collapses due to safety tuning.
- **Cultural Scope**: Scenarios primarily reflect Western social norms. High-context cultures or different hierarchical orders may follow different mechanisms not fully captured.
- **Judge Bias**: Despite the high correlation of GPT-5, LLM-as-a-judge may still exhibit stylistic preferences in evaluating open-ended social explanations.

## Related Work & Insights
- **vs. Individual ToM**: Differs from works by Sap, Wu, and others by shifting the target from individual mental states to structural group outcomes.
- **vs. Multimodal/Embodied Social Reasoning**: While previous work emphasizes visual grounding, GroupToM requires using those visual cues specifically to explain group-level dynamics.
- **vs. Multi-agent Reasoning**: Moves beyond simple cooperation or negotiation tasks to focus on social psychological phenomena like groupthink and nonlinear emergence.
- **Inspiration for Model Design**: Future social intelligence may require explicit modeling of structural variables (power, topology, normative pressure) rather than concatenating dialogues into context.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Distinctive focus on Group ToM and nonlinear social emergence.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad model coverage and clear diagnostic layers, though multimodal dependency could be stronger.
- **Writing Quality**: ⭐⭐⭐⭐ Insightful theoretical framework and failure mode analysis.
- **Value**: ⭐⭐⭐⭐⭐ Significant for social intelligence, multimodal evaluation, and agent safety; highlights that individual ToM is insufficient for social world modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[CVPR 2026\] IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting](../../CVPR2026/multimodal_vlm/if-bench_benchmarking_and_enhancing_mllms_for_infrared_images_with_generative_vi.md)
- [\[ICML 2025\] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models](../../ICML2025/multimodal_vlm/from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)
- [\[ACL 2026\] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding](do_mllms_capture_how_interfaces_guide_user_behavior_a_benchmark_for_multimodal_u.md)

</div>

<!-- RELATED:END -->
