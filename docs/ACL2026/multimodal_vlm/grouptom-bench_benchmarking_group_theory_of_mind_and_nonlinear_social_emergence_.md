---
title: >-
  [Paper Note] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs
description: >-
  [ACL2026][Multimodal VLM][Group Theory of Mind] This paper introduces GroupToM-Bench, which utilizes 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Group Theory of Mind"
  - "Multimodal Evaluation"
  - "Social World Models"
  - "Nonlinear Emergence"
  - "Structural Constraints"
date: 2026-05-08
content_hash: 9f2f3a41f8f22323
---

# GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs

**Conference**: ACL2026  
**arXiv**: [2606.04184](https://arxiv.org/abs/2606.04184)  
**Code**: Not found in cache  
**Area**: Multimodal VLM / Social Cognition Evaluation / Theory of Mind  
**Keywords**: Group Theory of Mind, Multimodal Evaluation, Social World Models, Nonlinear Emergence, Structural Constraints  

## TL;DR
This paper introduces GroupToM-Bench, which utilizes 240 expert-designed multimodal group interaction scenarios and a 7-layer cognitive audit framework to evaluate whether MLLMs can reason from individual beliefs/desires/intentions to group tensions, structural constraints, and nonlinear collective outcomes. The results indicate that current models generally exhibit a significant "group cognitive gap."

## Background & Motivation
**Background**: Theory of Mind (ToM) evaluation has expanded from static individual narratives to multimodal, interactive, and multi-agent scenarios. Many benchmarks focus on whether models can identify the beliefs, desires, and intentions of a single character or track information asymmetry in local interactions.

**Limitations of Prior Work**: Real-world social behavior is not a linear summation of individual intentions. Power structures, cultural norms, information asymmetry, and conformity pressure can distort true individual intentions, leading the group toward outcomes that no individual truly desired, such as the Abilene Paradox or groupthink. Existing benchmarks often fail to measure this nonlinear emergence.

**Key Challenge**: Models may perform well at the individual level yet still predict collective outcomes as "rational consensus among all parties." This suggests that individual ToM capability does not equate to a social world model, specifically regarding the understanding of how structural constraints transform private mental states into public behaviors.

**Goal**: Construct a benchmark where the model must simultaneously process private mental states, public dialogues, and visual cues (expressions, poses, and spatial relationships in images) to perform reasoning along the causal chain of "individual state - group tension - structural constraint - collective outcome."

**Key Insight**: The authors model group interaction as a Constrained Dynamic Field: the micro-layer represents the BDI (Belief-Desire-Intention) states of each character, the meso-layer represents group tension and structural constraints, and the macro-layer represents collective outcome prediction and mechanistic attribution.

**Core Idea**: Instead of asking the model "what a specific person believes," the task asks how multiple individuals form nonlinear collective failures under the pressures of power, conformity, information flow, and public positioning.

## Method
GroupToM-Bench is essentially a diagnostic benchmark. Its value lies not in proposing a new model, but in decomposing group social reasoning into measurable cognitive levels and exposing linear summation biases in models using multimodal scenarios.

### Overall Architecture

The benchmark consists of 240 expert-curated scenarios covering 8 overlapping socio-psychological domains, generating over 3K reasoning tasks. Each scenario includes the private states of multiple characters, public dialogues, structural constraints, and an overall scene image. The image is not decorative; it provides social cues such as expressions, body language, and spatial positioning.

Evaluation utilizes a 7-layer cognitive audit framework. L1-L3 are at the individual level: Belief, Desire, and Intention. L4-L7 are at the group level: Group Tension, Structural Constraint, Collective Outcome Prediction, and Mechanistic Attribution. L1, L2, L3, L4, and L6 use exact-match multiple-choice questions, while L5 and L7 use open-ended responses scored by GPT-5 based on expert reference answers.

### Key Designs

1. **Three-layer Social Causal Chain Modeling**:
    - **Function**: Expands group social reasoning from "identifying individual psychology" to "explaining how individual psychology is distorted by group structure."
    - **Mechanism**: The micro-layer focuses on BDI states, the meso-layer on latent tension and authority/communication/cultural constraints, and the macro-layer on final outcomes and structural attribution.
    - **Design Motivation**: If only individual beliefs are measured, models might pass through text pattern matching; only by incorporating structural pressure into the causal chain can the model's understanding of group emergence be tested.

2. **Seven-layer Cognitive Audit Framework**:
    - **Function**: Layer-by-layer localization of where the model’s social reasoning breaks down.
    - **Mechanism**: L1 tests belief, L2 tests desire, L3 tests intention, L4 tests latent group tension, L5 tests structural constraints, L6 tests collective outcomes, and L7 tests mechanistic attribution. The first 3 layers are individual cognitive foundations, while the last 4 are group emergence dynamics.
    - **Design Motivation**: A single aggregate score cannot distinguish between a model's inability to read individual psychology and its inability to infer group structure. The seven-layer decomposition directly reveals the cognitive cliff from L3 to L4/L6.

3. **Human-in-the-loop Data Construction and Verification**:
    - **Function**: Ensures scenarios possess socio-psychological logic, multimodal dependency, and evaluable gold standards.
    - **Mechanism**: Experts first design seeds defining private intentions, structural constraints, and key decision points. Frontier MLLMs and diffusion models then expand these into complete multimodal interactions. Finally, two stages of manual review check for factual/logical consistency and visual reasoning value, establishing an independent human baseline.
    - **Design Motivation**: Unlike physical tasks, group social reasoning lacks naturally clear mechanical ground truths; therefore, social causal structures must be authored by experts and manually verified to prevent scenarios from degrading into text-based common sense problems.

### Loss & Training

As this is an evaluation benchmark paper, no new models were trained. For evaluation, L1, L2, L3, L4, and L6 are multiple-choice tasks where answers can be any non-empty combination of four options; thus the random guessing baseline is 6.7% ($1/15$). Any missing or incorrect selection results in a score of 0. L5 and L7 are open-ended responses scored from 0-100 by GPT-5 according to gold references. The authors also conducted judge meta-evaluation: the Pearson correlation between GPT-5 and human experts on a subset of 100 responses was $r=0.76, p<0.001$, higher than Gemini-3-pro's $r=0.68$ and Qwen3-Max's $r=0.71$.

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

The most significant breakpoint occurs from L3 to L4/L6: models maintain relatively high scores at the individual intention layer but drop sharply at group tension and collective outcome prediction. Qwen3 VL-8B drops from 73.3% at L1 to 34.3% at L6; InternVL 3.5-8B scores only 26.2% at L6.

### Ablation Study

| Evaluated Object | Base Average Trend | Change after Text-only | L6 Drop | Explanation |
|------|---------------|------------------|---------|------|
| Human | L1-L7 approx. 88-92 under image+text | 3.7-4.3 decrease per layer | 3.9 | Humans indeed utilize visual social cues |
| GPT-4o | Base L1-L7: 79.8/75.3/72.7/50.3/47.2/48.6/53.4 | 1.8-2.1 decrease per layer | 2.0 | Images are helpful but dependency is insufficient |
| Qwen3 VL-8B | Base L1-L7: 73.3/68.8/69.6/37.3/47.8/34.3/53.6 | 0.3-0.7 decrease per layer | 0.5 | Primarily relies on text heuristics; obvious visual blind spots |

| Failure Mode | GPT-4o L6 Error % | Qwen3 VL-8B L6 Error % | Meaning |
|------|------------------|------------------------|------|
| Optimistic consensus prediction | 48% | 61% | Tendency to predict rational consensus, missing groupthink/Abilene Paradox |
| Misattributed non-optimality | N/A | N/A | Recognizes poor outcomes but attributes them to individual issues rather than structural forces |
| Random/incoherent selection | <8% | <8% | Errors are not random but reflect systematic linear summation biases |

### Key Findings

- The Cognitive Transition Gap across 11 models ranges from 18.8% to 27.3%, with a median of approximately 24.5%. This indicates that the group-level cognitive gap is a systematic shortcoming of current architectures rather than an isolated phenomenon.
- L5 Structural Constraint is a key bottleneck. Gemini 3-pro scored 59.7 on L5, higher than GPT-4o's 47.2, suggesting that the ability to articulate structural mechanisms is not entirely consistent with general individual ToM ability.
- Text-only ablation shows models do not sufficiently use images: removing images for Qwen3 VL-8B resulted in an average drop of only about 0.5 points, compared to a 4.0-point drop for humans. This reflects a lack of visual social integration rather than multimodal robustness.

## Highlights & Insights
- **Advancing ToM Evaluation to the Group Emergence Layer**: The paper moves beyond repeating individual false belief tests to measuring how structural pressures alter group outcomes. This definition of a "social world model" is closer to real social interaction.
- **Seven-layer Framework for Diagnostic Failure Mapping**: The hierarchy from L1 to L7 clearly shows where a model transitions from individual understanding to group misjudgment, rather than providing just a mixed average score.
- **Linear Summation Bias as an Explanatory Failure Mode**: Models fail by tending to average the public positions of multiple individuals into a rational consensus, failing to simulate conformity, silence, power suppression, or information cascades. This is more specific than simply saying "models are poor at social reasoning."
- **Multimodal Benchmarks Require Stronger Visual Causal Control**: The authors' text-only ablation shows that even with scenarios emphasizing vision, models still guess correctly via text. Future benchmarks should force correct answers to rely on visual cues that cannot be recovered from text.

## Limitations & Future Work

- **Uneven Multimodal Dependency**: The authors admit some samples can still be partially solved from text. GPT-4o's average score dropped only 1.9 points without images, while Qwen3 VL-8B dropped only 0.5, indicating current data does not entirely prevent text bypasses.
- **Potential Interference from Alignment-Induced Conservatism**: Models can identify negative individual intentions but are often reluctant to predict destructive or irrational collective collapses. Future work needs to compare base models with safety/instruction-tuned models to distinguish between safety alignment effects and true reasoning limitations.
- **Limited Cultural Scope**: Scenarios primarily reflect Western social norms and decision-making protocols. Mechanisms like high-context culture, hierarchical orders, and collective "face" manifest differently across cultures, and current ground truths may not generalize.
- **Bias Risks in Open-ended Judging**: Although GPT-5 correlates best with humans, LLM-as-a-judge may have stylistic preferences. Open-ended explanations of group social reasoning are particularly susceptible to phrasing and cultural assumptions.

## Related Work & Insights
- **vs. Individual ToM Benchmarks**: Works by Sap, Wu, Xu, Chen, Gu, etc., primarily evaluate individual beliefs/desires; GroupToM-Bench shifts the target to structural group outcomes.
- **vs. Multimodal/Embodied Social Reasoning**: Existing multimodal ToM and embodied multi-agent benchmarks emphasize visual grounding; this work further requires models to use visual cues to explain group-level dynamics.
- **vs. General Multi-agent Reasoning**: Multi-agent tasks often measure collaboration, negotiation, or hidden information strategies; GroupToM emphasizes social psychology concepts like conformity, power, groupthink, and nonlinear emergence.
- **Inspiration for Model Design**: Future social intelligence models may require explicit structural variable modeling—such as character power, communication topology, normative pressure, and private/public state differences—rather than simply concatenating all dialogues into the context.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The focus on Group ToM and nonlinear social emergence is distinctive and clearly differentiates it from traditional individual ToM.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad model coverage, clear hierarchical diagnostics, and inclusion of text-only ablation and judge meta-evaluation; however, multimodal dependency could be stronger.
- Writing Quality: ⭐⭐⭐⭐ The theoretical framework and failure mode descriptions are insightful; the nomenclature for some future models/settings may require additional context for readers.
- Value: ⭐⭐⭐⭐⭐ Valuable for social intelligence, multimodal evaluation, and agent safety, reminding researchers that individual ToM is not a sufficient condition for a social world model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[ACL 2026\] Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning](can_mllms_reason_beyond_language_visreason_a_comprehensive_benchmark_for_vision-.md)
- [\[ACL 2026\] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](do_mllms_understand_pointing_benchmarking_and_enhancing_referential_reasoning_in.md)

</div>

<!-- RELATED:END -->
