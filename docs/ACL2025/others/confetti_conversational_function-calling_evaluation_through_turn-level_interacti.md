---
title: >-
  [Paper Note] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions
description: >-
  [ACL 2025][Function calling] CONFETTI proposes a function-calling evaluation benchmark for multi-turn conversational scenarios, containing 109 human-simulated conversations, 313 user turns, and 86 APIs. Through off-policy turn-level evaluation and dialog act annotation, it systematically tests the tool-calling capability of LLMs in complex conversational scenarios. The study reveals that even the strongest model (Nova Pro) only achieves around 40% accuracy…
tags:
  - "ACL 2025"
  - "Function calling"
  - "conversational evaluation benchmark"
  - "multi-turn interaction"
  - "tool use"
  - "LLM evaluation"
date: 2026-05-08
content_hash: 452ac408c6b50d9e
---

# CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions

**Conference**: ACL 2025  
**arXiv**: [2506.01859](https://arxiv.org/abs/2506.01859)  
**Code**: None  
**Area**: Others  
**Keywords**: Function calling, conversational evaluation benchmark, multi-turn interaction, tool use, LLM evaluation

## TL;DR

CONFETTI proposes a function-calling evaluation benchmark for multi-turn conversational scenarios, containing 109 human-simulated conversations, 313 user turns, and 86 APIs. Through off-policy turn-level evaluation and dialog act annotation, it systematically tests the tool-calling capability of LLMs in complex conversational scenarios. The study reveals that even the strongest model (Nova Pro) only achieves around 40% accuracy, with chained calling being a universal weakness.

## Background & Motivation

**Background**: As the application of LLMs as intelligent assistants becomes increasingly widespread, function-calling has become a core capability for LLMs to interact with external tools and APIs. Existing function-calling evaluation benchmarks (such as ToolBench, API-Bench, BFCL, etc.) mainly focus on single-turn calling or simple multi-turn scenarios, where the evaluation emphasis is on "whether a single API can be called correctly."

**Limitations of Prior Work**: Real-world interactions between users and LLMs are far more complex than single-turn calling. Users ask follow-up questions, correct goals, switch topics, present vague requirements, or express implicit intentions. Existing benchmarks exhibit clear deficiencies in the following aspects: (1) **Lack of real conversational complexity**, failing to involve scenarios such as goal correction and goal switching; (2) **Coarse-grained evaluation**, which typically only assesses the correctness of the final call while ignoring the quality of intermediate dialogue management; (3) **Limited API scale**, failing to test the model's ability to select from a large number of candidate APIs.

**Key Challenge**: Evaluating function-calling requires simultaneous consideration of both "the accuracy of tool selection" and "the appropriateness of dialogue management." However, almost all existing benchmarks focus solely on the former and ignore the latter. In multi-turn conversations, a superior agent should not only call the correct API but also correctly understand vague requirements, handle goal switching, and perform reasonable clarifying follow-ups.

**Goal**: Build a function-calling benchmark covering multiple dimensions of dialogue complexity to evaluate both tool-calling accuracy and dialogue response quality, thereby revealing the true performance of current LLMs in realistic multi-turn tool-use scenarios.

**Key Insight**: Beginning from the complexity of real conversational interactions, the authors manually design test scenarios that cover various conversational challenges and introduce dialog act annotations to evaluate response quality in non-function-calling turns.

**Core Idea**: Construct a multi-dimensional evaluation framework that comprehensively characterizes the conversational function-calling capabilities of LLMs through meticulously designed human-simulated dialogues, off-policy turn-level evaluation, and dialog act analysis.

## Method

### Overall Architecture

The construction process of CONFETTI consists of three phases: (1) **Dialogue design and collection**—annotators simulate multi-turn interactions between users and the agent, deliberately design various complex scenarios; (2) **Annotation system establishment**—labeling the expected function-calls and dialog acts for each turn; (3) **Evaluation framework**—given the dialogue history, letting the evaluated LLM respond independently at each turn (off-policy), and then comparing the responses against reference annotations across multiple dimensions.

### Key Designs

1. **Multi-dimensional Dialogue Complexity Coverage**:

    - Function: Ensures the benchmark can test various conversational difficulties that LLMs encounter in real scenarios.
    - Mechanism: Defines multiple types of dialogue complexity, including: follow-ups (asking questions based on previous results), goal correction (the user revising previous goals, e.g., "No, I'm changing to tomorrow's flight"), goal switching (completely switching to a new topic/requirement), ambiguous goals (vague requirements where the agent needs to clarify actively), and implicit goals (intentions not explicitly stated but inferable). The 109 dialogues systematically cover these dimensions.
    - Design Motivation: Real users do not cooperatively formulate clear requests as in typical evaluation datasets; dialogues are replete with corrections, shifts, and vague expressions. Only by covering these complexities can the actual capabilities of models be truly evaluated.

2. **Off-Policy Turn-Level Evaluation**:

    - Function: Independently evaluates the model at each conversational turn to prevent error accumulation from affecting the fairness of the assessment.
    - Mechanism: Provides the model with standard dialogue history (rather than the model's own previous outputs) as context, allowing the model to generate responses independently at each user turn. This ensures that different models are evaluated under identical context, eliminating the issue where "prior errors in on-policy evaluation lead to subsequent failures." It also supports the evaluation of chained function-calls (scenarios where multiple APIs must be called sequentially within a single turn).
    - Design Motivation: Although on-policy evaluation (allowing the model to complete the entire dialogue on its own) is closer to real scenarios, a failure in an intermediate turn causes subsequent evaluation to fail. The off-policy approach allows independent assessment of each turn's capability, yielding more fine-grained and fairer results.

3. **Dialog Act Annotation System**:

    - Function: Evaluates the dialogue management quality of models in non-function-calling turns.
    - Mechanism: Annotates the standard responses of each turn with dialog acts (such as inform, request, confirm, clarify, etc.), ensuring the model not only "does the right thing" (calling the correct API) but also "says the right thing" (generating appropriate dialogue actions). For example, when facing vague requirements, the model should generate a clarification rather than blindly guessing a call; when confirmation is needed, it should confirm instead of executing directly.
    - Design Motivation: Function-calling is more than just API calling—a good conversational agent needs to perform appropriate information gathering, confirmation, and clarification prior to execution. Dialog act evaluation addresses the limitation of existing benchmarks that only focus on function-calling accuracy.

### Loss & Training

CONFETTI is an evaluation benchmark rather than a training method, and thus does not involve loss functions or training strategies. Regarding evaluation metrics, it primarily uses precision/recall/F1 of function calls (checking the match of function names and parameters), as well as the accuracy of dialog acts. For chained calling, it also evaluates the sequence completeness and order correctness.

## Key Experimental Results

### Main Results

The paper evaluates the performance of several SOTA LLMs on CONFETTI, ranked by overall F1 score:

| Model | Overall F1 (%) | Function Name Accuracy | Parameter Accuracy | Chained Call Success Rate |
|------|---------------|------------|-----------|-------------|
| Nova Pro | 40.01 | Relatively High | Medium | Low |
| Claude Sonnet v3.5 | 35.46 | High | Medium | Low |
| Llama 3.1 405B | 33.19 | Medium-High | Medium | Low |
| Command-R-Plus | 31.18 | Medium | Medium | Low |
| Mistral-Large-2407 | 30.07 | Medium | Medium-Low | Very Low |

### Ablation Study

Performance variation analysis across different dimensions:

| Analysis Dimension | Phenomenon | Explanation |
|---------|------|------|
| Increase in API count (5→20+) | Dramatic drop in some models' performance | Selection difficulty intensifies with large candidate APIs |
| Increase in dialogue length | Diverse performance (some stable, some degraded) | Exposes differences in long-context processing |
| Chained function-calling | Extremely poor performance across all models | The most common weakness; multi-calls per turn are highly challenging |
| Goal switching scenarios | Significantly harder than follow-ups | Goal switching requires stronger context understanding |
| Ambiguous goals | Most models tend to guess directly rather than clarify | Dialogue management capabilities are generally insufficient |

### Key Findings

- **Chained function-calling is a common weakness of all models**: When consecutive API calls are required within a single turn, even the strongest models perform poorly. This indicates that current LLMs still have obvious deficiencies in planning multi-step operations.
- **API count is a key bottleneck**: When the number of available APIs exceeds 20, the accuracy of many models drops significantly, suggesting "selection difficulty" when models face large-scale APIs. However, Nova Pro and Claude remain relatively robust in this regard.
- **Long dialogues are not a universal issue**: Some models (such as Claude, Nova Pro) handle long dialogues well, whereas others degrade significantly as context grows, reflecting differences in long-context capabilities among models.
- **Even the best model scores only 40%**: This result indicates that conversational function-calling remains an unresolved challenge. Current SOTA models are still far from being ready for actual production deployment.
- **Dialog act analysis reveals blind spots in dialogue management**: Models are better at executing explicit instructions but perform poorly when they need to actively clarify, confirm, or guide the conversation flow.

## Highlights & Insights

- **Systematic design of dialogue complexity dimensions is the biggest highlight**: Unlike simple API calling tests, CONFETTI explicitly defines dimensions such as follow-ups, goal correction, and goal switching, providing a framework closer to real scenarios for function-calling evaluation. This taxonomy can guide the construction of future tool-calling datasets.
- **Off-policy turn-level evaluation is highly practical**: It resolves the fairness issue regarding error accumulation in multi-turn evaluations, allowing different models to be compared under identical context. This paradigm deserves adoption in other multi-turn interaction evaluations.
- **Low performance in chained calling is an important community signal**: It highlights that the true bottleneck of current LLM function-calling lies not in single calls, but in multi-step planning and sequential execution, pointing out directions for subsequent research.

## Limitations & Future Work

- **Small data scale**: The scale of 109 dialogues and 313 turns limits statistical significance and coverage of long-tail scenarios.
- **Limited API coverage**: Although 86 APIs are substantial, it is still insufficient compared to the diversity of the real-world tool ecosystem.
- **Lack of accompanying training data**: CONFETTI is solely an evaluation benchmark and does not offer corresponding training data or mitigation methods.
- **Evaluation is mostly reference-matching based**: The evaluation of function-calling relies mainly on comparing against reference annotations. However, in multi-turn dialogues, there can be multiple reasonable calling strategies, meaning reference matching may underestimate certain valid alternatives.
- **Model version timeliness**: The evaluated models (e.g., Claude Sonnet v3.5, Llama 3.1) have newer versions available, and the results might not reflect current up-to-date performance.

## Related Work & Insights

- **vs BFCL (Berkeley Function-Calling Leaderboard)**: BFCL primarily focuses on single-turn function-calling accuracy, while CONFETTI emphasizes complex interaction scenarios in multi-turn dialogues. They are complementary—BFCL tests foundational capabilities, while CONFETTI tests application-level abilities.
- **vs ToolBench**: ToolBench is larger but suffers from lower automatic generation quality, whereas CONFETTI is smaller but manually and meticulously designed with high annotation quality. CONFETTI's dialogue complexity design is more systematic.
- **vs API-Bench**: API-Bench focuses on API selection accuracy but lacks dialogue management evaluation. The dialog act annotation in CONFETTI bridges this gap.
- **Insights**: CONFETTI's evaluation framework (complexity dimensions + turn-level off-policy + dialog act) can serve as a blueprint for building larger conversational agent evaluation benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-dimensional dialogue complexity design and dialog act evaluation are pioneered in function-calling benchmarks, offering an innovative approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates multiple SOTA models with rich analysis dimensions, though the small data scale slightly affects statistical power.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-defined problems and in-depth analysis of evaluation results.
- Value: ⭐⭐⭐⭐ Fills the gap in conversational function-calling evaluation. The findings on chained calling are highly valuable for the community, though the lack of mitigation strategies slightly reduces immediate utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](../../ICML2025/others/regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
