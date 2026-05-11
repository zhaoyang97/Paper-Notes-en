---
title: >-
  [Paper Note] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect
description: >-
  [AAAI 2026][Robotics][Explanatory Interactive Learning] This paper systematically evaluates the impact of **order effects** on Explanatory Interactive Learning (XIL) through two large-scale user studies (713 participants…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Explanatory Interactive Learning"
  - "Cognitive Bias"
  - "Order Effect"
  - "User Study"
  - "XIL"
date: 2026-05-08
content_hash: 8aa46a46359298a1
---

# Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect

**Conference**: AAAI 2026
**arXiv**: [2512.04764](https://arxiv.org/abs/2512.04764)
**Code**: None
**Area**: Robotics (Human-Computer Interaction / Explainable AI)
**Keywords**: Explanatory Interactive Learning, Cognitive Bias, Order Effect, User Study, XIL

## TL;DR

This paper systematically evaluates the impact of **order effects** on Explanatory Interactive Learning (XIL) through two large-scale user studies (713 participants in total). The findings show that order effects have a **limited and inconsistent** influence on user feedback quality, with a statistically significant but weak effect observed only within sessions (not between sessions). The overall conclusion is that order effects do not constitute a major obstacle to the practical deployment of XIL.

## Background & Motivation

### Explanatory Interactive Learning (XIL)

XIL is a powerful interactive learning framework:
1. An AI model makes predictions on a set of samples and generates explanations (e.g., highlighted attention regions in image classification).
2. Users evaluate whether the explanations are reasonable, and provide corrective feedback if not.
3. The algorithm uses the feedback to improve the model.

XIL has demonstrated value in scenarios such as spam filter customization and confound correction. However, in the XIL loop, samples are presented to users in a **machine-determined order**—which may trigger cognitive biases.

### The Threat of Order Effects

**Order effects** are a class of cognitive biases in which human judgments are systematically influenced by the order in which information is presented.
- **Primacy effect**: Earlier information is given disproportionately high weight.
- **Recency effect**: Later information is given disproportionately high weight.

Specific risks in XAI/XIL contexts:
- If users first see samples where the AI performs well → they may over-trust and fail to correct subsequent errors.
- If users first see samples where the AI performs poorly → they may over-distrust and underestimate AI capability.

### Limitations of Prior Work

- Nourani et al. (2021) identified order effects, but their experimental design differed substantially from typical XIL scenarios (users self-selected samples and could not correct outputs).
- Honeycutt et al. (2020) found no significant order effect, but only examined between-session effects and relied solely on self-reported measures.
- The two studies yield contradictory conclusions and neither closely resembles real-world XIL usage.

## Method

### Overall Architecture

Two controlled user studies are designed to simulate realistic XIL debugging tasks:
- **Study 1 (Within-session)**: Order effects within a single debugging session.
- **Study 2 (Between-session)**: Order effects across two consecutive debugging sessions.

### Key Designs

#### 1. **Debugging Task Design**

Participants interact with a (fictitious) face detection model. The system presents **blurred images** (85×85 Gaussian kernel, $\sigma=40$) along with model-placed bounding boxes. Within 6 seconds, participants must:
- Confirm the bounding box if it is correct.
- Drag it to the correct position if it is wrong.

**Rationale for blurring**: To prevent the task from being trivially easy—if correct/incorrect placements were immediately obvious, order effects might not be observable.

**Bounding box placement accuracy is defined at three levels**:
- Correct: complete overlap with ground truth.
- Partially wrong: 25% overlap.
- Wrong: 0% overlap.

#### 2. **Independent Variable Design**

**Order condition**: Between-subjects variable, 3 levels:
- **Increasing (Inc)**: Low model accuracy in the first half, high in the second half.
- **Constant (Const)**: Accuracy remains unchanged throughout (control group).
- **Decreasing (Dec)**: High model accuracy in the first half, low in the second half.

**Placement correctness**: Within-subjects variable — correct vs. wrong.

**Image difficulty**: Within-subjects variable — easy vs. difficult (determined via a pilot study).

#### 3. **Dependent Variables**

- **User feedback accuracy**: Overlap ratio between user-placed bounding boxes and ground truth.
- **User–model agreement**: Overlap ratio between user placement and model placement — a behavioral measure of trust.
- **Perception questionnaire**: 4 items on a 7-point Likert scale assessing users' perceived model accuracy and trustworthiness.

#### 4. **Statistical Analysis**

Mixed linear models are employed, with fixed effects including order condition, placement correctness, image difficulty, and their interactions; random intercepts include participant and image. Significant interaction effects are further examined via post-hoc comparisons with Bonferroni correction. Questionnaire data are analyzed using the Kruskal–Wallis rank-sum test.

### Additional Details for Study 1

- Each participant evaluated 40 images plus 6 warm-up trials.
- Overall accuracy was 60% across conditions; Inc: 40% in the first half / 80% in the second half; Dec: the reverse.
- A priori power analysis indicated 330 participants are sufficient to detect small-to-medium effect sizes (82% statistical power).

### Additional Details for Study 2

- Each participant completed two sessions of 40 images each.
- Session 1 accuracy: Inc = 40%, Const = 60%, Dec = 80%.
- Between sessions, participants were told the model was "updating based on their feedback" (fictitious).
- Session 2 was identical across all three groups (60% accuracy).

## Key Experimental Results

### Study 1 (Within-session) Main Results

**User feedback accuracy**:

| Condition | Overall Accuracy | Correct Images | Wrong Images |
|-----------|-----------------|----------------|--------------|
| Inc (Increasing) | 0.76±0.10 | 0.79±0.11 | 0.70±0.08 |
| Const (Constant) | 0.75±0.08 | 0.80±0.10 | 0.67±0.11 |
| Dec (Decreasing) | 0.76±0.08 | 0.79±0.11 | 0.70±0.08 |

- Image difficulty showed a strong main effect ($F(1,36)=63.33, p<.001$).
- Placement correctness showed a significant main effect ($F(1,36)=10.65, p=.002$).
- Order condition was significant only in a two-way interaction with placement correctness ($F(2,12875)=4.56, p=.011$), with a very small effect size.

**User–model agreement**:

| Scenario | Inc | Const | Dec |
|----------|-----|-------|-----|
| Correct + Easy | ~0.92 | ~0.92 | ~0.92 |
| Correct + Difficult | ~0.67 | ~0.67 | ~0.67 |
| Wrong + Easy | 0.14±0.06 | 0.15±0.12 | 0.13±0.03 |
| **Wrong + Difficult** | **0.19±0.08** | **0.24±0.11** | **0.24±0.12** |

- The three-way interaction was significant ($F(2,12862)=7.99, p<.001$).
- **Key finding**: The Inc group exhibited the lowest reliance on the model for "wrong + difficult" images — interpretable as a weak **primacy effect**: early exposure to model errors rendered users more cautious.

**Questionnaire**: No significant differences across groups ($p=.909$); perceived model quality was entirely unaffected by order.

### Study 2 (Between-session) Key Results

**Session 2 feedback accuracy**:

| Condition | Overall Accuracy | Correct Images | Wrong Images |
|-----------|-----------------|----------------|--------------|
| Inc | 0.78±0.08 | 0.84±0.11 | 0.68±0.09 |
| Const | 0.78±0.07 | 0.84±0.10 | 0.69±0.09 |
| Dec | 0.78±0.08 | 0.83±0.11 | 0.70±0.09 |

**Session 2 agreement**: No significant differences across groups.
- Inc: 0.63±0.10
- Const: 0.62±0.10
- Dec: 0.61±0.09

**Questionnaire**: No significant differences ($p=.821$).

### Key Findings

1. **Weak within-session order effects exist**: Early exposure to model errors (Inc condition) led users to rely less on the model for difficult + wrong samples — possibly a small primacy effect.
2. **No between-session order effects**: Session 1 model performance did not influence user behavior in Session 2 — users appear to "reset their expectations" upon model updates.
3. **Feedback quality remains consistently high**: User accuracy ranged from 0.75 to 0.78 across all conditions.
4. **Self-reported perceptions are unaffected**: Questionnaire scores were nearly identical across groups.

## Highlights & Insights

- **Large scale by HCI/XAI standards**: 713 participants, substantially exceeding comparable studies.
- **Parallel behavioral and self-report measurement**: The study captures not only what users *say* (perception) but what they *do* (accuracy, agreement), providing more reliable evidence.
- **Direct practical guidance for XIL deployment**:
    - Randomizing sample presentation order within a session can mitigate the weak within-session effect.
    - Between-session effects need not be a concern — users naturally adapt to model updates.
- **Elegant experimental design**: The constant condition serves as a baseline, with increasing and decreasing conditions as mutual contrasts, ensuring any observed differences can be attributed to order rather than content.

## Limitations & Future Work

- Only order effects arising from the distribution of errors are examined; order effects due to difficulty distribution or explanation type are not considered.
- Only bounding-box explanations for image classification are studied; other explanation modalities (concept-level explanations, counterfactual examples, etc.) are untested.
- The task is relatively simple (face localization); order effects may be more pronounced in complex domain tasks such as medical imaging.
- Participants did not interact with a genuinely learning model — model updates were fictitious. Real closed-loop XIL may produce different outcomes.

## Related Work & Insights

- Compared to Nourani et al. (2021): The automation bias they observed (early exposure to correct predictions → over-reliance) is only weakly replicated in the present study.
- Compared to Honeycutt et al. (2020): Both studies find no between-session order effects, but the present work additionally examines within-session effects and employs behavioral measures.
- Implication: Query selection strategies in XIL algorithms (e.g., uncertainty sampling) naturally induce order effects — but based on the present findings, this need not be a major concern.

## Rating

- Novelty: ⭐⭐⭐ — The research question is important but not entirely new; the work represents a more rigorous replication of an ongoing debate.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 713-participant large-scale experiment, a priori power analysis, dual behavioral and self-report measures, within- and between-session dual design.
- Writing Quality: ⭐⭐⭐⭐ — Follows psychological experiment reporting conventions; statistical analysis is thorough.
- Value: ⭐⭐⭐⭐ — Provides solid empirical evidence that order effects need not be a major concern in the practical deployment of the XIL framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)
- [\[AAAI 2026\] Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)
- [\[AAAI 2026\] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)

</div>

<!-- RELATED:END -->
