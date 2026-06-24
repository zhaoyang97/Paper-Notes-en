---
title: >-
  [Paper Note] Dialogue Systems for Emotional Support via Value Reinforcement
description: >-
  [ACL 2025][Dialogue Systems][emotional support] This paper proposes ES-VR, the first method that integrates human value reinforcement into emotional support dialogue systems. By leveraging a target value detector and a reference generator (both trained on Reddit data), combined with a two-stage SFT + DPO training scheme, the supporter model not only alleviates the seeker's negative emotions but also explores and reinforces their positive values, achieving a deeper…
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "emotional support"
  - "value reinforcement"
  - "dialogue system"
  - "DPO"
  - "seeker simulator"
date: 2026-05-08
content_hash: 6bb7cc4a0f5faddd
---

# Dialogue Systems for Emotional Support via Value Reinforcement

**Conference**: ACL 2025  
**arXiv**: [2501.17182](https://arxiv.org/abs/2501.17182)  
**Code**: [GitHub](https://github.com/holi-lab/ES-Value)  
**Area**: Text Generation / Dialogue Systems  
**Keywords**: emotional support, value reinforcement, dialogue system, DPO, seeker simulator

## TL;DR

This paper proposes ES-VR, the first method that integrates human value reinforcement into emotional support dialogue systems. By leveraging a target value detector and a reference generator (both trained on Reddit data), combined with a two-stage SFT + DPO training scheme, the supporter model not only alleviates the seeker's negative emotions but also explores and reinforces their positive values, achieving a deeper, internal transformation.

## Background & Motivation

**Background**: Emotional support dialogue systems aim to help seekers alleviate daily emotional difficulties. Recently, LLMs have accelerated the development of such systems, with many models focusing on reinforcing the seeker's positive emotions. However, merely focusing on emotional changes may fail to capture deeper, internal transformations.

**Limitations of Prior Work**: Emotional changes do not equate to real internal transformations—a superficial "thank you" from a seeker can score higher (0.758) in emotional classifiers than a response expressing genuine willingness to change (0.583), yet the latter is the truly effective support outcome. Most existing methods ignore the role of **human values** in emotional support.

**Key Challenge**: Values are core beliefs that shape personal priorities and play a crucial role in modern psychotherapies like ACT (Acceptance and Commitment Therapy). However, existing dialogue systems have rarely explored integrating value reinforcement into emotional support.

**Goal**: How to train a dialogue system that not only provides emotional comfort but also identifies and reinforces the seeker's positive values, promoting long-term internal change.

**Key Insight**: Utilize Reddit r/offmychest community data to train a value detector and a reference generator, and then train the supporter model through simulated dialogues + DPO.

**Core Idea**: Replace traditional "emotion reinforcement" with "value reinforcement" as the optimization objective of emotional support, achieved through target value detection, reference response generation, and value reward-based DPO training.

## Method

### Overall Architecture

Three core components: (1) **Target Value Detector**—predicts which values should be reinforced in each turn; (2) **Reference Generator**—generates reference responses that promote target values; (3) **Supporter Model**—combines target values and reference responses, selects strategies, and generates final responses.

### Key Designs

1. **Value Classification Taxonomy**: Adopts the taxonomy from Kiesel et al. (2022), integrating Schwartz's Core Value theory and three other major value lists, totaling 20 value categories. Analysis of the ESConv dataset reveals that: seekers in the high-intervention-effect group (emotions dropping from 5 to 1-2 post-support) express an average of **7.9 positive values** in the last 4 turns, significantly higher than the **6.5** in the low-effect group.

2. **Reddit Data Construction**: Collects posts and comments from r/offmychest from 2019-2023, annotated using an emotion intensity model and a value detection model (Schroter et al., 2023). Values expressed by the OP in positive comments are treated as successful target values, and prior commenter replies are treated as effective support utterances. The final dataset exceeds 20,000 entries.

3. **Target Value Detector**: Given a dialogue history $(o_1, c_1, ..., c_{t-1}, o_t)$, predicts the values to be reinforced in the next turn $v_{t+1} = \text{LM}_{\text{TVD}}(o_1, c_1, ..., c_{t-1}, o_t)$. The ground truth labels are the top-3 values with the highest probabilities in $o_{t+1}$.

4. **Reference Generator**: Two-stage training—The SFT stage learns to generate supporter responses $c_t = \text{LM}_{\text{RG}}(o_1, c_1, ..., o_t; v_{t+1})$ based on dialogue history and target values. The DPO stage constructs preference data (original replies as preferred, other comments under the same post as rejected, excluding those with overlapping shared values) to further optimize generation quality.

5. **Supporter Model**: Processes four reasoning steps in each turn: (a) Identify the seeker's issues and current state; (b) Analyze key content of the reference response; (c) Decide whether to adopt the reference response (generating Yes/No and reasoning); (d) Select an emotional support strategy and generate the final response.

    - **SFT Stage**: Uses GPT-4o-mini as both the supporter and seeker simulator to generate dialogue data (33,130 training, 2,367 validation), which is distilled into Llama-3-8B-Instruct. To prevent the model from inheriting GPT's 90% preference for *not* using reference responses, "alternative responses" are additionally simulated in each supporter turn.
    - **DPO Stage**: Constructs preference data based on value rewards: $R(u_t^{\text{sup}}) = \sum_{k=1}^h \gamma^{k-1} N_{t+k}$, where $N_{t+k}$ is the frequency of target values appearing in the seeker's subsequent $k$-th turn of utterances, $\gamma$ is the discount factor, and $h$ is the look-ahead step size. Pairs are added to the preference dataset when the reward difference exceeds a threshold $T_{\text{diff}}$.

6. **Seeker Simulator**: Based on GPT-4o-mini, using GPT-4o/4o-mini to generate 2,036 unique personas (including problem types, emotions, and contexts). Human evaluation shows its naturalness is no less than that of real human seekers.

### Loss & Training

- SFT: Standard language model negative log-likelihood
- DPO: Direct Preference Optimization (Rafailov et al., 2023) with implicit reward modeling

## Key Experimental Results

### GPT-4o-mini Ablation (Effect of Value + Reference)

| Settings | Sugg.↑ | Expe.↑ | Info.↑ | Overall↑ | Intensity↓ | Seeker-Value↑ | Supporter-Value↑ |
|------|--------|--------|--------|----------|------------|---------------|------------------|
| GPT Baseline | 4.03 | 2.34 | 4.11 | 4.44 | 2.19 | 0.43 | 0.36 |
| + Target values | 4.38 | 2.48 | 4.27 | 4.59 | 1.96 | 0.48 | 0.48 |
| + Reference | 4.34 | 2.54 | 4.29 | 4.61 | 1.89 | 0.47 | 0.42 |
| + Both | **4.57** | **3.11** | **4.42** | **4.72** | **1.89** | — | — |

### Main Results

| Method | Sugg.↑ | Expe.↑ | Info.↑ | Over.↑ | Intensity↓ | Seeker-V↑ | Supp-V↑ |
|------|--------|--------|--------|--------|------------|-----------|---------|
| GPT-4o-mini (+Both) | 4.57 | 3.11 | 4.42 | 4.72 | 1.89 | 0.49 | 0.42 |
| Llama-Psych8k | 4.75 | 2.89 | 4.63 | 4.75 | 1.53 | 0.49 | 0.62 |
| PPDPP | 4.45 | 2.49 | 4.26 | 4.54 | 1.83 | 0.44 | 0.31 |
| Emotion-DPO | 4.74 | 4.05 | 4.61 | 4.82 | 1.86 | 0.49 | 0.51 |
| **ES-VR (DPO)** | **4.80** | **4.20** | **4.65** | **4.86** | 1.81 | **0.52** | **0.56** |

ES-VR (DPO) significantly outperforms GPT-4o-mini in Experience (4.20 vs 3.11) and value reinforcement (Supporter 0.56 vs 0.42).

### Psychotherapist Evaluation

Therapists performed pairwise comparisons of ES-VR against GPT and Emotion-DPO. The advantages of ES-VR lie in:
- **Validating challenges**
- **Emphasizing positive aspects**
These two elements are the core components of value reinforcement.

### Key Findings

- **Value Reinforcement > Emotion Reinforcement**: ES-VR (DPO) comprehensively outperforms Emotion-DPO on ES-Value metrics, proving the effectiveness of shifting the target from emotions to values.
- **Reddit Crowdsourced Knowledge has Value**: The reference generator trained on Reddit data significantly enhances support effects (Experience increases from 2.34 to 3.11).
- **DPO Phase is Crucial**: SFT to DPO shows a leap in Experience from 3.76 to 4.20, and Overall from 4.78 to 4.86.
- **High-Quality Seeker Simulator**: Human evaluation shows that the GPT-4o-mini-based simulator is as natural as real human seekers.
- **Selective Adoption of Reference Responses is Important**: Using reference responses only when appropriate (~10%) rather than forcing usage every time.

## Highlights & Insights

- **Pioneering Nature**: First to explicitly integrate value reinforcement into emotional support dialogue systems, opening up a new research direction backed by psychological theory.
- **ESConv Dataset Analysis** provides strong motivational evidence—positive value expression is significantly higher in high-effect support sessions.
- **Elegant Reward Function Design**: Uses a look-ahead window $R = \sum \gamma^{k-1} N_{t+k}$ to evaluate value reinforcement, capturing long-term impact better than single-step rewards.
- **Thoroughly Validated Simulator**: Performed not only automatic evaluations but also human evaluations on simulator naturalness, enhancing the credibility of experimental findings.
- **Therapist-involved Evaluation** (licensed clinical psychologists) adds professional authority.

## Limitations & Future Work

- Relies on GPT-4o-mini for dialogue simulation and evaluation, posing a risk of circular dependency—data generated by GPT may favor GPT's evaluation preferences.
- Unclear whether the 2,036 personas in the seeker simulator sufficiently cover the diversity of real-world emotional support scenarios.
- Accuracy of the value detection model (Schroter et al., 2023) directly affects the entire pipeline, but its error propagation is not thoroughly analyzed.
- Evaluations were conducted in simulated environments without deployment in real-user scenarios for validation.
- Reddit data quality and representation may be biased—the r/offmychest community has a specific demographic user base.

## Related Work & Insights

- **ESConv** (Liu et al., 2021) is the most critical emotional support dataset, from which this paper discovered the correlation between values and support outcomes.
- **PPDPP** (Deng et al., 2024) represents simulation-based emotional support training methods, upon which ES-VR introduces the value dimension.
- Psychological theory of **ACT (Acceptance and Commitment Therapy)** provides the core motivation for this work.
- **DPO** (Rafailov et al., 2023) direct preference optimization is highly suitable here—value reinforcement rewards naturally form preference pairs.
- Insight: Translating validated therapeutic frameworks (e.g., value-directed intervention) into optimizable NLP objectives is a promising interdisciplinary path.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to integrate value reinforcement into emotional support systems, presenting a highly pioneering direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional evaluation (skills/emotions/values), therapist evaluation, and simulator validation, but lacks real-user experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Strong motivational proof (ESConv data analysis) and clear methodology description.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new research direction, well-supported by psychological theory, with data and code fully open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](../../ACL2026/dialogue/stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_.md)
- [\[ACL 2025\] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)
- [\[ACL 2025\] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](../../ACL2026/dialogue/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)

</div>

<!-- RELATED:END -->
