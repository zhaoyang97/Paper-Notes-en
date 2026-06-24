---
title: >-
  [Paper Note] SocialEval: Evaluating Social Intelligence of Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Social Intelligence] Proposes SocialEval, a bilingual social intelligence benchmark based on narrative scripts. By manually constructing 153 "World Trees" that model social interactions as goal-conditioned MDPs, it integrates outcome-oriented Goal Achievement Evaluation (GAE) and process-oriented Interpersonal Ability Evaluation (IAE) to systematically evaluate the social intelligence of LLMs in multi-turn social scenarios and their gaps with humans.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Social Intelligence"
  - "Benchmark"
  - "World Tree"
  - "Interpersonal Ability"
  - "Goal Achievement"
  - "BESSI"
date: 2026-05-08
content_hash: 78447367ced87c31
---

# SocialEval: Evaluating Social Intelligence of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.00900](https://arxiv.org/abs/2506.00900)  
**Code**: [https://github.com/thu-coai/SocialEval](https://github.com/thu-coai/SocialEval)  
**Area**: LLM NLP / Social Intelligence Evaluation  
**Keywords**: Social Intelligence, Benchmark, World Tree, Interpersonal Ability, Goal Achievement, BESSI

## TL;DR

Proposes SocialEval, a bilingual social intelligence benchmark based on narrative scripts. By manually constructing 153 "World Trees" that model social interactions as goal-conditioned MDPs, it integrates outcome-oriented Goal Achievement Evaluation (GAE) and process-oriented Interpersonal Ability Evaluation (IAE) to systematically evaluate the social intelligence of LLMs in multi-turn social scenarios and their gaps with humans.

## Background & Motivation

- **Background**: LLMs exhibit emergent social intelligence (SI) in social simulations (such as Generative Agents) and are widely used in social science research and interpersonal scenario training. However, the fundamental question of "how strong LLMs' SI actually is and how large the gap with humans remains" has not been fully answered.
- **Limitations of Prior Work**: Existing evaluation endeavors (e.g., SOTOPIA, AgentSense) suffer from two major limitations: (1) they focus solely on single-turn social dynamics, ignoring the sequentially dependent process of multi-turn social interactions; (2) they assess only final goal achievement, lacking fine-grained evaluation of the deployment of interpersonal abilities during the goal-pursue process.
- **Key Challenge**: Social psychology indicates that SI is inherently a dynamic process—individuals continuously adjust interpersonal skills in evolving narrative social activities (script theory) to achieve social goals. Existing methods fail to capture this complete process.
- **Key Insight**: Drawing inspiration from script theory (Schank & Abelson) and dramaturgical theory (Goffman), this work models social activities as "World Trees"—tree structures interwoven with multiple plot lines driven by interpersonal abilities. This allows the simultaneous evaluation of LLM navigation outcomes (goal achievement) and navigation processes (ability application) within the tree.

## Method

### Taxonomy of Social Worlds

Based on interdependence theory (Kihlstrom & Cantor), a taxonomy of social worlds is constructed using the Cartesian product of two dimensions: self-interest and altruism. Each dimension takes values in $\{1, 0, -1\}$, producing 9 orientations. After excluding chitchat $(0,0)$ due to lack of goals and the rare withdrawal $(-1,0)$, 7 social orientations are retained:

| Category | Orientation | (Self-interest, Altruism) | Typical Behavior |
|------|------|-------------|---------|
| Prosocial | Cooperation | (1, 1) | Mutually beneficial cooperation between both parties |
| Prosocial | Negotiation | (1, 0) | Striving for one's own optimal outcome |
| Prosocial | Assistance | (0, 1) | Actively helping others |
| Prosocial | Altruism | (-1, 1) | Sacrificing self to support others |
| Pro-self | Competition | (1, -1) | Benefiting oneself at the expense of others |
| Antisocial | Induction | (0, -1) | Manipulating others to cause them harm |
| Antisocial | Conflict | (-1, -1) | Adversarial and destructive behaviors |

### Interpersonal Ability Inventory (BESSI Framework)

Adopting the psychological BESSI framework, this work defines 5 major ability dimensions and 32 specific interpersonal abilities as the process evaluation dimensions:

- **Social Engagement** (5 abilities): Leadership, persuasion, conversation, expressiveness, energy management
- **Cooperation** (5 abilities): Teamwork, trust, empathy, social warmth, ethics
- **Self-Regulation** (12 abilities): Task/time/detail management, goal regulation, decision-making, adaptability, independence, self-reflection, etc.
- **Emotional Resilience** (5 abilities): Stress regulation, optimism, anger management, confidence, impulse control
- **Innovation** (5 abilities): Abstract thinking, creativity, artistry, cultural competence, information processing

### World Tree Construction

Each World Tree consists of the following components:

- **Characters**: One protagonist + several supporting characters, each having public information, private information, and social goals.
- **Scene**: The root node of the World Tree, referring to interactive video creations on Bilibili/YouTube.
- **Episodes**: Composed of conversational interactions among characters, averaging 6.5 episodes per tree.
- **Episode Transitions**: Critical decision points where the protagonist selects from multiple candidate utterances (averaging 2.17), each reflecting different interpersonal abilities and leading to different subsequent plotlines.
- **Episode Endings**: Annotated with whether social goals are achieved, used for GAE.

Quality control process: Inspectors are trained first (piloting 2 World Trees), followed by a three-phase cross-checking, reaching a final consensus rate of **95%**. Translations use GPT-4o and are verified by professional translators, with an acceptance rate of **97%**.

### Dual-Task Evaluation Framework

Formulating each World Tree as a goal-conditioned MDP $(S, A, T, R)$, two evaluation tasks are defined:

**Task 1 — GAE (Goal Achievement Evaluation)**: The LLM plays the protagonist, choosing an utterance at each episode transition point to advance the plot. After multi-step decisions, an ending is reached, and whether the social goal is met is determined. The metric is the goal achievement rate.

**Task 2 — IAE (Interpersonal Ability Evaluation)**: For each candidate utterance, a multiple-choice question is constructed, containing the correct utterance and several plausible distractors reflecting incorrect abilities, testing whether the LLM correctly identifies the targeted ability. The metric is the ability selection accuracy.

To eliminate position bias, the option order for each sample is randomly shuffled 3 times, and majority voting is applied. The human baseline was completed by 20 Chinese and English native-speaking graduate students.

## Key Experimental Results

### GAE Goal Achievement Evaluation (Goal Achievement Rate %)

| Model | Prosocial zh/en | Pro-self zh/en | Antisocial zh/en | Overall zh/en |
|------|-------------|-------------|-------------|-----------|
| Human (best) | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| Human (avg) | 64.9/59.9 | 55.0/40.0 | 51.3/50.0 | 61.8/55.2 |
| DeepSeek-R1 | 54.3/52.8 | 32.2/30.8 | 28.0/26.4 | 47.1/45.7 |
| o1 | 54.3/52.7 | 32.5/31.0 | 27.5/25.7 | 47.0/45.4 |
| Claude-3-opus | 54.0/52.3 | 31.5/29.8 | 29.7/27.6 | 47.0/45.2 |
| DeepSeek-V3 | 53.4/51.9 | 30.1/28.5 | 25.7/23.5 | 46.4/44.8 |
| GPT-4o | 52.8/51.6 | 27.6/25.5 | 23.2/17.6 | 44.6/42.7 |
| Qwen-2.5-72B | 47.5/44.4 | 26.6/23.4 | 20.2/16.1 | 40.3/37.1 |
| Llama-3.1-8B | 37.9/33.7 | 21.2/18.7 | 13.0/10.8 | 31.8/28.2 |

### IAE Interpersonal Ability Evaluation (Selection Accuracy %)

| Model | Social Engagement zh/en | Cooperation zh/en | Self-Regulation zh/en | Emotional Resilience zh/en | Innovation zh/en | Overall zh/en |
|------|--------------|----------|-------------|-------------|----------|----------|
| Human (best) | 84.9/85.7 | 89.6/92.9 | 86.3/80.6 | 81.8/86.7 | 81.5/85.7 | 85.7/85.3 |
| Human (avg) | 79.5/82.2 | 82.7/84.6 | 80.5/74.5 | 78.6/79.1 | 76.8/79.9 | 80.2/79.1 |
| DeepSeek-R1 | 77.3/76.3 | 83.5/81.9 | 75.2/73.4 | 78.1/76.6 | 75.2/73.2 | 77.6/75.4 |
| Claude-3-opus | 77.5/76.5 | 84.5/82.5 | 74.6/72.5 | 76.9/72.4 | 75.2/72.1 | 77.6/75.3 |
| o1 | 76.2/74.7 | 82.4/80.8 | 75.1/73.1 | 78.1/76.6 | 75.0/73.1 | 77.3/75.5 |
| DeepSeek-V3 | 75.6/74.2 | 81.4/78.5 | 74.9/72.4 | 77.6/76.3 | 74.9/72.9 | 76.5/73.8 |
| Qwen-2.5-72B | 68.9/64.6 | 75.7/73.5 | 68.3/64.6 | 71.2/65.2 | 69.0/64.4 | 70.4/66.5 |
| Mistral-7B | 59.4/55.2 | 64.2/62.1 | 58.5/52.6 | 55.4/52.5 | 49.9/47.7 | 58.8/54.7 |

## Key Findings

1. **LLMs lag significantly behind humans**: The best LLM (DeepSeek-R1) still lags behind Human-avg by 14.7/9.5 percentage points (zh/en) on GAE, with the gap mainly originating from pro-self and antisocial scenarios.

2. **LLMs exhibit a strong prosocial preference**: The performance gap for humans between prosocial and antisocial scenarios is about 13.7 percentage points, whereas for DeepSeek-R1, this gap is as high as 26.3 percentage points. LLMs prefer choosing positive and helpful behaviors even when it leads to goal failure, whereas humans flexibly adjust behavioral strategies.

3. **Significant cross-lingual SI variance**: Wilcoxon signed-rank tests ($p<0.001$) show a significant difference in LLM performance on Chinese vs. English SI evaluations, with Chinese generally outperforming English, aligning with human cross-lingual patterns.

4. **Positive correlation with model scale**: The SI of open-source LLMs correlates positively with parameter size. For the Qwen-2.5 series from 7B to 72B, overall GAE improved from 33.9/29.9 to 40.3/37.1.

5. **Multiple-choice questions reflect generative capacities**: The semantic similarity between LLM-generated utterances and the candidate utterances reaches up to 67% (zh), and the similar utterance is selected at a rate of over 80%, validating the effectiveness of the multiple-choice evaluation.

6. **Brain-like functional partitioning**: Representational space analysis of LLaMA-3.1 shows that the 5 ability dimensions in the 8B model are initially clustered but overlap, while the 70B model forms clearly segregated clusters. Neuronal activation analysis indicates that neurons corresponding to interpersonal abilities in larger models gradually become denser and more isolated, resembling functional partitioning in the human brain (lobes hypothesis).

## Highlights & Insights

- **World Tree Modeling Paradigm**: Modeling social interactions as World Trees (MDPs) enables both outcome and process evaluations; this dual-dimension design is significantly superior to existing methods that only measure terminal outcomes.
- **Alignment Implications of Prosocial Preferences**: LLMs would rather fail than take aggressive actions, suggesting that current alignment training may over-constrain behavioral flexibility in social contexts.
- **Novel Brain-inspired Analytical Approach**: Using t-SNE clustering and Wanda Score to analyze the representations and neuronal distributions of interpersonal abilities in LLMs concretizes the analogy of "brain functional partitioning" for the first time.

## Rating

- ⭐⭐⭐⭐ Novelty: The World Tree + dual-task evaluation paradigm possesses strong originality.
- ⭐⭐⭐⭐ Experimental Thoroughness: 19+ LLMs, human baselines, behavioral analysis, representation analysis, and neuronal analysis.
- ⭐⭐⭐ Scalability: High manual construction cost (approx. 12 hours/$40 per tree); only 20 trees for antisocial scenarios.
- ⭐⭐⭐⭐ Value: Provides a comprehensive evaluation tool and in-depth insights for understanding and improving the social behaviors of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models](catching_shortcuts_a_framework_for_evaluating_shortcuts_in_large_language_models.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)

</div>

<!-- RELATED:END -->
