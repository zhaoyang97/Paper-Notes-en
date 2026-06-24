---
title: >-
  [Paper Note] Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games
description: >-
  [ACL 2025][personality traits] The PANDA method is proposed to project human personality traits (a total of 8 traits from the Big Five and Dark Triad) into the policy learning of text-based game agents. By guiding Q-value adjustment through a personality classifier, it is discovered that the High Openness personality significantly outperforms other personality types in adventure-based text games.
tags:
  - "ACL 2025"
  - "personality traits"
  - "text-based games"
  - "Big Five"
  - "Dark Triad"
  - "DRRN"
  - "reinforcement learning"
date: 2026-05-08
content_hash: d1df0d2cf18bd514
---

# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games

## Basic Information

**Conference**: ACL 2025  
**Code**: [pull-ups/PANDA](https://github.com/pull-ups/PANDA)  
**Model**: [mirlab/PersonalityClassifier](https://huggingface.co/mirlab/PersonalityClassifier)  
**Area**: Others  
**Keywords**: personality traits, text-based games, Big Five, Dark Triad, DRRN, reinforcement learning  

## TL;DR

The PANDA method is proposed to project human personality traits (a total of 8 traits from the Big Five and Dark Triad) into the policy learning of text-based game agents. By guiding Q-value adjustment through a personality classifier, it is discovered that the High Openness personality significantly outperforms other personality types in adventure-based text games.

## Background & Motivation

- **Challenges of Text-based Games**: As a classic challenge in AI, text-based games differ from games with pre-defined action spaces like Atari or Go. They require understanding environments described in natural language and generating actions in text format, involving complex natural language understanding and generation capabilities.
- **Expansion of Value Alignment**: Existing work focuses on integrating general value systems (such as ethics and social norms) into agent behaviors but has yet to explore how diverse **intrinsic personality traits** affect agent decision-making.
- **Core Problem**: How do different personality traits affect the behavior and performance of agents in interactive environments? Are there certain "advantageous" personality types?
- **Research Significance**: Evaluating agent behavior by introducing the dimension of personality provides a new perspective for developing more human-centered and behavior-controllable AI systems.

## Method

### Overall Architecture: PANDA (Personality-Adapted Neural Decision Agents)

The core idea of PANDA is to achieve personality-guided agent decision-making through two steps:

1. **Training a personality classifier**: Determining which personality traits an action exhibits under a given context.
2. **Integrating personality guidance into policy learning**: Achieving personality-constrained action selection by adjusting Q-values.

### Modeling Text-Based Game Environments

- Text-based games are modeled as a **Partially Observable Markov Decision Process (POMDP)**: $(S, T, A, O, R)$
- Under the latent state $S$, the agent receives a textual observation $O$, generates a textual action $A$, and changes the state according to the transition function $T$.
- The **Jiminy Cricket** benchmark is used, which consists of 25 complex text adventure games, covering over 1,800 locations and nearly 5,000 interactive objects.

### Base Agent Architecture: DRRN

- The **Deep Reinforcement Relevance Network (DRRN)** is adopted as the base framework.
- The neural network is trained to predict the Q-value $Q(s_t, a_t)$, which represents the value function of an action given a state.
- Actions that maximize the Q-value are selected using a softmax policy.

### Key Designs: Personality-Guided Q-Value Adjustment

The core formula for calculating the adjusted Q-value is:

$$Q'(s_t, a_t^i) = Q(s_t, a_t^i) + \gamma \cdot C(s_t, a_t^i \mid p)$$

- $Q(s_t, a_t^i)$: Action value from the original policy network.
- $C(s_t, a_t^i \mid p) \in \{-1, 0, 1\}$: The output of the personality classifier, indicating high/medium/low valence of the action on personality $p$.
- $\gamma$: Guidance intensity and direction. $\gamma > 0$ enhances behaviors matching the personality, while $\gamma < 0$ suppresses behaviors matching the personality.

Action selection policy:

$$\pi(a_t = a_t^i | s_t) = \frac{\exp(Q(s_t, a_t^i))}{\sum_{j=1}^{|A_t|} \exp(Q(s_t, a_t^j))}$$

### Personality Classifier Construction

**Dataset Construction (120,000 samples)**:

1. **Initial Validated Personality Descriptions**: Using validation questionnaire items from the BFI (Big Five Inventory) and SD-3 (Short Dark Triad), 10 instances (5 high-valence + 5 low-valence) are generated for each trait, totaling 80 items.
2. **Scenario Seed Augmentation**: GPT-4 is used to generate 300 diverse scenarios (divided into 30 subsets, with 10 scenarios per subset).
3. **Final Scale**: $80 \times 300 \times 5 = 120,000$ samples with personality labels.

**8 Personality Traits Covered**:
- Big Five: Openness (Ope.), Conscientiousness (Con.), Extraversion (Ext.), Agreeableness (Agr.), Neuroticism (Neu.)
- Dark Triad: Psychopathy (Psy.), Narcissism (Nar.), Machiavellianism (Mac.)

**Classifier Training**:
- Fine-tuned using Flan-T5-XL (3 billion parameters).
- Input: Context + Action + Personality Type $\rightarrow$ Output: High/Medium/Low valence.
- Achieves **98.59%** accuracy on the validation set.

## Experiments

### Main Experimental Setup

- **Environment**: 25 text adventure games from the Jiminy Cricket benchmark.
- **Agent Configurations**: 16 personality profiles (8 personalities $\times$ 2 directions [High/Low]) + 1 no-personality baseline.
- **Evaluation**: Average score of the last 50 episodes for each configuration, using 3 random seeds.

### Main Results (Table 3)

Performance in 15 valid games (none of the personality agents could score in the remaining 10 games):

| Metric | High Openness | No Personality | Low Openness |
|------|--------------|----------------|--------------|
| Average Score (Avg.) | **6.5** | 5.6 | 4.4 |
| Number of Dominant Games (Cnt.) | **11** | - | 0 |
| Difference (Diff.) | +2.1 | - | - |

**Key Findings**: High Openness agents lead across all three metrics, with an average score increase of 16%, achieving the best performance in 11/15 games.

### Statistical Analysis (Table 4)

- **Wilcoxon Signed-Rank Test**: High Openness vs No Personality, $p = 0.002$; High Openness vs Low Openness, $p = 0.000$.
- **Friedman Test**: The test statistic for Openness is $Fr = 25.2$, $p = 0.000$, far exceeding other personality traits.
- No other personality traits showed a statistically significant consistent advantage.

### Trajectory Analysis (Table 5)

| Metric | No Personality | High Openness | Low Openness |
|------|----------------|---------------|--------------|
| Trajectory Length | 45.85 | **57.04** | 39.86 |
| Visited Nodes (Total) | 9.49 | **10.16** | 8.32 |
| Unexplored Nodes | 0.83 | **1.20** | 0.30 |

High Openness agents exhibit longer trajectories and visit more locations (especially unexplored new locations), which reflects their curiosity and tendency to explore new things.

### Ablation Study

- **Guidance Intensity $\gamma$**: The impact of different $\gamma$ values was tested. It was found that a proper guidance intensity can effectively steer behavior without excessively disrupting policy learning.
- **Classifier Comparison**: Flan-T5-XL achieves 100% and 96.29% on the BFI and SD-3 validation sets respectively, significantly outperforming zero-shot GPT-4o-mini (81.81% and 22.22%).

## Highlights & Insights

1. **Intriguing Correlation Between Personality and Game Performance**: The High Openness trait (curiosity, adventurous spirit) has an inherent advantage in text adventure games, validating the applicability of psychological theories to AI behavior.
2. **Simple and Efficient Guidance Mechanism**: Personality guidance is achieved through a simple additive modification of Q-values, requiring no alterations to the network architecture or training process.
3. **Large-Scale Personality-Labeled Dataset**: A dataset of 120,000 personality-labeled samples ensures diversity and coverage through scenario-seed expansion.
4. **Quantitative Validation of Personality Effects**: Beyond demonstrating the impact of personality differences on performance, the underlying behavioral mechanisms are uncovered via trajectory analysis (High Openness $\rightarrow$ more exploration $\rightarrow$ higher scores).

## Limitations & Future Work

1. **Environmental Limitations**: Validated only in text adventure games. The advantage of openness might apply only to exploration-oriented tasks and may not generalize to scenarios requiring cautious decision-making.
2. **Limitations in the LLM Era**: The base framework relies on DRRN (a traditional RL method) without exploring the effect of personality guidance when using LLMs as agents.
3. **Personality Data Source**: The training data is generated by GPT-4, which may introduce bias.
4. **Zero Scores in 10 Games**: This indicates that in excessively difficult games, the performance of personality guidance is constrained by the capability of the base policy.
5. **Single $\gamma$ Value**: The guidance intensity was not dynamically adjusted for different games or personalities.

## Related Work & Insights

- **Text-Based Game AI**: RL-based methods such as DRRN (He et al., 2016), CALM (Yao et al., 2020), etc.
- **Agent Value Alignment**: Ethical evaluation from Jiminy Cricket (Hendrycks et al., 2021), moral benchmarks from MACHIAVELLI (Pan et al., 2023).
- **Personality Computing**: Applications of psychological scales like Big Five (McCrae & Costa, 1987) and Dark Triad (Jones & Paulhus, 2014) in NLP.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: Applying psychological personality theory to policy learning of game agents, presenting a novel perspective (+1)
- **Method Simplicity**: The Q-value additive modification approach is simple and direct, making it easy to implement and understand (+0.5)
- **Experimental Thoroughness**: 25 games, 16 personality configurations, statistical tests, and trajectory analyses, ensuring a comprehensive evaluation (+0.5)
- **Practical Impact**: Uncovering that the Openness personality favors exploration, providing a valuable reference for value alignment and persona-based AI (+0.5)
- **Deductions**: Handled without validation on LLM agents, ineffective in some games, and highly specific environment setting (-1)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Deviation Dynamics in Cardinal Hedonic Games](../../AAAI2026/others/deviation_dynamics_in_cardinal_hedonic_games.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)

</div>

<!-- RELATED:END -->
