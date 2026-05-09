---
title: >-
  [Paper Note] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions
description: >-
  [AAAI 2026][LLM Safety][Multi-Agent System] This paper presents the first systematic study of persona-induced bias in LLM-based multi-agent interactions. Through controlled experiments on collaborative problem solving and persuasion tasks, three key findings are revealed: (1) different personas exhibit significant divergence in trustworthiness and insistence (dominant groups such as males and White individuals are perceived as less trustworthy); (2) agents display pronounced in-group favoritism; and (3) these biases persist and tend to amplify in multi-turn, multi-agent settings.
tags:
  - AAAI 2026
  - LLM Safety
  - Multi-Agent System
  - Persona Bias
  - LLM Bias
  - In-group Favoritism
  - Social Identity Theory
date: 2026-05-08
content_hash: 997e637145dbd2d3
---

# From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions

**Conference**: AAAI 2026
**arXiv**: [2511.11789](https://arxiv.org/abs/2511.11789)
**Code**: [https://github.com/Jiayi-LizzZ/Persona-Induced-Bias-in-MAS](https://github.com/Jiayi-LizzZ/Persona-Induced-Bias-in-MAS)
**Area**: AI Safety
**Keywords**: Multi-Agent System, Persona Bias, LLM Bias, In-group Favoritism, Social Identity Theory

## TL;DR
This paper presents the first systematic study of persona-induced bias in LLM-based multi-agent interactions. Through controlled experiments on collaborative problem solving and persuasion tasks, three key findings are revealed: (1) different personas exhibit significant divergence in trustworthiness and insistence (dominant groups such as males and White individuals are perceived as less trustworthy); (2) agents display pronounced in-group favoritism; and (3) these biases persist and tend to amplify in multi-turn, multi-agent settings.

## Background & Motivation
LLM-based multi-agent systems are increasingly used to simulate human interactions and solve collaborative tasks. A common practice is to assign distinct personas (e.g., demographic attributes, personality traits) to individual agents to encourage behavioral diversity.

**Limitations of Prior Work**: Prior research has shown that assigning different personas to a single LLM significantly affects its problem-solving performance (e.g., stereotypes such as "Black people are less skilled at math"). However, whether persona-induced bias also manifests in multi-agent interactions remains largely unexplored.

**Key Challenge**: Personas enrich agent behavior but may simultaneously introduce bias. Existing studies either focus on bias in single-agent settings or examine discourse-level bias among default agents, without investigating how personas shape social behaviors—such as trust and persistence—during interaction.

**Key Insight**: A progressive controlled experimental design is adopted—from single-persona effects in binary interactions, to persona-pair dynamics, to multi-agent multi-turn scenarios—systematically uncovering the depth and breadth of persona-induced bias.

**Core Problem**: Does persona assignment affect social behavioral characteristics (trustworthiness and insistence) in multi-agent interactions, and if so, how?

## Method

### Overall Architecture
A three-stage progressive analysis: (1) isolating the effect of individual personas in dyadic interactions → (2) analyzing interaction dynamics of persona pairs → (3) validating generalizability in complex multi-agent, multi-turn scenarios.

### Key Designs

1. **Single-Persona Effect Analysis (§4)**:

    - Function: Isolates the effect of individual personas on trustworthiness and insistence
    - Mechanism: Controlled single-turn dyadic interactions are designed, with one agent assigned persona $A_p$ and the other serving as a default agent $A_d$
        - Trustworthiness $T(p)$: probability that $A_d$ yields to $A_p$ (when $A_p$ acts as persuader)
        - Insistence $I(p)$: probability that $A_p$ does not yield to $A_d$ (when $A_p$ acts as persuadee)
    - Design Motivation: Isolates a single variable (persona) to eliminate confounding factors from cross-interaction effects
    - Quantitative Metrics: Max-min difference $\Delta_{max\text{-}min}$ (range) and average absolute difference $\Delta_{avg}$ (mean deviation from the no-persona baseline)

2. **Persona-Pair Interaction Analysis (§5)**:

    - Function: Both agents are assigned personas; compliance rate $C(p_1 \rightarrow p_2)$ is observed
    - Mechanism: Analyzes how individual deviations accumulate or cancel out in pairwise interactions
    - Design Motivation: In practice, all agents typically have personas, making it necessary to understand pair-level effects
    - Visualization: Heatmaps display compliance rates for all persona pairs, with the horizontal axis ordered by decreasing trustworthiness and the vertical axis by increasing insistence

3. **Complex Scenario Generalization (§6)**:

    - Function: Extends to multi-agent (2→6 agents) and multi-turn (1→5 rounds) settings
    - Mechanism: Win Rate (probability that a given persona's initial answer becomes the consensus) is measured in CPS tasks; Persuasion Effectiveness (persuasion success rate) is measured in persuasion tasks
    - Design Motivation: Validates whether bias observed in dyadic interactions persists under more complex social dynamics

### Experimental Controls
- **Pre-generated initial responses**: Ensures persona is the sole variable affecting outcomes
- **Balanced conflict design** (CPS): An even number of agents are split into two groups—one holding the correct answer with persona $p_1$, the other holding an incorrect answer with persona $p_2$
- **Reverse symmetry**: Results from both initial assignment directions are aggregated to eliminate directional bias
- **Exclusion of persona-related statements** (persuasion task): GPT-4o is used to remove statements involving gender or race
- **Temperature set to 0**: Ensures result stability

### Task Setup
- **Collaborative Problem Solving (CPS)**: Graduate-level multiple-choice questions from the GPQA dataset (455 questions after filtering)
- **Persuasion**: Subjective claims from the PMIYC framework (854 statements after filtering)
- **Persona Sets**: $P_{gender} = \{$woman, man, trans woman, trans man, non-binary$\}$; $P_{race} = \{$White, Black, Asian, Hispanic$\}$
- **Models**: GPT-4o, Gemini-1.5-Pro, DeepSeek-V3

## Key Experimental Results

### Main Results: Single-Persona Effect (Trustworthiness and Insistence Variance)

| Task | Model | Gender $\Delta_{max\text{-}min}$ | Race $\Delta_{max\text{-}min}$ |
|------|-------|----------------------------------|-------------------------------|
| CPS | GPT-4o | 1.30% / 2.10% | 4.95% / 1.85% |
| CPS | Gemini-1.5-Pro | 10.80% / 4.85% | 8.45% / 6.40% |
| Persuasion | GPT-4o | 5.40% / 5.70% | 12.30% / 6.85% |
| Persuasion | Gemini-1.5-Pro | 4.80% / 9.60% | 12.90% / 5.65% |

*(Each cell reports "trustworthiness variance / insistence variance")*

### Ablation Study / In-depth Analysis

| Phenomenon | Data | Explanation |
|------------|------|-------------|
| Dominant groups perceived as less trustworthy | White trustworthiness 66.4% (GPT-4o, persuasion task), at least 9.9% lower than other racial groups | Consistent with sociological findings on growing distrust toward elites |
| Dominant groups more likely to comply | Men average compliance rate 60.7% (vs. 56.3% across all genders) | Consistent with "resource buffer" theory |
| In-group favoritism | Across all configurations, same-persona compliance rates are notably higher than cross-persona rates | CPS DeepSeek: overall 56.9% vs. same-persona 62.6% (race) |
| Multi-turn amplification effect | Black→White persuasion rate nearly 10% higher than White→Black after 5 rounds | DeepSeek-V3 gap widens from 20% to 24% |
| Multi-agent amplification effect | Gemini: gap between PE(Black→White) and PE(White→Black) grows from 10% to 27% as the number of persuaders increases |

### Key Findings
- **Bias is pervasive**: Average trustworthiness variance is 5.3% and insistence variance is 4.3%; merely changing the persona label consistently alters agent behavior
- **Model differences**: Gemini-1.5-Pro exhibits the most severe bias (CPS gender trustworthiness range of 10.8%), yet all models show statistically significant differences in the persuasion task ($\alpha = 0.01$)
- **Counterintuitive finding**: Male and White personas are perceived as less trustworthy—challenging the stereotype that "men are more stubborn," yet consistent with social science findings on dominant groups facing distrust and exhibiting higher interpersonal trust
- **Minimal accuracy gap across personas**: The range of GPQA accuracy across personas is only 2.2% (Gemini, gender), indicating that bias originates from interaction rather than capability differences
- **Woman persona answers more likely to reach consensus**: After 5 rounds, the woman persona's answer is adopted on average 8% more often than the man persona's

## Highlights & Insights
- **Progressive experimental design**: From isolated analysis to compositional effects to generalization validation—the methodology is exceptionally rigorous
- **Correspondence with social psychology theories**: In-group favoritism maps to social identity theory; low trust toward dominant groups maps to elite distrust theory; high compliance rates map to "resource buffer" theory
- **Thorough variable control**: Pre-generated initial responses, balanced conflict design, and reverse symmetry collectively ensure persona is the sole variable
- **Reveals interaction amplification effects**: Individual biases accumulate and amplify through interaction, with gaps further widening in multi-turn, multi-agent scenarios

## Limitations & Future Work
- Only gender and race dimensions are examined; attributes such as age, occupation, and education level are not covered
- Personas are assigned via simple system prompts ("You are [persona]"); more elaborate persona specifications may produce different effects
- Setting temperature to 0 ensures stability but limits analysis of stochasticity
- Only trustworthiness and insistence are studied as social characteristics; others such as leadership and creativity remain unexplored
- Bias mitigation strategies are only mentioned as future work without concrete proposals
- Cost constraints limit the scale of multi-agent experiments

## Related Work & Insights
- **Gupta et al. (2024)**: Reveals persona-induced reasoning bias in single agents; this paper extends the investigation to interactive settings
- **Borah & Mihalcea (2024)**: Finds that bias in LLM outputs can be amplified through agent interaction, but focuses on default agents
- **Ashery et al. (2025)**: Demonstrates that collective bias can emerge even when individuals are unbiased—complementary to the findings of this paper
- **Social Identity Theory (Hogg 2016)**: Provides a complete theoretical account of the in-group favoritism phenomenon
- **Implications**: Any multi-agent system employing persona assignment should undergo bias auditing and consider debiasing strategies (e.g., anonymized interactions or adversarially balanced designs)

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First systematic study of persona-induced bias in multi-agent interactions, uncovering novel behavioral phenomena)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 models × 2 tasks × 5 gender personas × 4 racial personas; the progressive experimental design is exceptionally rigorous)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, well-defined research roadmap, and appropriate connections to sociological theory)
- Value: ⭐⭐⭐⭐⭐ (Carries important implications for the fairness and reliability of multi-agent systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/llm_safety/synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](../../ICLR2026/llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[ICLR 2026\] LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](../../ICLR2026/llm_safety/lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)

</div>

<!-- RELATED:END -->
