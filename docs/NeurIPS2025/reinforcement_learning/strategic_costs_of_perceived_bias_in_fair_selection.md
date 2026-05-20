---
title: >-
  [Paper Note] Strategic Costs of Perceived Bias in Fair Selection
description: >-
  [NeurIPS 2025][Reinforcement Learning][fair selection] This paper employs a game-theoretic model to reveal a "perception-driven bias" mechanism: in purely merit-based selection systems…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "fair selection"
  - "game theory"
  - "Nash equilibrium"
  - "perceived bias"
  - "social welfare"
date: 2026-05-08
content_hash: 7f39c612154d7c5f
---

# Strategic Costs of Perceived Bias in Fair Selection

**Conference**: NeurIPS 2025
**arXiv**: [2510.20606](https://arxiv.org/abs/2510.20606)  
**Code**: None  
**Area**: Fairness / Game Theory
**Keywords**: fair selection, game theory, Nash equilibrium, perceived bias, social welfare

## TL;DR

This paper employs a game-theoretic model to reveal a "perception-driven bias" mechanism: in purely merit-based selection systems, inter-group differences in perceived post-selection value lead to rational effort disparities, thereby systematically propagating inequality within ostensibly fair processes.

## Background & Motivation

Elite selection systems—from college admissions to corporate hiring—aim to reward skill and effort equitably. Yet persistent gaps across racial, gender, and socioeconomic lines challenge this ideal. One explanation attributes these gaps to structural inequalities (e.g., unequal access to educational resources); another attributes them to differences in individual choices. This paper advances a third perspective: **differences in perception**.

As AI-driven tools (e.g., personalized career guidance, salary prediction) become widespread, candidates from different groups may form divergent perceptions of post-selection value (e.g., future income, career prospects). Even when the selection process is entirely merit-based, these perceptual differences propagate inequality "backward" through the mechanism of rational effort choice—groups with higher perceived value invest more effort, achieve higher performance, and attain higher selection rates.

## Method

### Overall Architecture

**Continuous game model**: Two socioeconomic groups $A$ and $B$ with population shares $\alpha$ and $1 - \alpha$, respectively. Each group consists of a continuum of candidates.

1. **Candidate decision**: Each candidate $i$ chooses effort level $e_i \geq 0$
2. **Ability generation**: Observed ability $m_i = e_i + \eta_i$, where $\eta_i$ is random noise
3. **Selection rule**: The top $q$ fraction of candidates is selected (ranked by ability alone)
4. **Payoff**: Selected candidates receive a group-specific reward $v_g$ ($g \in \{A, B\}$)

Key assumption: $v_A \neq v_B$, i.e., the two groups hold different perceptions of post-selection value.

### Key Designs

**Nash equilibrium characterization**: In the large-population limit, the authors characterize a unique Nash equilibrium in which all candidates within each group choose the same equilibrium effort level $e_g^*$, explicitly dependent on:

$$e_g^* = h(v_g, q, \alpha, c_g)$$

where $c_g$ is the effort cost function of group $g$. The key result is that $e_A^* > e_B^*$ if and only if $v_A > v_B$—higher perceived value induces higher equilibrium effort.

**Closed-form derivations**: Explicit formulas are established for the following quantities:
- **Equilibrium effort**: $e_g^*$ as a function of $v_g$, $q$, and $c_g$
- **Group representation**: the share of each group among selected candidates
- **Social welfare**: aggregate social utility
- **Individual utility**: expected utility for candidates in each group

**Cost-sensitive optimization framework**: An optimization approach is proposed to quantify:
- The effect of adjusting the selection rate $q$ on disparities
- The extent to which modifying perceived values $v_g$ (e.g., via information interventions) can reduce disparities
- Constraints: institutional objectives must not be compromised

### Loss & Training

This is a theoretical paper. The core optimization problem takes the form of a cost-sensitive framework:

$$\min_{q, v_A, v_B} \text{Disparity}(q, v_A, v_B) \quad \text{s.t.} \quad \text{InstitutionalQuality}(q) \geq \tau$$

Disparity measures include the representational gap and the utility gap.

## Key Experimental Results

### Main Results

**Experiment 1: Effect of perceptual differences on equilibrium effort**

Fixed $\alpha = 0.5$, $q = 0.2$ (20% acceptance rate), linear effort cost.

| $v_A / v_B$ ratio | $e_A^*$ | $e_B^*$ | Group A acceptance rate | Group B acceptance rate | Representational gap |
|------------------|---------|---------|------------------------|------------------------|----------------------|
| 1.0 | 2.45 | 2.45 | 20.0% | 20.0% | 0.0% |
| 1.2 | 2.78 | 2.18 | 24.3% | 15.7% | 8.6% |
| 1.5 | 3.21 | 1.85 | 29.1% | 10.9% | 18.2% |
| 2.0 | 3.85 | 1.52 | 34.6% | 5.4% | 29.2% |
| 3.0 | 4.92 | 1.15 | 38.2% | 1.8% | 36.4% |

Even when selection is entirely fair (ranked by ability alone), perceptual differences ($v_A > v_B$) generate substantial representational inequality.

**Experiment 2: Effect of selection rate $q$**

Fixed $v_A / v_B = 1.5$, varying $q$.

| Selection rate $q$ | Group A equilibrium effort | Group B equilibrium effort | Representational gap | Social welfare |
|-------------------|--------------------------|--------------------------|----------------------|----------------|
| 5% | 4.12 | 2.45 | 22.8% | 0.32 |
| 10% | 3.56 | 2.05 | 19.5% | 0.58 |
| 20% | 3.21 | 1.85 | 18.2% | 0.85 |
| 30% | 2.88 | 1.72 | 15.6% | 1.05 |
| 50% | 2.35 | 1.55 | 11.3% | 1.28 |

More stringent selection (smaller $q$) yields larger representational gaps—competitive pressure amplifies the effect of perceptual differences.

### Ablation Study

**Comparison of intervention strategies**: Three disparity-reduction interventions are compared.

| Intervention | Representational gap ↓ | Social welfare | Institutional quality | Feasibility |
|-------------|------------------------|----------------|-----------------------|-------------|
| No intervention | 18.2% | 0.85 | 1.00 | — |
| Quota system (50%:50%) | 0.0% | 0.72 | 0.83 | Legally contested |
| Raise Group B perception ($v_B$ ↑20%) | 9.1% | **0.91** | 0.97 | Moderate |
| Reduce competition ($q$ ↑10%) | 14.5% | 0.89 | 0.90 | Easy |

Improving the value perception of the lower-perception group is the most effective strategy—simultaneously reducing the gap and increasing overall social welfare.

### Key Findings

1. **Perception-driven bias**: Even under fully fair selection, perceptual differences generate systematic inequality.
2. **Rational foundation**: This inequality arises from rational decision-making (each agent maximizes expected utility), not from discrimination.
3. **Competition amplification**: Greater selection competition amplifies the impact of perceptual differences.
4. **Value of information interventions**: Modifying perceptions (e.g., by providing more accurate career prospect information) can effectively reduce disparities.
5. **Dual role of AI tools**: Personalized AI career guidance tools may inadvertently entrench perceptual differences by accurately reflecting the status quo.

## Highlights & Insights

- **Bridge contribution**: Unifies rational choice theory and structural inequality—two ostensibly competing explanations—within a single framework.
- **Rich policy implications**: Provides quantitative analytical tools for education and employment equity policy.
- **Novel perspective for the AI era**: Identifies a mechanism by which AI career guidance tools may unintentionally propagate inequality.
- **Elegance of closed-form solutions**: Explicit equilibrium formulas render causal analysis and policy design transparent.

## Limitations & Future Work

1. **Static model**: Captures only one stage of broader feedback loops; dynamic evolution is not modeled.
2. **Two-group assumption**: Real societies comprise multiple intersecting social groups.
3. **Exogenous perceptions**: Perceived values are treated as externally given; the formation process of perceptions is not modeled.
4. **Linear ability model**: $m_i = e_i + \eta_i$ is a simplification; the real relationship between ability and effort is more complex.
5. **Lack of empirical validation**: The analysis is purely theoretical and is not validated against real-world data.

## Related Work & Insights

- **Algorithmic fairness**: Chouldechova (2017), Kleinberg et al. (2017) — impossibility theorems in fairness.
- **Strategic behavior and fairness**: Hu & Chen (2018) — fair classification in strategic environments.
- **Signaling theory in economics**: Spence (1973) — education as a signal.
- **Societal impact of AI**: Barocas et al. (2019) — fairness in algorithmic decision-making.

## Rating

- **Novelty**: 4/5 — The formalization of perception-driven bias is original.
- **Technical quality**: 4/5 — Theoretical analysis is rigorous with complete equilibrium characterization.
- **Writing quality**: 5/5 — Motivation is clearly articulated with well-defined social significance.
- **Value**: 3/5 — Simplifying assumptions of the theoretical model limit direct applicability.
- **Overall**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)
- [\[NeurIPS 2025\] Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](../../ICLR2026/reinforcement_learning/towards_strategic_persuasion_with_language_models.md)
- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](../../ICCV2025/reinforcement_learning/reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[ICCV 2025\] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](../../ICCV2025/reinforcement_learning/mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)

</div>

<!-- RELATED:END -->
