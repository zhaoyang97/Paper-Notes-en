---
title: >-
  [Paper Note] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution
description: >-
  [AAAI 2026][LLM personality simulation] This paper proposes the first systematic comparative framework that directly contrasts strategic behavioral differences between humans and personality-prompted LLMs in paired dispute mediation scenarios, finding significant divergence in personality-behavior mapping and challenging the assumption that personality prompting can serve as a proxy for human behavior.
tags:
  - AAAI 2026
  - LLM personality simulation
  - dispute resolution
  - Big Five personality
  - behavioral alignment
  - social simulation
date: 2026-05-08
content_hash: 958450ba3cd727fb
---

# Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution

**Conference**: AAAI 2026
**arXiv**: [2602.07414](https://arxiv.org/abs/2602.07414)
**Code**: [Available](https://github.com/DSincerity/Personality-LLM-BehavAlign-Dispute)
**Area**: Interpretability
**Keywords**: LLM personality simulation, dispute resolution, Big Five personality, behavioral alignment, social simulation

## TL;DR

This paper proposes the first systematic comparative framework that directly contrasts strategic behavioral differences between humans and personality-prompted LLMs in paired dispute mediation scenarios, finding significant divergence in personality-behavior mapping and challenging the assumption that personality prompting can serve as a proxy for human behavior.

## Background & Motivation

LLMs are increasingly deployed to simulate high-stakes social scenarios (legal mediation, negotiation, dispute resolution), yet a fundamental question remains unvalidated: when prompted with Big Five Inventory (BFI) personality traits, can LLMs reproduce the personality-driven behavioral differences observed in humans?

Human personality dimensions (e.g., agreeableness, neuroticism, extraversion) have been extensively shown to systematically influence conflict resolution strategies (e.g., cooperation, competition, avoidance). However, two critical gaps persist in the literature: (1) most personality-conflict studies rely on static questionnaires rather than dynamic conversational behavior observation; and (2) prior work examines either humans or LLMs in isolation, lacking direct human-AI behavioral comparisons in matched scenarios. This paper addresses these gaps by constructing parallel human-LLM dialogue datasets to systematically examine the degree of behavioral alignment.

## Method

### Overall Architecture

This paper proposes an **evaluation framework** (not a new model), consisting of three core components:

1. **Human baseline dataset KODIS**: Crowdsourced role-playing dispute resolution dialogues from the Prolific platform, comprising 248 human-human conversations in which participants act as "buyer" and "seller" negotiating over three issues: refunds, review deletion, and formal apologies.
2. **LLM-to-LLM (L2L) simulation dataset construction**: LLM dialogues driven by matched scenarios and personality traits.
3. **Interpretable behavioral metric system**: Quantitative indicators of strategic behavior and conflict outcomes grounded in the IRP (Interests-Rights-Power) framework.

### Key Designs

**LLM personality configuration**: Each LLM is assigned a BFI personality vector $\{P_{AGR}, P_{EXT}, P_{CON}, P_{NEU}, P_{OPE}\}$ using a six-level polarity-intensity scale. To ensure fair comparison, personality distributions are sampled from the empirical BFI distribution of human participants. Personality prompts are generated using 70 bipolar adjective pairs (Goldberg, 1992), with three adjectives per trait and intensity expressed via "very / a bit / no modifier."

**Issue importance personalization**: Based on regression results from human data ($B=2.13$, $p=.02$), the importance of the apology issue is tied to agreeableness. Importance values for other issues are assigned randomly.

**Behavioral metric system** (based on the IRP framework):

- **Outcome metrics**: Score, Accept, Not Walk-Away
- **Strategic behavior metrics**:
  - IRP ratio: frequency of cooperative/competitive strategy use $\text{IRP}_{\text{ratio}}^{X} = \frac{N_S^X}{N_S^{\text{all}}}$
  - IRP reciprocity: proportion of times a speaker follows opponent's strategy $X$ with the same strategy $\text{IRP}_{\text{recip}}^{X} = \frac{N_S^{X=X_P}}{N_P^X}$
  - Escalation ratio: frequency of competitive responses to non-competitive utterances
  - De-escalation ratio: frequency of non-competitive responses to competitive utterances

**Experimental models**: GPT-4o mini (500 simulations), Claude Sonnet 3.7 (250), Gemini 2.0 Flash (250), all using default parameters (temperature=1).

### Loss & Training

This paper presents an evaluation framework and involves no model training. Analysis methods consist of linear regression for continuous dependent variables and logistic regression for binary outcomes. Independent variables include both the focal LLM's and the opponent's BFI traits, with position (buyer/seller) as a control variable using effects coding (Buyer=–1, Seller=1).

## Key Experimental Results

### Main Results (Effect of Personality on Outcomes and Strategic Behavior)

**Table 2: Regression analysis of personality traits on negotiation outcomes**

| Dependent Variable | GPT-4 Significant IVs | Claude Significant IVs | Gemini Significant IVs | KODIS (Human) |
|---|---|---|---|---|
| Score | S-EXT ($B=1.67^{**}$), S-AGR ($B=-4.38^{***}$) | S-AGR ($B=-2.50^{***}$), P-EXT ($B=-1.42^{*}$) | S-AGR ($B=-4.48^{***}$), S-CON ($B=1.72^{*}$) | Position effect only |
| Accept | POS ($B=-0.22^{*}$) | S-EXT ($B=-0.17^{**}$), P-EXT ($B=0.17^{**}$) | — | S-NEU ($B=-0.26^{*}$), P-NEU ($B=0.27^{*}$) |
| Not Walk-Away | S-OPE ($B=-0.18^{*}$), P-OPE ($B=0.18^{*}$) | — | S-NEU ($B=-0.18^{*}$), P-NEU ($B=0.18^{*}$) | No significant personality effects |

**Table 3: IRP strategic behavior regression analysis (selected)**

| Metric | GPT-4 | Gemini | Claude | KODIS |
|---|---|---|---|---|
| Cooperation ratio | P-EXT ($B=0.49^{*}$) | None | None | None |
| Competition ratio | S-EXT, S-NEU, P-EXT, P-NEU (multiple significant) | S-EXT, P-EXT significant | POS only | S-EXT, P-EXT, P-CON, S-OPE |
| Escalation ratio | S-AGR ($B=-1.37^{**}$), P-EXT ($B=1.56^{**}$) | None (aligned with KODIS) | S-AGR ($B=-2.45^{***}$) | None |
| De-escalation ratio | S-AGR ($B=1.83^{***}$) | S-EXT, P-EXT significant | None (aligned with KODIS) | None |

### Ablation Study

**IRP strategy distribution heatmap**: Humans rely most heavily on "fact statements" with the most balanced distribution; LLMs prefer "proposals" and "concessions," exhibiting a formulaic transactional style. Claude most closely resembles humans (more facts), Gemini shows the most skewed distribution (high "power" + "residual"), and GPT-4 is the most balanced but consistently favors "rights."

**Temporal dynamics analysis**: Human strategies evolve dynamically across dialogue phases (early: facts → mid: interests and proposals → late: concessions), whereas LLMs exhibit flat trajectories with minimal change in strategy composition over time. Claude partially replicates human dynamic patterns.

### Key Findings

1. **Fundamental divergence in personality-behavior mapping**: Neuroticism is the strongest predictor of strategic outcomes in humans, whereas extraversion and agreeableness yield stronger effects in LLMs.
2. **Cross-LLM variation**: Claude and Gemini align more closely with human behavior on strategic metrics than GPT-4o mini.
3. **Polarized LLM reciprocity patterns**: LLMs more consistently reciprocate cooperative strategies, while humans respond more flexibly.
4. **GPT-4o mini is more prone to escalation**; Claude exhibits a strong preference for de-escalation; humans maintain a more balanced escalation-de-escalation dynamic.

## Highlights & Insights

- The first study to directly compare personality-behavior relationships between humans and multiple LLMs in paired conflict scenarios.
- Proposes a generalizable evaluation framework and dataset construction methodology.
- Reveals an important negative finding: personality-prompted LLMs cannot serve as reliable proxies for human behavior in social influence scenarios.
- Novel application of the IRP framework to LLM evaluation.

## Limitations & Future Work

- IRP annotations are performed by LLMs without comprehensive human validation.
- Robustness to prompt phrasing and instruction variants is not evaluated.
- Only the BFI five-factor model is used; other dimensions such as emotional intelligence and Machiavellianism may offer additional explanatory power.
- The study involves a single scenario (jersey dispute), leaving generalizability unverified.
- Open-source LLMs and more recent versions of closed-source models are not included.

## Related Work & Insights

- **Personality-conflict mapping**: Wood & Bell (2008) demonstrate that extraversion and agreeableness predict cooperative conflict styles; this paper extends such findings to dynamic conversational behavior.
- **LLM social behavior simulation**: Multi-agent social simulations from Park et al. (2022) and Zhou et al. (2023).
- **Inspiration**: The proposed framework could be extended to other social-psychological constructs (e.g., trust, power dynamics) or used as a psychological alignment objective for LLM fine-tuning.

## Rating

- **Novelty**: ★★★★☆ — First systematic comparison of personality-conflict behavioral alignment between humans and LLMs.
- **Technical Depth**: ★★★☆☆ — Rigorous framework design, but no novel model or algorithmic contribution.
- **Experimental Thoroughness**: ★★★★☆ — Three LLMs plus a human baseline, with multi-dimensional regression analysis.
- **Practical Value**: ★★★★☆ — Carries important cautionary implications for LLM social simulation applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[AAAI 2026\] CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](crosscheck-bench_diagnosing_compositional_failures_in_multim.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[AAAI 2026\] SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)

</div>

<!-- RELATED:END -->
