---
title: >-
  [Paper Note] Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work
description: >-
  [ICML2025][democratic AI] This paper proposes the "Democracy Levels" framework, which categorizes the transfer of AI decision-making authority from unilateral power to democratic systems into six levels (L0–L5). Equipped with a multidimensional evaluation system and practical tools, it provides a systematic roadmap for the democratization of AI governance.
tags:
  - "ICML2025"
  - "democratic AI"
  - "participatory AI"
  - "pluralistic alignment"
  - "collective decision-making"
  - "AI governance"
date: 2026-05-08
content_hash: 3ffbf01c64806251
---

# Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work

**Conference**: ICML2025  
**arXiv**: [2411.09222](https://arxiv.org/abs/2411.09222)  
**Code**: None (position paper / framework proposal)  
**Area**: AI Governance / Democratic AI  
**Keywords**: democratic AI, participatory AI, pluralistic alignment, collective decision-making, AI governance

## TL;DR

This paper proposes the "Democracy Levels" framework, which categorizes the transfer of AI decision-making authority from unilateral power to democratic systems into six levels (L0–L5). Equipped with a multidimensional evaluation system and practical tools, it provides a systematic roadmap for the democratization of AI governance.

## Background & Motivation

AI systems are exerting a profound influence on the lives of billions—ranging from financial risk assessment to recommendation systems, and from autonomous interactions of AI agents to large-scale infrastructure decisions. This raises several core challenges:

**Risk of Power Concentration**: AI development is primarily driven by a handful of corporations and governments, potentially leading to an unprecedented concentration of power.

**Governance Vacuum**: The generality of AI, its rapid pace of change, market incentives, and cross-jurisdictional regulatory arbitrage make it difficult for traditional regulation to keep pace.

**Ambiguous Concept of Democratization**: Currently, "AI democratization" is mostly understood as "making AI open and accessible" rather than actual democratic governance.

**Unsystematic Early Experiments**: Although experiments like Anthropic's Collective Constitutional AI, OpenAI's Democratic Inputs, and Meta's Community Forums are inspiring, they lack a systematic evaluation framework.

The core thesis of the paper: **Effective AI democratization requires democratic governance and alignment of AI**, especially for decisions with systemic societal impacts.

## Method

### Core Framework: Democracy Levels (L0–L5)

The framework categorizes the **degree of decision-making authority transfer from unilateral authority to democratic systems** into six levels:

| Level | Name | Role of Democratic System | Example (e.g., AI persuasion rules) |
|------|------|-------------|------------------------|
| L0 | Unilateral Decision-making | None | Company establishes AI persuasion rules on its own |
| L1 | Information Input | Provides reference information for decisions | Public forums gather opinions, which the company interprets freely |
| L2 | Decision Prescription | Generates directly implementable decisions (can be vetoed) | Citizens' assembly formulates rules, while the company retains veto power |
| L3 | Binding Decision-making | Makes binding decisions | Rules established through a democratic process are binding on the company |
| L4 | Automated Triggering | Automated triggering of binding decision-making processes under specific conditions | Democratic deliberation is automatically initiated when AI capabilities exceed a threshold |
| L5 | Meta-governance | Governs the democratic system itself | The democratic system decides how to run L4 processes |

This taxonomy draws inspiration from the levels of driving automation (SAE Levels), with the core idea being the progressive transfer of power and responsibility from a unilateral authority to a new decision-making system.

### Evaluation Dimension Framework

The framework defines three main categories and 13 dimensions to evaluate the quality of a democratic system:

**Process Quality**:

$$Q_{process} = f(\text{代表性}, \text{知情度}, \text{审议性}, \text{实质性}, \text{鲁棒性}, \text{可读性})$$

- **Representation**: Whether key decisions represent the relevant population.
- **Informedness**: Whether decisions incorporate critical information from domain experts, authoritative organizations, and diverse stakeholders.
- **Deliberation**: Whether decisions are based on thoughtful deliberation rather than superficial reactions.
- **Substantiveness**: Whether decisions are actionable and have actual impact.
- **Robustness**: Whether the process can withstand adversarial behaviors and non-ideal conditions.
- **Legibility**: Whether the processes and decisions are accessible, comprehensible, and verifiable.

**Delegation**:

- **Integration**: Whether the authority integrates democratic processes into its operations.
- **Ability to bind**: Whether the authority can technically and legally bind itself to adhere to democratic decisions.
- **Commitment**: The extent to which the authority commits to respecting democratic decisions.

**Trust**:

- **Awareness**: Whether the public is aware of the existence and operation of the democratic process.
- **Participation**: Whether the public is willing and able to participate.
- **Accountability**: Whether there are external oversight and accountability structures.
- **Buy-in**: Whether the public and key stakeholders recognize the legitimacy of the process.

### Practical Tools

**1. Levels Decision Tool**: Helps unilateral authorities (and advocates) assess whether and to what extent to delegate decision-making power to a democratic system. Considerations include: value of legitimacy, potential for collective intelligence, feasibility of power transfer, speed and adaptability requirements, and resource constraints.

**2. Democratic System Card**: Similar to Model Cards and AI System Cards, used for structural documentation, evaluation, and comparison of democratic systems. It consists of three components: a description of how the system works, evaluating dimensions, and a qualitative assessment of the highest decision-making level at which the system can be trusted.

### Case Studies

**Anthropic Collective Constitutional AI**: A representative panel of the US public provides and evaluates AI principles → processed through deduplication and transformation for training → falls under the L0 → L1 transition (information input only). If preset acceptance/rejection criteria existed, it could reach L2; with binding commitments, it could reach L3.

**Meta Oversight Board**: Content moderation decisions → L4 (regular binding decisions); policy advisory opinions → L1 (non-binding). However, it is flawed in the representation dimension—the board was not designed with democratic representation in mind.

## Key Experimental Results

This is a position paper without traditional quantitative experiments. Its core contributions lie in the framework design and case studies:

| Aspect | Evaluation |
|------|------|
| Number of Framework Levels | 6 levels (L0–L5) |
| Number of Evaluation Dimensions | 13 dimensions across 3 major categories |
| Current Cases Reviewed | Anthropic CCAI, Meta Community Forums, Meta Oversight Board, OpenAI Democratic Inputs, Google DeepMind STELA |
| Companion Tools | 2 (Levels Decision Tool + Democratic System Card) |
| Core Argumentative Logic | Democracy → Legitimacy + Distribution of Power + Epistemic Advantages → Better AI Governance |

The paper systematically addresses five common objections (libertarian critique, sufficiency of government regulation, shareholder primacy, decelerationism, and technological immaturity).

## Highlights & Insights

1. **Ingenious Level Analogy**: Drawing inspiration from the autonomous driving levels (SAE Levels) to classify the degree of AI democratization makes abstract concepts concrete and easy to communicate.
2. **Principle-driven yet Practical**: The framework not only offers theoretical levels but also provides actionable evaluation tools (Decision Tool and System Card), lowering the barrier to practice.
3. **Pragmatic yet Ambitious**: Acknowledging that democratization is an incremental process ("building democratic muscle"), it does not demand a one-step jump but clearly outlines the ultimate goal (L5 Meta-governance).
4. **Business-friendly Argumentation**: Arguing the benefits of democratization for enterprises from the perspectives of reducing compliance costs, avoiding antitrust scrutiny, and maintaining market value increases its practical feasibility.
5. **Dual-empowerment Perspective**: It discusses both "democracy for AI" (using democracy to govern AI) and "AI for democracy" (using AI to improve democratic processes), establishing a positive feedback loop.

## Limitations & Future Work

1. **Lack of Quantitative Validation**: As a position paper, the efficacy of the framework has not been empirically verified, and the dimension scoring lacks quantitative standards.
2. **Underspecified Scaling Challenges**: Global AI decisions involve participants across cultures, languages, and legal systems. How to achieve meaningful representation and deliberation remains underexplored.
3. **Tension between Speed and Democracy**: AI technology evolves rapidly, whereas high-quality deliberative democratic processes are inherently time-consuming; the framework's response to this tension ("democratic innovation") is somewhat generic.
4. **Asymmetrical Power Dynamics**: The framework assumes that unilateral authorities possess the willingness to devolve power. However, in reality, tech companies lack such incentives, and deep analyses of power struggles are missing.
5. **Lack of Technical Detail**: Specific solutions are missing regarding how to technically bind the outputs of democratic processes (e.g., model specifications) to model training and deployment.
6. **Insufficient Discussion on Applicability**: The applicability to developing countries and authoritarian regimes is under-discussed.

## Related Work & Insights

- **Arnstein's (1969) "Ladder of Citizen Participation"**: A classic framework of participation levels, which directly inspired the leveling logic in this paper.
- **Anthropic CCAI (2023)**: One of the most influential experiments in democratic AI alignment.
- **SAE Levels of Driving Automation**: Direct inspiration for the framework's level design.
- **Model Cards (Mitchell et al., 2019)**: Inspiring the design of the Democratic System Card.
- Complements research directions such as **Pluralistic AI** (Sorensen et al.) and **Participatory AI** (Birhane et al.).

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically combines democratic theory with AI governance; the leveling framework design is highly original.
- Experimental Thoroughness: ⭐⭐ — Position paper, lacks quantitative experiments, with limited case analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rigorous arguments, and adequacy in addressing objections.
- Value: ⭐⭐⭐⭐ — Provides much-needed conceptual tools and evaluation frameworks for the democratization of AI governance, offering high practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](../../NeurIPS2025/others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[ICML 2025\] Position: AI Evaluation Should Learn from How We Test Humans](position_ai_evaluation_should_learn_from_how_we_test_humans.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](../../CVPR2025/others/which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[ICML 2025\] If Open Source Is to Win, It Must Go Public](if_open_source_is_to_win_it_must_go_public.md)

</div>

<!-- RELATED:END -->
