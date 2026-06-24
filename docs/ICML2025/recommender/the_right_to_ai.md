---
title: >-
  [Paper Note] Position: The Right to AI
description: >-
  [ICML 2025][Recommender Systems][AI Governance] This position paper introduces the concept of the "Right to AI," advocating that individuals and communities affected by AI systems should have the right to participate in their development and governance. Drawing on the "right to the city" theory from urban planning, the paper constructs a four-tiered citizen participation model.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "AI Governance"
  - "Participatory Design"
  - "Right to AI"
  - "Democratic Legitimacy"
  - "Data Sovereignty"
date: 2026-05-08
content_hash: 807cea9eb67556aa
---

# Position: The Right to AI

**Conference**: ICML 2025  
**arXiv**: [2501.17899](https://arxiv.org/abs/2501.17899)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: AI Governance, Participatory Design, Right to AI, Democratic Legitimacy, Data Sovereignty

## TL;DR

This position paper introduces the concept of the "Right to AI," advocating that individuals and communities affected by AI systems should have the right to participate in their development and governance. Drawing on the "right to the city" theory from urban planning, the paper constructs a four-tiered citizen participation model.

## Background & Motivation

AI systems are increasingly penetrating critical domains such as healthcare, education, finance, and urban planning. However, decision-making power in their design and deployment is highly concentrated within a small number of corporations and government agencies. This "top-down" model leads to various issues: persistent algorithmic bias (e.g., racial discrimination in mortgage approvals), large-scale data extraction lacking transparency ("data enclosure" phenomenon), and the exclusion of the public from AI governance, rendering them passive recipients.

Existing AI governance frameworks—whether the OECD guidelines, the EU AI Act, or corporate self-regulation—mostly remain at the levels of "informing" and "consultation," lacking mechanisms to empower the public with genuine decision-making authority. Measured by Arnstein's "ladder of citizen participation," most current AI governance practices reside at the lower-to-middle rings (informing/consultation), far from the top rung of "citizen control."

The **Key Challenge** of this paper is: **AI has become a de facto social infrastructure** (akin to electricity and transportation), yet its governance model remains stuck in the "product/commodity" paradigm. Just as urban planning shifted from Le Corbusier-style elite design to Jane Jacobs-style community participation, AI governance requires a similar paradigm shift.

The **Core Idea** of the "Right to AI" is: **this is not merely a right to access AI services, but a "power right" to participate in shaping the objectives, constraints, and risk thresholds of AI**—a collective right regarding co-governance rather than individual protection.

## Method

### Overall Architecture

The paper establishes a complete chain of argumentation, from theoretical foundations and justifications to model construction and case studies:

1. **Theoretical Foundations**: Lefebvre's "right to the city" + Arnstein's "ladder of citizen participation" + Jacobs' grassroots participation
2. **Four Justifications**: Democratic legitimacy, social justice, epistemic autonomy, and data production
3. **Four-Tiered Participation Model**: From the "consumer model" to "citizen control"
4. **Nine Case Studies**: Verifying lessons learned from participatory practices

### Key Designs

1. **AI as Social Infrastructure: Three-Fold Justification**: The authors justify that AI exhibits the characteristics of social infrastructure from three dimensions: (a) **Broad societal impact**—AI is embedded in high-impact areas like medical diagnosis and educational assessment; (b) **Core role in daily life**—credit approval, job screening, and social benefit distribution increasingly rely on AI; (c) **Requirement for collective management**—like other infrastructures, AI embeds political, economic, and cultural assumptions. The **Design Motivation** is: only by redefining AI as "infrastructure" rather than a "product" can a legitimate foundation for public participation in governance be established.

2. **Four Pillars of Justification**:

    - **Democratic Legitimacy**: Decisions affecting the public should be co-created with public participation (citing Dahl, Habermas).
    - **Social Justice and Pluralism**: ML models generalize from big data and may neglect the values and cultural norms of marginalized groups, necessitating inclusive governance.
    - **Epistemic Autonomy**: AI systems filter information and recommend decisions, affecting the epistemic ecosystem; centralized control may lead to cultural homogenization.
    - **Data Production**: Data is generated within diverse social contexts and is a collective product, which should be collectively governed according to Ostrom's theory of common-pool resources.

3. **Four-Tiered Participation Model (adapted from Arnstein's ladder)**:
   
   | Tier | Model | Public Role | Characteristics |
   |------|------|---------|------|
   | Tier 1 | Consumer Model | Passive Consumer | No substantive input, participating only through surveys/feedback |
   | Tier 2 | Private Corporate Ownership | Limited Feedbacker | Corporations integrate some user feedback, but core decision-making remains with the company |
   | Tier 3 | State Regulation | Regulated/Consulted Party | The government sets standards but may overlook local knowledge |
   | Tier 4 | Citizen Control | Co-governor | Local data trusts, cooperative ownership, and oversight via citizen assemblies |

   Transition mechanisms between tiers include: structured feedback channels (1→2), statutory community audits/advisory boards (2→3), and cooperative data trusts + technical education (3→4).

4. **Analysis of Nine Case Studies**: Extracting lessons from actual participatory AI projects. For example:

    - **WeBuildAI**: Enabling donors, volunteers, and other stakeholders to collaboratively design allocation algorithms, yielding outcomes fairer than manual methods.
    - **Maori Data Sovereignty initiatives**: Protecting Maori linguistic data and ensuring community-led technology development.
    - **Machine translation for low-resource African languages**: Community-driven data collection to create new datasets and benchmarks for 30+ languages.

### Cross-Tier Comparison

| Dimension | Consumer Model | Private Corporate Ownership | State Regulation | Citizen Control |
|------|----------|---------|---------|---------|
| Subjectivity | Lowest | Limited | Moderate | Highest |
| Transparency | Low | Partial | Moderate-to-High | Highest |
| Inclusivity | None | Selective | Policy-based | Broad |
| Governance Structure | Corporate-internal | Corporation + User Board | Government Agency | Citizen Assemblies / Data Trusts |

## Key Experimental Results

### Main Results

As a position paper, this work replaces traditional experiments with case studies:

| Case Project | Domain | Key Outcomes |
|---------|------|---------|
| Anthropic Collective Constitutional AI | AI Alignment | Exposed tensions within ethical frameworks |
| PRISM Alignment Dataset | AI Ethics | Revealed cross-cultural alignment divergence |
| MID-Space | Urban Planning | Incorporated viewpoints of marginalized groups |
| WeBuildAI | Algorithm Governance | Produced fairer allocation algorithms |
| Māori Data Sovereignty | Language Tech | Achieved community-led data governance |

### Ablation Study

| Alternative Governance Model | Advantages | Disadvantages |
|-------------|------|------|
| Market-driven (Status Quo) | High efficiency, rapid innovation | Concentrated power, persistent bias, lack of accountability |
| State-driven | Enforceable standards, increased accountability | Tends to overlook local knowledge, influenced by political agendas |
| Participatory (Ours) | Balances efficiency with democratic legitimacy | Faces scalability, resource, and institutional challenges |

### Key Findings

- Most existing participatory practices remain at the level of "knowledge sharing" rather than "power sharing."
- Success factors: early and sustained stakeholder involvement, transparent objective communication, and clear acknowledgment of resource disparities.
- Risk: "participation-washing"—inviting public participation without granting genuine decision-making authority.
- In high-risk scenarios (such as healthcare or aviation), citizen control requires a hybrid format that integrates expert knowledge.

## Highlights & Insights

- **Precise conceptual transfer**: The analogy from urban planning's "right to the city" to the digital era's "Right to AI" is well-suited and inspiring.
- The four-tiered model not only describes the current and ideal states but also provides transition mechanisms between tiers, offering practical guidance.
- The warning against "participation-washing" is critical—many corporate "user feedback" mechanisms essentially exist at the lowest tier of Arnstein's ladder.
- The introduction of a Jane Jacobs-style "bottom-up" perspective challenges the pervasive techno-elitism in the AI field.

## Limitations & Future Work

- Insufficient discussion on the operability of the "citizen control" tier—establishing data trusts and citizen assemblies in the AI domain faces massive institutional and technical challenges in practice.
- Under-discussed efficiency losses and decision delays that may result from participatory approaches.
- The nine case studies span different domains and scales, making them difficult to compare systematically.
- Insufficient focus on AI governance scenarios in the Global South (although Māori and African language projects are mentioned, the specific challenges of developing nations are not discussed in depth).
- The legal characterization and enforceability of the "Right to AI" remain vague—is it a moral advocacy or a legally actionable right?

## Related Work & Insights

- Complementary to calls for AI safety governance by Bengio et al. (2024)—this paper emphasizes a "bottom-up" pathway.
- Echoes the technical work on pluralistic alignment by Sorensen et al. (2024)—this paper provides a broader macro-institutional framework.
- Insights for recommender system scenarios: user participation in recommendation algorithms is typically limited to a "feedback" button—a classic lower rung of Arnstein's ladder; more meaningful participation could involve users in defining recommendation objectives (e.g., "what I want the algorithm to optimize for").

## Rating

- Novelty: ⭐⭐⭐⭐ Creative interdisciplinary concept transfer, well-organized four-tiered model
- Experimental Thoroughness: ⭐⭐⭐ Valuable case studies but lacks systematic empirical evidence
- Writing Quality: ⭐⭐⭐⭐ Clear and complete arguments, extensively cited
- Value: ⭐⭐⭐⭐ Important contribution to AI governance discourse, though the gap between theory and practice still needs to be bridged

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ICML 2026\] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](../../ICML2026/recommender/position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)
- [\[ICML 2026\] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](../../ICML2026/recommender/position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)
- [\[ICML 2025\] Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints](position_dont_use_the_clt_in_llm_evals_with_fewer_than_a_few_hundred_datapoints.md)
- [\[ICML 2025\] Deprecating Benchmarks: Criteria and Framework](deprecating_benchmarks_criteria_and_framework.md)

</div>

<!-- RELATED:END -->
