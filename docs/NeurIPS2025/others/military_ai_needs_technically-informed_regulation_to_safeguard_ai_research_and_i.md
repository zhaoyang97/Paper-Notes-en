---
title: >-
  [Paper Note] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications
description: >-
  [NeurIPS 2025][Autonomous Weapons Systems] This paper proposes a behavior-oriented definition and regulatory framework for AI-powered Lethal Autonomous Weapons Systems (AI-LAWS). It identifies systems requiring enhanced regulation through two technical criteria, puts forward five concrete policy recommendations, and calls on AI researchers to participate actively throughout the full lifecycle of military AI governance.
tags:
  - NeurIPS 2025
  - Autonomous Weapons Systems
  - AI-LAWS
  - Military AI Regulation
  - Behavior-Oriented Definition
  - Technically-Informed Policy
  - AI Research Freedom
date: 2026-05-08
content_hash: 726385339df14130
---

# Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications

**Conference**: NeurIPS 2025
**arXiv**: [2505.18371](https://arxiv.org/abs/2505.18371)
**Code**: None
**Area**: AI Policy / AI Safety / Military AI Governance
**Keywords**: Autonomous Weapons Systems, AI-LAWS, Military AI Regulation, Behavior-Oriented Definition, Technically-Informed Policy, AI Research Freedom

## TL;DR

This paper proposes a behavior-oriented definition and regulatory framework for AI-powered Lethal Autonomous Weapons Systems (AI-LAWS). It identifies systems requiring enhanced regulation through two technical criteria, puts forward five concrete policy recommendations, and calls on AI researchers to participate actively throughout the full lifecycle of military AI governance.

## Background & Motivation

- **Background**: AI-enhanced military weapons systems—including drones, unmanned vessels, and battlefield coordination platforms—have undergone rapid global development and operational deployment. Representative AI-LAWS such as Russia's Lancet loitering munition, Israel's Lavender targeting system, and the U.S. Project Maven intelligence analysis platform are already in active use, spanning four domains: air, land, sea, and command-and-control. Governance frameworks, however, lag severely behind technological development.

- **Limitations of Prior Work**: (1) Existing LAWS definitions are either overly broad or rely on extreme thresholds such as "full autonomy" (e.g., the UN definition requires a system to "identify, select, and attack targets without further operator intervention"), rendering them unable to cover already-deployed semi-autonomous systems. (2) Policymakers lack AI technical backgrounds, and current frameworks (e.g., U.S. DoD Directive 3000.09) are grounded in high-level principles rather than actual system behavior. (3) The three dominant regulatory narratives—the humanitarian "Meaningful Human Control" (MHC), the defense-oriented "Appropriate Human Judgment" (AHJ), and AGI existential risk—all lack operationalizable technical metrics.

- **Key Challenge**: The civil–military technology gap in modern AI is minimal; civilian AI research can be rapidly repurposed by the military, often without the researchers' knowledge. This creates a triple-layered risk: insufficient validation and brittleness in the military domain, geopolitical conflict escalation and arms race dynamics, and institutional erosion of scientific freedom. Meanwhile, the group most capable of assessing AI system behavior—AI researchers—is conspicuously absent from military governance discussions.

- **Goal**: To provide an operationalizable regulatory framework for AI-LAWS grounded in system behavior rather than labels, intent, or extreme assumptions; to offer concrete policy recommendations; and to argue why and how AI researchers should engage in military AI policymaking.

- **Key Insight**: The paper begins from the technical properties of AI systems—opacity, brittleness, out-of-distribution generalization degradation, and post-deployment drift—and systematically maps these concepts onto military risks and policy requirements.

- **Core Idea**: Regulation of AI-LAWS must be anchored to the technical behavior of systems rather than to policy labels, and AI researchers must become central participants in the regulatory lifecycle.

## Method

### Overall Architecture

This is a position paper structured in three stages: status analysis → behavior-oriented definition → policy recommendations. The paper first demonstrates the inadequacy of existing frameworks by systematically surveying globally deployed and in-development AI-LAWS (Table 2) and the unique risks they introduce (Table 1). It then proposes two behavior-based regulatory criteria for identifying AI-LAWS requiring enhanced oversight, and finally derives five technically-informed policy recommendations from this definition. Although not a conventional technical paper, its analysis is deeply grounded in core AI concepts.

### Key Designs

1. **Dual-Criterion Behavior-Oriented Definition of AI-LAWS**

   - **Function**: To provide an operationalizable system classification standard for the contested military AI terminology landscape, distinguishing AI-LAWS requiring enhanced regulation from conventional automated weapons.
   - **Mechanism**: Systems satisfying both criteria are identified as AI-LAWS. **Criterion 1** (AI Technology Requirement) requires that the system employs AI/ML methods integral to its function (e.g., neural networks) and exhibits AI-specific risks such as target misidentification, unpredictable escalation, post-deployment drift, and inadequate out-of-distribution generalization. **Criterion 2** (Lethal Force Involvement) requires that at least one AI/ML-dependent capability participates in semi-autonomous or fully autonomous strike and force-application decisions.
   - **Design Motivation**: Existing definitions are either too broad (the UN definition would encompass naval mines and heat-seeking missiles) or set thresholds too high (conditions such as "superhuman learning" allow most deployed systems to escape regulation). A behavior-oriented definition avoids label-based disputes and focuses directly on the actual risks posed by system behavior.

2. **Three-Dimensional Risk Analysis Framework**

   - **Function**: To systematically argue why AI-LAWS require a distinct regulatory regime separate from conventional LAWS.
   - **Mechanism**: Risks are decomposed into **military risks** (brittleness—sharp performance degradation outside the training distribution; opacity—black-box decisions resistant to audit; overtrust—operators defaulting to AI recommendations under pressure), **geopolitical risks** (unpredictable behavior triggering conflict escalation; arms race dynamics; policymakers over-investing due to uncertainty about unknown capabilities), and **institutional risks** (military funding penetrating academic research; publication restrictions and impeded international collaboration; researchers unknowingly redirected toward military applications).
   - **Design Motivation**: Most military AI discourse focuses on battlefield performance or ethical principles in isolation. The three-dimensional analysis reveals that these risks are deeply intertwined—for example, unpredictability in military deployment drives policy overreaction that in turn erodes scientific freedom—making piecemeal regulatory solutions inherently inadequate.

3. **Five Concrete Policy Recommendations**

   - **Function**: To translate the behavior-oriented definition into actionable policy directions spanning the full spectrum from nuclear scenarios to everyday institutional boundaries.
   - **Mechanism**: **(1) Prohibit AI control over nuclear weapons deployment**, including launch decisions and nuclear strategy advisory systems. **(2) Develop international verification standards for AI-LAWS**, establishing a voluntary international coalition to coordinate behavioral benchmarks and contextual transfer performance thresholds, following an iterative evolution model analogous to Internet protocols. **(3) Prohibit "AI generals"**, banning AI systems from autonomously commanding human soldiers (the "Minotaur warfare" model), with command authority reserved for humans. **(4) Clarify the legal status of civilian AI infrastructure**, defining under the Geneva Convention framework when AI models, data, and compute become legitimate military targets (illustrated by the 2025 strike on the Weizmann Institute). **(5) Establish institutional civil–military boundaries**, with universities and companies publicly declaring policies delineating civil and military research, protecting students from being involuntarily directed into classified work.
   - **Design Motivation**: Each recommendation addresses risks specific to AI-LAWS rather than generalized AI ethics principles, and each has nascent precedents in existing policy dialogue but lacks technical grounding and detail. Together, the five recommendations constitute a multi-layered regulatory architecture spanning strategic to institutional levels.

### Loss & Training

Not applicable (this is a policy position paper, not a technical experimental paper). The paper's argumentative logic is analogous to a "training strategy"—policy arguments are constructed using established technical concepts from the AI safety literature (brittleness, distribution shift, adversarial robustness, interpretability, overtrust), with publicly deployed system cases and documented failure modes serving as "experimental data."

## Key Experimental Results

### Main Results

This paper contains no quantitative experiments; instead, two core reference tables serve as the basis for analysis.

**Table 2: Overview of Globally Deployed/In-Development AI-LAWS**

| Domain | System | Developer | Country | Purpose | Status |
|--------|--------|-----------|---------|---------|--------|
| Command & Control | Defense Llama | Scale AI | USA | Command/targeting/report synthesis | Demo |
| Command & Control | Lattice | Anduril | USA | Battlefield coordination | Deployed |
| Command & Control | Project Maven | NGA/DoD | USA | Intelligence analysis | Deployed |
| Command & Control | Lavender | IDF | Israel | Targeting | Deployed |
| Command & Control | ChatBIT | PLA | China | Command decision-making | Demo |
| Air | Lancet | ZALA Aero | Russia | Loitering munition (autonomous targeting) | Deployed |
| Air | Saker Scout | Saker | Ukraine | Quadrotor drone | Deployed |
| Air | Kargu | STM | Turkey | Loitering munition | Deployed |
| Air | XQ-58A Valkyrie | Kratos | USA | Stealth autonomous wingman | In development |
| Land | THeMIS | Milrem | Estonia | Ground unmanned vehicle | Deployed |
| Land | Uran-9 | Kalashnikov | Russia | Artillery unmanned vehicle | Deployed |
| Sea | Orca XLUUV | Boeing | USA | Long-range submersible | Demo |
| Sea | Ghost Fleet | DARPA/Leidos | USA | Surface vessel fleet | Deployed |

### Ablation Study

**Table 1: Classification of Unique Risks Introduced by AI-LAWS**

| Risk Type | Description | Representative Scenario |
|-----------|-------------|------------------------|
| Undetected operational failure | Insufficient validation + overtrust → post-deployment failures difficult to detect | Target AI trained in forests misidentifies vehicles in desert; commanders trust flawed AI planning recommendations |
| Black-box decision opacity | Decision basis of AI strike systems difficult to understand or audit | AI strike system selects target based on faulty sensor; operator cannot intervene in time due to inability to understand the decision |
| Erosion of research freedom | Classified funding flows and dual-use restrictions → declining academic openness | University AI labs brought under military classified management, restricting publication and international collaboration |
| Militarization of AI talent | Civilian researchers recruited into military projects, sometimes without explicit notification or opt-out mechanisms | Scientists discover their projects have been taken over by defense funding, shifting from open-source AI to classified applications |
| Accelerated arms race | Widespread proliferation of AI-LAWS lowers conflict escalation threshold | Drone swarms deployed without adequate testing, creating security dilemmas due to technological uncertainty |

### Key Findings

1. **AI-LAWS are a present reality, not a hypothetical**: Multiple AI-LAWS spanning four military domains are already operationally deployed, with development nations including both major military powers and smaller states.
2. **Triple failure of existing regulatory frameworks**: MHC lacks measurable standards for "meaningful control"; AHJ does not define metrics for "adequate judgment"; the AGI risk narrative ignores already-deployed systems.
3. **Overtrust effects are amplified in military contexts**: Command advisory systems nominally preserve "human in the loop," yet operators tend to defer to AI recommendations under time pressure or ambiguous authority.
4. **Civilian AI research is being rapidly militarized**: The U.S. DIU has proposed embedding military AI research centers in universities; China implements a military-civil fusion strategy—the blurring of boundaries erodes academic freedom and international collaboration.
5. **The 2025 strike on the Weizmann Institute** suggests that civilian AI research institutions may become targets of attack as a consequence of AI militarization.

## Highlights & Insights

- **Systematic mapping from technical concepts to policy language**: The paper precisely maps established AI concepts—out-of-distribution generalization, adversarial brittleness, model drift, overtrust—onto military risks, avoiding the vague generalities common in policy papers.
- **Pragmatic behavior-oriented definition**: The dual-criterion design circumvents the definitional trap of "full autonomy," enabling coverage of semi-autonomous systems such as Lancet and Lavender that are actually in need of regulation.
- **Iterative design for international coalition**: Drawing on governance models from Internet protocols and financial risk auditing, the paper proposes a voluntary coalition rather than treaty law, balancing sovereignty concerns with shared responsibility.
- **Distinctive appeal to the AI research community**: The paper specifically warns against overstating benchmark results, noting that the military and think tanks closely monitor AI technical publications, and that hype may lead to the premature militarization of immature technologies.
- **Analysis of the "Minotaur warfare" concept**: The paper dissects proposals for LLMs serving as battlefield commanders, identifying the particular dangers of hallucination, drift, adversarial brittleness, and human overtrust in command contexts.

## Limitations & Future Work

- The **feasibility analysis of the five policy recommendations is underdeveloped**: for example, how a voluntary international coalition would respond to non-participating or actively defecting states (the policy environments of major AI-LAWS developers such as China and Russia are not examined in depth).
- The dual-criterion definition **remains ambiguous at the operational level**: how can one objectively determine whether AI is "integral" to a system's function? How can the boundary between "semi-autonomous" and "fully autonomous" be quantified?
- **No concrete technical framework for verification standards is provided**: the paper argues for the necessity of international verification standards but does not propose specific designs for behavioral benchmarks or performance thresholds.
- **Insufficient discussion of regulatory evasion driven by commercial interests**: the commercial incentives of defense technology companies (e.g., Anduril, Palantir) and their influence on regulatory compliance are not addressed.
- The system information cited is current as of mid-2025; given the **extremely rapid pace of AI militarization**, some analyses may already require updating.

## Related Work & Insights

- **Scharre (2023)**'s comprehensive discussion of LAWS and **Bode & Huelss (2023)**'s UN-level analysis provide the policy dialogue foundation on which this paper builds, adding an AI technical perspective.
- The U.S. **DoD Directive 3000.09** and the **UK 2024 Guidance** represent the AHJ and stronger-regulation pathways respectively; the paper argues that both lack AI-specific indicators.
- **Rivera et al. (2024)**'s simulations of AI-induced conflict escalation and **Lamparth et al. (2024)**'s research on unpredictable LLM behavior in military scenarios provide empirical support for the paper's risk arguments.
- Implications for AI safety researchers: foundational research on model robustness, interpretability, out-of-distribution detection, and adversarial robustness carries **direct and substantial policy implications** for military AI governance—the value of such technical work far exceeds its academic publication impact.
- Academic institutions should **proactively establish and publicly declare norms delineating civil and military research**, rather than passively awaiting government regulation or silently forfeiting academic freedom in ambiguity.

## Rating

- Novelty: ⭐⭐⭐⭐ The behavior-oriented dual-criterion definition and the systematic mapping of AI technical concepts to policy language are original; the five policy recommendations are more nascent in existing discourse but receive their most technically rigorous treatment here.
- Experimental Thoroughness: ⭐⭐⭐ No quantitative experiments, as expected of a policy position paper, but the paper systematically surveys the global AI-LAWS deployment landscape and risk cases with broad evidentiary coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentation is logically rigorous, technical and policy language are fluently integrated, the structure is clear and progressive, and the tables effectively support the analysis.
- Value: ⭐⭐⭐⭐⭐ The paper carries significant real-world importance for military AI governance, explicitly calls on the research community to engage rather than disengage, and the recommendations on nuclear red lines and the prohibition of AI generals carry particular urgency.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[NeurIPS 2025\] Emergency Response Measures for Catastrophic AI Risk](emergency_response_measures_for_catastrophic_ai_risk.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)

<!-- RELATED:END -->
