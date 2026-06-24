---
title: >-
  [Paper Note] Build Agent Advocates, Not Platform Agents
description: >-
  [ICML 2025][LLM (Other)][LLM Agent] A position paper arguing that language model agents (LMAs), if controlled by platform companies, will become "platform agents" that exacerbate surveillance, lock-in, and attention manipulation. The authors propose developing user-controlled "agent advocates" to protect individual autonomy, recommending three key interventions: open models/compute, interoperability standards, and market regulation.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "LLM Agent"
  - "Platform Economy"
  - "User Autonomy"
  - "Digital Rights"
  - "Agent Advocates"
date: 2026-05-08
content_hash: d2528c3041538af4
---

# Build Agent Advocates, Not Platform Agents

**Conference**: ICML 2025  
**arXiv**: [2505.04345](https://arxiv.org/abs/2505.04345)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: LLM Agent, Platform Economy, User Autonomy, Digital Rights, Agent Advocates

## TL;DR
A position paper arguing that language model agents (LMAs), if controlled by platform companies, will become "platform agents" that exacerbate surveillance, lock-in, and attention manipulation. The authors propose developing user-controlled "agent advocates" to protect individual autonomy, recommending three key interventions: open models/compute, interoperability standards, and market regulation.

## Background & Motivation

**Background**: LLM Agents (LMAs) are becoming the core product direction for AI companies. Companies like Anthropic, Google DeepMind, and OpenAI have released agent products capable of using browsers, executing programming tasks, and conducting deep research. Platform companies such as Meta, Google, Amazon, and Microsoft are actively positioning themselves in LMA development.

**Limitations of Prior Work**: The current digital economy is dominated by platform companies. Acting as intermediaries, these companies suffer from four major issues:
   - **Surveillance harm**: Users are tracked comprehensively, with data used for advertising and other purposes.
   - **Pseudo-market design**: Platforms manipulate bilateral transactions to extract value rather than facilitate fair competition.
   - **Attention control**: Google holds nearly 90% of the search market, and Meta's three platforms occupy three of the top four spots in social media.
   - **Governing power**: Platforms unilaterally set and enforce rules without legitimacy.

**Key Challenge**: The default evolutionary path of LMAs will be shaped by the existing structure of the platform economy, leading to "platform agents"—agents controlled by platform companies serving platform interests rather than user interests. This will exacerbate surveillance, lock-in, and manipulation. Platform agents will understand users far more deeply than current recommendation systems, enabling highly personalized manipulation.

**Goal**: How to disrupt the default path of platform agents? How to promote the development of agents that genuinely serve users? What technical and institutional interventions are required?

**Key Insight**: Leveraging political economy and mediation theory, mediation is conceptualized across three dimensions: representative vs. go-between, interest orientation (self-interested vs. other-interested), and whether they constructively shape relationships. Existing platforms are characterized as "constructive, self-interested go-betweens."

**Core Idea**: Construct "agent advocates"—user-controlled AI agents that act as loyal representatives of the user, realized through three major interventions: open models, interoperability standards, and market regulation.

## Method

### Overall Architecture

This is a position paper without technical methodology, presenting instead a conceptual framework and policy recommendations. The argumentative chain progresses as follows: Analysis of platform power $\rightarrow$ How platform agents exacerbate four major risks $\rightarrow$ How agent advocates mitigate these risks $\rightarrow$ Three major interventions to realize agent advocates $\rightarrow$ Responses to three categories of objections.

### Key Designs

1. **Four Major Risks of Platform Agents**:

    - **Exacerbation of Surveillance Harm**: Platform agents can accumulate intimate knowledge of users across diverse interactions, moving beyond statistical patterns to understand users like an "intimate friend." Deployment by self-interested commercial entities could lead to far more severe manipulation than exists today.
    - **Manipulation of Market Design**: As users rely increasingly on agents for daily transactions, they lose opportunities for independent research. Platforms can manipulate market choices more precisely, prioritizing self-operated products or those of paying advertisers.
    - **Escalation of Attention Control**: When platform agents browse the open web on behalf of the user, they control not only the information flow but also the presentation format, enabling "pervasive editing"—subtly nudging the user within conversations.
    - **Strengthened Governing Power**: Achieving fine-grained behavioral control through guardrails and nudges, where "alignment" technologies can be co-opted to restrict user liberty rather than safeguard user safety.

2. **Definition and Advantages of Agent Advocates**:

    - Acting as loyal representatives of users rather than platform go-betweens.
    - Operating on local hardware or encrypted private clouds, giving users full control over their data and agent actions.
    - Actively defending against platform surveillance: browsing platforms on behalf of users to retrieve information and avoid tracking.
    - Mitigating platform lock-in: providing underlying cross-platform interoperability (e.g., merging group chats across different messaging services).
    - Transforming the attention-to-revenue mapping: agents can judge on behalf of users whether content is worth paying for.

3. **Three Key Interventions**:

    - **Open Models and Compute**: Fostering open-weight models independent of platform control. Open models from Meta or Google carry commercial licensing restrictions (which may prohibit use in agents undermining platform revenues), while Chinese models are subject to state censorship. The authors recommend that liberal democracies establish public AI infrastructure.
    - **Interoperability and Technical Standards**: Designing agent-to-agent communication protocols (efficient protocols beyond natural language), certification systems (to certify agent qualifications and norms), and transaction clearinghouses (to prevent collusion and fraud among agents).
    - **Market Regulation**: Prohibiting platforms from blocking GUI access for agents, mandating APIs, and protecting data portability, resembling "net neutrality" principles.

### Loss & Training

Not applicable (position paper, no technical experiments).

## Key Experimental Results

### Main Results

There are no active experiments in this paper. A hypothetical comparative table (from the appendix) categorizes the agent spectrum into three positions:

| Dimension | Platform Agent | Intermediate State (Hardware Ecosystem) | Agent Advocate |
|------|---------------|------------------|----------------|
| Lock-in | Deep lock-in, high switching costs | Hardware ecosystem lock-in | Full interoperability |
| Surveillance | Comprehensive intrusive data collection | Limited privacy protection | Local data storage |
| Market Design | Manipulates transactions, favors self-owned products | Transparent with ecosystem preference | Open market interaction |
| Attention | Driven by platform interests | Focused on complementary hardware services | Absolute user autonomy |

### Ablation Study

The robustness of the arguments is evaluated by addressing three main classes of counterarguments:

| Counterargument | Core Argument | Authors' Response |
|---------|---------|---------|
| Platform agents are not a concern | LMAs lack performance / Markets self-correct / Platform agents offer convenience advantages | Even if only a subset of risks materialize, advancing advocates is warranted |
| Agent advocates are infeasible | Technical hurdles (difficulty of "loyal AI") / Institutional issues (companies may decay) / Historical precedents (decentralization often fails) | LMAs favor decentralization more than the existing web by actively weakening network effects |
| Agent advocates cannot solve all problems | Malicious usage / Non-cooperative multi-agent dynamics / Broad societal impacts | Agent advocates specifically target platform threats; paired with security frameworks, they yield a Pareto improvement |

### Key Findings

- Four main reasons platform agents are the default trajectory: path dependency, pre-existing user bases, pressure to realize ROI on AI, and standard competition/acquisition capacities.
- The greatest opportunity for agent advocates: LMAs can "bypass" the platforms' walled gardens, directly representing users in transactions with opposing parties.
- Relying solely on regulation is insufficient (especially given a lack of legislative will in the US); a dual-track strategy combining technical and institutional design is required.
- Key bottleneck: Open-source models face licensing restrictions from platform companies, and truly independent high-performance models remain highly scarce.

## Highlights & Insights

- **Three-Dimensional Analytical Framework of Mediation**: Categorizing mediation into representative vs. go-between, self-interested vs. other-interested, and neutral vs. constructive. This precisely captures the core issue of platforms, framing them as "constructive, self-interested go-betweens." The framework itself holds high analytical value.
- **Vision of an Agent Micro-transaction Economy**: LMAs can inspect content first to decide whether to purchase, utilizing "provable forgetting"—introducing a brand-new transaction model for the information economy.
- **Certification Systems for Ground-level Governance**: Users can mandate transactions strictly with agents exhibiting specific attributes (e.g., interacting only with agents of similar capability), enabling market-based bottom-up governance.
- **Insight on "Alignment Restricting Liberty"**: Pointing out that alignment technologies can be misappropriated by platforms, acting not to protect users from AI harm but to mandate adherence to platform rules.

## Limitations & Future Work

- **Lack of Technical Validation**: All arguments are speculative and analytical, lacking any prototype implementations or experimental validation.
- **Unresolved Governance Paradox**: Developers of agent advocates also possess commercial incentives, which may ultimately pull them toward platform-agent paradigms; proposed credentialing systems and clearinghouses remain highly preliminary.
- **Optimistic Assumptions on Agent Capabilities**: Assuming agent advocates can effectively browse platforms on behalf of users, detect tracking behaviors, and automate micro-transaction decisions, capabilities that are not yet fully proven.
- **Absence of a Global Perspective**: Discussions focus on the US/EU, neglecting the digital divide and context of the Global South.
- **Future Directions**: Implementing a prototype system for agent advocates to validate the feasibility of core functions such as cross-platform interoperability, privacy-preserving browsing, and credential systems.

## Related Work & Insights

- **vs. EU AI Act**: Regulatory pathways are viable in the EU but lack political feasibility in the US; agent advocates serve as a technical alternative.
- **vs. Platform Regulation (DSA)**: Regulation merely constrains existing actions but fails to shift the fundamental power dynamics.
- **vs. Cooperative AI Community**: This work is related to multi-agent competition/cooperation literature but anchors its analysis on political economy rather than game theory.
- **vs. "CERN for AI" Proposals**: This paper broadens the definition of public AI's mission—beyond understanding AI hazards to actively countering platform dominance.

## Rating

- Novelty: ⭐⭐⭐⭐ Invokes platform economics and mediation theory in LMA discussions, offering a highly unique and profound perspective.
- Experimental Thoroughness: ⭐⭐ A position paper with no empirical experiments; arguments are strictly deductive and policy-oriented.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured and rigorously argued; handles the three classes of rebuttals in a comprehensive and honest manner.
- Value: ⭐⭐⭐⭐ Offers important insights for the developmental trajectory and policy orchestration of the AI agent landscape.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](../../ACL2025/llm_nlp/axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](../../ACL2025/llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](../../ACL2025/llm_nlp/membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](../../ACL2025/llm_nlp/act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](../../ACL2025/llm_nlp/agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)

</div>

<!-- RELATED:END -->
