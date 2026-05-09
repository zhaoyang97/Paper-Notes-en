---
title: >-
  [Paper Note] With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems
description: >-
  [AAAI 2026][LLM Agent][agent governance] This paper proposes the Agentic Risk & Capability (ARC) framework, which systematically identifies, assesses, and mitigates safety and security risks in agentic AI systems from a **capability** perspective, providing organizations with an actionable and structured methodology for governance.
tags:
  - AAAI 2026
  - LLM Agent
  - agent governance
  - risk assessment framework
  - capability analysis
  - technical controls
  - AI safety
date: 2026-05-08
content_hash: 9fecd0f0052e38cf
---

# With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems

**Conference**: AAAI 2026
**arXiv**: [2512.22211](https://arxiv.org/abs/2512.22211)
**Code**: None (framework openly available)
**Area**: LLM Agent / AI Governance
**Keywords**: agent governance, risk assessment framework, capability analysis, technical controls, AI safety

## TL;DR

This paper proposes the Agentic Risk & Capability (ARC) framework, which systematically identifies, assesses, and mitigates safety and security risks in agentic AI systems from a **capability** perspective, providing organizations with an actionable and structured methodology for governance.

## Background & Motivation

**Background**: 2025 has been heralded as the "year of AI agents," with major companies releasing LLM-based agent systems capable of autonomous reasoning, planning, and executing tasks such as writing code, browsing the web, and modifying files. However, agent systems are considerably more prone to unsafe behavior than foundation models, making governance substantially more challenging.

**Limitations of Prior Work**: Existing approaches are either too coarse-grained (e.g., the EU AI Act and NIST RMF offer only principled guidance without technical detail), too narrow (e.g., MAESTRO and OWASP focus on specific security threats and require cybersecurity expertise), or too micro-level (e.g., benchmarks such as AgentHarm test specific scenarios and cannot comprehensively identify risks).

**Key Challenge**: Agentic systems possess autonomy over a wide range of actions, introducing a risk surface far exceeding that of traditional LLM systems; yet performing customized, in-depth risk assessments for every agent system is not sustainable at scale.

**Goal**: To establish a systematic, scalable, and adaptable technical governance framework that enables organizations to apply differentiated risk management across heterogeneous agent systems.

**Key Insight**: The framework analyzes agent systems through the lens of **capabilities** rather than tools—because the same capability may be realized by multiple tools, and a single tool may support multiple capabilities, making tool-level governance both redundant and prone to obsolescence.

**Core Idea**: The ARC framework decomposes agent systems along three dimensions—Components, Design, and Capabilities—and constructs a complete mapping from elements → failure modes → harms → technical controls.

## Method

### Overall Architecture

The ARC framework consists of three major parts: **Elements → Risks → Controls**, supplemented by **Implementation** guidance. The core workflow is: first analyze the elements of the agent system, then identify potential risks, and finally recommend corresponding technical controls.

### Key Designs

#### Module 1: Three-Dimensional Element Analysis of Agent Systems

- **Function**: Decomposes agent systems into three analytical dimensions.
- **Mechanism**:
    - **Component dimension**: Analyzes the LLM engine, tools (MCP), instructions, and memory of individual agents.
    - **Design dimension**: Analyzes multi-agent architectural patterns (hierarchical delegation / parallel / sequential), roles and access control, and monitoring and traceability.
    - **Capability dimension**: Divided into cognitive capabilities (planning / agent delegation / tool use), interaction capabilities (natural language / multimodal / official communication / commercial transactions / internet access / computer operation / API interfaces), and operational capabilities (code execution / file and data management / system administration).
- **Design Motivation**: The capability perspective offers three advantages over a tool-level perspective: (1) a more comprehensive unit of analysis that avoids the redundancy and obsolescence of tool-level controls; (2) a natural support for differentiated risk management, as more capabilities entail higher risk; (3) action-based risks are more intuitive for non-technical stakeholders, facilitating cross-departmental collaboration.

#### Module 2: Risk Identification and Risk Register

- **Function**: Systematically identifies all potential risks in an agent system and constructs an organizational-level risk register.
- **Mechanism**: Each risk entry must satisfy three criteria: (1) it originates from a specific element (component / design / capability); (2) it corresponds to a failure mode (agent failure / external manipulation / tool or resource failure); (3) it leads to a safety or security harm.
    - Security harm types: data breach / application failure / infrastructure and network attacks / identity and access management.
    - Safety harm types: illegal and CBRNE activities / discriminatory content / inappropriate content / user safety compromise / misinformation dissemination.
- **Design Motivation**: The three-criteria cross-product enables systematic enumeration of risks; however, not all combinations are meaningful, and organizations must determine which risks to include in the register—supported by academic research or industry case evidence.

#### Module 3: Layered Technical Control System

- **Function**: Recommends technical controls for each identified risk and stratifies them by priority.
- **Mechanism**: A three-tier control system—
    - Level 0 (Cardinal): Baseline requirements that must be adopted as stated.
    - Level 1 (Standard): Should be adopted or meaningfully adapted.
    - Level 2 (Best Practice): Recommended for high-risk systems.
    - Each control aims to reduce the scope/severity of impact or reduce the probability of a specific failure mode.
- **Design Motivation**: The layered design allows organizations to prioritize control implementation according to their risk tolerance and resource constraints.

### Loss & Training

This paper presents a governance framework and does not involve model training. At the implementation level, however, several key strategies are proposed:

- **Risk contextualization**: Risks are assessed along two dimensions—impact (5 levels: minimal → catastrophic) and likelihood (5 levels: almost certain → rare)—taking into account domain sensitivity, use-case type, data sensitivity, and system criticality.
- **Residual risk assessment**: The framework acknowledges that technical controls cannot fully eliminate risk and requires evaluation of whether residual risk after controls is acceptable.
- **Continuous update mechanism**: The framework is designed to be iteratively updated to keep pace with the rapidly evolving agentic AI landscape.

## Key Experimental Results

### Main Results

This paper presents a framework and does not include conventional experiments. The following empirical support is provided:

| Element | Risk Analysis Coverage | Corresponding Controls |
|---------|----------------------|----------------------|
| Components (4 categories) | Complete risk register | Tiered controls |
| Design (3 categories) | Complete risk register | Tiered controls |
| Capabilities (13 categories) | Complete risk register | Tiered controls |

### Ablation Study

The paper demonstrates the framework's process through illustrative analyses, for example:
- Internet search capability + external manipulation failure mode → malicious website prompt injection attack → corresponding controls include input guardrails, escape filtering, and structured retrieval APIs.
- File and data management capability + agent failure → inefficient/redundant queries degrading database performance.
- Tool component + tool failure → tool fails to correctly authenticate user identity or permissions.

### Key Findings

1. The same capability may be subject to three distinct failure modes, each requiring different control strategies.
2. Some controls overlap (particularly those targeting prompt injection), which represents a justified defense-in-depth strategy.
3. Combinatorial risk across capabilities is a significant source of residual risk.

## Highlights & Insights

- **Novelty of the capability perspective**: Compared to tool-level analysis, the capability perspective balances comprehensiveness, scalability, and interpretability—particularly critical given the explosive growth in agent types and MCP tools.
- **Practical design**: The framework targets organizational governance teams and provides a complete risk register template that can serve directly as an implementation starting point.
- **Complementarity with existing work**: Rather than replacing benchmarks or AI control techniques, the framework provides an upper-level governance structure that organically integrates various technical mechanisms.

## Limitations & Future Work

1. The framework relies on human judgment to determine which risk combinations are meaningful; automated risk discovery mechanisms are absent.
2. Validation of the effectiveness of technical controls is insufficient—quantitative methods for residual risk assessment require further investigation.
3. Analysis of combinatorial risks and cascading failures among multiple agents remains insufficiently deep.
4. The framework is primarily oriented toward intra-organizational deployment and offers limited coverage of cross-organizational agent interactions (e.g., A2A protocols).
5. The capability taxonomy may require ongoing expansion as agentic AI technology continues to evolve.

## Related Work & Insights

- **TRiSM framework** (Raza et al.): Provides general metrics but lacks contextualized methods.
- **Dimensional governance** (Engin et al.): Tracks systems via three axes—decision authority / process autonomy / accountability—but thresholds are difficult to operationalize.
- **MAESTRO / OWASP**: Security-oriented; high barrier for developers without a security background.
- **Progent / AgentSpec**: Runtime permission control languages; applicable as concrete control mechanisms within the ARC framework.
- **AI Control paradigm**: Complementary to ARC—the former focuses on mechanism design, the latter on organizational governance.

## Rating

⭐⭐⭐

As an AI governance framework paper, the ARC framework's capability perspective represents a valuable contribution, and the classification system is comprehensive and systematic. However, quantitative validation and empirical evidence from real-world deployment are lacking. The practical value of the framework depends heavily on organizational execution capacity, and the paper is closer to a position paper than a technical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](../../ACL2026/llm_agent/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](../../ACL2026/llm_agent/how_adversarial_environments_mislead_agentic_ai.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](../../ICLR2026/llm_agent/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)

</div>

<!-- RELATED:END -->
