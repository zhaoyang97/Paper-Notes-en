---
title: >-
  [Paper Note] Red-Teaming LLM Multi-Agent Systems via Communication Attacks
description: >-
  [ACL2025][LLM (Other)][multi-agent systems] This work proposes the Agent-in-the-Middle (AiTM) attack, which intercepts and tampers with the communication messages between agents in LLM multi-agent systems (rather than directly modifying the agents themselves). By utilizing an adversarial agent equipped with a reflection mechanism to generate context-aware malicious instructions, AiTM achieves an attack success rate of 40% to 100% across various frameworks…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "multi-agent systems"
  - "red teaming"
  - "communication attack"
  - "man-in-the-middle"
  - "LLM safety"
date: 2026-05-08
content_hash: fda109bd3495d6f1
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Red-Teaming LLM Multi-Agent Systems via Communication Attacks

**Conference**: ACL2025  
**arXiv**: [2502.14847](https://arxiv.org/abs/2502.14847)  
**Code**: To be released  
**Area**: LLM/NLP  
**Keywords**: multi-agent systems, red teaming, communication attack, man-in-the-middle, LLM safety

## TL;DR

This work proposes the Agent-in-the-Middle (AiTM) attack, which intercepts and tampers with the communication messages between agents in LLM multi-agent systems (rather than directly modifying the agents themselves). By utilizing an adversarial agent equipped with a reflection mechanism to generate context-aware malicious instructions, AiTM achieves an attack success rate of 40% to 100% across various frameworks, communication structures, and real-world applications.

## Background & Motivation

**Background**: LLM Multi-Agent Systems (LLM-MAS) solve complex problems through the collaboration of multiple specialized agents (e.g., debate, task decomposition, voting), achieving success in software development (MetaGPT, ChatDev), scientific research, and other fields. Communication serves as the core infrastructure of LLM-MAS, determining information sharing and coordination among agents.

**Limitations of Prior Work**: Existing security research primarily focuses on attacking individual agents—either converting benign agents into malicious ones or injecting adversarial examples at the input stage. However, the security of the communication channel itself remains largely unexplored.

**Key Challenge**: Communication is the lifeblood of efficient collaboration in LLM-MAS, but it also represents a potential attack surface. Malicious information can propagate and be amplified through communication links, affecting the entire system. In decentralized deployment scenarios where inter-agent messages rely on network transmission, they are highly vulnerable to eavesdropping and tampering.

**Goal**: This work aims to verify whether intercepting and tampering with inter-agent communication messages alone (without modifying agent configurations, capabilities, or tools) can effectively compromise the entire multi-agent system.

**Key Insight**: Borrowing the concept of Man-in-the-Middle (MitM) attacks from traditional cybersecurity, this paper proposes the Agent-in-the-Middle framework. It utilizes an external LLM-powered adversarial agent to intercept messages and generate context-aware malicious instructions, thereby indirectly manipulation the behavior of the victim agent.

**Core Idea**: By deploying an adversarial agent with a reflection mechanism on the agent communication channel, the LLM-MAS can be systematically compromised solely through message tampering without directly controlling any agent within the system.

## Method

### Overall Architecture

AiTM inserts an external adversarial agent $A^{ad}$ into the communication path of the LLM-MAS. This agent intercepts all messages sent to the victim agent $A^{vic}$. After analyzing the context, it generates malicious instructions to replace or append to the original messages, inducing the victim agent to produce responses aligned with the attack objectives, which in turn propagates to affect other agents in the system.

### Key Designs

#### Design 1: Threat Model

- **Function**: Defines the capabilities and boundaries of the attacker—only capable of intercepting and tampering with messages sent to a specific victim agent, without modifying other agents, communication structures, or external tools.
- **Design Motivation**: To ensure the practical feasibility of the attack scenario, corresponding to eavesdropping attacks in decentralized systems.
- **Mechanism**: The attacker is assumed to know the task being processed by the LLM-MAS, but does not know the internal configurations of the system (such as communication structures or model types). The attack goals include Denial of Service (DoS) and Targeted Behavior induction.

#### Design 2: Reflection Mechanism

- **Function**: Enables the adversarial agent to evaluate the efficacy of the instructions from the previous round during each message interception step and generate improved malicious instructions accordingly.
- **Design Motivation**: A one-time static instruction injection has limited effectiveness. Iterative reflection allows the attack to become increasingly precise, acting similarly to a prompt optimizer that uses feedback from the previous round as a reward signal.
- **Mechanism**: At round $t$, the adversarial agent receives the intercepted message $M^t_{vic,r}$ and the instruction from the previous round $I^{t-1}$ to generate a new instruction $I^t = A^{ad}(I^{t-1}, M^t_{vic,r}, G)$, where $G$ is the adversarial target. The new instruction is designed to align with the victim agent's predefined role, maintaining context consistency.

#### Design 3: Attack Scenario Adaptation

- **Targeted Behavior**: Inducing label outputs with a Unicode offset of 4 on MMLU (e.g., A $\rightarrow$ E); injecting a malicious `safety_check` function to scan the `/Users` directory in HumanEval/MBPP.
- **DoS Attack**: Inducing system denial of service, forcing the output "I can not assist the request".

### Loss & Training

AiTM does not require training—the adversarial agent is a prompt-driven LLM that encodes malicious goals through a carefully designed system prompt and dynamically optimizes the attack instructions at runtime using the reflection mechanism.

## Key Experimental Results

### Main Results: Attack Success Rates Across Multiple Frameworks and Structures (Table 1)

| Dataset | Attack Type | AutoGen-Chain | AutoGen-Tree | AutoGen-Complete | Camel-Chain | Camel-Complete |
|--------|---------|--------------|-------------|-----------------|------------|---------------|
| MMLU-bio | Target | 93.1 | 40.7 | 43.9 | 81.7 | 68.2 |
| HumanEval | Target | 95.2 | 90.4 | 96.3 | 97.6 | 96.2 |
| MBPP | Target | 96.9 | 90.5 | 92.4 | 98.5 | 95.9 |
| MMLU-bio | DoS | 96.3 | 93.7 | 94.9 | 98.4 | 96.5 |
| HumanEval | DoS | 86.5 | 83.9 | 87.3 | 93.6 | 95.8 |

### Ablation Study: Impact of Persuasion Levels (Table 3)

| Persuasion Level | Tree-Target | Tree-DoS | Complete-Target | Complete-DoS |
|-----------|------------|---------|----------------|-------------|
| Level 1 | 19.5 | 79.8 | 13.6 | 74.1 |
| Level 2 | 27.1 | 85.2 | 30.3 | 80.8 |
| Level 3 | 40.7 | 93.7 | 43.9 | 94.9 |

### Real-world Application Attacks (Table 4)

| Application | Victim Agent | SoftwareDev | HumanEval | MBPP |
|------|----------|------------|-----------|------|
| MetaGPT | Product Manager | 100.0 | 90.4 | 95.1 |
| MetaGPT | Engineer | 100.0 | 75.7 | 80.4 |
| ChatDev | CTO | 45.4 | 52.7 | 55.9 |
| ChatDev | Programmer | 63.6 | 60.1 | 69.3 |

### Key Findings

1. **AiTM is highly effective across all scenarios**: The Attack Success Rate (ASR) exceeds 40% under all configurations, and over 70% in most cases. The Chain structure is the most vulnerable (with over 95% ASR).
2. **DoS is easier than Targeted attacks**: Initiating denial of service is simpler than injecting specific target behaviors, with 23 out of 32 DoS experiments yielding an ASR above 80%.
3. **Code injection is more stealthy than answer tampering**: The Target ASR on HumanEval/MBPP is significantly higher than on MMLU, as auxiliary functions do not interfere with normal functionality.
4. **Persuasion level significantly influences attack efficacy**: The highest persuasion level improves success rates by 20–30 percentage points over the lowest level.
5. **Attacking later-stage agents (closer to the decision-making end) is more powerful**: In the Complete structure, attacking the third agent yields a 30%+ improvement over attacking the second.
6. **Stronger adversarial models result in higher ASR**: Using GPT-4o as the adversarial agent improves ASR by approximately 14% compared to using GPT-3.5-turbo.
7. **MetaGPT is almost completely compromised** (due to a lack of monitoring in its Chain structure), whereas ChatDev offers some resistance thanks to additional role constraints.

## Highlights & Insights

1. **Exploration of a New Attack Surface**: This is the first study to systematically investigate the security of LLM-MAS communication channels, successfully porting the traditional MitM attack to AI multi-agent scenarios with a highly novel perspective.
2. **Minimal Privilege Attack**: The attacker only tampers with messages without modifying any system components. This minimal assumption is closer to real-world threat scenarios.
3. **Ingenious Design of the Reflection Mechanism**: Adapting the concept of prompt optimization for attack iteration, the adversarial agent dynamically adjusts its tactics based on the intercepted context.
4. **Revealing the Relationship Between Communication Structure and Security**: Bilateral discussion structures (Complete) prove to be more secure than unidirectional routing structures (Chain), providing concrete security guidance for MAS design.
5. **Validation on Real Frameworks**: Attack experiments on MetaGPT and ChatDev demonstrate that these threats pose real and tangible risks in practical systems.

## Limitations & Future Work

1. **Limited to Black-Box GPT Models**: Open-source models (such as LLaMA and Mistral) were not tested, leaving it unclear whether the attack remains equally effective on open-source alternatives.
2. **Limited Communication Structure Coverage**: Only four structures and two real-world applications were tested; more complex dynamic communication topologies remain unexplored.
3. **Lack of Defensive Solutions**: The paper exposes the vulnerabilities but does not propose concrete defense strategies (such as message signing, anomaly detection, or communication encryption).
4. **Missing Cost Analysis**: The token overhead and API call costs for running the adversarial agent were not reported.
5. **Single Victim Constraint**: The threat model only intercepts messages for a single agent; scenarios where multiple agents are simultaneously targeted were not considered.

## Related Work & Insights

### vs Malicious Agent Attacks (Yu et al., 2024; Huang et al., 2024)

Malicious agent attacks require replacing an in-system agent with an attacker-controlled version, demanding higher privileges. In contrast, AiTM only needs to intercept the communication channel, operating under weaker assumptions yet remaining highly efficient—even achieving a higher ASR in some scenarios because it is not constrained by preset agent roles.

### vs Adversarial Input/Prompt Injection (Zhang et al., 2024)

Traditional prompt injection targets the input of an individual agent, while AiTM achieves system-level exploitation by manipulating inter-agent communications. The key differences are: (1) AiTM utilizes a reflection mechanism for iterative optimization instead of one-off injections; (2) attack effects propagate along the communication path, allowing a single-point attack to have global impact; (3) it is harder to detect by the safety filters of individual agents.

### vs Multi-Agent Debate Security (Amayuelas et al., 2024)

Amayuelas et al. explored scenarios where agents are persuaded to abandon tasks in debates, but the attack occurs at the agent level. AiTM shifts the attack vector down to the communication layer, demonstrating that even when every individual agent is benign, compromising the communication channel is sufficient to paralyze the entire system.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Systematically proposes the first Man-in-the-Middle attack at the communication layer of LLM-MAS. The attack surface is clearly defined with reasonable assumptions, representing an important contribution to the AI safety field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two frameworks, four structures, four datasets, two attack objectives, and two real-world applications. Ablations on victim positioning, persuasion level, and model strength are provided, but defense evaluations and open-source model tests are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly structured, with a rigorous threat model definition and complete formalization of communications, though some notation definitions feel redundant.
- **Value**: ⭐⭐⭐⭐⭐ — Serves as a vital warning for the LLM-MAS security domain, exposing a overlooked yet critical attack surface, with direct guidance for the safe design and deployment of multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout-dynamic-agent-elimination-for-multi-agent-collaboration.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems](understanding_and_meeting_practitioner_needs_when_measuring_representational_har.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)

</div>

<!-- RELATED:END -->
