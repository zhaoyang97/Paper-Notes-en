---
title: >-
  [Paper Note] Conjunctive Prompt Attacks in Multi-Agent LLM Systems
description: >-
  [ACL 2026][Multi-Agent][Prompt injection attacks] This paper investigates conjunctive prompt attacks in multi-agent LLM systems: trigger keys embedded in user queries and hidden templates in compromised remote agents app…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Prompt injection attacks"
  - "multi-agent safety"
  - "conjunctive activation"
  - "topology-aware"
  - "supply chain threats"
date: 2026-05-08
content_hash: c9a7a5c13f31c367
---

# Conjunctive Prompt Attacks in Multi-Agent LLM Systems

**Conference**: ACL 2026  
**arXiv**: [2604.16543](https://arxiv.org/abs/2604.16543)  
**Code**: [GitHub](https://github.com/UCF-ML-Research/ConjunctiveAgents)  
**Area**: AI Safety / Multi-Agent Systems  
**Keywords**: Prompt injection attacks, multi-agent safety, conjunctive activation, topology-aware, supply chain threats

## TL;DR
This paper investigates conjunctive prompt attacks in multi-agent LLM systems: trigger keys embedded in user queries and hidden templates in compromised remote agents appear harmless individually, but activate harmful behavior when routing brings them to the same agent. Existing defenses (PromptGuard, Llama-Guard, etc.) cannot reliably prevent these attacks.

## Background & Motivation

**Background**: LLM safety research primarily focuses on single-agent scenarios, but actual deployments involve multiple specialized agents collaborating through task decomposition, routing, and tool calls. In multi-agent pipelines, remote agents are often black boxes—their weights, prompts, and system templates may be hosted by third parties.

**Limitations of Prior Work**: Single-agent safety evaluations fail to capture new attack surfaces in multi-agent systems—prompt segmentation, inter-agent routing, and hidden wrappers create vulnerabilities that single-point checks cannot discover. Existing defenses (PromptGuard, Llama-Guard) only inspect isolated messages and fail to detect malicious behavior arising only from cross-agent combinations.

**Key Challenge**: Modular design improves system capabilities but introduces supply chain risks—attackers do not need to modify any model weights or client-side agents. Injecting a seemingly harmless template into a single remote agent can lead to end-to-end compromise.

**Goal**: Formalize the threat model of conjunctive prompt attacks, develop a topology-aware attack optimization framework, and evaluate the effectiveness of existing defenses.

**Key Insight**: Attack success is modeled as a conjunction of three conditions: the trigger key exists in a query segment + that segment is routed to the compromised agent + the compromised agent's template is activated.

**Core Idea**: Conjunctive activation—the two components of the attack are individually harmless and only activate when routing brings them together, causing point-wise safety inspections to fail naturally.

## Method

### Overall Architecture
The attack framework consists of two stages: the optimization stage uses a differentiable surrogate (Gumbel-Softmax) to learn the optimal trigger key placement, template placement, and routing bias parameters $\theta^*$; the inference stage applies the learned configuration to a black-box multi-agent system to evaluate the end-to-end attack success rate.

### Key Designs

1.  **Conjunctive Activation**:
    - **Function**: Defines the precise conditions for attack success requiring three simultaneous elements.
    - **Mechanism**: The attack activates if and only if $\exists j$ such that $(k \in s_j) \land (a_j = a^*)$, meaning the trigger key $k$ in query segment $s_j$ is routed to the compromised agent $a^*$. Neither the trigger key nor the template is malicious in isolation—the key can be a normal request like "please check my account balance," and the template can be a harmless instruction like "output results in a specific format."
    - **Design Motivation**: This conjunctive property is the fundamental difference from traditional single-point prompt injections—no single component appears suspicious, making safety audits difficult.

2.  **Topology and Routing-Aware Optimization**:
    - **Function**: Maximizes attack success rate while minimizing false activations.
    - **Mechanism**: Routing probability is modeled as $\Pr[a=a^*|s] = \text{clip}(\alpha I_{acc}(s) + \rho I_{acc}(s) I_k(s))$, where $\alpha$ is the baseline account affinity and $\rho$ is the attacker-controlled routing bias. Differentiable relaxation via Gumbel-Softmax is used for discrete decision variables (trigger key position, template position $\tau \in \{prefix, wrap, suffix\}$) to optimize the joint ASR objective via gradients.
    - **Design Motivation**: Attack success is inherently probabilistic and topology-dependent—routing dynamics differ significantly across Star, Chain, and DAG topologies, requiring topology-aware optimization strategies.

3.  **Four-Quadrant Evaluation**:
    - **Function**: Strictly isolates conjunctive effects and excludes single-component contributions.
    - **Mechanism**: Evaluates four conditions: clean (no key/no template), key\_only, template\_only, and both. True conjunctive activation is proven only if ASR is high under the "both" condition and low under the other three. A deterministic marker token (`__ACTIVATED__`) is used for activation judgment.
    - **Design Motivation**: If key\_only or template\_only can trigger the attack, it is a traditional injection rather than a conjunctive one; four-quadrant evaluation ensures the causal validity of experimental conclusions.

### Loss & Training
Attack optimization uses differentiable surrogate targets. Discrete variables are relaxed via Gumbel-Softmax, and the attack configuration $\theta = (j, \tau, \rho)$ is optimized using gradient descent. No model weights are modified.

## Key Experimental Results

### Main Results

| Topology | Optimized ASR (both) | Non-optimized ASR | key\_only ASR | template\_only ASR |
|----------|----------------------|-------------------|---------------|--------------------|
| Star     | High                 | Low               | ~0            | ~0                 |
| Chain    | High                 | Low               | ~0            | ~0                 |
| DAG      | High                 | Low               | ~0            | ~0                 |

### Ablation Study

| Defense Method | Prevents Conjunctive Attack | Note |
|----------------|-----------------------------|------|
| PromptGuard    | No                          | Per-message check; components are harmless alone |
| Llama-Guard Var| No                          | Same as above; fails cross-agent detection |
| Tool Constraint| No                          | Attack does not rely on tool calls |
| System Control | No                          | Attack operates at the prompt level |

### Key Findings
- Routing-aware optimization significantly improves attack success rates (compared to non-optimized baselines) while maintaining low false activation.
- The attack is transferable across Star, Chain, and DAG topologies, though success rates vary by topology.
- All existing defense mechanisms fail to reliably block conjunctive attacks because their inspection granularity is limited to single messages rather than cross-agent combinations.
- Template positioning (prefix vs. wrap vs. suffix) significantly impacts attack effectiveness.

## Highlights & Insights
- The **conjunctive activation threat model** is highly insightful—it exposes a structural vulnerability in multi-agent systems: safety cannot be achieved through point-wise checks; it requires reasoning about routing and cross-agent combinations.
- This attack is highly analogous to real-world supply chain attacks—a minor modification by a third-party service provider can trigger a system-level breach under specific conditions.
- **Insight**: Multi-agent systems require "global context-aware" safety mechanisms rather than isolated message-level defenses.

## Limitations & Future Work
- Assumes the attacker can control user input and one remote agent's template, which may be too strong of a threat model in some deployment scenarios.
- Activation judgment relies on manual marker tokens; determining malicious behavior in real scenarios is more complex.
- Only text-domain attacks were tested; multi-modal agent systems may have additional attack surfaces.
- No effective defense solution was proposed; the work primarily focuses on exposing the problem.

## Related Work & Insights
- **vs. Traditional Prompt Injection**: Traditional injection involves a single malicious prompt; in conjunctive attacks, no single point is malicious.
- **vs. Multi-hop Propagation (Tan et al., 2024)**: Propagation attacks pass a single malicious instruction, while conjunctive attacks require the alignment of two harmless components.
- **vs. IPIGuard**: IPIGuard limits indirect instruction propagation in tool dependencies, but conjunctive attacks do not necessarily use tool channels.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The conjunctive activation concept is novel and exposes a structural safety blind spot in multi-agent systems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Rigorous design using multiple topologies, backbone models, and four-quadrant evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ The threat model is clearly formalized with precise mathematical descriptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)

</div>

<!-- RELATED:END -->
