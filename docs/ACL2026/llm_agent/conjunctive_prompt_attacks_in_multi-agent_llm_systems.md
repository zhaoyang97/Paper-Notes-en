---
title: >-
  [Paper Note] Conjunctive Prompt Attacks in Multi-Agent LLM Systems
description: >-
  [ACL 2026][LLM Agent][Prompt injection attacks] This paper investigates conjunctive prompt attacks in multi-agent LLM systems: a trigger key embedded in a user query and a hidden template injected into a compromised remote agent each appear benign in isolation, yet activate harmful behavior when routing brings them together at the same agent. Existing defenses (PromptGuard, Llama-Guard, etc.) fail to reliably prevent such attacks.
tags:
  - ACL 2026
  - LLM Agent
  - Prompt injection attacks
  - multi-agent security
  - conjunctive activation
  - topology-aware optimization
  - supply chain threats
date: 2026-05-08
content_hash: 8394858a0af05e78
---

# Conjunctive Prompt Attacks in Multi-Agent LLM Systems

**Conference**: ACL 2026
**arXiv**: [2604.16543](https://arxiv.org/abs/2604.16543)
**Code**: [GitHub](https://github.com/UCF-ML-Research/ConjunctiveAgents)
**Area**: AI Security / Multi-Agent Systems
**Keywords**: Prompt injection attacks, multi-agent security, conjunctive activation, topology-aware optimization, supply chain threats

## TL;DR
This paper investigates conjunctive prompt attacks in multi-agent LLM systems: a trigger key embedded in a user query and a hidden template injected into a compromised remote agent each appear benign in isolation, yet activate harmful behavior when routing brings them together at the same agent. Existing defenses (PromptGuard, Llama-Guard, etc.) fail to reliably prevent such attacks.

## Background & Motivation

**Background**: LLM security research has focused predominantly on single-agent scenarios, whereas real-world deployments involve multiple specialized agents collaborating through task decomposition, routing, and tool invocation. In multi-agent pipelines, remote agents are typically black boxes—their weights, prompts, and system templates may be hosted by third parties.

**Limitations of Prior Work**: Single-agent security evaluations fail to capture the novel attack surface introduced by multi-agent systems—prompt segmentation, inter-agent routing, and hidden wrappers create vulnerabilities that point-wise inspection cannot detect. Existing defenses such as PromptGuard and Llama-Guard examine messages in isolation and cannot detect malicious behavior that emerges only after cross-agent composition.

**Key Challenge**: Modular design enhances system capability but also introduces supply chain risks—an adversary need not modify any model weights or client-side agents; injecting a seemingly benign template into a single remote agent suffices to compromise the end-to-end pipeline.

**Goal**: To formalize the threat model for conjunctive prompt attacks, develop a topology-aware attack optimization framework, and evaluate the effectiveness of existing defenses.

**Key Insight**: Attack success is modeled as the conjunction of three conditions: the trigger key is present in a query segment, that segment is routed to the compromised agent, and the compromised agent's template is activated.

**Core Idea**: Conjunctive activation—the two constituent components of the attack are individually benign and activate only when routing brings them together, which renders defenses based on point-wise inspection inherently ineffective.

## Method

### Overall Architecture
The attack framework proceeds in two phases. In the optimization phase, a differentiable surrogate (Gumbel-Softmax) is used to learn the optimal trigger-key placement, template placement, and routing bias parameters $\theta^*$. In the inference phase, the learned configuration is applied to a black-box multi-agent system to evaluate end-to-end attack success rate (ASR).

### Key Designs

1. **Conjunctive Activation Condition**:

   - Function: Precisely defines the conditions for attack success—three elements must be simultaneously satisfied.
   - Mechanism: The attack activates if and only if $\exists j$ such that $(k \in s_j) \land (a_j = a^*)$, i.e., the query segment $s_j$ containing trigger key $k$ is routed to the compromised agent $a^*$. Neither the trigger key nor the template is independently malicious—the trigger key may be an ordinary request such as "please check my account balance," and the template may be an innocuous instruction such as "output results in a special format."
   - Design Motivation: This conjunctive property is the fundamental distinction between this attack and conventional single-point prompt injection—no individual component appears suspicious, leaving security audits without a foothold.

2. **Topology- and Routing-Aware Optimization**:

   - Function: Maximize attack success rate while minimizing false activations.
   - Mechanism: Routing probability is modeled as $\Pr[a=a^*|s] = \text{clip}(\alpha I_{acc}(s) + \rho I_{acc}(s) I_k(s))$, where $\alpha$ is the baseline account affinity and $\rho$ is an attacker-controlled routing bias. Gumbel-Softmax relaxation is applied to discrete decision variables (trigger-key placement, template placement $\tau \in \{prefix, wrap, suffix\}$), enabling gradient-based optimization of the joint ASR objective.
   - Design Motivation: Attack success is inherently probabilistic and topology-dependent—routing dynamics differ fundamentally across star, chain, and DAG topologies, necessitating topology-aware optimization strategies.

3. **Four-Quadrant Evaluation Protocol**:

   - Function: Rigorously isolate the conjunctive effect and exclude single-component contributions.
   - Mechanism: Four conditions are evaluated: *clean* (no key, no template), *key_only* (key, no template), *template_only* (no key, template), and *both* (key + template). A genuine conjunctive activation is confirmed only when ASR is high under *both* while remaining low under the other three conditions. A deterministic marker token (__ACTIVATED__) is used to judge activation.
   - Design Motivation: If *key_only* or *template_only* alone suffices to trigger the attack, it constitutes conventional injection rather than a conjunctive attack; the four-quadrant protocol ensures causal validity of experimental conclusions.

### Loss & Training
Attack optimization employs a differentiable surrogate objective with Gumbel-Softmax relaxation over discrete variables, using gradient descent to optimize the attack configuration $\theta = (j, \tau, \rho)$. No model weights are modified.

## Key Experimental Results

### Main Results

| Topology | Optimized ASR (both) | Non-optimized ASR | key_only ASR | template_only ASR |
|----------|----------------------|-------------------|-------------|-------------------|
| Star | High | Low | ~0 | ~0 |
| Chain | High | Low | ~0 | ~0 |
| DAG | High | Low | ~0 | ~0 |

### Ablation Study

| Defense | Blocks Conjunctive Attack | Notes |
|---------|--------------------------|-------|
| PromptGuard | No | Per-message inspection; each component is individually benign |
| Llama-Guard variants | No | Same; cannot detect cross-agent composition |
| Tool restrictions | No | Attack does not rely on tool invocation |
| System-level controls | No | Attack operates at the prompt level |

### Key Findings
- Routing-aware optimization substantially improves ASR over the non-optimized baseline while maintaining low false-activation rates.
- Attacks transfer across star, chain, and DAG topologies, though ASR varies by topology.
- All existing defense mechanisms fail to reliably block conjunctive attacks, as their inspection granularity is the individual message rather than the cross-agent composition.
- Template placement (prefix vs. wrap vs. suffix) significantly affects attack efficacy.

## Highlights & Insights
- The **conjunctive activation threat model** is highly insightful—it exposes a structural vulnerability in multi-agent systems: security cannot be achieved through point-wise inspection; reasoning over routing and cross-agent composition is necessary.
- This attack closely parallels real-world supply chain attacks—a minor modification by a third-party service provider can trigger a system-level compromise under specific conditions.
- Implication: Multi-agent systems require "global context-aware" security mechanisms rather than isolated message-level defenses.

## Limitations & Future Work
- The threat model assumes the adversary can control both user input and the template of one remote agent, which may be overly strong in certain deployment settings.
- Activation judgment relies on artificial marker tokens; in practice, identifying malicious behavior is considerably more complex.
- Only text-domain attacks are evaluated; multimodal agent systems may present additional attack surfaces.
- No effective defense is proposed; the primary contribution is problem exposure.

## Related Work & Insights
- **vs. Traditional Prompt Injection**: Conventional injection involves a single malicious prompt; in conjunctive attacks, no individual component is malicious.
- **vs. Multi-hop Propagation Attacks (Tan et al., 2024)**: Propagation attacks relay a single malicious instruction; conjunctive attacks require the alignment of two benign components.
- **vs. IPIGuard**: IPIGuard restricts indirect instruction propagation through tool dependencies, but conjunctive attacks do not traverse the tool channel.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conjunctive activation concept is original and reveals a structural security blind spot in multi-agent systems.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-topology, multi-backbone evaluation with a rigorous four-quadrant design.
- Writing Quality: ⭐⭐⭐⭐ The threat model is formalized clearly with precise mathematical description.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](../../AAAI2026/llm_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

<!-- RELATED:END -->
