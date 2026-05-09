---
title: >-
  [Paper Note] Which LLM Multi-Agent Protocol to Choose?
description: >-
  [ICLR 2026][LLM Evaluation][Multi-Agent Protocol] This paper introduces ProtocolBench and ProtocolRouter, presenting the first systematic comparison of multi-agent communication protocols (A2A, ACP, ANP, Agora, etc.) across four dimensions—task success rate, latency, message overhead, and robustness—and proposes a learnable protocol router for scenario-adaptive protocol selection, reducing fault recovery time by up to 18.1%.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Multi-Agent Protocol
  - ProtocolBench
  - ProtocolRouter
  - A2A
  - Communication Protocol Evaluation
date: 2026-05-08
content_hash: cb5cd954dfa7bc57
---

# Which LLM Multi-Agent Protocol to Choose?

**Conference**: ICLR 2026
**arXiv**: [2510.17149](https://arxiv.org/abs/2510.17149)
**Code**: Available (benchmark artifacts included)
**Area**: LLM Evaluation
**Keywords**: Multi-Agent Protocol, ProtocolBench, ProtocolRouter, A2A, Communication Protocol Evaluation

## TL;DR

This paper introduces ProtocolBench and ProtocolRouter, presenting the first systematic comparison of multi-agent communication protocols (A2A, ACP, ANP, Agora, etc.) across four dimensions—task success rate, latency, message overhead, and robustness—and proposes a learnable protocol router for scenario-adaptive protocol selection, reducing fault recovery time by up to 18.1%.

## Background & Motivation

As large-scale multi-agent systems evolve, the communication protocol layer has become a critical yet underexplored factor affecting system performance and reliability:

**Protocol Proliferation**: A diverse set of agent communication protocols has emerged in recent years, including Google's A2A (Agent-to-Agent), ACP (Agent Communication Protocol), ANP (Agent Network Protocol), and Agora, yet no unified evaluation standard exists.

**Selection Difficulty**: In practice, protocol selection is typically based on intuition or experience, lacking data-driven decision support.

**Underestimated Performance Gaps**: Protocols are often regarded as mere "pipes" with limited impact on system performance; however, protocol differences can lead to completion time gaps of up to 36.5%.

**Lack of Standardized Evaluation**: Different protocols are evaluated using different tasks and metrics in their respective publications, making direct comparisons infeasible.

**Limitations of Single-Protocol Approaches**: No single protocol is optimal across all scenarios, yet existing systems typically rely on a single protocol.

The paper aims to establish a standardized protocol evaluation framework and achieve optimal protocol selection through adaptive routing.

## Method

### Overall Architecture

The work comprises two core components:

1. **ProtocolBench**: A standardized protocol evaluation benchmark that systematically compares agent protocols across four dimensions under multiple scenarios.
2. **ProtocolRouter**: A learnable protocol router that dynamically selects the optimal protocol based on scenario requirements and runtime signals.

### Key Designs

1. **ProtocolBench Evaluation Framework**:

    - **Four Evaluation Axes**:
        - Task Success Rate: Measures whether a protocol can support agents in correctly completing tasks.
        - End-to-End Latency: Total time from task assignment to completion.
        - Message/Byte Overhead: Additional communication cost incurred by the protocol.
        - Robustness Under Failures: Recovery capability in the presence of network failures, agent crashes, and other anomalies.
    - **Evaluation Scenarios**:
        - Streaming Queue: Streaming task processing scenario, testing throughput and latency.
        - Fail-Storm Recovery: Large-scale failure recovery scenario, testing robustness.
        - GAIA: General agent intelligence evaluation.
    - **Design Motivation**: To provide a comprehensive, fair, and reproducible protocol comparison framework.

2. **Protocol Implementation and Comparison**:

    - **A2A (Agent-to-Agent)**: Google's inter-agent communication protocol, emphasizing interoperability.
    - **ACP (Agent Communication Protocol)**: A structured messaging protocol based on FIPA standards.
    - **ANP (Agent Network Protocol)**: A distributed protocol designed for large-scale agent networks.
    - **Agora**: An open protocol supporting flexible message routing.
    - All protocols are evaluated in a unified test environment to ensure fair comparison.
    - **Design Motivation**: Coverage of mainstream protocols ensures broad applicability of conclusions.

3. **ProtocolRouter Dynamic Router**:

    - **Input Signals**: Scenario requirement descriptions (e.g., latency sensitivity, fault tolerance requirements) and runtime monitoring signals (e.g., current network status, agent load).
    - **Routing Granularity**: Supports scenario-level routing (one protocol for the entire scenario) and module-level routing (different protocols for different modules).
    - **Learning Approach**: A lightweight routing model trained on historical performance data, mapping scenario features to protocol selections.
    - **Design Motivation**: Since no universal protocol exists, adaptive selection enables full exploitation of each protocol's strengths.

4. **ProtocolRouterBench**:

    - A standardized benchmark specifically designed to evaluate protocol router performance.
    - Encompasses diverse scenario configurations and performance metrics.
    - **Design Motivation**: Provides reproducible evaluation standards for router research.

### Loss & Training

- ProtocolRouter is trained using supervised learning, with training data derived from historical execution records in ProtocolBench.
- Online learning enables continuous adaptation to new scenarios and runtime conditions.
- The latency overhead of routing decisions is minimal and does not affect overall system performance.

## Key Experimental Results

### Main Results

| Scenario | Metric | Best Protocol | Worst Protocol | Gap |
|----------|--------|---------------|----------------|-----|
| Streaming Queue | Completion Time | — | — | Up to 36.5% difference |
| Streaming Queue | End-to-End Latency | — | — | Mean difference of 3.48s |
| Fail-Storm Recovery | Recovery Time | — | — | Significant difference |
| GAIA | Task Success Rate | Scenario-dependent | Scenario-dependent | Consistent inter-protocol gap |

### ProtocolRouter Performance

| Baseline | Metric | ProtocolRouter | Best Single Protocol | Gain |
|----------|--------|----------------|----------------------|------|
| Fail-Storm Recovery | Recovery Time | Optimal | Second-best | 18.1% reduction |
| GAIA | Success Rate | Higher | Second-best | Scenario-dependent gain |
| Overall | Weighted Score | Optimal | Protocol-dependent | Consistent improvement |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|------------|---------|
| Scenario-level vs. Module-level Routing | Module-level superior | Finer-grained routing yields better adaptation |
| With/Without Runtime Signals | With signals superior | Real-time monitoring improves routing accuracy |
| Router Model Size | Lightweight sufficient | Small models achieve effective routing |
| Fixed Protocol vs. Router | Router consistently superior | Validates the necessity of adaptive selection |

### Key Findings

1. **Protocol Selection Significantly Affects Performance**: Performance gaps between protocols in the same scenario can reach 36.5%, far exceeding expectations.
2. **No Universal Protocol Exists**: The optimal protocol varies across scenarios; a single-protocol strategy inevitably involves trade-offs.
3. **Latency Gaps Are Prominent**: End-to-end latency differences of 3.48 seconds in the Streaming Queue scenario have substantial impact on real-time applications.
4. **Robustness Gaps Are Consistent**: In failure scenarios, different protocols exhibit stable and distinguishable recovery capability patterns.
5. **Adaptive Routing Is Effective**: ProtocolRouter outperforms the best single protocol across all scenarios, demonstrating the value of dynamic routing.
6. **Module-Level Routing Is Superior**: Different modules within the same system may benefit from different protocols; fine-grained routing yields better results.

## Highlights & Insights

1. **First Protocol Benchmark**: ProtocolBench fills a critical gap in multi-agent protocol evaluation, providing empirical support for protocol design and selection.
2. **Practical Routing Mechanism**: ProtocolRouter transforms protocol selection from a manual decision into a data-driven one, lowering the barrier to deployment.
3. **Comprehensive Four-Dimensional Evaluation**: Task success rate, latency, overhead, and robustness collectively cover the core concerns in real-world deployment.
4. **Module-Level Routing Insight**: The finding that different components within the same system may suit different protocols provides theoretical support for heterogeneous protocol architectures.
5. **36.5% Performance Gap**: This figure compellingly demonstrates that protocol selection is not a trivial implementation detail but a critical system design decision.

## Limitations & Future Work

1. **Protocol Coverage**: The number of evaluated protocols is currently limited; emerging protocols (e.g., MCP-related protocols) have not yet been incorporated.
2. **Scenario Diversity**: Although representative, the evaluation scenarios may not cover all real-world usage patterns.
3. **Routing Overhead**: While the router itself is lightweight, the additional routing cost in ultra-low-latency scenarios still warrants attention.
4. **Security Considerations**: Differences between protocols in terms of security (e.g., message encryption, authentication) have not been thoroughly evaluated.
5. **Large-Scale Validation**: The number of agents evaluated is limited; performance at the thousands-to-tens-of-thousands scale remains to be verified.
6. **Protocol Mixing Compatibility**: Module-level routing implies the coexistence of multiple protocols within a single system; compatibility and debugging complexity require further discussion.
7. **Dynamic Environment Adaptation**: The router's ability to adapt to runtime environmental changes (e.g., network topology changes, dynamic agent join/leave) needs further strengthening.

## Related Work & Insights

- **A2A Protocol (Google)**: A protocol standard for agent interoperability, emphasizing cross-platform compatibility.
- **MCP (Model Context Protocol)**: Anthropic's model context protocol, which, while not directly targeting inter-agent communication, has influenced protocol design thinking.
- **FIPA-ACL**: The traditional agent communication language standard, upon which ACP is built.
- **AutoGen / CrewAI**: Multi-agent frameworks that typically rely on fixed communication patterns.
- **Insights**:
    - Protocol-layer research may emerge as a new frontier for performance optimization in multi-agent systems.
    - The idea of adaptive protocol routing can be extended to other system-level decisions (e.g., model selection, tool selection).
    - Establishing a layered agent protocol standard analogous to the network protocol stack is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](../../ACL2026/llm_evaluation/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[AAAI 2026\] MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](../../AAAI2026/llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[NeurIPS 2025\] Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](../../NeurIPS2025/llm_evaluation/model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)

<!-- RELATED:END -->
