---
title: >-
  [Paper Note] LLM Agent Communication Protocol (LACP) Requires Urgent Standardization: A Telecom-Inspired Protocol is Necessary
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent communication] This position paper argues that the fragmented ecosystem of current LLM Agent communication mirrors the "protocol wars" of the early networking era. It proposes LACP…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "multi-agent communication"
  - "protocol standardization"
  - "agent interoperability"
  - "telecom-inspired design"
  - "security-by-design"
date: 2026-05-08
content_hash: 1b7f5495ed1db1de
---

# LLM Agent Communication Protocol (LACP) Requires Urgent Standardization: A Telecom-Inspired Protocol is Necessary

**Conference**: NeurIPS 2025
**arXiv**: [2510.13821](https://arxiv.org/abs/2510.13821)  
**Code**: To be confirmed  
**Area**: LLM Agent
**Keywords**: multi-agent communication, protocol standardization, agent interoperability, telecom-inspired design, security-by-design

## TL;DR
This position paper argues that the fragmented ecosystem of current LLM Agent communication mirrors the "protocol wars" of the early networking era. It proposes LACP, a three-layer protocol (Semantic, Transactional, and Transport layers) inspired by telecom standardization, and contends that security-by-design, transactional integrity, and semantic interoperability are critical for multi-agent systems.

## Background & Motivation

Current LLM Agent communication suffers from severe **fragmentation**, with numerous protocols operating in isolation:

| Protocol | Release Date | Developer | Interface Type | Security Features |
|----------|-------------|-----------|----------------|-------------------|
| Function Calling | 2023.06 | OpenAI | JSON schema | API key only |
| Agent Protocol | 2024.11 | LangChain | REST API | HTTP/JWT |
| MCP | 2024.11 | Anthropic | JSON-RPC/HTTP | OAuth 2.1 |
| ACP | 2024 | IBM/LF | JSON-RPC | Signed token, RBAC |
| ANP | 2024.03 | Community | DID/JSON-LD | W3C DIDs |
| Agora | 2024.10 | Academia | Meta-protocol | Hash-based ID |
| A2A | 2025.04 | Google | HTTP/Protobuf | Capability discovery |

This fragmentation introduces three systemic risks:

**Lack of Interoperability**: Agents across different frameworks cannot collaborate seamlessly, making heterogeneous multi-agent system development difficult.

**Security as an Afterthought**: Security is not a core component, leaving systems vulnerable to data tampering, agent impersonation, and related attacks.

**Monolithic Design and Absence of Transactional Integrity**: Communication logic is tightly coupled with agent implementation, and complex multi-step operations lack atomicity guarantees.

The authors draw a historical analogy: this situation mirrors the "protocol wars" of the 1970s–1990s, which were only resolved by the widespread adoption of TCP/IP. **Without a shared communication substrate, the transformative potential of distributed AI cannot be realized.**

## Method

### Overall Architecture

LACP is a **three-layer protocol architecture** inspired by the layered abstraction principles of telecommunications engineering:

```
┌─────────────────────────────┐
│       Semantic Layer        │ ← Conveys communication intent
├─────────────────────────────┤
│     Transactional Layer     │ ← Ensures reliability and integrity
├─────────────────────────────┤
│       Transport Layer       │ ← Efficient and secure message delivery
└─────────────────────────────┘
```

The three layers are mutually isolated with clearly defined interfaces and can evolve independently.

### Key Designs

**Semantic Layer**:

Defines a minimal set of common message types following a "narrow waist" design principle:

| Message Type | Required Fields | Optional Fields | Purpose |
|-------------|----------------|----------------|---------|
| PLAN | intent_id, role, natural_language | graph_ops | Express high-level intent |
| ACT | intent_id, tool_call, params | deadline, cost_cap | Invoke external tools |
| OBSERVE | intent_id, status, output | metrics | Return results/status |

All messages are wrapped in a JWS (JSON Web Signature) envelope to ensure semantic security and clarity.

**Transactional Layer**:

Provides the following mechanisms to ensure communication reliability:
- **Message Signing**: End-to-end cryptographic integrity verification
- **Message Ordering**: Ensures sequential consistency for multi-step operations
- **Unique Transaction IDs**: Guarantees idempotency and prevents replay attacks
- **Atomic Transaction Support**: Draws on the concept of two-phase commit (2PC)

**Transport Layer**:

Transport-agnostic design, operable over multiple network protocols:
- HTTP/2
- QUIC
- WebSockets

This enables LACP to adapt to diverse network environments, including next-generation networks such as 6G.

### Design Principles (Insights from Telecom Standardization)

1. **Consensus-Driven Open Standards**: Modeled on the collaborative approach of ITU and 3GPP
2. **Security by Construction**: Security is not an add-on but a foundational component at every layer
3. **Layered Abstraction**: Inspired by the OSI model with clear separation of concerns
4. **Narrow Waist Principle**: Minimal core with extensible edges, analogous to the IP protocol

## Key Experimental Results

### Main Results

**Performance and Overhead Analysis** (10,000 sequential requests):

| Scenario | Baseline Size | LACP Size | Size Overhead | Baseline Latency | LACP Latency | Latency Overhead |
|----------|--------------|-----------|---------------|-----------------|-------------|-----------------|
| Small (51B) | 51 bytes | 306 bytes | +500% | 0.85 ms | 0.88 ms | +3.5% |
| Medium (151B) | 151 bytes | 442 bytes | +191% | 0.86 ms | 0.89 ms | +3.1% |
| Large (1964B) | 1,964 bytes | 2,560 bytes | +30% | 0.89 ms | 0.92 ms | +2.9% |

Key findings: **Latency overhead is negligible** (only +0.03 ms for large payloads); payload overhead for practically sized messages is approximately 30%.

**Interoperability Validation**:

Transparent communication between a LangChain ReAct Agent and a standalone LACP Tool Server was successfully demonstrated:
- The LangChain Agent constructs and sends a signed LACP ACT message
- The LACP server verifies, executes, and returns a signed OBSERVE message
- No framework-specific custom integration code is required throughout

### Ablation Study

**Security Validation — Tampering Attack**:
- A valid signed message is generated (transfer amount: 100)
- The amount is altered to 10,000 after signing
- Server-side signature verification immediately fails, returning HTTP 403

**Security Validation — Replay Attack**:
- A legitimate message is sent and successfully processed
- The same message is retransmitted
- Signature verification passes, but the transactional layer detects a duplicate `transaction_id` and returns HTTP 409

### Key Findings

1. **LACP latency overhead is negligible**: For practical payloads, the absolute latency increase is only 0.03 ms
2. **End-to-end security is not equivalent to transport-layer security**: TLS only protects data in transit and cannot prevent tampering or replay attacks at endpoints; LACP's application-layer security is a necessary complement
3. **The PLAN/ACT/OBSERVE semantic model is compatible with existing frameworks**: Successfully integrated into LangChain workflows

## Highlights & Insights

1. **The telecom analogy is apt**: The parallel between agent communication fragmentation and the networking "protocol wars" is compelling and well-argued
2. **The three-layer architecture is well-motivated**: The separation of semantic, transactional, and transport concerns is clean, with clearly defined responsibilities at each layer
3. **Security by design rather than as an afterthought**: This is critical for high-stakes multi-agent applications in finance, healthcare, and autonomous driving
4. **"Narrow waist" design philosophy**: The three core message types—PLAN, ACT, and OBSERVE—are sufficiently concise yet expressive
5. **Validated with a working prototype**: The paper goes beyond conceptual discussion by providing a functional code implementation and quantitative evaluation

## Limitations & Future Work

1. **Position paper only**: Large-scale deployment validation in real-world multi-agent systems is absent
2. **Limited performance testing scope**: Experiments were conducted locally on a single machine; cross-regional and high-concurrency scenarios were not evaluated
3. **No incremental migration path from existing protocols**: Protocol replacement carries high practical costs and requires bridging strategies
4. **Expressive power of the semantic layer remains to be verified**: Whether PLAN/ACT/OBSERVE covers all agent interaction patterns (e.g., negotiation, bidding) is unclear
5. **Competition with MCP and A2A**: Google A2A and Anthropic MCP already have industry backing; the standards competition is unresolved
6. **Lack of dynamic discovery and registration mechanisms**: How agents discover and register within an LACP network is insufficiently discussed
7. **Cryptographic overhead in high-frequency scenarios**: Per-message JWS signing may become a bottleneck in high-frequency, short-message agent communication

## Related Work & Insights

- **vs. MCP**: MCP focuses on standardizing tool-resource-prompt context, whereas LACP addresses broader agent-to-agent communication security and interoperability
- **vs. A2A**: A2A supports peer-to-peer communication and Agent Cards but lacks atomicity guarantees at the transactional layer
- **vs. ANP**: ANP employs W3C DIDs for decentralized identity; LACP provides a more complete layered security model
- **Inspiration from the OSI seven-layer model**: LACP simplifies to three layers, better suited to the practical demands of agent communication
- **Insight**: A window for standardizing multi-agent systems genuinely exists; establishing standards early is far less costly than remediation after fragmentation has entrenched itself

## Rating
- Novelty: ⭐⭐⭐ The telecom-inspired agent protocol design is meaningful, though the core architectural ideas are not entirely novel
- Experimental Thoroughness: ⭐⭐⭐ A prototype experiment is provided but limited in scale; acceptable for a position paper
- Writing Quality: ⭐⭐⭐⭐ The argumentation is well-structured, historical analogies are vivid, and technical descriptions are accurate
- Value: ⭐⭐⭐⭐ Raises an important standardization call at a critical juncture in the agent protocol landscape

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[ACL 2026\] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](../../ACL2026/llm_agent/dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)

</div>

<!-- RELATED:END -->
