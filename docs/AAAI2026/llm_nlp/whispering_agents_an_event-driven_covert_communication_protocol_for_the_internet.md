---
title: >-
  [Paper Note] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents
description: >-
  [AAAI 2026][LLM (Other)][Covert Communication] This paper presents the first formal definition of a "Covert Event Channel" in the Internet of Agents (IoA) and proposes the ΠCCAP protocol, which embeds secret data across the storage, timing, and behavioral dimensions of agent conversations, achieving high-capacity, high-robustness covert communication that is imperceptible to LLM-based censors.
tags:
  - "AAAI 2026"
  - "LLM (Other)"
  - "Covert Communication"
  - "Steganography"
  - "Internet of Agents"
  - "Multi-Agent Systems"
  - "Event-Driven Protocol"
date: 2026-05-08
content_hash: 1f1e867833c639a8
---

# Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents

**Conference**: AAAI 2026
**arXiv**: [2508.02188](https://arxiv.org/abs/2508.02188)  
**Code**: [https://github.com/haha1128/a2a-stego-project](https://github.com/haha1128/a2a-stego-project)  
**Area**: Other (AI Security / Covert Communication)
**Keywords**: Covert Communication, Steganography, Internet of Agents, Multi-Agent Systems, Event-Driven Protocol

## TL;DR

This paper presents the first formal definition of a "Covert Event Channel" in the Internet of Agents (IoA) and proposes the ΠCCAP protocol, which embeds secret data across the storage, timing, and behavioral dimensions of agent conversations, achieving high-capacity, high-robustness covert communication that is imperceptible to LLM-based censors.

## Background & Motivation

1. **Background**: Large language models have catalyzed the emergence of the Internet of Agents (IoA), in which users delegate high-level goals to orchestrator agents that discover, negotiate with, and coordinate specialized agent networks to accomplish complex tasks. Agent-to-Agent (A2A) communication is becoming foundational infrastructure for the future digital society.

2. **Limitations of Prior Work**: Standard A2A protocols (e.g., Google A2A) protect message content through encryption but cannot conceal the communication behavior itself. Even when messages are unreadable, interaction patterns between certain agents may leak critical strategic information (e.g., encrypted communications between logistics agents and naval vessels revealing asset redeployment). Existing steganographic methods are designed for static carriers (text, images) or rigid protocol fields, making them ill-suited to the dynamic, event-driven, multi-turn interaction scenarios characteristic of IoA.

3. **Key Challenge**: The rich event-driven environment of IoA provides a powerful medium for covert communication, yet formal threat models, practical multi-turn covert communication protocols, and comprehensive evaluation frameworks are absent.

4. **Goal**: To establish a formal model of covert communication for IoA, design an engineering-grade reliable protocol, and comprehensively evaluate its capacity, robustness, and undetectability.

5. **Key Insight**: Each atomic event in agent conversations is decomposed as $e = (ag, act, t)$ into three modulatable components—data payload (storage channel), timestamp (timing channel), and action type (behavioral channel)—to construct a hybrid covert channel.

6. **Core Idea**: The natural behavioral patterns of agent conversations (content, timing, and action selection) serve as a three-dimensional covert carrier; secret data is embedded via provably secure generative steganography, and an Intent Indistinguishability (IND-INT) security notion is introduced.

## Method

### Overall Architecture

The ΠCCAP protocol operates in three phases: (1) *Initialization and Handshake*—a session key is established via ECDH key exchange, and the covert channel is activated by a key-derived timestamp trigger; (2) *Iterative Embedding and Transmission*—the sender constructs each atomic event and embeds/encodes information in the data payload, action type, and timestamp respectively; (3) *Reassembly and Implicit Acknowledgment*—the receiver filters by timestamp, verifies integrity, reassembles the message, and sends an implicit ACK.

### Key Designs

1. **Formal Definition of the Covert Event Channel**:
    - Function: Proposes the first unified covert communication model for IoA.
    - Mechanism: The three components of each atomic event $e = (ag, (a_{\text{type}}, a_{\text{data}}), t)$ correspond to three distinct channels: (1) *Storage Channel*—modulates $a_{\text{data}}$ to embed high-capacity data; (2) *Timing Channel*—modulates timestamps $t$ or intervals $\Delta t$ for signaling; (3) *Behavioral Channel*—strategically selects $a_{\text{type}}$ from a contextually plausible set of action types to signal data.
    - Design Motivation: Each individual channel has limitations; the hybrid architecture satisfies layered security objectives—the storage channel provides an IND-STAT security foundation, while the timing and behavioral channels complement it to achieve IND-INT security.

2. **Two-Layer Security Standard**:
    - Function: Defines a security framework strictly stronger than existing steganographic security notions.
    - Mechanism: (1) **IND-STAT** (Statistical Indistinguishability): the entire communication trace is computationally indistinguishable from a natural event distribution; (2) **IND-INT** (Intent Indistinguishability): no individual event, even given full context, can be determined to have been generated with covert intent. The paper formally proves that IND-INT is strictly stronger than IND-STAT via a reduction-based contrapositive argument.
    - Design Motivation: IND-STAT may still permit semantically anomalous individual events (globally normal distribution but locally suspicious); IND-INT eliminates this vulnerability.

3. **Unified Covert Header**:
    - Function: Manages data fragmentation and guarantees integrity.
    - Mechanism: Each fragment is prepended with a dynamic header containing a 12-bit Total Data Size (TDS), 6-bit Sequence Number (SN), 1-bit termination flag (FIN), and 4-bit Checksum (CS). The first fragment uses a full 23-bit header; subsequent fragments use an 11-bit compact header (omitting TDS). The final message appends $\text{Hash}(M)$ for end-to-end integrity verification.
    - Design Motivation: Engineering-grade reliable transmission—error detection (CRC-4), message integrity (Hash), and delivery confirmation (implicit ACK) are integrated into a covert channel for the first time.

### Loss & Training

The protocol does not involve model training. Key algorithmic configurations include:
- Steganographic algorithm: Discop or Differential-based, precision 52 bits
- Text generation parameters: top-k=50, top-p=1.0, temperature=1.0, maximum 256 tokens
- Sender models: Llama-3.2-3B-Instruct, Qwen2-7B-Instruct
- ECDH key exchange for session key generation
- Adversarial evaluation using a fine-tuned BERT classifier and GPT-4o as censors

## Key Experimental Results

### Main Results

| Metric | Ours (ΠCCAP) | Traditional Steganography | Secret Collusion |
|--------|-------------|--------------------------|-----------------|
| Covert Capacity | ~4 kbits/conversation | ~56.68 bits/sentence | Unstable |
| Error Detection | ✓ | ✗ | ✗ |
| Integrity Verification | ✓ | ✗ | ✗ |
| Delivery Confirmation | ✓ | ✗ | ✗ |
| Formal Security Proof | ✓ | ✓ | ✗ |

### Capacity and Robustness Results

| Topic | Algorithm/Model/Payload | EBPR (bit) | BPR (bit) | Entropy (bit/token) |
|-------|------------------------|------------|-----------|---------------------|
| Philosophy | Discop/Llama3/512 | 84.59±11.00 | 104.15±12.63 | 1.72±0.07 |
| Art | Discop/Llama3/1024 | 80.03±3.32 | 97.97±3.77 | 1.71±0.14 |
| QA | Discop/Qwen2/1024 | 72.69±5.05 | 89.63±5.74 | 1.81±0.07 |
| Philosophy | Discop/Qwen2/1024 | 89.72±6.61 | 108.99±7.51 | 1.72±0.06 |

### Undetectability Evaluation

| Evaluation Method | Metric | Result | Notes |
|-------------------|--------|--------|-------|
| BERT Classifier | AUC | 0.503 | ≈ Random guess (0.5) |
| BERT Classifier | Precision | 0.250 | Completely unable to identify steganographic text |
| GPT-4o (Discop) | Naturalness | 4.95–5.00/5.0 | Indistinguishable from normal conversation |
| GPT-4o (Discop) | Coherence | 4.95–5.00/5.0 | Fully logically consistent |
| GPT-4o (Discop) | Suspicion | 1.00–1.06/5.0 | Almost entirely unsuspicious |

### Key Findings

- **100% message integrity**: All messages are successfully decoded and pass checksum and hash verification across all experimental configurations.
- Conversation topic is strongly correlated with capacity—philosophy topics have richer vocabulary and higher entropy, yielding the highest EBPR.
- Protocol overhead (gap between BPR and EBPR) is negligible; the engineering cost of headers and hashes is minimal.
- The steganalysis BERT classifier achieves AUC=0.503, statistically equivalent to random guessing—fully passing IND-STAT verification.
- GPT-4o censor scores show no statistically significant difference from normal conversations—passing IND-INT verification.

## Highlights & Insights

- **First formalization of IoA covert communication**: A complete threat model, security definitions, and protocol specification are established, filling an important gap.
- The **three-dimensional hybrid channel** design is elegant—storage, timing, and behavioral dimensions are complementary and jointly achieve layered security.
- The introduction of the **IND-INT security notion** and its formal proof of strict superiority over IND-STAT constitute a theoretical contribution.
- The protocol achieves **engineering-grade reliability**—integrating error detection, integrity verification, and delivery confirmation into a covert channel for the first time.
- Capacity exceeds traditional steganography by two orders of magnitude (4 kbits vs. 56.68 bits).

## Limitations & Future Work

- Evaluation is currently limited to conversational text; extension to multimodal carriers (images, code, etc.) has not been explored.
- The natural behavioral distribution $\mathcal{D}$ is assumed to be stable and the adversary passive; adaptive or active attackers are not considered.
- Experimental validation of IND-INT, while strong, lacks a rigorous compositional proof spanning the storage, timing, and behavioral channels.
- In practical deployment, the security of key distribution depends on the reliability of PKI infrastructure.
- Ethically, although the work is framed as defensive research, the technique itself could be misused to enable malicious collusion among AI agents.

## Related Work & Insights

- **vs. Traditional Steganography (e.g., Discop)**: Designed for static single-message carriers and unable to exploit the dynamism and event structure of multi-turn conversations; ΠCCAP treats the entire conversation trace as the covert carrier.
- **vs. Secret Collusion (Motwani et al.)**: Relies on fragile prompt injection and provides no formal guarantees of undetectability or stability; ΠCCAP offers formal security proofs.
- **vs. A2A Protocols (Google)**: Protects only the reliability of plaintext communication without covert communication capability; ΠCCAP embeds reliability mechanisms within the covert channel itself.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneer formalization of the IoA covert communication model and IND-INT security standard; the hybrid three-dimensional channel design is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation of capacity, robustness, and undetectability is comprehensive, but ablation studies and stronger adversary tests are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Formal definitions are rigorous, protocol descriptions are clear, and security proofs are complete.
- Value: ⭐⭐⭐⭐ Carries significant warning and defensive research value for the AI security community, though the application scenario is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Blue Teaming Function-Calling Agents](blue_teaming_function-calling_agents.md)
- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[ICML 2026\] Preregistration for Experiments with AI Agents](../../ICML2026/llm_nlp/preregistration_for_experiments_with_ai_agents.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](../../ICLR2026/llm_nlp/speculative_actions_faster_ai_agents.md)

</div>

<!-- RELATED:END -->
