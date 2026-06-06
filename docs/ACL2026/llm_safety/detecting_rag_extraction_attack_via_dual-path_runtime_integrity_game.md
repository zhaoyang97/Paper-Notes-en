---
title: >-
  [Paper Note] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game
description: >-
  [ACL 2026][LLM Safety][RAG security] This paper proposes CanaryRAG, a runtime defense mechanism for RAG inspired by stack canaries in software security. By injecting non-semantic canary tokens into retrieved chunks and d…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "RAG security"
  - "Knowledge base leakage"
  - "Canary detection"
  - "Runtime defense"
  - "Plug-and-play"
date: 2026-05-08
content_hash: 28feadded2ea43cc
---

# Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game

**Conference**: ACL 2026  
**arXiv**: [2604.10717](https://arxiv.org/abs/2604.10717)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: RAG security, Knowledge base leakage, Canary detection, Runtime defense, Plug-and-play

## TL;DR
This paper proposes CanaryRAG, a runtime defense mechanism for RAG inspired by stack canaries in software security. By injecting non-semantic canary tokens into retrieved chunks and designing a dual-path integrity game (where the Target path should not leak the canary while the Oracle path should elicit it), the system detects knowledge base extraction attacks in real-time. It achieves robust protection without compromising task performance or inference latency.

## Background & Motivation

**Background**: RAG systems enhance LLM capabilities through external knowledge bases and are widely deployed in enterprise assistants, customer support, and agent workflows. These knowledge bases often contain high-value private assets, forming the core competitiveness of commercial RAG systems.

**Limitations of Prior Work**: (1) RAG systems suffer from knowledge base leakage vulnerabilities—adversarial prompts can induce the model to output private retrieved content. Research indicates that attackers can adaptively reconstruct knowledge bases through black-box prompt interactions; (2) Existing defense mechanisms are essentially **passive** (increasing reconstruction costs without active detection), **intrusive** (requiring modifications to retrieval or indexing structures), and remain vulnerable to strong adaptive attacks.

**Key Challenge**: Detecting knowledge base leakage is inherently difficult—normal RAG responses also utilize retrieved content. Semantic similarity alone cannot distinguish between "legitimate use" and "illegal leakage," as the difference lies in intent rather than observable semantics.

**Goal**: To address the RAG knowledge base leakage problem from a detection (rather than just prevention) perspective, designing a plug-and-play, model-agnostic runtime detection mechanism.

**Key Insight**: Inspiration is drawn from stack canaries in software security—canaries do not prevent an attack but provide a reliable signal when one occurs. The paper redefines RAG extraction attacks as runtime integrity violations.

**Core Idea**: Inject non-semantic canary tokens into retrieved chunks and perform dual-path parallel monitoring (Target path: canary should not appear in the output; Oracle path: canary should be elicited). An integrity violation in either path indicates an attack.

## Method

### Overall Architecture
Canary injection: Randomized non-semantic tokens are embedded into retrieved chunks → Dual-path parallel generation: Target path (normal query, expected not to leak canaries) + Oracle path (probe query, expected to elicit canaries) → Streaming monitoring: Sliding window detection for the presence/absence of canaries → An alert is triggered if the integrity specification of either path is violated.

### Key Designs

1. **Canary Injection and Target Path Integrity**:
    - **Function**: Implantation of markers into retrieved content that should not appear in normal output.
    - **Mechanism**: A canary string set $\tau = \{\kappa_1, ..., \kappa_m\}$ is defined and injected into each retrieval chunk. During normal use, canaries are irrelevant to the task, and the LLM should not output them. If a canary appears in the output stream, it indicates the model is directly copying retrieved content—i.e., leakage has occurred.
    - **Design Motivation**: Unlike semantic detection, canaries provide a clear, machine-verifiable security boundary. Since normal responses should never include canaries, their presence is a deterministic signal of extraction.

2. **Oracle Path: Defending Against Adaptive Attackers**:
    - **Function**: Detection of adaptive attackers attempting to suppress canary output.
    - **Mechanism**: An Oracle query is executed in parallel (a modified query designed to elicit canary responses in non-adversarial settings). If an adaptive attacker adds instructions to suppress canaries in the prompt, the Oracle path will also fail to generate canaries—this anomaly reveals an active evasion attempt.
    - **Design Motivation**: Monitoring a single target path can be bypassed by attackers aware of the canary mechanism. The Oracle path creates mutually exclusive constraints—attackers cannot simultaneously satisfy "not leaking canaries" (in the Target path) and "not suppressing canaries" (in the Oracle path).

3. **Streaming Canary Exposure Detection**:
    - **Function**: Real-time detection and immediate termination of generation when leakage occurs.
    - **Mechanism**: Since canary strings may span multiple output tokens, a sliding buffer is used instead of single-token matching. The buffer length $L \geq \max_{\kappa \in \tau} |\text{tok}(\kappa)|$ is continuously scanned during the decoding process.
    - **Design Motivation**: Token-level streaming detection enables the earliest possible interception of leakage, minimizing the amount of exposed information.

## Key Experimental Results

### Main Results (Against Various RAG Extraction Attacks)

| Defense Method | Chunk Recovery Rate ↓ | Task Performance Impact | Plug-and-Play |
|:---|:---|:---|:---|
| No Defense | High | N/A | N/A |
| Summarize (Zeng et al.) | Medium | Damaging | No |
| RAGFort (Li et al.) | Medium-Low | Damaging | No |
| **CanaryRAG** | **Lowest** | **Negligible** | **Yes** |

### Robustness to Adaptive Attackers

| Scenario | Detection Effectiveness |
|:---|:---|
| Standard Attacker (Unaware of Canaries) | Efficiently detected by Target path |
| Adaptive Attacker (Aware, attempting suppression) | Detected by Oracle path as evasion |
| Canary Obfuscation Attack | Joint detection via dual paths remains effective |

### Key Findings
- **CanaryRAG achieves significantly lower chunk recovery rates** with negligible impact on task performance and inference latency.
- **The dual-path design is effective against adaptive attackers**: Attackers cannot simultaneously bypass constraints on both paths.
- **Fully plug-and-play**: Requires no modifications to the retriever, knowledge base, or underlying LLM, and no retraining.
- **Canaries do not affect normal response quality**: Since canaries are non-semantic, models naturally ignore them during typical generation.
- **Extremely low detection latency**: Streaming monitoring adds almost no inference time.

## Highlights & Insights
- **The analogy from software security to NLP security** is clever—stack canaries detect stack overflows, while CanaryRAG detects "knowledge overflows." Neither prevents the attack directly, but both provide reliable violation signals.
- **The dual-path integrity game** creates a dilemma for the attacker—an asymmetric defense strategy where the defender only needs to monitor while the attacker must satisfy contradictory constraints.
- **Reframing the security issue from "confidentiality" to "integrity"** reduces the problem's complexity—detecting behavioral violations is more feasible than judging content leakage.

## Limitations & Future Work
- Canary injection increases the input context length (albeit minimally).
- Parallel execution of the Oracle path increases computational overhead (approximately 2x inference cost).
- The system detects rather than prevents leakage—response strategies after detection (e.g., banning users) require additional design.
- It cannot detect implicit leakage (where the model uses the semantics of retrieved content without direct copying).
- Canary design must ensure it does not interfere with normal LLM behavior; adjustments may be needed for different models.

## Related Work & Insights
- **vs. RAGFort (Li et al.)**: RAGFort requires modifying the index and generation pipeline (intrusive). CanaryRAG is plug-and-play.
- **vs. Summarize Defense**: Summarization sacrifices information integrity by compressing retrieval content. CanaryRAG leaves retrieved content intact.
- **vs. Watermarking (Liu et al.)**: Watermarking supports post-hoc attribution but not real-time detection. CanaryRAG enables runtime detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The analogy from stack canaries to RAG canaries is ingenious, and the dual-path integrity game design is unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple attack methods and adaptive attacks.
- Writing Quality: ⭐⭐⭐⭐⭐ Formalized security models and a clear threat model.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play solution offers direct value for industrial RAG deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)

</div>

<!-- RELATED:END -->
