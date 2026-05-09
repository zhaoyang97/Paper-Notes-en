---
title: >-
  [Paper Note] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game
description: >-
  [ACL 2026][RAG Security] This paper proposes CanaryRAG, a RAG runtime defense mechanism inspired by stack canaries in software security. By injecting non-semantic canary tokens into retrieved chunks and designing a dual-path integrity game — the target path should not leak canary tokens, while the Oracle path should be able to elicit them — CanaryRAG detects knowledge base extraction attacks in real time, achieving state-of-the-art protection without compromising task performance or inference latency.
tags:
  - ACL 2026
  - RAG Security
  - Knowledge Base Extraction
  - Canary Detection
  - Runtime Defense
  - Plug-and-Play
date: 2026-05-08
content_hash: c6e92ba7132e544f
---

# Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game

**Conference**: ACL 2026
**arXiv**: [2604.10717](https://arxiv.org/abs/2604.10717)
**Code**: None
**Area**: Information Retrieval
**Keywords**: RAG Security, Knowledge Base Extraction, Canary Detection, Runtime Defense, Plug-and-Play

## TL;DR
This paper proposes CanaryRAG, a RAG runtime defense mechanism inspired by stack canaries in software security. By injecting non-semantic canary tokens into retrieved chunks and designing a dual-path integrity game — the target path should not leak canary tokens, while the Oracle path should be able to elicit them — CanaryRAG detects knowledge base extraction attacks in real time, achieving state-of-the-art protection without compromising task performance or inference latency.

## Background & Motivation

**Background**: RAG systems augment LLMs with external knowledge bases and have been widely deployed in enterprise assistants, customer support, and agentic workflows. Knowledge bases typically contain high-value proprietary assets that constitute the core competitive advantage of commercial RAG systems.

**Limitations of Prior Work**: (1) RAG systems are vulnerable to knowledge base extraction — adversarial prompts can induce models to output retrieved private content, and research has shown that attackers can adaptively reconstruct knowledge bases through black-box prompt interactions. (2) Existing defenses are fundamentally **passive** (raising reconstruction cost without actively detecting attackers), **intrusive** (requiring modifications to the retrieval or indexing components of the RAG pipeline), and remain vulnerable to strong adaptive attacks.

**Key Challenge**: Detecting knowledge base leakage is inherently difficult — normal RAG responses also consume retrieved content, and semantic similarity alone cannot distinguish "legitimate use" from "unauthorized disclosure," since the difference lies in intent rather than observable semantics.

**Goal**: Address the RAG knowledge base leakage problem from a detection (rather than prevention) perspective, and design a plug-and-play, model-agnostic runtime detection mechanism.

**Key Insight**: The design is inspired by stack canaries in software security — canaries do not prevent attacks but provide reliable signals that an attack has occurred. RAG extraction attacks are reframed as runtime integrity violations.

**Core Idea**: Inject non-semantic canary tokens into retrieved chunks, and employ dual-path parallel monitoring — the target path (canary tokens should not appear in the output) and the Oracle path (canary tokens should be elicitable). Any path deviating from expected behavior signals an attack.

## Method

### Overall Architecture
Canary injection: embed random non-semantic tokens into retrieved chunks → Dual-path parallel generation: target path (normal query, canary leakage not expected) + Oracle path (probe query, canary elicitation expected) → Streaming monitoring: sliding-window detection of canary presence/absence → Violation of either path's integrity specification triggers an alert.

### Key Designs

1. **Canary Injection and Target Path Integrity**:

    - Function: Plant markers in retrieved content that should never appear in normal outputs.
    - Mechanism: Define a canary string set $\tau = \{\kappa_1, ..., \kappa_m\}$ and inject it into each retrieved chunk. Under normal usage, canary tokens are task-irrelevant and should not appear in LLM responses. Canary tokens appearing in the output stream indicate that the model is directly copying retrieved content — i.e., knowledge leakage has occurred.
    - Design Motivation: Unlike semantic detection, canary tokens provide an explicit, machine-verifiable security boundary. Normal responses should never contain canary tokens; their presence is therefore a deterministic signal of extraction behavior.

2. **Oracle Path: Defense Against Adaptive Attackers**:

    - Function: Detect adaptive attackers who attempt to suppress canary output.
    - Mechanism: A parallel Oracle query (a modified query designed to elicit a canary response under non-adversarial conditions) is executed concurrently. If an adaptive attacker includes instructions to suppress canary tokens in the prompt, the Oracle path will also fail to produce canary tokens — this anomaly reveals active evasion attempts.
    - Design Motivation: Single target-path monitoring can be bypassed by an attacker who knows the canary mechanism. The Oracle path creates mutually exclusive constraints — an attacker cannot simultaneously satisfy "do not leak canaries" (on the target path) and "do not suppress canaries" (on the Oracle path).

3. **Streaming Canary Exposure Detection**:

    - Function: Detect leakage in real time and immediately terminate generation upon detection.
    - Mechanism: Since canary strings may span multiple output tokens, a sliding buffer rather than single-token matching is employed. The buffer length satisfies $L \geq \max_{\kappa \in \tau} |\text{tok}(\kappa)|$ and continuously scans during decoding.
    - Design Motivation: Token-level streaming detection enables the earliest possible interception of leakage, minimizing the amount of information exposed.

## Key Experimental Results

### Main Results (Against Multiple RAG Extraction Attacks)

| Defense Method | Chunk Recovery Rate↓ | Task Performance Impact | Plug-and-Play |
|---|---|---|---|
| No Defense | High | N/A | N/A |
| Summarize (Zeng et al.) | Moderate | Lossy | No |
| RAGFort (Li et al.) | Moderate-Low | Lossy | No |
| **CanaryRAG** | **Lowest** | **Negligible** | **Yes** |

### Robustness Against Adaptive Attackers

| Scenario | Detection Effectiveness |
|---|---|
| Standard attacker (unaware of canaries) | Target path detects effectively |
| Adaptive attacker (aware of canaries, attempts suppression) | Oracle path detects evasion behavior |
| Canary obfuscation attack | Dual-path joint detection remains effective |

### Key Findings
- **CanaryRAG achieves significantly lower chunk recovery rates** while imposing negligible impact on task performance and inference latency.
- **The dual-path design is effective against adaptive attackers**: an attacker cannot simultaneously circumvent the constraints of both paths.
- **Fully plug-and-play**: no modifications to the retriever, knowledge base, or underlying LLM are required, and no retraining is needed.
- **Canary tokens do not affect response quality for normal queries**: because they are non-semantic, the model naturally ignores them during normal generation.
- **Detection latency is minimal**: streaming monitoring introduces virtually no additional inference overhead.

## Highlights & Insights
- **The analogy from software security to NLP security** is particularly elegant — stack canaries detect stack overflows; CanaryRAG detects knowledge overflows. Both provide reliable violation signals without preventing the attack.
- **The dual-path integrity game** places attackers in a dilemma — an asymmetric defensive strategy in which the defender only monitors while the attacker must simultaneously satisfy contradictory constraints.
- **Recasting the security problem from "confidentiality" to "integrity"** reduces problem difficulty — detecting behavioral violations is more tractable than judging whether content has been leaked.

## Limitations & Future Work
- Canary injection marginally increases input context length.
- Parallel execution of the Oracle path incurs additional computational overhead (approximately 2× inference cost).
- The system detects rather than prevents leakage — post-detection response strategies (e.g., user blocking) require additional design.
- Implicit leakage (where the model leverages the semantics of retrieved content without direct copying) cannot be detected.
- Canary design must ensure no interference with normal LLM behavior, and may require adjustment for different models.

## Related Work & Insights
- **vs. RAGFort (Li et al.)**: RAGFort requires modifications to the indexing and generation pipeline and is therefore intrusive. CanaryRAG is plug-and-play.
- **vs. Summarize Defense**: Summarization defenses sacrifice information completeness by compressing retrieved content. CanaryRAG leaves retrieved content unmodified.
- **vs. Watermarking Methods (Liu et al.)**: Watermarking supports post-hoc attribution but not real-time detection. CanaryRAG enables runtime detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The analogy from stack canaries to RAG canaries is highly creative; the dual-path integrity game design is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple attack methods including adaptive attacks.
- Writing Quality: ⭐⭐⭐⭐⭐ Security model formalization is rigorous; threat model is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ The plug-and-play solution has direct practical value for industrial RAG deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation](../../AAAI2026/information_retrieval/ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](../../AAAI2026/information_retrieval/cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)

</div>

<!-- RELATED:END -->
