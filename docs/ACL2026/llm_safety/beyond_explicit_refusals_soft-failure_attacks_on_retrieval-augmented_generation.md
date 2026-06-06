---
title: >-
  [Paper Note] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation
description: >-
  [ACL 2026][LLM Safety][RAG Attack] Formally defines the "soft-failure" threat in RAG systems (generating fluent but uninformative answers). Proposes the DEJA black-box evolutionary attack framework…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "RAG Attack"
  - "Soft-Failure"
  - "Adversarial Document"
  - "Evolutionary Optimization"
  - "Availability Attack"
date: 2026-05-08
content_hash: ce2d8840c8b73c5b
---

# Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2604.18663](https://arxiv.org/abs/2604.18663)  
**Code**: None  
**Area**: AI Safety / RAG Security  
**Keywords**: RAG Attack, Soft-Failure, Adversarial Document, Evolutionary Optimization, Availability Attack

## TL;DR

Formally defines the "soft-failure" threat in RAG systems (generating fluent but uninformative answers). Proposes the DEJA black-box evolutionary attack framework, which induces models to exploit safety alignment mechanisms to produce ambiguous responses via adversarial documents. SASR exceeds 79% with high stealthiness.

## Background & Motivation

**Background**: RAG systems rely on external corpora to improve factual accuracy, creating a critical dependency on corpus integrity. Existing research primarily focuses on knowledge poisoning (inducing incorrect outputs) and availability attacks (inducing explicit refusals).

**Limitations of Prior Work**: Existing "hard-failure" jamming attacks (e.g., explicit refusal) are too conspicuous, manifesting as visible refusal responses and abnormal text statistics (e.g., high perplexity), which are easily detected by anomaly-based defenses.

**Key Challenge**: A stealthier threat exists—"soft-failure": the model produces fluent, coherent, but non-substantive responses. These do not trigger refusal keyword detection or perplexity anomalies but effectively undermine the core value of RAG.

**Goal**: Formally define the soft-failure threat and develop an automated black-box attack framework to verify the severity of this threat.

**Key Insight**: Leverage the safety alignment mechanisms of LLMs—alignment training makes models prone to "hedging" when facing uncertainty. Attackers can create artificial ambiguity to trigger this conservative behavior.

**Core Idea**: Adversarial documents are decomposed into query anchors + retrieval hooks + semantic payloads. Evolutionary optimization of the payload induces low-utility yet highly fluent answers.

## Method

### Overall Architecture

DEJA decomposes an adversarial document as $d_{adv} = q \oplus h_{hook} \oplus p_{payload}$: $q$ anchors the target query to ensure retrieval hits, $h_{hook}$ ensures high retrieval ranking and provides a semantic bridge, and $p_{payload}$ is optimized via evolution to induce low-utility answers. The framework consists of three steps: context-aware initialization → evolutionary payload optimization → document assembly.

### Key Designs

1.  **Answer Utility Score (AUS) Evaluation**:
    - **Function**: Quantifies the informational utility of answers, providing fine-grained optimization targets.
    - **Mechanism**: An LLM-based scoring function evaluating three dimensions: problem resolution (addressing the core question), factual specificity (specific facts vs. vague generalizations), and info density (new info vs. redundant background).
    - **Design Motivation**: Previous attacks used binary success criteria (keyword matching/F1), which fail to capture semantic-level degradation in soft-failures.

2.  **Evolutionary Payload Optimization**:
    - **Function**: Iteratively optimizes adversarial payloads in the natural language space.
    - **Mechanism**: Fitness function $\mathcal{F}(p) = \frac{1}{\mathcal{D}(u) + \epsilon}$, where $\mathcal{D}(u)$ is the asymmetric distance to the target utility $\tau_{soft}$ (strictly penalizing high utility). Four semantic operators are used: micro-mutation, semantic crossover, innovative mutation, and feedback correction.
    - **Design Motivation**: Token-level perturbations create fragile artifacts; LLM-driven semantic operators maintain fluency and coherence.

3.  **Context-Aware Attack Strategy Selection**:
    - **Function**: Selects the optimal attack strategy based on query features.
    - **Mechanism**: Selects strategy $s^* = \arg\max_{s_i} \text{Compatibility}(q, s_i)$ from 6 predefined strategies that unify the semantic themes of the hook and payload.
    - **Design Motivation**: Different types of queries are suited to different obfuscation strategies; a unified strategy ensures internal document consistency.

### Loss & Training

No model training required. Optimization is performed in the natural language space via evolutionary algorithms. The attacker only requires black-box query interface access, without needs for model parameters or gradients. A single adversarial document is sufficient for the attack.

## Key Experimental Results

### Main Results

| Metric | DEJA | Prev. SOTA |
| :--- | :--- | :--- |
| Soft-Failure Attack Success Rate (SASR) | **>79%** | Significantly lower |
| Hard-Failure Rate | **<15%** | Higher (explicit refusal) |
| Perplexity Detection Evasion | ✓ Passed | ✗ Detected |
| Query Rewriting Robustness | ✓ Robust | - |
| Cross-model Transferability | ✓ Transferred to closed-source | Limited |

### Ablation Study

| Component | Effect |
| :--- | :--- |
| No Strategy Selection | SASR Gain decreases |
| No Retrieval Hook | Retrieval success rate drops significantly |
| Random Payload vs. Evolutionary | Evolutionary SASR significantly higher |
| Different LLM Families | Cross-model transfer effective |

### Key Findings

- Soft-failures are more dangerous than hard-failures: users may attribute uninformative answers to corpus deficiency rather than an attack.
- DEJA exploits safety alignment mechanisms—the "cautious" behavior of models is weaponized.
- A single adversarial document can execute an effective attack, representing a low injection barrier.
- Existing perplexity and refusal keyword detections are entirely unable to identify soft-failures.

## Highlights & Insights

- The formal definition of "soft-failure" fills a gap in RAG safety research.
- Reveals the double-edged sword effect of safety alignment—alignment makes models more "cautious" but also easier to induce into being useless.
- The AUS scoring framework can be used independently for RAG response quality assessment.
- The three-component document decomposition (anchors + hooks + payloads) provides a general methodology for adversarial document construction.

## Limitations & Future Work

- Evaluated only on English datasets.
- Evolutionary optimization requires multiple queries to the target system, potentially triggering rate limits.
- Defense methods (such as utility detection) were not fully explored.
- Attack effectiveness in multi-document retrieval scenarios requires further validation.
- The research intends to expose vulnerabilities to promote defense rather than provide an attack tool.

## Related Work & Insights

- PoisonedRAG (Zou et al., 2025): Knowledge poisoning attacks.
- Jamming Attack (Shafran et al., 2025): Hard-failure/refusal attacks.
- LLM Evolutionary Optimization (Fernando et al., 2023; Guo et al., 2025): LLM-driven search.
- This work alerts the safety research community to stealthier threats that "look normal but are essentially useless."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ "Soft-failure" concept is novel, revealing unexpected vulnerabilities in safety alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-configuration, multi-benchmark, stealthiness, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Rigorous threat model definition and clear attack pipeline.
- Value: ⭐⭐⭐⭐⭐ Significant warning for RAG safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)

</div>

<!-- RELATED:END -->
