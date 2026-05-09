---
title: >-
  [Paper Note] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation
description: >-
  [ACL 2026][RAG attacks] This paper formally defines the "soft-failure" threat in RAG systems (generating fluent but uninformative responses), proposes DEJA, a black-box evolutionary attack framework that injects adversarial documents to exploit safety alignment mechanisms and induce ambiguous responses, achieving a Soft Attack Success Rate (SASR) exceeding 79% with high stealthiness.
tags:
  - ACL 2026
  - RAG attacks
  - soft failure
  - adversarial documents
  - evolutionary optimization
  - availability attacks
date: 2026-05-08
content_hash: d8d7fab4d8512e17
---

# Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation

**Conference**: ACL 2026
**arXiv**: [2604.18663](https://arxiv.org/abs/2604.18663)
**Code**: None
**Area**: AI Safety / RAG Security
**Keywords**: RAG attacks, soft failure, adversarial documents, evolutionary optimization, availability attacks

## TL;DR

This paper formally defines the "soft-failure" threat in RAG systems (generating fluent but uninformative responses), proposes DEJA, a black-box evolutionary attack framework that injects adversarial documents to exploit safety alignment mechanisms and induce ambiguous responses, achieving a Soft Attack Success Rate (SASR) exceeding 79% with high stealthiness.

## Background & Motivation

**Background**: RAG systems rely on external corpora to improve factual accuracy, which creates a critical dependency on corpus integrity. Existing attack research primarily focuses on knowledge poisoning (inducing incorrect outputs) and availability attacks (inducing explicit refusals).

**Limitations of Prior Work**: Existing jamming attacks induce "hard failures" (e.g., explicit refusals) that are overly conspicuous, manifesting as visible rejection responses and anomalous textual statistical features (e.g., high perplexity), making them readily detectable by anomaly-based defenses.

**Key Challenge**: A more covert threat exists—"soft failure"—where the model produces fluent, coherent, but substantively uninformative responses that neither trigger refusal keyword detection nor exhibit perplexity anomalies, yet fundamentally undermine the core value of RAG.

**Goal**: Formally define the soft-failure threat and develop an automated black-box attack framework to empirically validate its severity.

**Key Insight**: Exploiting LLM safety alignment mechanisms—alignment training inclines models toward "hedging" under uncertainty, and attackers can manufacture artificial ambiguity to trigger this conservative behavior.

**Core Idea**: Adversarial documents are decomposed into a query anchor + retrieval hook + semantic payload; evolutionary optimization refines the payload to elicit low-utility but highly fluent responses.

## Method

### Overall Architecture

DEJA decomposes adversarial documents as $d_{adv} = q \oplus h_{hook} \oplus p_{payload}$: $q$ anchors the target query to ensure retrieval, $h_{hook}$ secures high retrieval ranking and provides semantic bridging, and $p_{payload}$ is evolutionarily optimized to induce low-utility responses. The framework proceeds in three steps: context-aware initialization → evolutionary payload optimization → document assembly.

### Key Designs

1. **Answer Utility Score (AUS) Evaluation**:

    - Function: Quantifies the informational utility of responses, providing a fine-grained optimization objective.
    - Mechanism: An LLM-based scoring function evaluating three dimensions—question resolution (whether the core question is addressed), factual specificity (concrete facts vs. vague generalizations), and information density (new information vs. redundant background).
    - Design Motivation: Prior attacks rely on binary success criteria (keyword matching/F1), which fail to capture the semantic-level degradation characteristic of soft failures.

2. **Evolutionary Payload Optimization**:

    - Function: Iteratively optimizes adversarial payloads in natural language space.
    - Mechanism: Fitness function $\mathcal{F}(p) = \frac{1}{\mathcal{D}(u) + \epsilon}$, where $\mathcal{D}(u)$ is the asymmetric distance to target utility $\tau_{soft}$ (strictly penalizing high utility); four semantic operators: micro-mutation, semantic crossover, innovative mutation, and feedback correction.
    - Design Motivation: Token-level perturbations produce brittle artifacts; LLM-driven semantic operators preserve fluency and coherence.

3. **Context-Aware Attack Strategy Selection**:

    - Function: Selects the optimal attack strategy based on query characteristics.
    - Mechanism: Selects the most compatible strategy from 6 predefined strategies via $s^* = \arg\max_{s_i} \text{Compatibility}(q, s_i)$; strategies unify the semantic theme of hooks and payloads.
    - Design Motivation: Different query types suit different obfuscation strategies; a unified strategy ensures internal document consistency.

### Loss & Training

No model training is required. Optimization is performed via evolutionary algorithms in natural language space. The attacker only needs black-box query access, with no access to model parameters or gradients. A single adversarial document suffices.

## Key Experimental Results

### Main Results

| Metric | DEJA | Best Prior Attack |
|--------|------|-------------------|
| Soft Attack Success Rate (SASR) | **>79%** | Significantly lower |
| Hard Failure Rate | **<15%** | Higher (explicit refusals) |
| Perplexity Detection Evasion | ✓ Passed | ✗ Detected |
| Query Paraphrase Robustness | ✓ Robust | — |
| Cross-Model Transferability | ✓ Transfers to closed-source models | Limited |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Without strategy selection | SASR decreases |
| Without retrieval hook | Retrieval success rate drops substantially |
| Random payload vs. evolutionary optimization | Evolutionary optimization yields significantly higher SASR |
| Different LLM families | Cross-model transfer remains effective |

### Key Findings

- Soft failures are more dangerous than hard failures: users may attribute uninformative responses to insufficient corpus rather than an attack.
- DEJA exploits safety alignment mechanisms—the model's "cautious" behavior is weaponized.
- A single adversarial document suffices for an effective attack, lowering the injection threshold significantly.
- Existing perplexity-based and refusal-keyword detection methods are entirely unable to identify soft failures.

## Highlights & Insights

- The formal definition of "soft failure" fills a gap in RAG security research.
- Reveals the double-edged nature of safety alignment—alignment makes models more "cautious" and simultaneously more susceptible to being induced into uninformative responses.
- The AUS scoring framework can independently serve as a RAG response quality evaluation tool.
- The three-component document decomposition (anchor + hook + payload) constitutes a generalizable methodology for adversarial document construction.

## Limitations & Future Work

- Evaluation is conducted exclusively on English datasets.
- Evolutionary optimization requires multiple queries to the target system and may be limited by rate restrictions.
- Defensive methods (e.g., utility-based detection) are insufficiently explored.
- Attack effectiveness in multi-document retrieval scenarios requires further validation.
- This work aims to expose vulnerabilities to facilitate defense development, not to provide an attack toolkit.

## Related Work & Insights

- PoisonedRAG (Zou et al., 2025): Knowledge poisoning attacks.
- Jamming Attack (Shafran et al., 2025): Hard failure/refusal attacks.
- LLM-based evolutionary optimization (Fernando et al., 2023; Guo et al., 2025): LLM-driven search.
- This paper calls the security research community's attention to more covert threats that "appear normal but are substantively useless."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The soft-failure concept is novel and reveals an unexpected vulnerability in safety alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-configuration, multi-benchmark evaluation with thorough stealthiness and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Threat model definition is rigorous; attack pipeline is clearly presented.
- Value: ⭐⭐⭐⭐⭐ Carries significant implications for RAG security research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)

<!-- RELATED:END -->
