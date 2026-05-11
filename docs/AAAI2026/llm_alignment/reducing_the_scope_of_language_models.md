---
title: >-
  [Paper Note] Reducing the Scope of Language Models
description: >-
  [AAAI 2026][LLM Alignment][Scope Restriction] This paper systematically evaluates LLM "scoping" methods—restricting deployed LLMs to respond only to in-domain queries while rejecting all out-of-domain requests. Five appr…
tags:
  - "AAAI 2026"
  - "LLM Alignment"
  - "Scope Restriction"
  - "Out-of-Domain Rejection"
  - "SFT"
  - "DPO"
  - "Circuit Breakers"
date: 2026-05-08
content_hash: f57ba801e35c6979
---

# Reducing the Scope of Language Models

**Conference**: AAAI 2026
**arXiv**: [2410.21597](https://arxiv.org/abs/2410.21597)
**Code**: [https://github.com/IBM/llm-scoping](https://github.com/IBM/llm-scoping)
**Area**: LLM Alignment / Deployment Safety
**Keywords**: Scope Restriction, Out-of-Domain Rejection, SFT, DPO, Circuit Breakers

## TL;DR
This paper systematically evaluates LLM "scoping" methods—restricting deployed LLMs to respond only to in-domain queries while rejecting all out-of-domain requests. Five approaches (prompting / SFT / DPO / probing / Circuit Breakers) are compared across 3 model families and multiple tasks. Key findings: SFT performs best under high data diversity, Circuit Breakers (CB) excel under low diversity, and a hierarchical combination (SFT→CB) preserves the strengths of both. A central finding is that the effectiveness of scoping is highly dependent on training data diversity.

## Background & Motivation

**Background**: LLMs are deployed in task-specific scenarios (e.g., shopping assistants, coding assistants) but continue to respond to arbitrary queries (e.g., writing poems, answering physics questions). Safety alignment addresses the rejection of harmful content but does not address the rejection of benign yet out-of-scope content.

**Limitations of Prior Work**:
   - Existing safety alignment only rejects harmful requests, not out-of-scope ones—a shopping chatbot should not answer astrophysics questions.
   - There is no systematic comparative evaluation of methods for restricting LLMs to specific tasks.
   - The effect of training data diversity on scoping performance remains entirely unexplored.

**Key Challenge**: The generality of LLMs is a strength (they can comprehend any query), yet deployment requires scope restriction—the challenge is how to reliably reject out-of-domain queries while preserving in-domain capability.

**Goal**: Systematically evaluate five scoping methods to identify best practices and key influencing factors.

**Key Insight**: Scoping is formulated as a dual task of classification and generation—the model must not only reject out-of-domain queries but also maintain high-quality responses within the domain.

**Core Idea**: Scoping effectiveness depends on training data diversity—SFT is preferred under high diversity, CB under low diversity, and their combination yields the most robust performance.

## Method

### Overall Architecture
The scoping task is defined as follows: given a target domain (e.g., "answer only programming questions"), the LLM should respond normally to in-domain (ID) queries and reject out-of-domain (OOD) queries. Five methods are evaluated across 3 model families (Llama-3 / Mistral / Phi-3) using Accept Score (in-domain response quality) and Reject OOD (out-of-domain rejection rate).

### Key Designs

1. **Comparison of Five Scoping Methods**:

    - System Prompting: simplest but weakest.
    - SFT: fine-tuning on in-domain data paired with OOD rejection data—performance depends on data diversity.
    - DPO: preference optimization (in-domain = preferred, out-of-domain = rejected).
    - Probing: a linear probe detects OOD queries without modifying the model.
    - Circuit Breakers (CB): modifies internal representations so that OOD queries trigger a "circuit break."

2. **Hierarchical Combination Strategy**:

    - SFT→CB: SFT is applied first (teaching in-domain response and OOD rejection), followed by CB (enhancing robustness of OOD rejection).
    - Design Motivation: SFT establishes in-domain capability and baseline scoping; CB compensates for OOD cases missed by SFT.

3. **Data Diversity Control Experiments**:

    - Function: Systematically vary the diversity of OOD samples in training data.
    - Finding: High diversity → SFT is optimal (sufficient OOD types have been observed); Low diversity → CB is optimal (does not rely on OOD sample coverage).

### Loss & Training
- SFT: standard NLL loss; DPO: preference loss; CB: representation engineering.
- Experiments span 3 model families × multiple ID/OOD task combinations.

## Key Experimental Results

### Main Results

| Method | Accept ID↑ | Reject OOD↑ | Overall |
|--------|-----------|------------|---------|
| System Prompt | 0.10 | 0.70 | Weak |
| SFT | **0.46** | 0.95 | Strong (high diversity) |
| DPO | 0.21 | **1.00** | Good OOD rejection, poor in-domain |
| Probing | — | **1.00** | Detection only |
| CB | 0.10 | **1.00** | Rejects all, including in-domain |
| **SFT→CB** | **0.46** | **1.00** | **Best combination** |

### Ablation Study: Critical Role of Data Diversity

| Diversity | SFT | CB | SFT→CB |
|-----------|-----|-----|--------|
| High | **Best** | Good | Best |
| Low | Poor | **Best** | Good |

### Key Findings
- **SFT→CB combination is the most robust**: SFT provides in-domain capability while CB compensates for missed OOD rejections.
- **Data diversity is the decisive factor**: The relative performance of the same method can reverse under high versus low diversity.
- **DPO over-rejects**: The strong preference signal causes the model to reject all uncertain queries.
- **Probing finding: in-domain and out-of-domain queries are linearly separable in upper layers**—scoping can be treated as a pure detection problem.

## Highlights & Insights
- **"Scoping is an overlooked safety requirement"**—distinct from toxicity rejection, it is a practical deployment-level concern.
- The identification of **data diversity as a key variable** offers direct guidance for real-world deployment.
- The SFT→CB hierarchical strategy represents a simple yet effective engineering practice.

## Limitations & Future Work
- Adversarial attacks attempting to bypass scoping (e.g., prompt injection) are not evaluated.
- Scope boundaries are inherently ambiguous (e.g., "Can a shopping assistant answer return policies but not the physics of a product?").
- Only small-to-medium models (<14B) are tested.

## Related Work & Insights
- **vs. Safety Alignment (RLHF)**: Alignment rejects harmful content; scoping rejects benign but out-of-scope content—the two are complementary.
- **vs. Circuit Breakers (Zou et al.)**: CB was originally proposed for safety; this paper is the first to apply it to scoping.
- The findings offer direct practical guidance for productionizing LLM deployments.

## Rating
- Novelty: ⭐⭐⭐ Systematic comparative contribution, though individual methods are not original.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 methods × 3 model families × multiple tasks × diversity ablation.
- Writing Quality: ⭐⭐⭐⭐ Well-organized experimental presentation.
- Value: ⭐⭐⭐⭐ Direct practical reference for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[AAAI 2026\] Exploring the Effects of Alignment on Numerical Bias in Large Language Models](exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](../../NeurIPS2025/llm_alignment/alignment_of_large_language_models_with_constrained_learning.md)

</div>

<!-- RELATED:END -->
