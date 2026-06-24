---
title: >-
  [Paper Note] PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance
description: >-
  [ACL 2025][AI Safety][contextual integrity] Proposes PrivaCI-Bench, the largest contextual privacy evaluation benchmark to date (154K instances) built upon Contextual Integrity theory. It covers real court cases, privacy policies, and synthetic data from EU AI Act compliance checkers to evaluate the legal compliance capabilities of LLMs under HIPAA, GDPR, and the AI Act.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "contextual integrity"
  - "privacy evaluation"
  - "legal compliance"
  - "GDPR"
  - "AI Act"
date: 2026-05-08
content_hash: c274a165d4e2d930
---

# PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance

**Conference**: ACL 2025  
**arXiv**: [2502.17041](https://arxiv.org/abs/2502.17041)  
**Code**: [https://github.com/HKUST-KnowComp/PrivaCI-Bench](https://github.com/HKUST-KnowComp/PrivaCI-Bench)  
**Area**: AI Safety  
**Keywords**: contextual integrity, privacy evaluation, legal compliance, GDPR, AI Act

## TL;DR
Proposes PrivaCI-Bench, the largest contextual privacy evaluation benchmark to date (154K instances) built upon Contextual Integrity theory. It covers real court cases, privacy policies, and synthetic data from EU AI Act compliance checkers to evaluate the legal compliance capabilities of LLMs under HIPAA, GDPR, and the AI Act.

## Background & Motivation

**Background**: LLM privacy evaluation primarily focuses on PII pattern matching (i.e., whether models can identify/protect phone numbers, emails, etc.). Although several privacy benchmarks exist, they are small in scale and narrow in scope.

**Limitations of Prior Work**: (1) PII matching does not equate to genuine privacy protection—for instance, a doctor sharing a patient's medical records for treatment is permissible. (2) Existing contextual privacy benchmarks either rely on synthetic data that fails to reflect real-world distributions, or are limited to a single domain with a restricted number of samples. (3) There is no evaluation benchmark yet for the latest regulations, such as the EU AI Act (which entered into force in August 2024).

**Key Challenge**: Privacy concerns not only the type of information (what data) but, more importantly, the information flow (who to whom, for what purpose, and under what conditions the transmission occurs)—which refers to Contextual Integrity (CI) theory. However, existing benchmarks lack modeling of the complete context.

**Goal**: To build a large-scale, multi-regulatory, real-data-based contextual privacy compliance evaluation benchmark.

**Key Insight**: Leveraging the five-parameter framework of CI theory (sender, recipient, subject, information type, transmission principle) to parse regulatory documents and evaluation cases, and constructing an auxiliary knowledge graph to facilitate reasoning.

**Core Idea**: Integrating CI theory, legal regulations, real-world cases, and a knowledge graph into a comprehensive contextual privacy compliance benchmark.

## Method

### Overall Architecture
Data collection (court cases + privacy policies + AI Act synthetic data) $\rightarrow$ regulatory parsing (CI parameter extraction) $\rightarrow$ knowledge graph construction (role/attribute hierarchical graphs) $\rightarrow$ multiple-choice question generation (CI parameter probing) $\rightarrow$ three evaluation strategies (DP/CoT/RAG).

### Key Designs

1. **Data Source Diversity**:

    - **HIPAA**: 214 real court cases (medical domain)
    - **GDPR**: 2,462 real EU court cases + 675 privacy policies
    - **EU AI Act**: 3,000 synthetic cases enumerated from the official compliance checker
    - **ACLU**: 70 cases related to privacy and technology
    - Labels: permit / prohibit / not applicable

2. **Auxiliary Knowledge Graph**:

    - **Function**: Constructing a role knowledge graph $\mathcal{R}$ (8,993 roles, 91,876 edges) and an attribute knowledge graph $\mathcal{A}$ (7,875 attributes, 176,999 edges), which are 20 times larger than prior work.
    - **Mechanism**: To bridge the domain gap between terms in regulations (e.g., "covered entity") and concrete instances in cases (e.g., "Samsung"), WordNet and GPT-4o are utilized to construct parent-child hierarchical relations.
    - **Design Motivation**: Enabling LLMs to map concrete cases to legal regulatory articles.

3. **CI Parameter Probing (MCQ)**:

    - **Function**: Generating 147,840 multiple-choice questions (at 3 difficulty levels) to test whether LLMs understand CI parameters in context.
    - Easy: Large semantic discrepancy between distractors and the correct answer. Medium: Randomly selected distractors. Hard: Distractors semantically closest to the correct answer.

4. **Three Evaluation Strategies**:

    - **Direct Prompt (DP)**: Direct judgment.
    - **Chain-of-Thought (CoT)**: Step-by-step analysis.
    - **RAG**: Explains contextual legal terms first, retrieves relevant sub-articles using BM25, and then reasons with the retrieved context.

## Key Experimental Results

### Main Results

| Model | HIPAA | GDPR | AI Act | MCQ (Easy) | MCQ (Hard) |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | Medium-High | Medium | Medium | High | Medium |
| QwQ-32B | Medium | Medium | Medium-Low | Medium-High | Low-Medium |
| DeepSeek R1 | Medium | Medium | Medium-Low | Medium-High | Low-Medium |
| Small Open-Source Models | Low-Medium | Low | Low | Medium | Low |

### Ablation Study

| Finding | Explanation |
|------|------|
| LLMs can identify CI parameters | The MCQ Easy accuracy is relatively high |
| Insufficient privacy compliance judgment | The final compliance judgment accuracy is far lower than CI parameter recognition |
| RAG > CoT > DP | External knowledge retrieval significantly assists legal reasoning |
| AI Act is the most challenging | It is the latest regulation and is least covered in the pre-training data of LLMs |
| Reasoning models are not significantly better | QwQ and R1 show limited performance gain in compliance tasks |

### Key Findings
- **LLMs can identify CI parameters but struggle with compliance judgment**: This suggests that the bottleneck is not context understanding, but rather insufficient legal reasoning capability.
- **The EU AI Act is the biggest challenge**: Since it recently went into effect, almost no relevant cases exist in the LLM pre-training data.
- **The knowledge graph is crucial for RAG**: The parent-child hierarchical mapping of roles and attributes helps LLMs map concrete cases to regulatory articles.
- **Reasoning models (e.g., QwQ/R1) show no obvious advantage in legal compliance**: This might be because legal reasoning requires domain expertise rather than generic reasoning capabilities.

## Highlights & Insights
- **In-depth integration of CI theory and AI evaluation**: Rather than using simple PII matching, this work models the entire information flow context, aligning closer with real privacy demands.
- **First evaluation benchmark covering the EU AI Act**: Seizing the initiative in evaluating under the newest regulation, offering vital importance to AI compliance research.
- **Knowledge-graph-assisted legal reasoning**: The constructed role/attribute KG provides reusable infrastructure for legal AI.

## Limitations & Future Work
- The EU AI Act data is synthetic and might lack realism.
- Only three regulatory systems are covered; other key regulations (e.g., CCPA, PIPL) are not included.
- CI parameters are annotated by GPT-4o and verified manually, meaning annotation quality is constrained by LLM capability.
- The MCQ format may not fully reflect real-world legal reasoning scenarios.

## Related Work & Insights
- **vs. Mireshghallah et al. (2024)**: Their CI benchmark contains only 1,326 synthetic data points, whereas PrivaCI-Bench contains 154K instances and includes real-world cases.
- **vs. Privacy Checklist (Li et al. 2024)**: Their work only covers 214 HIPAA cases; PrivaCI-Bench expands to GDPR and the AI Act, with a knowledge graph 20 times larger.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of CI theory, legal compliance, and a large-scale benchmark is profound
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-regulation + real/synthetic data + CI probing + 3 evaluation strategies
- Writing Quality: ⭐⭐⭐⭐ Clean framework figures and solid theoretical foundations
- Value: ⭐⭐⭐⭐⭐ Provides a standardized evaluation tool for AI privacy compliance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal–Moral Responsibility](../../ICLR2026/ai_safety/formalising_human-in-the-loop_computational_reductions_failure_modes_and_legal-m.md)
- [\[ACL 2025\] CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](../../ICML2026/ai_safety/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity](../../ICML2026/ai_safety/abc-bench_an_agentic_bio-capabilities_benchmark_for_biosecurity.md)
- [\[ACL 2026\] OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](../../ACL2026/ai_safety/omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)

</div>

<!-- RELATED:END -->
