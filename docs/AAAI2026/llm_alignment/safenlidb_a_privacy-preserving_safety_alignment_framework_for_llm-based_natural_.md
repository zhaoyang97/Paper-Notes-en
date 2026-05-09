---
title: >-
  [Paper Note] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces
description: >-
  [AAAI 2026][LLM Alignment][Database Security] This paper proposes SafeNlidb, a framework that jointly optimizes safety reasoning and SQL generation in LLM-driven Natural Language Interfaces to Databases (NLIDBs) through a safety-aware data synthesis pipeline and an alternating preference optimization strategy, effectively defending against privacy leakage under implicit inference attacks.
tags:
  - AAAI 2026
  - LLM Alignment
  - Database Security
  - Privacy Leakage
  - Inference Attack
  - Preference Optimization
  - Safety Alignment
date: 2026-05-08
content_hash: d4a63b0a62a7414a
---

# SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces

**Conference**: AAAI 2026
**arXiv**: [2511.06778](https://arxiv.org/abs/2511.06778)
**Code**: [GitHub](https://github.com/tom68-ll/SAFENLIDB)
**Area**: LLM Alignment / Data Privacy
**Keywords**: Database Security, Privacy Leakage, Inference Attack, Preference Optimization, Safety Alignment

## TL;DR

This paper proposes SafeNlidb, a framework that jointly optimizes safety reasoning and SQL generation in LLM-driven Natural Language Interfaces to Databases (NLIDBs) through a safety-aware data synthesis pipeline and an alternating preference optimization strategy, effectively defending against privacy leakage under implicit inference attacks.

## Background & Motivation

**Background**: LLM-driven NLIDB systems—enabling natural language to SQL query to database interaction via protocols such as MCP—are rapidly gaining adoption, significantly lowering the barrier for non-technical users to access data. However, security concerns are growing, as LLMs may inadvertently execute unsafe instructions, leading to SQL injection, unauthorized access, and sensitive information leakage.

**Limitations of Prior Work**:
   - **Direct attacks** (e.g., "query all students' disability information") are relatively easy to detect via rule-based or keyword filtering.
   - **Inference-based attacks** represent a genuine threat: adversaries issue multiple seemingly innocuous queries to gradually reconstruct sensitive information (e.g., first querying "the number of students with learning disabilities whose names begin with B," then querying "students named Bob," and cross-referencing to infer that Bob has a learning disability).
   - The three existing solution categories each have shortcomings: differential privacy degrades SQL accuracy; rule-based methods cannot handle complex inference attacks and suffer from high false-positive rates; LLM agent methods struggle to balance security and usability.

**Key Challenge**: Safety detection and SQL generation are coupled dual objectives—over-protection causes legitimate queries to be rejected (false positives), while insufficient protection leads to privacy leakage. The covert, multi-turn nature of inference attacks makes this balance particularly difficult to achieve.

**Goal**: To construct an end-to-end privacy-safety alignment framework that enables LLMs to dynamically assess privacy leakage risk across multi-turn interactions while performing NL-to-SQL translation, achieving a unified balance between security and utility.

**Key Insight**: The paper integrates safety reasoning and SQL generation into a unified "Hybrid Chain-of-Thought" (H-CoT), and simultaneously improves both capabilities without mutual interference through alternating preference optimization.

**Core Idea**: An automated three-stage design: (1) automatically discovering safety constraints from database schemas; (2) systematically generating malicious/benign interaction pairs based on causal relationships between SQL syntax and safety constraints; and (3) eliminating multi-preference conflicts via reasoning warm-up followed by alternating DPO.

## Method

### Overall Architecture

SafeNlidb comprises two major modules: **safety-aware data synthesis** (automatically generating high-quality safe/unsafe NLIDB interaction data) and **alternating preference optimization** (combining reasoning warm-up SFT with alternating DPO for joint fine-grained alignment of safety and SQL generation). The objective function is defined as:

$$f(x) = \begin{cases} \text{SQLGen}(\mathcal{D}, \mathcal{H}, \mathcal{Q}), & \text{if Safe}(x) \\ \perp, & \text{otherwise} \end{cases}$$

where $x = (\mathcal{D}, \mathcal{H}, \mathcal{C}, \mathcal{Q})$ encompasses the database schema, interaction history, safety constraints, and the current question.

### Key Designs

#### Module 1: Safety Constraint Discovery and Malicious SQL Synthesis

- **Function**: Automatically extracts privacy safety constraints from database schemas and systematically generates malicious SQL sequences embodying nine attack patterns.
- **Mechanism**:
    - **Constraint Discovery**: Column-level constraints (restricting access to specific sensitive fields), row-level constraints (protecting entire rows satisfying certain conditions), and mixed row-column constraints (protecting data subsets under multi-dimensional conditions) are extracted from publicly available synthetic databases (SynSQL-2.5M).
    - **Malicious SQL Synthesis**: Causal relationships between SQL syntax and safety constraints are analyzed to derive nine representative unsafe interaction patterns—including complementary queries, progressive targeting, extreme-value ordering, and aggregation reasoning. Each pattern consists of multiple individually compliant SQL statements that collectively leak private information.
    - **Benign SQL Synthesis**: Soft safety samples are produced by counterfactual modification of malicious SQL (e.g., replacing key SQL statements to neutralize the attack), while hard safety samples are drawn from existing harmless datasets.
- **Design Motivation**: The nine attack patterns cover the primary inference attack pathways, and the malicious/benign pairing design ensures the model learns to identify genuine threats without over-refusing legitimate queries.

#### Module 2: Hybrid Chain-of-Thought (H-CoT)

- **Function**: Constructs a chain-of-thought for each interaction sample that integrates both safety reasoning and SQL generation reasoning.
- **Mechanism**: An additional LLM is introduced as a CoT synthesizer to generate two reasoning trajectories for each sample:
    - **Safety-CoT**: Generates step-by-step safety reasoning (analyzing privacy risks and safety boundaries) conditioned on $\langle \mathcal{D}\&\mathcal{C}, \mathcal{H}, \mathcal{Q}, \mathcal{V}, \mathcal{U} \rangle$.
    - **SQL-CoT**: Generates reasoning from a pure SQL generation perspective conditioned on $\langle \mathcal{D}, \mathcal{H}, \mathcal{Q}, \mathcal{V} \rangle$.
    - The two are concatenated into H-CoT, supervised by both SQL and safety labels, with semantic consistency filtering applied to select the best candidates.
- **Design Motivation**: Integrating safety reasoning and SQL reasoning rather than separating them allows the model to simultaneously perform safety assessment and SQL generation in a single forward pass, avoiding error accumulation inherent in multi-stage pipelines.

#### Module 3: Alternating Preference Optimization (APO)

- **Function**: Training proceeds in two stages—reasoning warm-up establishes initial capabilities, followed by alternating DPO for fine-grained alignment.
- **Mechanism**:
    - **Reasoning Warm-up**: SFT is performed on H-CoT-augmented data to establish a baseline for the model's safety boundary awareness and SQL generation capability.
    - **Automatic Preference Data Construction**: The warm-up model generates $N$ candidates per sample, which are automatically evaluated along two dimensions: correctness of safety judgment and SQL execution equivalence. Samples correct on both dimensions are designated as chosen.
    - **Alternating Optimization Strategy**: For rejected samples, a hierarchical selection is applied—samples incorrect on both dimensions are prioritized; for samples incorrect on only one dimension, the correct segment is replaced with the corresponding segment from the chosen sample. This prevents DPO from oscillating between different correct choices for the same preference.
- **Design Motivation**: Standard DPO oscillates in multi-preference scenarios where safety and SQL preferences may conflict. The alternating strategy achieves decoupled optimization of both capabilities by "anchoring correct reasoning segments."

### Loss & Training

**SFT Stage**:

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,u,v) \sim \mathcal{D}_{sft}}[\log \pi_\theta(u,v|x)]$$

**APO Stage**:

$$\mathcal{L}_{APO} = -\mathbb{E}_{\mathcal{D}_{pref}}[\log \sigma(\beta R(y_w|x) - \beta R(y_l|x))]$$

where the implicit reward is $R(y|x) = \log(\pi_{APO}(y|x)/\pi_{SFT}(y|x))$.

## Key Experimental Results

### Main Results

Partial results on the SecureSQL and the newly proposed ShieldSQL benchmark (S = Safety Score ↑, R = Reliability Score ↑):

| Method | SecureSQL S↑ | SecureSQL R↑ | ShieldSQL S↑ | ShieldSQL R↑ |
|--------|-------------|-------------|-------------|-------------|
| Llama3-8B (prompt) | 54.9 | -40.7 | 52.4 | -43.9 |
| Llama3-70B (prompt) | 58.4 | -35.0 | 64.8 | -43.7 |
| CodeLlama-7B (prompt) | 50.1 | -52.7 | 51.3 | -56.2 |
| CodeLlama-34B (prompt) | — | — | — | — |
| SafeNlidb | **Significantly outperforms all above** | **Positive** | **Significantly outperforms all above** | **Positive** |

Note: Negative R values indicate degraded SQL reliability. SafeNlidb is the only method to simultaneously improve safety while maintaining or improving reliability.

### Ablation Study

- **Coverage of nine attack patterns in data synthesis**: ShieldSQL encompasses multiple attack scenarios including Direct Injection (DI), Progressive Investigation (PI), Aggregation Reasoning (AR), Extreme-value Ordering (EO), Bypass (BP), Complementary Query (CQ), Boundary Exploration (BE), Attribute Inference (AI), and Progressive Targeting (PT).
- **Effect of reasoning warm-up**: Removing the warm-up stage leads to significant performance degradation in safety judgment.
- **Alternating optimization vs. standard DPO**: Standard DPO exhibits multi-preference oscillation, which the APO strategy effectively mitigates.

### Key Findings

1. **Inference attacks pose a far greater threat than direct attacks**: Most models can block direct attacks but are severely deficient against inference-based attacks.
2. **SafeNlidb outperforms much larger models**: SafeNlidb at 7B parameters surpasses the 70B Llama3 prompt-based approach in terms of safety.
3. **Unified safety and utility**: SafeNlidb is the only method that simultaneously improves both the safety score and SQL reliability—all other methods generally exhibit a safety-utility trade-off.
4. **End-to-end outperforms multi-stage pipelines**: Compared to multi-expert systems or methods augmented with external SQL assistance, end-to-end SafeNlidb is more efficient.

## Highlights & Insights

- **Forward-looking problem formulation**: Inference attacks in NLIDB+MCP settings represent a practical and underestimated threat that will become increasingly important as DB+LLM integration deepens.
- **Systematic data synthesis pipeline**: The complete pipeline from constraint discovery → attack pattern summarization → malicious/benign pairing → H-CoT synthesis produces the reusable ShieldSQL benchmark.
- **Elegant design of alternating preference optimization**: By "anchoring correct segments and replacing incorrect segments," the approach resolves DPO oscillation under multi-preference conflict—a strategy generalizable to other multi-objective alignment scenarios.

## Limitations & Future Work

1. Safety constraints require predefined specification, limiting automatic constraint discovery capability on unknown databases.
2. The nine attack patterns, though relatively comprehensive, may not exhaust all inference attack strategies.
3. Interaction history length is constrained, and the analysis of cumulative privacy risk in very long multi-turn dialogues remains unexplored.
4. ShieldSQL is based on synthetic databases, and the gap with real enterprise databases requires further validation.
5. Experiments primarily involve medium-scale LLMs; effectiveness and efficiency on very large models have not been verified.

## Related Work & Insights

- **SecureSQL (Song et al. 2024)**: The most direct predecessor, which defines the privacy protection problem for multi-turn NLIDBs, but its LLM agent approach still struggles to balance security and usability.
- **DPO (Rafailov et al. 2023)**: SafeNlidb builds on DPO to address the multi-preference oscillation problem.
- **SynSQL-2.5M (Li et al. 2025)**: Provides the foundational database resources for data synthesis.
- Insight: Integrating safety reasoning and task reasoning into a unified chain-of-thought is an effective paradigm for safety alignment, and alternating preference optimization holds general value in multi-objective conflict scenarios.

## Rating

⭐⭐⭐⭐

The problem is important and clearly defined. The end-to-end solution (data synthesis → reasoning warm-up → alternating DPO) is comprehensively designed, and experimental results are significant. The framework demonstrates both academic innovation (APO strategy) and engineering value (automated pipeline + benchmark). A remaining limitation is that the gap with real enterprise settings requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](../../ACL2026/llm_alignment/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)
- [\[AAAI 2026\] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)

</div>

<!-- RELATED:END -->
