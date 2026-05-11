---
title: >-
  [Paper Note] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction
description: >-
  [NeurIPS 2025][LLM Safety][Text-to-SQL] This paper proposes MaskSQL, a framework that protects privacy by replacing sensitive table names, column names…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Text-to-SQL"
  - "Privacy Protection"
  - "Prompt Abstraction"
  - "LLM-SLM Hybrid"
  - "Database Security"
date: 2026-05-08
content_hash: f99afdeba45b397a
---

# MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction

**Conference**: NeurIPS 2025  
**arXiv**: [2509.23459](https://arxiv.org/abs/2509.23459)  
**Code**: [https://github.com/sepideh-abedini/MaskSQL](https://github.com/sepideh-abedini/MaskSQL)  
**Area**: AI Security / Privacy Protection  
**Keywords**: Text-to-SQL, Privacy Protection, Prompt Abstraction, LLM-SLM Hybrid, Database Security

## TL;DR

This paper proposes MaskSQL, a framework that protects privacy by replacing sensitive table names, column names, and data values with abstract symbols before sending prompts to a remote LLM. Combined with a local SLM for schema linking and SQL reconstruction, MaskSQL preserves privacy while surpassing SLM-only approaches in SQL generation accuracy.

## Background & Motivation

**Background**: In Text-to-SQL, LLMs (e.g., GPT-4) achieve the strongest performance but require remote API access, exposing sensitive database schemas and user data. SLMs can be deployed locally but perform poorly on complex queries.

**Limitations of Prior Work**: (1) Cryptographic methods (HE/MPC) incur prohibitive computational overhead at LLM scale; (2) differential privacy degrades utility; (3) existing prompt sanitization methods (Portcullis, PP-TS) target general text and cannot maintain SQL schema alignment.

**Key Challenge**: LLMs require schema information to generate correct SQL, yet the schema itself constitutes sensitive data.

**Key Insight**: What LLMs actually need for SQL generation is the **mapping relationship** between the question and the schema — the specific names are irrelevant and can be replaced with abstract symbols.

**Core Idea**: A three-stage pipeline — Abstraction (local SLM performs schema linking and symbol substitution) → SQL Generation (remote LLM processes the abstracted prompt) → SQL Reconstruction (local restoration and self-correction).

## Method

### Overall Architecture

Given a natural language question $\mathcal{Q}$ and database schema $\mathcal{S}$: local SLM performs schema ranking/filtering → SLM performs value/reference linking → abstract symbol substitution → remote LLM generates abstract SQL $\mathcal{Y}'$ → local restoration + SLM self-correction → executable SQL $\mathcal{Y}$.

### Key Designs

1. **Schema Ranking and Filtering**

    - **Function**: A RoBERTa cross-encoder ranks schema elements by relevance to the question, retaining top-$k$ tables and top-$j$ columns.
    - **Mechanism**: Reduces the schema footprint sent to the LLM, lowering noise and exposure surface.
    - **Design Motivation**: Real-world databases often contain hundreds of tables, most of which are irrelevant to a given query.

2. **Value and Reference Linking**

    - **Function**: A local SLM identifies tokens in the question that correspond to database values and schema elements.
    - **Mechanism**: Three steps — (1) SLM identifies value tokens (e.g., "New York Hospital"); (2) maps them to corresponding columns/tables; (3) identifies reference tokens (e.g., "patients" → Patients table).
    - **Design Motivation**: Accurate linking is critical to abstraction quality — missed tokens will leak sensitive information.

3. **Abstracting Concrete Tokens**

    - **Function**: A bijective mapping replaces table names with $T_i$, column names with $C_i$, and values with $V_i$, producing $\mathcal{Q}'$ and $\mathcal{S}'$.
    - **Mechanism**: Simple text substitution augmented with value-column correspondence annotations (e.g., "$V_1$ is a value of column $C_7$").
    - **Design Motivation**: The logical mapping between question and schema is preserved, allowing the LLM to understand query intent despite seeing only abstract symbols.

4. **SQL Reconstruction + Self-Correction**

    - **Function**: A symbol lookup table restores the abstract SQL to concrete form; a local SLM then performs error correction.
    - **Mechanism**: The restored SQL is executed; its result is provided alongside the original question to the SLM for final correction.
    - **Design Motivation**: Abstraction noise may introduce value type mismatches (e.g., string "positive" vs. numeric 1).

### Loss & Training

No training is required. A local SLM (Qwen-2.5-7B-Instruct) handles trusted-side inference, while a remote LLM (GPT-4.1) handles SQL generation.

## Key Experimental Results

### Main Results (BIRD Benchmark, 300 Complex Queries)

| Framework | Execution Accuracy | Token Usage | Privacy |
|------|----------|-----------|---------|
| Direct Prompting + Qwen-7B | 34.33% | 1,380 | ✓ Local |
| DAIL-SQL + Qwen-7B | 44.33% | 3,492 | ✓ Local |
| Fine-Tuned MSc-SQL | 48.33% | 8,342 | ✓ Local |
| DIN-SQL + GPT-4.1 | 65.66% | 24,812 | ✗ Exposed |
| **MaskSQL ($\Psi_C$)** | **62.00%** | ~5,000 | **✓ Partial Protection** |
| **MaskSQL ($\Psi_F$)** | **58.33%** | ~5,000 | **✓ Full Protection** |

### Ablation Study

| Component | Accuracy | Note |
|------|--------|------|
| w/o Schema Filtering | 52.0% | Excessive noise |
| w/o Self-Correction | 54.7% | Value type errors |
| w/o Value Linking | 50.3% | Incomplete abstraction |
| **Full MaskSQL** | **58.33%** | **Full protection** |

### Key Findings

- MaskSQL (full protection) outperforms all SLM-only approaches by 10+ percentage points and approaches unprotected LLM performance (7.3pp gap).
- Masking Recall exceeds 92%; Re-identification Score (adversary failure rate) exceeds 85%.
- The partial strategy $\Psi_C$, which protects only person names, locations, and occupations, achieves higher accuracy (62%), illustrating the privacy–utility trade-off.

## Highlights & Insights

- **Abstraction vs. Editing/Generalization**: Abstraction preserves mapping relationships rather than deleting or obfuscating information, making it the optimal privacy-preservation strategy for SQL tasks.
- **Hybrid Architecture**: The SLM handles trust-sensitive steps (schema linking, reconstruction) while the LLM handles reasoning (SQL generation), leveraging the strengths of each.
- **Configurable Privacy Policy**: Users can define $\Psi$ to select which elements to protect, enabling flexible adaptation to different regulatory requirements.

## Limitations & Future Work

- Schema linking quality depends on the local SLM; complex schemas may result in missed elements.
- Evaluation is limited to 300 complex queries from the BIRD benchmark, which is a relatively small scale.
- The adversary model is weak (considering only re-identification) and does not account for side-channel attacks.

## Related Work & Insights

- **vs. Portcullis**: Portcullis applies general-purpose NER-based sanitization and does not preserve SQL schema alignment.
- **vs. DIN-SQL**: DIN-SQL exposes the full schema; MaskSQL achieves comparable performance while protecting privacy.

## Rating
- Novelty: ⭐⭐⭐⭐ — The prompt abstraction approach is clear and practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual evaluation of both privacy and utility metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ — Precise problem formulation and well-structured pipeline description.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core privacy challenge in real-world LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[NeurIPS 2025\] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)
- [\[NeurIPS 2025\] Music Arena: Live Evaluation for Text-to-Music](music_arena_live_evaluation_for_text-to-music.md)

</div>

<!-- RELATED:END -->
