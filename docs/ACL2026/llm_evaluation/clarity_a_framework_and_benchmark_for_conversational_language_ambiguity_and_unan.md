---
title: >-
  [Paper Note] CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems
description: >-
  [ACL 2026][LLM Evaluation][NL2SQL Evaluation] CLARITY is the first NL2SQL diagnostic benchmark proposed by Oracle that supports "**multi-facet ambiguity + unanswerability** + single/multi-turn conversation + diverse user…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "NL2SQL Evaluation"
  - "Multi-facet Ambiguity"
  - "Unanswerability"
  - "Schema Grounding"
  - "Conversational Clarification"
date: 2026-05-08
content_hash: d4fc3c954697fe26
---

# CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems

**Conference**: ACL 2026  
**arXiv**: [2604.22313](https://arxiv.org/abs/2604.22313)  
**Code**: None (Oracle internal framework)  
**Area**: LLM Evaluation / Conversational NL2SQL / Ambiguity Detection  
**Keywords**: NL2SQL Evaluation, Multi-facet Ambiguity, Unanswerability, Schema Grounding, Conversational Clarification

## TL;DR
CLARITY is the first NL2SQL diagnostic benchmark proposed by Oracle that supports "**multi-facet ambiguity + unanswerability** + single/multi-turn conversation + diverse user clarification behaviors." It extends Spider/BIRD into approximately 30,000 instances through a controllable LLM pipeline (SQL → pivot term → rewriting → conversation → screening). Using schema-level pivot/group annotations, it reveals a failure mode where SOTA LLMs "can detect ambiguity but fail to locate specific schema elements."

## Background & Motivation

**Background**: NL2SQL has been deployed in industrial-grade query interfaces (Oracle, Snowflake, etc.), but user queries are frequently ambiguous or unanswerable. Benchmarks like AMBROSIA, AmbiQT, NoisySP, and SQUAB only cover single-type ambiguity and single-turn interactions. While PRACTIQ, MMSQL, and BIRD-INTERACT extend to multi-turn, they assume cooperative users with clean clarifications and only provide instance-level labels (binary ambiguous/unanswerable).

**Limitations of Prior Work**: ใน real-world production environments: (1) a single query often contains **multiple interacting ambiguity sources** (involving several columns and values simultaneously); (2) user clarification responses are frequently **partially helpful, vague, or irrelevant**; (3) even when systems correctly detect "ambiguity," they often **mislocate the specific schema elements**, a schema-level failure masked by current benchmarks that only measure detection rates.

**Key Challenge**: There is a significant gap between the real distribution of ambiguity/unanswerability (multi-faceted, multi-turn, user-noisy) and existing evaluation assumptions (uni-faceted, single-turn, cooperative-user). This gap allows NL2SQL systems with "high ambiguity detection rates" to fail frequently in industrial deployments without identifiable causes.

**Goal**: To construct a diagnostic NL2SQL benchmark that: (1) covers four modes (column/value ambiguity and unanswerability); (2) supports uni-/multi-facet scenarios; (3) covers three types of user behavior (helpful/partial/unhelpful); (4) provides fine-grained metadata including span-level pivot terms and candidate schema groups, moving evaluation beyond "knowing there is a problem" to "precisely pointing to the problem."

**Key Insight**: Reverse engineering from executable SQL. Since the column/value reference structure of SQL explicitly defines the "true intent," rule-based parsing combined with LLM rewriting can intentionally distort it into ambiguous or unanswerable versions. This constraint-driven generation ensures both ground-truth integrity and control over ambiguity patterns.

**Core Idea**: Utilizing a modular pipeline—"SQL Parser → Adaptive Schema Retriever → Pivot Term Generator/Evaluator → NL Query Generator/Evaluator → Conversation Generator → Automated Screening"—to automatically extend existing clean NL2SQL data into schema-grounded ambiguous/unanswerable multi-turn interaction instances.

## Method

### Overall Architecture
CLARITY is an end-to-end generation framework. For a triplet $(u, q, \mathcal{S})$ (NL query / SQL / schema), a specified mode $m$, user response type $R$, and target count $K$, it outputs a conversation $C=\{(r_t, t_t)\}_{t=1}^T$ and a verification flag $v$:

- **Inputs**: (NL query $u$, SQL $q$, schema $\mathcal{S}$, A/U mode $m \in M$, response type $R$, multi-facet count $K$)
- **Pipeline Steps**:
    - Step 1: SQL Parser parses $q$ to extract columns/values and samples $K$ A/U targets.
    - Step 2: Adaptive Schema Retriever constructs the target space $\mathcal{X}_j$ and candidate group $G_j$ for each target.
    - Step 3: Pivot Generator/Evaluator generates and validates pivot terms via multi-LLM consensus.
    - Step 4: NL Query Generator/Evaluator injects pivots into $u$ to rewrite it as an A/U query.
    - Step 5: Conversation Generator extends the query into multi-turn dialogue based on response type.
    - Step 6: Automated Data Screening performs final filtering via majority voting.
- **Outputs**: ~6.4k Spider + ~3.5k BIRD single-turn instances; ~12.1k Spider + ~7.2k BIRD multi-turn instances, each labeled with pivot terms and target schema groups.

The mode set is $M = \{\texttt{col\_amb}, \texttt{val\_amb}, \texttt{col\_unans}, \texttt{val\_unans}\}$, where $K=1$ denotes uni-facet and $K\ge 2$ denotes multi-facet scenarios.

### Key Designs

1.  **Constraint-driven SQL → Ambiguity Reverse Engineering**:
    - **Function**: Ensures generated ambiguous/unanswerable queries have clean ground truth derived from executable SQL, where each pivot strictly follows the formal definition of "mapping to multiple schema elements (amb)" or "mapping to none (unans)."
    - **Mechanism**: The SQL Parser resolves $q$ into referenced columns $\mathcal{C}$ and column-value pairs $(\mathcal{C}, \mathcal{V})$, sampling $K$ targets $\mathcal{T} = \{t_j\}$ without replacement. The Adaptive Schema Retriever constructs candidate group $G_j$ using lexical (token overlap), semantic (sentence-embedding cosine similarity), or hybrid methods. A two-stage filter separates ambiguity types by first identifying lexical overlaps then retrieving semantic matches under lexical constraints. The Pivot Generator synthesizes pivots given $(\mathcal{X}_j, G_j)$, while the Pivot Evaluator verifies boolean constraints (mode compliance, $G_j$ compatibility, schema absence).
    - **Design Motivation**: Free LLM generation introduces distributional bias; pure template methods lack diversity. Reverse engineering ensures ground truth precision while LLM generation under strict constraints provides natural variety. Human validation shows 93–100% accuracy across most categories.

2.  **Multi-facet and Three Types of User Response Modeling**:
    - **Function**: Transitions from "uni-facet + cooperative-user" to industrial reality, where multiple ambiguities coexist and user answers may be vague.
    - **Mechanism**: (a) **Multi-facet**: By setting $K \ge 2$, a query contains multiple A/U instances simultaneously. (b) **Three Conversation Types** $R \in \{\texttt{helpful}, \texttt{partial}, \texttt{unhelpful}\}$: The generator simulates an agent asking clarification questions for pivot $p_j$. A "helpful" user provides complete information, "partial" provides some clues, and "unhelpful" provides irrelevant or vague responses. 
    - **Design Motivation**: In production, users rarely provide perfect clarifications. By modeling these behaviors, CLARITY quantifies the performance gap between cooperative and noisy environments.

3.  **Schema-grounded Fine-grained Annotation + Automated Data Screening**:
    - **Function**: Upgrades evaluation from instance-level ("Is it ambiguous?") to schema-level ("Which columns/values are ambiguous?").
    - **Mechanism**: Each instance includes (i) span-level pivot terms $\{p_j\}$ and (ii) candidate schema target groups $\{G_j\}$. Beyond Detection Accuracy (DA), the Match Accuracy (MA) metric evaluates whether detected ambiguity corresponds to correct schema elements. 
    - **Design Motivation**: Experiments reveal that while LLM DA is near ~100%, MA drops significantly in multi-facet scenarios (e.g., 5.4% on BIRD M-Col Amb). Without schema-grounding, this critical localization failure remains hidden.

## Key Experimental Results

### Main Results (GPT-5 Single-turn Column Ambiguity, Spider Few-shot)

| Setting | Uni LEM | Uni SEM | Multi LEM | Multi SEM |
|---------|---------|---------|-----------|-----------|
| Zero-shot | 19.2 | 15.0 | 15.2 | 5.2 |
| Few-shot no-meta (uni examples) | 56.9 | 42.9 | 55.3 | 13.4 |
| Few-shot no-meta (uni+multi examples) | 62.7 | 50.3 | 61.4 | 23.4 |
| Few-shot meta (uni examples) | 60.5 | 53.9 | 67.0 | 60.2 |
| **Few-shot meta (uni+multi examples)** | **61.3** | **55.8** | **70.1** | **60.1** |

SEM (Strict Exact Match) measures if all correct SQL interpretations are enumerated; LEM (Lenient Exact Match) requires at least one. On multi-facet tasks, SEM jumps from 5.2% (zero-shot) to 60.1% with metadata, a **+55 point gain**.

### Multi-turn SQL Prediction (Spider, EA %)

| Model | U-Lex Amb | M-Lex Amb | U-Sem Amb | M-Sem Amb | U-Col Unans | M-Col Unans | Concise | Verbose | Partial | Not |
|-------|-----------|-----------|-----------|-----------|-------------|-------------|---------|---------|---------|-----|
| GPT-4o | 74.2 | 73.2 | 74.2 | 67.2 | 99.0 | 100.0 | 73.1 | 80.5 | 74.6 | 62.6 |
| GPT-4.1 | 75.2 | 73.6 | 81.0 | 73.2 | 95.0 | 99.0 | 68.3 | 82.2 | 78.1 | 71.7 |
| GPT-5 | 73.0 | 71.6 | 77.6 | 69.4 | 83.0 | 93.0 | 62.9 | 79.3 | 77.2 | 70.0 |

### Key Findings
- **Detection $\neq$ Localization**: DA is consistently 99–100%, but MA for multi-facet column ambiguity is only 5.4–20.4%. Systems "sense" ambiguity but **cannot pinpoint which specific columns** are ambiguous.
- **Multi-facet Difficulty**: Multi-facet SEM improvement from 5.2% to 60.1% highlights a heavy reliance on teaching signals compared to uni-facet tasks.
- **Unanswerable Cases are Easier**: EA for unanswerable cases is typically 90%+, as models only need to output null, whereas ambiguous cases require resolving complex resource contention.
- **Conversation Quality Matters**: Performance follows the trend: Verbose > Partial > Not. The interaction structure impacts performance more than the model's inherent reasoning capacity.

## Highlights & Insights
- **Defining the "Detection-Localization Gap"**: Introducing the MA metric reveals a fundamental flaw in SOTA models that instance-level benchmarks miss, which is critical for industrial deployment transparency.
- **Constraint-driven Reverse Engineering**: Methodologically more controllable than forward synthesis and more natural than template-based methods. This 6-module pipeline is adaptable to other "clean-to-noisy" synthesis tasks like NL2API or NL2Code.
- **Multi-judge Feedback Loops**: Enforcing consensus with feedback for regeneration ensures high-quality large-scale benchmark synthesis.
- **User Cooperation Modeling**: Explicitly treats "user clarification ability" as an evaluation variable, providing a diagnostic coordinate for robust conversational agents.

## Limitations & Future Work
- **Limitations**: Value-based evaluation is limited due to DB exposure constraints. Synthesis relies on LLMs, which may introduce systematic bias (e.g., human validation for multi val_amb is 81.2%). Real-world patterns like topic shifts are not yet covered.
- **Future Work**: Incorporating industrial query logs for better user behavior alignment; extending the pipeline to NL2Code/API; and fine-tuning NL2SQL models specifically on schema-level localization using CLARITY data.

## Related Work & Insights
- **Comparison to AMBROSIA/AmbiQT**: CLARITY provides superior dimensionality by adding multi-facet and multi-turn schema-grounded annotations.
- **Comparison to PRACTIQ/BIRD-INTERACT**: CLARITY's pivot/group metadata allows for localization evaluation, moving beyond simple binary instance-level labels.
- **Synergy with AmbiSQL**: Using CLARITY as few-shot exemplars significantly improved AmbiSQL's MA (e.g., 50.1 to 57.7 on BIRD), proving it serves as both an evaluation benchmark and a high-quality training resource.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)
- [\[ACL 2026\] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->
