---
title: >-
  [Paper Note] TabReX: Tabular Referenceless eXplainable Evaluation
description: >-
  [ACL 2026][Interpretability][Tabular evaluation metrics] This paper proposes TabReX, a graph-reasoning-based referenceless evaluation framework for tabular generation. It converts source text and generated tables into kn…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Tabular evaluation metrics"
  - "referenceless evaluation"
  - "knowledge graph alignment"
  - "explainable evaluation"
  - "structured generation"
date: 2026-05-08
content_hash: f7e51c7c7a9e04b2
---

# TabReX: Tabular Referenceless eXplainable Evaluation

**Conference**: ACL 2026
**arXiv**: [2512.15907](https://arxiv.org/abs/2512.15907)
**Code**: [GitHub](https://github.com/TabReX)
**Area**: Interpretability
**Keywords**: Tabular evaluation metrics, referenceless evaluation, knowledge graph alignment, explainable evaluation, structured generation

## TL;DR

This paper proposes TabReX, a graph-reasoning-based referenceless evaluation framework for tabular generation. It converts source text and generated tables into knowledge graph triples and aligns them to compute interpretable, attribute-driven scores. TabReX substantially outperforms existing methods in correlation with human judgments, and the authors also introduce TabReX-Bench, a large-scale evaluation benchmark.

## Background & Motivation

**Background**: As LLMs are increasingly used to generate or transform structured outputs (e.g., converting reports into financial tables or synthesizing patient data), automatic evaluation of table quality has become a critical need. Existing evaluation metrics fall into several categories: n-gram metrics (BLEU, ROUGE), embedding-based metrics (BERTScore, BLEURT), token-level exact matching (Exact Match, PARENT), QA-based referenceless metrics (QuestEval), and recent LLM-judge metrics (TabEval, TabXEval).

**Limitations of Prior Work**: (1) N-gram and embedding metrics flatten tables into plain text, entirely ignoring row/column structure and unit semantics; (2) token-level methods cannot distinguish between harmless formatting adjustments and genuine factual errors; (3) QA-based metrics excessively penalize layout changes (e.g., row reordering); (4) most metrics require reference tables, limiting their generality; (5) existing benchmarks are small in scale and cover only limited perturbation types, preventing comprehensive robustness testing.

**Key Challenge**: Table evaluation must simultaneously account for structural fidelity and factual accuracy, while distinguishing data-preserving transformations (e.g., row reordering, unit conversion) from data-altering transformations (e.g., value tampering, row/column insertion or deletion). No existing metric performs well on both dimensions.

**Goal**: Design a referenceless, attribute-driven, explainable table evaluation framework that provides cell-level error traceability and a tunable sensitivity–specificity trade-off.

**Key Insight**: Table evaluation is reformulated as a graph alignment problem — both the source text and the generated table can be represented as knowledge graph triples [subject, predicate, object], and aligning these triples enables precise localization of matched, missing, and spurious information.

**Core Idea**: Text2Graph and Table2Graph are used to unify both modalities into a shared triple space. LLM-guided graph alignment identifies correspondences and discrepancies, and an attribute-driven scoring function computes interpretable evaluation scores.

## Method

### Overall Architecture

TabReX is a three-stage pipeline: (1) converting the source text and candidate table into knowledge graphs (Text2Graph + Table2Graph); (2) using LLM-guided graph alignment to identify correspondences between triples; and (3) computing attribute-driven structural and content scores from the alignment results. The final output includes a table-level score and cell-level error traceability.

### Key Designs

1. **Dual-Modality Knowledge Graph Conversion**:

    - Function: Unify text and tables into a comparable triple representation.
    - Mechanism: For text, an LLM extracts atomic factual triples $\mathcal{G}_S = \{(s_i, p_i, o_i)\}$ following entity-centric syntax, enforcing consistent granularity, normalized predicates, and unit-aware values. For tables, lightweight deterministic rules generate triples using column headers as predicates, row identifiers as subjects, and cell values as objects.
    - Design Motivation: Unifying both modalities into the same representation space reduces evaluation to a graph alignment problem, eliminating bias introduced by modality differences. The deterministic rule-based approach on the table side (without LLM) ensures speed and consistency.

2. **LLM-Guided Graph Alignment**:

    - Function: Precisely match triples from the source text and the generated table.
    - Mechanism: A two-step alignment procedure — (1) Deterministic matching: triples with identical subject-predicate pairs or those equivalent after schema normalization are directly aligned; (2) LLM-assisted refinement: handles remaining cases involving paraphrasing, abbreviations, and compound attributes (e.g., "GDP growth (YoY)" ↔ "growth_rate_2021"). Each matched pair is annotated with a difference vector $\Delta$ recording unit-aware numerical deviation, categorical mismatch, and missing/spurious tokens.
    - Design Motivation: Deterministic matching handles simple cases efficiently, while the LLM addresses difficult cases requiring semantic understanding, balancing efficiency and accuracy.

3. **Attribute-Driven Scoring**:

    - Function: Compute interpretable and tunable evaluation scores from alignment results.
    - Mechanism: Two components are defined — TablePenalty computes the normalized proportion of row/column-level missing (MI) and extraneous (EI) entities; CellPenalty computes cell-level missing, extraneous, and partial matches (numerical deviation $\Gamma$). The final score is $\mathcal{S}_{\text{TabReX}} = \text{TablePenalty} + \text{CellPenalty}$. Weight parameters $(\alpha, \beta)$ provide a tunable sensitivity–specificity trade-off: increasing $\beta_{\text{MI}}$ biases toward sensitivity (rewarding comprehensive coverage), while increasing $\beta_{\text{EI}}$ biases toward specificity (penalizing hallucinations).
    - Design Motivation: Different domains have different error tolerance (financial applications require precision; clinical applications require recall), and the tunable weights allow the same framework to adapt to diverse requirements.

### Loss & Training

TabReX requires no training and operates purely as an inference-time evaluation framework. LLMs are only invoked in the Text2Graph and graph alignment steps; the scoring function is entirely deterministic.

## Key Experimental Results

### Main Results

Correlation with human rankings (Table 2):

| Metric Category | Method | Spearman ρ (↑) | Kendall τ (↑) | Tie ratio (↓) |
|----------------|--------|---------------|-------------|-------------|
| Non-LLM (reference-based) | EM | 45.88 | 39.38 | 58.40 |
| Non-LLM (reference-based) | BERTScore | 36.21 | 30.66 | 0.92 |
| LLM (reference-based) | TabXEval | **80.27** | **72.37** | 45.33 |
| Referenceless | QuestEval | 62.93 | 52.29 | 3.03 |
| Referenceless | **TabReX** | **74.51** | **64.24** | **13.59** |

Under referenceless conditions, TabReX approaches the correlation of the strongest reference-based method TabXEval, while achieving a substantially lower tie ratio (13.6% vs. 45.3%).

### Ablation Study

| Ensemble Method | Spearman ρ | Kendall τ | Note |
|----------------|-----------|----------|------|
| Lex-Emb (Mean) | 38.43 | 32.65 | Lexical + embedding ensemble |
| LLM (Harmonic) | 56.00 | 46.93 | LLM metric ensemble |
| Hybrid (Harmonic) | 54.03 | 42.71 | Hybrid ensemble |
| **TabReX** | **74.51** | **64.24** | Single method |

### Key Findings

- TabReX as a single metric outperforms all ensemble methods, demonstrating that the graph alignment paradigm itself is more effective than simple aggregation.
- From easy to hard perturbations, TabReX maintains a stable sensitivity–specificity trade-off (small shift in the operating point), whereas EM, H-Score, and others degrade substantially.
- Although TabXEval achieves the highest correlation, its tie ratio of 45.3% means nearly half of distinct variants receive identical scores — indicating insufficient discriminative precision.
- TabReX-Bench (710 tables × 12 perturbations = 9,120 instances, 6 domains, 3 difficulty levels) constitutes the largest tabular evaluation benchmark to date.

## Highlights & Insights

- The use of **knowledge graph triples as an intermediate representation** is an elegant design choice — it reduces the modality alignment problem to a graph matching problem and naturally supports dual evaluation of structure and semantics.
- The **practical value of the tunable trade-off** is significant — financial domains can increase $\beta_{\text{EI}}$ to strictly penalize hallucinations, while clinical domains can increase $\beta_{\text{MI}}$ to ensure information completeness.
- **Planner-driven perturbation generation** ensures benchmark diversity and reproducibility — generating 12 perturbation types in a single LLM call yields more consistent results than generating them individually.

## Limitations & Future Work

- Text2Graph relies on LLMs for triple extraction, which may lack robustness for complex nested tables or non-standard formats.
- Evaluation costs depend on the number of LLM calls, making latency and cost non-negligible at scale.
- Only GPT-5-nano is evaluated as the backbone; performance with open-source models remains to be verified.
- Cross-table reasoning (requiring facts jointly derived from multiple tables) is not yet covered.

## Related Work & Insights

- **vs. TabXEval**: TabXEval is reference-based and achieves the highest correlation, but its high tie ratio leads to insufficient discriminative precision; TabReX is referenceless and provides finer-grained discrimination.
- **vs. QuestEval**: Both are referenceless methods, but QuestEval relies on generic QA signals and excessively penalizes table-specific structural transformations (e.g., row reordering); TabReX's graph alignment naturally abstracts away the effects of formatting changes.
- **vs. PARENT/BLEU**: These metrics are largely ineffective for structured output evaluation; TabReX represents a fundamental paradigm shift in this evaluation landscape.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The graph alignment paradigm for referenceless table evaluation is a fundamentally novel approach, and the attribute-driven scoring mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ TabReX-Bench is large-scale and rigorously designed, baselines are comprehensive, and human evaluation is thorough.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly, though the density of mathematical notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ The work makes an important contribution to the evaluation of structured generation, and the framework design is general and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[ICLR 2026\] STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings](../../ICLR2026/interpretability/stride_subset-free_functional_decomposition_for_xai_in_tabular_settings.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)

</div>

<!-- RELATED:END -->
