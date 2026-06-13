---
title: >-
  [Paper Note] TabReX: Tabular Referenceless eXplainable Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Tabular Evaluation Metrics] Ours proposes TabReX, a referenceless tabular generation evaluation framework based on graph reasoning. It transforms source text and generated tables into knowledge…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tabular Evaluation Metrics"
  - "Referenceless Evaluation"
  - "Knowledge Graph Alignment"
  - "Explainable Evaluation"
  - "Structured Generation"
date: 2026-05-08
content_hash: 7041471db0edf5e4
---

# TabReX: Tabular Referenceless eXplainable Evaluation

**Conference**: ACL 2026  
**arXiv**: [2512.15907](https://arxiv.org/abs/2512.15907)  
**Code**: [GitHub](https://github.com/TabReX)  
**Area**: Interpretability  
**Keywords**: Tabular Evaluation Metrics, Referenceless Evaluation, Knowledge Graph Alignment, Explainable Evaluation, Structured Generation

## TL;DR

Ours proposes TabReX, a referenceless tabular generation evaluation framework based on graph reasoning. It transforms source text and generated tables into knowledge graph triples and aligns them to calculate explainable attribute-driven scores. It significantly outperforms existing methods in correlation with human judgment and introduces TabReX-Bench, a large-scale benchmark.

## Background & Motivation

**Background**: As LLMs are increasingly utilized to generate or transform structured outputs (e.g., converting reports into financial tables, synthesizing patient data), automatic evaluation of table quality has become a critical requirement. Existing evaluation metrics mainly include: n-gram metrics (BLEU, ROUGE), embedding metrics (BERTScore, BLEURT), token-level exact matching (Exact Match, PARENT), QA-based referenceless metrics (QuestEval), and recent LLM-as-a-judge metrics (TabEval, TabXEval).

**Limitations of Prior Work**: (1) N-gram and embedding metrics flatten tables into text, completely ignoring row-column structures and unit semantics; (2) Token-level methods fail to distinguish between harmless formatting adjustments and genuine factual errors; (3) QA metrics over-penalize layout changes (e.g., row reordering); (4) Most metrics require a reference table, limiting generalizability; (5) Existing benchmarks are small in scale with single perturbation types, failing to comprehensively test metric robustness.

**Key Challenge**: Tabular evaluation must simultaneously consider structural fidelity and factual accuracy while distinguishing between data-preserving transformations (e.g., row reordering, unit conversion) and data-modifying transformations (e.g., numerical tampering, row/column addition/deletion). Existing metrics fail to perform well across both dimensions simultaneously.

**Goal**: Design a referenceless, attribute-driven, and explainable tabular evaluation framework capable of providing cell-level error tracing and an adjustable sensitivity-specificity trade-off.

**Key Insight**: Convert tabular evaluation into a graph alignment problem—both source text and generated tables can be represented as knowledge graph triples $[subject, predicate, object]$. Aligning these triples allows for precise localization of matching, missing, and redundant information.

**Core Idea**: Use Text2Graph and Table2Graph to unify both modalities into a triple space, identify correspondences and differences via LLM-guided graph alignment, and then compute explainable scores using an attribute-driven scoring function.

## Method

### Overall Architecture

TabReX is a three-stage pipeline: (1) Transforming source text and candidate tables into knowledge graphs (Text2Graph + Table2Graph); (2) Identifying correspondences between triples via LLM-guided graph alignment; (3) Calculating attribute-driven structural and content scores from the alignment results. The final output includes table-level scores and cell-level error tracing.

### Key Designs

1.  **Dual-modal Knowledge Graph Conversion**:
    *   **Function**: Unifies text and tables into comparable triple representations.
    *   **Mechanism**: For text, an LLM extracts atomic factual triples $\mathcal{G}_S = \{(s_i, p_i, o_i)\}$ using entity-centric syntax, enforcing consistent granularity, normalized predicates, and unit-aware values. For tables, a lightweight rule-based approach deterministically generates triples by using headers as predicates, row identifiers as subjects, and cell values as objects.
    *   **Design Motivation**: Unifying both modalities into the same representation space transforms evaluation into a graph alignment problem, eliminating biases caused by modal differences. Deterministic rules (without LLM) on the table side ensure speed and consistency.

2.  **LLM-guided Graph Alignment**:
    *   **Function**: Precisely matches triples from text and tables.
    *   **Mechanism**: Two-step alignment—(1) Deterministic matching: Triples with identical subject-predicate pairs or identical pairs after schema normalization are aligned directly. (2) LLM-assisted refinement: Handles remaining paraphrases, abbreviations, and composite attributes (e.g., "GDP growth (YoY)" $\leftrightarrow$ "growth_rate_2021"). Each matched pair is labeled with a difference vector $\Delta$, recording unit-aware numerical differences, category mismatches, and missing/redundant markers.
    *   **Design Motivation**: Deterministic matching handles simple cases (fast), while the LLM handles difficult cases requiring semantic understanding, balancing efficiency and accuracy.

3.  **Attribute-driven Scoring**:
    *   **Function**: Calculates explainable and adjustable evaluation scores from alignment results.
    *   **Mechanism**: Composed of two components—TablePenalty calculates the normalized ratio of missing (MI) and extra (EI) entities at the row/column level; CellPenalty calculates cell-level missing/extra/partial matches (numerical deviation $\Gamma$). The final score is $\mathcal{S}_{\text{TabReX}} = \text{TablePenalty} + \text{CellPenalty}$. Weight parameters $(\alpha, \beta)$ provide a tunable sensitivity-specificity trade-off: increasing $\beta_{\text{MI}}$ favors sensitivity (rewarding comprehensive coverage), while increasing $\beta_{\text{EI}}$ favors specificity (penalizing hallucinations).
    *   **Design Motivation**: Different domains have varying error tolerances (precision for finance, recall for clinical settings); adjustable weights allow the same framework to adapt to diverse requirements.

### Loss & Training

TabReX is a pure inference-time evaluation framework and requires no training. LLMs are only used in the Text2Graph and graph alignment steps, while the scoring function is entirely deterministic.

## Key Experimental Results

### Main Results

Correlation comparison with human rankings (Table 2):

| Metric Category | Method | Spearman $\rho$ (↑) | Kendall $\tau$ (↑) | Tie ratio (↓) |
| :--- | :--- | :--- | :--- | :--- |
| Non-LLM (Reference) | EM | 45.88 | 39.38 | 58.40 |
| Non-LLM (Reference) | BERTScore | 36.21 | 30.66 | 0.92 |
| LLM (Reference) | TabXEval | **80.27** | **72.37** | 45.33 |
| Referenceless | QuestEval | 62.93 | 52.29 | 3.03 |
| Referenceless | **TabReX** | **74.51** | **64.24** | **13.59** |

TabReX approaches the correlation of the strongest reference-based method (TabXEval) under referenceless conditions, with a significantly lower tie ratio (13.6% vs 45.3%).

### Ablation Study

| Ensemble Method | Spearman $\rho$ | Kendall $\tau$ | Description |
| :--- | :--- | :--- | :--- |
| Lex-Emb (Mean) | 38.43 | 32.65 | Lexical + Embedding Ensemble |
| LLM (Harmonic) | 56.00 | 46.93 | LLM Metric Ensemble |
| Hybrid (Harmonic) | 54.03 | 42.71 | Hybrid Ensemble |
| **TabReX** | **74.51** | **64.24** | Single Method |

### Key Findings

*   The single TabReX metric outperforms all ensemble methods, suggesting that the graph alignment paradigm is inherently more effective than simple aggregation.
*   Across easy to hard perturbations, the sensitivity-specificity trade-off of TabReX remains stable (minimal arrow shifts), whereas EM and H-Score degrade significantly.
*   Although TabXEval has the highest correlation, its tie ratio reaches 45.3%, implying nearly half of the different variants are assigned identical scores—indicating insufficient discriminative precision.
*   TabReX-Bench (710 tables $\times$ 12 perturbations = 9120 instances, 6 domains, 3 difficulty levels) is currently the largest tabular evaluation benchmark.

## Highlights & Insights

*   **Knowledge Graph Triples as Intermediate Representation**: This design is elegant—simplifying modal alignment into a graph matching problem, naturally supporting dual evaluation of structure and semantics.
*   **Practical Value of Tunable Trade-offs**: Prominent utility—financial domains can increase $\beta_{\text{EI}}$ to strictly penalize hallucinations, while clinical domains can increase $\beta_{\text{MI}}$ to ensure information completeness.
*   **Planner-driven Perturbation Generation**: Ensures benchmark diversity and reproducibility—a single LLM call generates 12 types of perturbations, providing more consistency than sequential generation.

## Limitations & Future Work

*   Text2Graph relies on LLMs for triple extraction, which may not be robust enough for complex nested tables or non-standard formats.
*   Evaluation cost depends on the number of LLM calls; cost and latency are non-negligible for large-scale usage.
*   Evaluations were only conducted with GPT-5-nano as the backbone; effectiveness with open-source models remains to be verified.
*   Cross-table reasoning (requiring joint facts from multiple tables) is not yet covered.

## Related Work & Insights

*   **vs TabXEval**: TabXEval requires a reference and has the highest correlation, but its high tie ratio leads to poor discriminative precision; TabReX is referenceless and provides finer-grained discrimination.
*   **vs QuestEval**: Both are referenceless, but QuestEval relies on general QA signals and over-penalizes table-specific structural transformations (e.g., row reordering); TabReX's graph alignment naturally excludes the impact of formatting changes.
*   **vs PARENT/BLEU**: These metrics are almost ineffective in structured output evaluation; TabReX represents a fundamental shift in the evaluation paradigm.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The referenceless tabular evaluation paradigm using graph alignment is a fresh perspective; the attribute-driven scoring mechanism is sophisticatedly designed.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ TabReX-Bench is large-scale and rigorously designed, with comprehensive baseline comparisons and sufficient human evaluation.
*   Writing Quality: ⭐⭐⭐⭐ The framework description is clear, though the numerous formulas require careful reading.
*   Value: ⭐⭐⭐⭐⭐ Provides a significant push for the field of structured generation evaluation; the framework design is general and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](../../ICLR2026/llm_evaluation/human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/llm_evaluation/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)

</div>

<!-- RELATED:END -->
