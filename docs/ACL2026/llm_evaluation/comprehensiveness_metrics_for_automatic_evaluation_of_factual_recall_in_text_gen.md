---
title: >-
  [Paper Note] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation
description: >-
  [ACL 2026][LLM Evaluation][Comprehensive Evaluation] Addressing the difficulty of quantifying "omitted key information" in long-form generation…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Comprehensive Evaluation"
  - "Factual Recall"
  - "NLI Graph"
  - "Q&A Comparison"
  - "End-to-End LLM Evaluation"
date: 2026-05-08
content_hash: 3c091d8bb71a9ca5
---

# Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation

**Conference**: ACL 2026  
**arXiv**: [2510.07926](https://arxiv.org/abs/2510.07926)  
**Code**: None provided (N/A)  
**Area**: LLM Evaluation / Factuality / Information Coverage  
**Keywords**: Comprehensive Evaluation, Factual Recall, NLI Graph, Q&A Comparison, End-to-End LLM Evaluation

## TL;DR
Addressing the difficulty of quantifying "omitted key information" in long-form generation, the authors propose three comprehensiveness metrics—NLI decomposition + graph analysis, QA comparison, and end-to-end (E2E) LLM identification. These metrics calculate coverage $S = |\mathcal{A}_{in}| / (|\mathcal{A}_{in}| + |\mathcal{A}_{out}|)$ based on a reference corpus $\mathcal{C}$. Meta-evaluation on WikiContradict and ConflictBank reveals that the simplest E2E method is strongest on average (best LMR=0.85), but Q&A offers better robustness (cross-model std of only 0.009 vs 0.044 for E2E), with each suitable for different scenarios.

## Background & Motivation
**Background**: LLM factuality evaluation primarily focuses on precision—tools like FActScore and FacTool decompose responses into atomic facts for individual verification. Rare recall-focused works (such as SAFE's $R_K(y) = \min(S(y)/K, 1)$) rely on a preset number of "required facts" $K$, failing to locate specific missing content.

**Limitations of Prior Work**: (1) Precision-only metrics cannot identify "half-truths" where LLMs selectively answer or deliberately omit critical information, which can be as fatal as hallucinations in high-risk scenarios like medical or legal domains. (2) Existing recall metrics use coarse-grained ratios and cannot pinpoint "which specific fact was missed," hindering diagnosis or real-time feedback. (3) There is a lack of specialized meta-benchmarks for evaluating comprehensiveness evaluators themselves.

**Key Challenge**: To identify "what was missed," one theoretically needs to know the "complete set of atomic facts that should be included"—yet a complete fact set is non-exhaustive, as any response could involve infinite related facts.

**Goal**: (1) Transform the comprehensiveness evaluation problem into "coverage relative to a reference corpus $\mathcal{C}$" to make evaluation operational. (2) Design three metrics of varying granularities and meta-evaluate their reliability. (3) Measure the comprehensiveness of mainstream open-source LLMs in real-world RAG scenarios.

**Key Insight**: Using modern retrievers and search engines to create a reference corpus is sufficiently effective; the unattainability of "absolute completeness" does not affect the utility of "relative coverage." Evaluators can employ NLI, QA, or direct LLMs as internal mechanisms for different granularities.

**Core Idea**: Define comprehensiveness as the coverage of the response atomic fact set $\mathcal{A}_R$ over relevant items in the reference corpus atomic fact set $\mathcal{A}_\mathcal{C}$, implemented through three pipelines of varying complexity.

## Method

### Overall Architecture
Input: User prompt $P$, candidate response $R$, reference corpus $\mathcal{C}$. Output: Covered set $\mathcal{A}_{in}$, uncovered set $\mathcal{A}_{out}$, and a scalar score $S = |\mathcal{A}_{in}| / (|\mathcal{A}_{in}| + |\mathcal{A}_{out}|)$. The NLI and Q&A methods also produce a fact graph $G_F$ (where nodes are atomic statements/QA pairs and edges represent entailment), supporting a fine-grained "uncovered context basis" $\mathcal{A}_{basis}$ (the minimal set required for completion). All three metrics share a uniform interface but differ significantly in internal pipeline complexity.

### Key Designs

1.  **NLI-based comprehensiveness (Multi-stage + Graph Analysis)**:

    - **Function**: Decomposes the response and corpus into atomic statements, using NLI to identify which corpus statements are entailed by the response.
    - **Mechanism**: A 5-stage pipeline—(a) **Atom Extraction**: Uses LLM few-shot to split $R$ and each document in $\mathcal{C}$ into atomic facts; (b) **Atom Revision**: Resolves pronouns and splits compound sentences ("A wrote X and Y" → "A wrote X" + "A wrote Y"); (c) **Relevance Filtering**: Estimates a relevance score (1-5) for each context atom, filtering with $T_{rel} = 3.5$; (d) **NLI Relation Extraction**: Uses general LLMs instead of specialized NLI models to judge entailment between response↔context and context↔context (a total of $2|\mathcal{A}_R||\mathcal{A}_\mathcal{C}| + |\mathcal{A}_\mathcal{C}|(|\mathcal{A}_\mathcal{C}|-1)$ pairs); (e) **Score Calculation**: Constructs a fact graph $G_F$, performs Strongly Connected Component (SCC) compression to get $G_C$, and defines $\mathcal{A}_{in} = \{A_i \in \mathcal{V}_C \cap \mathcal{A}_\mathcal{C} | \exists A_j \in \mathcal{A}_R, \text{path}(A_j, A_i) \in G_F\}$.
    - **Design Motivation**: Graph structures naturally handle the transitive closure of entailment (e.g., "X entails Y, Y entails Z"), avoiding redundant counting. General LLMs are used because NLI models struggle with regulatory or knowledge-intensive statements. The trade-off is $O(|\mathcal{A}|^2)$ computational cost and a lack of context during the atom comparison phase.

2.  **Q&A-based comprehensiveness (Question Mediation + Answer Comparison)**:

    - **Function**: Replaces atomic statement comparison with "asking questions + comparing answers," leveraging question alignment for natural comparability.
    - **Mechanism**: A 4-stage process—(a) **Question Mining**: Extracts self-contained factual questions from $R$ and $\mathcal{C}$ separately; (b) **Question Refinement**: De-duplicates, generalizes, filters ambiguous questions, and assigns relevance scores; (c) **Answer Generation**: Answers all refined questions using each source; each question may have multiple or no answers, with a confidence score filtered at $T_{conf} = 2$; (d) **Answer Comparison**: Uses LLM + Pint tool (to handle physical units) to classify relationships as "equivalent / first implies second / contradictory / neutral," then converts to entailment to construct a fact graph and calculate $S$.
    - **Design Motivation**: NLI comparison requires iterating over all atom pairs ($O(N^2)$), while Q&A only compares answers under the same question ($O(M \cdot k)$), improving efficiency. Generating answers allows the model to see the full context, making it more accurate than isolated NLI atom comparisons. The Pint tool fixes LLM weaknesses in dimensional analysis.

3.  **End-to-End comprehensiveness (Single LLM Call)**:

    - **Function**: Directly prompts the LLM to output $\mathcal{A}_{in}$ and $\mathcal{A}_{out}$ within a single $(P, \mathcal{C}, R)$ context.
    - **Mechanism**: A single LLM call $(\mathcal{A}_{in}, \mathcal{A}_{out}) = \text{CoverageEvaluator}(P, \mathcal{C}, R)$, with scores calculated as before. No fact graphs, intermediate steps, or pairwise relation classification.
    - **Design Motivation**: The long-context reasoning of modern LLMs (e.g., Llama 4 17Bx128E) is powerful enough to list uncovered portions directly after seeing the original text and the answer. This eliminates the cascading errors of a pipeline but sacrifices interpretability and stability (results depend heavily on the evaluator LLM choice).

### Loss & Training
This work does not train any models. All LLM calls use existing open-source models (gpt-oss-20b/120b, Llama 3.3 70B, Llama 4 17Bx128E, Qwen 2.5 72B) with temperature 0 and top-p 1. Llama 4 uses FP8 quantization. The Pint tool is only used with Llama/Qwen (as the gpt-oss engine does not support tool calls).

## Key Experimental Results

### Meta-evaluation Results (Label Match Rate, LMR↑)

| Metric × LLM | WikiContradict LMR | ConflictBank LMR | Average |
|--------------|-------------------|------------------|------|
| E2E × Llama 4 17Bx128E | — | — | **0.85** (Best) |
| Q&A × gpt-oss-20b | — | — | **0.81** (Best Stability) |
| NLI (All LLMs) | Sig. lower than Q&A/E2E | Same | — |
| E2E × gpt-oss-120b on ConflictBank | — | Sig. lower than Q&A | — |

Cross-model stability: The Q&A method's LMR std across 5 LLMs is only **0.009**, while E2E's std reaches **0.044** (5× more sensitive to evaluator LLM selection). On WikiContradict, E2E significantly outperformed Q&A (except for gpt-oss-120b, p < 0.05); results on ConflictBank were mixed, with Q&A significantly better than E2E on 3 LLMs.

### Human Evaluation (50 WikiContradict Samples)

| Metric | % Completely Correct Output | Main Error Types |
|--------|----------------|-------------|
| NLI | 48.0% | CLASS_ERR + MISS_ATOM + IRR_ATOM |
| Q&A | 66.0% | MISS_ATOM + DUP_ATOM |
| **E2E** | **88.0%** | CLASS_ERR + MISS_ATOM (minimal) |

The agreement rate between automatic LMR judgments and human labels reached **81.3%**, proving the reliability of the meta-evaluation protocol.

### LLM Comprehensiveness Measurements (ELI5 500 tasks, Best Evaluator)

| Model | Q&A Score | E2E Score |
|------|---------|---------|
| **gpt-oss-120b** | **0.71** (Highest) | **0.83** (Highest) |
| Llama 4 17Bx128E | 0.69 | 0.78 |
| gpt-oss-20b | 0.67 | 0.76 |
| Llama 3.3 70B | 0.68 | 0.75 |
| **Qwen 2.5 72B** | **0.66** (Lowest) | **0.73** (Lowest) |

Both Q&A and E2E identified gpt-oss-120b as the most comprehensive and Qwen 2.5 72B as the least; however, absolute Q&A scores were significantly lower than E2E (due to more fine-grained questions increasing the denominator).

### Key Findings
- **The brute-force E2E method is strongest on average**: E2E using Llama 4 17Bx128E achieved the highest average LMR of 0.85, whereas the complex NLI pipeline performed the worst. This suggests that "modern long-context LLMs in one step" are sufficient for many evaluation tasks, and overly fine-grained pipeline designs may be a hindrance.
- **E2E variance is 5x that of Q&A**: Choosing the wrong evaluator LLM leads to a sharp performance drop, indicating that E2E's simplicity comes at the cost of high dependency on a specific model's capabilities. Q&A should be preferred in production if the evaluator choice is uncertain.
- **Core failure of NLI = Context loss in isolated atoms**: Human error analysis showed NLI had the highest CLASS_ERR rate, confirming that entailment between two atoms requires the original context—which pure NLI lacks. This is a warning for all factuality methods that "extract atoms first, analyze independently later."
- **Q&A redundancy issues**: Although question refinement demands de-duplication, semantically similar questions (e.g., "Who is the author?" vs "Who wrote it?") often remain, causing some coverage to be underestimated.
- **gpt-oss-120b is the most comprehensive**: It performed best in RAG scenarios on ELI5, consistent with community assessments of the gpt-oss series.

## Highlights & Insights
- **Bypassing the completeness problem with "relative coverage"**: Converting the absolute question of "what should be said" into a relative question of "what was said in the reference and how much was covered" makes comprehensiveness evaluation actionable. This approach of defining evaluation targets via existing datasets can migrate to any metric where "complete sets are non-enumerable" (e.g., helpfulness, safety).
- **Three-tier granularity comparison (NLI/QA/E2E)**: Clearly demonstrates the trade-off between evaluator complexity and accuracy. The most complex (NLI) was the worst, while the simplest (E2E) was best on average but least stable. It serves as a reminder that "more complex pipelines aren't always better."
- **Unit comparison via Pint tool**: Identifying LLM weaknesses in physical unit conversion and fixing them with specialized tools is a classic hybrid neuro-symbolic evaluation design that can be generalized to other LLM weaknesses.
- **Graph-theoretic definition of uncovered context basis $\mathcal{A}_{basis}$**: Using the transitive closure of the fact graph to define the "minimal set for completion" allows the evaluation to provide not just a score, but specific suggestions for improvement, useful for real-time feedback and fine-tuning.
- **0.009 vs 0.044 cross-model std**: Including "evaluator robustness" as a metric alongside "accuracy" is a significant methodological contribution.

## Limitations & Future Work
- Authors acknowledge that NLI and Q&A are computationally intensive (NLI requires $O(N^2)$ LLM calls), making them suitable for scenarios requiring fine-grained diagnosis, while E2E is for large-scale evaluation with less interpretability.
- The fact graph is a simple structure based on atoms and cannot express complex factual relations (conditions, quantifiers, negations); however, complex graph computation would be prohibitively expensive.
- The evaluator assumes the reference corpus $\mathcal{C}$ is reliable. If $\mathcal{C}$ contains misinformation, a model might be penalized for not repeating errors.
- Recursive dependency of LLM-evaluating-LLM: The authors mitigate this by restricting the LLM to simple tasks and specific contexts, but it cannot be entirely eliminated.
- Thresholds $T_{rel} = 3.5$ and $T_{conf} = 2$ are manually set; they may require adjustment for different domains (e.g., medical vs. general), but no automatic tuning process is provided.
- The ELI5 experiment included only 500 tasks and 5 open-source models, lacking evaluation of closed-source models (GPT-4/Claude), which limits the scope.

## Related Work & Insights
- **vs FActScore (Min et al. 2023) / FacTool (Chern et al. 2023)**: These methods focus on precision (verifying correctness), while this work specifically addresses the missing recall dimension.
- **vs SAFE (Wei et al. 2024) $R_K$ metric**: SAFE uses a preset $K$ for coarse recall; this work uses a reference corpus and three methods to output specific uncovered fact sets, offering much stronger diagnostic capabilities.
- **vs FEQA / QuestEval (Eyal 2019 / Scialom 2021)**: These QA-based summarization consistency metrics inspired the Q&A pipeline here, but this work extends from fixed source documents to multi-source RAG.
- **vs Marinescu et al. (2025) FactReasoner**: Borrowed its atom extraction + graph analysis logic, but while FactReasoner targets precision, this work applies it to recall.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly splitting comprehensiveness into three granularities and meta-evaluating the evaluator itself is a rare methodological contribution in factuality evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 meta-eval datasets × 5 LLMs × 3 metrics + human evaluation + real-world ELI5 application + BCa bootstrap + permutation tests. Solid, though it lacks closed-source LLM testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formal definitions, explicit pipeline steps, and full prompts in the appendix provide excellent reproducibility.
- Value: ⭐⭐⭐⭐ Provides an actionable toolkit for evaluating whether LLMs miss key information; the NLI vs. QA vs. E2E comparison provides methodological insights for other evaluation protocol designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)

</div>

<!-- RELATED:END -->
