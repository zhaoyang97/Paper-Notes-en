---
title: >-
  [Paper Note] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization
description: >-
  [ACL 2026][Information Retrieval & RAG][GEO] This paper treats "optimizing a single document for multiple potential queries simultaneously" as a constrained multi-objective optimization problem and proposes IF-GEO. Follo…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "GEO"
  - "Generative Search"
  - "Multi-query Optimization"
  - "Conflict-Aware Instruction Fusion"
  - "Risk-Aware Stability"
date: 2026-05-08
content_hash: 3d32a27545c6d6b0
---

# IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization

**Conference**: ACL 2026  
**arXiv**: [2601.13938](https://arxiv.org/abs/2601.13938)  
**Code**: Not declared by authors (as of acceptance)  
**Area**: Information Retrieval / Generative Engine Optimization (GEO) / RAG  
**Keywords**: GEO, Generative Search, Multi-query Optimization, Conflict-Aware Instruction Fusion, Risk-Aware Stability

## TL;DR
This paper treats "optimizing a single document for multiple potential queries simultaneously" as a constrained multi-objective optimization problem and proposes IF-GEO. Following a "diverge-then-converge" strategy, it first uses an LLM to reverse-mine representative queries and generate structured edit requests. Then, through **Priority × Necessity scoring + Dedup + Conflict Resolution + Global Revision Blueprint**, it fuses conflicting edit instructions into an executable modification blueprint. The authors also introduce three risk-aware stability metrics: WCP, DR, and WTR. On GEO-Bench, IF-GEO increases the Objective Overall from 7.59 (Auto-GEO) to 11.03, while reducing the worst-case single-query drop from -0.0511 to -0.0090.

## Background & Motivation

**Background**: Generative Search Engines (GSEs, such as ChatGPT Search and Perplexity) are replacing traditional ranking-based search engines. Visibility is no longer determined by rank but by "whether it is selected and cited by the LLM in the answer." "Generative Engine Optimization (GEO)," proposed at KDD'24, focuses on rewriting document content to enhance its exposure in generated responses.

**Limitations of Prior Work**: Existing methods (e.g., the 9 heuristic rules of GEO, preference rules of Auto-GEO, and single-intent trajectories of RAID) treat the multi-query visibility problem as a **one-dimensional** optimization—optimizing for a single target. In reality, a document often needs to satisfy 3–5 heterogeneous queries (e.g., "What is X" / "Pros and cons of X" / "Usage of X"), which frequently **conflict** under limited content budgets: adding examples for A might squeeze out the statistics required for B.

**Key Challenge**: Using the "mean" or a "single aggregated intent" as the optimization objective masks the true failure modes—significant degradation for a minority of queries—under the guise of improved averages. Heuristic methods (e.g., "adding citations") provide positive mean gains but cannot handle query-level trade-offs.

**Goal**: (a) Propose a framework capable of "generating divergent instructions and then converging them via fusion"; (b) Establish explicit risk-aware evaluation protocols (WCP/DR/WTR) to measure "tail degradation."

**Key Insight**: Treat each candidate query as an independent "stakeholder." Let the LLM first propose individual edit requests (with necessity scores), then use a "coordinator" to perform global scoring, ranking, deduplication, and conflict arbitration. Finally, output a JSON blueprint aggregated by document sections as a strong constraint contract for subsequent rewriting.

**Core Idea**: Replace "one-query-one-rewrite" with "diverge-then-converge + conflict-aware instruction fusion"—integrating the multi-objective optimization coordinator into the LLM editing process.

## Method

### Overall Architecture
IF-GEO is a pure LLM-API pipeline (using GPT-4o-mini for all calls), divided into two Phases:

1.  **Phase I — Diverge**: (a) **Query Discovery**: The LLM acts as a "Search Analyst" to perform reverse retrieval on document $D$, outputting a weighted set of representative queries $Q(D) = \{(q_i, w_i)\}_{i=1}^m$, where $w_i \in [0,100]$ is the "popularity" score given by the LLM, with paraphrasing explicitly prohibited; (b) **Request Generation**: Each $q_i$ is analyzed independently to identify "what the document lacks," generating structured requests $r_{i,j} = \langle e_{i,j}, u_{i,j}, s_{i,j} \rangle$, where $e_{i,j}$ is a local anchor snippet, $u_{i,j}$ is the rewrite suggestion, and $s_{i,j} \in [0,100]$ is the necessity score (G-EVAL style).
2.  **Phase II — Converge**: (a) **Prioritization & Dedup**: Calculate global priority $g_{i,j} = w_i \cdot s_{i,j}$. Requests below a threshold $\tau$ are discarded (default $\tau = 0.7 \times 100 \times 100$ equivalent rule). Semantically duplicate requests are merged into meta-requests, retaining the highest $s$; (b) **Conflict Resolution**: For mutually exclusive requests on the same anchor, the LLM performs "Selection (choosing the high score if the gap is large)" or "Synthesis (creating a compromise if scores are close)" based on $g$ values; (c) **Blueprint Construction**: Retained instructions are aggregated by document **section** rather than query into an ordered JSON blueprint; (d) **Blueprint-Guided Revision**: Another LLM acts as a "constrained editor" to modify the document strictly according to the blueprint, with an explicit order to **keep unmentioned sections as-is** to prevent free-form rewriting.
3.  **Objective Function**: In addition to maximizing $\mathbb{E}[\Delta v]$, WCP, DR, and WTR (see below) are introduced as equally important optimization constraints.

### Key Designs

1.  **Diverge — Weighted Representative Queries + Structured Requests with Necessity Scores**:
    *   **Function**: Uses consistent structured "edit requests" to externalize and make "serving different queries" comparable.
    *   **Mechanism**: $Q(D)$ uses reverse retrieval instead of paraphrasing to approximate the "true potential user distribution." The LLM assigns two independent scores: $w_i$ for query importance among users and $s_{i,j}$ for the criticality of an edit to that query. The product $g_{i,j} = w_i \cdot s_{i,j}$ directly drives fusion and arbitration.
    *   **Design Motivation**: Traditional GEO blurs different query needs at the start into an "engine preference," losing query variance. Explicit structured requests preserve these differences so the fusion stage can "see conflicts."

2.  **Converge — Prioritize → Dedup → Conflict-Resolve → Blueprint**:
    *   **Function**: Converges the divergent request pool into an **executable** global modification blueprint.
    *   **Mechanism**: First, $g_{i,j}$ thresholding removes noise. Then, semantic deduplication occurs. Mutually exclusive requests are handed to the LLM for "semantic arbitration" rather than relying on hard thresholds—selecting the best if scores differ significantly or synthesizing if close. Finally, instructions are reordered into a JSON blueprint **by section instead of by query**, changing the process from "serial patching by query" to "one-time editing by section."
    *   **Design Motivation**: Ablations show that **Conflict Resolution** is the most critical stage for performance (removing it drops the Mean from 9.24 to 6.14). Organizing by section avoids the disaster of "re-editing the same paragraph repeatedly and eventually overwriting useful info."

3.  **Risk-Aware Stability Objective (WCP / DR / WTR)**:
    *   **Function**: Incorporates "stability across multiple queries" into the objective and evaluation to prevent averages from masking tail degradation.
    *   **Mechanism**: (i) **Worst-Case Performance** $\text{WCP} = \min_{i=1}^m \Delta v_i$ provides a safety floor; (ii) **Downside Risk** $\text{DR} = \frac{1}{m}\sum_{i=1}^m (\min(0, \Delta v_i))^2$ penalizes only the square of negative gains, distinguishing "benign fluctuations" from "harmful ones"; (iii) **Win-Tie Rate** $\text{WTR} = \frac{1}{m}\sum_{i=1}^m \mathbb{I}(\Delta v_i \ge 0)$ quantifies the "proportion of coverage without regression" as a proxy for Pareto safety.
    *   **Design Motivation**: Standard variance (VAR) treats both positive and negative fluctuations as risks. Higher visibility upside is good for GEO; only "significant drops in minority queries" constitute true failure, which DR and WCP capture accurately.

### Loss & Training
**No model training**—IF-GEO is a pure inference-time framework where all steps are prompt calls with fixed schemas. Default hyperparameters: query expansion $N_q = 5$, suggestions per query $N_s = 5$, internal temperature = 0.2, $\tau = 0.7$. Rewriting is performed by the same LLM and evaluated via the GPT-4o-mini simulation engine from GEO-Bench.

## Key Experimental Results

### Main Results

GEO-Bench / RAID multi-query benchmark (1k queries, 5 relevant queries per document). Visibility improvement (higher is better):

| Method | Objective Overall | Objective Word | Objective Position | Subjective Average |
| :--- | :--- | :--- | :--- | :--- |
| Trans. SEO | 1.84 | 1.83 | 1.77 | 1.51 |
| Cite Sources (strongest heuristic) | 4.71 | 4.47 | 4.59 | 3.31 |
| Quotation Addition | 4.23 | 4.29 | 4.19 | 2.71 |
| Statistics Addition | 3.49 | 3.28 | 3.39 | 2.31 |
| RAID (Single intent) | 0.88 | 1.06 | 0.78 | 1.36 |
| Auto-GEO (Prev. SOTA) | 7.59 | 7.80 | 7.64 | 5.30 |
| **IF-GEO (Ours)** | **11.03** | **11.07** | **11.15** | **5.87** |

Cross-query stability metrics (for Objective Overall):

| Method | VAR ↓ | WCP ↑ | WTR ↑ | DR ↓ |
| :--- | :--- | :--- | :--- | :--- |
| Cite Sources | 0.0165 | -0.0785 | 72.06% | 0.0044 |
| Auto-GEO | 0.0159 | -0.0511 | 73.56% | 0.0043 |
| **IF-GEO** | 0.0189 | **-0.0090** | **80.50%** | **0.0023** |

IF-GEO reduces the "worst-case single-query drop" from -0.0511 (Auto-GEO) to -0.0090 (≈ -82% reduction), halves DR, and increases WTR from 73.56% to 80.50%.

### Ablation Study

Subset of 250 queries (values slightly lower than main results due to sample size):

| Variant | Mean ↑ | VAR ↓ | WCP ↑ | WTR ↑ | DR ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| IF-GEO (Full) | **9.24** | 0.0156 | **-0.0328** | **80.80%** | **0.0021** |
| w/o Blueprint Construction | 8.18 | 0.0167 | -0.0517 | 81.20% | 0.0021 |
| w/o Instruction Fusion | 7.07 | 0.0156 | -0.0569 | 74.80% | 0.0043 |
| w/o Conflict Resolution | 6.14 | 0.0174 | -0.0713 | 77.20% | 0.0032 |

### Key Findings
*   **Conflict Resolution is the most critical safety guardrail**: Removing it causes the largest Mean drop (3.1pt) and the deepest WCP drop, showing that LLM-led "dynamic conflict arbitration" is why IF-GEO avoids regression. In contrast, Blueprint Construction mainly affects "execution efficiency" rather than "stability."
*   **Instruction Fusion treats the tail**: Without it, WTR drops from 80.8% to 74.8% and DR doubles to 0.0043, proving that fusion is about "reducing fighting rules" rather than "adding more rules." Its value lies in tail stability rather than the mean.
*   **N=5 is the sweet spot**: Increasing expanded queries from 1 to 9 shows a monotonic Mean increase (8.06 to 10.02), but WTR/DR/WCP plateau after $N=5$. Given linear cost increases, $N=5$ is the default.
*   **Cross-Engine Generalization**: When the target GE is switched to Gemini-2.0-Flash (with no method tuning), IF-GEO still leads Auto-GEO in WCP/WTR, suggesting "explicit coordination" is more universal than "engine-specific preference rules."
*   **Initial Rank Robustness**: Analyzing by document initial rank buckets shows IF-GEO maintains stable gains even in low-rank buckets, indicating it genuinely improves "content robustness" rather than benefiting from positional bias.

## Highlights & Insights
*   **Bringing the "coordination mechanism" of multi-objective optimization directly into LLM editing** is the greatest conceptual innovation. GEO is no longer just prompt engineering or heuristic stacking; it's an optimization problem with a formalized objective function (constrained by WCP/DR).
*   **The WCP/DR/WTR trio is a reusable evaluation language**: Many LLM applications face the "average-good but tail-bad" problem (recommendations, personalization, alignment). Upgrading from an "average-only" view to a risk-aware view should become an industry standard.
*   **The "Reverse Retrieval + Necessity Scoring" structured request** is elegant—it translates vague optimization intents into comparable, arbitrable structured objects, allowing LLMs to engage in "semantic negotiation." This can be migrated to prompt rewriting, PR reviews, etc.
*   **Letting the LLM judge "if the score gap is large"** rather than using hard thresholds for arbitration is a low-cost, flexible design that avoids the pain of hyperparameter tuning for trade-offs without massive human labels.

## Limitations & Future Work
*   **Inference Cost**: The full pipeline requires $N_q$ query mining steps + $N_q \times N_s$ request generation steps + multi-step fusion + rewriting. Token consumption is much higher than single-pass baselines; cost-per-second is a barrier to deployment.
*   **Simulation Gap**: Evaluation is based on GPT-4o-mini as a simulated GE. Real-world tests on commercial GSEs like Perplexity or Bing AI are missing, leaving visibility transferability in question.
*   **Query Discovery Single-Point-of-Failure**: The quality of the blueprint depends on the first step. If the query distribution is biased (e.g., cold-start tail domains), the fusion will be inaccurate. There is no study on "recovery from bad discovery."
*   Personal observations: (a) The multiplicative $g_{i,j} = w_i \cdot s_{i,j}$ is rough; future work could use LP/softmax normalization; (b) "Adversarial GEO"—analyzing the equilibrium when multiple publishers use IF-GEO simultaneously—is not addressed; (c) There is clearly a Pareto trade-off between WCP/DR/WTR and Mean, but the Pareto front is not provided.

## Related Work & Insights
*   **vs GEO (KDD'24)**: GEO uses 9 manual heuristics (add citations, stats, authority, etc.) and is query-agnostic. IF-GEO is a query-aware coordination framework, upgrading GEO from heuristics to an "optimization algorithm."
*   **vs Auto-GEO (Wu et al., 2025)**: Auto-GEO learns search engine preference rules from large-scale ranking data but remains a single aggregated target. IF-GEO re-diagnoses and edits each document without pre-learned rules, explicitly optimizing risk-aware targets.
*   **vs RAID (Chen et al., 2025b)**: RAID uses 4W multi-role reflection to infer a single intent trajectory. IF-GEO preserves multiple intents and explicitly arbitrates conflicts. RAID's Mean is only 0.88 in multi-query scenarios, far behind IF-GEO's 11.03.
*   **vs General Multi-Objective Optimization (Pareto / ε-constraint)**: IF-GEO implements the "coordinator" as an LLM prompt instead of using classical theories. This is a successful case of the "LLM-as-decision-maker" paradigm.

## Rating
*   Novelty: ⭐⭐⭐⭐ "Diverge-then-converge + LLM conflict arbitration + risk-aware metrics" is new in GEO, though the underlying paradigm aligns with multi-agent debate/G-EVAL.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison with 11 baselines + 4 stability metrics + full ablation + query sweep + cross-model/rank robustness. Missing a real-world commercial GSE test.
*   Writing Quality: ⭐⭐⭐⭐ Motivation is clear (Fig 1 illustrates conflicts well); definitions are standardized; physical meaning of metrics is explained effectively.
*   Value: ⭐⭐⭐⭐ GEO is an emerging field. This paper contributes both a method and an evaluation protocol (WCP/DR/WTR), which will likely influence subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/information_retrieval/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)

</div>

<!-- RELATED:END -->
