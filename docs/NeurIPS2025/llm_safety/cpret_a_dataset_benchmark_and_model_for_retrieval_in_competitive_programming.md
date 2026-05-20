---
title: >-
  [Paper Note] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming
description: >-
  [NeurIPS 2025][LLM Safety][competitive programming retrieval] To address the prevalence of duplicate and near-duplicate problems in competitive programming—which compromises contest fairness and inflates LLM evaluation s…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "competitive programming retrieval"
  - "duplicate problem detection"
  - "embedding model"
  - "Group-InfoNCE"
  - "data contamination"
  - "benchmark"
date: 2026-05-08
content_hash: 212e3e56a41c02b6
---

# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming

**Conference**: NeurIPS 2025
**arXiv**: [2505.12925](https://arxiv.org/abs/2505.12925)  
**Code**: [https://github.com/coldchair/CPRet](https://github.com/coldchair/CPRet)  
**Area**: AI Safety
**Keywords**: competitive programming retrieval, duplicate problem detection, embedding model, Group-InfoNCE, data contamination, benchmark

## TL;DR

To address the prevalence of duplicate and near-duplicate problems in competitive programming—which compromises contest fairness and inflates LLM evaluation scores—this work constructs CPRet, a large-scale benchmark spanning four retrieval tasks, and proposes CPRetriever, a domain-specific retrieval model trained with Group-InfoNCE loss. CPRetriever surpasses 20+ existing embedding models across all tasks and reveals systematic evaluation bias in LiveCodeBench attributable to problem similarity.

## Background & Motivation

**Background**: Competitive programming contests (ICPC, IOI, etc.) have become standard benchmarks for evaluating LLM code generation and reasoning capabilities (LiveCodeBench, HumanEval, etc.), with thousands of new problems added annually.

**Prevalence of Duplicate Problems**: A large number of duplicate or highly similar problems have accumulated across online judge (OJ) platforms, with community discussions about "repeated problems" increasing noticeably in recent years. This introduces two harms: (a) contestants who have previously encountered the original problem gain an unfair advantage; (b) LLM evaluation scores are inflated—models may rely on memorization rather than genuine reasoning.

**Gap in Retrieval Tasks**: Existing code retrieval benchmarks (e.g., CoIR) cover only Text-to-Code and Code-to-Code dimensions, lacking **problem-level** similarity retrieval tasks, as well as corresponding training data and evaluation standards.

**Temporal Leakage**: Analysis of the APPS dataset partitioned by time reveals that model performance on problems predating 2022 significantly exceeds that on newer problems, demonstrating that training data leakage substantially undermines evaluation integrity.

**Goal**: (a) Define four retrieval tasks and construct a large-scale benchmark; (b) train a competitive programming–specific retrieval model; (c) quantify the systematic impact of problem similarity on LLM evaluation.

**Core Idea**: Apply Group-InfoNCE loss to align problems with their multiple correct solutions, fine-tune the resulting model into a problem-level retriever, and establish the first comprehensive benchmark for competitive programming retrieval.

## Method

### Four Retrieval Task Definitions

Four retrieval tasks spanning both code and problem dimensions are defined:

- **Text-to-Code (T2C)**: Given a natural language problem description, retrieve correct code solutions. Evaluates the model's ability to align problem semantics with code implementations. Training set: 38.8K problems / 2.93M code pairs; test set: 4.9K queries / 41.6K corpus.
- **Code-to-Code (C2C)**: Given one correct solution, retrieve other correct solutions to the same problem. Evaluates the model's ability to understand code functionality and identify syntactically distinct but semantically equivalent implementations. Test set: 4.8K queries / 39.8K corpus.
- **Problem-to-Duplicate (P2Dup)** (newly proposed): Given a problem description, retrieve duplicate or highly similar problems from a corpus of 10,900+ problems. Training: 491 pairs; test: 168 queries / 202 relevant pairs.
- **Simplified-to-Full (S2Full)** (newly proposed): Given a simplified or paraphrased version of a problem, retrieve the corresponding original full description. Training: 7.6K pairs; test: 10K queries / 10K corpus.

### Dataset Construction (CPRet-PCPCD)

The dataset comprises 42.2K problems and 2.9M code submissions from 12 OJ platforms, covering problem statements in 3 languages (English/Chinese/Japanese) and 20+ programming languages, including both ICPC-style (all-or-nothing scoring) and OI-style (partial scoring) problems, with a cutoff date of December 2024. This substantially surpasses existing datasets:

| Dataset | Problems | Code | Cutoff |
|---|---|---|---|
| Description2Code | 7.8K | 309K | 2016/08 |
| APPS | 10K | 232K | 2020/10 |
| CodeContests | 13.6K | 4.5M | 2021/07 |
| TACO | 26.4K | 1.55M | 2023/02 |
| **CPRet-PCPCD** | **42.2K** | **2.9M** | **2024/12** |

Key data collection procedures:
- **Duplicate problem pairs**: All public discussion posts and blogs on Codeforces and Luogu are crawled; keyword-based heuristics combined with LLM classification yield ~5,000 candidates, which are then manually annotated by multiple experienced competitive programmers. Three-level duplicate criteria are defined: Exact Match (solutions pass directly), Near Match (minor modifications suffice), and Method Match (same core algorithm but different implementation details). Approximately 700 pairs are ultimately annotated; after clustering, 30% of clusters are randomly selected for the test set.
- **Simplified problem pairs**: User-contributed Chinese translations/simplifications on Luogu (originating from Codeforces, AtCoder, etc.) are crawled; after filtering low-quality machine translations, ~17K pairs are obtained.
- **Temporal split**: Test sets for code retrieval tasks strictly use problems published after 2023 to prevent training data leakage.

### Group-InfoNCE Loss

**Design Motivation**: In competitive programming, a single problem typically admits multiple correct solutions (different algorithms, different programming languages). Standard InfoNCE handles only single positive pairs, while the Multi-Pos extension naively averages positive sample similarities without enforcing intra-group consistency.

**Core Idea**: All correct solutions to the same problem are treated as a "positive group" $G_i = \{x_i^{1+}, \ldots, x_i^{m+}\}$, with group similarity defined as:

$$\mathrm{sim}_G(x_i, G_j) = \frac{1}{m}\sum_{k=1}^{m}\mathrm{sim}(x_i, x_j^{k+})$$

The full loss function consists of three components: a contrastive term, a group contrastive term, and a variance regularization term:

$$\mathcal{L}_{\text{Group}} = -\log\frac{\exp(\mathrm{sim}_G(x_i,G_i)/\tau)}{\exp(\mathrm{sim}_G(x_i,G_i)/\tau) + \sum_{j\neq i}[\exp(\mathrm{sim}(x_i,x_j)/\tau) + \exp(\mathrm{sim}_G(x_i,G_j)/\tau)]} + \frac{\text{Penalty}_G(x_i,G_i)}{\tau^2}$$

where the variance regularization $\text{Penalty}_G = \mathrm{Var}_{k=1}^{m}(\mathrm{sim}(x_i, x_i^{k+}))$ ensures that all solutions to the same problem maintain a consistent distance from the problem embedding. A key insight is that the variance penalty constrains the consistency of similarities rather than the solution embeddings themselves—this allows solutions based on different algorithmic strategies to remain diverse in the embedding space while all "orbiting" around the problem embedding.

### Format Masking

During training, non-essential portions of problem statements—including I/O format descriptions, sample inputs/outputs, and data range constraints—are randomly masked, forcing the model to focus on algorithmic semantics rather than formatting cues (e.g., numerical patterns in sample inputs), thereby improving generalization across diverse problem formats.

### CPRetriever-Prob Fine-tuning

Built upon CPRetriever-Code, the model is jointly fine-tuned on P2Dup and S2Full training data using a triplet loss:

$$\mathcal{L}_{\text{triplet}} = \max(0, \mathrm{sim}(x, x^-) - \mathrm{sim}(x, x^+) + \alpha)$$

Hard negative mining strategy: SFR-Embedding-Code-2B first pre-retrieves the top-10 most similar problems per query as candidate negatives; Qwen-2.5-Max then automatically verifies and removes false negatives (i.e., samples that are actually duplicate problems), ensuring reliable negatives without additional manual annotation.

## Key Experimental Results

### Main Results (NDCG@10, across Tiny/Small/Medium/Large scale tiers)

| Model (Scale) | T2C | C2C | P2Dup | S2Full | Avg |
|---|---|---|---|---|---|
| gte-modernbert-base (149M) | 14.99 | 36.22 | 21.12 | 77.45 | 37.44 |
| SFR-Emb-Code-400M (400M) | 9.43 | 43.59 | 19.40 | 75.31 | 36.93 |
| SFR-Emb-Code-2B (2B) | 39.60 | 68.05 | 45.26 | 86.43 | 59.84 |
| Qodo-Embed-1-7B (7B) | 36.47 | 51.91 | 47.15 | 91.17 | 56.68 |
| Qwen3-Embedding-0.6B (600M) | 48.96 | 60.49 | 36.26 | 81.63 | 56.83 |
| Qwen3-Embedding-4B (4B) | 66.62 | 71.97 | 56.59 | 89.39 | 71.15 |
| Qwen3-Embedding-8B (8B) | 60.54 | 72.97 | 53.23 | 87.95 | 68.67 |
| **CPRetriever-Code (2B)** | **70.40** | 70.59 | 38.68 | 81.45 | 65.28 |
| **CPRetriever-Prob (2B)** | 56.50 | 70.68 | 60.06 | 90.74 | 69.50 |
| **CPRetriever-Code-Qwen3 (4B)** | **86.22** | **86.70** | 41.14 | 88.10 | 75.54 |
| **CPRetriever-Prob-Qwen3 (4B)** | 80.84 | 87.10 | **74.33** | **96.15** | **84.60** |

### Ablation Study (Group-InfoNCE vs. Alternative Losses)

| Loss Function | T2C | C2C | Avg |
|---|---|---|---|
| InfoNCE (single positive) | baseline | baseline | baseline |
| Multi-Pos | marginal gain | marginal gain | +1–2% |
| **Group-InfoNCE** | **best** | **best** | consistent improvement across both base models |

### Key Findings
- CPRetriever-Code substantially outperforms the strongest same-scale baseline SFR-Emb-Code-2B on code retrieval (T2C: 70.4 vs. 39.6; C2C: 70.6), a relative improvement of approximately 78%.
- Fine-tuning into CPRetriever-Prob yields a jump on P2Dup from 38.7 to 60.1 (+55%) and on S2Full from 81.4 to 90.7 (+11%), while T2C drops to 56.5—indicating an inherent tension between the two task categories.
- Code domain–specific models consistently outperform general-purpose embedding models (e.g., the 7B Qodo model achieves a lower average than the 2B CPRetriever-Prob).
- Using Qwen3-4B as the backbone proves superior to SFR-2B; CPRetriever-Prob-Qwen3 achieves an average of 84.6 across all four tasks.

### Impact of Problem Similarity on LLM Evaluation

Analysis of 388 LiveCodeBench problems published after September 2024:

- **Pass rate increases with similarity**: Problems with higher maximum cosine similarity to historical problems yield higher average pass rates across all models. The positive correlation is strongest for medium-difficulty problems; hard problems have generally low pass rates, but an upward trend is still observed in the high-similarity range.
- **High similarity compresses inter-model differences**: Performance gaps among the Low/Medium/High variants of O3-Mini and O4-Mini nearly vanish in the high-similarity range (0.80–0.90), whereas differences are pronounced in the low-similarity range. This demonstrates that highly similar problems allow weaker models to perform well through memorization, masking true reasoning capability gaps.
- **Difficulty and similarity are partially independent**: Easy problems exhibit slightly higher historical similarity than hard problems, but the difference is limited, suggesting that similarity should be controlled as an independent factor in benchmark construction.

## Highlights & Insights
- **Group-InfoNCE** constitutes a general solution for handling one-to-many positive sample relationships. Its core design—group similarity combined with variance regularization—is transferable to other multi-positive contrastive learning scenarios, such as multiple paraphrases of the same sentence or multiple images of the same concept.
- **The similarity analysis of LiveCodeBench** exposes a hidden systematic bias in community benchmarks: high-similarity problems inflate model evaluation scores and obscure real capability differences among models. Future benchmarks should be stratified or filtered by similarity.
- **Extending competitive programming retrieval from the code level to the problem level** represents a forward-looking problem formulation—grounded in practical needs (contest plagiarism detection, educational search)—that fills a gap in existing benchmarks.
- **The two-stage training strategy** (Code first, then Prob) reveals an interesting trade-off: T2C relies on implementation details, while P2Dup and S2Full depend on high-level semantics. Given this inherent tension, releasing two specialized models is a well-justified engineering decision.

## Limitations & Future Work
- The annotated duplicate problem pair set is limited in scale (~700 pairs), and the P2Dup test set contains only 168 queries, constraining evaluation robustness and statistical significance.
- Similarity is computed solely from text embeddings, without exploiting structured information within problems (e.g., constraint ranges, algorithm tags, time/memory limits).
- Although the test set applies a temporal split, these problems may still be exposed as new model training data is updated; the authors plan to refresh the benchmark every 6–12 months.
- Analysis is limited to LiveCodeBench and has not been extended to HumanEval, APPS, or mathematical competition settings (IMO, AIMO).
- Future work could incorporate RAG techniques, using retrieved similar problems to assist LLMs in solving new problems, forming a retrieval-generation closed loop.

## Related Work & Insights
- **vs. CoIR/APPS benchmarks**: These cover only T2C and C2C tasks, lack temporal splits, and have cutoff dates of 2020–2023. CPRet introduces two problem-level tasks, enforces strict temporal separation, and extends data coverage to December 2024.
- **vs. SFR-Embedding-Code-2B**: A strong code embedding model that achieves only 39.6 on competitive programming T2C. CPRetriever-Code reaches 70.4 at the same scale, a 78% improvement.
- **vs. Qwen3-Embedding-4B/8B**: General-purpose large models achieve only 56.6/53.2 on P2Dup; after domain-specific fine-tuning, CPRetriever-Prob-Qwen3 (4B) reaches 74.3, a 31% improvement.
- **vs. Qodo-Embed-1-7B**: This 7B code embedding model achieves 91.2 on S2Full but only 36.5 on T2C, illustrating the trade-off between general high performance and task-specific optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic definition of problem-level retrieval tasks in competitive programming; Group-InfoNCE offers theoretical inspiration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comparison against 20+ models, full coverage of four tasks, and a compelling similarity analysis on LiveCodeBench.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated contributions, rich figures and tables, rigorous experimental design.
- Value: ⭐⭐⭐⭐ — Dataset, models, and online demo are all open-sourced, with direct practical utility for both the competitive programming community and the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)
- [\[NeurIPS 2025\] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)
- [\[NeurIPS 2025\] Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery](unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [\[ICCV 2025\] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](../../ICCV2025/llm_safety/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)

</div>

<!-- RELATED:END -->
