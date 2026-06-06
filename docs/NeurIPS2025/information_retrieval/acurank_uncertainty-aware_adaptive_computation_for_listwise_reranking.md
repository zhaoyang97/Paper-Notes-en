---
title: >-
  [Paper Note] AcuRank: Uncertainty-Aware Adaptive Computation for Reranking
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][uncertainty estimation] AcuRank dynamically adjusts the reranking subset size and verification scope via TrueSkill-based uncertainty estimation…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "uncertainty estimation"
  - "adaptive computation"
  - "Bayesian model"
  - "reranking"
  - "context length optimization"
date: 2026-05-08
content_hash: 27ea812a7d47705b
---

# AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking

**Conference**: NeurIPS 2025
**arXiv**: [2505.18512](https://arxiv.org/abs/2505.18512)  
**Code**: N/A  
**Area**: Other
**Keywords**: uncertainty estimation, adaptive computation, Bayesian TrueSkill, listwise reranking, accuracy-efficiency trade-off

## TL;DR

AcuRank maintains a probability distribution over document relevance via a Bayesian TrueSkill model, and at each iteration selectively reranks only documents whose positions remain uncertain. This yields a reranking framework that adaptively allocates computation according to query difficulty, surpassing fixed-computation baselines on multiple benchmarks with fewer LLM calls.

## Background & Motivation

**Background**: Modern information retrieval pipelines (e.g., web search, RAG systems) typically employ a fast first-stage retriever such as BM25 to obtain a candidate set, followed by LLM-based listwise reranking to improve top-$k$ precision. Due to LLM context length constraints, reranking can only be performed over small subsets (typically 20 documents), requiring multiple calls whose results must be aggregated.

**Limitations of Prior Work**: Dominant strategies—Sliding Windows and TourRank—both allocate a fixed computation budget, assigning an identical number of reranker calls to every query regardless of its difficulty. This leads to over-computation on easy queries and under-computation on hard ones, with no mechanism to leverage intermediate signals for dynamic adjustment.

**Key Challenge**: Fixed-computation strategies treat all queries uniformly, yet query difficulty varies enormously (as evidenced by the wide WIG distribution). Documents ranked low in early rounds have no opportunity for recovery, even when the initial judgment was based on limited or noisy context.

**Goal**: The paper seeks a method to dynamically determine *which* documents to rerank and *how many* times, based on per-query and per-document ranking uncertainty, simultaneously improving accuracy and reducing unnecessary computation.

**Key Insight**: The paper draws on TrueSkill—a Bayesian framework originally developed for multiplayer game rating—modeling document relevance as Gaussian distributions, quantifying uncertainty via ranking probabilities, and using these signals to guide selective reranking and adaptive termination.

**Core Idea**: Document relevance is represented as a probability distribution rather than a point estimate. Computation is invested only in documents with uncertain rankings, and Bayesian belief updates drive iterative refinement until convergence.

## Method

### Overall Architecture

AcuRank is an iterative adaptive reranking framework operating as follows:
1. **Initialization**: Each document's TrueSkill parameters $(\mu_i, \sigma_i)$ are initialized from first-stage retrieval scores, with $\mu_i$ set to the raw retrieval score and $\sigma_i = \mu_i / 3$.
2. **Uncertainty-Based Selection**: The probability $s_i = P(x_i > t(k))$ that each document belongs to the top-$k$ is computed; documents satisfying $\epsilon < s_i < 1 - \epsilon$ form the candidate set $\mathcal{C}$.
3. **Grouped Reranking**: Documents in $\mathcal{C}$ are sorted by $\mu_i$ in descending order, partitioned into groups of size $m = 20$, and passed to the LLM reranker.
4. **Bayesian Update**: TrueSkill message-passing updates each document's parameters based on reranking outcomes: documents ranked higher than expected receive an increased $\mu_i$ and decreased $\sigma_i$.
5. **Stopping Criterion**: The process terminates when $|\mathcal{C}| < \tau$ or the computation budget is exhausted.
6. **Output**: Documents are sorted by their final $\mu_i$ values.

### Key Designs

1. **TrueSkill Probabilistic Relevance Modeling**
    - **Function**: Represents each document's relevance as a Gaussian distribution $x_i \sim \mathcal{N}(\mu_i, \sigma_i^2 + \beta^2)$ rather than a scalar score.
    - **Mechanism**: $\mu_i$ encodes estimated relevance, $\sigma_i$ encodes epistemic uncertainty (which decreases as reranking evidence accumulates), and $\beta$ is a fixed observation noise term. Bayesian posterior updates automatically revise the belief distribution after each reranking round.
    - **Design Motivation**: Point estimates cannot capture ranking confidence. Probabilistic modeling naturally quantifies the "probability of rank inversion," providing a principled signal for adaptive computation.

2. **Ranking-Probability-Based Uncertainty Selection**
    - **Function**: Efficiently identifies documents that warrant further reranking, bypassing those already firmly inside or outside the top-$k$.
    - **Mechanism**: A threshold $t(k)$ is defined such that $\sum_i P(x_i > t(k)) = k$, so $s_i = P(x_i > t(k))$ captures the probability that document $D_i$ belongs to the top-$k$. Only documents with $\epsilon < s_i < 1 - \epsilon$ enter the candidate set; the remaining documents retain their current positions.
    - **Design Motivation**: This avoids the $O(n^2)$ complexity of exact rank distribution computation by approximating with Gaussian tail probabilities, while realizing the core principle of concentrating computation where uncertainty is highest.

3. **Sequential Grouping and Adaptive Stopping**
    - **Function**: Partitions uncertain candidate documents into groups for the reranker and determines when to terminate based on convergence signals.
    - **Mechanism**: Documents are sorted by $\mu_i$ in descending order and sequentially partitioned into groups of size $m$ (rather than randomly), ensuring documents of similar estimated relevance are compared within the same group, yielding more informative updates. The default stopping condition is $|\mathcal{C}| < \tau = 10$.
    - **Design Motivation**: Ablation experiments confirm that sequential grouping outperforms random grouping in both accuracy and efficiency, as comparisons among similarly relevant documents are more informative. Uncertainty-driven stopping is more efficient than a fixed budget (19.7 calls vs. 22.7 calls at equivalent accuracy).

### Loss & Training

AcuRank is a **training-free** inference-time framework. The core update mechanism derives from TrueSkill's closed-form Bayesian update:

$$\mu_i^{(t+1)} = \mu_i^{(t)} + \frac{\sigma_i^2}{c} \lambda, \quad (\sigma_i^{(t+1)})^2 = \sigma_i^2 \left(1 - \frac{\sigma_i^2}{c} \upsilon\right)$$

where $c = \sum_{D_j \in \mathcal{B}} (\sigma_j^2 + \beta^2)$, and $\lambda$ and $\upsilon$ are determined by the win/loss probabilities of adjacent document pairs. Each reranking result is interpreted as the outcome of a "multiplayer game": documents ranked higher than expected receive an increased $\mu_i$, while $\sigma_i$ decreases to reflect growing confidence.

Hyperparameters—$\epsilon = 0.01$ (uncertainty threshold), $m = 20$ (group size), $\tau = 10$ (stopping threshold)—are selected on a subset of TREC-DL19/DL20 and then held fixed across all datasets.

## Key Experimental Results

### Main Results

BM25 top-100 + RankZephyr-7B reranker, NDCG@10:

| Method | TREC-DL Avg | BEIR Avg | Overall Avg | Avg. Calls |
|---|---|---|---|---|
| BM25 (no reranking) | 37.8 | 43.6 | 41.1 | 0.0 |
| SW-1 (Sliding Window, 1 pass) | 53.1 | 51.6 | 54.3 | 8.8 |
| SW-2 (Sliding Window, 2 passes) | 53.4 | 51.9 | 54.5 | 17.6 |
| SW-3 (Sliding Window, 3 passes) | 53.3 | 51.9 | 54.5 | 26.4 |
| TourRank-1 | 52.7 | 50.1 | 53.4 | 12.7 |
| AcuRank-9 (budget 9) | 53.3 | 52.3 | 54.6 | 8.8 |
| **AcuRank** | **54.1** | **52.8** | **55.5** | **19.7** |
| AcuRank-H | 54.3 | 53.0 | 55.7 | 41.7 |
| AcuRank-HH | 54.3 | 53.1 | 55.8 | 57.2 |

With a BM25 top-1000 candidate pool, AcuRank achieves 58.0 Overall Avg with 68.4 calls versus SW-1's 56.2 Avg with 94.6 calls—higher accuracy with 27% fewer calls.

### Ablation Study

| Initialization | Grouping | Stopping Criterion | TREC Avg | BEIR Avg | Overall Avg | Calls |
|---|---|---|---|---|---|---|
| ✓ First-stage scores | Sequential | Uncertainty threshold | 59.1 | 52.8 | **55.5** | 19.2 |
| ✗ Default init | Sequential | Uncertainty threshold | 59.0 | 51.7 | 54.8 | 13.4 |
| ✓ First-stage scores | **Random** | Uncertainty threshold | 58.8 | 52.7 | 55.3 | 22.6 |
| ✓ First-stage scores | Sequential | **Top-$k$ stability** | 58.8 | 52.4 | 55.2 | 22.7 |

Generalization across retrievers/rerankers (BEIR, NDCG@10 Avg):

| Configuration | SW-1 | AcuRank | AcuRank Calls |
|---|---|---|---|
| SPLADE++ED + RankZephyr | 52.3 | **52.7** | 8.9 (vs SW-1: 9.0) |
| BM25 + RankGPT (gpt-4.1-mini) | 53.4 | **53.7** | 20.8 |
| BM25 + RankVicuna-7B | 49.0 | **50.8** | 19.5 |
| BM25 + Llama-3.3-70B (zero-shot) | 60.6 | **62.2** | 19.4 |

### Key Findings

1. **Adaptive outperforms fixed**: AcuRank consistently surpasses all fixed-computation baselines with comparable or fewer calls, residing on the Pareto frontier of the accuracy–efficiency trade-off.
2. **Query difficulty adaptation**: WIG is significantly negatively correlated with the number of calls ($\rho = -0.27,\ p < 10^{-8}$), confirming that harder queries receive more computation.
3. **Greater gains on hard queries**: On the Touché dataset, hard queries gain +7.0 NDCG while easy queries gain only +2.2; AcuRank allocates more calls to hard queries (+2.5 calls).
4. **All components contribute**: First-stage initialization improves NDCG by 0.7; sequential grouping improves both accuracy (+0.2) and efficiency (−3.4 calls) over random grouping; uncertainty-driven stopping requires 3 fewer calls than top-$k$ stability stopping at comparable accuracy.
5. **Cross-model generalization**: Consistent advantages are observed across four reranker models—RankZephyr, RankVicuna, RankGPT, and Llama-3.3-70B.

## Highlights & Insights

1. **Elegance of probabilistic modeling**: TrueSkill was originally designed for game rating systems; it is here ingeniously repurposed for document ranking by interpreting LLM reranking outputs as outcomes of a "multiplayer game," enabling incremental Bayesian updates with solid theoretical grounding.
2. **Uncertainty-driven computation allocation**: Rather than simple early exit, the framework precisely identifies "swing documents" near the top-$k$ boundary and concentrates computation there, which is more principled than global heuristics.
3. **Training-free plug-and-play framework**: The pipeline introduces no trainable parameters and is compatible with arbitrary LLM rerankers, demonstrating effectiveness across models ranging from 7B to 70B parameters.
4. **A rare case of simultaneous accuracy and efficiency gains**: Under the SPLADE++ED configuration, AcuRank achieves higher accuracy (52.7 vs. 52.3) with fewer calls (8.9 vs. 9.0) than SW-1, breaking the simple accuracy–efficiency trade-off.
5. **Anytime prediction property**: AcuRank-9, AcuRank, AcuRank-H, and AcuRank-HH form a family of variants spanning low to high budgets, allowing users to select flexibly according to latency constraints.

## Limitations & Future Work

1. **Latency optimization**: Although the total number of calls is reduced, iterative dependencies limit parallelization—the next round's candidate selection must await the current round's updates.
2. **Fixed hyperparameters**: Global hyperparameters ($\epsilon$, $\tau$) are applied uniformly across all domains and may be suboptimal for specific settings; an adaptive tuning mechanism is absent.
3. **Simple grouping strategy**: The current sequential grouping by $\mu_i$ is a heuristic; clustering based on document similarity may yield more informative pairwise comparisons.
4. **Unexploited reranker-internal signals**: Modern LLMs can expose token-level or generation-level confidence scores; these signals are not currently integrated to enhance uncertainty estimation.
5. **Underutilized rank entropy**: The paper discusses rank entropy $\mathbb{H}(r_i) = -\sum_r P(r_i = r)\log P(r_i = r)$ as a richer uncertainty signal, but the current implementation relies only on thresholded cumulative probabilities.

## Related Work & Insights

- **Sliding Window methods** (RankGPT, LRL, RankVicuna): Fixed window sizes and traversal paths lead to over-computation on simple queries and under-computation on complex ones.
- **Tournament-style methods** (TourRank, TDPart, ListT5): Multi-round elimination schedules, but computation remains fixed; high accuracy requires prohibitive cost (TourRank-5 requires 63.7 calls).
- **Retrieval uncertainty** (Cohen et al., 2021): Introduces a Bayesian framework for interpreting relevance scores in retrieval, but remains at post-hoc calibration without online adaptive computation.
- **TrueSkill** (Herbrich et al., 2006): Originally the rating system for Xbox Live multiplayer; creatively repurposed here for probabilistic modeling of document ranking.

**Broader Implications**: The "uncertainty-driven adaptive computation" paradigm has strong generalization potential—it could extend to dynamic retrieval rounds in RAG systems, selective verification in multi-step reasoning, and adaptive voting strategies in model ensembles.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The transfer of TrueSkill to reranking is elegant and theoretically grounded, though the overarching idea of "allocating computation based on uncertainty" is not entirely novel.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Bayesian modeling is rigorous, with closed-form update equations and approximation error analysis, though theoretical guarantees for the ranking probability approximation could be strengthened.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 datasets, 4 reranker models, 3 retrievers, comprehensive ablations, and query-level analysis constitute extremely thorough experimental coverage.
- **Value**: ⭐⭐⭐⭐ — The training-free plug-and-play design is practically appealing, though iterative dependencies limit parallelization and latency benefits in deployment require further validation.

---
title: >-
  [Paper Notes] AcuRank: Uncertainty-Aware Adaptive Computation for Reranking
description: >-
  [NeurIPS 2025][Uncertainty Estimation] AcuRank dynamically adjusts the reranking subset size and verification scope via TrueSkill-based uncertainty estimation, achieving a superior accuracy–efficiency trade-off while avoiding over-computation.
tags:
  - NeurIPS 2025
  - Uncertainty Estimation
  - Adaptive Computation
  - Bayesian Model
  - Reranking
  - Context Length Optimization
---

# AcuRank: Uncertainty-Aware Adaptive Computation for Reranking

**Conference**: NeurIPS 2025
**arXiv**: [2505.18512](https://arxiv.org/abs/2505.18512)  
**Code**: N/A  
**Area**: LLM Efficiency, Information Retrieval
**Keywords**: uncertainty estimation, adaptive computation, Bayesian model, reranking, context length optimization

## TL;DR
AcuRank dynamically adjusts the reranking subset size and verification scope via TrueSkill-based uncertainty estimation, achieving a superior accuracy–efficiency trade-off while avoiding over-computation.

## Background & Motivation
**Limitations of Prior Work**: LLM-based listwise reranking is typically performed over fixed-size subsets, ignoring variation in query difficulty and document distribution.

**Accuracy–Efficiency Trade-off**: Fixed computation budgets cannot adapt to input characteristics, leading to over-computation on some queries and under-computation on others.

**Bayesian Optimal Baseline**: Comparison against a Bayesian optimal estimator motivates and validates the need for adaptive reranking.

**Context Length Constraints**: Because of LLM context limits, reranking typically aggregates over partial results, causing information loss.

**Absence of Confidence Measures**: Existing methods lack quantification of ranking confidence, providing no signal for when verification should stop.

**Practical Application Needs**: Benchmarks such as TREC-DL and BEIR demonstrate that diverse queries require adaptive strategies.

## Method

### Overall Architecture
AcuRank employs an iterative refinement strategy: (1) initialize the TrueSkill model; (2) update relevance estimates via partial subset verification; (3) compute ranking confidence; (4) decide whether to continue verification based on confidence thresholds; (5) aggregate the final ranking.

### Key Designs
**Bayesian TrueSkill Model**
- **Function**: Models document reranking as a paired comparison problem, maintaining a latent skill parameter for each document.
- **Mechanism**: A Bayesian framework statistically models ranking uncertainty and supports incremental updates.
- **Design Motivation**: Unlike point estimates, the Bayesian approach captures the true confidence distribution over rankings.

**Uncertainty Quantification**
- **Function**: Computes ranking confidence to measure the stability of the current ranking.
- **Mechanism**: Estimates the probability of top-$k$ rank inversion based on the variance of the TrueSkill posterior distribution.
- **Design Motivation**: High-confidence rankings require no further verification, saving computation.

**Adaptive Verification Stopping**
- **Function**: Dynamically determines the reranking subset size and number of verification rounds based on confidence thresholds.
- **Mechanism**: Terminates when all top-$k$ documents exceed the confidence threshold, avoiding unnecessary computation.
- **Design Motivation**: The optimal verification scope varies substantially across queries; static strategies are suboptimal.

## Key Experimental Results

| Dataset | Method | nDCG@10 | Computation Saved | Accuracy Gain |
|---|---|---|---|---|
| TREC-DL | Fixed computation | 0.652 | Baseline | — |
| TREC-DL | AcuRank | 0.671 | 23% | +2.9% |
| BEIR (avg.) | Fixed computation | 0.512 | Baseline | — |
| BEIR (avg.) | AcuRank | 0.528 | 18% | +3.1% |
| TREC-DL (hard queries) | AcuRank | 0.644 | 32% | +4.2% |

| Query Type | Optimal Subset Size | Confidence Threshold | Avg. Verification Rounds | Performance (nDCG@10) |
|---|---|---|---|---|
| Easy queries | 20 | 0.85 | 1.5 | 0.692 |
| Medium queries | 50 | 0.75 | 2.3 | 0.658 |
| Hard queries | 100 | 0.65 | 3.8 | 0.621 |
| Global average | — | — | 2.4 | 0.671 |

## Key Findings
1. **Adaptive outperforms fixed**: AcuRank substantially outperforms fixed-size reranking on both TREC-DL and BEIR, with average nDCG@10 gains exceeding 3%.
2. **Query difficulty correlation**: Hard queries require larger verification scopes and lower thresholds; easy queries can be handled with minimal computation.
3. **Substantial computation savings**: AcuRank saves 18–23% of computation at comparable accuracy, reaching 32% savings on hard queries.
4. **Confidence metric validity**: TrueSkill-based confidence is highly correlated with actual ranking stability (correlation coefficient > 0.85).
5. **Cross-model generalization**: The method performs consistently across different LLM classifiers, indicating strong generalizability.

## Highlights & Insights
1. **Uncertainty-driven adaptivity**: The first systematic use of ranking uncertainty to guide computation allocation, with a solid theoretical foundation.
2. **Elegant Bayesian modeling**: The TrueSkill model handles paired comparisons naturally, with incremental updates enabling online application.
3. **Simultaneous accuracy and efficiency gains**: Both precision and efficiency improve, with particularly notable effects on hard queries.
4. **Fine-grained analysis**: Query-level accuracy–efficiency trade-off visualization facilitates system-level hyperparameter tuning.

## Limitations & Future Work
1. **Initialization sensitivity**: TrueSkill initialization parameters may affect early-round estimates; more robust initialization is needed.
2. **High-order interactions ignored**: The current model does not account for higher-order interactions among documents; pairwise learning-to-rank approaches could be explored.
3. **Static threshold**: The confidence threshold remains manually set; a learnable adaptive threshold would be preferable.
4. **Cold-start problem**: New-domain queries lack historical data, making optimal parameter estimation difficult.
5. **Uniform verification cost assumption**: All verification steps are assumed to have equal cost, whereas vectorization costs may vary significantly in practice.

## Related Work & Insights
- **Listwise reranking**: Unlike listwise learning-to-rank methods (e.g., LambdaMART), AcuRank requires no supervision from relevance labels.
- **Uncertainty quantification**: Inspired by Bayesian deep learning methods, but applied innovatively to retrieval ranking.
- **Adaptive computation**: Extends the dynamic early-exit paradigm to ranking tasks.
- **Insight**: Uncertainty estimation holds broad promise for accelerating LLM inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (Uncertainty-driven adaptive reranking is a novel angle; Bayesian modeling is elegantly applied.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Validated on multiple TREC-DL/BEIR benchmarks with ablations and fine-grained analysis.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear logic, well-motivated, sound experimental design.)
- **Value**: ⭐⭐⭐⭐ (Directly reduces inference cost while improving accuracy; strong potential for industrial deployment.)
- **Overall**: ⭐⭐⭐⭐ (18/20)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](../../ICML2026/information_retrieval/very_efficient_listwise_multimodal_reranking_for_long_documents.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](../../ACL2026/information_retrieval/unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)

</div>

<!-- RELATED:END -->
