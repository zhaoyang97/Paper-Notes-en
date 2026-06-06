---
title: >-
  [Paper Note] Competition-Aware CPC Forecasting with Near-Market Coverage
description: >-
  [CVPR 2026][Time Series][CPC forecasting] This paper reformulates CPC forecasting in search advertising as a time series prediction problem under partially observable competition states. Three observable proxies — semant…
tags:
  - "CVPR 2026"
  - "Time Series"
  - "CPC forecasting"
  - "search advertising auction"
  - "competition proxy"
  - "spatiotemporal graph network"
  - "time series foundation model"
date: 2026-05-08
content_hash: 2b84cc76fdeec389
---

# Competition-Aware CPC Forecasting with Near-Market Coverage

**Conference**: CVPR 2026
**arXiv**: [2603.13059](https://arxiv.org/abs/2603.13059)  
**Code**: None  
**Area**: Time Series
**Keywords**: CPC forecasting, search advertising auction, competition proxy, spatiotemporal graph network, time series foundation model

## TL;DR
This paper reformulates CPC forecasting in search advertising as a time series prediction problem under partially observable competition states. Three observable proxies — semantic similarity, CPC trajectory alignment, and geographic intent — are constructed to approximate latent competition, and are subsequently injected into forecasters as covariates and graph priors respectively. The proposed framework achieves substantial improvements over purely autoregressive baselines on medium- and long-term forecasting horizons.

## Background & Motivation
CPC in search advertising is not a stable business metric but an outcome variable of the auction process. For advertisers, it directly determines how many clicks a given budget can purchase; consequently, forecasting errors quickly translate into media planning deviations, budget waste, and profit compression.

Existing CPC forecasting methods face a fundamental difficulty: advertisers can only observe their own impressions, clicks, spend, and the resulting CPC, but have no visibility into competitors' bids, quality scores, budget pacing, or the platform's complete internal auction state. In other words, the "competitive environment" that actually governs price formation is a latent variable.

The authors argue that many prior approaches fail on medium-to-long horizons not because of insufficient model capacity, but because their inputs lack structured signals capable of reflecting competitive dynamics. Purely autoregressive methods excel at extrapolating short-term inertia, yet when competitors shift budgets, local market demand rises, or keyword substitution patterns change, univariate history alone is insufficient to maintain prediction stability.

The paper's logical chain is clear:

**Background**: Research on search advertising has devoted considerable attention to GSP auctions, ranking mechanisms, and CTR modeling, but the problem of forecasting future CPC from the perspective of an individual advertiser remains understudied.

**Limitations of Prior Work**: What is observed are outcomes, not competitive causes; yet CPC is highly sensitive to competitive dynamics.

**Key Challenge**: The true competitive state is not directly observable, but it leaves traces in observable variables.

**Goal**: To determine whether a set of high-quality proxy signals can make implicit competitive structure explicit, enabling forecasting models to absorb it.

**Key Insight**: Competition is approximated from three complementary perspectives — keyword semantic substitutability, historical CPC trajectory behavioral similarity, and local market structure captured by geographic intent.

**Core Idea**: Rather than inventing a fundamentally new forecasting architecture, the paper first constructs "competition" as priors and covariates usable by different model families, enabling more stable medium-to-long-term prediction under partial observability.

The motivation presented in this paper is notably well-grounded. Rather than packaging contributions as a universal large model or a novel GNN, the authors explicitly acknowledge that the genuine novelty lies in proxy variable construction and the choice of proxy representation. This yields stronger business interpretability than work that merely substitutes a different backbone.

## Method
The proposed method is not a single model but a competition-aware forecasting design framework. The core procedure consists of two steps: distilling competition proxies from observable data, and deciding in what form to feed these proxies to the forecaster.

The forecasting target is multi-step weekly CPC prediction over 1,811 keywords, with forecasting horizons $h \in \{1, 6, 12\}$ weeks. The authors do not attempt to explicitly reconstruct the true auction mechanism; instead, they emphasize that sufficiently stable proxies that closely approximate competitive structure are adequate to improve forecasting.

### Overall Architecture
The overall pipeline can be summarized in five steps:

1. Construct a keyword-level weekly panel from Google Ads logs spanning 2021 to 2023.
2. Generate three categories of competition proxies: semantic proxies, behavioral proxies, and geographic proxies.
3. Route the proxies into models via two pathways:
    - Covariate route: organize proxies into leakage-free exogenous features.
    - Relational prior route: encode inter-keyword competition relationships as a fixed semantic graph.
4. Compare three model families — traditional/neural baselines, time series foundation models (TSFMs), and spatiotemporal graph neural networks (STGNNs) — under a unified evaluation protocol.
5. Analyze performance across the 1-week, 6-week, and 12-week horizons and within the high-risk competitive frontier subset.

The framework input is not simply historical CPC but a richer weekly panel encompassing clicks, impressions, spend, device mix, search type mix, and competition signals derived from keyword text and trajectories. The output is predicted CPC values for future weeks.

### Key Designs
The most important contributions of the paper are not specific formulas but the construction of three proxy categories and two representation pathways.

1. **Semantic Neighborhood and Semantic Keyword Graph**
    - Function: Approximate the set of keywords likely competing for the same traffic using semantic similarity of keyword text.
    - Mechanism: Each keyword is encoded into a 384-dimensional vector $e_i \in \mathbb{R}^{384}$ using all-MiniLM-L6-v2, and cosine similarity is computed to identify semantic neighbors. Each keyword is connected to its $k=10$ most similar neighbors to form a fixed semantic graph, with the adjacency matrix row-normalized.
    - Design Motivation: In ad auctions, true competitive relationships often do not arise from surface-level string matching but from substitutable search intent. For example, two differently worded "airport car rental" keywords may compete for the same high-intent traffic. Text semantics can recover this substitutability.
    - Novelty: Many spatiotemporal graph models rely on naturally available physical graphs such as road networks or power grid topologies. Since no such topology exists here, the authors actively transform "semantic substitutability" into graph structure.

2. **DTW-Based Behavioral Neighborhood**
    - Function: Identify keywords whose historical CPC trajectories exhibit similar dynamics, potentially with temporal misalignment.
    - Mechanism: Dynamic Time Warping (DTW) with a Sakoe-Chiba band constraint is used to measure pairwise CPC sequence similarity, avoiding pathological alignments. The resulting behavioral neighborhood is leakage-free, relying solely on historical trajectory statistics to form behavioral competition features.
    - Design Motivation: Some keywords are textually dissimilar yet co-move under seasonal effects, budget adjustments, or market shocks. Text-based proxies cannot capture such co-movement; trajectory alignment supplements this by identifying keywords with correlated competitive exposure.
    - Relationship to Semantic Neighborhood: The semantic proxy reflects static substitution relationships, while the DTW behavioral proxy captures dynamic co-resonance; the two are complementary rather than redundant.

3. **Geographic Intent Proxy**
    - Function: Extract hierarchical geographic information (continent, country, city) from keyword text as a proxy for local demand and competitive heterogeneity.
    - Mechanism: Keyword text is combined with a geographic lexicon and hierarchical mapping to assign multi-scale geographic labels to each keyword, effectively transforming "geographic attribution of user intent" into structured variables.
    - Design Motivation: Search demand in the car rental industry is highly localized; differences in airport, city, or country correspond to entirely different demand intensities and competitive densities. Geographic structure inherently determines local market congestion.
    - Empirical Finding: Finer geographic granularity does not necessarily improve performance; coarser granularity such as continent proves more stable because it avoids fragmenting training samples excessively.

4. **Two Proxy Representation Pathways**
    - Function: Decouple the same competition proxies from any single model, injecting them as either covariates or graph priors.
    - Mechanism: The covariate route feeds neighborhood history aggregates, geographic variables, and core operational variables jointly into TSFMs or traditional models. The graph route encodes semantic keyword edges as a fixed graph structure for STGNNs, enabling cross-keyword information propagation.
    - Design Motivation: The authors seek to answer not which backbone is strongest, but which competition representation is most useful. This requires decoupling representation choice from model family for a controlled comparison.
    - Implication: If the covariate route proves superior, it suggests competition proxies are better suited as exogenous conditions; if the graph route dominates, it suggests relational propagation itself is the key mechanism. The paper finds each pathway holds advantages at different horizons.

5. **Competitive Frontier Evaluation**
    - Function: Segment keywords into four quadrants by mean CPC and volatility, with focused analysis on the high-risk upper-right quadrant of expensive and highly volatile keywords.
    - Mechanism: Mean CPC represents value and coefficient of variation represents volatility. The paper emphasizes the frontier region of high CPC and high volatility.
    - Design Motivation: Average error fails to reflect business risk; what must be stabilized are expensive, volatile keywords, because forecasting errors on these terms inflict the greatest budget damage.
    - Value: This step connects prediction improvements directly to business risk rather than merely reporting average sMAPE.

### Loss & Training
The training and evaluation setup is largely unified, with emphasis on preventing temporal leakage and accommodating heavy-tailed distributions:

- Data is split strictly in temporal order, with the final 20% reserved as out-of-sample test.
- Forecasting horizons are 1 week, 6 weeks, and 12 weeks, corresponding to short-term bid adjustment, medium-term tactical planning, and long-term budget allocation respectively.
- The primary evaluation metric is sMAPE; RMSE serves as a secondary metric.
- STGNNs are trained globally across the keyword panel in a joint learning setup.
- Graph models use MAE as the optimization objective, since the CPC distribution exhibits pronounced heavy tails that make squared-error objectives susceptible to domination by extreme values.

The paper does not emphasize complex loss engineering; the primary focus is on input structure design. This reflects the authors' belief that, for this problem, correctly representing competitive information matters more than adopting a more sophisticated training objective.

## Key Experimental Results
The data construction is central to the paper, as it motivates the need for competition-aware modeling.

- Raw log scale: approximately 1.66 billion records from Google Ads car rental industry data spanning 2021 to 2023.
- Each record includes keyword, matched query, landing page URL, device type, search type, and numeric metrics including impressions, clicks, and cost.
- After vertical filtering, domain quality filtering, and keyword normalization, 1,811 keyword time series are retained.
- Each keyword is required to appear in at least 110 of the 127-week window to avoid spurious signals from extremely short-lived keywords.
- Weekly aggregation yields 218,924 keyword-week observations.
- Weekly CPC is defined as $\mathrm{CPC}_{k,t} = \frac{\mathrm{cost}_{k,t}}{\mathrm{clicks}_{k,t}}$, computed only when clicks exceed zero.
- Mean CPC is 2.86, maximum is 80.16, p99 is 12.13, and skewness is 3.34, confirming a pronounced heavy-tailed price distribution.
- The competitive frontier high-risk quadrant contains 402 keywords, constituting the primary region of business-critical analysis.

### Main Results
The following family-level summary across horizons illustrates the distinct roles of different model families at different planning horizons.

| Forecasting Horizon | Best Traditional/ML Baseline sMAPE | Best TSFM sMAPE | Best STGNN sMAPE | Conclusion |
|---|---|---|---|---|
| 1 week | 30.42 | 27.94 | **25.82** | Graph models dominate at short horizons; local competition propagation aids immediate forecasting |
| 6 weeks | 35.04 | **27.14** | 30.42 | TSFMs with competition covariates are most stable at medium horizons |
| 12 weeks | 40.23 | **29.14** | 37.46 | TSFMs lead substantially at long horizons; graph structure advantage weakens |

The authors further expand the 6-week horizon — the most business-critical — as competition-aware designs produce the largest differentiation at this timescale.

| Model Family | Architecture | Best Competition-Aware Configuration | sMAPE | RMSE |
|---|---|---|---|---|
| Statistical/ML | SARIMAX | Univariate lags | 43.93 ± 23.55 | 1.660 ± 1.759 |
| Statistical/ML | XGBoost | Core operational features | 36.64 ± 17.51 | 1.301 ± 1.119 |
| Statistical/ML | TabPFN | Core operational features | 35.04 ± 17.77 | 1.250 ± 1.133 |
| TSFM | Moirai | Leakage-free lags + calendar stabilization | 30.14 ± 18.24 | 1.000 ± 0.970 |
| TSFM | TimeGPT | Calendar conditioning + growth clamp | 29.29 ± 17.07 | 1.002 ± 1.008 |
| TSFM | **Chronos-2** | **Geographic intent covariates** | **27.14 ± 15.04** | **0.841 ± 0.846** |
| STGNN | GraphWaveNet | Semantic graph + search mix | 30.57 ± 20.57 | 1.005 ± 0.941 |
| STGNN | GConvLSTM | Semantic graph + continental geography | 30.69 ± 20.42 | 1.001 ± 0.955 |
| STGNN | DCRNN | Semantic graph + geography + semantic neighborhood CPC | 30.42 ± 20.42 | 1.000 ± 0.926 |

Three observations emerge from this table:

- Pure baselines have clearly plateaued, with the best reaching only 35.04.
- Competition covariates yield the largest gains for TSFMs, with Chronos-2 + geographic intent reducing the 6-week sMAPE to 27.14.
- Although graph models do not outperform Chronos-2 at 6 weeks, they systematically outperform non-graph baselines, confirming that relational priors carry meaningful information.

### Ablation Study
The most informative ablations are not architectural layer removals but comparisons of different competition proxy effectiveness and granularity choices.

| Configuration | Horizon | Key Metric | Description |
|---|---|---|---|
| Core only | 6 weeks | 31.61 sMAPE | Reference for graph model with core inputs only |
| **Core + Geo + Sem CPC** | 6 weeks | **30.71 sMAPE** | Best 6-week configuration; geographic proxy and semantic neighborhood CPC are complementary |
| All proxies naive stacking | 6 weeks | 34.0 sMAPE | Indiscriminate feature stacking performs worst, 3.3 points below the best configuration |
| Core only | 12 weeks | 38.32 sMAPE | Reference baseline for long-horizon forecasting |
| **Core + Continent** | 12 weeks | **37.93 sMAPE** | Coarse geographic prior is most stable at long horizons |
| All proxies naive stacking | 12 weeks | 43.13 sMAPE | 5.2 points below the best configuration; more features do not monotonically help |

The authors also compare geographic granularities, demonstrating the principle that coarse priors are more robust than fine-grained ones.

| Geographic Resolution | 1-week sMAPE | 6-week sMAPE | 12-week sMAPE | Interpretation |
|---|---|---|---|---|
| **Continent (7 dummies)** | **26.36** | **30.90** | **37.93** | Most stable; balances structural information with sample density |
| Country (63 dummies) | 26.72 | 31.51 | 38.70 | Finer information but fragmented samples reduce robustness |
| City (268 dummies) | 27.16 | 31.82 | 39.04 | Excessive granularity amplifies noise and sparsity issues |

### Key Findings
- **Finding 1**: Competition proxies are not optional auxiliary information but a determining factor for medium-to-long-horizon prediction performance. Gains at 6 and 12 weeks substantially exceed those at 1 week, indicating that proxies primarily operate by addressing regime shifts and local market changes.
- **Finding 2**: Different horizons favor different representation forms. STGNNs dominate at 1 week, reflecting greater dependence on immediate cross-keyword relational propagation; TSFMs dominate at 6 and 12 weeks, where stable exogenous priors are needed to suppress long-term drift.
- **Finding 3**: Geographic proxies prove more impactful than the authors initially anticipated from text-based intuition. The strongest overall result comes from Chronos-2 + geographic intent, not from any complex graph structure.
- **Finding 4**: Selective augmentation outperforms exhaustive stacking. Adding all proxies indiscriminately degrades performance, demonstrating that in noisy auction environments, high-quality priors must be used selectively.
- **Finding 5**: Improvements concentrate in the competitive frontier region of high CPC and high volatility. The authors report that at 6 weeks, Core + Geo + Sem CPC reduces the error of this high-risk subset by 1.3 percentage points relative to Core only.

## Highlights & Insights
- The paper's greatest strength is the precision of its problem formulation. Rather than treating CPC forecasting as a generic time series task, it explicitly characterizes the problem as forecasting under a partially observable competitive system, a framing that directly shapes the method design.
- The three proxy categories are well-layered. Semantic proxies capture substitutability, DTW proxies capture behavioral co-movement, and geographic proxies capture local market structure — each corresponding to a distinct source of competitive traces in observable data.
- The "proxy representation pathway" perspective is particularly instructive. Whereas most work only compares backbone architectures, this paper isolates the question of whether the same information is more effective as a covariate or as a graph prior, yielding a cleaner research design.
- The competitive frontier evaluation methodology is practically valuable. In advertising, average error is not the sole objective; the genuinely dangerous cases involve expensive and volatile keywords. Focusing the evaluation on these keywords produces results that more closely reflect real media planning decisions.
- One particularly instructive finding is that coarse geographic granularity outperforms fine granularity, reminding practitioners that in commercial time series settings, priors need not be finer to be better — what matters is whether they form stable statistical signals within finite samples.
- The paper also delivers an important engineering insight: in high-noise auction environments, the key to effective feature engineering is not quantity but precision. This carries broad implications for industrial forecasting.

## Limitations & Future Work
- The authors acknowledge that the data originates from a single vertical — car rental — in a relatively concentrated market. Conclusions may not directly generalize to industries with more competitive participants or more dispersed query intent.
- The semantic graph is static and cannot represent the fact that keyword substitution relationships change dynamically with seasons, events, or competitor strategy shifts — an important limitation given the dynamic nature of advertising markets.
- Because the framework is built on data visible to a single advertiser, even the best proxies remain approximations and cannot be equivalent to the true auction state.
- The behavioral neighborhood constructed via DTW is computed offline and statically; the paper does not explicitly model how neighborhood relationships evolve through rolling updates over time.
- The paper focuses primarily on forecasting error and does not close the loop to actual bidding or budget allocation outcomes, leaving the business decision loop incomplete.

Promising directions for future work include:

1. Upgrading the fixed semantic graph to a dynamic graph whose edge weights evolve with time and market conditions.
2. Explicitly distinguishing substitution relationships from complementary relationships on the graph, rather than using a single similarity metric for all edges.
3. Incorporating event signals or external demand signals such as travel peak seasons, holidays, and airport traffic volumes to further stabilize long-horizon forecasting.
4. Directly evaluating whether more accurate CPC forecasts translate into superior campaign ROI, connecting the forecasting task to downstream decision optimization.

## Related Work & Insights
- **vs. Traditional CPC Forecasting Methods**: Traditional approaches typically rely on autoregression, tree-based models, or a small number of business features, implicitly assuming that the current series contains sufficient information; this paper argues that single-series observations are incomplete and that competitive structure must be explicitly supplemented.
- **vs. Common TSFM Applications**: Many TSFM papers demonstrate that large models can perform zero-shot or few-shot forecasting but do not investigate which structured covariates should be provided. This paper's contribution lies in demonstrating that foundation models do not inherently understand competitive relationships and still require well-designed exogenous proxies.
- **vs. Common STGNN Work**: Traffic and weather tasks typically come with natural spatial topologies; advertising CPC does not. This paper provides the insight that, as long as business "substitutability" can be converted into a stable relational graph, STGNNs can be transferred to commercial auction systems.
- **Research Insight 1**: In many industrial problems, what is genuinely lacking is not a deeper network but the ability to convert invisible mechanisms into observable proxies. This paradigm transfers to recommendation, bidding, supply chain, and financial market forecasting.
- **Research Insight 2**: Evaluation should prioritize high-risk subsets rather than relying solely on global averages. For heavy-tailed data, subgroup robustness often carries greater decision value than aggregate metrics.
- **Research Insight 3**: Decoupling "proxy construction" from "representation choice" as separate ablation dimensions facilitates clearer attribution of performance gains and provides more actionable guidance for downstream system design.

## Rating
- Novelty: ⭐⭐⭐⭐ — Explicitly modeling CPC forecasting as a partially observable competition problem and systematically comparing proxy construction with representation pathways constitutes a genuinely novel framing.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Coverage spans traditional models, TSFMs, STGNNs, multiple horizons, and competitive frontier analysis, yielding a fairly comprehensive experimental scope; cross-industry generalization remains unaddressed.
- Writing Quality: ⭐⭐⭐⭐ — Motivation, problem framing, and experimental conclusions are articulated clearly; readers can readily identify what the authors set out to demonstrate.
- Value: ⭐⭐⭐⭐☆ — Highly practical for high-value advertising deployment scenarios, and offers broader methodological inspiration for industrial forecasting tasks through the "proxy recovery of latent variable structure" paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](../../AAAI2026/time_series/recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[NeurIPS 2025\] Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](../../NeurIPS2025/time_series/improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](../../ICML2026/time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)

</div>

<!-- RELATED:END -->
