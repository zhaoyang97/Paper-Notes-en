---
title: >-
  [Paper Note] Evolutionary Learning in Spatial Agent-Based Models for Physical Climate Risk Assessment
description: >-
  [NeurIPS 2025][geospatial ABM] This paper proposes an Agent-Based Model (ABM) that integrates geospatial climate hazard data with evolutionary learning mechanisms. Using a simplified economic network comprising a three-t…
tags:
  - "NeurIPS 2025"
  - "geospatial ABM"
  - "evolutionary learning"
  - "physical climate risk"
  - "supply chain systemic risk"
  - "RCP8.5"
  - "flood damage functions"
date: 2026-05-08
content_hash: c1a1918ada30f1cb
---

# Evolutionary Learning in Spatial Agent-Based Models for Physical Climate Risk Assessment

**Conference**: NeurIPS 2025
**arXiv**: [2509.18633](https://arxiv.org/abs/2509.18633)
**Code**: [GitHub](https://github.com/yaramohajerani/spatial-climate-ABM) (open source)
**Area**: Other
**Keywords**: geospatial ABM, evolutionary learning, physical climate risk, supply chain systemic risk, RCP8.5, flood damage functions

## TL;DR
This paper proposes an Agent-Based Model (ABM) that integrates geospatial climate hazard data with evolutionary learning mechanisms. Using a simplified economic network comprising a three-tier commodity–manufacturing–retail supply chain, the model simulates economic responses from 2025 to 2100 under RCP8.5 flood projections. Results demonstrate that evolutionary adaptation enables firms to maintain significantly higher levels of production, capital, liquidity, and employment under climate stress, while revealing supply chain systemic risks that traditional asset-level assessments fail to capture.

## Background & Motivation
Climate risk assessment faces three methodological challenges:

1. **Extrapolation to new regimes**: Econometric damage functions fitted to historical data (e.g., temperature–GDP polynomial relationships) are extrapolated to future climate conditions far beyond historical experience, raising concerns about reliability.
2. **Limited functional forms**: Polynomial damage functions cannot capture the dynamic behaviors, threshold effects, and path dependencies of economic agents in changing environments.
3. **Absence of cascade effects**: Static econometric approaches struggle to capture nonlinear tipping points and cascading transmission effects within socioeconomic networks.

Existing ABM approaches also have shortcomings:
- Post-disaster recovery models (Henriet et al., Wenz et al.) focus only on short time horizons (days to weeks) with fixed-parameter responses.
- Integrated assessment ABMs (Lamperti et al.) primarily address transition risks and carbon pricing.
- A Bank of England survey found that most central bank ABMs focus on transition rather than physical risk.

**Gap addressed**: This paper integrates asset-level damage functions, long-term physical risk scenarios, and firm-level evolutionary learning into a unified geospatial ABM framework.

## Core Problem
How to construct a long-term climate risk assessment framework capable of simultaneously capturing direct climate damages and supply chain cascade effects, while evaluating how adaptive strategies endogenously emerge to mitigate climate impacts?

## Method

### Network Architecture
- **Spatial resolution**: 0.25-degree global grid ($1440 \times 720$ cells)
- **Economic network**: Simplified economy with 65 firms and 650 households
    - 20 commodity firms (blue, upstream)
    - 30 manufacturing firms (red, midstream)
    - 15 retail firms (green, downstream)
- **Supply chain structure**: commodity → manufacturing → retail → households, with intra-sector trade permitted
- **Trophic Level**:

$$TL_i = 1 + \sum_j w_{ij} \cdot TL_j, \quad w_{ij} = \frac{I_{ij}}{\sum_k I_{ik}}$$

### Acute Risk Modeling
- **Data source**: RCP8.5 riverine flood depth projections from the WRI Aqueduct database
- **Damage calculation**: JRC global flood depth–damage functions matched by sector and location
- **Damage transmission**: (1) reduction in capital stock, (2) temporary productivity loss (50% recovery per step), (3) inventory destruction

### Production Function
A Leontief production function is adopted:

$$Q = \min\left(\frac{L}{\alpha_L^s}, \frac{I}{\alpha_I^s}, \frac{K}{\alpha_K^s}\right) \cdot \phi$$

where $\phi \in [0,1]$ is the post-flood productivity damage factor. Sector-specific technical coefficients:

| Sector | $\alpha_L$ | $\alpha_I$ | $\alpha_K$ | Characteristics |
|--------|-----------|-----------|-----------|-----------------|
| Commodity | 0.6 | 0.0 | 0.7 | Capital-intensive, no upstream inputs |
| Manufacturing | 0.3 | 0.6 | 0.6 | Highly automated, high intermediate input demand |
| Retail | 0.5 | 0.4 | 0.2 | Moderately labor-intensive, low capital |

### Household Behavior
- **Labor supply**: Utility maximization $U_h = w_f - \delta \cdot d_{hf}$ (wage minus commuting cost)
- **Consumption budget**: $B_c = L_h \cdot \bar{w} + \max(0, M_h - 50) \cdot 0.1$, capped at 80% of current liquidity
- **Consumption allocation**: commodity 25% / manufacturing 45% / retail 30%
- **Migration**: Households migrate to lower-risk areas when flood depth exceeds 0.5 m

### Wage Dynamics
A persistent shortage mechanism prevents wage–price spirals:

$$w_{t+1} = \begin{cases} w_t \cdot (1 + \gamma_w \cdot \eta_1) & \text{persistent shortage for 4 periods and } M_f \geq 2w_t \\ w_t \cdot (1 - \gamma_w \cdot \eta_2) & \text{not labor-constrained and } u > 0.2 \\ w_t & \text{otherwise} \end{cases}$$

### Evolutionary Learning System

Each firm maintains **5 evolvable parameters**:
- Budget weights $\beta_L, \beta_I, \beta_K$: allocation multipliers for labor, inputs, and capital
- Risk sensitivity $\gamma_r$: scaling factor for increased capital demand following a disaster
- Wage responsiveness $\gamma_w$: scaling factor for wage adjustment magnitude

**Fitness function**: 10-step rolling window, four-dimensional weighted sum:
- Liquidity growth (log + tanh transformation, 35%)
- Production level (25%)
- Peak maintenance (20%, asymmetric — recovering firms are not penalized)
- Survival bonus (20%)

**Evolutionary mechanism**:
- **Individual mutation** (every 5 steps): fitness improvement → small mutation (2.5% std), fitness decline → large mutation (10% std)
- **Population replacement** (every 10 steps): firms whose liquidity falls below the survival threshold are replaced by offspring of successful firms, inheriting strategies with per-parameter mutation

## Key Experimental Results

### Four-Scenario Comparison (2025–2100, RCP8.5, semi-annual steps)

| Metric | Baseline + Learning | Baseline + No Learning | Disaster + Learning | Disaster + No Learning |
|--------|--------------------|-----------------------|--------------------|-----------------------|
| Production (post-2080 mean) | High & stable | Moderate | **4.3±2.4** | 1.0±0.6 |
| Capital (last decade) | 90.8±3.7 | 60.6±3.5 | **28.8±5.8** | 1.6±0.3 |
| Liquidity (last decade) | $901±90 | $814±12 | **$491±43** | $71±3 |
| Employment rate (last decade) | ~1.0 | ~1.0 | 0.45–1.0 | **0.2±0.2** |
| Wages (last decade) | $9.3±0.2 | $8.1±0.2 | $2.6±0.2 | $0.42±0.03 |
| Prices (last decade) | Moderate | $35.8±3.4 | **Below baseline** | $42.2±1.7 |

### Key Findings

**Production maintenance**: Disaster + learning firms maintain 4.3 units of production after 2080 (vs. 1.0 for no-learning), demonstrating that evolutionary learning enables firms to adapt to changing climate conditions and sustain productive capacity over time.

**Capital dynamics**: Learning firms exhibit high-volatility capital trajectories (flood damage → adaptive rebuilding cycles), yet still reach 28.8 units in the final decade (vs. 1.6 for no-learning).

**Systemic risk exposure**: Even when few firms are directly exposed to floods, supply chain shocks propagate throughout the network and affect firms with no direct hazard exposure — a type of risk that traditional asset-level assessments would severely underestimate.

**Bottleneck evolution**: In the disaster + no-learning scenario, firms are primarily constrained by **labor bottlenecks** (not due to labor unavailability, but to insufficient liquidity to pay wages); in the learning scenario, the binding constraint shifts to **input bottlenecks**.

**Unexpected finding — lower prices under disaster + learning**: Learning firms under disaster conditions exhibit prices even lower than the baseline no-learning scenario, because the decline in supply is more manageable than the decline in demand.

## Highlights & Insights
- **Framework completeness**: Geospatial hazard data, asset-level damage functions, multi-tier supply chains, and evolutionary strategy learning are integrated into a unified framework.
- **Quantitative demonstration of systemic risk**: Bottleneck analysis clearly shows how indirect and cascading risks dominate total damages.
- **Emergent effects of evolutionary learning**: Adaptive strategies emerge without explicit programming, revealing cost-effective adaptation pathways that static models cannot identify.
- **Practical application orientation**: The framework is designed as a climate risk portfolio assessment tool for financial institutions and corporations.
- **Open-source implementation**: Full code is publicly available, facilitating follow-up research and application.

## Limitations & Future Work
- **Highly simplified economic network**: A global network of only 65 firms is a proof-of-concept; real firm-level or sector-level input–output data are needed for calibration.
- **Single hazard type**: Only RCP8.5 riverine flooding is considered; other acute and chronic hazards such as heat waves, droughts, and hurricanes are excluded.
- **Single damage function**: Only the JRC function is used; alternative damage function frameworks may yield different conclusions.
- **Simplified agent behavior**: Although an evolutionary mechanism is included, the initial strategy space (5 parameters) may be overly restrictive.
- **Lack of historical calibration**: No historical data on firm responses to disasters are available for validation.
- **Fitness function weight selection**: The 35/25/20/20% weights are set heuristically; sensitivity analysis remains to be completed.
- **No financial system modeling**: The absence of banks, credit, and insurance intermediaries limits the framework's ability to assess financial climate risks.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of geospatial ABM, evolutionary learning, and long-term climate scenarios is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐ The proof of concept is clear, but the small economic network scale limits the credibility of quantitative conclusions.
- Writing Quality: ⭐⭐⭐⭐ Methods are described in detail with complete equations; the four-scenario comparison is systematically presented.
- Value: ⭐⭐⭐⭐ Fills an important gap in long-term ABM assessment of physical climate risk, with a clear application pathway.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergency Response Measures for Catastrophic AI Risk](emergency_response_measures_for_catastrophic_ai_risk.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](evolutionary_prediction_games.md)
- [\[ICLR 2026\] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](../../ICLR2026/others/completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](../../ICLR2026/others/neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)

</div>

<!-- RELATED:END -->
