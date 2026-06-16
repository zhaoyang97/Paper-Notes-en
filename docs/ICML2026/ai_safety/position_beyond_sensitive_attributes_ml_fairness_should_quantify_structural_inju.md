---
title: >-
  [Paper Note] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants
description: >-
  [ICML 2026][AI Safety][Paper Note] This is an ICML position paper: the authors argue that ML fairness research must move beyond focusing solely on "sensitive attributes" like race/sex and instead incorporate "social determinants" (contextual variables such as neighborhood, ADI, school funding, healthcare accessibility) into auditing. Using a theoretical
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: c3f9a821d24623de
---
# Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants

**Conference**: ICML 2026  
**arXiv**: [2508.08337](https://arxiv.org/abs/2508.08337)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Algorithmic Fairness / Position Paper  
**Keywords**: Algorithmic Fairness, Structural Injustice, Social Determinants, Sensitive Attributes, Causal Fairness

## TL;DR
This is an ICML position paper: the authors argue that ML fairness research must move beyond focusing solely on "sensitive attributes" like race/sex and instead incorporate "social determinants" (contextual variables such as neighborhood, ADI, school funding, healthcare accessibility) into auditing. Using a theoretical model for university admissions, U.S. Census data, and semi-synthetic breast cancer screening experiments, they demonstrate that mitigation strategies centered only on sensitive attributes may inadvertently create new forms of structural injustice.

## Background & Motivation

**Background**: Current ML fairness literature almost equates "injustice" with "discrimination along sensitive attributes." Most fairness metrics (Demographic Parity, Equal Opportunity, Conditional Demographic Parity, Causal Path Effects, etc.) require a pre-specified sensitive attribute $A$ (race/sex/age) and then mandate that predictions or decisions be decoupled from $A$ or satisfy some conditional independence. Reference datasets like Adult, Folktables, and Communities and Crimes often intentionally discard contextual fields such as address or geolocation.

**Limitations of Prior Work**: Interdisciplinary literature (political philosophy, sociology, public health) has long identified that what truly shapes individual opportunities and outcomes are social determinants—contextual variables like neighborhood deprivation, school funding, air pollution, distance to hospitals, and community resources. These variables create heterogeneity within the same demographic group (e.g., the median annual income of African American women drops from \$38k to \$18.8k across different PUMAs) and impose shared burdens across different groups (non-URM and URM applicants in impoverished areas face identical community disadvantages). Focusing exclusively on sensitive attributes erases both types of structural signals.

**Key Challenge**: Sensitive attributes are individual-level, (quasi-)stable intrinsic identifiers, whereas social determinants are context-level, structural variables that drift across space and time. Existing individual-level causal graphs ($A \to Y$, $A \to M \to Y$) and de-sensitization losses cannot accommodate community-level structures like "neighborhood-individual" bidirectional influences or community aggregate statistics, resulting in the normalization of context as noise.

**Goal**: To establish "social determinants" as first-class auditing objects by addressing three questions: (i) How to conceptually distinguish social determinants from sensitive attributes and proxies? (ii) Why do current technical paradigms fail to support them? (iii) What new structural injustices are caused by mitigation strategies that focus solely on sensitive attributes?

**Key Insight**: Starting from a specific scenario—historical redlining forced Black families into specific communities, causing long-term correlations between race, zip code, and community ethnic composition. However, these three differ fundamentally in fairness implications: zip code is an administrative label that cannot be "improved," while school funding and air quality are actionable structural variables. The authors classify them using three criteria: context-level definition, social-structural content, and exogenous stratification.

**Core Idea**: Auditing must precede mitigation. Before "fixing" models, structural injustice must be explicitly quantified via social determinants. Otherwise, blindly applying racial quotas might push the most disadvantaged sub-groups (e.g., non-URM individuals in impoverished areas) into even worse positions.

## Method

### Overall Architecture

The core thesis of this position paper is that ML fairness research must not stop at de-biasing along sensitive attributes like race/sex, but must treat "social determinants"—contextual variables like neighborhood deprivation, school funding, and healthcare accessibility—as first-class auditing objects. Furthermore, auditing must precede mitigation. To solidify this claim, the authors follow a progression from "Conceptual → Theoretical → Empirical → Implementation": they first define social determinants using a three-criterion framework to distinguish them from sensitive attributes, proxies, and administrative labels; then, they use a closed-form theorem of university admissions to prove how race-only quotas can backfire against non-URM applicants in poor areas; next, they conduct semi-synthetic experiments using Census data and OSF HealthCare breast cancer screening records to show that social determinants create systemic gaps even under uniform guidelines; finally, they synthesize their arguments into three actionable pillars (data governance, the Social Determinant Parity metric, and multi-level causal models) to translate "what to audit" into "how to do it."

### Key Designs

**1. Three-Criterion Definition of Social Determinants (Definition 2.2): Defining the Audit Object**

A variable $S$ is considered a social determinant only if it satisfies three criteria: (a) **Context-level definition**: It is defined at a contextual level (neighborhood/institution/jurisdiction) where multiple individuals share the same $S$ value; (b) **Social-structural content**: Cross-contextual differences are primarily shaped by resource allocation, institutional policies, and systemic investments (e.g., school funding ✓, vs. zip code as a purely administrative label ✗); (c) **Exogenous stratification**: The boundaries used for aggregation (neighborhood/postal zone) are exogenously defined rather than based endogenously on the characteristics of the group being described. Based on these criteria, Table 1 categorizes variables: race = sensitive attribute; zip code = non-social determinant (administrative label); ethnic composition of a HOLC redlined area = proxy for a sensitive attribute (endogenous boundary); whereas ethnic composition of a zip code and school funding = true social determinants.

**2. Theorem of Structural Injustice in Quota-based Admissions (Theorem 4.5): Formalizing Backfire Risks**

The authors use a closed-form model of university admissions to transform the intuition of how race-based affirmative action might harm non-URM applicants in poor areas into a provable proposition. Under four assumptions—imbalanced regional ethnic distribution, Academic Preparedness $\perp$ Race $\mid$ Region, rich region scores stochastically dominate poor region scores, and limited university capacity $g$—the URM quota is defined as $\eta_{\mathrm{quota}} \cdot \frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n} g$. The theorem provides a counterexample condition: only when $\max_q \frac{F^{(\mathrm{rich})}(q)}{F^{(\mathrm{poor})}(q)} \ge \frac{\eta_{\mathrm{quota}}}{1+(1-\eta_{\mathrm{quota}})\frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n_{a'}^{(\mathrm{poor})}+n_{a'}^{(\mathrm{rich})}}}$ is satisfied will the score threshold for non-URM applicants in poor regions not be pushed higher than that for URM applicants in rich regions. This paradox suggests that as structural injustice increases (larger stochastic dominance ratio), the quota system is paradoxically less likely to cause additional harm; however, as structural justice improves, the same quota system is more likely to create new injustices.

**3. Breast Cancer Screening Semi-synthetic Experiment (Section 5.2): Empirical Evidence for Auditing**

Using real records of ~54k screenings and ~45k patients from OSF HealthCare (2012–2022), the authors plot the "age of first screening" for white women in poor (ADI ∈ [75,100)) vs. rich (ADI ∈ [0,25)) areas. Despite identical guidelines, the mean difference exceeds 3 years and the median nearly 5 years, attributable to structural factors like transportation and accessibility. A simulation with 100k particles sampled cancer onset using SEER age-specific incidence rates. Allocating 10k screening slots across four policy combinations ("Current Distribution" vs. "Improved Distribution") × ("All to Poor Area" vs. "Both Areas Equally"), the study counted "early detection" events (age of first screening ≤ age of onset). Results showed that adopting the improved screening pattern for the poor area increased early detection from $1367 \pm 33$ to $1461 \pm 36$.

**4. Three Actionable Pillars and Social Determinant Parity: Implementation Roadmap**

The authors propose three pillars for implementation:
- **Pillar 1: Data Governance**: Retain contextual fields like address and geolocation for auditing rather than discarding them.
- **Pillar 2: Social Determinant Parity**: A new metric that replaces the demographic variable in Demographic Parity with structural variables like ADI, infrastructure accessibility, or policy exposure. 
- **Pillar 3: Multi-level Causal Models**: Incorporate social determinants as explicit intervention nodes in causal representation learning rather than treating them as mediators of race.

## Key Experimental Results

### Main Results

| Scenario | Key Data | Description |
|------|----------|------|
| Census PUMS, Median Income of African American Women in CA | Low ADI \$38,000 / Mid ADI \$23,800 / High ADI \$18,800 | Even within the same race × sex intersection, social determinants cause >2× income disparity. |
| OSF HealthCare, Age of First Breast Cancer Screening for White Women | Rich vs. Poor Area: Mean diff >3 years, Median diff ≈5 years | Uniform guidelines fail to eliminate gaps caused by structural conditions. |
| Breast Cancer Semi-synthetic Simulation (10k slots to poor area, 500 runs) | Current $1367 \pm 33$ → Improved $1461 \pm 36$ detections | Improving just one social determinant proxy (screening distribution) yields a ~7% gain in early detection. |

### Ablation Study / Policy Comparison

| Quota Variable $\eta_{\mathrm{quota}}$ | Eq. (1) Right-hand Threshold | Risk to Poor-region non-URM |
|---|---|---|
| $\eta=1$ (Natural Proportion) | Right-hand = 1 | Directly tied to severity of structural injustice. |
| Increasing $\eta$ | Threshold increases monotonically | Higher risk of violation → increased pressure on poor-region non-URM. |
| Improved Structural Justice (CDF ratio decrease) | Left-hand decreases | More likely to create new injustice under the same $\eta$. |

### Key Findings
- **Intra-group Disparity**: Figure 1 demonstrates that "intersectional sensitive attributes" are insufficient; African American women face massive income gaps based on ADI.
- **Guideline Insufficiency**: OSF data shows that white women in rich and poor areas follow the same guidelines, but gaps persist due to structural barriers (transport, trust), proving that "context-unaware guidelines" are a source of injustice.
- **The Quota Paradox**: The theoretical model proves that as structural justice improves, quotas are more likely to harm non-URM individuals in poor areas; aggressive quotas amplify this squeeze.
- **Intervenability**: The semi-synthetic experiment shows that improving structural factors (like screening age distribution) provides a measurable policy lever, whereas race is not a variable that can be similarly "intervened" upon for individual outcomes.

## Highlights & Insights
- **Precise Categorization**: The three-criterion definition (context-level / social-structural / exogenous) provides a rigorous framework to distinguish between race, zip codes, and actual social determinants, exposing the flaws in simply using neighborhood as a proxy for sensitive attributes.
- **Formalized Counter-intuition**: While debates on affirmative action are often philosophical, this paper uses a 4-assumption model and a single inequality to mathematically define when quotas harm the most vulnerable sub-groups.
- **Auditing Must Precede Mitigation**: This methodological mantra can be extended to any Responsible AI scenario, such as RLHF data governance or medical algorithm deployment.
- **Integrating SDoH into ML Fairness**: Building on the work of Obermeyer et al. (2019), this paper provides a systematic entry point for Social Determinants of Health (SDoH) into the ML fairness framework with empirical范式.

## Limitations & Future Work
- The theoretical model focuses on "inter-regional" structural injustice and does not account for intra-institutional racial discrimination. The breast cancer experiment focuses on a single lever (first screening age) and is not a comprehensive causal claim.
- The three-criterion definition includes "grey areas," such as whether institutional membership is truly exogenous in scenarios like college admissions where selection is endogenous.
- **Social Determinant Parity** is proposed conceptually but lacks a differentiable form for optimization or extensive empirical comparison with existing in-processing/post-processing algorithms.
- Multi-level causal models require high-density community-level covariates and must address the potential violation of SUTVA (no interference), leaving the engineering path for future work.

## Related Work & Insights
- **vs. Conditional Demographic Parity (Žliobaite et al., 2011)**: While CDP uses region as a mediator to explain residual bias, this paper argues for auditing the structural variables within the region directly.
- **vs. Path-specific Causal Fairness (Zhang & Bareinboim, 2018a)**: These methods often treat social determinants as mediators downstream of race. This paper argues that environmental variables are not ancestors of race and treating them as mediators obscures their status as direct policy intervention points.
- **vs. Domain Adaptation for Fairness (Madras et al., 2018)**: These approaches treat contextual heterogeneity as distribution shift to be normalized. This paper opposes treating context as noise, arguing it is the primary signal for auditing.
- **vs. Obermeyer et al. (2019)**: While that work empirically showed how "cost as a proxy for need" creates racial bias, this paper scales that insight into a generalized methodology with theoretical models and actionable pillars.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefines the "object of audit" in a crowded field; fundamental position-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Combines theoretical proofs, Census data, real medical data, and simulations; lacks end-to-end comparison with existing de-biasing algorithms.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear argumentation; Table 1 and the Quota Theorem are highly effective.
- **Value**: ⭐⭐⭐⭐⭐ Provides a replicable conceptual framework and actionable technical pillars; highly significant for directing ICML community research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](../../AAAI2026/ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)

</div>

<!-- RELATED:END -->
