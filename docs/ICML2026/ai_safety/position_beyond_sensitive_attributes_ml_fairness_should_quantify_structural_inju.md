---
title: >-
  [Paper Note] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants
description: >-
  [ICML 2026][AI Safety][Paper Note] This is an ICML position paper: the authors argue that ML fairness research must move beyond focusing solely on "sensitive attributes" like race/sex and must include "social determinants" (contextual variables such as neighborhood, ADI, school funding, and healthcare accessibility) in audits. Using a theoretical model
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: 7daede883f1503e8
---
# Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants

**Conference**: ICML 2026  
**arXiv**: [2508.08337](https://arxiv.org/abs/2508.08337)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Algorithmic Fairness / Position Paper  
**Keywords**: Algorithmic Fairness, Structural Injustice, Social Determinants, Sensitive Attributes, Causal Fairness

## TL;DR
This is an ICML position paper: the authors argue that ML fairness research must move beyond focusing solely on "sensitive attributes" like race/sex and must include "social determinants" (contextual variables such as neighborhood, ADI, school funding, and healthcare accessibility) in audits. Using a theoretical model of university admissions, US Census data, and semi-synthetic experiments on breast cancer screening, they demonstrate that mitigation strategies centered only on sensitive attributes may inadvertently create new forms of structural injustice.

## Background & Motivation

**Background**: Current ML fairness literature almost equates "injustice" with "discrimination along sensitive attributes." Most fairness metrics (Demographic Parity, Equal Opportunity, Conditional Demographic Parity, Causal Path Effects, etc.) first specify $A$ (race/sex/age) and then require predictions or decisions to be decoupled from $A$ or satisfy some conditional independence. Benchmark datasets such as Adult, Folktables, and Communities and Crimes even actively discard contextual fields like address or geolocation.

**Limitations of Prior Work**: Interdisciplinary literature (political philosophy, sociology, public health) has long pointed out that what truly shapes individual opportunities and outcomes are social determinants—contextual variables like neighborhood deprivation, school funding, air pollution, distance to hospitals, and community resources. These variables create heterogeneity within the same group (e.g., median annual income for African American women drops from \$38k to \$18.8k across different PUMAs) and impose shared burdens across different groups (e.g., non-URM applicants in impoverished areas face the same community disadvantages as URM applicants). Focusing only on sensitive attributes erases both types of structural signals.

**Key Challenge**: Sensitive attributes are individual-level, (quasi-)stable intrinsic identifiers, whereas social determinants are context-level, structural variables that drift across space and time. Existing individual-level causal graphs ($A \to Y$, $A \to M \to Y$) and de-sensitization losses cannot accommodate community-level structures like "neighborhood-individual" reciprocal influences or community aggregate statistics, often normalizing context as noise.

**Goal**: To establish "social determinants" as first-class objects for auditing by answering three questions: (i) How to conceptually distinguish social determinants from sensitive attributes and their proxies? (ii) Why do existing technical paradigms fail to accommodate them? (iii) What new structural injustices arise from mitigation strategies centered only on sensitive attributes?

**Key Insight**: Starting from a specific scenario—historical redlining forced Black families into specific neighborhoods, making race, zip code, and community racial composition highly correlated over time. However, these three differ significantly in their fairness implications: zip code is an administrative label that cannot be "improved," while school funding and air quality are genuinely intervenable structural variables. The authors use three criteria (context-level definition / social-structural content / exogenous stratification) to categorize them clearly.

**Core Idea**: Auditing must precede mitigation. Before "fixing" a model, structural injustice must be explicitly quantified via social determinants. Otherwise, blindly following race-based quotas may push even more disadvantaged subgroups (e.g., non-URM individuals in poor areas) into worse positions.

## Method

### Overall Architecture

The core thesis of this position paper is that ML fairness research must not stop at removing discrimination along sensitive attributes like race/sex but must treat "social determinants"—contextual variables such as neighborhood deprivation, school funding, and healthcare accessibility—as first-class auditing objects, and that auditing must precede mitigation. To solidify this claim, the authors proceed through "Concept → Theory → Empiricism → Implementation": first, they define social determinants using a three-criteria framework to separate them from sensitive attributes, proxies, and administrative labels; second, they use a closed-form theorem for university admissions to prove that race-only quota-based mitigation can backlash against non-URM applicants in poor neighborhoods; third, they conduct semi-synthetic experiments using Census data and real OSF HealthCare breast cancer screening records to show that social determinants create systemic gaps even under uniform guidelines within the same group, and that intervening on these determinants yields quantifiable gains in early detection; finally, they summarize the argument into three actionable pillars (data governance, a new "Social Determinant Parity" metric, and multi-level causal models) to translate "what to audit" into "how to do it."

### Key Designs

**1. Three-criteria Definition of Social Determinants (Definition 2.2): Defining the boundaries of the audit object to prevent conceptual ambiguity.**

A variable $S$ is considered a social determinant only if it satisfies three criteria: (a) **Context-level definition**—it is defined over a context (neighborhood / institution / jurisdiction) where multiple individuals share the same $S$ value; (b) **Social-structural content**—disparities across contexts are primarily shaped by resource allocation, institutional policies, or systemic investment (e.g., school funding ✓, vs. zip code as a mere administrative label ✗); (c) **Exogenous stratification**—the boundaries used for aggregation (neighborhood / census tract) are determined exogenously rather than endogenously based on the characteristics of the group being described. Based on these yes/no criteria, Table 1 categorizes variables: race = sensitive attribute; zip code = non-social determinant (administrative label); racial composition of a HOLC redlined area = proxy for sensitive attribute (endogenous boundary); whereas racial composition of a zip code area or school funding = true social determinants. This distinction clarifies a common confusion: "using neighborhood as a proxy for race" is an extension of redlining, while "auditing the structural conditions of the neighborhood itself" is auditing structural injustice.

**2. Theorem of Structural Injustice in Quota-based Admissions (Theorem 4.5): Elevating the intuition of "when sensitive-attribute-centric mitigation harms the most disadvantaged" to a provable proposition.**

The authors use a closed-form model of university admissions to express the conditions under which affirmative action based solely on race harms non-URM applicants in impoverished areas as an inequality. Under four assumptions—imbalanced regional racial distribution, Academic Preparedness $\perp$ Race $\mid$ Region, rich-area score CDF stochastically dominating poor-area scores, and a limited quota $g$ for selective universities—the URM quota is denoted as $\eta_{\mathrm{quota}} \cdot \frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n} g$. The theorem provides a counterexample condition: only when $\max_q \frac{F^{(\mathrm{rich})}(q)}{F^{(\mathrm{poor})}(q)} \ge \frac{\eta_{\mathrm{quota}}}{1+(1-\eta_{\mathrm{quota}})\frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n_{a'}^{(\mathrm{poor})}+n_{a'}^{(\mathrm{rich})}}}$ holds, will the score threshold for non-URM applicants in poor areas not be pushed higher than that for URM applicants in rich areas. The resulting paradox is counterintuitive: the more severe the structural injustice (larger stochastic dominance ratio on the left), the easier the inequality is satisfied, and the less harm the quota causes; however, once structural justice improves, using the same quota is more likely to create new injustices. Furthermore, a larger $\eta_{\mathrm{quota}}$ yields a higher threshold on the right, meaning more aggressive sensitive-attribute-centric mitigation amplifies the squeeze on poor non-URM individuals.

**3. Semi-synthetic Breast Cancer Screening Experiment (Section 5.2): Grounding theory in a high-stakes healthcare scenario to respond to "fairness through unawareness."**

The authors use real records of approximately 54,000 screenings from 45,000 patients at OSF HealthCare (2012–2022) to plot the "age at first screening" distribution for white women in poor (ADI ∈ [75,100)) vs. rich (ADI ∈ [0,25)) areas. Even under the same screening guidelines and within the same racial group, the mean difference exceeds 3 years and the median difference is nearly 5 years—disparities attributable only to structural conditions like transportation, accessibility, and trust. Using a 100k-particle simulation sampling cancer onset by SEER age-specific incidence, they evaluate four policy combinations: "status quo distribution vs. improved distribution (applying rich-area first-screening age to poor areas)" × "all slots to poor areas vs. equal split." After 500 runs, they count "age at first screening ≤ age at onset = early detection." The results show that adopting the improved screening pattern in poor areas increases early detections from $1367 \pm 33$ to $1461 \pm 36$. This experiment proves that disparities persist regardless of race or guidelines and demonstrates that "intervening on social determinants" is a viable policy lever compared to merely adjusting sensitive attribute quotas.

**4. Three Actionable Pillars and the Social Determinant Parity Metric: Translating the proposal into a technical roadmap.**

While this position paper does not specify training objectives, it provides three implementation pillars. Pillar 1: Data Governance—stop discarding contextual fields like address and geolocation; preserve social determinants in audit data. Pillar 2: **Social Determinant Parity**—replace the conditioning variable in Demographic Parity with structural variables (e.g., ADI, infrastructure access, policy exposure). A longitudinal version requires tracking these metrics over time as contextual variables shift. Pillar 3: Multi-level Causal Models + Causal Representation Learning—treat social determinants as explicit intervention nodes rather than downstream mediators of race. Only when they are intervention targets can actions like "improving school funding" enter the causal framework.

## Key Experimental Results

### Main Results

| Scenario | Key Data | Description |
|------|----------|------|
| Census PUMS, Median Income of African American Women in CA | Low ADI \$38,000 / Mid ADI \$23,800 / High ADI \$18,800 | Within the same race × gender intersection, social determinants still cause a >2× median income gap. |
| OSF HealthCare, White Women's Age at First Breast Cancer Screening | Rich vs. Poor: Mean diff >3 years, Median diff ≈5 years | Under a uniform guideline, the gap is solely due to structural conditions. |
| Breast Cancer Semi-synthetic Simulation (10k slots to poor areas, 500 runs) | Status Quo $1367 \pm 33$ → Improved $1461 \pm 36$ detections | Improving just one proxy (first screening age distribution) yielded a +7% gain in early detection. |

### Ablation Study

| Quota Multiplier $\eta_{\mathrm{quota}}$ | Right-hand side threshold of Eq. (1) | Probability of harm to poor-area non-URM |
|---|---|---|
| $\eta=1$ (Natural proportion) | RHS = 1 | Directly linked to the severity of structural injustice. |
| Increasing $\eta$ | Monotonically increases | More likely to violate the inequality → poor non-URM are squeezed further. |
| Improved Structural Injustice | LHS decreases | For the same $\eta$, new injustices are more likely to be created. |

### Key Findings
- **2x Difference Within the Same Group**: Figure 1 refutes the idea that "intersectional sensitive attributes are enough"—African American women, a frequently discussed intersectional group, show massive income gaps based on ADI.
- **Uniform Guidelines Fail to Close Gaps**: In OSF data, white women in rich and poor areas follow the same guideline; the gap stems entirely from structural conditions (transportation, access, trust), implying that "guidelines blind to social determinants" are themselves a source of injustice.
- **The Quota Paradox**: The theoretical model proves that as structural justice improves, quotas are more likely to backlash against non-URM individuals in poor areas; aggressive quotas amplify this harm.
- **Intervenability**: Semi-synthetic experiments show that "improving the first-screening age distribution" yields +94 early detections per 10k screenings, indicating that social determinants are practical policy levers whereas race is not.

## Highlights & Insights
- **Clean Three-criteria Definition**: By using context-level / social-structural / exogenous stratification, the authors categorize variables like race, zip code, and school funding into distinct buckets, exposing lazy approaches that simply treat neighborhood as a sensitive attribute.
- **Counterintuitive Quota Paradox**: While debates on "fairness of affirmative action" are usually philosophical, this paper uses an inequality to define when quotas harm the most vulnerable subgroups, bringing the debate back to mathematics.
- **Auditing Must Precede Mitigation**: This methodological slogan is transferable to any Responsible AI scenario, such as reward data governance in RLHF or healthcare algorithm deployment.
- **Integrating SDoH into ML Fairness**: Since Obermeyer et al. (2019), the community has lacked a systematic entry point for Social Determinants of Health (SDoH). This paper provides the conceptual framework, theoretical model, and experimental paradigm to fill that gap.

## Limitations & Future Work
- The theoretical model only characterizes "inter-regional" structural injustice and ignores intra-institutional racial discrimination. The breast cancer experiment covers only one lever (first screening age).
- The three-criteria definition has gray areas; for example, whether "institutional membership" is exogenous is unclear in employment or education settings where college enrollment is an endogenous choice.
- **Social Determinant Parity** is proposed conceptually but lacks a differentiable form for optimization and has not been compared empirically against existing in-processing/post-processing algorithms.
- Multi-level causal models require significant community-level covariates and must address the failure of SUTVA/no interference, leaving the engineering path for future work.

## Related Work & Insights
- **vs. Conditional Demographic Parity (Žliobaite et al., 2011; Wachter et al., 2021)**: These use region as a mediator to explain residual dependence on race. This paper reverses the view by targeting the structural variables carried by the region itself.
- **vs. Path-specific Causal Fairness (Zhang & Bareinboim, 2018a; Chiappa, 2019)**: These treat social determinants as downstream mediators of race; this paper argues that environmental variables are not ancestors of race and that treating them as mediators loses their lever-like attribute for policy intervention.
- **vs. Domain Adaptation for Fairness (Madras et al., 2018)**: These normalize cross-contextual heterogeneity as distribution shift. This paper argues context should be the signal audited, not noise to be removed.
- **vs. Obermeyer et al. (2019)**: That work showed how using cost as a proxy for need causes racial bias. This paper generalizes that single case into a broad methodology with concept definitions and theoretical models.
- **vs. Kasirzadeh (2022)**: While previous work called for structural justice at a philosophical level, this paper provides the mathematical criteria and experimental pillars to bridge philosophy and ML engineering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefines the "audit object" in a saturated field; position-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes theory, Census data, real medical data, and simulation. One star deducted for lack of end-to-end comparison with existing fairness algorithms.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain; Table 1 and the Quota Paradox are highly effective.
- Value: ⭐⭐⭐⭐⭐ Provides a conceptual framework, computable tools, and actionable pillars for the ICML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](../../AAAI2026/ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)

</div>

<!-- RELATED:END -->
