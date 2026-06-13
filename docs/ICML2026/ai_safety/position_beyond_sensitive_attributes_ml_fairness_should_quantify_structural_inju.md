---
title: >-
  [Paper Note] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants
description: >-
  [ICML 2026][AI Safety][Algorithmic Fairness] This is an ICML position paper: the authors argue that ML fairness research cannot focus solely on "sensitive attributes" such as race/sex. Instead…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Algorithmic Fairness"
  - "Structural Injustice"
  - "Social Determinants"
  - "Sensitive Attributes"
  - "Causal Fairness"
date: 2026-05-08
content_hash: e7b1971c5ab0389e
---

# Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants

**Conference**: ICML 2026  
**arXiv**: [2508.08337](https://arxiv.org/abs/2508.08337)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Algorithmic Fairness / Position Paper  
**Keywords**: Algorithmic Fairness, Structural Injustice, Social Determinants, Sensitive Attributes, Causal Fairness

## TL;DR
This is an ICML position paper: the authors argue that ML fairness research cannot focus solely on "sensitive attributes" such as race/sex. Instead, it must incorporate "social determinants" (contextual variables like neighborhood, ADI, school funding, and healthcare accessibility) into auditing. Using a theoretical college admission model, U.S. Census data, and semi-synthetic breast cancer screening experiments, they demonstrate that mitigation strategies centered only on sensitive attributes may inadvertently create new structural injustices.

## Background & Motivation

**Background**: Current ML fairness literature almost equates "unfairness" with "discrimination along sensitive attributes." Most fairness metrics (Demographic Parity, Equal Opportunity, Conditional Demographic Parity, causal path effects, etc.) first specify $A$ (race/sex/age) and then require predictions or decisions to be decoupled from $A$ or satisfy some form of conditional independence. Benchmark datasets such as Adult, Folktables, and Communities and Crimes often actively remove contextual fields like address or geolocation.

**Limitations of Prior Work**: Interdisciplinary literature (political philosophy, sociology, public health) has long pointed out that what truly shapes individual opportunities and outcomes are social determinants—contextual variables such as neighborhood deprivation, school funding, air pollution, distance to hospitals, and community resources. These variables create heterogeneity within the same group (e.g., for African American women, median annual income drops from \$38k to \$18.8k depending on the PUMA) and produce shared burdens across different groups (e.g., non-URM applicants in impoverished areas face the same community disadvantages as URM applicants). Focusing only on sensitive attributes erases both types of structural signals.

**Key Challenge**: Sensitive attributes are individual-level, (quasi-)stable intrinsic identifiers, whereas social determinants are context-level structural variables that drift across space and time. Existing individual-level causal graphs ($A \to Y$, $A \to M \to Y$) and de-sensitization losses cannot accommodate community-level structures like "neighborhood-individual" bidirectional influences or community aggregate statistics, effectively normalizing context as noise.

**Goal**: To treat "social determinants" as first-class auditing objects by answering three questions: (i) How to conceptually distinguish social determinants from sensitive attributes and their proxies? (ii) Why do existing technical paradigms fail to accommodate them? (iii) What new structural injustices are introduced by mitigation strategies centered solely on sensitive attributes?

**Key Insight**: Starting from a specific scenario—historical redlining forced Black families into specific neighborhoods, making race, zip code, and community racial composition highly correlated over time. However, these three differ completely in their fairness implications (zip code is an administrative label that cannot be "improved"; school funding and air quality are true intervenable structural variables). The authors classify them using three criteria: context-level definition, social-structural content, and exogenous stratification.

**Core Idea**: Auditing must precede mitigation. Before "fixing" models, structural injustice must be explicitly quantified through social determinants. Otherwise, blindly applying quotas based on race may push more disadvantaged subgroups (such as non-URM individuals in impoverished areas) into even worse positions.

## Method

As a position paper, the "Method" consists of three components: (1) A definitional framework to separate social determinants from sensitive attributes, proxies, and administrative labels; (2) A closed-form theoretical model of university admissions proving conditions under which quota-based affirmative action harms non-URM applicants in impoverished areas; (3) Semi-synthetic experiments using U.S. Census PUMS and real OSF HealthCare breast cancer screening data to validate empirical gaps caused by social determinants.

### Overall Architecture
Input: Interdisciplinary conceptual foundations of structural injustice + existing ML fairness metrics + real census/healthcare data.  
Process: (I) Conceptual gap analysis → (II) Three reasons for the mismatch of existing technical paradigms → (III) Closed-form theoretical model (quota-based admission) → (IV) Census population heterogeneity + semi-synthetic breast cancer screening → (V) Addressing two counter-arguments ("Social determinants are just another sensitive attribute" / "Causal effects of sensitive attributes already include them").  
Output: Three actionable pillars—data governance, dynamic fairness metrics (Social Determinant Parity), and a causal framework treating social determinants as explicit intervention targets.

### Key Designs

1.  **Three-Criteria Definition of Social Determinants (Definition 2.2)**:
    - **Function**: Strictly separates social determinants from sensitive attributes, proxies, and administrative labels to prevent terminological ambiguity.
    - **Mechanism**: A variable $S$ is considered a social determinant only if it satisfies: (a) **Context-level definition**: Defined at a context level (neighborhood / institution / jurisdiction), where multiple individuals share the same $S$ value; (b) **Social-structural content**: Variance across contexts is primarily shaped by resource allocation, institutional policies, or systemic investment (e.g., school funding ✓, zip code label ✗); (c) **Exogenous stratification**: The boundaries used for aggregation (neighborhood / postal zone) are defined exogenously, rather than endogenously based on the characteristics of the group being described. Based on this, Table 1 provides a clean classification: race = sensitive attribute; zip code = non-social determinant (administrative label); racial composition of HOLC redlined areas = proxy for sensitive attribute (endogenous boundary); racial composition of zip code area / school funding = social determinant.
    - **Design Motivation**: Prior literature often conflates "using neighborhoods as proxies for racial discrimination" with "auditing the structural conditions of the neighborhood itself." The former is an extension of redlining, while the latter audits structural injustice. Failing to distinguish them excludes intervenable actions, such as improving school funding, from the scope of research.

2.  **Structural Injustice Theorem for Quota-based Admissions (Theorem 4.5)**:
    - **Function**: Formalizes the conditions under which affirmative action using only race-based quotas harms non-URM applicants in impoverished areas, proving that sensitive-attribute-centric mitigation $\neq$ advancing structural justice.
    - **Mechanism**: Under four assumptions (imbalanced regional racial distribution + Academic Preparedness $\perp$ Race $\mid$ Region + rich area score CDF stochastically dominates poor area + limited selective university slots $g$), the URM quota is written as $\eta_{\mathrm{quota}} \cdot \frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n} g$. The theorem provides a counter-example inequality: only when $\max_q \frac{F^{(\mathrm{rich})}(q)}{F^{(\mathrm{poor})}(q)} \ge \frac{\eta_{\mathrm{quota}}}{1+(1-\eta_{\mathrm{quota}})\frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n_{a'}^{(\mathrm{poor})}+n_{a'}^{(\mathrm{rich})}}}$ will the score threshold for non-URM applicants in poor areas not be pushed higher than that for URM applicants in rich areas.
    - **Design Motivation**: Reveals a counter-intuitive paradox: the more severe the structural injustice (the larger the stochastic dominance ratio on the left), the easier the inequality holds and the smaller the quota damage. Conversely, as structural justice improves, maintaining the same quota is more likely to create new injustices. Furthermore, a larger $\eta_{\mathrm{quota}}$ increases the threshold on the right, meaning "more aggressive sensitive-attribute-centric mitigation amplifies harm to non-URM individuals in poor areas." This elevates the necessity of auditing social determinants from intuition to a provable proposition.

3.  **Semi-synthetic Breast Cancer Screening Experiment (Section 5.2)**:
    - **Function**: Applies the theory to a high-stakes medical scenario, proving that even with uniform screening guidelines, social determinants still create systemic gaps, and intervening on social determinants yields quantifiable gains in early detection.
    - **Mechanism**: Using real records of approximately 54,000 screenings from 45,000 patients (OSF HealthCare 2012–2022), the authors plot the "age at first screening" distribution for White women in poor areas (ADI ∈ [75,100)) and rich areas (ADI ∈ [0,25)). Despite the same guidelines and race, the mean difference is >3 years, and the median difference is nearly 5 years. Then, using 100k particle simulations + SEER age-specific incidence sampling for cancer onset, 10k screening slots are allocated according to four policy combinations: "Current Distribution vs. Improved Distribution" (using the rich area distribution for poor areas) + "All to Poor Area vs. Equal Split." Each is run 500 times to count "Age at First Screening ≤ Onset Age = Early Detection." Results: implementing the improved screening pattern in poor areas increased early detections from $1367 \pm 33$ to $1461 \pm 36$.
    - **Design Motivation**: (a) Empirically responds to "fairness through unawareness"—disparities persist even with the same race and guidelines, necessitating the explicit auditing of social determinants. (b) Quantifies the difference between "intervening on social determinants" and "adjusting sensitive attribute quotas," providing empirical support for Pillar 3's goal of treating social determinants as intervention targets.

### Loss & Training
This position paper does not involve training objectives. However, in Pillar 2, the authors propose a new metric, **Social Determinant Parity**: swapping the conditioning variables of existing Demographic Parity from ethnicity to structural variables such as area deprivation index, infrastructure accessibility, or policy exposure. A longitudinal version further requires the metric to track changes over time relative to contextual variables. Pillar 3 further advocates for multi-level causal models + causal representation learning to make social determinants explicit intervention nodes rather than mediators.

## Key Experimental Results

### Main Results

| Scenario | Key Data | Description |
| :--- | :--- | :--- |
| Census PUMS, Median Annual Income of African American Women in CA | Low ADI \$38,000 / Mid ADI \$23,800 / High ADI \$18,800 | Same ethno-gender intersection, social determinants still result in >2× median income gap |
| OSF HealthCare, Age at First Breast Cancer Screening for White Women | Rich vs. Poor: Mean difference >3 years, median difference ≈5 years | Under the same uniform screening guidelines, the gap is solely attributable to structural conditions |
| Breast Cancer Semi-synthetic Simulation (10k slots all to poor areas, 500 runs) | Current Pattern $1367 \pm 33$ → Improved Pattern $1461 \pm 36$ early detections | Improving just one social determinant proxy (age at first screening distribution) yields a Gain of ~+7% in early detection |

### Ablation Study / Policy Comparison

| Quota Multiplier $\eta_{\mathrm{quota}}$ | Inequality (1) Right-hand Threshold | Probability of Harm to Poor Non-URMs |
| :--- | :--- | :--- |
| $\eta=1$ (Natural Proportion) | Right-hand side = 1 | Directly linked to severity of structural injustice |
| Increasing $\eta$ | Right-hand side increases monotonically | Higher $\eta$ makes violation more likely → poor non-URMs are further squeezed |
| Improved structural justice (Lower CDF ratio) | Left-hand side decreases | More likely to create new injustice under the same $\eta$ |

### Key Findings
- **2x Difference within Same Group**: Figure 1 refutes the idea that "intersecting sensitive attributes are sufficient." African American women, the most frequently discussed intersectional group, face massive income gaps due to ADI.
- **Uniform Guidelines Fail to Close Gaps**: In OSF data, White women in rich and poor areas follow the same screening guidelines. Disparities stem entirely from structural conditions (transportation, accessibility, trust), showing that "guidelines unaware of social determinants" are themselves a source of unfairness.
- **The Quota Paradox**: The theoretical model proves that as structural justice improves, quotas are more likely to backlash against poor non-URMs; aggressive quotas amplify this harm—a formalized warning regarding whether "affirmative action is always better."
- **Intervenability**: The semi-synthetic experiment shows that improving only the "age at first screening distribution" results in +94 early detections per 10k screenings, proving that social determinants are true policy levers, whereas race is not.

## Highlights & Insights
- **Clean Three-Criteria Definition**: Using context-level / social-structural / exogenous stratification as yes/no criteria allows race, zip code, HOLC racial composition, zip code racial composition, and school funding to be placed into distinct categories. This immediately exposes the issues with lazy approaches of "just adding neighborhood as a sensitive attribute."
- **The Counter-intuitive Quota Paradox**: Long-standing debates on the "fairness of affirmative action" have remained at the level of philosophical or value judgments. This paper uses 4 assumptions and 1 inequality to turn the question of "when quotas harm the most disadvantaged subgroups" into a calculable condition, bringing philosophical debate back to mathematics.
- **Auditing Must Precede Mitigation**: This methodological slogan can be transferred to any Responsible AI scenario involving "assessment before intervention"—such as reward data governance in RLHF or medical algorithm deployment.
- **Reintegrating SDoH into ML Fairness**: Since Obermeyer et al.'s (2019) work on racial bias in healthcare algorithms, the ML community has lacked a systematic entry point for integrating Social Determinants of Health (SDoH) into fairness frameworks. This paper uses real OSF data and semi-synthetic simulations to establish a concrete experimental paradigm for future work.

## Limitations & Future Work
- The authors acknowledge that the theoretical model only captures "inter-regional" structural injustice and ignores racial discrimination within the same school or institution. The semi-synthetic breast cancer experiment only covers one lever (age at first screening) and should not be viewed as a causal assertion on all structural barriers.
- There are gray areas in the application of the three-criteria definition—for example, whether "institutional membership is exogenous" is unclear in many employment and education scenarios (college admission is an endogenous choice). Stable classification in practice requires more case studies.
- Social Determinant Parity is proposed only at a conceptual level. A specific differentiable form for optimization has not been provided, nor has it been empirically compared with existing in-processing or post-processing fairness algorithms.
- The vision of multi-level causal models + causal representation learning requires observing many community-level covariates and addressing the failure of SUTVA / no interference, leaving the engineering path for future work.

## Related Work & Insights
- **vs. Conditional Demographic Parity (Žliobaite et al., 2011; Wachter et al., 2021)**: The latter uses region as a mediator to explain residual dependence between race and outcome, remaining "race-centric." This paper advocates for auditing the structural variables carried by the region itself and explicitly distinguishes itself from Conditional Demographic Parity variants.
- **vs. Path-specific Causal Fairness (Zhang & Bareinboim, 2018a; Chiappa, 2019; Wu et al., 2019)**: While these methods can insert social determinants into a race → SD → Y path, they default to SD being a downstream mediator of race. This paper points out that environmental variables are not ancestors of race and that treating SD as a mediator overlooks its attribute as a lever for direct policy intervention.
- **vs. Domain Adaptation for Fairness (Madras et al., 2018; Creager et al., 2021)**: These treat cross-context heterogeneity as distribution shift to be normalized. This paper opposes the paradigm of "viewing context as noise" and argues that context is the signal to be audited.
- **vs. Obermeyer et al. (2019)**: The latter empirically revealed that "using cost as a proxy for need introduces racial bias." This paper goes further by providing conceptual definitions, theoretical models, and universal pillars to upgrade a single case into a methodology.
- **vs. Kasirzadeh (2022)**: These philosophical discussions of structural injustice primarily call for change at a high level. This paper provides three criteria, a closed-form theorem, and semi-synthetic experiments, serving as the first bridge from philosophical appeal to ML engineering practice.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A position-level innovation that redefines and formalizes the "auditing object" itself in the heavily explored field of ML fairness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes theoretical derivation, Census data, real medical data, and semi-synthetic simulations. One star deducted for the lack of end-to-end comparison with existing fairness algorithms.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The argumentation chain (I)–(V) is extremely clear. Table 1, the three criteria, and the quota theorem complement each other. The "Alternative Views" section directly addresses strong counter-arguments.
- **Value**: ⭐⭐⭐⭐⭐ Provides a conceptual framework (three criteria), calculable tools (Social Determinant Parity / quota inequality), and actionable pillars, offering significant guidance for the ICML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](../../AAAI2026/ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)

</div>

<!-- RELATED:END -->
