---
title: >-
  [Paper Note] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives
description: >-
  [ICML 2026][AI Safety][Organ Allocation] This ICML 2026 position paper argues, using historical UNOS data, that the next-generation ML strategies for the U.S. heart transplant allocation system must model the incentive misalignment among "organ procurement organizations (OPOs), transplant centers, physicians, patients, and regulators" as a first-class citizen. It calls for integrating mechanism design, strategic classification, causal inference…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Organ Allocation"
  - "Incentive Alignment"
  - "Strategic Classification"
  - "Mechanism Design"
  - "RLHF"
date: 2026-05-08
content_hash: 125300f61bcce1ef
---

# Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives

**Conference**: ICML 2026  
**arXiv**: [2602.04990](https://arxiv.org/abs/2602.04990)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Mechanism Design / Strategic Classification / Healthcare Policy  
**Keywords**: Organ Allocation, Incentive Alignment, Strategic Classification, Mechanism Design, RLHF  

## TL;DR
This ICML 2026 position paper argues, using historical UNOS data, that the next-generation ML strategies for the U.S. heart transplant allocation system must model the incentive misalignment among "organ procurement organizations (OPOs), transplant centers, physicians, patients, and regulators" as a first-class citizen. It calls for integrating mechanism design, strategic classification, causal inference, and social choice into the ML pipeline; otherwise, even the strongest predictive models will be undermined by strategic behaviors during deployment.

## Background & Motivation

**Background**: The historical U.S. heart transplant allocation is a manually designed, rigid priority queue based on medical urgency. It is rapidly transitioning toward data-driven ML/optimization methods (e.g., the "Continuous Distribution" framework already deployed for lung transplants, with heart allocation in preparation). Demand significantly outweighs supply—over 100,000 people are on the waiting list in the U.S. alone.

**Limitations of Prior Work**: Existing ML solutions treat allocation as a static optimization problem (learning an optimal policy table or scoring function), completely ignoring the fact that allocation is a multi-party game. Hospitals, OPOs, clinicians, and patients each have their own goals and will respond strategically to policy changes. Predictors trained accurately on historical data may fail or even produce adverse effects post-deployment due to distribution shifts.

**Key Challenge**: Supervised learning is essentially "learning a mapping," whereas real-world features are "actively shaped by participants." This is a manifestation of Goodhart’s Law—"When a measure becomes a target, it ceases to be a good measure." For instance, the 2018 policy change placed IABP (Intra-Aortic Balloon Pump) patients in high-priority Status 2, causing the proportion of patients bridged with IABP to skyrocket from 7.0% to 24.9%—a more than three-fold increase accompanied by real clinical risks such as poor organ perfusion and bleeding.

**Goal**: To identify incentive misalignment points across the entire decision pipeline (feature gaming, out-of-sequence allocation, performance evaluation distortion, strategic entry/exit from the list, preference aggregation manipulation) and provide a corresponding research agenda for the ML community.

**Key Insight**: By translating each stage of the pipeline into a mechanism design, strategic classification, or social choice problem, next-generation allocation strategies can remain robust, effective, fair, and trustworthy under strategic behavior.

**Core Idea**: Next-generation organ allocation ML must be "incentive-aware"—it should not only learn "who should be prioritized" but also "who should be prioritized under the condition that all participants will respond strategically."

## Method

As this is a position paper rather than a methodology paper, it does not present a single algorithm. Instead, its "method" involves a systematic scan of the heart transplant decision pipeline—from patient feature reporting and OPO bidding (including out-of-sequence offers) to transplant center acceptance, list entry/exit, and top-level policy preference aggregation. In each stage, the authors locate incentive misalignments, provide quantitative evidence using UNOS 2010–2024 registry data, translate the issue into a specific mechanism design/strategic classification/social choice problem, and summarize these into a research agenda for the ML community. The core claims are expanded below:

**1. Claim: Urgency tiers are prone to feature gaming and must be modeled as strategic classification.** Current urgency tiers rely on device usage, allowing clinicians to push patients across decision boundaries by initiating or withholding specific devices. The paper formalizes this as strategic classification: a patient’s features $x$ shift to $x'$ at a cost $c(x, x')$ to cross a classifier threshold, where optimal manipulation balances the "gains of high priority" against "manipulation costs" (including clinical harm). The most typical empirical evidence is the 2018 policy change that moved IABP patients to Status 2, leading to a jump in IABP bridging from 7.0% to 24.9%—a three-fold increase representing a Goodhart-style collapse. The proposed ML remedy includes using repeated risk minimization to handle actively shaped feature distributions (known to converge under certain assumptions [Perdomo et al., 2020]), using causal inference to distinguish features with real medical effects from mere correlates, and using selective verification (random audits) to raise manipulation costs.

**2. Claim: Out-of-sequence allocation and periodic performance evaluations are system-level misalignments; evaluation mechanisms must be co-designed with ML.** Out-of-sequence (OOS) offers allow OPOs to bypass the priority queue, intended to rescue organs at risk of wastage, but the trigger thresholds are opaque. Since CMS began monitoring OPOs by waste rates in 2021, kidney OOS rates surged from 2% in 2020 to 18% in 2023, systematically favoring wealthier groups. On the performance side, SRTR ranks centers into 5 tiers based on metrics like 1-year survival, incentivizing centers to risk-aversely reject marginal offers. UNOS data shows a statistically significant rebound in acceptance rates in May after the April reporting window closes, consistent with a "horizon effect" where centers accept higher-risk cases at the start of a new window. The ML agenda suggests using computer vision and ex-vivo perfusion for real-time organ assessment to learn optimal OOS triggers, replacing semi-annual evaluations with CUSUM-style continuous monitoring, and using improved risk-adjustment models to evaluate small centers fairly.

**3. Claim: Optimization targets themselves are learned from strategic parties; therefore, social choice and mechanism design must be at the source of the ML pipeline.** Top-level policies currently use AHP (Analytic Hierarchy Process) to extract preference weights from the community, but the Gibbard–Satterthwaite theorem proves any "reasonable" voting rule is manipulable. Small rural centers push for broader sharing, while large urban centers do the reverse. Multiple listing is a more explicit inequity—only 2.16% of patients list at multiple centers, yet their transplant rate of 80.44% is far higher than the 73.06% for single-listing patients. The paper advocates for: using frugal preference elicitation with RLHF instead of AHP to distinguish "normative ends" from "attributes as means"; using counterfactual modeling to quantify where multiple listing truly helps the system; and treating the pipeline as multi-agent mechanism design by introducing credit systems to encourage offer acceptance and randomized audits to raise the cost of manipulation.

## Key Experimental Results

Note: As a position paper, it uses statistical observations from UNOS 2010–2024 registry data rather than traditional algorithmic experiments to support its claims.

### Outcomes of Highest Urgency (Status 1) Patients (2010–2024)

| Metric | Value | Implications |
|------|------|------|
| Death within 3 days of listing | 6.5% | Significant mortality within one week |
| Death within 7 days of listing | 13.7% | Significant mortality within one week |
| Median time to transplant | 26 days | Only 10 days earlier than median death time |
| Median time to death | 36 days | Razor-thin margin of safety |
| Time to death IQR | 13–118 days | Massive heterogeneity within Status 1 |

### System-level Evidence of Incentive Misalignment

| Phenomenon | Key Figures | Explanation |
|------|----------|------|
| IABP bridging share (Post-2018 policy) | 7.0% → 24.9% | Three-fold increase, suspected feature gaming |
| Kidney OOS allocation rate | 2020: 2% → 2023: 18% → Early 2026: 9% | Spiked after CMS regulation, dropped after federal scrutiny |
| Multiple listing patients / Transplant rate | 2.16% / 80.44% (vs. 73.06% single-list) | Significant arbitrage by wealthier groups |
| Avg. distance for multiple listing centers | 379 nautical miles (Max > 2200) | Cross-regional arbitrage impacting equity |

### Key Findings

- Once evaluation metrics are made public (e.g., CMS monitoring OPOs, SRTR ranking centers), participants immediately "reshape behavior to the metric" rather than "what is best for the patient."
- Waiting time priority disadvantages the most critically ill, as they cannot survive long enough to accumulate priority; this institutionally makes "early listing to stockpile time" the optimal strategy.
- Federal scrutiny reduced OOS rates from 20% to 9% in less than two years, showing that response to incentives is immediate and powerful in both directions.

## Highlights & Insights

- **Concrete Research Agenda for Goodhart's Law**: The paper does not merely state that ML is prone to gaming; it maps specific pipeline stages to types of gaming and provides corresponding mechanism design tools. This "system-level misalignment map" is highly actionable for the ML community.
- **"Means vs. Ends" Value**: All current preference aggregation tasks (including AI alignment and RLHF) tend to have humans vote on "specific solutions." This paper suggests humans should vote on "ends," while algorithms search for the "means" under those constraints—a concept directly transferable to RLHF preference data design.
- **Acknowledging Counter-arguments**: Section 7 discusses the view that clinician manipulation might be a way to "correct imperfect policies." By engaging with opposing views, the paper increases the credibility of its position.

## Limitations & Future Work

- The paper focuses almost entirely on U.S. heart transplants; specific misalignments (IABP gaming, OOS allocation, SRTR cycles) may differ in other countries or organ types (liver/kidney/lung).
- Many "incentive explanations" are currently correlational (e.g., the May rebound); the authors admit that "a more rigorous causal analysis remains necessary."
- The proposed ML agendas are mostly directional ("should use strategic classification," "should use RLHF"). End-to-end deployable systems have not yet been built on actual organ allocation data.
- Randomized audits, while effective in theory, face institutional hurdles regarding who performs the audit and who bears the cost.

## Related Work & Insights

- **vs. Papalexopoulos et al. (2023) (Continuous Distribution Framework)**: While they proposed continuous scoring to mitigate cliff-edge effects, this paper notes that continuous scores still rely on classifiers/regressors that are susceptible to feature manipulation.
- **vs. Hardt et al. (2016), Perdomo et al. (2020) (Strategic Classification)**: Existing work provides general frameworks; this paper grounds these frameworks in survival analysis and dynamic waiting lists—settings the strategic classification community has not yet fully explored.
- **vs. Anagnostides et al. (2025) (Dynamic Heart Allocation)**: Unlike previous work focused on optimizing allocation rules, this paper argues for the "co-design" of upstream performance evaluations and downstream preference aggregation.
- **vs. Conitzer et al. (2024) (Social Choice and AI Alignment)**: Both emphasize social choice theory; this paper provides a more specific, high-stakes application (life-and-death) that makes the research goals of RLHF/social choice more concrete.

## Rating
- Novelty: ⭐⭐⭐⭐ The technical tools are not new, but packaging them as an "Organ Allocation ML Agenda" with UNOS evidence is novel for the ML community.
- Experimental Thoroughness: ⭐⭐⭐⭐ The "experiments" are UNOS data observations across multiple misalignment points from 2010–2024; the evidence chain is complete, though causal analysis can be further refined.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent structure following "phenomenon → data → incentive explanation → ML solution"; the dialogue with alternative views in Section 7 is a benchmark for position papers.
- Value: ⭐⭐⭐⭐⭐ Points out that the real bottleneck for healthcare ML is incentive modeling rather than model capacity; provides a high-stakes real-world scenario for researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[ICML 2026\] Two Blind Spots in Machine Unlearning: Over-Unlearning and Prototype Re-learning Attacks](unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)
- [\[ICML 2026\] When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery](when_should_an_ai_scientist_stop_verifiable_experiment_steering_and_refusal_for_.md)

</div>

<!-- RELATED:END -->
