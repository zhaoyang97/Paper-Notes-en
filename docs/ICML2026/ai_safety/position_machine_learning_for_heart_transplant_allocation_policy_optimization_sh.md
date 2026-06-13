---
title: >-
  [Paper Note] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives
description: >-
  [ICML 2026][AI Safety][Organ Allocation] This ICML 2026 position paper argues, using historical UNOS data, that next-generation ML strategies for the U.S. heart transplant allocation system must model incentive misalignm…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Organ Allocation"
  - "Incentive Alignment"
  - "Strategic Classification"
  - "Mechanism Design"
  - "RLHF"
date: 2026-05-08
content_hash: 777da9d2004e983d
---

# Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives

**Conference**: ICML 2026  
**arXiv**: [2602.04990](https://arxiv.org/abs/2602.04990)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Mechanism Design / Strategic Classification / Healthcare Policy  
**Keywords**: Organ Allocation, Incentive Alignment, Strategic Classification, Mechanism Design, RLHF

## TL;DR
This ICML 2026 position paper argues, using historical UNOS data, that next-generation ML strategies for the U.S. heart transplant allocation system must model incentive misalignment between "Organ Procurement Organizations (OPOs), transplant centers, physicians, patients, and regulators" as a first-class citizen. It calls for the integration of mechanism design, strategic classification, causal inference, and social choice into the ML pipeline; otherwise, even the most powerful predictive models will be undermined by the strategic behavior of stakeholders upon deployment.

## Background & Motivation

**Background**: Historically, heart transplant allocation in the U.S. has been a manually designed, rigid priority queue based on medical urgency. Recently, it has been rapidly transitioning toward data-driven ML and optimization methods (e.g., the "Continuous Distribution" framework already deployed for lung transplants, with heart allocation preparations underway). Demand severely exceeds supply—over 100,000 individuals are on the waiting list in the U.S. alone.

**Limitations of Prior Work**: Existing ML solutions treat allocation as a static optimization problem (learning an optimal policy table or scoring function), completely ignoring the fact that allocation is a multi-agent game. Hospitals, OPOs, clinicians, and patients each have their own objectives and will respond strategically to policy changes. Predictors trained accurately on historical data may fail due to distribution shift after deployment, or even produce counterproductive effects.

**Key Challenge**: Supervised learning essentially "learns a mapping," while real-world features are "actively shaped by participants." This is the embodiment of Goodhart's Law— "When a measure becomes a target, it ceases to be a good measure." For example, the 2018 policy change prioritized patients using IABP (Intra-aortic balloon pump) as Status 2; consequently, the proportion of patients bridged with IABP surged from 7.0% to 24.9%—a more than three-fold increase accompanied by real clinical risks such as poor organ perfusion and bleeding.

**Goal**: To identify incentive misalignment points across the entire decision pipeline (feature gaming, out-of-sequence allocation, performance evaluation distortion, strategic list entry/exit, and preference aggregation manipulation) and provide a corresponding research agenda for the ML community.

**Key Insight**: Every stage of the pipeline can be translated into a mechanism design, strategic classification, or social choice problem. Only by explicitly modeling incentives can the next generation of allocation strategies remain robust, efficient, fair, and trustworthy under strategic behavior.

**Core Idea**: Next-generation organ allocation ML must be incentive-aware—it should not just learn "who should be prioritized," but "who should be prioritized under the condition that all participants will respond strategically."

## Method

This is not a methodology paper, but a position paper. Its "method" involves decomposing the pipeline into five diagnostic segments of misalignment and five ML research agendas, validated empirically using UNOS 2010–2024 historical data.

### Overall Architecture

The authors scan the entire heart transplant decision pipeline from top to bottom: patient feature reporting → OPO bidding (including out-of-sequence) → transplant center acceptance → list entry/exit → preference aggregation of the top-level policy. Each segment follows a fixed structure: "Phenomenon → Data Evidence → Incentive Explanation → ML Fix," finally culminating in an overarching research agenda for the ML community.

### Key Designs

1.  **Feature Gaming and Strategic Classification (Section 2)**:
    - **Function**: Diagnoses the risk where "patient-level features or statuses are strategically manipulated to gain high priority" and formalizes this as a strategic classification problem.
    - **Mechanism**: Current 6-tier urgency levels depend on device usage—clinicians can push patients across decision boundaries by choosing to use or not use certain devices. This is modeled as: feature $x$ shifts to $x'$ via cost $c(x, x')$ to cross a classifier threshold, where optimal manipulation balances the "gains from high priority" against the "manipulation cost (including clinical harm)." This can be addressed through repeated risk minimization (known to converge under certain assumptions [Perdomo et al., 2020]), combined with causal inference to distinguish which features have causal effects on medical urgency rather than mere correlations. Additionally, selective verification (randomized audits) is introduced—mechanism design has proven that a small amount of random auditing can significantly align incentives. Table 1 provides empirical evidence: 6.5% of the highest urgency group die within 3 days of listing, and 13.7% within 7 days. Waiting time based sorting rewards stable patients who "list early to hoard time," displacing those in true critical need.
    - **Design Motivation**: Current "fair" multi-tier systems treat "proxy variables for urgency" as the targets themselves, inevitably leading to Goodhart-style collapses. Only by acknowledging the existence of manipulation and explicitly modeling it can scoring functions be designed to withstand feature drift.

2.  **Out-of-Sequence Allocation and Performance Evaluation Distortion (Section 3 + Section 4)**:
    - **Function**: Exposes two types of systemic misalignment: "OPOs skipping the priority queue to direct organs to specific centers" (out-of-sequence / open offers) and the distortion of acceptance rates by SRTR's semi-annual performance evaluations.
    - **Mechanism**: Out-of-sequence allocation is intended to rescue organs at risk of being discarded, but trigger thresholds are opaque and subjectively judged by OPOs. Since CMS began monitoring OPO waste rates in 2021, the proportion of kidney out-of-sequence offers surged from 2% in 2020 to 18% in 2023, systematically favoring wealthier groups—turning a supposedly rare "safety valve" into a primary channel. Regarding performance, centers are rated in 5 tiers based on "waitlist mortality, transplant rate, and 1-year survival," incentivizing risk aversion (rejecting marginal offers). The paper uses UNOS data to show a statistically significant rebound in acceptance rates and transplant volumes in May after the April reporting window closes, consistent with a "horizon effect" of taking higher-risk cases at the start of a new window. ML fixes: Use computer vision + ex vivo perfusion to assess organ status in real-time to learn optimal thresholds for "when to trigger out-of-sequence"; replace semi-annual evaluations with continuous monitoring (CUSUM style) to weaken cyclical gaming; and use more accurate risk-adjustment models to ensure fair evaluation of small and large centers.
    - **Design Motivation**: Improving algorithms alone cannot save this pipeline—if the incentives of upstream OPOs and downstream centers remain misaligned, even the best priority queue will be bypassed. The "evaluation mechanism" itself must be included in the ML co-design scope.

3.  **Preference Aggregation and Mechanism-Level Research Agenda (Section 5 + Section 6 + Section 8)**:
    - **Function**: Includes the highest level of the pipeline—"what exactly are we optimizing"—into incentive modeling and unifies the agenda as a call to the ML community.
    - **Mechanism**: Top-level policies currently use AHP (Analytic Hierarchy Process) to sample preference weights from the community, but the Gibbard–Satterthwaite theorem has long proven that any "reasonable" voting rule can be manipulated. The paper points out: small rural centers push for "broader sharing and relaxed geographic constraints," while large urban centers do the opposite; patient families inflate weights for attributes that benefit them (e.g., the weight for "prior living donor" was set to 13.9%, while the theoretical weight under a fixed patient pool should be 0). Multiple listing is also an explicit inequity—only 2.16% of patients multi-list, yet their transplant rate is 80.44% vs. 73.06% for single-listing, concentrated among young, white, university-educated groups. The paper advocates for a three-fold ML agenda: first, replacing AHP with frugal preference elicitation + RLHF to distinguish "normative ends" from "attributes as means," letting humans vote only on ends while leaving means to algorithmic optimization; second, using counterfactual modeling to quantify when and where multi-listing actually helps the system, otherwise migrating to single-entry mechanisms; third, viewing the entire pipeline as multi-agent mechanism design, introducing credit systems to encourage centers to accept offers and using random audits to increase manipulation costs.
    - **Design Motivation**: The "objective function" of allocation is learned from gaming parties. If the preference aggregation stage is already manipulated, further ML engineering downstream only achieves local optima on the wrong target. Social choice and mechanism design must be integrated at the source of the ML pipeline.

### Empirical Basis

The paper is not purely theoretical—it systematically uses UNOS 2010–2024 heart transplant registry data to provide quantitative evidence for each diagnostic segment (IABP ratio 7.0% → 24.9%, kidney out-of-sequence 2% → 18%, May rebound after the April reporting window, 2.16% multi-listed patients achieving an 80.44% transplant rate, etc.), grounding the claim that "incentive misalignment has real-world consequences" from theoretical judgment into data observation.

## Key Experimental Results

Note: As a position paper, there are no traditional methodology experiments. The tables below summarize the key statistics used to support the arguments.

### Outcomes of Highest Urgency (Status 1) Patients (2010–2024)

| Metric | Value | Implications |
|------|------|------|
| Deaths within 3 days of listing | 6.5% | Significant mortality within one week |
| Deaths within 7 days of listing | 13.7% | Significant mortality within one week |
| Median time to transplant | 26 days | Only 10 days earlier than median time to death |
| Median time to death | 36 days | Extremely thin safety margin |
| IQR of time to death | 13–118 days | Massive heterogeneity within Status 1; single tier is insufficient |

### Systemic Evidence of Incentive Misalignment

| Phenomenon | Key Figure | Explanation |
|------|----------|------|
| IABP bridging share (Post-2018 policy) | 7.0% → 24.9% | Over three-fold growth, suspected feature gaming |
| Kidney out-of-sequence allocation ratio | 2020: 2% → 2023: 18% → Early 2026: 9% | Surged after CMS regulation; dropped after federal scrutiny |
| Multi-listed patient ratio / Transplant rate | 2.16% / 80.44% (vs. single-list 73.06%) | Significant arbitrage by wealthy groups |
| Average distance to multi-listing centers | 379 nautical miles (Max > 2200) | Cross-regional arbitrage, fairness compromised |

### Key Findings

- Once evaluation metrics are made public (CMS monitoring OPOs, SRTR rating centers), all parties immediately "reshape behavior according to indices"—rather than in the direction of "what is best for the patient."
- Sorting by waiting time disadvantages those in true critical need—they do not live long enough to accumulate priority; this institutionally makes "listing early to hoard time" the optimal strategy.
- Federal scrutiny squeezed the out-of-sequence ratio from 20% to 9% in less than two years, demonstrating that responses to incentives are immediate and powerful—in both positive and negative directions.

## Highlights & Insights

- **Materializing Goodhart's Law into a Research Agenda**: The paper does not just vaguely state that "ML will be gamed." Instead, it provides a step-by-step pipeline showing "which step is arbitraged by which game, and which mechanism design tool should be used to fix it." This "system-level misalignment map" is highly actionable for the ML community entering healthcare policy.
- **The "means vs ends" distinction has high transfer value**: Current preference aggregation tasks (including AI alignment and RLHF) tend to have humans vote on "specific solutions," which the paper argues pushes the optimization task onto humans. A better approach is to have humans vote only on "ends" and let algorithms search for "means" under those constraints. This is directly transferable to the design of preference data collection in RLHF.
- **Acknowledging the "Manipulation is not necessarily evil" counterargument**: Section 7 provides a rare, serious discussion of the counterview that "clinician feature manipulation actually corrects imperfect policies," using kidney exchange as an example to argue that "highly efficient but unexplainable" systems can be accepted by the community. This willingness to engage directly with opposing views is commendable in a position paper and enhances its credibility.

## Limitations & Future Work

- The paper is almost entirely focused on U.S. heart transplants; the specific incentive misalignments proposed (IABP gaming, out-of-sequence allocation, SRTR evaluation cycles) need to be reassessed in liver/kidney/lung systems in other countries.
- Many "incentive explanations" currently rely on correlational evidence (e.g., the May rebound), and the authors admit that "a more rigorous causal analysis remains necessary"—providing an entry point for future rigorous work.
- The proposed ML agendas are mostly directional ("should use strategic classification," "should use RLHF for aggregation") and have not yet produced end-to-end deployable systems on actual organ allocation data; these remain open questions for the community.
- While randomized audits are effective in mechanism design, their clinical implementation involves institutional questions of who audits and who bears the cost; the paper admits that "poorly designed audits can backfire."

## Related Work & Insights

- **vs. Papalexopoulos et al. (2023) (Continuous Distribution Framework)**: They pushed allocation from discrete tiers to continuous scoring to mitigate cliff-edge effects. This paper acknowledges this as an improvement but points out that the classifiers/regressors relied upon by continuous scoring are equally susceptible to feature manipulation, making continuousness a necessary but insufficient condition.
- **vs. Hardt et al. (2016), Perdomo et al. (2020) (Strategic Classification and Performative Prediction)**: Existing work provides general frameworks (manipulation cost models + convergence of repeat risk minimization). This paper "grounds" these frameworks in survival analysis and dynamic waiting lists—settings not yet fully explored by the strategic classification community—opening up new technical problems.
- **vs. Anagnostides et al. (2025) (Dynamic Heart Allocation Policy Optimization)**: Previous work by the same author team optimized allocation rules directly and discussed "whether to eliminate center rejection rights." This paper takes a broader view—single-point strategy optimization is insufficient; upstream performance evaluation and downstream preference aggregation must be co-designed.
- **vs. Conitzer et al. (2024) (Social Choice and AI Alignment)**: Both emphasize using social choice theory to support multi-party preference aggregation. This paper provides a more specific, high-stakes (life-and-death) application scenario compared to general AI alignment, which in turn makes the research goals of RLHF/social choice more concrete.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The technical tools (strategic classification, mechanism design, RLHF) are not new individually, but packaging them as an "Organ Allocation ML Agenda" supported by UNOS empirical evidence represents a new system-level problem statement for the ML community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ As a position paper, the "experiments" are UNOS data observations—covering multiple misalignment points from 2010–2024 with a complete chain of evidence. One star is withheld as several causal claims require further rigorous analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure with a consistent four-part "Phenomenon→Data→Incentive Explanation→ML Solution" format per section. Section 7's proactive dialogue with alternative views is a benchmark for position papers.
- **Value**: ⭐⭐⭐⭐⭐ It points out to the medical ML community that the real bottleneck is incentive modeling rather than model capacity, while providing researchers in strategic classification, mechanism design, and RLHF with a high-risk, high-reward real-world application scenario. Its potential impact extends far beyond the paper itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](../../NeurIPS2025/ai_safety/machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)

</div>

<!-- RELATED:END -->
