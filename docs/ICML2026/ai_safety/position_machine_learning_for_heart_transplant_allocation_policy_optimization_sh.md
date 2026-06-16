---
title: >-
  [Paper Note] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives
description: >-
  [ICML 2026][AI Safety][RLHF] This is an ICML 2026 position paper: Using historical UNOS data, the authors argue that next-generation ML strategies for the U.S. heart transplant allocation system must treat incentive misalignments among "Organ Procurement Organizations (OPOs), transplant centers, physicians, patients, and regulators" as first-class
tags:
  - ICML 2026
  - AI Safety
  - RLHF
date: 2026-05-08
content_hash: 63b8328552a920ee
---
# Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives

**Conference**: ICML 2026  
**arXiv**: [2602.04990](https://arxiv.org/abs/2602.04990)  
**Code**: None (Position Paper)  
**Area**: AI Safety / Mechanism Design / Strategic Classification / Medical Policy  
**Keywords**: Organ Allocation, Incentive Alignment, Strategic Classification, Mechanism Design, RLHF

## TL;DR
This is an ICML 2026 position paper: Using historical UNOS data, the authors argue that next-generation ML strategies for the U.S. heart transplant allocation system must treat incentive misalignments among "Organ Procurement Organizations (OPOs), transplant centers, physicians, patients, and regulators" as first-class citizens. The paper calls for the integration of mechanism design, strategic classification, causal inference, and social choice into the ML pipeline; otherwise, even the most powerful predictive models will be undermined by the strategic behavior of stakeholders upon deployment.

## Background & Motivation

**Background**: The historical U.S. heart transplant allocation system was a manually designed, rigid priority queue based on medical urgency. Recently, it has been rapidly transitioning toward data-driven ML/optimization methods (e.g., the "Continuous Distribution" framework already deployed for lung transplants, with heart allocation preparations underway). Demand severely exceeds supply—over 100,000 people are on the waiting list in the U.S. alone.

**Limitations of Prior Work**: Existing ML solutions treat allocation as a static optimization problem (learning an optimal policy table or scoring function), completely ignoring the fact that allocation is a multi-agent game. Hospitals, OPOs, clinicians, and patients each have their own objectives and will respond strategically to policy changes. Predictors trained accurately on historical data may fail due to distribution shifts or even produce counterproductive effects after deployment.

**Key Challenge**: Supervised learning is essentially "learning a mapping," whereas real-world features are "actively shaped by participants." This is a manifestation of Goodhart’s Law: "When a measure becomes a target, it ceases to be a good measure." For example, the 2018 policy change gave high priority (Status 2) to patients with IABP (Intra-aortic balloon pump); subsequently, the proportion of patients bridged with IABP surged from 7.0% to 24.9%—a more than three-fold increase accompanied by real clinical risks such as poor organ perfusion and bleeding.

**Goal**: To identify points of incentive misalignment across the entire decision pipeline (feature gaming, out-of-sequence allocation, performance evaluation distortion, strategic listing/de-listing, preference aggregation manipulation) and provide a corresponding research agenda for the ML community.

**Key Insight**: Translate each stage of the pipeline into a mechanism design, strategic classification, or social choice problem. Only by explicitly modeling incentives can the next generation of allocation policies remain robust, effective, fair, and trustworthy under strategic behavior.

**Core Idea**: Next-generation organ allocation ML must be "incentive-aware"—not just learning "who should be prioritized," but learning "who should be prioritized under the condition that all participants will respond strategically."

## Method

As a position paper rather than a methodology paper, this work does not introduce new algorithms. Instead, its "method" involves a systematic demonstration: scanning the heart transplant decision pipeline from top to bottom—patient feature reporting $\rightarrow$ OPO offering (including out-of-sequence) $\rightarrow$ transplant center acceptance $\rightarrow$ listing/de-listing $\rightarrow$ top-level policy preference aggregation. At each stage, the authors locate an incentive mismatch, provide quantitative evidence using UNOS 2010–2024 registry data, translate it into a specific mechanism design/strategic classification/social choice problem, and synthesize a research agenda for the ML community. The core claims are expanded below:

**1. Claim: Urgency tiers are subject to arbitrage through feature gaming and must be modeled as strategic classification.** Current six-tier urgency levels depend on device usage. Clinicians can push a patient across decision boundaries by choosing whether to install a specific device. The paper formalizes this as strategic classification: a patient with feature $x$ incurs a cost $c(x, x')$ to shift to $x'$ to cross the classifier threshold. Optimal manipulation balances the "gains from higher priority" against the "costs of manipulation" (including clinical harm like bleeding). A typical empirical example is the 2018 policy change for IABP; the surge from 7.0% to 24.9% represents a "Goodhart collapse" where the "proxy for urgency" became the goal itself. The proposed ML remedy includes using repeated risk minimization to handle actively shaped distributions (known to converge under certain assumptions [Perdomo et al., 2020]), applying causal inference to distinguish features with true medical effects from mere correlations, and using selective verification (random audits) to raise the cost of manipulation. Mechanism design has proven that even a small amount of random auditing can significantly align incentives. Table 1 also reveals deeper issues: 6.5% of the highest urgency group die within 3 days of listing, yet waiting time rewards stable patients who "hoard time" by listing early, crowding out those in actual crisis.

**2. Claim: Out-of-sequence allocation and periodic performance evaluations are system-level misalignments; the "evaluation mechanism" itself must be part of the ML co-design.** Out-of-sequence (open) offers allow OPOs to bypass the priority queue and direct organs to specific centers, intended to save organs at risk of waste. However, trigger thresholds are opaque and subjective. Since CMS began monitoring OPOs for waste rates in 2021, the proportion of out-of-sequence kidney allocations jumped from 2% in 2020 to 18% in 2023, with systemic bias toward wealthier groups—a "safety valve" turned into a primary channel. Performance metrics are similarly distorted: SRTR rates centers on 5 tiers every six months based on waitlist mortality and 1-year survival. This incentivizes risk-aversion, causing centers to reject marginal offers. UNOS data shows acceptance rates and transplant volumes bounce back significantly in May after the April reporting window closes, consistent with a "horizon effect" as centers take higher risks at the start of a new window. The ML agenda here involves: using computer vision and ex-vivo perfusion for real-time organ assessment to learn optimal thresholds for triggering out-of-sequence offers; replacing semi-annual evaluations with CUSUM-style continuous monitoring to weaken periodic gaming; and using better risk-adjustment models to ensure fair evaluation of small vs. large centers. The argument is that if upstream OPO and downstream center incentives remain misaligned, the best priority queue will be bypassed.

**3. Claim: Since optimization objectives themselves are learned from strategic actors, social choice and mechanism design must enter at the source of the ML pipeline.** Top-level policies currently use AHP (Analytic Hierarchy Process) to extract preference weights from the community. However, the Gibbard–Satterthwaite theorem proves that any "reasonable" voting rule can be manipulated. Small rural centers push for "broader sharing and fewer geographic constraints," while large urban centers do the opposite. Patient advocates push for attributes beneficial to them (e.g., the weight for "prior living donors" was adjusted to 13.9%, even though its theoretical weight in a fixed pool should be near 0). Multi-listing is a more explicit form of unfairness: only 2.16% of patients multi-list, yet their transplant rate of 80.44% is significantly higher than the 73.06% for single-listing patients; these patients are predominantly young, white, and college-educated, with an average cross-region distance of 379 nautical miles. The paper advocates for: replacing AHP with frugal preference elicitation plus RLHF; distinguishing "normative ends" from "attributes as means" (letting humans vote on ends while algorithms optimize means); using counterfactual modeling to quantify when multi-listing actually helps the system; and treating the entire pipeline as multi-agent mechanism design, potentially introducing credit systems to encourage offer acceptance. The core argument is that if the preference aggregation stage is manipulated, downstream ML engineering is merely performing local optimization on the wrong objective.

The methodological backbone is empirical: the paper systematically uses UNOS 2010–2024 heart transplant registry data to provide quantitative evidence for every diagnosis (IABP ratios, kidney out-of-sequence surges, May rebounds, multi-listing transplant rates), grounding theoretical claims of "incentive misalignment" in data observation.

## Key Experimental Results

Note: As a position paper, there are no traditional algorithmic experiments. The tables below summarize the key statistics used to support the arguments.

### Outcomes for Highest Urgency (Status 1) Patients (2010–2024)

| Metric | Value | Meaning |
|------|------|------|
| Death within 3 days of listing | 6.5% | Significant mortality within one week |
| Death within 7 days of listing | 13.7% | Significant mortality within one week |
| Median time to transplant | 26 days | Only 10 days earlier than median time to death |
| Median time to death | 36 days | Extremely thin margin of safety |
| Time to death IQR | 13–118 days | Massive heterogeneity within Status 1; single tier is insufficient |

### System-level Evidence of Incentive Misalignment

| Phenomenon | Key Figure | Explanation |
|------|----------|------|
| IABP Bridge Ratio (Post-2018 Policy) | 7.0% → 24.9% | Three-fold increase, suspected feature gaming |
| Kidney Out-of-Sequence Proportion | 2020: 2% → 2023: 18% → early 2026: 9% | Surged after CMS regulation; dropped after federal audit |
| Multi-listing Patient Ratio / Transplant Rate | 2.16% / 80.44% (vs 73.06% single-listing) | Significant arbitrage by wealthy groups |
| Average Distance for Multi-listing Centers | 379 nmi (Max > 2200 nmi) | Cross-regional arbitrage; impaired fairness |

### Key Findings

- Once evaluation metrics are made public (CMS monitoring OPOs, SRTR rating centers), stakeholders immediately "reshape behavior according to metrics" rather than focusing on what is best for the patient.
- Waiting time-based ranking disadvantages the most critical patients who cannot survive long enough to accumulate priority; this institutionally makes "listing early to hoard time" the optimal strategy.
- Federal audits were able to reduce out-of-sequence rates from 20% to 9% in less than two years, proving that stakeholder responses to incentives are immediate and powerful—in both positive and negative directions.

## Highlights & Insights

- **Visualizing Goodhart’s Law as a Research Agenda**: Instead of vaguely stating "ML will be gamed," the paper maps out which step is arbitraged by which type of behavior and which mechanism design tool should fix it. This "system-level mismatch map" is highly actionable for the ML community entering healthcare policy.
- **Value of "Means vs. Ends" Distinction**: Current preference aggregation tasks (including AI Alignment and RLHF) often ask humans to vote on "specific solutions," which effectively pushes the optimization task onto the human. The paper argues for having humans vote only on "ends," allowing algorithms to search for "means" within those constraints—an insight directly applicable to RLHF preference data collection.
- **Acknowledging Dissenting Views**: Section 7 seriously discusses the counter-argument that clinician "manipulation" might actually be correcting imperfect policies. Using kidney exchange as an example, it demonstrates that "efficient but non-interpretable" systems can still be accepted by the community. This adds significant credibility to the position.

## Limitations & Future Work

- The paper focuses almost entirely on U.S. heart transplants. Specific misalignments (IABP gaming, out-of-sequence allocation) need to be reassessed in liver/kidney/lung systems and other national contexts.
- Many "incentive explanations" currently rely on correlational evidence (e.g., the May rebound). The authors concede that "a more more rigorous causal analysis remains necessary."
- The proposed ML agenda is largely directional (e.g., "should use strategic classification"). No end-to-end deployable system has yet been run on organ allocation data, leaving this as an open problem for the community.
- While randomized audits are effective in theory, practical implementation requires addressing who performs the audit and who bears the cost. Improperly designed audits could be counterproductive.

## Related Work & Insights

- **vs. Papalexopoulos et al. (2023) (Continuous Distribution Framework)**: They suggest moving from discrete tiers to continuous scoring to mitigate cliff-edge effects. Ours acknowledges this improvement but notes that continuous scorers still rely on classifiers/regressors that are susceptible to feature manipulation; continuity is necessary but not sufficient.
- **vs. Hardt et al. (2016), Perdomo et al. (2020) (Strategic Classification and Performative Prediction)**: Existing work provides general frameworks (cost models + convergence of RRM). Ours "grounds" these frameworks in survival analysis and dynamic waitlists—settings not yet fully explored by the strategic classification community.
- **vs. Anagnostides et al. (2025) (Dynamic Heart Allocation Policy Optimization)**: Previous work by the same team optimized allocation rules and discussed center "rejection rights." Ours broadens the scope: optimizing the policy in isolation is insufficient; upstream performance evaluation and downstream preference aggregation must be co-designed.
- **vs. Conitzer et al. (2024) (Social Choice and AI Alignment)**: Both emphasize using social choice theory for multi-agent preference aggregation. Ours provides a more specific, high-stakes application scenario (life and death), which in turn makes RLHF/social choice research goals more concrete.

## Rating
- Novelty: ⭐⭐⭐⭐ The technical tools (strategic classification, mechanism design, RLHF) are established, but packaging them as an "Organ Allocation ML Agenda" backed by UNOS evidence is a novel system-level problem statement for the ML community.
- Experimental Thoroughness: ⭐⭐⭐⭐ The "experiments" are UNOS data observations covering multiple mismatch points over 15 years. The evidence chain is complete, though causal analysis could be strengthened.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure; each section follows a "phenomenon $\rightarrow$ data $\rightarrow$ incentive explanation $\rightarrow$ ML solution" pattern. Section 7's inclusion of alternative views is a gold standard for position papers.
- Value: ⭐⭐⭐⭐⭐ Points out that the real bottleneck for healthcare ML is incentive modeling rather than model capacity. It provides strategic classification and mechanism design researchers with a high-stakes real-world application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/ai_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](../../NeurIPS2025/ai_safety/machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)

</div>

<!-- RELATED:END -->
