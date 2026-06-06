---
title: >-
  [Paper Note] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems
description: >-
  [ICML2026][Institutional readiness] Addressing the widespread phenomenon of public sector AI systems being "technically feasible but deployment failures…
tags:
  - "ICML2026"
  - "Institutional readiness"
  - "AI deployment"
  - "public sector"
  - "responsible AI"
  - "deployment governance"
date: 2026-05-08
content_hash: b6378bedd00e5e70
---

# Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems

**Conference**: ICML2026  
**arXiv**: [2605.17203](https://arxiv.org/abs/2605.17203)  
**Code**: None  
**Area**: AI Governance/Deployment Policy  
**Keywords**: Institutional readiness, AI deployment, public sector, responsible AI, deployment governance  

## TL;DR
Addressing the widespread phenomenon of public sector AI systems being "technically feasible but deployment failures," this paper proposes the **Institutional Alignment Readiness (IAR)** framework. This five-dimensional assessment tool evaluates whether recipient institutions are equipped for responsible AI deployment across institutional compatibility, data ecosystem maturity, human oversight capacity, fiscal sustainability, and regulatory alignment.

## Background & Motivation

**Background**: The responsible AI field has produced numerous principles, checklists, and documentation tools (e.g., Model Cards, Datasheets for Datasets, NIST AI RMF) to evaluate the technical attributes of models and datasets. These tools are mature in assessing accuracy, robustness, and fairness.

**Limitations of Prior Work**: AI systems in the public sector frequently stall between "prototype" and "scale," where bottlenecks are often not the model quality itself. Systems performing well in internal tests may fail to launch because recipient institutions lack approval processes, data-sharing agreements, human oversight capacity, operational budgets, or legal mandates. Existing frameworks evaluate artifacts and developer-side processes rather than the institutional readiness of the end-users.

**Key Challenge**: There is a systemic misalignment between existing evaluation tools and real-world deployment needs—they assess the "artifact," while the "institution" determines success. A system passing all technical evaluations may still fail due to legal ambiguities in cross-agency data sharing, missing referral pathways, or under-trained frontline staff.

**Goal**: To construct a practical, decision-oriented institutional readiness framework that helps teams answer a critical question before scaling: "Is this institution ready to deploy this system at this scope right now?"

**Key Insight**: Based on two anonymized large-scale AI deployment cases in public education systems (an image-based anthropometric screening tool and a speech-analysis early learning risk identification system), the authors induce common dimensions of institutional barriers from actual deployment failures.

**Core Idea**: Shift the evaluation of deployment readiness from the "AI artifact" to the "recipient institution," proposing the IAR framework as a complementary layer to existing model evaluation tools.

## Method

### Overall Architecture
IAR is a pre-deployment assessment framework that adds a secondary layer of evaluation above traditional artifact-level assessments (model and dataset evaluations). It focuses on whether the recipient institution possesses the institutional conditions for responsible AI use. Its output is not a single score but phased deployment recommendations: **no-go**, **pilot-only**, or **broader deployment**.

### Key Designs

1.  **Five-Dimensional Readiness Assessment System**:
    - **Function**: Systematically evaluates institutional deployment capacity across five independent and necessary dimensions.
    - **Mechanism**: Each dimension corresponds to a category of institutional constraints observed in the case studies. The dimensions are: (1) Institutional & Operational Compatibility (approval chains, workflow fit, operator training, deployment windows); (2) Data Ecosystem Maturity (target group representation, data-sharing agreements, labeling capacity); (3) Human Oversight Capacity (qualified reviewers, referral pathways, anti-discretion protocols); (4) Fiscal Sustainability (post-pilot budget, maintenance, and retraining plans); (5) Regulatory Alignment Readiness (privacy compliance, consent procedures, grievance paths).
    - **Design Motivation**: Existing frameworks (Model Cards for models, Datasheets for datasets, NIST RMF for governance processes) fail to answer whether the recipient institution is ready; these five dimensions specifically fill those blind spots.

2.  **Phased Deployment Decision Logic**:
    - **Function**: Transforms deployment readiness from a binary judgment into incremental phase management.
    - **Mechanism**: Rather than using hard thresholds or weighted scores, the framework categorizes deficiencies as **blocking** (must stop), **scoping** (limit to pilot), or **monitoring** (proceed with tracking). A system can reside in any of four stages: "Not Ready → Internal Validation → Limited Pilot → Broader Deployment."
    - **Design Motivation**: Public sector AI deployment is incremental and conditional in practice, not a one-size-fits-all binary decision; a unified threshold would decrease the framework's applicability across different institutions.

3.  **Inductive Construction via Dual Case Studies**:
    - **Function**: Provides empirical support for the framework through real-world deployment cases.
    - **Mechanism**: Based on two anonymized AI projects in large public education systems that reached technical feasibility but stalled institutionally. Case A (Image-based screening) was hindered by data representation gaps, missing referral paths, and legal hurdles in cross-departmental sharing. Case B (Speech analysis) was forced to pivot due to data infeasibility and was later constrained by stakeholder alignment.
    - **Design Motivation**: Instead of abstract theoretical derivation, dimensions are extracted from actual failure modes to ensure practical relevance.

## Key Experimental Results

### IAR Five-Dimension Evaluation Matrix

| IAR Dimension | Observable Indicators | Typical Failure Mode |
| :--- | :--- | :--- |
| Institutional Compatibility | Documented approval chains, workflow adaptation, training plans, deployment windows | Tech is ready but launch fails due to pending approvals, workflow mismatch, or unprepared operators |
| Data Ecosystem Maturity | Dataset representation, sharing agreements, labeling capacity, retention/deletion policies | Model performs well in dev but cannot scale due to missing or slow access to target population data |
| Human Oversight Capacity | Qualified reviewers, explicit veto power, referral pathways, personnel continuity | Human-in-the-loop becomes pro-forma; edge cases go unreported; harmful outputs lack expert intervention |
| Fiscal Sustainability | Post-pilot budget, maintenance/retraining plans, infrastructure cost estimates, leadership transition contingency | Runs well during pilot but cannot be maintained or updated once initial funding is exhausted |
| Regulatory Alignment | Privacy compliance, legal basis for collection, ethical review, consent/notification, appeal paths | Deployment is delayed or halted due to legal classification, consent issues, or cross-agency data usage |

### Comparison of Evaluation Blind Spots (Existing Frameworks vs. IAR)

| IAR Dimension | Example Existing Mechanisms | Target of Evaluation | Usually Missed in Deployment |
| :--- | :--- | :--- | :--- |
| Institutional Compatibility | Model Cards, NIST AI RMF | Model behavior, intended use, governance advice | Presence of specific approval chains, frontline workflow fit, training feasibility |
| Data Ecosystem | Datasheets, Fairness metrics | Dataset attributes, distributional robustness | Whether target population data can be accessed, shared, and updated at the required scale |
| Human Oversight | HITL guidelines, Impact assessments | Whether a review step is designed | Whether qualified reviewers, referral paths, and veto powers actually exist and are sustainable |
| Fiscal Sustainability | **No standard ML evaluation** | Outside technical scope | Survival post-pilot, including maintenance, retraining, and continuity across leadership cycles |
| Regulatory Alignment | Privacy-preserving ML, Legal checklists | Privacy at the data processing level | Jurisdiction-specific consent, data classification, and cross-agency sharing requirements |

### Key Findings
- **Case A** (Image-based screening): Initial development took only 2 months to reach technical readiness, but scaling data collection to more schools required over 6 additional months because approvals and access had to be negotiated site-by-site and were constrained by school calendars.
- **Case B** (Speech analysis): Forced to pivot entirely before deployment because required data was unavailable; data feasibility acted as a decisive institutional constraint. Stakeholder alignment remained a core challenge after the pivot.
- **General Pattern**: Technical evaluations do not explain deployment trajectories. Institutional factors like approval delays, referral gaps, and data-sharing restrictions determine whether a system moves from validation to pilot to scale.
- **Interdependencies**: Dimensions have prerequisite relationships; for example, regulatory alignment is often a prerequisite for data ecosystem maturity (e.g., establishing the legal basis for sharing health-related student data in Case A).

## Highlights & Insights
- **Paradigm Shift in Evaluation**: Moving the focus of readiness from the "artifact" to the "institution." This shift, while conceptually simple, precisely fills structural blind spots in current responsible AI frameworks—none of which answer "is the institution ready?"
- **Pragmatic Non-Quantitative Design**: Intentionally avoiding a weighted scoring system, instead categorizing flaws as blocking/scoping/monitoring. This aligns with the incremental decision-making reality of the public sector.
- **Unique Contribution of "Fiscal Sustainability"**: Among the five dimensions, fiscal sustainability is the only one with absolutely no corresponding standard ML evaluation mechanism, highlighting a non-technical risk frequently overlooked by technical teams.

## Limitations & Future Work
- **Limited Validation Scope**: The framework is built on only two anonymized cases within public education systems in one country; it has yet to be validated in healthcare, social services, or international contexts.
- **Lack of Quantitative Tools**: Currently a qualitative framework without standardized scales, thresholds, or dimensional weights, which may limit consistency and comparability in application.
- **Supplier-Side Readiness Excluded**: Focuses only on the recipient institution, not the developer team’s capacity for maintenance, audit response, or knowledge transfer.
- **Future Directions**: Customizing readiness expectations for different AI risk levels (e.g., screening vs. administrative tools) and cross-domain validation to determine universal vs. context-specific dimensions.

## Related Work & Insights
- **Sociotechnical Critiques (Selbst et al., 2019)**: Provides the theoretical foundation by warning that systems cannot be assumed to move across contexts without rebuilding organizational supports.
- **Data Cascades (Sambasivan et al., 2021)**: Demonstrates that data failures in high-stakes AI reflect upstream organizational conditions rather than dataset flaws.
- **Distinction from AI Maturity Models (Dreyling et al., 2024)**: While maturity models assess macro-level organizational AI capability, IAR evaluates specific deployment conditions for a specific system—an organization may be "AI-ready" generally but lack the specific referral paths or legal basis for a particular model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](comprehensive_ai_governance_requires_addressing_non-model_gains.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICML 2026\] Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](../../AAAI2026/others/designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[CVPR 2026\] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](../../CVPR2026/others/rethinking_snn_online_training_and_deployment_grad.md)

</div>

<!-- RELATED:END -->
