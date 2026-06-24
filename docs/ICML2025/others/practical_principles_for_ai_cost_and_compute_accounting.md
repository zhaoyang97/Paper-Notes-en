---
title: >-
  [Paper Note] Practical Principles for AI Cost and Compute Accounting
description: >-
  [ICML 2025][Compute Auditing] To address the ambiguity of accounting standards for compute/cost thresholds in AI regulation, this paper proposes seven principles to close evasion loopholes (such as the distillation loophole), avoid disincentivizing safety measures, and achieve consistent implementation across firms, providing a theoretical framework for operationalizing regulations like the EU AI Act.
tags:
  - "ICML 2025"
  - "Compute Auditing"
  - "AI Regulation"
  - "Compute Thresholds"
  - "Distillation Loophole"
  - "Cost Accounting"
date: 2026-05-08
content_hash: f28106a192be103c
---

# Practical Principles for AI Cost and Compute Accounting

**Conference**: ICML 2025  
**arXiv**: [2502.15873](https://arxiv.org/abs/2502.15873)  
**Code**: None  
**Area**: AI Governance / Policy  
**Keywords**: Compute Auditing, AI Regulation, Compute Thresholds, Distillation Loophole, Cost Accounting

## TL;DR

To address the ambiguity of accounting standards for compute/cost thresholds in AI regulation, this paper proposes seven principles to close evasion loopholes (such as the distillation loophole), avoid disincentivizing safety measures, and achieve consistent implementation across firms, providing a theoretical framework for operationalizing regulations like the EU AI Act.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: **The Threshold Dilemma of AI Regulation**: The EU AI Act defines high-risk AI systems based on compute thresholds (such as $10^{25}$ FLOPs), and bills like California's SB 53 and New York's A6453A adopt similar strategies. The advantages of compute/cost thresholds are: they correlate with capability risks, are quantifiable, can be measured early in development, and are externally verifiable.

**Core Problem**: Currently, there is a lack of unified accounting standards—severe ambiguity exists regarding "what to calculate and how to calculate it." When developers are incentivized to make their models look "cheap," narrow accounting standards are creatively exploited. **The most representative vulnerability is the distillation loophole**: DeepSeek-V3 quoted a cost of 6 million USD but did not include the training cost of DeepSeek-R1. Narrowly defined accounting conceals the true development investment.

**Insufficiencies of Existing Guidelines**: The Frontier Model Forum (2024) and the European Commission (2025) each provide guidelines, but both contain exploitable exclusion clauses—the former allows the exclusion of "discarded branches" and the cost of distilling teacher models, leaving loopholes for bypassing the thresholds. This paper quotes Jón Danielsson's maxim: "Any statistical relationship used for policy collapses," emphasizing that accounting standards must be game-resistant.

## Method

### Overall Architecture

The framework proposed in this paper is designed around three goals: (1) reducing the space for strategic gaming; (2) avoiding disincentivizing responsible risk mitigation practices; (3) achieving consistent implementation across firms and jurisdictions. The framework consists of seven principles, each targeting a specific accounting ambiguity.

### Key Designs

1. **Principle 1: Account for all upstream costs and compute of the project**:
    - Function: Requires developers to report all upstream technical costs and compute of the final AI system.
    - Mechanism: Includes data curation/compression, teacher model training (distillation), zero operations in dropout/sparse operations, etc. The key is to close the distillation loophole—if the student model relies on the knowledge of the teacher model, the teacher's training compute must be accounted for.
    - Design Motivation: Calculating only "theoretically necessary" or "directly relevant" compute can be exploited; narrow accounting is the root cause of loopholes.

2. **Principles 3+6: Exclude pure safety activities + use independent dual thresholds**:
    - Function: Allows developers to exempt activities intended solely to mitigate societal risks (CSAM filtering, safety testing, etc.); simultaneously requires independent thresholds for both cost and compute.
    - Mechanism: Exemptions for safety activities require auditable proof (demonstrating they do not enhance capabilities) to prevent "safety washing"; dual thresholds act as mutual failsafes—machine-generated data is cheap but compute-intensive, while human-generated data is expensive but compute-sparse; adjusting their proportions can bypass a single threshold.
    - Design Motivation: A single threshold can be bypassed by adjusting data sources (human vs. machine); without a safety exemption, responsible development would be disincentivized.

3. **Principles 5+7: Itemized accounting reporting + regular updates of standards**:
    - Function: Requires developers to submit auditable itemized reports (including descriptions, purposes, estimation methods, and ultimate compute for each activity); standards should be updated quarterly or semi-annually.
    - Mechanism: Analogous to SEC reporting requirements in financial accounting, reporting has both direct oversight effects and the utility of indirectly incentivizing due diligence; the rapid evolution of AI technology makes any static framework quickly obsolete.
    - Design Motivation: Transparency and accountability are the foundations of effective regulation; efficiency gains and new technical paradigms require standards to adapt continuously.

### Loss & Training

Since this is a policy analysis paper, it does not involve model training. The core contribution is the design of a principled framework rather than a technical implementation.

## Key Experimental Results

### Main Results

| Dimension | Ours | Frontier Model Forum (2024) | European Commission (2025) |
|------|------|---------------------------|------------------|
| Account for all upstream compute | ✓ | ✗ (Excludes discarded branches) | Reference Only |
| Account for distilling teacher models | ✓ | ✗ (Distillation loophole) | ✓ |
| Exclude public resources | ✓ | ✓ | — |
| Exclude safety activities | ✓ | — | — |
| Reasonable estimation | ✓ | ✓ | — |
| Itemized accounting reporting | ✓ | — | — |
| Independent dual thresholds (cost + compute) | ✓ | — | — |
| Regular updates of standards | ✓ | — | — |

### Ablation Study

| Loophole Type | Exploitable Exclusion Clause | Response of Our Principles |
|----------|-----------------|--------------|
| Distillation loophole | Excluding teacher model training | Principle 1: Account for all upstream |
| Public release loophole | Open-sourcing first then using to "reset" | Principle 2: Resources publicly released by developers themselves within 6 months are still accounted for |
| Gaming a single threshold | Adjusting the ratio of human/machine data | Principle 6: Independent dual thresholds for cost + compute |
| "Safety washing" | Disguising capability-enhancing activities as safety measures | Principle 3: Requires auditable proof of no capability enhancement |

### Key Findings

- The recommendations of the Frontier Model Forum present the highest risk of loopholes due to the exclusion of distilling teacher models.
- The case of DeepSeek-V3 (quoted at 6 million USD but excluding R1 training) is a typical empirical demonstration of the distillation loophole.
- Cost and compute can be intentionally decoupled by adjusting the proportion of data sources, making dual thresholds act as mutual failsafes.
- Itemized reporting not only provides regulatory information but also has an indirect incentive effect—firms will be more cautious when knowing they have to submit detailed reports.

## Highlights & Insights

- **Clear identification and systematic closure of the distillation loophole**: Taking DeepSeek-V3 as an example, pointing out how narrow accounting conceals the true cost is the most powerful argument in the paper.
- **Failsafe design of dual thresholds**: Recognizing that cost and compute can be intentionally decoupled (machine-generated data vs. human-labeled data), a dual triggering mechanism is employed as a safeguard.
- **Balance between safety activity exemption and prevention of "safety washing"**: It neither disincentivizes responsible development nor laxly allows loopholes, requiring auditable proof through a sophisticated design.
- **Quoting Danielsson's maxim**: "Any statistical relationship used for policy collapses"—a profound warning about the risks of regulatory reliance on a single proxy metric.
- Analogizing to SEC reporting in financial accounting, "fair value" concepts, etc., makes the framework understandable across domains.

## Limitations & Future Work

- Specific threshold values (e.g., how high the FLOP threshold should be set) are not discussed.
- Treatment of gray areas is insufficiently clear—for instance, there are no clear rules on whether foundational research investments should be accounted for in model development.
- Accounting attribution in multi-entity collaborations (federated learning, crowdsourcing) is only briefly discussed.
- The costs of accounting enforcement and auditing itself could be significant, imposing a disproportionate burden on small and medium-sized developers.
- Rapidly evolving AI technologies (such as test-time compute scaling) may challenge the applicability of the existing framework.
- Information security risks—sharing sensitive development information with regulators may lead to competitive leakage.

## Related Work & Insights

- **EU AI Act Article 51**: Defines high-impact general-purpose AI models using a threshold of $10^{25}$ FLOPs.
- **Frontier Model Forum (2024)**: Compute accounting guidelines released by an industry coalition, which excludes distilling teacher models.
- **European Commission (2025)**: Draft guidelines accompanying the EU AI Act, supporting the inclusion of data curation and distillation.
- **Heim & Koessler (2024)**: Discusses the properties and functions of training compute thresholds as a regulatory tool.
- **Hooker (2024)**: Points out the limitations of compute thresholds as a governance strategy.
- **Kaplan et al. (2020)**: Scaling laws provide a scientific basis for the relationship between compute and capability.
- Insights: AI governance requires technical infrastructure similar to financial regulation (standards, reporting mechanisms, auditing processes).

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically propose principles for AI compute accounting, with in-depth identification and analysis of the distillation loophole.
- Experimental Thoroughness: ⭐⭐⭐ As a policy analysis paper, it lacks traditional experiments, but the comparative analysis with existing guidelines and case studies of loopholes are comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly logical arguments, well-organized principles, and appropriate citations.
- Value: ⭐⭐⭐⭐ Fills a theoretical gap in compute auditing within AI governance, with the seven principles having direct policy influence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](../../CVPR2025/others/practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[ICML 2025\] Position: AI Evaluation Should Learn from How We Test Humans](position_ai_evaluation_should_learn_from_how_we_test_humans.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](../../NeurIPS2025/others/obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](../../ACL2025/others/a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)

</div>

<!-- RELATED:END -->
