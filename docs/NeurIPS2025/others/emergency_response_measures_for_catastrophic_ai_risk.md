---
title: >-
  [Paper Note] Emergency Response Measures for Catastrophic AI Risk
description: >-
  [NeurIPS 2025][Catastrophic AI risk] This paper systematically analyzes how Frontier Safety Policies (FSPs) can be integrated into the first two stages of China's four-phase emergency response framework (prevention–early…
tags:
  - "NeurIPS 2025"
  - "Catastrophic AI risk"
  - "emergency response"
  - "frontier safety policies"
  - "AI governance"
  - "dangerous capability evaluation"
date: 2026-05-08
content_hash: 3bd5fed8dbde092a
---

# Emergency Response Measures for Catastrophic AI Risk

**Conference**: NeurIPS 2025
**arXiv**: [2511.05526](https://arxiv.org/abs/2511.05526)  
**Code**: None  
**Area**: Other
**Keywords**: Catastrophic AI risk, emergency response, frontier safety policies, AI governance, dangerous capability evaluation

## TL;DR

This paper systematically analyzes how Frontier Safety Policies (FSPs) can be integrated into the first two stages of China's four-phase emergency response framework (prevention–early warning–response–recovery), employing dangerous capability evaluations, tiered thresholds, and pre-established safety measures to address catastrophic AI risks. The analysis is further contextualized through comparisons with international practices such as the EU AI Act and California SB 53.

## Background & Motivation

**Background**: China's senior leadership has repeatedly emphasized the importance of AI emergency preparedness in recent years. The April 2025 Politburo study session, the State Council's security white paper, and TC260's *Guidelines for Security Emergency Response of Generative AI Services* collectively indicate that China is incorporating AI safety emergency management into its national emergency governance system. Internationally, the EU AI Act, California SB 53, and New York's RAISE Act are establishing analogous frameworks.

**Limitations of Prior Work**: Although the general structure of the four-phase emergency response framework (prevention and preparedness, monitoring and early warning, response and rescue, recovery and reconstruction) has been established, the concrete technical implementation of the first two phases remains underdeveloped. Existing regulations—such as the *Interim Measures* and GB/T 45654-2025—primarily address content safety and general service security, and do not yet systematically cover catastrophic risks such as weapons of mass destruction (WMDs) or loss-of-control scenarios.

**Key Challenge**: Catastrophic AI risks are characterized by unprecedented novelty and uncertainty, yet China possesses a mature emergency management architecture. The central challenge is how to integrate the emerging international frontier AI safety practices (e.g., FSPs) with China's institutional strengths.

**Goal**: To analyze how the FSP model can provide concrete technical implementation pathways for the first two proactive phases—prevention and early warning—of China's AI emergency response framework.

**Key Insight**: The authors observe that the core elements of FSPs (dangerous capability evaluations, tiered thresholds, and pre-established safety measures) closely correspond to the prevention and early warning phases of China's emergency response framework, and that major global AI companies and governments are converging toward similar safety mechanisms.

**Core Idea**: The FSP "evaluation–threshold–contingency plan" model is the most viable candidate for implementing the first two phases of China's AI emergency response framework.

## Method

### Overall Architecture

This paper is a policy analysis study and does not involve algorithm design. Its analytical framework consists of: (1) a review of China's existing AI regulatory legislation and industry self-governance frameworks; (2) a systematic comparison with international practices, including the EU Code of Practice, California SB 53, New York's RAISE Act, and the AI Seoul Summit commitments; and (3) concrete proposals for integrating FSPs into China's existing regulatory processes.

### Key Designs

1. **Core Structure of Frontier Safety Policies (FSPs)**:

    - **Function**: To provide AI developers with a systematic mechanism for managing catastrophic risks.
    - **Mechanism**: FSPs comprise three core elements—(a) *Dangerous capability threshold classification*: defining specific tiers of AI capability, such as the ability to engineer CBRN weapons or conduct autonomous AI research; (b) *Pre-deployment evaluation*: testing a model's dangerous capabilities prior to release; (c) *Tiered safety measures*: when evaluation results meet or exceed a threshold, pre-established safety measures are automatically triggered (e.g., enhanced safety filtering, restricted API access, or deployment suspension).
    - **Design Motivation**: To shift from reactive post-hoc response to proactive pre-emptive prevention. FSPs require developers to define in advance "if the model reaches capability X, implement measure Y," thereby avoiding the disorder of improvising responses during a crisis.

2. **Mapping to the Four-Phase Emergency Framework**:

    - **Function**: To demonstrate how FSPs naturally embed within China's emergency management system.
    - **Mechanism**: Prevention phase → FSPs require pre-defined thresholds and contingency plans; early warning phase → continuous capability evaluation provides real-time risk indicators (analogous to seismic monitoring); response phase → threshold-triggered measures execute automatically; recovery phase → evaluation data supports post-incident analysis and policy iteration.
    - **Design Motivation**: To leverage China's institutionalized emergency management architecture, thereby reducing the institutional cost of deploying FSPs.

3. **Registration-Based Extension Scheme**:

    - **Function**: To embed FSPs within the existing registration and approval process for generative AI service providers.
    - **Mechanism**: FSP maintenance obligations are added to the compliance documentation requirements under GB/T 45654-2025; catastrophic risk capability evaluations (e.g., WMDP-Bio, LAB-Bench benchmarks) are incorporated into the existing 31-item safety assessment checklist. At registration, developers must submit an FSP specifying the dangerous capability thresholds their current model has not yet reached, along with corresponding contingency plans.
    - **Design Motivation**: To reuse the existing "review before deployment" regulatory infrastructure and minimize additional compliance costs.

### Loss & Training

This paper is a policy analysis study and does not involve model training.

## Key Experimental Results

### Main Results

This paper contains no experiments; however, it provides a systematic comparative policy analysis:

| Regulatory Framework | Covers CBRN Risks | Covers Loss-of-Control Risks | Mandatory | Incident Reporting Deadline |
|---|---|---|---|---|
| China TC260 Framework | ✓ | ✓ | Standard (non-statutory) | Not specified |
| EU Code of Practice | ✓ | ✓ | Voluntary code | 2 days (severe) / 5 days (cybersecurity) |
| California SB 53 | ✓ | ✓ | Statutory | 24 hours |
| New York RAISE Act | ✓ | ✓ | Statutory (pending signature) | 72 hours |
| Anthropic RSP | ✓ | ✓ | Voluntary | Government notification |

### Ablation Study

| Emergency Response Phase | Current Implementation Status in China | Recommended Enhancements |
|---|---|---|
| Prevention & Preparedness | Basic regulatory framework established | Mandatory FSPs + information sharing + standardized benchmarks |
| Monitoring & Early Warning | TC260 guideline draft | Continuous capability evaluation + anomalous query monitoring |
| Response & Rescue | Basic framework | Model access suspension + compute restrictions + external experts |
| Recovery & Improvement | Not yet specified | No-fault investigations + policy updates + tabletop exercises |

### Key Findings

- Major global AI regulatory frameworks show a high degree of convergence in identifying catastrophic risk categories: CBRN weapons, cyberattacks, and loss-of-control incidents are common focal points.
- China's AI ecosystem already exhibits a substantial foundation of voluntary self-governance: 17 Chinese AI companies have signed safety commitments, and the Shanghai AI Laboratory has published a detailed frontier risk management framework.
- The FSP "evaluation–threshold–contingency plan" mechanism naturally aligns with China's principle of "tiered management."
- Early draft versions of GB/T 45654-2025 explicitly referenced catastrophic risks such as AI self-replication and malware generation, demonstrating that TC260 has prior conceptual groundwork in this area.

## Highlights & Insights

- **Precise Alignment Analysis**: The paper maps each technical element of FSPs onto the four phases of China's emergency response framework. This institutional analysis approach is both persuasive and operationally actionable. The key insight is that technical safety measures need not be designed from scratch—they can be embedded within existing institutional channels.
- **Global Convergence Perspective**: The paper systematically surveys the convergence among China, the EU, and the United States in governing catastrophic AI risks, providing an international reference for the necessity and feasibility of a "Chinese FSP." This mitigates the risk of insular policy design.
- **Elegant Registration-Based Extension**: The proposed scheme reuses the existing "review before deployment" process, requiring only incremental additions to documentation requirements and the evaluation checklist, thereby substantially reducing implementation friction.

## Limitations & Future Work

- FSPs primarily address foreseeable risks and are less effective against the most novel and unpredictable threats—yet unpredictability is precisely a defining characteristic of catastrophic AI risk.
- The effectiveness of FSPs depends heavily on evaluation quality, the reasonableness of thresholds, and corporate willingness to comply. Self-auditing may result in opacity or underreporting.
- The paper focuses on the prevention and early warning phases; the treatment of the response and recovery phases is comparatively limited.
- No quantitative framework is offered for assessing the real-world effectiveness of FSP implementation.

## Related Work & Insights

- **vs. Anthropic RSP**: Anthropic's Responsible Scaling Policy is the most influential industry FSP and the first to propose the ASL tiered safety framework. This paper uses it as the primary reference for analyzing how analogous mechanisms might be implemented in a Chinese context.
- **vs. Shanghai AI Laboratory Frontier Risk Management Framework**: This is the closest domestic Chinese analog to an FSP, incorporating a yellow-line/red-line threshold system. The paper treats it as a pioneering case of FSP adoption among Chinese AI organizations.
- **vs. EU Code of Practice**: The two frameworks exhibit strong similarities in risk categorization and incident reporting mechanism design. The EU framework places greater emphasis on cross-border compliance, whereas China's approach prioritizes integration with its existing emergency management system.

## Rating

- Novelty: ⭐⭐⭐ The policy recommendations are not entirely novel, but the analytical approach of precisely aligning FSPs with China's institutional framework is of considerable value.
- Experimental Thoroughness: ⭐⭐⭐ No experiments are included, as appropriate for a policy analysis paper; however, the institutional comparative analysis is systematic.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with rigorous argumentation and accurate, detailed policy citations.
- Value: ⭐⭐⭐⭐ Provides an operationally viable institutional integration scheme for governing catastrophic AI risks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evolutionary Learning in Spatial Agent-Based Models for Physical Climate Risk Assessment](evolutionary_learning_in_spatial_agent-based_models_for_physical_climate_risk_as.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](../../AAAI2026/others/designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)

</div>

<!-- RELATED:END -->
