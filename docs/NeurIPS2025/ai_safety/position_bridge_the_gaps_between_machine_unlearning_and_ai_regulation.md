---
title: >-
  [Paper Note] Position: Bridge the Gaps between Machine Unlearning and AI Regulation
description: >-
  [NeurIPS 2025 Oral][AI Safety][Machine Unlearning] This paper systematically analyzes six potential application scenarios of Machine Unlearning (MU) in compliance with the EU AI Act (AIA), identifies the technical gaps between the state of the art and actual regulatory requirements in each scenario, and calls on the research community to bridge these gaps in order to realize the potential of MU in AI governance.
tags:
  - "NeurIPS 2025 Oral"
  - "AI Safety"
  - "Machine Unlearning"
  - "AI Regulation"
  - "EU AI Act"
  - "Data Privacy"
  - "Compliance"
date: 2026-05-08
content_hash: 5a3837b63b999343
---

# Position: Bridge the Gaps between Machine Unlearning and AI Regulation

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2502.12430](https://arxiv.org/abs/2502.12430)  
**Code**: Unavailable  
**Area**: AI Safety
**Keywords**: Machine Unlearning, AI Regulation, EU AI Act, Data Privacy, Compliance

## TL;DR

This paper systematically analyzes six potential application scenarios of Machine Unlearning (MU) in compliance with the EU AI Act (AIA), identifies the technical gaps between the state of the art and actual regulatory requirements in each scenario, and calls on the research community to bridge these gaps in order to realize the potential of MU in AI governance.

## Background & Motivation

### State of the Field

**Background**: Since its inception, MU has been primarily motivated by the "right to be forgotten" under the GDPR. A new wave of AI regulations, exemplified by the EU AI Act (AIA), is now being enforced, and researchers have begun exploring whether MU can assist compliance with these frameworks.

The position of this paper, however, is that **the potential of MU to support AI regulatory compliance can only be realized if researchers proactively bridge the existing technical gaps**. There remain considerable gaps between the current state of MU and its anticipated use in regulatory compliance. The rationale for selecting the AIA as the case study is as follows:

### Limitations of Prior Work

**Limitations of Prior Work**: The AIA is the world's first comprehensive AI regulatory act, having entered into force in 2024.

### Root Cause

**Key Challenge**: The AIA adopts a risk-based, tiered regulatory approach with detailed requirements for high-risk AI systems and general-purpose AI (GPAI) models.

### Starting Point

**Key Insight**: The principles of the AIA are representative of global AI regulation, and the findings of this analysis are transferable to other regulatory frameworks.

## Method

### Overall Architecture

Rather than proposing a new algorithm, this paper **establishes a comprehensive mapping between MU and AIA compliance**, identifies six key application scenarios, and analyzes the technical challenges within each.

### Key Designs (Analysis of Six Application Scenarios)

1. **Accuracy**:
    - AIA requirement: High-risk AI systems must achieve an appropriate level of accuracy consistent with their intended purpose and the state of the art.
    - Potential MU role: Unlearning mislabeled, outdated, or anomalous training data to improve accuracy.
    - Technical gaps: Identifying all data points responsible for inaccuracies is inherently difficult; partial unlearning may be counterproductive; approximate unlearning should not be expected to yield higher accuracy than exact retraining.
    - Formal definition: An unlearning algorithm $U$ is an $(\epsilon, \delta)$-unlearner if the distribution of $U(M; D_f, D_r)$ is $(\epsilon, \delta)$-close to the distribution of $A(D_r)$.

2. **Bias**:
    - AIA requirement: High-risk AI systems and GPAI models posing systemic risks must mitigate discriminatory bias.
    - Potential MU role: Unlearning data points or training patterns that introduce bias.
    - Technical gaps: If bias originates from data absence rather than data presence, MU cannot help; how to evaluate bias remains an "open problem"; MU is primarily a post-processing approach and cannot address the root causes of bias.

3. **Confidentiality Attacks**:
    - AIA requirement: Detection and mitigation of confidentiality attacks, including membership inference and data reconstruction.
    - Potential MU role: Unlearning confidential information that is vulnerable to such attacks.
    - Technical gaps: Unlearning certain data points may inadvertently expose the privacy of neighboring data (the "Onion Effect"); over-unlearning may expose the membership of the forgotten data itself (the "Streisand Effect"); approximate unlearning involves trade-offs with accuracy and bias.
    - Alternative: Differential privacy (DP) may be preferable in certain scenarios.

4. **Data Poisoning**:
    - AIA requirement: Prevention, detection, and mitigation of data poisoning attacks.
    - Potential MU role: Removing the influence of identified poisoned data.
    - Technical gaps: Identifying the complete set of poisoned samples is highly challenging; poisoned data may be visually indistinguishable from clean data; some methods incur significant accuracy trade-offs.

5. **Generative AI Risk**:
    - AIA requirement: Mitigation of harmful generative outputs, including harmful medical advice, CBRN-related knowledge, and discriminatory content.
    - Potential MU role: Unlearning data or concepts in the training set responsible for harmful outputs.
    - Technical gaps: Broad concepts such as non-discrimination are difficult to operationalize as discrete forget sets; even after removing directly harmful data, models may reconstruct dangerous outputs from latent information in the remaining data; dual-use issues further complicate forget set identification.

6. **Copyright**:
    - AIA requirement: GPAI providers must implement copyright compliance policies and respect data mining opt-out requests.
    - Potential MU role: Preventing reproduction of copyrighted training data at the output stage.
    - Technical gaps: Even exact retraining with copyrighted data removed cannot guarantee non-infringing outputs, as models may generalize similar representations from remaining data; approximate unlearning is deemed "insufficient" for copyright scenarios.

### Loss & Training

Formal framework: Given model $M = A(D)$, an unlearning algorithm $U(M; D_f, D_r)$ produces an unlearned model $M_u$. When $\epsilon = \delta = 0$, this constitutes exact unlearning; otherwise, it is approximate unlearning. Three objectives must be balanced: model utility, unlearning quality, and efficiency.

## Key Experimental Results

### Summary of MU SOTA Capabilities (by Scenario)

### Main Results

| Application Scenario | MU SOTA Feasibility | Primary Obstacles | Alternatives |
|---|---|---|---|
| Accuracy | Relatively high | Identifying problematic data points | Full retraining, improved data pipelines |
| Bias | Moderate | Inconsistent evaluation criteria; data absence is not addressable | Pre/in/post-processing debiasing methods |
| Confidentiality Attacks | Moderate | Onion/Streisand effects; poor adaptability to new attacks | Differential privacy, access control |
| Data Poisoning | Low | Identifying the poison set; accuracy trade-offs | Robust training, data cleaning |
| GenAI Risk | Low | Difficulty of concept-level unlearning; dual-use issues | RLHF, guardrail mechanisms |
| Copyright | Low | Generalization of latent representations; lack of formal guarantees | Data governance, output filtering |

### Core Challenge Analysis

### Ablation Study

| Cross-Cutting Challenge | Scope | Current Status |
|---|---|---|
| Forget set identification | All 6 scenarios | Remains unsolved in most scenarios |
| Auditability / Verification | All 6 scenarios | Approximate MU lacks formal guarantees |
| Utility–unlearning trade-off | Accuracy, bias, confidentiality | Accuracy degrades as more data is forgotten |
| Privacy–unlearning conflict | Confidentiality | Unlearning may paradoxically expose information |
| Concept-level unlearning | GenAI risk, copyright | Non-discrete targets in distributed representations |

### Key Findings

- MU holds **potential** value for AIA compliance but is far from a silver bullet — considerable technical gaps exist in the majority of the six scenarios.
- **Auditability** is a central challenge across all scenarios: regulators need to verify unlearning effectiveness, yet current approximate MU methods rely solely on empirical proxy metrics.
- Interdependencies exist among scenarios: unlearning to restore accuracy may affect fairness, while debiasing unlearning may alter the attack surface for privacy attacks.
- In many scenarios, MU is better suited as a "reactive remedy" rather than a "preventive safeguard."

## Highlights & Insights

- This paper provides the **first comprehensive mapping** of MU techniques to specific AIA regulatory provisions, offering the MU research community a clear panorama of regulatory requirements.
- Each application scenario is analyzed with equal attention to both the potential of MU and its available alternatives, ensuring a balanced and objective assessment.
- The paper foregrounds **auditability** — a requirement broadly overlooked in the MU community — as a core concern: without demonstrable proof of effective unlearning to regulators, technical achievements are of limited practical value.
- Two counterintuitive risks in MU are identified: the "Onion Effect" and the "Streisand Effect."

## Limitations & Future Work

- As a position paper, this work does not include new algorithms or experiments.
- The analysis focuses primarily on the AIA; applicability to other regulatory frameworks (e.g., U.S. state-level AI laws, Canada's AIDA) is only briefly discussed.
- The combined use of MU with complementary techniques such as DP and RLHF is not examined in depth.
- Parts of the analysis remain qualitative in nature, lacking quantitative assessments of the identified gaps.

## Related Work & Insights

- Cooper et al. (2024), "Machine unlearning doesn't do what you think," raises critical concerns about MU in generative models; this paper extends that perspective to the regulatory compliance context.
- The WMDP benchmark (Li et al., 2024c), designed to measure the unlearning of CBRN-related knowledge, may serve as a useful verification tool.
- For researchers working in MU, this paper identifies six research directions directly relevant to regulatory requirements, which may enhance the broader societal impact of their work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic alignment of MU with AI regulatory frameworks; highly original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ — No experiments, but literature analysis is comprehensive and rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Regulatory provisions are cited with precision; technical analysis is balanced and objective.
- **Value**: ⭐⭐⭐⭐ — Identifies regulation-driven research directions for the MU community and serves as an important bridge between AI safety research and policy needs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](../../CVPR2025/ai_safety/towards_source-free_machine_unlearning.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)

</div>

<!-- RELATED:END -->
