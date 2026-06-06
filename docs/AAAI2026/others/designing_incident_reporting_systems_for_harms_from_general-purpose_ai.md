---
title: >-
  [Paper Note] Designing Incident Reporting Systems for Harms from General-Purpose AI
description: >-
  [AAAI 2026][AI Incident Reporting] Through a literature review and case studies of nine safety-critical industries (nuclear energy, aviation, healthcare, etc.)…
tags:
  - "AAAI 2026"
  - "AI Incident Reporting"
  - "AI Governance"
  - "Safety-Critical Systems"
  - "Policy Design"
  - "General-Purpose AI"
date: 2026-05-08
content_hash: 923b836d5f96e2ce
---

# Designing Incident Reporting Systems for Harms from General-Purpose AI

**Conference**: AAAI 2026
**arXiv**: [2511.05914](https://arxiv.org/abs/2511.05914)  
**Code**: None  
**Area**: AI Safety & Governance
**Keywords**: AI Incident Reporting, AI Governance, Safety-Critical Systems, Policy Design, General-Purpose AI

## TL;DR
Through a literature review and case studies of nine safety-critical industries (nuclear energy, aviation, healthcare, etc.), this paper proposes a seven-dimensional institutional design framework for AI incident reporting systems, providing systematic guidance for policy design of general-purpose AI incident reporting in the United States.

## Background & Motivation
General-purpose AI (GPAI) systems, particularly large language models (LLMs), are causing a growing number of real-world harms:

- Facilitating a $25.6 million financial fraud
- Assisting in planning explosive attacks
- Generating deepfake pornographic content
- Accidentally deleting an entire company codebase
- Exhibiting extortion and deception capabilities
- Spreading election misinformation in the United States and globally

**Key Challenge**: According to Normal Accident Theory, serious incidents are inevitable over time in complex systems such as GPAI. Pre-deployment audits cannot prevent all incidents — emergent capabilities of LLMs may manifest unexpectedly after deployment, creating unforeseen incident types. Yet research on the institutional design of AI incident reporting mechanisms remains nearly absent.

**Background**:
- As of July 2025, only China and the European Union mandate general-purpose AI incident reporting
- Multiple legislative proposals exist in the United States but have not yet been enacted
- Non-governmental databases (AIID, AIAAIC, AVID) exist but lack stakeholder buy-in
- The literature broadly supports incident reporting, but systematic institutional design analysis is lacking

This paper fills this gap by providing the first systematic examination of institutional design choices for AI incident reporting systems and the conditions under which they apply.

## Method

### Overall Architecture
This paper employs a three-step case study methodology:
1. **Selection of nine safety-critical industries** as cases: nuclear energy, aviation, pesticides, pharmaceuticals, cybersecurity, dams, railways, occupational safety, and healthcare
2. **Development of a seven-dimensional institutional design framework** through literature review
3. **Extraction of design considerations from the case industries**, with discussion of the applicability of specific design choices to the AI context

### Key Designs

**Design 1: Seven-Dimensional Institutional Design Framework**

The seven dimensions of the framework are as follows:

| Dimension | Definition | Options |
|-----------|------------|---------|
| **Policy Goal** | The objective the system seeks to achieve | Safety learning or accountability |
| **Reporting & Receiving Entities** | Participants who submit and receive reports | Users, victims, third parties, companies, industry employees, government |
| **Incident Types** | Categories of events to be reported | Safety incidents, rights incidents, security vulnerability incidents |
| **Risk Materialization Stage** | The risk stage at which events are reported | Hazard → Situation → Near miss → Harm event |
| **Reporting Mandatoriness** | Mechanisms for incentivizing reporting | Mandatory (legally required) or voluntary |
| **Reporter Anonymity** | Who can know the reporter's identity | Public, confidential, anonymous |
| **Post-Report Actions** | Actions taken upon receiving a report | Information sharing, information disclosure, auditing, regulatory action |

An important conceptual clarification offered in this paper is the distinction between **AI issues/flaws** and **AI incidents**: issues are system-level conditions (hazards) that serve as preconditions for incidents when exposed to external environments; incidents are events that have caused or may cause harm.

**Design 2: Incident Lifecycle Model**

The complete pathway from risk emergence to materialization:
- **Hazard**: An intrinsic system condition that may lead to harm
- **Situation**: A hazard exposed to an external environment
- **Near Miss**: An event that could have caused harm but ultimately did not
- **Harm Event**: An event that has actually caused harm

Hospital studies indicate that fewer than 1% of safety events cause serious injury, 18% cause minor injury, and 82% cause no injury. The frequency of near misses is estimated to be 300 times that of harm events. This suggests that AI near-miss reporting could be highly valuable for safety learning.

**Design 3: Design Considerations Derived from Nine Industry Cases**

Key design considerations extracted from the case studies include:

**(1) Dual nature of policy goals**: While safety learning and accountability can be pursued within the same system, dual-objective systems are rare in practice because the two goals tend to call for opposing design choices — for example, encouraging voluntary reporting conflicts with punitive enforcement. Establishing **multiple single-objective systems** may be necessary.

**(2) Regulatory vs. non-regulatory operators**:
- Regulatory agency operation: typically used for mandatory reporting and accountability-oriented systems
- Non-regulatory agency operation: typically non-punitive and learning-oriented. The FAA fully delegated its voluntary reporting system ASRS to NASA — an agency that does not regulate airlines — thereby fostering trust. ASRS has received over 2 million reports since 1975.

**(3) Reporting coverage**: Given the diverse forms of GPAI harms and complex supply chains, information must be collected from multiple parties (users, company employees, third parties, the public). The FAA maintains no fewer than 8 separate voluntary reporting programs for different occupational groups (dispatchers, air traffic controllers, pilots, etc.).

### Loss & Training
*(This paper is a policy research study and does not involve model training. The methodological strategy is described here.)*

The study employs qualitative case analysis, with a methodology adapted from Raji et al. (2022), Ayling and Chapman (2022), and Stein et al. (2024):
1. Identification of nine safety-critical industries through seed articles
2. Literature review of incident reporting institutions in each industry
3. Classification and coding of each industry's reporting system according to the seven-dimensional framework
4. Extraction of common patterns and best practices as design considerations for AI incident reporting

## Key Experimental Results

### Main Results

*(This paper is a policy analysis study; "main results" correspond to the systematic comparative analysis of the case industries.)*

**Summary comparison of incident reporting systems across nine industries:**

| Industry | Mandatory Reporting | Voluntary Reporting | Operating Entity | Coverage |
|----------|--------------------|--------------------|-----------------|----------|
| Nuclear | ✓ | ✓ | Regulatory (NRC) + International (IAEA) | Companies + employees |
| Aviation | ✓ | ✓ | Regulatory (FAA) + Non-regulatory (NASA/ASRS) | Companies + employees + public |
| Pesticides | ✓ | ✓ | Multi-level (federal + state + local) | Farmers + physicians + consumers |
| Pharmaceuticals | ✓ | ✓ | Regulatory (FDA/MedWatch) | Companies (90% mandatory) + consumers |
| Cybersecurity | ✓ | ✓ | 22 federal agencies, 45+ requirements | Companies + government |
| Dams | ✓ | ✓ | Federal + state + industry | Engineers + government |
| Railways | ✓ | ✓ | FRA + NASA (C3RS) | Companies + employees |
| Occupational Safety | ✓ | ✗ | OSHA | Employers |
| Healthcare | ✓ | ✓ | State agencies + federal | Hospitals + physicians + patients |

**Key quantitative findings**:
- 90% of reports in the FDA MedWatch database originate from mandatory corporate reporting
- 91% of incidents are reported in the FAA wildlife strike database, but via 10+ distinct systems
- A hospital study found that <1% of events caused serious injury and 82% caused no injury
- C3RS (railway voluntary reporting) attracted only 23 of 800 railroad companies, reflecting a failure of industry buy-in
- Kesari (2023) found that mandatory cybersecurity incident reporting reduced identity theft complaints by an average of 10.1%

### Ablation Study

*(In this policy analysis paper, this section corresponds to an analysis of design dimension variants.)*

**Comparison of mandatory vs. voluntary reporting effectiveness:**

| Reporting Type | Advantages | Disadvantages | Typical Case |
|---------------|------------|---------------|--------------|
| Mandatory | High coverage, strong regulatory visibility | May suppress reporting of low-severity events | FDA: 90% of reports from mandatory requirements |
| Voluntary | Covers near misses, promotes learning | Difficult industry buy-in, competitive concerns | ASRS: 2M+ reports vs. C3RS: only 23 participants |
| Anonymous | Reduces fear of retaliation, increases reporting rates | Impedes accountability, complicates follow-up investigation | ASRS de-identification + FAA immunity protection |
| Public | Facilitates accountability and follow-up investigation | Suppresses reporting (72% of physicians indicate legal protection would increase reporting) | OSHA mandatory public reporting |

### Key Findings
- Safety learning and accountability goals are difficult to reconcile within the same system and often require separate systems
- The success of ASRS (2M+ reports) depends on multiple factors: non-regulatory operation, de-identification, immunity protection, and early stakeholder engagement. This model failed when replicated in the railway sector (C3RS)
- In cybersecurity, 22 federal agencies and 45+ reporting requirements create severe fragmentation, impeding data aggregation and learning — a pitfall AI governance should avoid
- Ambiguous reporting thresholds lead companies to evade compliance (as seen in automotive manufacturer cases), making precise definitions critical
- The multi-domain deployment and complex supply chains of GPAI mean that incident information is dispersed across multiple parties, necessitating multi-system, multi-level reporting

## Highlights & Insights
- The **seven-dimensional framework** provides a unified vocabulary for analyzing and comparing incident reporting systems, filling a gap in the AI governance literature
- The **incident lifecycle model** clearly distinguishes hazards, situations, near misses, and harm events, laying a foundation for standardizing AI incident definitions
- Design principles derived from the experience of nine mature industries help the AI field avoid reinventing the wheel
- The critical analysis of the applicability of the ASRS model to the AI industry offers practical guidance

## Limitations & Future Work
- The analysis is U.S.-centric, and some lessons may not transfer readily to other jurisdictions
- No cost-benefit analysis is conducted comparing incident reporting systems against alternative governance mechanisms
- AI security vulnerability incidents (distinct from safety/rights incidents) are not examined in depth, and information sharing in this category carries additional risks
- The framework focuses on institutional design and does not address technical implementation (incident monitoring/detection methods), safety culture, user interaction, or other sociotechnical factors
- Empirical data from practitioner interviews, user experiments, and industry surveys are absent

## Related Work & Insights
- **vs. Raji et al. (2022)**: Raji et al. propose a citizen–regulator reporting system to promote audit accountability; this paper provides a more comprehensive analysis of institutional design dimensions, arguing that accountability and learning may require separate systems
- **vs. McGregor (2021) (AIID)**: AIID represents an important first step but lacks the stakeholder buy-in and information needed for safety learning — eight of the nine industries studied implement additional reporting mechanisms beyond standalone databases
- **vs. Shrishak (2023)**: Shrishak proposes a voluntary AI reporting system modeled on the FAA/ASRS, but this paper draws on the failure of C3RS to argue that the ASRS model's applicability to the AI industry is questionable, given competitive dynamics, the absence of unions, and the complexity of multi-domain deployment

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic institutional design analysis framework for AI incident reporting
- Experimental Thoroughness: ⭐⭐⭐ Qualitative case analysis is detailed, but lacks quantitative empirical data and practitioner evidence
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous argumentation, and comprehensive literature review
- Value: ⭐⭐⭐⭐ Directly useful to AI safety governance policymakers; highly timely

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/others/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [\[ICML 2026\] Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](../../ICML2026/others/mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)
- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](../../ICML2026/others/beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)
- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)

</div>

<!-- RELATED:END -->
