---
title: >-
  [Paper Note] Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems
description: >-
  [ACL2025][LLM (Other)][Representational harms] Through semi-structured interviews with 12 practitioners responsible for evaluating representational harms in LLM-based systems, this work reveals that publicly available measurement tools generally fail to meet practitioner needs—being either "not useful" due to insufficient validity/specificity or "not used" due to organizational/institutional barriers. Based on measurement theory and pragmatic measurement frameworks…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Representational harms"
  - "LLM evaluation"
  - "measurement tools"
  - "semi-structured interviews"
  - "practitioner needs"
  - "measurement theory"
date: 2026-05-08
content_hash: 5e32a77b4fd31da4
---

# Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems

**Conference**: ACL2025  
**arXiv**: [2506.04482](https://arxiv.org/abs/2506.04482)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Representational harms, LLM evaluation, measurement tools, semi-structured interviews, practitioner needs, measurement theory

## TL;DR

Through semi-structured interviews with 12 practitioners responsible for evaluating representational harms in LLM-based systems, this work reveals that publicly available measurement tools generally fail to meet practitioner needs—being either "not useful" due to insufficient validity/specificity or "not used" due to organizational/institutional barriers. Based on measurement theory and pragmatic measurement frameworks, systematic recommendations for improvement are proposed.

## Background & Motivation

1. **Severity of Representational Harms**: LLM-based systems can depict certain social groups in detrimental ways (stereotyping, demeaning, or erasing). Measuring and mitigating these harms is critical for deployment.
2. **Proliferation of Measurement Tools vs. Low Adoption**: Although the NLP community has released numerous datasets, metrics, tools, and benchmarks (e.g., StereoSet, BBQ, ToxiGen), very few are actually adopted by practitioners in practice.
3. **Mismatch Between Research Assumptions and Practical Needs**: Existing HCI research shows that Responsible AI tools developed by researchers often mismatch the actual needs of practitioners, though no prior work has specifically focused on the domain of LLM representational harms.
4. **Abstract and Contested Nature of Representational Harms**: Their meanings vary across different use cases, languages, and cultures, making them difficult to define and measure precisely, which renders general-purpose tools unsuitable.
5. **Lack of Practitioner Perspective**: Prior critiques regarding the deficiencies of measurement tools mostly originate from researchers' technical evaluations, lacking a systematic investigation into the practical challenges and institutional constraints faced by practitioners themselves.
6. **Underutilization of Measurement Theory in NLP**: Mature measurement theories from the social sciences (validity and reliability frameworks) and pragmatic measurement concepts have not been fully adopted by NLP tool developers.

## Method

### Overall Architecture

Qualitative research paradigm: defining 7 desiderata of measurement tools $\to$ designing a semi-structured interview protocol $\to$ recruiting practitioners $\to$ interviewing until saturation $\to$ conducting thematic analysis coding $\to$ proposing recommendations based on measurement theory. The core contribution is not an algorithm but a systematic diagnosis of the practice-research gap.

### Key Designs

#### 1. 7 Desiderata of Measurement Tools

- **Function**: Defines the 7 properties that measurement tools should satisfy—validity, reliability, specificity, extensibility, scalability, interpretability, and actionability.
- **Design Motivation**: To provide a structured scaffolding for the interviews, ensuring systematic coverage of various challenges practitioners might encounter, while also defining analytical dimensions for the subsequent recommendations.
- **Mechanism**: Derived from the authors' own experience in measuring representational harms combined with a systematic review of tool evaluation papers in the NLP literature.

#### 2. Semi-Structured Interview Design and Execution

- **Function**: Conducting 1-hour interviews with 12 practitioners (covering roles of research engineers, applied scientists, data engineers, etc., from big tech, AI startups, and non-profits).
- **Design Motivation**: Semi-structured interviews allow gathering structured information around preset dimensions while leaving space for practitioners to raise novel, spontaneous challenges. The sample size follows the qualitative saturation principle (recruitment was stopped when consecutive interviews yielded no new insights).
- **Mechanism**: Participants first described their roles and evaluation workflows, followed by guided probing using the 7 desiderata regarding their reasons for using or abandoning measurement tools. Recruitment was conducted via professional networks, social media, cold emailing, and snowball sampling. Each participant was compensated $75, with IRB approval from Microsoft Research.

#### 3. Thematic Analysis

- **Function**: Doing dual inductive-deductive coding on interview transcriptions to identify themes of challenges faced by practitioners.
- **Design Motivation**: Inductive coding captures emergent findings outside the preset framework (e.g., data contamination), while deductive coding ensures systematic coverage of the 7 desiderata.
- **Mechanism**: The first author initial-coded the transcripts, expanded the codebook upon discovering new categories, and re-coded the text. All authors then discussed to synthesize the final themes. At least one other author performed independent coding verification on each transcript, resolving discrepancies through discussion.

#### 4. Recommendations Based on Measurement Theory

- **Function**: Maps practitioner challenges to measurement theory (systematization $\rightarrow$ operationalization $\rightarrow$ application $\rightarrow$ interrogation) and pragmatic measurement frameworks to propose targeted improvement recommendations.
- **Design Motivation**: NLP tool developers frequently skip "systematically defining concepts" and jump directly to "operationalization" (building tools), resulting in unassessable validity.
- **Mechanism**: Researchers are advised to: (a) avoid bypassing the systematization phase and explicitly document the definition of the construct being measured; (b) provide validity and reliability evidence along with self-assessment tools; (c) release reference distributions of measurement results to improve interpretability; and (d) design modular, extensible tools that allow customization by practitioners.

## Key Experimental Results

### Table 1: Participant Profile (12 Participants)

| Role Type | Organization Type | Count |
|----------|-------------|------|
| Research Engineer | Big Tech | 2 |
| Applied/Research Scientist | Big Tech | 3 |
| Consultant/NLP Specialist | AI Startups/Non-Tech Enterprises | 3 |
| Researcher | AI Startups/Non-profit | 3 |
| Data Engineer | Large Non-Tech Company | 1 |

Participants are distributed across North America and Europe, representing diverse organizational types such as big tech, AI startups, and non-profits.

### Table 2: Summary of Practitioner Challenges in Each Desideratum

| Desideratum | Key Finding | Number of Mentions |
|----------|----------|----------|
| **Validity** | Vague concept definition, incorrect data labeling, and trust breakdown driven by data contamination. | Almost all |
| **Specificity** | General-purpose tools fail to align with specific systems, scenarios, or cultural contexts, forcing in-house development. | All 12 |
| **Interpretability** | Tools produce a "single score" but fail to explain its meaning or thresholds. | 6 |
| **Actionability** | Deprioritization of measurement in the absence of clear mitigation strategies. | 6 |
| **Reliability** | Theoretically important but not prioritized in practice. | 0 (none spontaneously mentioned) |
| **Scalability** | Only a barrier in online evaluation scenarios. | Few |
| **Institutional Barriers** | Security and compliance requirements, data licensing issues, and unsupportive organizational culture. | Multiple |

Key findings: Validity and specificity are "showstoppers"—tools are abandoned immediately if they fail to meet these; while reliability is theoretically crucial, practitioners get weeded out at the validity check and never progress to evaluating reliability.

### Other Key Findings

- **Data Contamination as a Unique Pain Point**: 50% of the participants expressed unease about using any public benchmarks due to the opacity of LLMs' training datasets.
- **Lack of Cultural Context**: Representational harms heavily depend on cultural context, and general-purpose stereotyping datasets often fail to adapt to specific system scenarios.
- **Low-Resource Languages**: 1/3 of the participants mentioned that evaluating non-English languages suffers from a near-total void of available tools.
- **In-house Tools as the New Normal**: Multiple participants were forced to construct customized measurement methods from proprietary data due to the inapplicability of public tools.

## Highlights & Insights

- **Filling a Key Gap**: The first work to systematically investigate the gap between LLM representational harm measurement tools and practitioner needs.
- **Double Challenge Model**: An elegant and powerful classification framework distinguishing between 'not useful' (inherent tool deficiencies) and 'not used' (institutional/practical barriers).
- **Interdisciplinary Perspective**: Incorporates measurement theory from the social sciences and the concept of pragmatic measurement into NLP tool design, offering concrete and actionable suggestions.
- **Methodological Rigorousness**: Strictly adheres to qualitative research standards (saturation principle, IRB approval, dual coding, positionality statements), which is rare and highly commendable for NLP conferences like ACL.

## Limitations & Future Work

- **Selection Bias**: Recruitment was challenging (73 cold emails yielded only 1 interview), possibly skewing the sample toward practitioners who have already faced these challenges, preventing broad statistical generalization.
- **English-centric Interviews**: While some participants mentioned low-resource languages, the study remains heavily centered around English-based evaluations.
- **Lack of Quantitative Validation**: Qualitative findings (e.g., 'validity being the primary concern') were not validated via large-scale quantitative surveys.
- **Limited Harm Types**: The focus is strictly on representational harms, leaving the applicability of tools for allocative harms unexamined.

## Related Work & Insights

### vs. Blodgett et al. (2020)'s review of bias critiques

Blodgett et al. critiqued NLP bias papers from a **researcher perspective**, pointing out poorly motivated concepts and mismatches between measurements and conceptual claims. This work complementarily approaches from a **practitioner perspective**. The identified issues of validity and specificity align closely with these academic critiques, while additionally highlighting institutional barriers (e.g., safety compliance, organizational culture) that surface exclusively in practice.

### vs. Holstein et al. (2019)'s survey on fairness tool needs

Holstein et al. investigated practitioner requirements for ML fairness tools, focusing on **allocative harms** and **traditional predictive models**. This paper extends the scope to **representational harms** and **LLM systems**, revealing that many of the observed challenges (e.g., data contamination, cultural specificity) are novel issues native to the LLM era.

### vs. Delobelle et al. (2024)'s actionability analysis

Delobelle et al. analyzed the lack of actionability in measurement tools from a technical standpoint. In contrast, this study complements this by exploring actionability barriers on a **practical level** through practitioner interviews (e.g., deprioritizing measurement when mitigation strategies cannot be anticipated), collectively providing a more comprehensive view of actionability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic investigation into practitioner needs for LLM representational harm measurement tools, introducing an innovative interdisciplinary perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Interviewing 12 participants reached saturation, utilizing standard thematic analysis, though lacking quantitative validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-structured, richly cited, and complete with positionality and ethical declarations; a model for qualitative research writing.
- **Value**: ⭐⭐⭐⭐ — Directly guides the NLP community to design more pragmatic evaluation tools, offering concrete and actionable recommendations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)
- [\[ACL 2025\] Structural Reasoning Improves Molecular Understanding of LLM](structural_reasoning_improves_molecular_understanding_of_llm.md)
- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
