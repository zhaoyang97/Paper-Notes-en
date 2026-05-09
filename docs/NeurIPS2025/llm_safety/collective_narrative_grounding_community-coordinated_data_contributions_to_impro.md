---
title: >-
  [Paper Note] Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems
description: >-
  [NeurIPS 2025][LLM Safety][local knowledge] This paper proposes the Collective Narrative Grounding protocol, which collects community narratives through participatory workshops and structures them into "narrative units." A RAG pipeline then injects this local knowledge into LLM-based QA systems. Experiments on LocalBench reveal that 76.7% of errors can be directly remediated by local narratives, and GPT-5 achieves only 21% accuracy on the participatory QA set, highlighting the severity of the local knowledge gap.
tags:
  - NeurIPS 2025
  - LLM Safety
  - local knowledge
  - LLM grounding
  - participatory design
  - narrative unit
  - community governance
  - RAG
date: 2026-05-08
content_hash: 5aa24e3948b7b237
---

# Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems

**Conference**: NeurIPS 2025
**arXiv**: [2601.04201](https://arxiv.org/abs/2601.04201)
**Code**: None
**Area**: AI Safety
**Keywords**: local knowledge, LLM grounding, participatory design, narrative unit, community governance, RAG

## TL;DR

This paper proposes the Collective Narrative Grounding protocol, which collects community narratives through participatory workshops and structures them into "narrative units." A RAG pipeline then injects this local knowledge into LLM-based QA systems. Experiments on LocalBench reveal that 76.7% of errors can be directly remediated by local narratives, and GPT-5 achieves only 21% accuracy on the participatory QA set, highlighting the severity of the local knowledge gap.

## Background & Motivation

**State of the Field**: LLM-based QA systems perform well on general knowledge but frequently fail on community-specific local knowledge — including local historical events, cultural traditions, and place-specific information that constitute "knowledge blind spots."

**Limitations of Prior Work**: (a) LLM training data exhibits systematic geographic bias, with severe underrepresentation of low-income and rural communities; (b) data voids can be exploited by misinformation; (c) existing RAG and fine-tuning approaches rely on pre-existing (often scarce) local text, and cannot capture orally transmitted community knowledge.

**Root Cause**: The source of LLM knowledge — internet text — is itself skewed, making the knowledge gap not a technical limitation but a structural form of epistemic injustice, in which the knowledge of marginalized communities is systematically excluded.

**Paper Goals**: (a) Quantify the failure modes of LLMs on local knowledge; (b) design a complete protocol for collecting, structuring, and governing narrative data from communities; (c) validate whether community narratives can directly address the major error categories.

**Starting Point**: Community members' oral narratives are treated as first-class data sources. Stories are collected through participatory mapping workshops and then structured into queryable narrative units.

**Core Idea**: A community participatory protocol collects local narratives, structures them into a knowledge layer, and uses RAG to fill LLM blind spots regarding local knowledge.

## Method

### Overall Architecture

The system consists of three closed loops: **Elicitation** → **Structuring** → **Governance & Application**.

- **Input**: Oral narratives from community members, map annotations, and stories/photos/audio submitted via mobile devices.
- **Processing**: An NLP pipeline combined with human review segments narratives into narrative units, populates a structured schema, and extracts entities, timestamps, and locations.
- **Output**: (1) A RAG-powered local QA system with provenance citations; (2) a community governance dashboard supporting browsing, flagging, retraction, and auditing.

### Key Designs

1. **Participatory Mapping Workshop**:

    - *Function*: Elicit place-specific, experiential narratives from community members.
    - *Mechanism*: Follows four principles — (a) explicit expert positioning (participants are explicitly framed as "community experts"); (b) physical scaffolding (large-scale satellite maps projected on tables for direct annotation); (c) asset-based questioning ("Where is your favorite place?" rather than "What are the problems?"); (d) ethical engagement (informed consent, de-identification, and the right to withdraw at any time).
    - *Design Motivation*: Traditional participatory workshops suffer from power imbalances between technocrats and communities; these principles invert the conventional facilitator–participant power dynamic.

2. **Narrative Unit Schema**:

    - *Function*: Transform unstructured oral narratives into computationally processable structured objects.
    - *Mechanism*: Each narrative unit contains 10 fields: `narrative_id`, `author_pseudonym`, `timestamp`, `geocode` (GeoJSON), `narrative_text`, `embedded_claims[]` (array of factual claims), `media_links[]`, `verification_status` (unverified / community_verified / disputed / retracted), `community_flags[]`, and `relationships[]` (inter-unit relations: corroborates / disputes / extends / near-in-space / near-in-time).
    - *Design Motivation*: The schema must preserve narrative richness while enabling entity/temporal/spatial extraction, verification, and provenance control.

3. **Failure Taxonomy**:

    - *Function*: Systematically quantify error types of LLMs on local knowledge.
    - *Mechanism*: 1,000 model failures are audited on LocalBench (14,782 county-level QA pairs across 526 U.S. counties) and labeled into 8 mutually exclusive categories. Two trained annotators label independently (raw agreement 87%, Cohen's $\kappa = 0.852$).
    - *Design Motivation*: Understanding where LLMs fail is a prerequisite for targeted remediation via community narratives.

### Loss & Training

This paper does not involve model training; the core contribution is protocol and system design. RAG integration uses a vector index over narrative text and embedded claims, combined with graph context to provide provenance-traceable evidence.

## Key Experimental Results

### Main Results — Error Category Distribution

| Error Category | Proportion | Remediable by Narratives? |
|---|---|:---:|
| Factual knowledge absence | 31.8% | ✓ |
| Cultural misunderstanding | 23.4% | ✓ |
| Geographic confusion | 12.4% | ✓ |
| Temporal displacement | 9.1% | ✓ |
| **Top four subtotal** | **76.7%** | ✓ |
| Other (reasoning/calibration, etc.) | 23.3% | Partial |

### Participatory QA Benchmark (GPT-5)

| Outcome | Count (of 20) | Proportion |
|---|:---:|:---:|
| Fully correct | 4 | 20% |
| Partially correct / ambiguous | 12 | 60% |
| Incorrect / hallucinated | 3 | 15% |
| Refused to answer | 1 | 5% |

Annotator agreement: 84.2%, Cohen's $\kappa = 0.812$.

### Key Findings
- 76.7% of LLM local knowledge errors concentrate in four categories that are directly remediable by community narratives.
- In most partially correct responses, the missing facts already exist in workshop-collected narratives, confirming a clear remediation pathway via narrative grounding.
- Typical errors include misidentification of local officials, confusion between adjacent locations, and provision of outdated event details.
- Knowledge voids simultaneously create attack surfaces for misinformation (67.3% of errors could be exploited by fabricated narratives).

## Highlights & Insights
- **Participatory Design × AI Systems**: Rather than patching LLM knowledge deficits through technical means, this work addresses the root of knowledge production — enabling community members to become knowledge contributors and governors of AI systems.
- **Elegant Narrative Unit Schema**: The schema strikes a balance between structure and narrative richness; the `relationships[]` field supports corroboration, dispute, and extension relations among units, forming a knowledge graph.
- **Transferable Failure Taxonomy**: The finding that 76.7% of errors concentrate in four remediable categories is generalizable and can guide prioritization in other LLM grounding efforts.

## Limitations & Future Work
- Workshop scale is small (N=24, 3 workshops, single city: Atlanta), limiting representativeness.
- No end-to-end RAG system is constructed and quantitatively evaluated for accuracy gains after narrative injection; the current work constitutes a proof-of-concept only.
- A fundamental tension exists between privacy and utility: more specific narratives are more useful but also more susceptible to re-identification of contributors.
- The practical complexity of community governance is high; the proposed DAO-style voting and reputation-weighted mechanisms remain unvalidated.
- Narrative quality control is challenging: ensuring factual accuracy of subjective oral accounts is non-trivial.
- Scalability is unclear: the resources required to scale from a 24-person workshop to city- or national-level coverage are unspecified.

## Related Work & Insights
- **vs. WorldBench (Moayeri et al., 2024)**: WorldBench quantifies geographic knowledge bias in LLMs but provides no remediation; this paper offers a complete protocol for proactively supplementing knowledge from the community side.
- **vs. Localness-Aware LLM (Gao et al., 2025)**: The latter constructs knowledge graphs from social media videos; this paper emphasizes participatory governance and narrative rather than passive social media content.
- **vs. Standard RAG approaches**: Standard RAG relies on existing documents; the innovation here lies in generating new community knowledge sources and granting communities governance rights over them.
- The work carries important implications for localized AI systems, community AI governance, and data justice.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of participatory workshops, narrative schema, and LLM grounding presents a genuinely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ The failure taxonomy and benchmark design are sound, but the scale is small and no end-to-end system validation is provided.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated and design tensions are discussed in depth.
- Value: ⭐⭐⭐⭐ Makes an important methodological contribution to fair AI, community governance, and local knowledge augmentation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)
- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[NeurIPS 2025\] When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)
- [\[NeurIPS 2025\] DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)
- [\[NeurIPS 2025\] A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)

<!-- RELATED:END -->
