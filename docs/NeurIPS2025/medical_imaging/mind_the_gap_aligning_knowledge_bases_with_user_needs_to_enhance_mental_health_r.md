---
title: >-
  [Paper Note] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval
description: >-
  [NeurIPS 2025][Medical Imaging][RAG] This paper proposes a knowledge base augmentation framework grounded in "demand gap" analysis. By overlaying real user data (forum posts) onto existing mental health resource reposito…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "RAG"
  - "knowledge base augmentation"
  - "mental health"
  - "gap analysis"
  - "corpus expansion"
date: 2026-05-08
content_hash: 442a1cc4755930c5
---

# Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2509.13626](https://arxiv.org/abs/2509.13626)
**Code**: None
**Area**: Medical Imaging
**Keywords**: RAG, knowledge base augmentation, mental health, gap analysis, corpus expansion

## TL;DR

This paper proposes a knowledge base augmentation framework grounded in "demand gap" analysis. By overlaying real user data (forum posts) onto existing mental health resource repositories to identify content voids, the framework applies targeted augmentation strategies to achieve near-full-corpus RAG retrieval quality with minimal document additions.

## Background & Motivation

Retrieval-augmented generation (RAG) systems in the mental health domain rely on professionally curated knowledge bases, yet face the following core challenges:

**High cost of knowledge base expansion**: Mental health resources are authored and reviewed by licensed clinicians, resulting in long production cycles and high costs.

**Misalignment between user needs and content supply**: Users expressing concerns on forums and similar platforms tend to employ informal, colloquial language, whereas knowledge base content primarily consists of formal psychoeducational materials — creating a dual gap in both topic coverage and linguistic register.

**Low efficiency of indiscriminate expansion**: Randomly adding documents without targeting specific gaps wastes resources and yields limited improvements in retrieval quality.

The core insight is that **content gaps can be identified by overlaying naturalistic user demand data** — when certain topics are frequently raised by users but are poorly covered in the knowledge base, actionable gaps emerge.

## Method

### Overall Architecture

The framework consists of four stages: (1) data collection and preprocessing; (2) content gap analysis; (3) synthetic document generation and corpus augmentation; and (4) automated evaluation across multiple RAG pipelines.

### Key Designs

1. **Topic taxonomy**: Built upon the clinical CLICC framework (Clinician Index of Client Concerns) with 46 topics, further expanded into 368 sub-topics. GPT-4o is used to annotate both user queries and knowledge base documents with topic and sub-topic labels.

2. **Coverage Gap metric**: Borrowing from TF-IDF, the paper maps "term–document" relationships onto "sub-topic–query" relationships. User mention frequency (demand) serves as TF, and knowledge base document coverage (supply) serves as IDF:

$$\text{Gap}(t) = \frac{\log(1 + f_p(t))}{\max_w \log(1 + f_p(w))} \cdot \left[\log\left(\frac{D + c}{df(t) + c}\right)\right]^\alpha$$

where $\alpha = 1.5$ amplifies the influence of weakly covered topics.

3. **Usefulness Gap metric**: An LLM-as-a-Judge approach (GPT-4o-mini) evaluates query–document pairs on contextual relevance (1–50) and practical helpfulness (1–50). Scores are averaged per sub-topic and inverted so that high scores correspond to large gaps.

4. **Hybrid metric**: A 50/50 blend of Coverage Gap and Usefulness Gap. Sensitivity analysis shows that varying the weights changes corpus composition by no more than 24.2%.

5. **Directed vs. Non-Directed augmentation**: Document quotas are allocated to sub-topics proportionally according to the hybrid gap score (Directed) and compared against randomly sampled document sets of equivalent size (Non-Directed).

### Data & Evaluation Pipelines

- **Knowledge base**: 387 curated documents from the Singapore mental health platform mindline.sg
- **User data**: 1,223 anonymized posts from the *let's talk* forum (80% train / 20% test)
- **Reference corpus**: 7,640 synthetic documents generated from 7 global mental health platforms (GPT-4o-mini; avg. 16.1 s/doc, \$0.0007/doc)
- **Four RAG pipelines**: Baseline, Hierarchical, Reranking, and Query Transformation

## Key Experimental Results

### Main Results: Minimum document additions required to reach ~95% of reference corpus performance

| RAG Pipeline | Directed Increase | Non-Directed Increase | Directed Doc Count | Non-Directed Doc Count | Doc Savings |
|---|---|---|---|---|---|
| Baseline | +318% | +763% | 1,230 | 2,954 | 58.4% |
| Hierarchical | +74% | +403% | 288 | 1,560 | 81.5% |
| Reranking | +74% | +318% | 288 | 1,230 | 76.5% |
| Query Transformation | +42% | +232% | 162 | 898 | 81.9% |

### Gap Analysis Results

| Rank | Topic | Sub-topic |
|---|---|---|
| 1 | Depression | Self-criticism and low self-worth |
| 2 | Relationship | Erosion of trust and boundary issues |
| 3 | Anxiety | Health anxiety and illness phobia |
| 4 | Relationship | Attachment insecurity and emotional detachment |
| 5 | Emotional dysregulation | Rapid mood swings and reactivity |
| 6 | Family | Parental conflict and family tension |
| 7 | Depression | Social isolation and disconnection |
| 8 | Depression | Anhedonia and withdrawal |
| 9 | Anxiety | Fear of social evaluation and avoidance |
| 10 | Anxiety | Sleep disturbances |

### Key Findings

- **The Query Transformation pipeline performs best**: Adding only 42% more documents (162 articles) suffices to reach 95% of reference performance, as query rewriting bridges the gap between users' colloquial language and the formal register of the knowledge base.
- **Directed augmentation consistently outperforms random augmentation**: Across all pipelines, the Directed strategy achieves superior quality with fewer documents.
- Aggregated at the topic level, the five most underserved categories are: addiction (non-substance/alcohol), depression, anxiety, family, and racial/ethnic/cultural concerns.

## Highlights & Insights

- The **transfer of TF-IDF logic to demand–supply analysis** is a particularly elegant idea, applying a classical information retrieval methodology to content gap detection.
- The **two-dimensional gap metric** (coverage + usefulness) is more comprehensive than a single dimension, since a topic may have documents that are too generic or overly clinical to be truly helpful.
- The experimental design is rigorous, with 10 corpus size configurations per condition (Directed and Non-Directed) across four RAG pipelines, yielding 88 total experimental conditions.
- In this high-stakes domain (mental health), the authors appropriately note that synthetic documents serve only as a proof of concept; production content must be authored by qualified experts.

## Limitations & Future Work

- **Population bias**: Forum users skew younger (academic stress, identity concerns) and are not representative of the full age range targeted by the knowledge base.
- **Absence of human evaluation**: All assessments rely on LLM-as-a-Judge without clinical expert validation, potentially overlooking professional dimensions such as therapeutic framing or emotional appropriateness.
- **Synthetic document quality**: Documents have not undergone medical review and serve only as feasibility demonstrations.
- The choice of a 50/50 hybrid weighting, while shown to be relatively insensitive, may require adjustment in other domains.
- The paper does not explore dynamic feedback loops — as the knowledge base is updated, user needs may shift accordingly.

## Related Work & Insights

- The framework could be integrated with active learning for RAG systems to enable a closed loop of "gap detection → generation → expert review → deployment."
- The framework is domain-agnostic and applicable to knowledge base construction in other high-stakes verticals such as law and education.
- For knowledge base operators, this work provides a data-driven methodology for content prioritization.

## Rating

- **Novelty**: ⭐⭐⭐ — The TF-IDF transfer is creative, but the overall methodology is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 pipelines × 22 corpus configurations = 88 experimental conditions; highly systematic.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ — Directly actionable guidance for knowledge base operators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)
- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](../../ICLR2026/medical_imaging/biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[NeurIPS 2025\] MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](../../AAAI2026/medical_imaging/voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](../../ACL2026/medical_imaging/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)

</div>

<!-- RELATED:END -->
