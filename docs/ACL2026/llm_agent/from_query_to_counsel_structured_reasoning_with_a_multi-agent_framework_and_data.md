---
title: >-
  [Paper Note] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation
description: >-
  [ACL 2026][LLM Agent][Legal consultation QA] This paper introduces JurisCQAD—a large-scale dataset of 43,000+ real Chinese legal consultations—and proposes the JurisMA multi-agent framework, which performs structured task decomposition via a legal element graph and dynamic multi-agent collaboration (Manager Agent + Format Check + Law Search), achieving significant improvements over both general-purpose and law-specialized LLMs on LawBench.
tags:
  - ACL 2026
  - LLM Agent
  - Legal consultation QA
  - multi-agent
  - legal element graph
  - task decomposition
  - Chinese law
date: 2026-05-08
content_hash: 8e1fe5d1e6abb7ad
---

# From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation

**Conference**: ACL 2026
**arXiv**: [2604.10470](https://arxiv.org/abs/2604.10470)
**Code**: None
**Area**: LLM Agent / Legal NLP
**Keywords**: Legal consultation QA, multi-agent, legal element graph, task decomposition, Chinese law

## TL;DR
This paper introduces JurisCQAD—a large-scale dataset of 43,000+ real Chinese legal consultations—and proposes the JurisMA multi-agent framework, which performs structured task decomposition via a legal element graph and dynamic multi-agent collaboration (Manager Agent + Format Check + Law Search), achieving significant improvements over both general-purpose and law-specialized LLMs on LawBench.

## Background & Motivation

**Background**: Legal consultation question answering (Legal CQA) is a core task in legal AI, requiring the generation of well-grounded and actionable legal advice from personalized legal predicaments. Existing approaches primarily rely on continued pretraining on legal corpora or retrieval-augmented generation with statutory provisions.

**Limitations of Prior Work**: (1) Lack of high-quality training data—existing legal LLMs (e.g., LawGPT) are mainly trained on synthetic data, creating a domain shift from real consultation scenarios; (2) Legal CQA involves complex task combinations—requiring identification of legal relationships, causal reasoning, core issue localization, and statutory matching, which end-to-end models struggle to fully cover; (3) High context dependency—precise interpretation of legal entities, relationships, and user intent is required.

**Key Challenge**: Real legal consultations are typically ambiguous and multifaceted, requiring dynamic interpretation of facts, parties, and legal implications. Existing methods either rely on coarse continued pretraining (with low-quality supervision signals) or sentence-level statutory retrieval (which easily conflates legally distinct but linguistically similar concepts).

**Goal**: Construct a large-scale real-world legal consultation dataset and design an interpretable task decomposition and multi-agent collaboration framework.

**Key Insight**: Decompose legal consultations into a structured legal element graph—extracting entities, events, relationships, user intent, and legal issues—then iteratively refine legal opinions through multi-agent collaboration.

**Core Idea**: The element graph provides a semantic foundation → the Manager Agent dynamically coordinates sub-tasks → the Format Check Agent and Law Search Agent iteratively refine outputs → the Content Check Agent performs final polishing.

## Method

### Overall Architecture
JurisMA operates in three stages: (1) **Legal Semantic Graph Construction**—an Element Agent parses the query into a graph structure containing entities, events, relationships, intent, and legal issues; (2) **Multi-Agent Iterative Refinement**—a Manager Agent dynamically evaluates draft quality and invokes the FormatCheck Agent and LawSearch Agent as needed for format correction and statutory grounding; (3) **Content Revision**—a Content Check Agent performs final linguistic quality and professionalism edits.

### Key Designs

1. **Legal Element Graph Construction**:

    - **Function**: Transforms free-text queries into structured legal semantic representations, providing global context for downstream reasoning.
    - **Mechanism**: Defines a graph $G = (V, E)$, where nodes $V$ include legal entities (individuals/organizations with role/status/temporal attributes), legal events, user claims, key facts, and inferred legal issues. Edges $E$ represent semantic relationships (e.g., kinship, contractual obligations). The graph is serialized to JSON and concatenated with the query as semantic input for generation.
    - **Design Motivation**: Legal reasoning centers on identifying key facts, involved parties, and legal relationships. Graph structures capture this structured information more effectively than flat text. Inspired by Hart's primary/secondary rules theory and Kelsen's hierarchical norm model.

2. **Manager Agent Dynamic Coordination**:

    - **Function**: Acts as a central controller, dynamically evaluating draft quality and selectively activating sub-agents.
    - **Mechanism**: At each iteration, the Manager Agent inspects the draft for (1) linguistic adequacy—clarity and conciseness, and (2) legal completeness—presence of authoritative statutory citations. If structural or expressive issues are detected, it invokes the Format Check Agent to generate revision suggestions, which the Draft Agent integrates; if legal citations are insufficient, it invokes the Law Search Agent to retrieve relevant provisions from a statutory database. The process runs for at most 5 iterations or until the Manager Agent returns "Pass."
    - **Design Motivation**: Not all drafts require every type of revision. Dynamic routing ensures only necessary sub-agents are activated, maintaining efficiency and controllability. This mirrors the multi-role collaborative revision workflow in real law firms.

3. **JurisCQAD Dataset**:

    - **Function**: Provides large-scale, high-quality real legal consultation data for training and evaluation.
    - **Mechanism**: 43,000+ instances, each organized as a (question, positive response, negative response) triplet. Sourced from real user legal consultations and verified by domain experts, covering high-frequency legal domains. Positive responses are reviewed by professional lawyers; negative responses are model-generated but annotated as inadequate or incorrect.
    - **Design Motivation**: Existing Chinese legal datasets are mostly synthetic QA pairs or statutory retrieval tasks and fail to reflect the complexity and linguistic diversity of real consultations. The triplet format supports contrastive learning and preference optimization.

### Loss & Training
SFT training is conducted on JurisCQAD with element-graph-augmented inputs. Evaluation uses a revised LawBench incorporating multiple lexical and semantic metrics.

## Key Experimental Results

### Main Results

| Model Category | Representative Models | LawBench Performance |
|---|---|---|
| General LLMs | GPT-4, Qwen | Moderate |
| Legal LLMs | LawGPT, ChatLaw | Moderate–Low |
| JurisMA | Trained on JurisCQAD | **Significantly Best** |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Full JurisMA | Best | Complete framework |
| w/o Element Graph | Degraded | Loss of structured semantic foundation |
| w/o Multi-Agent Iteration | Degraded | Incomplete legal citations |
| w/o LawSearch Agent | Significantly Degraded | Loss of statutory grounding |

### Key Findings
- Models trained on JurisCQAD significantly outperform both general-purpose and law-specialized LLMs, validating the value of high-quality data.
- The structured context provided by the element graph is more effective than raw text input.
- Multi-agent iteration converges on average within 2–3 rounds, indicating that the Manager Agent's quality judgments are effective.
- Notably, law-specialized LLMs sometimes underperform general-purpose LLMs, likely due to inconsistent pretraining data quality.

## Highlights & Insights
- **Legal Element Graph as Semantic Representation**: Transforming free-text queries into entity–relation–event graph structures provides an interpretable semantic foundation for legal reasoning. This idea is generalizable to other domains requiring structured understanding (e.g., medicine, finance).
- **Dynamic Agent Routing**: The Manager Agent activates sub-agents on demand rather than following a fixed pipeline, balancing efficiency and output quality.
- **The Power of Real Data**: The 43,000-instance dataset of real legal consultations constitutes a significant infrastructure contribution to the field.

## Limitations & Future Work
- Coverage is limited to the Chinese legal system; cross-jurisdictional applicability has not been validated.
- Element graph quality depends on the comprehension capability of the Element Agent.
- The statutory retrieval scope of the Law Search Agent may be incomplete.
- The multi-agent system incurs high inference costs due to multi-round iteration and multi-agent invocations.

## Related Work & Insights
- **vs. LawGPT**: Performs continued pretraining on legal corpora, but with coarse data processing. JurisMA employs structured task decomposition and high-quality data instead.
- **vs. LawLuo**: A legal multi-agent system with a fixed pipeline. JurisMA's Manager Agent provides dynamic routing.
- **vs. RAG-based methods (e.g., LSIM)**: Sentence-level statutory retrieval may conflate linguistically similar but legally distinct concepts. The element graph provides more precise contextual grounding.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of legal element graphs and multi-agent collaboration represents a novel design in legal NLP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison with multiple baselines and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Methodological description is systematic, with accurate references to legal concepts.
- **Value**: ⭐⭐⭐⭐ Both the dataset and the framework make direct contributions to legal AI research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[ACL 2026\] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)

<!-- RELATED:END -->
