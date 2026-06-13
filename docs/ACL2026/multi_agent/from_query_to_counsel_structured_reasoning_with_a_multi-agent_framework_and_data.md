---
title: >-
  [Paper Note] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation
description: >-
  [ACL 2026][Multi-Agent][Legal Question Answering] This paper introduces JurisCQAD—a large-scale dataset containing over 43,000 real-world Chinese legal consultations—and proposes the JurisMA multi-agent framework. By uti…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Legal Question Answering"
  - "Legal Element Graph"
  - "Task Decomposition"
  - "Chinese Law"
date: 2026-05-08
content_hash: d391dd6c3ad6b610
---

# From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation

**Conference**: ACL 2026  
**arXiv**: [2604.10470](https://arxiv.org/abs/2604.10470)  
**Code**: None  
**Area**: LLM Agent / Legal NLP  
**Keywords**: Legal Question Answering, Multi-Agent, Legal Element Graph, Task Decomposition, Chinese Law

## TL;DR
This paper introduces JurisCQAD—a large-scale dataset containing over 43,000 real-world Chinese legal consultations—and proposes the JurisMA multi-agent framework. By utilizing a legal element graph for structured task decomposition and dynamic multi-agent collaboration (Manager Agent, Format Check, and Law Search), the framework significantly outperforms both general-purpose and legal-specific LLMs on LawBench.

## Background & Motivation

**Background**: Legal Consultation Question Answering (Legal CQA) is a core task in legal AI, requiring the generation of evidence-based and actionable legal advice from personalized legal dilemmas. Existing methods primarily rely on continued pre-training on legal corpora or utilizing legal article retrieval to assist generation.

**Limitations of Prior Work**: (1) Lack of high-quality training data—existing legal LLMs (e.g., LawGPT) are mainly trained on synthetic data, leading to a domain shift from real consultation scenarios; (2) Legal CQA involves complex task combinations—requiring the identification of legal relationships, causal reasoning, pinpointing core issues, and matching legal articles, which end-to-end models struggle to cover entirely; (3) High context dependency—demanding precise interpretation of legal entities, relationships, and user intent.

**Key Challenge**: Real legal consultations are often vague and multifaceted, requiring dynamic interpretation of facts, subjects, and legal implications. Existing methods either rely on coarse continued pre-training (with low-quality supervision signals) or sentence-level retrieval (which easily confuses concepts that are legally distinct but linguistically similar).

**Goal**: To construct a large-scale dataset of real legal consultations and design an interpretable framework for task decomposition and multi-agent collaboration.

**Key Insight**: Decomposing a legal consultation into a structured legal element graph—extracting entities, events, relationships, user intent, and legal issues—followed by iterative optimization of legal advice through multi-agent collaboration.

**Core Idea**: The element graph provides the semantic foundation $\rightarrow$ the Manager Agent dynamically coordinates sub-tasks $\rightarrow$ Format Check and Law Search Agents perform iterative refinement $\rightarrow$ the Content Check Agent provides final polishing.

## Method

### Overall Architecture
JurisMA operates in three stages: (1) Legal Semantic Graph Construction—the Element Agent parses the query into a graph structure containing entities, events, relationships, intent, and legal issues; (2) Multi-Agent Iterative Optimization—the Manager Agent dynamically evaluates draft quality and invokes the FormatCheck Agent or LawSearch Agent as needed for structural corrections or legal supplementation; (3) Content Revision—the Content Check Agent performs final modifications for linguistic quality and professionalism.

### Key Designs

1.  **Legal Element Graph Construction**:
    - **Function**: Transforms free-text queries into structured legal semantic representations to provide global context for downstream reasoning.
    - **Mechanism**: Defines a graph $G = (V, E)$, where nodes $V$ include legal entities (individuals/organizations with roles/statuses/temporal attributes), legal events, user claims, key facts, and inferred legal issues. Edges $E$ represent semantic relationships (e.g., kinship, contractual obligations). The graph is serialized into JSON and concatenated with the query as semantic input for generation.
    - **Design Motivation**: Legal reasoning centers on identifying key facts, involved subjects, and legal relationships. A graph structure captures this structured information better than flat text, inspired by Hart's primary/secondary rules and Kelsen's hierarchy of norms.

2.  **Manager Agent Dynamic Coordination**:
    - **Function**: Acts as a central controller to dynamically assess draft quality and selectively activate sub-agents.
    - **Mechanism**: In each iteration, the Manager Agent checks the draft for (1) linguistic sufficiency (clarity and conciseness) and (2) legal completeness (inclusion of authoritative citations). If structural or expression issues are detected, the Format Check Agent is invoked to provide suggestions for the Draft Agent; if legal citations are insufficient, the Law Search Agent retrieves relevant articles from a statutory database. The process continues for up to 5 iterations or until the Manager Agent returns "Pass."
    - **Design Motivation**: Not all drafts require every type of revision. Dynamic routing ensures efficiency and controllability by activating only necessary sub-agents, mimicking the collaborative workflow of a professional law firm.

3.  **JurisCQAD Dataset**:
    - **Function**: Provides a large-scale, high-quality dataset of real legal consultations for training and evaluation.
    - **Mechanism**: Contains 43,000+ instances, each organized as a triplet (question, positive answer, negative answer). Derived from real user consultations and verified by experts, it covers high-frequency legal domains. Positive answers are reviewed by professional lawyers, while negative answers are model-generated but labeled as inadequate or incorrect.
    - **Design Motivation**: Existing Chinese legal datasets are mostly synthetic Q&A or article retrieval tasks, failing to reflect the complexity and linguistic diversity of real consultations. The triplet format supports contrastive learning and preference optimization.

### Loss & Training
The model is trained using SFT on JurisCQAD with element-graph-augmented inputs. Evaluation is conducted on a revised LawBench using various lexical and semantic metrics.

## Key Experimental Results

### Main Results

| Model Category | Representative Model | LawBench Performance |
| :--- | :--- | :--- |
| General LLM | GPT-4, Qwen | Moderate |
| Legal LLM | LawGPT, ChatLaw | Low to Moderate |
| JurisMA | Trained on JurisCQAD | **Significantly Superior** |

### Ablation Study

| Configuration | Performance | Explanation |
| :--- | :--- | :--- |
| Full JurisMA | Optimal | Complete framework |
| w/o Element Graph | Decrease | Loss of structured semantic foundation |
| w/o Multi-Agent Iteration | Decrease | Incomplete legal citations |
| w/o LawSearch Agent | Significant Decrease | Loss of legal grounding capability |

### Key Findings
- Models trained on JurisCQAD significantly outperform both general and legal-specific LLMs, validating the value of high-quality data.
- Structured context provided by the element graph is more effective than raw text input.
- Multi-agent iterations typically converge in 2-3 rounds, indicating the effectiveness of the Manager Agent's quality judgment.
- Interestingly, legal-specific LLMs sometimes underperform general LLMs, likely due to inconsistent quality in their pre-training data.

## Highlights & Insights
- **Semantic Representation via Legal Element Graphs**: Transforming free-text queries into entity-relationship-event graphs provides an interpretable semantic foundation for legal reasoning. This approach is generalizable to other domains requiring structured understanding (e.g., medicine, finance).
- **Dynamic Agent Routing**: The Manager Agent activates sub-agents on demand rather than following a fixed pipeline, balancing efficiency and quality.
- **Power of Real-World Data**: The high-quality dataset of 43,000 real legal consultations serves as a significant infrastructural contribution.

## Limitations & Future Work
- The framework currently covers only the Chinese legal system; cross-jurisdictional applicability remains unverified.
- The quality of the element graph depends heavily on the comprehension capabilities of the Element Agent.
- The scope of legal article retrieval by the LawSearch Agent may be incomplete.
- The inference cost of the multi-agent system is relatively high due to multiple iterations and agent calls.

## Related Work & Insights
- **vs. LawGPT**: LawGPT uses continued pre-training on legal corpora but features coarse data processing. JurisMA employs structured task decomposition and higher-quality data.
- **vs. LawLuo**: LawLuo uses a fixed-pipeline multi-agent system. JurisMA's Manager Agent provides dynamic routing.
- **vs. RAG Methods (e.g., LSIM)**: Sentence-level retrieval can confuse linguistically similar but legally distinct concepts. The element graph provides more precise context.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of legal element graphs and multi-agent collaboration is a novel design in legal NLP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Extensive comparisons against various baselines and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Systematic description of methodology with accurate references to legal concepts.
- **Value**: ⭐⭐⭐⭐ The dataset and framework provide direct contributions to legal AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/multi_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)

</div>

<!-- RELATED:END -->
