---
title: >-
  [Paper Note] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh
description: >-
  [ACL 2026][LLM Agent][Legal Assistant] Mina is developed as a multilingual LLM legal assistant for the Bangladesh legal context. Through a two-stage RAG pipeline for precise act and section retrieval…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Legal Assistant"
  - "Multilingual Agent"
  - "RAG"
  - "Bangladesh Law"
  - "Low-resource Language"
date: 2026-05-08
content_hash: 286c342ff9ee94db
---

# Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh

**Conference**: ACL 2026  
**arXiv**: [2511.08605](https://arxiv.org/abs/2511.08605)  
**Code**: [GitHub](https://github.com/)  
**Area**: LLM Agent / Legal NLP  
**Keywords**: Legal Assistant, Multilingual Agent, RAG, Bangladesh Law, Low-resource Language

## TL;DR
Mina is developed as a multilingual LLM legal assistant for the Bangladesh legal context. Through a two-stage RAG pipeline for precise act and section retrieval, combined with a toolchain and multilingual embeddings, it achieved a 75-80% pass rate on the Bangladesh Bar Council MCQ exam. The legal consultation cost is only 0.12-0.61% of traditional methods.

## Background & Motivation

**Background**: The Bangladesh judicial system faces a backlog of 3.7-4.4 million cases with only 2,100 judges (1 per 90,000 people). Civil disputes can drag on for decades, lawyer fees are high and unregulated, and public legal aid funding is limited.

**Limitations of Prior Work**: (1) AI legal assistants lack support for Bengali and are not adapted to the Bangladesh jurisdiction; (2) The legal system is rooted in colonial-era codes containing numerous Persian-origin terms, which English-dominated models fail to process effectively; (3) Low-income populations face triple barriers: complex legal language, opaque procedures, and high costs.

**Key Challenge**: The triple low-resource nature of language, legal system, and resources—scarcity of Bengali NLP tools, highly specialized cross-lingual mixed legal terminology, and lack of legal and digital literacy among target users.

**Goal**: To construct a localized multilingual legal assistant capable of drafting legal documents, citing statutes, and translating complex legal language into plain Bengali explanations.

**Key Insight**: Combine mature components (multilingual embeddings, RAG, LangGraph Agent) with deep adaptation for a bilingual low-resource legal environment, rather than seeking innovation in a single module.

**Core Idea**: A two-stage RAG pipeline (retrieving act summaries followed by specific sections) + a custom legal dictionary + a multi-agent workflow to achieve jurisdiction-specific precise legal answering.

## Method

### Overall Architecture
The system centers on an Orchestrator Agent that evaluates user input, conversation history, and documents to determine the response path. If internal context is insufficient, a two-stage RAG is triggered: Cohere multilingual embeddings first retrieve relevant acts, followed by a search for specific sections within those acts. External tools (web search, document parsing, etc.) are called as needed.

### Key Designs

1.  **Two-Stage RAG Pipeline**:
    *   **Function**: Precisely retrieve Bangladesh legal provisions while avoiding confusion across different acts.
    *   **Mechanism**: Two independent vector databases are constructed: an Act Database (LLM summaries of 595 acts) and a Section Database (18,023 indexed chunks). Queries first use semantic keywords to retrieve the top-5 acts, then use Act IDs to filter the Section Database for the top-10 relevant sections. Retrieval results undergo relevance checks; if insufficient, queries are rewritten and retried.
    *   **Design Motivation**: Naive RAG often mixes sections from unrelated acts to generate answers. The two-stage design ensures breadth at the act level and precision at the section level.

2.  **Dual-Agent Architecture**:
    *   **Function**: Separate decision-making and retrieval responsibilities to enhance maintainability.
    *   **Mechanism**: The Orchestrator Agent evaluates query context to decide whether to answer directly or trigger retrieval; the RAG Agent manages the end-to-end retrieval process. Both operate on a LangGraph state machine, supporting persistent memory across turns.
    *   **Design Motivation**: Separation of concerns makes the system modular, facilitating independent debugging and expansion.

3.  **Toolchain and Legal Dictionary**:
    *   **Function**: Handle auxiliary tasks and colonial-era legal terminology.
    *   **Mechanism**: Eight specialized tools include document parsers (supporting pptx/docx/pdf), keyword generators (LLM + regex fallback), web search (DuckDuckGo), query relevance analyzers, a custom legal dictionary (explaining Persian-origin and colonial-era terms), and a socio-economic simulation module.
    *   **Design Motivation**: The high frequency of Persian-origin and colonial-era terms in Bangladesh legal texts is a primary barrier to model understanding; the legal dictionary directly bridges this domain knowledge gap.

### Loss & Training
This work does not involve model training. The system is built on prompt engineering and RAG using pre-trained LLMs. Thirteen models, including GPT-4o, Gemini series, Llama series, and Qwen, were evaluated.

## Key Experimental Results

### Main Results (Bar Exam MCQ, 2-Step RAG + Tools)

| Model | 2022 | 2023 | Description |
| :--- | :--- | :--- | :--- |
| Gemini-2.5-Flash | **77.0%** | **77.0%** | Highest score, matching/exceeding human average |
| GPT-4o | 73.6% | 72.2% | Strong baseline |
| Llama3.1-70B | 42.4% | 46.2% | Best open-source |
| Qwen3-30B | 70.8% | 72.4% | Second best open-source |
| w/o RAG (GPT-4o) | 18.6% | 19.2% | Importance of RAG |

### Ablation Study

| Configuration | MCQ Accuracy | Description |
| :--- | :--- | :--- |
| w/o RAG | 18.6% | No retrieval, nearly random guessing |
| Naive RAG | 62.4% | Single-stage retrieval, frequent act confusion |
| 2-Step RAG | 69.2% | Two-stage retrieval, precision improvement |
| 2-Step RAG + Tools | 73.6% | Full system |

### Key Findings
*   RAG is the core of the system: Without RAG, GPT-4o achieves only 18.6% (close to the 25% random choice), which surges to 69.2% with 2-Step RAG.
*   Two-stage RAG consistently outperforms naive RAG by approximately 7-10 percentage points, validating the necessity of hierarchical retrieval.
*   Small models ($<4B$) perform poorly even with RAG, indicating a threshold for model scale in legal reasoning.
*   Cost analysis shows Mina's operational cost is only 0.12-0.61% of traditional legal consultation, representing a 99.4-99.9% cost saving.

## Highlights & Insights
*   A model for system-level rather than module-level innovation—while each component utilizes existing technology, deep adaptation to a bilingual low-resource legal context yields high practical value.
*   The two-stage RAG design is simple yet effective: positioning acts macroscopically before locating sections microscopically avoids the critical issue of cross-act confusion.
*   Passing the Bar exam serves as a persuasive evaluation method, directly validating the system's utility in real-world legal scenarios.

## Limitations & Future Work
*   The legal database currently covers only 595 acts, while many regulations and precedents in the Bangladesh legal system remain non-digitized.
*   Performance in real user scenarios (non-exam) has not been evaluated; exam questions may not fully represent actual legal consultation needs.
*   The effectiveness and evaluation details of the socio-economic simulation module are insufficient.
*   The model's legal advice lacks legal validity, posing a risk of misuse.

## Related Work & Insights
*   **vs. General Legal AI**: Existing systems (e.g., Harvey AI) target common law systems and cannot handle the specificities of the Bangladesh legal framework.
*   **vs. General RAG**: Naive RAG leads to severe errors in legal contexts due to act confusion; the two-stage design is key to domain adaptation.
*   **vs. Multilingual LLMs**: Even LLMs supporting Bengali lack jurisdiction-specific knowledge; RAG and the legal dictionary fill this domain expertise gap.

## Rating
*   Novelty: ⭐⭐⭐ Primarily system integration innovation, limited module-level innovation.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 13 models, three-stage exam evaluation, and legal expert review.
*   Writing Quality: ⭐⭐⭐⭐ Detailed background and clear motivation.
*   Value: ⭐⭐⭐⭐⭐ Effectively addresses low-resource legal accessibility with significant social impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery](moose-copilot_a_web-based_interactive_assistant_for_unified_exploratory_and_fine.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)
- [\[ACL 2026\] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)

</div>

<!-- RELATED:END -->
