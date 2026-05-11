---
title: >-
  [Paper Note] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh
description: >-
  [ACL 2026][LLM Agent][Legal Assistant] This work presents Mina, a multilingual LLM-powered legal assistant for the Bangladeshi legal domain. Through a two-stage RAG pipeline that accurately retrieves relevant acts and sp…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Legal Assistant"
  - "Multilingual Agent"
  - "RAG"
  - "Bangladesh Law"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: f65adeac5a05e4ff
---

# Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh

**Conference**: ACL 2026
**arXiv**: [2511.08605](https://arxiv.org/abs/2511.08605)
**Code**: [GitHub](https://github.com/)
**Area**: LLM Agent / Legal NLP
**Keywords**: Legal Assistant, Multilingual Agent, RAG, Bangladesh Law, Low-Resource Languages

## TL;DR
This work presents Mina, a multilingual LLM-powered legal assistant for the Bangladeshi legal domain. Through a two-stage RAG pipeline that accurately retrieves relevant acts and specific provisions, combined with a tool chain and multilingual embeddings, Mina achieves 75–80% passing rates on the Bangladesh Bar Council exam while reducing legal consultation costs to just 0.12–0.61% of traditional methods.

## Background & Motivation

**Background**: Bangladesh's judicial system faces a backlog of 3.7–4.4 million cases, with only 2,100 judges (one per 90,000 people), civil disputes dragging on for decades, unregulated and expensive legal fees, and limited public legal aid funding.

**Limitations of Prior Work**: (1) Existing AI legal assistants lack Bengali language support and are not adapted to Bangladeshi jurisdiction; (2) Bangladesh's legal system is rooted in colonial-era codes containing substantial Persian-origin terminology that English-dominant models cannot handle effectively; (3) Low-income populations face a triple barrier of complex legal language, opaque procedures, and high costs.

**Key Challenge**: A triple low-resource challenge spanning language, legal system, and infrastructure—scarce Bengali NLP tools, highly specialized and cross-lingual legal terminology, and a target user base with limited legal and digital literacy.

**Goal**: To construct a localized, multilingual legal assistant capable of drafting legal documents, citing statutes, and translating complex legal language into plain Bengali explanations.

**Key Insight**: Rather than pursuing innovation in a single module, the work combines proven components (multilingual embeddings, RAG, LangGraph agents) and deeply adapts them to the bilingual, low-resource legal environment.

**Core Idea**: A two-stage RAG pipeline (first retrieving act summaries, then retrieving specific provisions) combined with a custom legal dictionary and a multi-agent workflow to deliver jurisdiction-specific, precise legal answers.

## Method

### Overall Architecture
The system centers on an Orchestrator Agent that evaluates user input, conversation history, and documents to determine the response path. When internal context is insufficient, a two-stage RAG is triggered: Cohere multilingual embeddings first retrieve relevant acts, then specific provisions are retrieved within those acts. External tools (web search, document parsing, etc.) are invoked on demand.

### Key Designs

1. **Two-Stage RAG Pipeline**:

    - **Function**: Accurately retrieves Bangladeshi legal provisions and prevents cross-act content confusion.
    - **Mechanism**: Two independent vector databases are constructed—an Act Database (LLM-generated summaries of 595 acts) and a Provision Database (18,023 chunked and indexed provisions). At query time, semantic keywords are used to retrieve the top-5 acts, which then filter the provision database to return the top-10 relevant provisions. Retrieval results undergo a relevance check; if insufficient, the query is rewritten and retried.
    - **Design Motivation**: Naive RAG frequently mixes provisions from unrelated acts. The two-stage design ensures breadth at the act level and precision at the provision level.

2. **Dual-Agent Architecture**:

    - **Function**: Separates decision-making and retrieval responsibilities to improve maintainability.
    - **Mechanism**: The Orchestrator Agent evaluates query context to decide whether to answer directly or trigger retrieval; the RAG Agent manages the end-to-end retrieval process. Both operate as LangGraph state machines with persistent cross-turn memory.
    - **Design Motivation**: Separation of concerns makes the system modular, facilitating independent debugging and extension.

3. **Tool Chain and Legal Dictionary**:

    - **Function**: Handles auxiliary tasks and colonial-era legal terminology.
    - **Mechanism**: Eight specialized tools include a document parser (supporting pptx/docx/pdf), a keyword generator (LLM with regex fallback), web search (DuckDuckGo), a question relevance analyzer, a custom legal dictionary (explaining Persian-origin and colonial-era terms), a socioeconomic simulation module, and others.
    - **Design Motivation**: Persian-origin and colonial-era terms in Bangladeshi legal texts represent a primary barrier to model comprehension; the legal dictionary directly bridges this domain knowledge gap.

### Loss & Training
No model training is involved. The system is built upon prompt engineering and RAG over pre-trained LLMs. Thirteen models are evaluated, including GPT-4o, Gemini series, Llama series, and Qwen.

## Key Experimental Results

### Main Results (Bar Council Exam MCQ, 2-Step RAG + Tools)

| Model | 2022 | 2023 | Notes |
|-------|------|------|-------|
| Gemini-2.5-Flash | **77.0%** | **77.0%** | Highest score, matches/exceeds human average |
| GPT-4o | 73.6% | 72.2% | Strong baseline |
| Llama3.1-70B | 42.4% | 46.2% | Best open-source |
| Qwen3-30B | 70.8% | 72.4% | Second-best open-source |
| w/o RAG (GPT-4o) | 18.6% | 19.2% | Demonstrates importance of RAG |

### Ablation Study

| Configuration | MCQ Accuracy | Notes |
|---------------|-------------|-------|
| w/o RAG | 18.6% | No retrieval, near-random guessing |
| Naive RAG | 62.4% | Single-stage retrieval, frequent act confusion |
| 2-Step RAG | 69.2% | Two-stage retrieval, improved precision |
| 2-Step RAG + Tools | 73.6% | Full system |

### Key Findings
- RAG is the core of the system: without it, GPT-4o achieves only 18.6% (near the 25% random baseline); adding 2-Step RAG raises this to 69.2%.
- Two-stage RAG consistently outperforms naive RAG by approximately 7–10 percentage points, validating the necessity of hierarchical retrieval.
- Small models (<4B) perform poorly even with RAG, indicating that legal reasoning imposes a minimum requirement on model scale.
- Cost analysis shows Mina's operating cost is only 0.12–0.61% of traditional legal consultation, representing a 99.4–99.9% cost reduction.

## Highlights & Insights
- This work exemplifies system-level innovation over module-level innovation—although each component relies on existing techniques, deep adaptation to a bilingual, low-resource legal setting yields substantial practical value.
- The two-stage RAG design is simple yet effective: macro-level act localization followed by micro-level provision localization avoids the critical problem of cross-act confusion.
- Using the bar council exam as an evaluation benchmark is highly convincing, directly validating system usability in realistic legal scenarios.

## Limitations & Future Work
- The legal database currently covers only 595 acts; a large portion of the Bangladeshi legal system remains undigitized, including regulations and case law.
- Performance in real user scenarios (as opposed to exam settings) has not been evaluated; exam questions may not fully represent actual legal consultation needs.
- The socioeconomic simulation module lacks sufficient evaluation detail regarding its practical effectiveness.
- Legal advice generated by the model carries no legal standing and poses a risk of misuse.

## Related Work & Insights
- **vs. General Legal AI**: Existing legal AI systems (e.g., Harvey AI) target common law jurisdictions and cannot handle the specificities of the Bangladeshi legal system.
- **vs. General RAG**: Naive RAG produces serious errors in legal settings due to act-level confusion; the two-stage design is essential for domain adaptation.
- **vs. Multilingual LLMs**: Even LLMs with Bengali support lack jurisdiction-specific knowledge; RAG combined with a legal dictionary bridges this gap.

## Rating
- Novelty: ⭐⭐⭐ Primarily system integration innovation; limited novelty at the individual module level.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 13 models, three-stage exam evaluation, and legal expert review.
- Writing Quality: ⭐⭐⭐⭐ Detailed background with clearly motivated problem statement.
- Value: ⭐⭐⭐⭐⭐ Genuinely addresses legal accessibility in a low-resource setting with significant societal impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)

</div>

<!-- RELATED:END -->
