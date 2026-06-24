---
title: >-
  [Paper Note] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases
description: >-
  [AAAI 2026][Multi-Agent][Multi-Agent System] This paper presents Thucy, the first multi-agent claim verification system supporting cross-database and cross-table reasoning. Led by a Verifier agent, it coordinates three specialized agents (Data/Schema/SQL Expert) with zero prior knowledge of the data sources, enabling autonomous discovery, reasoning, and SQL evidence generation. Thucy surpasses the previous SOTA by 5.6 percentage points on TabFact (94.3%).
tags:
  - "AAAI 2026"
  - "Multi-Agent"
  - "Multi-Agent System"
  - "Claim Verification"
  - "Relational Databases"
  - "SQL Evidence Generation"
  - "Cross-Database Reasoning"
date: 2026-05-08
content_hash: 4c77f69767f5e5f5
---

# Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases

**Conference**: AAAI 2026
**arXiv**: [2512.03278](https://arxiv.org/abs/2512.03278)  
**Code**: [GitHub](https://github.com/michaeltheologitis/thucy)  
**Area**: Audio & Speech
**Keywords**: Multi-Agent System, Claim Verification, Relational Databases, SQL Evidence Generation, Cross-Database Reasoning

## TL;DR

This paper presents Thucy, the first multi-agent claim verification system supporting cross-database and cross-table reasoning. Led by a Verifier agent, it coordinates three specialized agents (Data/Schema/SQL Expert) with zero prior knowledge of the data sources, enabling autonomous discovery, reasoning, and SQL evidence generation. Thucy surpasses the previous SOTA by 5.6 percentage points on TabFact (94.3%).

## Background & Motivation

**Background**: Everyday discourse is filled with claims verifiable against structured data (e.g., declining crime rates, economic growth), yet most people lack the technical means to check them. Advances in LLMs and agentic AI make automated verification increasingly feasible.

**Limitations of Prior Work**: Existing fact verification systems suffer from three major limitations: (a) support only single-table queries (no cross-table or cross-database reasoning); (b) typically handle only small tables (hundreds of rows) that fit within an LLM context window; (c) provide no traceable evidence, leaving users unable to audit the system's reasoning.

**Key Challenge**: Real-world claim verification often requires multi-table joins across multiple databases and autonomous exploration of previously unknown data sources—assumptions that existing methods do not support.

**Goal**: To build a multi-agent system with zero prior knowledge of the data environment, capable of cross-database and cross-table verification, and able to provide concrete SQL evidence.

**Key Insight**: Drawing on the rigorous evidentiary standards of the ancient Greek historian Thucydides, the system is designed to produce not only a verdict but also evidence in the form of SQL queries and their execution results.

**Core Idea**: The verification task is decomposed into four stages—data exploration → schema understanding → SQL querying → verdict synthesis—each handled by a specialized agent, coordinated by the Verifier. The key design principles are **expert decoupling** and **Verifier context protection**: experts handle low-level, noisy database interactions, while the Verifier receives only concise, high-quality information.

## Method

### Overall Architecture

Thucy adopts an **Agents-as-Tools** architecture, in which a high-level **Verifier** agent orchestrates three specialist agents: **Data Expert** (surveys available data sources), **Schema Expert** (answers schema-related questions), and **SQL Expert** (writes and executes SQL queries, returning evidence). The three specialists do not communicate with each other directly and interact only with non-AI tools. The Verifier wraps each specialist as a callable tool, while the data layer is accessed via Google MCP Toolbox in a plug-and-play fashion.

### Key Designs

**Module 1: Data Expert**

- **Function**: Performs a high-level scan of all available data sources before verification begins, producing a concise single-paragraph summary informing the Verifier of what data is available.
- **Mechanism**: Subscribes to the schema tool set and autonomously browses all connected databases, traversing table names, column names, data types, and other metadata, then distills the noisy low-level information into a clear natural-language summary.
- **Design Motivation**: Data exploration involves substantial low-level detail (database names, table names, column names, types, metadata) that would severely pollute the Verifier's precious context window if exposed directly. The Data Expert "absorbs this noise" and propagates only refined information upward.

**Module 2: Schema Expert**

- **Function**: Answers arbitrary questions about database schemas—from the meaning of column names to foreign-key relationships and constraints.
- **Mechanism**: Also subscribes to the schema tool set but is encouraged to explore low-level structural details thoroughly. It accepts two inputs: (1) a natural-language schema question and (2) a contextual hint (e.g., "Seattle, WA") to direct the agent toward relevant databases. It outputs a formatted Markdown summary that precisely answers the schema question.
- **Design Motivation**: Real-world databases often have opaque column names (e.g., "NIBRS Group AB," "Beat") requiring dedicated exploration and interpretation. The Schema Expert shields the Verifier from noisy metadata, delivering accurate schema information in refined form.

**Module 3: SQL Expert**

- **Function**: Translates natural-language questions into SQL queries, executes them against the database, and returns the answer along with supporting SQL.
- **Mechanism**: Accepts two inputs: (1) a natural-language data query and (2) relevant schema information provided by the Schema Expert. The execution process mirrors a real data analyst's workflow—first exploring the data to resolve ambiguities (e.g., does "quarter" refer to calendar or fiscal year?), then writing the query, handling syntax errors and semantic mismatches, and finally returning the answer with the corresponding SQL. The agent is instructed to exclude exploratory or failed queries, retaining only evidence relevant to the final answer.
- **Design Motivation**: NL2SQL is not a one-shot process; it requires iterative interaction with the database. The SQL Expert is designed to enter a "tunnel-vision" mode—focusing solely on the current question and relevant data fragments to avoid distraction from irrelevant information.

**Verifier**

- **Function**: Coordinates all specialist agents, drives the verification workflow, and produces a final verdict (Verified / Partly Verified / Partly Inaccurate / Inaccurate) along with an analytical report containing explanatory SQL.
- **Mechanism**: An iterative workflow—first queries the Data Expert for a data overview → consults the Schema Expert for relevant schema → calls the SQL Expert to query data → repeats the loop if information is insufficient → synthesizes the final report. The Verifier **never directly accesses the underlying data**, only asking the right questions and receiving refined answers.
- **Design Motivation**: The Verifier uses the most capable model (GPT-5), and its context must remain lightweight and information-dense. Delegating "dirty work" to specialists allows the Verifier to focus on high-level reasoning and judgment.

### Loss & Training

This paper involves no model training. The system is built on the OpenAI Agents SDK. The Verifier uses GPT-5; specialist agents use GPT-5-mini (with fallback to GPT-4o-mini). The data layer is connected via Google MCP Toolbox, supporting PostgreSQL, MySQL, and others. Each specialist agent's memory persists only within a single tool call (not across calls), preserving atomicity and reusability.

## Key Experimental Results

### Main Results

Accuracy comparison on the TabFact benchmark (small test split, ~2k fact–table pairs):

| Method | Model | Accuracy (%) |
|--------|-------|-------------|
| BINDER | Codex | 85.1 |
| DATER | Codex | 85.6 |
| CoTable | PaLM 2 | 86.6 |
| ReAcTable | Codex | 86.1 |
| AutoTQA | GPT-4-turbo | 88.7 |
| POS | GPT-4o-mini | 82.7 |
| **Thucy** | GPT-4o-mini | **93.7** |
| **Thucy** | GPT-5-mini | **94.3** |

### Ablation Study

- **Robustness to model downgrade**: Downgrading specialist agents from GPT-5-mini to GPT-4o-mini reduces accuracy by only 0.6% (94.3% → 93.7%), still substantially outperforming the SOTA (+5.0%). This suggests that architectural design matters more than the capability of individual agents.
- **Cross-table and cross-database capability**: Thucy is the only system that simultaneously supports Cross-Table + Cross-Database + Source-Agnostic + Verifiable verification.

### Key Findings

1. **Specialist decomposition is highly effective**: Even with a weaker model (GPT-4o-mini), Thucy substantially outperforms baselines using equivalent or stronger models.
2. **Source-agnostic design is feasible**: The system autonomously discovers and explores databases with no prior knowledge.
3. **Value of SQL evidence**: In a Seattle crime rate verification case, Thucy not only issues a verdict but also returns directly executable SQL queries that users can modify to explore further.
4. **Validation on real cases**: For a Seattle city attorney's claim that "crime rates fell in 2024," Thucy finds that the actual data show a slight increase (property crime +0.7%, violent crime +0.8%), returning a verdict of "Inaccurate."

## Highlights & Insights

- **Clean realization of the Agents-as-Tools pattern**: Expert encapsulation as tools, atomic execution (no cross-call memory), and Verifier context protection together constitute a scalable multi-agent architecture.
- **Engineering value of MCP Toolbox**: Databases can be plugged in via YAML configuration, demonstrating the practical utility of the MCP standardization protocol in real-world agent systems.
- **Real-world news verification cases are highly compelling**: The revival of Thucydidean evidentiary rigor in modern information verification. The step-by-step investigation of the Seattle crime rate and the "36% drop in violent crime" cases demonstrates the system's ability to handle ambiguous definitions (e.g., differing interpretations of "downtown").
- **Necessity of the Schema Expert**: Opaque column names and complex schemas in real databases are the primary obstacle to NL2SQL; a dedicated schema-understanding agent is a key contribution.

## Limitations & Future Work

1. **Evaluation limited to a single-table benchmark**: TabFact is itself a single-table dataset and does not fully showcase Thucy's cross-database and cross-table advantages, which are demonstrated only qualitatively through case studies.
2. **High operational cost**: The Verifier uses GPT-5 and the three specialists each execute numerous tool calls; deployment costs merit attention.
3. **Restricted to SQL data sources**: The system cannot handle non-relational data (e.g., JSON, CSV, graph databases) or verify claims requiring unstructured information (e.g., text, images).
4. **Manual claim submission**: No automated claim extraction (e.g., automatically identifying verifiable claims from news articles) is integrated.
5. **Error propagation risk**: Errors made by specialist agents (e.g., schema misinterpretation or SQL semantic errors) may propagate to the Verifier.

## Related Work & Insights

- **Fact verification systems**: BINDER (NL→program synthesis), DATER (decompose-and-aggregate), CoTable (multi-step reasoning), ReAcTable (ReAct paradigm)—all limited to single tables.
- **AutoTQA**: The first system to support cross-table reasoning, but uses a cyclic orchestration pattern (execute→critique→refine loop) and provides no traceable evidence.
- **POS**: Focuses on interpretability (returns a sequence of atomic steps) but is limited to single queries and not directly verifiable.
- **MCP Protocol**: Anthropic's standardized AI-to-data-source connection protocol, adopted by PayPal, Databricks, Snowflake, and others.
- **Insights**: The success of multi-agent systems depends less on the capability ceiling of individual agents than on sound task decomposition and context management strategies.

## Rating

⭐⭐⭐⭐

The system is well-designed—the combination of source-agnostic operation, cross-database and cross-table reasoning, and SQL evidence is a first in the fact verification domain. The Agents-as-Tools architecture is clean and pragmatic, and the real-world cases are highly convincing. The SOTA results on TabFact are persuasive. The main weaknesses are that the benchmark does not fully demonstrate the system's core advantages (cross-database and cross-table) and that deployment costs are relatively high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](../../ACL2026/multi_agent/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[ICLR 2026\] CoAct-1: Computer-using Multi-agent System with Coding Actions](../../ICLR2026/multi_agent/coact-1_computer-using_multi-agent_system_with_coding_actions.md)

</div>

<!-- RELATED:END -->
