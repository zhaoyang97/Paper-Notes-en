---
title: >-
  [Paper Note] PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts
description: >-
  [ACL2026][LLM Evaluation][agentic retrieval] PolitNuggets introduces a multilingual agentic discovery benchmark encompassing over 10…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "agentic retrieval"
  - "political biographies"
  - "long-tail facts"
  - "multilingual retrieval"
  - "FactNet"
date: 2026-05-08
content_hash: 66d0fe41dac9827d
---

# PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts

**Conference**: ACL2026  
**arXiv**: [2605.14002](https://arxiv.org/abs/2605.14002)  
**Code**: https://github.com/yifeifrank/poli_searcher  
**Area**: information_retrieval  
**Keywords**: agentic retrieval, political biographies, long-tail facts, multilingual retrieval, FactNet

## TL;DR
PolitNuggets introduces a multilingual agentic discovery benchmark encompassing over 10,000 political career facts for 400 global political figures. Using the FactNet dynamic evidence verification protocol, the study finds that current agents exhibit high precision but low recall, with the primary bottlenecks being long-tail fact discovery, non-English evidence, and efficient tool usage.

## Background & Motivation
**Background**: Long-context LLMs enable models to perform Reasoning in Context within given materials. Tool-augmented agents further allow models to actively search the web, read data, and organize evidence, gradually forming Reasoning through Context. Deep Research workflows in production systems have demonstrated the potential of this approach.

**Limitations of Prior Work**: Many existing benchmarks still lean toward short-form QA, single-fact retrieval, or static long-text extraction. Real-world research tasks are more akin to "reconstructing a person's professional trajectory": facts are scattered across government websites, news archives, non-English materials, and legacy web pages. Models must decide what to search, what to read, when to stop, and how to synthesize fragmented evidence into a structured timeline.

**Key Challenge**: Strong long-context performance does not equate to strong agentic discovery. A model might extract facts from a clean piece of evidence, but when evidence must be independently sourced, languages are inconsistent, sources conflict, and relevant facts are weakly connected, failures often occur in search strategy and evidence coverage rather than final generation.

**Goal**: The authors aim to establish a reproducible benchmark to measure the discovery capability of long-tail political facts, fine-grained attribute extraction, and search costs. It further analyzes whether agent success stems from short-context extraction, long-context recall, parametric knowledge, multilingual capability, or tool-calling reliability.

**Key Insight**: Political biographies serve as an excellent real-world task. Wikipedia covers US and high-profile figures well but lacks fine-grained tenure months, formal titles, and organizational changes for non-US officials. PolitNuggets views these gaps as a latent fact network, requiring agents to traverse weakly connected fact nodes in the open web.

**Core Idea**: Use an evidence-conditioned dynamic fact network, FactNet, to evaluate whether an agent has truly discovered verifiable political biography nuggets outside of Wikipedia, rather than relying on static context QA or simple string matching.

## Method
PolitNuggets encompasses benchmark construction, agent systems, and evaluation protocols. The focus is not on proposing a new retrieval algorithm but on establishing a measurement method for "long-tail fact discovery in the open web" that closely mirrors real workflows. The authors model political biographies as sets of timestamped events, each containing roles, organizations, and time intervals. Portions already covered by Wikipedia are filtered out; the evaluation target is the uncovered but evidence-verifiable long-tail facts.

### Overall Architecture
Data is sourced from WhoGov: 200 non-US cabinet politicians and 200 US legislators/senators, totaling 400 entities. The system operates under two conditions: With Wiki enhancement (inputting existing Wikipedia text to fill gaps) and Without Wiki reconstruction (cold-starting biographies from the open web using only entity names). Each agent run produces a structured biography and an evidence archive. Subsequently, FactNet determines if predicted nuggets are supported by evidence and calculates Event-Level F1, Attribute-Level F1, and search costs.

### Key Designs
1.  **Supervisor-Searcher-Archive-Coder Architecture**:
    - **Function**: Decomposes the open-ended search task into four roles: global planning, local retrieval, evidence persistence, and structured output.
    - **Mechanism**: The Supervisor maintains a global search summary and task list, decomposing the biography task for the Searcher. The Searcher performs searching, browsing, and page retrieval, storing relevant evidence chunks in the Archive. Finally, the Coder reads both the Supervisor's summary and the Archive's raw evidence to output a strict JSON schema. The system allows up to 3 focused search-retrieve cycles per subtask and a global limit of 100 LLM calls to control budget.
    - **Design Motivation**: Long-tail facts often require multi-step queries and referencing original texts. Relying solely on summaries leads to lost details. The Archive's storage of source-linked chunks prevents "contextual amnesia" and provides evidence for subsequent dynamic verification.

2.  **FactNet Dynamic Evidence Evaluation**:
    - **Function**: Prevents misclassifying true new facts discovered by the model as false positives.
    - **Mechanism**: A Consolidated Ground Truth is aggregated from multiple agent runs, and a Wikipedia coverage filter yields a Novel set $G = G_e \setminus W_e$. When a predicted nugget is not in the current $G$, it is not penalized directly; instead, a GPT-5-mini judge checks if the nugget is supported by the system's own Archive evidence. If supported and not covered by Wikipedia, it is added to the dynamic ground truth $G'$.
    - **Design Motivation**: Open-world fact discovery cannot exhaustively list all correct answers in advance. Dynamic novelty validation allows the benchmark to reward verifiable new findings while still requiring every claim to be supported by a source.

3.  **Two-Layer Granularity and Efficiency Evaluation**:
    - **Function**: Separates the ability to "find facts" from the ability to "fill details accurately" and explicitly measures cost.
    - **Mechanism**: Event-Level F1 only requires role, organization, and year matching to measure event discovery. Attribute-Level F1 further requires start/end month and exact title matching for fine-grained slot filling. Efficiency is measured by average search steps and token usage to expose systems that achieve high recall at excessive costs.
    - **Design Motivation**: In political biography tasks, a model might know someone was a minister but not the specific months or formal title. Hierarchical metrics clarify whether failure occurs at discovery or granularity, aiding system optimization.

### Loss & Training
This work presents a benchmark and evaluation system rather than training a new model. The experiment evaluates Grok-4-Fast, Gemini-2.5-Flash, Qwen-3-225B/80B, and Gemini DeepResearch. All agentic runs record token usage via OpenRouter. Searching uses Serper, and page retrieval uses Jina and Exa. Static LRM baselines use identical evidence collected from Grok-4-Fast With-Wiki runs, constructed as Short Archive context, Long raw pages context, and Memory-only bio to isolate the difference between active search and passive long-context extraction.

## Key Experimental Results

### Main Results
Main results show that Grok-4-Fast is the strongest agentic setting, maintaining similar F1 even under cold-start conditions without Wikipedia. Gemini approaches this in some settings but with higher search costs; the Qwen series lags significantly. Attribute-Level F1 is generally lower than Event-Level F1, indicating that fine-grained month and title extraction remains difficult.

| Context | Model | Region | EventF1 | AttrF1 | Main Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| With Wiki | Gemini DR | US / Non-US | $0.778$ / $0.701$ | $0.505$ / $0.489$ | High precision, conservative |
| With Wiki | Grok-4-Fast | US / Non-US | $0.768$ / $0.712$ | $0.501$ / $0.475$ | Strongest overall agentic setting |
| With Wiki | Gemini | US / Non-US | $0.638$ / $0.679$ | $0.407$ / $0.485$ | Non-US EventF1 increases, but high cost |
| With Wiki | Qwen-225B | US / Non-US | $0.499$ / $0.440$ | $0.335$ / $0.306$ | Weak in both discovery and granularity |
| Without Wiki | Grok-4-Fast | US / Non-US | $0.766$ / $0.708$ | $0.506$ / $0.475$ | Stable performance, increased steps |
| Without Wiki | Gemini | US / Non-US | $0.671$ / $0.618$ | $0.439$ / $0.468$ | Requires more search to sustain performance |

Efficiency analysis reveals that removing Wikipedia significantly increases search steps and tokens, though F1 does not necessarily collapse. Grok's average steps increased from $11.17$ (With-Wiki) to $14.52$ (Without-Wiki); Gemini rose from $13.53$ to $18.04$. Grok is described as being on a superior Pareto frontier, achieving higher F1 with fewer searches.

| Comparison | Metric | With Wiki Mean | Without Wiki Mean | Gain | 95% CI | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini | steps | $13.533$ | $18.043$ | $+4.510$ | $[3.032, 5.931]$ | Yes |
| Gemini | tokens | $770,151$ | $1,062,534$ | $+292,383$ | $[143,694, 449,363]$ | Yes |
| Grok-4-Fast | steps | $11.169$ | $14.519$ | $+3.350$ | $[2.314, 4.344]$ | Yes |
| Grok-4-Fast | tokens | $394,522$ | $461,227$ | $+66,705$ | $[32,970, 99,278]$ | Yes |

### Ablation Study
The key ablation in PolitNuggets is Archive memory versus static LRM baselines. Removing the Archive led to an Event-Level F1 drop of approximately $0.05$, confirming that original evidence snippets are more reliable than summaries. The static long-context baseline also revealed a counter-intuitive phenomenon: longer, noisier raw pages are not necessarily better than short, curated Archives.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full Supervisor-Searcher + Archive | Grok With-Wiki US EventF1 $0.768$ | Archive preserves original source-linked evidence for detail filling |
| No-Archive | Event-Level $\Delta F1 \approx -0.05$ | Summaries lose fine-grained evidence, causing contextual amnesia |
| Short Archive LRM, Gemini | US/Non-US EventF1 $0.667/0.674$ | Clean short evidence is better for extraction than long pages |
| Long raw pages LRM, Gemini | US/Non-US EventF1 $0.621/0.655$ | Long context suffers from noise, no guaranteed improvement |
| Memory-only LRM, Gemini | US/Non-US EventF1 $0.251/0.192$ | Model memory alone is insufficient; must be evidence-grounded |
| Grok short $\rightarrow$ long, US EventF1 | $0.626 \rightarrow 0.538$ | Raw long context is approx. $14.1\%$ lower than Archive |

### Key Findings
- The primary issue with current agents is recall, not precision. With-Wiki Grok-4-Fast achieves Event precision of $0.890/0.872$ (US/Non-US), but recall is only $0.703/0.620$; Attribute-Level recall is even lower.
- A significant International Evidence Gap exists. Grok-4-Fast's Non-US EventF1 is $0.0557$ lower than US (95% CI does not cross 0); the gap for Qwen-80B reaches $-0.0989$.
- Long-context capability is not a sufficient condition for agentic success. Success is supported by short-context extraction, tool-calling reliability, multilingual robustness, and parametric knowledge.
- Wiki removal increases cost but does not drastically reduce F1, suggesting agents can compensate for missing context through longer search trajectories, though efficiency issues are magnified.

## Highlights & Insights
- FactNet's dynamic ground truth design is highly suitable for open-world tasks. It avoids penalizing true new discoveries while maintaining a hard "self-supported" threshold, reducing the risk of rewarding hallucinations.
- The paper separately evaluates Reasoning in Context and Reasoning through Context. This analysis is crucial; many models score well on long-context benchmarks but fail in open-web research due to poor query planning, source selection, and tool instability.
- The political biography task places multilingualism at the heart of evaluation. Non-US entities are not "extra hard cases" but primary scenarios that real information systems must face.
- Reporting both F1 and cost avoids chasing high scores at any price. The most expensive part of real Deep Research systems is repeated searching and reading; efficiency curves provide more product value than single-point accuracy.

## Limitations & Future Work
- Due to budget constraints, the paper did not evaluate the most powerful and expensive frontier models; conclusions may shift as models and retrieval products update.
- The benchmark depends on search engines and web states. Although the authors release cached pages, real online runs are affected by ranking drift, page disappearances, and content updates.
- The static LRM baseline uses evidence collected by agent runs, thus it does not strictly prove Reasoning through Context is superior to Reasoning in Context, only demonstrating extraction differences on the same evidence snapshot.
- Fact verification relies on LLM judges. While human re-evaluation shows a correlation of $0.87$ and Exa spot checks reveal a false positive rate of roughly $3.66\%$, cross-lingual titles and historical organization names may still present edge cases.
- The task focuses on public political figures. Technically, it could be migrated to private profiling, so downstream usage requires clear ethical boundaries and fact audits.

## Related Work & Insights
- **vs LongBioBench / HELMET / MRCR**: These benchmarks focus on long-text understanding within a given context; PolitNuggets shifts the difficulty to active discovery, evidence selection, and open-web synthesis.
- **vs GAIA / BrowseComp / WebSailor**: These tasks emphasize tool use or hard-to-find facts; PolitNuggets focuses on longitudinal, multi-event, structured biography synthesis with multilingual political scenarios.
- **vs Deep Research System Evaluation**: Commercial Deep Research is often a black box and hard to reproduce. PolitNuggets releases code, cached pages, and an LRM evaluation package, enhancing reproducibility.
- **Insights for Retrieval Agents**: Future systems should explicitly optimize query planning, evidence persistence, source diversity, and multilingual routing, rather than merely expanding context windows.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Constructing an evidence-conditional agentic benchmark using political long-tail facts with FactNet dynamic evaluation is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid data scale, model coverage, efficiency statistics, significance testing, LRM baselines, and human audits.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem framing and complete tables; readers need care to follow the numerous model and context conditions.
- Value: ⭐⭐⭐⭐⭐ Directly relevant to agentic search, Deep Research evaluation, multilingual fact discovery, and political information systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[AAAI 2026\] Benchmarking LLMs for Political Science: A United Nations Perspective](../../AAAI2026/llm_evaluation/benchmarking_llms_for_political_science_a_united_nations_perspective.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)

</div>

<!-- RELATED:END -->
