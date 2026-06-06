---
title: >-
  [Paper Note] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review
description: >-
  [ACL 2026][LLM Agent][Content-grounded retrieval] IntrAgent decomposes the researcher's process of "reading papers to find information" into a two-stage pipeline: "ranking sections by structure…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Content-grounded retrieval"
  - "section ranking"
  - "iterative reading"
  - "sufficiency check"
  - "scientific QA"
date: 2026-05-08
content_hash: 651143dc06f2e0f1
---

# IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review

**Conference**: ACL 2026  
**arXiv**: [2604.22861](https://arxiv.org/abs/2604.22861)  
**Code**: https://github.com/FengboMa/IntrAgent (Available)  
**Area**: LLM Agent / Scientific Literature Retrieval  
**Keywords**: Content-grounded retrieval, section ranking, iterative reading, sufficiency check, scientific QA

## TL;DR
IntrAgent decomposes the researcher's process of "reading papers to find information" into a two-stage pipeline: "ranking sections by structure, then iteratively reading sections and stopping when sufficient." This allows the LLM to extract fine-grained answers faithfully aligned with queries from an entire scientific paper without relying on vector retrieval, outperforming RAG/scientific agent baselines by an average of 13.2% across five STEM domains on the new IntraBench benchmark.

## Background & Motivation
**Background**: In scientific research, accurately extracting fine-grained information such as "experimental setups, parameters, and conclusions" from papers is a high-frequency demand. Current mainstream solutions either feed the entire paper directly into an LLM (suffering from context length limits and information drowning) or use RAG to partition the paper into 500-token chunks and retrieve several blocks based on semantic similarity to concatenate into a prompt.

**Limitations of Prior Work**: RAG's "chunking + cosine similarity" completely ignores the hierarchical section structure of scientific papers—a query for "excitation laser wavelength" hardly matches the section title "Experiment Setup-SERS measurement" terminologically. Retrieved chunks are often paragraphs from the Introduction containing the word "laser," and feeding irrelevant context to the LLM decreases accuracy. Existing "scientific agents" like PaperQA2 / SciMaster, originally designed for web-search QA, degrade into RAG when applied to "single paper QA" scenarios, suffering similar drawbacks.

**Key Challenge**: Scientific papers have an explicit prior mapping of "problem $\rightarrow$ corresponding section," but the flat retrieve-then-generate architecture of RAG flattens this hierarchical prior. Meanwhile, relevant evidence may span multiple non-adjacent sections, making it difficult to "read enough" in a single retrieval, yet traditional pipelines lack an explicit stop criterion, often leading to premature termination and hallucinations.

**Goal**: (1) Formally define a new task, IntraView—given a full paper $C$ and a query $Q$, output a short answer strictly grounded in $C$; (2) Design an agent capable of utilizing hierarchical sections and judging "information sufficiency"; (3) Provide a cross-disciplinary benchmark for fair evaluation.

**Key Insight**: Mimic human literature reading behavior—scan the table of contents to locate sections most likely to contain the answer, read details section by section, and determine whether "information is sufficient" while reading, stopping once the answer is found.

**Core Idea**: Replace "semantic similarity chunking and retrieval" with "structure-aware section ranking + iterative reading with sufficiency checks," explicitly incorporating human reading strategies into the agent's action space.

## Method

### Overall Architecture
The input is a PDF paper $C$ and a research query $Q$, and the output is a brief answer $A$ faithful to $C$. The entire pipeline is divided into two main stages:

1. **Section Ranking**: First, minerU is used to convert the PDF into Markdown $C'$ with title markers. The LLM then infers the title hierarchy tree to construct a deduplicated title set $H=\{h_1,\dots,h_n\}$. Finally, the LLM performs reasoning-based ranking on $H$ based on $Q$, resulting in a reordered section sequence $C_R=[(h_{r_1},t_{r_1}),\dots]$.
2. **Iterative Reading**: Following the order of $C_R$, the agent executes a loop: "fetch next section $\rightarrow$ extract details $D_i \rightarrow$ check if accumulated $\{D_1,\dots,D_i\}$ is sufficient to answer $Q$." If YES, it stops; if NO, it continues. Finally, $A=\mathrm{LLM}(D_1,\dots,D_m,Q)$.

All intermediate LLM steps include detailed prompts (Appendix E). Section parsing and hierarchy trees are managed by deterministic Python code, while critical semantic judgments (ranking, extraction, sufficiency, synthesis) are handled by the LLM, ensuring a clear division of labor: "code manages structure, LLM manages understanding."

### Key Designs

1. **Hierarchy Preservation**:
    - **Function**: Restores a flat list of Markdown titles $H_0$ into a section tree, allowing subsequent ranking to perceive "parent-child" semantic relationships.
    - **Mechanism**: First, all titles starting with `#` are captured from minerU output as the initial set $H_0$. The LLM infers the parent node for each title, returning a set of paths from root to node. Deterministic rules then remove redundant paths where "no body text exists between parent and child," resulting in the refined set $H$. This step explicitly builds parent-child relationships like "Experiment Setup $\rightarrow$ SERS measurement," which otherwise would appear as an isolated "SERS measurement" during ranking.
    - **Design Motivation**: Flat ranking often misplaces sections that are semantically close but hierarchically distant. Removing HP caused accuracy to drop from 65.6/72.5/67.6 to 60.7/70.3/64.2 across GPT-4o/4.1/DS-R1 models, proving that hierarchical priors indeed assist the model in understanding "which major topic this section belongs to."

2. **Reasoning-Based Section Ranking**:
    - **Function**: Given $H$ and $Q$, the LLM outputs a relevance ranking $R=[r_1,\dots,r_n]$, which is used to reorder the sections into $C_R$.
    - **Mechanism**: Instead of calculating cosine similarity, the LLM is prompted: "which sections are most likely to contain the answer to $Q$? Rank them from highest to lowest probability and provide reasons." Examples show the model reasoning that "laser wavelength usually belongs to experimental settings, so Rank 1 is Experiment Setup-SERS measurement"—domain common sense that embedding retrieval cannot capture.
    - **Design Motivation**: The "problem $\rightarrow$ section" mapping in scientific papers relies heavily on domain knowledge rather than surface-level lexical similarity. Entrusting ranking to LLM reasoning instead of vector retrieval is the most critical step in addressing RAG's weaknesses.

3. **Information Sufficiency Check**:
    - **Function**: After reading each section, the LLM judges whether the current accumulated $\{D_1,\dots,D_i\}$ is sufficient to answer $Q$, outputting YES/NO. If NO, it reads the next section; if YES, it terminates and synthesizes the answer.
    - **Mechanism**: The prompt explicitly forbids guessing and requires the LLM to judge based only on extracted sentences. Three confidence levels (Conservative / Balanced / Aggressive) are introduced to allow users to trade off between "reading more for stability" and "reading less for efficiency." This mechanism solves two problems: (a) forcing continued reading when evidence spans sections, and (b) stopping immediately when evidence has clearly appeared to avoid diluting signals with irrelevant context.
    - **Design Motivation**: This was the most dramatic step in the ablation study—disabling Sufficiency Check and only reading the Top-1 section caused GPT-4o's accuracy on the Physics subset to plummet from 75.4% to 32.2%, nearly RAG levels, proving that "stopping only when enough is read" is the soul of an agent compared to one-shot retrieval pipelines.

### Loss & Training
IntrAgent is a training-free agent framework; all "decisions" rely on prompts and reasoning without any parameter updates. The only tunable components are the confidence level settings (affecting average iterations: Conservative $\approx$ 9.9 steps, Balanced $\approx$ 5.1 steps, Aggressive $\approx$ 3.9 steps) and the choice of the backbone LLM.

## Key Experimental Results

### Main Results
Evaluation Benchmark: IntraBench, covering 5 STEM domains (Physics SERS / Public Health Infectious Disease Modeling / Earth Science Remote Sensing / Engineering Human Factors / Material Additive Manufacturing) $\times$ 25 expert-selected papers $\times$ 63 expert-authored questions = 315 test instances. Scoring is based on multiple-choice questions, using GPT-4.1 to map free-form generated short answers to options to calculate accuracy.

| Backbone LLM | Prev. SOTA | IntrAgent | Gain |
|---|---|---|---|
| GPT-4o | 62.1 (LongRAG) | 70.0 | +7.9 |
| GPT-4.1 | 64.7 (LongRAG) | 75.8 | +11.1 |
| DeepSeek-R1 | 65.5 (LongRAG) | 74.4 | +8.9 |
| o3 | 60.4 (Vanilla RAG MiniLM) | 73.4 | +13.0 |
| o4-mini | 61.5 (Vanilla RAG MiniLM) | 73.8 | +12.3 |
| Gemini-2.5 Pro | 61.8 (Vanilla RAG MiniLM) | 75.9 | +14.1 |
| Llama-3.1-70B | 61.4 (Vanilla RAG GritLM) | 68.8 | +7.4 |
| **Average** | — | — | **+13.2** |

IntrAgent outperformed all baselines across all 7 backbones. The two strongest models, GPT-4.1 and Gemini-2.5 Pro, both reached approximately 75.8/75.9, suggesting model capacity is no longer the bottleneck.

### Ablation Study

| Configuration | GPT-4o | GPT-4.1 | DeepSeek-R1 | Description |
|---|---|---|---|---|
| Full (w/ Hierarchy Preservation) | 65.6 | 72.5 | 67.6 | Complete design |
| w/o Hierarchy Preservation | 60.7 | 70.3 | 64.2 | Ranking flat titles; drops 2.2–4.9 points |
| w/o Sufficiency Check (Top-1 only) | 32.2 | — | — | Physics subset; GPT-4o drops from 75.4 to 32.2 |

Confidence Levels (GPT-4o, Physics subset):

| Confidence | Accuracy | Avg. Iterations | Median Tokens |
|---|---|---|---|
| Conservative | 58.9 | 9.9 | 7853 |
| Balanced (Default) | 68.3 | 5.1 | 6376 |
| Aggressive | 62.7 | 3.9 | 2233 |

### Key Findings
- **Sufficiency check is the lifeblood**: Dropping it to 32.2% accuracy shows that simple "reasoning ranking + reading Top-1" is just another form of RAG; only "stopping when sufficient" elevates IntrAgent to 75%+.
- **Reading more is not necessarily more accurate**: The Conservative mode read nearly 10 sections but scored nearly 10 points lower than Balanced, consistent with the LongRAG finding that "overly long context dilutes signals." This serves as a counter-example to "the more aggressive/thorough the agent, the better"—Balanced is the sweet spot.
- **Model Agnosticism**: IntrAgent outperformed the strongest baselines across all 7 backbones, indicating performance gains stem from architectural design rather than model strength.
- **Title Robustness**: Changing section titles to "child-friendly/noisy/Shakespearean" versions only dropped GPT-4o accuracy from 89.2 to 84.6, showing structural reasoning is insensitive to title phrasing.
- **Evaluation Stability**: On 65 physics questions, the automatic mapping of GPT-4.1 matched human labels for 63/65 cases, providing strong evidence for the reliability of the evaluation protocol.

## Highlights & Insights
- **Explicit Structural Priors**: Writing the explicit prior "scientific paper = title tree" into the agent workflow—rather than hoping for implicit encoding in vector space—is a direct challenge to the flat RAG paradigm and valuable for all "structured document QA" tasks (e.g., contracts, manuals, technical reports).
- **Sufficiency Check = Retrieval with a Stop Token**: Changing "how much to read" from a hard-coded top-k to an LLM-evaluated open loop effectively installs an adaptive stop criterion for retrieval; this idea could be integrated back into RAG as "iterative-RAG with a sufficiency gate."
- **Productized Confidence Design**: Exposing "how many sections to read" as a user-controlled knob is more controllable in engineering than letting the model decide autonomously; it is friendly for cost-quality trade-offs in production services.
- **Filling the Benchmark Gap**: Existing benchmarks like LitQA only cover biology. IntraBench covers five STEM domains with 315 questions, emphasizing that "answers must be grounded in the paper, no extrapolation allowed," setting a new standard for evaluating serious scientific assistants.

## Limitations & Future Work
- Non-text modalities like charts and formulas are not yet handled, even though they often carry critical experimental trends. The coverage is limited to original papers and excludes review articles.
- Section parsing relies on minerU; it may fail for scanned or oddly formatted older papers, propagating errors to the hierarchy tree.
- Sufficiency Check relies entirely on LLM self-assessment. Confidence levels act like prompt-engineered "temperature knobs" rather than theoretical guarantees, potentially leading to premature stops when evidence is scattered.
- The evaluation uses multiple-choice mapping, which may mask the actual quality of free-form generation (e.g., verbosity or hallucinations).
- Future Work: Implementing hard thresholds for sufficiency checks based on uncertainty metrics (e.g., self-consistency or token logprob); introducing multimodal capabilities after multi-modal OCR; and using rerankers to accelerate the retrieval portion for fewer API calls.

## Related Work & Insights
- **vs LongRAG**: LongRAG fills more chunks into a long context; ours uses structure and iteration to actively select chunks. IntrAgent outdistances LongRAG by 6–14 points, verifying that "selecting accurately is more important than filling more."
- **vs PaperQA2 / SciMaster**: These agents rely on web search for scientific QA and degrade to RAG without it. Ours proves that "agents designed specifically for grounded retrieval" far outperfrom "general agents" adapted to the task.
- **vs DRAGIN / R²AG**: While these also improve RAG via dynamic retrieval or reranking, IntrAgent elevates reasoning to the "section level," which closer mimics human reading strategies and avoids the difficulty of semantic alignment at the chunk granularity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining "hierarchical ranking + sufficiency check" for scientific literature QA is a first, though individual components have precursors in RAG literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covered 7 backbones $\times$ 5 domains $\times$ 12+ baselines, plus three ablation studies, title robustness, and evaluation protocol verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive case figures, though some terminology (e.g., "mindset bionics") feels slightly contrived.
- **Value**: ⭐⭐⭐⭐ Provides a new task, benchmark, and method simultaneously. It has clear directional significance for industrial scientific assistants, and IntraBench serves as a valuable community asset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ACL 2026\] SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](safemcp_proactive_power_regulation_for_llm_agent_defense_via_environment-grounde.md)

</div>

<!-- RELATED:END -->
