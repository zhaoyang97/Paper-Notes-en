---
title: >-
  [Paper Note] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking
description: >-
  [ICLR 2026][LLM Evaluation][Unindexed Information Seeking] This paper identifies and formalizes the problem of *Unindexed Information Seeking* (UIS)—dynamic web pages, embedded files…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Unindexed Information Seeking"
  - "Multi-Agent Framework"
  - "Dual-Mode Browser"
  - "SFT+RFT Training"
  - "Information Retrieval Benchmark"
date: 2026-05-08
content_hash: be89d29a8f8fe54d
---

# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking

**Conference**: ICLR 2026
**arXiv**: [2603.08117](https://arxiv.org/abs/2603.08117)  
**Code**: [https://huggingface.co/datasets/UIS-Digger/UIS-QA](https://huggingface.co/datasets/UIS-Digger/UIS-QA)  
**Area**: LLM Evaluation
**Keywords**: Unindexed Information Seeking, Multi-Agent Framework, Dual-Mode Browser, SFT+RFT Training, Information Retrieval Benchmark

## TL;DR
This paper identifies and formalizes the problem of *Unindexed Information Seeking* (UIS)—dynamic web pages, embedded files, and interactive content that cannot be directly retrieved by search engines—and proposes the first UIS benchmark UIS-QA (110 questions) along with the multi-agent framework UIS-Digger. A ~30B parameter model trained with SFT+RFT achieves 27.27% accuracy, surpassing systems integrating O3/GPT-4.1.

## Background & Motivation
**Background**: LLM-based information retrieval agents (WebSailor, OWL, DDv2, etc.) have achieved remarkably high scores on GAIA (70.90%) and BrowseComp-zh (46.70%). However, these benchmarks primarily evaluate retrieval of **indexed information** directly accessible via search engines.

**Limitations of Prior Work**: A substantial portion of critical information on the internet constitutes **unindexed information** (UIS): deep pages within government portals, product specifications requiring multiple navigation steps, data embedded in PDF/XLSX files, and dynamic content visible only through interaction with date pickers or filters. Current agents are fundamentally incapable of accessing such information.

**Key Challenge**: Existing evaluation frameworks **do not distinguish** between indexed and unindexed information, causing agent capabilities to be systematically overestimated. State-of-the-art agents suffer a dramatic performance drop from ~70% on GAIA to 24.55% on UIS-QA, exposing two bottlenecks: (a) **insufficient action space**—search-engine-based agents lack web interaction capabilities; and (b) **limited backbone model capacity**—models struggle to make correct decisions within a large action space.

**Key Insight**: UIS is not a peripheral issue but a **fundamental blind spot** in the evaluation of information retrieval agents. The authors formally partition internet information into indexed information $\mathcal{II}$ and unindexed information $\mathcal{UI}$ with mathematical definitions, and introduce the first UIS-QA benchmark and UIS-Digger system.

**Core Idea**: Expose the severity of the UIS problem through the first dedicated benchmark, and address UIS challenges via a multi-agent system combined with domain-specific training.

## Method

### Overall Architecture
UIS-Digger is a **four-agent collaborative system** based on the ReAct paradigm, communicating via request-response messages. Given a user query, it produces a final answer. A Planner decomposes the query into sub-tasks and coordinates three subordinate agents: Web Searcher (indexed information retrieval), Web Surfer (deep webpage browsing), and File Reader (file parsing).

### Key Designs
1. **UIS-QA Benchmark (110 questions)**:

    - **Function**: The first benchmark specifically evaluating agents' ability to acquire unindexed information.
    - **Mechanism**: Expert annotators navigate deep websites → LLM generates QA pairs → triple UIS filtering (manual Google search verification + z.ai automated verification + DeepSeek-R1 internal knowledge check), ensuring answers cannot be obtained directly via search engines.
    - **Design Motivation**: Existing benchmarks (GAIA, BrowseComp) ignore UIS, inflating agent evaluation scores. UIS-QA covers government announcements, product pages, code repositories, games, and corporate annual reports (84 Chinese + 26 English), requiring answers to be objective, authoritative, and temporally stable.

2. **Dual-Mode Browser (Web Surfer)**:

    - **Function**: Dynamically switches between text mode and visual mode to comprehend different types of web content.
    - **Mechanism**: Text mode efficiently handles structured text; visual mode (screenshots) interprets complex UI layouts (date pickers, charts, etc.). Both modes **share memory and browser state**, eliminating synchronization overhead.
    - **Design Motivation**: Pure text agents cannot handle interactive elements requiring visual understanding, while pure visual mode is inefficient. Dynamic switching achieves an optimal balance between functionality and efficiency.
    - **Action space**: click, scroll, type, select dropdown, navigate, submit form, download file, screenshot, etc.

3. **Parallel Tool Execution and File Parsing**:

    - Web Searcher can invoke search engines and crawlers simultaneously.
    - File Reader supports PDF/XLSX/DOCX parsing; oversized files are read incrementally in chunks (following Yu et al., 2025b).

### Loss & Training
Two-stage synthetic data construction and training:
- **Data Construction**: (a) Collect information by deep-browsing 100+ real websites → LLM generates QA pairs → LLM Judge filters; (b) Construct three types of virtual websites (flight booking, statistical query scenarios) to generate targeted training data addressing agent weaknesses such as date pickers, radio buttons, and filters.
- **SFT Stage**: A strong teacher model $\mathcal{X}^*$ (temperature=0) generates one trajectory per question; LLM Judge verifies correctness and non-triviality before reject sampling.
- **RFT Stage**: The SFT model $\mathcal{X}^s$ (temp=0.4, 4 sampled trajectories per question) performs self-sampling with the same reject sampling procedure, **weighted by difficulty**—trajectories from harder questions (fewer correct samples) are preferentially retained, yielding the final model $\mathcal{X}^r$.

## Key Experimental Results

### Main Results

| System | Backbone | UIS-QA | GAIA | BrowseComp-zh |
|--------|----------|--------|------|---------------|
| GPT-5 Direct Inference | GPT-5 | 0.9% | — | — |
| WebSailor | 32B | 7.3% | 53.2% | 25.5% |
| OWL | GPT-4.1 | 25.45% | 70.90% | 46.70% |
| DDv2 | — | 24.55% | — | — |
| **UIS-Digger** | **~30B** | **27.27%** | — | — |

### Ablation Study

| Configuration | UIS-QA Accuracy | Notes |
|---------------|----------------|-------|
| Search only (no browsing) | ~7% | Theoretically unsolvable due to insufficient action space |
| Text mode only | ~20% | Lacks visual mode for dynamic UI |
| Full system (no training) | ~18% | Backbone model cannot effectively utilize tools |
| SFT only | ~23% | Cold-start is effective but insufficiently exploratory |
| **SFT + RFT** | **27.27%** | Difficulty-weighted RFT yields a final +4pp gain |

### Key Findings
- State-of-the-art agents experience a severe performance drop on UIS-QA (GAIA 70% → UIS-QA 25%), confirming that UIS represents an independent and significant challenge.
- A ~30B parameter model with domain-specific training surpasses general-purpose systems integrating O3/GPT-4.1, demonstrating that UIS requires **dedicated optimization**.
- Failure mode analysis: incorrect search strategies 42%, tool usage errors 28%, reasoning errors 30%.
- The dual-mode browser and file parsing capabilities are the key differentiators for UIS problem-solving ability.

## Highlights & Insights
- **First formal treatment of the UIS problem**: The internet information set $\mathcal{P}$ is rigorously partitioned into indexed $\mathcal{II}$ and unindexed $\mathcal{UI}$, with a clear distinction between ideal definitions and practical approximations, laying a theoretical foundation for this previously overlooked direction.
- The **shared-state design** of the dual-mode browsing strategy is particularly elegant—it avoids the mode-switching synchronization issues common in multimodal agents and is transferable to other agents requiring multimodal perception.
- The virtual website data generation strategy is worth adopting: training environments are designed to directly target agent weaknesses (e.g., date picker interaction), replacing expensive real-world annotation with simulation.
- The difficulty-weighted RFT strategy is simple yet effective—correct trajectories for harder questions carry stronger learning signal, and prioritizing their retention more efficiently improves the agent's weaker capabilities.

## Limitations & Future Work
- UIS-QA contains only 110 questions; the scale is limited and 84/110 are in Chinese, restricting language and domain coverage.
- Absolute accuracy of only 27.27% indicates that UIS remains far from solved, requiring stronger backbone models and more complete toolchains.
- Websites requiring login or CAPTCHA are not considered, despite being common in real-world scenarios.
- Evaluation is limited to accuracy, with no analysis of efficiency metrics such as interaction steps or time cost.
- Training data construction relies on a specific teacher model, raising questions about generalizability.

## Related Work & Insights
- **vs. GAIA/BrowseComp**: These benchmarks do not distinguish UIS; high scores may only reflect retrieval capability within the coverage of search engine indices.
- **vs. WebArena/Mind2Web**: These focus on browser operations but evaluate in controlled environments; UIS-QA evaluates on the real open internet.
- **vs. ReAct/Reflexion**: Single-agent action spaces are limited; UIS-Digger's multi-agent architecture covers the complete space of search + browsing + file parsing.
- **Insight**: Agent evaluation must be stratified by information source (indexed vs. unindexed) to faithfully reflect the true capability boundaries of agents.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First identification and formalization of the UIS problem; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-system comparison, but UIS-QA scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear with complete formalization.
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental evaluation blind spot in information retrieval agents and establishes the foundation for UIS research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/llm_evaluation/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ICLR 2026\] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)

</div>

<!-- RELATED:END -->
