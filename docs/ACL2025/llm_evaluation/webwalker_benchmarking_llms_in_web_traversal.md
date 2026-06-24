---
title: >-
  [Paper Note] WebWalker: Benchmarking LLMs in Web Traversal
description: >-
  [ACL 2025][LLM Evaluation][Web Traversal] This paper proposes the WebWalkerQA benchmark to evaluate the capabilities of LLMs in deep web traversal for information gathering. It also designs the WebWalker multi-agent framework to mimic human web navigation behaviors via an Explore-Critic paradigm, significantly improving complex QA performance by integrating horizontal and vertical retrieval with RAG.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Web Traversal"
  - "Retrieval-Augmented Generation"
  - "Multi-Agent Framework"
  - "Deep Information Retrieval"
  - "Benchmark Evaluation"
date: 2026-05-08
content_hash: d3930ec7895498ed
---

# WebWalker: Benchmarking LLMs in Web Traversal

**Conference**: ACL 2025  
**arXiv**: [2501.07572](https://arxiv.org/abs/2501.07572)  
**Code**: [https://github.com/Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)  
**Area**: LLM Evaluation  
**Keywords**: Web Traversal, Retrieval-Augmented Generation, Multi-Agent Framework, Deep Information Retrieval, Benchmark Evaluation

## TL;DR

This paper proposes the WebWalkerQA benchmark to evaluate the capabilities of LLMs in deep web traversal for information gathering. It also designs the WebWalker multi-agent framework to mimic human web navigation behaviors via an Explore-Critic paradigm, significantly improving complex QA performance by integrating horizontal and vertical retrieval with RAG.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has demonstrated outstanding performance in open-domain question answering. Existing methods typically rely on page snippets returned by search engines as knowledge sources, which LLMs use to generate answers.

**Limitations of Prior Work**: Traditional search engines can only retrieve shallow web content (such as homepage snippets) and fail to traverse deep subpages of websites to obtain more detailed and deeper information. When questions involve multi-level, cross-subpage complex information, relying solely on shallow search engine results severely limits the answer quality of LLMs.

**Key Challenge**: Web information is organized hierarchically—a website often contains multi-level structures such as homepages, subpages, and nested subpages. Existing RAG systems formulate responses after viewing only the "first layer," which is analogous to writing a report after reading only the table of contents. This leads to incomplete or even incorrect answers in scenarios requiring deep information.

**Goal**: (1) To construct a benchmark dataset specifically designed to evaluate the web traversal capabilities of LLMs; (2) to design a multi-agent framework that allows LLMs to navigate among subpages of websites like humans, systematically extracting deep-level information.

**Key Insight**: When humans browse a website, they navigate between different subpages based on their needs—first browsing first-level pages to understand the overall framework, and then diving into specific subpages for details. This "exploration-judgment" navigation pattern can be modeled as a collaborative process among multiple agents.

**Core Idea**: Utilizing an Explorer Agent to navigate websites and extract information, and a Critic Agent to evaluate whether the collected information is sufficient to answer the question. The iterative collaboration between these two agents enables human-like deep web traversal.

## Method

### Overall Architecture

WebWalker adopts a multi-agent architecture. Given a question and a starting URL, the system first accesses the webpage and parses the content using the Explorer Agent to extract useful information and potential navigation links. Subsequently, the Critic Agent evaluates whether the collected information is sufficient to decide whether to continue exploration or to stop and generate the final answer. The entire process systematically covers website information through both "horizontal traversal" (navigating between sibling subpages) and "vertical depth-first" (following links to deeper subpages).

### Key Designs

1. **Explorer Agent**:

    - **Function**: Responsible for webpage content parsing and navigation decision-making.
    - **Mechanism**: The Explorer receives the current webpage content and the question to perform two tasks: extracting question-related snippets from the current page and identifying hyperlinks that might contain further relevant information. It maintains a list of "visited pages" to avoid redundant visits, ranks hyperlinks based on relevance, and selects the next page to visit. After each new page visit, the extracted information is accumulated into an information pool.
    - **Design Motivation**: To mimic human information extraction and navigation behaviors while browsing the web. Instead of aimlessly traversing all links, it purposefully selects paths most likely to contain relevant information.

2. **Critic Agent**:

    - **Function**: Responsible for evaluating the sufficiency of the collected information and determining the exploration-stopping strategy.
    - **Mechanism**: At the end of each exploration round, the Critic receives the currently accumulated information pool and the original question to judge whether the information is sufficient to answer the question. If sufficient, it triggers a stop signal and forwards the information to the answer generation module. If not, it provides feedback to the Explorer regarding what additional info is needed, guiding the next round of exploration. This forms an iterative closed loop until either the information is sufficient or the maximum traversal depth is reached.
    - **Design Motivation**: To prevent excessive exploration from wasting computational resources while avoiding hasty answers based on insufficient information. The guiding mechanism of the Critic makes the exploration process more efficient, preventing the Explorer from wasting exploration steps on irrelevant pages.

3. **Integrated Horizontal-Vertical Traversal Strategy**:

    - **Function**: Combining breadth-first traversal on the same hierarchy and depth-first traversal across levels to achieve comprehensive information coverage.
    - **Mechanism**: Horizontal traversal refers to browsing multiple parallel subpages at the same hierarchy level of a website (such as different product pages in a product list), while vertical traversal refers to diving deeper into nested subpages along hyperlinks (such as going from a product list to a specific product detail page, and then to reviews). The system automatically adjusts the ratio of these two strategies based on the question type.
    - **Design Motivation**: Different types of questions require different navigation strategies—comparison-type questions require more horizontal traversal, whereas deep-understanding questions demand more vertical traversal. This flexible integration strategy enables WebWalker to adapt to diverse information needs.

### Loss & Training

As a framework-level method, WebWalker does not involve model training. Both the Explorer and Critic are implemented utilizing existing LLMs (e.g., GPT-4, Qwen) in a prompt-driven manner, with the core lying in the prompt design and the multi-agent collaboration workflow. The construction of the WebWalkerQA benchmark, however, requires manual annotation, where annotators navigate real websites and label answerable question-answer pairs.

## Key Experimental Results

### Main Results

| Method | Data Source | EM | F1 | Acc |
|------|---------|-----|-----|-----|
| Direct RAG (Shallow Retrieval) | Search Engine Homepage | 18.2 | 32.5 | 24.1 |
| BM25 + RAG | Search Engine Pages | 21.3 | 35.8 | 27.0 |
| WebWalker + RAG (GPT-4o) | Deep Subpages | 33.5 | 48.7 | 39.2 |
| WebWalker + RAG (Qwen-Max) | Deep Subpages | 29.1 | 43.2 | 35.8 |
| Human Performance | - | 72.8 | 85.3 | 78.6 |

### Ablation Study

| Configuration | EM | F1 | Description |
|------|-----|-----|------|
| Full WebWalker | 33.5 | 48.7 | Full model |
| w/o Critic Agent | 27.8 | 41.3 | Exploration efficiency drops without the Critic Agent |
| w/o Vertical Traversal | 28.9 | 42.1 | Only horizontal traversal; deep information is missed |
| w/o Horizontal Traversal | 30.2 | 44.5 | Only vertical traversal; coverage is insufficient |
| Fixed Traversal Depth = 2 | 31.1 | 45.6 | The impact of limiting the maximum depth |

### Key Findings

- The introduction of the Critic Agent contributes the most (+5.7 EM), demonstrating that targeted exploration is much more effective than blind traversal.
- Both horizontal and vertical traversals are indispensable, but vertical traversal has a larger impact on overall performance since many answers are hidden within deep subpages.
- The WebWalkerQA benchmark is highly challenging: even WebWalker powered by GPT-4o exhibits a huge gap compared to human performance (a difference of 39.3 in EM), indicating substantial room for improvement in deep web information retrieval.
- Performance varies significantly when using different LLMs as backbones, showing that web navigation capability is highly dependent on the instruction-following and reasoning abilities of the models.

## Highlights & Insights

- The **Explore-Critic paradigm** is highly intuitive and effective—decomposing information gathering into two distinct roles of "exploration" and "criticism," and gradually approaching a comprehensive answer through iterative feedback. This design can be transferred to any scenario requiring multi-step information gathering (e.g., document analysis, code repository understanding).
- **WebWalkerQA benchmark fills the gap**: There was previously a lack of datasets specifically designed to evaluate the web traversal capabilities of LLMs. This benchmark is built upon real websites, and the questions require deep-level navigation to be answered.
- Inspiration from the integrated horizontal-vertical strategy: Complex information retrieval should not search along only one dimension but should flexibly adjust the search strategy based on task requirements.

## Limitations & Future Work

- The current evaluation only covers English websites, with web structures and language diversity not yet fully considered.
- The Explorer relies on HTML parsing to retrieve links and content, which might fail to parse correctly on highly dynamic (JS-rendered) webpages.
- Traversal efficiency still has room for improvement—each page visit requires an LLM inference call, resulting in significant computational overhead when traversal depth is large.
- The human-machine gap remains massive (~40 points difference in EM), indicating that the capabilities of current LLMs on complex web navigation tasks are far from mature.

## Related Work & Insights

- **vs WebGPT**: WebGPT also attempts to let LLMs browse the web, but it focuses more on search engine interactions rather than deep website traversal. The key difference of WebWalker lies in its emphasis on the systematic traversal of multi-level subpages.
- **vs ReAct Agent**: ReAct is a general reasoning-acting framework. WebWalker can be viewed as a concrete instantiation of ReAct in the web traversal scenario, with an added Critic mechanism to control the exploration process.
- **vs MRAG / Auto-RAG**: These methods focus on multi-turn retrieval in RAG systems, but their information sources are still limited to search engine results. In contrast, WebWalker's information source is the complete subpage structure of a website.

## Rating

- Novelty: ⭐⭐⭐⭐ Deep web traversal is an important but overlooked direction, and the Explore-Critic paradigm design is highly reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ The WebWalkerQA benchmark is solidly constructed, and the comparison across multiple models and configurations is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive methodology description.
- Value: ⭐⭐⭐⭐ The benchmark dataset and framework hold significant reference value for subsequent Web Agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)
- [\[ACL 2025\] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)

</div>

<!-- RELATED:END -->
