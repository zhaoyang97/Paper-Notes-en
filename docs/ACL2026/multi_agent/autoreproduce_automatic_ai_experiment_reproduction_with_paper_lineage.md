---
title: >-
  [Paper Note] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage
description: >-
  [ACL 2026][Multi-Agent][paper reproduction] AutoReproduce proposes a multi-agent framework that mines implicit domain knowledge from cited references via a "Paper Lineage" algorithm…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "paper reproduction"
  - "paper lineage"
  - "code generation"
  - "research automation"
date: 2026-05-08
content_hash: a34796fa74cd2407
---

# AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage

**Conference**: ACL 2026
**arXiv**: [2505.20662](https://arxiv.org/abs/2505.20662)  
**Code**: [https://github.com/AI9Stars/AutoReproduce](https://github.com/AI9Stars/AutoReproduce)  
**Area**: LLM Evaluation
**Keywords**: paper reproduction, paper lineage, multi-agent, code generation, research automation

## TL;DR

AutoReproduce proposes a multi-agent framework that mines implicit domain knowledge from cited references via a "Paper Lineage" algorithm, enabling end-to-end automatic reproduction of paper experiments. On the self-constructed benchmark ReproduceBench, it achieves a code execution rate of 94.87% with a performance gap of only 19.72%.

## Background & Motivation

**Background**: Reproducing paper experiments is critical for accelerating scientific progress, yet as methods grow increasingly complex, reproduction demands deep domain expertise and substantial human effort. LLMs have been applied to discrete tasks such as paper analysis, idea generation, and environment configuration, but no end-to-end automatic reproduction framework has emerged.

**Limitations of Prior Work**: (1) Papers frequently omit critical experimental details, and different research domains rely heavily on tacit knowledge (e.g., specific module architectures, data processing pipelines). (2) Concurrent work such as Paper2Code generates code without considering executability, making it impossible to verify reproduction correctness. (3) Existing methods do not systematically exploit the domain conventions and implementation practices embedded in cited references.

**Key Challenge**: Successful reproduction requires not only understanding the methodological descriptions in a paper, but also mastering domain conventions that are left unstated—tacit knowledge scattered across cited references and related codebases.

**Goal**: (1) Systematically mine implicit knowledge from cited references. (2) Build an end-to-end executable code reproduction framework. (3) Establish a reproduction evaluation benchmark with execution verification.

**Key Insight**: The proposed "Paper Lineage" algorithm traces cited references and associated codebases, leveraging implementation conventions accumulated in prior work as a knowledge source for reproduction.

**Core Idea**: Paper reproduction = paper understanding + domain knowledge mining + code generation + execution verification. The lineage algorithm compensates for gaps in a paper's own description by propagating tacit knowledge through citation chains.

## Method

### Overall Architecture

AutoReproduce operates in three phases: (1) **Literature Review** — a research agent produces three-level summaries of the target paper (overall / method / experiment); (2) **Paper Lineage** — top-$k$ relevant papers are identified from citations, their codebases are retrieved, and key files are extracted; (3) **Code Development** — a research agent and a code agent collaborate through three steps (data acquisition, method reproduction, experiment execution) to generate executable code.

### Key Designs

1. **Paper Lineage Algorithm**

    - **Function**: Mine implicit domain knowledge and implementation conventions from cited references.
    - **Mechanism**: The research agent identifies the top-$k$ (default 3) most relevant papers from the citations of the source paper, prioritizing comparison baselines in the main experimental sections. Papers are retrieved and summarized via the ArXiv API; codebases are cloned via the GitHub API. The code agent selectively extracts key source files from each codebase based on paper summaries and task descriptions, constructing $\langle\text{summary, code}\rangle$ tuples as reference exemplars. For papers without publicly available code, only their summaries serve as the knowledge source.
    - **Design Motivation**: Research is cumulative; new methods build on prior work, and the codebases in citation chains contain implementation standards not explicitly stated in papers.

2. **Three-Stage Code Development**

    - **Function**: Complete reproduction from data processing through method implementation to experiment execution.
    - **Mechanism**: (a) *Data Acquisition* — distinguishes standard benchmarks from custom datasets; infers key data attributes (tensor shape, dtype) via mini-batch sampling. (b) *Method Reproduction* — the code agent generates implementation code; the research agent cross-checks against paper summaries and provides corrective feedback. (c) *Experiment Execution* — validates the full experimental pipeline using an early-exit mechanism for rapid verification. Error diagnosis and code editing are decoupled into two separate steps.
    - **Design Motivation**: Decoupling error analysis from code modification significantly improves debugging success rates.

3. **Sampling-Based Unit Testing**

    - **Function**: Rapidly verify the executability of generated code.
    - **Mechanism**: During the method reproduction phase, the code agent infers data-flow properties via mini-batch sampling and applies precise line-level modifications using an EDIT command rather than regenerating entire files.
    - **Design Motivation**: Reduces token generation overhead and avoids the instability of full-file regeneration.

### Loss & Training

No model training is involved. GPT-4o, Claude-3.5-Sonnet, o3-mini, and Gemini-2.5-Pro are used as the backbone LLMs for the agents.

## Key Experimental Results

### Main Results

**ReproduceBench Evaluation**

| Method | LLM | Align-Score | Exec Rate | Perf Gap (↓) |
|---|---|---|---|---|
| ChatDev | GPT-4o | 43.33 | 2.56% | 99.62% |
| Agent Lab | GPT-4o | 48.64 | 23.08% | 82.31% |
| PaperCoder | o3-mini | 60.26 | 17.94% | 89.23% |
| AutoReproduce | GPT-4o | 56.24 | **76.92%** | 41.77% |
| AutoReproduce | o3-mini | 75.21 | **92.31%** | 24.31% |
| AutoReproduce | Gemini-2.5-Pro | **77.56** | **94.87%** | **19.72%** |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Full AutoReproduce | Best | Lineage + three-stage development |
| w/o Paper Lineage | Degraded | Implementation deviation due to missing domain knowledge |
| w/o Unit Testing | Exec Rate drops | Loss of executability verification |

### Key Findings

- AutoReproduce's execution rate (94.87%) substantially surpasses all baselines (highest: 23.08%), demonstrating the critical importance of end-to-end executability verification.
- The Paper Lineage algorithm is a key contribution — its removal leads to significant degradation in both Align-Score and Perf Gap.
- Gemini-2.5-Pro performs best as the backbone LLM; however, even with GPT-4o, AutoReproduce greatly outperforms PaperCoder.
- A performance gap of 19.72% remains, indicating that fully automated high-fidelity reproduction remains challenging.

## Highlights & Insights

- The concept of "Paper Lineage" is highly insightful — it operationalizes the cumulative nature of scientific research into an actionable knowledge mining algorithm.
- The emphasis on end-to-end executability addresses a critical gap in prior work (e.g., Paper2Code): code that cannot be executed has no reproduction value.
- The strategy of decoupling error diagnosis from code modification represents an important engineering insight.

## Limitations & Future Work

- ReproduceBench comprises only 13 papers, limiting its scale.
- The method relies on cited papers having publicly available codebases; otherwise, the lineage algorithm degrades to using textual knowledge only.
- A performance gap of approximately 20% remains, and high-fidelity reproduction still requires human involvement.
- Coverage is limited to the AI domain; extension to other disciplines would require additional adaptation.

## Related Work & Insights

- **vs. Paper2Code / PaperCoder**: These methods do not account for code executability, whereas AutoReproduce emphasizes end-to-end execution.
- **vs. Agent Laboratory**: Agent Laboratory achieves an execution rate of only 23%, compared to AutoReproduce's 95%.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Paper Lineage algorithm and the end-to-end reproduction framework are both significant innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-LLM comparisons and broad baseline coverage, though the benchmark scale is small.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly, with intuitive pipeline diagrams.
- **Value**: ⭐⭐⭐⭐⭐ Represents a substantial advancement toward automated scientific research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](../../AAAI2026/multi_agent/hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [\[AAAI 2026\] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](../../AAAI2026/multi_agent/assemble_your_crew_automatic_multi-agent_communication_topol.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)

</div>

<!-- RELATED:END -->
