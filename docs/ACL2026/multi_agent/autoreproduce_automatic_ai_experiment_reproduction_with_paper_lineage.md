---
title: >-
  [Paper Note] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage
description: >-
  [ACL 2026][Multi-Agent][Paper Reproduction] AutoReproduce proposes a multi-agent framework that mines implicit domain knowledge from references via a "paper lineage" algorithm to achieve end-to-end automatic reproduction…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Paper Reproduction"
  - "Paper Lineage"
  - "Code Generation"
  - "Scientific Automation"
date: 2026-05-08
content_hash: 29f3a83cd1ad2ac3
---

# AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage

**Conference**: ACL 2026  
**arXiv**: [2505.20662](https://arxiv.org/abs/2505.20662)  
**Code**: [https://github.com/AI9Stars/AutoReproduce](https://github.com/AI9Stars/AutoReproduce)  
**Area**: LLM Evaluation  
**Keywords**: Paper Reproduction, Paper Lineage, Multi-agent, Code Generation, Scientific Automation

## TL;DR

AutoReproduce proposes a multi-agent framework that mines implicit domain knowledge from references via a "paper lineage" algorithm to achieve end-to-end automatic reproduction of paper experiments, achieving a 94.87% code execution rate and only a 19.72% performance gap on the self-constructed ReproduceBench.

## Background & Motivation

**Background**: Reproducing paper experiments is vital for accelerating scientific progress, but as methods become increasingly complex, reproduction requires deep domain expertise and significant labor. LLMs have been used for discrete tasks such as paper analysis, idea generation, and environment configuration, but an end-to-end automatic reproduction framework has yet to emerge.

**Limitations of Prior Work**: (1) Key experimental details are often missing from papers—different research areas rely on vast amounts of implicit knowledge (e.g., specific module architectures, data processing pipelines); (2) Concurrent work such as Paper2Code only generates code without considering executability, making it impossible to verify the correctness of the reproduction; (3) Existing methods do not systematically utilize the domain conventions and implementation practices contained within cited references.

**Key Challenge**: Successful reproduction requires not only understanding the methodological descriptions in the paper itself but also mastering domain-specific regular practices that are not explicitly stated—this "tacit knowledge" is scattered throughout references and related code repositories.

**Goal**: (1) Systematically mine implicit knowledge from references; (2) Build an end-to-end executable code reproduction framework; (3) Establish a reproduction evaluation benchmark that includes execution verification.

**Key Insight**: The authors propose a "Paper Lineage" algorithm to trace references and associated code repositories, using implementation conventions accumulated in historical research as knowledge sources for reproduction.

**Core Idea**: Paper Reproduction = Paper Understanding + Domain Knowledge Mining + Code Generation + Execution Verification. The lineage algorithm compensates for the lack of description in the paper itself through implicit knowledge passed through the citation chain.

## Method

### Overall Architecture

AutoReproduce consists of three stages: (1) Literature Review—a research agent performs a three-level summary of the paper (Overall/Method/Experiment); (2) Paper Lineage—identifies top-k related papers from citations, retrieves their codebases, and extracts key files; (3) Code Development—research and code agents collaborate through data acquisition, method reproduction, and experiment execution to generate executable code.

### Key Designs

1.  **Paper Lineage Algorithm**:
    - **Function**: Mines implicit domain knowledge and implementation conventions from cited references.
    - **Mechanism**: The research agent identifies top-k (default 3) most relevant papers from the source paper's citations, prioritizing baselines from the main experimental sections. Papers are retrieved via the ArXiv API and summarized, while repositories are cloned via the GitHub API. The code agent selectively extracts key source files from the repositories based on paper summaries and task instructions, building $\langle\text{summary, code}\rangle$ tuples as reference examples. Papers without public code only use their summaries as knowledge sources.
    - **Design Motivation**: Scientific research is an incremental process; new methods are built upon existing ones, and the codebases in the citation chain contain implementation standards not explicitly detailed in the paper text.

2.  **Three-stage Code Development**:
    - **Function**: complete reproduction from data processing to method implementation to experimental execution.
    - **Mechanism**: (a) Data Acquisition—distinguishes between standard benchmarks and custom datasets, using mini-batch sampling to infer key data attributes (tensor shape, dtype); (b) Method Reproduction—the code agent generates implementation code, and the research agent verifies it against paper summaries to provide corrective feedback; (c) Experiment Execution—verifies the entire experimental pipeline using an early-exit mechanism for rapid validation. Error diagnosis and code editing are decoupled into two distinct steps.
    - **Design Motivation**: Decoupling error analysis from code modification significantly improves the success rate of debugging.

3.  **Sampling-based Unit Testing**:
    - **Function**: Rapidly verifies the executability of generated code.
    - **Mechanism**: During the method reproduction stage, the code agent infers data flow properties through mini-batch sampling and uses EDIT commands for precise line-level modifications instead of regenerating entire files.
    - **Design Motivation**: Reduces token generation overhead and avoids the instability associated with full-file regeneration.

### Loss & Training

No model training is involved. LLMs such as GPT-4o, Claude-3.5-Sonnet, o3-mini, and Gemini-2.5-Pro are used as the agent backbones.

## Key Experimental Results

### Main Results

**ReproduceBench Evaluation**

| Method | LLM | Align-Score | Exec Rate | Perf Gap (↓) |
| :--- | :--- | :--- | :--- | :--- |
| ChatDev | GPT-4o | 43.33 | 2.56% | 99.62% |
| Agent Lab | GPT-4o | 48.64 | 23.08% | 82.31% |
| PaperCoder | o3-mini | 60.26 | 17.94% | 89.23% |
| **Ours** | GPT-4o | 56.24 | **76.92%** | 41.77% |
| **Ours** | o3-mini | 75.21 | **92.31%** | 24.31% |
| **Ours** | Gemini-2.5-Pro | **77.56** | **94.87%** | **19.72%** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full AutoReproduce | Optimal | Lineage + Three-stage development |
| w/o Paper Lineage | Decrease | Implementation bias due to lack of domain knowledge |
| w/o Unit Tests | Exec Rate Decrease | Missing executability verification |

### Key Findings

- AutoReproduce's code execution rate (94.87%) far exceeds all baselines (max 23.08%), indicating that end-to-end executability verification is crucial.
- The Paper Lineage algorithm is a key contribution—Align-Score and Perf Gap both significantly deteriorate when it is removed.
- Gemini-2.5-Pro performs best as the backbone LLM, but even with GPT-4o, AutoReproduce substantially outperforms PaperCoder.
- A performance gap of 19.72% remains, indicating that high-fidelity fully automated reproduction is still a challenging task.

## Highlights & Insights

- The concept of "Paper Lineage" is highly insightful—transforming the cumulative nature of scientific research into an actionable knowledge mining algorithm.
- The emphasis on end-to-end executability fills a critical gap in existing work (such as Paper2Code)—code that cannot execute provides no value for reproduction.
- The strategy of decoupling error diagnosis from code modification is a significant engineering insight.

## Limitations & Future Work

- ReproduceBench contains only 13 papers, which is a relatively small scale.
- The method depends on citations having public code repositories; otherwise, the lineage algorithm degrades to using only textual knowledge.
- A performance gap of approximately 20% still exists, and high-precision reproduction likely still requires human intervention.
- The framework currently only covers the AI domain; expansion to other disciplines requires additional adaptation.

## Related Work & Insights

- **vs. Paper2Code/PaperCoder**: These methods do not consider code executability; AutoReproduce emphasizes end-to-end execution.
- **vs. Agent Laboratory**: Agent Lab's execution rate is only 23%, while AutoReproduce reaches 95%.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Paper Lineage algorithm and end-to-end reproduction framework are significant innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons across multiple LLMs and baselines, though the benchmark scale is small.
- Writing Quality: ⭐⭐⭐⭐ Clear framework descriptions and intuitive process charts.
- Value: ⭐⭐⭐⭐⭐ Represents a major advancement for the automation of scientific research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)
- [\[ACL 2026\] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)

</div>

<!-- RELATED:END -->
