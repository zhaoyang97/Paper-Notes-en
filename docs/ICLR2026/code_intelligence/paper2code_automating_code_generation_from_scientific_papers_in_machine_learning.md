---
title: >-
  [Paper Note] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning
description: >-
  [ICLR 2026][Code Intelligence][paper-to-code] This work proposes PaperCoder — a multi-agent LLM framework that automatically converts machine learning papers into executable code repositories via a three-stage pipeline:…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "paper-to-code"
  - "multi-agent framework"
  - "repository-level code generation"
  - "scientific reproducibility"
  - "LLM"
date: 2026-05-08
content_hash: 150097f14c85dfa7
---

# Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning

**Conference**: ICLR 2026
**arXiv**: [2504.17192](https://arxiv.org/abs/2504.17192)  
**Code**: [github.com/going-doer/Paper2Code](https://github.com/going-doer/Paper2Code)  
**Area**: Code Intelligence
**Keywords**: paper-to-code, multi-agent framework, repository-level code generation, scientific reproducibility, LLM

## TL;DR

This work proposes PaperCoder — a multi-agent LLM framework that automatically converts machine learning papers into executable code repositories via a three-stage pipeline: Planning, Analysis, and Coding. 88% of the generated repositories are rated as best by the original paper authors, and the framework substantially outperforms baselines on the PaperBench benchmark.

## Background & Motivation

1. Reproducibility is central to scientific progress, yet only ~19.5% of top-venue papers from 2024 provide available code, forcing researchers to reverse-engineer methods from papers at considerable cost.
2. LLMs have demonstrated strong capabilities in understanding scientific documents and generating high-quality code, but existing scientific workflow automation approaches (e.g., ideation, experiment improvement) typically rely on pre-existing code implementations.
3. Repository-level code generation is far more complex than single-file generation, requiring coordinated architectural design, module structure, and cross-file dependency management.
4. Papers are written to communicate ideas to humans, containing high-level motivation and narrative content that is noisy, loosely structured, and ambiguous from a software engineering perspective.
5. Existing multi-agent code generation frameworks (ChatDev, MetaGPT) adopt bottom-up strategies, incrementally expanding from short requirement descriptions, making them ill-suited to handle lengthy scientific documents.
6. Core challenge: Starting solely from a paper (without code, APIs, or supplementary materials), can a faithful code implementation be generated?

## Method

### Overall Architecture

PaperCoder simulates a typical human developer lifecycle, decomposing the paper-to-code task into three structured stages:

$$\text{Planning: } P = M_{\text{plan}}(R), \quad \text{Analysis: } A = M_{\text{analysis}}(R, P), \quad \text{Coding: } C = M_{\text{code}}(R, P, A)$$

### Key Designs

#### 1. Planning Stage

Comprises four sequential sub-components that progressively transform unstructured paper content into implementation-level abstractions:

- **Overall Plan**: Extracts a high-level summary of core components and functionality (model components, training objectives, data processing, evaluation protocol).
- **Architecture Design**: Defines the repository-level architecture — file list, class diagrams (static structure), and sequence diagrams (dynamic interactions).
- **Logic Design**: Specifies file dependencies and execution order, ensuring that code generation does not fail due to inter-file dependencies (e.g., generating file A before file B when B requires modules from A).
- **Configuration Generation**: Synthesizes a `config.yaml` containing hyperparameters, model settings, and runtime options, enabling researchers to easily adjust experimental configurations.

#### 2. Analysis Stage

For each file $f_i$ identified during planning, a detailed file-level analysis $a_i$ is generated, covering: functional objectives, input/output behavior, intra- and inter-file dependencies, and algorithmic specifications derived from the paper.

#### 3. Coding Stage

Files are generated sequentially in execution order, with each file conditioned on all accumulated context:

$$c_i = \text{LLM}(\mathcal{T}_{\text{code}}(R, P, f_i, a_i, \{c_1, ..., c_{i-1}\}))$$

This ensures that when generating the $i$-th file, the model is fully aware of its dependencies and the current state of the repository.

### Loss & Training

- Non-training framework; a prompt-engineering-based multi-agent system.
- Uses o3-mini-high as the default backbone LLM.
- Supports a Self-Refine verify-and-refine step to further improve output quality at each stage.
- Evaluation: reference-based (when author code is available) + reference-free (when no code exists) + human evaluation (scored by first authors).

## Key Experimental Results

### Main Results

**Paper2CodeBench (90 papers, ICLR/ICML/NeurIPS 2024)**:

| Method | Ref-Based (ICLR) | Ref-Based (ICML) | Ref-Based (NeurIPS) | Ref-Free (ICLR) | Ref-Free (ICML) | Ref-Free (NeurIPS) |
|------|---------|---------|---------|---------|---------|---------|
| ChatDev | 2.70 | 2.97 | 2.96 | 4.00 | 4.12 | 4.01 |
| MetaGPT | 2.48 | 2.75 | 2.95 | 3.52 | 3.63 | 3.59 |
| Paper (one-shot) | 3.08 | 3.28 | 3.22 | 4.15 | 4.30 | 4.08 |
| **PaperCoder** | **3.68** | **3.72** | **3.83** | **4.73** | **4.73** | **4.77** |
| Oracle (author code) | N/A | N/A | N/A | 4.84 | 4.80 | 4.83 |

**PaperBench Code-Dev (20 ICML 2024 papers)**:

| Method | Replication Score (o3-mini) | Replication Score (Claude 3.5) |
|------|---------|---------|
| BasicAgent | 5.1% | 35.4% |
| IterativeAgent | 16.4% | 27.5% |
| **PaperCoder** | **45.14%** | **51.14%** |

### Ablation Study

| Cumulative Components | Ref-Based | Ref-Free |
|----------|-----------|----------|
| Paper only | 3.28 | 4.30 |
| + Overall Plan | 3.40 | 4.34 |
| + Arch. Design | 3.13 (↓) | 4.07 (↓) |
| + Logic Design | 3.60 | 4.50 |
| + Config File | 3.66 | 4.45 |
| + Analysis (Full) | **3.72** | **4.73** |

### Key Findings

1. **PaperCoder approaches author-level quality**: The Ref-Free score (~4.74) shows no statistically significant difference from the Oracle (~4.82).
2. **Advantage of the top-down strategy**: Systematically analyzing the full paper before generation outperforms the bottom-up expansion strategies of ChatDev and MetaGPT.
3. **Logic Design is a critical turning point**: Adding Architecture Design alone decreases performance (execution order ambiguity causes confusion), but performance recovers substantially once Logic Design is incorporated.
4. **Human evaluation consistency**: 88% of PaperCoder-generated repositories are rated best by first authors, and 92% report them as "genuinely helpful."
5. **Executability**: On average, only 0.81% of code lines require modification for successful execution.

## Highlights & Insights

1. This work defines and systematizes the novel "paper-to-code" task and constructs a comprehensive evaluation framework (Paper2CodeBench).
2. The three-stage pipeline elegantly mirrors the human developer workflow of Plan → Analyze → Code, with each stage executed by a specialized agent.
3. The evaluation framework is thorough: reference-based, reference-free, and first-author human evaluation are used in conjunction, with high correlation validated between model-based and human evaluation (r = 0.79).
4. Self-Refine experiments demonstrate that refining early planning outputs propagates quality improvements downstream to code generation (Config File refinement yields the largest gain of +1.00).

## Limitations & Future Work

1. Heavy dependence on backbone LLM capability — open-source models (DS-Coder, Qwen-Coder) perform significantly worse than o3-mini-high, limiting practical accessibility due to API costs.
2. Data processing achieves the lowest coverage — papers typically provide insufficient descriptions of data formats and preprocessing steps, which is the primary source of generation errors.
3. Evaluation is limited to ML papers; generalizability to other scientific domains (physics, biology) remains unknown.
4. Evaluation metrics are primarily LLM-based; while highly correlated with human scores, they do not fully substitute human judgment.

## Related Work & Insights

- **ChatDev / MetaGPT**: Multi-agent code development frameworks employing bottom-up strategies, unsuitable for processing lengthy paper inputs.
- **PaperBench (Starace et al.)**: A concurrent work providing an evaluation benchmark with human-annotated rubrics (20 ICML papers), focused on evaluation rather than methodology.
- **Self-Refine (Madaan et al.)**: The verify-and-refine paradigm, integrated into PaperCoder's planning and analysis stages.
- Insight: Automating paper-to-code conversion can substantially accelerate scientific reproducibility, but requires a top-down methodology of "understand the whole before coding."

## Rating

- ⭐ Novelty: 4/5 — New task definition (Paper2Code) + structured three-stage framework + comprehensive evaluation system
- ⭐ Experimental Thoroughness: 4.5/5 — Automatic evaluation on 90 papers + human evaluation on 21 papers + external validation on PaperBench + extensive ablations
- ⭐ Writing Quality: 4.5/5 — Clear structure, precise formalization, high-quality figures and tables
- ⭐ Value: 4.5/5 — Directly addresses reproducibility pain points in the ML community; code is open-sourced with broad potential impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[NeurIPS 2025\] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](../../NeurIPS2025/code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)

</div>

<!-- RELATED:END -->
