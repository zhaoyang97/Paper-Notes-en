---
title: >-
  [Paper Note] CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance
description: >-
  [NeurIPS 2025][LLM Evaluation][code assistant] This paper proposes CodeAssistBench (CAB), the first fully automated benchmark for evaluating multi-turn, repository-level programming assistance. CAB automatically constructs 3,286 real-world programming help scenarios from GitHub Issues, spanning 7 languages and 214 repositories, and reveals a substantial performance gap: state-of-the-art models achieve 70–83% on StackOverflow-style questions but only 7–16% on post-cutoff repositories.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - code assistant
  - multi-turn dialogue
  - benchmark
  - programming assistance
  - GitHub Issues
  - repository-level evaluation
  - LLM Agent
date: 2026-05-08
content_hash: 93fe9f6719fd74db
---

# CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance

**Conference**: NeurIPS 2025
**arXiv**: [2507.10646](https://arxiv.org/abs/2507.10646)
**Code**: [amazon-science/CodeAssistBench](https://github.com/amazon-science/CodeAssistBench/)
**Area**: LLM Evaluation
**Keywords**: code assistant, multi-turn dialogue, benchmark, programming assistance, GitHub Issues, repository-level evaluation, LLM Agent

## TL;DR

This paper proposes CodeAssistBench (CAB), the first fully automated benchmark for evaluating multi-turn, repository-level programming assistance. CAB automatically constructs 3,286 real-world programming help scenarios from GitHub Issues, spanning 7 languages and 214 repositories, and reveals a substantial performance gap: state-of-the-art models achieve 70–83% on StackOverflow-style questions but only 7–16% on post-cutoff repositories.

## Background & Motivation

**Background**: Evaluation of LLM-based coding assistants has evolved from isolated code synthesis (HumanEval, MBPP) to repository-level maintenance (SWE-Bench, BigCodeBench). However, existing benchmarks are either single-turn (InfiBench, StackEval) or restrict multi-turn dialogue to code synthesis settings (ConvCodeWorld, MINT, TICODER), leaving a systematic gap in evaluating realistic multi-turn programming assistance scenarios.

**Neglected Real-World Developer Needs**: The 2024 Stack Overflow Developer Survey (34,168 respondents) shows that 77.9% of developers need AI for "searching answers," 77.3% for "debugging and error resolution," and 73.6% for "understanding unfamiliar codebases." These tasks require iterative clarification, environment-aware reasoning, and integration of project-specific details—far beyond the coverage of existing benchmarks.

**Fundamental Limitations of Single-Turn Evaluation**: Real-world programming assistance is inherently multi-turn. For instance, resolving a Docker port-mapping issue requires the assistant to (1) understand the repository's network architecture, (2) explain that proxy ports are internally hardcoded, and (3) reassure the user that no additional mapping is needed—each response shaping the next question. Single-turn correctness metrics cannot capture such reasoning trajectories.

**Scalability Limitations of Manual Curation**: InfiBench and StackEval rely on human-curated StackOverflow data, which is costly and loses discriminative power as LLM training data is updated. An automatically and continuously scalable benchmark construction paradigm is therefore necessary.

**Key Challenge**: Existing benchmarks measure "the ability to generate code given a well-specified requirement," whereas developers truly need "the ability to resolve practical problems under ambiguous descriptions within specific project contexts."

**Key Insight**: The paper leverages closed GitHub Issues labeled *question* or *help-wanted*, along with their multi-turn resolution dialogues, to automatically construct an executable and judgeable multi-turn programming assistance benchmark. A three-agent User–Maintainer–Judge framework is employed to simulate and evaluate the real-world assistance capabilities of LLMs.

## Method

### Overall Architecture

CAB comprises two core components:

- **(1) Automated Dataset Construction Pipeline**: GitHub repository collection → issue filtering and structuring → preparation of three elements per instance (Docker environment + satisfaction conditions + user reply references).
- **(2) Environment-Aware Multi-Agent Evaluation System**: a User Agent poses questions → a Maintainer Agent (the model under evaluation) responds within a containerized environment → upon dialogue termination, a Judge Agent evaluates responses against the satisfaction conditions.

The entire pipeline is end-to-end automated, involving 44,628 Sonnet 3.7 API calls with zero human intervention.

### Key Design 1: Repository Collection and Two-Stage Issue Filtering

- **Function**: Selects high-quality repositories with active communities from the vast GitHub corpus and extracts structured multi-turn help dialogues from them.
- **Mechanism**: Repository eligibility is defined as $R = \{r \in \mathcal{R}_{GH} \mid s(r) > S_{\min},\; t(r) > t_0,\; \ell(r) \in \mathcal{L}\}$, where $s(r)$ is the star count (e.g., threshold of 10), $t(r)$ is the creation date, and $\ell(r)$ is the license type. A community score $CS(r) = Q(r) + H(r)$ (the number of *question* and *help-wanted* issues) is further used to select the top-$N$ repositories. Issue filtering adopts a two-stage architecture: the first stage applies regex rules to remove media-rich content and single-author issues; the second stage employs an LLM classifier to assess seven criteria including resolution status, technical specificity, and safety. At the message level, an LLM removes low-value comments (e.g., "+1," "Thanks").
- **Structured Processing**: Consecutive messages from the same role are merged into logical segments and paired into turn structures $\text{turn}_{i,k} = (m_{i,k}^{\text{author}}, m_{i,k}^{\text{maintainer}})$.
- **Design Motivation**: The two-stage filtering ensures data quality while maintaining full automation—regex rules enable rapid denoising, while LLM-based classification provides precise semantic quality assessment.

### Key Design 2: Three-Element Data Preparation

- **Function**: Prepares an executable environment, evaluation criteria, and user simulation references for each filtered issue.
- **Docker Environment Generation**: Sonnet 3.7 analyzes repository artifacts (README, Dockerfile, GitHub workflows, file structure) and automatically generates and tests Docker build scripts against the commit $sha_i$ closest to the issue creation date, iterating until a successful configuration is found.
- **Satisfaction Condition Extraction**: An LLM extracts a concrete set of resolution criteria $s_i = \{s_{i,1}, \ldots, s_{i,K}\}$ from the complete dialogue, serving as an objective evaluation basis. Human validation yields a precision of 86.3% and a recall of 65.7%.
- **User Reply References**: A BM25 index is constructed over historical maintainer–user message pairs, and the top-$N$ most similar interactions are retrieved per issue as behavioral references for user simulation.
- **Design Motivation**: The three-element design ensures that each benchmark instance is executable (Docker), judgeable (satisfaction conditions), and simulatable (reference replies).

### Key Design 3: Three-Role Multi-Agent Evaluation Framework

- **Function**: Simulates realistic developer–maintainer help interactions to systematically evaluate LLM capabilities for repository-level assistance.
- **User Agent**: Initiates programming queries based on a GitHub issue, evaluates model responses against satisfaction conditions, provides realistic follow-up clarifications or questions, and emits a termination signal upon detecting issue resolution. The agent observes execution results but does not directly operate the environment.
- **Maintainer Agent** (model under evaluation): Analyzes problems, executes commands, generates responses, and adapts strategies based on user feedback within the containerized environment.
- **Judge Agent**: After dialogue termination (user satisfaction or reaching the 10-turn limit), evaluates responses along three dimensions—technical correctness, satisfaction condition completeness, and interaction quality. For issues with a Docker environment, successful execution is treated as a hard requirement.
- **Design Motivation**: This framework provides a more comprehensive assessment of LLM reasoning capability, contextual understanding, and communication quality in multi-turn interactions than single-turn correctness evaluation.

### Loss & Training

- Stratified sampling is employed to avoid computational bottlenecks: at most 5 Docker issues per language and 10 issues per turn-length bucket (1/2/3/4/5+), with underrepresented buckets filled forward.
- Final evaluation sets: 350 All-Time instances + 194 Recent instances.
- Two repository groups: **All-Time** (700 high-star repositories without temporal restriction) and **Recent** (3,500 repositories created after 2024-11-01), the latter serving to test model generalization to knowledge beyond training cutoffs.

## Key Experimental Results

### Model Accuracy and Dialogue Turn Counts

| Model | Recent Accuracy | All-Time Accuracy | Recent Error Rate | Avg. Turns (Correct) | Avg. Turns (Incorrect) |
|-------|:---:|:---:|:---:|:---:|:---:|
| ChatGPT 4.1 Mini | **16.49%** | **29.14%** | 53.09% | 2.94 / 2.35 | 5.70 / 4.28 |
| DeepSeek R1 | 11.34% | 27.14% | 55.15% | 2.82 / 2.24 | 4.50 / 4.28 |
| Sonnet 3.7 (Think) | 13.40% | 27.43% | 59.28% | 2.50 / 2.20 | 4.95 / 4.26 |
| Sonnet 3.7 | 11.34% | 25.71% | 57.73% | 2.36 / 2.30 | 5.71 / 4.21 |
| Llama 3.3 70B | 9.33% | 13.58% | 64.77% | 3.22 / 2.68 | 4.67 / 4.50 |
| Haiku 3.5 | 7.22% | 16.86% | 61.86% | 3.86 / 2.73 | 6.76 / 5.63 |

> Turn counts are reported as Recent / All-Time. Correctly resolved issues average 2–3 turns (consistent with real GitHub dialogues); incorrectly resolved issues require 1–2 additional turns.

### Dataset Construction Statistics

| Metric | Value |
|--------|-------|
| Raw GitHub Issues | 25,656 |
| Issues after filtering | 3,342 |
| Final retained Issues | 3,286 (56 excluded due to Docker build failures) |
| Contributing repositories | 214 / 770 (repositories passing filtering) |
| Programming languages | 7 (Python, Java, C++, C#, JS, TS, C) |
| Successful Docker builds | 238 (Recent 97.6% vs. All-Time 78.2%) |
| LLM API calls | 44,628 (Sonnet 3.7, fully automated, zero human intervention) |

### Key Findings

- **Striking Performance Gap**: Models achieve 70–83% accuracy on StackOverflow-style questions but only 7–16% on CAB's Recent repositories. ChatGPT 4.1 Mini performs best (16.49%) and Haiku 3.5 performs worst (7.22%).
- **Significant Temporal Gap**: Recent repositories yield accuracy 10–15 percentage points lower than All-Time repositories across models. A synthetic ablation study indicates this gap stems primarily from post-cutoff framework/API changes rather than characteristics of AI-generated code (Sonnet 3.7 achieves 74% on synthetic repositories vs. only 11.34% on real Recent ones).
- **Notable Language Differences**: Statically typed languages (C#, C++, Java) generally achieve less than 13% accuracy on the Recent set, while dynamically typed languages (JS, Python) perform relatively better but remain low in absolute terms.
- **Verbosity Tendency**: 40–60% of responses are judged as verbose; Sonnet 3.7 Think exhibits the most balanced output, while Haiku 3.5 and Llama 3.3 are the most verbose.
- **Human Evaluation Validation**: Judge–human agreement is 65.92% (84.2% of the 78.28% inter-human agreement rate; Cohen's κ = 0.68); satisfaction condition precision is 86.3% and recall is 65.7%.

## Highlights & Insights

- **Fully Automated Benchmark Construction Paradigm**: The entire pipeline—from repository collection to Docker environment construction to evaluation judgment—is fully automated (44,628 LLM calls, zero human intervention) and can be continuously updated to track model progress. CAB is not merely a benchmark but a methodology for benchmark construction.
- **Execution Verification > Text-Based Evaluation**: Docker containerization enables evaluation that checks not only whether a response is "semantically correct" but also whether it "works in practice." Issues with Docker environments require actual successful execution, making evaluation more rigorous than pure text matching.
- **Elegant Design of Satisfaction Condition Extraction**: Automatically extracting multiple specific satisfaction conditions from raw issue resolution dialogues (rather than applying binary correct/incorrect labels) supports partial credit and enables fine-grained failure diagnosis.
- **A New Dimension of LLM "Knowledge Hallucination"**: Models do not lack general programming ability; rather, they lack knowledge of *the specific project*. The contrast between 74% on synthetic repositories and 11% on real Recent repositories demonstrates that project context understanding is the core bottleneck.

## Limitations & Future Work

- Satisfaction condition extraction is conservative—precision of 86.3% but recall of only 65.7%, potentially missing critical conditions for issue resolution and resulting in overly lenient evaluation.
- User simulation relies on templates and BM25 retrieval, which cannot fully replicate the complex follow-up strategies and emotional dynamics of real developers.
- Only 7 programming languages are covered; increasingly popular languages such as Rust, Go, and Kotlin are absent.
- Judge LLM–human agreement stands at 65.92% in multi-turn settings, posing greater challenges than single-turn evaluation.
- Docker environment build success rate is only 78.2% for All-Time repositories, causing some issues with Docker requirements to be excluded.
- The evaluation sample is stratified (544/3,286) and does not cover all issues.

## Related Work & Insights

- **vs. SWE-Bench**: SWE-Bench evaluates "code modification ability—writing a patch given an issue," while CAB evaluates "interactive ability—assisting in resolving programming problems through multi-turn dialogue." The latter more closely reflects the everyday use case of AI coding assistants.
- **vs. InfiBench / StackEval**: Both are single-turn StackOverflow QA benchmarks with manual curation; CAB features multi-turn, repository-level assistance that is fully automatically generated and continuously scalable.
- **vs. ConvCodeWorld / MINT**: The former focuses on multi-turn code synthesis under relatively stable environment assumptions; CAB covers a broader range of assistance types—debugging, configuration, comprehension, and troubleshooting—embedded within real Docker environments.
- **Insights**: The recency gap revealed by CAB has important implications for RAG-augmented coding agents: when framework APIs fall beyond the model's training knowledge, effective document retrieval and contextual adaptation become essential.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first multi-turn, repository-level programming assistance benchmark; the fully automated pipeline design is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 6 state-of-the-art models, 7 languages, synthetic ablation, and human evaluation triple validation; relatively comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with a complete structure; the appendix is extremely detailed, including all prompt templates and human annotation protocols.
- Value: ⭐⭐⭐⭐⭐ — Reveals the true capability boundaries of LLMs in programming assistance (project context understanding is the core bottleneck), providing important guidance for coding agent development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](../../ACL2026/llm_evaluation/odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[NeurIPS 2025\] PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations](pfδ_a_benchmark_dataset_for_power_flow_under_load_generation_and_topology_variat.md)
- [\[NeurIPS 2025\] Benchmarking is Broken — Don't Let AI be its Own Judge](benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)

</div>

<!-- RELATED:END -->
