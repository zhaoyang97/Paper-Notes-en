---
title: >-
  [Paper Note] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale
description: >-
  [ICML 2026][Code Intelligence][SWE Agent] The authors developed a language-agnostic unified construction pipeline, an interactive installation agent…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "SWE Agent"
  - "Executable Training Environments"
  - "Multilingual"
  - "Automated Data Pipeline"
  - "Issue Quality Filtering"
date: 2026-05-08
content_hash: aefd6b236d599312
---

# SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale

**Conference**: ICML 2026  
**arXiv**: [2602.23866](https://arxiv.org/abs/2602.23866)  
**Code**: HuggingFace `nebius/SWE-rebench-V2` and `nebius/SWE-rebench-V2-PRs`  
**Area**: Code Intelligence / SWE Agent Training Data / Multilingual Code Benchmarks  
**Keywords**: SWE Agent, Executable Training Environments, Multilingual, Automated Data Pipeline, Issue Quality Filtering

## TL;DR
The authors developed a language-agnostic unified construction pipeline, an interactive installation agent, and a three-model ensemble for issue clarity filtering to automatically mine 32,079 executable SWE tasks across 20 languages and 3,617 repositories from GitHub (supplemented by 120k+ PR-derived tasks). Each task includes pre-built Docker images, fail-to-pass (F2P) tests, and instance-level diagnostic metadata, providing a stable, training-oriented foundation for large-scale reinforcement learning of SWE Agents rather than just evaluation.

## Background & Motivation

**Background**: "Repo-level issue solving" benchmarks, exemplified by SWE-bench, have become the mainstream evaluation protocol for SWE Agents. Reinforcement learning (RL) using test passage as a reward signal is the primary driver for advancing agent capabilities. Recently, efforts like SWE-Gym, Multi-SWE-RL, SWE-Factory, SetUpAgent, and SWE-Bench++ have attempted to automate task collection and environment setup.

**Limitations of Prior Work**: Executable tasks suitable for training (not just evaluation) remain scarce. Manually annotated benchmarks are too small and heavily biased toward Python. While automated pipelines have increased the scale, most remain "evaluation-first"—lacking pre-built images, stable cross-language F2P signals, and diagnostic information to align descriptions with tests. This leads to high reward noise and difficulty in curriculum design during RL training.

**Key Challenge**: To ensure stable RL, one needs (i) reproducible dependency installation, (ii) deterministic test execution, and (iii) consistency between natural language specifications and test oracles. Achieving these across languages is prohibitively expensive, as each ecosystem has different build systems, package managers, and test runners, making manual per-repo configuration unscalable.

**Goal**: Construct a language-agnostic end-to-end pipeline where a single workflow handles 20 languages. By using a few reusable language templates (base images, runners, log parsers), the objective is to produce large-scale, reproducible, and diagnostically labeled executable SWE training tasks.

**Key Insight**: Breakdown the entire process into five stages (preliminary mining → interactive setup synthesis → dual-pass execution validation → LLM-integrated issue clarity filtering → metadata enrichment). Quantify the "yield vs. failure modes" at each stage to embed quality control directly into the construction pipeline.

**Core Idea**: Replace "per-instance manual verification" with an "interactive setup agent + 3-LLM judge ensemble + instance-level diagnostic labels" to push data scale to the magnitudes required for training while maintaining executability.

## Method

### Overall Architecture
The pipeline funnels 29.5 million raw PRs down to 32,000 stable executable tasks. The five cereal stages are as follows:

1. **Preliminary Data Collection**: Aggregate issue/PR metadata from GitHub Archive, perform distributed cloning, and extract diffs from local git history. Filter through three layers: license, issue-PR linking, and presence of new/modified tests. High-resource languages (Python/Java/Go) require 25 stars + 15 closed issues; long-tail languages are relaxed to 10 stars + 1 closed issue. Approximately 21,700 repositories and 580,000 candidate tasks are retained.
2. **Setup Synthesis**: Perform installation and test script inference only **once** per repository (using the snapshot of the latest mined PR). This inference is then reused across all tasks, shifting from "per-task manual" to "per-repo automated" effort. Base Dockerfiles for each language are generated using Qwen3-Coder-480B, followed by an interactive agent loop (mini-SWE-agent v1.14.4 + Qwen3-Coder-480B) for closed-loop debugging.
3. **Execution-based Validation**: Use multi-stage Docker builds to separate base and repository layers. Run the full test suite with the test patch first, then re-run with the solution patch to obtain F2P pairs. Each task is run 3 times; it is only retained if the structured test results remain consistent to filter out flaky tests.
4. **Filtering by Issue Clarity**: Use three independent LLM judges (gpt-oss-120b, GLM-4.7, DeepSeek-V3.2) to score the issue text on whether it is "well-specified." High-confidence tasks are retained only when **all three models agree**, using SWE-bench Verified manual annotations as a calibration anchor.
5. **Metadata Enrichment + PR-Derived Expansion**: Define diagnostic labels (B1=test suite coupling, B2=implicit naming requirements, B3=external dependencies) based on failure mode analysis of seven frontier models on 300 tasks. Additionally, release 120k+ tasks where PR descriptions generate problem statements as a larger-scale but lower-confidence training resource.

### Key Designs

1. **Interactive Setup Agent (Language-Agnostic Setup Synthesizer)**:
    - **Function**: On top of pre-built base images for each language, an LLM agent automatically produces reproducible install/test scripts and log parsers through a "code exploration → run install → observe errors → fix scripts" loop.
    - **Mechanism**: Utilizes the mini-SWE-agent framework with Qwen3-Coder-480B. Structured reports (e.g., JUnit XML for JVM) are mandated to avoid stdout drift. Compiled languages like C/C++ require mandatory rebuild steps after patching. The log parser is bootstrapped by sampling a successful run and verifying it across other traces. Interactive pass@1 is $25.8\%$ vs. $12.1\%$ non-interactive. Even a smaller Qwen3-30B in interactive mode ($17.4\%$ pass@1) outperforms Qwen3-480B in non-interactive mode ($12.1\%$).
    - **Design Motivation**: Simple "analyze file list → generate command" approaches work for Python but fail for 20 languages and 3,600 repos. Trial-and-error loops are essential for long-tail build systems. Per-repo synthesis limits costs to approximately $\$1.9$K USD for 535K API calls.

2. **Multi-LLM Ensemble Issue Clarity Filter**:
    - **Function**: For tasks that pass execution validation, an LLM judge determines if the issue text is "self-contained and sufficient for implementation" to avoid polluting RL reward signals with poorly defined tasks.
    - **Mechanism**: Calibrated against 1,699 manual "well-specified" annotations from SWE-bench Verified. The ensemble uses a "consensus" strategy. While the three-model average gives the best F1 ($0.43$), the three-model consensus yields the best precision ($0.88$, although recall is only $0.06$). To protect downstream training purity, the **Verified-E prompt + three-model consensus** was selected.
    - **Design Motivation**: Single model judges often mistake "verbose but unclear" issues for qualified ones. Consensus acts as a cheap proxy for manual double-blind review. Execution validation precedes issue filtering because only F2P tasks are worth the token cost of quality assessment.

3. **Failure-Driven Instance-Level Diagnostic Metadata (A vs B\* Stratification)**:
    - **Function**: Tag tasks as "clean (A)" vs. "known deficiencies (B1/B2/B3)" to allow downstream users to select samples for curriculum learning (e.g., SFT on A, RL on B1 with partial rewards).
    - **Mechanism**: After analyzing trajectories from 7 frontier models across 300 tasks, three systematic failure modes were categorized. gpt-oss-120b with meta-prompting was then used to label the entire dataset. Validation showed that models achieve $5$–$8\times$ higher pass@3 on A tasks compared to B\* tasks (e.g., Gemini $34.0\%$ vs. $4.0\%$).
    - **Design Motivation**: Automated pipelines inevitably contain noise. Explicitly labeling defects allows for stratified filtering, balancing data scale and quality better than simply discarding all suspicious tasks.

## Key Experimental Results

### Main Results

| Stage | Input PRs | Output PRs | Output Repos | Description |
|------|---------|---------|---------|------|
| Raw | $2.95\text{e}7$ | — | $1.45\text{e}5$ | Global GitHub Archive |
| w/ Tests | $8.59\text{e}6$ | — | $1.02\text{e}5$ | Must add/modify tests |
| Issue-PR Link | $8.06\text{e}5$ | — | $5.08\text{e}4$ | Strong constraint, 10$\times$ loss |
| Repo Filter | $5.84\text{e}5$ | — | $2.17\text{e}4$ | Star/Issue thresholds |
| F2P Success | $4.13\text{e}4$ | — | 4006 | Install + Validation pass |
| Issue Clarity | $3.30\text{e}4$ | — | 3701 | Three-model consensus |
| Stable (3 runs)| $\mathbf{3.21\text{e}4}$ | — | $\mathbf{3617}$ | Final Release |

### Ablation Study

| Configuration | pass@1 | pass@10 | Description |
|------|--------|---------|------|
| Non-interactive (Qwen3-480B) | 12.1 | 15.7 | Baseline fixed process |
| mini-SWE-agent (Qwen3-30B, 32k) | 17.4 | 46.1 | Small model + interaction > large non-interactive |
| mini-SWE-agent (DeepSeek-V3.2, 32k) | 20.3 | 59.8 | |
| mini-SWE-agent (Qwen3-480B, 32k) | 25.8 | 58.8 | Primary config |
| mini-SWE-agent (Qwen3-480B, 128k) | **27.1** | **62.7** | Diminishing returns from long context |

### Key Findings
- **"Interaction" is more valuable than "Model Scaling"**: Qwen3-30B interactive pass@1 ($17.4\%$) significantly exceeds Qwen3-480B non-interactive ($12.1\%$). Closed-loop debugging is a necessity, not an luxury.
- **Issue linking is the primary bottleneck**: The jump from "8.6M PRs with tests" to "800k PRs with tests and issue links" represents an order of magnitude loss, motivating the release of 120k PR-derived tasks.
- **A/B\* metadata provides strong signals**: Models consistently perform $5$–$8\times$ better on A subsets than B\* subsets, validating the labels for curriculum design.
- **Issue filtering prioritizes high precision**: Choosing consensus ($0.88$ precision) ensures that training reward signals are not polluted by poorly specified tasks, even at the cost of recall.

## Highlights & Insights
- **"Per-repo vs per-task setup synthesis" is the ultimate leverage for scaling**: 21,692 repo setup agents cost roughly $\$1.9$K USD ($\$0.0873$ per repo). This "configure once, harvest all tasks" approach is applicable to any domain requiring complex environment setups.
- **Failure-driven metadata vs. Metadata-driven failure**: Instead of brainstorming potential defects, the authors induced failure trajectories from frontier models first and then labeled the dataset. This empirical approach is far more robust.
- **Data as a substrate, not just a gold standard**: The authors acknowledge the dataset contains noise (e.g., $10\%$ flaky rates, $2.4\%$ explicit solution leakage in PR-derivatives) and choose to label it transparently rather than aggressive cleaning. This empowers users to perform their own stratified filtering.

## Limitations & Future Work
- Lack of end-to-end RL training: The authors only verify "pre-requisite properties" due to the immense compute required for RL on 32k tasks.
- Docker images cannot eliminate external dependency drift: Package repositories and network resources change over time, requiring ongoing maintenance.
- Single container assumption: The pipeline excludes complex systems requiring multiple services, databases, or queues.
- Leakage in PR-derived tasks: $23\%$ contain some leakage. RL training on these requires leakage detectors to prevent the model from simply memorizing answers.

## Related Work & Insights
- **vs. SWE-rebench v1 (Badertdinov 2025)**: v1 was Python-only and evaluation-oriented. v2 generalizes to 20 languages and focuses on training readiness with pre-built images.
- **vs. SWE-Factory (Guo 2026)**: While Factory covers four languages, v2 provides greater language breadth and more granular instance-level diagnostic metadata.
- **vs. SPICE (Oliva 2025)**: SPICE uses consensus for labeling; v2 integrates this into the automated pipeline and calibrates it against human-verified ground truth.
- **Insight**: Transitioning from "per-task manual" to "per-repo automated + instance labeling" could reduce costs by an order of magnitude for other complex agent environments (e.g., browser-based agents or CTF security tasks).

## Rating
- Novelty: ⭐⭐⭐⭐ — While individual components exist, the combination of a language-agnostic pipeline, training-orientation, and diagnostic metadata is a first for the SWE data field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive funnel analysis and ablations for setup and filtering; missing end-to-end RL verification to confirm downstream utility.
- Writing Quality: ⭐⭐⭐⭐⭐ — Transparent discussion of failure modes, costs, and limitations.
- Value: ⭐⭐⭐⭐⭐ — Addresses the biggest bottleneck in SWE Agent RL and fully open-sources the dataset, images, and pipeline code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](../../ACL2026/code_intelligence/swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
