---
title: >-
  [Paper Note] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?
description: >-
  [NeurIPS 2025][LLM Agent][Research Agent] This paper proposes MLRC-Bench, a dynamic benchmark grounded in ML conference competition tasks, designed to objectively evaluate the ability of LLM agents to propose and implement novel research methods. The study finds that even the strongest agent (gemini-exp-1206) closes only 9.3% of the gap between the baseline and top human solutions, and that LLM subjective scores for "novelty" exhibit virtually no correlation with actual performance.
tags:
  - NeurIPS 2025
  - LLM Agent
  - Research Agent
  - Benchmark
  - ML Competition
  - Methodological Innovation
  - LLM-as-Judge
date: 2026-05-08
content_hash: 154f96df33dfe6ea
---

# MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?

**Conference**: NeurIPS 2025
**arXiv**: [2504.09702](https://arxiv.org/abs/2504.09702)
**Code**: [HuggingFace](https://huggingface.co/spaces/launch/MLRC_Bench)
**Area**: LLM Agent / AI for Science
**Keywords**: Research Agent, Benchmark, ML Competition, Methodological Innovation, LLM-as-Judge

## TL;DR

This paper proposes MLRC-Bench, a dynamic benchmark grounded in ML conference competition tasks, designed to objectively evaluate the ability of LLM agents to propose and implement novel research methods. The study finds that even the strongest agent (gemini-exp-1206) closes only 9.3% of the gap between the baseline and top human solutions, and that LLM subjective scores for "novelty" exhibit virtually no correlation with actual performance.

## Background & Motivation

**State of the Field**: Evaluation of LLM research agents has primarily followed two directions: (1) end-to-end scientific discovery pipelines akin to AI Scientist (idea generation → coding → experimentation → paper writing), which rely on LLM-as-Judge or human review and lack objective standards; and (2) Kaggle-style ML engineering competitions similar to MLE-Bench, which rarely require genuine methodological innovation and can be solved through hyperparameter tuning or ensemble of existing methods.

**Limitations of Prior Work**: (1) The subjectivity of end-to-end evaluation (e.g., AI Scientist)—LLM-as-Judge tends to produce overly optimistic assessments; (2) The shallowness of Kaggle-style evaluation (e.g., MLE-Bench)—no requirement to propose new methods; (3) Most existing benchmarks rely on single-file code, disconnected from the repository-level development characteristic of real research; (4) Many benchmarks lack computational constraints (runtime/GPU limits), failing to incentivize efficient solutions.

**Root Cause**: How to simultaneously evaluate an agent's capacity for methodological innovation and its objective performance within a single framework—requiring both novel method proposals and quantifiable performance metrics.

**Paper Goals**: To construct an objective, extensible, and frontier-oriented agent benchmark that assesses whether LLM agents can propose and implement genuinely effective novel methods.

**Starting Point**: ML conference competitions inherently embody the dual requirements for such evaluation—open-ended problems demand creativity, while public leaderboards provide objective comparison baselines. These competitions are directly repurposed as benchmark tasks.

**Core Idea**: ML conference competitions are restructured into agent-agnostic standardized environments. The relative progress from a baseline to the top human solution serves as the objective metric, while the correlation between LLM subjective scores and objective performance is systematically analyzed.

## Method

### Overall Architecture

MLRC-Bench comprises seven ML competition tasks spanning domains including LLM merging, backdoor trigger recovery, temporal action localization, rainfall prediction, machine unlearning, product recommendation, and cross-domain meta-learning. Each task provides a standardized code repository, a baseline solution, dev/test datasets, and evaluation metrics. Agents modify code within the `methods/` directory to implement new approaches, iterate on the development set, and are ultimately evaluated on the test set. The primary metric is Relative Improvement to Human:

$$s'_{\text{agent}} = \frac{s_{\text{agent}} - s_{\text{baseline}}}{s_{\text{top\_human}} - s_{\text{baseline}}} \times 100\%$$

### Key Designs

1. **Repository-Level Code Framework**:

    - Function: Enables agents to work within realistic research project structures rather than single-file submissions.
    - Mechanism: Each competition is restructured into a standardized project layout, launched via `python main.py --method my_method --phase dev/test`. Agents may only modify code within the `methods/` directory; evaluation scripts are read-only. Test set data is hidden from the agent during development via file permission controls.
    - Design Motivation: Real-world ML research involves multi-file collaboration and dependency reuse, which single-file formats oversimplify. Repository-level structure also supports multi-agent collaboration (e.g., literature review agent, coding agent, and evaluation agent operating in parallel).

2. **Anti-Overfitting Evaluation Protocol**:

    - Function: Prevents agents from overfitting to the test set.
    - Mechanism: Agents iteratively modify code and evaluate on the dev set; the system snapshots the codebase after each modification. Upon completion, the snapshot with the best dev-set performance is selected for final evaluation on the test set, strictly following standard ML model selection practice.
    - Design Motivation: Mirrors the model selection workflow in real ML research, preventing agents from "gaming" the test set through repeated submissions.

3. **Dual Objective–Subjective Evaluation**:

    - Function: Quantitatively assesses agent solution quality and empirically examines the reliability of LLM subjective evaluation.
    - Mechanism: Objective metrics include effectiveness (competition metric), efficiency (runtime), and conciseness (LLoC). Subjective metrics consist of scores assigned by an o1 model across five dimensions (validity, clarity, rigor, generalizability, novelty) under both code-visible and code-blind settings. Spearman correlation coefficients are computed to analyze the relationship between objective and subjective metrics.
    - Design Motivation: To empirically validate the reliability of LLM-as-Judge evaluations employed in works such as AI Scientist, providing an empirical foundation for evaluation methodology.

### Loss & Training

No model training is involved. Agents operate under the MLAB framework (ReAct style), with each trial limited to 50 steps or 5 hours. Eight trials are conducted per configuration, with the best result reported.

## Key Experimental Results

### Main Results

Relative Improvement to Human (%) for different LLMs under the MLAB framework:

| Agent/LLM | Temporal Loc. | LLM Merging | Meta-Learning | Recommendation | Rainfall | Unlearning | Backdoor | **Average** |
|-----------|--------------|-------------|---------------|----------------|----------|------------|----------|-------------|
| gemini-exp-1206 | -0.5 | 5.0 | -1.1 | 0.1 | 43.1 | 5.6 | 12.9 | **9.3** |
| llama-3.1-405b | 0.5 | -1.0 | -4.9 | 0.0 | 31.5 | 6.2 | 11.5 | 6.3 |
| o3-mini | 0.3 | -1.0 | -4.9 | 0.1 | 25.1 | 3.6 | 6.2 | 4.2 |
| claude-3.5-sonnet | 0.8 | 5.0 | -4.9 | 3.0 | 14.6 | -94.7 | 39.9 | -5.2 |
| gpt-4o | 0.3 | 2.0 | -4.9 | 0.6 | 47.5 | -18.0 | 10.4 | 5.4 |
| gpt-4o + Human Idea | 0.5 | -1.0 | -4.9 | 2.2 | 12.3 | 6.8 | 8.8 | 3.5 |
| gpt-4o + CoI Idea | 0.4 | -1.0 | -4.9 | 0.1 | 39.4 | 11.8 | 4.0 | 7.1 |

### Ablation Study

| Analysis | Finding |
|----------|---------|
| Providing AI-generated ideas | Does not consistently improve performance; sometimes degrades it |
| Providing human ideas | Also inconsistent; implementation capability is the bottleneck |
| Novelty vs. performance Spearman correlation | -0.06 (near-zero) |
| Iterative optimization trend | Code size and runtime grow steadily; performance gains diminish |
| Cost-effectiveness | llama-3.1-405b offers the best cost-performance ratio |
| Pass@k scaling | High-quality ideas with multiple attempts help, but human ideas outperform AI-generated ones |

### Key Findings

- **Agent capability is severely limited**: The strongest agent closes only 9.3% of the baseline-to-human gap; on most tasks, agents fail to surpass the baseline.
- **Providing ideas is insufficient; implementation is the bottleneck**: Supplying human or even expert ideas does not consistently improve performance, indicating that the agent's ability to translate ideas into code and optimize them is the critical weak link.
- **LLM-as-Judge is unreliable**: Novelty scores are nearly uncorrelated with actual performance (Spearman $\rho = -0.06$), suggesting LLM reviewers may produce overly optimistic evaluations.
- **Agents over-engineer solutions**: As iteration progresses, code complexity and runtime grow, but performance gains are disproportionately small.
- **Claude catastrophically fails on machine unlearning (−94.7%)**: A case study reveals the agent optimizes forgetting and retention as separate objectives rather than jointly, leading to severe performance degradation.
- **Rainfall prediction scores are inflated**: Likely because similar approaches (U-Net variants) are widely available online.

## Highlights & Insights

- **Three clever benchmark design choices**: (1) Normalization using competition baselines and top human solutions enables cross-task comparability; (2) Repository-level code structure combined with file permission controls simulates authentic research environments; (3) A dynamic update mechanism allows continuous addition of new competitions and retirement of saturated tasks.
- **An important warning for AI Scientist-style works**: If LLM-as-Judge novelty scores are uncorrelated with actual effectiveness, evaluations of research agents that rely entirely on LLM review (e.g., AI Scientist) may substantially overestimate agent research capabilities.
- **Insight into agent failure modes**: 11.5% of steps fail due to tool parameter errors (hallucinated argument names), and only 17.2% of code execution errors are self-corrected by the agent, exposing the fragility of LLMs operating within complex codebases.

## Limitations & Future Work

- **Only seven tasks**: The limited number of tasks may reduce representativeness, though the authors emphasize quality over quantity.
- **High computational cost**: Eight trials per configuration × five models × three framework settings entails substantial API expenditure, constraining exploration of additional models and trials.
- **Potentially weak baselines**: In certain tasks (e.g., backdoor trigger recovery), the baseline is intrinsically weak, meaning an agent that outperforms it does not necessarily demonstrate genuine methodological innovation.
- **Limited agent frameworks tested**: The evaluation primarily covers MLAB; other frameworks (AIDE, SELA, etc.) are incompatible due to their single-file assumptions.
- **Multi-agent collaboration not explored**: Although the repository-level design supports multi-agent division of labor, this setting is not experimentally investigated.

## Related Work & Insights

- **vs. AI Scientist**: AI Scientist targets end-to-end pipelines (idea → paper) evaluated by LLM/human reviewers. MLRC-Bench focuses on method proposal and implementation, evaluated with objective metrics. The two approaches are complementary.
- **vs. MLE-Bench / MLAgentBench**: These benchmarks use Kaggle-style tasks that do not require methodological innovation and accept single-file submissions. MLRC-Bench demands novel methods and operates at the repository level.
- **vs. RE-Bench**: RE-Bench also targets ML research capabilities but concentrates on older domains such as language models and CIFAR-10, relying on expert-curated tasks that are difficult to update. MLRC-Bench sources tasks directly from competitions, enabling continuous and scalable updates.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic evaluation of research agents' methodological innovation capacity using ML competitions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five LLMs, three framework settings, and multi-dimensional evaluation with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures and tables, rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Makes significant benchmark and methodological contributions to the LLM research agent community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](../../ICLR2026/llm_agent/comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[NeurIPS 2025\] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments](defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)
- [\[NeurIPS 2025\] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

<!-- RELATED:END -->
