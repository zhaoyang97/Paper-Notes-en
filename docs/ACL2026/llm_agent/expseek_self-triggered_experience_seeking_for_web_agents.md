---
title: >-
  [Paper Note] ExpSeek: Self-Triggered Experience Seeking for Web Agents
description: >-
  [ACL 2026][LLM Agent][Web Agent] ExpSeek proposes a step-level entropy self-triggered framework for proactive experience seeking…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Web Agent"
  - "experience intervention"
  - "entropy triggering"
  - "proactive guidance seeking"
  - "multi-turn interaction"
date: 2026-05-08
content_hash: ecd003aed2699335
---

# ExpSeek: Self-Triggered Experience Seeking for Web Agents

**Conference**: ACL 2026
**arXiv**: [2601.08605](https://arxiv.org/abs/2601.08605)
**Code**: [https://github.com/WYRipple/ExpSeek](https://github.com/WYRipple/ExpSeek)
**Area**: LLM Agent
**Keywords**: Web Agent, experience intervention, entropy triggering, proactive guidance seeking, multi-turn interaction

## TL;DR

ExpSeek proposes a step-level entropy self-triggered framework for proactive experience seeking, enabling web agents to determine when and what guidance is needed based on intrinsic model signals during interaction, achieving absolute improvements of 9.3% and 7.5% on Qwen3-8B and Qwen3-32B, respectively.

## Background & Motivation

**Background**: Web agents must conduct multi-turn interactions over the open web to retrieve information and answer complex queries. Experience intervention has proven to be an effective paradigm for enhancing agent capabilities, with existing methods primarily following two lines: offline experience distillation and online self-evolution.

**Limitations of Prior Work**: Existing experience injection approaches are passive — experiences are injected once as global context into the system prompt before the task begins. However, as the agent's contextual observations continuously evolve during multi-turn interaction with the environment, statically injected experiences struggle to adapt to dynamic scenarios, potentially leading to decision-making errors.

**Key Challenge**: The effectiveness of experience depends on precise matching of timing and content. Overly frequent intervention increases inference overhead, while overly sparse intervention misses critical guidance windows; global experiences cannot provide tailored guidance for the specific state of each individual step.

**Goal**: To construct a proactive experience-seeking framework that addresses two core questions — (1) *when* to seek experience: leveraging intrinsic model signals to determine the timing of intervention; and (2) *what* experience to seek: designing step-level customized experience content.

**Key Insight**: The authors observe a statistical correlation between a model's step-level entropy (mean token entropy) and reasoning quality — erroneous steps exhibit significantly higher entropy than correct ones. This intrinsic signal can serve as an indicator of the agent's degree of "confusion" without requiring an additional reward model.

**Core Idea**: Using the model's own step-level entropy as a self-triggering signal to determine intervention timing, combined with an experience base and an experience model that dynamically generate step-level customized guidance, thereby shifting the paradigm from passive global injection to proactive step-level seeking.

## Method

### Overall Architecture

ExpSeek comprises three stages: (1) experience base construction — extracting structured experience triplets from successful/failed trajectory pairs and organizing them by topic; (2) entropy self-triggering mechanism — estimating entropy threshold intervals for process steps and answer steps via logistic regression and bootstrap resampling; (3) guided intervention at inference — when step-level entropy exceeds the threshold, an experience model retrieves relevant experiences based on the current context and generates customized guidance.

### Key Designs

1. **Experience Base Construction**:

    - Function: Distilling reusable guiding experiences from training trajectories.
    - Mechanism: For each query in the training set, $k$ trajectories are sampled and paired as successful and failed trajectories $(\tau^+, \tau^-)$. A utility model analyzes the failed trajectory step by step and generates experience triplets for each erroneous step: a behavior description (Behavior), an error analysis (Mistake), and a corrective direction (Guidance, without directly revealing the answer). Topic labels are then inductively assigned to the triplets through iterative batching, forming a topic-organized experience base $\mathcal{E}_p$ (process steps) and $\mathcal{E}_a$ (answer steps).
    - Design Motivation: The triplet design emulates the human pattern of learning from mistakes; topic-based organization enables more efficient retrieval; and separating the base by step type accommodates the distinct entropy distribution characteristics of process steps and answer steps.

2. **Entropy as Self-Trigger**:

    - Function: Using intrinsic model signals to automatically determine when experience intervention is needed.
    - Mechanism: The mean token entropy at each step is computed as $\bar{H}_t = \frac{1}{|R_t|} \sum_{x \in R_t} H(x)$. Separate logistic regression models $P(y_t=0|\bar{H}_t) = 1/(1+e^{-(w \cdot \bar{H}_t + b)})$ are fitted for process steps and answer steps, and 1,000 bootstrap resamplings are used to estimate the 95% confidence interval $[\theta_{lower}, \theta_{upper}]$ as the threshold interval. At inference time, steps below the lower bound receive no intervention, those above the upper bound are always intervened upon, and those within the interval are intervened upon with linearly interpolated probability.
    - Design Motivation: The KS test confirms that the entropy distributions of correct and erroneous steps are statistically distinguishable (process steps: KS=0.1998; answer steps: KS=0.3809; $p<0.001$). Probabilistic intervention avoids the brittleness of hard thresholds, and bootstrap resampling provides robust interval estimates.

3. **Guided Intervention at Inference**:

    - Function: Generating context-matched, customized guidance at the time of triggering.
    - Mechanism: When entropy triggers an intervention and the preceding step was not intervened upon, the experience model $\mathcal{M}_e$ reads the historical context $h_t$ of the current step, selects the 3 most relevant topics from the corresponding experience base, and dynamically generates guidance $e_t$ for the current scenario based on the experience triplets under those topics. For process steps, guidance is appended after the environmental observation; for answer steps, the guidance enables the agent to continue reasoning or revise its answer.
    - Design Motivation: Generative guidance outperforms retrieval-based guidance (confirmed experimentally by a large margin), as generation can adapt general experiences to the specific current context. A one-step cooldown period (no intervention on the step immediately following an intervened step) prevents over-intervention.

### Loss & Training

ExpSeek is an inference-time framework and does not involve training. Qwen3-235B-A22B-Instruct is used as the utility model for experience base construction. The agent uses Qwen3-8B/32B with sampling temperature 1.0, top-p 0.95, and a maximum of 30 ReAct interaction steps.

## Key Experimental Results

### Main Results

**Accuracy (%) on four web agent benchmarks**

| Method | WebWalkerQA | GAIA | Seal | xbench | Avg. |
|--------|-------------|------|------|--------|------|
| **Qwen3-8B** | | | | | |
| No Experience | 38.47 | 29.13 | 23.23 | 25.60 | 32.23 |
| Training-Free GRPO | 40.62 | 29.32 | 25.59 | 26.00 | 33.79 |
| ReasoningBank+ | 40.78 | 32.04 | 26.38 | 28.00 | 34.80 |
| **ExpSeek** | **48.25** | **36.89** | **30.16** | **37.20** | **41.50** |
| **Qwen3-32B** | | | | | |
| No Experience | 45.01 | 36.50 | 27.80 | 27.40 | 37.79 |
| ReasoningBank+ | 45.60 | 33.01 | 29.84 | 36.33 | 39.33 |
| **ExpSeek** | **51.09** | **43.88** | **32.76** | **42.00** | **45.32** |

### Ablation Study

| Variant (8B) | GAIA | xbench |
|--------------|------|--------|
| Process-step guidance only | 33.01 (+3.9) | 28.40 (+2.8) |
| Answer-step guidance only | 30.29 (+1.2) | 34.80 (+9.2) |
| Full ExpSeek | **36.89** (+7.8) | **37.20** (+11.6) |

**Comparison of triggering and guidance strategies (8B, GAIA)**

| Trigger | Guidance | Acc. | Avg. Steps | Avg. Time |
|---------|----------|------|------------|-----------|
| Rule-based | Experience model | 38.81 | 9.52 | 329.71s |
| Claude-4 | Experience model | 39.47 | 8.55 | 370.82s |
| **Entropy** | **Experience model** | **36.89** | **5.75** | **127.57s** |
| Entropy | Retrieval embedding | 30.92 | 5.54 | 110.61s |

### Key Findings

- Entropy triggering offers a notable efficiency advantage: step count is only 60% and wall-clock time only 39% of rule-based triggering, while maintaining comparable accuracy.
- A 4B experience model can effectively guide a 32B agent (GAIA +5.2%, xbench +9.7%), validating the feasibility of weak-to-strong guidance.
- Experience guidance increases entropy at process steps (promoting exploration) and decreases entropy at answer steps (enhancing convergence), forming a "diverge-then-converge" behavioral pattern.
- Performance remains robust even when only one experience per topic is retained, indicating that the experience model can generalize from a small number of seed experiences.

## Highlights & Insights

- Shifting experience intervention from passive global injection to proactive step-level seeking represents a paradigm-level innovation.
- Leveraging the model's own entropy signal as a trigger requires no additional reward model, making the approach both elegant and practical.
- The "diverge-then-converge" entropy behavioral pattern provides an intuitive explanation of how ExpSeek operates internally.
- Strong cross-task generalization: an experience base built from only 25% of WebWalkerQA data remains significantly effective across three out-of-distribution benchmarks.

## Limitations & Future Work

- Threshold estimation relies on the training set and the utility model's assessment of step quality; more accurate estimation strategies remain to be explored.
- Effectiveness on non-web domains and broader toolsets has yet to be validated.
- ExpSeek could be explored as a rollout augmentation technique for agentic reinforcement learning to improve convergence speed and sampling quality.

## Related Work & Insights

- ExpSeek is complementary to offline/online experience accumulation methods such as ReasoningBank, as it focuses on how experiences are utilized (timing and content) rather than how they are collected.
- The successful application of entropy as an indicator of reasoning quality suggests broader possibilities for leveraging model uncertainty signals in other agent settings.
- The successful demonstration of weak-model-guided strong-model inference offers a new direction for reducing guidance costs in practical deployments.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Paradigm shift from passive injection to proactive seeking; entropy self-triggering mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four benchmarks, two model scales, and extensive ablations and analyses covering efficiency, scaling behavior, transferability, and internal mechanisms.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and in-depth experimental analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](../../ICLR2026/llm_agent/st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)

</div>

<!-- RELATED:END -->
