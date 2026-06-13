---
title: >-
  [Paper Note] ExpSeek: Self-Triggered Experience Seeking for Web Agents
description: >-
  [ACL 2026][LLM Agent][Web Agent] ExpSeek proposes a step-level entropy self-triggered proactive experience-seeking framework, enabling Web Agents to judge when guidance is needed and what guidance to obtain based on inte…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Web Agent"
  - "experience intervention"
  - "entropy triggering"
  - "proactive guidance seeking"
  - "multi-turn interaction"
date: 2026-05-08
content_hash: 072c2b1f87d5eda3
---

# ExpSeek: Self-Triggered Experience Seeking for Web Agents

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.08605](https://arxiv.org/abs/2601.08605)  
**Code**: [https://github.com/WYRipple/ExpSeek](https://github.com/WYRipple/ExpSeek)  
**Area**: LLM Agent  
**Keywords**: Web Agent, experience intervention, entropy triggering, proactive guidance seeking, multi-turn interaction

## TL;DR

ExpSeek proposes a step-level entropy self-triggered proactive experience-seeking framework, enabling Web Agents to judge when guidance is needed and what guidance to obtain based on internal signals during interactions, achieving absolute improvements of 9.3% and 7.5% on Qwen3-8B/32B respectively.

## Background & Motivation

**Background**: Web Agents need to perform multi-turn interactions in open networks to retrieve information and answer complex queries. Experience intervention has proven to be an effective paradigm for enhancing agent capabilities, with existing methods mainly following two routes: offline experience distillation and online self-evolution.

**Limitations of Prior Work**: Existing experience injection methods are passive—injecting experience into the system prompt as a global context all at once before the task starts. However, in multi-turn interactions between the agent and the environment, contextual observations change continuously, and the initially injected static experience fails to adapt to dynamic scenarios, potentially leading to decision bias.

**Key Challenge**: The effectiveness of experience depends on the precise matching of timing and content: overly frequent intervention increases reasoning burden, while overly sparse intervention misses critical guidance windows; global experience cannot provide customized guidance for the specific state of the current step.

**Goal**: To build a proactive experience-seeking framework to solve two core problems—(1) When to seek experience (when): utilizing the model's own signals to judge the timing of intervention; (2) What experience to seek (what): designing step-level customized experience content.

**Key Insight**: The authors observe a statistical correlation between the step-level entropy (mean token entropy) of the LLM and reasoning quality—the entropy of incorrect steps is significantly higher than that of correct steps. This internal signal can serve as an indicator of the agent's "confusion" level without requiring an additional reward model.

**Core Idea**: Use the model's own step-level entropy as a self-triggering signal to judge the timing of intervention, combined with an experience base and an experience model to dynamically generate step-level customized guidance, achieving a paradigm shift from passive global injection to proactive step-level seeking.

## Method

### Overall Architecture

ExpSeek consists of three stages: (1) Experience base construction—extracting structured experience triplets from successful/failed trajectory pairs and organizing them by topic; (2) Entropy self-trigger mechanism—estimating entropy threshold intervals for process steps and answer steps through logistic regression and bootstrap resampling; (3) Guided intervention at inference—when the step-level entropy exceeds the threshold, the experience model retrieves relevant experience based on the current context and generates customized guidance.

### Key Designs

1. **Experience Base Construction**:

    - **Function**: Distill reusable guidance experiences from training trajectories.
    - **Mechanism**: For each query in the training set, $k$ trajectories are sampled to pair successful and failed trajectories $(\tau^+, \tau^-)$. A tool model analyzes the failed trajectories step-by-step and generates a triplet for each incorrect step: Behavior description, Mistake analysis, and Guidance (correction direction without directly providing the answer). Finally, topic labels are summarized for the triplets through iterative batch processing, forming experience bases $\mathcal{E}_p$ (process steps) and $\mathcal{E}_a$ (answer steps) organized by topic.
    - **Design Motivation**: The triplet design simulates how humans learn from mistakes. Topic organization makes retrieval more efficient, and splitting bases by step type matches the distinct entropy distribution characteristics of process and answer steps.

2. **Entropy as Self-Trigger**:

    - **Function**: Use internal model signals to automatically determine when experience intervention is required.
    - **Mechanism**: The average token entropy of each step is calculated as $\bar{H}_t = \frac{1}{|R_t|} \sum_{x \in R_t} H(x)$. Logistic regression models $P(y_t=0|\bar{H}_t) = 1/(1+e^{-(w \cdot \bar{H}_t + b)})$ are fitted separately for process and answer steps. A 95% confidence interval $[\theta_{lower}, \theta_{upper}]$ is estimated via 1000 bootstrap resamplings to serve as the threshold interval. During inference, no intervention occurs below the lower bound, intervention is mandatory above the upper bound, and intervention occurs with linear probability within the interval.
    - **Design Motivation**: KS tests confirm that the entropy distributions of correct and incorrect steps are statistically distinguishable (process steps KS=0.1998, answer steps KS=0.3809, p<0.001). Probabilistic intervention avoids the fragility of hard thresholds, and bootstrap provides robust interval estimation.

3. **Guided Intervention at Inference**:

    - **Function**: Generate customized guidance matching the current context upon triggering.
    - **Mechanism**: When entropy triggers an intervention and the previous step was not intervened, the experience model $\mathcal{M}_e$ reads the history context $h_t$ of the current step and selects the three most relevant topics from the corresponding base. It then dynamically generates guidance $e_t$ for the current scene based on triplets under these topics. Guidance for process steps is appended after environment observations, while guidance for answer steps allows the agent to continue reasoning or correct the answer.
    - **Design Motivation**: Generative guidance is superior to retrieval-based guidance (as experiments show retrieval is significantly less effective) because generation adapts general experience to the specific context. A one-step cooling period (no intervention in the immediate next step) prevents over-intervention.

### Loss & Training

ExpSeek is an inference-time framework and does not involve training. Qwen3-235B-A22B-Instruct is used as the tool model for experience base construction. The agents used are Qwen3-8B/32B, with a sampling temperature of 1.0, top-p of 0.95, and a maximum of 30 ReAct interaction steps.

## Key Experimental Results

### Main Results

**Accuracy (%) on Four Web Agent Benchmarks**

| Method | WebWalkerQA | GAIA | Seal | xbench | Avg. |
|------|-------------|------|------|--------|------|
| **Qwen3-8B** | | | | | |
| No Experience | 38.47 | 29.13 | 23.23 | 25.60 | 32.23 |
| Training-Free GRPO | 40.62 | 29.32 | 25.59 | 26.00 | 33.79 |
| ReasoningBank+ | 40.78 | 32.04 | 26.38 | 28.00 | 34.80 |
| **Ours (ExpSeek)** | **48.25** | **36.89** | **30.16** | **37.20** | **41.50** |
| **Qwen3-32B** | | | | | |
| No Experience | 45.01 | 36.50 | 27.80 | 27.40 | 37.79 |
| ReasoningBank+ | 45.60 | 33.01 | 29.84 | 36.33 | 39.33 |
| **Ours (ExpSeek)** | **51.09** | **43.88** | **32.76** | **42.00** | **45.32** |

### Ablation Study

| Variant (8B) | GAIA | xbench |
|-----------|------|--------|
| Process guidance only | 33.01 (+3.9) | 28.40 (+2.8) |
| Answer guidance only | 30.29 (+1.2) | 34.80 (+9.2) |
| Full ExpSeek | **36.89** (+7.8) | **37.20** (+11.6) |

**Comparison of Trigger and Guidance Methods (8B, GAIA)**

| Trigger Method | Guidance Method | Acc. | Avg. Steps | Avg. Time |
|----------|----------|------|----------|----------|
| Rule-based | Experience Model | 38.81 | 9.52 | 329.71s |
| Claude-4 | Experience Model | 39.47 | 8.55 | 370.82s |
| **Entropy Trigger** | **Experience Model** | **36.89** | **5.75** | **127.57s** |
| Entropy Trigger | Retrieval (Emb) | 30.92 | 5.54 | 110.61s |

### Key Findings

- Entropy triggering offers a significant efficiency advantage: it uses only 60% of the steps and 39% of the time required by rule-based triggering while maintaining comparable accuracy.
- A 4B experience model can effectively guide a 32B Agent (GAIA +5.2%, xbench +9.7%), validating the feasibility of using weak models to guide strong models.
- Experience guidance increases entropy in process steps (promoting exploration) and decreases entropy in answer steps (enhancing convergence), forming a "divergence-convergence" behavioral pattern.
- Performance remains robust even when only one experience is kept per topic, indicating the experience model's ability to generalize from a few seed experiences.

## Highlights & Insights

- Transforming experience intervention from passive global injection to proactive step-level seeking is a fundamental paradigm innovation.
- Utilizing the model's own entropy signal as a trigger is both elegant and practical, as it requires no additional reward models.
- The "divergence-convergence" entropy behavioral pattern provides an intuitive explanation for the working mechanism of ExpSeek.
- Strong cross-task generalization: building the experience base with only 25% of WebWalkerQA data remains significantly effective across three OOD benchmarks.

## Limitations & Future Work

- Threshold estimation depends on the training set and the tool model's assessment of step quality; more precise strategies remain to be explored.
- Effectiveness in non-web domains and with broader toolsets has not yet been verified.
- ExpSeek could be explored as a rollout enhancement technique for Agentic RL to improve convergence speed and sampling quality.

## Related Work & Insights

- Complementary to offline/online experience accumulation methods like ReasoningBank, ExpSeek focuses on the utilization of experience (timing and content).
- The successful application of entropy as an indicator for reasoning quality inspires the possibility of utilizing model uncertainty signals in other agent scenarios.
- The success of weak models guiding stronger models provides a new strategy for reducing guidance costs in practical deployments.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Paradigm shift from passive injection to proactive seeking; the entropy self-triggering mechanism is ingeniously designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, two model scales, and extensive ablation and analysis (efficiency, scaling laws, transferability, internal mechanisms).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-justified motivation, and in-depth experimental analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](../../ICML2026/llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

</div>

<!-- RELATED:END -->
