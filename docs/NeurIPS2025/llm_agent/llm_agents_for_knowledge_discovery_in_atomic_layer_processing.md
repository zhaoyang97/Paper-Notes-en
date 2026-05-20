---
title: >-
  [Paper Note] LLM Agents for Knowledge Discovery in Atomic Layer Processing
description: >-
  [NeurIPS 2025][LLM Agent][knowledge discovery] By having an LLM agent control a simulated chemical reactor (a black-box function), this work demonstrates that agents can explore, discover…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "knowledge discovery"
  - "atomic layer processing"
  - "scientific exploration"
  - "tool-augmented reasoning"
date: 2026-05-08
content_hash: bbbbd98c60fdaa26
---

# LLM Agents for Knowledge Discovery in Atomic Layer Processing

**Conference**: NeurIPS 2025
**arXiv**: [2509.26201](https://arxiv.org/abs/2509.26201)  
**Code**: [https://github.com/awwerbro/ALDReactor](https://github.com/awwerbro/ALDReactor)  
**Area**: LLM Agent
**Keywords**: LLM agent, knowledge discovery, atomic layer processing, scientific exploration, tool-augmented reasoning

## TL;DR
By having an LLM agent control a simulated chemical reactor (a black-box function), this work demonstrates that agents can explore, discover, and summarize the rules of an unknown chemical system through trial and error without any prior knowledge, revealing both the capabilities and limitations of agents for open-ended scientific discovery.

## Background & Motivation

Applications of LLM agents in materials science primarily fall into two categories:

**Knowledge integration**: synthesizing, validating, and organizing domain knowledge

**Self-driven laboratories**: serving as components in automated experimental pipelines to pursue specific optimization objectives

Both categories make it difficult to distinguish among three distinct agent capabilities: **recall of training-time knowledge**, **latent knowledge discovery**, and **genuinely novel knowledge discovery**.

This paper raises a critical question: **Can an LLM agent discover entirely new knowledge solely by "exploring a system," without any concrete objective function?**

To address the evaluation challenge, the authors cleverly design **entirely fictitious systems** — a custom-rule "alien market" and a fabricated chemical reactor — ensuring the agent cannot rely on training-time knowledge, thereby genuinely testing its discovery capacity.

This question is highly relevant to **Atomic Layer Processing (ALP)** in the semiconductor industry: discovering and characterizing novel chemical reactions is a time-consuming and expensive process that AI/ML may help accelerate.

## Method

### Overall Architecture

Core idea: repurpose LangGraph's tool functionality to provide **black-box functions**, allowing the agent to explore freely without any prescribed objective.

```
LLM Agent
    ├── Tool 1: perform_experiment(recipe) → sensor data summary
    ├── Tool 2: retrieve_experiment(id) → full experimental data
    └── Objective: describe system rules (no other optimization target)
```

**Two test systems**:

1. **Alien market** (proof of concept): the agent must discover rules through trial and error — the alien refuses to sell items whose names contain the letters "p" or "m"
2. **ALP reactor simulation** (main experiment): the agent controls a virtual chemical reactor and explores reactions among fictitious chemical species via pressure sensors and a quartz crystal microbalance (QCM)

### Key Designs

**ALP reactor simulation system**:

The reactor is modeled as a tube discretized into $N$ segments with uniform temperature. Core equations:

Gas-phase species reaction-transport equation:
$$\frac{\partial c_i(x,t)}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} - v\frac{\partial c_i}{\partial x} - \frac{4}{d}\sum_j r_{ij}$$

Surface coverage evolution:
$$\frac{\partial \theta_i}{\partial t} = \frac{1}{\sigma_i}\sum_j r_{ij}$$

Reaction rate $r_i = k_i c_i \theta_i \sigma$, where $k_i$ follows Arrhenius kinetics.

**Three experimental configurations**:

| Config | # Species | Discoverable Reactions | Difficulty |
|--------|-----------|----------------------|------------|
| I | 2 (A, B) | ALD deposition | Easy; saturation achievable with 0.5 s exposure |
| II | 2 (A, B) | ALD deposition (slower kinetics) | Hard; requires 40 s exposure + elevated temperature |
| III | 4 (A, B, C, D) | ALD + ALE + passivation + CVD | Complex; large experimental space |

In Configuration II, the Arrhenius pre-exponential factor for Reaction 2 is reduced by 4×, and the vapor pressure of species B is lowered (Antoine equation parameter B halved), requiring greater patience in exploration.

**Information restriction design**:
- The agent only observes **pressure** and **QCM mass-change** signals (analogous to the limited sensing available to real experimentalists)
- The full reactor state (a 560-dimensional vector) is not exposed to the agent
- Experimental outputs are processed through deterministic narrative generation followed by LLM summarization, rather than being passed as raw data

### Loss & Training

There is no training or loss function in the conventional sense. Agent evaluation is based on:
- **Alien market**: correctness score for rule discovery (1 point per correctly identified letter; −0.5 for spurious rules)
- **ALP reactor**: UMAP projections of the agent's experimental trajectory are compared against expert trajectories to assess coverage of the experimental space

Models used: the alien market tests employ gpt-5, gemini-2.5-pro, gpt-5-mini, gemini-2.5-flash, and gemini-2.0-flash; the ALP reactor experiments use gemini-2.5-pro.

## Key Experimental Results

### Main Results

**Alien market results**:

| Model | No experiment-count limit | After specifying experiment count |
|-------|--------------------------|----------------------------------|
| gpt-5 | Best performance (conducts more experiments autonomously) | Further improved |
| gemini-2.5-pro | Poor (stops too early) | Substantially improved |
| gpt-5-mini | Poor | Substantially improved |
| gemini-2.5-flash | Poor | Substantially improved |
| gemini-2.0-flash | Poor | Moderately improved |

Key finding: gpt-5's advantage stems not from superior reasoning but from **spontaneously conducting more experiments**; other models show large performance gains when instructed to execute a sufficient number of experiments.

**ALP Configuration I (simple, two-species)**: all 3/3 iterations successfully identify the self-limiting nature of the reactions, execute ALD deposition, and further explore kinetic limits and decomposition growth.

**ALP Configuration II (difficult, two-species)**:

| Prompt condition | Time | Outcome |
|-----------------|------|---------|
| IIa (standard prompt) | 3600 s | 3/3 stuck in CVD low-growth local minimum |
| IIb (standard prompt) | 7200 s | 3/3 still stuck in local minimum |
| IIc (+QCM reference value) | 7200 s | 1/3 local minimum, 1/3 mislabeled, 1/3 successful discovery |

**ALP Configuration III (four species, full system)**:

| Iteration | Reactions Discovered | Reactions Missed |
|-----------|---------------------|-----------------|
| 1 | ALD (B+C), passivation (D vs. B–C surface), CVD (high temperature) | ALE (unrecognized; A used only as "cleaner") |
| 2 | ALD, ALE, passivation (D vs. single-pulse B/C) | CVD co-dosing, complete passivation |
| 3 | ALE (A+C), C decomposition | ALD (B+C), CVD |

### Ablation Study

**Effect of persistence**:
- Without a specified experiment count, most models stop exploring prematurely
- Simply increasing experiment time has limited effect (no improvement from Configuration IIa→IIb)
- Providing additional context (QCM reference values) proves more effective than merely increasing time

**Path dependence**:
- In the alien market, nearly all agents begin with "apple," sometimes leading to an erroneous "double-p rule"
- The three iterations of ALP Configuration III explore different portions of the experimental space
- This motivates a swarm strategy — multiple sub-agents explore independently while a supervising agent aggregates their findings

### Key Findings

1. **Persistence is essential**: agents (like humans) require sufficient experimental resources to overcome noise and local minima
2. **Signals trigger curiosity**: agents need to detect some signal before investigating further; in the absence of signals, two strategies are effective — providing more resources or richer context
3. **Inherent path dependence**: different starting points lead to different discovery trajectories; this is a feature rather than a flaw, and can be exploited for swarm exploration strategies
4. **Credibility of knowledge discovery**: because the system is entirely fictitious, it can be confirmed that the agent is performing genuine novel knowledge discovery rather than memory recall

## Highlights & Insights

1. **Elegant experimental design**: the use of fictitious systems cleanly resolves the evaluation challenge of determining whether an agent is drawing on training-time knowledge
2. **Genuinely open-ended exploration**: rather than giving the agent a specific optimization target, the paper asks it to characterize the system — a framing far closer to real scientific discovery
3. **Direct transferability to physical experiments**: the reactor recipe format is compatible with real LabVIEW control systems; switching from simulation to physical experimentation requires only a change of API endpoint
4. **Value of ignorance**: an agent's lack of prior knowledge may open more novel exploration paths — a counterintuitive yet profound insight
5. **Connection to Duan et al.**: concurrent work in biological systems suggests that the scientific discovery capability of LLM agents generalizes across domains

## Limitations & Future Work

1. **Single-agent only**: multi-agent collaborative exploration is not investigated (though swarm strategies are mentioned as a possibility)
2. **Simulation only**: deployment in a real laboratory has not been demonstrated
3. **Subjective evaluation**: conclusions for the ALP section rely on manual inspection of agent statements and experimental trajectories
4. **Model temperature not systematically tuned**: temperature is noted as potentially affecting exploration diversity, but this is not thoroughly investigated
5. **Input space remains limited**: four chemical species already poses a challenge, whereas real chemical spaces are far more complex
6. **No cost–benefit analysis**: the number of experimental resources and time an agent requires to achieve coverage comparable to human experts is not quantified

## Related Work & Insights

- **Distinction from Boiko et al. (2023)**: works such as Coscientist focus on automating known experimental workflows, whereas this paper focuses on open-ended discovery without a prescribed objective
- **Complementarity with Duan et al.**: Duan et al. have LLMs model biological reaction systems in Python, while this paper has LLMs explore chemical systems through experimentation — two complementary approaches
- **"Great without goals" (Stanley & Lehman, 2015)**: the authors invoke the central thesis of this book — the most important discoveries often arise without goal-directed search
- **Broader implications**: analogous frameworks could be used to evaluate agent discovery capability in any simulated environment (physics, biology, economic models, etc.)

## Rating
- Novelty: ⭐⭐⭐⭐ Using fictitious systems to test LLMs' knowledge discovery capacity is a methodologically creative contribution
- Experimental Thoroughness: ⭐⭐⭐ Multiple configurations and prompting strategies are explored, but only 3 iterations per condition limits statistical power
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, experimental descriptions are thorough, and the discussion section is insightful
- Value: ⭐⭐⭐⭐ Opens "goal-free exploration" as an important research direction for AI-assisted scientific discovery

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)
- [\[NeurIPS 2025\] AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)

</div>

<!-- RELATED:END -->
