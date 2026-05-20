---
title: >-
  [Paper Note] SIGN: Schema-Induced Games for Naming
description: >-
  [AAAI 2026][Model Compression][Naming Game] SIGN introduces lightweight message Schemas (e.g., `@say {name: Ck}`) into LLM multi-agent naming games…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Naming Game"
  - "LLM Multi-Agent"
  - "Convention Formation"
  - "Schema Guidance"
  - "Multi-Agent Coordination"
date: 2026-05-08
content_hash: b766dfae9c1afba3
---

# SIGN: Schema-Induced Games for Naming

**Conference**: AAAI 2026
**arXiv**: [2510.21855](https://arxiv.org/abs/2510.21855)  
**Code**: [github.com/ryanzhangofficial/llm-naming-game-steering](https://github.com/ryanzhangofficial/llm-naming-game-steering)  
**Area**: Model Compression
**Keywords**: Naming Game, LLM Multi-Agent, Convention Formation, Schema Guidance, Multi-Agent Coordination

## TL;DR
SIGN introduces lightweight message Schemas (e.g., `@say {name: Ck}`) into LLM multi-agent naming games, demonstrating that structured priors can improve group convention agreement by up to 5.8×, reduce convergence token cost by an order of magnitude, and provide a simple, controllable "tuning knob" for efficient multi-agent coordination.

## Background & Motivation

### State of the Field
Large language models are transitioning from monolithic applications toward multi-agent collaboration scenarios (collaborative coding, distributed planning, etc.). In multi-agent systems, agents must form consistent communication conventions. Prior work has shown that LLM populations can spontaneously develop shared conventions through interaction, analogous to human language evolution.

### Limitations of Prior Work

**Inconsistent conventions**: LLM agents tend to produce inconsistent conventions under unconstrained natural language communication, leading to coordination failures.

**Slow convergence**: Without structural constraints, large numbers of tokens are wasted on redundant expressions, making group convention convergence extremely slow.

**Poor scalability**: As system scale increases, the efficiency problems of unstructured communication become more pronounced.

### Root Cause
LLMs can be instructed to use structured formats such as JSON Schema to improve reasoning and collaborative efficiency, yet it remains unclear whether such structured priors can **guide convention formation itself** (rather than merely improving individual interaction quality).

### Starting Point
This paper systematically investigates the effect of lightweight Schema priors on convention formation in LLM populations through the classical Naming Game framework.

## Method

### Overall Architecture
The naming game is defined over $N$ agents and a fixed lexicon $\mathcal{L} = \{C_1, \ldots, C_M\}$. Each round, two agents are randomly paired; each generates a message that is mapped to a name in the lexicon via a decoder. Each agent maintains a memory window of size $K$, recording the most recent $K$ interactions with partners.

### Key Designs

#### 1. **Three Experimental Conditions**
- **NL (Natural Language)**: Agents generate unconstrained natural language output; a decoder attempts to extract valid tokens.
- **NL-SW (Natural Language Sliding Window)**: Adds a memory window $K$ to NL, where recent interactions influence subsequent proposals.
- **Schema**: Responses are required to match the format `@say {name: Ck}`, parsed via regular expressions; non-compliant outputs trigger one retry, and if still invalid, a random selection is made by default.
- **Design Motivation**: Schema provides an explicit, easily parsed handle for lexicon entries, making responses transparent to listeners at minimal overhead.

#### 2. **Lose-Shift Mechanism**
- When $y_i \neq y_j$ (two agents disagree), the agent adopts the partner's choice with probability $\alpha$.
- $\alpha$ is a key parameter controlling the speed of convention propagation.
- Experiments explore $\alpha \in \{0.5, 0.75, 0.9\}$.

#### 3. **Non-Compliance Handling Strategy**
- First non-compliance: a brief reminder is sent and the agent retries.
- Second non-compliance: free text is decoded; if undecodable, it is marked as None.
- This ensures the experiment runs correctly under edge cases.

### Experimental Setup
- **Models**: Phi-3 Mini 4K Instruct (main experiments), LLaMA 3.2 3B Instruct (appendix validation)
- **Decoding parameters**: max\_tokens=32, temperature=0.7, top-p=0.9, repeat\_penalty=1.1
- **Scale**: $N \in \{12, 24\}$ agents, lexicon size $M=12$, $T=300$ rounds
- **Evaluation metrics**: group agreement rate, tokens required for convergence

## Key Experimental Results

### Main Results

| N | K | NL | NL-SW | Schema | Gain |
|---|---|-----|-------|--------|------|
| 12 | 0 | 0.111 ± 0.048 | — | — | — |
| 24 | 0 | 0.125 ± 0.042 | — | — | — |
| 12 | 5 | — | 0.278 ± 0.127 | **0.611 ± 0.293** | 2.2× |
| 24 | 5 | — | 0.292 ± 0.042 | **0.556 ± 0.064** | 1.9× |
| 12 | 10 | — | 0.333 ± 0.144 | **0.639 ± 0.096** | 1.9× |
| 24 | 10 | — | 0.295 ± 0.039 | **0.588 ± 0.085** | 2.0× |

### Ablation Study

| Configuration | Tokens to 50% Agreement | Tokens to 60% Agreement | Tokens to 70% | Notes |
|------|-------------------|-------------------|----------|------|
| NL | Not reached | Not reached | Not reached | Fundamentally unable to converge to high agreement |
| NL-SW | ~10× Schema | ~100× Schema | Not reached | Extremely slow convergence |
| Schema | Baseline | Baseline | **Only condition to reach** | Order-of-magnitude token advantage |

### Cross-Model Validation

| Model Configuration | Schema Agreement | NL/NL-SW Agreement |
|---------|-------------|---------------|
| Phi-3 only | 0.6–0.65 | <0.3 |
| LLaMA only | 0.75–0.8 | 0.65–0.7 |
| Phi+LLaMA mixed | Clear advantage | Lower than Schema |

### Key Findings
1. Under Schema conditions, group agreement reaches 0.6–0.65; NL-SW achieves only ~0.3 and NL below 0.2, with a maximum improvement of **5.8×**.
2. Schema requires an order of magnitude fewer tokens than NL/NL-SW to converge to 50% agreement, and nearly two orders of magnitude fewer at 60%.
3. Increasing $\alpha$ slightly reduces agreement for both NL-SW and Schema; $\alpha=0.5$ yields the most consistent results.
4. The agreement improvement is primarily attributable to Schema guidance rather than group size or memory window.
5. LLaMA achieves higher overall agreement than Phi-3, but Schema provides additional gains for both.
6. Schema remains effective in mixed-model populations, demonstrating applicability to heterogeneous systems.

## Highlights & Insights
1. **Minimalist design philosophy**: A single lightweight tag `@say {name: Ck}` significantly improves multi-agent coordination, illustrating the power of "minimal structuring."
2. **From "improving individual interactions" to "guiding convention formation"**: The value of structured formats is extended from micro-level interaction to macro-level social dynamics.
3. **Model-agnostic control knob**: Schema serves as a universal coordination mechanism independent of any specific model.
4. **Connection to language evolution**: The convention formation process in LLM populations resembles the evolution of human language communities; Schema is analogous to grammatical norms.

## Limitations & Future Work
1. The lexicon size is fixed at 12; performance under larger lexicons remains unexplored.
2. Population size is tested only up to 24; scalability to hundreds or thousands of agents has not been validated.
3. Only the naming game, a simple task, is evaluated; effectiveness in more complex collaborative scenarios (e.g., joint planning, multi-turn negotiation) is unknown.
4. Whether Schema consistency limits expressiveness and creativity in broader tasks warrants further investigation.
5. Dynamic or adaptive Schema designs are not considered.

## Related Work & Insights
- The naming game theory (Baronchelli et al. 2008) provides the classical framework for studying convention formation.
- LLM structured output (JSON Schema, etc.) has proven effective in single-turn inference; this paper extends its utility to group dynamics.
- Implication for LLM multi-agent system design: even the simplest communication protocol can significantly improve efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ — Combining structured formats with convention formation is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ — The task is relatively simple and scale is small, but cross-model validation is solid.
- Writing Quality: ⭐⭐⭐⭐ — Concise and clear with well-designed experiments.
- Value: ⭐⭐⭐⭐ — Provides actionable insights for multi-agent LLM system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](../../ICCV2025/model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[AAAI 2026\] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)](satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)
- [\[AAAI 2026\] Renormalization Group Guided Tensor Network Structure Search](renormalization_group_guided_tensor_network_structure_search.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)

</div>

<!-- RELATED:END -->
