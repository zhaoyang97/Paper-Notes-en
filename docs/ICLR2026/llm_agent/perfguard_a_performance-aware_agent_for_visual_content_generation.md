---
title: >-
  [Paper Note] PerfGuard: A Performance-Aware Agent for Visual Content Generation
description: >-
  [ICLR 2026][LLM Agent][visual content generation] This paper proposes PerfGuard, a performance-aware agent framework for visual content generation. It replaces textual tool descriptions with a multi-dimensional performance scoring matrix to model tool capability boundaries, and incorporates adaptive preference updating and capability-aligned planning optimization, substantially improving tool selection accuracy (error rate reduced from 77.8% to 14.2%) and visual generation quality.
tags:
  - ICLR 2026
  - LLM Agent
  - visual content generation
  - agent
  - tool selection
  - performance-aware
  - AIGC
  - preference optimization
  - image generation
  - image editing
date: 2026-05-08
content_hash: b8ec8e5b930c0dee
---

# PerfGuard: A Performance-Aware Agent for Visual Content Generation

**Conference**: ICLR 2026
**arXiv**: [2601.22571](https://arxiv.org/abs/2601.22571)
**Code**: [GitHub](https://github.com/FelixChan9527/PerfGuard)
**Area**: LLM Agent
**Keywords**: visual content generation, agent, tool selection, performance-aware, AIGC, preference optimization, image generation, image editing

## TL;DR

This paper proposes PerfGuard, a performance-aware agent framework for visual content generation. It replaces textual tool descriptions with a multi-dimensional performance scoring matrix to model tool capability boundaries, and incorporates adaptive preference updating and capability-aligned planning optimization, substantially improving tool selection accuracy (error rate reduced from 77.8% to 14.2%) and visual generation quality.

## Background & Motivation

LLM agents have demonstrated strong potential in automated task processing, yet exhibit critical deficiencies in visual content generation (AIGC):

**Ambiguous tool capability descriptions**: Existing systems rely on generic textual descriptions (e.g., "capable of generating images semantically aligned with text"), which fail to distinguish performance differences across models along multiple dimensions.

**Idealized assumptions**: Most frameworks assume that "tool invocations always succeed," lacking systematic evaluation of actual tool success rates.

**Static tool selection**: Benchmark scores may diverge from real-world task performance, and cannot adapt to tool updates.

**Disconnect between planning and execution**: Task planning does not account for tool performance boundaries, causing generated subtasks to exceed tool capabilities.

Although systems such as CompAgent and GenArtist enhance generation through multi-model scheduling, their tool descriptions remain coarse-grained and lack performance awareness.

## Method

### Overall Architecture

PerfGuard builds upon a standardized agent system (Analyst → Planner → Worker → Self-Evaluator), driven by user input for iterative visual generation:

1. **Analyst** parses multimodal input → task summary $\tau^*$, target image semantics $s^*$, evaluation objective $g$
2. **Planner** decomposes the task into subtasks $u_t$ based on $\tau^*$, $s^*$, and the tool performance profile $\mathcal{B}$
3. **Worker** selects the most suitable tool from the tool library to execute $u_t$, producing image output $o_t$
4. **Self-Evaluator** assesses the alignment of $o_t$ with objective $g$ across multiple dimensions and feeds back to the Planner

### Key Designs

#### 1. Performance-Aware Selection Modeling (PASM)

Tool performance boundaries are precisely defined via a multi-dimensional scoring matrix rather than textual descriptions:

**Image generation tools**: 7 dimensions based on T2I-CompBench (color, shape, texture, 2D spatial, 3D spatial, non-spatial, numeracy)

**Image editing tools**: 7 dimensions based on ImgEdit-Bench (addition, removal, replacement, attribute alteration, motion change, style transfer, background change)

The Worker generates preference weights $\mathcal{W}_{task} \in \mathbb{R}^{1 \times d}$ for subtask $u_t$ and computes tool fitness scores:

$$S_{tools} = \mathcal{W}_{task} \cdot \text{Normalize}(M_p)^\top$$

$$\mathcal{R} = \text{argsort}(S_{tools}, \text{descending})$$

where $M_p \in \mathbb{R}^{d \times l}$ is the performance boundary matrix of $l$ tools across $d$ dimensions.

#### 2. Adaptive Preference Updating (APU)

The performance boundary matrix is iteratively updated by comparing theoretical rankings with actual execution rankings:

$$\mathcal{R}_{theory} = \text{top}_m(S_{tools}) \cup \text{rand}_n(S_{tools}[m+1:l])$$

$$M_p^{new} = \text{Normalize}\left(M_p + \mathcal{W}_{task} \cdot \eta \cdot \Delta\right)$$

$$\Delta = \frac{\mathcal{R}_{theory} - \mathcal{R}_{actual}}{m+n}$$

An exploration-exploitation strategy is adopted: top-$m$ high-scoring tools are selected alongside $n$ randomly sampled low-scoring tools, increasing the probability of discovering latent high-performing tools. New tools are initialized with the average score of tools of the same category.

#### 3. Capability-Aligned Planning Optimization (CAPO)

Step-aware Preference Optimization (SPO) is extended to the Planner's autoregressive planning process:

At each step, $k$ candidate subtasks $\{u_t^1, \ldots, u_t^k\}$ are generated; the Self-Evaluator selects the best $u_t^w$ and worst $u_t^l$, and the Planner is optimized as follows:

$$\mathcal{L}(\theta) = -\mathbb{E}\left[\log\sigma\left(\alpha\left(\log\frac{p_\theta(u_t^w|ctx)}{p_{ref}(u_t^w|ctx)} - \log\frac{p_\theta(u_t^l|ctx)}{p_{ref}(u_t^l|ctx)}\right)\right)\right]$$

where $ctx = (\tau^*, s^*, \mathcal{B}, h_{t-1})$ and $h_{t-1}$ denotes the history of subtasks and evaluation results.

### Loss & Training

**Self-Evaluator scoring**:

$$e_t = \sum_{i=0}^L \gamma_i^{local} \pi_{Eval}(o_t, g_i^{local}) + \gamma^{global} \pi_{Eval}(o_t, g^{global})$$

A weighted evaluation over global and local semantics guides the winning/losing sample selection in CAPO.

## Key Experimental Results

### Main Results

**Basic image generation (T2I-CompBench)**:

| Method | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|--------|--------|--------|----------|----------|-------------|----------|
| FLUX | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3127 | 0.3771 |
| SD3 | 0.8132 | 0.5885 | 0.7334 | 0.3200 | 0.3140 | 0.3703 |
| GenArtist | 0.8482 | 0.6948 | 0.7709 | 0.5437 | 0.3346 | 0.4499 |
| T2I-Copilot | 0.8039 | 0.6120 | 0.7604 | 0.3228 | 0.3379 | 0.3985 |
| **PerfGuard** | **0.8753** | **0.7366** | **0.8148** | **0.6120** | **0.3754** | **0.5007** |

PerfGuard achieves the best performance across all 6 dimensions.

**Advanced image generation (OneIG-Bench)**:

| Method | Type | Alignment↑ | Text↑ | Reasoning↑ | Style↑ |
|--------|------|-----------|-------|-----------|--------|
| SD3 | Diffusion | 0.801 | 0.648 | 0.279 | 0.361 |
| T2I-Copilot | Agent | 0.821 | 0.679 | 0.318 | 0.386 |
| **PerfGuard** | Agent | **0.834** | **0.684** | **0.350** | **0.395** |

**Complex image editing (Complex-Edit Level-3)**:

| Method | IF↑ | PQ↑ | IP↑ | Overall↑ |
|--------|-----|-----|-----|----------|
| Step1X_Edit | 7.95 | 8.66 | 7.70 | 8.10 |
| OmniGen | 7.52 | 8.86 | 8.01 | 8.13 |
| **PerfGuard** | **8.95** | **9.02** | **8.56** | **8.84** |

### Ablation Study

**Module ablation (T2I-CompBench)**:

| CAPO | PASM | APU | Color↑ | Spatial↑ | Complex↑ |
|------|------|-----|--------|----------|----------|
| ✗ | ✗ | ✗ | 0.8239 | 0.5600 | 0.4327 |
| ✓ | ✗ | ✗ | 0.8466 | 0.5756 | 0.4493 |
| ✗ | ✓ | ✗ | 0.8521 | 0.5919 | 0.4412 |
| ✗ | ✓ | ✓ | 0.8596 | 0.6005 | 0.4738 |
| ✓ | ✓ | ✓ | **0.8753** | **0.6120** | **0.5007** |

PASM contributes the most (Color +3.42%, Texture +5.7%); APU further refines performance (Complex 0.4412→0.4738); CAPO provides an additional layer of overall optimization.

**Tool selection error rate comparison**:

| Method | Error Rate |
|--------|-----------|
| Text description only + QWen3-14B | 77.8% |
| Text description only + GPT-4o | 72.2% |
| External experience module + QWen3-14B | 68.1% |
| PASM (benchmark score matrix) + QWen3-14B | 30.5% |
| PASM + APU (η=0.13, 800 steps) | **14.2%** |

**Update step size η ablation**: η=0.10 converges slowly; η=0.15 converges fast initially but oscillates severely later; **η=0.13** achieves the optimal balance.

### Key Findings

1. Tool selection error rate with pure text descriptions reaches 77.8%; even GPT-4o reduces it only to 72.2%.
2. The performance-aware matrix reduces the error rate to 30.5%, and adaptive updating further brings it to 14.2% (a 5.5× improvement).
3. After CAPO training, the Planner becomes aware of tool performance boundaries and understands how operation ordering affects outcomes.
4. PerfGuard's token consumption does not scale with the number of tools, whereas conventional approaches exhibit catastrophic growth.

## Highlights & Insights

1. **Addresses a genuine pain point**: The framework precisely models the core problem of ambiguous tool capability boundaries in AIGC, offering an intuitive and effective solution.
2. **Efficient tool management**: PASM's dimension-matching approach decouples token consumption from the number of tools, conferring a significant advantage in large-scale tool libraries (200+ tools).
3. **Adaptive closed loop**: APU continuously corrects the performance matrix via actual execution ranking feedback, mitigating the bias inherent in static benchmarks.
4. **Planner training**: CAPO enables the Planner to learn that tool limitations can adversely affect planning accuracy (e.g., editing the background first may reduce the success rate of subsequent steps).
5. **Engineering practicality**: The framework is modular; PASM can be directly applied to any tool library with available benchmark scores.

## Limitations & Future Work

1. The performance boundary dimensions depend on specific benchmarks (T2I-CompBench, ImgEdit-Bench); new task types require redesigning the dimension schema.
2. CAPO requires generating and evaluating multiple candidates, increasing inference cost (although the paper demonstrates faster execution than GenArtist, absolute runtime figures are not reported).
3. PerfGuard underperforms AnySD on the Identity Preservation (IP) metric in image editing, as AnySD targets minimal editing.
4. The tool library caps the upper bound — PerfGuard does not substantially outperform T2I-Copilot on alignment/text metrics.
5. APU convergence depends on sufficient tool usage history; the cold-start problem is only partially addressed through average-score initialization.

## Related Work & Insights

**vs. GenArtist**: GenArtist lacks a performance-aware tool selection strategy, leading to planning errors and missing elements. **vs. T2I-Copilot**: T2I-Copilot achieves strong performance through multi-module semantic decomposition but offers limited tool diversity. **vs. CLOVA**: CLOVA improves success rates via self-reflection and prompt tuning, but does not model tool performance boundaries.

**Core insight**: In agent systems, tool selection is a severely underestimated bottleneck. Replacing natural language descriptions with structured performance matrices is a simple yet highly effective approach that generalizes to any multi-tool agent scenario, including code generation and data analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ (Performance-aware tool selection modeling is a novel and practical approach)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three benchmarks, detailed ablations, efficiency analysis, and tool-scale expansion experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous method presentation, rich visualizations)
- Value: ⭐⭐⭐⭐ (Directly informative for agent tool selection; the framework exhibits strong generalizability)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)

<!-- RELATED:END -->
