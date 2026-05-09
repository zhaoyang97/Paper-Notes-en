---
title: >-
  [Paper Note] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments
description: >-
  [CVPR 2026 (Oral)][Multimodal VLM][Vision-Language Models] This paper introduces VS-Bench, a multimodal benchmark comprising ten visual game environments, which systematically evaluates VLMs' strategic abilities in multi-agent settings across three dimensions—perception, strategic reasoning, and decision-making. Results reveal that even the strongest current models exhibit significant gaps from optimal performance in reasoning and decision-making.
tags:
  - CVPR 2026 (Oral)
  - Multimodal VLM
  - Vision-Language Models
  - Multi-Agent Evaluation
  - Game Theory
  - Strategic Reasoning
  - Benchmark
date: 2026-05-08
content_hash: 9d283b23009bcb5b
---

# VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments

**Conference**: CVPR 2026 (Oral)  
**arXiv**: [2506.02387](https://arxiv.org/abs/2506.02387)  
**Code**: [GitHub](https://github.com/VS-Bench/VS-Bench)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Multi-Agent Evaluation, Game Theory, Strategic Reasoning, Benchmark

## TL;DR
This paper introduces VS-Bench, a multimodal benchmark comprising ten visual game environments, which systematically evaluates VLMs' strategic abilities in multi-agent settings across three dimensions—perception, strategic reasoning, and decision-making. Results reveal that even the strongest current models exhibit significant gaps from optimal performance in reasoning and decision-making.

## Background & Motivation

1. **Background**: VLM evaluation has evolved from static tasks (image captioning, VQA) to interactive agent benchmarks spanning software engineering, GUI operation, games, and embodied control. Nevertheless, existing VLM benchmarks almost exclusively focus on single-agent settings.
2. **Limitations of Prior Work**: The real world is inherently multi-agent, involving cooperative, competitive, and mixed-motive interactions. Existing multi-agent LLM evaluations (e.g., GT-Bench, MAgIC) are confined to text-only environments and cannot assess models' ability to process visual observations. Reducing visual information to textual descriptions discards critical cues such as spatial layouts and motion patterns.
3. **Key Challenge**: The three distinctive challenges of multi-agent environments—non-stationary dynamics, interdependent decision-making, and equilibrium selection—are entirely absent from single-agent benchmarks, while the additional complexity introduced by visual observations is ignored by text-only benchmarks.
4. **Goal**: To construct a comprehensive multimodal, multi-agent evaluation platform covering cooperative, competitive, and mixed-motive interaction types, and to thoroughly assess VLMs' strategic capabilities along three dimensions: perception, strategic reasoning, and decision-making.
5. **Key Insight**: Adapting classical environments from game theory and multi-agent reinforcement learning into visual games, where VLMs receive multimodal observations (image + text) and produce actions.
6. **Core Idea**: By carefully designing ten visual environments spanning three game types, paired with three hierarchical evaluation dimensions (perception → reasoning → decision-making), VS-Bench provides a comprehensive and standardized assessment of VLMs' strategic capabilities.

## Method

### Overall Architecture
VS-Bench formalizes multi-agent environments as Partially Observable Markov Games (POMGs), where each agent receives multimodal observations $\mathcal{O}_i = (\mathcal{I}_i, \mathcal{T}_i)$ (image + text), outputs textual actions, and converts them to environment actions via a mapping function $\mathcal{M}$. The system comprises ten visual game environments and three evaluation dimensions.

### Key Designs

1. **Ten Visual Game Environments**:
    - **Function**: Provide standardized test scenarios covering three types of multi-agent dynamics.
    - **Mechanism**: Curated and adapted from classical game-theoretic and MARL environments: cooperative (Hanabi, Overcooked, KAZ), competitive (Breakthrough, Kuhn Poker, Atari Pong, MPE), and mixed-motive (Coin Dilemma, Monster Hunt, Battle of Colors). Each environment is adapted to support multimodal observations (image + text). The environments span diverse properties including fully/partially observable, deterministic/stochastic, synchronous/asynchronous, and symmetric/asymmetric dynamics.
    - **Design Motivation**: Coverage of diverse game types and required capabilities (spatial perception, theory of mind, long-term planning, team coordination) ensures comprehensive evaluation.

2. **Three-Dimensional Evaluation Framework**:
    - **Function**: Progressively assess VLMs' strategic capabilities from low-level to high-level.
    - **Mechanism**: (1) **Perception** — measured by visual element recognition accuracy, with 400 annotated samples collected per environment; (2) **Strategic Reasoning** — measured by accuracy in predicting other agents' next actions (theory-of-mind capability), also with 400 balanced samples per environment; (3) **Decision-Making** — measured by normalized episode return (random = 0, oracle = 100) to capture long-term decision quality. The three dimensions are hierarchically structured to precisely localize capability bottlenecks.
    - **Design Motivation**: A single metric cannot distinguish between "fails to perceive," "perceives but reasons incorrectly," and "reasons correctly but decides poorly"; layered evaluation enables diagnosis of specific failure modes.

3. **Large-Scale Model Comparison and In-Depth Analysis**:
    - **Function**: Provide comprehensive experimental analysis and interpretable insights.
    - **Mechanism**: Evaluation of 15 frontier VLMs (6 commercial reasoning models, 6 commercial chat models, 3 open-source models) with unified settings: temperature 1.0, maximum output 8K tokens. In-depth analyses cover multimodal vs. text-only observations, test-time scaling (CoT prompting vs. IO vs. reasoning), social behavior and persona assignment, human baseline experiments, and failure case analysis.
    - **Design Motivation**: Beyond reporting results, the goal is to understand *why* and *where* models underperform, providing clear directions for future research.

### Loss & Training
This paper presents a benchmark study and involves no model training. All models are evaluated under unified inference settings: temperature 1.0, maximum output 8K tokens; reasoning models are additionally allowed 16K reasoning tokens.

## Key Experimental Results

### Main Results

**Perception Evaluation** (element recognition accuracy %):

| Model | Overall | Hanabi | Overcooked | Breakthrough | Pong |
|-------|---------|--------|------------|--------------|------|
| o3 | **84.9** | 79.7 | 69.8 | 97.2 | 64.6 |
| Gemini-2.5-pro | 83.4 | 79.9 | 54.5 | 98.5 | 86.5 |
| Qwen2.5-VL-72B | 80.3 | 76.0 | 72.9 | 75.1 | 65.2 |

**Strategic Reasoning** (next-action prediction accuracy %):

| Model | Overall | Best Environment | Worst Environment |
|-------|---------|-----------------|------------------|
| o3 | **46.6** | Poker 67.0% | Pong 25.8% |
| Claude-3.7-sonnet | 40.4 | Poker 65.2% | Overcooked 26.0% |
| Random | 23.0 | — | — |

**Decision-Making Evaluation** (normalized return %):

| Model | Overall | Best Cooperative | Best Competitive | Best Mixed-Motive |
|-------|---------|-----------------|-----------------|-------------------|
| o3 | **31.4** | Hanabi 55.8 | Board 65.0 | Hunt 24.0 |
| Gemini-2.5-pro | 23.2 | Overcooked 17.1 | Board 55.0 | Battle 33.8 |
| Human Average | **62.7** | — | — | — |

### Ablation Study

| Configuration | Decision-Making Overall | Notes |
|--------------|------------------------|-------|
| Multimodal observations | 31.4% | Standard setting (o3) |
| Text-only observations | Marginally higher | Still far below oracle after removing visual challenges |
| Chat + IO prompting | ~4.8% | GPT-4.1 |
| Chat + CoT prompting | Substantial gain | CoT substantially improves chat models |
| Reasoning model | 31.4% | Reasoning models consistently best |

### Key Findings
- **Perception is largely adequate**: All models achieve overall accuracy ≥67.8%; the best model (o3) reaches 84.9%, with no significant advantage for reasoning models.
- **Strategic reasoning is the critical bottleneck**: The best model (o3) achieves only 46.6% overall accuracy, with particularly poor performance on video-game-style environments (Overcooked, Pong, Monster Hunt), highlighting the compounded challenge of visual observations and strategic interaction.
- **Decision-making capability is severely lacking**: 4 out of 15 models perform worse than a random policy overall; o3 achieves only 31.4%, surpassing only 12.9% of human participants.
- **Open-source models are competitive with reasoning models in mixed-motive games**: InternVL3 on Coin Dilemma and Qwen2.5-VL on Monster Hunt achieve strong results through cooperative strategies.
- **Persona assignment significantly affects social games**: Assigning self-interested or cooperative personas to o3 produces notable changes in behavior and performance.

## Highlights & Insights
- The **three-dimensional hierarchical evaluation design** is particularly elegant: the perception → reasoning → decision-making progression enables precise diagnosis of whether failures stem from "cannot see," "cannot reason," or "cannot act," a paradigm transferable to benchmark design for other complex tasks.
- The **social behavior analysis** yields the most intriguing finding: open-source models, by virtue of a stronger cooperative tendency, outperform more capable reasoning models in mixed-motive games, revealing that "intelligence" and "cooperativeness" may constitute distinct capability dimensions.
- The **multimodal vs. text-only comparison** is instructive: removing visual input yields only marginal improvement, indicating that the core bottleneck lies in strategic reasoning rather than visual perception, thereby pointing to a clear direction for future improvement.

## Limitations & Future Work
- The ten games, while covering three game types, remain limited in scale and lack more complex real-world scenarios (e.g., autonomous driving, financial trading).
- Evaluation is restricted to 2-player settings (with limited 3-player experiments in the appendix), leaving large-scale multi-agent scenarios unexplored.
- All models are evaluated under uniform parameter settings, without fully exploring each model's optimal configuration.
- Future directions: (1) training specialized multi-agent strategic models; (2) exploring in-context learning for VLMs in game-theoretic settings; (3) developing methods to enhance VLMs' theory-of-mind capabilities.

## Related Work & Insights
- **vs. GT-Bench / GAMA-Bench**: These benchmarks evaluate LLMs' game-playing ability in text-only environments; VS-Bench extends evaluation to the multimodal domain with a more comprehensive framework (perception + reasoning + decision-making vs. decision-making only).
- **vs. MAgIC / LLMArena**: Also text-only multi-agent evaluations; VS-Bench's visual environments more closely approximate real-world multi-agent interaction settings.
- **vs. VisualWebArena / OSWorld**: These are single-agent VLM benchmarks; VS-Bench uniquely introduces the challenges of multi-agent interaction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First multimodal multi-agent VLM benchmark, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 15 models × 10 environments × 3 dimensions, complemented by human baselines and in-depth analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-designed tables, and substantive analysis.
- **Value**: ⭐⭐⭐⭐⭐ Provides a standardized evaluation platform for VLMs' multi-agent strategic capabilities; experimental findings offer important guidance for future research.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[AAAI 2026\] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection](../../AAAI2026/multimodal_vlm/multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)

<!-- RELATED:END -->
