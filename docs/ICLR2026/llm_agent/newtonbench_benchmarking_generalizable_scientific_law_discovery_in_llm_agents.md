---
title: >-
  [Paper Note] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents
description: >-
  [ICLR 2026][LLM Agent][scientific discovery] This paper proposes NewtonBench, a benchmark for LLM-based scientific law discovery comprising 324 tasks across 12 physical domains. Novel tasks resistant to memorization are…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "scientific discovery"
  - "benchmark"
  - "counterfactual physical laws"
  - "symbolic regression"
  - "interactive exploration"
date: 2026-05-08
content_hash: 62cc79d4eddd3a7c
---

# NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2510.07172](https://arxiv.org/abs/2510.07172)
**Code**: Available
**Area**: LLM Agent
**Keywords**: scientific discovery, benchmark, counterfactual physical laws, symbolic regression, interactive exploration

## TL;DR
This paper proposes NewtonBench, a benchmark for LLM-based scientific law discovery comprising 324 tasks across 12 physical domains. Novel tasks resistant to memorization are generated via "counterfactual law shifts," requiring agents to discover hidden physical equations through interactive experimentation. GPT-5 achieves the best performance (75.9% symbolic accuracy) but degrades sharply on complex systems (40.3%), and code tools surprisingly hurt stronger models.

## Background & Motivation

**Background**: LLM-driven scientific discovery is an emerging frontier, yet existing benchmarks (e.g., SRBench) face a "methodological trilemma"—scientific relevance, scalability, and anti-memorization cannot be simultaneously satisfied.

**Limitations of Prior Work**:
- Most existing benchmarks involve static function fitting without interactive exploration
- Synthetic benchmarks are scalable but lack scientific grounding
- Real physical equations may be memorized by LLMs from training data
- Systematic evaluation across levels of system complexity is absent

**Key Challenge**: Satisfying scientific grounding, anti-memorization, and scalability simultaneously is inherently contradictory—directly using real laws risks memorization, while fully synthetic laws lose scientific meaning.

**Goal**: Resolve the trilemma via counterfactual law shifts and construct an interactive scientific discovery benchmark.

**Key Insight**: Systematically mutate the expression trees of known physical laws (operator/constant mutations) to generate laws that are scientifically grounded yet never encountered by LLMs.

**Core Idea**: Generate counterfactual physical laws through expression tree mutation combined with an interactive experimental environment, forming the first scalable, memorization-resistant scientific discovery benchmark.

## Method

### Overall Architecture
12 physical domains × 3 difficulty levels (Easy/Medium/Hard, 3 variants each) × 3 system complexity levels (Vanilla/Simple/Complex) = 324 tasks. Agents iteratively design experiments by submitting variable values via a `<run_experiment>` tool and observing outputs to discover the hidden equation.

### Key Designs

1. **Counterfactual Law Shifts**

    - Function: Starting from original physical laws, generate new equations through cumulative mutations.
    - Two mutation types: operator mutation (e.g., $+$ → $\times$) and constant mutation (e.g., square → cube).
    - Three difficulty levels: Easy (1–2 mutations) → Medium (further mutations on Easy) → Hard (further mutations on Medium).
    - Dimensional consistency is maintained by adjusting physical constant units after mutation.
    - Design Motivation: Generated equations never appear in training corpora, providing inherent anti-memorization guarantees.

2. **Three-Level System Complexity**

    - Vanilla: Target equation only, no confounding variables.
    - Simple: Target equation embedded in a simple system with auxiliary equations.
    - Complex: Maximum confounding, with multiple equations forming an interconnected system.

3. **Interactive Experimental Environment**

    - Agents propose input values; the simulator returns system outputs.
    - An optional Python code interpreter is available for numerical regression.

### Evaluation Metrics
- Symbolic Accuracy (SA): Mathematical equivalence checked via LLM-as-judge (98.3% agreement with human annotators).
- RMSLE: Data fitting quality metric.

## Key Experimental Results

### Main Results (11 Models)

| Model | Vanilla Easy | Vanilla Hard | Complex Hard | Avg. SA |
|-------|-------------|-------------|-------------|---------|
| GPT-5 | 90.3% | 87.5% | **40.3%** | **75.9%** |
| Gemini-2.5-pro | 96.5% | 69.4% | 16.7% | 65.4% |
| o4-mini | 88.9% | 52.8% | 2.8% | 47.8% |
| DeepSeek-R1 | 88.2% | 36.8% | 2.8% | 43.4% |
| GPT-4.1 | 16.7% | 1.4% | 0.7% | 5.8% |

### Ablation Study

| Configuration | Key Findings |
|--------------|-------------|
| Code tools on strong models | GPT-5: 75.9% → drops 2–3%; GPT-5-mini: 53.1% → 48.1% — **code is harmful** |
| Code tools on weak models | Models with SA < 40%: code yields clear improvements |
| Noise level 0.0001 | Accuracy drops 12–16% across all models |
| Increasing noise | Performance degrades proportionally with noise level |

### Key Findings
- **Reasoning capability is a prerequisite**: All non-reasoning models (e.g., GPT-4.1) achieve < 10% accuracy.
- **Complexity collapse**: GPT-5 drops from 90.3% (Vanilla Easy) to 40.3% (Complex Hard); second-order and higher complexity is the core bottleneck.
- **Paradoxical effect of code tools**: Strong models exhibit sharply reduced exploration rates when using code (over-exploitation); weak models benefit from computational offloading.
- **Large cross-domain variance**: Bose-Einstein distributions are hardest (18.1% SA); heat conduction is easiest.
- **Reasoning token scaling**: Reasoning models significantly increase token consumption with task complexity; non-reasoning models do not.

## Highlights & Insights
- **Counterfactual law shifts offer an elegant solution to memorization**: Rather than constructing entirely synthetic equations (which lose scientific grounding), controlled mutations are applied to real equations, preserving scientific meaning while preventing memorization.
- **Exploration–exploitation trade-off with code tools**: Strong models given code tools tend toward local fitting (exploitation) at the expense of global exploration, a profound behavioral insight that echoes the classic dilemma in reinforcement learning.
- **Interactive evaluation paradigm**: The benchmark advances from "fit an equation to given data" to "design experiments to discover laws," more faithfully reflecting the real scientific discovery process.

## Limitations & Future Work
- Coverage is limited to physics; generalization to chemistry and biology remains unvalidated.
- Counterfactual laws, while scientifically grounded, do not correspond to real physical phenomena.
- Even minimal noise (0.0001) causes a 12–16% accuracy drop, raising concerns about applicability to real-world scenarios.
- Evaluation is restricted to single-target equation discovery with scalar outputs.

## Related Work & Insights
- **vs. SRBench**: A traditional symbolic regression benchmark based on static data fitting, lacking interactive exploration and anti-memorization design.
- **vs. AI Feynman**: Uses real Feynman equations but is susceptible to memorization; NewtonBench addresses this via counterfactual shifts.
- **vs. BALSA/Funsearch**: Program search methods that are complementary to NewtonBench's equation discovery paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Counterfactual law shifts combined with an interactive discovery benchmark represent a wholly original contribution; the code tool paradox is a profound finding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 11 models, 12 domains, and multiple ablations, though improvement pathways for non-reasoning models are not explored.
- Writing Quality: ⭐⭐⭐⭐ Benchmark design motivation is clearly articulated and experimental analysis is in-depth.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous evaluation tool for LLM scientific discovery capabilities; the code tool paradox carries important implications for agent design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](membership_privacy_risks_of_sharpness_aware_minimization.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

</div>

<!-- RELATED:END -->
