---
title: >-
  [Paper Note] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
description: >-
  [NeurIPS 2025][Reinforcement Learning][RLVR] This work releases Reasoning Gym, a library of 100+ procedurally generated reasoning tasks spanning algebra, arithmetic, algorithms, logic, geometry, graph theory, games, and more. Each task supports infinite data generation and parameterized difficulty control. Experiments demonstrate that RLVR training achieves significant skill transfer both within and across domains, and improves performance on external benchmarks such as MATH and GSM8K.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - RLVR
  - verifiable rewards
  - procedural generation
  - curriculum learning
  - reasoning transfer
  - difficulty cliff
date: 2026-05-08
content_hash: 0114cd689a76e3aa
---

# Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2505.24760](https://arxiv.org/abs/2505.24760)
**Code**: [GitHub](https://github.com/open-thought/reasoning-gym/)
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: RLVR, verifiable rewards, procedural generation, curriculum learning, reasoning transfer, difficulty cliff

## TL;DR
This work releases Reasoning Gym, a library of 100+ procedurally generated reasoning tasks spanning algebra, arithmetic, algorithms, logic, geometry, graph theory, games, and more. Each task supports infinite data generation and parameterized difficulty control. Experiments demonstrate that RLVR training achieves significant skill transfer both within and across domains, and improves performance on external benchmarks such as MATH and GSM8K.

## Background & Motivation

**Background**: LLM reasoning capabilities have recently advanced substantially (o1, DeepSeek-R1, QwQ-32B), driven primarily by reinforcement learning with verifiable rewards (RLVR), which uses outcome-based feedback to guide models toward developing open-ended reasoning processes.

**Limitations of Prior Work**: (1) **Data bottleneck** — current RLVR relies on manually curated question–answer pairs or web-scraped content, which is costly, unsustainable, and increasingly scarce as reasoning models improve. (2) **Memorization** — fixed datasets risk models memorizing answers rather than learning to reason. (3) **Uncontrollable difficulty** — existing benchmarks do not support on-demand difficulty adjustment or curriculum learning. (4) **Reliance on human judgment** — correctness evaluation for some reasoning tasks requires subjective assessment.

**Key Challenge**: The success of RLVR hinges on large quantities of high-quality, automatically verifiable training data, yet this is precisely the scarcest resource available. Fixed datasets simultaneously impose a scale ceiling, memorization risk, and difficulty rigidity.

**Goal**: To construct a reasoning task library that supports procedural generation of unlimited training data, automatic verification, and parameterized difficulty control, fundamentally addressing the data bottleneck of RLVR.

**Key Insight**: Reasoning tasks are designed as *environments* rather than *datasets* — each task is a generation algorithm whose parameters control problem properties and whose verifier automatically assesses answer correctness, analogous to the environment concept in reinforcement learning.

**Core Idea**: Construct 100+ verifiable reasoning environments via procedural generation, supporting unlimited data, dynamic difficulty, and automatic verification to provide scalable infrastructure for RLVR training.

## Method

### Overall Architecture
Reasoning Gym (RG) is organized into three layers: (1) **Task generators** — each reasoning task is a parameterized generation algorithm capable of producing an unlimited number of distinct problem instances; (2) **Verifiers** — each task is equipped with an automatic verifier that determines whether model outputs are correct and provides binary rewards; (3) **Parameter control** — each task exposes three categories of parameters: difficulty parameters (controlling complexity), structural parameters (controlling problem properties), and stylistic parameters (altering presentation without affecting difficulty).

Task domains include: mathematics (algebra, arithmetic, geometry), algorithmic thinking (search, optimization, procedures), logical reasoning (formal proofs, inference rules), pattern recognition (sequences, visual analogies), and constraint satisfaction (games, puzzles, planning).

### Key Designs

1. **Procedural Generation and Algorithmic Verifiability**:

    - Function: Eliminates dataset size limits and memorization risk.
    - Mechanism: Each task is not a fixed collection of question–answer pairs but a generation function — given a seed and parameters, it produces a new problem instance. All tasks have deterministic correctness verification algorithms requiring no human judgment. The solution space is sufficiently large to make reward hacking difficult.
    - Design Motivation: Procedural generation fundamentally resolves three issues: (1) unlimited data scale — no dataset ceiling exists; (2) no memorization — every instance is novel; (3) sustainability — no dependence on manual annotation or web scraping.

2. **Parameterized Difficulty Control**:

    - Function: Supports dynamic curriculum learning and fine-grained capability diagnosis.
    - Mechanism: Difficulty parameters directly control problem complexity (e.g., number of graph nodes, polynomial degree, word length); structural parameters control problem properties (e.g., dimensionality, constraint types, proof depth); stylistic parameters vary presentation (e.g., variable names, number formats, phrasing) without affecting difficulty.
    - Design Motivation: Parameterization allows researchers to precisely control experimental variables — for example, varying only the number of graph nodes to study model performance at different scales, or varying only the representation to test whether a model genuinely understands a problem.

3. **Multi-Domain Coverage and Standardized Evaluation**:

    - Function: Comprehensively evaluates diverse reasoning capabilities.
    - Mechanism: Covers broad categories including algebra, arithmetic, geometry, algorithms, logic, pattern recognition, and constraint satisfaction, each containing multiple concrete tasks. Each task provides easy and hard parameter configurations for standardized evaluation.
    - Design Motivation: Reasoning ability is not unidimensional — a model may excel at mathematics but struggle with spatial reasoning. Broad coverage enables researchers to identify specific model strengths and weaknesses.

### Loss & Training
GRPO is used for RLVR training. Rewards consist of an accuracy reward (1.0) and a format reward (0.2). A curriculum learning strategy automatically advances the difficulty level when model accuracy exceeds 70% over 20 consecutive training steps.

## Key Experimental Results

### Main Results (Zero-Shot Evaluation on Hard Config)

| Model | Overall Accuracy | Type |
|-------|-----------------|------|
| o3-mini | 63.5% | Reasoning-optimized |
| DeepSeek-R1 | 59.5% | Reasoning-optimized |
| Grok 3 Mini | 55.1% | Reasoning-optimized |
| Llama 4 Maverick | 41.5% | General-purpose |
| Claude 3.5 Sonnet | 40.3% | General-purpose |
| Gemma 3 27B | 20.3% | General-purpose |

### Cross-Domain Transfer (Qwen2.5-3B-Instruct after training)

| Train Domain → Test Domain | Gain | Note |
|---------------------------|------|------|
| Algorithmic → Algebra | +29.1% | Procedural reasoning transfers to algebra |
| Algorithmic → Geometry | +22.3% | Procedural reasoning transfers to geometry |
| Games → Algebra | +21.8% | Constraint satisfaction transfers to algebra |
| Logic → Cognition | +13.3% | Logical reasoning transfers to cognition |
| RG-Math → MATH benchmark | +9.7% | Transfer to external benchmark |
| RG-Math → Big-Bench Hard | +7.66% | Substantial gains on external benchmark |

### Ablation Study: Curriculum Learning vs. Fixed Difficulty

| Task | Fixed Difficulty | Curriculum Learning | Note |
|------|-----------------|---------------------|------|
| Spell Backwards | Baseline | Higher | Progressive difficulty effective |
| Count Primes | Baseline | Comparable | Model unable to break initial difficulty |
| Most tasks | Baseline | Superior or equal | Curriculum learning broadly beneficial |

### Key Findings
- **~22% capability gap between reasoning and general-purpose models**: The best reasoning model (63.5%) substantially outperforms the best general-purpose model (41.5%), indicating that RLVR induces a capability leap rather than a marginal improvement.
- **Difficulty cliff phenomenon**: Transitioning from easy to hard configurations causes performance to drop by 62% on coding tasks, 30% on graph tasks, and 33% on geometry tasks, suggesting that current models possess brittle rather than robust reasoning capabilities.
- **Significant cross-domain transfer**: Models trained on algorithmic tasks improve by 29% on algebra and 22% on geometry, indicating that RLVR develops transferable reasoning primitives rather than domain-specific pattern matching.
- **Emergent capability from zero**: Games-category tasks yield a baseline accuracy of 0%, which rises to 3.3% after RLVR — while the absolute value is modest, this demonstrates that RLVR can bootstrap entirely novel reasoning capabilities.
- **Effective transfer to external benchmarks**: RG-Math training improves MATH by 9.7% and Big-Bench Hard by 7.66%, with gains across multiple MMLU-Pro subjects, confirming transferability to real-world scenarios.

## Highlights & Insights
- The **"environment rather than dataset" design philosophy** shifts the paradigm for reasoning training: datasets have ceilings, environments do not. This conceptual contribution alone may inspire more researchers to design reasoning environments rather than collect reasoning data.
- The **cross-domain transfer results** are among the most striking findings: models trained on algorithmic tasks improve by 20%+ on algebra and geometry, suggesting that different reasoning domains share underlying reasoning primitives (e.g., decomposition, backtracking, verification) and that RLVR can develop these primitives.
- The **systematic documentation of the difficulty cliff phenomenon** provides important evidence for understanding the capability boundaries of current reasoning models — models are more likely to have acquired solution templates within a specific complexity range than genuine reasoning strategies.

## Limitations & Future Work
- Current tasks emphasize reasoning with clear-cut answers (e.g., mathematics, graph algorithms); coverage of open-ended reasoning (e.g., strategic planning, creative thinking) is limited.
- Procedurally generated tasks do not necessarily cover the distribution of naturally occurring reasoning problems — real-world problems may exhibit structures that generators cannot simulate.
- Verifiers are limited to exact matching and cannot provide fine-grained feedback (e.g., process rewards) for partially correct reasoning trajectories.
- Experiments are conducted only on 3B-scale Qwen models; RLVR effects and transfer patterns may differ at larger scales.

## Related Work & Insights
- **vs. DeepSeek-R1 RLVR data**: DeepSeek-R1 applies RLVR to fixed mathematics and programming datasets; RG provides a broader, more controllable alternative across diverse domains.
- **vs. MATH/GSM8K**: These are fixed evaluation sets; RG supports both training and evaluation with far greater task diversity than any single mathematics benchmark.
- **vs. ARC Benchmark**: ARC is a fixed visual reasoning benchmark; RG includes ARC-style tasks in text format with the added capability of procedural generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of "reasoning environments" and its large-scale implementation represent a clear contribution, though procedural generation of test data is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Zero-shot evaluation, within-domain and cross-domain transfer, external benchmark transfer, and curriculum learning ablations are comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐⭐ Experimental findings are organized with exceptional clarity; each experiment is accompanied by a well-defined research question and takeaway.
- Value: ⭐⭐⭐⭐⭐ As open-source infrastructure, this work directly accelerates RLVR research; the cross-domain transfer findings offer important insights into the nature of reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](../../ICLR2026/reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](generalizing_verifiable_instruction_following.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)

</div>

<!-- RELATED:END -->
