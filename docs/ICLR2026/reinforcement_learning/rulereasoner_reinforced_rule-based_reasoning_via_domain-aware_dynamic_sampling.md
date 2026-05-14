---
title: >-
  [Paper Note] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling
description: >-
  [Reinforcement Learning] RuleReasoner constructs a diverse rule reasoning dataset, RuleCollection-32K, and proposes a domain-aware dynamic sampling strategy (Dads). Under the RLVR framework…
tags:
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 2da0d0bb715bd30c
---

# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

- **Conference**: ICLR2026
- **arXiv**: [2506.08672](https://arxiv.org/abs/2506.08672)
- **Code**: To be released
- **Area**: LLM Reasoning / Reinforcement Learning
- **Keywords**: rule-based reasoning, RLVR, dynamic sampling, GRPO, domain reweighting, OOD generalization

## TL;DR

RuleReasoner constructs a diverse rule reasoning dataset, RuleCollection-32K, and proposes a domain-aware dynamic sampling strategy (Dads). Under the RLVR framework, an 8B model trained with this approach outperforms OpenAI-o1 by 4.1% on in-distribution reasoning tasks and by 10.4% on out-of-distribution tasks, while achieving approximately 1.4× training efficiency improvement.

## Background & Motivation

**Background**: Rule-based reasoning is a foundational capability in AI, spanning domains such as law, mathematics, and medical diagnosis. Large reasoning models (LRMs) have acquired long-chain thinking abilities through reinforcement learning (RL), yet they continue to face challenges from diverse rule formats, complex rule types, and combinatorial explosion in real-world scenarios.

**Limitations of Prior Work**: (1) Traditional approaches rely on model scaling or distillation from stronger models, which is costly and unsustainable. (2) As context windows expand, models exhibit the "lost in the middle" phenomenon, struggling to attend to relevant rules and facts. (3) Existing RLVR methods adopt coarse sampling strategies for multi-domain training data—static mixing leads to inter-domain imbalance, over-optimizing easy domains while under-optimizing difficult ones.

**Key Challenge**: The success of RLVR in mathematical and code reasoning has not transferred to rule reasoning—high-quality diverse training data is lacking, and the data scheduling problem for multi-domain training remains insufficiently studied.

**Goal**: To train a compact (4B/8B) specialized reasoning model that surpasses frontier LRMs (o1, R1) on rule reasoning tasks while improving training efficiency.

**Key Insight**: Improving RLVR along two dimensions simultaneously—**data** and **sampling strategy**: (1) constructing RuleCollection-32K, covering 8 categories of rule reasoning tasks; (2) designing Dads, a history-reward-based dynamic domain sampling algorithm that automatically schedules the proportion of each domain in training batches.

**Core Idea**: Replace static data mixing with domain-aware dynamic sampling (Dads)—at each training step, compute the "under-optimization degree" of each domain based on its historical rewards, dynamically increase the sampling weight of under-optimized domains, and realize adaptive online data scheduling.

## Method

### Overall Architecture

The RuleReasoner training pipeline:
1. Initialize standard RLVR training on RuleCollection-32K
2. At each training step: Rollout → compute per-domain average reward → update domain reward estimates via exponential moving average → compute domain weights via softmax → resample the next batch according to weights
3. Policy optimization using a GRPO variant (with KL term and entropy reward removed)
4. Rule order is randomly shuffled to prevent memorization

### Key Designs

#### 1. Domain-aware Dynamic Sampling (Dads)

**Function**: Automatically adjusts the sampling probability of each reasoning domain at every RLVR training step, prioritizing under-optimized domains.

**Mechanism**: For domain $d_i$, maintain an exponential moving average reward $\widetilde{r}_{s,d_i}$, compute its under-optimization degree $v_{s,d_i} = 1 - \widetilde{r}_{s,d_i}$ (the gap relative to the target reward of 1), and convert to sampling weights via softmax:

$$\widetilde{r}_{s,d_i} = \alpha \widetilde{r}_{s-1,d_i} + (1-\alpha) \bar{r}_{s,d_i}$$

$$w_{s,d_i} = \frac{\exp(v_{s,d_i}/\tau) + \epsilon}{\sum_{j=1}^n [\exp(v_{s,d_j}/\tau) + \epsilon]}$$

where $\alpha=0.5$ is a smoothing factor to prevent reward estimate fluctuation, $\tau=0.5$ controls weight sharpness, and $\epsilon=0.1$ guarantees a minimum sampling probability for each domain.

**Design Motivation**: Static data mixing cannot adapt to training dynamics—easy domains (e.g., ProntoQA) converge quickly and continued sampling wastes computation; difficult domains (e.g., AR-LSAT) require more training but are undersampled. Dads acts as an **online scheduler**, automatically redirecting computational resources from converged domains to under-optimized ones.

#### 2. RuleCollection-32K Dataset

**Function**: Construct a training dataset covering diverse rule formats, reasoning types, and complexity levels.

**Mechanism**: Built following five principles:
- **Varying depth**: 0–7-hop reasoning, supporting curriculum learning from simple to complex
- **Different formats**: Explicit rules (premises/constraints) and implicit rules (principles/context)
- **Multiple reasoning types**: Deductive, inductive, and analytical reasoning
- **Context dependency**: Rules must be adaptively applied in conjunction with specific questions, precluding mere memorization
- **Robust evaluation**: Boolean/multiple-choice questions preferred for precise assessment

Covers 8 tasks: Clutrr (inductive), ProntoQA/ProofWriter (deductive), FOLIO/LogicNLI (first-order logic), AR-LSAT/Logical Deduction/LogiQA (others).

#### 3. Training Regularization

**Function**: Prevent the model from identifying specific datasets or memorizing rule patterns.

**Mechanism**: Three regularization measures:
- **Remove entropy reward**: Avoids entropy explosion when bootstrapping without cold start
- **Remove KL divergence term**: The rule reward function eliminates distribution shift concerns; removing KL saves computation and encourages exploration
- **Shuffle rule order**: The order of context rules in each training sample is randomly permuted to prevent positional memorization

### Loss & Training

An exact-match-based **rule reward function** is used:

$$\mathcal{R}_{\text{EM}}(\hat{y}, y) = \begin{cases} 1 & \text{is\_equivalent}(\hat{y}, y) \\ -1 & \text{otherwise} \end{cases}$$

Policy optimization adopts the GRPO objective with the KL term and entropy reward removed, retaining only the clipped policy gradient. Strict on-policy training—only one gradient update is performed after each rollout.

## Key Experimental Results

### Main Results: In-distribution Pass@1 on 8 Tasks

| Method | Clutrr | ProntoQA | ProofWriter | FOLIO | LogicNLI | AR-LSAT | Log.Ded. | LogiQA | **Avg** |
|------|--------|----------|-------------|-------|----------|---------|----------|--------|---------|
| OpenAI o1 | 52.2 | 91.0 | 91.0 | 77.0 | 60.0 | 98.0 | 88.0 | 82.1 | 79.9 |
| Claude-3.7 | 65.7 | 92.8 | 90.0 | 74.7 | 58.0 | 76.2 | 97.0 | 81.5 | 79.5 |
| DeepSeek-R1 | 71.6 | 40.0 | 27.0 | 72.7 | 49.0 | 89.7 | 98.3 | 85.0 | 66.7 |
| DAPO | 86.5 | 96.0 | 94.8 | 80.9 | 65.8 | 40.0 | 95.3 | 74.6 | 79.2 |
| AdaRFT | 92.5 | 96.0 | 97.4 | 81.8 | 64.4 | 44.6 | 96.6 | 80.5 | 81.7 |
| **RuleReasoner-8B** | **95.5** | **96.4** | **97.0** | **84.7** | 70.4 | 46.8 | 98.3 | 83.5 | **84.0** |

> RuleReasoner-8B achieves an average accuracy of 84.0%, surpassing OpenAI-o1 (79.9%) by 4.1 percentage points, with the lowest variance across domains.

### Ablation Study: OOD Generalization

| Method | BBH | BBEH | ProverQA | **OOD Avg** |
|------|-----|------|----------|------------|
| Qwen3-8B (base) | 22.9 | 13.0 | 15.3 | — |
| SFT w/ Long CoT | 31.3 | 28.0 | 43.8 | 34.4 |
| GRPO | 35.5 | 24.3 | 34.1 | 31.3 |
| DAPO | 39.8 | 27.3 | 42.8 | 36.6 |
| OpenAI o1 | 46.4 | 33.5 | 52.5 | 44.1 |
| **RuleReasoner-8B** | **52.3** | **45.8** | **65.4** | **54.5** |
| **RuleReasoner-4B** | — | — | — | **54.5** (Δ+7.3 vs o1) |

> RuleReasoner-8B outperforms o1 by an average of 10.4 percentage points across three OOD benchmarks. Even the 4B variant achieves a 78.3% average across the three benchmarks.

### Key Findings

1. **Dads outperforms static curriculum learning**: Compared to data-balance RL (79.1%) and easy-to-hard RL (80.4%), Dads (84.0%) surpasses them by 3–5 percentage points on ID tasks—online scheduling substantially outperforms static ordering.
2. **Training efficiency**: RuleReasoner achieves equivalent OOD performance to DAPO with approximately 72 fewer steps (~1.4× speedup), without additional rollout computation.
3. **SFT vs. RLVR**: SFT approaches RLVR on ID tasks (81.9 vs. 84.0) but lags significantly on OOD tasks (34.4 vs. 54.5), confirming that **RL generalizes while SFT memorizes**.
4. **Emergence of meta-reflection**: After training, the model exhibits self-verification and logical consistency checking—self-correcting errors even on unseen rules.

## Highlights & Insights

- The design of Dads is elegant and minimalist: domain weights are estimated solely from historical rewards, requiring neither a proxy model nor human priors, making it a drop-in data scheduling plugin for RLVR.
- An 8B model surpassing frontier reasoning models such as o1 and R1 demonstrates that, in rule reasoning settings, **specialized data combined with intelligent scheduling** matters more than model scale.
- The construction principles of RuleCollection-32K (varying depth, multiple formats, context dependency) are worth adopting in other reasoning dataset designs.
- DeepSeek-R1 collapses on certain tasks (ProntoQA 40.0%, ProofWriter 27.0%), exposing the fragility of general-purpose reasoning models on structured rule reasoning.

## Limitations & Future Work

- Training data is constrained by currently available rule reasoning tasks and may not cover all rule formats and complexity levels found in natural language.
- Rule filtering quality is limited—noisy or redundant rules may degrade reasoning quality.
- Validation beyond 8B models is absent; larger-scale models may yield greater benefits but are computationally prohibitive.
- Evaluation is primarily based on exact match, which is insufficient for rule application scenarios requiring free-text output.

## Related Work & Insights

- **Logic-RL** (Xie et al., 2025): RLVR for logical reasoning, but targets "rule-free" reasoning, differing from this paper's "rules given" setting.
- **DAPO** (Yu et al., 2025): Improves RLVR efficiency via oversampling and filtering, but lacks fine-grained domain scheduling.
- **AdaRFT** (Shi et al., 2025): Curriculum-based sampling that depends on human priors or model success rate estimates; Dads is fully adaptive.
- **GRPO** (Shao et al., 2024): The base policy optimization algorithm upon which RuleReasoner adds domain-aware sampling.
- **Insight**: The "domain-aware dynamic sampling" paradigm is generalizable to any multi-domain RLVR training scenario, including mathematical reasoning and code generation.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Dads as a general data scheduling method for RLVR is meaningfully novel, though the core mechanism of softmax reweighting is not a fundamentally new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Baselines cover frontier LRMs, SFT, multiple RLVR variants, and curriculum learning; ablations are thorough; mean and standard deviation reported over 5 runs.
- **Value**: ⭐⭐⭐⭐ — The method is concise and efficient, directly integrable into existing RLVR pipelines.
- **Writing Quality**: ⭐⭐⭐⭐ — Algorithm pseudocode is clear, though the dense table layout makes certain sections visually heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)

</div>

<!-- RELATED:END -->
