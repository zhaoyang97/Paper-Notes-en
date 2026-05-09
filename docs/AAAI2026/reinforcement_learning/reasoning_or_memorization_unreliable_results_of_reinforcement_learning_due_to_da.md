---
title: >-
  [Paper Note] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination
description: >-
  [AAAI 2026][Reinforcement Learning][Data Contamination] This paper conducts a systematic data leakage audit revealing severe data contamination of the Qwen2.5 series on standard math benchmarks such as MATH-500. It demonstrates that recent findings claiming "spurious rewards can improve mathematical reasoning" are artifacts of contamination, and constructs a fully uncontaminated benchmark, RandomCalculation, to verify that only correct reward signals yield genuine reasoning improvements.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Data Contamination
  - LLM Reasoning
  - Spurious Rewards
  - RLVR
  - Math Reasoning Evaluation
date: 2026-05-08
content_hash: 7e5b4e1256d6a4a0
---

# Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

**Conference**: AAAI 2026
**arXiv**: [2507.10532](https://arxiv.org/abs/2507.10532)
**Code**: [github](https://github.com/wumingqi/LLM-Math-Evaluation)
**Area**: Reinforcement Learning
**Keywords**: Data Contamination, LLM Reasoning, Spurious Rewards, RLVR, Math Reasoning Evaluation

## TL;DR

This paper conducts a systematic data leakage audit revealing severe data contamination of the Qwen2.5 series on standard math benchmarks such as MATH-500. It demonstrates that recent findings claiming "spurious rewards can improve mathematical reasoning" are artifacts of contamination, and constructs a fully uncontaminated benchmark, RandomCalculation, to verify that only correct reward signals yield genuine reasoning improvements.

## Background & Motivation

Recent years have seen significant progress in using reinforcement learning (RL) to enhance the mathematical reasoning capabilities of LLMs. In particular, Reinforcement Learning with Verifiable Rewards (RLVR)—which assigns a reward of 1 when the predicted answer matches the ground truth and 0 otherwise—has attracted wide attention due to its independence from learned reward models.

**A puzzling phenomenon**: Several recent works report that even random or incorrect rewards improve MATH-500 performance on Qwen2.5-Math-7B. More extremely, using only 1 labeled example (1-shot-RL) or no labels at all (Absolute-Zero) yields substantial gains. Crucially, these "magical" effects are almost exclusively observed on the Qwen2.5 series and do not transfer to the Llama series—suggesting a model-specific issue.

**Two competing hypotheses**:

**Data contamination hypothesis**: Qwen2.5's large-scale pretraining corpus (up to 36T tokens) includes GitHub repositories containing benchmark questions and their official solutions. Spurious rewards merely "activate" memorized answers.

**Strong math ability hypothesis**: Qwen's pretraining confers stronger mathematical capabilities than Llama, so even noisy gradient updates improve MATH-500 performance. If true, however, spurious rewards should also be effective on clean benchmarks.

**Core mission of this paper**: Distinguishing between these two hypotheses requires simultaneous leakage auditing and rigorous out-of-distribution RLVR evaluation.

## Method

### Overall Architecture

The research framework proceeds in four progressive stages:
1. **Memorization audit**: Two novel metrics are proposed to detect the degree of data contamination in Qwen on standard benchmarks.
2. **Controlled replication**: The "success" of spurious rewards on MATH-500 is reproduced.
3. **Clean benchmark construction**: A RandomCalculation generator is designed to produce fully uncontaminated arithmetic problems.
4. **Causal verification**: RLVR is conducted on the clean benchmark to demonstrate that spurious rewards fail.

### Key Designs

#### 1. Data Contamination Detection: Two Novel Metrics

**Partial-Prompt Completion Rate**: The problem is truncated (retaining the first 40%/60%/80%), and the model is prompted to generate the remainder. ROUGE-L and exact match (EM) scores against the original problem measure the degree of match. High completion accuracy indicates prior exposure to the problem.

**Partial-Prompt Answer Accuracy**: Using the same truncated prompts, this metric checks whether the correct answer appears in the model's generated continuation. Producing a correct answer from a partial prompt is a strong indicator of data leakage.

**Striking findings**: Given the first 60% of a MATH-500 problem, Qwen2.5-Math-7B achieves a completion rate of 54.60% EM and an answer accuracy of 53.6%. In contrast, Llama3.1-8B achieves only 3.8% EM and 2.4% accuracy. A key validation: on the recent LiveMathBench (May 2025 edition), Qwen's completion rate drops to 0.0%, consistent with Llama, confirming that the difference is due to contamination rather than capability.

#### 2. RandomCalculation Benchmark Construction

An automated generator constructs fully uncontaminated arithmetic expressions:

- **Base elements**: Integers from 0 to 100, along with derived fractions, squares, and cubes.
- **Composition**: The four arithmetic operations are used to randomly generate expressions of 1 to 20 steps.
- **Dataset composition**: 20 sub-datasets, each containing 1,000 unique problems.
- **Key property**: All instances are generated after Qwen's release, guaranteeing zero contamination.

**Zero-shot testing validates absence of memorization**: Qwen2.5's accuracy on RandomCalculation decreases monotonically with the number of computation steps, with no memorization patterns observed.

#### 3. Continuous Reward Function for RandomCalculation

Since ground-truth answers to random arithmetic expressions frequently contain high-precision decimals, standard binary RLVR cannot provide effective positive feedback. A continuous reward function is designed:

$$r = 1 - \underbrace{0.5 \cdot \min(|a-b|, 1)}_{\text{absolute distance}} - \underbrace{0.5 \cdot \min\left(\frac{|a-b|}{|b| + \epsilon}, 1\right)}_{\text{relative distance}}$$

where $a$ is the model output, $b$ is the reference answer, and $\epsilon = 10^{-6}$.

#### 4. Gradient Analysis of Spurious Rewards: The Exploitation Bias Mechanism

The paper provides a theoretical explanation for why spurious rewards appear to "work" on contaminated data. The GRPO gradient is:

$$\nabla_\theta J_{\text{CLIP}} = \nabla_\theta r_{i,t} \cdot G(r_{i,t})$$

For high-probability tokens ($\pi_{\text{old}} = 0.85$), the upper clipping bound $1.02 \cdot \pi_{\text{old}} = 1.02$ exceeds the probability ceiling of 1.0, so the gradient is non-negative and high-probability tokens are continuously reinforced. Due to data contamination, correct-answer tokens on MATH-500 already carry high probabilities—thus, random rewards "retrieve" these memorized answers through GRPO's exploitation bias.

For medium-probability tokens ($\pi_{\text{old}} = 0.5$, typical of RandomCalculation), the clipping bounds $[0.4, 0.6]$ cause most gradients from random rewards to be clipped away, yielding $G(r_{i,t}) \approx 0$ and no meaningful improvement.

### Loss & Training

- **RL algorithm**: GRPO (Group Relative Policy Optimization)
- **Spurious reward types**:
    - Random: reward of 1 assigned with probability $\gamma = 0.5$
    - Inverted: the correct signal is flipped ($1 - \text{correct}$)
    - Mv-incorrect: incorrect labels produced by majority voting
- **Training configuration**: learning rate 5e-7, temperature 1.0, 16 sampled responses per prompt, batch size 128
- **Hardware**: 8 × NVIDIA A800 80G GPUs

## Key Experimental Results

### Main Results

**Data contamination detection (Greedy w/o Template)**:

| Model | Dataset | 80% Prompt EM | 60% Prompt EM | 40% Prompt EM |
|-------|---------|--------------|--------------|--------------|
| Qwen2.5-Math-7B | MATH-500 | **65.80%** | **54.60%** | **39.20%** |
| Qwen2.5-Math-7B | AMC | 55.42% | 42.17% | 36.14% |
| Qwen2.5-Math-7B | AIME2024 | 56.67% | 20.00% | 16.67% |
| Qwen2.5-Math-7B | AIME2025 | 16.67% | 0.00% | 0.00% |
| Qwen2.5-Math-7B | LiveMathBench | 5.00% | **0.00%** | **0.00%** |
| Llama3.1-8B | MATH-500 | 17.80% | 3.80% | 0.60% |

**RLVR performance on RandomCalculation**:

| Reward Type | MATH-500 (Qwen) | RandomCalc 5-step (Qwen) | RandomCalc 10-step (Qwen) |
|------------|----------------|------------------------|--------------------------|
| Correct reward | Consistent gain | Consistent gain, exceeds Max@16 | Consistent gain, exceeds Max@16 |
| Random reward | Substantial gain | Unstable, no reliable improvement | No improvement |
| Inverted reward | Slight degradation | Rapid collapse | Rapid collapse |

### Ablation Study

**Response similarity (ROUGE-L) before and after RL**:

| Dataset | Correct Reward | Random Reward | Mv-incorrect | Note |
|---------|---------------|--------------|-------------|------|
| MATH-500 | 0.555 | **0.601** | 0.563 | Spurious reward ≈ memory retrieval |
| RandomCalc 5-step | 0.225 | 0.247 | 0.251 | Low similarity = genuine reasoning |
| RandomCalc 10-step | 0.193 | 0.251 | 0.279 | More steps → less memorization |

**Template effect (Qwen base models)**:

| Configuration | Qwen2.5-7B Accuracy | Qwen2.5-Math-7B Accuracy | Note |
|--------------|--------------------|-----------------------|------|
| Greedy (w/o Template) | High | **72.20%** | Highest without template |
| Greedy (w/ Template) | Substantially lower | 50.60% | Template severely degrades performance |
| RLVR starting point (w/ Template) | ~35% | ~50% | True starting point underestimated |

### Key Findings

1. **Data contamination is the root cause of spurious reward "success" on Qwen**: After removing contamination, the "magic" of spurious rewards disappears.
2. **Only correct rewards genuinely improve reasoning ability**: On RandomCalculation, correct rewards enable Qwen to surpass the Max@16 upper bound, whereas spurious rewards cannot.
3. **Baseline underestimation due to template effects**: Qwen base model performance drops substantially after applying the chat template, meaning a portion of RLVR's apparent "gains" stems from adaptation to the template format.
4. **KL divergence provides further confirmation**: Token-level KL divergence before and after RL on MATH-500 is significantly lower than on RandomCalculation, indicating that RL on MATH-500 primarily exploits existing memorization.
5. **Correct rewards also fail for Llama**: Llama cannot surpass Max@16 on RandomCalculation even with correct rewards, confirming that Qwen genuinely has stronger mathematical training—but this is not the reason spurious rewards "succeed."

## Highlights & Insights

- **Highly timely research question**: Directly addresses the most contentious debate in RL for LLM reasoning, offering substantial value to the community.
- **Simple yet effective detection method**: The partial-prompt completion rate is a general-purpose data contamination detection approach applicable to other benchmarks.
- **Elegant design of RandomCalculation**: Controllable difficulty, guaranteed zero contamination, and support for continuous rewards make it an ideal evaluation tool for RL.
- **Theoretical analysis of GRPO exploitation bias**: Explains at the gradient level why high-probability tokens (memorized answers) are further reinforced under random rewards.
- **Important warning to the community**: Researchers evaluating RL methods must use uncontaminated benchmarks and test across multiple model families.

## Limitations & Future Work

1. **Computational constraints**: Only a subset of Qwen2.5 and Llama3.1 models are tested; broader model families (e.g., Gemma, Mistral) are not covered.
2. **RandomCalculation is relatively simple**: It involves only basic arithmetic and cannot fully represent mathematical reasoning ability (lacking algebra, geometry, probability, etc.).
3. **Contamination types are not distinguished**: The analysis does not differentiate among complete problem leakage, leakage of similar problem types, or leakage of solution patterns.
4. **Limited RL method coverage**: Only GRPO is tested; whether the same phenomenon occurs with PPO, REINFORCE, or other RL algorithms remains unverified.
5. **Generalizability of positive conclusions**: Whether the finding that "correct rewards are effective" holds for more complex reasoning tasks (e.g., theorem proving, code generation) requires further investigation.

## Related Work & Insights

- Directly responds to the conclusions of works such as Spurious Rewards by providing counter-evidence.
- The "successes" of TTRL (Test-time RL), Few-Shot-RL, and related methods on Qwen may all be influenced by data contamination.
- Implications for future research: RL for reasoning studies should routinely include: (1) multi-model-family evaluation, (2) clean benchmark validation, and (3) data leakage auditing.
- Benchmarks with timestamps postdating the pretraining cutoff, such as LiveMathBench and AIME2025, are more trustworthy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic demonstration of how data contamination produces spurious conclusions in RL research.
- Experimental Thoroughness: ⭐⭐⭐⭐ — A complete four-stage pipeline of detection, replication, construction, and verification, with in-depth gradient analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous argumentation, intuitive figures and tables, and clear conclusions.
- Value: ⭐⭐⭐⭐⭐ — Carries important methodological implications for the current RL for LLM reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [\[ICLR 2026\] Learning from Synthetic Data Improves Multi-hop Reasoning](../../ICLR2026/reinforcement_learning/learning_from_synthetic_data_improves_multi-hop_reasoning.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](../../ICLR2026/reinforcement_learning/textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)

</div>

<!-- RELATED:END -->
