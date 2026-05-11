---
title: >-
  [Paper Note] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation
description: >-
  [ICLR 2026][Robotics][prompting] This paper proposes String Seed of Thought (SSoT), a concise prompting method that instructs LLMs to first generate a random string and then extract randomness from it to select an answer…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "prompting"
  - "probabilistic instruction following"
  - "diversity"
  - "LLM reasoning"
  - "randomness"
date: 2026-05-08
content_hash: 395183a5cc1d4da6
---

# String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation

**Conference**: ICLR 2026
**arXiv**: [2510.21150](https://arxiv.org/abs/2510.21150)
**Code**: None
**Area**: Robotics
**Keywords**: prompting, probabilistic instruction following, diversity, LLM reasoning, randomness

## TL;DR

This paper proposes String Seed of Thought (SSoT), a concise prompting method that instructs LLMs to first generate a random string and then extract randomness from it to select an answer. SSoT significantly improves distribution faithfulness in probabilistic instruction following (PIF) and response diversity in open-ended generation (DAG). The paper theoretically proves that TV distance decays exponentially with string length, and experiments show that reasoning-capable LLMs approach the performance of pseudo-random number generators.

## Background & Motivation

1. **LLMs exhibit systematic bias in probabilistic selection**: LLMs excel at deterministic single-answer tasks, but perform poorly when required to select answers according to a specified distribution. For instance, when asked to simulate a fair coin flip, LLMs often produce heavily skewed outputs rather than approximating a 50-50 split.

2. **Numerous real-world applications require probabilistic behavior**: Scenarios such as human behavior simulation, content diversification, and mixed strategies in game theory (e.g., the Nash equilibrium in rock-paper-scissors) all require the empirical distribution of LLM outputs to align with a target distribution, rather than converging to a single optimal answer.

3. **Response diversity is critical for test-time scaling**: Generating a large pool of candidate solutions and selecting the best is a core strategy in test-time scaling, yet LLM outputs tend to collapse into a limited set of answers, constraining candidate diversity.

4. **Existing debiasing methods have limited effectiveness**: Approaches such as temperature scaling, few-shot examples, and prompt ensembling partially mitigate bias but perform inconsistently on skewed-distribution tasks, and most require task-specific tuning.

5. **LLMs can describe distributions but cannot sample from them**: Prior work shows that LLMs accurately describe probability distributions yet lag significantly when asked to actually sample from them—a gap between knowing and doing.

6. **Long chain-of-thought in reasoning LLMs presents new opportunities**: Reasoning models such as DeepSeek-R1 and QwQ-32B produce extended chains of thought, providing a potential entropy source during inference for generating sufficient randomness.

## Method

### Core Idea

SSoT is remarkably simple—it adds a two-stage instruction to the prompt:

1. **Generate a random string**: The LLM is instructed to output a random string; this task-agnostic instruction produces sufficient entropy.
2. **Extract randomness from the string**: The LLM is instructed to manipulate the string (e.g., sum-mod, hashing) to derive the final answer.

For PIF tasks, the core instruction is "generate a random string, then manipulate it to sample from the target distribution"; for DAG tasks, it is "generate a random string, then manipulate it to produce a diverse response."

### Key Design Principles

- **Bias isolation**: Selecting answers directly from the prompt is susceptible to training biases such as option position and label frequency; SSoT decouples randomness generation from answer selection. The task-agnostic instruction "generate a random string" is less prone to selection bias.
- **Full parallelizability**: Each generation is independent and requires no history maintenance, unlike sequential sampling methods.
- **Unified prompt framework**: The same prompt framework applies to all PIF/DAG tasks, with the LLM autonomously selecting the optimal strategy.

### Theoretical Analysis

**Theorem 4.1 (2-universal Hash Function Bound)**: Assuming that the conditional probability of each character in the LLM-generated string is bounded ($\delta \leq P(x_i|\{x_j\}_{j<i}) \leq 1-(A-1)\delta$), when a 2-universal hash function is used to extract randomness, the TV distance satisfies:

$$d_{TV} \leq \frac{\sqrt{M}}{2\delta''} 2^{-\frac{n}{2}\log_2 \frac{1}{(1-(A-1)\delta)^2+(A-1)\delta^2}} + \sqrt{\frac{\ln((2^M-2)/\delta')}{K\phi(\pi_{P_X})}}$$

The first term decays exponentially with string length $n$; the second term represents finite-sample error.

**Theorem 4.2 (Sum-Mod Strategy Bound)**: When the LLM employs a sum-mod strategy (summing ASCII values of characters and taking the result modulo $M$), the TV distance likewise decays exponentially with string length, provided the marginal distribution of each character does not deviate severely from uniform.

### Autonomous Strategy Selection by LLMs

Analysis of chain-of-thought traces reveals that LLMs autonomously select strategies based on task complexity:
- **Sum-Mod**: Simple ASCII sum-mod for uniform distribution tasks.
- **Rolling Hash**: Automatically switches to more complex polynomial hashing ($\sum_i B^i \text{ord}(c_i)$ followed by mod) for skewed distribution tasks.
- **DAG tasks**: Creative categories automatically adopt a template-plus-local-sampling strategy; other categories use a list-plus-global-sampling approach.

## Key Experimental Results

### PIF Performance: Systematic Evaluation Across Five Frontier LLMs

| Model | Method | 2-choice | Biased 2-choice | 3-choice | Biased 3-choice | Biased 9-choice |
|------|------|:-:|:-:|:-:|:-:|:-:|
| deepseek-v3 | Baseline | 5.97 | 111.45 | 136.03 | 117.28 | 297.33 |
| deepseek-v3 | **SSoT** | **2.91** (↓51%) | **3.54** (↓97%) | **15.33** (↓89%) | **15.65** (↓87%) | **44.90** (↓85%) |
| deepseek-r1 | Baseline | 36.09 | 69.58 | 106.30 | 49.53 | 138.21 |
| deepseek-r1 | **SSoT** | **3.03** (↓92%) | **1.51** (↓98%) | **4.98** (↓95%) | **4.30** (↓91%) | **18.06** (↓87%) |
| QwQ-32B | **SSoT** | 3.39 | **2.47** (↓98%) | **1.82** (↓98%) | **1.30** (↓99%) | **11.48** (↓96%) |
| PRNG (ideal) | — | 1.85 | 1.93 | 3.36 | 2.85 | 13.72 |

(JS divergence ×10³, lower is better)

**Key Findings**: DeepSeek-R1 and QwQ-32B with SSoT achieve JS divergence close to that of a pseudo-random number generator (PRNG). Notably, QwQ-32B attains a JS divergence of only 1.30 on Biased 3-choice, even surpassing the PRNG baseline of 2.85.

### DAG Performance and Adversarial Game

| Method | NoveltyBench Overall (Distinct / Utility) |
|------|:-:|
| Baseline | 4.70 / 5.17 |
| Paraphrase | 5.63 / 5.57 |
| T=1.0 | 5.57 / 6.03 |
| **SSoT** | **6.19** / 5.92 |

SSoT achieves the highest Distinct score (6.19) and simultaneously improves both Distinct and Utility on the Creativity category. On the WildChat dataset, SSoT raises the Distinct score from 3.39 to 5.25 (+55%).

**Rock-Paper-Scissors Adversarial Experiment**: SSoT enables LLMs to achieve near-zero average scores against 10 "black-belt" rock-paper-scissors bots (approaching the ideal mixed-strategy equilibrium), whereas the Baseline and Simple Prompt are systematically defeated.

### CoT Scaling Analysis

Controlling reasoning length via budget forcing reveals that:
- As the number of thinking tokens increases, the uniformity of generated integers improves significantly (JS divergence decreases monotonically).
- Even under T=0 (fully deterministic decoding), longer reasoning chains produce strings of higher complexity (as measured by Lempel-Ziv complexity and zlib compression ratio, both of which increase with reasoning length).

## Highlights & Insights

- **Extreme simplicity**: Adding a single instruction to the prompt substantially improves probabilistic behavior, requiring neither training nor external tools.
- **Theoretical and empirical consistency**: Rigorous convergence guarantees for TV distance are proved, and experimental results closely align with theoretical predictions.
- **Autonomous strategy selection by LLMs**: The work reveals that LLMs can autonomously invent appropriate randomness extraction strategies (Sum-Mod vs. Rolling Hash) based on task complexity.
- **Reasoning-length scaling law for PIF**: This paper is the first to demonstrate that PIF performance scales with CoT length, offering a new dimension for understanding the probabilistic capabilities of reasoning models.

## Limitations & Future Work

- **Dependence on model reasoning capability**: Smaller models (below 8B parameters) may fail to correctly perform modular arithmetic or hashing operations, leading to degraded performance.
- **Risk of bias propagation**: If LLM-generated random strings exhibit strong positional bias and the model adopts a "lazy" strategy (e.g., using only the first character), the output distribution will be biased.
- **Inapplicability to single-answer tasks**: SSoT is designed for scenarios with multiple valid answers or probabilistic requirements; applying it to single-answer tasks such as mathematics or factual retrieval may distract the model.
- **Increased inference overhead**: Generating random strings and performing string manipulations extend CoT length and increase inference cost.

## Related Work & Insights

### vs. Prompt Ensemble
Prompt Ensemble uses 50 paraphrased prompts with randomized option ordering to reduce positional bias. It performs well on uniform-distribution PIF tasks but degrades markedly on skewed distributions—because eliminating positional bias alone is insufficient for precise distribution alignment. SSoT approaches ideal PRNG performance on both uniform and skewed settings, demonstrating strong robustness to distributional skew.

### vs. Few-shot Examples
Few-shot methods provide $k$ examples sampled from the target distribution ($k=3/10/50$), expecting LLMs to calibrate their output distribution through in-context learning. Experiments show that few-shot performance degrades rapidly as the number of actions increases (especially in biased settings), whereas SSoT maintains consistently low JS divergence across 2 to 64 choices, exhibiting far superior scalability.

### vs. Sequential Sampling
Sequential Sampling appends the history of previous selections to the prompt, expecting the LLM to adjust subsequent choices based on the accumulated distribution. This approach breaks independence across generations, precludes parallelization, and causes severe prompt bloat over long sequences. SSoT generates each response independently, naturally supporting parallel sampling.

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: The idea of "first generating a random string, then extracting randomness from it" as a prompting strategy is highly creative and opens a new direction for research on probabilistic behavior in LLMs.
- ⭐⭐⭐⭐ **Technical Quality**: Theoretical analysis is rigorous (two formal theorems); experiments cover five models, multiple task settings, and adversarial scenarios.
- ⭐⭐⭐⭐ **Practicality**: Zero-cost deployment; applicable to diverse scenarios including games, simulation, and content diversification.
- ⭐⭐⭐⭐ **Writing Quality**: Well-structured with a clear progression from theory to experiments to analysis; the CoT strategy analysis is particularly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)

</div>

<!-- RELATED:END -->
