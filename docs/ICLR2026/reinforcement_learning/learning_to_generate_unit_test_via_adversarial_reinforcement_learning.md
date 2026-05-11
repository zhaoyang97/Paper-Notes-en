---
title: >-
  [Paper Note] Learning to Generate Unit Test via Adversarial Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Unit Test Generation] This paper proposes UTRL, a framework that iteratively trains a unit test generator and a code generator via adversarial RL — the test generator learns to produce…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Unit Test Generation"
  - "Adversarial Training"
  - "RLVR"
  - "Self-Play"
  - "Discrimination Reward"
date: 2026-05-08
content_hash: c16dbea62d52137a
---

# Learning to Generate Unit Test via Adversarial Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2508.21107](https://arxiv.org/abs/2508.21107)
**Code**: [Project Page](https://dgjun32.github.io/UTRL)
**Area**: Code Generation / Reinforcement Learning
**Keywords**: Unit Test Generation, Adversarial Training, RLVR, Self-Play, Discrimination Reward

## TL;DR
This paper proposes UTRL, a framework that iteratively trains a unit test generator and a code generator via adversarial RL — the test generator learns to produce discriminative test cases that distinguish LLM-generated code from correct solutions, while the code generator learns to pass those tests. A Qwen3-4B model trained with UTRL surpasses GPT-4.1 in test generation quality.

## Background & Motivation

**Background**: Unit testing is a core software engineering practice. High-quality tests are used for best-of-N sampling and as reward functions in RLVR. LLMs have been applied to automate test generation, but methods for training LLMs to generate high-quality tests remain underdeveloped.

**Limitations of Prior Work**: (1) SFT requires annotated instruction–test pairs, which are expensive to obtain and difficult to scale across domains; (2) evaluating test quality is itself an open problem with no unique ground truth; (3) defining verifiable rewards for test generation is non-trivial — unlike code generation, there is no clear pass/fail signal.

**Key Challenge**: A method is needed to train a test generator without test annotations, yet defining a reward signal for "good tests" requires some form of reference standard.

**Key Insight**: Given a dataset of instruction–code pairs, the ability to distinguish LLM-generated code from correct solutions can serve as a proxy for test quality — good tests should expose bugs in LLM-generated code.

**Core Idea**: The test generator is rewarded for catching bugs in the code generator's outputs, while the code generator is rewarded for passing the tests. The two models co-evolve adversarially.

## Method

### Overall Architecture
Training alternates between two steps: Step 1 trains the test generator $\mathcal{M}_{\text{UT}}$ (maximizing a discrimination reward and a validity reward), and Step 2 trains the code generator $\mathcal{M}_{\text{code}}$ (maximizing test pass rate). Both share a Qwen3-4B backbone and are optimized with GRPO.

### Key Designs

1. **Discrimination Reward**:

    - Function: Measures how well the test suite $\mathcal{T}$ distinguishes LLM-generated code from the correct solution.
    - Mechanism: $R_{\text{disc}}(\mathcal{T}, \mathcal{C}, C^*) = \frac{1}{|\mathcal{C}|}\sum_{C \in \mathcal{C}}[1 - \prod_{T \in \mathcal{T}}(1-\text{Pass}(C,T))^{\text{Pass}(C^*,T)}]$. Invalid test cases (those failing the correct solution) are first filtered out; the reward then measures the proportion of LLM-generated solutions caught by at least one valid test.
    - Design Motivation: Good tests should expose bugs in LLM code; a higher discrimination rate implies greater discriminative power.

2. **Validity Reward**:

    - Function: Measures the functional correctness of test cases (correct input–output mapping).
    - Mechanism: $R_{\text{valid}}(\mathcal{T}, C^*, \tau) = \frac{\sum_{T} \text{Pass}(C^*, T)}{\max(|\mathcal{T}|, \tau)}$, where the denominator is clamped to $\tau$ to prevent a small number of test cases from achieving a high validity score.
    - Design Motivation: Prevents the generator from gaming the reward by producing only a handful of trivial test cases.

3. **Code Generator Training**:

    - Function: Trains the code generator to pass tests produced by the test generator.
    - Mechanism: $R_{\text{code}} = \frac{\sum_T \text{Pass}(C,T) \cdot \text{Pass}(C^*,T)}{\sum_T \text{Pass}(C^*,T)}$, computing the pass rate only over valid test cases.
    - Design Motivation: As the test generator evolves and produces harder tests, passing them drives continuous improvement in the code generator.

### Loss & Training
- Test generator: $r_{\text{UT}} = \lambda R_{\text{disc}} + (1-\lambda) R_{\text{valid}}$, optimized with GRPO.
- Code generator: $R_{\text{code}}$, optimized with GRPO.
- The two models are updated alternately in an iterative training loop.

## Key Experimental Results

### Main Results
Evaluated on the TACO benchmark (competitive programming) via Best-of-N improvement:

| Method | Model | Best-of-N Code Accuracy Gain | Test Fidelity |
|--------|-------|------------------------------|---------------|
| Base Qwen3-4B | 4B | 1× | Baseline |
| SFT (w/ GT tests) | 4B | Moderate | Moderate |
| SFT (w/ reasoning) | 4B | Moderate+ | Moderate+ |
| **UTRL** | **4B** | **3.1×** | **Highest** |
| GPT-4o | ~Trillion | Moderate | Moderate |
| **GPT-4.1** | ~Trillion | High | High |
| **UTRL (Qwen3-4B)** | **4B** | **Surpasses GPT-4.1** | **Surpasses** |

### Ablation Study

| Iteration | Discrimination Rate | Validity Rate | Notes |
|-----------|---------------------|---------------|-------|
| Round 0 | Baseline | Baseline | Untrained |
| Round 1 | Improved | Improved | Initial adversarial training |
| Round 2 | ↑ continued | ↑ continued | Sustained improvement |
| Round 3 | ↑ continued | ↑ continued | No saturation observed |

### Key Findings
- A 4B model trained with UTRL surpasses GPT-4.1 in test generation, demonstrating that adversarial RL matters more than model scale.
- UTRL requires no test annotations — only instruction–code pairs — making it substantially cheaper and more effective than SFT-based methods.
- The adversarially trained code generator achieves code quality comparable to models trained with ground-truth tests, confirming that the test generator provides an effective reward proxy.
- Iterative training continuously improves both models: as the code generator produces outputs closer to correct solutions, the test generator is forced to detect increasingly subtle bugs.

## Highlights & Insights
- **Elegant Design of the Discrimination Reward**: Rather than defining what makes a "good test," the framework requires only that tests discriminate between LLM-generated and correct code. This reframes test evaluation from an open-ended problem into a measurable discrimination task.
- **Natural Curriculum Learning**: As the code generator improves and its outputs approach correct solutions, bugs become more subtle, forcing the test generator to learn to cover harder edge cases. Adversarial training automatically induces an easy-to-hard curriculum.
- **4B Surpassing GPT-4.1**: Test generation is a niche scenario well-suited to UTRL — a 4B model, through adversarial training, substantially outperforms trillion-parameter models on this specific task, demonstrating the power of targeted RL.

## Limitations & Future Work
- Validation is limited to competitive programming (TACO); generalization to other programming domains (e.g., web, systems) remains to be explored.
- The code generator and test generator share the same backbone — using independent models may yield further gains.
- The discrimination reward depends on the code generator's output distribution — if the code generator is too weak, discrimination becomes trivial and provides insufficient learning signal.
- A direct and fair comparison with CURE is absent due to differences in base models.

## Related Work & Insights
- **vs. SFT methods (CodeRM/UTGEN)**: SFT requires test annotations; UTRL requires only code annotations and achieves superior performance.
- **vs. CURE**: CURE also uses RL but relies on annotated test datasets; UTRL requires no such annotations.
- **vs. AZR Self-Play**: AZR trains a model to pose and solve problems; UTRL trains one model to generate tests and another to write code — the analogy and distinction are clear.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Applying discrimination reward and adversarial training to test generation is a novel and effective idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple baselines, diverse evaluation metrics, and thorough iterative analysis.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear with complete pseudocode.
- Value: ⭐⭐⭐⭐⭐ — Direct practical value for code evaluation and automated testing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)

</div>

<!-- RELATED:END -->
