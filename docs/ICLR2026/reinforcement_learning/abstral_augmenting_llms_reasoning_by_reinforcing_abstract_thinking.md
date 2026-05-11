---
title: >-
  [Paper Note] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking
description: >-
  [ICLR 2026][Reinforcement Learning][abstract reasoning] This paper proposes AbstRaL, which uses reinforcement learning to teach LLMs to construct mathematical abstractions of reasoning problems (replacing concrete number…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "abstract reasoning"
  - "reinforcement-learning"
  - "GSM robustness"
  - "symbolic reasoning"
  - "distribution shift"
date: 2026-05-08
content_hash: 7104003ef257f817
---

# AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking

**Conference**: ICLR 2026
**arXiv**: [2506.07751](https://arxiv.org/abs/2506.07751)
**Code**: Available
**Area**: Reinforcement Learning
**Keywords**: abstract reasoning, reinforcement-learning, GSM robustness, symbolic reasoning, distribution shift

## TL;DR
This paper proposes AbstRaL, which uses reinforcement learning to teach LLMs to construct mathematical abstractions of reasoning problems (replacing concrete numbers/names with symbolic variables and extracting general formulas), then employs a symbolic solver to derive answers. AbstRaL nearly eliminates performance degradation caused by distribution shift on GSM perturbation benchmarks, and also yields implicit improvements on OOD mathematical and general reasoning tasks.

## Background & Motivation
**Background**: LLMs perform well on elementary mathematics benchmarks such as GSM, but exhibit significant performance drops under distribution shift (altered numbers, changed names, or inserted distractor conditions), exposing the brittleness of their reasoning.

**Limitations of Prior Work**: A common approach to improving robustness is to synthesize additional instantiated variants for data augmentation—yet this incurs high computational cost with limited gains. Alternative methods based on abstract reasoning (CoA, AoT) either rely on in-context learning (weak performance) or use SFT (producing unfaithful abstractions).

**Key Challenge**: The autoregressive objective of SFT forces the model to learn the specific surface context of each training instance, impeding the acquisition of instance-agnostic abstract thinking. A training paradigm is needed that directs the model's attention toward abstract structure rather than superficial context.

**Goal**: How can LLMs be trained to construct faithful mathematical abstractions such that reasoning becomes invariant to surface-level contextual perturbations in the input?

**Key Insight**: Rather than augmenting training data, the paper directly teaches LLMs the skill of "abstraction"—variabilizing problem conditions, performing symbolic reasoning, and computing answers via a solver. RL, rather than SFT alone, is used to ensure the faithfulness of the abstractions.

**Core Idea**: Use RL with fine-grained abstraction rewards to teach LLMs to "think abstractly"—transforming concrete reasoning problems into symbolic formulas before solving them.

## Method

### Overall Architecture
AbstRaL operates as a four-step pipeline: (1) **Condition Identification**—an LLM parses the numerical values and variables in the problem and assigns abstract symbols; (2) **Abstract Reasoning**—the LLM generates a reasoning chain and mathematical formulas using the abstract symbols; (3) **Abstraction Extraction**—formulas are extracted via regex matching; (4) **Symbolic Derivation**—the SymPy solver computes the final answer. Step 2 is the core learning target.

### Key Designs

1. **GranulAR Training Data**:

    - **Function**: Constructs fine-grained abstract reasoning training data.
    - **Mechanism**: Concrete values in existing Socratic CoT data are replaced with abstract symbols (e.g., `in0`, `out0`), preserving the reasoning structure. Llama-3.3-70B is used for rewriting, and SymPy subsequently verifies whether the rewritten formulas correctly derive the target answers.
    - **Design Motivation**: Embedding abstract reasoning within the CoT + step-by-step decomposition format that LLMs are already familiar with (close to the pretraining distribution) reduces the difficulty of learning.

2. **RL Abstraction Reward Design**:

    - **Function**: Two rewards, requiring no trained reward model, are used to reinforce abstract reasoning.
    - **Answer Correctness Reward $r_{answer}$**: SymPy verifies whether the model-generated abstraction $\tilde{\mathcal{A}}$ combined with the original conditions $\mathcal{C}$ derives the correct answer; a positive reward is given for success, and 0 otherwise.
    - **Symbolic Distance Reward $r_{symbolic}$**: The token-level edit distance between the generated abstraction $\tilde{\mathcal{A}}$ and the gold-standard abstraction $\mathcal{A}$ is computed and normalized to $[0,1]$, providing a finer-grained learning signal.
    - **Design Motivation**: The autoregressive objective of SFT causes the model to also learn surface context; the RL reward focuses exclusively on the correctness and faithfulness of the abstraction.
    - The GRPO algorithm is employed.

3. **Robustness of Symbolic Derivation**:

    - Once the model learns faithful abstractions, the symbolic formulas remain unchanged regardless of variations in input numbers or names, and SymPy solves them correctly.
    - Robustness to distractor conditions arises from the model learning to identify which conditions are relevant to the reasoning.

### Loss & Training
SFT on GranulAR data + GRPO with $r_{answer}$ + $r_{symbolic}$. Validated on Qwen2.5 and Llama3 series (0.5B–7B). Training data is constructed using Llama-3.3-70B.

## Key Experimental Results

### Main Results (GSM Robustness)

| Method | GSM-Symbolic Vary Both | Δ↓ | GSM-Plus Distract | Original |
|------|----------------------|---|-------------------|----------|
| CoT-8S (Qwen-0.5B) | 34.0 | 10.6 | 22.7 | 42.4 |
| CoT-RL | 32.3 | 7.77 | 15.2 | 38.0 |
| SyReLM | 36.8 | 5.54 | 21.1 | 41.5 |
| **AbstRaL** | **44.6** | **-1.27** | **25.3** | **46.3** |

### Key Findings
- **Δ < 0**: AbstRaL achieves *higher* performance on perturbed variants than on the original problems, demonstrating that abstraction not only eliminates distribution shift but also enhances baseline reasoning.
- On Qwen2.5-Math-7B, AbstRaL also significantly improves robustness, with the largest gains on GSM-Plus Distract—because abstraction naturally ignores distractor conditions.
- SFT-only (without RL) frequently produces unfaithful abstractions that are misaligned with the problem. RL corrects this via the reward signal.
- OOD transfer: AbstRaL yields zero-shot improvements on AIME (math competition) and BBH (general reasoning), indicating that abstract thinking generalizes across domains.

## Highlights & Insights
- **"Abstraction" is a more efficient strategy than "instantiation" for improving reasoning robustness**: Rather than synthesizing large numbers of variant instances, the model is directly taught to learn general patterns. By analogy, instead of training the model on more addition problems, it is taught the concept of "addition" itself.
- **The unique value of RL for abstraction learning**: SFT is inherently compelled to learn the surface context of each sample (an intrinsic limitation of the autoregressive objective), whereas the RL reward function can focus exclusively on the structural correctness of abstractions.
- **Fine-grained signal from the symbolic distance reward**: Rather than a binary correct/incorrect reward, the model receives a signal indicating "how close the abstraction is to correct"—accelerating learning convergence.

## Limitations & Future Work
- Validation is currently limited to GSM (elementary mathematics); abstraction for more complex problems (e.g., geometric reasoning, proof-based tasks) may be substantially more challenging.
- The condition identification step relies on few-shot prompting with a 70B model; the ability of smaller models to perform condition identification autonomously remains underexplored.
- SymPy's coverage of non-equation problems (e.g., combinatorics, probability) is limited.
- Training data quality depends on the rewriting quality of the oracle LLM.

## Related Work & Insights
- **vs. CoA / AoT (abstract reasoning methods)**: These methods perform abstraction via in-context learning, yielding weak performance. AbstRaL uses SFT+RL training and substantially outperforms them.
- **vs. data augmentation strategies**: Synthesizing additional instances requires large amounts of data and computation. AbstRaL learns abstraction directly from the same training set, making it more efficient.
- **vs. CoT-RL (standard RL)**: CoT-RL applies RL to standard GSM without learning abstractions, resulting in limited robustness gains. The abstraction component is the key differentiator in AbstRaL.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The principle of "teaching models to learn abstractions rather than more instances" is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models and scales, two robustness benchmarks, OOD transfer, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the framework diagram is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for reasoning robustness; the transferability of abstract thinking is particularly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[ICLR 2026\] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [\[AAAI 2026\] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction](../../AAAI2026/reinforcement_learning/thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](../../NeurIPS2025/reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)

</div>

<!-- RELATED:END -->
