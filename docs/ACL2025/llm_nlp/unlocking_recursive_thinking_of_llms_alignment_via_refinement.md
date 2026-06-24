---
title: >-
  [Paper Note] Unlocking Recursive Thinking of LLMs: Alignment via Refinement
description: >-
  [ACL2025][LLM (Other)][Recursive thinking] This work proposes AvR (Alignment via Refinement), a two-stage framework that leverages refinement-aware rewards and differential learning to equip LLMs with "critique $\rightarrow$ refinement" recursive thinking capabilities. Using only 10k data points, it improves the win rate of LLaMA-3-8B-Instruct on AlpacaEval 2 by over 26 percentage points.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Recursive thinking"
  - "Alignment"
  - "Refinement"
  - "Long-form CoT"
  - "Test-time scaling"
  - "DPO"
date: 2026-05-08
content_hash: 9422bcb4094fe1a6
---

# Unlocking Recursive Thinking of LLMs: Alignment via Refinement

**Conference**: ACL2025  
**arXiv**: [2506.06009](https://arxiv.org/abs/2506.06009)  
**Code**: [Banner-Z/AvR](https://github.com/Banner-Z/AvR)  
**Area**: LLM/NLP  
**Keywords**: Recursive thinking, Alignment, Refinement, Long-form CoT, Test-time scaling, DPO

## TL;DR

This work proposes AvR (Alignment via Refinement), a two-stage framework that leverages refinement-aware rewards and differential learning to equip LLMs with "critique $\rightarrow$ refinement" recursive thinking capabilities. Using only 10k data points, it improves the win rate of LLaMA-3-8B-Instruct on AlpacaEval 2 by over 26 percentage points.

## Background & Motivation

1. **Long-form CoT drives test-time scaling**: The OpenAI o1 series demonstrates that long-form CoT significantly enhances performance on complex tasks. However, most existing LLMs lack autonomous multi-turn correction capabilities, making it difficult to iteratively optimize outputs during inference.
2. **Traditional alignment neglects process signals**: Methods like DPO/RLHF only provide preference rewards for final outputs, lacking supervision over intermediate processes such as reflection and refinement. Consequently, models fail to learn from "where the improvement was made."
3. **Difficulty in self-correction**: Prior studies indicate that LLMs struggle to correct their own mistakes without external reward functions. Simply prompting models to refine can even degrade quality (with the baseline LLaMA-3's win rate dropping by 2.6% after refinement).
4. **High cost of o1-like approaches**: Methods based on MCTS or large-scale RL require powerful backbone models, massive sampling, and heavy training overhead, which is impractical in resource-constrained scenarios.
5. **Redundancy in parallel sampling**: Traditional preference optimization cannot distinguish differences in quality between generations, which easily leads to repeating similar errors during parallel sampling.
6. **Inspiration from differential learning**: Sutton's differential learning philosophy suggests that optimizing the reward difference between consecutive states can more effectively guide decision improvement. This naturally fits refinement scenarios where the goal is sequential improvement.

## Method

### Overall Architecture

- **Function**: A two-stage training framework—Stage I learns single-step refinement (multi-turn interactive), and Stage II internalizes recursive thinking into autonomous long-form CoT.
- **Design Motivation**: Stage I first enables the model to master the basic paradigm of "critique $\rightarrow$ refinement", and Stage II then employs trajectory distillation so the model can perform autonomous recursive reasoning without external prompts, achieving test-time scaling.
- **Mechanism**:
  1. Model the query and each turn of response as a multi-step MDP, defining the state transition as the concatenation of context.
  2. Introduce a refinement-aware reward $R(s_{t+1}, s_t)$ that requires positive modification in each step and superiority over the initial response.
  3. Discard trajectories that do not satisfy the conditions via rejection sampling.

### Key Designs

#### Stage I — Single-Step Refinement Optimization

- **Function**: Construct a refinement tree and score the initial and refined responses of each query using a Bradley-Terry reward model to obtain critique/refinement paired data.
- **Design Motivation**: Enable the model to learn to "improve progressively each time" rather than merely learning standard good/bad preferences; maximizing the reward difference before and after refinement via DPO is more precise than traditional DPO.
- **Mechanism**:
  1. Use Qwen2.5-32B as a corrector to generate critiques and refinements for the initial outputs of LLaMA-3-8B.
  2. RSFT: Select the refinement trajectories with the highest rewards from the refinement tree for supervised fine-tuning (10k samples).
  3. DPO: Construct preference pairs for the three behaviors—"generation", "critique", and "refinement". The generation step uses the best refinement vs. the original output; the critique and refinement steps select the highest/lowest-scoring pairs respectively.

#### Stage II — Multi-Step Recursive Thinking

- **Function**: Use the Stage I model to automatically synthesize recursive CoT trajectories, training the model to autonomously complete the entire loop of "generation $\rightarrow$ critique $\rightarrow$ refinement $\rightarrow$ re-critique $\rightarrow$ re-refinement $\rightarrow$ end".
- **Design Motivation**: Stage I still requires explicit prompts to drive each refinement step; Stage II internalizes recursive thinking, eliminating the reliance on external instructions and step-by-step supervision.
- **Mechanism**:
  1. Greedy search: Generate $x$ critiques $\times$ $y$ refinements in each turn (step=2 in experiments), selecting the optimal path using the BT model.
  2. Stop when no refinement is better than the current best response, then concatenate the steps into a complete recursive reasoning trajectory.
  3. Run RSFT to train the model to generate such trajectories, where the model uses `<think>` tags to wrap intermediate steps during inference.
  4. Length-controlled DPO: Sample 5 outputs and construct DPO pairs (4k pairs) using the highest-scoring shorter output vs. the lowest-scoring longer output, mitigating the length bias of the reward model.

## Key Experimental Results

### Main Results

| Method | Data Size | Win Rate | LC Win Rate |
|------|--------|----------|-------------|
| LLaMA-3-8B-Instruct (Seed) | - | 25.0% | 25.0% |
| DPO (Traditional) | 60k | 37.9% | 40.3% |
| SimPO | 60k | 40.5% | 44.7% |
| Meta-Rewarding Iter 4 | 20k | 39.5% | 39.4% |
| **AvR Stage I DPO + refine r2** | 20k | **50.8%** | 35.5% |
| **AvR Stage II RSFT** | 10k | **51.0%** | 42.5% |
| **AvR Stage II + Length Control** | 14k | 49.0% | **51.4%** |

**Key Findings**: AvR outscores SimPO and DPO (which require 60k data points) using only 10k-14k data. Incorporating length control achieves the highest LC Win Rate of 51.4%, gaining 26.4 percentage points over the Seed model.

### Experiment 2: Arena-Hard v0.1

| Method | Score | 95% CI |
|------|-------|--------|
| GPT-3.5-turbo | 23.3% | (-2.2, 1.9) |
| GPT-4-0613 | 37.9% | (-2.8, 2.4) |
| SimPO | 33.8% | - |
| Meta-Rewarding Iter 4 | 29.1% | (-2.3, 2.1) |
| **AvR Stage II** | **34.5%** | (-2.5, 2.3) |

**Key Findings**: AvR Stage II outperforms GPT-3.5-Turbo and all other LLaMA-3-8B-based baselines on Arena-Hard, using only 10k SFT data points.

### Experiment 3: Cross-Model Refinement Capability

The AvR Stage I model can be applied to refine the outputs of GPT-4o and GPT-4o-mini, yielding significant improvements in their AlpacaEval 2 scores. This demonstrates that the learned refinement capability generalizes across different models.

## Highlights & Insights

- **Extremely high data efficiency**: Just 3k samples yield a 20% win-rate improvement. The complete pipeline requires only 10k-14k data points, which is substantially lower than the 60k required by traditional RL methods.
- **Novel concept**: Introduces "differential learning" into alignment training, defining a refinement-aware reward to realize recursive thinking that aims to "improve at each step."
- **Elegant two-stage design**: Stage I externally boots the foundational ability, while Stage II internalizes it into autonomous CoT, progressively unlocking recursive reasoning.
- **Cross-model transferability**: The 8B model trained in Stage I can successfully enhance the outputs of GPT-4o, validating the value of refinement capability as a generalizable skill.

## Limitations & Future Work

- **Dependence on external reward models**: The entire pipeline heavily relies on a BT reward model (27B Skywork-Reward). Its biases (e.g., length bias) directly affect the quality of the synthesized data.
- **Evaluation limited to open-ended generation**: AlpacaEval and Arena-Hard focus primarily on chat quality, lacking validation on structured reasoning tasks such as mathematics or coding.
- **Stage I requires strong model guidance**: The initial stage relies on Qwen2.5-32B to generate critiques and refinement data, failing to be fully self-bootstrapped.
- **Trade-off in length control**: Although length-controlled DPO increases the LC Win Rate, the overall Win Rate drops by 2%. The trade-off between the two warrants further investigation.

## Related Work & Insights

### vs SCoRe (Kumar et al., 2024)
SCoRe is an online RL method that requires extensive online sampling and training on mathematics/coding benchmarks to enhance self-correction capabilities. AvR achieves comparable performance via offline synthesis of refinement data + DPO, offering significantly lower training costs while being tailored for open-ended generation rather than just reasoning tasks.

### vs Meta-Rewarding LLM (Wu et al., 2024)
Meta-Rewarding requires 4 iterations of training (aggregating 20k data points) to achieve a 39.5% win rate. In contrast, a single Stage II training run of AvR reaches a 51.0% win rate using only 10k data points. The core difference lies in AvR optimizing the "improvement process" rather than merely searching for better final responses.

### vs DeepSeek-R1 / o1-like Methods
o1-like approaches rely on large-scale RL paired with strong backbone models (e.g., 70B+), which is extremely costly. AvR achieves comparable recursive thinking effects on an 8B model with minimal data, presenting a cost-effective pathway.

## Rating

- Novelty: ⭐⭐⭐⭐ — Refinement-aware rewards and the two-stage recursive thinking framework present meaningful new concepts.
- Experimental Thoroughness: ⭐⭐⭐ — The AlpacaEval and Arena-Hard results are solid, but validation on reasoning-oriented tasks is lacking.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clear, with complete mathematical formulations and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — It provides a practical and cost-effective solution for small models to acquire recursive reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)

</div>

<!-- RELATED:END -->
