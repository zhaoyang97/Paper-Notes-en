---
title: >-
  [Paper Note] Learning to Reason from Feedback at Test-Time
description: >-
  [ACL 2025][Test-Time Compute] This paper proposes the FTTT (Feedback at Test-Time Training) paradigm, which formulates the environment feedback utilization of LLMs during the inference phase as an optimization problem. It designs a learnable test-time optimizer, OpTune, achieving superior scalability and performance compared to existing feedback utilization methods across four reasoning datasets.
tags:
  - "ACL 2025"
  - "Test-Time Compute"
  - "Feedback Utilization"
  - "Test-Time Optimization"
  - "Reasoning Scalability"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 85587e4cb33eb05b
---

# Learning to Reason from Feedback at Test-Time

**Conference**: ACL 2025  
**arXiv**: [2502.15771](https://arxiv.org/abs/2502.15771)  
**Code**: Yes  
**Area**: Other  
**Keywords**: Test-Time Compute, Feedback Utilization, Test-Time Optimization, Reasoning Scalability, Reinforcement Learning

## TL;DR

This paper proposes the FTTT (Feedback at Test-Time Training) paradigm, which formulates the environment feedback utilization of LLMs during the inference phase as an optimization problem. It designs a learnable test-time optimizer, OpTune, achieving superior scalability and performance compared to existing feedback utilization methods across four reasoning datasets.

## Background & Motivation

**Background**: Large language models often fail to generate correct answers in a single attempt when dealing with complex reasoning tasks. Iterative interaction with the environment and utilizing feedback (such as error prompts, execution results, and grading signals) to gradually refine solutions is a critical strategy to improve success rates.

**Limitations of Prior Work**: Existing feedback utilization methods suffer from two primary issues. The first class of methods (e.g., concatenating all historical feedback into the prompt for refinement) leads to prompt length inflation as iterations increase, resulting in severe length generalization issues—the model's performance degrades or even collapses under excessively long inputs. The second class of methods (e.g., simple retry strategies) samples new answers independently each time, completely failing to utilize the feedback obtained from previous attempts, thereby wasting valuable error experiences.

**Key Challenge**: How to utilize historical feedback while avoiding input length inflation? An ideal solution should be able to "compress" and absorb feedback details into the model, allowing the model to efficiently utilize cumulative feedback experience within a limited context window.

**Goal**: (1) Propose a feedback utilization paradigm that does not require history concatenation; (2) Design a learnable module to optimize model behavior at test-time based on feedback.

**Key Insight**: The authors analogize feedback utilization to an optimization problem—each feedback acts like a gradient signal, indicating the direction in which the model should adjust. Similar to gradient descent in traditional optimization, a "test-time optimizer" can be designed to adjust the model's hidden states or parameters based on feedback signals.

**Core Idea**: Formulate test-time feedback utilization as an optimization problem (the FTTT paradigm) and train a lightweight test-time optimizer, OpTune, which iteratively updates the model's internal representations during inference based on each feedback without expanding the input length.

## Method

### Overall Architecture

The overall workflow of FTTT: Given a reasoning task, the LLM generates an initial solution $\rightarrow$ submits it to the environment to obtain feedback $\rightarrow$ OpTune (the test-time optimizer) updates the model's internal states based on the feedback $\rightarrow$ the LLM generates a new solution based on the updated state $\rightarrow$ repeat until success or the maximum number of iterations is reached. The crucial innovation lies in the fact that feedback is not concatenated to the input prompt, but rather "written" into the hidden states of the model via OpTune.

### Key Designs

1. **FTTT Paradigm—Formulating Feedback Utilization as an Optimization Problem**:

    - **Function**: Provides a unified framework to shift feedback utilization from "prompt concatenation" to "optimizing model states".
    - **Mechanism**: Defines the objective function to maximize the probability of generating the correct answer given the feedback sequence. In each iteration, the model generates a candidate answer $y_t$, the environment returns feedback $f_t$, and the optimizer converts the $(y_t, f_t)$ pair into update signals for the model's parameters or hidden states. Formally, it is analogous to SGD: $\theta_{t+1} = \theta_t + \text{OpTune}(y_t, f_t, \theta_t)$
    - **Design Motivation**: Traditional prompt concatenation couples the feedback utilization problem with the length generalization problem. Once formulated as an optimization problem, the accumulation of feedback information is no longer constrained by the context window.

2. **OpTune—A Learnable Test-Time Optimizer**:

    - **Function**: A lightweight neural network module that dynamically adjusts the parameters or hidden states of the LLM during test-time based on feedback signals.
    - **Mechanism**: OpTune takes the current model state and feedback signal as input and outputs a parameter update. It is trained to learn from mistakes—given a series of (attempt, feedback) pairs, OpTune should be able to infer the correct refinement direction. The training of OpTune is conducted on the training set by simulating test-time scenarios, using reinforcement learning or supervised learning signals.
    - **Design Motivation**: Handcrafted optimization rules find it difficult to adapt to different types of reasoning tasks and feedback formats. A learnable optimizer can adaptively process various feedback signals and learn effective "feedback compression" and "state update" strategies during training.

3. **Scalable Test-Time Compute Mechanism**:

    - **Function**: Allows continuous performance improvement by increasing the number of test-time iterations, achieving positive scaling in compute-performance.
    - **Mechanism**: Since each iteration only needs to process the current feedback (rather than all accumulated history), the compute cost scales linearly instead of superlinearly. Meanwhile, OpTune's update mechanism ensures that each iteration effectively utilizes new information, leading to monotonic performance gains with the number of iterations.
    - **Design Motivation**: Existing methods either saturate or degrade in performance after a certain number of iterations due to length inflation, or scale very inefficiently due to not utilizing historical information. FTTT combines the advantages of both.

### Loss & Training

The training of OpTune utilizes a two-stage strategy: first, collect (task, sequence of attempts, sequence of feedback, ground-truth answer) data on the training set; then, train OpTune to maximize the probability of generating the correct answer given the feedback sequence. This can be achieved using supervised learning (if ground-truth answers are available) or reinforcement learning (using task success rate as the reward).

## Key Experimental Results

### Main Results (Four Reasoning Datasets, Two LLMs)

| Method | Dataset 1 Success Rate | Dataset 2 Success Rate | Dataset 3 Success Rate | Dataset 4 Success Rate |
|------|-------------|-------------|-------------|-------------|
| Single Generation (No Feedback) | Baseline | Baseline | Baseline | Baseline |
| Prompt Concatenation (5 turns) | Moderate Gain | Moderate Gain | Moderate Gain | Moderate Gain |
| Prompt Concatenation (10 turns) | Performance Saturation/Degradation | Performance Saturation/Degradation | Performance Saturation/Degradation | Performance Saturation/Degradation |
| Simple Retry (5 turns) | Limited Gain | Limited Gain | Limited Gain | Limited Gain |
| FTTT + OpTune (5 turns) | **Significant Gain** | **Significant Gain** | **Significant Gain** | **Significant Gain** |
| FTTT + OpTune (10 turns) | **Continuous Gain** | **Continuous Gain** | **Continuous Gain** | **Continuous Gain** |

### Scalability Analysis

| Iterations | Prompt Concatenation | Simple Retry | FTTT + OpTune |
|---------|-----------|---------|---------------|
| 1 | Baseline | Baseline | Baseline |
| 3 | Improves then Plateaus | Slow Improvement | Steady Improvement |
| 5 | Starts to Saturate | Slow Improvement | Steady Improvement |
| 10 | Degradation | Near Saturation | Still Improving |
| 15+ | Obvious Degradation | Fully Saturated | Continues to Improve |

### Key Findings

- FTTT + OpTune is the only method that continuously improves performance with increasing iterations, validating its superior scalability.
- Prompt concatenation methods suffer from performance saturation or even degradation after roughly 5-10 turns due to length inflation.
- Although the simple retry method is not constrained by length, its scaling efficiency is extremely low because it does not utilize historical information.
- Experimental results are consistent across two LLMs of different scales, highlighting the generality of the FTTT paradigm.
- The computational overhead introduced by OpTune is relatively small, but the resulting performance improvement is significant.

## Highlights & Insights

- **Formulating test-time feedback utilization as an optimization problem**: This conceptual framework is highly elegant—it unifies seemingly distinct feedback utilization strategies (prompt concatenation, retry, refinement, etc.) under a "test-time optimization" framework, making it possible to analyze and improve feedback utilization using tools from optimization theory.
- **Decoupling feedback utilization from length generalization**: Prior methods either utilized feedback but were limited by length, or were free from length constraints but failed to utilize feedback. OpTune solves both problems simultaneously by "internalizing" feedback information into the model state. This idea can be transferred to any generation scenario that requires iterative optimization.
- **Paradigm potential of learnable optimizers**: The design of OpTune opens the door to the intersection of "meta-learning meets test-time compute". In the future, specialized test-time optimizers can be trained for different task types.

## Limitations & Future Work

- OpTune requires pre-collecting feedback data on a training set for training, which may encounter a cold-start issue for entirely new task types.
- The current method assumes that feedback information is structured (e.g., correct/incorrect signals, error types), and its capability to handle unstructured feedback (e.g., natural language critique) is unknown.
- The generalization capability of OpTune—whether an OpTune trained on one reasoning domain can transfer to other domains—requires further study.
- Integration with other test-time compute methods (e.g., Best-of-N, Tree Search, etc.) is worth exploring.

## Related Work & Insights

- **vs Self-Refine**: Self-Refine concatenates feedback into prompts, which is limited by length. FTTT avoids this issue through model state updates.
- **vs Best-of-N Sampling**: Best-of-N independently samples without utilizing feedback, whereas FTTT updates the model using historical feedback before each sampling.
- **vs Test-Time Training (TTT)**: TTT typically performs self-supervised training on the test data distribution, whereas FTTT is specialized in utilizing task feedback signals, making it more task-specific.
- The optimizer paradigm in this paper may inspire new designs for "adaptive reasoning" systems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulating feedback utilization as an optimization problem and designing a learnable optimizer is conceptually novel and elegantly executed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient validation on four datasets and two models, with convincing scalability analysis.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly explained, and formal definitions are accurate.
- Value: ⭐⭐⭐⭐⭐ Significantly advances test-time compute and feedback utilization paradigms, showcasing high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)
- [\[ACL 2025\] HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)
- [\[ACL 2025\] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](../../ICML2025/others/ranked_entropy_minimization_for_continual_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
