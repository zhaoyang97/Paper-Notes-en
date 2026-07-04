---
title: >-
  [Paper Note] Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors
description: >-
  [ACL 2025][Reasoning][dialogue tutoring] This work proposes the **Traver** (Trace-and-Verify) agent workflow, which explicitly estimates student knowledge states via **knowledge tracing** and scores and selects optimal candidate tutoring utterances using a **turn-by-turn verifier**. To evaluate this, the **Dict** automated evaluation protocol (simulated student + code generation test) is introduced. In programming tutoring scenarios, it improves the student pass rate from 38.…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "dialogue tutoring"
  - "coding tutor"
  - "knowledge tracing"
  - "turn-by-turn verifier"
  - "reward model"
  - "inference-time scaling"
date: 2026-05-08
content_hash: 433ffe529f7858c5
---

# Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors

**Conference**: ACL 2025  
**arXiv**: [2502.13311](https://arxiv.org/abs/2502.13311)  
**Code**: [iwangjian/Coding-Tutor](https://github.com/iwangjian/Coding-Tutor)  
**Area**: LLM Inference / Dialogue Tutoring  
**Keywords**: dialogue tutoring, coding tutor, knowledge tracing, turn-by-turn verifier, reward model, inference-time scaling

## TL;DR

This work proposes the **Traver** (Trace-and-Verify) agent workflow, which explicitly estimates student knowledge states via **knowledge tracing** and scores and selects optimal candidate tutoring utterances using a **turn-by-turn verifier**. To evaluate this, the **Dict** automated evaluation protocol (simulated student + code generation test) is introduced. In programming tutoring scenarios, it improves the student pass rate from 38.7% to 43.7% (a relative gain of 106.5%), significantly outperforming Vanilla Instruct, Self-Refine, and TreeInstruct.

## Background & Motivation

**Background**: LLM-driven Intelligent Tutoring Systems (ITS) have been widely applied to knowledge-transfer scenarios such as language learning, mathematical reasoning, and science concept education. However, prior works focus primarily on "explaining concepts / recommending exercises," and their evaluations rely heavily on closed-ended QA or multiple-choice questions.

**Limitations of Prior Work**:
   - Existing tutoring systems are **passive and reactive**—answering only what the student asks, lacking target-driven proactive guidance capabilities.
   - LLMs do not understand what students "already know versus what they lack," which prevents personalized adaptation.
   - Multi-turn tutoring dialogues lack a turn-by-turn quality control mechanism, making LLMs prone to generating ineffective or redundant tutoring utterances.
   - Tutoring outcomes are difficult to evaluate automatically; human evaluation is costly and irreproducible.

**Key Challenge**: Programming tutoring is a **target-driven + open-ended** task—the tutor must proactively guide students of varying knowledge foundations to complete a predefined coding task rather than simply giving them the answer. This requires epistemic grounding (mastering domain knowledge and assessing the student's level) and communicative grounding (adapting communication strategies to the student's current state), both of which are major shortcomings of existing LLMs.

**Goal**: How to equip LLM tutoring strength with knowledge-state awareness and turn-by-turn quality control capabilities, thereby effectively guiding students of different proficiency levels to complete complex programming tasks.

**Key Insight**: Introduce Knowledge Tracing (KT) into dialogue tutoring as epistemic grounding, and introduce value-guided search (process verifiers) into multi-turn dialogues as communicative grounding, allowing the two to work collaboratively.

**Core Idea**: For each turn, first estimate the student's knowledge state $\rightarrow$ generate multiple candidate utterances based on this estimate $\rightarrow$ use a trained turn-by-turn verifier to score and select the optimal utterance, achieving test-time compute scaling.

## Method

### Overall Architecture: Traver (Trace-and-Verify)

Traver is an agent workflow comprising two core components that work collaboratively:

1. **Knowledge Tracing (KT)**: Before each dialogue turn, explicitly estimate the student's mastery level of each knowledge component (KC) based on the dialogue context and the previous belief state, driving the tutor to focus on student knowledge gaps.
2. **Turn-by-Turn Verifier**: Generate $N$ candidate utterances for each turn, score each candidate using a trained verifier, and select the utterance with the highest reward.

Inference Flow: `KT estimates student state → Prompt tutor to focus on knowledge gaps → Sample N candidate utterances in parallel → Verifier scores and ranks them → Output the optimal utterance`

### Key Designs

#### Key Design 1: Student State Modeling Based on Knowledge Tracing

- **Function**: Decompose task-specific knowledge $\mathcal{K}$ into a set of Knowledge Components (KCs), including reference dependencies (cross-file/class/function dependencies) and problem-solving steps. Each turn, assess the student's mastery belief $B_t$ for each KC based on the dialogue context.
- **Mechanism**: Implement textualized knowledge tracing via LLM prompting—instructing the tutor to infer the current belief $B_t$ based on the dialogue history and the previous belief $B_{t-1}$ at each turn, and then prompting the tutor to prioritize generating tutoring utterances for unmastered KCs.
- **Design Motivation**: The core of effective tutoring is "knowing where the student is." While knowledge tracing (e.g., BKT) has proven effective in traditional ITS, this mechanism is entirely absent in LLM-based dialogue tutoring. Integrating KT via prompting avoids additional training while providing explicit personalization signals for the tutor.

#### Key Design 2: Turn-by-Turn Verifier

- **Function**: Train a Mistral-7B-based verifier $V_\theta$ to produce a reward score $v_t \in [0,1]$ for each candidate tutoring utterance, reflecting the degree to which the utterance advances the tutoring objective.
- **Mechanism**: Define the turn-level reward as an iterative combination of cumulative progress and current contribution:
  $$v_t = \max(v_{t-1} + w_{r_t}, 0)$$
  where the weighted reward is:
  $$w_{r_t} = \frac{1 - v_{t-1}}{d_t + 1}(2o_{s_t} - 1)$$
  Here, $d_t = T - t$ represents the remaining number of turns (guiding distance), and $o_{s_t} \in \{0,1\}$ is a binary label indicating whether the utterance in this turn helps the student eventually complete the task. As the dialogue progresses, the remaining turns decrease, giving later turns higher weights.
- **Design Motivation**: Inspired by process reward models (such as Math-Shepherd, Let's Verify Step by Step), but existing PRMs target static reasoning chains and cannot handle interactive multi-turn dialogues. This design incorporates guiding distance into reward calculation, ensuring that $v_t$ is bounded while assigning higher weights to key later turns.

### Loss & Training

- **Verifier Training**: Train the verifier using MSE loss:
  $$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\sum_{t=1}^{T_i}(V_\theta([\mathcal{T}^{(i)};\mathcal{C}_t^{(i)};r_t^{(i)}]) - v_t^{(i)})^2$$
- **Training Data**: Synthesize tutoring dialogues using Vanilla Instruct with various backbone LLMs, where the post-test Pass score provides the outcome reward labels $o_{s_t}$.
- **Model Architecture**: Mistral-7B with an additional linear layer, outputting a $[0,1]$ reward score.
- **Inference Strategy**: A plug-and-play module that samples $N$ candidates in parallel at each turn and selects the one with the highest reward. Scaling $N$ from 1 to 20 continuously improves performance (inference-time scaling).
- **Evaluation Protocol Dict**: Simulates three tiers of students (Low/Medium/High). It automates evaluation through a pre-test $\rightarrow$ tutoring dialogue $\rightarrow$ post-test (code generation + unit tests) pipeline. A cognitive load truncation (retaining at most $M=60$ words per turn) is introduced to simulate realistic memory limitations.

## Key Experimental Results

### Main Results (Table 1): Comparison of Tutoring Performance of Different Methods

| Method | Backbone | Recall | Δ%R | Pass | Δ%P |
|------|----------|--------|-----|------|-----|
| Pre-Test (No Tutoring) | – | 45.9 | – | 21.2 | – |
| Vanilla Instruct | Qwen2-7B | 56.4 | 22.8 | 26.4 | 24.5 |
| Vanilla Instruct | Llama-3.1-70B | 62.5 | 36.0 | 34.9 | 65.0 |
| Vanilla Instruct | GPT-4o | 64.2 | 39.9 | 38.7 | 82.8 |
| Vanilla Instruct | o1-mini | 61.3 | 33.4 | 35.9 | 69.4 |
| Self-Refine | GPT-4o | 64.0 | 39.5 | 40.6 | 91.7 |
| TreeInstruct | GPT-4o | 64.3 | 40.1 | 39.8 | 88.1 |
| **Traver (Ours)** | **Llama-3.1-70B** | **66.8** | **45.5** | **39.3** | **85.7** |
| **Traver (Ours)** | **GPT-4o** | **68.8** | **49.8** | **43.7** | **106.5** |
| Oracle (Upper Bound) | – | 74.0 | 61.2 | 51.9 | 144.8 |

Note: Δ%R/Δ%P indicates the relative Turn-on-Post-test Rate (TOR) of improvement before and after tutoring; higher values indicate more effective tutoring.

### Ablation Study (Table 2)

| Configuration | Backbone | Pass | Δ%P |
|------|----------|------|-----|
| Traver (Full) | Llama-3.1-70B | 39.3 | 85.7 |
| w/o Knowledge Tracing | Llama-3.1-70B | 35.8 | 69.2 |
| w/o Verifier | Llama-3.1-70B | 35.1 | 65.8 |
| Traver (Full) | GPT-4o | 43.7 | 106.5 |
| w/o Knowledge Tracing | GPT-4o | 41.7 | 97.1 |
| w/o Verifier | GPT-4o | 39.8 | 88.2 |

### Human Evaluation (Table 3)

| Comparison | Proactiveness Win/Lose | Adaptability Win/Lose | Coherence Win/Lose |
|------|----------------|----------------|----------------|
| Traver vs Vanilla | 42.2/26.7 | 40.0/33.3 | 24.4/25.6 |
| Traver vs Self-Refine | 38.9/26.7 | 41.1/30.0 | 27.8/18.9 |
| Traver vs TreeInstruct | 34.4/22.2 | 37.8/25.6 | 28.9/20.0 |

Fleiss' κ = 0.45 (moderate agreement); Traver significantly outperforms all baselines in proactiveness and adaptability.

### Key Findings

1. **Verifier Contribution > KT**: Removing the Verifier drops Δ%P by approximately 20 percentage points (Llama-70B: 85.7 $\rightarrow$ 65.8), while removing KT drops it by about 16 percentage points (85.7 $\rightarrow$ 69.2), demonstrating that turn-by-turn verification is the core driver.
2. **Inference-Time Scaling is Effective**: As candidate count $N$ increases from 1 to 20, the Pass rate monotonically rises from 35.1% to 39.3%. Furthermore, Traver's scaling curve outperforms both the CoT-selection and random-selection baselines.
3. **Greatest Benefit to Low-Level Students**: Traver (GPT-4o) achieves a Δ%P of 242.5% for Low-level students, compared to only 44.8% for High-level students. This aligns with intuition, as students with a weaker foundation have more room for improvement.
4. **Small Models Can Tutor but Make Mistakes**: Qwen2-7B and GPT-3.5 yield Δ%P < 0 for high-level students, suggesting that weaker models might "do more harm than good" and mislead students who already have some foundation.
5. **Existing Methods are Far From Saturated**: Even the best configuration Traver (GPT-4o) with a Δ%P of 106.5% is still far below the Oracle's 144.8%, indicating substantial room for improvement in coding tutoring.

## Highlights & Insights

- **Paradigm Shift from "Answering" to "Tutoring"**: Shifting LLMs from passive QA roles to target-driven proactive tutors is a valuable formulation in itself. The use of coding tutoring as a testbed for open-ended task tutoring is well-designed.
- **Extending Process Reward from Reasoning to Dialogues**: Creatively transferring step-level verification to turn-level dialogue verification. The introduction of guiding distance successfully addresses the key challenge of credit assignment in multi-turn dialogues.
- **Exquisite Design of Dict Evaluation Protocol**: Simulating three tiers of students + cognitive load truncation + pre/post-test comparison + code unit testing forms a self-consistent and reproducible automatic evaluation loop.
- **First Verification of Inference-Time Scaling in Dialogues**: Proving that the best-of-$N$ + verifier strategy remains effective in multi-turn interactive scenarios, opening up a new direction for test-time compute in dialogue systems.

## Limitations & Future Work

1. **Simulated Students vs. Real Humans**: The core experiments rely entirely on simulated students via LLMs (Mixtral-8x7B). Such role-playing behaviors may distinctively differ from real students' learning processes, and the work lacks human-in-the-loop validation.
2. **Single Scenario**: Only evaluated on a single scenario of programming tutoring (100 tasks from EvoCodeBench), leaving generalization to other tutoring domains like mathematics or science unverified.
3. **Simplified KT Implementation**: Knowledge tracing relies entirely on LLM prompting without quantifying the accuracy of KT itself. Real-world student knowledge states could be far more complex.
4. **Verifier Generalizability**: The verifier is trained and evaluated using a 5-fold cross-validation on the same dataset; whether it generalizes to new domains or task types remains unknown.
5. **Over-simplified Cognitive Load Modeling**: Simulating cognitive load solely through "truncating to $M=60$ words" is overly simplistic compared to the complexity of actual cognitive processes.

## Related Work & Insights

- **Relationship with Math PRM**: This work's turn-by-turn verifier is directly inspired by "Let's Verify Step by Step" (Lightman et al., 2024) and Wang et al. (2024). However, the innovation lies in extending from static reasoning chains to dynamic multi-turn dialogues, which requires handling interactivity and personalization.
- **Relationship with TreeInstruct**: TreeInstruct (Kargupta et al., 2024) also employs the Socratic method to estimate student knowledge for tree-structured questioning but lacks quality control via a verifier. Traver builds upon this by adding a verifier to achieve superior performance.
- **Comparison with AlgoBo**: AlgoBo (Jin et al., 2024) is a teachable agent (where the student teaches the LLM), whereas this work proposes a tutoring agent (where the LLM teaches the student)—opposite but complementary directions.
- **Insight**: The concept of turn-by-turn verifiers can be extended to all target-driven multi-turn interactive scenarios (e.g., task-oriented dialogue, collaborative programming, psychological counseling). It represents a promising extension of process reward models in dialogue systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of KT and turn-by-turn verifier is novel in dialogue tutoring, and migrating the concept of PRM to multi-turn dialogue is creative; however, the individual components themselves (KT, best-of-$N$) are not new techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experimental design comparing 7 backbones, 3 baselines, along with ablation studies, human evaluation, inference-time scaling analysis, and simulated student validation. However, the scale of 100 tasks is relatively small, and user-study/real-human experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The problem definition is clear. The designs of Traver and Dict are logically self-consistent, with complete mathematical derivations, and rich visualizations (framework diagram, TOR curves, scaling curves). The overall narrative flows smoothly.
- **Value**: ⭐⭐⭐⭐ — Provides direct guidance for LLM applications in education. The Dict evaluation protocol is highly reusable, and the validation of inference-time scaling in dialogue is inspiring. However, limited by the simulated student setup, real-world deployment still requires human verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)
- [\[ACL 2026\] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](../../ACL2026/llm_reasoning/mtr-bench_a_comprehensive_benchmark_for_multi-turn_reasoning_evaluation.md)
- [\[ICLR 2026\] SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning](../../ICLR2026/llm_reasoning/simpletir_end-to-end_reinforcement_learning_for_multi-turn_tool-integrated_reaso.md)
- [\[ACL 2025\] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)
- [\[ICML 2026\] When the Chain of Thought Knows Better: Failure Modes in Multi-Turn Reasoning Models](../../ICML2026/llm_reasoning/when_the_chain_of_thought_knows_better_failure_modes_in_multi-turn_reasoning_mod.md)

</div>

<!-- RELATED:END -->
