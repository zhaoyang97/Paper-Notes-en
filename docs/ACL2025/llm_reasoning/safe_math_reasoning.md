---
title: >-
  [Paper Note] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification
description: >-
  [ACL 2025][Reasoning][Formal Verification] The Safe framework is proposed, which for the first time utilizes the Lean 4 formal language to perform retrospective step-by-step verification of LLM mathematical reasoning. It detects hallucinations through auto-formalization and automated theorem proving (ATP), and integrates with prospective PRM scores to achieve SOTA performance on multiple mathematical datasets. Furthermore, the FormalStep benchmark containing 30…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Formal Verification"
  - "Lean 4"
  - "Process-supervised Reward Models"
  - "Mathematical Reasoning"
  - "Automated Theorem Proving"
date: 2026-05-08
content_hash: 5884027201d90217
---

# Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification

**Conference**: ACL 2025  
**arXiv**: [2506.04592](https://arxiv.org/abs/2506.04592)  
**Code**: [https://github.com/liuchengwucn/Safe](https://github.com/liuchengwucn/Safe)  
**Area**: LLM Reasoning  
**Keywords**: Formal Verification, Lean 4, Process-supervised Reward Models, Mathematical Reasoning, Automated Theorem Proving  

## TL;DR

The Safe framework is proposed, which for the first time utilizes the Lean 4 formal language to perform retrospective step-by-step verification of LLM mathematical reasoning. It detects hallucinations through auto-formalization and automated theorem proving (ATP), and integrates with prospective PRM scores to achieve SOTA performance on multiple mathematical datasets. Furthermore, the FormalStep benchmark containing 30,809 formal statements is released.

## Background & Motivation

**Background**:
   - CoT reasoning has become the standard method to enhance LLM reasoning capabilities.
   - Process Reward Models (PRMs) verify reasoning quality by scoring each step, enabling Best-of-N sampling.
   - Automated Theorem Proving (ATP) has made progress in formal environments like Lean/Coq (e.g., DeepSeek-Prover).
   - Auto-formalization utilizes LLMs to translate natural language mathematics into formal languages.

**Limitations of Prior Work**:
   - **PRMs are black boxes**: They do not provide verifiable evidence, lacking interpretability and correctness guarantees.
   - **CoT hallucinations are difficult to detect**: Minor errors in intermediate reasoning steps severely impact subsequent reasoning and the final result.
   - **Formal proof and natural language reasoning are isolated**: Research in these two directions is independent, lacking an effective bridge.
   - **Formal proof of complete problems is extremely difficult**: On miniF2F, DeepSeek-Prover only achieves 60.2% under a 16×6400 generation budget.

**Key Challenge**:
   - Formal verification provides correctness guarantees but is hard to apply to entire complex problems.
   - PRMs are practical but unreliable—they predict a prospective probability (whether the correct answer can be reached) rather than a retrospective confirmation (whether the current step is logically correct).
   - These two are complementary: one evaluates step-level correctness, while the other assesses prospective reachability.

**Goal**:
   - To verify the correctness of natural language reasoning steps utilizing formal methods.
   - To effectively integrate retrospective formal verification with prospective PRM scoring.

**Key Insight**:
   - Verifying single-step correctness (rather than the whole problem) significantly reduces ATP difficulty.
   - Using auto-formalization to translate each reasoning step into a Lean 4 theorem, and then attempting to prove it using ATP.
   - Mapping reasoning steps into four verification states $\rightarrow$ aggregating with LSTM $\rightarrow$ integrating with PRM scores.

**Core Idea**:
   - "The gold standard of a mathematical claim is to provide a proof" — providing formal verification evidence for each step of LLM reasoning using Lean 4.

## Method

### Overall Architecture

The pipeline of Safe consists of:
1. **Sampling and Decomposition**: Samples $n$ reasoning trajectories using zero-shot CoT and decomposes them into step sequences.
2. **Step Verifier**: Performs auto-formalization and automated theorem proving for each step.
3. **State Aggregator**: Aggregates the verification states of each step using an LSTM to output a retrospective score.
4. **Score Fusion**: Integrates retrospective scores with prospective PRM scores to perform final Best-of-N selection.

### Key Designs

1. **Step Verifier**:

    - **Function**: Generates one of four states for each reasoning step: (1) no verification needed, (2) formalization failure, (3) successful formalization and successful proof, (4) successful formalization but failed proof.
    - **Mechanism**: Uses the previous step as the premise and the current step as the proof goal to generate a Lean 4 theorem.
    - **Design Motivation**: Single-step verification is **much simpler** than proving the entire problem, achieving an 80%+ proof success rate with a sample budget of only 16.

2. **Auto-Formalization Module**:

    - **Function**: Translates natural language reasoning steps into Lean 4 formal statements using LLM in-context learning (ICL).
    - **Mechanism**: For example, in an inequality transformation, the inequality before transformation serves as the premise, and the one after serves as the goal.
    - **Design Motivation**: Unlike traditional auto-formalization of complete problems, this only requires formalizing single-step correctness.

3. **ATP Module**:

    - **Function**: Proves formal statements using COPRA or DeepSeek-Prover-V1.5.
    - **Mechanism**: Employs DeepSeek-Prover-V1.5 (7B parameters) with a sample budget of 16 (without MCTS), achieving a proof rate $>80\%$.
    - **Design Motivation**: Although single-step statements are out-of-distribution (OOD) data, they are significantly less challenging than complete mathematical problems.

4. **State Aggregator**:

    - **Function**: Maps the sequence of verification states to a retrospective score using a tiny LSTM (2 layers, hidden size = 64).
    - **Mechanism**: $score_{retro}^i = \sigma(W \cdot \text{LSTM}(state_{i1}, ..., state_{ij}) + b)$
    - **Design Motivation**: Verification states are noisy (formalization or proof may fail); the model needs to learn to infer correctness from noisy states.

5. **Retrospective-Prospective Fusion**:

    - **Function**: Integrates retrospective scores from the LSTM with prospective scores from the PRM.
    - **Mechanism**: $score_i = (score_{retro}^i)^\alpha \cdot (score_{pro}^i)^{1-\alpha}$, with the final selection being $A^* = A_{i^*}, i^* = \arg\max_i score_i$.
    - **Design Motivation**: The two scores are complementary—retrospective scores measure whether the steps are correct, while prospective scores measure whether a correct final answer can be reached.

### Loss & Training

- **LSTM Training**: 2-layer LSTM, hidden_size=64, batch_size=32, learning_rate=0.0001, 200 epochs.
- **Extremely Small Training Data**: MATH (500 problems) + GSM8K (1000 problems) + complete CollegeMath training set.
- **No Extra Training for ATP**: Direct usage of off-the-shelf DeepSeek-Prover-V1.5.
- **Auto-Formalization**: Driven by GPT-4o with few-shot ICL.

## Key Experimental Results

### Main Results

BoN@5 accuracy comparison (Llama-3.1-8B-Instruct):

| Method | MATH-500 | GSM8K | CollegeMath |
|------|----------|-------|-------------|
| ZS-CoT@1 | 49.1 | 85.4 | 52.6 |
| Majority@5 | 50.5 | 87.8 | 54.3 |
| Skywork (ORM) | 48.9 | 90.2 | 53.2 |
| ArmoRM (ORM) | 55.1 | 90.0 | 57.1 |
| Shepherd (PRM) | 58.1 | 90.2 | 58.3 |
| RLHFlow (PRM) | 51.7 | 89.9 | 53.6 |
| LSTM (Ours) | 55.1 | 88.7 | 55.9 |
| **Safe (Ours)** | **60.0** | **90.8** | **59.0** |
| Pass@5 (Upper Bound) | 70.8 | 95.5 | 72.3 |

Summary of cross-model results:
- Llama-3.0-8B: Safe achieves 36.3% on MATH, yielding a 1.6% Gain over Shepherd PRM (34.7%) and an 11.5% Gain over Majority.
- GPT-4o: Safe achieves 80.4% on MATH (vs. 79.8% for Shepherd) and 74.2% on CollegeMath (vs. 73.5%).
- DeepSeek-Math: Safe achieves 52.4% on MATH (vs. 49.7% for Shepherd).

### Key Findings

1. **Extremely lightweight yet effective LSTM verifier**: Having minimal parameters (2 layers × 64 dimensions) and trained on minimal data, its performance is comparable to SOTA ORM/PRM.
2. **Safe (LSTM + PRM) consistently outperforms all baselines**: The integration of retrospective and prospective scores brings consistent improvements.
3. **Larger gains on challenging datasets**: Gains on MATH-500 and CollegeMath are prominent, whereas GSM8K shows less room for improvement due to its lower difficulty and resulting data imbalance.
4. **Single-step ATP proof rate $>80\%$**: Achieved with a sample budget of only 16 without MCTS, dramatically lowering computational costs.
5. **Scope of verification**: Capable of verifying numerical calculations, equation solving, as well as abstract mathematical properties like "sufficient/necessary conditions."

### FormalStep Dataset Statistics

| Category | Count | Statement Length | Proof Length | Proof Rate |
|------|------|---------|---------|-------|
| Geometry | 873 | 147.5 | 51.1 | 72.3% |
| Number Theory | 11,515 | 79.5 | 36.4 | 82.1% |
| Algebra | 5,525 | 107.8 | 39.8 | 81.7% |
| Combinatorics | 9,414 | 125.0 | 49.8 | 81.4% |
| Others | 3,482 | 112.8 | 25.5 | 79.1% |
| **Total** | **30,809** | 104.2 | 41.0 | **81.2%** |

- Geometry has the longest statements and the lowest proof rate (72.3%), aligns with intuition.
- The overall proof rate of 81.2% suggests that single-step verification is indeed much easier than proving complete problems.

## Highlights & Insights

- **First work utilizing Lean 4 to verify LLM natural language reasoning**: Pioneering the introduction of formal verification to CoT output assessment.
- **Insightful distinction between "retrospective" and "prospective"**: Elucidating the essential difference between formal verification (step-level correctness) and PRMs (reachability of the correct final answer).
- **Key insight of single-step verification**: Proving complete problems is overly tough, but single-step verification is straightforward enough for existing ATP models.
- **Extremely lightweight LSTM aggregator**: With a vocabulary size of only 4 and minimal training data, this tiny model is exceptionally effective.
- **Contribution of the FormalStep benchmark**: Over 30K formal statements, filling the blank in step-level ATP evaluation.
- **Exemplary practice of Neuro-Symbolic AI**: Offering a complementary fusion of natural language reasoning and formal verification.

## Limitations & Future Work

1. **Limited expressiveness of Lean 4**: Weaker formalization capabilities for geometry and combinatorics prevent a subset of steps from being verified.
2. **Dependency on GPT-4o for auto-formalization**: Higher API costs, and formalization quality is fundamentally capped by the LLM's capacity.
3. **Noisy verification states**: Formalization failure does not imply step incorrectness; proof failure does not strictly imply statement falsehood.
4. **Strict step correctness focus**: Cannot detect "correct but mathematically dead-end / wrong-path" intermediate reasoning steps.
5. **BoN@5 setup**: Relying on only 5 candidate sequences heightens the demand on sampling diversity.
6. Future directions include exploring stronger ATP backbones, broader formalization coverage, and combination with RLHF.

## Related Work & Insights

- **Lightman et al. (2024)**: PRM training methodologies; Ours offers a complementary perspective on formal verification.
- **DeepSeek-Prover-V1.5 (Xin et al., 2024)**: ATP infrastructure; Ours applies it directly to single-step verification.
- **Draft, Sketch and Prove (Jiang et al., 2022)**: Natural-language-assisted formal reasoning, showing an opposite workflow to ours.
- **Logic-LM (Pan et al., 2023)**: Representative work enhancing natural language reasoning with formal logic representation.
- Insight: Formal methods do not necessarily have to solve complete math problems; leveraging them to verify intermediate steps is a more practical approach.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Innovation | 9 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Practical Value | 8 |
| **Total Score** | **8.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[ACL 2025\] Large Language and Reasoning Models are Shallow Disjunctive Reasoners](large_language_and_reasoning_models_are_shallow_disjunctive_reasoners.md)

</div>

<!-- RELATED:END -->
