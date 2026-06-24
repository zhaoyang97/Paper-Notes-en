---
title: >-
  [Paper Note] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis
description: >-
  [ACL 2025][Reasoning][inference scaling] This paper proposes CoT-based Synthesizer—a novel inference scaling strategy that leverages CoT reasoning to analyze complementary information from multiple candidate responses to synthesize a superior final answer. Even when all candidate answers are incorrect, it can still synthesize the correct answer, achieving an 11.8% Gain for Llama3-8B and a 10.3% Gain for GPT-4o on MATH500.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "inference scaling"
  - "answer synthesis"
  - "chain-of-thought"
  - "self-consistency"
  - "reward model"
date: 2026-05-08
content_hash: 6fb44f5a3e1b7d0e
---

# CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis

**Conference**: ACL 2025  
**arXiv**: [2501.01668](https://arxiv.org/abs/2501.01668)  
**Code**: [https://github.com/RUCKBReasoning/CoT-based-Synthesizer](https://github.com/RUCKBReasoning/CoT-based-Synthesizer)  
**Area**: LLM Inference  
**Keywords**: inference scaling, answer synthesis, chain-of-thought, self-consistency, reward model

## TL;DR
This paper proposes CoT-based Synthesizer—a novel inference scaling strategy that leverages CoT reasoning to analyze complementary information from multiple candidate responses to synthesize a superior final answer. Even when all candidate answers are incorrect, it can still synthesize the correct answer, achieving an 11.8% Gain for Llama3-8B and a 10.3% Gain for GPT-4o on MATH500.

## Background & Motivation

**Background**: LLMs often generate incorrect answers in single attempts on complex reasoning tasks. Consequently, inference scaling strategies, including Best-of-N and Self-Consistency, have been widely investigated.

**Limitations of Prior Work (Best-of-N)**: Scores each candidate answer independently and fails to exploit the relationship between candidates. It is highly prone to reward hacking and cannot rectify errors when all candidates are incorrect.

**Limitations of Prior Work (Self-Consistency)**: Relies on majority voting via exact matching, which fails when the correct answer has low frequency among candidates. It also cannot yield a correct answer when all candidates are wrong.

**Key Challenge**: Traditional methods assume that the correct answer must exist within the candidate pool, whereas in reality, all candidates may contain local errors.

**Goal**: To go beyond the "selecting from candidates" paradigm and shift toward "synthesizing from candidates"—leveraging complementary information from multiple candidates to synthesize a superior answer.

**Core Idea**: Train a lightweight Synthesizer model that uses CoT reasoning to analyze the strengths and weaknesses of candidate answers, combining their merits to synthesize the correct answer.

## Method

### Overall Architecture
1. **Inference Phase**: The policy model generates N diverse candidate answers for a query → The query and candidates are fed into the CoT-based Synthesizer → The Synthesizer analyzes each candidate and synthesizes the final answer.
2. **Data Generation Phase**: An automated pipeline constructs the training data (two stages: synthetic answer generation + LLM repair).
3. **Training Phase**: SFT is used to train Synthesizer-8B on the generated data.

### Key Designs

1. **Diverse Candidate Generation**

    - Generate diverse candidates using a higher sampling temperature $t=0.9$ and Top-P $p=0.9$.
    - Truncate low-probability tokens to ensure generation quality while maintaining diversity.

2. **Response Analysis and Synthesis**

    - **Analysis Phase**: The Synthesizer analyzes the relationship between the query and each candidate, considering answer frequency, relevance, and accuracy, but without over-relying on frequency, prioritizing logical coherence and factual accuracy.
    - **Synthesis Phase**: When correct candidates exist, it extracts valid reasoning steps from other candidates to further enhance the answer; when all candidates are flawed, it integrates reasonable components from multiple candidates to construct a more complete answer.
    - Structured reasoning is achieved by designing specific prompts.

3. **Two-Stage Data Generation Pipeline**

    - **Stage 1: Synthetic Answer Generation**
        - The sampling LLM (Llama3-8B-Instruct) generates candidate answer sets.
        - The response LLM (Llama3.1-70B-Instruct) analyzes and synthesizes the candidates using CoT.
        - The response LLM is sampled multiple times (N=50 times), and filtered using gold answers to retain correct synthesized answers.
    - **Stage 2: LLM Repair**
        - When all candidates are incorrect, the response LLM is explicitly informed that all candidates are wrong.
        - The model is prompted to reflect on the errors, extract valid reasoning steps from the incorrect answers, and synthesize a corrected response.
        - Correct answers are filtered and retained again using gold answers.

### Loss & Training

- SFT training is performed using the standard autoregressive language modeling objective: $p_\phi(y|x,R) = \prod_{i=1}^T p_\phi(y_i|x,R,y_{<i})$
- Base model: Llama3-8B-Instruct
- Training data: MATH 12k samples expanded to 295k, WikiTQ 18k expanded to 87k
- Mathematical tasks are filtered using exact matching, and TableQA is filtered via CritiqueLLM scoring.

## Key Experimental Results

### Main Results

**Average accuracy on MATH500 dataset** (averaged across 7 policy models):

| Method | Average Accuracy |
|------|-----------|
| CoT-prompting | 55.2% |
| Self-Consistency | 60.8% |
| ArmoRM (Best-of-N) | 60.0% |
| Scalar RM | 59.7% |
| **Synthesizer-8B (Ours)** | **62.6%** (+7.4 Gain) |

**Single Model Highlights**:
- Llama3-8B on MATH500: 24.2% → 36.0% (+11.8% Gain)
- GPT-4o on MATH500: 62.5% → 72.8% (+10.3% Gain)
- On WikiTQ, the average is 83.6%, defeating all baselines.
- On FeTaQA, the average is 86.0%, also in the lead.

**Key Features**:
- The 8B Synthesizer can enhance the performance of 70B or even API models (e.g., GPT-4o).
- Despite not including GSM8k and FeTaQA in the training data, it still performs exceptionally well on these tasks (zero-shot generalization).

### Ablation Study

| Setting | GSM8k Avg | MATH500 Avg |
|------|-----------|-------------|
| Synthesizer-8B | 89.3 | 62.6 |
| w/o CoT training | 86.9 (-2.4) | 57.7 (-4.9) |
| w/o training | 84.1 (-5.2) | 57.5 (-5.1) |

- Performance drops without CoT training, and actually degrades on stronger models like Llama3.1-70B, showing that CoT analysis prevents overfitting to the sampling LLM.
- Performing synthesis directly using Llama3-8B without training is basically ineffective for strong models.

### Key Findings

1. **Data Scalability**: The training data size scales log-linearly with performance, showing no signs of saturation, whereas the performance of Scalar RM degrades as the data size increases (due to overfitting caused by repetitive instructions).
2. **Inference Scalability**: As the number of candidate answers increases from 5 to 25, the performance of the Synthesizer continues to improve, whereas the Best-of-N method degrades with more candidates due to reward hacking.
3. **Performance with Zero Correct Candidates**: When all 5 candidates are incorrect, the Synthesizer can still synthesize 7 correct answers (while SC and ArmoRM yield 0 correct answers in this case).
4. **Cost Efficiency**: It requires only 5 candidates to achieve performance comparable to SC with 10+ candidates.

## Highlights & Insights

1. **Paradigm Shift**: Transitioning from "selecting the best" to "synthesizing a new answer by combining strengths", breaking the fundamental limitations of traditional inference scaling methods.
2. **Small Models Powering Large Models**: The 8B Synthesizer significantly improves models far larger than itself, such as GPT-4o, demonstrating the value of "specialized synthesis".
3. **Ingenious Repair Phase**: Designing the LLM Repair phase for cases where all candidates are incorrect, explicitly informing the model of candidate errors and prompting reflection, which effectively improves training data coverage.
4. **Strong Generalization**: It works effectively on unseen datasets and models with different architectures.

## Limitations & Future Work

1. Group-wise iterative synthesis is required when candidate quantities are large, increasing inference complexity.
2. Generating training data relies on a strong Response LLM (Llama3.1-70B), which remains costly.
3. Currently validated only on mathematical reasoning and TableQA, and has not yet been extended to other tasks such as code generation and open-ended generation.
4. Higher inference latency is incurred due to multiple samplings and synthesis model inference per run.

## Related Work & Insights

- **Self-Consistency / USC**: Classic inference scaling methods, but they rely on the assumption that the correct answer is present in the candidates.
- **Best-of-N / Reward Model**: Another mainstream direction, but independent scoring fails to exploit correlations among candidates.
- **LMCOR**: Also performs answer synthesis but directly trains using the gold answer, lacking the CoT analysis process.
- **Insight**: The concept of synthesis can be generalized to multi-LLM ensembles — rather than selecting the best single model's response, it integrates the reasoning steps of all models.

## Rating

| Metric | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| **Overall Rating** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/llm_reasoning/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[ICLR 2026\] VoG: Enhancing LLM Reasoning through Stepwise Verification on Knowledge Graphs](../../ICLR2026/llm_reasoning/vog_enhancing_llm_reasoning_through_stepwise_verification_on_knowledge_graphs.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)

</div>

<!-- RELATED:END -->
