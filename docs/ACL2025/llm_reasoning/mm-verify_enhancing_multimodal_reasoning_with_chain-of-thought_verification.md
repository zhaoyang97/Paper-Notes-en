---
title: >-
  [Paper Note] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification
description: >-
  [Reasoning] This paper proposes two models, MM-Verifier and MM-Reasoner. By synthesizing long-chain CoT verification data through simulation-based search combined with rejection sampling, and creating multimodal reasoning data via text distillation, the proposed 7B parameter models achieve an accuracy of 65.3% on MathVista, outperforming GPT-4o (63.8%) and human performance (60.3%).
tags:
  - "Reasoning"
date: 2026-05-08
content_hash: 07d4884dfc4acf85
---

# MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification

| Attribute | Content |
|------|------|
| Title | MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification |
| Conference | ACL2025 |
| arXiv | [2502.13383](https://arxiv.org/abs/2502.13383) |
| Code | [github.com/Aurora-slz/MM-Verify](https://github.com/Aurora-slz/MM-Verify) |
| Area | LLM Reasoning / Multimodal Math |
| Keywords | Multimodal Verification, Chain-of-Thought, MCTS, Reward Model, Math Reasoning |

## TL;DR

This paper proposes two models, MM-Verifier and MM-Reasoner. By synthesizing long-chain CoT verification data through simulation-based search combined with rejection sampling, and creating multimodal reasoning data via text distillation, the proposed 7B parameter models achieve an accuracy of 65.3% on MathVista, outperforming GPT-4o (63.8%) and human performance (60.3%).

## Background & Motivation

- **Inspiration from Test-Time Scaling**: In text-only LLMs, combining external slow-thinking and verification mechanisms has been proven to enhance multi-turn reasoning (e.g., DeepSeek-R1, s1).
- **Lack of Multimodal Verifiers**: Self-criticism methods used in text-only domains perform poorly in multimodal models (validated by experiments in Table 6), highlighting an urgent need to develop powerful multimodal verifiers.
- **Lack of Long-Chain CoT Reasoning Data**: While long CoT data exists in text-only domains (e.g., DeepSeek-R1), most mathematical problems in multimodal domains are not in long CoT format.
- **Two Core Challenges**:
  1. How to synthesize high-quality multimodal verification data to train MM-Verifier?
  2. How to efficiently synthesize multimodal long-chain CoT reasoning data to train MM-Reasoner?

## Method

### Stage 1: Simulation-Based Search for Long-Chain CoT MM-Verifier

#### Data Source Collection

A total of 59,772 questions across seven categories are selected from MATH360V to serve as the source data pool: Geometry3K (33.84%), Super-CLEVR (24.17%), TabMWP (22.45%), FigureQA (18.07%), GEOS (1.48%), etc.

#### Simulation-Based Search Algorithm

Inspired by MCTS but avoiding traditional MCTS (as multimodal models struggle to generate reliable rewards):

1. Starting from the root node $q_i$, $k$ child nodes are simulated for each node.
2. For each child node, the model directly generates an answer based on the current path:
   $$\text{Simulation Answer} = LLM\left(\bigoplus_{i=1}^{d-1} u_i\right)$$
3. Repeat the simulation $l$ times, using the accuracy rate as the reward signal.
4. Finally, collect $n$ leaf-node solution pairs $\langle q_i, p_j^i \rangle$ as positive/negative samples.

This method generates longer CoT answers than direct sampling (validated in Figure 3).

#### Long-Chain CoT Verification Data Synthesis

1. GPT-4o is employed to perform "step-by-step verification" on each $\langle q_i, p_j^i \rangle$ pair, generating verification text $v_i$.
2. Data Cleaning: LLaMA-3.2-3B-Instruct is used to extract answers, checking two conditions:
    - If the extracted answer matches the golden label and the verification conclusion is "correct" $\rightarrow$ Retain
    - If the extracted answer does not match and the verification conclusion is "incorrect" $\rightarrow$ Retain
    - Other cases $\rightarrow$ Discard
3. Qwen2-VL-7B-Instruct is fine-tuned (SFT) using the cleaned data $D_{clean}$ $\rightarrow$ MM-Verifier (Stage 1) is obtained.

### Stage 2: Rejection Sampling for Further Verification Enhancement

1. Leak the long-chain CoT reasoning capabilities of the Stage 1 Verifier to generate more verification data.
2. Perform cleaning by comparing with ground-truth answers via string matching.
3. Cleaned and filtered data is used to continue training the Stage 1 model $\rightarrow$ MM-Verifier (Stage 2) is obtained.

Key Advantage: Reduces API costs (no longer requiring GPT-4o) while further enhancing verification capabilities.

### MM-Reasoner: Cross-Modal Knowledge Distillation

**Mechanism**: Leverage the strong reasoning capability of text-only reasoning models (Qwen-QwQ) by bridging multimodal data through textual descriptions.

1. Select the MAVIS-GEOMETRY dataset, which contains geometric diagram drawings and corresponding text description instructions.
2. Input the geometric text description and the original question into the text-only reasoning model QwQ.
3. Use the output of QwQ as the training target for MM-Reasoner.
4. After filtering out incorrect reasoning results, utilize the remaining data for SFT on Qwen2-VL-7B-Instruct.

Data Statistics: There are 32,146 training data samples for MM-Reasoner (all sourced from MAVIS-Geo).

## Key Experimental Results

### MM-Verifier Performance: MathCheck Outcome-Judging

The 7B MM-Verifier surpasses all large proprietary models (including GPT-4o, Gemini, Claude) and 72B open-source models (Figure 1).

### MathVista Results

| Method | Sample 4 ALL | Sample 8 ALL | Sample 12 ALL |
|------|-------------|-------------|---------------|
| **Qwen2-VL + Majority Voting** | 57.1 | 61.1 | 62.9 |
| Qwen2-VL + Qwen2-VL-72B Judge | 53.4 | 56.2 | 55.7 |
| Qwen2-VL + MM-Verifier(S2) | **59.8** | **62.5** | **64.1** |
| **MM-Reasoner + Majority Voting** | 59.4 | 62.2 | 64.8 |
| MM-Reasoner + MM-Verifier(S2) | **61.5** | **65.3** | **65.2** |

Key Findings:
- MM-Verifier(S2) consistently outperforms Majority Voting and Qwen2-VL-72B Judge across all settings.
- MM-Reasoner + MM-Verifier(S2) with Sample 12 reaches **65.3%**, outperforming GPT-4o (63.8%) and human performance (60.3%).
- When acting as a judge, Qwen2-VL-72B exhibits a performance decline when verifying the long outputs of MM-Reasoner, indicating that conventional models struggle to verify long-chain CoT.

### MathVerse Results

MM-Verifier + MM-Reasoner achieves 25.7% (ALL), outperforming Math-LLaVA-13B (22.9%) and LLaVA-OneVision (20.7%).

### Final Comprehensive Comparison

| Model | MathVista ALL | MathVerse ALL |
|------|-------------|---------------|
| Human | 60.3 | 64.9 |
| GPT-4o | 63.8 | 50.8 |
| Qwen2-VL-7B | 52.5 | 20.1 |
| Math-LLaVA-13B | 46.6 | 22.9 |
| **Ours (7B)** | **65.3** | **25.7** |

### MM-Reasoner Scalability

As the training data scale increases from 6,952 to 32,146, the performance continues to steadily improve (Figure 4), validating the scalability of the proposed data synthesis method.

### Stage 1 vs Stage 2

Stage 2 outperforms Stage 1 in almost all settings, verifying the effectiveness of rejection sampling in further enhancing the verifier.

## Highlights & Insights

1. **7B Surpasses GPT-4o and Humans**: Achieving 65.3% on MathVista with only a 7B model, outperforming GPT-4o (63.8%) and human performance (60.3%), which fully demonstrates the power of the Verifier+Reasoner combination.
2. **Ingenious Data Synthesis Strategy**: Simulation search generates long CoT $\rightarrow$ GPT-4o verification $\rightarrow$ rejection sampling self-enhancement, forming a progressive data-quality improvement pipeline.
3. **Text-Multimodal Bridging**: The capability of the text-only reasoning model is distilled into the multimodal model through textual descriptions from MAVIS, avoiding expensive multimodal tree search.
4. **Revealing the Limitations of LLMs in Verifying Long Outputs**: Qwen2-VL-72B's performance in verifying MM-Reasoner declines as the number of samples increases, indicating that traditional models lack the capability to verify long-chain CoT.
5. **Two-Stage Verifier Design**: Stage 1 utilizes an external API to guarantee quality, whereas Stage 2 bootstraps via its own capabilities, progressively lowering costs.

## Limitations & Future Work

1. **Resource Constraints**: Failed to scale MM-Verifier and MM-Reasoner to the 72B parameter size.
2. **Limited Scalability Testing**: Data volume scalability checks were only carried out up to <100K samples.
3. **Narrow Reasoning Domain**: Primarily focuses on mathematical reasoning; effectiveness has not been verified on other multimodal reasoning tasks (such as scientific reasoning or commonsense reasoning).
4. **Dependency on External Models**: Stage 1 requires GPT-4o to generate verification texts, leading to a relatively high data synthesis cost.
5. **Single Data Source for MM-Reasoner**: Employs only MAVIS-Geometry data, offering limited domain coverage.

## Related Work & Insights

- **Multimodal Math Models**: UniMath, G-LLaVA, MAVIS, EAGLE, etc.
- **Reward Models**:
    - ORM (Outcome Reward Model): Qwen2.5-Math-RM-72B, which only evaluates the final outcome.
    - PRM (Process Reward Model): Math-Shepherd, EurusPRM, Qwen2.5-Math-PRM, which evaluate the reasoning process step-by-step.
- **LLM-as-a-Judge**: Various approaches utilizing LLMs as evaluators.
- **Test-Time Scaling**: Scaling compute during inference, such as DeepSeek-R1, s1, and LIMO.

## Rating ⭐⭐⭐⭐⭐

**Strengths**: Impressive results (7B outperforming GPT-4o and humans); the data synthesis pipeline is carefully designed and validated with thorough ablation studies at each step; introduces a new paradigm for verifiers in the multimodal domain; open-source code.

**Weaknesses**: The approach relies on external APIs (GPT-4o) to some extent; the training sources for MM-Reasoner are relatively narrow (only geometry problems); generalization to non-mathematical tasks remains unproven.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ICML 2025\] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](../../ICML2025/llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)

</div>

<!-- RELATED:END -->
