---
title: >-
  [Paper Note] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs
description: >-
  [ACL 2025][Reasoning][Chain-of-Thought] Proposes the Diverse Chain of Thought (DCoT) training method, which enables "within-inference refinement" by generating multiple sequential reasoning chains in a single inference session. It consistently outperforms standard CoT baselines across models ranging from 1.3B to 70B, with particularly significant improvements in large output space tasks (numerical/extractive).
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Chain-of-Thought"
  - "Diverse Reasoning Chains"
  - "Test-time Self-Correction"
  - "Instruction Fine-tuning"
  - "Reasoning Enhancement"
date: 2026-05-08
content_hash: 55945a8510700c3f
---

# Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs

**Conference**: ACL 2025  
**arXiv**: [2407.03181](https://arxiv.org/abs/2407.03181)  
**Code**: [https://github.com/UKPLab/acl2025-diverse-cot](https://github.com/UKPLab/acl2025-diverse-cot)  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought, Diverse Reasoning Chains, Test-time Self-Correction, Instruction Fine-tuning, Reasoning Enhancement  

## TL;DR
Proposes the Diverse Chain of Thought (DCoT) training method, which enables "within-inference refinement" by generating multiple sequential reasoning chains in a single inference session. It consistently outperforms standard CoT baselines across models ranging from 1.3B to 70B, with particularly significant improvements in large output space tasks (numerical/extractive).

## Background & Motivation
**Background**: Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of LLMs. Furthermore, methods like Self-Consistency improve performance by sampling multiple independent CoT paths and selecting the optimal one via voting.

**Limitations of Prior Work**: Multiple CoTs in methods like Self-Consistency are **generated independently and in parallel**. The model cannot reference the reasoning process of the first chain while generating the second, making it impossible to perform corrections and improvements during the inference process itself.

**Key Challenge**: Ideally, a model should be able to inspect the first reasoning chain and sequentially refine errors or supply missing details based on the existing reasoning. However, existing frameworks lack this capability—research on self-correction also indicates that LLMs struggle to self-correct without external feedback.

**Goal**: How to enable LLMs to generate multiple reasoning chains within a single inference run and leverage information from preceding reasoning chains to improve subsequent ones?

**Key Insight**: Concatenating multiple CoTs into a single sequence for training, allowing the model to learn to sequentially generate multiple CoTs within a single inference session, where subsequent CoTs can "see" the preceding reasoning process.

**Core Idea**: By using $(question, [CoT_1, CoT_2, ..., CoT_k])$ as training samples, the model learns to sequentially generate multiple reasoning chains within a single inference run, thereby achieving within-inference self-correction without requiring external feedback.

## Method

### Overall Architecture
The input is a question (optionally with candidate options), and the output is $k$ sequential CoT reasoning chains followed by a final answer. The format is:

```plaintext
Prompt: [Question] Q [Options] O [Number of answers] k
Response: [Answer 1] CoT₁ [Answer 2] CoT₂ ... [Answer k] CoTₖ [Final answer] ans
```

The training data is generated using GPT-3.5 turbo with different CoT triggers (e.g., "Let's think step by step") to produce 4 correct reasoning chains for each question, which are then concatenated into a single sequence as the training target.

### Key Designs

1. **DCoT Training Format**:

    - **Function**: Concatenates multiple correct CoTs for the same question into a single sequence for SFT.
    - **Mechanism**: Compared to standard CoT training (where each CoTs is treated as an independent sample), DCoT concatenates $k$ CoTs into $(Q, [CoT_1, CoT_2, ..., CoT_k, \text{Final Answer}])$. The key is that during training, the model learns through self-attention to reference the preceding $i-1$ CoTs while generating the $i$-th CoT.
    - **Design Motivation**: To enable the model to naturally "review" prior reasoning processes during inference, realizing sequential refinement rather than independent sampling.

2. **CoT Trigger Diversity**:

    - **Function**: Uses multiple different CoT prompt triggers to generate diverse reasoning chains.
    - **Mechanism**: Adopts 4 different trigger prompts (e.g., "Let's think step by step", "Let's break this down") to generate multiple correct reasoning chains with distinct styles for each question.
    - **Design Motivation**: Diversity is a prerequisite for effective correction—if the multiple reasoning chains are identical, no new information can be leveraged to fix mistakes.

3. **Final Answer Convergence Mechanism**:

    - **Function**: Generates the final answer using the `[Final answer]` instruction after multiple CoTs.
    - **Mechanism**: This special token forces the model to make a final judgment after evaluating all reasoning chains, acting like an implicit voting or integration process.
    - **Design Motivation**: To prevent the model from simply repeating the output of the last CoT, instead compelling it to synthesize information across multiple reasoning chains to make the final decision.

4. **Fair Comparison Design**:

    - DCoT and the CoT baseline use the **exact same reasoning chain data**, with the only difference being the organization: DCoT concatenates them into a single sequence, while CoT splits them into independent samples.
    - Base models without instruction tuning (Phi 1.5/2, LLaMA-2 7B/13B/70B) are selected to avoid interference from prior instruction-tuned CoT knowledge.

### Loss & Training
- **Training Data**: 9 datasets, covering numerical reasoning (GSM8K), extractive reasoning (ConditionalQA, HotpotQA), multiple-choice (ARC, BoardgameQA, Quartz), binary classification (StrategyQA), and symbolic reasoning (LLC).
- **Data Filtering**: Retains only reasoning chains that lead to the correct answer.
- **Selection of $k$ during Inference**: Search $k \in [1,4]$ on the development set; experiments show that $k=2$ (i.e., one-step refinement) is usually sufficient.

## Key Experimental Results

### Main Results

| Model | Method | Average | Numerical | Extractive | Multi-Choice |
|------|------|------|--------|--------|--------|
| Phi 1.5 (1.3B) | CoT | 47.2 | 34.95 (GSM8K) | 46.88 | 51.26 |
| Phi 1.5 (1.3B) | **DCoT** | **49.39** | **36.85** | **48.64** | **55.34** |
| Phi 2 (2.7B) | CoT | 60.85 | 56.71 | 58.89 | 64.42 |
| Phi 2 (2.7B) | **DCoT** | **62.6** | **60.73** | **61.88** | **68.00** |
| LLaMA2-7B | CoT | 58.97 | 28.51 | 59.80 | 61.36 |
| LLaMA2-7B | **DCoT** | **60.8** | **29.57** | **63.62** | **61.99** |
| LLaMA2-13B | CoT | 64.39 | 42.53 | 65.24 | 66.41 |
| LLaMA2-13B | **DCoT** | **66.18** | **44.28** | **67.54** | **68.30** |
| LLaMA2-70B | CoT | 66.96 | 56.00 | 64.76 | 69.34 |
| LLaMA2-70B | **DCoT** | **68.63** | **66.00** | 59.67 | **71.11** |

### Ablation Study: Impact of Different k

| Method | Phi 1.5 | Phi 2 | LLaMA-7B | LLaMA-13B |
|------|---------|-------|----------|-----------|
| CoT | 47.51 | 63.51 | 59.30 | 65.41 |
| DCoT@1 | 47.87 | 63.91 | 61.28 | 65.80 |
| DCoT@2 | **48.63** ↑ | **65.33** ↑ | **62.46** ↑ | **67.30** ↑ |
| DCoT@3 | 48.96 | 65.30 | 62.37 | 66.92 |
| DCoT@4 | 48.76 | 64.89 | 62.42 | 66.70 |

### Key Findings
- **$k=2$ is the optimal efficiency-performance trade-off**: A single refinement yields consistent improvements across all models, while $k=3,4$ generally yields no further performance gains (excluding GSM8K).
- **DCoT@1 ≈ CoT**: When generating only one reasoning chain, the DCoT-finetuned model performs nearly identical to CoT, proving that DCoT training does not degrade standard CoT capabilities.
- **Large output space tasks benefit the most**: Numerical and extractive tasks show the most significant gains because their large output spaces make revision more meaningful; binary classification tasks (only yes/no) show limited improvement.
- **Complementary to Self-Consistency**: DCoT+SC further boosts performance, demonstrating that the two approaches improve different dimensions of reasoning.
- **Human evaluation confirms self-correction**: In 56% of cases, the second CoT corrected the errors of the first rather than acting as a random perturbation.

## Highlights & Insights
- **Training format as the method**: No additional reward models or external feedback are needed. By simply altering the organization of the training data (concatenation vs. independence), the model acquires within-inference self-correction capabilities. This design is remarkably elegant and simple.
- **First to prove self-improvement without external feedback**: Prior works like Huang et al. 2024 suggested that LLMs could not self-correct on their own. This work challenges this notion through DCoT training—the key is exposing the model to the "generate one, then generate another" pattern during the training phase.
- **Highly practical**: Many existing instruction-tuning datasets already contain multiple CoTs for each question. Constructing DCoT data only requires concatenating them, incurring virtually zero additional cost.

## Limitations & Future Work
- The training data only originates from CoTs generated by GPT-3.5, leaving the reasoning chain quality bounded by the teacher model.
- Performance is limited on binary classification and symbolic reasoning tasks, which likely require tailored designs.
- Performance saturates or even declines when $k>2$; how to maintain effectiveness in multi-turn refinement remains an open question.
- Incorporating erroneous CoTs into training (e.g., via contrastive learning) was not explored, which might further enhance correction capabilities.

## Related Work & Insights
- **vs Self-Consistency (Wang et al. 2023)**: SC independently samples multiple CoTs and votes on them, whereas DCoT sequentially generates and integrates them. DCoT inherits the advantage of letting subsequent CoTs reference previous reasoning paths.
- **vs Self-Correction (Madaan et al. 2023)**: Traditional self-correction requires a two-stage process (generation + critique), while DCoT implicitly embeds correction within a single inference, making it more streamlined.
- **vs STaR (Zelikman et al. 2022)**: STaR utilizes reinforcement learning to improve reasoning, whereas DCoT purely relies on SFT, which is simpler to implement.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet effective; the insight that "changing the training format equals acquiring a new ability" is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Spans multiple model families (Phi, LLaMA), scales (1.3B to 70B), and task types, supplemented by human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous experimental design.
- Value: ⭐⭐⭐⭐ Highly practical, directly applicable to existing instruction fine-tuning pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)
- [\[ACL 2025\] CoT-Valve: Length-Compressible Chain-of-Thought Tuning](cot-valve_length-compressible_chain-of-thought_tuning.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/llm_reasoning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)

</div>

<!-- RELATED:END -->
