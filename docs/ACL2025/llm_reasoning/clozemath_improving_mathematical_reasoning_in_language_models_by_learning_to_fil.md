---
title: >-
  [Paper Note] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations
description: >-
  [ACL 2025][Reasoning][Mathematical Reasoning] ClozeMath proposes a fine-tuning strategy inspired by human cloze learning. By masking equations in mathematical solutions and training the model to predict them (a text-infilling objective) jointly with standard language modeling objectives, ClozeMath significantly outperforms the strong baseline Masked Thought on GSM8K and MATH. It also demonstrates superior generalization in test-time scaling and robustness evaluations.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Mathematical Reasoning"
  - "Text Infilling"
  - "Equation Masking"
  - "PrefixLM"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: da978245cd7b7dab
---

# ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations

**Conference**: ACL 2025  
**arXiv**: [2506.03763](https://arxiv.org/abs/2506.03763)  
**Code**: None (Qualcomm AI Research)  
**Area**: LLM Reasoning  
**Keywords**: Mathematical Reasoning, Text Infilling, Equation Masking, PrefixLM, Chain-of-Thought

## TL;DR

ClozeMath proposes a fine-tuning strategy inspired by human cloze learning. By masking equations in mathematical solutions and training the model to predict them (a text-infilling objective) jointly with standard language modeling objectives, ClozeMath significantly outperforms the strong baseline Masked Thought on GSM8K and MATH. It also demonstrates superior generalization in test-time scaling and robustness evaluations.

## Background & Motivation

Current LLMs enhance their mathematical reasoning capabilities primarily through training on Chain-of-Thought (CoT) styled data, where the model learns to generate intermediate reasoning steps before producing the final answer. However, this training paradigm faces a fundamental issue:

**Limitations of the Prediction Paradigm**: The standard next-token prediction objective may not align with human learning processes. When humans learn mathematical derivations, they tend to understand the general methodology first and then handle specific details (grasping the "probing logic" before computing concrete numbers), rather than sequentially memorizing "which step follows which."

**Limitations of Masked Thought**: The recent strong baseline, Masked Thought Fine-tuning (MFT), forces the model to attend to more distant problem definition information by randomly masking tokens in the solutions. However, its random masking strategy suffers from **spurious correlation** issues: when consecutive mathematical transformation steps are tightly linked, masking can force the model to predict subsequent steps without having the prerequisite steps defined (such as when a variable definition is masked), leading to the learning of incorrect dependencies.

**Key Insight**: In mathematical solutions, textual descriptions (rationales) represent general methods, whereas equations correspond to problem-specific calculations. Inspired by human cloze exercises, the authors propose that given the textual reasoning of a solution, the model should **fill in the missing equations**, thereby strengthening its ability to infer mathematical relationships.

## Method

### Overall Architecture

ClozeMath incorporates a text-infilling objective on top of standard language model training. During training, the model jointly optimizes two losses:
- $\mathcal{L}_{\text{lm}}$: Standard language modeling objective (predicting the solution given the question).
- $\mathcal{L}_{\text{tf}}$: Equation-filling objective (predicting the masked equations given the question and the masked solution).

Final objective: $\mathcal{L}_{\text{ClozeMath}} = \mathcal{L}_{\text{lm}} + \mathcal{L}_{\text{tf}}$

During inference, standard token-by-token generation is used without requiring any special handling.

### Key Designs

1. **Equation Masking**:

    - **Function**: Identify all equations in a mathematical solution and replace them with special mask tokens (\<X\>, \<Y\>, etc.), while keeping the textual rationales intact.
    - **Progressive Demasking**: Given a solution containing $|F^i|$ equations, $|F^i|$ text-infilling samples are generated: the first sample masks all equations, the second preserves the first equation but masks the rest, and so forth. This complies with the causal dependency of equations in math solutions—each equation only depends on the preceding ones.
    - **Why not random masking**: Ablation studies demonstrate that random masking (similar to T5's span corruption) disrupts the logical coherence of the textual rationales, significantly degrading performance (74.22% → 71.19%).

2. **PrefixLM Architecture**:

    - **Mechanism**: PrefixLM is implemented on pretrained decoder-only models. Bidirectional attention is applied to the prompt segment (question + masked solution), while causal attention is applied to the target sequence (masked equations), separated by a \<SEP\> token.
    - **Design Motivation**: Bidirectional attention allows the model to better comprehend the contextual information in the prefix (problem definition + textual rationales), thereby inferring the missing equations more effectively.
    - **Key Finding**: Utilizing PrefixLM alone yields negligible improvement (71.57% → 71.79%), yet its combination with the equation-filling objective exhibits a substantial improvement (71.57% → 74.22%). This indicates that leverage of bidirectional context requires an appropriate training target.

3. **Sample Balancing**:

    - Since the number of text-infilling samples depends on the number of equations, the language modeling samples are duplicated in practice to maintain an approximate 50:50 ratio between the two training objectives.

### Loss & Training

- Joint training loss: $\mathcal{L}_{\text{ClozeMath}} = \mathcal{L}_{\text{lm}} + \mathcal{L}_{\text{tf}}$
- Fine-tune base language models (DeepSeek-Math-7B-base, Llama-3.1-8B, Llama-3.2-3B, Llama-3.2-1B) using LoRA (rank=32).
- Expand the vocabulary to support \<SEP\> and mask tokens.

## Key Experimental Results

### Main Results

| Dataset | Model | Baseline (Base) | MFT | ClozeMath | ClozeMath vs MFT |
|--------|------|-----------|-----|-----------|-----------------|
| GSM8K | DeepSeek-Math-7B | 59.21 | 70.20 | **74.22** | +4.02 |
| GSM8K | Llama-3.1-8B | 49.58 | 64.82 | **70.00** | +5.18 |
| GSM8K | Llama-3.2-3B | 17.66 | 45.03 | **53.15** | +8.12 |
| GSM8K | Llama-3.2-1B | 4.62 | 21.15 | **27.89** | +6.74 |
| MATH | DeepSeek-Math-7B | 31.68 | 33.42 | **36.90** | +3.48 |
| MATH | Llama-3.1-8B | 18.06 | 20.94 | **22.88** | +1.94 |

### Ablation Study

| Settings | GSM8K Accuracy | Description |
|------|------------|------|
| Full ClozeMath | **74.22%** | Equation Masking + PrefixLM |
| W/o Text-infilling | 71.79% | PrefixLM only, without infilling target |
| W/o PrefixLM | 72.71% | Equation filling + CausalLM |
| W/o both (Standard IT) | 71.57% | Traditional instruction tuning |
| Random masking (non-equation) | 71.19% | Logically incoherent, lowest performance |

### Key Findings

- ClozeMath consistently outperforms MFT across **all model sizes** and is more sample-efficient during training (performing better at every checkpoint).
- **Test-Time Scaling**: Under CoT decoding (k=9), ClozeMath also performs better than MFT (e.g., DeepSeek-Math: 77.10% vs 76.50%), showing its scalability with increased inference compute.
- **Robustness Evaluation (GSM-Symbolic)**: On variant problems with newly added constraints, ClozeMath's advantage is even more pronounced (DeepSeek-Math GSM-P1: 49.25% vs 44.25%, a gain of 5%).
- The smaller the model, the more substantial ClozeMath's improvement over MFT (e.g., +8.12 on GSM8K for Llama-3.2-3B).

## Highlights & Insights

- **Precise Analogy to Human Learning**: The cloze test is a classic approach in language learning. Porting this concept to mathematical reasoning is an elegant design. The model first grasps the solving trajectory (with the retained textual rationale) before deriving specific equations, aligning perfectly with the paradigm of "mastering methodology before focusing on details."
- **Insightful Analysis of Spurious Correlations in Masked Thought**: The paper concretely highlights issues with MFT where incorrect dependencies are learned when transformation steps are tightly coupled (e.g., predicting 4*b=24 without the prior definition of variable b).
- **Synergy of PrefixLM and Equation Infilling**: PrefixLM alone is marginally effective, but delivers a significant boost when paired with the equation-filling objective. This underscores the necessity of co-designing network architectures and training objectives.

## Limitations & Future Work

- Experiments are restricted to models under 10B. The performance on larger models remains unverified.
- The current focus is limited to mathematical reasoning. Generalizability to other domains requiring structured reasoning (such as code generation or logical reasoning) needs further validation.
- Equation identification relies on heuristic rules. More complex mathematical expressions (such as complex formulas in LaTeX format) may necessitate more robust parsing.
- Due to formatting issues in the MATH dataset, the evaluation of CoT decoding is restricted to GSM8K.

## Related Work & Insights

- Directly compares with Masked Thought (Chen et al., 2024), illustrating the discrepancy between targeted masking strategies (equations vs. random tokens).
- The text-infilling methodology is inherited from T5 (Raffel et al., 2020) and UL2 (Tay et al., 2023), but innovates by masking only equations rather than arbitrary spans.
- The application of PrefixLM is inspired by Liu et al. (2018), validating its effectiveness when paired with a target-specific objective on reasoning tasks.
- Insights for the research community: **The design of training objectives should take the structural characteristics of tasks into account**, rather than naively adopting generic solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ The analogy to human cloze tests is novel and elegant, and the equation-masking strategy is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, covering multiple models, datasets, ablations, robustness, and test-time scaling.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, deep analysis of MFT's limitations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play fine-tuning strategy that can be widely applied to the training of mathematical reasoning LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [\[ICLR 2026\] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](../../ICLR2026/llm_reasoning/mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)

</div>

<!-- RELATED:END -->
