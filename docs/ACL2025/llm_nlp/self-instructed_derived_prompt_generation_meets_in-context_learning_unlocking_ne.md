---
title: >-
  [Paper Note] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs
description: >-
  [ACL 2025][LLM (Other)][Prompt Optimization] A self-instructed reinforcement learning framework is proposed to train a "derived prompt generation model." By utilizing the derived prompt-response pairs as in-context learning (ICL) examples to enhance queries of the original prompt, the response quality is significantly improved without modifying the parameters of black-box LLMs (such as GPT-4).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Prompt Optimization"
  - "Black-box LLM Alignment"
  - "In-Context Learning"
  - "Reinforcement Learning"
  - "Derived Prompt"
date: 2026-05-08
content_hash: c487e3d132ff7d32
---

# Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs

**Conference**: ACL 2025  
**arXiv**: [2409.01552](https://arxiv.org/abs/2409.01552)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Prompt Optimization, Black-box LLM Alignment, In-Context Learning, Reinforcement Learning, Derived Prompt

## TL;DR

A self-instructed reinforcement learning framework is proposed to train a "derived prompt generation model." By utilizing the derived prompt-response pairs as in-context learning (ICL) examples to enhance queries of the original prompt, the response quality is significantly improved without modifying the parameters of black-box LLMs (such as GPT-4).

## Background & Motivation

The performance of LLMs is highly dependent on the quality of input prompts—vague or imprecise prompts lead to low-quality responses. For black-box LLMs like GPT-4, fine-tuning for alignment is infeasible due to the lack of access to parameters, making guiding models through better prompts a core technique.

Existing prompt optimization methods suffer from three main limitations:

**Semantic Inconsistency**: Refined prompts may deviate from the original intent. For example, if the original question is "Human knowledge is a collection of definitions and observations, what do you think?", the refined version becomes "What is the basis of human knowledge, and how is it acquired and organized?"—the focus and depth are completely altered.

**Over-narrowing**: Methods like BPO refine "Describe the health benefits of green tea" into "Discuss the antioxidant properties of green tea and its role in cancer prevention"—ignoring other health benefits.

**Data Collection Burden**: Training prompt refinement models typically requires a large amount of paired (original prompt, refined prompt) data.

Core Problem: **Is there a more effective way to improve the responses of black-box LLMs than directly replacing the original prompt with a refined prompt?**

## Method

### Overall Architecture

The method consists of two core components:
1. **Self-instructed RL derived prompt generation model training**
2. **Intent-consistent in-context query inference framework**

Key Idea: Instead of directly replacing the original prompt with a derived prompt, the derived prompt and its response are used as ICL examples to construct an informative context environment for the original prompt.

### Key Designs

1. **Derived Prompt Generation (DPG)**: A derived prompt $x'$ is a transformation of the original prompt $x$, which maintains semantic relevance but is expressed better. Unlike "refined prompts" that directly replace the original prompt, the purpose of a derived prompt is to generate high-quality exemplar responses for ICL to use.

   To activate the instruction-following capability of the pre-trained LLM, a DPG instruction $x_{\text{DPG}}$ is utilized:
    $X = \text{Concat}([x_{\text{DPG}}, x])$
    $x' \sim \pi_\theta(X)$
   
   This eliminates the need for paired data during the SFT stage—the model naturally possesses instruction-following and rewriting capabilities.

2. **Self-Instructed Reinforcement Learning Training**: The response model $\mathcal{M}$ (such as GPT-4) is integrated into the training process. Generate derived prompt $x'$ $\rightarrow$ generate response $y'$ using $\mathcal{M}$ $\rightarrow$ evaluate quality of $(x', y')$ using reward model $\mathcal{R}$ $\rightarrow$ optimize the generation model $\pi_\theta$ using feedback.
   
   Training objective:
    $\max \mathbb{E}_{(x, x', y')} \left[\mathcal{R}(x', y') - \beta \log \frac{\pi_\theta(x'|X)}{\pi_{\text{ref}}(x'|X)}\right]$
   
   where KL divergence regularization ensures training stability. Core advantage: No requirement for collecting paired data, training signals are obtained directly through interaction.

3. **Intent-Consistent ICL Query Framework**: Instead of directly replacing $x$ with $x'$ during inference, the following steps are performed:

    - Generate a derived prompt $x'$
    - Generate a response $y'$ to $x'$ using the LLM
    - Use $(x', y')$ as an example to populate the ICL template
    - Query the LLM with the ICL template to generate the final response to the original prompt $x$
   
   This preserves the information of the original prompt while leveraging a high-quality ICL context to activate the LLM's intrinsic knowledge.

### Loss & Training

- **Reward Model**: Trained on the hh-rlhf helpful dataset based on GPT2-Large
- **Training Data**: Only requires a collection of original prompts (14K samples from the BPO training set), without paired data
- **Reference Model**: A frozen copy $\pi_{\text{ref}}$ initialized identically to $\pi_\theta$
- **PPO Optimization**: Trained using the PPO algorithm with KL regularization

## Key Experimental Results

### Main Results (Ours vs OP/BPO, trained with Llama3, querying GPT-4)

| Evaluation Dataset | Ours vs OP (Win%) | Ours vs BPO (Win%) |
|-----------|------------------|-------------------|
| Vicuna Eval | **90.0** vs 3.8 | **88.8** vs 7.5 |
| BPO-test | **71.0** vs 24.5 | **74.0** vs 25.5 |
| Dolly Eval | **80.5** vs 15.5 | **71.0** vs 27.0 |
| Self-Instruct | **76.2** vs 5.6 | **71.4** vs 21.0 |

### Ablation Study (Querying GPT-4, Vicuna Eval / Self-Instruct)

| ID | Comparison A | Comparison B | Vicuna A Win | Self-Inst A Win |
|------|--------|--------|-------------|----------------|
| #1 | OD | OP | 78.8% | 66.3% |
| #2 | BPO | OP | 68.8% | 66.3% |
| #5 | BPO+ICL | OP | 76.3% | 69.4% |
| #7 | OD+ICL | OD | 60.0% | 52.4% |
| #9 | OD+ICL | BPO+ICL | 75.0% | 68.9% |

### Key Findings

1. **Significant Improvements on Black-Box Models**: On GPT-4, the method trained on Llama3 achieves an average Win Rate of 67.1% (vs OP) and 56.1% (vs BPO). It is even higher on GPT-3.5: 74.3% and 69.9%.

2. **OD Quality Outperforms BPO**: Looking solely at prompt refinement (#1 vs #2), OD already exhibits higher quality. On Vicuna, the net win of OD vs OP is 67.6%, whereas BPO is only 53.8%.

3. **ICL Framework is a Universal Enhancement Module**: Even when replacing OD with BPO, BPO+ICL significantly outperforms BPO alone (#5 vs #4), demonstrating that the ICL query is a plug-and-play, universal framework.

4. **Cross-Model Transferability**: The $\pi_\theta$ trained with Llama3 generates ICL examples that can improve response quality on Llama2, Qwen2, and even GPT-4.

5. **Self-Instructed RL Outperforms SFT**: After RL training, the win rate of OD vs OP increases from 17.6% to 67.6%; and OD+ICL vs OP increases from 39.1% to 86.2%.

6. **Training-Free Baselines Already Perform Well**: Directly using an untrained LLM to generate derived prompts + ICL already yields a 39.1% improvement, demonstrating the effectiveness of the ICL framework itself.

## Highlights & Insights

- **Paradigm Shift**: Shifting from "replacing the original prompt with a better one" to "constructing an ICL environment with derived prompts to assist the original prompt" preserves the user's original intent. This is a critical transition in perspective.
- **Zero Data Collection Training**: Self-instructed RL leverages DPG instructions and the instruction-following capabilities of LLMs, eliminating the need for (original, refined) paired data.
- **Black-Box Usable**: The entire method does not modify the parameters of the target LLM, making it fully applicable to black-box models like GPT-4. Training only requires a reward model and a trainable generation model.
- **Clear Case Analysis**: The case of health benefits of green tea in the paper vividly demonstrates the semantic drift problem in BPO and highlights the advantages of the proposed method.

## Limitations & Future Work

- Increased inference overhead: Each query requires generating a derived prompt first, querying the LLM to get a response, and then constructing the ICL template to query again—effectively calling the LLM twice.
- The quality of the reward model (GPT-2 Large-based) directly determines the training efficacy; a stronger reward model may bring further improvements.
- Not tested on tasks requiring exact formatting outputs (such as code generation or structured data extraction).
- The design of the ICL template may affect the performance, which is not thoroughly explored with different template variations in this paper.
- Evaluation largely relies on GPT-4o as a judge, which might introduce evaluation bias.

## Related Work & Insights

- The core difference from BPO (direct prompt refinement) lies in preserving the original intent + ICL enhancement.
- The key difference from RLHF is that the optimization target is the prompt generation model rather than the response model.
- As a plug-and-play module, the ICL query framework can enhance any prompt optimization method, including future new methods.
- This framework can be extended to multi-turn conversation scenarios.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3.5 |
| Value | 4 |

The paradigm shift of "derived prompt + ICL" has high novelty. The experiments validate consistency across multiple models and multiple datasets. There are some minor issues with inconsistent notation usage in the writing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Length Controlled Generation for Black-box LLMs](length_controlled_generation_for_black-box_llms.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ICLR 2026\] Prompt-MII: Meta-Learning Instruction Induction for LLMs](../../ICLR2026/llm_nlp/prompt-mii_meta-learning_instruction_induction_for_llms.md)

</div>

<!-- RELATED:END -->
