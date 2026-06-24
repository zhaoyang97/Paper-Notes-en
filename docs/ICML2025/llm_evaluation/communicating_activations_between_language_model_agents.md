---
title: >-
  [Paper Note] Communicating Activations Between Language Model Agents
description: >-
  [ICML 2025][LLM Evaluation][Multi-agent communication] A method is proposed to allow LLM agents to communicate via intermediate layer activations (instead of natural language) by injecting the activation vector of Model A into the intermediate layers during the forward pass of Model B. This requires zero additional parameters or training data, while improving performance by up to 27% compared to natural language communication across multiple reasoning benchmarks…
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Multi-agent communication"
  - "activation space"
  - "model grafting"
  - "LLM reasoning"
  - "computational efficiency"
date: 2026-05-08
content_hash: d60c34783fcd800b
---

# Communicating Activations Between Language Model Agents

**Conference**: ICML 2025  
**arXiv**: [2501.14082](https://arxiv.org/abs/2501.14082)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Multi-agent communication, activation space, model grafting, LLM reasoning, computational efficiency

## TL;DR
A method is proposed to allow LLM agents to communicate via intermediate layer activations (instead of natural language) by injecting the activation vector of Model A into the intermediate layers during the forward pass of Model B. This requires zero additional parameters or training data, while improving performance by up to 27% compared to natural language communication across multiple reasoning benchmarks, using only 1/4 of the computation.

## Background & Motivation

**Background**: Multiple LLM agents can improve reasoning capabilities through natural language dialogue (e.g., debates), but the computational cost grows rapidly with the number of agents and message length.

**Limitations of Prior Work**: (a) Natural language communication requires a complete generation-and-parsing cycle, which incurs significant computational overhead; (b) The decoding process compresses rich internal representations into a single token, discarding substantial information; (c) Studies show that the intermediate layers of a model contain richer entity representations than the output layer.

**Key Challenge**: Natural language is a communication medium designed for humans and may not be the optimal communication method for LLMs.

**Goal**: Can LLMs communicate using a more efficient, higher-density information medium—directly transmitting activation vectors?

**Key Insight**: Hernandez et al. found that models construct rich entity representations around the halfway point, but compress these into next-token predictions in later layers—implying that intermediate activations carry more information than the final output.

**Core Idea**: Pause the computation of Model B at layer $j$, fuse the activation of Model A's $k$-th layer using a function $f$, and then resume the forward pass of Model B.

## Method

### Overall Architecture
1. Model A receives prompt $x_A$ and performs forward propagation to layer $k$ to obtain activation $h_{A,k}$.
2. Model B receives prompt $x_B$ and performs forward propagation to layer $j$ to obtain activation $h_{B,j}$.
3. Fuses the activation of the last token using function $f$: $h_{B,j}^{\text{new}} = f(h_{A,k}, h_{B,j})$.
4. Resumes the forward propagation of the remaining layers of B to decode the output.

### Key Designs

1. **Activation Fusion Function**:

    - **Function**: Fuses the intermediate activations of the two models.
    - **Mechanism**: Tests four functions: sum ($a+b$), mean ($\frac{a+b}{2}$), replace ($a$), and a learned linear layer.
    - **Design Motivation**: Simple functions (such as mean) require zero extra parameters while already providing significant improvements.

2. **Layer Selection Strategy**:

    - **Function**: Selects the optimal injection layer $j$ and extraction layer $k$.
    - **Mechanism**: Experiments show that intermediate layers (around 40-60% depth) yield the best results.
    - **Design Motivation**: Shallow representations are too primitive, while deep layer representations start to degenerate into next-token predictions.

### Loss & Training
- Completely training-free (sum, mean, and replace have no learnable parameters).
- The learnable linear layer version only requires training on a small amount of task-agnostic data.

## Key Experimental Results

### Main Results

| Task | Natural Language Debate | Activation Comm. (mean) | Gain |
|------|---------------|---------------|------|
| GSM8K | 52.3% | 66.5% | +14.2% |
| MMLU (Avg) | 49.8% | 53.1% | +3.3% |
| Biographies | 68.2% | 86.7% | +18.5% |
| Coordination Game | 41.0% | 68.0% | +27.0% |

Computational cost: <1/4 of natural language communication

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| sum | Good | Summing information might be too aggressive |
| **mean** | **Best** | Balances information between the two models |
| replace | Medium | Completely discards information from B |
| learned linear | Slightly better than mean | Requires a small amount of training |
| Shallow Injection (10%) | Poor | Information is too primitive |
| Intermediate Injection (50%) | **Best** | Equilibrium point |

### Key Findings
- Activation communication is also effective on smaller models (1-7B), unlike natural language debate which is only effective on large models.
- Intermediate layer activations contain richer information than the final output.
- It requires only a partial forward pass of A + a complete forward pass of B, resulting in extremely high computational efficiency.

## Highlights & Insights
- **"Telepathy" between LLMs**—skipping the bottleneck of language to directly transmit high-dimensional representations, which is conceptually elegant.
- The finding that intermediate layers are the "information-richest" layers holds independent research value.
- The method is extremely lightweight—zero parameters, zero training, and plug-and-play.

## Limitations & Future Work
- Requires the two models to have the same hidden dimension size (otherwise projection layers are needed).
- Only transmits the activation of the last token, failing to leverage information from all tokens in the sequence.
- Sensitive to the choice of model pairs—performance varies significantly across different combinations.
- Safety implications have not been analyzed (whether activation injection might lead to harmful outputs).

## Related Work & Insights
- **vs Multi-agent Debate**: Natural language communication, 4×+ computational cost, poorer performance.
- **vs CIPHER**: Communication via tokenizer embeddings, causing greater information loss than activations.
- **vs CALM**: Fusion of activations via learned attention layers, requiring training, whereas ours does not.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering use of activations instead of language for LLM-to-LLM communication.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple tasks, multiple fusion functions, and multiple models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and concise methodology.
- **Value**: ⭐⭐⭐⭐⭐ Holds significant importance for multi-agent LLM systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust](../../ICML2026/llm_evaluation/the_acute_protocol_operationalizing_language_model_activations_for_better_calibr.md)
- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ICML 2025\] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)
- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](../../ICML2026/llm_evaluation/hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)

</div>

<!-- RELATED:END -->
