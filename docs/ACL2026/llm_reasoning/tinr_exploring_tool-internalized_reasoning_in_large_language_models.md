---
title: >-
  [Paper Note] TInR: Exploring Tool-Internalized Reasoning in Large Language Models
description: >-
  [ACL 2026][LLM Reasoning][Tool Internalization] This paper proposes the TInR-U framework, which internalizes tool knowledge into LLM parameters (instead of relying on external documentation) to achieve efficient and reli…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Tool Internalization"
  - "Reasoning"
  - "LLM"
  - "Reinforcement Learning"
  - "Tool Calling"
date: 2026-05-08
content_hash: d388afddfaa532f5
---

# TInR: Exploring Tool-Internalized Reasoning in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.10788](https://arxiv.org/abs/2604.10788)  
**Code**: https://github.com/travis-xu/TInR  
**Area**: LLM Reasoning / Tool Use  
**Keywords**: Tool Internalization, Reasoning, LLM, Reinforcement Learning, Tool Calling

## TL;DR
This paper proposes the TInR-U framework, which internalizes tool knowledge into LLM parameters (instead of relying on external documentation) to achieve efficient and reliable tool-assisted reasoning, outperforming existing methods in both in-domain and out-of-domain tests.

## Background & Motivation

**Background**: Tool-Integrated Reasoning (TIR) has become a mainstream direction for extending LLM capabilities, allowing models to solve tasks beyond their original abilities—such as knowledge updates and real-time queries—by calling external tools during the reasoning process. Existing TIR methods primarily rely on two training strategies: In-Context Learning (ICL) and Supervised Fine-Tuning (SFT). Subsequently, Reinforcement Learning (RL) methods were introduced to enhance exploration and adaptability.

**Limitations of Prior Work**: Despite significant progress, existing TIR methods still face three fundamental issues. First, tool documentation formats are diverse and inconsistent, making it difficult for LLMs to quickly master heterogeneous tool knowledge, creating a gap between external documentation and internal understanding. Second, as the number of tools increases, it is impossible to fit all documentation into the context window; while retrieval strategies can partially mitigate this, they increase pipeline complexity and may lead to a mismatch between retrieval and actual usage. Third, long tool documentation significantly increases prompt length, leading to higher inference latency and computational overhead.

**Key Challenge**: The current paradigm involves an irreconcilable trade-off between efficiency, scalability, and accuracy. The external dependency mode reflects a passive query mindset rather than active mastery of tools.

**Goal**: To explore Tool-Internalized Reasoning (TInR), where LLMs encode tool knowledge into their parameters, enabling the model to reason without relying on external documentation. This requires satisfying two key requirements: (1) Tool internalization—encoding tool functions and usage rules into parameters; (2) Tool-reasoning coordination—seamlessly integrating tool knowledge for adaptive tool use during reasoning.

**Key Insight**: Inspired by how humans internalize tool knowledge into their brains for continuous application, this paper proposes implementing a similar internalization mechanism in LLMs. The core observation is that if a model can "understand" tools instead of "querying" documentation every time, it can simultaneously achieve knowledge unification, context efficiency, and reasoning fluidity.

**Core Idea**: A three-stage training pipeline (Tool Internalization → SFT Warmup → Reinforcement Learning) is used to gradually empower the LLM with tool-internalized reasoning capabilities. Specialized tool tokens serve as parameterized representations, and bi-directional knowledge alignment ensures a balance between fine-grained fidelity and holistic understanding.

## Method

### Overall Architecture

The TInR-U framework consists of three progressive training stages. First, in the tool internalization stage, the model learns to map tool documentation to exclusive tokens and vice versa through a bi-directional knowledge alignment strategy, while undergoing tool-use training to ensure alignment with actual applications. Second, in the SFT warmup stage, high-quality reasoning trajectories constructed via rejection sampling are used to perform supervised fine-tuning, establishing a foundation for reasoning capabilities. Finally, in the reinforcement learning stage, optimization is performed using a composite reward function specifically designed for tool reasoning. During inference, the model only needs to utilize internalized tool knowledge for multi-step reasoning and tool calls, without requiring external documentation.

### Key Designs

1.  **Exclusive Tool Tokens and Bi-directional Knowledge Alignment**:
    -   **Function**: Assigns a unique token to each tool as a parameterized representation and achieves deep internalization of tool knowledge through three complementary loss functions.
    -   **Mechanism**: The tool memorization objective $\mathcal{L}_{\text{memorization}}=-\sum_{t\in\mathcal{T}}\log P(I(t)\mid D(t))$ enables the model to learn the mapping from documentation to tokens; the tool recall objective $\mathcal{L}_{\text{recall}}=-\sum_{t\in\mathcal{T}}\sum_{s=1}^{|D(t)|}\log P(D(t)_s\mid I(t),D(t)_{<s})$ encourages the model to reconstruct the original documentation from tokens; the tool usage objective $\mathcal{L}_{\text{usage}}$ directly trains on instruction-tool call pairs. The total loss is $\mathcal{L}_{\text{Phase1}}=\mathcal{L}_{\text{memorization}}+\alpha\mathcal{L}_{\text{recall}}+\beta\mathcal{L}_{\text{usage}}$.
    -   **Design Motivation**: The bi-directional design preserves fine-grained tool details while establishing holistic understanding. Memorization and recall tasks provide complementary learning signals, while the usage task ensures that internalized knowledge can be effectively applied in practical reasoning.

2.  **Two-Step Tool Call Generation**:
    -   **Function**: Decomposes the tool call generation process via two control tags, `<tool_token>` and `<tool_call>`, first predicting the tool token set and then generating specific parameters based on the documentation.
    -   **Mechanism**: The first step generates the tool token set $\{I(t_i)\}_{i=1}^K$ within the `<tool_token>` scope, and the second step pairs each token with parameters according to the corresponding documentation $\{D(t_i)\}_{i=1}^K$ within the `<tool_call>` scope to generate the full tool call.
    -   **Design Motivation**: This design decouples the complex tool calling task into two relatively simple sub-tasks, reducing the model's burden. Ablation studies show that removing the two-step design causes the EM metric to drop from 61.31% to 43.40%.

3.  **Composite Reward Function and GRPO Optimization**:
    -   **Function**: Employs a composite reward combining format rewards and correctness rewards during the reinforcement learning stage, optimized via Group Relative Policy Optimization (GRPO).
    -   **Mechanism**: The format reward $R_{\text{format}}$ checks whether trajectories contain special tags in the correct sequence; tool reward $r_{\text{tool}}$ and parameter reward $r_{\text{param}}$ are measured using Jaccard similarity; the final reward is $R=R_{\text{format}}+R_{\text{correct}}$. The training objective uses a PPO-style target, where the relative advantage function is normalized within the same batch to stabilize training.
    -   **Design Motivation**: Separating format and correctness rewards ensures both structural validity and semantic accuracy.

### Loss & Training

In the SFT stage, 10 candidate tools (including ground-truth, retrieved, and random tools) are collected for each instruction from datasets like ToolACE, xLAM, and BFCL. A Large Reasoning Model (LRM) is used to generate multiple reasoning trajectories, keeping only the correct ones verified by ground-truth tools. Further data formatting replaces tool names in the reasoning content with their corresponding tokens.

## Key Experimental Results

### Main Results (Seen/Unseen Tools)

| Method | Seen-Tool EM | Seen-Tool Call EM | Unseen-Tool EM | Unseen-Tool Call EM |
| :--- | :--- | :--- | :--- | :--- |
| ToolRetriever+ToolRL | 63.78 | 61.08 | 59.66 | 51.72 |
| ToolGen | 83.78 | 71.89 | 73.79 | 55.86 |
| **Ours (TInR-U)** | **85.95** | **74.05** | **75.86** | **57.24** |

### BFCL Out-of-Domain Results

| Method | Tool Ident.-EM | Tool Call-EM | Tool Acc. | Param Acc. |
| :--- | :--- | :--- | :--- | :--- |
| ToolRetriever+ToolRL | 30.56 | 16.63 | 37.35 | 28.45 |
| ToolGen | 34.89 | 22.01 | 30.91 | 48.24 |
| **Ours (TInR-U)** | **38.06** | **26.00** | **35.83** | **50.12** |
| **Gain** (Rel.) | +9.09% | +18.13% | -4.07% | +3.90% |

### Ablation Study

| Configuration | Tool Ident.-EM | Tool Call-EM | Tool Acc. |
| :--- | :--- | :--- | :--- |
| Full Model | 78.30 | 61.31 | 77.25 |
| w/o Bi-align | 49.67 | 40.39 | 48.63 |
| w/o RL | 76.47 | 59.61 | 74.90 |
| w/o 2-step design | — | 43.40 | 72.94 |
| Memorization only | 58.43 | 45.49 | 57.25 |

### Key Findings

*   **Outstanding Generalization**: On unseen tool sets, TInR-U achieves a 2.81% improvement in tool identification EM compared to ToolGen, with relative out-of-domain generalization gains reaching 18.13%.
*   **Significant Inference Efficiency**: As the tool set size increases, the inference speed of ToolRL continuously decreases, whereas TInR-U maintains constant efficiency.
*   **Strong Multi-step Reasoning**: TInR-U consistently outperforms ToolGen in multi-step/multi-turn tool usage scenarios.
*   **Good Model Compatibility**: TInR-U outperforms ToolGen across three mainstream models: Qwen-2.5B, LLaMA-3.1-8B, and Mistral-7B.
*   **Ablation Insights**: Both bi-directional knowledge alignment and RL training are essential components; the two-step tool call generation is critical (removing it drops tool call EM by over 30%).

## Highlights & Insights

*   **Novelty**: The combination of exclusive tool tokens and bi-directional alignment is an ingenious design that avoids the ambiguity problematic in traditional embedding methods. Compared to pure semantic, numeric, or hierarchical indexing, exclusive tokens lead by over 30 percentage points in tool identification EM.
*   **Decoupled Reasoning and Tool Calling**: The two-step generation framework decomposes complex tasks into two more constrained sub-problems. This "select tool then fill parameters" design philosophy is transferable to any multi-step decision-making problem.
*   **Efficient Inference Paradigm**: Unlike the dynamic querying of retrieval or agent-based frameworks, TInR-U implements a fully parameterized "reasoning as encoding" mode, decoupling inference speed from toolset size.
*   **Clever Use of RL**: The composite reward design constrains both format and semantics simultaneously, and GRPO's group relative normalization avoids sparse reward issues.

## Limitations & Future Work

**Limitations acknowledged by the authors**:
*   Evaluation datasets may not fully cover the diversity of real-world tools.
*   Potential "false negatives" in the dataset (functionally similar tools not labeled as valid).

**Self-discovered limitations**:
*   **Tool Update Costs**: Model parameterization means introducing new tools requires fine-tuning; this may lack flexibility for frequently updated tool ecosystems.
*   **Documentation Fidelity Trade-off**: Parameterized compression always involves some information loss.
*   **Base Model Dependency**: Absolute performance on small models still has an upper bound.

**Specific improvement ideas**: Develop incremental adaptation algorithms to integrate new tools more efficiently; introduce verifiable knowledge distillation methods; extend to multimodal tool scenarios; design hybrid paradigms that retain a small amount of document recall as a fallback mechanism for extreme resource constraints.

## Related Work & Insights

**vs. Traditional TIR methods**: Traditional methods rely on external documentation during inference. This paper avoids context bloating and frequent retrieval costs through parameterized internal knowledge.

**vs. Other internalization attempts (ToolkenGPT, etc.)**: Previous work was limited to small toolsets, simple reasoning strategies, or unstable LLM evaluations. This paper provides more rigorous evaluation, validation in complex tool environments, and a three-stage pipeline specifically designed for reasoning.

**vs. Agent frameworks (DeepAgent, etc.)**: Agent frameworks improve tool selection through iterative reasoning and scalable tool searching but still rely on dynamic retrieval and introduce extra latency. TInR-U avoids such overhead through full parameterization.

**Value**: This paper proves that LLMs can effectively internalize large-scale heterogeneous knowledge and apply it flexibly during reasoning, opening up possibilities for parameterized knowledge representation across multiple domains.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift to tool internalization (from "querying docs" to "parameter internalization") proposes a fundamentally different technical route.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three large-scale datasets, evaluates both in-domain and out-of-domain generalization, and includes detailed ablations, multi-base model validation, and inference efficiency analysis.
*   **Writing Quality**: ⭐⭐⭐⭐ The paper logic is clear and motivation is well-articulated, though some mathematical expressions could be more intuitive.
*   **Value**: ⭐⭐⭐⭐⭐ High practical industrial value; significantly improves inference efficiency with direct potential for application in large-scale tool environments like data centers and API gateways.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)

</div>

<!-- RELATED:END -->
