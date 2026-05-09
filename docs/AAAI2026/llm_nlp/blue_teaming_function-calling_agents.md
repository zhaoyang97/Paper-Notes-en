---
title: >-
  [Paper Note] Blue Teaming Function-Calling Agents
description: >-
  [AAAI 2026][LLM/NLP][function-calling] This paper systematically evaluates the robustness of four open-source function-calling LLMs against three attack types, and assesses the effectiveness of eight defense mechanisms, revealing that current models are insecure by default and that existing defenses remain difficult to deploy in practice.
tags:
  - AAAI 2026
  - LLM/NLP
  - function-calling
  - prompt injection attack
  - tool poisoning
  - blue teaming defense
  - open-source LLM
date: 2026-05-08
content_hash: 07f59841d8b0bfd4
---

# Blue Teaming Function-Calling Agents

**Conference**: AAAI 2026
**arXiv**: [2601.09292](https://arxiv.org/abs/2601.09292)
**Code**: None
**Area**: LLM/NLP
**Keywords**: function-calling, prompt injection attack, tool poisoning, blue teaming defense, open-source LLM

## TL;DR

This paper systematically evaluates the robustness of four open-source function-calling LLMs against three attack types, and assesses the effectiveness of eight defense mechanisms, revealing that current models are insecure by default and that existing defenses remain difficult to deploy in practice.

## Background & Motivation

**Background**: Function-calling capabilities enable LLMs to interact with external tools, extending their utility beyond text generation. With the proliferation of protocols such as A2A and MCP, agentic applications are becoming increasingly prevalent.

**Limitations of Prior Work**: Function-calling capability does not guarantee robustness against adversarial attacks; even with defenses in place, models may still be induced to invoke malicious functions. Existing research has primarily demonstrated attack feasibility on closed-source models, leaving a systematic evaluation of open-source models largely absent.

**Key Challenge**: Open-source function-calling models must expose tool implementation details (e.g., source code), creating a unique attack surface that does not exist in closed-source models. Meanwhile, existing defense mechanisms suffer from high false-positive rates and other practical deployment challenges.

**Goal**: To systematically evaluate the security of open-source function-calling models, quantify attack success rates (ASR) and defense effectiveness, and provide empirical evidence to guide the design of more secure agentic systems.

**Key Insight**: Adopting a blue-team (defender) perspective, the paper simultaneously implements attacks and defenses within a unified end-to-end evaluation framework, with particular focus on novel attack vectors introduced by tool implementation visibility.

**Core Idea**: The visibility of tool implementations creates a unique attack surface for adversaries (e.g., Renaming Tool Poisoning), while current defense solutions—both preventive and reactive—fail to provide comprehensive protection, necessitating multi-layered, combined defenses.

## Method

### Overall Architecture

Using the Ollama and DSPy frameworks, attack-defense evaluations are conducted on four models—Qwen3:8B, Llama-3.2:3B, Granite3.2:8B, and Granite3.3:8B—using the Berkeley Function Calling Leaderboard dataset. A malicious target function `get_result` containing SQL injection code serves as the unified measure of ASR.

### Key Designs

1. **Three Attack Types**:

    - **Direct Prompt Injection (DPI)**: Malicious instructions are directly embedded in user queries, disguised as system administrator messages to override original instructions. ASR ranges from 56% to 94% across most models, making it the most effective attack.
    - **Simple Tool Poisoning (STP)**: Tool descriptions are modified by appending a malicious payload to each, while the malicious function is added to the tool list. ASR reaches 95% on Qwen3:8B.
    - **Renaming Tool Poisoning (RTP)**: A novel attack proposed in this paper that simultaneously manipulates both tool descriptions and implementation code, employing a dual-payload strategy to redirect model attention toward malicious variables in the implementation. Effective only against Qwen3:8B (ASR 74%), suggesting that model attends more closely to tool implementations.

2. **Four Preventive Defenses**:

    - **Cosine Similarity**: An embedding model computes query-tool similarity, delegating tool selection to the embedding model. Effectiveness varies; ASR drops to 0 in some scenarios, but accuracy may also degrade significantly.
    - **Tool Obfuscation**: A novel defense proposed in this paper that applies code obfuscation to tool names and implementations, removing variable and tool names as attack vectors. Generally beneficial across most model-attack combinations.
    - **Description Rewriting**: An LLM (Granite-Code:8B) regenerates tool descriptions from implementations, establishing a strong binding between descriptions and actual functionality. Reduces ASR for tool poisoning attacks to 0.

3. **Four Reactive Defenses**:

    - **Watermarking**: Uses HMAC-SHA256 to generate watermarks for legitimate tool names, verifying tool authenticity prior to execution. Detects 100% of malicious function calls; however, Llama-3.2:3B fails to reliably reproduce watermarks.
    - Additional defenses including Prompt Shields, LLM-as-a-Judge, and Intention Analysis are evaluated, all of which exhibit high false-positive rates.

### Loss & Training

This paper is an experimental evaluation study and does not involve model training. Evaluation metrics are Accuracy (proportion of correct tool calls) and ASR (proportion of queries successfully inducing malicious function calls), assessed on 172 query-answer pairs.

## Key Experimental Results

### Main Results

| Attack Type | Qwen3:8B ACC/ASR | Llama3.2:3B ACC/ASR | Granite3.2:8B ACC/ASR | Granite3.3:8B ACC/ASR |
|-------------|-----------------|--------------------|-----------------------|-----------------------|
| No Attack | 0.92/0 | 0.66/0 | 0.84/0 | 0.78/0 |
| DPI | 0.06/0.94 | 0.20/0.58 | 0.34/0.56 | 0.80/0 |
| STP | 0.04/0.95 | 0.50/0.23 | 0.72/0.12 | 0.39/0.51 |
| RTP | 0.24/0.74 | 0.69/0.02 | 0.84/0.01 | 0.83/0 |

### Ablation Study

- Description Rewriting reduces ASR for STP and RTP to 0 with negligible impact on accuracy.
- Watermarking intercepts 100% of malicious function calls (pre-execution detection).
- Cosine Similarity yields unstable results; in some scenarios ASR increases to 0.64.
- Tool Obfuscation is generally beneficial for all models except Llama3.2:3B.

### Key Findings

- Qwen3:8B is the most vulnerable to all attacks (likely due to greater reliance on tool implementation details), while Granite3.3:8B demonstrates the strongest robustness.
- **No single defense covers all attack types**; every evaluated defense has significant practical limitations.
- Tool implementation visibility in open-source models introduces unique attack vectors absent in closed-source counterparts.
- Security is generally insufficient at smaller model scales (3B–8B).

## Highlights & Insights

- This work provides the first systematic security evaluation of open-source function-calling models, filling a gap beyond closed-source assessments.
- A novel attack (RTP) and a novel defense (Tool Obfuscation) are proposed, both simple yet effective.
- An important finding is revealed: different models vary in the degree to which they attend to tool descriptions versus implementation code, which substantially influences attack effectiveness.
- Granite3.3:8B is found to be completely immune to DPI (ASR = 0), warranting further investigation into its safety alignment mechanism.

## Limitations & Future Work

- Only small open-source models (3B–8B) are evaluated; comparisons with larger models (70B+) and closed-source models are absent.
- Attack scenarios are limited to single function calls; multi-step reasoning and chained tool invocations are not addressed.
- Defense combination experiments are insufficient; the optimal strategy for stacking multiple defenses remains unexplored.
- Adaptive adversaries—attackers who adjust their strategies upon learning the deployed defenses—are not considered.

## Related Work & Insights

- ASB (Zhang et al. 2025) proposes a comprehensive evaluation framework with 400+ tools, achieving a maximum ASR of 84.30%.
- DRIFT (Li et al. 2025) proposes dynamic rule-based protection, and Meta SecAlign constructs LLMs with built-in defenses.
- The findings of this paper can inform security design in MCP/A2A protocols; tool registration should incorporate signature verification mechanisms.

## Rating

⭐⭐⭐ (3/5)

The experimental design is systematic and comprehensive, but technical depth is limited, as the contribution is primarily empirical. The findings are valuable—open-source models are insecure by default and no single defense is universally effective—but the work lacks theoretical analysis and stronger defense proposals. It nonetheless serves as a useful reference for the agentic security community.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/llm_nlp/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](../../ICLR2026/llm_nlp/meta-rl_induces_exploration_in_language_agents.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](../../ICLR2026/llm_nlp/enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_f.md)

<!-- RELATED:END -->
