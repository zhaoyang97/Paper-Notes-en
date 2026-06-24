---
title: >-
  [Paper Note] AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs
description: >-
  [ACL 2025][LLM Alignment][agent safety alignment] This paper proposes the AgentAlign framework, which leverages abstract behavior chains as an intermediary to synthesize high-quality agent safety alignment data (both harmful and benign) in simulated environments. Through Supervised Fine-Tuning (SFT), AgentAlign improves the agent safety of three open-source model families by 35.8%–79.5% while maintaining or even enhancing their task capabilities.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "agent safety alignment"
  - "behavior chain"
  - "agentic LLM"
  - "tool use"
  - "safety-utility trade-off"
date: 2026-05-08
content_hash: 23422fb3feef9b78
---

# AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.23020](https://arxiv.org/abs/2505.23020)  
**Code**: [https://github.com/](https://github.com/) (Open-sourced; see paper for the specific link)  
**Area**: LLM Alignment / Agent Safety  
**Keywords**: agent safety alignment, behavior chain, agentic LLM, tool use, safety-utility trade-off

## TL;DR
This paper proposes the AgentAlign framework, which leverages abstract behavior chains as an intermediary to synthesize high-quality agent safety alignment data (both harmful and benign) in simulated environments. Through Supervised Fine-Tuning (SFT), AgentAlign improves the agent safety of three open-source model families by 35.8%–79.5% while maintaining or even enhancing their task capabilities.

## Background & Motivation

LLMs are transitioning from functioning as "information providers" to "action executors"—possessing capabilities to search for information, operate browsers, execute code, or even directly control computers. This role shift introduces completely new safety risks: whereas prior LLM misuse was primarily about "providing harmful advice", they can now **execute harmful tasks end-to-end**, such as completing a multi-step DDoS attack (searching for scripts $\rightarrow$ downloading $\rightarrow$ installing dependencies $\rightarrow$ executing).

Key Challenge: Existing LLMs perform well in traditional text safety alignment (such as AdvBench), with Gemini/GPT-4o-mini achieving refusal rates near 90%. However, their safety drops sharply in agent scenarios (such as AgentHarm)—**the same models exhibit refusal rates of less than 20%**. This indicates a severe lack of safety alignment specifically targeted at agent use cases during the post-training phase.

Two major challenges hinder the advancement of agent safety alignment:

**Difficulty in acquiring high-quality agent instructions**: Human annotation is prohibitively expensive, and generating them directly with LLMs often results in low-quality, impractical instructions (the instructions cannot be associated with specific tools or lack crucial information required for execution).

**Difficulty in balancing the safety-utility trade-off**: Simple safety training often leads to over-refusal, where the model unnecessarily rejects benign requests.

Core Idea: Leveraging the observation that "harmful activities often follow similar behavioral patterns," this work proposes using an **abstract behavior chain** as an intermediary to generate alignment data. It first constructs behavioral patterns and then instantiates them into concrete executable instructions within simulated environments, thereby ensuring both data authenticity and executability.

## Method

### Overall Architecture
AgentAlign consists of four closely linked modules:
1. **Abstract Behavior Chain Construction** $\rightarrow$ Capturing general behavioral patterns of harmful activities
2. **Grounded Instruction Synthesis** $\rightarrow$ Instantiating behavior chains into concrete executable instructions
3. **Quality Control Pipeline** $\rightarrow$ Ensuring the semantic correctness and executability of instructions
4. **Response Generation** $\rightarrow$ Generating execution trajectories for benign instructions and refusal responses for harmful instructions

The final output is the AgentAlign dataset containing 18,749 data instances (9,783 benign + 4,956 harmful + 4,010 third-party supplementary).

### Key Designs

1. **Abstract Behavior Chain Construction**:

    - **Function**: Constructing abstract action sequences that represent the behavioral patterns of harmful activities.
    - **Mechanism**: By selecting 7 high-risk categories from 49 groups in RapidAPI Hub, and supplementing them with System_Tools and Local_Services, 42 abstract tool capabilities (e.g., `web_search`, `manage_files`, `send_email`) are extracted to construct the action space $\mathcal{A}$. Combining an 8-category, 64-subcategory taxonomy of harmful behaviors, seed behavior chains are manually created and then expanded and synthesized using LLMs. Each behavior chain is denoted as $\beta = (a_1, \ldots, a_k)$, where $a_i \in \mathcal{A}$ and the length is $k \in [1, 5]$. After manual audit and filtering, **240 high-quality behavior chains** are obtained.
    - **Design Motivation**: The behavior chains provide a "reusable skeletal structure of harmful behaviors." A single behavior chain can be instantiated into countless concrete instructions, greatly enhancing the efficiency of data generation. Designing at the abstract level also avoids the limitations of directly generating explicit harmful content.

2. **Grounded Instruction Synthesis**:

    - **Function**: Grounding abstract behavior chains into concrete, executable instructions.
    - **Mechanism**: Implementing multiple concrete tools for each abstract action (e.g., `web_search` $\rightarrow$ `google_search` / `bing_search` / `baidu_search`) to construct simulated environments. For a behavior chain with $N$ steps and $M$ tool choices per step, $M^N$ combinations are possible. By sampling from these combinations, LLMs are used to generate harmful instructions from a red-teaming perspective (ensuring tool parameters can be inferred from the instruction context), benign instructions from a product manager perspective (non-malicious interpretations of the same behavior chain), and supplementary edge cases (e.g., lawful but sensitive scenarios like security testing or medical research).
    - **Design Motivation**: The simulated environment addresses the issues of uncontrollable real APIs and poor data quality (RapidAPI Hub was evaluated but abandoned due to excessively low quality). The $M^N$ combination space guarantees instruction diversity. Sharing the same behavior chain skeleton between benign and harmful instructions helps the model accurately learn safety boundaries.

3. **Double Quality Control**:

    - **Function**: Ensuring the semantic correctness and executability of the generated instructions.
    - **Mechanism**:
        - **Semantic Verification**: Performing asymmetric validation via LLMs—using a lenient standard on harmful instructions to check if they have benign interpretations (reducing misclassification), and a strict standard on benign instructions to check if they have harmful interpretations (reducing missed detections).
        - **Execution Verification**: Passing instructions that pass semantic validation to an almost non-refusal model (Mistral-Large) for execution, filtering out instructions with missing parameters or incoherent logic.
    - **Design Motivation**: The asymmetric design precisely calibrates safety boundaries, avoiding boundary blurring caused by simple binary classification.

4. **Response Generation & Data Balancing**:

    - **Function**: Generating (instruction, response) pairs for alignment training.
    - **Mechanism**: For benign instructions, Mistral-Large is allowed to interact with the simulated environment to generate multi-step execution trajectories (averaging 3.48 steps). For harmful instructions, Claude-3.5-Haiku is used to generate refusal responses. Incorrect responses (e.g., false refusals for benign prompts, false execution for harmful ones) are filtered. Supplementary third-party data from ToolACE (1,840) and Glaive (2,170) are added to increase diversity.
    - **Design Motivation**: Playing to the strengths of different models—models with strong execution capabilities generate trajectories, while models with superior alignment generate refusals. The data ratio was determined via pilot experiments to find the optimal balance.

### Loss & Training
- The three model families are fine-tuned uniformly using **LoRA/QLoRA**.
- Ministral-8B: LoRA rank=128, lr=3e-5, max_steps=800
- Qwen-2.5-7B: LoRA rank=128, alpha=256, lr=3e-5, 1 epoch
- Functionary-Small-v3.2: QLoRA, lr=2e-5, 1 epoch
- **Key Finding**: Excessive training steps lead to safety overfitting (over-refusal); approximately 1 epoch is recommended.

## Key Experimental Results

### Main Results (AgentHarm Benchmark)

| Model | Method | Harmful Score↓ | Harmful Refusal↑ | Benign Score↑ | Benign Refusal↓ |
|------|------|-----------|-------------|-----------|-------------|
| Ministral-8B | Standard | 67.4% | 0.0% | 69.1% | 0.0% |
| Ministral-8B | **+AgentAlign** | **10.5%** | **79.5%** | 63.3% | 2.8% |
| Qwen-2.5-7B | Standard | 41.9% | 21.6% | 53.4% | 0.0% |
| Qwen-2.5-7B | **+AgentAlign** | **6.7%** | **85.8%** | **64.2%** | 5.7% |
| Functionary | Standard | 21.7% | 52.8% | 45.9% | 0.6% |
| Functionary | **+AgentAlign** | **5.5%** | **88.6%** | **53.5%** | 1.7% |
| Claude-3.5-Haiku | - | 10.4% | 86.4% | 68.6% | **15.9%** |

### Ablation Study (Based on Qwen-2.5-7B)

| Configuration | Harmful Score↓ | Harmful Refusal↑ | Benign Score↑ | Description |
|------|-----------|-------------|-----------|------|
| Full AgentAlign | 6.7% | 85.8% | 64.2% | Optimal balance |
| Remove benign samples | ~10% | ~90% | ~35% | Benign capability drops significantly, over-refusal |
| Remove harmful samples | ~40% | ~20% | ~55% | Safety awareness almost completely lost |
| Remove third-party data | ~8% | ~88% | ~58% | Small impact, mainly increases refusal rate |

### Orthogonality with Prompting Methods (AgentAlign + Prompting)

| Method | Ministral Harmful Refusal↑ | Qwen Harmful Refusal↑ |
|------|---------------------|-----------------|
| AgentAlign | 79.5% | 85.8% |
| AgentAlign + Refusal Prompt | **88.6%** | **92.0%** |
| AgentAlign + ReAct | 75.6% | 83.5% |

### Key Findings
- The scale of safety improvement is **negatively correlated** with the initial safety of the base model: less secure models exhibit larger gains (e.g., Ministral improves from 0% to 79.5%).
- AgentAlign has minimal impact on benign tasks; performance on Qwen even increases (53.4% $\rightarrow$ 64.2%).
- Compared to Claude-3.5-Haiku, AgentAlign aligned models achieve comparable safety levels while maintaining a **significantly lower false refusal rate** (1.7%–5.7% vs. 15.9%).
- Generalizability is also validated on the ToolSword benchmark: Ministral improves from 58.2% to 100%.

## Highlights & Insights
- **Behavior chain abstraction** is the core novelty: abstracting harmful activities into tool action sequences enables an "execute once, instantiate infinitely" data amplification effect.
- **Asymmetric semantic validation** cleverly addresses the safety boundary calibration issue—applying strict criteria to benign queries and lenient criteria to harmful ones.
- **Simulated Environments > Real APIs**: RapidAPI Hub has poor quality and lacks write operations; building custom simulation environments yielded better results.
- The ratio of safety alignment data is crucial; excessive safety data severely degrades utility.
- A human evaluation majority-pass rate of 93% validates the feasibility of low-cost automatic synthesis schemes.

## Limitations & Future Work
- Discrepancies between simulated environments and real APIs may affect transferability.
- Dynamic multi-turn interaction scenarios (e.g., users changing requirements midway) are not considered.
- Approximately 7% of LLM-generated data still contains defects (according to human evaluation).
- Evaluation is restricted to AgentHarm and ToolSword, inheriting the limitations of these native benchmarks.
- The behavior chain design depends on manual seeds and auditing, leaving room for further automation.

## Related Work & Insights
- **vs. ToolAlign (Chen et al., 2024)**: ToolAlign is based on modifying existing tool-use datasets; its instructions remain heavily biased toward information retrieval (mostly GET methods) and lack critical write operations for agent scenarios. AgentAlign constructs behavior chains and simulated environments from scratch, covering a wider range of multi-step agent actions.
- **vs. AgentHarm (Andriushchenko et al., 2024)**: AgentHarm serves as an evaluation benchmark, which identified agent safety vulnerabilities but did not provide solutions; AgentAlign directly offers an alignment training scheme.
- **vs. General RLHF/DPO Alignment**: Traditional alignment focuses on safety in single-turn text dialogues, failing to cover safety in multi-step tool-invocation scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The behavior-chain-mediated data synthesis framework is creative, but the core mechanism (synthesizing safety data $\rightarrow$ SFT) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates three model families, various baselines, ablation studies, orthogonal experiments, human evaluation, and cross-benchmark validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Strong problem formulation, systematic and comprehensive method descriptions, and well-designed experiments.
- Value: ⭐⭐⭐⭐ Fills a crucial void in agent safety alignment training; the open-source dataset and codebase are highly beneficial to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[ACL 2025\] MPO: Multilingual Safety Alignment via Reward Gap Optimization](mpo_multilingual_safety_alignment.md)

</div>

<!-- RELATED:END -->
