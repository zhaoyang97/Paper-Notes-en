---
title: >-
  [Paper Note] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky
description: >-
  [ACL 2026][Dialogue Systems][Tool Calling] Proposes the DiaFORGE framework, which utilizes a disambiguation-centric synthetic data generation pipeline, chain-of-thought fine-tuning…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Tool Calling"
  - "Disambiguation"
  - "Multi-turn Dialogue"
  - "Enterprise API"
  - "Fine-tuning"
date: 2026-05-08
content_hash: aaad2207cfb15063
---

# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky

**Conference**: ACL 2026  
**arXiv**: [2507.03336](https://arxiv.org/abs/2507.03336)  
**Code**: [HuggingFace](https://huggingface.co/SAP/diaforge-utc-r-0725)  
**Area**: Dialogue Systems / LLM Agent  
**Keywords**: Tool Calling, Disambiguation, Multi-turn Dialogue, Enterprise API, Fine-tuning

## TL;DR

Proposes the DiaFORGE framework, which utilizes a disambiguation-centric synthetic data generation pipeline, chain-of-thought fine-tuning, and a dynamic evaluation system. This enables open-source LLMs to achieve tool-calling success rates 27 percentage points higher than GPT-4o and 49 percentage points higher than Claude-3.5-Sonnet when encountering near-duplicate enterprise APIs.

## Background & Motivation

**Background**: LLMs are evolving from conversational assistants into operational agents capable of calling APIs. Enterprise environments manage thousands of APIs, many of which are slight variants of core functions (e.g., different versions for customer support, finance, or supply chain).

**Limitations of Prior Work**: In reality, approximately 35-38% of queries retrieve highly similar distractor APIs, 71% of APIs contain mandatory parameters, and 76-81% of calls lack at least one required field. However, existing tool-calling benchmarks (BFCL, ToolBench, API-Bank) utilize static evaluation with pre-written user scripts, failing to expose cascading failure modes involving "incomplete requests + near-duplicate tools."

**Key Challenge**: Enterprise tool-calling requires two tightly intertwined capabilities: multi-turn dialogue to complete missing parameters and fine-grained disambiguation across a dense, overlapping API surface. Existing training data and evaluation methods ignore this intersection.

**Goal**: (1) Construct disambiguation-centric training data, (2) fine-tune open-source models to learn proactive questioning and precise tool selection, and (3) design a dynamic evaluation framework to measure end-to-end goal completion rates.

**Key Insight**: Drawing from real-world enterprise API production telemetry at SAP Labs, the authors identified disambiguation as the core bottleneck in tool calling.

**Core Idea**: Use a "bottom-up" multi-agent data engine to synthesize disambiguation-centric dialogues. By providing the assistant with sets of near-duplicate tools and deliberately withholding critical information, the assistant is forced to learn to disambiguate before execution.

## Method

### Overall Architecture

DiaFORGE is a three-stage pipeline: (1) UTC-Gen data engine for synthesizing training dialogues, (2) supervised fine-tuning with reasoning chains, and (3) a dual-track static and dynamic evaluation. The input is an enterprise tool catalog $\mathcal{T}$ (comprising ~5,000 production-grade API specifications), and the output is a fine-tuned tool-calling model.

### Key Designs

1.  **UTC-Gen Multi-agent Data Engine**:
    - **Function**: Synthesizes disambiguation-centric multi-turn dialogue training data from the bottom up.
    - **Mechanism**: For each seed tool $\tau^*$, an enterprise user persona $p$ is sampled, and a semantic encoder $\phi$ retrieves $k=5$ nearest-neighbor distractor tools to form a candidate pool $\mathcal{C}_k(\tau^*)$. Dialogues unfold in two phases: the Tool Selection Phase (where the user is intentionally vague, forcing the assistant to ask questions to exclude distractors) and the Parameter Completion Phase (where the assistant requests missing mandatory fields sequentially). All dialogues undergo a three-stage validation (format, relevance, LLM-based critique) before inclusion.
    - **Design Motivation**: Existing datasets assume user requests are fully specified, failing to train models for disambiguation scenarios. By injecting near-duplicate distractors and a two-phase mandatory dialogue protocol, the assistant is structurally forced to learn disambiguation.

2.  **Supervised Fine-tuning with Reasoning Chains**:
    - **Function**: Enables the model to generate interpretable reasoning processes before calling a tool.
    - **Mechanism**: Employs a turn-slicing strategy. For each assistant turn, input-target pairs are constructed as $x_{i,t} = [\text{SYS}]\;u_1\;a_1\;\ldots\;u_t$ and $y_{i,t} = a_t$. Each assistant response is divided into a private reasoning chain (thought process) and a public response; both are treated as learning targets during training. Fine-tuning is performed using LoRA, with loss calculated only on the completion tokens.
    - **Design Motivation**: Tool selection requires not just "getting it right" but "knowing why." Reasoning chains allow the model to explicitly exclude distractor tools rather than relying on pattern matching.

3.  **DiaBENCH Dynamic Evaluation Protocol**:
    - **Function**: Evaluates end-to-end goal completion rates within a real-time dialogue loop.
    - **Mechanism**: The fine-tuned model is inserted as an assistant into the UTC-Gen loop, while the user agent policy remains frozen. Interactions continue for up to $T_{max}$ turns to generate a complete trajectory. Three core metrics are tracked: Accuracy (Acc, both tool and parameters are correct), False Trigger Rate (FTR, wrong tool called), and Total Abstention Rate (TAR, no tool called). The user agent employs a multi-sampling and voting strategy to reduce evaluation noise.
    - **Design Motivation**: Static evaluation cannot capture the cascading effects of how assistant outputs influence subsequent user behavior. Dynamic evaluation more closely reflects real-world scenarios.

### Loss & Training

Standard SFT + LoRA using the AdamW optimizer for one epoch. Training data consists of 13,649 turn-sliced completion samples generated from 5,000 DiaFORGE dialogues. Loss masking is applied to ensure loss is only computed on completion tokens.

## Key Experimental Results

### Main Results

DiaBENCH dynamic evaluation results (Accuracy Acc↑ / False Trigger Rate FTR↓ / Total Abstention Rate TAR↓):

| Model | Acc↑ | FTR↓ | TAR↓ |
|------|------|------|------|
| GPT-4o | 0.62 | 0.02 | 0.36 |
| GPT-4o-fc | 0.56 | 0.59 | 0.05 |
| Claude-3.5-Sonnet | 0.39 | 0.03 | 0.55 |
| Gemma-3-DiaFORGE-27B | **0.89** | 0.03 | 0.03 |
| Nemotron-DiaFORGE-49B | **0.89** | 0.06 | 0.03 |
| Gemma-3-DiaFORGE-4B | 0.81 | 0.09 | 0.05 |
| Llama-3.2-DiaFORGE-3B | 0.80 | 0.08 | 0.06 |

### Ablation Study

Ablation based on Gemma-3-27B (Dynamic Eval Acc):

| Setting | Acc↑ | FTR↓ | TAR↓ |
|------|------|------|------|
| Full DiaFORGE | 0.89 | 0.03 | 0.03 |
| w/o Validation Cascade | 0.56 | 0.06 | 0.35 |
| w/o Near-duplicate Sampling | 0.63 | 0.18 | 0.19 |
| w/o Reasoning Chain | 0.77 | 0.16 | 0.04 |

### Key Findings

- With only 5,000 synthetic dialogues for fine-tuning, a 3B small model can outperform GPT-4o in dynamic evaluation (0.80 vs. 0.62).
- Native function-calling modes (indicated by the -fc suffix) actually increase the false trigger rate: GPT-4o-fc's FTR reaches 0.59.
- In a scenario with 10K tool calls per day, GPT-4o would result in ~3,500-3,800 abstentions or 5,500-6,000 miscalls, whereas the DiaFORGE model incurs only 250-350 total failures.
- Near-duplicate distractor sampling is the most critical component; removing it causes the FTR to jump from 0.03 to 0.18.

## Highlights & Insights

- Data-driven insights from SAP production environments are highly compelling: 35-38% of queries encounter near-duplicate tools, and 76-81% of calls lack parameters.
- Shifting the perspective of disambiguation from a "secondary requirement" to a "core training objective" is highly instructive.
- The dynamic evaluation framework fills a major gap in existing tool-calling benchmarks.

## Limitations & Future Work

- DiaBENCH contains only 119 seed tools, which is limited in scale.
- User agents are still simulated by LLMs, which may differ from real user behavior.
- Retrieval-augmented tool selection was not explored; as tool counts grow further, retrieval quality will become a new bottleneck.

## Related Work & Insights

- ReAct and HuggingGPT established the basic paradigm of LLMs as tool-calling agents; DiaFORGE builds on this by adding disambiguation capabilities.
- APIGen and ToolACE focus on data validation but assume fully specified requests; DiaFORGE’s disambiguation-centric strategy is a crucial complement.
- **Insight**: The core challenge for enterprise-grade AI agents is not just "can it call a tool," but "can it safely refrain from calling or clarify first when faced with ambiguity."

## Rating

- Novelty: ⭐⭐⭐⭐ The disambiguation-centric problem definition and systematic solution are unique in the tool-calling field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 6 open-source and 2 closed-source models with both static and dynamic tracks and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, persuasive industrial perspective, and consistent logic despite heavy use of notation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)

</div>

<!-- RELATED:END -->
