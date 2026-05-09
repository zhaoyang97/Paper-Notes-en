---
title: >-
  [Paper Note] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky
description: >-
  [ACL 2026][Dialogue Systems][tool calling] This paper proposes DiaFORGE, a disambiguation-centric synthetic data generation pipeline combined with chain-of-thought fine-tuning and a dynamic evaluation framework, enabling open-source LLMs to achieve tool-calling success rates 27 percentage points higher than GPT-4o and 49 percentage points higher than Claude-3.5-Sonnet when facing near-duplicate enterprise APIs.
tags:
  - ACL 2026
  - Dialogue Systems
  - tool calling
  - disambiguation
  - multi-turn dialogue
  - enterprise API
  - fine-tuning
date: 2026-05-08
content_hash: ba04565a840dc889
---

# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky

**Conference**: ACL 2026
**arXiv**: [2507.03336](https://arxiv.org/abs/2507.03336)
**Code**: [HuggingFace](https://huggingface.co/SAP/diaforge-utc-r-0725)
**Area**: Dialogue Systems / LLM Agent
**Keywords**: tool calling, disambiguation, multi-turn dialogue, enterprise API, fine-tuning

## TL;DR

This paper proposes DiaFORGE, a disambiguation-centric synthetic data generation pipeline combined with chain-of-thought fine-tuning and a dynamic evaluation framework, enabling open-source LLMs to achieve tool-calling success rates 27 percentage points higher than GPT-4o and 49 percentage points higher than Claude-3.5-Sonnet when facing near-duplicate enterprise APIs.

## Background & Motivation

**State of the Field**: LLMs are evolving from conversational assistants into operational agents capable of invoking APIs. Enterprise environments manage tens of thousands of APIs, many of which are subtle variants of core functionalities (e.g., different versions for customer support, finance, and supply chain).

**Limitations of Prior Work**: In practice, approximately 35–38% of queries retrieve highly similar distractor APIs, 71% of APIs have required parameters, and 76–81% of calls are missing at least one required field. Existing tool-calling benchmarks (BFCL, ToolBench, API-Bank) rely on pre-scripted user queries for static evaluation and fail to expose the cascading failure patterns arising from incomplete requests combined with near-duplicate tools.

**Root Cause**: Enterprise tool calling demands two tightly intertwined capabilities—multi-turn dialogue to elicit missing parameters, and fine-grained disambiguation across densely overlapping API surfaces—yet both existing training data and evaluation methodologies neglect this coupling.

**Paper Goals**: (1) Construct disambiguation-centric training data; (2) fine-tune open-source models to learn proactive clarification and precise tool selection; (3) design a dynamic evaluation framework to measure end-to-end task completion rates.

**Starting Point**: The authors, from SAP Labs, draw on production telemetry from real enterprise API environments, from which they identify disambiguation as the central bottleneck in tool calling.

**Core Idea**: A bottom-up multi-agent data engine synthesizes disambiguation-centric dialogues by providing the assistant with near-duplicate tool sets and deliberately withholding critical information, structurally compelling the assistant to learn to disambiguate before invoking any tool.

## Method

### Overall Architecture

DiaFORGE is a three-stage pipeline: (1) the UTC-Gen data engine synthesizes training dialogues; (2) supervised fine-tuning with chain-of-thought reasoning; (3) dual-track static and dynamic evaluation. The input is an enterprise tool catalog $\mathcal{T}$ (approximately 5,000 production-grade API specifications), and the output is a fine-tuned tool-calling model.

### Key Designs

1. **UTC-Gen Multi-Agent Data Engine**:

    - **Function**: Bottom-up synthesis of disambiguation-centric multi-turn dialogue training data.
    - **Mechanism**: For each seed tool $\tau^*$, an enterprise user persona $p$ is sampled, and a semantic encoder $\phi$ retrieves $k=5$ nearest-neighbor distractor tools to form a candidate pool $\mathcal{C}_k(\tau^*)$. Dialogues unfold in two phases: a tool selection phase (in which the user is intentionally vague and the assistant must ask clarifying questions to eliminate distractor tools) followed by a parameter completion phase (in which the assistant elicits each missing required field). All dialogues pass three-level validation—format, relevance, and LLM critique—before being added to the dataset.
    - **Design Motivation**: Existing datasets assume fully specified user requests and cannot train models for disambiguation scenarios. By injecting near-duplicate distractor tools and enforcing a two-phase dialogue protocol, the engine structurally compels the assistant to learn disambiguation.

2. **Supervised Fine-Tuning with Chain-of-Thought**:

    - **Function**: Train the model to produce interpretable reasoning prior to tool invocation.
    - **Mechanism**: A turn-slicing strategy is adopted, constructing input–target pairs for each assistant turn as $x_{i,t} = [\text{SYS}]\;u_1\;a_1\;\ldots\;u_t$ and $y_{i,t} = a_t$. Each assistant response consists of a private reasoning chain (thinking process) and a public response, both included as learning targets. LoRA is used for fine-tuning, with loss computed only over completion tokens.
    - **Design Motivation**: Tool selection requires not only correctness but also explainability. The reasoning chain enables the model to explicitly rule out distractor tools rather than relying on pattern matching.

3. **DiaBENCH Dynamic Evaluation Protocol**:

    - **Function**: Assess end-to-end task completion rates within a live dialogue loop.
    - **Mechanism**: The fine-tuned model is inserted as the assistant in the UTC-Gen loop, with the user agent policy kept frozen. Up to $T_{max}$ interaction turns are conducted to generate complete trajectories. Three core metrics are tracked: accuracy Acc (both tool and parameters correct), false trigger rate FTR (wrong tool invoked), and tool abstention rate TAR (no tool invoked). The user agent employs a multi-sampling and voting strategy to reduce evaluation noise.
    - **Design Motivation**: Static evaluation cannot capture the cascading effects of how assistant outputs influence subsequent user behavior; dynamic evaluation more faithfully reflects real-world scenarios.

### Loss & Training

Standard SFT with LoRA and the AdamW optimizer, trained for a single epoch. The training data consists of 13,649 turn-sliced completion samples derived from 5,000 DiaFORGE dialogues. Loss masking is applied so that loss is computed only over completion tokens.

## Key Experimental Results

### Main Results

DiaBENCH dynamic evaluation results (tool-calling accuracy Acc↑ / false trigger rate FTR↓ / abstention rate TAR↓):

| Model | Acc↑ | FTR↓ | TAR↓ |
|-------|------|------|------|
| GPT-4o | 0.62 | 0.02 | 0.36 |
| GPT-4o-fc | 0.56 | 0.59 | 0.05 |
| Claude-3.5-Sonnet | 0.39 | 0.03 | 0.55 |
| Gemma-3-DiaFORGE-27B | **0.89** | 0.03 | 0.03 |
| Nemotron-DiaFORGE-49B | **0.89** | 0.06 | 0.03 |
| Gemma-3-DiaFORGE-4B | 0.81 | 0.09 | 0.05 |
| Llama-3.2-DiaFORGE-3B | 0.80 | 0.08 | 0.06 |

### Ablation Study

Ablation based on Gemma-3-27B (dynamic evaluation Acc):

| Configuration | Acc↑ | FTR↓ | TAR↓ |
|---------------|------|------|------|
| Full DiaFORGE | 0.89 | 0.03 | 0.03 |
| w/o validation cascade | 0.56 | 0.06 | 0.35 |
| w/o near-duplicate distractor sampling | 0.63 | 0.18 | 0.19 |
| w/o chain-of-thought | 0.77 | 0.16 | 0.04 |

### Key Findings

- Fine-tuning on only 5,000 synthetic dialogues enables a 3B model to surpass GPT-4o in dynamic evaluation (0.80 vs. 0.62).
- Native function-calling mode (the `-fc` suffix) increases the false trigger rate: GPT-4o-fc reaches an FTR of 0.59.
- In a scenario with 10K daily tool calls, GPT-4o incurs approximately 3,500–3,800 abstentions or 5,500–6,000 erroneous invocations per day, whereas DiaFORGE models produce only 250–350 total failures.
- Near-duplicate distractor sampling is the most critical component; removing it causes FTR to jump from 0.03 to 0.18.

## Highlights & Insights

- The data-driven insights derived from SAP production environments are highly compelling: 35–38% of queries encounter near-duplicate tools, and 76–81% of calls have missing parameters.
- The conceptual shift of elevating disambiguation from an ancillary requirement to a primary training objective is particularly inspiring.
- The dynamic evaluation framework addresses a significant gap in existing tool-calling benchmarks.

## Limitations & Future Work

- DiaBENCH covers only 119 seed tools, limiting its scale.
- The user agent is still simulated by an LLM, which may diverge from real user behavior.
- Retrieval-augmented tool selection is not explored; as the number of tools grows further, retrieval quality will become a new bottleneck.

## Related Work & Insights

- ReAct and HuggingGPT establish the foundational paradigm of LLMs as tool-calling agents; DiaFORGE extends this by incorporating disambiguation capabilities.
- APIGen and ToolACE focus on data validation but assume fully specified requests; the disambiguation-centric strategy of DiaFORGE serves as an important complement.
- Key insight: the central challenge for enterprise-grade AI agents is not whether they can call tools, but whether they can safely refrain from calling—or first seek clarification—when faced with ambiguity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The disambiguation-centric problem formulation and systematic solution stand out distinctively in the tool-calling literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six open-source models and two closed-source models are evaluated under dual-track static and dynamic evaluation with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear and the industrial perspective is persuasive; mathematical notation is somewhat dense but the overall logic is coherent.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[AAAI 2026\] Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](../../AAAI2026/dialogue/emergent_persuasion_will_llms_persuade_without_being_prompted.md)
- [\[AAAI 2026\] Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](../../AAAI2026/dialogue/chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)
- [\[AAAI 2026\] Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](../../AAAI2026/dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

<!-- RELATED:END -->
