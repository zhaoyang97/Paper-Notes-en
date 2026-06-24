---
title: >-
  [Paper Note] PersonaVLM: Long-Term Personalized Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Personalization] This paper proposes PersonaVLM, a multimodal agent framework for long-term personalization. By utilizing active memory management (four memory databases), multi-step reasoning retrieval, and a momentum-based personality evolution mechanism, it transforms general MLLMs into personalized assistants capable of adapting to evolving user preferences, outperforming GPT-4o by 5.2% in 128K context scenarios.
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Personalization"
  - "Long-term Memory"
  - "Multimodal Agent"
  - "Big Five Personality"
  - "Agent Framework"
date: 2026-05-08
content_hash: 6f486eef1a4b5c92
---

# PersonaVLM: Long-Term Personalized Multimodal LLMs

**Conference**: CVPR 2026  
**arXiv**: [2604.13074](https://arxiv.org/abs/2604.13074)  
**Code**: [Project Homepage](https://PersonaVLM.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Personalization, Long-term Memory, Multimodal Agent, Big Five Personality, Agent Framework

## TL;DR
This paper proposes PersonaVLM, a multimodal agent framework for long-term personalization. By utilizing active memory management (four memory databases), multi-step reasoning retrieval, and a momentum-based personality evolution mechanism, it transforms general MLLMs into personalized assistants capable of adapting to evolving user preferences, outperforming GPT-4o by 5.2% in 128K context scenarios.

## Background & Motivation

1. **Background**: Multimodal Large Language Models (MLLMs) are being used by millions as assistants, creative partners, and companions. User expectations are shifting from general problem-solving toward personalized, empathetic, long-term experiences. Existing personalization methods are categorized into three types: adaptation-based (fine-tuning like Yo'LLaVA, MyVLM), augmentation-based (retrieval like RAP), and alignment-based (preference methods like ALIGNXPERT, PAS).
2. **Limitations of Prior Work**: Adaptation methods require fine-tuning for every new concept and fail to capture evolving preferences; augmentation methods use predefined databases lacking active management and update mechanisms; alignment methods assume static user traits and cannot adapt to personality changes over time. All methods are designed for static interactions and struggle with preference drift (e.g., switching from Sprite to Coke) and personality evolution.
3. **Key Challenge**: User preferences and personalities are inherently diverse and dynamic, yet existing methods use fixed windows and "one-size-fits-all" paradigms on the model side, failing to track continuously evolving traits on the user side.
4. **Goal**: To design a unified framework that simultaneously achieves three core capabilities: memory (active extraction and management of multimodal memories), reasoning (retrieval-based multi-turn reasoning), and alignment (adjusting outputs according to evolving personality).
5. **Key Insight**: Drawing from memory classification in cognitive science (Core/Semantic/Episodic/Procedural memory) and the Big Five personality model in psychology to construct a structured personalized memory architecture.
6. **Core Idea**: "Knowing what the user knows" is provided through four memory databases, and "understanding who the user is" is provided through the PEM momentum update mechanism. These two collaborate to achieve true long-term personalization.

## Method

### Overall Architecture
PersonaVLM aims to solve the issue where general MLLMs treat every user as the same stranger, failing to remember past interactions or perceive changing preferences. Using Qwen2.5-VL-7B as the backbone, an external personalized memory architecture is attached—comprising a Big Five personality profile and four types of memory databases—allowing the model to operate around this external storage. Each interaction cycle consists of two phases: a Response Phase that retrieves relevant segments from memory and generates user-aligned answers after reasoning; and an Update Phase that reviews the interaction to write new facts into memory and slightly shift the personality vector toward the user's latest behavior. Thus, "what the user knows" is carried by memory banks, while "who the user is" is carried by the persona profile, both being refreshed continuously while model weights remain fixed after a two-stage SFT + GRPO training.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 26, 'padding': 6, 'wrappingWidth': 420, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["User Query: Text + Optional Image + Timestamp"] --> T

    subgraph RESP["Response Phase: Multi-step Agent Retrieval"]
        direction TB
        T["think: determine retrieval necessity"] -->|insufficient info| RET["retrieve: by time period + keywords<br/>parallel top-k from semantic/episodic/procedural memory"]
        RET -->|max 3 rounds| T
        T|sufficient info| ANS["answer: generate personalized response"]
    end

    subgraph MEM["Personalized Memory Architecture"]
        direction TB
        P["Persona Profile P: Big Five 5D Vector"]
        DB["Four Memory Banks M: Core / Semantic / Episodic / Procedural"]
    end

    subgraph UPD["Update Phase: Personality Evolution & Memory Refreshing"]
        direction TB
        PEM["PEM: Cosine-decay EMA for long-term personality"]
        MU["Refresh four memory types<br/>semantic per turn · core/procedural per session · episodic by topic"]
    end

    MEM -.Read.-> RET
    ANS --> UPD
    UPD -.Write back.-> MEM
    TRAIN["Two-stage Training (SFT + GRPO): SFT cold start → GRPO strengthens multi-step reasoning"] -.Offline backbone training.-> T
```

### Key Designs

**1. Personalized Memory Architecture: Decomposition of User Profile into CRUD-enabled Categories**

To address the limitations where old methods use fixed windows or static databases, PersonaVLM splits the profile into a Persona Profile $\mathcal{P}$ (a quantitative vector of the Big Five: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, each 1–5) and a multi-type memory bank $\mathcal{M}$. Memories are classified into four types: Core Memory (basic attributes like name/occupation, keeping only the latest version), Semantic Memory (abstract knowledge independent of events, like entities and relations), Episodic Memory (timestamped atomic events including summaries and keywords), and Procedural Memory (plans/goals/habits). All support CRUD operations, with episodic and semantic memories being appended chronologically, while core and procedural memories maintain only the latest version. This covers the full spectrum from "who the user is" to "what the user has done" and "what the user is used to."

**2. Response Phase: Multi-step Agent Retrieval—Autonomous Decision on Whether, What, and When to Retrieve**

User queries often rely on context and contain anaphoras ("that thing we talked about last time"). A single semantic retrieval often misses the target. The Response Phase treats "fetching memory" as a multi-turn agent interaction: the model receives the instruction, context, and a consolidated profile (Core Memory + Persona), then outputs reasoning and an `action`. If more info is needed, it generates retrieval conditions—`time period` and `keywords`. The agent filters memory by time, then performs parallel top-$k$ retrieval across semantic, episodic, and procedural banks to backfill the model. This iterates until a final response $\mathcal{R}_m$ is generated (max 3 retrievals per trajectory).

**3. Update Phase: Personality Evolution Mechanism (PEM) and Memory Refreshing**

After response generation during user idle time, the update phase analyzes the interaction $U(\mathcal{Q}_m, \mathcal{R}_m, \mathcal{M}_{m-1})$. For personality, PEM maintains a long-term vector $\mathbf{p} \in \mathbb{R}^5$. It infers an instantaneous personality $\mathbf{p}'_m$ from the current query and integrates it into the long-term vector via Exponential Moving Average:

$$\mathbf{p}_m \leftarrow \lambda \cdot \mathbf{p}_{m-1} + (1-\lambda) \cdot \mathbf{p}'_m$$

The weight $\lambda$ follows a cosine decay schedule: $\lambda$ is small in early stages for rapid adaptation, and increases as interactions accumulate to stabilize the profile against single-turn fluctuations. Simultaneously, memories are refreshed: semantic memory extracts preferences per turn, core/procedural memories are updated via agent analysis at session ends, and episodic memory segments the conversation by topic.

**4. Loss & Training (SFT + GRPO): Learning Memory Management and Strategic Retrieval**

To enable the model to decide "when to retrieve" rather than just "how to format," training follows two steps. The SFT phase uses 78K synthetic samples to instill memory mechanisms (persona inference + CRUD) and QA with full multi-step reasoning trajectories. The RL phase uses GRPO to reinforce multi-turn reasoning, enforcing an output structure of `<think>` → `<retrieve>` / `<answer>`. The reward function:

$$r_i = f_{\text{acc}} \cdot f_{\text{cons}} + 0.5 \cdot f_{\text{format}}$$

computes the product of accuracy $f_{\text{acc}}$ and reasoning-answer consistency $f_{\text{cons}}$, plus a half-weighted format compliance term $f_{\text{format}}$. Scores are provided by Qwen3-30B-A3B as a zero-shot LLM judge. Training data was generated via PersonaHub, sampling 500 profiles for 30K+ long-term multimodal interactions.

### Example: Preference Drift from Sprite to Coke

A user previously liked Sprite (recorded in Core Memory and Episodic events). If the user sends an image of Coke saying "I've started drinking this lately," the **Response Phase** sees the model `<think>` and `<retrieve>` historical context. Detecting a conflict, it generates an `<answer>` acknowledging the shift in taste. During the **Update Phase**, the agent performs an `update` on Core Memory (Sprite $\rightarrow$ Coke) and `appends` to Episodic Memory. PEM integrates the turn's data; as long-term interactions are already high, $\lambda$ is large, ensuring the personality remains stable despite the single behavioral change.

## Key Experimental Results

### Main Results

**Persona-MME Benchmark (128K Context)**:

| Model | Overall | Memory | Intent | Preference | Behavior | Growth |
|------|---------|--------|--------|------------|----------|--------|
| GPT-4o | 72.35% | 86.99 | 83.87 | 63.12 | 57.14 | 73.87 |
| Qwen2.5-VL-7B (Baseline) | 64.84% | 66.13 | 66.85 | 59.75 | 59.24 | 70.69 |
| **PersonaVLM (Ours)** | **77.5%** | — | — | — | — | — |

**Comparison with GPT-4o**:

| Benchmark | PersonaVLM | GPT-4o | Gain |
|------|-----------|--------|------|
| Persona-MME (128K) | 77.5% | 72.35% | +5.2% |
| PERSONAMEM (128K) | ~49% | 39.20% | +9.8% |

### Ablation Study

| Configuration | Persona-MME | Description |
|------|------------|------|
| PersonaVLM (SFT+RL) | 77.5% | Full method |
| SFT Only | ~72% | RL gain approx. 5% |
| Without PEM | ~73% | Evolution mechanism contribution approx. 4% |
| Full context (No RAG) | Lower | Low info utilization in long contexts |
| RAG mode | Higher | Structured retrieval outperforms raw long context |

### Key Findings
- **7B Model outperforms GPT-4o**: PersonaVLM exceeds GPT-4o by 5.2% on Persona-MME and 9.8% on PERSONAMEM, demonstrating the value of specialized personalized training.
- **Greater advantage in 128K context**: The structured memory architecture becomes more significant as long-term interaction memory accumulates.
- **RL is vital for reasoning strategy**: GRPO training enables the model to learn when to retrieve and how to select reasoning paths.

## Highlights & Insights
- **Cognitive Science Inspiration**: The mapping to four memory types (Core/Semantic/Episodic/Procedural) is well-justified and provides complementary functionality.
- **PEM Cosine Decay**: Effectively balances initial rapid learning with long-term stability without manual learning rate tuning.
- **Data Synthesis Pipeline**: Sampling 500 profiles for 30K+ interactions addresses the core issue of scarce personalized training data.

## Limitations & Future Work
- Personality modeling is based on the Big Five, which may not capture all cultural nuances.
- Distribution shift may exist between synthetic training data and real user interactions.
- Validated only on Qwen2.5-VL-7B; larger scales remain untested.
- Memory CRUD operations may introduce errors (e.g., false deletions) without a robust error-correction mechanism.
- Future work: privacy-preserving personalization (federated learning) and multi-user shared memories.

## Related Work & Insights
- **vs Yo'LLaVA/MyVLM**: These learn user concepts via fine-tuning embeddings but cannot update memory dynamically. PersonaVLM's agent architecture supports active CRUD.
- **vs MemGPT**: MemGPT provides OS-like memory management but is text-only and depends on commercial models. PersonaVLM is self-contained, multimodal, and persona-driven.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multimodal agent framework for long-term dynamic personalization; original PEM design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ New Persona-MME benchmark, 10+ models compared, extensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Comprehensive framework, though many components require careful reading.
- Value: ⭐⭐⭐⭐⭐ Opens new directions for long-term dynamic interaction in MLLM personalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Unified Personalized Understanding, Generating and Editing](unified_personalized_understanding_generating_and_editing.md)
- [\[CVPR 2026\] Personalized Image Descriptions from Attention Sequences](personalized_image_descriptions_from_attention_sequences.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)

</div>

<!-- RELATED:END -->
