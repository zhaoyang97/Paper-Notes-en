---
title: >-
  [Paper Note] PersonaVLM: Long-Term Personalized Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Personalization] This paper proposes PersonaVLM, a multimodal agent framework for long-term personalization. Through proactive memory management (four-type memory database), multi-step reasoning-based retrieval, and a momentum-based personality evolution mechanism, it transforms a general-purpose MLLM into a personalized assistant capable of adapting to shifting user preferences, surpassing GPT-4o by 5.2% under a 128K context.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Personalization
  - Long-term Memory
  - Multimodal Assistant
  - Big Five Personality
  - Agent Framework
date: 2026-05-08
content_hash: df5b86d692c13725
---

# PersonaVLM: Long-Term Personalized Multimodal LLMs

**Conference**: CVPR 2026
**arXiv**: [2604.13074](https://arxiv.org/abs/2604.13074)
**Code**: [Project Page](https://PersonaVLM.github.io)
**Area**: Multimodal VLM
**Keywords**: Personalization, Long-term Memory, Multimodal Assistant, Big Five Personality, Agent Framework

## TL;DR
This paper proposes PersonaVLM, a multimodal agent framework for long-term personalization. Through proactive memory management (four-type memory database), multi-step reasoning-based retrieval, and a momentum-based personality evolution mechanism, it transforms a general-purpose MLLM into a personalized assistant capable of adapting to shifting user preferences, surpassing GPT-4o by 5.2% under a 128K context.

## Background & Motivation

1. **State of the Field**: Multimodal large language models are being used by millions as assistants, creative partners, and companions. User expectations are shifting from general-purpose problem-solving toward personalized, empathetic, long-term experiences. Existing personalization methods fall into three categories: adaptation-based (fine-tuning methods such as Yo'LLaVA and MyVLM), augmentation-based (retrieval methods such as RAP), and alignment-based (preference methods such as ALIGNXPERT and PAS).
2. **Limitations of Prior Work**: Adaptation-based methods require fine-tuning for each new concept and cannot capture evolving preferences; augmentation-based methods rely on predefined databases and lack proactive management and update mechanisms; alignment-based methods assume static user characteristics and cannot adapt to personality changes over time. All existing methods are designed for static interactions and fail to handle preference drift (e.g., a user switching from preferring Sprite to Coca-Cola) or personality evolution.
3. **Root Cause**: User preferences and personalities are inherently diverse and dynamic, yet existing methods apply fixed context windows and one-size-fits-all paradigms on the model side, while failing to track continuously evolving user characteristics on the user side.
4. **Paper Goals**: To design a unified framework that simultaneously achieves three core capabilities — memory (proactively extracting and managing multimodal memories), reasoning (multi-turn reasoning based on retrieval), and alignment (adapting outputs according to evolving personality).
5. **Starting Point**: Drawing on cognitive science's taxonomy of memory (core/semantic/episodic/procedural memory) and the psychological Big Five personality model to construct a structured personalized memory architecture.
6. **Core Idea**: The four-type memory database provides "what is known about the user," while the PEM momentum update mechanism provides "what kind of person the user is." The two components work in concert to achieve genuine long-term personalization.

## Method

### Overall Architecture
PersonaVLM is built on Qwen2.5-VL-7B as the backbone and comprises a personalized memory architecture (personality profile + four-type memory database) and two collaborative phases: a response phase (input → retrieval → reasoning → personalized response generation) and an update phase (interaction analysis → memory and personality update). Training follows a two-stage pipeline: SFT (78K samples) followed by GRPO reinforcement learning.

### Key Designs

1. **Personalized Memory Architecture**:
    - Function: Construct and maintain a comprehensive long-term user profile.
    - Mechanism: Consists of two main components: (1) a user personality profile $\mathcal{P}$ — a quantitative vector over the Big Five personality dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, each scored 1–5); (2) a multi-type memory database $\mathcal{M}$ — core memory (basic attributes, retaining only the latest version), semantic memory (event-independent abstract knowledge, including entities, relations, and multimodal concepts), episodic memory (timestamped atomic events, including summaries, dialogue turns, and keywords), and procedural memory (plans, goals, and habitual behaviors). CRUD operations are supported; episodic and semantic memories are stored along a timeline, while core and procedural memories retain only the latest version.
    - Design Motivation: Existing memory architectures either rely on commercial models, handle only text, or lack user-centric design. The four-type memory taxonomy covers a complete user portrait — from "who the user is" to "what the user has done" to "what the user is accustomed to."

2. **Personality Evolution Mechanism (PEM)**:
    - Function: Dynamically track and update the user's personality traits.
    - Mechanism: Maintains a long-term personality vector $\mathbf{p} \in \mathbb{R}^5$. At each turn, the current personality vector $\mathbf{p}'_m$ is inferred and updated via exponential moving average (EMA): $\mathbf{p}_m \leftarrow \lambda \cdot \mathbf{p}_{m-1} + (1-\lambda) \cdot \mathbf{p}'_m$. A key innovation is the use of a cosine decay schedule for $\lambda$ — a low $\lambda$ in early interactions enables rapid adaptation, while a high $\lambda$ in later stages maintains stability. The updated numerical vector is converted into a textual description for generation.
    - Design Motivation: A static personality assumption cannot handle scenarios such as "a user who initially appears extroverted but later exhibits introverted traits." The cosine decay in EMA strikes a balance between rapid learning and long-term stability.

3. **Two-Stage Training (SFT + GRPO)**:
    - Function: Train a general-purpose MLLM to acquire personalization capabilities.
    - Mechanism: The SFT stage uses 78K synthetic samples to train foundational abilities in memory management and multi-turn reasoning. The RL stage applies GRPO to further enhance reasoning — outputs must follow the `<think>` → `<retrieve>/<answer>` structure, and the reward function $r_i = f_{\text{acc}} \cdot f_{\text{cons}} + 0.5 \cdot f_{\text{format}}$ jointly measures accuracy, reasoning consistency, and format compliance. Training data is generated via PersonaHub, synthesizing 500 diverse user profiles and simulating long-term multimodal interactions (30K+ interactions).
    - Design Motivation: SFT alone cannot teach strategic retrieval decision-making (when to retrieve, what to retrieve, and from which time period). The exploratory nature of RL training supplements this capability.

### Loss & Training
SFT uses standard cross-entropy loss. GRPO applies group-normalized advantage functions; accuracy and consistency scores are computed by Qwen3-30B-A3B as an LLM judge. Retrieval attempts are limited to a maximum of 3 per trajectory.

## Key Experimental Results

### Main Results

**Persona-MME Benchmark (128K context)**:

| Model | Overall | Memory | Intent | Preference | Behavior | Growth |
|------|---------|--------|--------|------------|----------|--------|
| GPT-4o | 72.35% | 86.99 | 83.87 | 63.12 | 57.14 | 73.87 |
| Qwen2.5-VL-7B (Baseline) | 64.84% | 66.13 | 66.85 | 59.75 | 59.24 | 70.69 |
| **PersonaVLM** | **77.5%** | — | — | — | — | — |

**Comparison with GPT-4o**:

| Benchmark | PersonaVLM | GPT-4o | Gain |
|------|-----------|--------|------|
| Persona-MME (128K) | 77.5% | 72.35% | +5.2% |
| PERSONAMEM (128K) | ~49% | 39.20% | +9.8% |

### Ablation Study

| Configuration | Persona-MME | Notes |
|------|------------|------|
| PersonaVLM (SFT+RL) | 77.5% | Full method |
| SFT only | ~72% | RL contributes ~5% |
| w/o PEM | ~73% | PEM contributes ~4% |
| Full context (no RAG) | Lower | Low information utilization efficiency under long context |
| RAG mode | Higher | Structured retrieval outperforms direct long-context |

### Key Findings
- **A 7B model surpasses GPT-4o**: PersonaVLM outperforms GPT-4o by 5.2% on Persona-MME and 9.8% on PERSONAMEM, demonstrating the value of specialized training for personalization.
- **Greater advantage under 128K context**: Long-term interactions accumulate more memories, making the advantages of the structured memory architecture more pronounced.
- **RL is critical for reasoning strategy**: GRPO training enables the model to learn when to retrieve and how to select reasoning paths.

## Highlights & Insights
- The **cognitive science inspiration behind the memory architecture** is highly compelling: the four memory types (core/semantic/episodic/procedural) map directly to human memory taxonomies, resulting in a design that is both principled and functionally complementary.
- The **cosine decay design in PEM** elegantly resolves the tension between "rapid early learning" and "long-term stability," naturally adapting to the interaction lifecycle without requiring manual learning rate tuning.
- The **data synthesis pipeline** is an underappreciated contribution: the synthetic dataset of 500 user profiles and 30K+ multimodal interactions directly addresses the core challenge of scarce personalization training data.

## Limitations & Future Work
- Personality modeling is grounded in the Big Five model, which may not capture all cultural and individual differences.
- Synthetic training data may exhibit distributional gaps relative to real user interactions.
- Validation is limited to Qwen2.5-VL-7B; larger-scale models remain untested.
- CRUD operations on memory may introduce errors (e.g., incorrectly deleting important memories), and no error-correction mechanism is in place.
- Future work could explore privacy-preserving personalization (federated learning) and multi-user shared memory.

## Related Work & Insights
- **vs. Yo'LLaVA/MyVLM**: These methods learn user-specific visual concepts by fine-tuning embeddings but cannot manage or update memories. PersonaVLM's agent architecture supports dynamic CRUD operations.
- **vs. MemGPT**: MemGPT provides OS-like memory management but is text-only and depends on commercial models. PersonaVLM is self-contained, supports multimodality, and has an explicit personalization objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multimodal agent framework targeting long-term dynamic personalization; PEM design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Introduces the Persona-MME benchmark, compares 10+ models, and conducts multi-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Framework description is comprehensive, though the large number of components requires careful reading.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for MLLM personalization through long-term dynamic interaction.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)

<!-- RELATED:END -->
