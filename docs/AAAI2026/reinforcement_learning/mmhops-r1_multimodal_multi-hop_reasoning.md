---
title: >-
  [Paper Note] MMhops-R1: Multimodal Multi-hop Reasoning
description: >-
  [AAAI 2026][Reinforcement Learning][Multimodal multi-hop reasoning] This paper proposes the MMhops benchmark (31K samples, 3–4 reasoning hops) and the MMhops-R1 framework, which trains MLLMs via reinforcement learning to autonomously plan reasoning paths and dynamically invoke image/text retrievers for multimodal multi-hop reasoning. A 7B model surpasses 72B baselines and existing mRAG methods.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Multimodal multi-hop reasoning
  - retrieval-augmented generation
  - knowledge-based VQA
  - dynamic planning
date: 2026-05-08
content_hash: b6c9770c7aa6f1d6
---

# MMhops-R1: Multimodal Multi-hop Reasoning

**Conference**: AAAI 2026  
**arXiv**: [2512.13573](https://arxiv.org/abs/2512.13573)  
**Code**: [https://github.com/taoszhang/MMhops-R1](https://github.com/taoszhang/MMhops-R1)  
**Area**: Reinforcement Learning  
**Keywords**: Multimodal multi-hop reasoning, reinforcement learning, retrieval-augmented generation, knowledge-based VQA, dynamic planning  

## TL;DR
This paper proposes the MMhops benchmark (31K samples, 3–4 reasoning hops) and the MMhops-R1 framework, which trains MLLMs via reinforcement learning to autonomously plan reasoning paths and dynamically invoke image/text retrievers for multimodal multi-hop reasoning. A 7B model surpasses 72B baselines and existing mRAG methods.

## Background & Motivation
Existing MLLM reasoning capabilities are primarily focused on single-step reasoning (spatial, mathematical, etc.), yet complex real-world questions often require multi-step integration of multimodal information and external knowledge. Existing knowledge-based VQA datasets (OK-VQA, INFOSEEK, etc.) typically require only shallow reasoning of the form "one-step visual recognition + one-step knowledge retrieval." Even E-VQA, which extends to two hops, is limited to the text modality with fixed reasoning path lengths. These constraints hinder the evaluation and training of multimodal multi-hop reasoning. Moreover, existing multimodal RAG methods adopt static pipelines that cannot dynamically adjust reasoning depth according to question complexity.

## Core Problem
How can multimodal large language models be endowed with dynamic multi-hop reasoning capability—i.e., the ability to autonomously decide when to invoke image retrieval, when to invoke text retrieval, and when to generate an answer, rather than relying on a predefined fixed reasoning pipeline? This problem is significant because real-world visual question answering often requires a model to begin from an image and arrive at an answer only after multiple steps of external knowledge retrieval and cross-modal information integration.

## Method

### Overall Architecture
MMhops-R1 consists of two components: (1) construction of the MMhops dataset; and (2) an RL-based dynamic multimodal retrieval-augmented generation framework.

The input is a set of images paired with a natural language question. At each timestep, the model first performs thinking, then selects one of three actions: invoke the image retriever (image_search), invoke the text retriever (text_search), or generate a final answer (answer). Retrieval results are fed back to the model as observations, and the interaction loop continues until an answer is generated or the maximum number of turns ($T=4$) is reached.

### Key Designs
1. **MMhops Dataset Construction**: The dataset contains two reasoning types—Bridging (85%) and Comparison (15%). Bridging samples are constructed via an iterative expansion method: starting from single-hop VQA pairs, the answer serves as a bridging entity; new sub-questions are generated from its Wikipedia page, and these sub-questions are then merged into a multi-hop question. Comparison samples are constructed by identifying semantically similar entity pairs from Wikipedia, generating comparative questions based on quantifiable attributes, and replacing entity names with corresponding images. All samples require 3–4 reasoning steps and must rely on external knowledge.

2. **Dynamic Action Space and Multi-step Retrieval Interaction**: The model's action space is {think, image_search, text_search, answer}, where image_search retrieves relevant visual information via CLIP, and text_search retrieves the top-3 relevant passages via the E5 model. Each action is formatted using specific XML tags; malformed outputs receive a penalty signal. This design enables the model to learn autonomous reasoning path planning rather than following a manually designed pipeline.

3. **Composite Reward Function Design**: The total reward is $R = \alpha \cdot R_{\text{outcome}} + \beta \cdot R_{\text{format}} + \gamma \cdot R_{\text{action}}$. $R_{\text{outcome}}$ is a binary reward for answer correctness; $R_{\text{format}}$ rewards well-formed outputs; $R_{\text{action}}$ is a tool-use reward. The key design is that $R_{\text{action}}$ is gated by both outcome and format—tool use is rewarded only when the answer is correct and the format is valid, preventing the degenerate behavior of invoking tools arbitrarily without producing correct answers.

### Loss & Training
- Policy optimization is based on the DAPO objective, using clipping and a dynamic sampling strategy (each group of $G=8$ responses must contain at least one correct sample).
- Loss masking: tokens returned by external retrieval are excluded from gradient computation; only the model's own reasoning and action tokens are optimized.
- Backbone: Qwen2.5-VL-7B-Instruct, trained for 1 epoch with $\text{lr}=1\text{e-6}$.
- Knowledge base: 100K Wikipedia articles, each paired with one image.

## Key Experimental Results

| Dataset | Metric | MMhops-R1 (7B) | Qwen2.5-VL-72B | GPT-4o | Gemini-2.5-Pro |
|--------|------|------|----------|------|------|
| MMhops Bridging | Overall | **51.35** | 34.39 | 36.62 | 53.98 |
| MMhops Comparison | Overall | **22.01** | 7.59 | 8.76 | 29.39 |
| INFOSEEK | Overall | **33.2** | - | - | - |
| E-VQA 2-hop | Acc | **23.3** | - | - | 22.8 (PaLM) |

### Ablation Study
- Removing $R_{\text{action}}$ reduces Bridging Overall from 51.35 to 47.57, and Comparison from 22.01 to 20.62.
- Removing $R_{\text{format}}$ causes a sharp drop in Comparison to 14.42, indicating that format constraints are especially critical for cross-image reasoning.
- Removing both $R_{\text{action}}$ and $R_{\text{format}}$ degrades Bridging to 41.75 and Comparison to 13.03.
- Performance improves consistently as the number of interaction turns increases from 2 to 4 (Overall: 39.93 → 51.35); a 5th turn yields no significant gain but increases computational cost.
- Four reasoning steps are optimal for the MMhops dataset.

## Highlights & Insights
- **RL-driven dynamic multimodal RAG**: This is the first work to apply RL to multimodal multi-hop reasoning, enabling the model to autonomously learn when and how to retrieve rather than relying on prompt engineering.
- **Gated tool-use reward**: $R_{\text{action}}$ is gated by both $R_{\text{outcome}}$ and $R_{\text{format}}$, elegantly preventing the degenerate strategy of invoking tools arbitrarily without producing correct answers.
- **7B outperforms 72B**: Through RL training, the 7B model surpasses the 72B direct-reasoning baseline by 16.96% on Bridging, demonstrating that training strategies can compensate for model scale gaps in reasoning capability.
- **Generalizable dataset construction**: The iterative expansion method provides a systematic and scalable approach to extending single-hop datasets into multi-hop benchmarks.

## Limitations & Future Work
- The knowledge base is limited to 100K Wikipedia articles; expanding coverage could further improve performance.
- Overall accuracy on the Comparison task remains low (even the best-performing Gemini-2.5-Pro achieves only 29.39%), indicating that cross-image reasoning remains a significant challenge.
- Experiments are conducted only on Qwen2.5-VL-7B; larger models or alternative architectures have not been evaluated.
- The maximum number of interaction turns is fixed at 4; dynamic termination strategies have not been explored.
- Dataset construction relies on GPT-4o, which may introduce annotation bias.

## Related Work & Insights
- **vs. Search-R1/ReSearch**: These methods apply RL to text-only multi-hop RAG; MMhops-R1 extends this paradigm to the multimodal setting by introducing an image retrieval action.
- **vs. OmniSearch (Gemini)**: OmniSearch relies on manually designed prompts or SFT-based planning pipelines and lacks the capacity to autonomously learn retrieval strategies; the RL-trained MMhops-R1 (7B) surpasses GPT-4o + OmniSearch.
- **vs. Traditional mRAG (Wiki-LLaVA, EchoSight)**: These methods are designed for single-hop settings; their static pipelines cannot handle multi-hop reasoning chains.

The dynamic mRAG + RL paradigm is transferable to other tasks requiring multi-step external interaction, such as complex code generation and scientific reasoning. The gated reward design—where tool-use reward is conditioned on answer correctness—is a generalizable RL reward shaping technique.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to apply RL to multimodal multi-hop RAG; the dataset construction method is systematic and scalable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-baseline comparisons, ablation studies covering key design choices, and cross-dataset generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich figures and tables; dataset construction pipeline is described in detail.
- Value: ⭐⭐⭐⭐ Introduces an important benchmark and an effective method, advancing research on multimodal reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning from Synthetic Data Improves Multi-hop Reasoning](../../ICLR2026/reinforcement_learning/learning_from_synthetic_data_improves_multi-hop_reasoning.md)
- [\[AAAI 2026\] TextShield-R1: Reinforced Reasoning for Tampered Text Detection](textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](../../ICLR2026/reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)
- [\[ICCV 2025\] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](../../ICCV2025/reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)

</div>

<!-- RELATED:END -->
