---
title: >-
  [Paper Note] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering
description: >-
  [NeurIPS 2025][Reinforcement Learning][KB-VQA] This paper proposes Wiki-PRF, a three-stage (Processing–Retrieval–Filtering) multimodal RAG framework that trains a VLM via reinforcement learning to autonomously invoke visual tools and filter retrieved results, achieving state-of-the-art performance on E-VQA and InfoSeek.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - KB-VQA
  - RAG
  - Multimodal Retrieval
  - Tool Calling
date: 2026-05-08
content_hash: e81be61340c0e7bb
---

# Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering

**Conference**: NeurIPS 2025
**arXiv**: [2510.14605](https://arxiv.org/abs/2510.14605)
**Code**: [GitHub](https://github.com/cqu-student/Wiki-PRF)
**Area**: Reinforcement Learning / Visual Question Answering / Retrieval-Augmented Generation
**Keywords**: KB-VQA, RAG, Multimodal Retrieval, Reinforcement Learning, Tool Calling

## TL;DR

This paper proposes Wiki-PRF, a three-stage (Processing–Retrieval–Filtering) multimodal RAG framework that trains a VLM via reinforcement learning to autonomously invoke visual tools and filter retrieved results, achieving state-of-the-art performance on E-VQA and InfoSeek.

## Background & Motivation

**Background**: Knowledge-intensive visual question answering (KB-VQA) requires models to simultaneously understand image content and retrieve external knowledge. Retrieval-augmented generation (RAG) approaches have demonstrated significant progress on this task.

**Limitations of Prior Work**:
- **Coarse retrieval granularity**: Existing methods typically use whole-image visual features for retrieval. In complex scenes (e.g., a small statue beside a clock tower), global features are dominated by the primary object, leading to retrieval of substantial irrelevant information.
- **Low filtering precision**: Passage-level re-ranking fails to remove irrelevant content within individual passages, leaving substantial noise in retrieved results.

**Key Challenge**: Fine-grained information extraction is needed for precise retrieval, while simultaneously filtering noise from large volumes of retrieved results.

**Goal**: Design more precise multimodal retrieval and more effective result filtering mechanisms.

**Key Insight**: Enable the VLM to autonomously decide which visual tools (captioning, grounding, flipping) to apply to the input, and train it via RL to filter irrelevant information from retrieved results.

**Core Idea**: Train a VLM-PRF model using answer accuracy and format consistency as reward signals. Through GRPO, the model learns to flexibly invoke tools and efficiently filter information—marking the first application of RL to multimodal RAG.

## Method

### Overall Architecture

Wiki-PRF consists of three core stages:

1. **Processing Stage**: VLM-PRF autonomously selects and invokes tools (captioning/grounding/flipping) based on the image and question, generating high-quality multimodal retrieval queries.
2. **Retrieval Stage**: Multimodal knowledge base retrieval based on visual features and textual descriptions.
3. **Filtering Stage**: VLM-PRF performs relevance filtering and information condensation on retrieved results, producing task-oriented, compact knowledge.

### Key Designs

1. **Tool Calling Mechanism**:

    - **Captioning Tool**: VLM-PRF generates an initial description $C_{init}$, which is then refined by VLM-base into a retrieval query $C_{query} = \text{VLM}_{\text{captioning}}(C_{init}, Q)$.
    - **Grounding Tool**: VLM-PRF specifies the target object; VLM-base localizes and crops the image region $I_{\text{grounding}} = \text{Crop}(I, \text{VLM}_{\text{grounding}}(\text{object}))$.
    - **Flipping Tool**: Applies horizontal flipping to the image to mitigate the effect of viewpoint variation on retrieval.
    - VLM-PRF reasons about which tools to use and in what order within `<think>` tags, and outputs tool calls within `<tool>` tags.

2. **Multimodal Retrieval**: Queries are encoded using EVA-CLIP 8B, and cosine similarity retrieval is performed via Faiss-GPU. Retrieved results are split into passage-level sections and ranked by similarity to the question, with Top-$k_s$ sections selected.

3. **Filtering Stage**: VLM-PRF receives direct retrieval information $D$ and tool-retrieved results $\mathcal{S}_{\text{search}}$, reasons within `<think>` tags, and outputs filtered task-oriented knowledge $F$ within `<answer>` tags:
    $$F = \text{VLM-PRF}(D, \mathcal{S}_{\text{search}}), \quad A = \text{VLM}(F, Q)$$

4. **Reinforcement Learning Training**: VLM-PRF is trained using GRPO (with KL constraint removed). The reward function is:
    $$r_\phi(x,y) = \alpha \cdot EM(a_{\text{pred}}, a_{\text{gt}}) + \beta \cdot M(a_{\text{tool}}, t_{\text{tool}}) + \gamma \cdot M(a_{\text{filter}}, t_{\text{filter}})$$
    - Answer reward ($\alpha=1$): exact match score
    - Tool format reward ($\beta=0.3$): regex-based format compliance verification
    - Filter format reward ($\gamma=0.7$): verification of filtering output format
    - LoRA (rank=64, alpha=128) is used to train only a small number of additional parameters.

### Loss & Training

- Base model: Qwen2.5-VL-3B/7B
- Training configuration: 8 generation samples, temperature 0.7, 2 epochs, lr=1e-5
- Training duration: approximately 15 hours (8 × A800 GPUs)
- Retrieval setting: Top-1 image for direct retrieval + Top-5 articles for tool-augmented retrieval

## Key Experimental Results

### Main Results

| Method | Model | E-VQA Single-Hop | E-VQA All | InfoSeek UQ | InfoSeek UE | InfoSeek All |
|------|------|------------------|-----------|-------------|-------------|-------------|
| GPT-4V | - | 26.9 | 28.1 | 15.0 | 14.3 | 14.6 |
| EchoSight | Mistral-7B | 19.4 | - | - | - | 27.7 |
| ReflectiVA | LLaMA-3.1-8B | 28.0 | 29.2 | 40.4 | 39.8 | 40.1 |
| MMKB-RAG | Qwen2-7B | 39.7 | 35.9 | 36.4 | 36.3 | 36.4 |
| **Wiki-PRF-7B** | **Qwen2.5-VL-7B** | **37.1** | **36.0** | **43.3** | **42.7** | **42.8** |
| **Wiki-PRF (InternVL3)** | **InternVL3-8B** | **40.1** | **39.2** | **43.5** | **42.1** | **42.5** |

Wiki-PRF-7B achieves a new state of the art on InfoSeek (42.8) and on E-VQA (36.0).

### Retrieval Recall

| Model | Retrieval Input | Recall |
|------|---------|--------|
| None | images | 45.56 |
| Qwen2.5-VL-7B | images + tools | 53.44 |
| **VLM-PRF-7B** | **images + tools** | **54.89** |

Tool calling substantially improves retrieval recall (+9.3%), with RL training yielding further gains.

### Ablation Study

**Module Ablation** (InfoSeek 10K samples):

| Configuration | VQA Accuracy |
|------|-------------|
| Baseline (no RAG) | 34.22 |
| + Processing | 36.24 |
| + Processing + Filtering | 39.48 |

**RL vs. SFT Comparison** (InfoSeek 2K samples):

| Training Method | UQ | UE | All |
|---------|-----|-----|-----|
| Base (no fine-tuning) | 39.1 | 40.5 | 40.2 |
| SFT | 41.5 | 41.9 | 41.8 |
| **RL** | **46.6** | **46.2** | **46.3** |

RL substantially outperforms SFT (+4.5%), as SFT tends to imitate surface-level patterns whereas RL enables the model to internalize the underlying principles of information filtering.

### Oracle Setting (Given Ground-Truth Articles)

| Method | VQA Accuracy |
|------|-------------|
| Wiki-LLaVA | 51.5 |
| ReflectiVA | 57.6 |
| **Wiki-PRF-7B** | **65.8** |

Even when given the correct articles, Wiki-PRF's filtering capability allows it to locate key information far more effectively than competing methods.

### Effect of Knowledge Base Scale

| Method | 10K | 50K | 100K |
|------|-----|-----|------|
| Vanilla-MRAG (7B) | 56.3 | 39.6 | 23.7 |
| **Wiki-PRF-7B** | **60.3** | **51.2** | **42.8** |

Wiki-PRF exhibits significantly slower performance degradation as knowledge base scale increases.

### Key Findings

- After RL training, the number of tool-calling combinations used by the model increases from 34 to 40–53, indicating that RL enhances flexibility and diversity in tool usage.
- The captioning tool is invoked most frequently and is the most effective tool for improving article recall.
- A new state of the art of 77.8 is achieved on OK-VQA, confirming the cross-dataset generalizability of the approach.
- The filtering stage contributes approximately as much as the processing stage (each approximately +2–3%).

## Highlights & Insights

- **First application of RL to multimodal RAG**: Using answer accuracy as the reward signal alone is sufficient for the model to learn both tool selection and information filtering strategies, without requiring annotations for intermediate steps.
- **Dual-role design of VLM-PRF**: The same RL-trained model handles both processing (tool planning) and filtering (information refinement), demonstrating the efficiency of a "few parameters, multiple functions" design.
- **Flexible tool invocation**: VLM-PRF autonomously reasons about which tools to use and in what order, offering greater flexibility than hard-coded pipelines.
- **Clear empirical evidence for RL > SFT**: RL significantly outperforms SFT in the absence of intermediate-step annotations, validating the unique value of RL for RAG tasks.

## Limitations & Future Work

- The current tool set is limited (captioning, grounding, and flipping only); incorporating additional tools (e.g., OCR, chart parsing) may yield further improvements.
- The retrieval model (EVA-CLIP 8B) is frozen; joint training of the retriever and VLM-PRF may provide additional gains.
- Performance still degrades noticeably as knowledge base scale increases (approximately 43% at 100K), indicating that retrieval quality remains a bottleneck.
- RL training requires approximately 15 hours on 8 × A800 GPUs, which remains costly for resource-constrained settings.
- The flipping tool is not sufficiently general-purpose (addressing only horizontal flips) and may offer limited benefit.
- Comparisons with recent general-purpose RAG methods (e.g., Self-RAG) are absent.

## Related Work & Insights

- EchoSight performs article-level retrieval using visual information followed by textual re-ranking—Wiki-PRF further introduces tool-based processing and passage-level filtering.
- ReflectiVA introduces a reflection mechanism—Wiki-PRF replaces manually designed reflection pipelines with RL training.
- The direction of training VLMs with RL (R1-OneVision, VisualThinker-R1-Zero) is gaining momentum—Wiki-PRF extends this paradigm to the RAG setting.
- The concept of tool-augmented LLMs (e.g., Toolformer) is successfully transferred to multimodal retrieval.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-stage PRF framework and the combination of RL training with RAG are novel; the tool-calling mechanism is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-dataset validation, detailed ablations, and a convincing RL vs. SFT comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and detailed method description; equation numbering and notation could be more consistent.
- **Value**: ⭐⭐⭐⭐ Addresses core pain points in KB-VQA retrieval and filtering; state-of-the-art results are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [\[NeurIPS 2025\] Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go](mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)

</div>

<!-- RELATED:END -->
