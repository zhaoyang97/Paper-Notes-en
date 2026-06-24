---
title: >-
  [Paper Note] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?
description: >-
  [ACL 2025][LLM (Other)][Scene Graph] Proposes TSG Bench to systematically evaluate the capability of 11 LLMs on scene graph understanding and generation tasks, revealing significant bottlenecks of LLMs in scene graph generation (especially in multi-action decomposition).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Scene Graph"
  - "Benchmarking"
  - "Large Language Models"
  - "Structured Representation"
  - "Multimodal Reasoning"
date: 2026-05-08
content_hash: 9bb9ef9d1881e22b
---

# LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?

**Conference**: ACL 2025  
**arXiv**: [2505.19510](https://arxiv.org/abs/2505.19510)  
**Code**: [GitHub](https://github.com/docworlds/tsg-bench)  
**Area**: LLM/NLP  
**Keywords**: Scene Graph, Benchmarking, Large Language Models, Structured Representation, Multimodal Reasoning

## TL;DR

Proposes TSG Bench to systematically evaluate the capability of 11 LLMs on scene graph understanding and generation tasks, revealing significant bottlenecks of LLMs in scene graph generation (especially in multi-action decomposition).

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Large language models perform exceptionally well on textual tasks, but extending their capabilities to multimodal environments that require spatial and temporal reasoning poses a challenge. A Scene Graph is a structured representation that encodes entities, attributes, and relations in a scene, widely applied in areas such as Embodied AI, robotics, and 3D environment modeling.

However, there is a lack of systematic evaluation of the LLMs' capacity to leverage scene graphs. Existing works primarily focus on image-scene graph pairs, while research on text-to-scene graphs is limited. Existing benchmarks, such as FACTUAL, only address static scenes and are unsuitable for dynamic, real-world scenes. Key questions include: Do LLMs truly understand the spatial and semantic structure of scene graphs? Do they experience misunderstandings when dealing with long contexts and complex triplets?

This paper introduces TSG Bench, aiming to systematically evaluate the capabilities of LLMs in two dimensions: scene graph **understanding** and **generation**, filling the evaluation gap in this field.

## Method

### Overall Architecture

TSG Bench is constructed based on narrative texts and corresponding dynamic scene graph sequences, comprising 120 real-world scenes, 2,041 descriptions, and 4,289 scene graphs. The benchmark designs four evaluation tasks: two understanding tasks (SGQA, SGDS) and two generation tasks (SA-SGG, MA-SGG).

Regarding data representation, a narrative $D = (d_1, \dots, d_n)$ consists of multiple coherent natural language descriptions. Each description $d_i$ corresponds to a set of scene graphs $G_i = (G_{i1}, \dots, G_{ik})$, where $k \in [1, 8]$ depends on the complexity of the description. Each scene graph $G_{ij} = (V_{ij}, E_{ij})$ contains nodes (of four types: person, action, object, hand) and edges (of three types: verb, dobj, preposition).

### Key Designs

**1. Scene Graph Understanding Tasks: SGQA + SGDS**

- **Function**: Evaluates the LLM's capability to reason about and interpret scene graphs
- **Mechanism**: SGQA requires the model to perform multi-hop reasoning based on a sequence of scene graphs to answer questions; SGDS is a multiple-choice task, requiring the model to select the correct description given the context and scene graphs
- **Design Motivation**: Understanding tasks require logical/temporal connections across multiple triplets, simulating contextual reasoning requirements in practical applications

**2. Scene Graph Generation Tasks: SA-SGG + MA-SGG**

- **Function**: Evaluates the LLM's capability to generate structured scene graphs from text
- **Mechanism**: SA-SGG generates scene graphs for single-action descriptions; MA-SGG targets complex descriptions containing multiple actions, requiring the model to first decompose the description into multiple discrete actions and then generate scene graphs for each individually
- **Design Motivation**: Generation tasks require models to parse semantically similar elements from natural language and construct triplets, while MA-SGG additionally tests action decomposition and sequencing capabilities

**3. Data Construction: Human-in-the-loop Multi-round Process**

- **Function**: Ensures the quality and diversity of benchmark data
- **Mechanism**: Based on the EASG dataset, narratives and scene graphs are constructed through a multi-round process of LLM generation combined with human verification, including steps such as sentence generation, graph generation, human verification, synonymous rewriting, and sentence merging
- **Design Motivation**: Scene graphs generated purely by LLMs are often incomplete, requiring element-by-element examination and refinement by humans

### Loss & Training

As this work introduces an evaluation benchmark, it does not involve model training. The evaluation is conducted under a zero-shot setting, utilizing Exact Match (SGQA), Accuracy (SGDS), and Precision/Recall/F1 (SGG tasks) as metrics.

## Key Experimental Results

### Main Results

| Model | SGDS Acc | SGQA EM | SA-SGG F1 | MA-SGG F1 |
|------|----------|---------|-----------|-----------|
| Human | 98.33 | 88.00 | 82.50 | 75.60 |
| Claude-3.5-Sonnet | 98.40 | 90.60 | 68.43 | 58.80 |
| GPT-4o | 96.40 | 84.80 | 59.23 | 43.99 |
| LLaMA-3.3-70B | 97.60 | 84.60 | 33.37 | 28.92 |
| Qwen-2.5-72B | 96.80 | 81.40 | 54.42 | 36.78 |
| DeepSeek-V3 | 96.40 | 79.60 | 54.45 | 39.34 |
| Qwen-2.5-7B | 93.60 | 73.40 | 9.39 | 6.34 |

### Ablation Study

| Method | SGQA | SA-SGG F1 | MA-SGG F1 |
|------|------|-----------|-----------|
| Claude-3.5-Sonnet (zero-shot) | 90.60 | 68.43 | 58.80 |
| + CoT | 94.00 | 69.57 | 64.36 |
| + 10-shot | 92.00 | 75.29 | 71.75 |
| GPT-4o (zero-shot) | 84.80 | 59.23 | 43.99 |
| + CoT | 90.00 | 67.13 | 44.79 |
| + 10-shot | 84.40 | 65.78 | 57.40 |

### Key Findings

1. **Performance Gap between Understanding vs Generation**: LLMs perform exceptionally in understanding tasks (SGDS $\ge 90\%$), but exhibit a huge gap compared to humans in generation tasks (with the highest MA-SGG being only 58.80 vs 75.60 for humans).
2. **Action Decomposition is the Core Bottleneck**: In multi-action scene graph generation, models struggle to correctly decompose implicit and repetitive actions.
3. **10-shot ICL is Most Effective for Generation Tasks**: On MA-SGG, Claude utilizing 10-shot learning can improve F1 from 58.80 to 71.75.
4. **Small Models Almost Fail on Generation Tasks**: Qwen-2.5-7B and Mistral-7B achieve F1 $< 12$ on MA-SGG.
5. **LLMs Suffer from Scene Graph Hallucinations**: Smaller models generate a large number of elements not in the predefined set $L$.

## Highlights & Insights

- The first benchmark to systematically evaluate the text-to-scene-graph understanding and generation capabilities of LLMs, filling an important gap.
- Precisely pinpoints performance bottlenecks through decomposition experiments (node generation, edge generation, and action decomposition).
- Discovers that LLMs' capacity to handle implicit and repetitive actions is severely deficient, revealing weaknesses in numerical awareness.
- Scene graph error correction experiments reveal that providing the error type significantly improves the model's correction capability (e.g., Claude improves from 60.03 to 88.28).

## Limitations & Future Work

- The current benchmark only involves a single actor and does not contain object attributes (such as color, size, etc.).
- The complexity of action decomposition is limited (at most 8 sub-scene-graphs), whereas real-world scenarios can be more complex.
- Only the textual modality is evaluated. Future work can combine VLMs to extract narratives from images/videos and generate scene graphs.
- Evaluation metrics are based on exact matching of triplets, potentially neglecting correct outputs that are semantically similar but differently expressed.

## Related Work & Insights

- **Scene Graph Representation**: Widely applied in tasks such as 3D reconstruction, interactive games, and robotic navigation, the integration of LLMs and scene graphs is emerging as a new direction.
- **Insight**: LLMs' capabilities in structured output generation are far inferior to their understanding capabilities, which serves as an important caveat for Embodied AI systems that rely on LLMs to understand environments.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first text-to-scene-graph LLM evaluation benchmark, with a clear problem definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 LLMs + multiple prompting strategies + fine-grained analysis + error correction/hallucination experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, standardized task definitions, and rich charts.
- **Value**: ⭐⭐⭐⭐ Provides crucial evaluation tools and findings for LLM application research related to scene graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](aad-llm_neural_attention-driven_auditory_scene_understanding.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)

</div>

<!-- RELATED:END -->
