---
title: >-
  [Paper Note] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
description: >-
  [ACL 2025][Multi-Agent][Multi-agent reasoning] GETReason is proposed as a hierarchical multi-agent framework that decomposes the context extraction of public event images into three sub-tasks: geospatial, temporal, and event. These tasks are collaboratively completed by specialized agents, achieving more accurate image context reasoning than existing methods.
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "Multi-agent reasoning"
  - "event understanding"
  - "vision-language models"
  - "spatio-temporal reasoning"
  - "image context extraction"
date: 2026-05-08
content_hash: ab88b3049bd7f086
---

# GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.21863](https://arxiv.org/abs/2505.21863)  
**Code**: [github](https://coral-lab-asu.github.io/getreason/)  
**Area**: Others  
**Keywords**: Multi-agent reasoning, event understanding, vision-language models, spatio-temporal reasoning, image context extraction

## TL;DR

GETReason is proposed as a hierarchical multi-agent framework that decomposes the context extraction of public event images into three sub-tasks: geospatial, temporal, and event. These tasks are collaboratively completed by specialized agents, achieving more accurate image context reasoning than existing methods.

## Background & Motivation

Public event images (such as presidential inaugurations, large-scale protests, international summits, etc.) are not only visual records but also rich carriers of contextual information. Understanding these images requires not only describing the visible content but also inferring the implicit geopolitical, temporal, and event-related information.

**Limitations of Prior Work**:

**Traditional Description Models**: Encoder-decoder architectures can only describe visible objects, people, and actions, failing to infer deeper meanings. Even advanced VLMs like BLIP-2 and InstructBLIP often only describe "what is seen" while ignoring "why it is important."

**Reasoning Methods**: For example, CogBench can only infer coarse event types (such as "ceremony") and lacks specific details.

**Retrieval-Augmented Generation (RAG)**: Introduces external knowledge but is prone to hallucinations and misinformation.

**Lack of Evaluation Standards**: Existing metrics cannot effectively measure reasoning capability, and metrics like F1 do not consider the closeness between predicted values and ground truth.

## Method

### Overall Architecture

GETReason consists of a three-layer architecture: Scene Graph Generation Layer → Prompt Generation Layer → Multi-Agentic Extraction Layer. Each layer contains VLM agents that generate output based on specific prompts, collaborating to produce comprehensive and contextually rich information.

### Key Designs

1. **Scene Graph Generation**:

    - **Scene Graph Agent**: Identifies entities in the image along with their attributes and relationships, constructing an open-ended structured representation (JSON format).
    - **Abstraction Agent**: Infers higher-level abstract concepts conveyed by the image on top of the initial scene graph (e.g., "women participating in Saudi Arabia's political process").

2. **Prompt Generation**:

    - **Prompt Agent**: Generates tailored prompts for each agent in the multi-agentic extraction layer, ensuring each agent operates within its area of expertise (e.g., guiding the geospatial analyzer to focus on signage and attire characteristics).

3. **Multi-Agentic Extraction**:

    - **Event Agent**: Infers the key events in the image by synthesizing the scene graph, abstract concepts, and world knowledge.
    - **Temporal Agent**: Extracts fine-grained temporal information (century, decade, year, month, day) by leveraging clues such as lighting, celestial bodies, and technology styles.
    - **Geospatial Agent**: Accurately locates the country, province/state, and city of the image, assessing signage, attire, architectural features, etc.
    - **Cross Extraction**: A two-stage iterative reasoning strategy—feeding the contextual clues of other agents back to each agent to reduce hallucinations through cross-validation.

### Loss & Training

This paper does not involve end-to-end training but is instead a reasoning framework based on prompt engineering and multi-agent collaboration. The core strategies include:
- Direct Extraction: Each agent processes independently.
- Cross Extraction: Information sharing and iterative refinement among agents.
- Partial Cross Extraction: Only feeds event information back to the temporal and geospatial agents.

## Key Experimental Results

### Main Results

Results using Gemini 1.5 Pro-002 on the TARA dataset (GREAT metric, %):

| Method | Geo | Temp | Event | Total |
|------|-----|------|-------|-------|
| COT Zero-shot | 51.1 | 37.7 | 66.5 | 53.3 |
| Good Guesser | 76.1 | 31.0 | 64.4 | 57.8 |
| GETReason | **69.4** | **38.1** | **70.3** | **60.4** |

On the WikiTiLo dataset (without event evaluation):

| Method | Geo | Temp | Total |
|------|-----|------|-------|
| Good Guesser | 40.2 | 29.9 | 35.0 |
| GETReason | **42.4** | **34.0** | **38.2** |

Cross-model comparison (TARA, Total): GETReason achieves the best results across all models, with Gemini (60.4) > GPT-4o mini (53.5) > QwenVL-7B (51.3).

### Ablation Study

| Configuration | Geo | Temp | Event | Total |
|------|-----|------|-------|-------|
| GETReason (Full) | 69.4 | 38.1 | 70.3 | 60.4 |
| Direct Extraction | 67.4 | 33.2 | 68.6 | 57.6 |
| Partial Cross Extraction | 68.2 | 35.9 | 70.3 | 59.3 |
| w/o Images in Multi-Agent | 44.1 | 34.4 | 68.5 | 51.2 |
| w/o Prompt Layer & Images in Multi-Agent | 44.2 | 33.7 | 68.2 | 50.8 |

### Key Findings

1. **Effectiveness of Cross Extraction**: Full Cross Extraction achieves significant improvements over both Direct Extraction and Partial Cross Extraction, demonstrating that information sharing among agents is effective.
2. **Image Input is Crucial**: Removing image inputs in multi-agent extraction leads to a dramatic drop in geospatial accuracy from 69.4% to 44.1%.
3. **Reasoning Quality**: The geospatial, temporal, and event reasoning accuracy of GETReason reaches 81.4%, 76.9%, and 70.2%, respectively.
4. **Competitiveness of Good Guesser**: In terms of geospatial reasoning, Good Guesser outperforms GETReason in 3 out of 6 runs, indicating room for improvement in a single dimension.

## Highlights & Insights

- **Systematic Decomposition**: The complex image context understanding problem is systematically decomposed into three manageable sub-problems, each handled by a specialized agent, presenting a clear design logic.
- **GREAT Evaluation Metric**: A comprehensive evaluation metric taking into account geospatial distance (Haversine), hierarchical temporal weights, and semantic similarity of events is proposed, which is more reasonable than a simple F1 score.
- **Dataset Enhancement**: Systematic enhancement of the TARA dataset (TARA*) was conducted, supplementing it with event information, fine-grained spatio-temporal annotations, and reasoning chains.
- **Hallucination Mitigation via Cross-Validation**: Factual accuracy is enhanced through iterative information sharing among agents.

## Limitations & Future Work

1. **Dependence on Large Commercial Models**: The framework relies completely on closed-source models such as Gemini and GPT-4o, which are costly and difficult to reproduce.
2. **Ground Truth Generated by VLM**: The enhanced annotations in TARA* are themselves generated by Gemini 1.5 Pro, posing a risk of circular validation.
3. **High Computational Overhead**: Multi-agent cascading reasoning leads to a significant increase in inference time and API call costs.
4. **No End-to-End Training**: The purely prompt-based approach limits the optimization space of the model.
5. **Limited Event Type Coverage**: Primarily targets public event images, and its generalization capability has not been fully verified.

## Related Work & Insights

- **Multi-Agent Frameworks**: Draws inspiration from the collaborative multi-agent paradigms that have successfully been applied to the text/code domains in recent years (Dinh & Chan 2025, Ng et al. 2024).
- **Difference from RAG**: GETReason avoids the noise and hallucination issues associated with external knowledge sources in RAG methods.
- **Inspiration**: The multi-agent framework can be extended to other visual understanding tasks requiring multi-dimensional reasoning (e.g., disaster assessment, historical event analysis).

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing multi-agent reasoning into event image understanding is a novel attempt, but the core of the framework is essentially prompt engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient ablation studies with evaluations across two datasets, three models, and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and detailed definition of the GREAT metric.
- Value: ⭐⭐⭐ Clear practical application scenarios (news, archiving), but costs and dependence on closed-source models limit its practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](../../NeurIPS2025/multi_agent/gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)
- [\[ICML 2026\] Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning](../../ICML2026/multi_agent/toward_culturally_aligned_llms_through_ontology-guided_multi-agent_reasoning.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](../../CVPR2025/multi_agent/collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[ICML 2026\] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](../../ICML2026/multi_agent/mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)

</div>

<!-- RELATED:END -->
