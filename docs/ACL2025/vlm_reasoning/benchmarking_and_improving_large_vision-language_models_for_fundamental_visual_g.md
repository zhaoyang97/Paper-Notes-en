---
title: >-
  [Paper Note] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning
description: >-
  [ACL 2025][VLM Reasoning][Visual Graph Structure Understanding] This paper constructs a systematic evaluation benchmark to assess large vision-language models (LVLMs) on basic visual graph structure understanding and reasoning, finding that existing models perform poorly on such tasks, and proposes targeted improvement methods.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Visual Graph Structure Understanding"
  - "Visual Reasoning"
  - "Graph Understanding Benchmark"
  - "Large Vision-Language Models"
  - "Structured Visual Reasoning"
date: 2026-05-08
content_hash: 7a4e13db1403084d
---

# Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning

**Conference**: ACL 2025  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Visual Graph Structure Understanding, Visual Reasoning, Graph Understanding Benchmark, Large Vision-Language Models, Structured Visual Reasoning

## TL;DR
This paper constructs a systematic evaluation benchmark to assess large vision-language models (LVLMs) on basic visual graph structure understanding and reasoning, finding that existing models perform poorly on such tasks, and proposes targeted improvement methods.

## Background & Motivation

**Background**: Large vision-language models (such as GPT-4V, LLaVA, etc.) have achieved excellent performance in tasks like image captioning and VQA. However, their ability to understand structured visual information, such as graphs, networks, and flowcharts, has not been systematically evaluated.

**Limitations of Prior Work**: Existing LVLM evaluation benchmarks primarily focus on natural image understanding (e.g., scene recognition, object detection), neglecting an important visual understanding capability—the fundamental understanding of graph structures, including node recognition, edge relationship determination, and path reasoning.

**Key Challenge**: Graph structures are ubiquitous in scenarios such as scientific papers, knowledge graph visualizations, and flowcharts, but whether LVLMs truly understand these visual graph structures remains unclear. Graph structure understanding requires combining visual perception (recognizing nodes and edges) with logical reasoning (pathfinding, connectivity determination), which places higher demands on models.

**Goal**: (1) To construct a comprehensive benchmark covering multiple graph types and reasoning tasks; (2) to systematically evaluate the performance of existing LVLMs; and (3) to propose improvement methods to enhance the graph structure understanding capability of models.

**Key Insight**: Graphs are a fundamental data structure. Visualized graph understanding involves combining low-level visual perception with high-level structural reasoning, making it an ideal touchstone for testing the comprehensive capabilities of LVLMs.

**Core Idea**: Build a systematic visual graph understanding benchmark (VGraphBench) to reveal the structural understanding shortcomings of LVLMs, and apply graph structure-aware training strategies to remedy these deficiencies.

## Method

### Overall Architecture
The work is divided into two parts: (1) Benchmark construction—containing multiple graph types (directed, undirected, weighted graphs, trees, etc.) and various tasks (node counting, edge detection, path determination, shortest path, connectivity, etc.); (2) Improvement methods—using structured data augmentation and targeted fine-tuning to enhance the graph understanding capability of LVLMs.

### Key Designs

1. **Visual Graph Benchmarking (VGraphBench)**:

    - **Function**: Systematically evaluate the performance of LVLMs in fundamental graph structure understanding and reasoning.
    - **Mechanism**: Design tasks with multiple difficulty levels, ranging from simple node/edge recognition to complex path reasoning and graph property determination. Images are programmatically generated to ensure controlled variables (graph size, layout, color, etc.), avoiding distracting factors in natural images. Every task has a clear ground-truth answer.
    - **Design Motivation**: It is necessary to eliminate semantic priors in natural images so that models truly rely on their visual structure understanding capabilities.

2. **Multi-task Graph Understanding Evaluation System**:

    - **Function**: Cover the complete spectrum of capabilities from perception to reasoning.
    - **Mechanism**: Tasks are divided into two main categories—perception tasks (node counting, edge detection, degree calculation) and reasoning tasks (connectivity determination, shortest path, cycle detection, topological sorting, etc.). Perception tasks test whether the model "sees" the graph structure, while reasoning tasks test whether the model "understands" the graph structure.
    - **Design Motivation**: Distinguish between perception and reasoning failure modes to help diagnose specific model weaknesses.

3. **Graph Structure-Aware Fine-Tuning Strategy**:

    - **Function**: Improve LVLM performance by constructing graph structure understanding training data.
    - **Mechanism**: Generate a large number of image-question-answer pairs containing graph structures, covering various graph types and task types, to perform instruction fine-tuning on LVLMs. The training data includes progressive structural understanding tasks from simple to complex, helping the model gradually build visual understanding of graph structures.
    - **Design Motivation**: Pre-training data of existing LVLMs lacks sufficient graph structure understanding samples, necessitating targeted data supplementation.

### Loss & Training
The fine-tuning phase uses standard instruction fine-tuning loss (cross-entropy), with the key lying in the construction strategy of the training data.

## Key Experimental Results

### Main Results

| Model | Node Counting | Edge Selection | Connectivity | Shortest Path | Average |
|------|---------|--------|--------|---------|------|
| GPT-4V | Medium | Medium | Low | Low | ~45% |
| LLaVA-1.5 | Low | Low | Low | Very Low | ~30% |
| Ours (Fine-tuned) | Significant Gain | Significant Gain | Gain | Gain | ~60%+ |
| Random Baseline | ~20% | ~50% | ~50% | ~10% | ~25% |

### Ablation Study

| Configuration | Average Accuracy | Description |
|------|-----------|------|
| Full Fine-tuning | Best | Complete graph structure training data |
| Perception Tasks Only | Medium | Limited gain in reasoning tasks |
| Reasoning Tasks Only | Low | Insufficient perception foundation affects reasoning |
| No Graph Layout Augmentation | Drop | Sensitive to layout variations |

### Key Findings
- All existing LVLMs perform far below human levels on graph structure reasoning tasks, especially on tasks like shortest path and topological sorting where performance is close to random.
- Graph size (number of nodes) is a critical factor; accuracy drops sharply when the number of nodes exceeds 10.
- Perception is the foundation of reasoning—if the model cannot accurately recognize nodes and edges, reasoning tasks are bound to fail.
- Simple fine-tuning brings significant improvements, indicating this is not a fundamental limitation at the architectural level, but rather a lack of training data coverage.

## Highlights & Insights
- **Filling the Evaluation Gap**: Graph structure understanding is an important dimension of LVLM capability evaluation that has been previously neglected. The systematic design of the benchmark helps the community pinpoint model weaknesses.
- **Programmatic Generation with Controlled Variables**: Generating images via code rather than collecting natural images eliminates interference from semantic priors, representing the correct way to evaluate structural understanding capability. This can be transferred to the evaluation of other structured visual content such as flowcharts, UML diagrams, and circuit diagrams.

## Limitations & Future Work
- The benchmark utilizes programmatically generated "clean" graph structures; real-world images (such as hand-drawn diagrams, chart figures in papers) are much more challenging.
- The generalization of the fine-tuning method remains to be verified—whether it is still effective on graph types outside the training distribution.
- It can be extended to more complex graph types such as hypergraphs and dynamic graphs.
- Combining the ideas of Graph Neural Networks (GNNs) to enhance the graph structure understanding of LVLMs is a promising direction.

## Related Work & Insights
- **vs. Benchmarks like MathVista/ChartQA**: These benchmarks focus on math/chart understanding, whereas this work focuses on more fundamental graph structure understanding, representing a lower-level capability.
- **vs. TextVQA/DocVQA**: Document understanding focuses on text layout, whereas this work focuses on topological structure, making them complementary.
- **vs. NLGraph (Text-based Graph Reasoning)**: NLGraph describes graph structures using text, whereas this work evaluates understanding graph structures from vision, which is closer to real-world scenarios.
- **vs. GNN-related Works**: GNNs operate directly on graphs, whereas this work evaluates the ability of LVLMs to extract graph structures from image renderings.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematical evaluation of LVLM's graph structure understanding capability
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-task, multi-difficulty evaluation
- Writing Quality: ⭐⭐⭐⭐ Clean problem definition, reasonable benchmark design
- Value: ⭐⭐⭐⭐ Reveals important capability gaps in LVLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](../../NeurIPS2025/vlm_reasoning/mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[ACL 2025\] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](../../AAAI2026/vlm_reasoning/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](../../CVPR2025/vlm_reasoning/insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
