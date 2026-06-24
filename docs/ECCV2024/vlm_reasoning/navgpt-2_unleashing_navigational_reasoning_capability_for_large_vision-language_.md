---
title: >-
  [Paper Note] NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models
description: >-
  [ECCV 2024][VLM Reasoning][Vision-Language Navigation] NavGPT-2 closes the performance gap between LM-based agents and VLN-specific models while retaining the LLM's interpretable navigational reasoning capabilities by feeding the hidden layer representations of a frozen LLM into a topological map navigation policy network as vision-language features, showcasing excellent data efficiency.
tags:
  - "ECCV 2024"
  - "VLM Reasoning"
  - "Vision-Language Navigation"
  - "Large Language Models"
  - "Navigational Reasoning"
  - "Topological Map Policy"
  - "InstructBLIP"
date: 2026-05-08
content_hash: c22846d0ea08f453
---

# NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.12366](https://arxiv.org/abs/2407.12366)  
**Code**: [GitHub](https://github.com/GengzeZhou/NavGPT-2)  
**Area**: Multimodal VLMs  
**Keywords**: Vision-Language Navigation, Large Language Models, Navigational Reasoning, Topological Map Policy, InstructBLIP

## TL;DR

NavGPT-2 closes the performance gap between LM-based agents and VLN-specific models while retaining the LLM's interpretable navigational reasoning capabilities by feeding the hidden layer representations of a frozen LLM into a topological map navigation policy network as vision-language features, showcasing excellent data efficiency.

## Background & Motivation

Vision-Language Navigation (VLN) requires agents to navigate in real 3D environments following natural language instructions, serving as a core task of embodied AI. Recently, Large Language Models (LLMs) have been introduced to VLN, primarily following two paradigms:

**Zero-shot methods** (e.g., NavGPT, MapGPT): Translate visual content into text via image captions, and then use complex prompt engineering to let GPT-4 reason about navigation actions. However, this suffers from severe information loss, high complexity, and insufficient understanding of spatial structures, resulting in a ~40% success rate gap compared to specialized models.

**Fine-tuning methods** (e.g., LangNav, NaviLLM): Directly fine-tune models like LLaMA for VLN. However, they suffer from insufficient training data and mismatched pre-training and VLN objectives. Moreover, fine-tuning compromises the LLM's general language capabilities, turning it into a "black box".

**Key Challenge**: Existing methods either sacrifice performance for interpretability (zero-shot) or sacrifice interpretability for performance (fine-tuning), failing to achieve both.

**Key Insight**: NavGPT-2 strikes a balance between these two extremes by freezing the LLM, utilizing its hidden layer features as vision-language representations to feed the navigation policy network, while retaining the LLM's language generation capability to provide interpretable navigational reasoning.

## Method

### Overall Architecture

NavGPT-2 consists of two major components: (1) a Large Vision-Language Model (VLM) based on the InstructBLIP architecture, and (2) a topological map-based navigation policy network. The VLM is responsible for processing visual observations and instructions and generating navigational reasoning, while the policy network handles action prediction. Training is conducted in two stages, and both the VLM and the LLM remain frozen throughout the entire process.

### Key Designs

1. **Visual Alignment and Multi-view Perception**:

    - **Function**: Encodes RGB images from multiple candidate views in the environment into fixed-length visual tokens.
    - **Mechanism**: Adapting the Q-former design (from BLIP-2), frozen ViT-g/14 is used to extract visual features for each candidate view image. These are then cross-attended with 32 learnable queries, self-attended with the instruction text to obtain instruction-aware image queries, and finally linearly projected to the LLM input.
    - **Design Motivation**: Q-former effectively controls the token length of multi-view images, avoiding the issue of excessively long context.

2. **Navigation System Prompts and Reasoning Data Generation**:

    - **Function**: Construct structured navigation prompts, injecting directional information (e.g., "Candidate i, facing angle, {direction}"), and use GPT-4V to generate 10K single-step navigational reasoning data points from the R2R training set.
    - **Mechanism**: Organize images and instructions using special tokens (`<IMG>`, `</IMG>`, `<INST>`, `</INST>`), and perform instruction tuning on the Q-former and the projection layer.
    - **Design Motivation**: Enable the frozen LLM to output environmental descriptions, progress judgments, and next-step reasoning.

3. **VLM Hidden Layers as Vision-Language Representations**:

    - **Function**: Extract the hidden representation of the last encoder/decoder layer from the LLM as input features for the downstream policy network.
    - **Mechanism**: For encoder-decoder models (e.g., FlanT5), representations of image tokens and instruction tokens are taken from the last encoder layer; for decoder-only models (e.g., Vicuna), they are taken from the last decoder layer. The 32 image tokens of each view are merged into a single token via an MLP.
    - **Design Motivation**: The hidden layers of the LLM have already completed deep fusion of vision and language, providing high-quality cross-modal representations.

4. **Topological Map-based Navigation Policy Network**:

    - **Function**: Maintain a dynamic topological map to enable global action prediction and historical rollback.
    - **Mechanism**:
        - **Node Embedding**: Visited nodes are represented by average pooling of all candidate view features, while unexplored nodes are represented by partial views from adjacent visited nodes. Each view feature = VLM visual feature + orientation embedding + step embedding, with spatial relationships between nodes modeled via multi-layer Transformers.
        - **Cross-modal Encoding**: Node embeddings first cross-attend with instructions encoded by the LLM, and then pass through Graph-Aware Self-Attention (GASA). GASA incorporates a spatial affinity matrix based on the L2 distance between nodes on top of standard self-attention.
        - **Global Action Prediction**: A two-layer FFN is used to calculate action scores on GASA outputs, selecting the node with the highest score and moving along the shortest path on the map.
    - **Design Motivation**: The topological map effectively models long-term navigation history and spatial structures, supporting rollback of erroneous paths.

### Loss & Training

A two-stage training strategy is adopted:
- **Stage One**: Freeze the LLM and the visual encoder, fine-tuning only the Q-former and the projection layer by performing instruction tuning on the navigational reasoning data generated by GPT-4V (200K steps, batch=8).
- **Stage Two**: Freeze the entire VLM, fine-tuning only the navigation policy network. A joint loss of Behavior Cloning and DAgger is utilized: $\mathcal{L} = \lambda \mathcal{L}_{BC} + \mathcal{L}_{DAG}$, where BC is trained on ground truth trajectories, and DAgger is trained using pseudo-labels on trajectories sampled by the agent itself.

All experiments are conducted on a single A100 GPU.

## Key Experimental Results

### Main Results (R2R Dataset)

| Method | Val Unseen SR↑ | Val Unseen SPL↑ | Test SR↑ | Test SPL↑ | Is LLM Frozen |
|------|:---:|:---:|:---:|:---:|:---:|
| NavGPT (GPT-4, Zero-shot) | 34 | 29 | - | - | ✓ |
| NavCoT (LLaMA2-7B) | 40 | 37 | - | - | ✗ |
| NaviLLM (Vicuna-7B) | 67 | 59 | 68 | 60 | ✗ |
| DUET (Specialized Model) | 72 | 60 | 69 | 59 |  - |
| **NavGPT-2 (FlanT5-XL, 1.5B)** | **68** | **56** | **71** | **60** | ✓ |
| **NavGPT-2 (FlanT5-XXL, 5B)** | **74** | **61** | **72** | **60** | ✓ |

### Ablation Study

| Description | Val Seen SR | Val Unseen SR | Description |
|------|:---:|:---:|------|
| NavGPT-2 Full Model | 69.44 | 67.52 | Baseline |
| w/o Policy Network | 25.27 | 21.46 | Frozen LLMs cannot make action decisions directly |
| w/o reasoning pre-trained Q-former | 67.58 | 66.75 | Reasoning pre-training brings slight improvement |

### Data Efficiency Experiment

| Method | Training Data Volume | Val Unseen SR |
|------|:---:|:---:|
| DUET | 100% R2R | 63.90 |
| NavGPT-2 | 50% R2R | 63.30 |
| NavGPT-2 | 100% R2R | 67.52 |

### Key Findings

- NavGPT-2 achieves comparable performance with only 50% of the training data to DUET with 100% data, demonstrating the data efficiency advantage of using LLM hidden layer representations.
- FlanT5 (encoder-decoder) significantly outperforms Vicuna (decoder-only) because full attention is better suited for multi-choice action prediction in VLN.
- In zero-shot cross-dataset generalization, NavGPT-2 outperforms DUET by 3.67% SR on RxR and 21.6% SR on HM3D.
- Human evaluation indicates acceptable reasoning quality from NavGPT-2 (Accuracy: 1.66/3, Informativeness: 1.93/3).

## Highlights & Insights

- The **frozen LLM + policy network** design is elegant: it leverages the strong cross-modal representation capabilities of the LLM while compensating for its lack of spatial understanding through a specialized policy network.
- Versatility of VLM hidden representations: The same representations are used for both language decoding (navigational reasoning) and action decoding (policy network), achieving a unified feature space.
- Quantitative proof of data efficiency: 50% training data $\approx$ 100% data performance of specialized models.
- Demonstrates a feasible path for using LLMs as "feature extractors" rather than "decision makers" in VLN.

## Limitations & Future Work

- The LLM remains frozen, preventing it from further learning spatial reasoning capabilities directly from navigation tasks.
- The current quality of reasoning generation (1.66/3) still has significant room for improvement.
- The approach relies on GPT-4V to generate training data, which incurs high costs.
- Decoder-only models like Vicuna perform poorly, suggesting a need to explore better adaptation methods.

## Related Work & Insights

- NavGPT (2023) first applied GPT-4 to zero-shot navigation in VLN, revealing the potential of LLMs for navigational reasoning, though its performance fell far short of specialized models.
- The topological map navigation policy proposed by DUET (2022) is a key design in VLN, which NavGPT-2 directly adapts for its global branch.
- The Q-former architecture from InstructBLIP provides flexible token-length control for multi-image inputs.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of frozen LLM + hidden layer representations + policy network is a novel approach in this field, with clear underlying concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, including main experiments, ablation studies, data efficiency, cross-dataset generalization, human evaluations, and comparisons across various LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, persuasive motivation, and fluent writing.
- Value: ⭐⭐⭐⭐ Provides a practical solution for applying LLMs to VLN, closing the performance gap with specialized models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](../../ACL2026/vlm_reasoning/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ICLR 2026\] Unleashing Perception-Time Scaling to Multimodal Reasoning Models](../../ICLR2026/vlm_reasoning/unleashing_perception-time_scaling_to_multimodal_reasoning_models.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](../../CVPR2026/vlm_reasoning/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](../../ACL2026/vlm_reasoning/addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](../../ACL2026/vlm_reasoning/geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)

</div>

<!-- RELATED:END -->
