---
title: >-
  [Paper Note] CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation
description: >-
  [CVPR 2025][LLM Alignment][Parametric CAD Generation] This paper proposes the CAD-Llama framework, which converts 3D CAD models into Python-style code rich in semantic descriptions (SPCC) via a hierarchical annotation pipeline. It then utilizes adaptive pre-training and instruction tuning to transform LLaMA3-8B into a parametric CAD model generator. This approach outperforms previous methods by approximately 14% in accuracy on the text-to-CAD task…
tags:
  - "CVPR 2025"
  - "LLM Alignment"
  - "Parametric CAD Generation"
  - "Large Language Models"
  - "Structured CAD Code"
  - "Hierarchical Annotation"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: 412a3d9a6561fab7
---

# CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation

**Conference**: CVPR 2025  
**arXiv**: [2505.04481](https://arxiv.org/abs/2505.04481)  
**Code**: None  
**Area**: Alignment RLHF / CAD Generation  
**Keywords**: Parametric CAD Generation, Large Language Models, Structured CAD Code, Hierarchical Annotation, Instruction Tuning

## TL;DR

This paper proposes the CAD-Llama framework, which converts 3D CAD models into Python-style code rich in semantic descriptions (SPCC) via a hierarchical annotation pipeline. It then utilizes adaptive pre-training and instruction tuning to transform LLaMA3-8B into a parametric CAD model generator. This approach outperforms previous methods by approximately 14% in accuracy on the text-to-CAD task, while supporting various CAD editing tasks such as completion, addition, and deletion.

## Background & Motivation

**Background**: Computer-Aided Design (CAD) generative modeling is a hot research topic in both academia and industry. In recent years, Large Language Models (LLMs) have demonstrated strong generalization capabilities in tasks like text and code generation, naturally sparking interest in exploring LLMs for parametric CAD sequence generation.

**Limitations of Prior Work**: Existing methods face several challenges. First, there is a giant gap between parametric CAD sequences and the natural language or code used in LLMs' pre-training—LLMs have never encountered CAD parameter sequences during pre-training, nor do they possess direct perception of 3D structures. Second, prior attempts to generate CAD using LLMs (e.g., CAD-LLM, LLM4CAD) can mostly handle only simple models, showing weak generalization and failing to generate precise CAD models based on complex textual instructions. Third, encoder-decoder-based methods (e.g., Text2CAD, CAD-Translator) are limited by restricted model capacity, leading to poor generalization on out-of-distribution samples.

**Key Challenge**: A semantic gap exists between the strong generative prior of LLMs and the specialized nature of CAD data—CAD parameter sequences themselves lack textual descriptions of design intent and geometry, making it impossible for LLMs to understand the meaning of these purely numerical operation sequences.

**Goal**: 1) How to enable LLMs to understand and generate parametric CAD operation sequences? 2) How to construct a bridge connecting natural language and CAD data? 3) How to support multiple CAD downstream tasks ranging from text generation to editing?

**Key Insight**: The authors observe that LLMs are particularly adept at code generation, which is well-understood by LLMs because code is typically accompanied by rich comments and functional descriptions. Therefore, the key lies in adding structured semantic annotations to CAD data, transforming it into an "annotated code" format. A VLM (such as GPT-4o) is leveraged to generate detailed textual annotations by hierarchically describing 3D renderings and 2D sketches of CAD models.

**Core Idea**: Convert CAD parameter sequences into Python-style code (SPCC) accompanied by hierarchical semantic descriptions, and then empower the LLM with CAD generation and editing capabilities through adaptive pre-training and instruction tuning.

## Method

### Overall Architecture

The CAD-Llama framework consists of two main parts: (1) SPCC data synthesis, which converts CAD sequences into SPCC representation via a hierarchical annotation pipeline; and (2) model training, which first performs adaptive pre-training on the SPCC corpus and then conducts instruction tuning on multi-task instruction data. The resulting model, CAD-Llama-INS, supports various tasks including text-to-CAD, completion, captioning, addition, and deletion.

### Key Designs

1. **Hierarchical Annotation Pipeline**:

    - **Function**: Generate multi-level, structured textual descriptions for CAD models.
    - **Mechanism**: Conducted in two stages. **Stage 1 - Component Description**: Render 3D orthographic projections and 2D sketches for each component of the CAD model, and feed them into a VLM (GPT-4o) to generate detailed geometric descriptions of the component (shape, dimensions, extrusion direction, etc.). **Stage 2 - Global Description**: Render silhouette views of each component (with the target component highlighted and other components semi-transparent). Feed these along with the component descriptions from Stage 1 into GPT-4o to generate two parts of global descriptions: (a) an abstract overview $\mathcal{A}$ (defining what the model is in one sentence) and (b) a detailed description $\mathcal{T}$ (spatial relationships between components and assembly process), as well as a short naming $\mathcal{S}$ for each component to link global and local descriptions. To improve output stability, CAD models are categorized into five complexity levels, with 50 high-quality examples provided per level using two-shot prompting.
    - **Design Motivation**: A single VLM call cannot simultaneously capture both fine-grained geometric attributes and the compositional relationships among components. Hierarchical annotation allows each layer to focus on information of different granularities, ensuring the completeness and accuracy of descriptions.

2. **Structured Parametric CAD Code (SPCC)**:

    - **Function**: Create an LLM-friendly CAD data representation.
    - **Mechanism**: Inspired by the code generation capability of LLMs, CAD parameter sequences are transformed into Python-style code. Each sketch is represented as a loop list (e.g., `sketch_i.append(loop1)`), where methods like `Line`, `Arc`, and `Circle` are called within the loops to draw. Extrusion operations reference corresponding sketches to execute. Coordinates use 8-bit quantized parameters (with the start point reset to (0,0)), and angles use discrete values from 0 to 360. The hierarchical descriptions are then embedded in the code: component descriptions are attached as code comments before the corresponding component code, and the global description serves as the overall prefix. The final SPCC corpus contains a detailed description version ($\tilde{\mathcal{D}}$) and an abstract-description-only version ($\dot{\mathcal{D}}$), allowing the LLM to handle inputs with varied levels of detail.
    - **Design Motivation**: The code format leverages the existing capabilities of LLMs on code datasets, while the hierarchical descriptions make up for the core defect of CAD sequences lacking semantic annotations.

3. **Adaptive Pre-training and Instruction Tuning**:

    - **Function**: Transform a generalist LLM into a CAD domain expert.
    - **Mechanism**: **Pre-training Stage**: Perform full parameter fine-tuning on the SPCC corpus using the standard autoregressive language modeling objective. The novelty lies in grouping similar CAD models into the same context: utilizing CLIP to encode CAD renderings, constructing a document graph based on cosine similarity, and building the training context by traversing the graph. This enables the LLM to compare differences between similar models in-context, facilitating more efficient learning. With a context window of 2048, CAD-Llama is obtained. **Instruction Tuning Stage**: Construct an instruction dataset containing six CAD tasks (text-to-CAD, captioning, completion, addition, deletion, and their SPCC-enhanced versions), with 12K instances per task, totaling 48K instances. Parameter-efficient fine-tuning is performed using LoRA (rank=256, $\alpha=128$) with a context window of 4096, yielding CAD-Llama-INS.
    - **Design Motivation**: Grouping similar models is inspired by curriculum learning, allowing the model to observe similar samples first to learn fine-grained differences. The multi-task design of instruction tuning enables the model to unifiedly handle various CAD-related downstream tasks.

### Loss & Training

The pre-training stage utilizes the standard next-token prediction loss: $\mathcal{L}(\mathcal{X}) = \sum_{i=1}^{n} \log P(x_i | x_{i-1}, ..., x_0; \Phi)$, computed over all tokens. The instruction tuning stage similarly uses next-token prediction but calculates the loss solely on the output portion: $\mathcal{L}(D) = \sum_{i=1}^{N} \log P(Y_i | X_i; \Theta)$. The base model is LLaMA3-8B-HF, the optimizer is AdamW (lr=2e-5), and DeepSpeed together with FlashAttention are used to accelerate training.

## Key Experimental Results

### Unconditional Generation

| Method | COV ↑ | MMD ↓ | JSD ↓ | SR ↑ | Novel ↑ |
|------|-------|-------|-------|------|---------|
| DeepCAD | 66.68 | 1.19 | 2.59 | 61.84 | 91.7 |
| SkexGen | 77.42 | 1.07 | 0.93 | 72.26 | 99.1 |
| HNC-CAD | 80.46 | 0.98 | 0.74 | 79.11 | 93.9 |
| **CAD-Llama** | **79.93** | **0.96** | **0.66** | **99.90** | **97.1** |

### Text-to-CAD

| Method | ACC_T ↑ | MCD ↓ | MMD ↓ | JSD ↓ |
|------|---------|-------|-------|-------|
| GPT-4 | 20.03 | 25.62 | 3.33 | 18.09 |
| LLaMA3 | 17.26 | 17.33 | 4.10 | 12.36 |
| Text2CAD | 69.91 | 20.64 | 3.02 | 9.98 |
| CAD-Translator | 70.36 | 21.29 | 2.94 | 10.92 |
| **CAD-Llama-INS** | **84.72** | **10.53** | **1.54** | **3.59** |

### Ablation Study (CAD Representations)

| Representation | ACC_cmd ↑ | ACC_param ↑ | SR ↑ |
|----------|-----------|-------------|------|
| SDCS (Single Description + Sequence) | 39.17 | 25.56 | 18.14 |
| SDCC (Single Description + Code) | 42.62 | 27.13 | 21.46 |
| SPCS (Hierarchical Description + Sequence) | 73.13 | 47.32 | 98.71 |
| **SPCC (Hierarchical Description + Code)** | **80.41** | **59.09** | **99.30** |

### Key Findings

- CAD-Llama-INS achieves an accuracy (ACC_T=84.72) on text-to-CAD that exceeds the previous best CAD-Translator by approximately 14 percentage points.
- Hierarchical description is the largest contributor to performance improvement—the ACC_cmd gap between SPCC and SDCC reaches 38 percentage points.
- Code format also contributes to the performance, though to a lesser extent than hierarchical description—the ACC_cmd gap between SPCC and SPCS is about 7 percentage points.
- The SR of unconditional generation reaches 99.90%, which is significantly higher than all baselines, indicating extremely high generation stability.
- In CAD editing tasks, using the SPCC format (deletion* / addition*) significantly outperforms pure CAD code format. GPT-4's EM on deletion* also improves from 66.20 to 90.41, proving that the structured information of SPCC benefits all LLMs.

## Highlights & Insights

- **Conceptualizing CAD as "annotated code" is the core insight**: By leveraging the existing code generation capabilities of LLMs, semantic annotations are added to bridge the gap between CAD and natural language.
- **The hierarchical annotation pipeline is elegantly designed**: The two-stage annotation from component to global levels and the few-shot strategy based on complexity levels guarantee the quality and consistency of VLM outputs.
- **The 99.90% generation success rate is highly impressive**: This indicates that the SPCC format indeed enables LLMs to "comprehend" the structural constraints of CAD models.
- **Practicality of multi-task instruction tuning**: A single model supports multiple operations simultaneously (generation, completion, editing, captioning, etc.), which aligns with actual CAD workflows.

## Limitations & Future Work

- Based on the DeepCAD dataset (178K models), the scale and complexity are limited—mostly mechanical parts, lacking more complex CAD categories such as architecture or organic shapes.
- The annotation pipeline relies on GPT-4o, leading to high cost and potential risks of hallucination.
- The coordinate precision under 8-bit quantization (256 levels) may not meet the accuracy requirements of industrial-grade CAD.
- Only 2D rendering images are utilized for annotation, which might not fully preserve 3D information.
- Comparisons with methods using multimodal inputs (point clouds, images -> CAD) are not conducted.
- Human evaluation is not performed, and qualitative validation of the practical design usability of the generated CAD models is lacking.

## Related Work & Insights

- **DeepCAD (Wu et al., 2021)**: Foundational work in parametric CAD generation, which uses Transformer to autoregressively generate CAD sequences.
- **Text2CAD / CAD-Translator**: Encoder-decoder-based text-to-CAD approaches, but suffer from limited generalization capability.
- **OpenECAD**: Leverages VLMs in conjunction with the PythonOCC CAD kernel for CAD generation.
- **CAD-GPT / CAD-MLLM**: Uses multimodal LLMs to generate CAD sequences, supporting various inputs such as images and point clouds.
- The SPCC approach in this paper can be extended to other specialized domains—adding hierarchical semantic descriptions to professional data and converting it into LLM-friendly formats stands as a universal strategy to unleash the capabilities of LLMs in domain-specific tasks.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Utility: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day](../../ICML2025/llm_alignment/instruction_tuning_of_large_language_models_for_tabular_data_generation-in_one_d.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](../../NeurIPS2025/llm_alignment/mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](../../ICML2025/llm_alignment/on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICML 2025\] PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](../../ICML2025/llm_alignment/poisonbench_assessing_large_language_model_vulnerability_to_data_poisoning.md)
- [\[ACL 2025\] Federated Data-Efficient Instruction Tuning for Large Language Models](../../ACL2025/llm_alignment/federated_data-efficient_instruction_tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
