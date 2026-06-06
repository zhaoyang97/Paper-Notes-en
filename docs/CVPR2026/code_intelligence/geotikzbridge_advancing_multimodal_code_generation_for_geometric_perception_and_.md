---
title: >-
  [Paper Note] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning
description: >-
  [CVPR 2026][Code Intelligence][Geometric Perception] GeoTikzBridge constructs the largest 2.5M image–TikZ code dataset and the first auxiliary-line instruction dataset…
tags:
  - "CVPR 2026"
  - "Code Intelligence"
  - "Geometric Perception"
  - "TikZ Code Generation"
  - "Multimodal Reasoning"
  - "Auxiliary Line Generation"
  - "Image-to-Code"
date: 2026-05-08
content_hash: 937af20fbc362f67
---

# GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning

**Conference**: CVPR 2026
**arXiv**: [2603.22687](https://arxiv.org/abs/2603.22687)  
**Code**: Available (public)  
**Area**: Code Intelligence
**Keywords**: Geometric Perception, TikZ Code Generation, Multimodal Reasoning, Auxiliary Line Generation, Image-to-Code

## TL;DR
GeoTikzBridge constructs the largest 2.5M image–TikZ code dataset and the first auxiliary-line instruction dataset, trains a code generation model capable of accurately reconstructing geometric figures, and serves as a plug-and-play module to enhance the geometric reasoning capabilities of arbitrary MLLMs/LLMs.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have made substantial progress in cross-modal perception and reasoning, yet geometric problems remain challenging. Such problems require integrating fine-grained visual perception with structured symbolic reasoning. Existing image-to-code methods focus primarily on Web UI→HTML/CSS or chart→Python conversions, with little attention to geometric content. In mathematical reasoning, prevailing approaches rely predominantly on textual reasoning and overlook the need for relational propagation in geometric visual reasoning.

**Limitations of Prior Work**: MLLMs exhibit limited capability in local geometric perception, struggling to precisely parse segment relationships, angle magnitudes, shape constraints, and other fine-grained visual details. This is mainly due to: (1) the absence of large-scale geometric image–code datasets (DaTikZ contains only 145k samples with limited geometric coverage); and (2) insufficient modeling of subtle geometric variations.

**Key Challenge**: Geometric reasoning demands precise symbolic representations of figure structures, yet existing data and methods cannot provide adequate geometric perception training signals for MLLMs. TikZ code is more suitable for geometric reasoning than SVG because its syntax inherently records the logical steps and dependencies of geometric construction.

**Goal**: (1) How to construct a sufficiently large geometric image–TikZ code dataset for model training? (2) How to direct the model's attention toward local geometric details rather than generating code in a coarse-grained manner? (3) How to transfer geometric perception capabilities to downstream reasoning tasks?

**Key Insight**: The authors propose an iterative self-refinement strategy for dataset scaling, a localized geometric transformation strategy for fine-grained perception enhancement, and instruction-guided auxiliary line generation to empower reasoning.

**Core Idea**: By combining iterative data expansion with localized code transformation to construct a 2.5M-scale geometric TikZ dataset, the work trains a geometric code generation model that can serve as a plug-and-play reasoning module.

## Method

### Overall Architecture
The GeoTikzBridge framework consists of three components: (1) an iterative self-refinement framework for constructing the GeoTikz-Base dataset and training the GeoTikzBridge-Base model; (2) an instruction-guided GeoTikz-Instruct dataset and GeoTikzBridge-Instruct model; and (3) a training-free plug-and-play visual reasoning pipeline. The input is a geometric image; the output is compilable TikZ code.

### Key Designs

1. **Iterative Self-Refinement Data Construction**:

    - **Function**: Scales from 145k seed samples to a 2.5M large-scale geometric TikZ dataset.
    - **Mechanism**: DaTikZ serves as the seed dataset $\mathcal{D}_0$ to train an initial model $M_0$. Candidate images are collected from 9 public geometric datasets. Each iteration proceeds in three steps: (a) the current model predicts TikZ code for candidate images; rendered images are compared against originals using CLIP score, and samples exceeding threshold $\tau=0.8$ are added to the self-refined set $\mathcal{D}_k^R$; (b) localized code transformations are applied to reliable samples to obtain the augmented set $\mathcal{D}_k^T$; (c) the model is retrained on the merged data $\mathcal{D}_k = \mathcal{D}_{k-1} \cup \mathcal{D}_{k-1}^R \cup \mathcal{D}_{k-1}^T$. After 4 iterations, 2.5M samples are obtained.
    - **Design Motivation**: Geometric image–code data is scarce and direct annotation is costly. Through a bootstrapping cycle of "generate → filter → train," the model grows stronger each round and can annotate more data, forming a virtuous cycle.

2. **Localized Geometric Transformation Strategy**:

    - **Function**: Enhances the model's perception of fine-grained geometric details.
    - **Mechanism**: Consists of code transformation and image transformation. Code transformation randomly deletes 1 to $n$ lines from TikZ code (no more than 40%), retaining compilable modified code $\tilde{C}$ and its rendered image $\tilde{I}$ as new training pairs. This forces the model to learn the structural semantics of code rather than memorizing specific text sequences, acting as code noise injection to improve generalization and robustness. This strategy reduces the code repetition prediction rate by 15%.
    - **Design Motivation**: Complex images often cause models to overlook fine-grained geometric details, resulting in omission or hallucination of critical code lines. Localized editing compels the model to attend to the presence or absence of each geometric element.

3. **Instruction-Guided Auxiliary Line Generation (GeoTikz-Instruct)**:

    - **Function**: Enables the model to add auxiliary lines to geometric figures according to instructions, providing intermediate steps for reasoning.
    - **Mechanism**: Code transformations are applied to samples in $\mathcal{D}_K$ to obtain transformed images; Qwen2.5-VL-72B annotates instructions $Q$ describing the auxiliary line changes; Doubao performs VLM-based filtering to ensure quality. The final training triplets are (instruction $Q'$, transformed image $\tilde{I}'$, original code $C'$), yielding 419k training samples. GeoTikzBridge-Instruct is obtained via SFT on top of GeoTikzBridge-Base.
    - **Design Motivation**: Many geometric problems require auxiliary lines to solve, yet existing MLLMs cannot generate accurate auxiliary lines. Code transformation naturally produces before-and-after contrasts of "adding/removing geometric elements," enabling automatic construction of auxiliary line training data.

### Loss & Training
A standard causal autoregressive modeling objective is used: $\mathcal{L}_{\text{gen}} = -\sum_i \log P_M(c_i | I, c_{<i})$. The 8B model undergoes full-parameter SFT (learning rate 4e-7); the 38B model is fine-tuned with LoRA (learning rate 1e-4). DeepSpeed ZeRO-3 and Flash Attention are employed. Training runs on 8× H100 GPUs for approximately 96 GPU hours (8B) and 488 GPU hours (38B). Greedy decoding (temperature=0) is used at inference.

## Key Experimental Results

### Main Results — Image-to-TikZ Generation

| Method | DaTikZ CLIP-S↑ | DaTikZ FID↓ | MathVista-GPS CLIP-S↑ | EDU CLIP-S↑ |
|--------|----------------|-------------|----------------------|------------|
| Qwen2.5-VL-72B | 0.795 | 49.8 | 0.858 | 0.781 |
| InternVL3-78B | 0.747 | 62.7 | 0.860 | 0.801 |
| FigCodifier-8B | 0.785 | 45.8 | 0.884 | 0.675 |
| **GeoTikzBridge-Base-8B** | **0.804** | 43.6 | **0.895** | **0.795** |
| **GeoTikzBridge-Base-38B** | **0.813** | **39.7** | **0.915** | **0.821** |

### Downstream Mathematical Reasoning Improvement

| Baseline VLM | MathVista-GPS | GAOKAO-MM-Math |
|--------------|--------------|----------------|
| GLM4.5-V-106B | 0.745 | 0.613 |
| +GeoTikzBridge-Base | **0.764** (+1.9%) | **0.663** (+5.0%) |
| Skywork-OR1-32B (LLM)+TikZ | **0.861** | 0.663 |
| GPT-OSS-120B (LLM)+TikZ | **0.880** | 0.688 |

### Ablation Study

| Configuration | MathVista-GPS Accuracy |
|---------------|----------------------|
| InternVL3.5-38B baseline | 0.688 |
| + TikZ code + auxiliary line image | 0.697 |
| + auxiliary line image + auxiliary line code | 0.707 |
| + TikZ code + auxiliary line image + code | **0.736** |

### Key Findings
- The combination of LLM + TikZ code generally outperforms VLM direct image observation, attributed to catastrophic forgetting of language reasoning capabilities in VLMs during visual-language alignment training.
- Auxiliary lines in TikZ code form are more effective than their rendered image form, indicating that symbolic representations are more critical for reasoning.
- The localized code transformation strategy yields significant improvements in both compilation success rate and CLIP score, with the code repetition rate reduced by 15%.
- GeoTikzBridge surpasses GPT-5.0 on geometric code perception.

## Highlights & Insights
- The dual use of "code transformation" is particularly elegant: it simultaneously serves as data augmentation to improve model robustness and as an automatic construction method for auxiliary line training data—one operation solving two problems.
- The finding that LLM + TikZ code outperforms VLM direct image observation is highly instructive. It suggests a new multimodal reasoning paradigm: rather than having the reasoning model directly observe images, a dedicated perception model first converts images into executable symbolic representations, which are then passed to a pure language reasoning model—effectively achieving a decoupled "perception–reasoning" architecture.
- The data flywheel effect of iterative self-refinement is worth emulating: small initial data → weak model training → weak model annotates more data → filtering and augmentation → stronger model training.

## Limitations & Future Work
- The current approach is limited to geometric figures and has not been extended to technical diagrams such as circuit diagrams or engineering drawings.
- Auxiliary line generation relies on a VLM to first determine whether auxiliary lines are needed; incorrect judgments cause the entire pipeline to fail.
- The dataset primarily covers planar and analytic geometry, with limited coverage of solid geometry and topological figures.
- Although the TikZ code compilation success rate exceeds 95%, the remaining ~5% failure rate may affect practical deployment.

## Related Work & Insights
- **vs. FigCodifier**: Both are image-to-TikZ models, but FigCodifier has only 8B parameters and limited training data. GeoTikzBridge comprehensively surpasses it through 16× more data and the localized transformation strategy.
- **vs. DaTikZ dataset**: DaTikZ is the largest existing image–TikZ dataset (145k) but has limited geometric samples. GeoTikz-Base reaches 2.5M samples with a dedicated focus on the geometric domain.
- **vs. Mathematical reasoning models (R1 series)**: These models excel at textual reasoning but cannot directly process geometric images. GeoTikzBridge "bridges" LLM reasoning capabilities and visual perception by converting images into TikZ code.

## Rating
- Novelty: ⭐⭐⭐⭐ The complete pipeline from geometric perception → TikZ code → reasoning enhancement is novel in design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers image-to-code generation, downstream reasoning, and auxiliary line generation across multiple dimensions, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive framework diagrams.
- Value: ⭐⭐⭐⭐⭐ The 2.5M dataset and plug-and-play reasoning module offer substantial practical value for the geometric reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](../../ACL2026/code_intelligence/storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)

</div>

<!-- RELATED:END -->
