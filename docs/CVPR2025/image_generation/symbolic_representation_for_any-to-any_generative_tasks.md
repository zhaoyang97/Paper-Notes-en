---
title: >-
  [Paper Note] Symbolic Representation for Any-to-Any Generative Tasks
description: >-
  [CVPR 2025][Image Generation][Symbolic Representation] This paper proposes a symbolic generative task description language (A-Language) and a training-free inference engine. It maps natural language instructions into executable symbolic flows composed of function-parameter-topology triplets, achieving unified processing across 12 categories of multimodal generation tasks and matching or exceeding end-to-end trained unified multimodal models in quality and flexibility.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Symbolic Representation"
  - "Any-to-Any Generation"
  - "Workflow Synthesis"
  - "Training-Free"
  - "Multimodal Tasks"
date: 2026-05-08
content_hash: 8aee816d2e584b75
---

# Symbolic Representation for Any-to-Any Generative Tasks

**Conference**: CVPR 2025  
**arXiv**: [2504.17261](https://arxiv.org/abs/2504.17261)  
**Code**: [https://github.com/Jiaqi-Chen-00/Any-2-Any](https://github.com/Jiaqi-Chen-00/Any-2-Any)  
**Area**: Image Generation  
**Keywords**: Symbolic Representation, Any-to-Any Generation, Workflow Synthesis, Training-Free, Multimodal Tasks

## TL;DR

This paper proposes a symbolic generative task description language (A-Language) and a training-free inference engine. It maps natural language instructions into executable symbolic flows composed of function-parameter-topology triplets, achieving unified processing across 12 categories of multimodal generation tasks and matching or exceeding end-to-end trained unified multimodal models in quality and flexibility.

## Background & Motivation

**Background**: Any-to-any multimodal generation (mapping any modality input to any modality output) is a core challenge in generative AI. Existing approaches are divided into two categories: implicit neural modeling (e.g., Show-o, Unified-IO), which learns cross-modal mappings through large-scale training, and agent-based methods (e.g., HuggingGPT, GenAgent), which rely on multi-agent collaboration and tool scheduling.

**Limitations of Prior Work**: Implicit neural modeling is constrained by the range of training data, cannot handle rare or unforeseen tasks (such as image blending), and features implicit representations that cannot be interrupted or edited. Agent-based methods rely on complex multi-agent coordination, introducing system instability and operational overhead, and lack a unified formal task representation.

**Key Challenge**: Existing methods present a severe trade-off between generalization ability and controllability—neural models are high-quality but inflexible, while agent-based methods are flexible but unstable.

**Goal**: To design a unified symbolic task representation combined with pretrained language models to achieve training-free any-to-any generation.

**Key Insight**: Any generative task can be decomposed into three primary components—functions (computational operations), parameters (behavioral control), and topology (data flow structure)—which together form a complete task description.

**Core Idea**: Replace implicit neural representations with explicit symbolic flows to describe generative tasks, using LLMs as the inference engine to automatically convert natural language into this symbolic representation.

## Method

### Overall Architecture

Given a natural language task description $s$, input data $\mathcal{X}$ (of any modality), and a constraint set $\mathcal{C}$, the inference engine generates a complete symbolic representation $\Omega(t) = (\mathcal{F}, \Phi, \mathcal{T})$ through two-stage inference, where $\mathcal{F}$ is the function set, $\Phi$ is the parameter space, and $\mathcal{T}$ represents the topological connections. The generated symbolic flow is compiled and executed, invoking actual generative models via a ComfyUI backend to complete the task.

### Key Designs

1. **A-Language Symbolic Representation Language**:

    - **Function**: Formalize any generative task into function-parameter-topology triplets.
    - **Mechanism**: Each function $f_i: \mathcal{I}_i \times \phi_i \rightarrow \mathcal{O}_i$ is an atomic computational unit (e.g., image encoding, blending, decoding); parameters $\phi_i$ control the function's behavior (e.g., blending strength); topology $d_k = (f_j, y_j) \rightarrow (f_i, x_i)$ defines the exact data flow direction. Their combination forms a symbolic flow $\mathcal{S} = \{(f_i, \phi_{f_i}, D_i)\}$.
    - **Design Motivation**: To provide an implementation-independent, unified task description that is both mathematically rigorous and LLM-friendly. It explores three syntactic styles: declarative, data-flow, and pseudo-natural language.

2. **Two-Stage Training-Free Inference Engine**:

    - **Function**: Automatically convert natural language instructions into executable symbolic flows.
    - **Mechanism**: The first stage $\psi_1$ leverages an LLM to infer the required functions and parameters $(\mathcal{F}, \Phi)$ based on instructions and constraints; the second stage $\psi_2$ constructs the topological connections $\mathcal{T}$ based on the identified components. Separating the two stages reduces the complexity of single-step inference; RAG is utilized to retrieve the 3 most relevant reference programs out of 16 as in-context examples.
    - **Design Motivation**: Compared to one-step generation, a multi-stage process handles dependencies between components more effectively. Ablation studies demonstrate that removing either stage leads to a significant performance drop (41% $\rightarrow$ 28.5%/22%).

3. **Iterative Refinement Module**:

    - **Function**: Automatically correct symbolic flows in the event of compilation or execution failures.
    - **Mechanism**: $\Omega_{i+1}(t) = R(\Omega_i(t), \epsilon_i)$, where $R$ is the refinement operator and $\epsilon_i$ represents the detected error message. The LLM analyzes the error signals and adjusts the symbolic flow accordingly (e.g., modifying parameters, adding missing components, or reconstructing topology), with a maximum number of iterations set to prevent infinite loops.
    - **Design Motivation**: Generative task workflows are complex, making single-shot correct generation impractical; an automatic recovery mechanism is an essential component of a practical system.

### Loss & Training

This method is entirely training-free. GPT-4o is employed as the inference engine, with text-embedding-3-large as the embedding model for RAG retrieval. All experiments are conducted on a single L4 GPU (24GB) utilizing ComfyUI as the execution backend.

## Key Experimental Results

### Main Results

| Task | Show-o | SEED-X | LWM | U-IO 2 | Ours |
|------|--------|--------|-----|--------|------|
| Inpaint Rank | 1.6 | ✗ | ✗ | - | **1.4** |
| T2I Rank | 2.8 | 2.0 | 4.2 | 4.5 | **1.5** |
| I2V Rank | ✗ | ✗ | ✗ | - | **1.0** |
| T2A Rank | ✗ | ✗ | ✗ | 2.0 | **1.0** |
| 3D Gen Rank | ✗ | ✗ | ✗ | ✗ | **1.0** |

ComfyBench Performance:

| Method | Vanilla | Complex | Creative | Total |
|------|---------|---------|----------|------|
| ComfyAgent | 46.00 | 21.67 | 15.00 | 32.50 |
| HuggingGPT | 21.00 | 0.00 | 5.00 | 11.50 |
| **Ours** | **61.00** | **28.89** | **19.17** | **43.00** |

### Ablation Study

| Two-Stage Inference | Iterative Refinement | Total Success Rate |
|------------|----------|----------|
| ✓ | ✗ | 28.50% |
| ✗ | ✓ | 22.00% |
| **✓** | **✓** | **41.00%** |

Compilation/execution success rate: Ours 98%/87% vs. GenAgent 84%/63%

### Key Findings

- Win rate of the symbolic method across 12 task categories: 94% against Show-o, 98% against LVM (T2I), and 67% against Gen-3 (I2V alignment).
- Declarative syntax achieves the lowest error rate in complex workflows, whereas the pseudo-natural language format produces the most errors.
- Symbolic representations inherently support workflow editing: users can directly modify functions (changing models) or parameters (tuning prompts), which is impossible with end-to-end models.
- The method can handle complex non-atomic tasks consisting of 11-13 components and 11-17 connections.

## Highlights & Insights

- Novel Perspective: Generative tasks are treated as "program execution" rather than "end-to-end mapping," transcending the constraints of training data.
- Training-Free Design: Adding new task types only requires extending the function library, bypassing the need for retraining.
- The explicit nature of symbolic flows brings interruptibility, editability, and debuggability, which are vital for safety-sensitive scenarios.
- Elegant Integration: Achieves an elegant integration of neural models (for executing functions) and symbolic reasoning (for workflow planning).

## Limitations & Future Work

- Overall performance is bounded by the capabilities of the underlying functions (e.g., the models available in the ComfyUI ecosystem).
- The success rate for complex tasks (Creative category) remains low (19.17%), with topological inference being the primary bottleneck.
- Integrating this approach with agent-based methods could potentially enhance flexibility further.
- The cost and latency of error recovery are not discussed; practical deployment may require limiting the number of refinement iterations.

## Related Work & Insights

- The neuro-symbolic approach of VISPROG [12] is a direct predecessor, upon which this paper builds a more complete formalization.
- The multi-agent collaborative approach of GenAgent [49] is actually less stable than the symbolic method on simple tasks.
- Insight: For structured tasks, combining explicit symbolic representations with LLM inference can be more efficient than end-to-end learning.
- A comparative study of three syntax styles provides valuable references for the design of LLM-friendly Domain-Specific Languages (DSLs).

## Rating

- **Novelty**: 8/10 — The concept of formalizing generative tasks as triplet symbolic representations is refreshing, though the execution layer still relies on existing tools.
- **Experimental Thoroughness**: 7/10 — The coverage of 12 task categories, ComfyBench, and user studies is relatively comprehensive, but the evaluation scale is somewhat small (120 cases).
- **Writing Quality**: 7/10 — Formal definitions are clear but slightly verbose; the experimental visualization is not sufficiently intuitive.
- **Value**: 8/10 — Provides a new paradigm for multimodal generation, and the open-source code enhances its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](../../ECCV2024/image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[CVPR 2026\] FEAT: Fashion Editing and Try-On from Any Design](../../CVPR2026/image_generation/feat_fashion_editing_and_try-on_from_any_design.md)

</div>

<!-- RELATED:END -->
