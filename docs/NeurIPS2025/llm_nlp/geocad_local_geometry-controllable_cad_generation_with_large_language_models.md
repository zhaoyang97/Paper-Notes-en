---
title: >-
  [Paper Note] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][CAD generation] GeoCAD is proposed as the first method for locally geometry-controllable CAD generation. It introduces a complementary captioning strategy to generate geometric instructions for lo…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "CAD generation"
  - "local geometry control"
  - "large language models"
  - "text-to-CAD"
  - "complementary captioning"
date: 2026-05-08
content_hash: 785d52822904c4c3
---

# GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.10337](https://arxiv.org/abs/2506.10337)  
**Code**: [https://github.com/Zhanwei-Z/GeoCAD](https://github.com/Zhanwei-Z/GeoCAD)  
**Area**: LLM/NLP
**Keywords**: CAD generation, local geometry control, large language models, text-to-CAD, complementary captioning

## TL;DR
GeoCAD is proposed as the first method for locally geometry-controllable CAD generation. It introduces a complementary captioning strategy to generate geometric instructions for local parts and fine-tunes an LLM to enable precise modification of local CAD components according to user-defined text instructions.

## Background & Motivation
The Sketch-Extrude Modeling (SEM) paradigm is widely adopted in industrial CAD design. Users typically need to modify local loops after drafting to ensure functional or aesthetic requirements are met. If deep learning methods could automatically modify the geometry of local parts according to user-defined geometric instructions (e.g., "isosceles right triangle" or "rectangle with a trimmed corner"), the labor cost of optimizing CAD products would be substantially reduced.

Existing methods face two core challenges:

**Lack of text instruction-following capability**: Traditional controllable CAD generation methods (e.g., SkexGen, SketchGen) accept partial CAD attributes as input but cannot understand natural language instructions.

**Inability to focus on local generation**: Most existing text-to-CAD methods (e.g., CAD-GPT, Text2CAD) generate complete CAD models from scratch, making precise control over individual local parts difficult.

**Inaccurate geometric descriptions**: Some methods collect textual descriptions from a global 3D perspective, where oblique viewpoints fail to capture precise geometric attributes such as lengths and angles.

**FlexCAD supports local editing but lacks geometric constraints**: Since geometric constraints are absent from training prompts, FlexCAD struggles to follow geometric instructions at inference time.

## Method

### Overall Architecture
GeoCAD takes three inputs: (1) the original CAD model (represented in the hierarchical text format proposed by FlexCAD), (2) the target local part to be modified, and (3) user-specified geometric instructions. The output is a new CAD model in which only the target local part has been modified.

The core pipeline consists of two stages:
- **Complementary Captioning Strategy** (Sec 3.1): Generates approximately 221k geometric instructions for local parts.
- **Two-Stage LLM Fine-Tuning** (Sec 3.2): Leverages these instructions to fine-tune an LLM for locally controllable generation.

### Key Designs

#### 1. Complementary Captioning Strategy
Local loops are collected from the DeepCAD dataset; duplicate and invalid samples are filtered out. The remaining parts are divided into two categories:

**Simple parts (~105k)**: Common geometric shapes (triangles, quadrilaterals, sectors, etc.), comprising approximately 50% of the total.
- Annotated using the **Vertex-based Captioning** method.
- Vertex coordinates are extracted from the CAD text representation, and geometric attributes are analyzed for precise classification.
- Example: a quadrilateral with four equal sides is classified as a rhombus; if it also contains a right angle, it is further classified as a square.
- Key dimensional parameters (e.g., circle radius, square side length) are additionally included for some simple parts.

**Complex parts (~116k)**: Parts exhibiting more complex visual patterns.
- Annotated using the **VLLM-based Captioning** method.
- Complex parts are rendered as 2D images, and descriptive captions are generated using VLLMs such as GPT-4 / Qwen-VL.
- VLLMs are insufficiently precise for fine-grained geometric descriptions of simple parts (e.g., they cannot reliably distinguish rhombuses), hence the complementary design.

#### 2. Two-Stage LLM Fine-Tuning

**Stage 1: CAD–Text Alignment Pre-training (Optional)**
- Objective: align CAD-specific geometric representations with textual geometric instructions.
- Random data augmentation is applied to each local part: translation, scaling, rotation, and flipping.
- Augmented samples retain the same geometric instructions (e.g., a rotated right-angled trapezoid remains a right-angled trapezoid).
- The LLM is fine-tuned on instruction–answer pairs from both original and augmented samples.

**Stage 2: Geometry-Controlled Instruction Fine-Tuning**
- At each epoch, one local part is randomly masked from a given CAD model.
- The corresponding geometric instruction and the remaining visible parts are used as the prompt for the LLM to predict the masked part.
- **Key distinction from FlexCAD**: geometric instructions are explicitly included in the prompt as constraints.
- FlexCAD's training prompts lack geometric constraints, preventing the model from following geometric instructions at inference time.

### Loss & Training
- Standard cross-entropy (CE) loss computed between predicted tokens and ground-truth tokens.
- LoRA fine-tuning (rank=8, alpha=32) with most parameters frozen.
- Backbone model: Llama-3-8B.
- 8 × A100 GPUs, AdamW optimizer, batch size 32.
- Cosine annealing learning rate schedule, initial rate $5\times10^{-4}$.
- Stage 1: 10 epochs; Stage 2: 30 epochs.
- Inference: temperature $\tau=0.9$, Top-p=0.9.

## Key Experimental Results

### Main Results
1k CAD models are randomly sampled from the DeepCAD test set; one local part is randomly masked per model. Each method generates new parts using 5 simple + 5 complex geometric instructions per model, yielding 10k generated samples in total.

| Model | COV↑ | MMD↓ | JSD↓ | PV↑ | Ver-score↑ | VLLM-score↑ | Realism↑ |
|-------|------|------|------|-----|------------|-------------|----------|
| OpenAI-o3 (5-shot) | 53.6% | 1.64 | 1.49 | 65.7% | 33.6% | 22.1% | 18.7% |
| FlexCAD | 58.3% | 1.40 | 1.58 | 86.7% | 19.8% | 6.93% | 13.6% |
| FlexCAD (5-shot) | 59.4% | 1.37 | 1.34 | 88.1% | 43.5% | 26.8% | 20.2% |
| **GeoCAD** | **64.9%** | **1.13** | **0.98** | **90.5%** | **76.4%** | **65.7%** | **40.9%** |
| **GeoCAD (5-shot)** | **66.0%** | 1.16 | **0.80** | **92.3%** | **82.2%** | **68.2%** | **43.6%** |

GeoCAD substantially outperforms baselines on text–CAD consistency metrics:
- Ver-score exceeds FlexCAD by **38.7%**
- VLLM-score exceeds FlexCAD by **41.4%**
- Human evaluation Realism exceeds FlexCAD by **23.4%**

### Ablation Study

| Variant | COV↑ | MMD↓ | JSD↓ | PV↑ | Ver-score↑ | VLLM-score↑ |
|---------|------|------|------|-----|------------|-------------|
| Vertex-based captioning only | 63.6% | 1.18 | 1.02 | 89.5% | 78.3% | - |
| VLLM-based captioning only | 61.8% | 1.26 | 1.05 | 89.1% | - | 64.2% |
| w/o Stage 1 | 61.3% | 1.21 | 1.16 | 89.6% | 71.5% | 60.4% |
| w/o data augmentation | 62.9% | 1.18 | 1.09 | 88.5% | 73.2% | 61.8% |
| **Full GeoCAD** | **64.9%** | **1.13** | **0.98** | **90.5%** | **76.4%** | **65.7%** |

### Key Findings
1. **Complementary captioning is indispensable**: Vertex-based captioning alone cannot handle complex parts; VLLM-based captioning alone cannot precisely describe simple parts.
2. **Stage 1 pre-training is critical**: Removing it yields the worst performance, confirming that preliminary CAD–text alignment is necessary.
3. **Data augmentation is effective**: Removing it degrades performance, indicating that diversified augmented samples enhance alignment.
4. **GeoCAD generalizes well**: It accurately interprets and executes semantically similar unseen instructions (e.g., "narrow rounded rectangle," "right triangle").
5. **Precise dimensional control is achievable**: Parameters such as circle radius, square side length, and rectangle aspect ratio can be accurately controlled.

## Highlights & Insights
1. **First work to propose locally geometry-controllable CAD generation**, filling a gap in the field.
2. **The complementary captioning strategy is elegantly designed**: Vertex-based captioning enables precise classification of simple parts, while VLLM-based captioning handles visual pattern description of complex parts — the two approaches are mutually complementary.
3. Local CAD editing is formulated as a **mask-then-predict** task, effectively leveraging the LLM's in-context completion capability.
4. The annotation corpus reaches 221k samples, providing a referenceable data construction paradigm for future work.
5. The simple addition of geometric constraints to the prompt yields dramatic performance gains over FlexCAD.

## Limitations & Future Work
1. **Restricted to the SEM paradigm**: The method is not extended to other CAD representations such as CSG or B-rep.
2. **Dependence on the DeepCAD dataset**: The scale and diversity of the data may limit generalization.
3. **VLLM annotation quality is bounded**: The quality of captions for complex parts depends on the visual understanding capability of the underlying VLLM.
4. **Inference efficiency is not discussed**: The computational overhead of LLM inference is high; real-time applicability in industrial settings remains to be verified.
5. **Control is limited to the 2D loop level**: Finer-grained control (e.g., individual edges or vertices) has not been explored.
6. **Integration with 3D visual models**: Directly specifying geometric constraints from 3D viewpoints could offer a more intuitive interface.

## Related Work & Insights
- **FlexCAD**: The most direct predecessor; GeoCAD extends it by incorporating geometric constraints.
- **Text2CAD / CAD-GPT**: Generate complete CAD models from scratch; GeoCAD focuses on local editing.
- **LoRA fine-tuning paradigm**: Enables efficient domain adaptation while preserving the benefits of large-scale pre-training.
- Inspiration: The notion of "locally controllable editing" may be generalized to other structured generation tasks (e.g., local code modification, local molecular structure optimization).

## Rating
- Novelty: ⭐⭐⭐⭐ (First to propose locally geometry-controllable CAD generation; complementary captioning strategy is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive quantitative, qualitative, and ablation results; efficiency analysis is lacking)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; rich figures and tables)
- Value: ⭐⭐⭐⭐ (Fills a gap in the field; strong industrial application prospects)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](../../ACL2026/llm_nlp/how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)

</div>

<!-- RELATED:END -->
