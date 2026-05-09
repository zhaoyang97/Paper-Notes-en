---
title: >-
  [Paper Note] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs
description: >-
  [NeurIPS 2025 (Spotlight)][Multimodal VLM][Multi-view understanding] This paper proposes E3VQA, the first multi-view VQA benchmark, and M3CoT, a prompting technique that fuses three complementary scene graphs, to enhance multi-view scene understanding in Large Vision-Language Models (LVLMs), achieving gains of 4.84% on GPT-4o and 5.94% on Gemini 2.0 Flash.
tags:
  - NeurIPS 2025 (Spotlight)
  - Multimodal VLM
  - Multi-view understanding
  - egocentric view
  - exocentric view
  - scene graph
  - VQA
  - CoT
date: 2026-05-08
content_hash: 79e7e443061b1362
---

# Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2505.21955](https://arxiv.org/abs/2505.21955)
**Code**: Available ([https://github.com/Leeinsu1/Towards-Comprehensive-Scene-Understanding](https://github.com/Leeinsu1/Towards-Comprehensive-Scene-Understanding))
**Area**: Multimodal VLM / Scene Understanding
**Keywords**: Multi-view understanding, egocentric view, exocentric view, scene graph, VQA, CoT

## TL;DR

This paper proposes E3VQA, the first multi-view VQA benchmark, and M3CoT, a prompting technique that fuses three complementary scene graphs, to enhance multi-view scene understanding in Large Vision-Language Models (LVLMs), achieving gains of 4.84% on GPT-4o and 5.94% on Gemini 2.0 Flash.

## Background & Motivation

### Deployment of LVLMs in Interactive Applications
Large vision-language models are increasingly deployed in interactive applications such as virtual/augmented reality, where the first-person (egocentric) view provided by head-mounted cameras serves as a key input. However, egocentric views carry inherent limitations:

**Narrow field of view**: Head-mounted cameras have a limited FOV and cannot capture the full scene.

**Lack of global context**: Only the local region the user is attending to is visible.

**Difficulty in spatial reasoning**: Answering questions requiring global spatial information is challenging.

### Complementary Value of the Third-Person View
The third-person (exocentric) view can provide:
- Global scene layout
- Complete object visibility
- A global view of spatial relationships
- Contextual information about user–environment interaction

### Core Problem
How can first- and third-person views be effectively fused to enable LVLMs to understand scenes more comprehensively? Existing work almost exclusively relies on a single viewpoint.

## Method

### Overall Architecture

```
Input: Synchronized ego-exo image pairs
  ├── E3VQA Benchmark: 4K high-quality QA pairs
  └── M3CoT Reasoning: Three-view scene graph fusion
       ├── Ego Scene Graph
       ├── Exo Scene Graph
       └── Cross-view Scene Graph
       → Unified scene representation → LVLM answer
```

### Key Designs

#### 1. E3VQA Benchmark

**Data Construction**:
- Synchronized ego-exo image pairs sourced from the Ego-Exo4D dataset
- 4,000 high-quality question–answer pairs covering multiple question types:
    - **Spatial reasoning**: Where is the object? What are the relative positions?
    - **Action understanding**: What is the user doing?
    - **Causal reasoning**: Why is the user attending to this object?
    - **Counting and attributes**: How many instances of a specific object are in the scene?
- **Quality control**: Manual annotation with multi-round verification

**Question Type Distribution**:
- Answerable from ego view only: ~30%
- Answerable from exo view only: ~25%
- Requiring both views for correct answer: ~45%

#### 2. M3CoT (Multi-view Multi-modal Chain-of-Thought)

M3CoT is a training-free prompting technique whose core contribution is constructing a unified multi-view scene representation:

**Step 1: Generate Three Scene Graphs**

- **Ego Scene Graph $G_e$**: Extracted from the egocentric image
    - Nodes: detected objects
    - Edges: spatial relations (left/right/above/below/front/behind) + interaction relations

- **Exo Scene Graph $G_x$**: Extracted from the third-person image
    - Nodes: globally visible objects
    - Edges: global spatial relations

- **Cross-view Scene Graph $G_c$**: Cross-view associations
    - Nodes: objects co-visible in both views
    - Edges: cross-view correspondences and complementary information

**Step 2: Fuse into a Unified Representation**

The three scene graphs are merged into a single textualized unified scene description, which is provided as additional context to the LVLM.

**Step 3: Chain-of-Thought Reasoning**

The fused scene representation guides the LVLM through multi-step reasoning:
```
[Ego Image] + [Exo Image] + [Unified Scene Representation]
→ Step 1: Understand the information provided by each view
→ Step 2: Identify the complementary relationship between the two views
→ Step 3: Synthesize and answer the question
```

### Loss & Training

M3CoT is a **completely training-free** method:
- It leverages the in-context learning capability of LVLMs
- Implemented solely through carefully designed prompts
- Applicable to any LVLM that supports multi-image input

## Key Experimental Results

### Main Results

Performance of different LVLMs on the E3VQA benchmark:

| Model | Ego Only | Exo Only | Ego+Exo (no CoT) | +Standard CoT | +M3CoT (Ours) |
|-------|----------|----------|------------------|---------------|----------------|
| GPT-4o | 52.3 | 48.7 | 58.4 | 61.2 | **66.0 (+4.84)** |
| Gemini 2.0 Flash | 49.8 | 46.2 | 55.1 | 58.3 | **64.2 (+5.94)** |
| Claude 3.5 Sonnet | 50.1 | 47.3 | 56.8 | 59.4 | **63.8 (+4.42)** |
| LLaVA-1.5-13B | 38.2 | 35.6 | 42.1 | 44.7 | **48.3 (+3.58)** |
| InternVL2-8B | 41.5 | 38.9 | 45.3 | 47.6 | **51.2 (+3.61)** |

Performance gain by question type (GPT-4o):

| Question Type | Ego+Exo Baseline | +M3CoT | Gain |
|---------------|------------------|--------|------|
| Spatial reasoning | 55.2 | **63.8** | +8.6 |
| Action understanding | 62.1 | **67.4** | +5.3 |
| Causal reasoning | 54.8 | **62.1** | +7.3 |
| Counting and attributes | 61.5 | **64.7** | +3.2 |
| Ego-only questions | 68.3 | **70.1** | +1.8 |
| Two-view questions | 48.6 | **58.2** | +9.6 |

### Ablation Study

Contribution of each M3CoT component (GPT-4o):

| Configuration | Accuracy | Gain |
|---------------|----------|------|
| Baseline (Ego+Exo, no CoT) | 58.4 | — |
| + Ego Scene Graph only | 61.5 | +3.1 |
| + Exo Scene Graph only | 60.8 | +2.4 |
| + Cross-view Scene Graph only | 62.3 | +3.9 |
| + Ego + Exo Scene Graphs | 63.7 | +5.3 |
| **+ Full M3CoT (all three)** | **66.0** | **+7.6** |

### Key Findings

1. **Multi-view significantly outperforms single-view**: Ego+Exo exceeds Ego-only by ~6%, and M3CoT yields an additional gain of 4–6%.
2. **Cross-view scene graph contributes the most**: Cross-view association is more valuable than either single-view scene graph alone.
3. **Spatial reasoning benefits the most**: Questions requiring global spatial information gain the most from multi-view fusion (+8.6%).
4. **Two-view questions see the largest improvement**: M3CoT achieves a gain of up to 9.6% on this question category.
5. **Closed-source models benefit more**: GPT-4o and Gemini obtain larger gains from M3CoT than open-source models.

## Highlights & Insights

1. **First multi-view VQA benchmark**: E3VQA fills the gap in evaluation of joint ego-exo scene understanding.
2. **Training-free improvement**: M3CoT requires no additional training and is plug-and-play.
3. **Spotlight acceptance**: Reflects high-quality problem formulation and systematic evaluation.
4. **Practical applicability**: Directly applicable to intelligent assistants in AR/VR scenarios.
5. **Revealing LVLM limitations**: Systematic evaluation exposes shortcomings of existing LVLMs in multi-view reasoning.

## Limitations & Future Work

1. **Limited dataset scale**: 4K QA pairs are relatively small and may not cover all scene types.
2. **Scene graph extraction quality**: The effectiveness of M3CoT depends on the accuracy of scene graph extraction.
3. **Two-image constraint**: Real AR/VR scenarios may involve more than two viewpoints.
4. **Static images only**: Temporal information from video sequences is not considered.
5. **Computational overhead**: M3CoT requires an additional scene graph extraction step, increasing inference latency.

## Related Work & Insights

- **Ego-Exo4D**: Provides synchronized ego-exo video data, serving as the data foundation for E3VQA.
- **Visual Question Answering (VQA)**: Classic single-view VQA benchmarks such as VQAv2 and GQA.
- **Scene graph generation**: Datasets such as Visual Genome have advanced scene graph generation techniques.
- **Multi-view understanding**: Classic multi-view learning methods; this paper is the first to integrate them with LVLMs.
- **Future directions**: Video-level multi-view reasoning, fusion of more than two views, and end-to-end training approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First ego-exo joint VQA benchmark + training-free multi-view fusion method
- **Theoretical Depth**: ⭐⭐⭐ — Primary contributions are methodological and benchmark-oriented
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple models, multiple question types, thorough ablation
- **Practical Impact**: ⭐⭐⭐⭐⭐ — Direct benefit for AR/VR applications
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich visualizations

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[NeurIPS 2025\] Visual Structures Help Visual Reasoning: Addressing the Binding Problem in LVLMs](visual_structures_helps_visual_reasoning_addressing_the_binding_problem_in_vlms.md)

</div>

<!-- RELATED:END -->
