---
title: >-
  [Paper Note] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail
description: >-
  [ICCV 2025][3D Vision][3D scene understanding] This paper proposes ExCap3D, a method for generating multi-granularity captions for objects in 3D indoor scenes at both object-level and part-level description layers. Throu…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D scene understanding"
  - "dense annotation"
  - "multi-granularity captioning"
  - "joint object-part generation"
  - "3D Gaussian"
date: 2026-05-08
content_hash: 7528483a411cfbd8
---

# ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail

**Conference**: ICCV 2025
**arXiv**: [2503.17044](https://arxiv.org/abs/2503.17044)  
**Code**: Coming soon  
**Area**: 3D Vision
**Keywords**: 3D scene understanding, dense annotation, multi-granularity captioning, joint object-part generation, 3D Gaussian

## TL;DR

This paper proposes ExCap3D, a method for generating multi-granularity captions for objects in 3D indoor scenes at both object-level and part-level description layers. Through part-to-object information sharing and semantic/textual consistency losses, the approach ensures caption accuracy and coherence. On a newly constructed dataset of 190K captions, CIDEr scores improve by 17% and 124% over the prior SOTA at the object and part levels, respectively.

## Background & Motivation

3D indoor scene understanding is fundamental to AR/VR and robotics applications. Natural language descriptions can encode complex information and support more natural human–scene interaction. However, existing 3D annotation methods suffer from critical limitations:

**Single-granularity captioning**: Methods such as Scan2Cap and ScanQA describe objects at only one level of detail, focusing primarily on spatial relationships between objects.

**Lack of part-level detail**: These methods cannot describe the appearance, material, or functional properties of individual object parts.

**Different applications require different granularities**: Robot navigation may only require "recliner," whereas an assistive AI needs "a recliner with a high backrest, wooden armrests, a soft seat cushion, and an adjustable footrest."

**Core contribution**: The paper introduces the task of Expressive 3D Captioning—generating, for each detected object, an object-level caption (semantic category and appearance) and a part-level caption (material, color, and function of each part).

## Method

### Overall Architecture (Fig. 3)

1. **3D instance segmentation**: Mask3D is used to detect objects in the scene.
2. **Joint caption generation**: Two independent captioning heads generate object-level and part-level descriptions.
3. **Consistency constraints**: Semantic and textual consistency losses enforce coherence between the two captioning levels.

### 3D Instance Segmentation

Given a 3D scene mesh $M=(\mathcal{V}, \mathcal{F})$, the input is voxelized and fed into Mask3D:
- A 3D sparse convolutional UNet encoder extracts dense features $F \in \mathbb{R}^{N_{vox} \times D}$.
- A parallel mask module iteratively refines query vectors $Q \in \mathbb{R}^{N_q \times D_q}$ via transformer encoder layers.
- Output: refined instance-aware queries $Q_r$ and instance masks $I \in \{0,1\}^{N_q \times N_{vox}}$.

### Joint Captioning and Information Sharing

Two transformer language models $\Psi_{obj}$ and $\Psi_{part}$ perform autoregressive token prediction, conditioned on two information sources:

**1. Caption-aware queries**: Refined queries are linearly projected into captioning initialization tokens:

$$Q_{c,o} = \Phi_{query}(Q_{r,o})$$

**2. Segment-level context features**: Cross-attention layers attend to 3D features corresponding to each object. To reduce computational complexity, features are aggregated within pre-computed segments on mesh faces, yielding $S_o \in \mathbb{R}^{n_{s,o} \times D_{caption}}$.

**Part-to-object information sharing** (key design): Part-level captions are generated first; the final-layer hidden states of $\Psi_{part}$ are linearly projected and concatenated with the object captioner's context features:

$$H_{part,o} = \Phi_{hidden}([h_{part,1} \ldots h_{part,i}])$$

Object-level captioning is then conditioned on $Q_{c,o}$ and $[H_{part,o}; S_{obj,o}]$.

### Semantic and Textual Consistency Losses

**Semantic consistency**: Hidden states from both levels are projected to a low-dimensional space and classified into $N_{sem}$ fine-grained categories, constrained by a symmetric cross-entropy loss:

$$\mathcal{L}_{semantic} = CE(sem_{obj}, SG(sem_{part})) + CE(sem_{part}, SG(sem_{obj}))$$

where $SG$ denotes the stop-gradient operator.

**Textual consistency**: Hidden state sequences are aggregated into single vectors, and the distance between the two captioning levels is minimized:

$$\mathcal{L}_{textual} = d(\bar{h}_{obj,text}, \bar{h}_{part,text})$$

### Total Loss

$$\mathcal{L} = w_1 \mathcal{L}_{caption} + w_2 \mathcal{L}_{semantic} + w_3 \mathcal{L}_{textual}$$

where $w_1=1,\ w_2=w_3=0.1$.

## ExCap3D Dataset

### Construction Pipeline

Built upon 947 scenes from ScanNet++, covering 34K objects, using an automated pipeline:

1. **Object-level**: Project 3D GT instance annotations onto DSLR images → crop object regions → generate multi-view descriptions with a VLM (LLaVA 1.6-7B) → aggregate with an LLM (Llama 3.1-8B).
2. **Part-level**: Generate part pseudo-masks with MaskClustering + SAM → describe each part with a VLM → aggregate into a unified part-level caption with an LLM.

### Dataset Statistics (Table 1)

| Dataset | # Captions | # Categories | # Objects | Granularity |
|---------|-----------|--------------|-----------|-------------|
| Scan2Cap | 46k | 265 | 9.9k | Scene + Object |
| ScanQA | 35k | 370 | 9.5k | Scene + Object |
| **ExCap3D** | **190k** | **2k** | **34.7k** | **Object + Part** |

The dataset contains more than four times as many captions as the largest existing dataset, with category coverage reaching 2,000 classes.

## Key Experimental Results

### Main Results: Comparison with SOTA (Table 2)

**Object-level captioning**:

| Method | CIDEr↑ | ROUGE↑ | METEOR↑ |
|--------|--------|--------|---------|
| D3Net | 6.7 | 5.4 | 6.7 |
| Vote2Cap-DETR | 13.3 | 12.9 | 17.2 |
| PQ3D | 27.9 | 11.6 | 12.5 |
| **ExCap3D** | **32.7** | **16.6** | **17.9** |

**Part-level captioning**:

| Method | CIDEr↑ | ROUGE↑ | METEOR↑ |
|--------|--------|--------|---------|
| D3Net | 10.5 | 7.9 | 7.9 |
| Vote2Cap-DETR | 13.3 | 20.7 | 22.7 |
| PQ3D | 14.4 | 16.3 | 15.6 |
| **ExCap3D** | **32.3** | **21.7** | **20.8** |

CIDEr improvements: +17% at the object level and +124% at the part level (both vs. PQ3D). The substantial gain at the part level demonstrates that existing methods are fundamentally ill-equipped to handle fine-grained captioning.

### Ablation Study (Table 3)

| Method | Object CIDEr | Part CIDEr |
|--------|-------------|-----------|
| Independent models (baseline) | 29.8 | 18.7 |
| + Semantic consistency | 30.2 | **24.8** |
| + Textual consistency | **32.2** | 19.6 |
| + Part→Object information sharing | **34.8** | 25.4 |
| **Full model** | 32.7 | **32.3** |

Key findings:
- Semantic consistency primarily benefits part-level captioning (18.7→24.8), while textual consistency primarily benefits object-level captioning (29.8→32.2)—demonstrating complementary effects.
- Part-to-object information sharing improves both levels, with a particularly large gain at the object level (29.8→34.8).
- When all components are combined, the part-level improvement is most pronounced (18.7→32.3, +73%).

### Comparison of Information Sharing Directions (Table 4)

| Direction | Object CIDEr | Part CIDEr |
|-----------|-------------|-----------|
| Object→Part | 32.8 | 15.6 |
| **Part→Object** | **32.7** | **32.3** |

Modeling objects as the sum of their parts is substantially superior to treating parts as derived from the object—object-level captions do not contain fine-grained part information and thus cannot effectively guide part caption generation.

### Context Feature Ablation (Table 5)

| Method | Object CIDEr | Part CIDEr |
|--------|-------------|-----------|
| Without context features | 33.7 | 27.0 |
| With context features | 32.7 | **32.3** |

Segment-level context features are critical for part-level captioning (27.0→32.3), as describing low-level part details requires finer-grained features.

## Highlights & Insights

1. **"Object as the sum of its parts" modeling philosophy**: The information flow of first describing parts and then synthesizing object-level descriptions yields a clear advantage, consistent with the "bottom-up" object perception pattern in human cognition.
2. **Complementarity of consistency losses**: Semantic consistency ensures that both levels refer to the same semantic entity, while textual consistency ensures overlap in descriptive content—each constraining consistency from a distinct dimension.
3. **Scalability of the VLM pipeline**: Leveraging VLM + LLM to automatically generate 190K high-quality annotations avoids the bottleneck of manual annotation, and the approach can be extended to other 3D datasets.
4. **End-to-end learning outperforms a disjoint pipeline** (Table 6): Even when using the same VLM, end-to-end trained captioning quality substantially surpasses a two-stage detect-then-describe approach.

## Limitations & Future Work

1. Two independent captioning heads share information via cross-attention, which may limit the thoroughness of information transfer.
2. The sparse convolutional backbone operates at a voxel resolution of approximately 2 cm, constraining captioning ability for small or thin objects.
3. Part pseudo-masks derived from MaskClustering + SAM are lower quality than manual annotations and may introduce noise.
4. The dataset is built on ScanNet++; generalization to other 3D scanning formats remains to be validated.

## Related Work & Insights

- Unlike Cap3D (Luo et al., 2023), which captions isolated 3D objects, ExCap3D jointly detects and generates multi-granularity captions within complete 3D scenes.
- The multi-granularity captioning paradigm can be combined with 3D vision–language alignment methods (e.g., 3D-VISTA) to enable finer-grained embodied understanding.
- The part-to-object information sharing paradigm is generalizable to other hierarchical generation tasks, such as scene graph generation.
- The ExCap3D dataset can serve as foundational training data supporting fine-grained instruction following in 3D embodied AI.

## Rating ⭐⭐⭐⭐

Novelty ★★★★☆: The definition of the multi-granularity captioning task and the information sharing mechanism are original.
Experimental Thoroughness ★★★★☆: Ablation studies are comprehensive, with clear attribution of each component's contribution.
Writing Quality ★★★★☆: Figures and tables are clear; the method is described systematically.
Value ★★★★☆: The 190K dataset is highly valuable, though it relies on the high-resolution DSLR imagery of ScanNet++.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[ICCV 2025\] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)
- [\[ICCV 2025\] Predict-Optimize-Distill: A Self-Improving Cycle for 4D Object Understanding](predict-optimize-distill_a_self-improving_cycle_for_4d_object_understanding.md)

</div>

<!-- RELATED:END -->
