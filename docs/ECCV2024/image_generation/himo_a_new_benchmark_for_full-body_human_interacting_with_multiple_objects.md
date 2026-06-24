---
title: >-
  [Paper Note] HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects
description: >-
  [ECCV 2024][Image Generation][Human-Object Interaction] This work proposes HIMO, the first large-scale full-body human-multi-object interaction 4D MoCap dataset (3.3K sequences, 4.08M frames) accompanied by detailed textual descriptions and temporal segment annotations. It also presents a dual-branch conditional diffusion model and an autoregressive pipeline to generate coordinated multi-object interaction motion sequences.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Human-Object Interaction"
  - "Multi-Object"
  - "Motion Generation"
  - "Diffusion Model"
  - "Temporal Segmentation"
date: 2026-05-08
content_hash: 4633cf6161703428
---

# HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects

**Conference**: ECCV 2024  
**arXiv**: [2407.12371](https://arxiv.org/abs/2407.12371)  
**Code**: [Project Page](https://lvxintao.github.io/himo)  
**Area**: Image Generation  
**Keywords**: Human-Object Interaction, Multi-Object, Motion Generation, Diffusion Model, Temporal Segmentation

## TL;DR

This work proposes HIMO, the first large-scale full-body human-multi-object interaction 4D MoCap dataset (3.3K sequences, 4.08M frames) accompanied by detailed textual descriptions and temporal segment annotations. It also presents a dual-branch conditional diffusion model and an autoregressive pipeline to generate coordinated multi-object interaction motion sequences.

## Background & Motivation

Generating human-object interactions (HOI) is crucial for digital humans, gaming, AR/VR, robotics, and embodied AI. However, existing datasets and models are almost entirely limited to **single-object interactions**, ignoring multi-object manipulation scenarios that are ubiquitous in daily life.

**Limitations of Prior Work**:

| Dataset | Hands | Multi-Object | Text | Temporal Segmentation | Sequences | Frames |
|--------|------|--------|------|----------|--------|------|
| GRAB | ✓ | ✗ | ✗ | ✗ | 1,334 | - |
| BEHAVE | ✗ | ✗ | ✗ | ✗ | 321 | 15K |
| InterCap | ✗ | ✗ | ✗ | ✗ | 223 | 67K |
| ARCTIC | ✓ | ✗ | ✗ | ✗ | 339 | 2.1M |
| CHAIRS | ✓ | ✗ | ✗ | ✗ | 1,390 | - |
| **HIMO** | ✓ | ✓ | ✓ | ✓ | 3,376 | 4.08M |

Key gaps:

**No multi-object interaction data**: All existing datasets only involve a single object.

**Lack of textual annotations**: Most HOI datasets lack textual descriptions, limiting research on text-driven generation.

**Lack of temporal segmentation**: Long interaction sequences are not decomposed into atomic actions, making fine-grained timeline control impossible.

HIMO aims to simultaneously address all three gaps.

## Method

### Overall Architecture

This work consists of two main parts:
1. **HIMO Dataset Construction**: Hybrid capture system + SMPL-X fitting + text/temporal annotations.
2. **HOI Synthesis Model**: Dual-branch diffusion model + mutual attention module + autoregressive composition.

Two new tasks are proposed:
- **HIMO-Gen**: Given a complete textual description, generate a complete multi-object interaction sequence.
- **HIMO-SegGen**: Given segmented language descriptions (timeline control), generate composable interaction sequences.

### Key Designs

#### 1. Hybrid Motion Capture System

**Body Capture**:
- Optitrack optical motion capture system with 20 PrimeX-22 infrared cameras.
- Each subject wears a motion capture suit with 41 reflective markers.
- 2K resolution, 120fps.

**Finger Capture**:
- Noitom PNS inertial gloves to accurately record fine finger poses.
- The inertial approach avoids hand-object and object-object self-occlusions.
- Frequent recalibration ensures capture quality.

**Spatio-temporal Alignment**:
- Space: A positioning plate on the back of the hand provides wrist rotation information, integrating full-body and finger movements.
- Time: Tentacle Sync devices provide timecode synchronization for both Optitrack and PNS.

**Object Design**:
- 53 common household objects selected from ContactDB and Sketchfab.
- 3D-printed after rescaling to ensure the mesh geometry perfectly matches the physical objects.
- Each object is stickered with 3-6 12.5mm reflective markers for rigid body tracking.
- Key principle: Marker placement should minimize side effects on operations.

**Capture Scale**:
- 34 subjects.
- Predefined combinations of 2 or 3 objects.
- Each interaction task is executed 3 times (with different manipulation styles).
- Total: 3,376 sequences, 9.44 hours, 4.08M frames.
- Among them, 2,486 sequences involve 2 objects, and 890 sequences involve 3 objects.

#### 2. SMPL-X Parameter Fitting

The SMPL-X parameterized human body model is adopted, where parameters include:
- Global orientation $g \in \mathbb{R}^3$
- Body pose $\theta_b \in \mathbb{R}^{21 \times 3}$
- Finger pose $\theta_h \in \mathbb{R}^{30 \times 3}$
- Root translation $t \in \mathbb{R}^3$
- Body shape parameters $\beta \in \mathbb{R}^{10}$

Optimization objective:
$$\mathbb{E} = \alpha \mathbb{E}_j + \lambda \mathbb{E}_s + \gamma \mathbb{E}_r$$

- $\mathbb{E}_j$: Joint position matching loss (L2 distance between MoCap joints and SMPL-X regressed joints)
- $\mathbb{E}_s$: Smoothness term (L2 norm of joint position differences between adjacent frames)
- $\mathbb{E}_r$: Regularization term ($\|\theta_b\|_2^2 + \|\theta_h\|_2^2$)
- Weights: $\alpha=1, \lambda=0.1, \gamma=0.01$

#### 3. Textual and Temporal Annotation

**Textual Annotation Guidelines**:
- Describe the entire interaction process, emphasizing the order of operations ("First...then...finally...").
- Describe interaction modes (e.g., "picking up", "rotating", "pouring").
- Label the participating body parts (left hand/right hand).

**Temporal Segmentation Guidelines**:
- Decompose long sequences into fine-grained atomic actions.
- E.g., "pouring tea from a teapot into a cup" $\rightarrow$ "picking up the teapot" + "pouring tea into the cup" + "putting down the teapot".
- HOI sequences and corresponding texts are simultaneously segmented into equal parts.
- Average of 2.60 segments per sequence.

#### 4. HIMO-Gen: Dual-Branch Diffusion Model

**Data Representation**:
- Human motion $\boldsymbol{H} \in \mathbb{R}^{T \times D_h}$: Global joint positions $P^i \in \mathbb{R}^{52 \times 3}$, 6D continuous rotations $Q^i \in \mathbb{R}^{52 \times 6}$, translation $t^i \in \mathbb{R}^3$.
- Multi-object motion $\boldsymbol{O} \in \mathbb{R}^{T \times D_o}$: Relative rotations $\mathbf{R}^i \in \mathbb{R}^6$, global translations $\mathbf{T}^i \in \mathbb{R}^3$.
- Object geometry $\boldsymbol{G} \in \mathbb{R}^{1024 \times 3}$: Represented using BPS, calculating direction vectors from 1024 sampled points to the base point set.

**Model Architecture**:
- **Human Branch**: Input = Noisy human motion $\boldsymbol{H}_t$ + initial state mask $\boldsymbol{H}_0$ + text embedding + timestep embedding.
- **Object Branch**: Input = Noisy object motion + initial object state mask + object geometry embedding + text embedding.
- **Text Encoder**: CLIP encoder + linear layer.
- **Denoising Network**: $\ell_{enc} = 8$-layer Transformer with 4-head attention.

**Mutual Attention Module**:
Each Transformer block contains two attention layers and one feed-forward layer:
1. Self-attention layer: Embeds human/object features separately.
2. Mutual attention layer: In the human branch, K and V come from object features, while in the object branch, K and V come from human features.

$$\boldsymbol{H}^{(i+1)} = FF(\text{softmax}(\frac{\mathbf{Q}_h \mathbf{K}_o^T}{\sqrt{C}} \mathbf{V}_o))$$
$$\boldsymbol{O}^{(i+1)} = FF(\text{softmax}(\frac{\mathbf{Q}_o \mathbf{K}_h^T}{\sqrt{C}} \mathbf{V}_h))$$

**Object Pairwise Distance Loss** ($\mathcal{L}_{dis}$):
- Explicitly constrains relative distance patterns between interacting objects.
- For instance, when cutting an apple, the knife gradually approaches the apple's surface, maintains position for a period, and then leaves.
- Uses L2 loss to maintain consistency between the generated and ground truth object distances.

### Loss & Training

**Total Loss**:
$$\mathcal{L} = \lambda_{vel}\mathcal{L}_{vel} + \lambda_{pos}\mathcal{L_{pos}} + \lambda_{pen}\mathcal{L}_{pen} + \lambda_{dis}\mathcal{L}_{dis}$$

- $\mathcal{L}_{vel}$: Joint velocity loss
- $\mathcal{L}_{pos}$: Joint position loss
- $\mathcal{L}_{pen}$: Human-object penetration loss
- $\mathcal{L}_{dis}$: Object pairwise distance loss
- Weights: $\lambda_{vel} = \lambda_{pos} = \lambda_{pen} = 1$, $\lambda_{dis} = 0.1$

**Training Configurations**:
- Adam optimizer, learning rate 0.0001, weight decay 0.99.
- Maximum motion length of 300 frames for HIMO-Gen and 100 frames for HIMO-SegGen.
- Maximum text length of 40 and 15, respectively.
- Training time: approx. 9 hours for HIMO-Gen, approx. 12 hours for HIMO-SegGen (on a single A100 GPU with a batch size of 128).
- Data splits: Training/Testing/Validation = 0.8/0.15/0.05.

#### 5. HIMO-SegGen: Autoregressive Generation Pipeline

- Train the HIMO-Gen model on short sequence-text pairs after temporal segmenting.
- Modify the conditional input to be the **past few frames** (rather than only the first frame).
- Inference: Take the last few frames of the previously generated segment as condition for the next segment, iteratively generating.

Experiments find that using 10 conditional frames yields the best results.

## Key Experimental Results

### Main Results

**Quantitative Comparison on 2-Objects Split**:

| Method | R-Precision(Top3) ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|------|---------------------|-------|-----------|-------------|-------------|
| Real | 0.7988 | 0.0176 | 3.5659 | 11.3973 | - |
| IMoS | 0.5013 | 7.5890 | 8.7402 | 7.0033 | 0.9920 |
| MDM | 0.6052 | 6.8457 | 8.0187 | **11.3891** | 1.2880 |
| priorMDM | 0.5891 | 7.8517 | 7.2509 | 12.5799 | 1.5911 |
| **HIMO-Gen** | 0.6369 | **1.4811** | 3.6491 | 11.6603 | 1.7863 |
| **HIMO-SegGen** | **0.6404** | 4.2004 | **3.6077** | 15.7317 | **2.0495** |

**Quantitative Comparison on 3-Objects Split**:

| Method | R-Precision(Top3) ↑ | FID ↓ | MM-Dist ↓ |
|------|---------------------|-------|-----------|
| IMoS | 0.4662 | 4.9902 | 7.7702 |
| MDM | 0.5025 | **4.5713** | 6.3144 |
| priorMDM | 0.5137 | 4.8210 | 5.8900 |
| **HIMO-Gen** | **0.5350** | 4.7712 | **5.0866** |

### Ablation Study

Ablations of the mutual attention module and object distance loss are in the supplementary materials. Based on the paper's discussion:

| Design Choice | Effect |
|---------|------|
| Without mutual attention | Spatial-temporal misalignment of human and object motions |
| Without object distance loss | Unrealistic contact between objects |
| Number of conditional frames = 10 | Optimal transition smoothness (vs 5/15/20) |
| Randomly shuffling object order | Mitigate the influence of object input sequence |

### Key Findings

1. **HIMO-Gen significantly leads in FID**: On 2-objects, the FID is only 1.48, which is 78% lower than the second-best, MDM (6.85).
2. **HIMO-SegGen improves text alignment**: R-Precision and MM-Dist are superior to HIMO-Gen, thanks to more refined condition descriptions.
3. **Mutual attention module is key**: Guarantees coordination of human and object motions.
4. **Generalization capability**: The model can be applied to unseen object geometries and can synthesize new HOI combinations not present in the training set.
5. **3-objects is more challenging**: Performance of all methods drops on 3-objects, narrowing down the FID gap.

## Highlights & Insights

1. **Fills a crucial data gap**: The first 4D HOI dataset featuring multi-object, full-body, text, and temporal segmentation, with each dimension contributing uniquely.
2. **Innovative value of temporal segmentation annotations**: Allows complex actions to be decomposed into atomic actions, supporting flexible timeline control and motion composition.
3. **Well-designed hybrid motion capture system**: An optical (body accuracy) + inertial (finger anti-occlusion) combination scheme balances accuracy and feasibility.
4. **Object pairwise distance loss**: Cleverly uses simple L2 constraints to model the distance patterns between objects, effectively reducing penetration.
5. **Simple and efficient autoregressive composition pipeline**: Smooth transitions can be achieved by simply modifying the conditional input to include the past few frames.

## Limitations & Future Work

1. **Relatively limited dataset scale**: 3.3K sequences is still small compared to image/video domains.
2. **Restricted object categories**: 53 items, mainly concentrated in dining and kitchen scenarios.
3. **2-3 objects constraint**: Real-world scenarios may involve simultaneous manipulation of more objects.
4. **SMPL-X fitting accuracy**: Fitting errors are inevitable when mapping MoCap data to parameterized models.
5. **Autoregressive error accumulation**: Segment-by-segment generation in long sequences may lead to drift.
6. **Lack of hand-object contact modeling**: Although there is a penetration loss, precise contact force modeling is missing.

## Related Work & Insights

- **GRAB / ARCTIC**: Pioneers of full-body HOI datasets, but limited to a single object.
- **MDM (Motion Diffusion Model)**: The source of the baseline framework in this paper.
- **priorMDM**: A human-human interaction generation method, which inspired the dual-branch design.
- **GMD**: Controllable motion generation conditioned on keyframes.
- **Insights**: The temporal segmentation + autoregressive paradigm can be extended to more complex multi-agent interaction scenarios.

## Rating

- **Novelty**: ★★★★★ — The dataset fills an important gap, and the multi-object + temporal segmentation setup is brand new.
- **Practicality**: ★★★★☆ — Direct application value for gaming/VR/robotics, but dataset scale still needs expansion.
- **Experimental Thoroughness**: ★★★★☆ — Comprehensive comparison with multiple baselines + generalization experiments + ablation studies, following established evaluation protocols.
- **Writing Quality**: ★★★★☆ — Clear structure and comprehensive system details, though the method section is slightly wordy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](../../CVPR2025/image_generation/visual_persona_foundation_model_for_full-body_human_customization.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)

</div>

<!-- RELATED:END -->
