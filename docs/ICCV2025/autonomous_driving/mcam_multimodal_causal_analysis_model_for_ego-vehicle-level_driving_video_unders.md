---
title: >-
  [Paper Note] MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding
description: >-
  [ICCV 2025][Autonomous Driving][driving video understanding] This paper proposes MCAM, which constructs a causal structure between visual and language modalities via a Driving State Directed Acyclic Graph (DSDAG)…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "driving video understanding"
  - "causal analysis"
  - "directed acyclic graph"
  - "vision-language"
  - "ego-vehicle behavior understanding"
date: 2026-05-08
content_hash: 47042448d8c95332
---

# MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding

**Conference**: ICCV 2025
**arXiv**: [2507.06072](https://arxiv.org/abs/2507.06072)
**Code**: [GitHub](https://github.com/SixCorePeach/MCAM)
**Area**: Autonomous Driving
**Keywords**: driving video understanding, causal analysis, directed acyclic graph, vision-language, ego-vehicle behavior understanding

## TL;DR

This paper proposes MCAM, which constructs a causal structure between visual and language modalities via a Driving State Directed Acyclic Graph (DSDAG), combined with a multi-level feature extractor and a causal analysis module, for behavior description and causal reasoning in ego-vehicle-level driving video understanding.

## Background & Motivation

Driving video understanding in autonomous driving aims to translate driving behaviors from visual data into textual narratives—not only describing *what* was done, but also explaining *why*. Existing approaches face three core issues:

**Shallow causal reasoning**: Existing vision-to-language methods (e.g., ADAPT, DriveGPT4) operate at the level of probabilistic correlations and lack deep causal reasoning over driving behaviors. For example, observing a lead vehicle's brake lights and concluding "stopped because the vehicle ahead decelerated" ignores the true cause, which may be a distant red light.

**Cross-modal spurious correlations**: Spurious correlations exist between visual features and language descriptions. In certain scenes, roadside stop signs and unrelated parked vehicles may form erroneous associations in feature space, causing the model to generate inaccurate inferences.

**Neglect of ego-vehicle-level causal modeling**: Prior work primarily focuses on event-level causality rather than causal understanding centered on ego-vehicle state transitions. The complete loop—starting from a safe state, encountering environmental changes that introduce potential hazards, taking driving actions to return to a safe state—lacks formal modeling.

The authors' core insight is that driving behavior understanding can be formalized as a state transition graph: starting from an initial safe state $X_s$, environmental changes $Z$ introduce potential hazards $W$, and driving action $Y$ transitions the vehicle to a new safe state $X_e$. The task is inverse inference: given observed behaviors and outcomes, identify which environmental factors $V$ are the key influencing variables.

## Method

### Overall Architecture

MCAM consists of three components:
1. **Multi-level Feature Extractor (MFE)**: Extracts global and local features from video.
2. **Causal Analysis Module (CAM)**: Constructs causal relationships based on DSDAG.
3. **Vision-Language Transformer (VLT)**: Integrates visual features and textual information to generate descriptions and reasoning.

### Key Designs

1. **Multi-level Feature Extractor (MFE)**:

    - **Mechanism**: VidSwin Transformer (capturing global dependencies) and 3DResNet (extracting local features) are used in parallel, each processing the full video, the first frame, and the last frame.
    - **Input processing**: Video frames of shape $B \times F \times H \times W \times 3$; VidSwin outputs global features and 3DResNet outputs local features. After dimensionality alignment via downsampling and $1\times1$ convolutions, features are fused with a linear layer.
    - **Design Motivation**: Transformers excel at global modeling but are insufficient for local feature extraction, while CNNs are well-suited for local pattern capture. The dual-path parallel design extracts complementary features. Separate extraction of the first and last frames corresponds to the initial and terminal states in DSDAG.

2. **Causal Analysis Module (CAM)**:

    - **Driving State DAG (DSDAG)**: Models the driving process as a directed acyclic graph with nodes: initial safe state $X_s$, environment $Z$, driving action $Y$, potential hazard $W$, and terminal safe state $X_e$.
    - **Causal inference formalization**: Based on Pearl's causal framework, the do-calculus is used to model intervention effects. Key equations:
        - Action is determined by state and environment: $Y_c = F_Y(Z_\xi | U_s)$
        - Hazard is determined by the environment absent the action: $W = F_W(Z_\xi | U_s, do(Y_c = \emptyset))$
        - Terminal state: $X_e = F_{X_e}(Z_\xi | U_s, do(Y_c = c))$
    - **Feature disentanglement and fusion**: The 6 feature groups output by MFE (first/last frame × global/local, full clip × global/local) are projected via separate linear layers into initial state feature $F_{init}$, terminal state feature $F_{end}$, potential hazard feature $F_{pot}$, action feature $F_{act}$, and original feature $F_{ori}$.
    - **Attention weighting**: Causal features are concatenated as $H = \text{Concat}(F_{init}, F_{end}, F_{pot}, F_{act})$; attention weights are computed as $\alpha = \text{Softmax}(W_H H + b_H)$; the original feature is reweighted as $F = \alpha \odot F_{ori}$.
    - **Design Motivation**: Disentangling video features into distinct components of the driving state graph (states, environment, actions, hazards) enables the model to explicitly reason about which environmental factors truly drive the observed behavior, thereby reducing spurious correlations.

3. **Vision-Language Transformer (VLT)**:

    - **Mechanism**: An MLP aligns causal features into the text embedding space; a Transformer decoder generates description and reasoning text.
    - **Sparse attention mask**: A sparsity constraint $L_{sparse} = \lambda \sum_{i,j} |V_{(i,j)}|$ is applied to the relationship matrix between causal features and word embeddings to prevent hallucination.
    - **Design Motivation**: Direct generation may cause mismatches between visual content and text (hallucination); the sparsity constraint ensures that only genuinely relevant visual features influence the generation of specific tokens.

### Loss & Training

- **Signal loss**: $L_{signal} = \frac{1}{2N} \sum_i |y_i - \hat{y}_i| + (y_i - \hat{y}_i)^2$ (L1 + L2 hybrid)
- **Text generation loss**: Cross-entropy + KL divergence: $L_{caption} = -\frac{1}{N}\sum_i \sum_c y_{i,c}\log(\hat{y}_{i,c}) + \beta \cdot D_{KL}(P \| Q)$
- **Total loss**: $L_{total} = L_{signal} + L_{caption}$
- Batch size 16, trained for 40 epochs; learning rate 0.0003 on BDD-X and 0.0001 on CoVLA.
- All video frames are preprocessed to $224\times224$, sampling 32 frames.
- Trained on a single A100 80GB GPU.

## Key Experimental Results

### Main Results

**BDD-X dataset (Table 2):**

| Method | Narration B4 | Narration CIDEr | Reasoning B4 | Reasoning CIDEr | Params | FPS |
|--------|-------------|-----------------|--------------|-----------------|--------|-----|
| DriveGPT4 | 30.0 | 214.0 | 9.4 | 102.7 | 7.85B | — |
| RAG-Driver | 34.3 | 260.8 | 11.1 | 109.1 | 7.08B | — |
| Baseline (ADAPT) | 33.4 | 241.6 | 8.2 | 75.5 | 620.2M | 365 |
| **MCAM** | **35.7** | **252.0** | **9.1** | **94.1** | **885.3M** | **336** |

**CoVLA dataset (Table 3):**

| Method | B1 | B4 | CIDEr | METEOR | ROUGE | Params |
|--------|----|----|-------|--------|-------|--------|
| Baseline | 81.9 | 74.2 | 236.9 | 48.8 | 80.7 | 620.2M |
| **MCAM** | **82.6** | **75.3** | **275.4** | **50.2** | **81.9** | **885.3M** |

### Ablation Study

**Module combination ablation (Table 4, BDD-X):**

| Configuration | Narration B4 | Narration CIDEr | Reasoning B4 | Reasoning CIDEr | Params |
|---------------|-------------|-----------------|--------------|-----------------|--------|
| VidSwin + VLT | 32.9 | 235.8 | 8.0 | 74.3 | 582.2M |
| 3DResNet + VLT | 32.8 | 221.8 | 6.4 | 55.3 | 494.0M |
| MFE + VLT | 33.4 | 218.1 | 7.1 | 79.8 | 845.8M |
| VidSwin + CAM + VLT | 34.2 | 242.7 | 8.2 | 80.3 | 588.3M |
| **MFE + CAM + VLT (MCAM)** | **35.3** | **251.6** | **9.0** | **92.9** | **885.3M** |

### Key Findings

- CAM yields substantially larger gains on the reasoning task (CIDEr: 75.5 → 92.9) than on the narration task (241.6 → 252.0), confirming that causal modeling is more critical for explanatory inference.
- MFE (global + local) outperforms single-encoder alternatives, and CAM alone brings notable improvement even over VidSwin-only (CIDEr: 235.8 → 242.7).
- MCAM has only 885.3M parameters—far fewer than LLM-based methods (7B+)—while achieving competitive performance.
- Qualitative analysis shows that MCAM correctly identifies a distant red light rather than a nearby brake light as the reason for stopping, demonstrating reduced spurious correlation.

## Highlights & Insights

- **The introduction of DSDAG provides a structured causal framework for driving behavior understanding**, transforming the vague "understanding" task into an explicit state-transition reasoning problem.
- **Lightweight design**: Without relying on LLMs, MCAM achieves performance close to 7B-parameter models with only ~900M parameters and a throughput of 336 FPS, suitable for practical deployment.
- **Causal disentanglement strategy**: Projecting video features into distinct causal components (states/environment/actions/hazards) makes causal reasoning interpretable.
- This work represents the first integration of causal analysis structures into ego-vehicle-level video understanding.

## Limitations & Future Work

- Dataset annotations contain noise (CoVLA labels are generated by LLaMA-7B and include labeling errors), which constrains the model's upper bound.
- The state definitions in DSDAG are relatively simplified and do not model complex multi-vehicle interaction causal chains.
- The Transformer decoder in VLT tends to drift when generating long-form text.
- No comparison is made with the latest MLLMs (e.g., GPT-4V, Qwen-VL).
- The interpretability of attention weights in the causal analysis module awaits further validation.

## Related Work & Insights

- Another successful application of Pearl's structural causal model in computer vision.
- Unlike event-level causal methods such as CMCIR, MCAM focuses on ego-vehicle behavior, a more practically relevant dimension.
- The causal feature disentanglement paradigm can inspire other scene understanding tasks, such as pedestrian intent prediction and traffic accident analysis.
- The sparse attention mask strategy for hallucination suppression is worth generalizing to other multimodal generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of DSDAG and the causal analysis module is a creative design, though the practical effect of causal reasoning is primarily realized through attention reweighting.
- **Experimental Thoroughness**: ⭐⭐⭐ Two datasets and ablation over module combinations are provided, but no comparison with recent MLLMs and no interpretability validation of causal analysis.
- **Writing Quality**: ⭐⭐⭐ The causal modeling section introduces many formal definitions, but the correspondence between the formalism and the actual implementation is insufficiently clear.
- **Value**: ⭐⭐⭐⭐ The lightweight design is practically useful, and the causal analysis paradigm offers meaningful insights for driving behavior understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](../../ICLR2026/autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)

</div>

<!-- RELATED:END -->
