---
title: >-
  [Paper Note] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence
description: >-
  [CVPR 2026][LLM Reasoning][Spatial Reasoning] EagleVision is a dual-stage framework in which the macro-perception stage employs Semantic-Pose Fusion DPP (SPF-DPP) to jointly optimize semantic relevance and viewpoint dive…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "Spatial Reasoning"
  - "BEV"
  - "Active Vision"
  - "Chain-of-Thought"
  - "DPP Frame Selection"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 995c2305832d79d8
---

# EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence

**Conference**: CVPR 2026
**arXiv**: [2512.15160](https://arxiv.org/abs/2512.15160)  
**Code**: [https://wallelwan.github.io/EagleVision](https://wallelwan.github.io/EagleVision)  
**Area**: LLM Reasoning / Spatial Intelligence
**Keywords**: Spatial Reasoning, BEV, Active Vision, Chain-of-Thought, DPP Frame Selection, Reinforcement Learning

## TL;DR
EagleVision is a dual-stage framework in which the macro-perception stage employs Semantic-Pose Fusion DPP (SPF-DPP) to jointly optimize semantic relevance and viewpoint diversity in SE(3) space for key-frame selection, while the micro-verification stage enables the model to actively query new viewpoint frames on the BEV plane for iterative spatial CoT reasoning (hypothesis → observe → verify loop). The query strategy is trained purely via RL without human annotation, achieving open-source SOTA on VSI-Bench and SQA3D.

## Background & Motivation

**Background**: Video spatial reasoning (distance estimation, direction judgment, layout understanding) requires integrating geometric cues across multiple viewpoints. Existing MLLMs are constrained by fixed token budgets and rely on uniform frame sampling, with no ability to request new viewpoints during inference.

**Limitations of Prior Work**: (1) Uniform sampling guarantees neither semantic relevance nor viewpoint diversity—critical geometric parallax frames may be entirely missed; (2) once initial frames are fixed, the model cannot request additional viewpoints upon discovering insufficient evidence mid-reasoning; (3) collecting multi-step spatial reasoning annotations is impractical—supervision must come from answer-level signals.

**Three Core Research Challenges**: (a) Initial frame selection must balance semantic relevance and viewpoint diversity; (b) spatial hypotheses must be verified in a shared 3D coordinate system, requiring abstract spatial queries to be mapped to concrete frame retrieval; (c) human-annotated CoT trajectories are unavailable.

**Key Insight**: Extending the "thinking with images" paradigm from single-image (crop/zoom) to multi-view 3D spatial reasoning.

**Core Idea**: BEV-grounded pose querying—the model predicts a pose on the BEV plane to retrieve the nearest real frame, alternating with textual reasoning to form a closed loop.

## Method

### Overall Architecture
Input video + query → SLAM reconstruction of camera poses and depth → **Macro-Perception (SPF-DPP frame selection)** → **Micro-Verification (Spatial MCoT + BEV pose querying)** → Output answer. SLAM is a one-time preprocessing step; subsequent inference uses a lightweight 2D VLM with nearest-neighbor lookup, yielding low amortized cost.

### Key Designs

1. **SPF-DPP Macro-Perception (Joint Geometric–Semantic Frame Selection)**:

    - Function: Select $k$ semantically relevant and viewpoint-diverse frames within the token budget.
    - **SE(3) pose distance**: $d_{ij}^2 = \|\mathbf{t}_i - \mathbf{t}_j\|^2/\sigma_t^2 + \beta^2 \theta(R_i, R_j)^2$, where $\theta$ is the rotation angle distance.
    - **Heat-kernel diffusion viewpoint kernel**: Sparse adjacency matrix $W$ → graph Laplacian $\mathcal{L}$ → heat-kernel diffusion $K_{view} = \exp(-\tau \mathcal{L})$, propagating local affinities into global geometric relationships.
    - **Semantic quality modulation**: FG-CLIP computes frame–query similarity → temperature softmax calibration → diagonal quality matrix $Q$, $q_i = (1-\alpha) + \alpha \tilde{s}_i$.
    - **DPP L-ensemble**: $L_{dpp} = Q K_{view} Q$, greedy MAP frame selection ($(1-1/e)$-approximation guarantee).
    - Design Motivation: DPP naturally models the quality–diversity trade-off; heat-kernel diffusion on SE(3) ensures the viewpoint kernel is positive semi-definite.

2. **BEV-Grounded Spatial MCoT Micro-Verification (Active Visual Reasoning)**:

    - Function: The model actively queries new viewpoint frames during reasoning to verify spatial hypotheses.
    - Mechanism: At each reasoning step, the model chooses one of three actions—(a) generate text to continue reasoning; (b) predict a pose query $(x, y, \theta)$ on the BEV plane, whereupon the system retrieves the nearest real frame via scale-aware distance and appends it to the context; (c) terminate and output the answer.
    - This forms a **hypothesis → observe → verify** closed loop: the model hypothesizes "A is 2 meters to the left of B" → predicts a pose that would verify this hypothesis → observes the retrieved frame → confirms or revises.
    - Design Motivation: Static frame selection cannot anticipate all evidence needed during reasoning; active acquisition is essential.

3. **GRPO Training of the Query Strategy (No CoT Annotation Required)**:

    - Function: Train the model on when and where to query new viewpoints.
    - Mechanism: A spatial grounding reward penalizes queries targeting pose regions with no camera coverage, ensuring queries point to areas with actual frames.
    - Design Motivation: Human annotation of multi-step spatial reasoning is impractical; pure RL learns the query strategy from answer-level supervision.

### Loss & Training
Macro-perception requires no training (DPP is computed at inference time). Micro-verification is trained with GRPO using a spatial grounding reward plus an answer accuracy reward. The design is backend-agnostic: SLAM can be replaced by other pose estimation methods such as VGGT.

## Key Experimental Results

### Main Results (VSI-Bench)

| Method | Rank | Obj. Count | Abs. Dist | Obj. Size | Rel. Dir | Route Plan |
|--------|------|-----------|----------|----------|---------|------------|
| Human | — | 79.2 | 94.3 | 47.0 | 94.7 | 95.8 |
| GPT-4o | 3 | 34.0 | 46.2 | 5.3 | 37.0 | 31.5 |
| Gemini-1.5 Pro | 1 | 45.4 | 56.2 | 30.9 | 51.3 | 36.0 |
| InternVL2-8B | 10 | 34.6 | 23.1 | 28.7 | 36.7 | 29.9 |
| **EagleVision** | **Open-source SOTA** | **Best** | **Best** | **Best** | **Best** | **Best** |

Open-source VLM SOTA is also achieved on SQA3D.

### Ablation Study

| Configuration | VSI-Bench Avg | Notes |
|--------------|-------------|-------|
| Uniform sampling (baseline) | Reference | Standard MLLM approach |
| SPF-DPP frame selection | +significant gain | Value of semantic + viewpoint diversity |
| + Spatial MCoT (text only) | +gain | Benefit of reasoning chain |
| + BEV pose querying (full) | **Best** | Core contribution of active visual acquisition |
| Remove viewpoint kernel (semantic only) | Degraded | Geometric diversity is indispensable |
| Remove semantic kernel (viewpoint only) | Degraded | Semantic relevance is equally necessary |
| SLAM → VGGT | On par | Backend-agnostic validation |

### Key Findings
- SPF-DPP frame selection vs. uniform sampling yields consistent improvements across all spatial tasks, particularly for distance estimation and route planning.
- BEV pose querying is the most critical component—spatial hypotheses require specific viewpoints for verification, and active acquisition far outperforms static initial frame selection.
- The human upper bound (95.8%) greatly exceeds the strongest model, indicating that spatial reasoning remains a substantial challenge for AI.
- The model learns a coarse-to-fine query strategy: wide-angle frames are first queried to establish global layout, followed by targeted queries at specific locations to verify distances.

## Highlights & Insights
- **Paradigm shift from passive to active reasoning**: EagleVision is the first to enable MLLMs to actively acquire cross-viewpoint visual evidence during inference, rather than passively consuming a fixed frame set. This has direct implications for embodied AI and robotic navigation.
- **Mathematical elegance of DPP + heat-kernel diffusion**: SE(3) pose distance → heat-kernel diffusion viewpoint kernel → DPP quality–diversity trade-off formalizes geometric intuition into a rigorous mathematical framework.
- **Pure RL training without CoT annotation**: The spatial grounding reward cleverly elicits a reasonable query strategy from answer-level supervision, circumventing the infeasibility of manual CoT annotation.
- **Backend-agnostic design**: SLAM can be replaced by any pose estimation method, ensuring the framework's practicality is not tied to a specific reconstruction system.

## Limitations & Future Work
- Dependency on SLAM preprocessing for poses and BEV maps precludes real-time inference in completely unknown scenes.
- BEV pose querying retrieves only the nearest real frame—if the required viewpoint is not present in the original video, it cannot be obtained.
- The current evaluation is predominantly on indoor scenes (ScanNet); generalization to large-scale outdoor scenes remains to be verified.
- The GRPO-trained query strategy may overfit to the scene types present in the training distribution.

## Related Work & Insights
- **vs. 3D feature augmentation methods (SpatialVLM, etc.)**: These inject 3D information but cannot actively acquire new viewpoints during reasoning.
- **vs. 3D reconstruction + LLM methods**: Reconstruction is an offline black-box step that does not support iterative refinement.
- **vs. ChatGPT-o3 / DeepEyes**: These operate by cropping/zooming within a single image; EagleVision operates across viewpoints in 3D space.
- **Insights**: The BEV-grounded reasoning paradigm is extensible to autonomous driving (querying blind-spot viewpoints) and robotics (planning exploratory paths).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ BEV-grounded active spatial reasoning paradigm is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-benchmark validation on VSI-Bench and SQA3D with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Framework design is clear, mathematical formalization is complete, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐⭐ Significant advancement for spatial intelligence and embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](../../ICML2026/llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](../../ICML2026/llm_reasoning/beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
