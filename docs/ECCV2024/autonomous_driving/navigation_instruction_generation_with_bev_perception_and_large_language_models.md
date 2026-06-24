---
title: >-
  [Paper Note] Navigation Instruction Generation with BEV Perception and Large Language Models
description: >-
  [ECCV 2024][Autonomous Driving][Navigation Instruction Generation] This paper proposes BEVInstructor, which integrates Bird's-Eye-View (BEV) features into multimodal large language models. Through a Perspective-BEV fusion encoder, parameter-efficient prompt tuning, and an instance-guided iterative refinement strategy, it achieves state-of-the-art performance on both indoor and outdoor navigation instruction generation tasks.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Navigation Instruction Generation"
  - "Bird's-Eye View Perception"
  - "Multimodal Large Language Models"
  - "Prompt Tuning"
  - "Iterative Refinement"
date: 2026-05-08
content_hash: f91995fa4ca0524f
---

# Navigation Instruction Generation with BEV Perception and Large Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.15087](https://arxiv.org/abs/2407.15087)  
**Code**: [Available](https://github.com/FanScy/BEVInstructor)  
**Area**: Autonomous Driving / Embodied AI  
**Keywords**: Navigation Instruction Generation, Bird's-Eye View Perception, Multimodal Large Language Models, Prompt Tuning, Iterative Refinement

## TL;DR

This paper proposes BEVInstructor, which integrates Bird's-Eye-View (BEV) features into multimodal large language models. Through a Perspective-BEV fusion encoder, parameter-efficient prompt tuning, and an instance-guided iterative refinement strategy, it achieves state-of-the-art performance on both indoor and outdoor navigation instruction generation tasks.

## Background & Motivation

Navigation instruction generation requires embodied agents to describe routes in natural language based on path trajectories. This task holds significant value in areas such as robotics and human-computer interaction, including assisting visually impaired navigation and autonomous search-and-rescue reporting.

**Limitations of Prior Work**:

**Lack of 3D Perception**: Existing methods (e.g., CCC-speaker, Lana) directly map 2D perspective observations to route descriptions, ignoring the geometric information and object semantics of 3D environments, which easily leads to ambiguous route descriptions.

**Domain Gap of MLLMs**: Multimodal Large Language Models (such as GPT-4V, InstructBLIP) are primarily pre-trained on third-person independent image-text pairs, making it difficult to directly understand the spatial context of first-person viewpoint sequences. Zero-shot generation of navigation instructions performs poorly.

**Lack of Step-by-step Refinement**: Research in cognitive science shows that humans first draft a route description based on landmarks and then progressively refine it, whereas existing methods lack this iterative refinement mechanism.

**Core Idea**: To introduce BEV perception to encode 3D spatial semantics and geometric structures, combining the powerful linguistic capability of MLLMs, and simulating the human process of "drafting landmarks first $\to$ then refining descriptions" to improve instruction quality.

## Method

### Overall Architecture

BEVInstructor is based on LLaMA-7B and consists of three main modules: (1) **Perspective-BEV Visual Encoder** to encode 3D scene information; (2) **Perspective-BEV Prompt Tuning** to achieve parameter-efficient cross-modal alignment; (3) **Instance-Guided Iterative Refinement** to progressively enhance instruction quality.

Task Definition: Given an observation sequence $\mathcal{O} = \{O_t\}_{t=1}^T$ and an action sequence $\mathcal{A} = \{a_t\}_{t=1}^T$ of a navigation path, the model autoregressively generates the instruction $\mathcal{X} = \{x_l\}_{l=1}^L$:

$$\max_\Theta \sum_{l=1}^L \log P_\Theta(x_l | x_{<l}, \mathcal{O}, \mathcal{A})$$

### Key Designs

1. **Perspective-BEV Visual Encoder**: Encodes 3D environmental semantics and geometric information.

    - **Perspective Embedding**: Combines multi-view image features $F_{t,k}$ with orientation angle encoding $\delta_{t,k}$ and timestep embeddings as $p_{t,k} = \mathcal{E}^p(F_{t,k}) + \mathcal{E}^\delta(\delta_{t,k}) + E_t + E_o$.
    - **BEV Embedding**: Aggregates multi-view features into a $15 \times 15$ BEV grid via a BEV encoder (6 layers of deformable attention), utilizing depth-consistency weights $w_{k,n}^c$ to distinguish reference point projections of different depths. The BEV encoder is frozen after pre-training under the supervision of a 3D detection task.
    - **Perspective-BEV Fusion**: Uses Transformer layers to fuse the BEV embedding $B_t$ and perspective embeddings $[P_t, a_t]$, then compresses the $H_b W_b$ tokens into $N_q = 10$ fixed-length tokens via a lightweight Transformer $\mathcal{Q}$ to avoid excessively long tokens when feeding into the MLLM.
    - **Design Motivation**: 2D perspective features retain rich visual cues but lack 3D geometry, while BEV features encode spatial structures but lack texture details. Their complementary fusion achieves comprehensive scene understanding.

2. **Perspective-BEV Prompt Tuning**: Parameter-efficiently utilizes the cross-modal capabilities of the MLLM.

    - Inserts $N_p$ learnable embeddings into the visual embeddings $O_{1:T}$ as Perspective-BEV Prompts: $O' = O_{1:T} \oplus E_v$.
    - Introduces zero-initialized attention and learnable scale vectors in the last $N_a = 31$ layers of LLaMA.
    - **Design Motivation**: Directly fine-tuning MLLMs is expensive and may degrade text generation performance. Parameter-efficient scene-instruction alignment is achieved by updating only 7.2% of the parameters.

3. **Instance-Guided Iterative Refinement**: Simulates the human cognitive process of describing routes.

    - First Stage: BEVInstructor first identifies key instances and generates landmark tokens $\mathcal{X}^I$.
    - Second Stage: Generates complete instructions conditioned on the drafted landmarks: $\mathcal{O} \times \mathcal{A} \times \mathcal{X}^I \rightarrow \mathcal{X}$.
    - **Design Motivation**: Cognitive science shows that key landmarks play a central role in human route descriptions. Two-stage generation progressively enriches object semantics in instructions.

### Loss & Training

- **BEV Encoder Pre-training**: Supervised training on a 3D detection task using $\ell_1$ loss + cross-entropy loss, and then frozen.
- **Instruction Generation Training**: Autoregressive cross-entropy loss, combined with joint optimization of landmark generation and instruction generation (Eq. 11).
- Uses AdamW optimizer, learning rate of $1e^{-4}$, batch size of 8, and 20K iterations.
- Most parameters of LLaMA (6.68B) are frozen, fine-tuning only <500M parameters.

## Key Experimental Results

### Main Results

Comparison with SOTA on three datasets:

| Dataset | Metric | BEVInstructor | Prev. SOTA (Lana) | Gain |
|--------|------|---------------|-----------------|------|
| R2R val seen | SPICE | **0.220** | 0.201 | +1.9% |
| R2R val seen | CIDEr | **0.549** | 0.503 | +4.6% |
| R2R val unseen | SPICE | **0.208** | 0.194 | +1.4% |
| R2R val unseen | CIDEr | **0.449** | 0.419 | +3.0% |
| REVERIE val seen | CIDEr | **0.745** | 0.619 | **+12.6%** |
| REVERIE val unseen | CIDEr | **0.489** | 0.406 | **+8.3%** |
| UrbanWalk test | SPICE | **0.679** | 0.566 | +11.3% |
| UrbanWalk test | Rouge | **0.786** | 0.655 | +13.1% |

### Ablation Study

Ablation of components on R2R val unseen:

| Configuration | SPICE | CIDEr | Description |
|------|-------|-------|------|
| Perspective Only | 0.154 | 0.209 | Baseline |
| BEV Only | 0.172 | 0.281 | BEV alone outperforms Perspective |
| Perspective + BEV (concat) | 0.180 | 0.342 | Simple concatenation of the two features |
| + Fusion Module | 0.190 | 0.373 | Transformer fusion outperforms simple concatenation |
| + Iterative Refinement | 0.192 | 0.419 | Iterative refinement yields CIDEr +7.7% |
| **Full Model** | **0.208** | **0.449** | All modules stack complementarily |

Fusion method comparison (R2R val unseen):

| Fusion Method | SPICE | CIDEr | Description |
|---------|-------|-------|------|
| Addition | 0.185 | 0.366 | Simple addition |
| Concat | 0.184 | 0.310 | Concatenation |
| **Ours (Transformer)** | **0.208** | **0.449** | Transformer fusion is optimal |

### Key Findings

- BEV features alone already outperform Perspective features (CIDEr 0.281 vs 0.209), indicating that 3D geometric information is crucial for instruction generation.
- Instance-guided iterative refinement consistently improves performance under all settings; one-step refinement works best, while further increasing steps yields limited gains.
- GPT-4V's zero-shot performance is significantly lower than fine-tuned methods (SPICE 0.098 vs 0.208), indicating that general MLLMs cannot directly handle the navigation instruction generation task.

## Highlights & Insights

- It is the first to introduce BEV perception into navigation instruction generation, effectively bridging 3D spatial understanding and language generation.
- A parameter-efficient design (with only 7.2% of the parameters being trainable) balances performance and efficiency.
- Significant improvements are achieved in both indoor (R2R, REVERIE) and outdoor (UrbanWalk) scenes, verifying the generalizability of the method.
- Generated instructions can practically guide downstream navigation agents (HAMT/DUET), demonstrating the real-world usability of the instructions.

## Limitations & Future Work

- The BEV encoder requires pre-training data from 3D detection tasks; adaptation to novel scenes still needs verification.
- Currently only validated on simulator datasets; performance in real-world physical environments remains to be explored.
- Combining BEV perception with stronger MLLMs (such as LLaMA-2/3) could be explored.
- The iterative refinement currently employs a fixed number of steps; adaptive stopping strategies can be considered.

## Related Work & Insights

- Compared with Lana (CVPR2023), BEVInstructor introduces 3D geometric priors, enhancing the depth of scene understanding.
- The design of the BEV encoder draws inspiration from BEVFormer in the autonomous driving domain. Applying it to indoor navigation tasks is a novel contribution.
- The idea of instance-guided iterative refinement can be generalized to other vision-language tasks that require step-by-step reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Demonstrates reasonable design by combining BEV perception with MLLMs for navigation instruction generation for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets + detailed ablation + downstream agent evaluation, extremely thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivation.
- **Value**: ⭐⭐⭐⭐ — Provides a new technical roadmap for language interactions in Embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] H-V2X: A Large Scale Highway Dataset for BEV Perception](h-v2x_a_large_scale_highway_dataset_for_bev_perception.md)
- [\[CVPR 2026\] TrafficAlign: Aligning Large Language Models for Traffic Scenario Generation](../../CVPR2026/autonomous_driving/trafficalign_aligning_large_language_models_for_traffic_scenario_generation.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)

</div>

<!-- RELATED:END -->
