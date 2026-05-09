---
title: >-
  [Paper Note] Watch and Learn: Learning to Use Computers from Online Videos
description: >-
  [CVPR2026][LLM Pretraining][computer-using agent] This paper proposes Watch & Learn (W&L), a framework that leverages an Inverse Dynamics Model (IDM) to automatically convert human computer-use tutorial videos from the internet into executable UI trajectory data. The system generates 53K+ high-quality trajectories that serve as either ICL demonstrations or SFT training data, significantly improving CUA performance across multiple models and platforms.
tags:
  - CVPR2026
  - LLM Pretraining
  - computer-using agent
  - inverse dynamics model
  - video-to-trajectory
  - in-context learning
  - supervised fine-tuning
  - UI grounding
date: 2026-05-08
content_hash: e70440f18b3ddf67
---

# Watch and Learn: Learning to Use Computers from Online Videos

**Conference**: CVPR2026
**arXiv**: [2510.04673](https://arxiv.org/abs/2510.04673)
**Code**: [Project Page](https://chanh.ee/wandl/)
**Area**: LLM Pretraining
**Keywords**: computer-using agent, inverse dynamics model, video-to-trajectory, in-context learning, supervised fine-tuning, UI grounding

## TL;DR

This paper proposes Watch & Learn (W&L), a framework that leverages an Inverse Dynamics Model (IDM) to automatically convert human computer-use tutorial videos from the internet into executable UI trajectory data. The system generates 53K+ high-quality trajectories that serve as either ICL demonstrations or SFT training data, significantly improving CUA performance across multiple models and platforms.

## Background & Motivation

**Background**: Computer-Using Agents (CUAs) require large-scale multi-step human-computer interaction trajectories for training, but manual annotation is prohibitively expensive — OpenCUA's AgentNet dataset of 22K tasks took 6 months and over \$32,000, with scaling to millions costing upwards of \$500K.

**Limitations of Prior Work**: Manually annotated UI datasets are limited in scale and domain coverage, making it difficult to generalize to diverse and constantly evolving applications and operating systems. Exploration-based synthesis (e.g., BAGEL, OS-Genesis) introduces significant noise, while tutorial-driven synthesis relies on fragile LLM annotations that misalign with actual user behavior.

**Key Challenge**: A vast repository of human-operated tutorial videos exists on platforms like YouTube, naturally encoding cross-application task workflows, but no effective method exists to convert these into structured trajectories. Prior video-to-trajectory approaches such as MONDAY achieve only ~70% accuracy through cascaded pipelines, and TongUI's reliance on MLLM action annotation is similarly unreliable, with errors compounding across steps.

**Goal**: Develop a scalable, automatic pipeline to harvest internet tutorial videos and convert them into high-fidelity UI trajectories, applicable across operating systems and usable for both ICL and SFT paradigms.

## Method

### Overall Architecture

The W&L framework operates in three stages: **(1)** constructing a large-scale state-transition corpus and training an Inverse Dynamics Model (IDM); **(2)** task-aware retrieval of tutorial videos combined with IDM annotation to generate trajectories; **(3)** deploying trajectories as either ICL demonstrations or SFT training data to empower CUAs.

### Key Designs

**1. Inverse Dynamics Model (IDM)**

- **Core Idea**: Given consecutive screenshot pairs $(O_t, O_{t+1})$, the model predicts the user action $a_t$ that caused the transition — reducing trajectory recovery to single-step inverse dynamics prediction
- **Action Space**: 6 atomic operations — Click (with coordinates), Release, Scroll, Type (with text), Wait, Move (with coordinates); Click + Move + Release compose drag operations
- **Architecture**: SigLIP-2 vision encoder + 4-layer Transformer backbone + three prediction heads:
    - Action classification head: 6-class action classifier
    - Coordinate head: discretizes (x, y) into 0–999 classification bins (more stable than regression)
    - Language head: lightweight GPT-2 decoder for autoregressive text generation
- **Training Data**: Web pages sampled from Common Crawl entry points, automatically browsed to record 600K+ $(O_t, a_t, O_{t+1})$ triplets
- **Training Objective**: Multi-task cross-entropy loss with action-type-conditional branch activation

**2. Video Retrieval and Trajectory Generation**

- **Inference-time retrieval**: Task description + initial screenshot → Gemini 2.5 Flash optimizes search query → YouTube Search API retrieves top-15 videos → filters out non-screencast/blurry segments → retains top-3
- **Training-time retrieval**: Covers 69 applications across 7 categories (productivity/programming/design/video editing/audio/system/science); Gemini generates diverse queries, yielding 53,125 tutorial videos
- **Filtering**: 1fps frame sampling; Gemini 2.5 Flash automatically removes non-screencast content, cropping/zooming artifacts, and blurry transitions
- **Trajectory Annotation**: IDM predicts actions frame-by-frame, assembling complete trajectories $\tau = (O_0, a_0, O_1, a_1, \ldots, O_T, a_T, O_{T+1})$

**3. Trajectory Application Modes**

- **ICL**: Each trajectory is converted into (observation, action, reasoning) triplet demonstrations, where reasoning is generated by Gemini 2.5 Flash as natural language explanations
- **SFT**: Annotated trajectories are aggregated into (state, action) sequence corpora and used for standard sequence modeling fine-tuning

## Key Experimental Results

### Main Results

| Setting | Model | Method | Success Rate (%) |
|---------|-------|--------|-----------------|
| **ICL** | Gemini 2.5 Flash | Base | 19.0 |
| | | + W&L | **22.0** (+3.0) |
| | OpenAI o3 | Base | 21.8 |
| | | + TongUI | 21.1 (-0.7) |
| | | + W&L | **24.3** (+2.5) |
| | Claude 4 Sonnet | Base | 43.9 |
| | | + TongUI | 43.4 (-0.5) |
| | | + W&L | **45.5** (+1.6) |
| | Jedi (o3) | Base | 50.6 |
| | | + W&L | **52.8** (+2.2) |
| **SFT** | Qwen 2.5VL 7B | Base | 1.9 |
| | | + TongUI | 5.4 (+3.5) |
| | | + W&L | **13.0** (+11.1) |
| | UI-TARS-1.5-7B | Base | 27.3 |
| | | + TongUI | 23.8 (-3.5) |
| | | + W&L | **31.1** (+3.8) |

> OSWorld-Verified (50-step). W&L consistently outperforms baselines and TongUI across both ICL and SFT paradigms.

### WindowsAgentArena Results (15-step)

| Model | Success Rate (%) |
|-------|-----------------|
| UI-TARS-1.5-7B (zero-shot) | 18.1 |
| + TongUI SFT | 12.9 (-5.2) |
| + **W&L SFT** | **24.0** (+5.9) |
| OpenCUA-7B | 13.5 |
| UltraCUA-7B | 21.7 |

> W&L achieves SOTA among 7B models, while TongUI annotation actually degrades performance.

### Ablation Study

**IDM Annotation Accuracy Comparison** (held-out test set, 100 transitions per action type):

| Metric | Gemini 2.5 Flash | TongUI | **W&L IDM** |
|--------|-----------------|--------|------------|
| ActionType Acc. | 81.5% | 84.3% | **95.8%** |
| Action Acc. | 70.5% | 72.3% | **91.7%** |

**ICL Component Ablation** (OSWorld):

| Configuration | Gemini Flash | o3 | Claude Sonnet |
|---------------|-------------|-----|--------------|
| No examples | 19.0 | 21.8 | 43.9 |
| + Frames | 18.4 | 21.8 | 43.9 |
| + Frames + Actions | 20.1 | 23.0 | 44.4 |
| + Frames + Actions + Reasoning | **22.0** | **24.3** | **45.5** |

**Retrieval Strategy Ablation**: Random retrieval provides no gain for o3 (21.8), while task-relevant retrieval yields +2.5 → 24.3.

### Key Findings

- IDM annotation accuracy (91.7%) substantially surpasses TongUI (72.3%) and Gemini (70.5%), with particularly large advantages on location-dependent actions such as click and scroll
- TongUI annotation performs worse on Windows (TongUI's underlying UI-TARS was trained on Ubuntu), causing a 5.2-point SFT performance drop
- In the ICL setting, action labels and reasoning traces contribute incremental gains, indicating that trajectories convey procedural and causal knowledge beyond visual context
- Random retrieval does not harm performance (since annotations are inherently accurate), but task-relevant retrieval is necessary for significant improvements

## Highlights & Insights

- **Inverse dynamics modeling is the core innovation**: reduces trajectory recovery from end-to-end generation to single-step prediction, drastically lowering learning difficulty while naturally generalizing across applications
- **Exceptional scalability**: 53K trajectories generated fully automatically without human annotation, at far lower cost than AgentNet-style approaches
- **Dual application pathway**: the same trajectory set supports both ICL and SFT, flexibly accommodating both closed-source and open-source models
- **Cross-OS generalization**: demonstrated effectiveness on both Ubuntu (OSWorld) and Windows (WAA), validating the platform robustness of IDM annotations
- **The +11.1 gain on Qwen 2.5VL 7B** demonstrates that general-purpose multimodal models can acquire operational capabilities they originally lacked through this data

## Limitations & Future Work

- The IDM supports only 6 atomic actions, with limited coverage of complex interactions (e.g., right-click context menus, multi-touch gestures, keyboard shortcut combinations)
- Dependence on YouTube video quality and availability; niche applications may lack tutorial coverage
- Filtering and retrieval rely on Gemini 2.5 Flash, introducing dependency on a commercial API
- The paper does not explore automatic segmentation of multi-task long videos; the current approach assumes one video corresponds to one trajectory
- Reinforcement learning is not explored (only ICL + SFT); the authors list RL as future work
- 1fps sampling may miss intermediate states of rapid operations (e.g., consecutive clicks, fast scrolling)

## Related Work & Insights

- **Exploration-based synthesis**: BAGEL, NNetNav, Explorer, OS-Genesis — random exploration + retrospective annotation, producing noisy data
- **Tutorial-driven synthesis**: Synatra, AgentTrek (text tutorials), TongUI (multimodal tutorials + MLLM annotation) — broad coverage but fragile annotation
- **Self-improving agents**: OpenWebVoyager, WebRL, ZeroGUI — require no human data but are limited in task distribution
- **IDM in robotics**: VPT (Minecraft pretraining), DreamGen — inspired the IDM design in this work
- **Agent ICL**: Workflow abstraction + example selection, complementary to this paper's ICL approach

## Rating

- Novelty: ⭐⭐⭐⭐ — Transferring inverse dynamics modeling from robotics to GUI agents is a genuinely original approach
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual benchmarks + ICL/SFT dual paradigms + multi-model evaluation + comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with coherent motivation-method-experiment logic
- Value: ⭐⭐⭐⭐⭐ — Opens a practical pathway for scalable CUA training data production from internet videos

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[CVPR 2026\] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)
- [\[CVPR 2026\] Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](lottiegpt_vector_animation_generation.md)

<!-- RELATED:END -->
