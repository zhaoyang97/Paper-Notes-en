---
title: >-
  [Paper Note] Self-Improving Loops for Visual Robotic Planning
description: >-
  [ICLR 2026][Image Generation][Visual Planning] This paper proposes SILVR, a framework that achieves continual self-improvement on unseen tasks by iteratively fine-tuning an in-domain video generation model on self-collec…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Visual Planning"
  - "Self-Improvement"
  - "Video Generation Models"
  - "Inverse Dynamics Model"
  - "Online Experience"
date: 2026-05-08
content_hash: 0781d922c6869bbb
---

# Self-Improving Loops for Visual Robotic Planning

**Conference**: ICLR 2026
**arXiv**: [2506.06658](https://arxiv.org/abs/2506.06658)
**Code**: [https://diffusion-supervision.github.io/silvr/](https://diffusion-supervision.github.io/silvr/)
**Area**: Image Generation
**Keywords**: Visual Planning, Self-Improvement, Video Generation Models, Inverse Dynamics Model, Online Experience

## TL;DR

This paper proposes SILVR, a framework that achieves continual self-improvement on unseen tasks by iteratively fine-tuning an in-domain video generation model on self-collected online trajectories. SILVR achieves up to 285% performance improvement on MetaWorld and real-robot benchmarks.

## Background & Motivation

### Limitations of Prior Work

**Background**: Video generation models have demonstrated strong capabilities as text-conditioned visual planners for robotic task planning. However, generalization to unseen tasks remains a significant challenge. Existing methods rely primarily on offline data (pre-collected demonstrations or internet videos) and lack the ability to continuously improve from online self-collected experience.

**Core Problem**: Can one design a visual planning agent that self-improves online?

## Method

### Overall Architecture

The core loop of SILVR:
1. Video model generates visual plans → 2. Inverse dynamics model converts plans to actions → 3. Environment interaction collects trajectories → 4. Successful trajectories are filtered → 5. Video model is fine-tuned → Repeat

### Key Design 1: Video Model as Visual Planner

Based on the UniPi framework:
- A text-to-video model predicts future frame sequences as task plans
- An inverse dynamics model (IDM) converts consecutive frame pairs into executable actions
- Supports AVDC-based in-domain video models with text cross-attention

### Key Design 2: Inverse Probability Adaptation (IPA)

Integrates the in-domain video model with an internet-pretrained video prior (AnimateDiff, ~2B parameters) via score composition:

$$\tilde{\epsilon}_{\text{inv}} = \epsilon_{\text{general}}(\tau_t, t) + \alpha(\epsilon_{\text{general}}(\tau_t, t|\text{text}) + \gamma\epsilon_\theta(\tau_t, t|\text{text}) - \epsilon_{\text{general}}(\tau_t, t))$$

- $\gamma$: prior strength
- $\alpha$: text guidance scale
- The in-domain model supplies environment-specific visual knowledge
- The pretrained model provides text generalization and motion priors

### Key Design 3: Self-Improving Loop

Each iteration (Algorithm 1):
1. Optionally adapt with internet video prior
2. Execute $N$ visual planning interactions
3. Filter trajectories using filter function $f_r$ (success signal)
4. Fine-tune the in-domain video model on accumulated data
5. Optionally distill into a lightweight policy

## Experiments

### MetaWorld Results (12 Unseen Tasks, Average Success Rate)

| Method | Iter 0 | Iter 1 | Iter 2 | Iter 3 | Iter 4 |
|--------|--------|--------|--------|--------|--------|
| DSRL (GT filter) | 9.4 | 8.3 | 7.4 | 7.5 | 7.7 |
| BCIL (GT filter) | 5.6 | 12.3 | 20.9 | 23.3 | 23.2 |
| **SILVR (GT filter)** | **14.7** | **27.7** | **33.5** | **43.5** | **44.2** |
| SILVR (VLM filter) | 17.0 | 24.4 | 28.7 | 34.4 | 38.4 |

After iteration 4, SILVR achieves a success rate of 44.2%, substantially outperforming BCIL (23.2%) and DSRL (7.7%).

### Real-Robot Experiments

- **Cup-pushing task**: Continual improvement on unseen object colors
- **Drawer-opening task**: SILVR with internet video prior successfully guides self-improvement
- Without the internet video prior, improvement stagnates or degrades in real-world experiments

### Distillation

Distilling the final SILVR video planner into a Diffusion Policy yields a marginal further gain (44.2% → 49.2%).

### Ablation Study

| Setting | Finding |
|---------|---------|
| No data filtering | Slow improvement on MetaWorld; improvement still observed in real world |
| VLM instead of GT filter | Gemini-2.5-Pro performs best; self-improvement remains achievable |
| Suboptimal initial data | SILVR still achieves continual improvement |
| 10 iterations | Performance saturates after iteration 5 |

### Key Findings

- The decoupled design of visual planning (dynamics modeling vs. action prediction) facilitates generalization
- Internet video priors are critical for real-world experiments
- Without filtering, suboptimal experience can still convey useful information through score composition
- SILVR is more sample-efficient than RL fine-tuning baselines

## Highlights & Insights

- First systematic self-improving framework for visual robotic planning
- Organically integrates offline data with online experience
- Introduction of internet video priors elegantly addresses real-world generalization
- Robust to the choice of filtering signal (GT / VLM / no filter all remain functional)
- Distillation balances planning quality and inference speed

## Limitations & Future Work

- Assumes the initial model has a reasonable baseline success rate (cold-start problem)
- The choice of large-scale pretrained video models introduces efficiency/quality trade-offs
- Performance saturates after 10 iterations, potentially converging to a local policy optimum
- Video generation inference speed remains a deployment bottleneck
- Exploration mechanisms for escaping unimodal behavior are not investigated

## Related Work & Insights

- **Video planning**: UniPi, AVDC, and related video models for decision-making
- **Self-improving models**: Self-improvement in LLMs, VideoAgent, etc.
- **RL fine-tuning of BC policies**: DPPO, DSRL, ResIP, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of visual planning and self-improving loops is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across simulation and real robot, with multiple ablations and baselines
- Value: ⭐⭐⭐⭐ — Real-robot validation and distillation scheme address practical deployment
- Writing Quality: ⭐⭐⭐⭐ — Ablation studies on various design decisions are thorough

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](../../CVPR2026/image_generation/image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](../../CVPR2026/image_generation/solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](../../CVPR2026/image_generation/solace_self_confidence_rewards_t2i.md)
- [\[ICLR 2026\] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)
- [\[ICLR 2026\] Steer Away From Mode Collisions: Improving Composition In Diffusion Models](steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
