---
title: >-
  [Paper Note] Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation
description: >-
  [ICLR2026][Medical Imaging][exoskeleton optimization] This paper proposes the Exo-plore framework, which combines neuromechanical simulation with deep reinforcement learning to optimize hip exoskeleton control parameters…
tags:
  - "ICLR2026"
  - "Medical Imaging"
  - "exoskeleton optimization"
  - "neuromechanical simulation"
  - "deep reinforcement learning"
  - "human-in-the-loop"
  - "surrogate optimization"
date: 2026-05-08
content_hash: ac37d5d6e594be83
---

# Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation

**Conference**: ICLR2026
**arXiv**: [2601.22550](https://arxiv.org/abs/2601.22550)  
**Code**: [Project Page](https://daebangstn.github.io/exo-plore/)  
**Area**: Medical Imaging
**Keywords**: exoskeleton optimization, neuromechanical simulation, deep reinforcement learning, human-in-the-loop, surrogate optimization

## TL;DR

This paper proposes the Exo-plore framework, which combines neuromechanical simulation with deep reinforcement learning to optimize hip exoskeleton control parameters without requiring human subject experiments, and generalizes to pathological gait scenarios.

## Background & Motivation

Exoskeletons have demonstrated great potential for enhancing human mobility, yet delivering appropriate assistance to individual users remains a significant challenge. The current state-of-the-art approach—Human-in-the-Loop Optimization (HILO)—requires participants to walk for hours while wearing an exoskeleton to iteratively optimize control parameters. This creates a paradox: the populations most in need of exoskeleton assistance (e.g., individuals with mobility impairments) are precisely those least capable of tolerating such intensive optimization protocols.

Furthermore, humans actively adapt to external forces imposed by exoskeletons, modifying gait patterns and muscle coordination strategies, which causes predictions based on "fixed gait" assumptions to be systematically inaccurate. Existing neuromechanical simulation methods either rely on motion capture data tracking to handle large observation/action spaces, or depend on hand-crafted biologically inspired controllers with limited generalizability. No unified framework has been established to simultaneously achieve: (i) fitting observed human adaptation behavior, and (ii) predicting responses under unobserved assistance conditions.

## Core Problem

How can simulation accurately reproduce human adaptive responses to exoskeleton assistive forces without conducting real human experiments, enabling efficient optimization of exoskeleton control parameters? In particular, how can this capability generalize to pathological gait scenarios to provide personalized assistance for individuals with mobility impairments?

## Method

### Overall Architecture

Exo-plore consists of two core components: a **Gait Data Generator** and an **Exoskeleton Optimizer**.

### 1. Exoskeleton Controller

The hip exoskeleton employs delayed-feedback control, with the assistive torque defined as:

$$\tau_{\text{exo}}(t) = \kappa \cdot u(t - \Delta t)$$

where $u(t) = \sin(\theta_r) - \sin(\theta_l)$ is a control signal based on the difference between left and right hip joint angles, $\kappa$ is the gain (equivalent stiffness parameter), and $\Delta t$ is the time delay. The optimization objective is to find the optimal $(\kappa, \Delta t)$ that minimizes the metabolic Cost of Transport (CoT).

### 2. Gait Data Generator

The human controller comprises three modules:

- **PoseNet**: Computes PD target joint positions $\mathbf{q}_d$, trained via Deep RL
- **PD Controller**: Generates joint torques to reduce the error between target and current positions, with exoskeleton assistive forces subtracted
- **Muscle Coordination Network (MCN)**: Maps target joint torques to muscle activations $\mathbf{a}$, trained via supervised learning

The total reward function is designed as:

$$r_{\text{total}} = w_{\text{gait}} \cdot r_{\text{gait}} + w_{\text{arm}} \cdot r_{\text{arm}} + w_{\text{energy}} \cdot r_{\text{energy}} + w_{\text{HEI}} \cdot r_{\text{HEI}}$$

where $r_{\text{gait}}$ encourages following target gait patterns, $r_{\text{arm}}$ penalizes unnatural arm motion, $r_{\text{energy}}$ regularizes energy consumption, and $r_{\text{HEI}}$ models human–exoskeleton interaction.

The MCN training loss incorporates a novel **Intra-Muscle Regularizer (IMR)**, which enforces coherent activation patterns among line muscles belonging to the same anatomical muscle group.

### 3. Sim-to-Real Matching

Two key designs align simulation results with real human experimental data:

**(a) Metabolic Energy Model Calibration**: Metabolic energy expenditure is modeled as $\frac{d}{dt}\text{MEE} = \sum_i m_i^\alpha a_i^\beta$. Algorithm 1 and Algorithm 2 search for optimal parameters $(\alpha, \beta)$ such that the simulated Preferred Walking Speed (PWS) matches real human data, yielding $(\alpha, \beta) = (1.5, 1.0)$.

**(b) Human–Exoskeleton Interaction (HEI) Reward**: Designed based on the **resistance minimization hypothesis**, reflecting the behavioral principle that humans are more sensitive to losses than gains (Loss Aversion):

$$r_{\text{HEI}} = 1 + \frac{1}{\kappa} \sum_{k \in \{L,R\}} \min(0, P_k)$$

When the exoskeleton imposes resistive power on the human body ($P_k < 0$), $r_{\text{HEI}}$ drops below 1, driving the policy to actively adjust kinematics to reduce resistance, thereby reproducing the adaptive behavior observed in real human experiments.

### 4. Exoskeleton Optimizer

An MLP surrogate network replaces Gaussian Processes to fully exploit the abundance of simulation data:

- Latin Hypercube Sampling (LHS) is used to sample the control parameter space, avoiding aliasing effects of grid sampling
- The surrogate network loss incorporates Huber Loss (for outlier robustness), gradient penalty (to smooth the CoT landscape), and L1/L2 regularization
- SLSQP and trust-region gradient optimization are applied to identify the optimal control parameters

## Key Experimental Results

### Unassisted Gait Validation

- Joint kinematics (ankle, knee, hip) qualitatively match real human experimental data (Boo et al., 2025)
- Muscle activation patterns resemble human EMG signals without explicit constraints
- The walking speed–CoT curve is consistent with trends in Browning et al. (2006), with accurate PWS prediction

### Assisted Gait Validation

- Under control parameters $(\kappa, \Delta t) = (8\text{Nm}, 0.25\text{s})$, the scaling trends of assistive torque/power with walking speed are consistent with Lim et al. (2019b)
- HEI reward vs. no HEI: at 4 km/h, when delay increases from 0.05s to 0.25s in real human experiments, assistive power increases by 1.88×; the HEI reward condition yields 1.73× (correlation coefficient 0.83), while the no-HEI condition yields only 0.67× (correlation coefficient 0.69)
- The HEI reward condition produces the maximum metabolic reduction rate closest to real human experiments

### Control Parameter Optimization

- Healthy population: the optimal delay $\Delta t$ decreases monotonically with increasing walking speed
- Pathological gait: among 5 pathological gait types (equinus, waddling, crouch, calcaneal, foot drop), 4 exhibit a strong linear relationship between optimal gain $\kappa$ and pathology severity
- Foot drop fails to converge stably due to excessive gait variability caused by frequent toe–ground collisions

## Highlights & Insights

- **Fills a critical gap**: The first work to unify neuromechanical simulation and Deep RL for both fitting and predicting exoskeleton-assisted conditions, enabling genuine optimization without human subject experiments
- **Clever HEI reward design**: Draws on loss aversion from behavioral economics to model human adaptive behavior via the resistance minimization hypothesis—simple yet effective
- **Rigorous sim-to-real matching**: Validation spans not only kinematics but also assistive torque/power scaling, muscle activation patterns, and ground reaction forces across multiple dimensions
- **Pathological gait generalization**: Demonstrates a linear relationship between pathology severity and optimal assistance, with direct clinical implications
- **Practical surrogate network**: MLP + LHS + gradient penalty outperforms traditional Bayesian Optimization in data-rich simulation settings and offers better scalability

## Limitations & Future Work

- **Lack of real human validation**: Control parameters optimized in simulation have not been validated on real human subjects, particularly patient populations
- **Simplified reward model**: The HEI reward is based on a single assumption and may not capture the full complexity of human adaptive behavior
- **No personalization**: The framework does not model individual-specific motor control characteristics
- **Muscle dynamics approximation**: Rigid tendons and simplified muscle models may fail to capture individual differences
- **Foot drop failure**: 1 out of 5 pathological gait types cannot be successfully optimized, revealing limitations of the framework under high-variability scenarios
- **Simplified foot model**: A box-shaped rigid foot leads to overestimated step frequency at low walking speeds

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|-----------------|-------------|
| HILO (Zhang et al., 2017; Slade et al., 2024) | Iterative optimization via real human experiments | Requires hours of walking; infeasible for mobility-impaired patients; <30 iterations |
| Luo et al. (2024) | Deep RL + exoskeleton, published in Nature | Relies on imitation policy, limiting adaptation to unseen conditions; no correlation validation against real human data |
| Generative GaitNet (Park et al., 2022) | Deep RL gait generation | Does not account for exoskeleton assistance or pathological gait |
| **Exo-plore (Ours)** | Unified fitting + prediction framework, HEI reward, surrogate optimization | No real human validation; simplified muscle model |

- **Loss Aversion in robotics**: Introducing behavioral economics concepts to model human–robot interaction rewards represents a cross-disciplinary approach worth adopting in other HRI scenarios (e.g., assistive robots, prosthetic control)
- **Surrogate networks vs. GP**: In data-rich simulation settings, MLP surrogate networks with gradient penalty are more efficient and scalable than traditional Bayesian Optimization—a transferable insight for other simulation-based optimization problems
- **Pathological gait linear relationship**: If validated in real human experiments, this linearity could greatly simplify clinical exoskeleton parameter configuration, allowing optimal parameters to be rapidly estimated from pathology severity alone

## Rating
- Novelty: 8/10 — First work to apply a sim-to-real-matched neuromechanical simulation framework to exoskeleton control optimization; HEI reward design is novel
- Experimental Thoroughness: 8/10 — Multi-dimensional validation and ablation studies are thorough, but real human experimental validation is absent
- Writing Quality: 9/10 — Structure is clear, methods are described in detail, and algorithmic pseudocode is well-formatted
- Value: 8/10 — Significant contribution to the exoskeleton assistance field; pathological gait generalization holds strong clinical promise

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)
- [\[CVPR 2026\] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](../../CVPR2026/medical_imaging/beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)
- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](../../AAAI2026/medical_imaging/human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)

</div>

<!-- RELATED:END -->
