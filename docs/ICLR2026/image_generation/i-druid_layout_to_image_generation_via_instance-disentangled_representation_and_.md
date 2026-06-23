---
title: >-
  [Paper Note] I-DRUID: Layout to Image Generation via Instance-Disentangled Representation and Unpaired Data
description: >-
  [ICLR 2026][Image Generation][Layout-to-Image] Addressing two major issues in Layout-to-Image (L2I) generation—"attribute leakage" caused by entangled instance features in attention layers and "poor cross-scenario generalization" due to insufficient paired data—I-DRUID introduces an **Instance-Disentangled Module + Disentangled Constraint** to extract clean semanti
tags:
  - ICLR 2026
  - Image Generation
  - Layout-to-Image
  - Reinforcement Learning
  - MM-DiT
date: 2026-05-08
content_hash: 64733b45bc1502ec
---
# I-DRUID: Layout to Image Generation via Instance-Disentangled Representation and Unpaired Data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yB7FIFwJwN](https://openreview.net/forum?id=yB7FIFwJwN)  
**Code**: To be confirmed  
**Area**: Image Generation / Layout-to-Image Generation  
**Keywords**: Layout-to-Image, Attribute Leakage, Instance Disentanglement, Reinforcement Learning, AI Feedback, MM-DiT  

## TL;DR
Addressing two major issues in Layout-to-Image (L2I) generation—"attribute leakage" caused by entangled instance features in attention layers and "poor cross-scenario generalization" due to insufficient paired data—I-DRUID introduces an **Instance-Disentangled Module + Disentangled Constraint** to extract clean semantic features. It then employs **prompt-only Reinforcement Learning without paired images** to adapt the model to new scenarios via AI feedback. This synergistic approach achieves SOTA performance on both UNet and MM-DiT architectures.

## Background & Motivation
**Background**: The L2I task requires generating multiple objects harmoniously at specific locations given bounding boxes and instance descriptions. Current methods are categorized into training-based (injecting layout info via adapters into attention) and training-free (manipulating attention maps during inference). Creati-Layout was the first to implement L2I on MM-DiT architectures like SD3.

**Limitations of Prior Work**: ① **Attribute Leakage**: Attention layers naturally mix features of adjacent instances; for example, a "red hotdog" next to a "brown hotdog" may result in the brown one being colored red. Prior work mitigated this via attention map manipulation, but CLIP struggles to isolate individual attributes from complex prompts, limiting effectiveness. ② **Poor Generalization**: Models trained on long descriptions (LayoutSAM) often fail when encountering short, concise descriptions (COCO-MIG). Addressing this typically requires collecting more paired image-text data, which is costly.

**Key Challenge**: Achieving feature disentanglement for each instance while transferring the model to new scenarios without relying on additional paired data.

**Goal**: To eliminate attribute leakage and enhance cross-scenario generalization across both UNet and MM-DiT architectures without requiring additional paired data collection.

**Core Idea**:
- **Instance-Disentangled Representation (IDR)**: An Instance-Disentangled Module (IDM) and a Disentangled Constraint (IDC) are designed to split attention features into "semantically relevant features $R^+$" and "spurious parts $R^-$". The key insight is that semantically relevant features should trigger more precise attention maps than spurious parts.
- **Unpaired Reinforcement Learning (UID)**: Using only prompt-only unpaired data, Grounding-DINO serves as AI feedback. PPO is used to encourage or reject generation trajectories, enabling scenario adaptation without paired data.
- **Synergistic Gain**: IDM provides more accurate generation policies for RL, while RL in turn enhances L2I precision.

## Method

### Overall Architecture
I-DRUID consists of two stages: **(a) Instance-Disentangled Learning**, where IDM refines instance features into semantic components under IDC supervision following layout injection; **(b) Reinforcement Learning**, which converts deterministic ODE sampling to SDE to introduce exploration. Grounding-DINO evaluates sampling trajectories, and PPO optimizes the generation policy for new scenarios. The final loss jointly optimizes diffusion, disentanglement, and RL objectives.

```mermaid
flowchart LR
    A[Global Prompt + Instance Descriptions + Layout Boxes] --> B[Adapter Injecting Layout<br/>Enhanced Features E]
    B --> C[IDM: Calculate Channel Weights α]
    C --> D["R+ = α⊙E Semantic Relevant<br/>R− = (1-α)⊙E Spurious"]
    D --> E[IDC: Constraint CAS(R+) < CAS(R−)]
    E --> F[Denoising / Velocity Prediction]
    F --> G[SDE Trajectory Sampling]
    G --> H[GDINO Reward<br/>IoU + Confidence]
    H --> I[PPO Actor-Critic Optimization]
    I --> F
```

### Key Designs

**1. Instance-Disentangled Module (IDM): "Branching" features into semantic and spurious paths via channel weights.** IDM receives $n+1$ enhanced features ($n$ instances + 1 global prompt) $E=\{e_1,...,e_{n+1}\}\in\mathbb{R}^{(n+1)\times C\times W\times H}$ and corresponding layout masks $M$ (1 inside the box, 0 outside; global prompt mask is all 1s). It calculates channel-level weights $\alpha = \text{IDM}(E,M) \in \mathbb{R}^{n+1}$ through feature and mask branches (Conv / AvgPool / FC / Softmax). Separation is performed via multiplication: $R^+ = \alpha \odot E$ represents "semantically relevant features," while $R^- = (1-\alpha) \odot E$ represents the "spurious part." This explicitly separates entangled attention features.

**2. Instance-Disentangled Constraint (IDC): Encouraging "clean features" to trigger more accurate attention maps.** To guide the separation, the authors define the Cross-Attention Score (CAS) to measure the dispersion of an instance's attention map on the **background outside the box**: $\text{CAS}(R^+_{CA}, M) = \sum_{i=1}^{n}|R^+_{CA,i} - \text{AVG}(R^+_{CA,i} \odot (1-M_i))| \odot (1-M_i)$. A higher CAS indicates more leakage outside the box. The intuition is that $R^+$ should trigger a lower CAS than $R^-$. This is formulated as a differentiable loss: $L_{dis} = \text{Softplus}[\text{CAS}(R^+_{CA}, M) - \text{CAS}(R^-_{CA}, M)]$. Minimizing this forces the model to learn clean semantic features.

**3. Unpaired Reinforcement Learning: Converting deterministic sampling to SDE with GDINO as the judge.** Since MM-DiT (SD3) uses deterministic ODEs, exploration is limited. The authors rewrite the policy into an SDE form: $x_{t+\Delta t} = x_t + [v_\theta(x_t,t,y) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t,t,y))] \Delta t + \sigma_t\sqrt{\Delta t}\epsilon$, where $\sigma_t = a\sqrt{t/(1-t)}$. The reward is calculated using Grounding-DINO on generated images by computing IoU and confidence for each instance: $r(o,o_{pred}) = \sum_i[\text{IoU}(b_{pred,i},b_i) + c_{pred,i}]$. This allows online scoring using only prompts.

**4. Actor-Critic PPO Synergy + RL Acceleration.** A lightweight MLP critic-net $\phi$ is introduced to predict scalar rewards, trained with $L_{critic} = [\phi(s_t) - r(o,o_{pred})]^2$ to obtain the advantage function $A(s_t) = r - \phi(s_t)$. The Actor uses standard PPO importance sampling with a probability ratio $\rho_t$ and loss $L_{rl} = \max[-\rho_t A_t, -\text{clip}(\rho_t, 1-\zeta, 1+\zeta)A_t] + \text{KL}(\cdot)$. For efficiency, RL is only applied during the **first 20% of timesteps**. The final joint loss is $L_{act} = L_{ldm} + \lambda_{dis}L_{dis} + \lambda_{rl}L_{rl}$.

## Key Experimental Results

### Main Results

COCO-MIG (testing cross-scenario generalization with concise descriptions) average metrics:

| Method | Avg ISR ↑ | Avg mIoU ↑ |
|------|-----------|------------|
| InstanceDiff | 51.98 | 47.33 |
| MIGC | 59.68 | 52.60 |
| Creati-Layout* | 57.67 | 50.94 |
| **Ours (SD-1.5)** | **69.13** | **68.18** |
| **Ours (SD-3)** | 62.75 | 55.60 |

LayoutSAM-eval (testing basic L2I capabilities, 5000 prompts):

| Method | Spatial ↑ | Color ↑ | Texture ↑ | Shape ↑ | FID ↓ | PickScore ↑ |
|------|-----------|---------|-----------|---------|-------|-------------|
| MIGC | 85.66 | 66.97 | 71.24 | 69.06 | 21.19 | 20.71 |
| InstanceDiff | 87.99 | 69.16 | 72.78 | 71.08 | 19.67 | 21.01 |
| Creati-Layout | 92.67 | 74.45 | 77.21 | 75.93 | 19.10 | 22.02 |
| **Ours (SD-3)** | **93.14** | **75.37** | **78.35** | **77.20** | **17.21** | **23.16** |

### Ablation Study

| No. | IDM | RL-PPO | RL-GRPO | SFT | COCO-MIG Avg ISR | LayoutSAM Spatial |
|-----|-----|--------|---------|-----|------------------|-------------------|
| 1 | × | × | × | × | 56.82 | 86.96 |
| 2 | ✓ | × | × | × | 57.64 | 88.53 |
| 3 | ✓ | × | × | ✓ | 66.92 | 89.75 |
| 4 | ✓ | × | ✓ | × | 61.64 | 92.86 |
| 5 | × | ✓ | × | × | 60.23 | 91.47 |
| 6 | ✓ | ✓ | × | × | **62.75** | **93.14** |

### Key Findings
- **Generalization is a critical challenge**: InstanceDiff and Creati-Layout, trained on long descriptions, underperform compared to MIGC on the concise COCO-MIG dataset, confirming that cross-scenario generalization is significantly more difficult than base capability.
- **Synergy between IDM and RL**: Both IDM (No. 2) and RL (No. 5) provide improvements individually, but their combination (No. 6) yields the best performance on both benchmarks, proving that disentanglement provides better policies for RL.
- **Unpaired RL ≈ Paired SFT**: RL using prompt-only data (62.75 in No. 6) approaches or even exceeds SFT using paired data (66.92 in No. 3) on some metrics, saving data collection costs.
- **PPO vs. GRPO**: PPO (No. 6) outperforms GRPO (No. 4) on COCO-MIG ISR and is more computationally efficient.
- **Visual Effects**: Mitigation of attribute leakage (e.g., colors of hotdogs/chairs no longer bleeding into each other) compared to Creati-Layout.

## Highlights & Insights
- **Converting "attribute leakage" into a differentiable inequality constraint**: Using CAS to measure "out-of-box leakage" and Softplus to enforce $\text{CAS}_{R^+} < \text{CAS}_{R^-}$ is a clean and interpretable approach.
- **Generalization with unpaired data**: By using GDINO as an off-the-shelf judge, the L2I adaptation problem is converted from data collection into online RL exploration, bypassing expensive paired annotations.
- **Enabling RL exploration for deterministic models**: The SDE rewrite allows flow-matching models (SD3/FLUX) to utilize reinforcement learning.
- **Architecture Agnostic**: The framework is compatible with both UNet (SD-1.5) and MM-DiT (SD3).

## Limitations & Future Work
- **Reward dependency on a single detector**: Rewards rely entirely on Grounding-DINO's IoU and confidence; detector bias or failure on small/rare objects can propagate to generation preferences.
- **Heuristic nature of CAS**: Equating "out-of-box dispersion" with localization error is an approximation that might mislead for instances naturally requiring cross-border context.
- **High Training Cost**: Requiring 8×H20 GPUs for 4 days suggests a high barrier to reproduction.
- **SD3 Performance Gap**: SD3's Avg ISR (62.75) is lower than SD-1.5 (69.13) on COCO-MIG, suggesting larger architectures may not inherently resolve short-description generalization.
- **Future Work**: Replacing rewards with multi-judge ensembles or learnable rewards, and extending the disentangled constraint to finer spatial controls like segmentation masks or sketches.

## Related Work & Insights
- **Layout-to-Image**: Standard works like GLIGEN, MIGC, and Creati-Layout form the basis; this work introduces explicit disentanglement constraints to combat attribute leakage across dual architectures.
- **Diffusion RL**: Methods like DPOK and DDPO often rely on DDPM sampling randomness. This work bypasses incompatibility with deterministic ODEs via SDE rewriting, facilitating RL for flow-matching L2I.
- **AI Feedback Alignment**: Inherits the "AI as judge" philosophy from ImageReward and Constitutional AI, transferring the LLM alignment paradigm to layout-based generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combines differentiable disentanglement constraints with unpaired RL via SDE rewriting for flow-matching models, targeting key pain points.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive testing on multiple benchmarks and architectures, including detailed ablations of PPO vs. GRPO vs. SFT.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-explained formulas, and intuitive CAS/IDC explanations.
- **Value**: ⭐⭐⭐⭐ Addresses attribute leakage and unpaired generalization simultaneously with a deployable dual-architecture framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ContextGen: Contextual Layout Anchoring for Identity-Consistent Multi-Instance Generation](contextgen_contextual_layout_anchoring_for_identity-consistent_multi-instance_ge.md)
- [\[NeurIPS 2025\] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](../../NeurIPS2025/image_generation/instanceassemble_layoutaware_image_generation_via_instance_a.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](../../ICML2026/image_generation/envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](../../ICML2026/image_generation/occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)

</div>

<!-- RELATED:END -->
