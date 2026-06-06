---
title: >-
  [Paper Note] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models
description: >-
  [ICLR 2026][Robotics][bimanual manipulation] TwinVLA is proposed as a modular framework that composes two pretrained single-arm VLAs into a bimanual VLA via joint attention and MoE…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "bimanual manipulation"
  - "VLA"
  - "modular composition"
  - "joint attention"
  - "data efficiency"
date: 2026-05-08
content_hash: ec0f006e56f0f574
---

# TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models

**Conference**: ICLR 2026
**arXiv**: [2511.05275](https://arxiv.org/abs/2511.05275)  
**Code**: [Project Page](https://jellyho.github.io/TwinVLA/)  
**Area**: Robot Manipulation / Bimanual
**Keywords**: bimanual manipulation, VLA, modular composition, joint attention, data efficiency

## TL;DR

TwinVLA is proposed as a modular framework that composes two pretrained single-arm VLAs into a bimanual VLA via joint attention and MoE, requiring only ~800h of public single-arm data, 50 episodes of bimanual fine-tuning data, and 25 H100 GPU-days—achieving performance comparable to π0, which relies on 10,900h of proprietary data and 1,000+ GPU-days.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models have achieved remarkable success in single-arm robotic manipulation, generalizing effectively across tasks, objects, and environments. However, bimanual manipulation—essential for complex tasks such as folding clothes and assembling parts—has seen limited progress due to the scarcity of publicly available bimanual data.

**Limitations of Prior Work**:

1. **Severe data bottleneck**: π0 relies on over 10,000 hours of proprietary bimanual data; RDT-1B requires approximately 2,400 hours of mixed datasets—both prohibitively expensive and non-reproducible.
2. **Enormous computational cost**: RDT-1B trains on 48 H100 GPUs for one month; π0 demands even greater compute, exceeding 1,000 H100 GPU-days.
3. **Limitations of monolithic architectures**: Existing methods mix both arms' actions within a single model, failing to exploit the naturally modular structure of bimanual manipulation.
4. **Difficulty of cross-embodiment transfer**: The observation and action spaces of single-arm and bimanual systems differ substantially, requiring monolithic models to train jointly on heterogeneous data.

**Key Challenge**: Publicly available bimanual data is extremely scarce, yet existing methods all demand large-scale bimanual pretraining. The central question is how to build high-performance bimanual policies from abundant single-arm data.

**Goal**: Inspired by neuroscience—human bimanual control is coordinated by the SMA and corpus callosum across two independent motor systems, rather than a single controller—TwinVLA adopts a modular approach: duplicate a pretrained single-arm VLA → fuse information across arms via joint attention → route shared inputs efficiently with MoE → fine-tune on a small amount of bimanual data.

## Method

### Overall Architecture

The core idea of TwinVLA is "duplicate rather than retrain":

1. **Pretrain SingleVLA** (0.8B parameters): pretrain a compact VLA on the OXE single-arm dataset (~800h).
2. **Duplicate into Twin**: create an exact copy of SingleVLA to serve as left-arm and right-arm policies, respectively.
3. **Joint Attention fusion**: enable information exchange between the two VLMs via Joint Attention.
4. **MoE routing for shared inputs**: route shared inputs (language instruction + egocentric view) to both arms efficiently via MoE.
5. **Bimanual fine-tuning**: only ~50 episodes of bimanual demonstrations are required.

The action head is trained with a Conditional Flow Matching objective:

$$\mathcal{L}^{T}(\theta) = \mathbb{E}_{p(A_t|o_t), q(A_t^\tau|A_t)} \|v_\theta(A_t^\tau, h_t, d_t) - \mathbf{u}(A_t^\tau|A_t)\|^2$$

At inference, actions are sampled from noise via forward Euler integration: $A_t^{\tau+\delta} = A_t^\tau + \delta v_\theta(A_t^\tau, h_t, d_t)$.

### Key Design 1: Selective Module Duplication

The entire SingleVLA is not duplicated wholesale. TwinVLA adopts the following strategy:

- **Shared components**: visual encoder + DiT action head → visual understanding and low-level motor control are embodiment-agnostic.
- **Duplicated components**: VLM backbone → the decision layer requires arm-specific representations.
- **Independent components**: proprioception encoders for each arm.

This design yields a total of 1.3B parameters (compared to RDT-1B's 1.2B), with minimal additional computational overhead.

### Key Design 2: Joint Attention with Causal Masking

The two VLMs exchange information via Joint Attention:

- The Q, K, V tensors from both VLMs are concatenated and passed through a unified self-attention operation.
- Outputs are split back into arm-specific streams.
- A dedicated causal attention mask is designed: within each arm's region, lower-triangular causality is preserved; shared modalities are fully accessible; each arm can attend to half of the other arm's tokens.

### Key Design 3: MoE for Efficient Shared Input Processing

Naively feeding shared inputs (language + egocentric view) redundantly into both VLMs would substantially increase VRAM usage. The solution is:

$$\text{MoE}(x) = w_{\text{left}} \cdot \text{FFN}_{\text{left}}(x) + (1-w_{\text{left}}) \cdot \text{FFN}_{\text{right}}(x)$$

where $w_{\text{left}}$ is computed by a linear layer followed by softmax. For other components (Projection, LayerNorm), an output averaging strategy is applied.

Additionally, **Attention Re-weighting** is introduced to preserve pretraining modality importance and prevent newly added arm-specific tokens from diluting attention—reducing the initial fine-tuning loss by 40%.

## Key Experimental Results

### Main Results: Five Real-World Bimanual Tasks

| Method | Parameters | Pretraining Data | Compute | Avg. Success Rate |
|--------|-----------|-----------------|---------|------------------|
| Diffusion Policy | 271M | None | — | Lowest |
| RDT-1B | 1.2B | ~2,400h | >1,000 GPU-days | Medium |
| **TwinVLA** | **1.3B** | **~800h** | **25 GPU-days** | **High** |
| π0 (upper bound) | 3.3B | ~10,900h | >1,000 GPU-days | Highest |

TwinVLA substantially outperforms RDT-1B (+26% average success rate) and approaches π0's performance, despite using only 7% of π0's data and less than 3% of its compute.

### Ablation Study: Component Contributions

| Ablation Setting | Sim Success Rate | Real-World Success Rate | Notes |
|-----------------|-----------------|------------------------|-------|
| Full TwinVLA | Baseline | Baseline | — |
| w/o Attention Re-weighting | −1.1% | −4.0% | Initial loss increases 40% |
| w/o MoE | −2.2% | −9.0% | VRAM increases 21% |
| w/o Joint Attention | −6.2% | −36.0% | Most critical component |
| Train from scratch (no pretraining) | −4.6% | −46.0% | Pretraining is essential |

Joint Attention is the most critical component; removing it leads to a 27% drop in real-world performance, demonstrating that cross-arm coordination is indispensable for bimanual manipulation.

### Data Efficiency

| Number of Demonstrations | TwinVLA | RDT-1B |
|--------------------------|---------|--------|
| 20 episodes | Starting point | Starting point |
| 35 episodes | Rapidly surpasses RDT-1B | Slow improvement |
| 50 episodes | Significantly ahead | Still catching up |

TwinVLA exhibits a steep learning curve, surpassing RDT-1B—which requires large-scale pretraining data—with only 50 demonstrations.

### Robustness and Language Following

| Scenario | RDT-1B | π0 | TwinVLA |
|---------|--------|-----|---------|
| Low lighting (Fold towel) | 15.0% | 40.0% | **45.0%** |
| Distractors (Fold towel) | 15.0% | **60.0%** | 25.0% |
| Language following (multi-task) | Baseline | Baseline+x | **Baseline+21.8%** |

TwinVLA is robust to lighting variations and outperforms both RDT-1B and π0 in language-following evaluations by an average of 21.8%.

## Highlights & Insights

- **Paradigm significance of "duplicate rather than retrain"**: TwinVLA demonstrates that appropriate architectural inductive biases are more effective than brute-force data collection—a 40× improvement in computational efficiency and 13× in data efficiency represents a paradigm-level leap, not an incremental gain.
- **Neuroscience-engineering correspondence**: the SMA/corpus callosum coordination mechanism in human bimanual control directly maps to TwinVLA's joint attention, with biological principles guiding the architectural design.
- **25 vs. 1,000+ GPU-days**: this gap democratizes bimanual VLA research from a small number of labs with proprietary data to any team with a modest set of bimanual demonstrations.
- **Transferability of single-arm priors**: fundamental manipulation skills (grasping, placing, moving) are shared between single-arm and bimanual settings; the Twin structure enables this transfer naturally.

## Limitations & Future Work

- Visual distribution shift: the visual inputs from both arms differ from the single-arm pretraining distribution, limiting generalization.
- Absolute end-effector (EEF) control: embodiment-agnostic but less flexible than relative action representations.
- Weaker performance under distractors (25% vs. π0's 60%).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic realization of modular bimanual VLA composition
- Experimental Thoroughness: ⭐⭐⭐⭐ Real-world + simulation + data/compute efficiency + ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation; neuroscience analogy is elegantly intuitive
- Value: ⭐⭐⭐⭐⭐ Paradigm-level impact on bimanual VLA research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](../../ICML2026/robotics/stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](../../ICML2026/robotics/seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](../../ICML2026/robotics/spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->
