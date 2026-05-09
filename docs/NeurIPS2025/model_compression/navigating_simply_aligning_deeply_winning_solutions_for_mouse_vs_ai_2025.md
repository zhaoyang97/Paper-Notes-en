---
title: >-
  [Paper Note] Navigating Simply, Aligning Deeply: Winning Solutions for Mouse vs. AI 2025
description: >-
  [NeurIPS 2025][Model Compression][Visual Robustness] In the NeurIPS 2025 Mouse vs. AI competition, this paper presents the counterintuitive finding that a lightweight two-layer CNN substantially outperforms deep networks on visual robustness tasks, while demonstrating that a deeper ResNet architecture is more advantageous for neural alignment, revealing a fundamental tension between behavioral robustness and biological plausibility.
tags:
  - NeurIPS 2025
  - Model Compression
  - Visual Robustness
  - Neural Alignment
  - Lightweight CNN
  - Gated Linear Units
  - Reinforcement Learning
date: 2026-05-08
content_hash: dffbb95d487fb6fd
---

# Navigating Simply, Aligning Deeply: Winning Solutions for Mouse vs. AI 2025

**Conference**: NeurIPS 2025
**arXiv**: [2602.00982](https://arxiv.org/abs/2602.00982)
**Code**: Unavailable
**Area**: Model Compression
**Keywords**: Visual Robustness, Neural Alignment, Lightweight CNN, Gated Linear Units, Reinforcement Learning

## TL;DR

In the NeurIPS 2025 Mouse vs. AI competition, this paper presents the counterintuitive finding that a lightweight two-layer CNN substantially outperforms deep networks on visual robustness tasks, while demonstrating that a deeper ResNet architecture is more advantageous for neural alignment, revealing a fundamental tension between behavioral robustness and biological plausibility.

## Background & Motivation

- **State of the Field**: Out-of-distribution robustness in visual navigation remains a core challenge for reinforcement learning. Biological systems (e.g., mice) maintain stable navigation performance under significant environmental variation, whereas artificial systems typically suffer sharp performance degradation when encountering visual perturbations outside the training distribution.
- **Limitations of Prior Work**: The NeurIPS 2025 Mouse vs. AI competition provides a unique benchmark for studying this robustness gap through two complementary tracks:
  - **Track 1 (Visual Robustness)**: Evaluates agent generalization under unseen visual perturbations (fog, lighting changes, etc.)
  - **Track 2 (Neural Alignment)**: Evaluates how well artificial visual representations predict the neural activity of 19,000+ neurons in the mouse visual cortex
- **Root Cause**: Following the conventional wisdom that "complex tasks require complex architectures," the authors initially explored InceptionNet, a 24-layer IMPALA ResNet, and LSTM models. These complex architectures consistently exhibited training instability, severe overfitting, and performance drops of up to 35% under perturbations.
- **Paper Goals**: These failures prompted a fundamental rethink: **can a simple architecture paired with carefully selected augmentation components achieve superior robustness?**

## Method

### Overall Architecture

The paper proposes two independent architectures optimized for each track: Track 1 uses a minimal two-layer CNN + GLU + observation normalization (1.4M parameters); Track 2 uses a 16-layer deep ResNet + GLU gating (17.8M parameters). Both are trained with PPO-based reinforcement learning.

### Key Designs

1. **Lightweight Visual Encoder (Track 1)**: Feature extraction is performed with only two convolutional layers. The first layer applies an $8 \times 8$ kernel with stride 4 to a $86 \times 155 \times 1$ grayscale input, producing 16 channels; the second applies a $4 \times 4$ kernel with stride 2, expanding to 32 channels, both using LeakyReLU (negative slope 0.2). The flattened output is projected to 256 dimensions via a fully connected layer. **Design Motivation**: A capacity-constrained shallow network cannot memorize training-specific patterns, forcing it to learn generalizable features.

2. **Gated Linear Unit (GLU) Module**: Applies selective information gating to encoded features. A feature transformation path with Swish activation and a gating path with Sigmoid activation are processed in parallel, with element-wise multiplication implementing the gate:
$$\mathbf{h}_{\text{GLU}} = \text{Swish}(\text{FC}(\mathbf{z})) \odot \sigma(\text{FC}(\mathbf{z}))$$
The gating mechanism learns to identify features that remain reliable under visual perturbations while suppressing noise-sensitive components.

3. **Observation Normalization**: Running statistics are maintained via exponential moving average, and channel-wise normalization is applied to inputs:
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{running}}}{\boldsymbol{\sigma}_{\text{running}} + \epsilon}$$
This provides invariance to global illumination changes, which are the primary source of visual perturbations in the evaluation protocol.

4. **Deep ResNet Architecture (Track 2)**: 16 convolutional layers organized into a residual structure with progressively expanding channels (64→128→256→512), with GLU gating using Softmax for feature routing. The 17.8M parameter count provides sufficient capacity to capture hierarchical visual representations that match the diverse tuning properties of the biological visual cortex.

### Loss & Training

Track 1 employs two-stage training: first training the convolutional backbone for 1,400,000 steps, then appending the GLU module from the best checkpoint and continuing for 350,000 steps. Track 2 involves systematic checkpoint analysis, saving checkpoints from 60K to 1.14M steps, with optimal performance found at approximately 200K steps rather than at convergence.

## Key Experimental Results

### Main Results — Track 1 Visual Robustness

| Architecture | ASR (%) | MSR (%) | Final Score (%) |
|---|---|---|---|
| IMPALA ResNet (24-layer) | 80.96 | 51.00 | 65.98 |
| IMPALA ResNet (4-layer) | 91.40 | 84.00 | 87.70 |
| + Data Augmentation | 72.60 | 47.00 | 59.80 |
| SimpleCNN (Ours) | 94.20 | 89.00 | 91.60 |
| SimpleCNN + GLU | 95.60 | 88.00 | 91.80 |
| **SimpleCNN + GLU + Norm** | **96.80** | **94.00** | **95.40** |

### Ablation Study — Track 1 Component Contributions

| Configuration | ASR (%) | MSR (%) | Final Score (%) | Notes |
|---|---|---|---|---|
| Full Model | 96.80 | 94.00 | 95.40 | All components |
| w/o Normalization | 95.60 | 88.00 | 91.80 | Norm contributes +3.6 pp |
| w/o GLU | 94.20 | 89.00 | 91.60 | GLU contributes +0.2 pp |
| w/o Both | 94.20 | 89.00 | 91.60 | Baseline |

### Key Findings

- **Depth is harmful**: The 24-layer ResNet exhibits a 30 percentage-point gap between ASR and MSR (80.96% vs. 51.00%), indicating that deep networks overfit to the visual patterns of the training distribution.
- **Data augmentation backfires**: Applying data augmentation to the ResNet causes performance to collapse from 87.70% to 59.80%, a drop of 27.9 percentage points.
- **Non-monotonic relationship between training duration and performance**: In Track 2, the 200K-step checkpoint nearly matches the best model at 1.14M steps (0.1507 vs. 0.1517), a difference of only 0.66%.
- **InceptionNet completely fails**: The multi-scale convolutional architecture fails to converge within 500K steps; LSTM models also exhibit training instability.
- **Parameter count contrast**: Track 1 requires only 1.4M parameters, while Track 2 requires 17.8M (a 12.8× difference), reflecting fundamentally different capacity requirements for the two objectives.

## Highlights & Insights

- The paper's central insight is that **behavioral robustness and biological plausibility require fundamentally different architectural choices**: simple models provide robustness (preventing overfitting), while deep models provide representational richness (matching neural responses).
- The effect of observation normalization is remarkable for its simplicity, contributing 3.8 percentage points to the final score on its own.
- The systematic documentation of failed approaches (InceptionNet, deep ResNet, LSTM, data augmentation) is highly valuable.

## Limitations & Future Work

- **Task specificity**: Results may not generalize to more complex visual navigation scenarios.
- **Metric limitations**: Linear readout and representational similarity capture only partial aspects of biological vision.
- **Unexplored architectures**: Transformer-based designs and other alternatives remain uninvestigated.
- **Unimodal input**: The competition environment provides only visual input, leaving multisensory integration unexplored.

## Related Work & Insights

This work is closely related to domain randomization, the IMPALA architecture, and GLU/SwiGLU. The "simple architecture + targeted augmentation" paradigm offers broad inspiration for efficient model design: in resource-constrained settings or scenarios requiring robustness, reducing model complexity may in fact be the correct strategy.

The Track 2 findings are consistent with the hierarchical organization of the visual cortex in neuroscience — multi-scale representations from V1 to higher visual areas require sufficient model capacity to approximate. Although SimpleCNN achieves excellent behavioral performance, its neural alignment rank of 13th suggests that task-optimal representations and biologically plausible representations may constitute **distinct optimization objectives**. The non-monotonic relationship between training duration and performance is also noteworthy — the 200K-step checkpoint nearly matches the 1.14M-step model, suggesting that practical development workflows should systematically evaluate multiple intermediate checkpoints rather than relying solely on the final model.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Competition-driven empirical findings with valuable counterintuitive conclusions
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations with thorough documentation of failed approaches
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and in-depth analysis
- **Value**: ⭐⭐⭐⭐ Practically informative for visual RL and efficient architecture design

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)
- [\[ICLR 2026\] Textual Equilibrium Propagation for Deep Compound AI Systems](../../ICLR2026/model_compression/textual_equilibrium_propagation_for_deep_compound_ai_systems.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/iapl_aigenerated_image_detection_adaptive_prompt.md)

<!-- RELATED:END -->
