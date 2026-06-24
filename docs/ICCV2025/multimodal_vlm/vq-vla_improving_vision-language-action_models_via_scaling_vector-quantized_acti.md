---
title: >-
  [Paper Note] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
description: >-
  [ICCV2025][Multimodal VLM][Action tokenizer] This paper proposes an action tokenizer based on a convolutional Residual VQ-VAE (RVQ-VAE). Trained on 100x more training data (including a large amount of synthetic data) compared to prior methods, it enables zero-shot transfer to various downstream VLA tasks, improving the success rate of long-horizon tasks on real robots by up to 30% and increasing inference speed by nearly 3x.
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "Action tokenizer"
  - "VQ-VAE"
  - "synthetic data scaling"
  - "long-horizon tasks"
  - "robotic manipulation"
date: 2026-05-08
content_hash: c11444919f446875
---

# VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers

**Conference**: ICCV2025  
**arXiv**: [2507.01016](https://arxiv.org/abs/2507.01016)  
**Code**: [https://github.com/VQ-VLA](https://github.com/VQ-VLA) (To be confirmed)  
**Area**: Robotics / VLA / Action Representation  
**Keywords**: Action tokenizer, VQ-VAE, synthetic data scaling, long-horizon tasks, robotic manipulation

## TL;DR
This paper proposes an action tokenizer based on a convolutional Residual VQ-VAE (RVQ-VAE). Trained on 100x more training data (including a large amount of synthetic data) compared to prior methods, it enables zero-shot transfer to various downstream VLA tasks, improving the success rate of long-horizon tasks on real robots by up to 30% and increasing inference speed by nearly 3x.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models combine vision-language understanding with robot control. Current mainstream approaches (such as OpenVLA and RT-2) adapt to the token prediction framework of LLMs by discretizing continuous actions into 256 bins (binning each dimension independently).

**Limitations of Prior Work**:
   - Simple per-dimension binning discretization has limited precision, leading to cumulative error propagation and failure in long-horizon tasks.
   - Predicting only one action token per step results in a slow inference speed (OpenVLA runs at only 4.16 Hz).
   - The quality of action representation is constrained by the scale and diversity of the training data, while scaling the entire VLA model remains extremely expensive.

**Key Challenge**: VLA models require high-precision and high-efficiency action representations, but simple bin discretization imposes a bottleneck on both accuracy and sequence compression. Moreover, training the entire VLA model is prohibitively expensive, requiring a low-cost pathway for performance enhancement.

**Goal**: Design a scalable and generalizable action tokenizer to (a) improve action representation accuracy and long-horizon robustness, (b) accelerate inference through action chunking, and (c) scale up training data at low cost using synthetic data.

**Key Insight**: The authors observe a critical phenomenon—**the domain gap of action trajectories between simulation and the real world is negligible** (unlike images or physical properties). Therefore, synthetic trajectories can be extensively utilized to train the tokenizer without sacrificing real-world performance.

**Core Idea**: Employ a convolutional Residual VQ-VAE as the action tokenizer. By training it on a 100x larger scale of mixed data (real and synthetic) using a progressive strategy, simultaneous improvements in VLA accuracy, speed, and long-horizon capabilities are achieved.

## Method

### Overall Architecture
The pipeline of VQ-VLA consists of two phases:
- **Phase 1**: Train a general-purpose convolutional Residual VQ-VAE action tokenizer. The input is an action sequence of length $n$, denoted as $\mathbf{a}_{t:t+n} \in \mathbb{R}^{n \times d}$ (7 dimensions: XYZ + Euler angles + gripper), and the output consists of $N_q$ discrete tokens (corresponding to $N_q$ layers of RVQ quantization).
- **Phase 2**: Freeze the VQ-VAE and substitute the original binning scheme of OpenVLA with this action tokenizer, fine-tuning OpenVLA using LoRA.

### Key Designs

1. **Convolutional Residual VQ-VAE Architecture**:

    - **Function**: Encodes continuous action sequences into discrete tokens and reconstructs them back during decoding.
    - **Mechanism**: The encoder and decoder utilize 2D temporal convolutional layers (instead of MLPs) to better capture local temporal relationships and hierarchical temporal dependencies. Residual Vector Quantization (RVQ) decomposes the latent variable into multi-layer quantization: $\mathbf{q}(\mathbf{x}) = \sum_{i=1}^{N_q} \mathbf{q}_i(\mathbf{r}_i)$, where each layer refines the residual of the previous layer.
    - **Training Loss**: $\mathcal{L} = \|\mathbf{a} - \hat{\mathbf{a}}\|_2^2 + \lambda(\|\text{sg}(\mathbf{x}) - \mathbf{q}(\mathbf{x})\|_2^2 + \|\mathbf{x} - \text{sg}(\mathbf{q}(\mathbf{x}))\|_2^2)$, with $\lambda=4$.
    - **Design Motivation**: 2D temporal convolutions improve the success rate on LIBERO from 53.4% to 60% compared to MLPs, indicating that local temporal modeling is crucial.

2. **Time Embeddings + Action Type Embeddings**:

    - **Function**: Adds two types of embeddings before feeding the action sequence into the encoder.
    - **Mechanism**: Sinusoidal temporal embeddings encode temporal information across different frequencies, while learnable action type embeddings differentiate the semantic roles of the 7 dimensions (XYZ, Euler angles, gripper).
    - **Design Motivation**: The 7 dimensions of the action vector have varying semantics and require prior information to help the model process them distinctly.

3. **Progressive Training Strategy + Synthetic Data Scaling**:

    - **Function**: Progressively scales the training resources from real-world data to synthetic data.
    - **Mechanism**: First, the model is trained on Open X-Embodiment (real-world but noisy data), and then simulation data from LIBERO and ManiSkill (cleaner and smoother) are progressively introduced. Three versions are developed: VQ_O (OpenX only), VQ_{O+L} (+LIBERO), and VQ_{O+L+M} (+ManiSkill).
    - **Design Motivation**: The authors find that the domain gap of action trajectories between sim-real is negligible (the performance of VQ_L trained purely on simulation matches that of VQ_{O+L}), which justifies the large-scale use of synthetic data.

4. **Integration of VQ-VAE and VLA**:

    - **Function**: Replaces the binning tokens of OpenVLA with the discrete tokens from VQ-VAE.
    - **Mechanism**: Token IDs across different RVQ layers use non-overlapping ranges—the token at the $i$-th layer $z_q^i \in [256(i-1), 256i-1]$, preventing semantic confusion between different layers. The VLM directly predicts these tokens using standard next-token cross-entropy loss.
    - **Design Motivation**: Implements action chunking with a compression ratio of 5 (predicting 5 actions in a single step), greatly reducing the number of inference steps.

### Loss & Training
- VQ-VAE training: reconstruction loss + VQ loss + commitment loss, with $\lambda=4$
- VLA fine-tuning: standard next-token prediction cross-entropy, fine-tuned with LoRA for 400K steps (simulation) or 100K steps (real-world)

## Key Experimental Results

### Main Results (LIBERO-90)

| Method | Training Data | LIBERO-90 Success Rate |
|------|---------|-----------------|
| OpenVLA baseline | - | 73.53% |
| VQ_M (ManiSkill only) | ManiSkill | 14.38% |
| VQ_{M+R} (ManiSkill+RLBench) | ManiSkill+RLBench | **80.98%** |

VQ_{M+R} improves upon the baseline by 7.45%. When data is insufficient (VQ_M), performance drops significantly, validating the importance of data scale.

### Real-Robot Experiments

| Task | Baseline | VQ_O | VQ_{O+L} | VQ_{O+L+M} |
|------|----------|------|----------|------------|
| Pull tissue | 5% | 20% | 25% | 25% |
| Pick toy (avg 3) | 30% | 46.7% | 43.3% | 50% |
| Flip pot upright | 30% | 45% | 45% | **60%** |
| Put toy in basket | 20% | 35% | 35% | **45%** |
| Put cups in basket (Long-horizon) | 15% | - | - | **50%** |
| Put toy in drawer (Long-horizon) | ~0% | 15% | 10% | **25%** |

### Ablation Study

| Action Chunking Method | LIBERO-90 | Flip pot | Put in basket |
|---------------------|-----------|----------|---------------|
| Baseline (Single-step) | 74.76% | 30% | 20% |
| Autoregressive Output | 66.53% | 10% | 0% |
| VQ-based (VQ_{O+L+M}) | **86.61%** | **60%** | **45%** |

Autoregressive action chunking leads to a severe performance drop (due to the shortcut learning phenomenon—multiple action values within a chunk are highly similar), while VQ-based chunking performs the best.

### Inference Speed

| Method | Frequency (Hz) |
|------|----------|
| OpenVLA | 4.16 |
| VQ-VLA | **11.84** |

Inference speed is improved by approximately 2.85x.

### Key Findings
- **Synthetic data scaling is effective**: The ManiSkill dataset is 50x larger than LIBERO. Adding ManiSkill improves the short-horizon average success rate from 37.5% to 46.25%.
- **Sim-to-real domain gap is negligible**: VQ_L trained purely on simulation achieves comparable performance to VQ_{O+L} (Flip pot: 55% vs 45%).
- **Long-horizon tasks benefit the most**: VQ-based action chunking reduces cumulative errors, improving "Put cups in basket" success rate from 15% to 50%.
- **Embeddings are helpful**: Integrating temporal and action type embeddings improves the LIBERO-90 success rate from 85.17% to 86.16%.

## Highlights & Insights
- **The insight that "the sim-to-real gap for action trajectories is negligible" is a profound finding**: Unlike modalities such as images or physical properties, the statistical distribution of action trajectories is highly consistent between simulation and the real world. This implies that low-cost simulation data can be used to improve action representation quality, providing an efficient pathway to enhance VLA performance.
- **VQ-based action chunking outperforms autoregressive chunking**: This reveals that autoregressive generation in LLMs is prone to shortcut learning on low-dimensional continuous signals (actions), whereas explicit compression-decompression via VQ preserves variations within the sequence much better.
- **Extremely low training cost for the tokenizer**: It only takes one week of training on a single A100 GPU to bring consistent performance and speed improvements to downstream VLAs. This "small component, large gain" approach can be transferred to tokenizer designs for other modalities.

## Limitations & Future Work
- **Validated only on OpenVLA**: Although architectural generalizability is claimed, only one VLA's tokenizer was replaced. Further validation is required across more VLAs (e.g., RT-2, Octo).
- **Action space limited to 7-DoF SE(3)**: Higher-dimensional action spaces, such as dextrous hands, are not addressed.
- **Lack of direct comparison with other action tokenizers**: For example, cosine transform methods (such as FAST).
- **Future work**: (a) Incorporate frequency characteristics of action data as an additional conditioning factor; (b) combine with VLM distillation/quantization to further accelerate inference; (c) expand to larger-scale simulation datasets (e.g., RLBench on CoppeliaSim).

## Related Work & Insights
- **vs OpenVLA (binning)**: OpenVLA applies 256-bin discretization to each action dimension, limiting precision and predicting only one action per step. VQ-VLA employs RVQ to achieve finer quantization and predicts five actions in a single step via action chunking.
- **vs FAST (cosine transform)**: FAST uses cosine transforms for action tokenization, which is an alternative technical route. This paper does not directly compare with it, but their ideas are complementary.
- **vs MiniVLA**: MiniVLA also focuses on VLA efficiency but approaches it from the perspective of model compression. VQ-VLA enhances efficiency from the perspective of action representation; the two approaches can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of convolutional Residual VQ-VAE and synthetic data scaling is novel, and the finding regarding the sim-to-real gap is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full coverage of both simulation and real-world experiments with comprehensive ablations, though it lacks direct comparison with methods like FAST.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, but the mathematical notation in some parts lacks consistency.
- Value: ⭐⭐⭐⭐ Provides a low-cost performance improvement pathway for VLA with high practical value.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](../../ICML2026/multimodal_vlm/vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](../../CVPR2026/multimodal_vlm/from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](../../CVPR2026/multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)

</div>

<!-- RELATED:END -->
