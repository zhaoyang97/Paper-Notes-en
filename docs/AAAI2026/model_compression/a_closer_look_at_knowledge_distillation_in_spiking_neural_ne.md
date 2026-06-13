---
title: >-
  [Paper Note] A Closer Look at Knowledge Distillation in Spiking Neural Network Training
description: >-
  [AAAI 2026][Model Compression][Knowledge Distillation] To address the overlooked distribution mismatch between teacher ANN continuous features/logits and student SNN discrete sparse spike features/logits in ANN→SNN knowl…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Spiking Neural Networks"
  - "Activation Map Alignment"
  - "Noise Smoothing"
  - "Energy-Efficient Training"
date: 2026-05-08
content_hash: 657556c450ce4dfb
---

# A Closer Look at Knowledge Distillation in Spiking Neural Network Training

**Conference**: AAAI 2026
**arXiv**: [2511.06902](https://arxiv.org/abs/2511.06902)  
**Code**: [https://github.com/SinoLeu/CKDSNN](https://github.com/SinoLeu/CKDSNN)  
**Area**: Interpretability
**Keywords**: Knowledge Distillation, Spiking Neural Networks, Activation Map Alignment, Noise Smoothing, Energy-Efficient Training

## TL;DR

To address the overlooked distribution mismatch between teacher ANN continuous features/logits and student SNN discrete sparse spike features/logits in ANN→SNN knowledge distillation, this paper proposes the CKDSNN framework based on Saliency-scaled Activation Map Distillation (SAMD) and Noise-smoothed Logits Distillation (NLD), achieving new state-of-the-art SNN training performance on CIFAR-10/100, ImageNet-1K, and CIFAR10-DVS.

## Background & Motivation

Spiking Neural Networks (SNNs), inspired by biological neurons, transmit information via event-driven binary spikes, replacing multiply-accumulate operations with additions and offering substantial energy efficiency on neuromorphic hardware (e.g., Intel Loihi). However, SNN training faces two primary challenges: (1) ANN-to-SNN conversion requires a large number of timesteps to preserve accuracy; (2) direct training methods reduce timesteps but still exhibit a notable accuracy gap relative to ANNs due to surrogate gradient estimation errors.

Recent work has introduced knowledge distillation (KD) using pretrained ANNs as teachers and SNNs as students to improve SNN training quality with some success. Nevertheless, existing methods (e.g., KDSNN, BKDSNN) overlook two critical issues when performing KD:

1. **Feature distribution mismatch**: Intermediate features of ANNs are continuous floating-point values, whereas SNN features are discrete binary spikes (0/1) accumulated over multiple timesteps, naturally concentrated in salient regions—resulting in an inherent distributional incompatibility.
2. **Logits distribution mismatch**: SNN classification logits exhibit sparser and sharper distributions due to their binary feature origins, differing significantly from the continuous, smooth logits distributions of ANNs.

Existing methods apply naive element-wise alignment, ignoring these fundamental differences and yielding suboptimal distillation.

## Core Problem

How to effectively bridge the distributional gap—at both the feature and output levels—between the teacher (continuous floating-point features + smooth logits) and the student (discrete binary spike features + sparse logits) in ANN→SNN knowledge distillation, so as to make distillation genuinely effective.

## Method

### Overall Architecture

The CKDSNN framework takes a pretrained ANN teacher and a trainable SNN student, performing knowledge distillation at two complementary levels:
- **Feature level**: SAMD distills the Class Activation Maps (CAM) of the teacher ANN into the Spike Activation Maps (SAM) of the student SNN, rather than directly aligning raw features.
- **Logits level**: NLD smooths the sparse student SNN logits with Gaussian noise to bring their distribution closer to the teacher ANN's continuous logits before alignment.

Total loss = standard cross-entropy loss + $\beta$ · SAMD loss + $\gamma$ · NLD loss.

### Key Designs

1. **Saliency-scaled Activation Map Distillation (SAMD)**: Completed in three steps.

    - **CAM generation (teacher side)**: Grad-CAM is applied to the pretrained ANN to extract class activation maps $M^{te} \in \mathbb{R}^{H \times W}$, obtaining spatially salient regions relevant to the target class via gradient-weighted channel summation.
    - **SAM generation (student side)**: Since surrogate gradient estimation errors in SNNs render Grad-CAM-style methods inaccurate, the paper takes an alternative approach by directly leveraging the SNN's own spikes—summing spikes across all timesteps and channel dimensions to obtain the spike activation map $M^{st} = \sum_{t}\sum_{c} F^{st}_{t,c}$. This exploits the natural property of SNNs: spikes are inherently generated only in salient regions.
    - **Saliency-scaled alignment**: Although CAM and SAM are semantically consistent (both highlighting salient regions), they differ greatly in numerical magnitude (one derived from floating-point weighting, the other from binary accumulation). The paper converts both to probability distributions $P^{te}$ and $P^{st}$ via softmax with temperature parameter $\mathcal{T}$, then aligns them using KL divergence loss: $\mathcal{L}_{SAMD} = \mathcal{T}^2 \cdot KL(P^{te} \| P^{st})$.

2. **Noise-smoothed Logits Distillation (NLD)**: SNN logits exhibit sparse and sharp distributions due to binary features, making direct alignment with smooth ANN logits ineffective. The core of NLD is to "soften" SNN logits using **adaptive Gaussian noise**, whose mean and standard deviation are derived directly from the statistics of the SNN logits themselves ($\epsilon \sim \mathcal{N}(\bar{z}^{st}, \sigma(z^{st})^2)$), preserving the original distributional characteristics while increasing continuity. The fused logits are $z^{soft} = z^{st} + \lambda \epsilon$, then aligned with teacher logits via softmax and KL divergence: $\mathcal{L}_{NLD} = \tau^2 \cdot KL(y^{te} \| y^{soft})$.

3. **Architecture-agnostic SAM**: Unlike CATKD and similar activation map distillation methods that are restricted to CNN architectures, SAM generation is independent of gradient information (unaffected by surrogate gradient errors) and does not rely on any specific network structure, making it applicable to both Spiking CNNs (e.g., ResNet) and Spiking Transformers (e.g., Spikformer).

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{SAMD} + \gamma \cdot \mathcal{L}_{NLD}$$

Hyperparameter settings: $\mathcal{T}=2.0$, $\tau=2.0$, $\lambda=0.1$, $\beta=1.0$, $\gamma=1.0$. SAMD is applied at the last stage of the network (deep features carry more precise semantic information). SGD optimizer (momentum 0.9, weight decay 1e-4) with Cosine Annealing learning rate scheduling is used.

## Key Experimental Results

| Dataset | Architecture | Timesteps | Ours (CKDSNN) | Prev. SOTA | Gain |
|---------|--------------|-----------|---------------|------------|------|
| CIFAR-10 | ResNet-19 | T=1 | 96.11% | 95.37% (EnOF) | +0.74% |
| CIFAR-100 | ResNet-19 | T=1 | 79.11% | 77.08% (EnOF) | +2.03% |
| CIFAR-10 | ResNet-19 | T=4 | 97.81% | 96.19% (EnOF) | +1.62% |
| CIFAR-100 | ResNet-19 | T=4 | 83.88% | 82.43% (EnOF) | +1.45% |
| CIFAR-100 | Spikformer-4-384 | T=1 | 83.07% | 81.26% (BKDSNN) | +1.81% |
| ImageNet-1K | SEW-R34 | T=4 | 73.05% | 71.24% (BKDSNN) | +1.81% |
| CIFAR10-DVS | ResNet-20 | T=10 | 81.55% | 80.50% (EnOF) | +1.05% |

Regarding energy efficiency: on ImageNet-1K with ResNet-34, CKDSNN (T=2) achieves 71.33% accuracy with only 8.0% firing rate and 3.61W power consumption, whereas BKDSNN (T=4) requires 15.0% firing rate and 3.98W to achieve 71.24%—CKDSNN surpasses the previous state-of-the-art with fewer timesteps and lower power.

### Ablation Study

- **Both SAMD and NLD are indispensable**: Removing either component leads to significant performance drops across all settings, confirming that feature-level and logits-level distillation are complementary.
- **Saliency scaling approach**: Softmax scaling (79.11%) substantially outperforms no scaling (75.56%), Z-score normalization (76.48%), and L2-norm (74.78%) by approximately 2–3 points. Softmax effectively normalizes and identifies the most salient regions.
- **CAM-SAM vs. conventional activation map KD**: Applying ANN-style gradient-based activation map methods (e.g., e2KD, CATKD using CAM-CAM) to SNNs performs significantly worse than the proposed CAM-SAM approach, validating the impact of surrogate gradient errors on Grad-CAM in SNNs. CATKD is also constrained to CNN architectures, while SAM is architecture-agnostic.
- **Adaptive noise vs. fixed noise**: Adaptive Gaussian noise (NLD) substantially outperforms random noise with any fixed standard deviation. Small noise is ineffective; large noise is detrimental. NLD adaptively preserves distributional characteristics by deriving noise parameters from the logits themselves.
- **SAMD application stage**: The last stage yields the best results (79.11%), with performance decreasing for earlier stages (Stage 1: 77.01%).
- **Loss landscape**: Models trained with CKDSNN exhibit flatter loss landscapes with fewer saddle points, facilitating convergence to better local optima.

## Highlights & Insights

- **Generating SAM from SNN's intrinsic spike properties rather than forcing Grad-CAM** is the core insight. Spikes in SNNs are naturally generated only in salient regions; directly accumulating spikes constructs meaningful activation maps, elegantly circumventing the surrogate gradient error problem. The idea is simple, elegant, and computationally efficient.
- **Adaptive noise-smoothed logits** is a practically sound design: noise statistics are derived from the logits themselves, softening the distribution while preserving original characteristics, with theoretical justification grounded in the maximum entropy principle.
- **The two strategies are highly complementary**—SAMD addresses feature-level distributional mismatch while NLD handles logits-level mismatch, resolving the ANN-SNN distillation alignment problem from two distinct perspectives.
- **Architecture agnosticism**: SAM generation is independent of gradients and specific architectures, applicable to both Spiking CNNs and Spiking Transformers, offering strong generalizability.
- A better balance between accuracy and energy efficiency is achieved: prior methods' full-timestep performance is surpassed with fewer timesteps.

## Limitations & Future Work

- **Only image classification tasks are evaluated**: The KD effectiveness of SNNs on NLP, multimodal, and other tasks remains unexplored, and cross-task generalizability of the method requires further validation.
- **CAM accuracy depends on teacher quality**: When the teacher ANN's CAM itself is inaccurate (e.g., on difficult samples), it may mislead the student's SAM learning. Although NLD is found to partially mitigate this issue, the fundamental limitation imposed by teacher quality remains.
- **Distillation is applied only at the last stage**: Progressive multi-level distillation may further improve performance, but experiments show that shallow stages yield weaker results, potentially requiring stage-specific alignment strategies.
- **Noise hyperparameter $\lambda$ still requires manual tuning**: Although noise distribution parameters are adaptive, the fusion weight is fixed; dynamically adjusting $\lambda$ during training could be explored.
- **Temporal information utilization**: SAM naively sums over the temporal dimension, discarding temporal dynamics. Spike importance may vary across timesteps, and temporal attention mechanisms could be incorporated.

## Related Work & Insights

- **vs. KDSNN (CVPR'23)**: KDSNN directly performs element-wise alignment of ANN and SNN features and logits, entirely ignoring distributional differences. CKDSNN explicitly addresses this mismatch through SAMD and NLD, achieving substantially higher performance across all datasets (e.g., 66.92% vs. 63.42% on ImageNet).
- **vs. BKDSNN (ECCV'24)**: BKDSNN processes SNN spike features via blur matrices to improve feature matching, but still operates at the raw feature level. CKDSNN elevates distillation to the semantic activation map level, achieving stronger semantic consistency. ImageNet: 73.05% vs. 71.24%.
- **vs. EnOF (NeurIPS'24)**: EnOF improves SNN training by enhancing output features. CKDSNN approaches the problem from the perspective of distillation alignment, proposing a more fundamental solution—79.11% vs. 77.08% on CIFAR-100 at T=1.
- **vs. ANN activation map KD methods (e2KD, CATKD)**: These methods are designed for ANN-to-ANN distillation; directly applying them to SNNs yields poor results due to surrogate gradient errors. CKDSNN designs the SAM alternative specifically for SNN characteristics.

The SAM design philosophy is transferable: directly leveraging a model's intrinsic properties (e.g., spike distributions in SNNs) to construct distillation signals—rather than relying on general but ill-suited methods such as Grad-CAM—represents a "domain-appropriate" paradigm generalizable to distillation between other heterogeneous model pairs. The noise-smoothed logits approach is worth exploring in distillation for other quantized or sparse models, as any model producing sparse distributions may benefit from similar strategies. The CAM-SAM framework may also be applicable beyond distillation, serving as feature representations in self-supervised or contrastive learning for SNNs. Combining distillation training with quantized SNNs (e.g., binary SNNs) or mixed-precision SNNs is a promising cross-cutting direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The insight of leveraging SNN spike properties to design SAM as a replacement for Grad-CAM is inspiring, though the overall framework remains a combination of feature KD and logits KD paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three static datasets and one neuromorphic dataset, multiple architectures (CNN + Transformer), with comprehensive ablation studies, energy efficiency analysis, visualizations, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is systematic; the theoretical analysis section is slightly lengthy but enhances persuasiveness.
- Value: ⭐⭐⭐⭐ Clear contribution to the SNN-KD field with significant state-of-the-art improvements, though impact is limited to the relatively niche direction of SNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](../../ICLR2026/model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[AAAI 2026\] Condensed Data Expansion Using Model Inversion for Knowledge Distillation](condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)

</div>

<!-- RELATED:END -->
