---
title: >-
  [Paper Note] UniSTD: Towards Unified Spatio-Temporal Learning Across Diverse Disciplines
description: >-
  [CVPR 2025][Self-Supervised Learning][Unified Spatiotemporal Learning] The authors propose the UniSTD framework, which utilizes a standard Transformer combined with a rank-adaptive mixture-of-experts (RA-MoE) and a lightweight temporal module. This design enables a single model to simultaneously handle 10 spatio-temporal predictive tasks across 4 diverse disciplines without performance loss, outperforming existing joint-training methods by 18.8 PSNR in multi-task scenarios.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Unified Spatiotemporal Learning"
  - "Mixture-of-Experts"
  - "Low-Rank Adaptation"
  - "Multi-Task Learning"
  - "Transformer"
date: 2026-05-08
content_hash: 7537a5e5373762fa
---

# UniSTD: Towards Unified Spatio-Temporal Learning Across Diverse Disciplines

**Conference**: CVPR 2025  
**arXiv**: [2503.20748](https://arxiv.org/abs/2503.20748)  
**Code**: [https://github.com/1hunters/UniSTD](https://github.com/1hunters/UniSTD)  
**Area**: Spatiotemporal Predictive Learning / Self-Supervised/Unified Learning  
**Keywords**: Unified Spatiotemporal Learning, Mixture-of-Experts, Low-Rank Adaptation, Multi-Task Learning, Transformer

## TL;DR

The authors propose the UniSTD framework, which utilizes a standard Transformer combined with a rank-adaptive mixture-of-experts (RA-MoE) and a lightweight temporal module. This design enables a single model to simultaneously handle 10 spatio-temporal predictive tasks across 4 diverse disciplines without performance loss, outperforming existing joint-training methods by 18.8 PSNR in multi-task scenarios.

## Background & Motivation

**Background**: Spatiotemporal predictive learning aims to predict future frames based on historical frame sequences and is widely applied in traffic management, weather forecasting, motion prediction, driving scenario prediction, and more. Existing methods can be categorized into recurrent-based architectures (ConvLSTM, PredRNN, etc.) and non-recurrent ones (SimVP, TAU, EarthFormer, etc.). However, almost all of them are designed as task-specific architectures for a single task.

**Limitations of Prior Work**: Each task requires customized architecture design and heavily relies on domain-specific knowledge. These task-specific models exhibit unstable performance when transferring across tasks, and deploying them in practice requires maintaining a separate, independent model for each task, causing massive computational and storage overhead. Even when attempting to jointly train merely 3 tasks, existing methods like SimVP experience significant performance degradation.

**Key Challenge**: Data patterns across different disciplines (e.g., weather forecasting vs. traffic control vs. motion prediction) vary drastically. Forcing them to share a single model easily triggers inter-task conflicts and leads to sub-optimal convergence, whereas maintaining a separate model for each task lacks scalability.

**Goal**: To design a unified framework that simultaneously supports multiple cross-disciplinary spatiotemporal predictive tasks and avoids performance degradation as the number of tasks increases.

**Key Insight**: Leverage the two-stage paradigm of "large-scale pre-training + downstream adaptation" commonly used in LLMs/VLMs—using task-agnostic pre-trained weights (such as OpenCLIP-ViT weights) to provide a general foundation, and then utilizing Parameter-Efficient Fine-Tuning (LoRA) + Mixture-of-Experts to inject domain knowledge for specific spatiotemporal tasks.

**Core Idea**: Build a rank-adaptive MoE (RA-MoE) mechanism on top of a standard Transformer. Through continuous relaxation, this allows the LoRA rank of each expert to be optimized in a differentiable manner, enabling different tasks to automatically obtain dedicated adapters with varying capacities. Simultaneously, introduce a lightweight temporal attention module to compensate for the lack of temporal modeling in 2D pre-training.

## Method

### Overall Architecture

The framework adopts an Encoder-Transformer-Decoder architecture. The Encoder uses 2D convolutions to unify the inputs of various tasks (which have different channel numbers, resolutions, and temporal lengths) into a $B \times N \times L$ format to be fed into the Transformer. The Transformer uses pre-trained weights (OpenCLIP-ViT / ImageNet-ViT) with frozen main parameters, specialized for training via the RA-MoE adapter and a lightweight temporal module. The Decoder then uses transposed convolutions to upsample back to the original resolution.

### Key Designs

1. **Rank-Adaptive Mixture-of-Experts (RA-MoE)**:

    - **Function**: Dynamically allocates low-rank adapters of varying capacities to different tasks/disciplines, balancing multi-task conflicts.
    - **Mechanism**: Several LoRA adapters are connected in parallel to the Q/K/V/Proj matrices of the Transformer to serve as experts, whose weights are adaptively allocated by a dynamic router $\mathcal{G}$ based on the input. The key innovation is that the rank $r$ of each expert can also be optimized. Specifically: (1) The LoRA product $\mathbf{AB}$ is reformulated as $\mathbf{A} \mathbf{I}_{r-1,r} \mathbf{B}$, where the rank is precisely controlled by regulating the non-zero elements on the diagonal of the identity matrix; (2) Fractional interpolation is employed to relax discrete ranks into a continuous variable $f_r(\mathbf{x}) = (\lceil r \rceil - r) g_{\lceil r \rceil}(\mathbf{x}) + (r - \lfloor r \rfloor) g_{\lfloor r \rfloor}(\mathbf{x})$, enabling differentiable optimization of the rank; (3) An L1 regularization term $|C - \sum_i r_i|$ is used to control the total parameter budget. The ranks are optimized during the first 10 epochs and then rounded and fixed.
    - **Design Motivation**: Experimental observations show that different tasks exhibit vastly different weight update patterns on Q/K/V (as shown in Figure 3), indicating that different tasks require adapters of varying capacities. Using a fixed rank is sub-optimal, while brute-forcing the combination space ($4^{5 \times 10}$) is intractable. Formulating the combinatorial optimization problem as a differentiable optimization via continuous relaxation is elegant and highly efficient.

2. **Lightweight Temporal Attention Module**:

    - **Function**: Injects temporal modeling capability into the 2D pre-trained Transformer that originally only models spatial dimensions.
    - **Mechanism**: A lightweight module is inserted after the self-attention layer. First, global average pooling compresses $\mathbb{R}^{N \times L}$ to $\mathbb{R}^{1 \times L}$ (where $L = T_i \times C_i'$ contains the temporal dimension), followed by processing through an FFN (down-projection factor of 6) + 1D RA-MoE layer, which is then added to the original sequence via a Sigmoid function. The second MLP layer employs a zero-initialization strategy to prevent disrupting the pre-trained state.
    - **Design Motivation**: The FFN layers of a Transformer are essentially mixers along the last dimension, which in UniSTD corresponds to the temporal dimension. Therefore, there is no need to fine-tune the computationally expensive FFN layers; instead, a lightweight extra temporal module is sufficient. Zero-initialization ensures that the spatial modeling capacity from pre-training is not disrupted in the early phases of training.

3. **Two-Stage Pre-training & Adaptation Paradigm**:

    - **Function**: Provides a general knowledge foundation for spatiotemporal learning.
    - **Mechanism**: In the first stage, the model is pre-trained on task-agnostic large-scale datasets (OpenCLIP-ViT pre-trained on image-text data / ImageNet-ViT pre-trained on image classification) to obtain general visual representations. In the second stage, the RA-MoE adapters and temporal modules are jointly trained on multiple spatiotemporal tasks while freezing the main parameters of the Transformer. Sinusoidal Position Embeddings (SPE) are utilized instead of learnable position embeddings to maintain cross-resolution generalization.
    - **Design Motivation**: Spatiotemporal data is relatively scarce, and data scales across different tasks vary significantly. Utilizing the general features provided by large-scale pre-training is more stable than training from scratch, while allowing the standard Transformer architecture to replace task-specific designs.

### Loss & Training

The overall loss is $\mathcal{L} = \mathcal{L}_{MSE} + \beta |C - \sum_i r_i|$, where MSE is the mean squared error between the predictions and ground truths, and the second term is the rank constraint regularization ($\beta=1$). ViT-Base (12 layers, 768 dimensions, 12 heads) is utilized. There are 6 experts for Q/K/V per layer and 2 experts for Proj and the temporal module. The initial rank is 4.5. Optimization is performed using the AdamW optimizer with a learning rate of 0.01, weight decay of 0.05, 90 training epochs, and a batch size of 16.

## Key Experimental Results

### Main Results

The 10 evaluation tasks cover 4 disciplines: traffic control (TaxiBJ, Traffic4Cast), trajectory/robotics (MMNIST, BAIR, Human3.1M, KTH), driving scenarios (Cityscapes, KITTI), and weather forecasting (SEVIR, ENSO).

| Method | TaxiBJ PSNR | MMNIST PSNR | Human3.1M PSNR | KITTI PSNR | Joint Tasks |
|------|-------------|-------------|-----------------|------------|-----------|
| SimVPv2 (3 Tasks) | 24.6 | 15.7 | 22.4 | - | 3 |
| SimVPv2 (5 Tasks) | 20.4 | 12.1 | 17.6 | 14.7 | 5 |
| **Ours (UniSTD - 10 Tasks)** | **33.2** | **19.2** | **33.3** | **19.7** | **10** |

### Ablation Study

| Configuration | TaxiBJ PSNR | Human PSNR |
|------|-------------|------------|
| W/o Pre-training + W/o MoE | ~27 | ~28 |
| W/ Pre-training + Fixed-Rank LoRA | ~30 | ~31 |
| W/ Pre-training + MoE (Fixed-Rank) | ~31 | ~32 |
| **W/ Pre-training + RA-MoE** | **~33** | **~33** |

### Key Findings

- UniSTD maintains single-task-level performance even when supporting 10 tasks, whereas SimVPv2 starts to degrade significantly with just 3 tasks.
- Compared to SimVPv2-5Tasks, UniSTD outperforms it by approximately 12.8 PSNR on TaxiBJ.
- Task-agnostic pre-training (especially OpenCLIP-ViT) provides massive performance gains, verifying the effectiveness of the "general foundation + specialized adaptation" paradigm.
- Adaptive rank optimization outperforms fixed-rank LoRA, confirming that different tasks indeed require adapters of varying capacities.
- Zero-initialized temporal modules are crucial for maintaining training stability.

## Highlights & Insights

- Achieves a true "one-model-for-all" in the field of spatiotemporal prediction for the first time, sharing a single model across 10 vastly different tasks.
- The continuous relaxation technique for discrete rank optimization is elegant, converting an intractable combinatorial search into standard gradient optimization.
- Distinctly validates the critical insight that the transferability of large-scale 2D visual pre-training is powerful enough to benefit spatiotemporal prediction tasks.
- The minimalist design of the lightweight temporal module reflects a deep understanding of the problem's nature (as the FFN already implicitly serves as a temporal mixer).

## Limitations & Future Work

- The current framework mainly targets video-style spatiotemporal forecasting on regular grids, and does not cover graph-structured spatiotemporal data (e.g., traffic networks).
- The encoder/decoder still need to maintain independent convolutional layers for each task, meaning true "zero task-specific parameters" has not yet been achieved.
- Scaling behaviors of the model (e.g., larger ViT backbones beyond ViT-Base) have not been explored.
- Trade-offs between task-specific metrics (e.g., CSI for weather forecasting) and general metrics (PSNR/SSIM) were not thoroughly discussed.
- Future work can extend the paradigm to more disciplines (e.g., biology, fluid dynamics) and longer forecasting horizons.

## Related Work & Insights

- The continuous rank optimization idea in RA-MoE can be generalized to other multi-task LoRA adaptation scenarios.
- The "2D pre-training + temporal adaptation" paradigm can inspire other spatiotemporal tasks such as video understanding.
- The success of a unified architecture suggests that domain-specific designs might eventually be replaced by the combination of general Transformers and adapters.

## Rating

| Area | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Technical Depth | 4.5 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4 |
| Overall Rating | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MaRI: Material Retrieval Integration across Domains](mari_material_retrieval_integration_across_domains.md)
- [\[ICLR 2026\] Diverse Dictionary Learning](../../ICLR2026/self_supervised/diverse_dictionary_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](../../CVPR2026/self_supervised/diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](../../CVPR2026/self_supervised/learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)

</div>

<!-- RELATED:END -->
