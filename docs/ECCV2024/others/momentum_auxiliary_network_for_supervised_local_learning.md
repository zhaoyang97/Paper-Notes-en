---
title: >-
  [Paper Note] Momentum Auxiliary Network for Supervised Local Learning
description: >-
  [ECCV 2024 (Oral)][Local Learning] This paper proposes the Momentum Auxiliary Network (MAN), which transfers parameter information from adjacent local blocks to the auxiliary network of the current block via Exponential Moving Average (EMA). By introducing learnable biases to compensate for cross-block feature discrepancies, MAN addresses the "myopia" problem caused by the lack of inter-block information exchange in supervised local learning. It achieves superior performance…
tags:
  - "ECCV 2024 (Oral)"
  - "Local Learning"
  - "Momentum Auxiliary Network"
  - "Exponential Moving Average"
  - "Gradient Isolation"
  - "GPU Memory Optimization"
date: 2026-05-08
content_hash: fd516854a9fc87b7
---

# Momentum Auxiliary Network for Supervised Local Learning

**Conference**: ECCV 2024 (Oral)  
**arXiv**: [2407.05623](https://arxiv.org/abs/2407.05623)  
**Code**: [https://github.com/JunhaoSu0/MAN](https://github.com/JunhaoSu0/MAN)  
**Area**: Other / Deep Learning Training Strategies  
**Keywords**: Local Learning, Momentum Auxiliary Network, Exponential Moving Average, Gradient Isolation, GPU Memory Optimization

## TL;DR

This paper proposes the Momentum Auxiliary Network (MAN), which transfers parameter information from adjacent local blocks to the auxiliary network of the current block via Exponential Moving Average (EMA). By introducing learnable biases to compensate for cross-block feature discrepancies, MAN addresses the "myopia" problem caused by the lack of inter-block information exchange in supervised local learning. It achieves superior performance on ImageNet using less than half of the GPU memory of E2E training.

## Background & Motivation

**Background**: Deep neural networks are traditionally trained using end-to-end (E2E) backpropagation, where loss signals propagate backward layer-by-layer from the final layer to update all parameters. This approach lacks biological plausibility (as biological synapses perform local signal processing) and, more importantly, creates a "lock deadlock" wherein parameters can only be updated after a full forward and backward pass, leading to massive GPU memory consumption (due to storing activations and gradients of all layers).

**Limitations of Prior Work**: (1) Supervised local learning divides a network into multiple gradient-isolated local blocks, with each block supervised by an independent auxiliary network. While this saves GPU memory (as gradients only propagate within each block), the accuracy is far inferior to E2E training because of the lack of information exchange between blocks. (2) Existing improvements primarily focus on designing better auxiliary network architectures (e.g., DGL) or local loss functions (e.g., InfoPro, PredSim), but they still fail to fundamentally resolve the lack of inter-block communication. (3) The performance gap becomes more pronounced when the network is split into more blocks.

**Key Challenge**: The parallelism and memory advantages of local learning stem from gradient isolation, yet gradient isolation itself cuts off the inter-block information flow. Each block focuses only on local objectives and may discard globally beneficial information—referred to as the "myopia" problem. A mechanism is needed that can enable inter-block information transfer without disrupting gradient isolation.

**Key Insight**: Inspired by the EMA updates in momentum contrastive learning (MoCo), the authors propose to let each auxiliary network not only receive the input of the current block but also incorporate the parameter information of the next block through EMA. EMA provides a "smooth" information transfer mechanism—it requires no gradient flow across block boundaries while allowing the current block to perceive the state of subsequent blocks.

**Core Idea**: Use EMA to permeate the parameter information of adjacent blocks into the auxiliary network of the current block, combined with learnable biases to reconcile feature discrepancies, thereby endowing local learning with an approximately global perspective.

## Method

### Overall Architecture

The deep network is divided into $K$ gradient-isolated local blocks. The first $K-1$ blocks are each equipped with an auxiliary network for local supervision. The core improvement of MAN lies in the parameter update scheme of the auxiliary networks: it first updates the auxiliary network parameters $\gamma_j$ using local gradients, and then incorporates the parameters of the next local block $\theta_{j+1}$ into $\gamma_j$ via EMA. Simultaneously, a learnable bias $b_j$ is introduced for each auxiliary block to enhance adaptation to feature discrepancies across different blocks. The $K$-th block directly connects to the output classifier and does not use an auxiliary network.

### Key Designs

1. **EMA-based Information Transfer**:

    - **Function**: Achieves inter-block information communication without disrupting gradient isolation.
    - **Mechanism**: The parameter update of the $j$-th auxiliary network consists of two steps—first, update $\gamma_j \leftarrow \gamma_j - \eta_a \nabla_{\gamma_j} \mathcal{L}(\hat{y}_j, y)$ with local gradients, and then fuse the parameters of the first layer of the next block $\gamma_j \leftarrow \text{EMA}(\gamma_j, \theta_{j+1})$ via EMA. The EMA formula is $\gamma_j = m \cdot \gamma_j + (1-m) \cdot \theta_{j+1}$, where the momentum coefficient is $m = 0.995$. Only the parameters of the first layer of the next block (instead of all parameters) are used to maintain a balance in GPU memory usage.
    - **Design Motivation**: EMA provides a progressive form of information exchange—it is less abrupt than directly copying parameters, allowing the auxiliary network to smoothly and gradually perceive the learning state of subsequent blocks. Ablation studies show that EMA alone reduces the test error of DGL on CIFAR-10 ResNet-32 (K=16) from 14.08% to 11.07%.

2. **Learnable Bias**:

    - **Function**: Compensates for feature discrepancies between different gradient-isolated blocks.
    - **Mechanism**: Introduces an additional learnable bias parameter $b_j$ for each auxiliary network, which is optimized along with $\gamma_j$ using local gradients after the EMA update: $(\gamma_j, b_j) \leftarrow (\gamma_j, b_j) - \eta_a \nabla_{(\gamma_j, b_j)} \mathcal{L}(\hat{y}_j, y)$. The bias parameters are lightweight, increasing GPU memory by only about 1%.
    - **Design Motivation**: Since feature distributions across different local blocks vary (as they learn different levels of features), directly transferring parameters via EMA may have limited effectiveness due to feature mismatch. The learnable bias provides adaptive adjustment capabilities. Feature visualization indicates that the contributions of EMA and the bias are complementary.

3. **Plug-and-Play Universal Architecture**:

    - **Function**: Seamlessly integrates into any supervised local learning method.
    - **Mechanism**: MAN does not modify the core architecture or loss function design of local learning methods; it only alters the parameter update scheme of the auxiliary networks. In experiments, it successfully integrates with three methods—PredSim, DGL, and InfoPro—yielding significant improvements across all configurations.
    - **Design Motivation**: A good method should be generalizable. The EMA + bias mechanism of MAN is decoupled from specific auxiliary network architectures and loss functions.

### Loss & Training

The local loss functions follow the design of the baseline methods (such as the cross-entropy loss in DGL, and the information entropy loss + reconstruction loss in InfoPro). Regarding the training configuration: SGD optimizer + Nesterov momentum 0.9 with a cosine annealing learning rate scheduler. CIFAR-10 is trained for 400 epochs, and ImageNet is trained for 90 epochs. The EMA momentum coefficient is fixed at 0.995.

## Key Experimental Results

### Main Results

Test errors (%) on CIFAR-10 with varying network depths and block numbers:

| Method | ResNet-32 K=8 | ResNet-32 K=16 | ResNet-110 K=32 | ResNet-110 K=55 |
|------|-------------|--------------|----------------|----------------|
| E2E | 6.37 | 6.37 | 5.42 | 5.42 |
| DGL | 11.63 | 14.08 | 12.51 | 14.45 |
| DGL + MAN | **8.42** (↓3.21) | **9.11** (↓4.97) | **9.65** (↓2.86) | **9.73** (↓4.72) |
| InfoPro | 11.51 | 12.93 | 12.26 | 13.22 |
| InfoPro + MAN | **9.32** (↓2.19) | **9.65** (↓3.28) | **9.06** (↓3.20) | **9.77** (↓3.45) |

Results on ImageNet:

| Network | Method | Top1-Error | Top5-Error | GPU Memory (GB) | Memory Saving |
|------|------|-----------|-----------|-----------|---------|
| ResNet-101 | E2E | 22.03 | 5.93 | 19.71 | - |
| ResNet-101 | InfoPro(K=4) | 22.81 | 6.54 | 10.37 | 47.3% |
| ResNet-101 | InfoPro*(K=4) | **21.73** (↓1.08) | **5.81** (↓0.73) | ~10.47 | ~46.9% |
| ResNet-152 | E2E | 21.60 | 5.92 | 26.29 | - |
| ResNet-152 | InfoPro(K=4) | 22.21 | 6.26 | 13.48 | 48.7% |
| ResNet-152 | InfoPro*(K=4) | **20.78** (↓1.43) | **5.56** (↓0.70) | ~13.61 | ~48.2% |

### Ablation Study

Using the DGL baseline on CIFAR-10 ResNet-32 (K=16):

| Configuration | EMA | Learnable Bias | Test Error (%) |
|------|-----|---------|-----------|
| Original DGL | ✗ | ✗ | 14.08 |
| + EMA Only | ✓ | ✗ | 11.07 |
| + Bias Only | ✗ | ✓ | — |
| + EMA + Bias (Full MAN) | ✓ | ✓ | **9.11** |

Direct parameter usage vs. EMA usage from the next block (CIFAR-10 DGL):

| Scheme | Test Error (%) |
|------|-----------|
| No parameter usage from the next block | 14.08 |
| Direct usage (w/o EMA) | Limited improvement |
| EMA usage | **11.07** |

### Key Findings

- MAN brings significant improvements across all methods, datasets, network depths, and block configurations.
- On ImageNet, InfoPro + MAN outperforms E2E training while using less than half of the GPU memory.
- The contributions of EMA and learnable biases are complementary—EMA provides global information while the bias compensates for feature discrepancies.
- Linear separability analysis indicates that after incorporating MAN, earlier blocks learn more transferable/general features (with a slight drop in local accuracy), while later blocks exhibit significant performance gains—serving as clear evidence of global information flow.
- CKA analysis confirms that MAN aligns the representational features of local learning closer to those of E2E training.

## Highlights & Insights

- **Simple yet Effective Information Transfer**: EMA is an elegant way to transfer information across boundaries without breaking gradient isolation, with negligible computational overhead.
- **Outperforming E2E on ImageNet**: Achieving superior performance with approximately 47% memory savings marks a milestone for local learning.
- **Plug-and-Play Generalizability**: It is effective for three distinct local learning methods, demonstrating the generality of the proposed approach.
- **Extensive Analyses**: Linear separability and CKA analyses provide deep insights into the underlying mechanism of MAN—earlier blocks learn general features, and subsequent blocks benefit from global information.

## Limitations & Future Work

- On standard datasets (such as CIFAR-10), local learning with MAN is still less accurate than E2E training, particularly when divided into a large number of blocks.
- Currently, only the first layer's parameters of the next block are used for EMA; exploring deeper parameter transfers might yield further improvements.
- The method is only validated on image classification tasks, leaving dense prediction tasks (like detection and segmentation) unexplored.
- Combining MAN with self-supervised local learning methods remains an avenue for future investigation.

## Related Work & Insights

- **DGL**: Decoupled Greedy Learning, one of the primary baselines of MAN.
- **InfoPro**: A local learning method that leverages information-theoretic principles to retain middle-layer information.
- **PredSim**: A pioneer in local learning using similarity matching and layer-wise losses.
- **MoCo**: The successful application of EMA in contrastive learning, which inspired the design of MAN.
- **Insight**: EMA is a versatile cross-boundary information transfer tool that could be valuable in other gradient-isolated scenarios, such as federated learning or model-parallel training.

## Rating

- Novelty: ⭐⭐⭐⭐ (Using EMA for inter-block information transfer in local learning is novel and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets, 3 baseline methods, multiple network architectures, and comprehensive ablations and analyses)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (ECCV Oral, significantly advances the field of local learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
