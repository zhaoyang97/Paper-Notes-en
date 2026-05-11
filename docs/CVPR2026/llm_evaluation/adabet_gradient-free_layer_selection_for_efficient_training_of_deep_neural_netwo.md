---
title: >-
  [Paper Note] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks
description: >-
  [CVPR2026][LLM Evaluation][Layer Selection] This paper proposes AdaBet, a gradient-free layer selection method grounded in algebraic topology…
tags:
  - "CVPR2026"
  - "LLM Evaluation"
  - "Layer Selection"
  - "Betti Number"
  - "Topological Data Analysis"
  - "Gradient-free Fine-tuning"
  - "Edge Devices"
  - "Transfer Learning"
date: 2026-05-08
content_hash: 6d37856fbb0c3308
---

# AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks

**Conference**: CVPR2026  
**arXiv**: [2510.03101](https://arxiv.org/abs/2510.03101)  
**Code**: [https://github.com/Nokia-Bell-Labs/efficient_layer_selection](https://github.com/Nokia-Bell-Labs/efficient_layer_selection)  
**Area**: Efficient Training / Edge Computing  
**Keywords**: Layer Selection, Betti Number, Topological Data Analysis, Gradient-free Fine-tuning, Edge Devices, Transfer Learning

## TL;DR
This paper proposes AdaBet, a gradient-free layer selection method grounded in algebraic topology, which uses the first Betti number $b_1$ to quantify the topological complexity of each layer's activation space via a single forward pass—requiring no labels, gradients, or backpropagation. By fine-tuning only 10% of layers on ResNet50/VGG16/MobileNetV2/ViT-B16, AdaBet surpasses full fine-tuning in accuracy while reducing peak memory by approximately 40%.

## Background & Motivation

**Background**: The demand for fine-tuning deep neural networks on edge devices (smartphones, IoT, embedded systems) is growing rapidly, yet such devices are severely constrained in memory and compute, rendering full fine-tuning infeasible.

**Limitations of Prior Work**: (1) Conventional transfer learning freezes most layers and trains only the final few, a heuristic that ignores the potential need to adapt intermediate layers to new tasks. (2) Fisher Information-based layer selection requires backpropagation and labeled data, making it inapplicable in label-free or privacy-sensitive scenarios. (3) Structured pruning (PruneTrain) and elastic training (ElasticTrainer) reduce computation but still rely on gradient information.

**Key Challenge**: Measuring each layer's "importance" to a new task requires costly metrics (Fisher information, gradient norms), all of which depend on backpropagation—the very bottleneck one seeks to avoid. Using expensive operations to decide how to save cost is fundamentally contradictory.

**Goal**: Determine which layers most need updating using only a forward pass, without any labels, gradients, or backpropagation.

**Key Insight**: Leveraging topological data analysis (TDA), the paper characterizes each layer's activation space through its topological structure—specifically the presence of 1-dimensional loop structures—as a proxy for feature complexity and entanglement.

**Core Idea**: High $b_1$ (many 1-dimensional loops) $\Rightarrow$ entangled activation manifold $\Rightarrow$ misalignment between pre-trained features and the new task $\Rightarrow$ update required; Low $b_1$ (few loops) $\Rightarrow$ approximately linearly separable $\Rightarrow$ features can be reused $\Rightarrow$ freeze.

## Method

### Overall Architecture
AdaBet operates in two stages: (1) **Layer importance assessment** — a single forward pass over unlabeled data computes the normalized first Betti number $\hat{b}_1$ for each layer; (2) **Layer selection and fine-tuning** — the top-$\rho$ fraction of layers (default 10%) are selected for fine-tuning while the rest are frozen. The entire procedure requires no server-side meta-training and can be executed entirely on the edge device.

### Key Designs

1. **First Betti Number $b_1$ Computation**:

    - **Function**: Quantifies the 1-dimensional topological features (loops) in each layer's activation space.
    - **Mechanism**: For the activation tensor $a_i \in \mathbb{R}^{B \times C \times H \times W}$ of layer $i$, the tensor is flattened to $\mathbb{R}^{B \times d}$ (where $d = C \times H \times W$), a Vietoris–Rips complex is constructed, and persistent homology is computed to extract the 1-dimensional Betti number $b_1^{(i)}$—the count of 1-dimensional loop structures that persist across scales.
    - **Design Motivation**: $b_1$ captures the degree of manifold entanglement. An abundance of 1-dimensional loops indicates that feature manifolds of different classes are mutually intertwined and not linearly separable, implying the layer must be adapted to disentangle them.

2. **Normalized Betti Number $\hat{b}_1$**:

    - **Function**: Balances topological importance against computational cost.
    - **Mechanism**: $\hat{b}_1^{(i)} = b_1^{(i)} / |a_i|$, where $|a_i|$ denotes the number of parameters in the $i$-th layer's activation tensor ($C \times H \times W$). Normalization removes the scale bias introduced by larger layers, which naturally tend to exhibit higher raw $b_1$.
    - **Design Motivation**: Without normalization, the selection criterion favors large layers whose fine-tuning incurs higher memory and computation costs. Normalization shifts the criterion to "topological complexity per unit parameter," enabling an optimal trade-off between performance and resource consumption.

3. **Layer Selection Strategy**:

    - **Function**: Ranks layers by $\hat{b}_1$ in descending order and selects the top-$\rho$ fraction for fine-tuning.
    - **Mechanism**: Given fine-tuning ratio $\rho$ (default 10%), gradients are enabled for the top $\lceil \rho \times L \rceil$ layers and disabled for all others. The classification head is always included in training.
    - **Design Motivation**: $\rho$ serves as a user-adjustable knob—it can be set as low as 5% under extreme resource constraints or up to 20% when resources permit. The default of 10% yields optimal performance across multiple architectures.

4. **Channel-level Extension $\rho_{ch}$**:

    - **Function**: Further restricts fine-tuning to the most important channels within selected layers.
    - **Mechanism**: For each selected layer, $b_1$ is computed per channel; only the top-$\rho_{ch}$ fraction of channels have their gradients enabled. This enables finer-grained resource savings.
    - **Design Motivation**: Some layers may require adaptation in only a subset of channels; channel-level selection further reduces the number of trainable parameters.

### Key Comparison with Fisher Information
- **Limitations of FI**: (1) Requires backpropagation and labeled data; (2) is highly sensitive to batch size and the number of backpropagation steps—different batches yield different layer rankings (illustrated in Fig. 3 of the paper); (3) is unreliable on edge devices with limited batch availability.
- **Advantages of $b_1$**: (1) Requires only a forward pass with no labels; (2) produces highly consistent rankings across different batches (Kendall-$\tau > 0.95$, validated in the paper); (3) computational complexity scales linearly with batch size.

### Loss & Training
- Standard cross-entropy loss is used during fine-tuning; frozen layers do not participate in gradient computation.
- Layer selection is performed once prior to training and remains fixed throughout.

## Key Experimental Results

### Main Results (Accuracy %)

| Method | ResNet50 Flowers | ResNet50 CIFAR-100 | MobileNetV2 Cars | ViT-B16 CIFAR-100 |
|------|-----------------|-------------------|-----------------|-------------------|
| Full Training | 82.3 | 75.8 | 80.1 | 85.2 |
| Transfer Learning (last layers) | 79.5 | 72.1 | 76.8 | 82.7 |
| PruneTrain | 80.1 | 73.6 | 77.4 | 83.5 |
| ElasticTrainer | 81.2 | 74.3 | 78.9 | 84.1 |
| Fisher-based Selection | 81.8 | 74.9 | 79.2 | 84.6 |
| **AdaBet (Ours, ρ=10%)** | **84.5** | **77.4** | **82.8** | **87.1** |

### Resource Efficiency

| Method | Trainable Params (%) | Peak Memory (relative) | Training Time (relative) |
|------|--------------|----------------|----------------|
| Full Training | 100% | 1.00× | 1.00× |
| Transfer Learning | ~20% | 0.72× | 0.45× |
| Fisher Selection (10%) | 10% | 0.62× | 0.88× (incl. FI computation) |
| **AdaBet (ρ=10%)** | **10%** | **0.60×** | **0.52×** |

### Ablation Study

| Configuration | ResNet50 Flowers | CIFAR-100 |
|------|-----------------|-----------|
| Random Layer Selection (10%) | 80.8 | 73.2 |
| Selection by Parameter Count | 81.1 | 73.8 |
| $b_1$ without Normalization | 82.9 | 75.6 |
| $b_1$ Normalized (AdaBet) | **84.5** | **77.4** |
| AdaBet + Channel ($\rho_{ch}$=50%) | 84.2 | 77.1 |

### Key Findings
- **Normalization is critical**: Omitting normalization degrades accuracy by 1.6–1.8%, as it biases selection toward large layers and neglects critical smaller ones.
- **Substantial margin over Fisher**: AdaBet outperforms Fisher-based selection by more than 2.5% while incurring lower total training time, as Fisher requires additional backpropagation passes for importance estimation.
- **10% layer fine-tuning outperforms full training**: While counterintuitive, this is attributable to freezing well-aligned layers, thereby mitigating catastrophic forgetting and overfitting.
- **$b_1$ rankings are highly stable across batches**: Kendall-$\tau > 0.95$, compared to $\tau < 0.7$ for Fisher information under varying batch conditions.
- **Strong generalization across architectures**: AdaBet is effective on both CNN-based models (ResNet50, VGG16, MobileNetV2) and Transformers (ViT-B16).

## Highlights & Insights
- **A topological perspective on layer adaptation**: Reformulating the question of "which layers to fine-tune" as a topological feature analysis problem is a genuinely novel perspective. The intuition that high $b_1$ $\Leftrightarrow$ manifold entanglement $\Leftrightarrow$ need for disentanglement is both elegant and theoretically grounded.
- **No labels or gradients required**: This property is highly valuable in privacy-preserving settings such as federated learning and edge device personalization, where data and gradients must not leave the device.
- **Single forward pass decision**: Layer selection requires only one forward pass; subsequent training proceeds identically to standard fine-tuning, making the method straightforward to implement.
- **Elegant normalization design**: The formulation $\hat{b}_1 = b_1 / |a_i|$ simultaneously addresses importance weighting and computational cost balancing—a single division resolves two design objectives.

## Limitations & Future Work
- Computing Betti numbers via Vietoris–Rips complex construction incurs non-trivial overhead; dimensionality reduction or sampling may be necessary for high-dimensional activations. The paper does not adequately address scalability to large models such as LLaMA.
- The fine-tuning ratio $\rho$ is a manually specified hyperparameter; the optimal value may vary across tasks, and no adaptive selection strategy is proposed.
- Evaluation is restricted to classification tasks; the effectiveness on detection, segmentation, and generative tasks remains unexplored.
- The channel-level selection variant $\rho_{ch}$ yields only marginal additional gains (~0.3%) while considerably increasing implementation complexity.
- No comparison is made against parameter-efficient fine-tuning methods such as LoRA or Adapters, which are natural alternatives for edge fine-tuning scenarios.

## Related Work & Insights
- **vs. Fisher Information Selection**: Fisher-based methods require backpropagation and labeled data and are sensitive to batch conditions; AdaBet is entirely gradient-free and demonstrates significantly greater ranking stability.
- **vs. ElasticTrainer**: ElasticTrainer employs an elastic search strategy for layer selection but still depends on gradient information; AdaBet achieves equivalent functionality through a single forward pass.
- **vs. PruneTrain**: PruneTrain performs dynamic pruning during training; AdaBet makes a one-time selection prior to training, offering a simpler and more efficient workflow.
- **vs. LoRA/Adapters**: LoRA inserts low-rank matrices into every layer, whereas AdaBet selectively freezes entire layers. The two approaches are complementary—LoRA fine-tuning could be applied specifically to layers selected by AdaBet.
- **TDA in deep learning**: Prior work has primarily applied TDA to analyze training dynamics and data complexity; AdaBet is the first to leverage TDA for guiding training strategy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First application of algebraic topology (Betti numbers) to layer selection; the perspective is entirely novel and the theoretical intuition is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across 4 architectures × 4 datasets with comprehensive ablations; however, comparison against PEFT methods such as LoRA is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; the comparative visualization of FI versus Betti numbers is persuasive.
- **Value**: ⭐⭐⭐⭐ Practically significant for edge device fine-tuning, with a simple and deployable design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](../../ICLR2026/llm_evaluation/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](../../NeurIPS2025/llm_evaluation/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](../../ICLR2026/llm_evaluation/disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)
- [\[CVPR 2026\] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)

</div>

<!-- RELATED:END -->
