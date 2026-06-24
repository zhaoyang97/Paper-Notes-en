---
title: >-
  [Paper Note] BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression
description: >-
  [ECCV 2024][Model Compression][Neural Image Compression] This paper proposes the BaSIC framework, which simultaneously controls backbone network complexity and the parallel computation capability of autoregressive units by learning the Bayesian network structure of neural image compression (NIC) systems, achieving computational scalability control over the entire NIC pipeline for the first time.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Neural Image Compression"
  - "Bayesian Network"
  - "Computational Scalability"
  - "Autoregressive Model"
  - "Structure Learning"
date: 2026-05-08
content_hash: f722d2e7cbbe723f
---

# BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression

**Conference**: ECCV 2024  
**Code**: [https://github.com/worldlife123/cbench_BaSIC](https://github.com/worldlife123/cbench_BaSIC)  
**Area**: Model Compression / Image Compression  
**Keywords**: Neural Image Compression, Bayesian Network, Computational Scalability, Autoregressive Model, Structure Learning

## TL;DR

This paper proposes the BaSIC framework, which simultaneously controls backbone network complexity and the parallel computation capability of autoregressive units by learning the Bayesian network structure of neural image compression (NIC) systems, achieving computational scalability control over the entire NIC pipeline for the first time.

## Background & Motivation

**Background**: Neural Image Compression (NIC) has surpassed traditional codecs (such as JPEG, BPG, VVC intra) in rate-distortion performance, but its huge computational overhead severely hinders practical deployment. A typical NIC system consists of an encoder backbone, a decoder backbone, and an autoregressive entropy coding module, which collectively determine the overall computational complexity.

**Limitations of Prior Work**: Most existing NIC acceleration works focus on accelerating a single module (e.g., only accelerating the autoregressive module or only compressing the backbone network), failing to precisely control the overall computational complexity. For instance, some methods can only convert the autoregressive steps from serial to partially parallel but cannot control the computational cost of the backbone networks; other methods reduce the backbone via pruning but neglect the acceleration of the autoregressive module. There is a lack of a unified framework to simultaneously control the computational complexity of all modules.

**Key Challenge**: The computational complexity of a NIC system is jointly determined by multiple interrelated components—the number of layers/channels in the backbone networks determines the feature extraction capacity, and the dependency structure of the autoregressive module determines the parallelism and compression rate. There is a complex coupling relationship between the two: simplifying the backbone might require a stronger autoregressive model to compensate, and vice versa. Currently, there is no theoretical framework to model and optimize this coupling in a unified manner.

**Goal**: (1) How to simultaneously control both the backbone complexity and the autoregressive parallelism of NIC under a unified framework? (2) How to automatically find the optimal complexity allocation scheme given a computational budget? (3) How to maintain competitive compression performance under computational constraints?

**Key Insight**: The authors model the NIC system as a Bayesian network (BayesNet), where nodes represent latent variables and edges represent dependencies. Controlling the computational complexity of NIC is equivalent to learning the structure of this BayesNet—by adjusting the connectivity of edges to simultaneously control the backbone complexity and the autoregressive parallelism.

**Core Idea**: By modeling the NIC system as a Bayesian network, the backbone complexity control and the autoregressive parallelism optimization are unified as solving two subproblems of BayesNet structure learning.

## Method

### Overall Architecture

BaSIC decomposes the computational scalability problem of NIC into two BayesNet structure learning subproblems: (1) **Inter-Node Structure Learning**—controlling backbone network complexity via a heterogeneous bipartite BayesNet; (2) **Intra-Node Structure Learning**—optimizing the parallel computation of autoregressive units via a multipartite BayesNet. The inputs are reference image data and computational budget constraints, and the output is the optimal NIC model configuration matching the constraints.

### Key Designs

1. **Heterogeneous Bipartite BayesNet for Backbone Complexity Control**:

    - Function: Regulates the computational complexity of the encoder-decoder backbone.
    - Mechanism: Models the inter-layer connections of the NIC backbone as a bipartite graph, where one side of the nodes represents the output features of the encoder layers, and the other side represents the input features of the decoder layers. The edges in the graph stand for data dependencies between layers. By learning the structure of this bipartite graph (i.e., deciding which edges to keep or prune), channel pruning and layer skipping of the backbone network can be equivalently achieved. The heterogeneity is reflected in that nodes at different layers can have different dimensions (number of channels), allowing non-uniform pruning strategies. The optimization objective of structure learning is to maximize rate-distortion performance subject to computational budget constraints.
    - Design Motivation: Traditional pruning methods typically impose uniform pruning ratios across all layers, ignoring the varying contributions of different layers to performance. The BayesNet structure learning framework naturally supports non-uniform, adaptive complexity allocation and provides theoretical guarantees from a probabilistic perspective.

2. **Multipartite BayesNet for Autoregressive Parallelization**:

    - Function: Optimizes the parallel computation structure of the autoregressive entropy coding module.
    - Mechanism: In a standard autoregressive model, each latent variable depends on all its predecessors, which forces strictly serial decoding and results in extremely slow computation. This work organizes the latent variables into a multipartite graph structure: grouping variables such that variables within the same group are mutually independent (can be computed in parallel) while maintaining necessary dependencies between groups. This achieves group-wise parallel decoding. The structure learning objective of the multipartite BayesNet is to find the optimal grouping scheme to minimize the rate loss by capturing as many statistical dependencies as possible between groups given a specified degree of parallelism (i.e., number of groups). The algorithm determines the grouping and the topological ordering of inter-group dependencies using a greedy search.
    - Design Motivation: Existing autoregressive parallelization methods (e.g., Checkerboard, Channel-wise grouping) use fixed, hand-crafted grouping schemes, which fail to adapt to data characteristics. BayesNet structure learning provides data-driven optimal grouping schemes.

3. **Joint Optimization and Computational Budget Control**:

    - Function: Automatically allocates the complexity between the backbone and the autoregressive module under a given total computational budget.
    - Mechanism: Defines the global computational budget as $C_{total} = C_{backbone} + C_{AR}$, where the backbone complexity is determined by the bipartite BayesNet structure, and the autoregressive complexity is determined by the number of groups in the multipartite BayesNet. By solving the two subproblems under different $(C_{backbone}, C_{AR})$ allocations and then selecting the allocation scheme with the optimal rate-distortion performance. In practice, a multi-level Slimmable Network technique is adopted, allowing a single training run to yield models of multiple computational points.
    - Design Motivation: Decoupling the two subproblems greatly reduces the complexity of joint optimization, and the adoption of the Slimmable Network avoids the massive overhead of training individual models for each computational point.

### Loss & Training

The training loss is the standard rate-distortion loss $L = R + \lambda D$, where $R$ is the bitrate, $D$ is the distortion (MSE or MS-SSIM), and $\lambda$ controls the rate-distortion trade-off. The Slimmable Network strategy is utilized during training, randomly sampling different computational configurations in each mini-batch so that a single model can support multiple computational levels. The BayesNet structure learning uses a greedy search algorithm and is performed after training, introducing no computational overhead to the training phase.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (BaSIC) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Kodak | BD-Rate (at equal computation) | Optimal | Slimmable NIC | More accurate complexity control + better RD performance |
| ImageNet subset | PSNR @ equivalent MACs | Competitive | Fixed-architecture NIC | Achieves full computational scalability |
| Kodak | Complexity control accuracy | High accuracy | Slimmable baseline | Smaller control error |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Backbone-only control (no AR optimization) | Moderate BD-Rate | Lack of AR parallelization results in slow decoding |
| AR-only control (fixed backbone) | Moderate BD-Rate | Unable to reduce backbone computation |
| Uniform channel pruning | Inferior | Ignoring inter-layer differences incurs large performance loss |
| BaSIC full scheme | Optimal | Joint optimization of the two subproblems yields the best results |
| Different AR grouping schemes | RD performance comparison | Learned grouping outperforms hand-crafted Checkerboard grouping |

### Key Findings

- The complexity allocation scheme discovered via BayesNet structure learning yields significant PSNR improvements compared to uniform pruning under the same computational budget.
- The grouping scheme discovered by the greedy grouping search in the autoregressive module outperforms all hand-crafted grouping strategies.
- BaSIC can provide a continuous RD performance curve across a very wide range of computation (from extremely low to full scale).
- The optimal computational allocation ratio between the backbone and the autoregressive module varies with the target bitrate: the autoregressive module is more critical at low bitrates, while the backbone is more important at high bitrates.

## Highlights & Insights

- **Unified Perspective**: For the first time, NIC backbone complexity and autoregressive parallelization control are unified under the framework of BayesNet structure learning, which is theoretically elegant.
- **Full Computational Scalability**: Unlike methods that can only accelerate a single module, BaSIC achieves precise computational control over the entire NIC pipeline.
- **Data-Driven Structure Discovery**: Autoregressive grouping no longer relies on hand-crafted designs but is automatically discovered through structure learning to achieve the optimal solution.
- **Practical Multi-Level Deployment**: Combining Slimmable Networks with structure learning enables a single training run to support diverse deployment scenarios.

## Limitations & Future Work

- The BayesNet structure learning currently relies on a greedy search, which may fall into local optima.
- Although decoupling the optimization of the two subproblems reduces complexity, it may lose coupling information between the backbone and the autoregressive module.
- The framework is currently validated mainly on Joint Autoregressive and Hyperprior NIC architectures, and its applicability to newer NIC architectures (such as Transformer-based models) has not been verified.
- Performance interference may exist among different computational configurations during Slimmable Network training, which could degrade performance under extreme configurations.
- The comparison with actual hardware latency is missing, and MACs do not fully equate to real inference speed.

## Related Work & Insights

- **Slimmable Networks** (Yu & Huang): A network training strategy with switchable widths. BaSIC incorporates structure learning on top of this.
- **FSAR** (Finite State Autoregressive Entropy Coding): An autoregressive acceleration method. BaSIC's codebase is developed based on FSAR.
- **Checkerboard Context Model**: An autoregressive parallelization method with fixed checkerboard grouping. BaSIC demonstrates that learned grouping is superior.
- **Survey on NIC Scalability**: Most works in this direction focus on single-module acceleration, whereas BaSIC proposes the first global scalability framework.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling the NIC scalability problem from a Bayesian network perspective is highly novel.
- Experimental Thoroughness: ⭐⭐⭐ The ablation studies are relatively complete, but the datasets and baseline methods in the main experiments could be richer.
- Writing Quality: ⭐⭐⭐⭐ The mathematical modeling is clear, and the breakdown logic of the two subproblems is well-structured.
- Value: ⭐⭐⭐⭐ Fully computationally scalable NIC is of great significance for practical deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)
- [\[ACL 2025\] Basic Reading Distillation](../../ACL2025/model_compression/basic_reading_distillation.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](../../ICML2025/model_compression/lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)

</div>

<!-- RELATED:END -->
