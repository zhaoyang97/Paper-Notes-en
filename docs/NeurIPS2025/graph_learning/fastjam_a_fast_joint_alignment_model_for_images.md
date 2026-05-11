---
title: >-
  [Paper Note] FastJAM: a Fast Joint Alignment Model for Images
description: >-
  [NeurIPS 2025][Graph Learning][Joint image alignment] FastJAM is a graph-based fast joint image alignment method that computes pairwise keypoint correspondences using off-the-shelf image matchers…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Joint image alignment"
  - "graph neural networks"
  - "homography estimation"
  - "non-parametric clustering"
  - "inverse-compositional loss"
date: 2026-05-08
content_hash: 77779b47710baabb
---

# FastJAM: a Fast Joint Alignment Model for Images

**Conference**: NeurIPS 2025
**arXiv**: [2510.22842](https://arxiv.org/abs/2510.22842)
**Code**: [https://github.com/BGU-CS-VIL/FastJAM](https://github.com/BGU-CS-VIL/FastJAM)
**Area**: Graph Learning / Image Alignment
**Keywords**: Joint image alignment, graph neural networks, homography estimation, non-parametric clustering, inverse-compositional loss

## TL;DR

FastJAM is a graph-based fast joint image alignment method that computes pairwise keypoint correspondences using off-the-shelf image matchers, constructs a keypoint graph via fast non-parametric clustering, employs a GNN to propagate and aggregate information for predicting per-image homography parameters, and adopts an inverse-compositional loss to eliminate the need for regularization hyperparameters. It reduces joint alignment time from hours/minutes to approximately 49 seconds while achieving alignment quality superior to or on par with existing methods.

## Background & Motivation

**Background**: Joint image alignment (JA) is a classical computer vision problem—given a collection of images from the same semantic category, the goal is to align them into a unified coordinate system such that semantically corresponding regions appear at the same spatial locations. This is useful for downstream tasks including object discovery, 3D reconstruction, and generative model pre-training. Several neural network-based methods have emerged in recent years: Neural Congealing (CVPR 2023) incorporates GANs into alignment, ASIC (ICCV 2023) performs alignment based on implicit templates, and SpaceJAM (ECCV 2024) introduces lightweight spatial transformations.

**Limitations of Prior Work**: These methods share three common drawbacks: (1) excessively long training times—Neural Congealing requires 78 minutes (8 GPUs) and ASIC requires 67 minutes (4 GPUs); (2) large model capacity—Neural Congealing has 28.7M parameters; and (3) all methods require regularization terms to constrain predicted transformations from becoming degenerate, introducing hyperparameters that require careful tuning.

**Key Challenge**: Existing methods treat joint alignment as an end-to-end optimization problem, where the transformation parameters for each image must be iteratively refined over thousands of iterations. This "heavy optimization" paradigm is inherently slow and sensitive to regularization hyperparameters.

**Goal**: Can a non-iterative approach directly and efficiently infer joint alignment parameters from pairwise correspondences within an image collection?

**Key Insight**: Pairwise keypoint matches between images naturally form a graph structure—keypoints are nodes and matches are edges. GNNs are well-suited to perform information propagation and aggregation over such graph structures, enabling one-shot inference of per-image transformation parameters from global correspondences.

**Core Idea**: Reformulate joint alignment as a graph inference problem—pairwise matches → keypoint graph → GNN-predicted transformation parameters—coupled with an inverse-compositional loss that eliminates the need for regularization tuning.

## Method

### Overall Architecture

FastJAM is a three-stage pipeline:

**Stage 1: Pairwise Matching.** An off-the-shelf pre-trained image matcher (e.g., SuperGlue, LoFTR) is applied to all image pairs in the collection to compute keypoint correspondences. Each pair yields a set of matched keypoint pairs $(k_i^a, k_j^b)$, indicating that keypoint $i$ in image $a$ semantically corresponds to keypoint $j$ in image $b$.

**Stage 2: Graph Construction.** Fast non-parametric clustering (requiring no pre-specified number of clusters) groups semantically corresponding keypoints across images into the same cluster. A keypoint graph is then constructed: nodes are keypoints within each image, and edges are of two types—intra-image spatial relation edges between keypoints within the same image, and inter-image correspondence edges between keypoints in the same cluster across different images.

**Stage 3: GNN Inference.** A graph neural network performs message passing over the keypoint graph to propagate and aggregate cross-image correspondence information. Image-level pooling (aggregating all keypoint nodes belonging to the same image) is then applied to regress the homography transformation parameters $H$ for each image.

### Key Designs

1. **Fast Non-Parametric Clustering**:

    - **Function**: Automatically groups semantically corresponding keypoints from different images into the same cluster without requiring a pre-specified number of clusters.
    - **Mechanism**: Leverages correspondences already provided by the pairwise matcher and applies transitive closure or a similar strategy to identify "keypoint 3 of A ↔ keypoint 7 of B ↔ keypoint 5 of C" as the same semantic part. The clustering outcome determines the connectivity of inter-image edges.
    - **Design Motivation**: Clustering methods with a fixed number of clusters (e.g., k-means) are unsuitable for joint alignment, as the number of semantic parts varies unpredictably across categories. The adaptive nature of non-parametric clustering enables FastJAM to generalize across different categories.

2. **GNN Message Passing on the Keypoint Graph**:

    - **Function**: Globally infers per-image transformation parameters from locally available pairwise information.
    - **Mechanism**: The GNN performs multiple rounds of message passing over the graph. Intra-image edges propagate spatial layout information (relative positions of keypoints within an image), while inter-image edges propagate correspondence information (semantic matches across images). After multiple rounds, each keypoint node aggregates globally consistent information from multiple images. Image-level pooling by image ID yields an image-level representation, which is then regressed into a $3 \times 3$ homography matrix.
    - **Design Motivation**: Compared to the conventional pipeline of estimating pairwise homographies followed by global optimization, GNN message passing naturally achieves multi-hop diffusion of global information, simultaneously inferring transformations for all images in a single forward pass.

3. **Inverse-Compositional Loss**:

    - **Function**: Eliminates the need for transformation regularization terms, simplifying training.
    - **Mechanism**: Conventional losses typically take the form $L_{\text{align}} + \lambda L_{\text{reg}}$, where the regularization term prevents predicted transformations from becoming degenerate (e.g., excessively large scale or rotation), and $\lambda$ requires careful tuning. The inverse-compositional loss measures alignment error by composing the current predicted transformation with prior transformations in keypoint space; its mathematical formulation naturally constrains transformation plausibility without requiring explicit regularization.
    - **Design Motivation**: Removing the hyperparameter $\lambda$ makes the method more readily applicable to new datasets and categories, substantially reducing the tuning burden in practice.

### Loss & Training

The inverse-compositional loss is computed over keypoints, avoiding the overhead of pixel-level reconstruction. Training requires only approximately 600 iterations (compared to 8,000 for Neural Congealing and 20,000 for ASIC) and runs on a single GPU. The model has only 0.13M parameters, making it extremely lightweight.

## Key Experimental Results

### Main Results

**Runtime Efficiency Comparison** (SPair-71k dataset, averaged over three categories):

| Method | Parameters | GPUs | Iterations | Runtime | Speedup |
|--------|-----------|------|------------|---------|---------|
| Neural Congealing (CVPR'23) | 28.7M | 8 | 8,000 | 01:18:30 | 1× |
| ASIC (ICCV'23) | 7.9M | 4 | 20,000 | 01:06:38 | ~1× |
| SpaceJAM (ECCV'24) | 0.016M | 1 | 700 | 00:06:00 | ~13× |
| **FastJAM** | **0.13M** | **1** | **600** | **00:00:49** | **~96×** |

**Alignment Quality Comparison** (SPair-71k semantic keypoint transfer accuracy PCK):

| Method | cat | dog | car | Average |
|--------|-----|-----|-----|---------|
| Neural Congealing | Competitive | Competitive | Lower | 2nd |
| SpaceJAM | Good | Good | Good | 2nd/3rd |
| **FastJAM** | **Best/Tied** | **Best/Tied** | **Best** | **1st** |

FastJAM matches or surpasses all existing methods in alignment quality while being ~7.4× faster than the previously fastest method, SpaceJAM.

### Ablation Study

| Ablation | Effect |
|----------|--------|
| Remove non-parametric clustering (replaced with fixed-$k$ clustering) | Degraded cross-image correspondence quality, reduced alignment accuracy |
| Remove inverse-compositional loss (replaced with conventional loss + regularization) | Requires additional $\lambda$ tuning; unstable performance |
| Reduce number of GNN propagation layers | Global consistency information insufficiently diffused; uneven alignment |
| Different matchers (SuperGlue vs. LoFTR) | Marginal performance difference, demonstrating robustness to matcher choice |

### Key Findings

- The 96× speedup transforms joint alignment from an "offline preprocessing" step into a near-real-time operation—a qualitative breakthrough rather than a marginal improvement.
- The 0.13M parameter count, far smaller than Neural Congealing's 28.7M, demonstrates that a well-designed graph structure is more efficient than brute-force scaling of model parameters.
- The inverse-compositional loss eliminates regularization hyperparameters, removing the need for per-category retuning and substantially improving practical usability.
- Strong results on the CUB-200-2011 bird dataset demonstrate cross-domain generalization capability.

## Highlights & Insights

- **The Power of Problem Reformulation**: Recasting joint alignment from an "iterative optimization problem" to a "graph inference problem" fundamentally shifts the computational paradigm from thousands of iterations to a single forward pass. This reformulation strategy is broadly instructive.
- **Practical Value of Modular Design**: The matcher, clustering algorithm, and GNN are independent and interchangeable modules. FastJAM can directly benefit from future advances in image matching without requiring architectural redesign.
- **Elegance of the Inverse-Compositional Loss**: A well-designed loss function can simultaneously eliminate hyperparameters and improve performance—this is of substantial practical value, as hyperparameter tuning is often the greatest obstacle to deploying methods in real-world settings.
- **GNN as a "Global Reasoning Engine"**: Using GNN message passing to perform globally consistent inference over pairwise information is a paradigm transferable to any task requiring the inference of globally consistent structures from local observations (e.g., multi-view reconstruction, co-localization).

## Limitations & Future Work

- **Transformation Model Limited to Homography**: The homography assumption restricts the method to planar projective transformations between images, precluding the handling of non-rigid deformations (e.g., animal pose variation) or complex 3D viewpoint differences. Extension to thin-plate splines (TPS) or dense optical flow is an important future direction.
- **Quality Dependent on the Matcher**: FastJAM does not train the matcher—if off-the-shelf matchers perform poorly in specialized domains (e.g., medical or satellite imagery), the entire pipeline is adversely affected. Domain adaptation may be required.
- **Intra-class Variation Limitations**: For categories with extremely large intra-class appearance variation (e.g., "chair," encompassing office chairs to recliners), clustering may produce noisy clusters and degrade GNN inference quality.
- **End-to-End Training Not Explored**: The current three-stage sequential pipeline treats the matcher and clustering as fixed. Whether end-to-end fine-tuning could further improve quality remains an open question.
- **Absence of Comparison with Dense Correspondence Methods**: Methods based on dense semantic correspondences (e.g., DINO-feature-based DenseCorrespondence) are not compared, and such approaches may be more robust in certain scenarios.

## Related Work & Insights

- **vs. Neural Congealing (CVPR 2023)**: Uses GANs for joint alignment, requiring 8 GPUs, 28.7M parameters, and 78 minutes. FastJAM achieves superior alignment quality with 1 GPU, 0.13M parameters, and 49 seconds.
- **vs. SpaceJAM (ECCV 2024)**: Already lightweight (0.016M parameters, 6 minutes), yet FastJAM is a further 7.4× faster with better alignment quality. The key difference is that FastJAM replaces per-image iterative optimization with GNN-based global inference.
- **vs. SuperGlue**: SuperGlue also employs GNNs for keypoint matching but only handles pairs of images. FastJAM extends this to multi-image joint alignment, shifting the GNN's role from "matching" to "global alignment inference."
- **Inspiration**: In any problem requiring globally consistent structure to be extracted from a large number of pairwise relations, "graph construction + GNN inference" is a fast alternative worth considering—for instance, in multi-view pose estimation or co-segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ Reformulating joint alignment as graph inference, combined with an inverse-compositional loss that eliminates regularization, with clear motivation at every design step.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark comparisons, ablation studies, and cross-dataset generalization experiments with comprehensive quantitative and qualitative results.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, comparisons with prior work are intuitive, and the runtime table is immediately interpretable.
- Value: ⭐⭐⭐⭐ The 96× speedup represents a practically significant breakthrough, making joint alignment feasible within real-world processing pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](heterogeneous_swarms_jointly_optimizing_model_roles_and_weights_for_multi-llm_sy.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](../../ACL2026/graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

</div>

<!-- RELATED:END -->
