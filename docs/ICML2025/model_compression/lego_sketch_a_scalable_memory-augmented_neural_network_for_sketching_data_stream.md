---
title: >-
  [Paper Note] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams
description: >-
  [ICML2025][Model Compression][data stream sketch] This paper proposes Lego Sketch, a scalable memory-augmented neural network (MANN) based on modular "memory bricks". It addresses the scalability bottleneck of existing neural sketches—which require retraining across different data domains and spatial budgets—using normalized multi-hash embedding, scalable memory, and a self-guided weighted loss. It also provides the first theoretical error upper bound for neural sketches.
tags:
  - "ICML2025"
  - "Model Compression"
  - "data stream sketch"
  - "frequency estimation"
  - "memory-augmented neural network"
  - "scalability"
  - "meta-learning"
date: 2026-05-08
content_hash: cbd0e1d87b38b5ce
---

# Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams

**Conference**: ICML2025  
**arXiv**: [2505.19561](https://arxiv.org/abs/2505.19561)  
**Code**: [FFY0/LegoSketch_ICML](https://github.com/FFY0/LegoSketch_ICML)  
**Area**: Model Compression  
**Keywords**: data stream sketch, frequency estimation, memory-augmented neural network, scalability, meta-learning

## TL;DR

This paper proposes Lego Sketch, a scalable memory-augmented neural network (MANN) based on modular "memory bricks". It addresses the scalability bottleneck of existing neural sketches—which require retraining across different data domains and spatial budgets—using normalized multi-hash embedding, scalable memory, and a self-guided weighted loss. It also provides the first theoretical error upper bound for neural sketches.

## Background & Motivation

### Problem Definition

Data stream frequency estimation: Given an infinite data stream $\mathcal{X} = (x_1, \dots, x_N)$ containing $n$ distinct elements, the goal is to accurately estimate the frequency $f_i$ of any element $e_i$ within sublinear space. A sketch is a classic probabilistic data structure designed for this problem.

### Limitations of Prior Work

**Handcrafted sketches** (such as CM-Sketch, C-Sketch, etc.): Rely on pre-defined 2D arrays, hash functions, and fixed strategies, showing an upper limit in the space-accuracy trade-off.

**Neural sketches** (such as the Meta-Sketch series): Replace the handcrafted core structures with MANNs, achieving higher accuracy under tight space budgets, but suffer from two major scalability bottlenecks:
   - **Poor domain scalability**: The embedding module uses MLP/CNN to extract domain-specific features, requiring retraining when switching data domains (e.g., from web clickstreams to text streams).
   - **Poor budget scalability**: Uses a single fixed-size memory block $M \in \mathbb{R}^{d_1 \times d_2}$; modifying the space budget requires retraining, and the training cost increases as the budget grows.
3. As shown in Figure 2, the advantage of Meta-Sketch over handcrafted sketches actually diminishes in high-budget regions.

### Core Motivation

To design a modular MANN architecture—analogous to building with Lego bricks—that allows neural sketches to freely adapt to different data domains and spatial budgets **without any retraining**.

## Method

### Overall Architecture

Lego Sketch consists of five modules that closely collaborate to execute Store and Query operations:

| Module | Symbol | Function |
|------|------|------|
| Scalable Embedding | $\mathcal{E}$ | Domain-agnostic normalized multi-hash embedding |
| Hash Addressing | $\mathcal{A}$ | Sparse address vector generation via hashing |
| Scalable Memory | $\mathcal{M}$ | Managing $K$ memory bricks with hash-based routing |
| Memory Scanning | $\mathcal{S}$ | Reconstructing stream features from compressed memory |
| Ensemble Decoding | $\mathcal{D}$ | Fusing all information to output frequency estimation |

Each Store/Query operation is a single forward pass with an $O(1)$ computational complexity, independent of the stream length.

### Module 1: Normalized Multi-Hash Embedding ($\mathcal{E}$)

Traditional neural sketches utilize MLP encoders to extract domain-specific features, which necessitates retraining when generalizing across domains. Conversely, the embedding module of Lego Sketch **does not extract any domain-specific features**; instead, it generates embedding vectors restricted to a specific skewness range, achieving domain-agnosticism.

The specific design is inspired by Hash Embeddings in NLP:
- Maintain a learnable vector $V$ and $d_1$ independent hash functions $\{\mathcal{H}_1, \dots, \mathcal{H}_{d_1}\}$.
- Each hash function maps $x_i$ to an index of $V$, retrieving the corresponding value to obtain $v_i \in \mathbb{R}^{d_1}$.
- Perform **$L_1$ normalization** on $v_i$.

The key role of $L_1$ normalization is to maintain cumulative stability during additive memory storage, which directly improves estimation accuracy. Section 4.1 of the paper provides a theoretical proof of domain-agnostic scalability.

### Module 2: Hash Addressing ($\mathcal{A}$)

Use another set of $d_1$ hash functions $\{\mathcal{H}'_1, \dots, \mathcal{H}'_{d_1}\}$ to generate sparse address vectors:

$$a_i = \mathcal{A}(x_i) = \text{SparseVector}(\mathcal{H}'_1(x_i), \dots, \mathcal{H}'_{d_1}(x_i)) \in \mathbb{R}^{d_1 \times d_2}$$

The hashed positions are assigned a value of 1, and the rest are 0. Compared to the learnable addressing in Meta-Sketch, this purely hash-based method is simpler and facilitates scalability.

### Module 3: Scalable Memory ($\mathcal{M}$)

Core Innovation — Replace the single fixed memory block with $K$ **memory bricks** $M_1, \dots, M_K$:

- **Store**: The hash function $\mathcal{H}$ routes $x_i$ to $M_{\mathcal{H}(x_i)}$, executing additive write-in:
  $$M_{\mathcal{H}(x_i)} = M_{\mathcal{H}(x_i)} + v_i \circ a_i$$
  where $\circ$ denotes element-wise multiplication.
- **Query**: Similarly, locate the corresponding brick and extract the information:
  $$m_i = M_{\mathcal{H}(x_i)}^T a_i$$

The original stream $\mathcal{X}$ is uniformly partitioned into $K$ sub-streams, with each sub-stream independently compressed into a memory brick. To expand the space budget, one only needs to increase $K$, **requiring absolutely no retraining**. In experiments, this can easily scale to 140MB to process data streams on a scale of hundreds of millions.

### Module 4: Memory Scanning ($\mathcal{S}$)

An innovative module based on the **DeepSets** architecture, designed to autonomously reconstruct statistical features of the stream (such as the number of elements $n$, distribution skewness $\alpha$, etc.) from compressed memory.

It outputs three features, $s_{\mathcal{H}(x_i)}, s^{(n)}_{\mathcal{H}(x_i)}, s^{(\alpha)}_{\mathcal{H}(x_i)}$, for the decoding module. This enables the model to perceive the distribution characteristics of the current sub-stream during queries, significantly improving estimation accuracy.

### Module 5: Ensemble Decoding ($\mathcal{D}$)

Fuses the memory reading $m_i$, embedding vector $v_i$, and scanning features $s$ to output the frequency estimation $\hat{f}_i$.

### Loss & Training

**Self-Guided Weighting Loss**: Traditional neural sketches utilize a uniformly weighted meta-learning loss $\mathcal{L}_o$, but the difficulty of different meta-tasks varies significantly. This paper proposes a self-guided loss $\mathcal{L}'$ that **dynamically adjusts weights** based on the model's own performance on each meta-task, effectively solving the issue of diminishing advantages under high-budget scenarios.

### theoretical guarantees

1. **Domain-agnostic scalability**: Proving that the skewness of the normalized multi-hash embedding remains within a controllable range, removing the need for domain-specific retraining.
2. **Budget-independent scalability**: Increasing the number of memory bricks $K$ is equivalent to linearly boosting the spatial budget without affecting already trained parameters.
3. **Error bound**: Providing the first theoretical upper bound of estimation error for neural sketches, filling a critical gap in the field.

## Key Experimental Results

### Datasets

| Dataset | Type | Scale | Description |
|--------|------|------|------|
| AOL | Real-world | Real-world | Web search query streams |
| CAIDA | Real-world | Real-world | IP packet streams |
| Zipf (synthetic) | Synthetic | Configurable | Zipf distributions with different skewness parameters |
| Large-scale streams | Multi-domain | 100 million scale | Verifying scalability |

### Main Baselines

- **Handcrafted sketches**: CM-Sketch, C-Sketch, CU-Sketch, A-Sketch
- **Neural sketches**: Meta-Sketch and its variants
- **Metrics**: AAE (Average Absolute Error), ARE (Average Relative Error)

### Core Results

1. **Space-accuracy trade-off** (Figure 2, AOL dataset): Lego Sketch achieves the lowest error across all spatial budgets, and its advantage continuously grows with larger budgets (unlike Meta-Sketch, whose performance advantage diminishes in high-budget regions).
2. **Cross-domain generalization**: The exact same pre-trained model can be directly deployed to a completely different data domain (e.g., AOL $\rightarrow$ CAIDA) **without any retraining**, maintaining competitiveness or even outperforming Meta-Sketch models trained specifically within that domain.
3. **Large-scale scalability**: By increasing the number of memory bricks $K$, the total memory can be scaled up to 140MB to process 100-million-scale data streams, with accuracy continuously improving.
4. **Ablation Study** (Section 5.5) validates the contributions of individual modules:
    - $L_1$ normalization $\rightarrow$ Significant boost in accuracy.
    - Memory Scanning $\rightarrow$ Contributes the most in high-budget scenarios.
    - Self-guided loss $\rightarrow$ Resolves the diminishing advantages at high budgets.

### Derivative Capabilities as a Core Structure

Section 3.3 demonstrates that Lego Sketch can serve as a core structure to replace CM-Sketch and others, combining with external enhancement modules like filters to build even stronger derivative sketch systems.

## Highlights & Insights

1. **Elegant metaphor of Lego bricks**: The modular memory design transforms scalability from "requiring retraining" to simply "adding bricks," which is intellectually and engineering-wise elegant.
2. **Domain-agnostic embedding is a key innovation**: Abandoning the extraction of domain-specific features and instead using hashing + normalization to generate domain-independent embeddings is counter-intuitive yet highly effective.
3. **First error bound for neural sketches**: Introducing theoretical analysis into the neural sketch field significantly enhances the credibility of the methodology.
4. **Self-reflective capacity of Memory Scanning**: Allowing the model to "perceive" the statistical characteristics of the stream currently stored in memory during query is an ingenious way of utilizing information.
5. **Open-source code**: High reproducibility.

## Limitations & Future Work

1. **Training still requires meta-learning**: Although cross-domain/cross-budget deployment does not require retraining, the initial training stage still relies on self-supervised meta-learning, which incurs a non-trivial training cost.
2. **Dependency on hash function quality**: Independent hash functions are used in several places. In real-world applications, the independence and uniformity of hash functions might impact performance.
3. **Sole focus on frequency estimation**: Data streams involve other query tasks (e.g., quantiles, heavy hitter detection). Whether the proposed method generalizes to them remains unverified.
4. **Incomplete details in context**: The complete mathematical details of Memory Scanning and Self-guided Loss were not fully provided in the current context, containing only structural descriptions.
5. **Inability to compare directly with counter-based methods**: The paper admits that sketches and counter-based summarization methods (such as SpaceSaving) apply to different scenarios, yet database advantages under insertion-only stream models warrant further discussion.
6. **Practical engineering deployment cost**: Compared to simple hash + array operations of handcrafted sketches, the forward inference of MANNs incurs additional overhead, which may limit its application in latency-sensitive scenarios.

## Related Work & Insights

- **CM-Sketch / C-Sketch**: Classic handcrafted sketch core structures; Lego Sketch is essentially their neural upgrade.
- **Meta-Sketch** (Cao et al., 2023, 2024): Direct predecessor, which uses fixed MANN for neural sketching. Lego Sketch directly resolves its scalability issues.
- **Hash Embeddings** (Svenstrup et al., 2017): NLP embedding technique that inspired the design of the normalized multi-hash embedding.
- **DeepSets** (Zaheer et al., 2017): Permutation-invariant architecture, utilized in the Memory Scanning module.
- **MANN / Meta-Learning** (Santoro et al., 2016; Graves et al., 2016): Memory-augmented neural network paradigm, which serves as the base architecture of Lego Sketch.

**Insights**: The integration of modular design and domain-agnostic representation could inspire other stream data processing tasks requiring cross-domain generalization (e.g., anomaly detection, changepoint detection).

## Rating

- Novelty: ⭐⭐⭐⭐ — Modular memory bricks + domain-agnostic embedding + first theoretical analysis of neural sketches; a combination of several novel aspects.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experiments including multiple datasets, ablation, and large-scale scenarios; the cross-domain generalization validation is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ — The Lego metaphor runs throughout the paper, with a clear structure and a well-balanced blend of theory and experiments.
- Value: ⭐⭐⭐⭐ — Resolves the core pain point in real-world deployment of neural sketches, offering explicit engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICML 2025\] Predictive Data Selection: The Data That Predicts Is the Data That Teaches](predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [\[ICLR 2026\] Beyond Student: An Asymmetric Network for Neural Network Inheritance](../../ICLR2026/model_compression/beyond_student_an_asymmetric_network_for_neural_network_inheritance.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](../../NeurIPS2025/model_compression/grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)

</div>

<!-- RELATED:END -->
