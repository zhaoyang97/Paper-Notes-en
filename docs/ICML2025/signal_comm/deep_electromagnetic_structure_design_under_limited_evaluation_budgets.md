---
title: >-
  [Paper Note] Deep Electromagnetic Structure Design Under Limited Evaluation Budgets
description: >-
  [ICML 2025][Signal & Communication][Electromagnetic structure design] Proposes the Progressive Quadtree-based Search (PQS) method, which compresses the high-dimensional design space of electromagnetic structures via a hierarchical quadtree representation and utilizes a consistency-based sample selection mechanism to efficiently search for high-quality designs under a limited simulation budget, saving 75–85% of evaluation costs compared to generative methods.
tags:
  - "ICML 2025"
  - "Signal & Communication"
  - "Electromagnetic structure design"
  - "Quadtree search"
  - "Surrogate model"
  - "Limited evaluation budget"
  - "Sample selection"
date: 2026-05-08
content_hash: 8df1e07a7041fae6
---

# Deep Electromagnetic Structure Design Under Limited Evaluation Budgets

**Conference**: ICML 2025  
**arXiv**: [2506.19384](https://arxiv.org/abs/2506.19384)  
**Code**: None  
**Area**: Signal & Communication  
**Keywords**: Electromagnetic structure design, Quadtree search, Surrogate model, Limited evaluation budget, Sample selection

## TL;DR

Proposes the Progressive Quadtree-based Search (PQS) method, which compresses the high-dimensional design space of electromagnetic structures via a hierarchical quadtree representation and utilizes a consistency-based sample selection mechanism to efficiently search for high-quality designs under a limited simulation budget, saving 75–85% of evaluation costs compared to generative methods.

## Background & Motivation

Electromagnetic structure (EMS) design is crucial in fields such as antennas, frequency selective surfaces, and metamaterials. However, this problem faces two core challenges:

**Massive Design Space**: For a $12 \times 24$ grid, each cell has two states (metal/vacuum), resulting in a search space of $2^{288} \approx 10^{86}$, which far exceeds NAS ($10^4 \sim 10^{18}$) and molecular design ($10^6$).

**Extremely Expensive Evaluation**: Each evaluation requires full-wave electromagnetic simulation solving Maxwell's equations. A single simulation takes between 660 and 42,780 seconds, making simple analytical approximations impractical.

Existing methods mainly fall into two categories:
- **Predictor-based methods**: Train a DNN surrogate model to approximate the simulation function, but still search in the pixel-level high-dimensional space; furthermore, highly accurate surrogates require a massive amount of training data (typically 10,000 to 2 million samples).
- **Generative methods**: Utilize models like cGAN or cVAE to directly generate designs that satisfy constraints, but likewise require large-scale datasets for training.

Unlike NAS and molecular design, EMS design **lacks public datasets, pre-trained models, and data augmentation methods**, and each task in industrial scenarios is highly customized, requiring optimization from scratch. Therefore, there is an urgent need for a method that can still find high-quality designs under a limited simulation budget (e.g., 1000 simulations).

## Method

### Overall Architecture

The PQS framework contains two complementary modules:
1. **Quadtree-based Search Strategy (QSS)**: Represents the EMS layout using a quadtree, transforming pixel-level search into a hierarchical progressive search that refines from global patterns to local details.
2. **Consistency-based Sample Selection (CSS)**: Dynamically allocates the simulation budget based on prediction consistency metrics, achieving a balance between exploitation and exploration.

Overall process: Initialize dataset $D_0$ $\to$ Train predictor $f_\theta$ $\to$ Search candidate designs with QSS $\to$ Select simulation samples with CSS $\to$ Run simulations to get feedback $\to$ Update dataset and predictor $\to$ Iterate until budget is exhausted.

### Key Designs

#### 1. **Quadtree-based Representation**

The traditional pixel-level layout matrix is converted into a quadtree structure. Core idea: Homogeneous regions are represented by a single leaf node (requiring only 1 bit), while complex regions are recursively split into 4 child nodes.

Each node $n$ corresponds to a rectangular sub-region of the layout matrix, determined by row indices $[r_n^{\text{start}}, r_n^{\text{end}}]$ and column indices $[c_n^{\text{start}}, c_n^{\text{end}}]$. Under refinement, it splits at the midpoint:

$$r_{\text{mid}} = \left\lfloor \frac{r_n^{\text{start}} + r_n^{\text{end}}}{2} \right\rfloor, \quad c_{\text{mid}} = \left\lfloor \frac{c_n^{\text{start}} + c_n^{\text{end}}}{2} \right\rfloor$$

Each leaf node stores a binary state $s_n \in \{0, 1\}$, and the complete matrix is reconstructed through all leaf nodes:

$$x_{i,j} = \sum_{n \in L} s_n \cdot \mathbb{I}_n(i,j)$$

where $L$ is the set of leaf nodes and $\mathbb{I}_n(i,j)$ is an indicator function. The design space is compressed from the original $2^{m \times n}$ to $2^{|L|}$. By controlling $|L|$, the complexity of the design space can be progressively managed.

**Design Motivation**: Pixel-level search faces the curse of dimensionality. The quadtree utilizes the spatial locality of EMS layouts, avoiding pixel-by-pixel search for homogeneous regions and significantly reducing the effective search dimension.

#### 2. **Progressive Tree Search**

The search starts from the simplest design space (the root node) and gradually increases complexity. Each iteration randomly selects a leaf node and executes one of two operations with a 0.5 probability:
- **Resampling**: Modifies the state of leaf node $s_n$ without expanding the space.
- **Splitting**: Subdivides the leaf node into 4 child nodes to increase search granularity.

Maintains a Top-K optimal design list, continuously updating it until the number of leaf nodes reaches the upper limit $N_{\max}$.

#### 3. **Depth-wise Importance Assignment**

Following the tree search stage, further optimization of the Top-K designs is performed. Unlike the uniform distribution during initial splits, this stage fine-tunes the partitioning parameters of the quadtree (the row/column range of each node) to more accurately characterize key regions:

$$\max_{\mathbf{s'} \in \mathcal{S'}} O(f_\theta(\mathbf{x_{s'}}))$$

where $\mathcal{S'} = \{(r_n^{\text{start}}, r_n^{\text{end}}, c_n^{\text{start}}, c_n^{\text{end}}) \mid n \in Q\}$.

#### 4. **Consistency-based Sample Selection (CSS)**

Uses Kendall's tau coefficient $\tau$ to measure prediction ranking consistency between adjacent iterations:

$$\tau = \frac{2}{n(n-1)} \sum_{i<j} \text{sign}(O(f_{\theta_{t-1}}(\mathbf{x}_i)) - O(f_{\theta_{t-1}}(\mathbf{x}_j))) \cdot \text{sign}(O(f_{\theta_t}(\mathbf{x}_i)) - O(f_{\theta_t}(\mathbf{x}_j)))$$

A $\tau$ close to 1 indicates high prediction consistency, while a value close to 0 or negative indicates low consistency.

**Design Motivation**: Ranking accuracy is more critical than numerical accuracy—a model that can correctly rank candidates can effectively guide optimization even if its predicted values are biased.

### Loss & Training

**Objective function** uses the minimum of multi-criteria to ensure all performance metrics are not lower than the threshold:

$$O(S(\mathbf{x})) = \min_{1 \leq k \leq p} S_k(\mathbf{x})$$

**Hybrid selection strategy**: Dynamically allocates simulation budget based on the $\tau$ value:
- $R_p = \tau \times R$ samples recommended by the predictor (exploitation)
- $R_r = (1-\tau) \times R$ samples randomly selected (exploration)

Automatically increases the proportion of random exploration when model predictions are unreliable, preventing bias accumulation from inaccurate predictions.

The predictor adopts a ResNet50 architecture, with an initial dataset of 300 samples and a total simulation budget limited to 1000.

## Key Experimental Results

### Main Results

Evaluated on two real-world engineering tasks: Dual-layer Frequency Selective Surface (DualFSS, $12 \times 12 \times 2$, space $10^{86}$) and High-Gain Antenna (HGA, $15 \times 20$, space $10^{90}$).

| Method | DualFSS Agg Obj↑ | # Sims | HGA Agg Obj↑ | # Sims |
|------|-----------------|-------|-------------|-------|
| Random Sampling | 7.28 | 1000 | 0.63 | 1000 |
| Surrogate-RS | 5.81 | 1000 | 3.09 | 1000 |
| Surrogate-GA | 4.19 | 1000 | 1.58 | 1000 |
| TS-DDEO | 5.56 | 1000 | 0.52 | 1000 |
| cGAN | 3.13 | 7000 | -1.09 | 4000 |
| cVAE | 8.93 | 7000 | -1.37 | 4000 |
| InvGrad | 2.89 | 7000 | 3.18 | 4000 |
| GenCO | 1.18 | 7000 | -5.30 | 4000 |
| **PQS (Ours)** | **15.20** | **1000** | **3.66** | **1000** |

PQS outperforms all baselines using only 1000 simulations. It achieves a 109% improvement over random search on DualFSS, and a 70% improvement over cVAE (a generative method using 7000 simulations).

### Ablation Study

Conducted on the HGA task.

**Impact of the number of variables $N_{\max}$:**

| $N_{\max}$ | Agg Obj↑ | Obj1↑ | Obj2↑ | Kendall's Tau↑ |
|-----------|---------|-------|-------|---------------|
| 16 | 3.09 | 3.09 | 6.03 | 0.284  0.053 |
| 32 | **3.66** | **3.66** | **6.48** | 0.232  0.052 |
| 64 | 3.02 | 3.02 | 5.18 | 0.126  0.030 |

$N_{\max} = 32$ achieves the best performance. Too small a value easily leads to local optima, while too large a value makes it difficult for the predictor to model accurately under limited data.

**Effectiveness of QSS and CSS modules:**

| QSS | CSS | Agg Obj↑ | Obj1↑ | Obj2↑ |
|-----|-----|---------|-------|-------|
| ✗ | ✗ | 3.09 | 3.09 | 3.18 |
| ✓ | ✗ | 3.22 | 3.22 | 6.34 |
| ✓ | ✓ | **3.66** | **3.66** | **6.48** |

Both modules contribute positively, with CSS further improving performance by 13.7% on top of QSS.

### Key Findings

1. **Predictor-based methods perform poorly in few-shot scenarios**: Surrogate-RS (5.81 dB) is even lower than pure random search (7.28 dB), indicating that 1000 samples are insufficient to train a reliable predictor to guide search.
2. **Generative methods require substantial budgets yet still underperform PQS**: Methods like cGAN/cVAE/InvGrad using 7 times the simulation budget all underperform PQS, highlighting the advantages of the hierarchical representation and consistency-driven mechanisms.
3. **Excellent robustness**: Across 10 independent runs, PQS achieves an Agg Obj of 4.34  0.34 with the lowest variance, whereas IDN yields -17.08  4.23 and cVAE yields -4.04  1.09.
4. **CSS improves search efficiency by 50%**: In experiments finding the optimal samples in the dataset, CSS improves efficiency by approximately 50% compared to Top-K and Random strategies.

## Highlights & Insights

1. **The design of combining quadtree and progressive search is highly elegant**: It transforms the curse of dimensionality into a hierarchical progressive refinement task, moving from coarse-grained exploration of the global structure to fine-grained local details. This aligns with engineering intuition and effectively reduces search complexity.
2. **Using Kendall's tau instead of numerical error for consistency measurement**: Recognizing that ranking correctness is more crucial than prediction accuracy, this insight allows the surrogate model to effectively guide the search even when it is not highly precise.
3. **Significant savings in evaluation costs**: Compared to generative methods, it saves 75–85% of simulation costs, translating to a reduction of 20–39 days in the product design cycle, showing prominent industrial application value.
4. **Clear problem modeling**: The analogy and comparison with NAS and molecular design (Table 1) accurately pinpoints the unique challenges of EMS design—no public data, no pre-trained models, and no data augmentation.

## Limitations & Future Work

1. **Binary design space limitation**: The current method assumes each cell has only 0/1 states, without considering multi-material or continuous parameterized designs.
2. **Symmetry is underutilized**: Many EMS structures possess rotational or reflection symmetry. The quadtree representation does not explicitly encode this prior, which could be combined with symmetry constraints to further compress the search space.
3. **Fixed predictor architecture (ResNet50)**: More lightweight models or models better suited for structured inputs (such as GNNs) have not been explored; they might achieve better ranking capabilities with smaller data volumes.
4. **Insufficient scalability validation**: It has only been evaluated on two engineering tasks and not validated on larger scales (e.g., $50 \times 50$ grids) or scenarios with more design objectives.
5. **Lack of comparison with Bayesian optimization methods**: Well-established low-sample optimization methods such as BOHB or TPE were not included in the comparison.

## Related Work & Insights

- **Connection to NAS**: The progressive search concept in PNAS (Liu et al., 2018a) shares similarities with the progressive quadtree search in this paper. However, NAS benefits from data augmentation techniques like weight sharing, whereas EMS design lacks such infrastructure.
- **Surrogate-assisted optimization**: SAHSO (Li et al., 2022) and TS-DDEO (Zheng et al., 2023b) represent the latest advances in surrogate-assisted evolutionary algorithms, yet they still struggle in low-sample scenarios.
- **GenCO (Ferber et al., 2024)**: Employs a combined strategy of VAE generation + gradient ascent optimization. However, its poor performance in the EMS scenario suggests that generative models struggle to capture high-dimensional structural patterns when large amounts of training data are lacking.
- **Insights**: The hierarchical search concept in this work can be transferred to other scenarios requiring optimization of high-dimensional discrete structures under limited budgets, such as FPGA layout, circuit design, and 3D printing structure optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The use of hierarchical quadtree representations for EMS design is pioneering, and the combination of progressive search and consistency-based selection is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two real-world engineering tasks against 11 baselines. The ablation studies and robustness analyses are comprehensive, although validation on larger scales is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definitions are clear, and the analogy to NAS/molecular design helps position the work. The overall structure is smooth.
- **Value**: ⭐⭐⭐⭐⭐ It dramatically reduces simulation costs (saving 20–39 days), providing direct practical value for industrial EMS design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](../../ICML2026/signal_comm/meta-learning_structure-preserving_dynamics.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[CVPR 2025\] Continuous Space-Time Video Resampling with Invertible Motion Steganography](../../CVPR2025/signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)

</div>

<!-- RELATED:END -->
