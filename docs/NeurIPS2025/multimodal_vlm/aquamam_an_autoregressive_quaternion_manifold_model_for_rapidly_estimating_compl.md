---
title: >-
  [Paper Note] AQuaMaM: An Autoregressive, Quaternion Manifold Model for Rapidly Estimating Complex SO(3) Distributions
description: >-
  [NeurIPS 2025][Multimodal VLM][SO(3) distribution] This paper proposes AQuaMaM—a Transformer-based autoregressive quaternion manifold model that represents each projected component of the unit quaternion as a geometrical…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "SO(3) distribution"
  - "quaternion"
  - "autoregressive model"
  - "mixture of uniforms"
  - "pose estimation"
date: 2026-05-08
content_hash: 01b2d50ff289418e
---

# AQuaMaM: An Autoregressive, Quaternion Manifold Model for Rapidly Estimating Complex SO(3) Distributions

**Conference**: NeurIPS 2025
**arXiv**: [2301.08838](https://arxiv.org/abs/2301.08838)  
**Code**: [GitHub](https://github.com/airalcorn2/aquamam)  
**Area**: 3D Rotation Estimation / Manifold Probabilistic Modeling
**Keywords**: SO(3) distribution, quaternion, autoregressive model, mixture of uniforms, pose estimation

## TL;DR

This paper proposes AQuaMaM—a Transformer-based autoregressive quaternion manifold model that represents each projected component of the unit quaternion as a geometrically constrained mixture of uniform distributions, enabling exact likelihood computation and fast sampling on the SO(3) rotation manifold. AQuaMaM achieves 52× faster inference and 14% higher log-likelihood compared to IPDF, with sampled distributions that closely match the ground truth.

## Background & Motivation

**Background**: Accurately modeling complex multimodal distributions over SO(3) (the 3D rotation group) is critical for applications such as robotic grasping and object pose estimation. Standard probability distributions (e.g., multivariate Gaussians) are ill-suited to the curvature of the rotation manifold. **Limitations of Prior Work**: The current strongest method, IPDF (implicit-PDF), implicitly models SO(3) distributions via negative sampling—concise and effective, but with a fundamental accuracy–speed trade-off: inference requires $N$ forward passes (where $N$ sets the accuracy ceiling), with typical configurations using $N_{\text{train}}=4096$ and $N_{\text{test}}=2{,}359{,}296$ (only 0.2% coverage), making it prohibitively slow in environments that cannot exploit massive parallelism. **Key Challenge**: IPDF's likelihood accuracy grows only linearly with grid size, requiring trillions of grid cells to approach ideal precision. Explicit parametric methods (Bingham distributions, von Mises mixtures) lack the flexibility to capture arbitrary multimodal distributions. **Goal**: Design an efficient model that computes exact likelihoods in a single forward pass and can learn arbitrary SO(3) distributions. **Key Insight**: Exploit the geometric property of unit quaternions—since $\mathbf{q}$ and $-\mathbf{q}$ encode the same rotation, one can restrict $q_w > 0$, whereupon $q_w$ is fully determined by $(q_x, q_y, q_z)$, reducing the manifold problem to probability estimation on the unit 3-ball $B^3$. **Core Idea**: Decompose the joint distribution of $(q_x, q_y, q_z)$ via the chain rule, and autoregressively model each component as a mixture of equal-width uniform distributions—essentially a "quaternion language model."

## Method

### Overall Architecture

AQuaMaM is built on a Vision Transformer backbone, taking image patch embeddings and quaternion component embeddings as input. Through a partial causal attention mask, the model autoregressively factorizes the conditional distribution as $p(q_x, q_y, q_z | \mathbf{X}) = p(q_x|\mathbf{X}) \cdot p(q_y|q_x, \mathbf{X}) \cdot p(q_z|q_x, q_y, \mathbf{X})$, where each conditional component distribution is modeled as a mixture of $N$ uniform distributions. The exact probability density on SO(3) is then recovered via a density transformation.

### Key Designs

1. **Mixture-of-Uniforms Modeling for Projected Quaternion Components**:

    - Function: Partition the interval $[-1, 1]$ into $N$ equal-width bins and model the conditional distribution of each quaternion component as a mixture of uniform distributions over these bins.
    - Mechanism: For $q_x$, the density is $p(q_x) = \sum_{i=1}^N \pi_i \mathcal{U}(q_x; a_i, b_i) = \pi_k \cdot N/2$, where $k$ is the bin index containing $q_x$. For the conditional component $q_y|q_x$, the unit-norm constraint $\|q\| = 1$ automatically zeroes out infeasible bins (e.g., bins where $a_i > \sqrt{1-q_x^2}$), and the upper boundary of the boundary bin is adjusted to $\hat{b}_k = \min(\sqrt{1-q_x^2}, b_k)$.
    - Design Motivation: Mixtures of uniforms naturally accommodate distributions of arbitrary complexity and are easy to sample from. Hard-coding geometric constraints (zeroing infeasible bins) injects a strong inductive bias, preventing the model from assigning probability mass to physically impossible configurations.

2. **Manifold Density Transform**:

    - Function: Convert the density $p(q_x, q_y, q_z)$ defined on the flat space $B^3$ into the correct density $p(\mathbf{q})$ on the curved manifold $\widetilde{\mathbb{H}}_1$.
    - Mechanism: The volume scaling factor is computed via the wedge product of the Jacobian of the mapping $f(q_x, q_y, q_z) = [q_x, q_y, q_z, q_w]$ from $B^3$ to $\widetilde{\mathbb{H}}_1$. The expansion factor is $s_{\mathbf{q}} = 1/q_w$, yielding $p(\mathbf{q}) = p(q_x, q_y, q_z) \cdot q_w = \pi_{q_x} \pi_{q_y} \pi_{q_z} \cdot \frac{N q_w}{2 \omega_{q_y} \omega_{q_z}}$.
    - Design Motivation: Directly modeling probability on $B^3$ ignores manifold curvature—points near the manifold boundary ($q_w \to 0$) are dense in $B^3$ but sparse on $\widetilde{\mathbb{H}}_1$. The $q_w$ multiplicative factor corrects this density distortion.

3. **"Quaternion Language Model" Training Paradigm**:

    - Function: Reformulate rotation probability estimation as a three-token autoregressive language model training problem.
    - Mechanism: After taking the negative log-likelihood, the constant terms from the density transform can be ignored, yielding the loss $\hat{\mathcal{L}} = -\sum_d (\ln \pi_{q_{d,x}} + \ln \pi_{q_{d,y}} + \ln \pi_{q_{d,z}})$, which is exactly the sum of cross-entropy losses over three classifiers. The model only needs to learn to assign high probability to the correct bin. The precision lower bound on likelihood scales as $N^3$ (i.e., $\geq N^3 q_w / 8$), compared to IPDF's linear scaling with grid size.
    - Design Motivation: Reformulating continuous distribution estimation on a manifold as a standard classification problem allows the use of mature Transformer architectures and cross-entropy training. Parameter sharing across the three components (using a shared Transformer body) effectively circumvents the curse of dimensionality.

### Loss & Training

The training loss is the sum of cross-entropy losses over three components. After taking the negative log-likelihood, the factor $\frac{Nq_w}{2\omega_{q_y}\omega_{q_z}}$ from the manifold density transform is constant for a given dataset and can be ignored during optimization, reducing the loss to a standard three-classifier cross-entropy: $\hat{\mathcal{L}} = -\sum_d (\ln \pi_{q_{d,x}} + \ln \pi_{q_{d,y}} + \ln \pi_{q_{d,z}})$. At inference, greedy decoding generates "quaternion sentences," and a KV-cache strategy reduces the attention complexity of three forward passes from $O(3(P+3)^2)$ to $O((P+1)^2 + (P+2) + (P+3))$, yielding approximately 2× throughput improvement in practice. Quaternion component embeddings are mapped from NeRF-style positional encodings ($1 + 2L$ dimensional input) to $d_{\text{model}}$-dimensional space via an MLP, providing high-frequency features for continuous values. The model is trained with the Adam optimizer. AQuaMaM has approximately 20M parameters for the dice experiment ($N=500$ bins) and approximately 3.5M parameters for the toy experiment ($N=50{,}257$ bins), with 93% of parameters concentrated in the final classification layers.

## Key Experimental Results

### Main Results

Quantitative comparison on the "dice" dataset (500K rendered images with varying degrees of view ambiguity):

| Method | Mean Log-Likelihood↑ | Prediction Error (°)↓ | Inference Speed | Parameters |
|--------|---------------------|----------------------|-----------------|------------|
| IPDF (2.4M grid) | 12.29 | 4.57° | 1× | ~26M |
| **AQuaMaM** ($N=500$) | **14.01** (+14%) | **4.32°** (−5.5%) | **52×** | ~20M |

### Ablation Study

Distribution learning quality on the "infinite" toy dataset (6 view classes, each with $2^i$ rotational modes):

| Method | Mean Log-Likelihood↑ | Sampled vs. True Distribution | Mean Geodesic Dist. (°) | Error Rate |
|--------|---------------------|-------------------------------|------------------------|------------|
| IPDF | 12.32 (theoretical max 12.38) | **Severely mismatched** | 0.84° | High |
| **AQuaMaM** | **27.12** | **Precisely matched** | **0.04°** | 0.06% |

IPDF would theoretically require approximately 6 trillion grid cells to match AQuaMaM's log-likelihood.

### Key Findings

1. **IPDF's sampled distribution catastrophically diverges from the true distribution**: Despite evaluation loss approaching the theoretical minimum, IPDF's sampled distribution severely mismatches the true uniform distribution—demonstrating that low evaluation loss does not imply good distributional learning.
2. **AQuaMaM's cubic likelihood scaling advantage**: Log-likelihood precision scales as $N^3$ (vs. IPDF's $N$); at GPT-2-scale vocabulary ($N=50{,}257$), $N^3 = 1.26 \times 10^{14}$.
3. **Effective modeling of view ambiguity**: On the dice dataset, AQuaMaM correctly assigns high multimodal probability to symmetric viewpoints and concentrates probability for unambiguous viewpoints (clearly illustrated in Figure 7 visualizations).
4. **52× speedup on a single GPU**: For deployment environments that cannot exploit large-scale parallelism (e.g., edge devices, robotic systems), this speedup is of decisive practical significance.

## Highlights & Insights

- **Recasting SO(3) distribution estimation as a language model is an exceptionally elegant idea**—the unit-norm constraint on quaternions naturally parallels vocabulary constraints in language models; binning naturally corresponds to tokenization; and the manifold density transform requires only a single constant correction.
- **Hard-coding geometric constraints (zeroing infeasible bins) constitutes a subtle yet powerful inductive bias**—the model need not learn that certain rotation combinations are impossible; this fact is encoded directly in the architecture, substantially reducing learning difficulty.
- **AQuaMaM exposes a fundamental flaw in IPDF**—IPDF can approach the theoretical minimum evaluation loss while its sampled distribution remains catastrophically misaligned with the true distribution, implying that IPDF's evaluation metrics do not reflect real deployment performance.
- **Elegant circumvention of the curse of dimensionality**—three parameter-sharing $N$-class classifiers replace a single $N^3$-class classifier, drastically reducing parameter count while enabling more thorough learning at each classifier.

## Limitations & Future Work

- Validation is limited to a constructed toy dataset and a rendered dice dataset; the absence of direct comparisons on standard pose estimation benchmarks (e.g., PASCAL3D+, T-LESS, SYMSOL I/II) limits the persuasiveness of the performance evaluation.
- As $q_w \to 0$ (near 180° rotations), the rotational range covered by a single bin increases significantly, degrading local precision—this can be mitigated by increasing $N$, but at a linear cost in the number of classification layer parameters.
- Autoregressive decoding requires 3 forward passes (despite KV-cache optimization), which may still be insufficient for ultra-low-latency scenarios (e.g., per-frame pose estimation in real-time robot control).
- The choice of component ordering $(q_x, q_y, q_z)$ may affect the complexity of conditional distributions and the difficulty of learning, yet no ablation study is conducted to verify this.
- Training requires ground-truth rotation matrices, and applicability to semi-supervised or self-supervised settings is unexplored.
- The mixture-of-uniforms representation may require very large $N$ to resolve all modes for highly multimodal distributions (e.g., symmetric objects with more than 100 symmetry modes).

## Related Work & Insights

- **vs. IPDF (Murphy et al., 2021)**: IPDF is the de facto standard for implicit SO(3) modeling, estimating rotation density via negative sampling and softmax normalization. AQuaMaM comprehensively surpasses it across three dimensions: likelihood (+14%), speed (52×), and sampling quality (0.06% error rate vs. severely mismatched sampled distributions). The key distinction is IPDF's linear precision scaling vs. AQuaMaM's cubic scaling.
- **vs. Bingham/von Mises mixtures (Gilitschenski et al., 2020; Prokudin et al., 2018)**: Parametric distribution mixtures are limited in expressiveness by the shape of the predefined distribution family—they lag significantly behind in IPDF's evaluations. AQuaMaM's nonparametric $N$-bin approach can approximate any distribution to arbitrary discretization precision.
- **vs. Deep Bingham Networks (Deng et al., 2020)**: Winner-takes-all training alleviates mixture density network training difficulties, but remains constrained by the parametric shape of the Bingham distribution and is insufficient for highly asymmetric or fine-grained multimodal distributions.
- **vs. Direct classification methods (Mahendran et al., 2018)**: Discretizing rotation space into a fixed grid for direct classification requires $O(N^3)$ classes (curse of dimensionality), with prior work supporting only 200 rotations. AQuaMaM reduces this to three $O(N)$ classifiers via autoregressive factorization.
- **Insights**: The autoregressive mixture-of-uniforms paradigm can be generalized to distribution estimation on other manifolds (e.g., $S^2$ spherical directions, SE(3) full poses, trajectory modeling), and to any scenario requiring complex distribution modeling in constrained spaces.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The "quaternion language model" concept is highly original; the combination of geometric constraint encoding and manifold density transformation is elegant.
- Experimental Thoroughness: ⭐⭐⭐ — Datasets are relatively simple (toy data and rendered dice); standard benchmark comparisons are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are complete and clear; visualizations are innovative (opacity encoding probability density).
- Value: ⭐⭐⭐⭐ — High methodological value with broad generalizability to other manifolds; direct impact is somewhat constrained by the limited breadth of experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs](../../AAAI2026/multimodal_vlm/vmfcoop_towards_equilibrium_on_a_unified_hyperspherical_manifold_for_prompting_b.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)
- [\[NeurIPS 2025\] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)

</div>

<!-- RELATED:END -->
