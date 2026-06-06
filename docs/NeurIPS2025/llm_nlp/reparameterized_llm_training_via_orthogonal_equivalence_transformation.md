---
title: >-
  [Paper Note] Reparameterized LLM Training via Orthogonal Equivalence Transformation
description: >-
  [NeurIPS 2025][LLM/NLP][reparameterized training] This paper proposes POET, a training framework that reparameterizes weight matrices as the product of two learnable orthogonal matrices and a fixed random weight matrix…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "reparameterized training"
  - "orthogonal transformation"
  - "spectrum preservation"
  - "LLM pretraining"
  - "efficient training"
date: 2026-05-08
content_hash: a708e5c8fa51133c
---

# Reparameterized LLM Training via Orthogonal Equivalence Transformation

**Conference**: NeurIPS 2025
**arXiv**: [2506.08001](https://arxiv.org/abs/2506.08001)  
**Code**: [spherelab.ai/poet](https://spherelab.ai/poet)  
**Area**: LLM/NLP
**Keywords**: reparameterized training, orthogonal transformation, spectrum preservation, LLM pretraining, efficient training

## TL;DR

This paper proposes POET, a training framework that reparameterizes weight matrices as the product of two learnable orthogonal matrices and a fixed random weight matrix, thereby preserving spectral properties throughout training to achieve more stable optimization and improved generalization with fewer trainable parameters than AdamW.

## Background & Motivation

- LLM pretraining typically employs AdamW to directly optimize weight matrices, giving rise to three core issues:
    - Computationally intensive and poorly scalable with model size
    - Requires careful hyperparameter tuning to ensure stable convergence
    - Generalization performance may remain suboptimal even when training loss is perfectly minimized
- The spectral properties (singular values) of weight matrices are closely linked to generalization—smaller spectral norms generally correspond to stronger generalization
- Limitations of existing spectral control methods:
    - **Ineffective spectral control**: constraining only the largest singular value fails to regularize the full singular value spectrum
    - **High computational overhead**: both spectral norm regularization and spectral normalization require computing the largest singular value (via power iteration)
- Core idea: orthogonal transformations do not alter singular values → transforming fixed weights with orthogonal matrices automatically preserves spectral properties

## Method

### Overall Architecture

POET reparameterizes a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ as:

$$\mathbf{W}_{RP} = \mathbf{R} \mathbf{W}_0 \mathbf{P}$$

where:
- $\mathbf{W}_0$: a randomly initialized weight matrix that remains **frozen** throughout training
- $\mathbf{R} \in \mathbb{R}^{m \times m}$: a learnable left orthogonal matrix (transforming the column space / left singular vectors)
- $\mathbf{P} \in \mathbb{R}^{n \times n}$: a learnable right orthogonal matrix (transforming the row space / right singular vectors)

The forward pass is $\mathbf{y} = (\mathbf{R}\mathbf{W}_0\mathbf{P})^\top \mathbf{x}$. After training, $\mathbf{R}$ and $\mathbf{P}$ can be merged into the weight matrix, leaving inference speed unchanged.

**Spectrum preservation**: Given $\mathbf{W}_0 = \mathbf{U}\mathbf{\Sigma}_0\mathbf{V}^\top$, the reparameterized matrix satisfies $\mathbf{W}_{RP} = \mathbf{RU}\mathbf{\Sigma}_0\mathbf{V}^\top\mathbf{P}$, so the singular values $\mathbf{\Sigma}_0$ remain entirely unchanged.

### Key Designs

**1. Stochastic Primitive Orthogonal Optimization (SPO)**

Directly optimizing an $m \times m$ orthogonal matrix is computationally prohibitive. SPO decomposes it into a product of "primitive orthogonal matrices":

- **Full stochastic SPO**: randomly samples a subset $\mathbf{S}$ of $b$ indices and embeds a small $b \times b$ orthogonal matrix into an identity matrix:
  $$\mathbf{R} = \prod_{i=1}^c \left(\mathbf{I}_m + \mathbf{D}(\mathbf{S}^i)(\tilde{\mathbf{G}}_i - \mathbf{I}_b)\mathbf{D}(\mathbf{S}^i)^\top\right)$$

- **Block stochastic SPO**: constructs a block-diagonal orthogonal matrix combined with random permutations, ensuring all dimensions are transformed:
  $$\mathbf{R} = \prod_{i=1}^c \left(\mathbf{\Psi}_i^\top \cdot \text{Diag}(\tilde{\mathbf{G}}_i^1, \ldots, \tilde{\mathbf{G}}_i^{\lceil m/b \rceil}) \cdot \mathbf{\Psi}_i\right)$$

**2. Cayley–Neumann Parameterization (CNP)**

A truncated Neumann series is used to approximate the matrix inverse in the Cayley parameterization:

$$\mathbf{R} = (\mathbf{I}+\mathbf{Q})(\mathbf{I}-\mathbf{Q})^{-1} \approx (\mathbf{I}+\mathbf{Q})\left(\mathbf{I} + \sum_{i=1}^k \mathbf{Q}^i\right)$$

where $\mathbf{Q}$ is a skew-symmetric matrix. Setting $k=3$ achieves a favorable performance–efficiency trade-off.

**3. Merge-then-Reinitialize**

At fixed intervals, the learned orthogonal matrices are merged into the weights ($\mathbf{W} \leftarrow \mathbf{RWP}$) and then reset to identity matrices. This:
- Substantially reduces GPU memory usage (only one primitive matrix is stored at a time)
- Prevents accumulation of Neumann approximation errors
- Allows resampling of index sets and permutations

**4. Initialization Schemes**

Two novel initialization strategies are proposed:
- **Normalized Gaussian**: normalizes neurons sampled from a standard Gaussian distribution (best empirical performance)
- **Uniform Spectrum**: applies SVD to a standard initialization and sets all singular values to 1

### Loss & Training

The complete training algorithm:
1. Initialize weights with Normalized Gaussian: $\mathbf{W} \leftarrow \mathbf{W}_0$
2. Randomly sample indices/permutations; initialize small orthogonal matrices to identity
3. Construct $\mathbf{R}$ and $\mathbf{P}$ via SPO + CNP
4. Inner training loop: forward pass uses $\mathbf{RWP}$; backpropagation updates the small orthogonal parameters
5. Merge and reinitialize; return to step 2

## Key Experimental Results

### Main Results: LLaMA Pretraining

| Model (tokens) | AdamW | GaLore | LoRA (r=64) | POET-BS b=256 | POET-FS b=1/2 |
|--------------|-------|--------|-------------|--------------|--------------|
| 60M (30B) | 26.68 (25.3M) | 29.81 (25.3M) | 39.70 (4.85M) | 25.29 (9.66M) | 25.37 (8.54M) |
| 130M (40B) | 20.82 (84.9M) | 22.35 (84.9M) | 32.07 (11.2M) | 19.88 (22.3M) | 19.94 (28.6M) |
| 350M (40B) | 16.78 (302M) | 17.99 (302M) | 25.19 (30.3M) | 16.27 (60.3M) | 15.95 (102M) |
| 1.3B (50B) | 14.73 (1.21B) | 18.33 (1.21B) | 20.55 (59.4M) | **14.56** (118M) | **13.70** (407M) |

POET-FS (b=1/2) surpasses AdamW on the 1.3B model with approximately one-third of the trainable parameters (13.70 vs. 14.73). The advantage is maintained on the 3B model as well (16.90 vs. 19.61).

### Downstream Fine-tuning (GLUE)

| Fine-tuning | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | STS-B |
|---------|------|------|------|------|-----|-----|-------|-------|
| Full FT + AdamW | 0.361 | 0.658 | 0.696 | 0.818 | 0.829 | 0.534 | 0.914 | 0.880 |
| Full FT + POET | 0.523 | 0.818 | 0.824 | 0.885 | 0.902 | 0.661 | 0.920 | 0.873 |
| POET FT + POET | **0.505** | **0.821** | **0.826** | **0.892** | **0.902** | **0.682** | **0.931** | **0.887** |

Models pretrained with POET consistently outperform AdamW-pretrained baselines across all tasks and fine-tuning configurations.

### Ablation Study

| Ablation | Finding |
|--------|------|
| Initialization scheme | Normalized Gaussian is optimal (25.37); Uniform Spectrum performs worst (27.29) |
| Merge frequency $T_m$ | 200 and 400 are optimal; performance degrades when too small (5) or too large (1600) |
| Neumann terms $k$ | $k=0$ causes training divergence; $k=3$ achieves the best performance–efficiency balance |
| $\mathbf{R}$:$\mathbf{P}$ parameter allocation | Equal 50:50 split is optimal |
| FS vs. BS | Block stochastic is more parameter-efficient (better coverage of weight parameters) |

### Key Findings

- POET training exhibits three distinct phases: (1) cone-shell search, (2) stable learning on the cone shell, and (3) late-stage fine-tuning
- POET maintains its advantage even when AdamW is trained on nearly 3× more tokens, indicating non-trivial generalization improvement
- POET sustains high SVD entropy (diverse spectral distribution), outperforming both AdamW and Muon in this regard
- POET maintains low hyperspherical energy across layers, indicating uniform neuron distribution
- Performance correlates strongly with parameter budget, exhibiting scaling-law-like behavior

## Highlights & Insights

1. **Theoretical elegance**: POET is derived from spectral preservation theory; Theorem 1 proves that linear transformations preserving spectral properties must be orthogonal equivalence transformations
2. **Generalization guarantee**: generalization bounds are provided via spectrally-normalized margin theory
3. **Efficiency breakthrough**: AdamW full-parameter training is surpassed using only 1/3–1/10 of the trainable parameters
4. **Zero inference overhead**: orthogonal matrices are merged into weights after training, leaving the inference architecture completely unchanged
5. **Natural generalization of OFT**: extends energy-preserving training to spectrum-preserving training by introducing the right orthogonal matrix $\mathbf{P}$ for additional flexibility
6. **Three-phase learning dynamics**: vector probing analysis reveals interesting patterns in how the orthogonal matrices evolve during training

## Limitations & Future Work

- POET converges more slowly than AdamW in early training (Phase II characteristic), potentially resulting in longer total training time
- The merge-then-reinitialize frequency $T_m$ is a hyperparameter requiring tuning
- The SPO block size $b$ must also be selected, with optimal settings varying across model scales
- Setting $k=0$ in CNP causes training divergence, indicating a strong dependency on maintaining orthogonality
- Combinations with more advanced optimizers (e.g., Muon, SOAP) remain underexplored
- The theoretical reason for the poor performance of Uniform Spectrum initialization requires further analysis

## Related Work & Insights

- **Alternative to LoRA**: POET significantly outperforms LoRA under comparable parameter budgets, suggesting that spectrum preservation is more effective than low-rank assumptions
- **Improvement over GaLore**: GaLore relies on low-rank gradient approximations, whereas POET avoids information loss through orthogonal transformations
- **Complementary to Muon**: Muon also promotes spectral diversity; POET can potentially be combined with it
- **Connection to random neural networks**: POET-trained weights are statistically indistinguishable from random initialization (Gaussian isometry invariance)
- **Implications for pretraining paradigms**: POET demonstrates that "learning a transformation" rather than "directly learning weights" is a promising direction

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — reparameterized training derived from spectral preservation theory; SPO + CNP design is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ — validated across scales from 60M to 3B; downstream evaluation and ablations are comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ — complete theoretical derivations, in-depth experimental analysis, and insightful three-phase learning dynamics
- Value: ⭐⭐⭐⭐⭐ — significant implications for LLM pretraining methodology; potential to reshape large-scale model training paradigms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](../../ICLR2026/llm_nlp/pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)

</div>

<!-- RELATED:END -->
