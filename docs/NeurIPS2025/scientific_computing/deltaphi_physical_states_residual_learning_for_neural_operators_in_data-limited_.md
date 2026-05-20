---
title: >-
  [Paper Note] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving
description: >-
  [NeurIPS 2025][Scientific Computing][neural operators] This paper proposes DeltaPhi, a framework that forgoes direct learning of the input-to-output mapping for PDEs and instead learns **residuals between similar physica…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "neural operators"
  - "residual learning"
  - "data efficiency"
  - "PDE solving"
  - "implicit data augmentation"
date: 2026-05-08
content_hash: 578d992fb5979ae2
---

# DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving

**Conference**: NeurIPS 2025
**arXiv**: [2406.09795](https://arxiv.org/abs/2406.09795)  
**Code**: [https://github.com/yuexihang/DeltaPhi](https://github.com/yuexihang/DeltaPhi)  
**Area**: Scientific Computing / Neural PDE Solvers
**Keywords**: neural operators, residual learning, data efficiency, PDE solving, implicit data augmentation

## TL;DR

This paper proposes DeltaPhi, a framework that forgoes direct learning of the input-to-output mapping for PDEs and instead learns **residuals between similar physical states**. By exploiting the stability of physical systems as implicit data augmentation, DeltaPhi significantly improves the performance of diverse neural operators under data-scarce regimes.

## Background & Motivation

**Background**: Neural operators (e.g., FNO, DeepONet) have achieved substantial progress in learning PDE solution operators, yet their performance is highly dependent on the amount of training data. In practical physics and engineering settings, acquiring high-quality numerical PDE solutions is prohibitively expensive, making data scarcity a central bottleneck.

**Limitations of Prior Work**: Existing approaches to mitigating data insufficiency fall into two main categories: (a) incorporating physical equations as soft constraints (e.g., PINN-based methods), which suffer from difficult PDE residual optimization and require knowledge of the exact equation form; and (b) data augmentation (e.g., rotation, flipping), where generic strategies struggle to preserve the physical consistency of PDE solutions.

**Key Challenge**: For well-posed PDE problems, solutions depend continuously on initial/boundary conditions: the closer two initial conditions, the closer their corresponding trajectories. This Lipschitz continuity $\|G(a_1) - G(a_2)\| \leq C\|a_1 - a_2\|$ provides a critical prior: **the difference (residual) between outputs corresponding to similar inputs is small and smooth, and thus easier to learn than the original mapping**.

**Residual vs. Direct Mapping**: Directly learning $a \mapsto u$ requires fitting a complex high-dimensional mapping, whereas learning the residual $\Delta u = G(a_i) - G(a_k)$ (where $a_k$ is a sample similar to $a_i$) yields a target of smaller magnitude and smoother variation, reducing the learning difficulty.

**Goal**: For $N$ training samples, conventional methods yield only $N$ input-output pairs. By pairing each sample with its $K$ nearest neighbors, DeltaPhi generates $K \times N$ residual training pairs, achieving combinatorial implicit data augmentation without any additional data collection.

**Key Insight**: An ideal data-efficient approach should be a plug-and-play external framework compatible with arbitrary existing neural operator architectures, rather than being tied to a specific model design.

## Method

### Overall Architecture

DeltaPhi is an architecture-agnostic three-step inference framework that wraps any neural operator $\mathcal{N}_\theta$:

1. **Retrieve**: Given input condition $a_i$, retrieve the most similar auxiliary sample $(a_k, u_k)$ from the training set via cosine similarity.
2. **Residual Prediction**: Concatenate $a_i$ and $a_k$ as input to the neural operator to predict the physical state residual $\Delta u = \mathcal{N}_\theta(a_i, a_k)$.
3. **Reconstruction**: Obtain the final prediction $\hat{u}_i = \Delta u + u_k$, i.e., add the predicted residual to the ground-truth solution of the auxiliary sample.

### Key Designs

1. **Similarity Retrieval Strategy**: Cosine similarity is used to measure the similarity between input functions. During training, an auxiliary sample is randomly drawn from the top-$K$ ($K=20$) nearest neighbors to increase training diversity; during inference, the single most similar sample ($K=1$) is selected to minimize the residual and maximize accuracy.
2. **Differentiated $K$ Strategy for Training vs. Inference**: Randomly sampling neighbors during training introduces a regularization effect—the model must handle residuals of varying magnitudes, enhancing generalization. At inference time, the nearest neighbor is strictly selected, leveraging Lipschitz continuity to guarantee a minimal residual.
3. **Cross-Resolution Alignment**: In practice, training samples may reside on grids of different resolutions (e.g., unstructured meshes over irregular domains). DeltaPhi employs a Fourier-based up/downsampling strategy: functions at different resolutions are transformed to the frequency domain, where truncation or zero-padding is applied, before transforming back to the spatial domain, ensuring that $a_i$ and $a_k$ are operated on a unified grid.
4. **Input Concatenation**: $a_i$ and $a_k$ are concatenated along the channel dimension as the input to the neural operator, enabling the model to simultaneously observe the query condition and the reference condition and thereby predict the difference between their solutions.
5. **Mathematical Basis for Implicit Augmentation**: $N$ training samples yield $K \times N$ training pairs, where each residual pair $(a_i \oplus a_k,\, u_i - u_k)$ constitutes a valid supervision signal. This combinatorial augmentation requires no new data generation and is entirely based on re-pairing existing samples.

### Loss & Training

The training objective is a standard $L_2$ residual loss:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| \mathcal{N}_\theta(a_i, a_{k_i}) - (u_i - u_{k_i}) \|^2$$

where $(a_{k_i}, u_{k_i})$ is an auxiliary sample randomly drawn from the top-$K$ neighbors of $a_i$. The loss function is concise and introduces no additional hyperparameters or regularization terms.

## Key Experimental Results

### Main Results

**Irregular Domains (Table 1)**: Experiments on three complex geometries using NORM as the backbone:

| Task | NORM | NORM + DeltaPhi | Relative Gain |
|------|------|-----------------|---------------|
| Pipe Turbulence | 0.1183 | 0.0698 | **+40.99%** |
| Heat Transfer | 0.0200 | 0.0100 | **+50.00%** |
| Blood Flow | 0.0090 | 0.0080 | **+11.11%** |

**Regular Domains (Table 2, 100 training samples)**:

| Method | Darcy Flow | Navier-Stokes |
|--------|-----------|---------------|
| FNO | 0.0744 | 0.2055 |
| FNO + DeltaPhi | 0.0666 (**+10.54%**) | 0.1955 (**+4.86%**) |
| FFNO | 0.0737 | 0.1871 |
| FFNO + DeltaPhi | 0.0415 (**+43.76%**) | 0.1711 (**+8.54%**) |

Consistent gains are validated across **7 distinct architectures** (FNO, FFNO, CFNO, GNOT, Galerkin Transformer, MiOnet, ResNet).

### Ablation Study

| Ablation | Darcy (100) | NS (100) |
|----------|------------|----------|
| DeltaPhi (full) | **0.0415** | **0.1711** |
| No retrieval (random pairing) | 0.0582 | 0.1842 |
| Random $K$ at inference | 0.0480 | 0.1769 |
| Direct learning (baseline) | 0.0737 | 0.1871 |

Key conclusions: (1) similarity-based retrieval is critical—random pairing degrades performance substantially but still outperforms the baseline; (2) selecting the nearest neighbor at inference versus a random neighbor yields a clear difference, validating the importance of the Lipschitz continuity assumption.

### Key Findings

1. **Greater gains under more severe data scarcity**: Relative improvements are most pronounced in extremely data-scarce regimes (e.g., 50–100 samples) and diminish gradually—yet remain positive—as data volume increases.
2. **Architecture-agnostic gains**: Positive improvements are obtained across all 7 tested architectures, indicating that the benefit of residual learning stems from the paradigm shift rather than architectural coupling.
3. **Negligible computational overhead**: Similarity retrieval requires only 0.2–0.3 ms (based on pre-computed embedding vectors), making it negligible compared to the forward pass of neural operators.
4. **Cross-resolution effectiveness**: In heterogeneous-mesh experiments on irregular domains, the Fourier alignment strategy enables DeltaPhi to handle training samples of varying resolutions.

## Highlights & Insights

- **Elegant exploitation of physical intuition**: The Lipschitz continuity of PDE solutions is transformed from a mathematical property into an actionable learning paradigm—residual learning. This insight is both elegant and profound, revealing how the choice of *what to learn* (residuals vs. absolute values) fundamentally affects data efficiency.
- **Zero-cost data augmentation**: Without generating new data, introducing noise, or requiring equation information, DeltaPhi achieves an effective $N \to K \times N$ amplification of training data purely through re-pairing existing samples—a genuine "free lunch."
- **Practical plug-and-play usability**: As an external wrapper, DeltaPhi modifies neither the internal structure of any backbone nor its parameters, only changing how inputs/outputs are constructed and the training pipeline, resulting in minimal engineering integration cost.
- **Differentiated training/inference strategy**: Random neighbors at training time (regularization + diversity) and the nearest neighbor at inference time (minimal residual) reflect a deep understanding of the method's theoretical foundations.

## Limitations & Future Work

1. **Dependence on the similarity metric**: Cosine similarity is not necessarily optimal in high-dimensional function spaces, particularly when inputs exhibit multi-scale structure. Learned retrieval mechanisms or physics-informed distance metrics are worth exploring.
2. **Diminishing returns with large datasets**: When training data is abundant, direct learning already has sufficient expressive capacity, and the advantage of residual learning may weaken. Adaptive switching strategies merit future investigation.
3. **Upper bound on retrieval quality**: If no sufficiently similar sample exists in the training set for a given test input (out-of-distribution scenarios), the residual may no longer be small, violating the fundamental assumption of the method.
4. **Extension to time-dependent PDEs**: Current experiments focus primarily on steady-state or single-step prediction; the interaction between error accumulation and retrieval strategy in long-horizon autoregressive prediction warrants further study.
5. **Limitations of frequency-domain alignment**: Fourier alignment assumes periodicity or sufficient smoothness of the field; more robust alignment strategies may be required for problems with sharp discontinuities (e.g., shock waves).

## Related Work & Insights

- **Neural Operator family** (FNO, DeepONet, GNOT, etc.): DeltaPhi directly augments these methods and represents an orthogonal direction of improvement.
- **Retrieval-Augmented Generation (RAG)**: DeltaPhi's "retrieve + residual" paradigm is analogous to retrieval-augmented generation in NLP, but operates in continuous function spaces, inspiring broader consideration of RAG in scientific computing.
- **Residual learning (ResNet philosophy)**: From ResNet's insight that "learning the residual from the identity mapping is easier" to DeltaPhi's "learning the residual between similar physical states is more efficient" represents a natural extension of the residual learning philosophy to operator learning.
- **Few-shot Learning / Meta-Learning**: In the data-scarce setting, DeltaPhi's nearest-neighbor retrieval combined with residual prediction can be viewed as a non-parametric few-shot inference strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first systematic proposal of a residual learning framework in the neural operator domain; the combination of physical intuition and method design is highly natural.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 architectures × multiple PDEs × irregular domains, with comprehensive ablation and analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-motivated exposition.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, minimal overhead, open-source code, and extremely low barrier to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)

</div>

<!-- RELATED:END -->
