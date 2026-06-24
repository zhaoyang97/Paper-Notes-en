---
title: >-
  [Paper Note] Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product
description: >-
  [ICML2025 Oral Spotlight][Optimization][grokking] This paper demonstrates that the grokking (delayed generalization) phenomenon is not unique to neural networks or gradient descent, but rather stems from **the progressive learning of task-relevant features**. By reproducing modular arithmetic grokking on kernel machines using the non-neural Recursive Feature Machines (RFM), it reveals that block-circulant feature matrices are central to generalization.
tags:
  - "ICML2025 Oral Spotlight"
  - "Optimization"
  - "grokking"
  - "modular arithmetic"
  - "feature learning"
  - "AGOP"
  - "circulant matrices"
  - "kernel methods"
  - "Recursive Feature Machines"
date: 2026-05-08
content_hash: 5b21c8573b244bcd
---

# Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product

**Conference**: ICML2025 Oral Spotlight  
**arXiv**: [2407.20199](https://arxiv.org/abs/2407.20199)  
**Code**: [nmallinar/rfm-grokking](https://github.com/nmallinar/rfm-grokking)  
**Area**: Optimization / Grokking / Feature Learning  
**Keywords**: grokking, modular arithmetic, feature learning, AGOP, circulant matrices, kernel methods, Recursive Feature Machines

## TL;DR

This paper demonstrates that the grokking (delayed generalization) phenomenon is not unique to neural networks or gradient descent, but rather stems from **the progressive learning of task-relevant features**. By reproducing modular arithmetic grokking on kernel machines using the non-neural Recursive Feature Machines (RFM), it reveals that block-circulant feature matrices are central to generalization.

## Background & Motivation

Grokking refers to the "emergent" phenomenon where test accuracy suddenly jumps from near-zero to perfect long after standard training accuracy has reached 100%. It is regarded as a quintessential case of "ability emergence" in deep learning. Prior studies generally attribute it to:

- Neural network architectures
- Implicit regularization of gradient-based optimizers such as SGD
- Gradual decrease in training loss

However, are these explanations sufficient? The authors raise three critical questions:

1. Can grokking only occur in neural networks?
2. Is it strictly dependent on gradient optimization?
3. Can traditional metrics (training/test loss) predict the timing of emergence?

This paper answers "no" to all three questions by reproducing grokking in a completely different learning framework: **kernel machines + RFM**.

## Method

### Core Framework: Recursive Feature Machines (RFM)

RFM is an iterative algorithm that empowers kernel machines, which otherwise lack feature learning capabilities, with feature learning via **Average Gradient Outer Product (AGOP)**. Each iteration consists of two steps:

**Step 1 — Train Predictor**: Solve kernel ridge regression

$$f^{(t)}(x) = k(x, X; M_t) \alpha, \quad \alpha = k(X, X; M_t)^{-1} y$$

**Step 2 — AGOP Update Feature Matrix**:

$$M_{t+1} = [G(f^{(t)})]^{s}, \quad s = \frac{1}{2}$$

Where AGOP is defined as the average outer product of the Jacobian matrix on the training data:

$$G(f) = \frac{1}{n} \sum_{j=1}^{n} \frac{\partial f(x^{(j)})}{\partial x} \frac{\partial f(x^{(j)})}{\partial x}^\top \in \mathbb{R}^{d \times d}$$

### Data Encoding

Inputs $(a, b) \in \mathbb{Z}_p \times \mathbb{Z}_p$ are concatenated into $\mathbf{e}_a \oplus \mathbf{e}_b \in \mathbb{R}^{2p}$ using one-hot encoding, and the label is $\mathbf{e}_{f^*(a,b)} \in \mathbb{R}^p$.

### Key Findings: Block-Circulant Feature Matrices

The $2p \times 2p$ feature matrix $M^*$ learned by RFM ultimately exhibits a block-circulant structure:

$$M^* = \begin{pmatrix} A & C^\top \\ C & A \end{pmatrix}$$

where $C \in \mathbb{R}^{p \times p}$ is a circulant matrix (each row is a circular shift of the preceding row) and $A = c_1 I + c_2 \mathbf{1}\mathbf{1}^\top$.

For multiplication/division tasks, the circulant structure is observed only after reordering the rows and columns via **discrete logarithms** (since discrete logarithms convert multiplication into addition).

### Two Measures of Hidden Progress

1. **Circulant Deviation**: Measures how far a feature submatrix is from being circulant.

$$\mathcal{D}(A) = \frac{1}{\|A\|_F^2} \sum_{j=0}^{p-1} \mathrm{Var}(\mathcal{S}(A)[j])$$

Diagonal elements of a circulant matrix are constant, so $\mathcal{D} = 0$ represents a perfect circulant structure.

2. **AGOP Alignment**: The cosine similarity between the features in the current iteration and the final features.

$$\rho(M_t, M^*) = \frac{\langle \tilde{M}_t, \tilde{M}^* \rangle}{\|\tilde{M}_t\| \|\tilde{M}^*\|}$$

### Theoretical Connection with Fourier Multiplication Algorithm (FMA)

The authors prove that a kernel machine equipped with circulant features essentially implements FMA:

1. Apply Discrete Fourier Transform to each digit vector: $\hat{x}_{[1]} = Fx_{[1]}, \hat{x}_{[2]} = Fx_{[2]}$
2. Element-wise multiplication: $\hat{x}_{[1]} \odot \hat{x}_{[2]}$
3. Project back to the label space: $y_{\mathrm{add}}(x; \ell) = \sqrt{p} \cdot \langle F\mathbf{e}_a \odot F\mathbf{e}_b, F\mathbf{e}_\ell \rangle_{\mathbb{C}}$

This is consistent with the generalization scheme that neural networks are believed to learn in prior work, suggesting that RFM and neural networks discover the **same algorithm** to solve modular arithmetic.

## Key Experimental Results

Experiments use modulus $p = 61$, a training fraction of 50%, and quadratic and Gaussian kernels.

| Experimental Setup | Train Loss | Iterations to Emergence | Final Test Accuracy |
|---------|---------|---------|------------|
| RFM (Quadratic Kernel) Addition | Always 0 | ~12 iterations | 100% |
| RFM (Quadratic Kernel) Subtraction | Always 0 | ~12 iterations | 100% |
| RFM (Quadratic Kernel) Multiplication | Always 0 | ~15 iterations | 100% |
| RFM (Quadratic Kernel) Division | Always 0 | ~15 iterations | 100% |
| Neural Network (Quadratic Activation) Add/Sub | Gradual decrease | ~25 epochs | 100% |
| Neural Network (Quadratic Activation) Mul/Div | Gradual decrease | ~35 epochs | 100% |

**Key Comparisons**:

- **Random Circulant Features vs. RFM-Learned Features**: When inputs are transformed with a random circulant matrix, the standard kernel machine generalizes directly, even outperforming the features learned by RFM over multiple iterations.
- **Random Circulant Features Accelerate Neural Networks**: At a training fraction of 17.5%, the transformed network reaches 100% test accuracy in a few hundred epochs, whereas the untransformed network fails to converge even after 3000 epochs.
- **Pearson Correlation between NFM and AGOP**: 0.955 for addition, 0.942 for subtraction, 0.924 for multiplication, and 0.929 for division, demonstrating high consistency.

**Hidden Progress Metrics vs. Standard Metrics**: During the first 8–10 iterations where the test loss and test accuracy remain completely unchanged, both circulant deviation and AGOP alignment already exhibit an almost linear, steady improvement.

## Highlights & Insights

1. **Decoupling Contributions**: Completely decouples feature learning from predictor training, proving that emergence stems purely from feature learning capabilities, independent of network architecture and optimizers.
2. **Unified Perspective**: The feature structures learned by RFM (kernel machine) and neural networks are highly consistent (both block-circulant), suggesting the existence of "correct features" inherent to the task.
3. **Hidden Progress Metrics**: Proposes two metrics—circulant deviation and AGOP alignment—which can detect the model's steady progress towards generalization even when standard metrics (loss/accuracy) remain completely stagnant.
4. **Regularization Explanation**: Through the Neural Feature Ansatz (NFA), the role of weight decay is interpreted as regularization on the trace of AGOP; indeed, AGOP regularization can substitute for weight decay to achieve grokking.
5. **Practical Implications**: Random circulant features alone enable kernel/neural networks to generalize rapidly, demonstrating that generalization does not require precisely learned features, but merely the correct **structural type**.

## Limitations & Future Work

1. **Task Constraints**: Only modular arithmetic (addition, subtraction, multiplication, division) was validated. Potential generalization to more complex algebraic structures or "skill emergence" in natural language remains unexplored.
2. **AGOP Alignment is a Posteriori**: Requires knowing the feature matrix of the final converged model, making it unsuitable as a real-time forecasting tool during training.
3. **Strong Theoretical Assumptions**: The FMA theorem requires training on the complete dataset and using specific kernel functions; theoretical guarantees under partial training data scenarios are currently lacking.
4. **Preliminary Multi-Skill Analysis**: Although different grokking rates across two tasks are shown in the appendix, the interaction and competition between skills have not been analyzed in depth.
5. **Computational Overhead**: RFM requires calculating and inverting $d \times d$ AGOP matrices at each step, posing scalability challenges for large-scale problems.

## Related Work & Insights

- **Power et al. (2022)**: First discovered and named the grokking phenomenon.
- **Nanda et al. (2023)**: Discovered via mechanistic interpretability that neural networks solve modular addition using the Fourier multiplication algorithm.
- **Radhakrishnan et al. (2024)**: Proposed RFM and the Neural Feature Ansatz, which serve as the foundation for this paper's core framework.
- **Schaeffer et al. (2024)**: Questioned whether emergence might be an "illusion" caused by the choice of metric; this paper provides a deeper counterargument—even continuous metrics (test loss) exhibit a sharp transition.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to reproduce grokking in non-neural models, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four operations, two model classes, and multiple metrics; thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, exquisite illustrations, and well-integrated theory and experiments.
- Value: ⭐⭐⭐⭐ — Provides crucial decoupled evidence for understanding emergence/grokking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](../../NeurIPS2025/optimization/natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[ICML 2025\] Grokking at the Edge of Linear Separability](grokking_at_the_edge_of_linear_separability.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](../../ICLR2026/optimization/egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](../../ICLR2026/optimization/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)

</div>

<!-- RELATED:END -->
