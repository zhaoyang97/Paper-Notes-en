---
title: >-
  [Paper Note] A Certified Unlearning Approach without Access to Source Data
description: >-
  [ICML 2025][AI Safety][Machine Unlearning] This paper proposes the first certified unlearning framework that does not require access to the original training data. By leveraging a surrogate dataset to approximate the statistical properties of the original data, and employing a noise scaling mechanism based on the statistical distance between the source and surrogate distributions, it achieves provable data deletion guarantees.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Machine Unlearning"
  - "Certified Unlearning"
  - "Surrogate Dataset"
  - "Differential Privacy"
  - "Noise Calibration"
date: 2026-05-08
content_hash: ffc0d623adfa7bd1
---

# A Certified Unlearning Approach without Access to Source Data

**Conference**: ICML 2025  
**arXiv**: [2506.06486](https://arxiv.org/abs/2506.06486)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Machine Unlearning, Certified Unlearning, Surrogate Dataset, Differential Privacy, Noise Calibration

## TL;DR

This paper proposes the first certified unlearning framework that does not require access to the original training data. By leveraging a surrogate dataset to approximate the statistical properties of the original data, and employing a noise scaling mechanism based on the statistical distance between the source and surrogate distributions, it achieves provable data deletion guarantees.

## Background & Motivation

With the enforcement of data privacy regulations such as GDPR, CCPA, and CPPA, the need to remove specific data points from trained models has become increasingly urgent. **Certified Unlearning** ensures that the unlearned model is statistically indistinguishable from a model retrained from scratch through rigorous probabilistic guarantees.

Existing certified unlearning methods face a core assumption issue: **they typically require access to the complete original training data**. However, in practical scenarios, this assumption often does not hold:

- **Privacy Constraints**: Models may be trained by a third party, and the original data cannot be shared.
- **Resource Limitations**: Historical data may have been deleted due to storage costs.
- **Regulatory Barriers**: Data retention policies may prohibit storing the original data.

Therefore, a key open question is: **When the unlearning mechanism has absolutely no access to the original training samples, how can one achieve certified unlearning relying solely on a surrogate dataset that approximates the original data distribution?**

Although some work has been done in zero-shot unlearning (e.g., zero-shot unlearning, JiT unlearning, adversarial sample generation, etc.), these methods all **lack formal certified unlearning guarantees**. This paper is the first to fill this theoretical gap.

## Method

### Overall Architecture

The method proposed in this paper is built on the certified unlearning framework of **second-order Newton updates**. The core innovation lies in replacing the original dataset $\mathcal{D}$ (sampled from distribution $\rho$) with a surrogate dataset $\mathcal{D}_s$ (sampled from distribution $\nu$), and achieving certified unlearning through three key steps:

1. **Hessian Estimation**: Approximate the Hessian of the original dataset using the Hessian of the surrogate dataset.
2. **Model Update**: Perform a one-step Newton update based on the estimated Hessian.
3. **Noise Calibration**: Inject Gaussian noise calibrated according to the statistical distance between the source and surrogate distributions.

**Symbol definitions**:
- $\mathcal{D}$: Original dataset ($n_1$ samples, distribution $\rho$)
- $\mathcal{D}_s$: Surrogate dataset ($n_2$ samples, distribution $\nu$)
- $\mathcal{D}_u$: Dataset to be forgotten ($m$ samples)
- $\mathcal{D}_r = \mathcal{D} \setminus \mathcal{D}_u$: Retained dataset
- $\boldsymbol{w}^*$: Model trained on $\mathcal{D}$
- $\boldsymbol{w}_r^*$: Model retrained on $\mathcal{D}_r$
- $\hat{\boldsymbol{w}}_r$: Model produced by the proposed unlearning mechanism

### Key Designs

#### Step 1: Hessian Estimation

In traditional methods, the Newton update requires the Hessian $\mathbf{H}_{\mathcal{D}_r}$ of the retained data $\mathcal{D}_r$. Since $\mathcal{D}_r$ is unavailable, this paper approximates it using the Hessian of the surrogate dataset $\mathbf{H}_{\mathcal{D}_s}$:

$$\hat{\mathbf{H}}_{\mathcal{D}_r} = \frac{n \cdot \mathbf{H}_{\mathcal{D}_s} - m \cdot \mathbf{H}_{\mathcal{D}_u}}{n - m}$$

This leverages the property that the Hessian of the complete data can be linearly combined from the Hessians of the retained set and the forget set, while the Hessian of the surrogate dataset approximates that of the complete dataset.

#### Step 2: Model Update

Utilizing the property that $\nabla \mathcal{L}(\mathcal{D}, \boldsymbol{w}^*) = 0$ when training converges, the gradient on the retained set can be expressed as:

$$\nabla \mathcal{L}(\mathcal{D}_r, \boldsymbol{w}^*) = \frac{-m \cdot \nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)}{n - m}$$

Substituting the estimated Hessian and gradient into the Newton update formula:

$$\hat{\boldsymbol{w}}_r = \boldsymbol{w}^* + \frac{m}{n - m} \hat{\mathbf{H}}_{\mathcal{D}_r}^{-1} \nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)$$

Note: This step only requires the forget set $\mathcal{D}_u$ and the surrogate dataset $\mathcal{D}_s$, requiring absolutely no original retained data.

#### Step 3: Noise Calibration and Certification Guarantees

To achieve $(\epsilon, \delta)$-certified unlearning, Gaussian noise is injected into the updated model:

$$\hat{\boldsymbol{w}}_r' = \hat{\boldsymbol{w}}_r + \boldsymbol{n}, \quad \boldsymbol{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

where the noise standard deviation is:

$$\sigma = \frac{\Delta}{\epsilon} \sqrt{2 \ln(1.25/\delta)}$$

**Core Theoretical Result (Theorem 4.2)**: Under the assumptions that the loss function is $L$-Lipschitz, $\alpha$-strongly convex, $\beta$-smooth, and $\gamma$-Hessian Lipschitz, the upper bound of the difference between the retrained model and the unlearned model is:

$$\|\boldsymbol{w}_r^* - \hat{\boldsymbol{w}}_r\|_2 \leq \Delta = \frac{2\gamma L m^2}{\alpha^3 n_1^2} + \frac{\|\nabla \mathcal{L}(\mathcal{D}_u, \boldsymbol{w}^*)\|_2 \cdot m(n_1 - n_2)\beta + 2mn_2\beta \cdot \text{TV}(\rho \| \nu)}{(n_1\alpha - m\beta)(n_2\alpha - m\beta)}$$

Key Observation: The upper bound $\Delta$ consists of two terms—the first term is the standard Newton approximation error, and the second term depends on the **total variation distance $\text{TV}(\rho \| \nu)$ between the source and surrogate distributions**. When the two distributions are identical ($\text{TV} = 0$), the upper bound degenerates to the result of existing methods.

#### Practical Estimation of Statistical Distance

In practice, $\text{TV}(\rho \| \nu)$ cannot be computed directly. This paper proposes a heuristic approach:

1. **TV $\rightarrow$ KL Transformation**: Utilize the Bretagnolle-Huber inequality $\text{TV}(\nu \| \rho) \leq \sqrt{1 - e^{-\text{KL}(\nu \| \rho)}}$.
2. **KL Divergence Decomposition** (Proposition 4.5): Decompose $\text{KL}(\nu \| \rho)$ into conditional distribution divergence and marginal distribution divergence.
3. **Conditional Distribution Divergence**: Approximate this using the output probabilities of the original model $\boldsymbol{w}^*$ and the model $\tilde{\boldsymbol{w}}^*$ trained on the surrogate dataset.
4. **Marginal Distribution Divergence**: Leverage the energy model (Energy-Based Model) implicit in the classifier, generate samples from $\rho(\boldsymbol{x})$ via SGLD sampling, and then estimate the KL divergence using the Donsker-Varadhan variational representation.

### Loss & Training

**Loss Function Assumptions (Assumption 4.1)**:
- $L$-Lipschitz: Bounded change of loss in the parameter space
- $\alpha$-strongly convex: Guarantees a unique global optimum
- $\beta$-smooth: Gradient is Lipschitz continuous
- $\gamma$-Hessian Lipschitz: Second-order derivative is Lipschitz continuous

**Experimental Hyperparameter Settings**: $\alpha = 1 + \lambda$, $L = 1$, $\beta = 1$, $\gamma = 1$, $L_2$ regularization constant $\lambda = 0.01$, privacy parameters $\epsilon = 5e3$, $\delta = 1$. The SGLD sampling step size is 0.02, generating 1000 samples, with 4000 iterations per sample. The KL estimation network is a three-layer linear network trained for 500 epochs with a learning rate of 0.0001 using the Adam optimizer.

## Key Experimental Results

### Main Results

**Synthetic Experiments**: Original dataset contains 15,000 samples from a 50-dimensional standard Gaussian distribution; the surrogate dataset controls the KL divergence by adjusting the off-diagonal elements of the covariance matrix $\zeta \in [0.01, 0.1]$.

| Method | $\zeta$ | Train Acc | Test Acc | Forget Acc | MIA | RT |
|------|---------|-----------|----------|------------|-----|-----|
| Retrain | – | 77.0% | 72.0% | 73.6% | 47.63% | 10 |
| Unlearn (+) | – | 77.1% | 72.6% | 74.1% | 47.63% | 10 |
| Unlearn (-) | 0.02 | 77.2% | 72.2% | 74.8% | 48.89% | 7 |
| Unlearn (-) | 0.06 | 77.3% | 72.2% | 74.3% | 48.37% | 10 |
| Unlearn (-) | 0.10 | 77.3% | 72.7% | 74.1% | 48.30% | 10 |

**Real-world Data Experiments (StanfordDogs, $\xi=100$)**:

| Method | Train Acc | Test Acc | Retain Acc | Forget Acc | MIA | RT |
|------|-----------|----------|------------|------------|-----|-----|
| Retrain | 82.9% | 76.0% | 84.2% | 71.1% | 52.5% | 19 |
| Unlearn (+) | 83.8% | 75.7% | 85.1% | 72.2% | 52.0% | 21 |
| Unlearn (-) | 83.7% | 75.6% | 85.0% | 72.0% | 51.9% | 16 |

### Ablation Study

**Different Forget Ratios (StanfordDogs, $\xi=100$)**:

| Forget Ratio | Method | Train Acc | Forget Acc | MIA | RT |
|---------|------|-----------|------------|-----|-----|
| 0.01 | Unlearn (-) | 87.1% | 74.1% | 53.1% | 10 |
| 0.1 | Unlearn (-) | 83.7% | 72.0% | 51.9% | 16 |
| 0.2 | Unlearn (-) | 85.0% | 72.6% | 52.0% | 40 |

**Different Model Architectures (CIFAR-10, $\xi=100$)**:

| Architecture | Method | Train Acc | Test Acc | MIA | RT |
|------|------|-----------|----------|-----|-----|
| Linear (L) | Unlearn (-) | 78.1% | 77.2% | 48.83% | 32 |
| Conv+Linear (C+L) | Unlearn (-) | 80.5% | 78.1% | 50.71% | 46 |
| 2Conv+Linear (2C+L) | Unlearn (-) | 82.9% | 80.5% | 50.05% | 22 |

### Key Findings

1. **Unlearn (-) performs comparably to Unlearn (+)**: Even without access to the original data, the proposed method performs on par with methods having data access across all metrics.
2. **MIA is close to 50%**: Membership Inference Attack accuracy is close to random guess, indicating complete unlearning.
3. **Necessity of Noise Scaling**: Forget Score experiments show that when using a surrogate dataset, the proposed noise scaling (rather than directly using the noise from methods with data access) must be used to achieve effective certified unlearning.
4. **Effective KL Approximation**: The heuristic KL estimates match the exact values closely, validating its practicality.
5. **Cross-Architecture Generalization**: The method remains effective across architectures, from linear models to multi-layer convolutional networks.

## Highlights & Insights

1. **First theoretical guarantee for source-free certified unlearning**: Fills an important blank in the source-free unlearning field regarding the lack of formal certified guarantees, greatly expanding the scoping and applicability of certified unlearning.
2. **Elegant theoretical framework**: The upper bound $\Delta$ clearly characterizes the impact of surrogate data quality (via $\text{TV}(\rho \| \nu)$) on the unlearning guarantee—the closer the distributions, the smaller the required noise, and the higher the model utility.
3. **Complete practical solution**: From theoretical guarantees to source-free estimation of KL divergence (energy-based models + SGLD + Donsker-Varadhan), it provides a complete path from theory to practice.
4. **MNIST-USPS cross-domain experiments**: Demonstrates the feasibility of using a truly cross-dataset (different sources) as a surrogate dataset, which possesses strong practical significance.

## Limitations & Future Work

1. **Strong convexity assumption**: The loss function requires strict assumptions such as $\alpha$-strong convexity, which limits its direct applicability to deep non-convex models (although mixed linear networks can partially alleviate this, it remains an approximation).
2. **Theoretical gap in KL estimation**: Although the heuristic KL estimation performs well in experiments, it may theoretically lead to weaker privacy guarantees than the exact values.
3. **SGLD sampling quality**: The quality of sampling from energy-based models directly affects the accuracy of KL estimation, which may degrade in high-dimensional complex scenarios.
4. **Applicability to large-scale models**: The experiments are mainly validated on linear models and shallow CNNs; the applicability to modern large models like Transformers remains to be verified.
5. **Limitations of single-step Newton updates**: For large-batch forget requests, the single-step approximation may not be precise enough.

## Related Work & Insights

- **Guo et al. (2019)**: Pioneering work in certified data removal, using influence functions and Newton updates.
- **Sekhari et al. (2021)**: Established the standard definition and theoretical framework of $(\epsilon, \delta)$-certified unlearning.
- **Zhang et al. (2024)**: Extended certified unlearning to deep neural networks.
- **Ahmed et al. (CVPR 2025)**: An empirical approach for source-free unlearning, but without certified guarantees.
- **Chundawat et al. (2023)**: Zero-shot unlearning, using noise maximization and gated knowledge distillation.
- **Grathwohl et al. (2019)**: The energy-based model perspective inspired the marginal distribution sampling method used in this paper.

**Inspirations for Future Work**: The concept of surrogate datasets can be generalized to federated unlearning, privacy compliance auditing, and other scenarios; the theoretical framework can attempt to relax assumptions to non-convex losses or more general model classes.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First to provide formal theoretical guarantees for certified unlearning without source data |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Complete theorem proof system, from upper bounds to certified guarantees |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Synthetic + real datasets, multiple ablations, but model scales are relatively small |
| Practicality | ⭐⭐⭐⭐ | The KL estimation scheme makes the method deployable, but the strong convexity assumption limits the scope of application |
| Writing Quality | ⭐⭐⭐⭐ | Clear logic, natural transition from theory to practice |
| Overall Recommendation | ⭐⭐⭐⭐☆ | Outstanding theoretical contribution, representing a significant advancement in the field of certified unlearning |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](../../NeurIPS2025/ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](../../CVPR2025/ai_safety/towards_source-free_machine_unlearning.md)
- [\[ICLR 2026\] Remaining-data-free Machine Unlearning by Suppressing Sample Contribution](../../ICLR2026/ai_safety/remaining-data-free_machine_unlearning_by_suppressing_sample_contribution.md)
- [\[ICML 2025\] Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss](enhancing_certified_robustness_via_block_reflector_orthogonal_layers_and_logit_a.md)
- [\[CVPR 2026\] Unlearning without Forgetting: Securely Removing Targeted Concepts from Large-Scale Vision-Language Open-Vocabulary Detectors](../../CVPR2026/ai_safety/unlearning_without_forgetting_securely_removing_targeted_concepts_from_large-sca.md)

</div>

<!-- RELATED:END -->
