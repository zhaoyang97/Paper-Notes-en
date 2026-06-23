---
title: >-
  [Paper Note] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization
description: >-
  [ICML 2026][Optimization & Theory][Edge of Stability] This is an analytical paper: the authors point out that Gradient Descent (GD) simultaneously exhibits two **conflicting** implicit biases—small learning rates (LR) tend to suppress the parameter norm, while large learning rates (Edge of Stability) tend to suppress the loss sharpness. The learning rate interpolates betw
tags:
  - ICML 2026
  - Optimization & Theory
  - Edge of Stability
date: 2026-05-08
content_hash: 08b174fe3e92546b
---
# Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization

**Conference**: ICML2026  
**arXiv**: [2505.21423](https://arxiv.org/abs/2505.21423)  
**Code**: Research code accompanying the paper (no independent repository link provided)  
**Area**: Optimization Theory / Implicit Bias / Generalization  
**Keywords**: Gradient Descent, Implicit Bias, Edge of Stability, Sharpness Regularization, Parameter Norm

## TL;DR
This is an analytical paper: the authors point out that Gradient Descent (GD) simultaneously exhibits two **conflicting** implicit biases—small learning rates (LR) tend to suppress the parameter norm, while large learning rates (Edge of Stability) tend to suppress the loss sharpness. The learning rate interpolates between the two, and the authors observe a phase transition divided by a critical learning rate $\eta_c$. They further use a theoretical counter-example of a diagonal linear network to prove that "any single implicit bias is insufficient to explain generalization."

## Background & Motivation
**Background**: The good generalization of over-parameterized networks is often attributed to the "implicit bias" of Gradient Descent (GD)—the tendency of the optimization algorithm to favor a certain class of "well-structured" solutions. This line of research is split into two camps: one studies the bias under a small learning rate (or even Gradient Flow, GF), proving that GD favors **minimum norm solutions** (e.g., findings by Woodworth et al. on diagonal linear networks, or norm regularization of compositional structures in feedforward networks by Chou et al.); the other studies the Edge of Stability (EoS) phenomenon under a large learning rate, proving that GD can only converge to minima where the **sharpness is sufficiently low** (Ahn et al.: at a fixed $\eta$, it can only stop at $S_L(\theta^\star)<2/\eta$).

**Limitations of Prior Work**: These two camps almost exclusively analyze a single bias in isolation—either assuming the learning rate vanishes to 0 or focusing solely on EoS. However, real-world training uses **finite and non-zero** learning rates, where both biases operate simultaneously. Studying them separately fails to answer "how they interact, which one dominates, and when."

**Key Challenge**: The objectives of low norm and low sharpness are **not consistent and even conflict** at finite learning rates. Suppressing sharpness often comes at the cost of increasing the norm, and vice versa. The knob controlling this trade-off is precisely the learning rate. The success of prior "single bias explanations for generalization" might only be because these two biases happened to point to the same solution in those specific cases.

**Goal**: (1) Characterize the rise and fall of these two biases with respect to the learning rate; (2) Clarify where the generalization optimum falls on the learning rate spectrum; (3) Provide hard evidence that "a single bias is not optimal" using a simplified model that allows for exact calculations.

**Key Insight**: Treat the learning rate as a continuous knob. Fix the initialization, sweep through learning rates, and compare the sharpness and norm of the final GD solutions, rather than looking at the temporal evolution of Progressive Sharpening → EoS along a single trajectory as Cohen et al. did.

**Core Idea**: The learning rate interpolates between "low norm" and "low sharpness"; the best generalization often falls at an **intermediate learning rate where the two are balanced**, neither at the minimum norm end nor necessarily at the minimum sharpness end.

## Method
This paper does not propose a new algorithm but is an analytical work of "experimental observation + theoretical counter-example." Therefore, the "Method" refers to how it systematically reveals and demonstrates the conflict between the two implicit biases.

### Overall Architecture
The paper first provides a unified conceptual landscape: starting from the same initialization $\theta_0$, full-batch GD $\theta_{k+1}=\theta_k-\eta\nabla L(\theta_k)$ is run with a fixed learning rate $\eta$ until a loss threshold $\varepsilon$ is reached, recording the **sharpness** and **norm** of the final solution. Sharpness is defined as the operator norm of the loss Hessian $S_L(\theta):=\|\nabla^2 L(\theta)\|=\max_{\lambda\in\sigma(\nabla^2 L(\theta))}|\lambda|$ (i.e., the maximum absolute eigenvalue). Classical theory tells us that monotonic descent in GD is only guaranteed when $\eta<2/L$. Thus, once the sharpness exceeds $2/\eta$, the iteration enters EoS, and the sharpness is suppressed and oscillates around $2/\eta$.

Based on this setting, the paper's argument tightens through three layers: **Layer 1** uses large-scale experiments to show that "sweeping the learning rate reveals a sharp phase transition"; **Layer 2** names the stages before and after the transition as flow-aligned and EoS regimes, characterizing their respective norm/sharpness trends; **Layer 3** retreats to a diagonal linear network that can be solved analytically to prove that the norm minimizer and sharpness minimizer neither coincide nor represent the generalization optimum.

### Key Designs

**1. Two conflicting implicit biases induced by learning rate: Pitting "low norm" against "low sharpness" in the same coordinate system.**

Addressing the pain point of "isolated analysis by two camps," this paper brings both biases into the same experiment for a direct comparison. At small learning rates, GD stays close to Gradient Flow, where the compositional structure of feedforward networks biases it toward **small parameter norm** solutions. At large learning rates, the EoS mechanism prevents GD from staying in overly steep regions, forcing convergence to **low sharpness** solutions (since reachable minima under a fixed $\eta$ must satisfy $S_L(\theta^\star)<2/\eta$). The key observation is that as the learning rate crosses a critical value, the sharpness of the final solution **decreases hyperbolically** with $\eta$ (sticking to $\eta\mapsto 2/\eta$), while the $\ell_1$ norm **increases approximately linearly**. The fact that they move in opposite directions indicates they are a pair of conflicting biases requiring a trade-off, rather than reinforcing each other. This directly refutes the implicit assumption that "focusing on a single bias can explain generalization."

**2. Critical learning rate $\eta_c=2/s_{GF}$ and sharp phase transition: Providing a computable boundary for "when the dominant bias switched."**

The authors discovered a sharp phase transition between the two regimes determined by the data and the model. The critical learning rate is approximately $\eta_c:=2/s_{GF}^{\varepsilon}$, where $s_{GF}^{\varepsilon}:=\max_{t\le t_\varepsilon}S_L(\theta(t_\varepsilon))$ is the maximum sharpness of the Gradient Flow solution before reaching the loss threshold $\varepsilon$. In the **flow-aligned regime** ($\eta<\eta_c$), GD is nearly synchronized with Gradient Flow, and both final sharpness and norm remain **mostly constant** with the learning rate. In the **EoS regime** ($\eta>\eta_c$), sharpness begins to decay hyperbolically with $\eta$, and the norm increases. It must be emphasized: this phase transition appears in the sense of "fixed initialization, sweeping learning rates, and comparing final solutions," which is **different** from the Progressive Sharpening → EoS temporal phase transition observed by Cohen et al. along a single trajectory. The paper verifies the universality of this transition by varying dataset size, network width/depth (FCN/CNN/ResNet/ViT), activations (ReLU/tanh), loss (CE/MSE), loss threshold (equivalent to early stopping), initialization, and parameterization (μP/kernel).

**3. Theoretical counter-example of a diagonal linear network: Proving that no single bias is optimal.**

Experiments only show "correlation." To provide hard evidence of the "insufficiency of a single bias," the authors retreat to the simplest analytical setting: a shallow **diagonal linear network** with weight sharing, performing regression on a single data point with squared loss. On the zero-loss solution manifold $L=0$, they characterize the positions of the **norm minimizer** and the **sharpness minimizer** and compare their expected generalization errors. The conclusion: there exist scenarios where the **minimum expected generalization error is achieved neither at the norm minimizer nor at the sharpness minimizer**, but is continuously regulated by the learning rate. This constitutes a clean counter-example—since a single bias fails to achieve optimal generalization in such a simple model, using any single implicit bias to explain neural network generalization is untenable. The appendix further extends the analysis to classification settings and multi-data point cases.

**4. The U-shaped curve of generalization: Optimal generalization often occurs at a "balanced learning rate" rather than at either extreme.**

Plotting the test loss as a function of the learning rate, the authors found that in many settings (e.g., MNIST-5k + MSE), the test loss exhibits a **U-shape**: the best generalization appears at an **intermediate learning rate where the norm bias and sharpness bias are balanced**, and it **never** falls at the minimum norm end. This reinterprets the learning rate as a **regularization hyperparameter** that regulates the generalization capacity of the resulting model. The paper honestly notes that this is not an ironclad law—for instance, CIFAR-10-5k + MSE does not show a U-shape, suggesting the relative strength of the two biases also depends on the loss function and data, though the primary conclusion that "a single bias is insufficient" remains unaffected.

## Key Experimental Results

### Comparison of the two regimes

| Dimension | flow-aligned regime ($\eta<\eta_c$) | EoS regime ($\eta>\eta_c$) |
|------|--------------------------------------|------------------------------|
| Dominant Bias | Norm Regularization (aligned with GF) | Sharpness Regularization (EoS oscillation) |
| Sharpness vs. $\eta$ | Approximately constant | Hyperbolic decay, follows $2/\eta$ |
| $\ell_1$ norm vs. $\eta$ | Approximately constant | Approximately linear increase |
| Dynamics | Stable convergence, follows GF trajectory | Non-monotonic loss decay, curvature oscillates above $2/\eta$ |

### Experimental scope and generalization observations

| Variable | Values | Main Conclusion |
|------|------|----------|
| Architecture | FCN / CNN / ResNet / ViT | Phase transition and trade-off are universal |
| Loss | CE / MSE | Similar phase transition patterns, different temporal evolution |
| Setting | MNIST-5k + MSE | Test loss shows a clear U-shape, optimal at intermediate $\eta$ |
| Setting | CIFAR-10-5k + CE | Similar but weaker trend |
| Setting | CIFAR-10-5k + MSE | **Does not** show a U-shape (counter-example showing it's not universal) |

### Key Findings
- **Sharpness following $2/\eta$ is the fingerprint of EoS**: At the end of training in the EoS stage, the final sharpness is nearly equal to $2/\eta$, thus decaying hyperbolically with the learning rate—this is the bridge directly linking "final sharpness" to "learning rate."
- **Norm minimizer is never the generalization optimum**: In all settings where a U-shape was observed, the best generalization was not at the minimum norm (minimum learning rate) end. This is a direct rebuttal to the narrative of "norm as a complexity proxy."
- **Predictable critical point**: $\eta_c\approx 2/s_{GF}$ is determined by the maximum sharpness of the Gradient Flow solution; thus, changing the initialization moves $s_{GF}$ and shifts the critical learning rate accordingly, which was verified experimentally.
- **flow-aligned is not exactly GF**: The authors specifically point out that, unlike the conclusions of Arora et al., the absolute deviation of GD from Gradient Flow in the flow-aligned regime is not necessarily negligible, but their final sharpness and norm values are nearly identical—so "alignment" refers to the alignment of final statistics, not the overlap of the entire trajectory.
- **Width independence under μP**: In μP parameterization, weight-independent behavior of spectral properties like sharpness was observed, suggesting the trade-off remains stable in the infinite-width limit rather than being a finite-size artifact of small models.

### Intuition from a simplified model
For a diagonal linear network (shared weights) under single-point regression + squared loss, the zero-loss solutions form a low-dimensional manifold. Moving along this manifold, parameter norm and loss sharpness trade off: one end minimizes the norm, the other minimizes sharpness, while the "generalization optimum" corresponding to the true signal typically falls in between. The learning rate is the knob determining where GD finally stops on this manifold—small $\eta$ pushes the solution toward the norm minimizer, while large $\eta$ via EoS pushes it toward the sharpness minimizer. Thus, the fact that "neither bias is optimal" is demonstrated clearly in this hand-solvable model.

## Highlights & Insights
- **Combining two independent research threads on one phase diagram**: Previously, norm bias and sharpness bias belonged to two separate academic circles. This paper uses the clean experiment of "fixed initialization sweeping learning rates" to visualize their conflict as a sharp phase transition, creating a highly persuasive narrative.
- **Learning rate = Regularization knob**: Re-positioning the learning rate as a hyperparameter for regulating generalization capacity (rather than just a convergence speed parameter) provides intuition for tuning as "finding the balance between norm and sharpness."
- **Transferable methodology**: Using the simplest diagonal linear network as a "counter-example machine"—when you suspect a widely accepted single-factor explanation, retreating to an analytical toy model to construct a counter-example is a valuable paradigm for falsification.

## Limitations & Future Work
- **Theory only covers toy models**: Analytical conclusions are limited to shallow diagonal linear networks, single data points, and squared loss; conclusions for real deep networks are extrapolated from experiments and lack rigorous guarantees.
- **Choice of sharpness metric remains controversial**: The paper uses only worst-case sharpness (Hessian operator norm), while the relationship between generalization and sharpness itself remains debated in literature. Although the appendix verifies that the trade-off persists under other sharpness metrics, "which sharpness best predicts generalization" is still an open question.
- **U-shape is not universal**: The counter-example of CIFAR-10-5k + MSE shows that the relative strength of the two biases depends on loss and data; the paper does not provide a criterion for "when a U-shape will appear."
- **Future Directions**: Extending the counter-examples of diagonal linear networks to more realistic network classes or providing computable criteria for predicting the "optimal learning rate location" would move these observations from descriptive to prescriptive.

## Related Work & Insights
- **vs. Cohen et al. (2021, Original EoS observation)**: They characterized the temporal phase transition of sharpness climbing to $2/\eta$ at a fixed $\eta$; this paper takes a different perspective, sweeping learning rates to compare final solutions, revealing a different (regime-level) phase transition. The two are complementary.
- **vs. Ahn et al. (2022, Sharpness regularization) / Chou et al. (2023, Norm regularization)**: This paper does not negate either but points out that neither is sufficient to explain generalization in isolation and must be analyzed jointly as a pair of conflicting biases regulated by the learning rate.
- **vs. Andriushchenko et al. (2023a)**: They observed that generalization is related to hyperparameters like learning rate; this paper provides a mechanistic explanation for this correlation through the "norm-sharpness trade-off," and the conclusions of both are consistent and mutually reinforcing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Analyzing the two implicit biases as a conflicting trade-off and falsifying single-bias narratives with counter-examples is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically sweeping learning rates across multiple architectures/losses/initializations, though limited to small models due to sharpness estimation constraints.
- Writing Quality: ⭐⭐⭐⭐⭐ Conceptual landscapes, phase diagrams, and theoretical counter-examples progress logically; the narrative is clear and honest (actively providing counter-examples to the U-shape).
- Value: ⭐⭐⭐⭐ Re-interpreting the learning rate as a regularization knob has conceptual value for understanding generalization and hyperparameter tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[ICML 2026\] RACO: Reward-free Alignment for Conflicting Objectives](reward-free_alignment_for_conflicting_objectives.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[ICML 2026\] AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](adagc_enhancing_llm_pretraining_stability_via_adaptive_gradient_clipping.md)

</div>

<!-- RELATED:END -->
