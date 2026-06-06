---
title: >-
  [Paper Note] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos
description: >-
  [ICML 2026][LLM Pretraining][Mean-field theory] The authors treat dropout as an "external field" $h$ that breaks the perfect alignment fixed point $c^*=1$ in mean-field signal propagation theory. They derive the Landau e…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Mean-field theory"
  - "edge-of-chaos"
  - "dropout scheduling"
  - "universality classes"
  - "scaling laws"
date: 2026-05-08
content_hash: cd31293d305036e4
---

# Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

**Conference**: ICML 2026  
**arXiv**: [2605.21648](https://arxiv.org/abs/2605.21648)  
**Code**: Yes (dropout-universality-experiments, commit-pinned repository in paper)  
**Area**: Training Theory / Dropout / Mean Field / Signal Propagation / Scheduling  
**Keywords**: Mean-field theory, edge-of-chaos, dropout scheduling, universality classes, scaling laws

## TL;DR
The authors treat dropout as an "external field" $h$ that breaks the perfect alignment fixed point $c^*=1$ in mean-field signal propagation theory. They derive the Landau equation, two-parameter scaling collapse, and two distinct universality classes for smooth and kinked activations. From this, they obtain a "zero-cost" practical conclusion: a **front-loaded schedule** reduces test loss by 18–35% compared to constant dropout in MLPs and ViTs under the same budget.

## Background & Motivation

**Background**: Mean-field theory (MFT) of randomly initialized deep networks (Poole et al. 2016; Schoenholz et al. 2017) divides networks into ordered, chaotic, and critical phases. At the "edge-of-chaos," the correlation length $\xi_c$ diverges, allowing signals to propagate deepest. He initialization $\sigma_w^2=2$ is essentially the criticality condition for ReLU.

**Limitations of Prior Work**: Dropout is the default regularizer in industry, but its treatment in MFT only suggests it "destroys the $c^*=1$ fixed point" without providing usable scaling laws. In terms of scheduling, industry relies on heuristics—either constant dropout or strategies like stochastic depth and curriculum dropout—lacking a first-principles explanation for **why** a specific schedule might be superior.

**Key Challenge**: Dropout provides regularization (reducing overfitting) while simultaneously **cutting off** the correlation of signal propagation along depth. These two effects are adjusted independently at each layer, but current theory cannot dictate how to distribute dropout along depth for a given budget, nor whether different strategies are required for smooth vs. kinked activations.

**Goal**: (i) Incorporate dropout into MFT to provide a scaling-law level description; (ii) Distinguish universality classes for smooth/kinked activations; (iii) Translate the theory into executable scheduling rules.

**Key Insight**: View dropout as an "external field" $h$ in statistical mechanics and the de-alignment $m\equiv 1-c^*$ as the "order parameter." The problem is then transformed into a standard paradigm of Landau critical phenomena, where existing tools such as RG, scaling collapse, and universality classes can be applied.

**Core Idea**: Dropout adds a **constant offset** to the correlation map at $c=1$, such that $c^*<1$ remains a fixed point but with a finite correlation length. This offset is defined as the "external field" $h$. Maximizing $\xi_{\rm eff}$ under a budget $\sum_\ell h_\ell = L\bar{h}$ is a concave optimization where a **saturated step solution** is optimal; "regularization reach" further selects the "front-loaded" branch.

## Method

### Overall Architecture

Theoretical Side: Starting from MFT without dropout—forward variance/correlation follows the recursion $c^l = F(c^{l-1})$, where the criticality condition is defined by $\chi_\perp \equiv F'(1) = 1$. After introducing inverted dropout with keep-probability $\rho$ (independent masks applied to two inputs), the correlation map becomes $\bar{F}_\rho(c)$. The authors prove that $\bar{F}_\rho(1) = 1-h$ ($h>0$), meaning $c=1$ is no longer a fixed point. They define $m\equiv 1-c^*$ as the order parameter, $t\equiv \chi_\rho - 1$ as the reduced temperature, and $h$ as the external field, deriving the Landau equation to extract scaling laws. Experimental Side: Comparison of constant dropout with various schedules (front-loaded, back-loaded, linear, step) on MLP and ViT using CIFAR-10/100, while fixing the total budget $\bar{h}$.

### Key Designs

1.  **Dropout as an Alignment Symmetry-Breaking Field $h$**:

    - **Function**: Quantifies the "criticality-destroying" effect of dropout in MFT as a scalar field conjugate to the order parameter.
    - **Mechanism**: By evaluating the correlation recursion $\bar{F}_\rho$ with independent masks at $c=1$, they obtain $\bar{F}_\rho(1) = 1 - \frac{1-\rho}{\rho \bar{q}^*}\sigma_w^2 \int Dz\,\phi^2(\sqrt{\bar{q}^*}z)$, defining the external field as $h \equiv 1-\bar{F}_\rho(1)$. Under weak dropout, $h \approx a(1-\rho)$ is linearly related to the dropout probability. The order parameter is $m\equiv 1-c^*$. Taylor expanding $\bar{F}_\rho(1-m)$ at $m=0$ and substituting it into the fixed-point condition $1-m = \bar{F}_\rho(1-m)$ yields the Landau equation $h = \tfrac{g_\rho}{2}m^2 - tm$, with the physical solution $m(t,h) = \frac{t+\sqrt{t^2+2g_\rho h}}{g_\rho}$.
    - **Design Motivation**: Prior work only noted that "dropout destroys the critical point." This work proves that the recursion deformed by dropout **still has a fixed point $c^* < 1$**, making the correlation length well-defined and allowing for RG flow analysis. This is the prerequisite for all subsequent scaling laws.

2.  **Smooth vs. Kinked Universality Classes + Two-Parameter Scaling Collapse**:

    - **Function**: Explains why tanh and ReLU exhibit distinctly different critical behaviors in ResNet MFT (Yang & Schoenholz 2017) and provides **different critical exponents**.
    - **Mechanism**: The analytic structure of the correlation map in the neighborhood of $c=1$ determines the behavior. Smooth activations (tanh, GELU) satisfy Price's theorem and allow Taylor expansion, where the second-order term $g_\rho m^2$ gives $m\sim\sqrt{h}$ ($\delta=2$). Kinked activations (ReLU) have $\phi''$ containing delta functions, leading to a branch point at $c=1$, where the equation degenerates to $h = \kappa m^{3/2} - tm$, giving $m\sim h^{2/3}$ ($\delta=3/2$). A complete table of critical exponents ($\nu_t, \beta, \theta_{\rm rel}, \gamma, \delta, \nu_\rho, \alpha$) is provided. Two-parameter scaling collapse: for smooth classes, defining $\tilde{m}\equiv m\sqrt{g_\rho/(2h)}$ and $\tilde{t}\equiv -t/\sqrt{2g_\rho h}$ collapses all curves into $\tilde{m} = \sqrt{1+\tilde{t}^2}-\tilde{t}$. For kinked classes, $m = (h/\kappa)^{2/3}\mathcal{F}(t/(\kappa^{2/3}h^{1/3}))$ with a crossover scale of $|t|\sim \kappa^{2/3}h^{1/3}$. Hermite spectral expansion provides a secondary diagnosis: exponents for smooth activations decay exponentially, while those for kinked activations decay via power law.
    - **Design Motivation**: This framework places the engineering decision of "activation function selection" into the context of statistical mechanics universality classes—details within a class are irrelevant, but different scaling laws must be used across different classes. This is the first explicit distinction made along the dropout axis.

3.  **Front-loaded Dropout Schedule = Saturated Step + Regularization Reach**:

    - **Function**: Derives the practical "early-layer dropout" scheduling rule from **first principles**, applicable for any given budget with zero additional compute.
    - **Mechanism**: Let the keep probability vary by layer $\ell$, and define the effective inverse correlation length as $\xi_{\rm eff}^{-1} \approx \frac{1}{L}\sum_\ell \sqrt{t^2+2g_\rho h_\ell}$. At $t=0$ (at criticality), $\xi_{\rm eff}^{-1} \propto \frac{1}{L}\sum_\ell h_\ell^{1/2}$ subject to the budget constraint $\sum_\ell h_\ell = L\bar{h}$ and upper bound $h_\ell \leq h_{\max}$. As $h^{1/2}$ is concave (Jensen), any **step** solution concentrated in $\{0, h_{\max}\}$ is optimal. The gain relative to a constant schedule is $\xi_{\rm step}/\xi_{\rm const} = \sqrt{h_{\max}/\bar{h}}$. Since the MFT objective is invariant to layer permutation, a second principle is needed to break degeneracy: the authors define "downstream exposure" $\mathcal{D}_\ell \approx h_\ell \xi_c(1-e^{-(L-\ell)/\xi_c})$ (masks in early layers are "seen" by more downstream layers), which is a weight monotonically decreasing with $\ell$. Thus, the linear programming solution is to **fill early layers**—resulting in the front-loaded schedule. The same logic applies to $\int h^{1/3}$ for kinked classes.
    - **Design Motivation**: Transforms the engineering problem of "where to place dropout" into a two-step standard optimization problem involving concave budget allocation and degeneracy breaking with monotonic weights.

### Loss & Training

The experimental side does not alter the training objective, only the distribution of dropout across layers. Fixed $\bar{h}$ is used to compare constant, linear-decreasing, step (front-loaded), and step (back-loaded) schedules. On the theoretical side, $\bar{F}_\rho$, $\chi_\rho$, and $g_\rho$ are solved numerically using Gaussian measure integration of the MFT recursion for comparison.

## Key Experimental Results

### Main Results

| Experimental Setup | Schedule | Loss Drop | Δacc (pp) | Relative Gain |
|---------|------|---------|-----------|----------|
| MLP Overfitting (Fig.6) | Step (early) | +17.9% | +0.83 | +2.0% |
| MLP Budget Control (Fig.7) | Big step (1/3) | +22.6% | +1.08 | +2.6% |
| ReLU $\bar{h}=0.1$ sweep | Big step (1/3) | **+35.4%** | +2.04 | +5.0% |
| GELU $\bar{h}=0.1$ sweep | Big step (1/3) | +29.8% | +0.62 | +1.5% |
| ViT CIFAR-100 | Linear (decreasing) | +4.2% | +0.66 | +1.4% |
| ViT CIFAR-10 ablation | Both blocks, step (early) | +6.3% | +0.52 | +0.7% |

On ViT CIFAR-100, the linear-decreasing schedule reached 49.38% vs. 48.69% for constant dropout ($p<0.05$).

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Smooth (tanh/GELU) MFT Recursion | $m\sim\sqrt{h}$ ($\delta=2$), $\xi\sim h^{-1/2}$ | Consistent with Landau $m^2$ term |
| Kinked (ReLU) MFT Recursion | $m\sim h^{2/3}$ ($\delta=3/2$), $\xi\sim h^{-1/3}$ | Dominated by branch point $m^{3/2}$ term |
| Two-parameter scaling collapse (Fig.2) | All $(t,h)$ curves collapse to a single universal function | Closed-form $\tilde{m}=\sqrt{1+\tilde{t}^2}-\tilde{t}$ for smooth class |
| Width much larger than depth | Front-loading advantage stable | Valid within the MFT applicability domain |
| High dropout / Narrow networks | Advantage weakens | This occurs exactly where the theory fails |

### Key Findings

- **Ours** achieved the **maximum** gain on ReLU MLPs (+35.4% loss drop), confirming that the low-order non-linear response to $h$ in kinked classes allows for more aggressive budget redistribution.
- Gains in the smooth class (GELU) were also significant at +29.8%, indicating that the conclusions are universal across activation functions.
- On ViT, the advantage of the schedule narrowed to 4–6%, which is consistent with the theory: attention and skip connections modify global depth dynamics but preserve local Gaussian kernels. The "priority" determined by the theory (favoring early layers) remains valid, though the magnitude of the effect decreases.
- When dropout is pushed into regimes where the theory fails (high $\bar{h}$, narrow networks), the advantage disappears—providing **inverse** evidence supporting the theory.

## Highlights & Insights

- By treating dropout as an "external field" $h$ and de-alignment as the "order parameter" $m$ from statistical mechanics, the entire toolkit of Landau theory, critical exponents, and scaling collapse becomes immediately applicable. This is an elegant "problem alignment" that provides a template for treating any hyperparameter as a "field."
- The criterion for determining universality classes is based on the **analytic structure** of the activation (Taylor-expandable vs. branch point), rather than the common "scale-invariant" property. This finer criterion explains the split in depth behavior between the ReLU and tanh families.
- The two-step decomposition—"Concave budget optimization → saturated step → breaking degeneracy with regularization reach → front-loading"—is highly insightful. MFT objectives are often invariant to permutation; a secondary principle is required to select the specific implementation.

## Limitations & Future Work

- **Forward MFT Only**: Backward gradient covariance also possesses diagonal/off-diagonal asymmetries due to mask independence. Although the authors provide recursion (18), they do not fully develop a backward critical theory. Finite-width gradient susceptibilities, mask correlations during training, and representation changes after the catapult phase remain unmodeled.
- **Architectural Constraints**: The dropout-deformed MFT for CNNs/ResNets is only presented as an argument (App. A.4) without empirical experiments. While the conclusions hold for ViT, the large-width limit of attention mechanisms offers space for more detailed treatment.
- **Initialization Theory**: All conclusions are drawn at the **moment of initialization**, without characterizing how representation learning during training interacts with the schedule—a common limitation of such work.
- **Mask Correlation**: When masks are shared within a batch, the $c=1$ fixed point is restored, weakening the regularization effect and requiring new analysis.

Future directions: (i) Applying the same perspective to weight decay, warm-up, and adaptive dropout; (ii) Developing a finite-width gradient critical theory; (iii) Universality class analysis of attention head dropout; (iv) Scheduling in the training time dimension (integrating curriculum dropout).

## Related Work & Insights

- **vs. Schoenholz et al. (2017)**: They first noted that dropout destroys the $c=1$ fixed point but stopped at "criticality disappears." This paper proves that $c^*<1$ remains a fixed point, allowing RG and scaling law tools to remain applicable—a crucial "half-step forward."
- **vs. Hayou et al. (2019)**: They observed behavior differences between smooth and ReLU activations at the edge-of-chaos; this work provides **critical exponents** and a formal criterion for universality classes.
- **vs. Stochastic Depth (Huang et al. 2016) / Curriculum Dropout (Morerio et al. 2017) / LayerDrop (Fan et al. 2020)**: Those are temporal or whole-layer schedules. This work focuses on dropout intensity scheduling in the **spatial depth dimension**, which is orthogonal and stackable with those methods.
- **vs. scale-invariant criteria in Roberts et al. (2022)**: The smooth/kinked classification here is based on analytic structure rather than scale-invariance, providing different coverage and better suitability for determining dropout scaling laws.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fully migrates universality tools from statistical physics to dropout scheduling and provides distinct critical exponents for smooth/kinked classes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete $\bar{h}$-sweep and activation function ablations on MLP/ViT via CIFAR, though CNN/ResNet experiments are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-part derivation (Landau equation + scaling collapse + scheduling), with precise alignment between theory and experiments.
- Value: ⭐⭐⭐⭐ The "zero-cost" front-loaded schedule is immediately applicable; the theoretical framework is foundational for future research treating hyperparameters as "fields."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICML 2026\] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)

</div>

<!-- RELATED:END -->
