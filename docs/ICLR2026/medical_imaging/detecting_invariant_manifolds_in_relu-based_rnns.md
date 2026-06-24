---
title: >-
  [Paper Note] Detecting Invariant Manifolds in ReLU-Based RNNs
description: >-
  [ICLR2026][Medical Imaging][Piecewise Linear RNN] This paper proposes a semi-analytical algorithm for ReLU-based Piecewise Linear RNNs (PLRNNs) that directly computes the stable and unstable manifolds of saddle points or saddle periodic points. This allows for mapping the boundaries of different attractor basins in state space, identifying homoclinic/heteroclinic intersections, and proving the existence of chaos within the RNN—filling a long-standing gap in how to analyze the…
tags:
  - "ICLR2026"
  - "Medical Imaging"
  - "Piecewise Linear RNN"
  - "Invariant Manifolds"
  - "Attractor Basins"
  - "Homoclinic Orbits"
  - "Chaos"
date: 2026-05-08
content_hash: 50fec281f1073c00
---

# Detecting Invariant Manifolds in ReLU-Based RNNs

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EAwLAwHvhk](https://openreview.net/forum?id=EAwLAwHvhk)  
**Code**: https://github.com/DurstewitzLab/DetectingManifolds  
**Area**: Time Series / Dynamical Systems Reconstruction / RNN Interpretability  
**Keywords**: Piecewise Linear RNN, Invariant Manifolds, Attractor Basins, Homoclinic Orbits, Chaos

## TL;DR
This paper proposes a semi-analytical algorithm for ReLU-based Piecewise Linear RNNs (PLRNNs) that directly computes the stable and unstable manifolds of saddle points or saddle periodic points. This allows for mapping the boundaries of different attractor basins in state space, identifying homoclinic/heteroclinic intersections, and proving the existence of chaos within the RNN—filling a long-standing gap in how to analyze the dynamical topology of discrete-time ReLU RNNs with discontinuous Jacobians.

## Background & Motivation
**Background**: RNNs have recently regained popularity due to more stable training algorithms (avoiding vanishing/exploding gradients) and new architectures (including State Space Models, SSMs, which are essentially linear RNNs with nonlinear readouts). They are widely used for time series prediction and **dynamical systems reconstruction (DSR)**—using a trained RNN as a "surrogate model" of an underlying physical or biological system to infer the dynamical mechanisms (e.g., in climate science or neuroscience) that generated the observed time series.

**Limitations of Prior Work**: While models can fit data well, the understanding of "what dynamics the trained RNN is actually performing" lags significantly. The behavior of a dynamical system is determined by the **topology and geometry of its state space**, where the **stable and unstable manifolds** of saddle points/saddle periodic points are particularly crucial. Stable manifolds partition the state space into different attractor basins (determining which attractor the system eventually settles into, i.e., multistability), while the **intersection** of stable and unstable manifolds generates homoclinic points, leading to fractal geometry and chaos. However, these invariant manifolds are almost never calculated in the machine learning community.

**Key Challenge**: Existing tools in dynamical systems theory (DST), such as numerical continuation methods, are almost entirely designed for **smooth, continuous-time** ODE systems. In contrast, RNNs are **discrete-time mappings**, and ReLU causes the Jacobian to be **discontinuous** at sub-region boundaries. Existing methods for discrete mappings, especially piecewise linear ones, suffer from the curse of dimensionality and can only handle systems with $\le 5$ dimensions, which is far smaller than the latent space dimensions of practical RNNs. Locating manifolds in ReLU RNNs has previously lacked a "usable method."

**Key Insight**: PLRNNs use ReLU to partition the state space into a large number of **linear sub-regions**. Within each sub-region, the dynamics are a simple linear mapping. Since the local dynamics are linear, local segments of the manifolds can be derived in **closed-form analytically** from the eigenvectors of the local Jacobian. The authors aim to implement the strategy of "local analytical derivation + recursive stitching across regions."

**Core Idea**: Construct manifold segments analytically within each linear sub-region using eigenvectors, and then recursively stitch these segments across adjacent regions by iterating the RNN mapping forward or backward—leveraging the piecewise linear structure of PLRNNs to achieve **precise localization** of manifolds.

## Method

### Overall Architecture
The PLRNN is formulated as $z_t = A z_{t-1} + W\Phi(z_{t-1}) + h$, where $A$ is diagonal, $W$ is off-diagonal, and $\Phi=\max(0,\cdot)$ is element-wise ReLU. A key reformulation absorbs the ReLU into a diagonal indicator matrix $D_{t-1}$:

$$z_t = F_\theta(z_{t-1}) = (A + W D_{t-1})\,z_{t-1} + h,\qquad D_t=\mathrm{diag}(d_t),\ d_{i,t}=\mathbb{1}[z_{i,t}>0].$$

This partitions the state space into up to $2^M$ linear sub-regions. **Within each sub-region, $F_\theta$ is an affine mapping** with a constant Jacobian $A+W D$. The algorithm is built entirely on this property: manifolds are affine subspaces within each sub-region (obtained locally analytically) and are stitched together via iterative mapping when crossing boundaries.

The pipeline is as follows: first, use the existing SCYFI algorithm (the lab's prior work, which locates fixed points/periodic points in polynomial time) to find a saddle point $p$ of interest. Then, starting from the sub-region containing $p$, initialize the manifold using local eigenvectors. Sample several support points and propagate them using $F_\theta$ (for unstable manifolds) or $F_\theta^{-1}$ (for stable manifolds) into adjacent sub-regions. In each new sub-region, redefine the manifold segments and repeat this recursively until the state space is covered. The final output is a set of eigenvectors and at least one support point for each visited sub-region, which together precisely define the manifold segment in that region.

### Key Designs

**1. Intra-region analytical construction + Inter-region recursive stitching: Transforming "manifold computation" into "local linear algebra + iteration"**

This is the backbone of the algorithm, addressing the core pain point that discrete, discontinuous mappings cannot use continuation methods. In the sub-region containing $p$, the stable (unstable) manifold is the affine subspace spanned by the eigenvectors of the local Jacobian $A+W D(p)$ whose eigenvalues are inside (outside) the unit circle; these can be computed in closed form. Complex eigenvalues appear in pairs and span planar subspaces, allowing spiral points to be handled similarly. To extend the manifold to adjacent regions, $N_s$ support points are randomly sampled on the current manifold segment and pushed across sub-region boundaries using $F_\theta$ (unstable) or $F_\theta^{-1}$ (stable). Manifold segments are then constructed in whichever new sub-regions these points land. While the manifold dimension remains constant across regions, it **bends** as it enters regions with different dynamics (e.g., entering a spiral region). Theoretically, $d+1$ support points are sufficient to anchor a $d$-dimensional manifold segment: PCA is used if the segment is linear, and kernel-PCA is used if it bends. Setting $N_s = k_{\text{neigh}}\times d$ (number of adjacent regions $\times$ manifold dimension) ensures a high probability of reaching all extension directions. Thus, the "global manifold" $= \bigcup_n F_\theta^{\pm n}(\text{local manifold})$ is assembled piece by piece.

**2. Self-consistent inverse mapping heuristic: Making $F_\theta^{-1}$ solvable under piecewise linearity**

Computing the stable manifold requires backward iteration, necessitating the solution for $z_{t-1}=(A+W D_{t-1})^{-1}(z_t-h)$. The difficulty lies in $D_{t-1}$ itself depending on the signs of $z_{t-1}$ components—the solved $z_{t-1}$ must be **self-consistent** with the assumed $D_{t-1}$, otherwise the inverse is invalid. The authors provide a simple and efficient heuristic: ① Perform a backward step using the current sub-region's $D_t$ to get a candidate $z^*_{t-1}$; ② Perform a forward step to verify if $z_t \stackrel{?}{=} F_\theta(z^*_{t-1})$; ③ If equal, the step is complete; ④ If not, retry using the $D(z^*_{t-1})$ of the candidate point; ⑤ If it still fails, check adjacent regions by flipping bits in $D_{t-1}$. This is efficient because candidate points **usually** fall into the correct sub-region, or at most cross into an adjacent one, keeping the search space very small.

**3. Enforcing $F_\theta$ invertibility during training via regularization: Providing a legitimate premise for "backward iteration"**

The backward construction relies on $F_\theta$ being invertible. Theoretically, this is reasonable—most underlying ODE flow maps we wish to approximate are invertible by the Picard-Lindelöf theorem—but a trained $F_\theta$ may not satisfy this. A sufficient condition for invertibility is that Jacobians of adjacent sub-regions have the **same sign** for their determinants. The authors incorporate this as a hinge-loss regularization term in the training loss:

$$L_{\text{reg}} = \lambda\cdot\frac{1}{|S_{\text{reg}}|}\sum_{i\in S_{\text{reg}}}\max\big(0,\,-\det(J_i)\big),\qquad J_i=A+W D_i.$$

The brilliance is that **sampling only 1% of the sub-regions** visited by the actual trajectories is sufficient to ensure near-everywhere invertibility in the state space, with almost no impact on runtime or reconstruction accuracy. Furthermore, since many scientific dynamical systems are inherently time-reversible, this regularization not only does not hinder performance but actually **accelerates training convergence** on some systems (e.g., reaching PE(20) $\le 0.5$ in significantly fewer steps for a 10D damped nonlinear oscillator, $p=0.023$).

**4. Dimension-independent reconstruction quality metric $\Delta_\sigma$: Enabling verification for high-dimensional manifolds**

In 2D, manifolds are lines that can be visually verified, but high-dimensional manifolds cannot be visualized and require a quantifiable metric. For points $x_0$ sampled from a neighborhood $U$ or the reconstructed manifold, the authors define:

$$\delta_\sigma(x_0):=\min_{k_\sigma\ge 0}\frac{\lVert F_\theta^{k}(x_0)-p\rVert_2^2}{\lVert x_0-p\rVert_2^2}\in[0,1],$$

The intuition is: points truly on the stable/unstable manifold will be pulled toward $p$ by (backward/forward) iteration, resulting in $\delta_\sigma\approx 0$; points not on the manifold will have $\delta_\sigma$ spread over a wider range. Based on this, they define an indicator $I_U=\mathbb{1}[\delta_\sigma(x_0\in U)>\delta_\sigma^{\max}]$ (where $\delta_\sigma^{\max}$ is the maximum $\delta_\sigma$ of points sampled on the manifold) and report $\Delta_\sigma:=\langle I_U\rangle-\tilde\delta_\sigma(x_0\in W^\sigma(p)\cap U)$ as a quality statistic. Values closer to 1 are better (experimental results range from 0.76 to 0.97). ⚠️ Refer to the original paper for precise notation.

### Loss & Training
The RNN itself is trained using shPLRNN (shallow PLRNN) or ALRNN (almost-linear RNN) from prior work. The main loss is standard MSE with sparse teacher forcing. The manifold algorithm serves as a **post-training analysis tool**; the only intervention during training is the invertibility regularization $L_{\text{reg}}$ (with $\lambda$ on the order of $0.1\,e^{M}$).

## Key Experimental Results

### Main Results
The paper uses the "accurate restoration of state space structure" as the primary evaluation axis, verifying performance across synthetic and real-world datasets.

| System / Model | Configuration | Verification Content | Manifold Quality $\Delta_\sigma$ |
|--------|------|------|----------|
| 2D PL Toy Map | Analytically known inverse/fixed points | Forward/backward iterated points all fall on the algorithmically derived manifolds; basin boundaries match perfectly with the "grid initial value $t\to\infty$" method | — |
| Duffing Oscillator | shPLRNN, $M{=}2,H{=}10$ | Stable manifold of the central saddle is the basin boundary for the two spiral attractors, matching the true system flow | ≈0.97 |
| 2-Choice Decision Task | ALRNN, $M{=}15,P{=}6$ | Stable manifold of the saddle separating the two decision basins (contains both flat and curved segments). Distances from attractors to boundary are ≈0.34 vs 0.32, consistent with balanced weights | ≈0.95 |
| Lorenz-63 | shPLRNN, $M{=}3,H{=}20$ | Detected stable/unstable manifolds **match closely** with those obtained by numerical continuation of the original Lorenz ODE, proving the RNN learned the geometry, not just the attractor | ≈0.78 |
| Cortical Neuron Membrane Potential | ALRNN, $M{=}25,P{=}6$ | Trained model contains a 38-period orbit (firing rhythm) + stable fixed point. A 24D stable manifold separates "firing" and "resting" basins; firing state distance to boundary is 2.3 > resting state 1.2, implying firing is more robust to perturbation | ≈0.76 |

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| $|S_{\text{reg}}|/|S|$ (Invertibility Reg. Sampling) | Constraining only **1%** of sub-regions ensures $\det(J)>0$ nearly everywhere | Robust as latent dimension $M$ increases; runtime/accuracy negligibly affected |
| Reg. ON vs. OFF (Damped Oscillator) | Convergence to PE(20) $\le 0.5$ is significantly faster ($p=0.023$) | Significant benefit for time-reversible systems |
| Reg. ON vs. OFF (Lorenz-63) | Performance remains mostly unchanged ($p=0.122$) | Harmless for chaotic systems |
| Homoclinic Point Detection (2D PL Map) | Detected homoclinic intersections match analytical derivations; chaos proven via Smale-Birkhoff theorem | Bifurcation diagrams and Lyapunov exponents confirm "robust chaos" |

### Key Findings
- Manifold quality $\Delta_\sigma$ decreases as system complexity increases (Toy/Duffing ≈ 0.95+ → Lorenz ≈ 0.78 → Real neurons ≈ 0.76), but even 24D curved manifolds can be quantitatively verified.
- The algorithm does more than "draw pictures": by calculating distances from states to manifolds (basin boundaries), it can quantitatively answer mechanistic questions like "which state is more robust" or "are two choices equally weighted."
- Detection of homoclinic/heteroclinic intersections changes the "is the system chaotic" question from empirical observation to a decidable geometric fact, confirming chaos even in high-dimensional physiological signals like human ECG.

## Highlights & Insights
- **Turning "piecewise linearity" from a liability into an advantage**: While ReLU makes Jacobians discontinuous and breaks traditional continuation methods, the authors exploit the "strict linearity within each sub-region" for local analytical construction. This is a classic example of "finding the key to analysis through the model's structural weaknesses."
- **Training-analysis loop**: Invertibility is not just assumed; it is **actively shaped** by a lightweight regularization that constrains only 1% of sub-regions. This validates the backward manifold construction at nearly zero cost—a "regularization for analyzability" strategy transferable to other PL system analyses requiring inverse maps.
- **Dimension-independent validation metric $\Delta_\sigma$**: Using "whether iteration pulls points toward the saddle" as a criterion for manifold membership elegantly bypasses the challenge of high-dimensional visualization and can be reused by any method claiming to reconstruct manifolds.
- **Reading mechanisms from surrogate models**: In real cortical neurons, the algorithm quantitatively concludes that "the firing state is more robust to perturbation than the resting state," demonstrating the potential of DSR + manifold analysis as a tool for scientific discovery.

## Limitations & Future Work
- **Breakdown in chaotic regions**: When invariant manifolds fold into fractal structures, the analytical construction using support vectors to span curved/flat segments collapses and cannot capture self-similar geometry. In such cases, the method can only confirm chaos by finding homoclinic/heteroclinic intersections.
- **Worst-case exponential complexity**: Technically, the algorithm could grow at $2^P$ relative to the number of linear sub-regions; however, the authors note that in trained PLRNNs, the number of utilized sub-regions saturates quickly, keeping it mostly polynomial within data-explored regions.
- **Dependency on saddle point detection**: The ability to recover all relevant manifolds depends on SCYFI finding all saddle points first—missing a saddle point means missing its corresponding manifolds.
- **Future Work**: The authors suggest generalizing the method to other piecewise linear systems (SLDS, TLN/Threshold-Linear Networks, Lur'e systems). However, SLDS have boundary discontinuities and TLNs are continuous-time, requiring additional handling (e.g., discrete $\leftrightarrow$ continuous PLRNN conversion).

## Related Work & Insights
- **vs. SCYFI (Eisenmann et al. 2023)**: SCYFI can **locate fixed points/periodic points** in polynomial time; this paper goes a step further to construct the **invariant manifolds** of those points. It is a relay from "finding skeleton points" to "growing manifold surfaces."
- **vs. Numerical Continuation Methods**: Continuation methods are for smooth continuous-time ODEs and are limited by the curse of dimensionality ($\le 5$D). This work focuses on discrete, Jacobian-discontinuous ReLU RNNs, leverages piecewise linearity for precise localization, and scales to high dimensions.
- **vs. Recent PL Mapping Methods (Simpson 2023)**: Also targets piecewise linear maps but still suffers from the curse of dimensionality. This paper circumvents the explosion in practice by using "local analysis + only visiting data-explored sub-regions."
- **vs. Fixed Point Analysis (Golub & Sussillo 2018, etc.)**: Previous RNN interpretability work focused almost exclusively on fixed/periodic points, ignoring the invariant manifolds that actually partition the state space and determine multistability and chaos. This paper fills that gap and claims to be the **first** algorithm to detect invariant manifolds in ReLU RNNs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First detection algorithm for invariant manifolds in ReLU RNNs with a clear approach and well-defined gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified from toy maps to Lorenz and real neurons/ECG, cross-checked with analytical solutions/numerical continuation, though high-D manifolds remain hard to visualize.
- Writing Quality: ⭐⭐⭐⭐⭐ Definitions, algorithms, and motivations are clearly structured with sufficient DST background.
- Value: ⭐⭐⭐⭐⭐ Provides a quantifiable tool for mechanistic analysis, moving toward "understanding trained RNNs as scientific surrogate models."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BioTamperNet: Affinity-Guided State-Space Model Detecting Tampered Biomedical Images](biotampernet_affinity-guided_state-space_model_detecting_tampered_biomedical_ima.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](../../NeurIPS2025/medical_imaging/geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[ECCV 2024\] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](../../ECCV2024/medical_imaging/a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)
- [\[ICLR 2026\] Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis](dual-kernel_adapter_expanding_spatial_horizons_for_data-constrained_medical_imag.md)
- [\[ICLR 2026\] Accelerating Benchmarking of Functional Connectivity Modeling via Structure-aware Core-set Selection](accelerating_benchmarking_of_functional_connectivity_modeling_via_structure-awar.md)

</div>

<!-- RELATED:END -->
