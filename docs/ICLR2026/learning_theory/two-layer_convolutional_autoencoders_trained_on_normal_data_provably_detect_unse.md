---
title: >-
  [Paper Note] Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies
description: >-
  [ICLR 2026][learning_theory][Paper Note] Utilizing feature learning tools, this paper provides the first provable theoretical explanation for why a two-layer convolutional autoencoder can detect unseen anomalies when trained only on normal data: during training, convolutional kernels are absorbed by a "cone set" of normal features and align with these feature
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 3052b03827bbf74b
---
# Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FnbGlnKbIU](https://openreview.net/forum?id=FnbGlnKbIU)  
**Code**: None  
**Area**: Learning Theory / Anomaly Detection / Feature Learning  
**Keywords**: Reconstruction-Based Anomaly Detection, Convolutional Autoencoder, Feature Learning, Cone Set, Training Dynamics

## TL;DR
Utilizing feature learning tools, this paper provides the first provable theoretical explanation for why a two-layer convolutional autoencoder can detect unseen anomalies when trained only on normal data: during training, convolutional kernels are absorbed by a "cone set" of normal features and align with these feature directions; consequently, they show almost no response to signals where patches are replaced by anomalies, leading to significantly higher reconstruction errors for anomalies compared to normal data.

## Background & Motivation

**Background**: A major class of mainstream approaches in anomaly detection is Reconstruction-Based Anomaly Detection (RBAD). It involves training a generative or reconstruction model (such as autoencoders or diffusion models) using only normal data. During inference, the reconstruction error is treated as the anomaly score. Empirically, it has been repeatedly observed that models trained on normal data produce larger errors when reconstructing anomalous samples; thus, anomalies can be identified by setting a threshold for the reconstruction error.

**Limitations of Prior Work**: While RBAD is empirically effective in scenarios like industrial defect detection and fraud identification, there is almost no theoretical support explaining why this occurs. How can a model that has never seen an anomaly inherently yield a larger reconstruction error for one? This core phenomenon has remained at the level of empirical observation, lacking a provable mechanistic explanation.

**Key Challenge**: The training objective is solely to minimize the reconstruction error on normal data, providing no supervisory signal regarding anomalies. Yet, at test time, the model "fails to reconstruct" anomalies well. To explain this, two things must be clarified: what internal representations the model learns from normal data during the training phase, and why these representations "discriminate" against anomalous patches during the reconstruction phase.

**Goal**: The authors decompose the problem into three research questions: (Q1) How does the model learn features from normal training data in RBAD? (Q2) What role do these learned features play during reconstruction? (Q3) Why are anomalies more difficult to reconstruct than normal data?

**Key Insight**: Borrowing from the feature learning framework of Allen-Zhu & Li, the study models an image as an array of $P$ patches, where each patch follows a signal-noise model of "one feature + noise." By treating convolutional kernels as patch vectors of the same dimension, "kernels learning features" is equivalent to "kernel directions gradually aligning with feature directions." From this geometric perspective, training dynamics can be precisely characterized.

**Core Idea**: The paper proposes the geometric object of a "cone set"—a hyper-cone around each feature direction that sharpens as training progresses. Convolutional kernels falling into the cone are "absorbed" and become increasingly aligned with that feature. Normal features are sufficiently learned by the kernels, while directions in anomalous patches do not fall within any cone. After max pooling, these signals are filtered out, rendering them unreconstructible.

## Method

This paper does not propose a new algorithm but performs an end-to-end theoretical analysis of a specific two-layer convolutional autoencoder. The logic chain is: define data and network $\rightarrow$ analyze kernel growth during training (Q1) $\rightarrow$ analyze how the converged network reconstructs normal vs. anomalous patches (Q2, Q3) $\rightarrow$ validate with synthetic and real data.

### Overall Architecture

**Data Model**: An image $z=(x_1,\dots,x_P)^T$ consists of $P$ patches, where each patch $x_i = v_k + \rho$ is "a specific feature vector + small noise" ($|\langle\rho,v_k\rangle|\le\varepsilon$). All features $V=\{v_1,\dots,v_d\}$ constitute an orthonormal basis of a Hilbert space $H$, divided into two types: **normal features** $V_{nor}=\{v_1,v_2\}$ (always present, $\beta_1=\beta_2=1$) and **auxiliary features** $V_{aux}=\{v_3,\dots,v_d\}$ (present with probability $\beta_k<1$, representing acceptable error or sample diversity). An **anomaly** is defined as "replacing one patch $x_{i_0}$ of normal data with an anomalous patch $x_a$": if $x_a$ contains a feature $v_k$, it is a semantic anomaly; otherwise, it is a non-semantic (sensory) anomaly.

**Network Structure**: A two-layer convolutional autoencoder $\phi=\phi_d\circ\phi_e$ with $C=c_2 d$ convolutional kernels $w=(w_1,\dots,w_C)$ (over-parameterized by $c_2>1$). The encoder applies convolution plus smooth ReLU to each patch: $\phi_e(z;w_j)=(\sigma(\langle w_j,x_1\rangle),\dots,\sigma(\langle w_j,x_P\rangle))$, where $\sigma$ is a smooth ReLU differentiable near 0. The decoder reconstructs using max pooling:

$$\phi_d^{(i)}(\phi_e(z)) = \sum_{j\in[C]} \sigma(\langle w_j, x_i\rangle)\,\delta(i,j)\, w_j,$$

where $\delta(i,j)=1$ if and only if patch $i$ is the patch with the largest convolution value for kernel $w_j$ ($i=\arg\max_{i'}\sigma(\langle w_j,x_{i'}\rangle)$). Intuitively, each patch $x_i$ is reconstructed by a weighted summation of kernels that "pay the most attention" to it.

**Training Paradigm**: Standard empirical risk minimization with reconstruction loss $\ell(z;w)=\|\phi(z;w)-z\|^2$. Gradient descent is applied to kernels with random Gaussian initialization $w_j^{t+1}=w_j^t-\eta_t\nabla_{w_j}R_N(w^t)$, where the learning rate $\eta_t$ decays to 0. The analysis establishes probability bounds for the growth rate and magnitude of these iterative kernels.

### Key Designs

**1. Signal-Noise P-patch Data Modeling + Cone Set Definition: Mapping "kernels learning features" to analyzable geometric objects**

The difficulty in analyzing anomaly detection lies in the abstraction of "learned features." This paper unifies images, patches, and kernels into patch vectors within the same Hilbert space. With features as orthogonal bases, "kernel $w_j$ learning feature $v_k$" is precisely translated into the quantifiable event of $w_j$ aligning with $v_k$. Building on this, the cone set $\tilde S_t(k)$ is defined: at iteration $t$, kernel $w_j$ belongs to the cone set of feature $v_k$ if and only if its component along $v_k$ is sufficiently large and its components along other directions are sufficiently small:

$$\langle w_j^t, v_k\rangle \ge t\sigma_0 c_1 f_h, \qquad \langle w_j^t, v_{k'}\rangle \le t\sigma_0 f_r \;\;(\forall k'\neq k),$$

where the radius parameter satisfies $f_r = f_h\cdot o(t)$. This implies the cone (height $t\sigma_0 c_1 f_h$, radius $t\sigma_0 f_r$) becomes **increasingly sharp** as training progresses. Kernels within the cone are constrained tighter to the direction of $v_k$. The cone set is the core intuition, linking training dynamics, feature extraction, and anomaly reconstruction into a provable chain.

**2. Training Phase: Cone sets "absorb" convolutional kernels, enabling feature extraction (Q1)**

To explain why the model learns normal features, it must be proved that kernels align with features. Based on gradient expressions (Proposition 3.1) and Lemma 3.1, a key fact is derived: the growth rate of a kernel is positively correlated with its magnitude $\|w_j\|$, and $\|w_j^t\|\le t\sigma_0 f_h$. Kernels with larger initial values grow faster and reach larger final magnitudes, thus dominating reconstruction. Combined with Lemma 3.2 (kernels in a cone always satisfy $\delta(i,j)=1$ for patches containing that feature), two main conclusions are proven:

- **Theorem 1 (Irreversible Absorption)**: If $|\tilde S_0(k)|\ge1$, then $\tilde S_t(k)\subseteq\tilde S_{t+1}(k)$. Once a kernel enters a feature's cone set, it remains there forever. Since $f_r=f_h\cdot o(t)$, the term $\langle w_j^t,v_{k'}\rangle$ becomes negligible relative to $\langle w_j^t,v_k\rangle$, and the direction continually approaches $v_k$.
- **Theorem 2 (Non-empty Cone Sets)**: The size of $\tilde S_0(k)$ is bounded between $Cp_1-\lambda$ and $Cp_2+\lambda$ with high probability, meaning each feature's cone set is likely non-empty under random initialization, ensuring kernels can be absorbed.

Together, these show that training is a process of "each feature's cone absorbing a batch of kernels, sharpening them, and aligning them." The network thus extracts all features (including auxiliary features). Corollary 3.1 further notes that the growth rate gap for kernels in normal feature cones is larger than for auxiliary features, meaning normal features are learned more thoroughly.

**3. Reconstruction Phase: Anomaly patch signals are filtered by max pooling (Q2, Q3)**

After convergence, the kernel directions $w_j^*$ are essentially aligned with the features. When an anomaly $z(i_0,x_a)$ is fed into the network, the encoding of non-replaced patches remains identical to normal data; the replaced patch $x_a$ is encoded after being expanded via the orthogonal basis $x_a=\sum_k\langle x_a,v_k\rangle v_k$. The core result is Theorem 3: for non-semantic anomalies ($x_a$ sampled uniformly on the unit sphere), if a kernel is highly aligned with a feature ($\langle w_j,v_k\rangle>(1-\theta)\|w_j\|$), then

$$\langle w_j, x_a\rangle < \|w_j\|\cdot\max\Big\{2\theta,\ \tfrac{1}{\theta\sqrt{d-1}}\Big\}$$

holds with a probability of at least $1-2\theta\exp(-1/(2\theta^2))$. That is, the convolution of a kernel aligned to a normal feature with a random anomalous patch is **very small with high probability**. This weak signal is overshadowed and filtered out during the max pooling stage by stronger (normal) responses at the same patch position. Consequently, the anomalous patch has no kernels to facilitate its reconstruction, leading to a higher reconstruction error—providing a provable source for the empirical RBAD phenomenon.

As a significant supplement, Remark 3.4 explains why RBAD is **ineffective at semantic anomalies**: the replacement patch of a semantic anomaly contains a learned feature $v_{k_0}$ ($\beta_{k_0}\neq0$), making its $\langle w_j,x_a\rangle$ significantly larger than that of non-semantic anomalies. Its reconstruction error is closer to normal data, making it harder to detect. This is consistent with existing survey conclusions and demonstrates the model's bidirectional consistency.

### Loss & Training

Training minimizes the empirical reconstruction error on normal data $R_N(w)=\frac1N\sum_n\|\phi(z_n;w)-z_n\|^2$ using gradient descent with a decaying learning rate and random Gaussian initialization $w_j^0\sim\mathcal N(0,\sigma_0 I)$. Crucially, the process **requires no anomalous samples**, which is the core advantage of RBAD. Over-parameterization ($c_2>1$) and max pooling are introduced to prevent the "ideal orthogonal reconstruction" shown in Equation (3) from falling into local minima too early.

## Key Experimental Results

The experiments specifically aim to "validate the theory" rather than achieve SOTA performance (the authors explicitly state the goal is to complement rather than compete with existing RBAD methods).

### Synthetic Experiments (Identical to Theoretical Setup)

| Setting | Phenomenon | Theoretical Consistency |
|:---:|:---:|:---:|
| Normal vs. Non-semantic Anomaly | Clear gap in reconstruction error | Confirms Theorem 3: Anomaly signals filtered, hard to reconstruct |
| Normal vs. Semantic Anomaly | Significantly smaller gap | Confirms Remark 3.4: Semantic anomalies contain learned features |
| Noise $\varepsilon\in\{0.1,0.01,0.001\}$, $P\in\{20,\dots,60\}$, $d=\text{int}(c_3 P)$, $c_3\in\{1.2,1.5,2\}$ | Gap consistently exists across settings | Robustness to parameters |

Data scale: 4000 training and 1000 testing samples each for normal, semantic anomaly, and non-semantic anomaly. Loss: MSE, Optimizer: SGD, Over-parameterization: $C=\text{int}(1.2d)$.

### Real-world Data Visualization (MNIST / CIFAR-10)

| Observation | Corresponding Theory |
|:---:|:---:|
| Some kernels show clear outlines by epoch 10 and stop changing by epoch 50 | Kernel directions stabilize after absorption (Theorem 1) |
| Some kernels remain noise-like even after convergence | Corresponds to kernels not absorbed by any cone ($j\notin\cup_k\tilde S(k)$) |
| Shapes/colors of kernels do not change drastically during training | The cone "sharpens" through alignment rather than drastic shifts |

### Key Findings
- **Reconstruction error gap is the provable cause of RBAD**: The gap for non-semantic anomalies stems from the chain of "high probability small convolution between anomaly and aligned kernels + max pooling filtration," rather than any training signals targeted at anomalies.
- **Auxiliary features are also learned**: Cone set absorption applies to all features with $\beta_k\neq0$. Thus, the network learns even auxiliary features, explaining why semantic anomalies (containing learned features) are difficult to detect.
- **"Noise kernels" are a natural theoretical deduction**: Kernels that never develop outlines in visualizations correspond precisely to those that do not fall into any cone set, aligning theory with observed phenomena.

## Highlights & Insights
- **Cone set is an elegant geometric abstraction**: It unifies training dynamics, feature extraction, and anomaly reconstruction into a single visual of "sharpening cones absorbing kernels," with each step supported by probability bounds.
- **Max pooling as a mechanism for "signal filtering" is clever**: Anomalies are not without response; rather, their response is highly likely to be smaller than normal responses at the same location, pinning "large reconstruction error" to a specific operator.
- **Bidirectional Consistency**: The same theory explains both why non-semantic anomalies are detectable and why semantic anomalies are not. The latter aligns with existing empirical survey findings, enhancing the credibility of the model.
- **Transferable Logic**: Treating "kernel alignment with features" as an analysis object can be extended to analyze other generative anomaly detectors trained only on normal data (such as diffusion models).

## Limitations & Future Work
- **Extremely Simplified Model**: Uses only two layers, orthogonal features, a signal-noise patch model, and fixed $P$ and $d$. There remains a gap between this and deep RBAD systems (multi-layer, non-orthogonal, diffusion-based); the theory provides "mechanistic intuition" rather than a tight characterization of actual systems.
- **Predefined Feature Hierarchy**: The division of normal vs. auxiliary features ($V_{nor}=\{v_1,v_2\}$) and values of $\beta_k$ are hypothetical presets. How features naturally stratify in reality is not addressed.
- **Ineffective at Detecting Semantic Anomalies**: The theory itself suggests this mechanism is powerless against semantic anomalies, which is an inherent limitation of RBAD rather than a solvable issue of this paper.
- **Experiment Scope**: Experiments focus on validating theory without performance comparisons against SOTA RBAD methods on standard benchmarks, making it difficult to judge the boundary of the explanation's power in practical systems.

## Related Work & Insights
- **vs. Empirical RBAD (Lv et al. 2024 / Li et al. 2024)**: These works propose or improve RBAD methods and report larger reconstruction errors for anomalies without explaining why. This paper complements them by providing a provable training dynamics explanation.
- **vs. Feature Learning Theory (Allen-Zhu & Li 2023)**: This work adopts their P-patch signal-noise model and "kernel alignment" paradigm but shifts the context from classification to autoencoder reconstruction and adds "cone set absorption + max pooling" to answer specific anomaly detection questions.
- **vs. Anomaly Detection Surveys (Ruff et al. 2021)**: Surveys empirically note that RBAD is unsuitable for semantic anomalies; Remark 3.4 of this paper provides a formalized explanation for this empirical conclusion using the cone set theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Provides the first provable training dynamics explanation for the core RBAD phenomenon; cone set abstraction is novel.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments strictly align with theory; real-world visualizations are appropriate, but lacks performance benchmarks (as stated by the authors).
- Writing Quality: ⭐⭐⭐⭐ Driven by three research questions with a clear roadmap; balances geometric intuition with formal theorems.
- Value: ⭐⭐⭐⭐ Establishes a theoretical foundation for RBAD and self-consistently explains its capability boundaries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training](on_the_convergence_of_two-layer_kolmogorov-arnold_networks_with_first-layer_trai.md)
- [\[ICLR 2026\] SVD Provably Denoises Nearest Neighbor Data](svd_provably_denoises_nearest_neighbor_data.md)
- [\[ICLR 2026\] Covariate-Guided Clusterwise Linear Regression for Generalization to Unseen Data](covariate-guided_clusterwise_linear_regression_for_generalization_to_unseen_data.md)
- [\[ICML 2026\] Two-Layer Linear Auto-Regressive Models Estimate Latent States](../../ICML2026/learning_theory/two-layer_linear_auto-regressive_models_estimate_latent_states.md)
- [\[ICLR 2026\] T-Tamer: Provably Taming Trade-offs in ML Serving](t-tamer_provably_taming_trade-offs_in_ml_serving.md)

</div>

<!-- RELATED:END -->
