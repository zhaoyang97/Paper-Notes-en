---
title: >-
  [Paper Note] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning
description: >-
  [ICML 2026][AI Safety][Differential Privacy] This paper proposes DP-KFC: based on the observation that "the scaling of the Fisher matrix is determined by architecture…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "KFAC"
  - "Fisher Information Matrix"
  - "Preconditioner"
  - "Synthetic Noise"
date: 2026-05-08
content_hash: 88a6aeab77039b23
---

# DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2605.13418](https://arxiv.org/abs/2605.13418)  
**Code**: https://github.com/molinamarcvdb/DP-KFC (available)  
**Area**: Differential Privacy / Medical Imaging / Second-Order Optimization  
**Keywords**: Differential Privacy, KFAC, Fisher Information Matrix, Preconditioner, Synthetic Noise

## TL;DR
This paper proposes DP-KFC: based on the observation that "the scaling of the Fisher matrix is determined by architecture, and its correlation structure can be approximated by modality-level spectral statistics," it uses structured synthetic noise (pink noise $1/f^\alpha$ for images, Zipf sampling for text) to probe the network and reconstruct the KFAC preconditioner, without consuming privacy budget or introducing distribution shift. Under strong privacy ($\varepsilon\le 3$), it consistently outperforms DP-SGD and public data preconditioning methods.

## Background & Motivation

**Background**: The standard approach for differentially private deep learning is DP-SGD—per-sample gradients are $L_2$-clipped and injected with isotropic Gaussian noise. The scale of privacy noise grows with model dimension $\sqrt{d}$, causing the advantages of overparameterization in non-private settings to disappear under DP. To address this, the community has turned to adaptive/second-order methods (DP-Adam, KFAC + DP), but estimating second-order statistics from private data consumes privacy budget, while using public data introduces distribution shift.

**Limitations of Prior Work**: (1) The "isotropic" noise in DP-SGD is geometrically mismatched with the highly "anisotropic" loss landscape of neural networks—low-sensitivity parameters are drowned in noise, high-sensitivity parameters are over-clipped (Fig. 1 SNR collapse); (2) Ganesh et al. (2025) show that unbiased second-order estimation under privacy is often counterproductive, as noise in the preconditioner can be detrimental; (3) The precondition-then-privatize paradigm relies on public proxy data, which is unavailable in specialized domains like medical imaging.

**Key Challenge**: To match the privacy noise in DP-SGD to the loss geometry, gradients must first be transformed into an isotropic coordinate system; but obtaining the required curvature information must **not** consume privacy budget and **not** rely on public data.

**Goal**: (1) Prove that key information of the KFAC preconditioner can be recovered from the architecture itself; (2) Design an algorithm to reconstruct the preconditioner using only synthetic noise; (3) Strictly maintain the formal guarantees of DP (no additional privacy budget consumption).

**Key Insight**: Based on Mean Field Theory, the variance of layer activations $q^l$ and backpropagated gradients $\tilde q^l$ in deep networks satisfy deterministic recursions (determined by initialization and nonlinearity), independent of specific inputs; Karakida et al. (2019) show $\text{Tr}(F_l)\propto d\cdot q^{l-1}\cdot \tilde q^l$, i.e., the trace of the Fisher block is entirely determined by the architecture.

**Core Idea**: Decouple the Fisher matrix into "architecture sensitivity (recoverable via synthetic noise) + input-related structure (approximated by modality-level spectrum $1/f^\alpha$)", use synthetic probes to construct the KFAC factor inverse square root $F^{-1/2}$, and perform a linear transformation before per-sample gradients are clipped and noised (scale-then-privatize). This step is transparent to private data and thus does not consume privacy budget.

## Method

### Overall Architecture
The complete DP-KFC workflow alternates between two phases:
1. **Preconditioner Construction** (every $T_{freq}$ steps): Use a synthetic batch (pink noise / Zipf sequences) for forward and backward passes, estimate each layer's $\hat A_{l-1}=\mathbb{E}[\tilde a_{l-1}\tilde a_{l-1}^\top]$ and $\hat G_l=\mathbb{E}[\tilde\delta_l\tilde\delta_l^\top]$ via KFAC, and obtain $U_{A,l}, U_{G,l}$ by eigendecomposition.
2. **Private Training Step**: For each private sample $i$ and layer $l$, first transform the gradient $\tilde g_l^{(i)}=U_{G,l}\cdot g_l^{(i)}\cdot U_{A,l}$, then clip + add noise + average → SGD update.

The crucial "scale-then-privatize" order means $P_t$ is independent of the current batch (depends only on architecture and previously released model parameters), so the RDP guarantee fully inherits that of standard DP-SGD.

### Key Designs

1. **Data-Independent KFAC Factor Estimation**:

    - **Function**: Recover each layer's KFAC covariance $\hat A_{l-1}, \hat G_l$ from synthetic probes, without accessing private/public data.
    - **Mechanism**: Algorithm 1—generate $M$ synthetic $(\tilde x, \tilde y)$, perform forward and backward passes to obtain activations and errors, aggregate outer products $\hat A_{l-1}=\frac{1}{M}\sum \tilde a_{l-1}\tilde a_{l-1}^\top+\pi I$, $\hat G_l=\frac{1}{M}\sum \tilde\delta_l\tilde\delta_l^\top+\pi I$, then eigendecompose to get $U_{X,l}=Q_X(\Lambda_X+\gamma I)^{-1/2}Q_X^\top$. The preconditioner $F_l^{-1/2}=U_{A,l}\otimes U_{G,l}$ is represented implicitly via Kronecker product, without materializing the full FIM.
    - **Design Motivation**: MFT indicates that $\text{Tr}(F_l)$ depends on architecture, not data, so synthetic probes only need to preserve the architecture's propagation chain; damping $\pi I$ and $\gamma I$ ensure invertibility and control the condition number (Theorem 5.4: $\lambda_{min}\ge\sqrt\gamma$).

2. **Modality-Specific Synthetic Probes**:

    - **Function**: Ensure synthetic inputs carry architectural information and mimic the low-dimensional manifold structure of data.
    - **Mechanism**: **Image domain** uses Pink Noise—weight white noise $Z$ in the frequency domain as $\tilde Z_\mathbf{u}=Z_\mathbf{u}/(\|\mathbf{u}\|_2^{\alpha/2}+\epsilon)$, then IFFT, with $\alpha\approx 1$ to simulate the $1/f^\alpha$ spectrum of natural images (Field 1987); **NLP domain** uses Zipfian distribution to sample tokens from the vocabulary, placing [CLS] [SEP] [PAD] according to sentence syntax, so attention and LayerNorm follow real paths.
    - **Design Motivation**: White noise distributes energy evenly across frequencies, but deep networks mainly transmit low-frequency features; modality-level priors push synthetic probes close to real data activation statistics, yet carry no semantics → no privacy leakage.

3. **Scale-then-Privatize and DP Integration**:

    - **Function**: Inject curvature information into DP-SGD without increasing RDP consumption.
    - **Mechanism**: Algorithm 2—gradient transformation $\tilde g_l=U_{G,l}\,g_l\,U_{A,l}$ is performed before clipping, global $L_2$ norm becomes $\nu_i=\sqrt{\sum_l\|\tilde g_l^{(i)}\|_F^2}$, clip threshold $C$ unchanged; noise $\mathcal{N}(0,\sigma^2 C^2 I)$ is added after clipping and summing. Proposition 5.6 proves: since $P_t$ is a batch-independent linear operator, the composite mechanism's RDP guarantee matches that of the standard Gaussian mechanism.
    - **Design Motivation**: Compared to "Precondition-after-Privatize" (add noise then multiply by $P_t$), scale-then-privatize ensures the privacy noise term $d\sigma^2 C^2/B^2$ is **not** amplified by $\lambda_{max}^2$ (Theorem 5.4), with the greatest advantage in high-privacy regimes.

### Loss & Training

- Standard CE/MSE loss + DP-SGD optimizer; KFAC damping $\pi=\gamma=10^{-2}$; Opacus for RDP accounting, A100 GPU.
- Preconditioner refresh frequency $T_{freq}$ (typically 100–1000 steps), per-step wall-clock about 2.2× slower than DP-SGD, but each step is more efficient (Remark 5.5 "privacy wall"—privacy budget limits steps $T$, so higher per-step efficiency is worthwhile).
- Models: CNN on MNIST, CrossViT on CIFAR-100, BERT on StackOverflow, Logistic Regression on IMDB; privacy budget $\varepsilon\in[0.5,10]$.
- Convergence guarantee: Theorem 5.4 $\min_t\mathbb{E}\|\nabla\mathcal{L}\|^2\le \frac{C_1}{\lambda_{min}\sqrt T}+\frac{C_2}{\lambda_{min}\sqrt T}(\lambda_{max}^2\sigma_{sgd}^2+\frac{d\sigma^2 C^2}{B^2})$, $O(T^{-1/2})$ non-convex optimal rate.

## Key Experimental Results

### Main Results

MNIST CNN (5 seeds, hyperparameters tuned per method):

| Method | $\varepsilon=1$ | $\varepsilon=2$ | $\varepsilon=8$ |
|--------|----------------|----------------|----------------|
| DP-SGD | 91.7 ± 0.2 | 92.5 ± 0.3 | 93.7 ± 0.3 |
| AdaDPS (Public) | 91.3 ± 0.8 | 93.2 ± 1.0 | 93.3 ± 1.4 |
| DiSK (post-priv) | 93.7 ± 0.4 | 94.1 ± 0.3 | 94.3 ± 0.2 |
| DP-AdamBC (post-priv) | 94.0 ± 0.3 | 94.8 ± 0.2 | 95.3 ± 0.1 |
| Public DP-KFC | 95.3 ± 0.4 | 95.7 ± 0.3 | 96.4 ± 0.3 |
| **Synthetic DP-KFC** | **94.2 ± 0.5** | **95.0 ± 0.4** | **95.9 ± 0.3** |
| Synthetic DP-KFC + DP-AdamBC | **95.5 ± 0.3** | **96.1 ± 0.2** | 96.4 ± 0.3 |

Cross-modality results ($\varepsilon=1$):
- CIFAR-100 + CrossViT: Synthetic DP-KFC is nearly identical to Public DP-KFC, both outperform DP-Adam by ≈1.4%;
- StackOverflow + BERT: Synthetic 91.8% vs. DP-SGD 89.5% (+2.3%), but still ≈4% below Public DP-KFC 96.1%;
- IMDB + LR: Both Synthetic and Public DP-KFC reach 85.8% vs. DP-SGD 83.1%.

### Ablation Study

Transfer / Domain Mismatch (MNIST training, $\varepsilon=1.0$):

| Method | Fashion←MNIST (Ideal) | Path←MNIST (Texture Disjoint) |
|--------|----------------|----------------|
| Oracle (Private) | 88.3 ± 0.2 | 78.4 ± 1.7 |
| DP-SGD | 83.5 ± 0.7 | 68.5 ± 2.3 |
| AdaDPS (Public) | 84.7 ± 0.3 | 70.5 ± 2.0 |
| Public DP-KFC | 87.6 ± 0.2 | 73.4 ± 1.3 |
| **Synthetic DP-KFC** | **87.8 ± 0.2** | **78.2 ± 1.9** |

### Key Findings
- **Architecture Dominates Curvature**: Fig. 2 shows that KFAC eigenvalue decay for MLP/CNN/Attention layers is nearly identical across synthetic, public, and private oracle, validating the MFT prediction.
- **Direction vs. Scale**: For shallow layers, Synthetic DP-KFC and oracle have cosine similarity >0.8; for deep layers, it drops below 0.6 (deep layer directions depend on labels), but Frobenius error remains minimal—the main advantage is in scale, not direction.
- **Domain Mismatch is the Achilles' Heel of Public DP-KFC**: On PathMNIST, Public degrades to 73.4%, while Synthetic reaches 78.2% (on par with Oracle), showing synthetic probes avoid negative transfer.
- **NLP Gap**: On StackOverflow, Synthetic lags behind Public because random token sequences do not lie on the real text manifold; the authors acknowledge this as future work.
- **Complementarity**: Scale-then-Privatize (DP-KFC) and Post-Privatize (DP-AdamBC, DiSK) are orthogonal; combined, they reach 95.5% at $\varepsilon=1$, surpassing either alone.

## Highlights & Insights
- **Theoretical Bridge**: Bridging Mean Field Theory's "activation variance is architecture-determined" to DP preconditioner design is a rare example of deep learning theory directly yielding practical DP algorithms.
- **Zero Privacy Cost**: The batch-independence of $P_t$ allows RDP to fully inherit the standard Gaussian mechanism, with no extra budget—this is the main advantage over other second-order DP methods.
- **Modality-Level Priors**: Using $1/f^\alpha$ pink noise and Zipf sampling as "task-agnostic but modality-aware" priors can be extended to audio ($1/f$ spectrum), point cloud (local isotropy), and other modalities.
- **"Privacy Wall" Insight**: Remark 5.5 reframes the DP training compute tradeoff—privacy budget limits steps $T$, so slower but more effective second-order methods are actually more cost-effective.

## Limitations & Future Work
- Poor alignment in deep layer directions means synthetic probes may be insufficient for very deep networks (e.g., ResNet-152); the authors use CrossViT, which is relatively deep, but do not test at ResNet-152 scale.
- The NLP gap shows that synthetic probes only recover architectural factors, not the real data manifold; token-frequency or embedding-space probes are a natural extension but not explored in the paper.
- Optimal values for $T_{freq}$ and synthetic batch size $M$ depend on architecture and task; the appendix ablation does not provide closed-form guidance.
- The 2.2× per-step overhead may be non-negligible for large models; KFAC's Kronecker structure requires K-FAC-reduce approximation for joint Q/K/V in attention, and the approximation error is not strictly validated.
- Lacks empirical results for federated learning scenarios (though mentioned in the introduction).

## Related Work & Insights
- **vs. Public DP-KFC (public data version)**: Same framework, public data version achieves higher upper bound but suffers negative transfer under domain mismatch (PathMNIST), while Synthetic is more robust.
- **vs. AdaDPS (Li et al. 2022)**: AdaDPS uses diagonal adaptive preconditioners; DP-KFC uses block-diagonal KFAC, extending from diagonal to block-diagonal second-order methods, with zero privacy cost.
- **vs. DP-AdamBC (Tang 2024) / DiSK (Zhang 2025)**: Both are post-privatize corrections, orthogonal to DP-KFC and can be combined for optimal results.
- **vs. DP-Newton (Ganesh 2023)**: Requires private Hessian, naive estimation amplifies noise by $O(d)$; DP-KFC completely avoids this.
- **vs. Mean Field Theory Initialization (Schoenholz, Yang 2017)**: Extends MFT from "data-independent init" to "data-independent preconditioning," following the same line of thought.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reconstructing KFAC preconditioners with synthetic noise and injecting into DP-SGD at zero privacy cost is a rare theory-to-practice closed loop
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 4 baselines × multiple $\varepsilon$ + domain transfer + ablation, 10 seeds
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative (geometry mismatch→architecture dominance→synthetic probe→scale-then-privatize) is coherent, theory and algorithm are clearly matched
- Value: ⭐⭐⭐⭐⭐ Provides a truly usable second-order DP method for domains like medical imaging where both privacy and lack of public proxies are critical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[ICML 2026\] Persuasive Privacy](persuasive_privacy.md)
- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](../../ICCV2025/ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)

</div>

<!-- RELATED:END -->
