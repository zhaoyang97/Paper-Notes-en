---
title: >-
  [Paper Note] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning
description: >-
  [ICML 2026][Medical Imaging][Differential Privacy] This paper proposes DP-KFC: based on the observations that "the scale of the Fisher matrix is determined by the architecture and the correlation structure can be approxi…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Differential Privacy"
  - "KFAC"
  - "Fisher Information Matrix"
  - "Preconditioner"
  - "Synthetic Noise"
date: 2026-05-08
content_hash: f4f9a7bb76758d34
---

# DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2605.13418](https://arxiv.org/abs/2605.13418)  
**Code**: https://github.com/molinamarcvdb/DP-KFC (Available)  
**Area**: Differential Privacy / Medical Imaging / Second-order Optimization  
**Keywords**: Differential Privacy, KFAC, Fisher Information Matrix, Preconditioner, Synthetic Noise

## TL;DR
This paper proposes DP-KFC: based on the observations that "the scale of the Fisher matrix is determined by the architecture and the correlation structure can be approximated by modality-level spectral statistics," structured synthetic noise ($1/f^\alpha$ pink noise for images, Zipf sampling for text) is used to probe the network and reconstruct KFAC preconditioners. This approach consumes no privacy budget and introduces no distribution shift, consistently outperforming DP-SGD and public-data preconditioning methods under strong privacy constraints ($\varepsilon\le 3$).

## Background & Motivation

**Background**: The standard practice for differentially private deep learning is DP-SGD—injecting isotropic Gaussian noise after $L_2$ clipping of per-sample gradients. The scale of privacy noise grows with the model dimension $\sqrt{d}$, causing the advantages of over-parameterization seen in non-private scenarios to disappear under DP. To mitigate this, the community has turned to adaptive/second-order methods (DP-Adam, KFAC + DP), which either consume privacy budget to estimate second-order statistics from private data or suffer from distribution shifts when using public data.

**Limitations of Prior Work**: (1) The "isotropic" noise of DP-SGD is mismatched with the highly "anisotropic" geometry of the neural network loss landscape—low-sensitivity parameters are drowned by noise, while high-sensitivity parameters are excessively clipped (Fig. 1 SNR collapse); (2) Ganesh et al. (2025) proved that unbiased second-order estimation under privacy often does more harm than good, as the noise in the preconditioner itself hinders performance; (3) The "Precondition-then-Privatize" paradigm relies on public proxy data, which is often unavailable in specialized domains like medical imaging.

**Key Challenge**: To match the privacy noise of DP-SGD with the loss geometry, gradients must first be transformed into an isotropic coordinate system. However, obtaining the curvature information required for this transformation must **not** consume any privacy budget and must **not** rely on public data.

**Goal**: (1) Prove that key information of the KFAC preconditioner can be recovered from the architecture itself; (2) Design an algorithm to reconstruct the preconditioner using only synthetic noise; (3) Strictly maintain the formal guarantees of DP (no additional privacy budget consumption).

**Key Insight**: According to Mean Field Theory, the layer activation variance $q^l$ and backpropagated gradient variance $\tilde q^l$ in deep networks follow deterministic recursions (determined by initialization and non-linearity) and are independent of specific inputs. Karakida et al. (2019) proved that $\text{Tr}(F_l)\propto d\cdot q^{l-1}\cdot \tilde q^l$, meaning the trace of Fisher blocks is entirely determined by the architecture.

**Core Idea**: Decouple the Fisher matrix into "architectural sensitivity (recoverable via synthetic noise) + input correlation structure (approximated by modality-level spectra $1/f^\alpha$)." Use synthetic probes to construct the inverse square root of KFAC factors $F^{-1/2}$ and perform a linear transformation (scale-then-privatize) **before** per-sample gradients are clipped and noised. This step is transparent to the private data and thus consumes no privacy budget.

## Method

### Overall Architecture
The DP-KFC workflow consists of two phases that alternate periodically:
1. **Preconditioner Construction** (every $T_{freq}$ steps): Perform forward and backward passes using a synthetic batch (pink noise / Zipf sequences). Estimate per-layer $\hat A_{l-1}=\mathbb{E}[\tilde a_{l-1}\tilde a_{l-1}^\top]$ and $\hat G_l=\mathbb{E}[\tilde\delta_l\tilde\delta_l^\top]$ using KFAC formulas. Apply eigen-decomposition to obtain $U_{A,l}, U_{G,l}$.
2. **Private Training Step**: For each private sample $i$ and each layer $l$, first transform the gradient $\tilde g_l^{(i)}=U_{G,l}\cdot g_l^{(i)}\cdot U_{A,l}$, then perform clip + noise + average → SGD update.

The critical "scale-then-privatize" sequence implies that $P_t$ is independent of the current batch (depending only on the architecture and previously released model parameters), so the RDP guarantee fully inherits the accounting of standard DP-SGD.

### Key Designs

1. **Data-Free KFAC Factor Estimation**:
    - **Function**: Recover per-layer KFAC covariances $\hat A_{l-1}, \hat G_l$ from synthetic probes without accessing private or public data.
    - **Mechanism**: Algorithm 1—Generate $M$ synthetic $(\tilde x, \tilde y)$, perform forward/backward passes to get activations and errors, and aggregate via outer products: $\hat A_{l-1}=\frac{1}{M}\sum \tilde a_{l-1}\tilde a_{l-1}^\top+\pi I$, $\hat G_l=\frac{1}{M}\sum \tilde\delta_l\tilde\delta_l^\top+\pi I$. After eigen-decomposition, set $U_{X,l}=Q_X(\Lambda_X+\gamma I)^{-1/2}Q_X^\top$. The preconditioner $F_l^{-1/2}=U_{A,l}\otimes U_{G,l}$ is represented implicitly via Kronecker products, avoiding the materialization of the full FIM.
    - **Design Motivation**: MFT suggests $\text{Tr}(F_l)$ depends on architecture rather than data, so synthetic probes only need to maintain consistency in the architectural propagation chain. Damping terms $\pi I$ and $\gamma I$ ensure invertibility and control the condition number ($\lambda_{min}\ge\sqrt\gamma$ in Theorem 5.4).

2. **Modality-Specific Synthetic Probes**:
    - **Function**: Ensure synthetic inputs carry architectural information while simulating the low-dimensional manifold structure of real data.
    - **Mechanism**: **Image domain**: Use Pink Noise—weight white noise $Z$ in the frequency domain as $\tilde Z_\mathbf{u}=Z_\mathbf{u}/(\|\mathbf{u}\|_2^{\alpha/2}+\epsilon)$ followed by IFFT, where $\alpha\approx 1$ simulates the $1/f^\alpha$ spectrum of natural images (Field 1987). **NLP domain**: Draw tokens from the vocabulary using a Zipfian distribution, placing [CLS], [SEP], and [PAD] according to syntactic positions to ensure attention and LayerNorm follow realistic paths.
    - **Design Motivation**: White noise energy is uniformly distributed, but deep networks primarily pass low-frequency features. Modality-level priors push synthetic probes toward activation statistics close to real data without carrying semantic information → zero privacy leakage.

3. **Scale-then-Privatize with DP Integration**:
    - **Function**: Inject curvature information into DP-SGD without increasing RDP consumption.
    - **Mechanism**: Algorithm 2—The gradient transformation $\tilde g_l=U_{G,l}\,g_l\,U_{A,l}$ is completed before clipping. The global $L_2$ norm becomes $\nu_i=\sqrt{\sum_l\|\tilde g_l^{(i)}\|_F^2}$, while the clipping threshold $C$ remains unchanged. Noise $\mathcal{N}(0,\sigma^2 C^2 I)$ is added to the sum after clipping. Proposition 5.6 proves that because $P_t$ is a batch-independent linear operator, the RDP guarantee of the composed mechanism is identical to the standard Gaussian mechanism.
    - **Design Motivation**: Compared to "Precondition-after-Privatize" (noising first, then multiplying by $P_t$), scale-then-privatize prevents the privacy noise term $d\sigma^2 C^2/B^2$ from being amplified by $\lambda_{max}^2$ (Theorem 5.4), offering the greatest advantage in high-privacy regimes.

### Loss & Training
- Standard CE/MSE loss with DP-SGD optimizer; KFAC damping $\pi=\gamma=10^{-2}$; Opacus for RDP accounting on A100 GPUs.
- Preconditioner refresh frequency $T_{freq}$ (typically 100–1000 steps). Per-step wall-clock time is ~2.2× slower than DP-SGD, but each step is more efficient (Remark 5.5 "privacy wall"—the privacy budget limits the number of steps $T$, making each step more valuable).
- Models: CNN on MNIST, CrossViT on CIFAR-100, BERT on StackOverflow, Logistic Regression on IMDB; privacy budget $\varepsilon\in[0.5,10]$.
- Convergence Guarantee: Theorem 5.4 $\min_t\mathbb{E}\|\nabla\mathcal{L}\|^2\le \frac{C_1}{\lambda_{min}\sqrt T}+\frac{C_2}{\lambda_{min}\sqrt T}(\lambda_{max}^2\sigma_{sgd}^2+\frac{d\sigma^2 C^2}{B^2})$, achieving the $O(T^{-1/2})$ non-convex optimal rate.

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
| **Synthetic DP-KFC (Ours)** | **94.2 ± 0.5** | **95.0 ± 0.4** | **95.9 ± 0.3** |
| Synthetic DP-KFC + DP-AdamBC | **95.5 ± 0.3** | **96.1 ± 0.2** | 96.4 ± 0.3 |

Cross-modality results ($\varepsilon=1$):
- CIFAR-100 + CrossViT: Synthetic DP-KFC is almost identical to Public DP-KFC, both exceeding DP-Adam by ≈1.4%.
- StackOverflow + BERT: Synthetic 91.8% vs. DP-SGD 89.5% (+2.3%), though still ≈4% behind Public DP-KFC (96.1%).
- IMDB + LR: Both Synthetic and Public DP-KFC reach 85.8% vs. DP-SGD 83.1%.

### Ablation Study

Transfer / Domain Mismatch (MNIST training, $\varepsilon=1.0$):

| Method | Fashion←MNIST (Ideal) | Path←MNIST (Texture Disjoint) |
|--------|----------------|----------------|
| Oracle (Private) | 88.3 ± 0.2 | 78.4 ± 1.7 |
| DP-SGD | 83.5 ± 0.7 | 68.5 ± 2.3 |
| AdaDPS (Public) | 84.7 ± 0.3 | 70.5 ± 2.0 |
| Public DP-KFC | 87.6 ± 0.2 | 73.4 ± 1.3 |
| **Synthetic DP-KFC (Ours)** | **87.8 ± 0.2** | **78.2 ± 1.9** |

### Key Findings
- **Architecture Dominates Curvature**: Fig. 2 shows that KFAC eigenvalue decay for MLP/CNN/Attention layers almost overlaps between synthetic, public, and private oracles, validating MFT inferences.
- **Direction vs. Scale**: In shallow layers, Synthetic DP-KFC achieves cosine similarity >0.8 with the oracle, dropping to <0.6 in deep layers (where direction depends on labels). However, Frobenius error remains minimized—the advantage is primarily in scale rather than direction.
- **Domain Mismatch is the Archilles' heel of Public DP-KFC**: On the PathMNIST task, Public performance degrades to 73.4%, while Synthetic remains at 78.2% (matching Oracle), proving synthetic probes avoid negative transfer.
- **NLP Gap**: On StackOverflow, Synthetic underperforms Public because random token sequences fail to fall on the low-dimensional manifold of real text; the authors acknowledge this as future work.
- **Complementarity**: Scale-then-Privatize (DP-KFC) and Post-Privatize (DP-AdamBC, DiSK) methods are orthogonal; their combination reaches 95.5% at $\varepsilon=1$, exceeding either alone.

## Highlights & Insights
- **Theoretical Bridge**: Bridging the Mean Field Theory conclusion—that activation variance is determined by architecture—to DP preconditioner design is a rare instance of deep learning theory yielding a practical DP algorithm.
- **Zero Privacy Cost**: The batch-independent property of $P_t$ ensures RDP inherits standard Gaussian guarantees without any extra budget—the strongest selling point compared to other second-order DP methods.
- **Modality-Level Priors**: Using $1/f^\alpha$ pink noise and Zipf sampling as "task-agnostic but modality-aware" priors provides a strategy transferable to other modalities like audio ($1/f$ spectrum) or point clouds (local isotropy).
- **"Privacy Wall" Insight**: Remark 5.5 reframes the compute cost of DP training—since the privacy budget limits the number of steps $T$, slower but more effective second-order methods per step are actually more cost-effective.

## Limitations & Future Work
- Poor direction alignment in deep layers suggests synthetic probes may be insufficient for extremely deep networks (e.g., ResNet-152); the authors used CrossViT but did not test ResNet-152 scale.
- The NLP gap indicates synthetic probes only recover architectural factors, not the real data manifold. Token-frequency or embedding-space probes are natural extensions but were not implemented.
- Optimal values for $T_{freq}$ and synthetic batch size $M$ depend on architecture and task; appendix ablations do not provide closed-form guidance.
- Per-step 2.2× overhead might be non-negligible for large models. The Kronecker structure of KFAC for Attention (Q/K/V) requires K-FAC-reduce approximations, which were not rigorously validated for error.
- Lack of empirical results for Federated Learning scenarios (though mentioned in the intro).

## Related Work & Insights
- **vs. Public DP-KFC (Public version of this work)**: Same framework; the public version has a higher upper bound but suffers from negative transfer in domain mismatch (PathMNIST), whereas Synthetic is more robust.
- **vs. AdaDPS (Li et al. 2022)**: AdaDPS uses diagonal adaptive preconditioning; DP-KFC uses block-diagonal KFAC, extending from diagonal to block-diagonal second-order methods at zero privacy cost.
- **vs. DP-AdamBC (Tang 2024) / DiSK (Zhang 2025)**: The latter are post-privatize corrections; they are orthogonal to DP-KFC and their combination is optimal.
- **vs. DP-Newton (Ganesh 2023)**: Requires private Hessians, causing $O(d)$ noise amplification; DP-KFC avoids this entirely.
- **vs. Mean Field Theory Initialization (Schoenholz, Yang 2017)**: Extends MFT from "data-independent initialization" to "data-independent preconditioning."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reconstructing KFAC preconditioners via synthetic noise for zero-privacy-cost injection into DP-SGD is a rare theory-to-practice closed loop.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 datasets × 4 baselines × multiple $\varepsilon$ + domain transfer + ablations, 10 seeds.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative (geometric mismatch → architecture dominance → synthetic probes → scale-then-privatize) is cohesive, with clear theoretical and algorithmic alignment.
- **Value**: ⭐⭐⭐⭐⭐ Provides a truly usable second-order DP method for fields like medical imaging where privacy is critical and public proxies are unavailable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](../../AAAI2026/medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](../../CVPR2026/medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[AAAI 2026\] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](../../AAAI2026/medical_imaging/qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)
- [\[ICML 2026\] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)

</div>

<!-- RELATED:END -->
