---
title: >-
  [Paper Note] Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation
description: >-
  [ICML2025][Image Generation][Discrete Diffusion] This paper proposes the Simple Iterative Denoising (SID) and Critical Iterative Denoising (CID) frameworks, which eliminate the compounding denoising error in discrete diffusion by assuming conditional independence of intermediate noise states. It introduces a Critic network to adaptively adjust element-wise re-noising probabilities, significantly outperforming standard discrete diffusion baselines on graph and molecule generat…
tags:
  - "ICML2025"
  - "Image Generation"
  - "Discrete Diffusion"
  - "Iterative Denoising"
  - "Critic"
  - "Graph Generation"
  - "Molecule Generation"
  - "Compounding Denoising Error"
  - "Flow Matching"
date: 2026-05-08
content_hash: 1328cec8fdd5494e
---

# Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation

**Conference**: ICML2025  
**arXiv**: [2503.21592](https://arxiv.org/abs/2503.21592)  
**Code**: [github.com/yoboget/sid](https://github.com/yoboget/sid)  
**Area**: Discrete Diffusion / Graph Generation  
**Keywords**: Discrete Diffusion, Iterative Denoising, Critic, Graph Generation, Molecule Generation, Compounding Denoising Error, Flow Matching

## TL;DR

This paper proposes the Simple Iterative Denoising (SID) and Critical Iterative Denoising (CID) frameworks, which eliminate the compounding denoising error in discrete diffusion by assuming conditional independence of intermediate noise states. It introduces a Critic network to adaptively adjust element-wise re-noising probabilities, significantly outperforming standard discrete diffusion baselines on graph and molecule generation tasks.

## Background & Motivation

Discrete Diffusion Models (DDM) and Discrete Flow Matching (DFM) have made significant progress in generative modeling of discrete data such as graph structures. However, these models face a key challenge—**Compounding Denoising Error**:

- **Root Cause**: In the reverse denoising process, the intermediate noisy state $z_s$ directly depends on the preceding step $z_t$, forming a Markovian chain dependency. Early in the generation, $Z_t$ has limited information, causing the model to output high-entropy distributions, and sampling errors accumulate and propagate across time steps.
- **Particularly Severe in Mask Diffusion**: Once an element is unmasked, subsequent steps cannot correct it ($p_{s|t}(z|Z_t) = \delta_{z_t}(z)$ if $z_t \neq \text{Mask}$), permanently locking in early errors.
- **Marginal/Uniform Distributions Also Affected**: Although elements can change at each step, because the probability of modifying elements during denoising is constrained by $\Delta_t \cdot \dot{\alpha_t}/(1-\alpha_t)$ (Equation 5), the correction probability is extremely low, and the risk of error propagation remains significant.

The authors substantiate this with molecular validity experiments on the Zinc250k dataset: standard discrete diffusion achieves at most ~80% molecular validity, whereas SID approaches 100% with only a few denoising steps.

## Method

### 1. Simple Iterative Denoising (SID)

**Core Idea**: Break the Markovian chain dependency of intermediate states, assuming that the noisy states at each time step depend only on the original clean data $z_1$ and are conditionally independent of each other.

**Noising Process** (same formulation as DDM but with different semantics):

$$q_{t|1}(z|z_1) = \alpha_t \delta_{z_1}(z) + (1-\alpha_t) q_0(z)$$

where $\alpha_t \in [0,1]$ is the noise schedule parameter, and $q_0(z)$ is the noise distribution (which can be uniform, mask, or marginal distribution).

**Key Assumption (Assumption 3.1)**: Conditional Independence—

$$q_{t|1}(z|z_1) = q_{t|1}(z|z_s, z_1) \quad \forall\, t \neq s$$

i.e., given $z_1$, the noisy states of any two time steps are conditionally independent.

**Denoising Process**: The conditional independence assumption derives a simplified denoising formulation (Proposition 3.3):

$$p_{s|t}(z|Z_t) = \alpha_s \cdot p_{1|t}(z|Z_t) + (1-\alpha_s) \cdot q_0(z)$$

This can be interpreted as a two-step operation: (1) predicting the clean instance $p_{1|t}(z_1|Z_t)$ from the noisy data; (2) re-noising the prediction as $q_{s|1}(z|z_1)$. The key difference is that $z_s$ only depends on the predicted clean instance, rather than the current noisy state $z_t$, thus severing the error propagation path.

**Equivalence (Proposition 3.4)**: One-step denoising in SID is equivalent to the maximal corrector step corrector sampling of discrete diffusion. Therefore, it can directly reuse any pre-trained DDM/DFM denoiser **without retraining**.

**Loss & Training**: Fully consistent with DDM/DFM, employing weighted negative log-likelihood:

$$\mathcal{L} = \mathbb{E}\left[\gamma \sum_{x_1^{(i)}} [-\log p_\theta(x_1^{(i)}|\mathcal{G}_t)] + (1-\gamma) \sum_{e_1^{(i,j)}} [-\log p_\theta(e_1^{(i,j)}|\mathcal{G}_t)]\right]$$

where $\gamma = n/(n+m)$ balances the node and edge weights.

### 2. Critical Iterative Denoising (CID)

**Design Motivation**: Under SID, all elements are re-noised with the equal probability $1-\alpha_t$, but different elements have varying likelihoods under the data distribution—re-noising probability should be increased for less likely elements.

**Critic Network**: A GNN $f_\phi$ is trained to estimate the element-wise adaptive noise rate $\hat{\alpha}_t = p_\phi(a|\hat{Z}_{1|t})$, predicting the probability of each denoised element originating from the true data distribution.

**Critic Loss**:

$$\mathcal{L}_\phi = -\mathbb{E}_{t, \hat{Z}_{1|t}} \sum_i \log p_\phi(a_t^{(i)} | \hat{Z}_{1|t})$$

**Optimal Critic (Theorem 4.1)**:

$$C^*({\hat{z}_{1|t}}) = \frac{\alpha_t \cdot p_{\text{data}}(\hat{z}_{1|t})}{\alpha_t \cdot p_{\text{data}}(\hat{z}_{1|t}) + (1-\alpha_t) \cdot p_{\text{pred}}(\hat{z}_{1|t})}$$

From this, two corollaries follow: if $p_{\text{data}} = p_{\text{pred}}$, then $\hat{\alpha}_t^* = \alpha_t$ (degenerates to SID); if $p_{\text{data}} > p_{\text{pred}}$, then $\hat{\alpha}_t^* > \alpha_t$ (retaining elements with high likelihood); otherwise, it increases the re-noising probability.

**Implementation**: The Critic is parameterized with residual logits $p_\phi(a|\hat{Z}_{1|t}) = \sigma(f_\phi(\hat{Z}_{1|t}, \alpha_t) + \sigma^{-1}(\alpha_t))$, trained post-hoc on top of the pre-trained denoiser without retraining it.

**Inference Pipeline**: Each step $\rightarrow$ (1) the denoiser predicts $\hat{Z}_{1|t}$ $\rightarrow$ (2) the Critic computes the adaptive $\hat{\alpha}_{t+\Delta t}$ $\rightarrow$ (3) re-noising is applied with $\hat{\alpha}_{t+\Delta t}$.

## Key Experimental Results

### Molecule Generation (QM9 & ZINC250k)

| Model | QM9 Val.↑ | QM9 FCD↓ | ZINC Val.↑ | ZINC FCD↓ |
|:---|:---:|:---:|:---:|:---:|
| GDSS | 95.72 | 2.90 | 97.01 | 14.65 |
| DruM | 99.69 | 0.11 | 98.65 | 2.25 |
| DiGress | 98.19 | 0.10 | 94.99 | 3.48 |
| Marg. DDM | 95.73 | 1.09 | 80.40 | 8.50 |
| Mask DDM | 48.38 | 3.76 | 8.96 | 24.98 |
| **Marg. SID** | **99.67** | **0.50** | **99.50** | **2.01** |
| Mask SID | 96.43 | 1.80 | 93.85 | 9.05 |
| **Mask CID** | **99.92** | 1.76 | **99.97** | 3.46 |

- Marg. SID achieves an FCD of 2.01 on ZINC250k, outperforming all baselines (including DruM's 2.25 with 1000 steps).
- Mask CID achieves a validity rate of **99.97%** on ZINC250k, reducing the number of invalid molecules to only 1/50 of the best baseline DruM.
- SID uses only 500 steps, whereas baselines use 1000 steps.

### General Graph Generation (Planar & SBM)

| Model | Planar Spect.↓ | Planar V.U.N.↑ | SBM Spect.↓ | SBM V.U.N.↑ |
|:---|:---:|:---:|:---:|:---:|
| DruM | **6.2** | 90 | **5.0** | **85** |
| DiGress | 10.6 | 75 | 40.0 | 74 |
| Marg. DDM | 83.57 | 0.0 | 11.82 | 0.0 |
| **Marg. SID** | **7.62** | **91.3** | **5.93** | **63.5** |
| **Mask CID** | **6.40** | 66.0 | 11.94 | 19.0 |

- Marg. SID achieves a V.U.N of 91.3% on Planar, exceeding DruM's 90%.
- **Mask CID** | **6.40** | 66.0 | 11.94 | 19.0 |

### NFE Ablation

- Mask CID requires only **32 steps** to achieve over 99% molecular validity.
- At an extremely low NFE of 16 steps, CID still outperforms other models across all metrics.

## Highlights & Insights

1. **Elegant and Effective Conditional Independence Assumption**: A seemingly simple assumption (that intermediate states are conditionally independent of each other) fundamentally eliminates the compounding error problem in discrete diffusion, offering a clean theory and simple implementation.
2. **Training-Free Upgrading**: SID can directly reuse any pre-trained DDM/DFM denoiser without retraining; the Critic is also trained post-hoc.
3. **Unified Perspective with Corrector Sampling**: SID is equivalent to corrector sampling with a maximal corrector step, providing a unified framework for understanding different sampling strategies.
4. **GAN-like Motivation for Critic**: The optimal Critic behaves similarly to a GAN discriminator (comparing $p_{\text{data}}$ and $p_{\text{pred}}$), but is used to guide re-noising instead of adversarial training.
5. **Empirical Insights on Noise Distributions**: Marginal distributions consistently outperform mask distributions for graph generation, demonstrating that the compounding error problem is particularly severe in mask diffusion.

## Limitations & Future Work

1. **Evaluation Limited to Graph/Molecule Tasks**: Experiments have not been conducted on other discrete structures such as text or proteins, leaving its generality to be verified.
2. **Limited Gains of Critic on General Graphs**: The improvement of CID over SID is marginal on Planar/SBM, which is hypothesized to be due to more dispersed data distribution deviations in general graphs.
3. **V.U.N on SBM Dataset Inferior to DruM**: The 63.5% of Marg. SID is still lower than DruM's 85%, indicating limitations of the method on certain graph categories.
4. **Additional Training and Inference Overhead for Critic**: Each step requires two forward passes (denoiser + Critic), and the absolute inference speed is not reported.
5. **Theoretical Limitations of the Conditional Independence Assumption**: Despite the strong empirical performance, this assumption violates the Markovian property of real data, and thus does not theoretically guarantee convergence to the correct data distribution.

## Related Work & Insights

- **Cold Diffusion (Bansal et al., 2023)**: Represents the general paradigm of degrade-and-restore strategies; this work can be seen as its instantiation on discrete graph structures.
- **DiGress (Vignac et al., 2023)**: A discrete diffusion graph generation baseline that utilizes marginal distributions.
- **DruM (Jo et al., 2024)**: A diffusion bridge model, which remains a strong baseline on general graphs.
- **MDLM/UDLM (Sahoo et al., 2024)**: Explores the time-invariance of mask denoisers, a property inherited by SID.
- **Planning Methods (Liu et al., 2025)**: Selective denoising strategies, which are complementary to the adaptive re-noising in CID.

## Rating

- Novelty: ⭐⭐⭐⭐ — The conditional independence assumption is simple yet profound, and the theoretical characterization of the Critic is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets + ablation studies, with a well-designed fair comparison, but lacking validation on non-graph domains.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, complete theoretical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Offers important contributions to the discrete diffusion community, with high practical value due to training-free upgrading.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICML 2025\] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling](bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](../../ICLR2026/image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[ICML 2025\] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](../../NeurIPS2025/image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)

</div>

<!-- RELATED:END -->
