---
title: >-
  [Paper Note] Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration
description: >-
  [ICLR 2026][Optimization & Theory][μP] The authors complete the μP/CompleteP scaling rules ("tune on small models, transfer to large models") across four critical training axes—width, depth, batch size, and training duration. They further demonstrate that under proper parameterization, even fine-grained "per-module" hyperparameters (individual learning rate
tags:
  - ICLR 2026
  - Optimization & Theory
  - μP
  - CompleteP
date: 2026-05-08
content_hash: 9087384617e45955
---
# Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=elB9k4nTL1](https://openreview.net/forum?id=elB9k4nTL1)  
**Code**: TBD  
**Area**: Optimization / Hyperparameter Transfer / Scaling Laws  
**Keywords**: Hyperparameter Transfer, μP, CompleteP, SDE Scaling Laws, Per-module Hyperparameters  

## TL;DR
The authors complete the μP/CompleteP scaling rules ("tune on small models, transfer to large models") across four critical training axes—width, depth, batch size, and training duration. They further demonstrate that under proper parameterization, even fine-grained "per-module" hyperparameters (individual learning rates, weight decays, and Adam parameters for each tensor/layer type) can be directly transferred from a 50M parameter proxy model to a 7.2B parameter model, achieving a ~1.3× training speedup.

## Background & Motivation
**Background**: Large model training is extremely sensitive to hyperparameters (learning rate, weight decay, initialization scale, AdamW $\beta_1, \beta_2, \epsilon$). Incorrect settings can lead to non-convergence or training instability. To avoid repeated trial-and-error in expensive large-scale training, "principled parameterization" schemes—represented by μP (Maximal Update Parameterization)—have been proposed. By using the infinite-width limit as a reference, finite-scale models approach the same well-posed limit, allowing **optimal global hyperparameters found on small models to be directly applied to large models**. Depth-μP and CompleteP extended this transfer from "width" to "depth."

**Limitations of Prior Work**: Current methods only address half of the problem. First, scaling rules for μP and its variants have **only been derived for model dimensions (width, depth)**. However, the other two levers of compute—batch size and the number of training tokens (duration)—also break hyperparameter transfer; changing training configurations along these axes renders small-model optimal hyperparameters suboptimal for large models. Second, prior work focuses on **global hyperparameters**, where the entire model shares a single learning rate. Since different tensors in μP are already scaled differently based on their architectural roles (e.g., embedding vs. hidden weights), there is no reason to believe their "base optimal values" should collapse to the same number. Per-module tuning likely offers significant untapped potential.

**Key Challenge**: Per-module tuning introduces a high-dimensional search problem. A Transformer block contains over a dozen module types; multiplied by the number of layers, the number of free hyperparameters is of order $|M| \times L$, making direct search on large models impossible. The question becomes: **Can we gain the benefits of per-module tuning by performing expensive searches only on small models and transferring the results via parameterization?**

**Goal**: (1) Complete scaling rules to enable hyperparameter transfer across width, depth, batch, and duration; (2) Verify that per-module hyperparameters can transfer under correct parameterization; (3) Provide a practical search recipe for high-dimensional per-module hyperparameter spaces prone to "divergence cliffs."

**Core Idea**: The authors propose **Complete(d)P** parameterization. Building on CompleteP (width + depth), they incorporate batch and training duration scaling rules derived from Stochastic Differential Equation (SDE) limits. They use a "depth × type" Kronecker factorization to compress per-module hyperparameters into $|M| + L$ transferable multipliers, enabling the "search on small + transfer to large" pipeline at a per-module granularity.

## Method

### Overall Architecture
The work consists of two layers. **The lower layer is Parameterization (Scaling Rules)**: It decouples training configurations that must change with scale—width $m_N$, depth $m_L$, batch ratio $m_B$, and token ratio $m_D$—from the hyperparameters intended for search (learning rate $\eta$, weight decay $\lambda$, $\beta_1, \beta_2, \epsilon$, initialization scale, and residual branch multipliers). Parameterization defines how each hyperparameter is scaled by a factor $\kappa$ when the configuration increases, ensuring model behavior follows the same infinite limit. Consequently, **optimal hyperparameters remain invariant across scales** and can be directly transferred. **The upper layer is Optimization**: Per-module hyperparameters are optimized on a 50M parameter / 1.6B token proxy model using trust-region random search. These are then transferred via the scaling rules to a 7.2B parameter target model without further tuning.

The comprehensive transfer rules are summarized in Table 1 of the original paper, providing scaling factors for initialization variance, learning rate, AdamW $\epsilon$, and weight decay for every tensor category (input embedding, hidden weights, QK norm, unembedding, etc.).

### Key Designs

**1. Complete(d)P: Completing and Correcting Width/Depth Parameterization**

The core idea of width/depth transfer is treating finite models as discretizations of an infinite limit. The difficulty is that different parameterizations lead to different limits, most of which are pathological (e.g., Standard Parameterization (SP) leads to exploding features). This work introduces three refinements to CompleteP: First, it **extends parameterization to QK normalization layers**, which are standard in modern Transformers but require separate scaling considerations due to weight sharing across attention heads. Second, it **corrects the AdamW $\epsilon$ scaling for input embeddings** in CompleteP—a small but critical fix to pass coordinate checks. Third, it **eliminates explicit scalar multipliers on the final linear projection output**, reparameterizing its role into the learning rate and initialization scale to maintain compatibility with memory-efficient algorithms like Cut Cross-Entropy.

For depth transfer, a depth-dependent rescaling factor on residual connections is used:

$$h_{\ell+1} = h_\ell + m_L^{-\alpha}\, F_\ell(h_\ell),\qquad \ell\in\{1,\dots,L\}$$

controlled by $\alpha\in[\tfrac12,1]$. The authors observe that **Complete(d)P allows depth transfer at both $\alpha=\tfrac12$ and $\alpha=1$**, contradicting prior reports that transfer degrades at $\alpha=0.5$. They attribute their success to the added stability of QK norm and point out implementation inconsistencies in prior work.

**2. Batch Scaling: Extending SDE Reparameterization to Weight Decay**

Following the perspective that training discretizes an SDE, the authors identify parameterizations that ensure consistent SDE limits across batch sizes. Using a simplified RMSPropW example where the gradient $g_k = g + \sigma e_k$ and weight decay is $\lambda$, the discretization of $d\Theta_t = \tfrac{1}{\eta\sigma}(g+\lambda\sigma\Theta_t)dt + dW_t$ leads to a **square-root scaling rule**:

$$\eta' = \sqrt{\kappa}\,\eta,\quad k' = \kappa k,\quad \lambda' = \sqrt{\kappa}\,\lambda$$

where $\kappa$ is the factor by which the batch size is multiplied. This work claims to be the first to extend SDE scaling to **AdamW weight decay**. They distinguish between the PyTorch implementation (which scales $\lambda$ by the learning rate) and the original AdamLH, noting that incorrect scaling leads to significant drift in the latter.

**3. Duration Scaling: Constant Time-Horizon SDE Rules**

The authors observe that the optimal learning rate decays as training duration increases, approximately proportional to $1/\sqrt{\kappa}$ (where $\kappa$ is the iteration multiplier). From an SDE perspective, scaling the learning rate by $1/\sqrt{\kappa}$ for a fixed batch size is equivalent to **decreasing the Signal-to-Noise Ratio (SNR) while keeping the time horizon constant**. They propose the **constant time-horizon scaling rule**: when scaling tokens, only the SNR parameters should be adjusted. This is validated by increasing tokens solely by increasing batch size (which affects only SNR); this results in near-perfect learning rate transfer and a lower asymptotic loss bound.

**4. Kronecker Factorization and Trust-Region Search for Per-Module HPs**

To make the $|M|\times L$ dimensional search feasible, the hyperparameter $\zeta_{m,\ell}$ for module $m$ at depth $\ell$ is factorized in log-space:

$$\log\zeta_{m,\ell}(T,N,L,B) = \underbrace{\log\zeta^{\text{type}}_m}_{\text{Type}} + \underbrace{\log\zeta^{\text{depth}}_\ell(L)}_{\text{Depth}} + \underbrace{\log\mathrm{SDE}(T,B)}_{\text{Batch/Duration}} + \underbrace{\log\mathrm{CP}_m(N,L)}_{\text{Complete(d)P}}$$

This reduces free multipliers to $|M|+L$. For transfer to a new depth $L'$, depth multipliers are linearly interpolated. Due to the complex, fractal-like boundaries and "divergence cliffs" in the per-module space, the authors use **trust-region random search** to stay within stable regions while optimizing.

### Loss & Training
Experiments use decoder-only Transformers with pre-norm and QK norm trained on RedPajama. The loss is cross-entropy plus Z-loss with a cosine LR schedule. The metric is the final validation loss on pre-training data.

## Key Experimental Results

### Main Results

| Scenario | Comparison | Result |
|----------|------------|--------|
| 50M / 1.6B tokens (Search) | Per-module vs. Optimal Global HPs | **2.3×** speedup to reach same performance |
| 7.2B (Target, ~1e4× FLOPs) | Transferred Per-module vs. Global HPs | Retains **1.32×** acceleration + benchmark gains |
| Token Duration Scaling | Complete(d)P+SDE vs. CompleteP | Superior loss lower bound (2.253 vs. 2.482) |

For width/depth transfer, optimal learning rates remain stable as width (128→2048) and depth (4→128) increase. For batch transfer, the square-root rule maintains optimality from $\tfrac{1}{8}\times$ to $4\times$ batch sizes.

### Ablation Study

| Configuration | Conclusion |
|---------------|------------|
| Remove per-depth multipliers | Still significantly better than global, but **worse than full version** |
| Further search for decoupled HPs from Kronecker | **Almost no gain**, indicating Kronecker captures most benefits |
| No scaling vs. SQRT scaling on batch axis | Transfer fails without scaling; LR/WD drift significantly |

### Key Findings
- **Per-module gains primarily stem from different module types within the residual block** receiving different learning rates; per-depth multipliers provide additional minor contributions.
- **Kronecker factorization (depth × type) does not sacrifice performance**, validating the reduction of dimensions from $|M|L$ to $|M|+L$.
- **The per-module speedup decays slowly with scale** (from 2.3× at 50M to 1.32× at 7.2B), though it's unclear if this is due to non-asymptotic imperfections or inherent to infinite-scale models.

## Highlights & Insights
- **Explicit decoupling of "training configuration" and "hyperparameters"** is the conceptual anchor. The parameterization's role is to absorb the effects of the former (data, size, batch) on the latter.
- **The SDE perspective unifies batch and duration axes**. Batch scaling maintains SDE consistency, while duration scaling is reinterpreted as an SNR adjustment at a constant time horizon.
- **Characterization of the per-module loss landscape** as "near-invex but with fractal boundaries" explains the failure of standard Bayesian optimization and the necessity of trust-region methods.

## Limitations & Future Work
- **Search Efficiency**: While effective, the authors suggest that trust-region Bayesian optimization or utilizing training structure could further improve search efficiency.
- **Limited Setup**: Only validated on autoregressive Transformers with RedPajama and a cosine schedule; scaling principles should be tested across more architectures and schedules.
- **Decay of Speedup**: The cause of the performance gain decay at larger scales remains unknown.

## Related Work & Insights
- **vs. μP / Depth-μP**: Extends model size transfer (two axes) to batch and duration (four axes) and moves from global to per-module HPs.
- **vs. CompleteP**: Fixes embedding AdamW $\epsilon$ scaling, extends to QK norm, and provides evidence that $\alpha=0.5$ does not necessarily lead to transfer degradation.
- **vs. SDE Scaling (Malladi et al., 2022)**: First to extend SDE reparameterization to AdamW weight decay.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Full 4-axis transfer and per-module granularity; first SDE scaling for AdamW weight decay)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Validated up to 7.2B and 1e4× FLOPs, though on a single setup)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear rules summarized in tables; high information density)
- **Value**: ⭐⭐⭐⭐⭐ (Directly reduces training costs for LLMs; provide practical rules and search recipes)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Wasserstein Transfer Learning](../../NeurIPS2025/optimization/wasserstein_transfer_learning.md)
- [\[ICLR 2026\] AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms](autoep_llms-driven_automation_of_hyperparameter_evolution_for_metaheuristic_algo.md)
- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] ConRep4CO: Contrastive Representation Learning of Combinatorial Optimization Instances across Types](conrep4co_contrastive_representation_learning_of_combinatorial_optimization_inst.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)

</div>

<!-- RELATED:END -->
