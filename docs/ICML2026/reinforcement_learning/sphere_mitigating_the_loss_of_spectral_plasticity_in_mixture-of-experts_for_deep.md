---
title: >-
  [Paper Note] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Plasticity loss] This paper formalizes the plasticity loss of MoE policies in continual reinforcement learning as a decline in the empirical NTK matrix spectral entropy effective rank…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Plasticity loss"
  - "MoE policy"
  - "NTK spectrum"
  - "effective rank"
  - "Parseval regularization"
date: 2026-05-08
content_hash: 60f2ead8fd129056
---

# SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.04712](https://arxiv.org/abs/2605.04712)  
**Code**: Not released  
**Area**: Reinforcement Learning / Mixture-of-Experts / Continual Learning  
**Keywords**: Plasticity loss, MoE policy, NTK spectrum, effective rank, Parseval regularization

## TL;DR
This paper formalizes the plasticity loss of MoE policies in continual reinforcement learning as a decline in the empirical NTK matrix spectral entropy effective rank, reduces it via Gauss-Newton and Kronecker decomposition to a computable proxy dependent only on the "expert feature Gram matrix," and finally uses a one-line Parseval penalty (SPHERE) to increase this proxy. On MetaWorld and HumanoidBench continual RL settings, task success rates are improved by 133% and 50%, respectively.

## Background & Motivation

**Background**: MoE architectures have expanded from LLMs into deep reinforcement learning (DRL)—multi-task robotic arms, humanoids, quadrupeds, and large-scale online RL all use sparsely routed experts to scale policy capacity (Top-$K$ MoE, Dense-MoE, DS-MoE). In continual RL (CRL), agents must learn multiple tasks sequentially, making MoE's capacity advantages especially relevant.

**Limitations of Prior Work**: Empirically, MoE policies in CRL often degrade significantly—Willi et al. 2024 reported "late-stage task success collapse." This is a typical manifestation of plasticity loss: the longer the training, the weaker the ability to learn new skills from new data. While the literature provides explanations for dense networks (dormant neurons, representation spectral collapse, Hessian spectral collapse), the specific form and countermeasures for plasticity loss in sparse, branched MoE structures remain largely unexplored.

**Key Challenge**: Direct tools for characterizing plasticity—e.g., the eNTK matrix $\mathbf{K} = \mathbf{J}\mathbf{J}^\top \in \mathbb{R}^{N \times N}$—are unwieldy for MoE. Forming $\mathbf{K}$ requires $O(N^2 P)$ time and $O(NP + N^2)$ memory, with $P$ covering all expert parameters, making it infeasible for direct monitoring or loss optimization. Thus, plasticity loss is either unobservable or unoptimizable; a proxy that both reflects MoE plasticity and is backpropagatable is needed.

**Goal**: (1) Provide a formal definition of MoE policy plasticity loss; (2) Reduce the intractable full eNTK effective rank to a differentiable small-matrix proxy; (3) Design a regularizer that contracts the spectrum of the proxy; (4) Validate on mainstream CRL benchmarks.

**Key Insight**: Starting from the function space gradient descent formula $\Delta f = -\eta \mathbf{K} \nabla_f L$, the eigen-spectrum of $\mathbf{K}$ directly determines "which directions the gradient can move." When the spectrum collapses (a few eigenvalues dominate), $\mathbf{K}$ imposes a strong prior on the gradient, locking updates into a few principal directions—this is the essence of plasticity. Thus, "high plasticity = spectral isotropy," naturally quantified by spectral entropy effective rank $r_e(\mathbf{K}) = \exp(-\sum p_i \log p_i)$, where $p_i = \sigma_i / \sum_j \sigma_j$.

**Core Idea**: Approximate $r_e(\mathbf{K})$ via block-diagonalization (Gauss-Newton + intra-layer independence + Kronecker decomposition, inspired by K-FAC), reducing it to the "last-layer expert-weighted feature Gram matrix $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$." Then, apply a Frobenius penalty that pushes $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ toward $\frac{\mathrm{Tr}}{m}\mathbf{I}$, which provably increases $r_e(\mathbf{K})$.

## Method

### Overall Architecture

Input: Forward output of a Top-$K$ MoE actor in a PPO training pipeline.  
Output: A differentiable regularization term added to the PPO loss.  
The derivation is a chain of "layer-wise proxies": $r_e(\mathbf{K}) \to r_e(G^{\mathrm{GN}}) \to$ block-diagonal approximation $\to$ per-layer expert block $r_e(\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell) \to$ Kronecker proxy condition number lower bound $\frac{k_\ell}{\kappa(\mathbf{A}^{\mathrm{exp}}_{\ell-1} \otimes \mathbf{G}^{\mathrm{exp}}_\ell)} \to$ spectral contraction only on $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ (last layer, heaviest expert block).  
The final loss: $\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \lambda^e \cdot \|\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}} - \tfrac{\mathrm{Tr}(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})}{m}\mathbf{I}_m\|_F^2$.

### Key Designs

1. **From eNTK to Gauss-Newton + Block-Diagonal Approximation**:

    - **Function**: Equates the effective rank of $\mathbf{K} \in \mathbb{R}^{N \times N}$ to that of the parameter-space GN matrix $G^{\mathrm{GN}} = \tfrac{1}{N}\mathbf{J}^\top \mathbf{J} \in \mathbb{R}^{P \times P}$, then partitions it into block-diagonal form by "gating + per-expert per-layer."
    - **Mechanism**: Using the fact that $\mathbf{J}\mathbf{J}^\top$ and $\mathbf{J}^\top \mathbf{J}$ share nonzero spectra, $r_e(\mathbf{K}) = r_e(G^{\mathrm{GN}})$; then, following K-FAC's block-diagonal approximation (ignoring cross-layer and cross-gate-expert blocks), $G^{\mathrm{GN}} \approx \bigoplus_\ell \mathbf{G}^{\mathrm{GN},\mathrm{g}}_\ell \oplus \bigoplus_\ell \mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell$. Lemma 4.1 shows the effective rank of a block-diagonal matrix decomposes as $r_e(M) = \exp(H(\alpha) + \sum \alpha_b \log r_e(M_b))$, where $\alpha_b = \|M_b\|_*/\sum_m \|M_m\|_*$. Since gating parameters are far fewer than expert parameters ($P^g \ll P^{\mathrm{exp}}$), gating blocks are treated as constants, focusing optimization on expert blocks.
    - **Design Motivation**: The original $\mathbf{K}$ is intractable; this step reduces the "global rank" problem to "per-layer expert block rank," with a strict inequality rather than a heuristic, providing theoretical support for subsequent optimization.

2. **Kronecker Proxy + Condition Number Lower Bound**:

    - **Function**: Further factorizes each expert layer block $\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell$ into "weighted expert feature Gram $\mathbf{A}^{\mathrm{exp}}_{\ell-1}$ ⊗ backprop gradient Gram $\mathbf{G}^{\mathrm{exp}}_\ell$," and provides an optimizable lower bound $r_e(\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell) \ge k_\ell / \kappa(\mathbf{A} \otimes \mathbf{G})$.
    - **Mechanism**: Under intra-layer independence, using K-FAC, each expert's input per sample $a^{\mathrm{exp}}_{e,\ell-1}(x_i)$ is weighted by Top-$K$ gating $h^{(K)}_{i,e}$ and concatenated across experts: $a_{\ell-1}(x_i) = [h^{(K)}_{i,1} a^{\mathrm{exp}}_{1,\ell-1}^\top | \dots | h^{(K)}_{i,E} a^{\mathrm{exp}}_{E,\ell-1}^\top]^\top$, stacking to form $\mathbf{A}^{\mathrm{exp}}_{\ell-1} = \tfrac{1}{N}\Phi_{\ell-1}^\top \Phi_{\ell-1}$. The same applies for the gradient Gram $\mathbf{G}^{\mathrm{exp}}_\ell$. The eigenvalues of a Kronecker matrix are products of those of the factors, so the condition numbers multiply, yielding the lower bound $r_e \ge k_\ell / \kappa$.
    - **Design Motivation**: $\mathbf{A}^{\mathrm{exp}}_{\ell-1}$ is a low-dimensional matrix (dimension $\sum_e d^{\mathrm{exp}}_{e,\ell-1}$, typically hundreds) available from the forward pass, without requiring backward construction; its condition number lower bound serves directly as the loss. "Cross-expert concatenation rather than per-expert separation" is key—off-diagonal Gram blocks capture cross-expert correlations, implicitly suppressing feature collapse into the same direction.

3. **SPHERE Parseval Penalty + Spectral Contraction Proof**:

    - **Function**: Defines $\mathcal{L}_{\mathrm{SPHERE}}(\mathbf{A}) = \|\mathbf{A} - \tfrac{\mathrm{Tr}(\mathbf{A})}{m}\mathbf{I}_m\|_F^2$, and proves it is spectrally contractive (eigenvalues contract toward the mean), so $\kappa(\mathbf{A})$ decreases monotonically and $r_e(\mathbf{K})$ increases monotonically.
    - **Mechanism**: Expanding $\mathcal{L}_{\mathrm{SPHERE}}$ yields $\|\mathbf{A}\|_F^2 - \tfrac{\mathrm{Tr}(\mathbf{A})^2}{m}$; taking the gradient w.r.t. $\mathbf{A}$ and performing one SGD step, it is shown that for $\eta \le \tfrac{1}{2}$, each eigenvalue contracts as $\lambda_i \to (1-\beta)\lambda_i + \beta \bar\lambda$ toward the mean. Using the Kronecker monotonicity lemma ($\kappa(A_{t+1} \otimes B) \le \kappa(A_t \otimes B)$), this contraction propagates to the Kronecker proxy, and via block-diagonal decomposition to $r_e(\mathbf{K})$. In practice, only the last-layer expert block is penalized (deep representations are most prone to collapse), leaving the gradient Gram untouched as it would require extra backward computation.
    - **Design Motivation**: The aim is not just an "empirically useful regularizer" but one that provably increases the plasticity measure. Parseval's push-to-identity penalty satisfies spectral contraction, and the chain of inequalities ensures "adding this term → $r_e(\mathbf{K})$ increases" is a theorem, not a heuristic.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \lambda^e \cdot \mathcal{L}_{\mathrm{SPHERE}}(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})$. At each gradient update, $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ is computed from the forward output. Top-$K$ MoE uses $E = 10$ experts, $K = 2$; MetaWorld trains $10^6$ env steps per task, HumanoidBench $10^7$ env steps per task.

## Key Experimental Results

### Main Results

| Benchmark | Method | Setting | Avg. Success Rate | Notes |
|-----------|--------|---------|-------------------|-------|
| MetaWorld CW10 | Top-$K$ MoE | CRL | baseline | Severe CRL degradation |
| MetaWorld CW10 | + SPHERE | CRL | **+133%** | RL-CRL gap reduced by 52% |
| HumanoidBench H1 | Top-$K$ MoE | RL | baseline | Also drops within single task |
| HumanoidBench H1 | + SPHERE | RL | **+36%** | Drift evident over long horizon ($10^7$ steps) |
| HumanoidBench H1 | Top-$K$ MoE | CRL | baseline | – |
| HumanoidBench H1 | + SPHERE | CRL | **+50%** | – |

### Ablation Study

| Configuration | HumanoidBench CRL Avg. Success Rate | Description |
|---------------|-------------------------------------|-------------|
| w/o SPHERE | $0.36 \pm 0.08$ | Baseline without regularization |
| **w/ SPHERE** | $\mathbf{0.54 \pm 0.12}$ | Full method |
| All hidden expert layers regularized | $0.42 \pm 0.07$ | Over-constrains shallow representations |
| Per-expert loss sum (no cross-expert concat) | $0.40 \pm 0.08$ | Validates importance of cross-expert terms |
| Regularize $\mathbf{G}^{\mathrm{exp}}_{\mathrm{last}}$ instead of $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ | $0.43 \pm 0.09$ | Feature Gram provides main gain |

### Key Findings

- **MoE requires plasticity intervention more than dense PPO**: Figure 3 shows $r_e(\mathbf{K})$ drops for PPO/Top-$K$/Dense-MoE/DS-MoE under CRL, but MoE variants drop more sharply, supporting the intuition that sparse gating amplifies representation collapse.
- **Cross-expert concatenation is a key design**: Per-expert regularization yields only minor improvement (0.40), while joint regularization after concatenation achieves 0.54, proving that cross-expert correlation (off-diagonal Gram blocks) is the main channel for plasticity loss.
- **$r_e(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})$ and $r_e(\mathbf{K})$ have Pearson correlation 0.846**: The effectiveness of the proxy is independently validated, not just a theoretical lower bound.
- **MetaWorld vs HumanoidBench benefit structure differs**: The former benefits mainly in CRL (task switching drives plasticity loss), while the latter benefits even in single-task RL (long horizon $10^7$ steps causes endogenous distribution drift and plasticity loss)—showing plasticity loss is not just a "task switching" issue, but also arises in long-horizon single tasks.

## Highlights & Insights

- Provides a mathematically defined, optimizable, and provably improvable proxy for the previously fuzzy phenomenon of "plasticity loss"—the paper's main contribution. The chain "$r_e(\mathbf{K}) \to G^{\mathrm{GN}} \to$ block-diagonal $\to$ Kronecker proxy $\to$ Parseval penalty" is rigorously formalized at each step (K-FAC, spectral entropy rank, Marshall majorization).
- "Gating-weighted cross-expert concatenation" is a MoE-specific, elegant design—it directly bakes Top-$K$ sparse routing into the Gram matrix, unlike treating experts as independent modules, and explicitly constrains experts to share a "diverse yet consistent" representation space.
- Achieving +36% even on RL single-task in HumanoidBench shows plasticity loss is not just a "continual learning" issue; long-horizon single-task distribution drift (on-policy data continually changing) is sufficient to trigger it—this may prompt a re-examination of long-horizon RL training paradigms.
- The theoretical derivation applies to dense MoE and DS-MoE, so the SPHERE proxy can be extended to LLM-as-policy, a recent popular setting.

## Limitations & Future Work

- Block-diagonal and Kronecker intra-layer independence approximations are classic K-FAC assumptions; the authors provide empirical validation only in the appendix, without non-asymptotic error bounds—if experts are strongly coupled (e.g., shared experts), the proxy may be inaccurate.
- Experiments are limited to continuous control MoE policies, not covering discrete actions or LLM-as-policy; the latter involves orders of magnitude more experts and dimensions than robotics, so feature Gram memory/computation may become a bottleneck.
- $\lambda^e$ is a fixed hyperparameter; task-adaptive or scheduled variants are unexplored. Early CRL may not need strong spectral constraints, which may only be necessary later.
- Only the last layer is regularized, but the choice of "which layer to regularize" is empirical (deep representations are most prone to collapse). Whether this holds for deeper/multi-layer expert architectures remains to be seen.

## Related Work & Insights

- **vs LayerNorm (Juliani & Ash 2024)**: Empirically, LN alleviates plasticity loss but only stabilizes forward numerics, not explicitly acting on the NTK spectrum. SPHERE directly optimizes the plasticity measure, with provable improvement.
- **vs Parseval Regularization (Chung et al. 2024)**: Original PW regularizes weight matrices to orthogonality in parameter space. SPHERE applies Parseval's idea to the expert feature Gram in representation space, specifically adapted for MoE's cross-expert structure.
- **vs Spectral Normalization (Miyato et al. 2018; Bjorck et al. 2021)**: SN controls only the largest singular value; SPHERE maintains "full-spectrum uniformity," directly targeting isotropic NTK spectra.
- **vs CBP (Dohare et al. 2024)**: CBP periodically reinitializes some neurons (structural perturbation), complementary to SPHERE's smooth gradient regularization—potentially combinable in future work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a formal NTK-based definition and optimizable proxy for MoE plasticity loss; clean derivation chain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks (MetaWorld + HumanoidBench), RL/CRL protocols, 5 baselines + 4 ablations, broad coverage; but no LLM-MoE tests.
- Writing Quality: ⭐⭐⭐⭐ Dense with mathematical formulas but clear derivations, each proposition references proof location; motivation-theory-algorithm-experiment flow is smooth.
- Value: ⭐⭐⭐⭐ Provides the first principled stabilization scheme for the emerging MoE-DRL direction, with potential for extension to large-model MoE fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](../../ICML2025/reinforcement_learning/mitigating_plasticity_loss_in_continual_reinforcement_learning_by_reducing_churn.md)
- [\[ICML 2026\] Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)

</div>

<!-- RELATED:END -->
