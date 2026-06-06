---
title: >-
  [Paper Note] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Plasticity loss] This paper formalizes the loss of plasticity in Mixture-of-Experts (MoE) policies during continual reinforcement learning (CRL) as a decline in the spectral entropy ef…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Plasticity loss"
  - "MoE policy"
  - "NTK spectrum"
  - "effective rank"
  - "Parseval regularization"
date: 2026-05-08
content_hash: 8e561a4d1df06f3c
---

# SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.04712](https://arxiv.org/abs/2605.04712)  
**Code**: Not publicly released  
**Area**: Reinforcement Learning / Mixture-of-Experts / Continual Learning  
**Keywords**: Plasticity loss, MoE policy, NTK spectrum, effective rank, Parseval regularization

## TL;DR
This paper formalizes the loss of plasticity in Mixture-of-Experts (MoE) policies during continual reinforcement learning (CRL) as a decline in the spectral entropy effective rank of the empirical NTK matrix. By employing Gauss-Newton and Kronecker decomposition, this is reduced to a computable proxy dependent only on the "expert feature Gram matrix." This proxy is then optimized using a one-line Parseval penalty (SPHERE), improving task success rates by 133% and 50% in MetaWorld and HumanoidBench CRL settings, respectively.

## Background & Motivation

**Background**: MoE architectures have expanded from LLMs into Deep Reinforcement Learning (DRL)—including multi-task robotic arms, humanoid, quadruped, and large-scale online RL—utilizing sparsely routed experts to increase policy capacity (e.g., Top-$K$ MoE, Dense-MoE, DS-MoE). Continual RL (CRL), where agents learn multiple tasks sequentially, is the ideal arena for leveraging MoE capacity.

**Limitations of Prior Work**: Empirically, MoE policies exhibit significant performance degradation in CRL; Willi et al. (2024) reported a "success rate collapse in late-stage tasks." This is a typical manifestation of plasticity loss: as training progresses, the ability to learn new skills from new data weakens. While explanations exist for dense networks (dormant neurons, representation spectral collapse, Hessian spectral collapse), the specific form of and countermeasures for plasticity loss in sparse, branched structures like MoE remain largely unexplored.

**Key Challenge**: The primary tool for characterizing plasticity—the eNTK matrix $\mathbf{K} = \mathbf{J}\mathbf{J}^\top \in \mathbb{R}^{N \times N}$—is prohibitively large for MoE. Forming $\mathbf{K}$ requires $O(N^2 P)$ time and $O(NP + N^2)$ memory, where $P$ includes all expert parameters, making it impossible to directly monitor or optimize. Consequently, plasticity loss is either unobservable or unoptimizable, necessitating a proxy that is both reflective of MoE plasticity and differentiable.

**Goal**: (1) Provide a formal definition of plasticity loss in MoE policies; (2) Reduce the intractable full eNTK effective rank to a differentiable small-matrix proxy; (3) Design a regularization term to perform spectral contraction on this proxy; (4) Validate performance on mainstream CRL benchmarks.

**Key Insight**: Starting from the gradient descent formula in function space $\Delta f = -\eta \mathbf{K} \nabla_f L$, the spectrum of $\mathbf{K}$ determines the directions in which gradients can propagate. When the spectrum collapses (a few eigenvalues dominate), $\mathbf{K}$ becomes a strong prior that locks updates into a few primary directions—this is the essence of plasticity loss. Therefore, "high plasticity = spectral isotropy," quantified by the spectral entropy effective rank $r_e(\mathbf{K}) = \exp(-\sum p_i \log p_i)$, where $p_i = \sigma_i / \sum_j \sigma_j$.

**Core Idea**: The $r_e(\mathbf{K})$ is approximated via block-diagonal decomposition (Gauss-Newton + layer independence + Kronecker decomposition, following K-FAC logic) to the "last-layer expert-weighted feature Gram matrix $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$." A Frobenius penalty is then used to push $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ towards $\frac{\mathrm{Tr}}{m}\mathbf{I}$ for "spectral contraction," which is proven to strictly increase $r_e(\mathbf{K})$.

## Method

### Overall Architecture

The input is the forward output of a Top-$K$ MoE actor within a PPO pipeline; the output is a differentiable regularization term added to the PPO loss. The derivation follows a chain of "layer-wise downward proxies": $r_e(\mathbf{K}) \to r_e(G^{\mathrm{GN}}) \to$ block-diagonal approximation $\to$ layer-wise expert blocks $r_e(\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell) \to$ Kronecker proxy condition number lower bound $\frac{k_\ell}{\kappa(\mathbf{A}^{\mathrm{exp}}_{\ell-1} \otimes \mathbf{G}^{\mathrm{exp}}_\ell)} \to$ spectral contraction on $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ (the last and most weight-heavy expert block). The final loss is: $\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \lambda^e \cdot \|\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}} - \tfrac{\mathrm{Tr}(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})}{m}\mathbf{I}_m\|_F^2$.

### Key Designs

1.  **From eNTK to Gauss-Newton + Block-Diagonal Approximation**:
    - **Function**: Converts the effective rank of $\mathbf{K} \in \mathbb{R}^{N \times N}$ into the effective rank of the parameter-space GN matrix $G^{\mathrm{GN}} = \tfrac{1}{N}\mathbf{J}^\top \mathbf{J} \in \mathbb{R}^{P \times P}$, fragmented into block-diagonal components corresponding to "gating + per-expert layers."
    - **Mechanism**: Utilizes the fact that $\mathbf{J}\mathbf{J}^\top$ and $\mathbf{J}^\top \mathbf{J}$ share non-zero spectra to obtain $r_e(\mathbf{K}) = r_e(G^{\mathrm{GN}})$. It then applies K-FAC's block-diagonal approximation (ignoring cross-layer and cross-gate-expert blocks) to yield $G^{\mathrm{GN}} \approx \bigoplus_\ell \mathbf{G}^{\mathrm{GN},\mathrm{g}}_\ell \oplus \bigoplus_\ell \mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell$. Using Lemma 4.1, the effective rank of a block-diagonal matrix is decomposed as $r_e(M) = \exp(H(\alpha) + \sum \alpha_b \log r_e(M_b))$, where $\alpha_b = \|M_b\|_*/\sum_m \|M_m\|_*$. Since gating parameters are few ($P^g \ll P^{\mathrm{exp}}$), gating blocks are treated as constants, and optimization focuses on expert blocks.
    - **Design Motivation**: The original $\mathbf{K}$ is uncomputable; this step reduces the "global rank" problem to a "per-layer expert block rank" problem, providing a theoretical basis for optimization through strict inequalities rather than heuristics.

2.  **Kronecker Proxy + Condition Number Bound**:
    - **Function**: Further factors each expert layer block $\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell$ into "weighted expert feature Gram $\mathbf{A}^{\mathrm{exp}}_{\ell-1} \otimes$ backpropagated gradient Gram $\mathbf{G}^{\mathrm{exp}}_\ell$" and provides an optimizable lower bound $r_e(\mathbf{G}^{\mathrm{GN},\mathrm{exp}}_\ell) \ge k_\ell / \kappa(\mathbf{A} \otimes \mathbf{G})$.
    - **Mechanism**: Under the layer-wise independence approximation, it uses K-FAC logic to weight the input $a^{\mathrm{exp}}_{e,\ell-1}(x_i)$ of each expert for each sample by Top-$K$ gating weights $h^{(K)}_{i,e}$, concatenating them across experts into $a_{\ell-1}(x_i) = [h^{(K)}_{i,1} a^{\mathrm{exp}}_{1,\ell-1}^\top | \dots | h^{(K)}_{i,E} a^{\mathrm{exp}}_{E,\ell-1}^\top]^\top$. This forms $\mathbf{A}^{\mathrm{exp}}_{\ell-1} = \tfrac{1}{N}\Phi_{\ell-1}^\top \Phi_{\ell-1}$. A gradient Gram $\mathbf{G}^{\mathrm{exp}}_\ell$ is constructed similarly. Since eigenvalues of a Kronecker matrix are products of factor eigenvalues, the condition numbers multiply, providing the $r_e \ge k_\ell / \kappa$ bound.
    - **Design Motivation**: $\mathbf{A}^{\mathrm{exp}}_{\ell-1}$ is a low-dimensional matrix (dimension $\sum_e d^{\mathrm{exp}}_{e,\ell-1}$, around several hundreds) obtainable during the forward pass without backpropagation. Its condition number lower bound serves as the proxy target. "Concatenation across experts" is the key—the off-diagonal blocks of the resulting Gram matrix capture cross-expert correlations, implicitly suppressing the collapse of features across different experts into the same direction.

3.  **SPHERE Parseval Penalty + Spectral Contraction Proof**:
    - **Function**: Defines $\mathcal{L}_{\mathrm{SPHERE}}(\mathbf{A}) = \|\mathbf{A} - \tfrac{\mathrm{Tr}(\mathbf{A})}{m}\mathbf{I}_m\|_F^2$ and proves it is spectrally contractive (eigenvalues shrink toward the mean), ensuring $\kappa(\mathbf{A})$ monotonically decreases and $r_e(\mathbf{K})$ monotonically increases.
    - **Mechanism**: Expanding $\mathcal{L}_{\mathrm{SPHERE}}$ yields $\|\mathbf{A}\|_F^2 - \tfrac{\mathrm{Tr}(\mathbf{A})^2}{m}$. Under SGD with $\eta \le \tfrac{1}{2}$, each eigenvalue moves as $\lambda_i \to (1-\beta)\lambda_i + \beta \bar\lambda$ toward the mean. This spectral contraction propagates to the Kronecker proxy via the Kronecker monotonicity lemma ($\kappa(A_{t+1} \otimes B) \le \kappa(A_t \otimes B)$) and finally to $r_e(\mathbf{K})$ via block-diagonal decomposition. In practice, this penalty is applied only to the actor's last expert layer where representation collapse is most severe.
    - **Design Motivation**: The authors sought a "provably strict plasticity-enhancing" regularizer rather than a heuristic one. The Parseval-like push-to-identity penalty satisfies the definition of spectral contraction, turning the relationship between the penalty and $r_e(\mathbf{K})$ into a theorem.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \lambda^e \cdot \mathcal{L}_{\mathrm{SPHERE}}(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})$. During each gradient update, $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ is calculated from the forward output. The Top-$K$ MoE uses $E = 10$ experts and $K = 2$. MetaWorld tasks are trained for $10^6$ env steps, and HumanoidBench tasks for $10^7$ steps.

## Key Experimental Results

### Main Results

| Benchmark | Method | Setup | Avg. Success Rate | Notes |
|-----------|------|------|-----------|------|
| MetaWorld CW10 | Top-$K$ MoE | CRL | baseline | Severe degradation in CRL |
| MetaWorld CW10 | + SPHERE | CRL | **+133%** | RL-CRL gap reduced by 52% |
| HumanoidBench H1 | Top-$K$ MoE | RL | baseline | Decay within single tasks |
| HumanoidBench H1 | + SPHERE | RL | **+36%** | Significant drift in long horizons ($10^7$ steps) |
| HumanoidBench H1 | Top-$K$ MoE | CRL | baseline | – |
| HumanoidBench H1 | + SPHERE | CRL | **+50%** | – |

### Ablation Study

| Configuration | HumanoidBench CRL Avg Success Rate | Description |
|------|---------------------------|------|
| w/o SPHERE | $0.36 \pm 0.08$ | Baseline without regularization |
| **w/ SPHERE** | $\mathbf{0.54 \pm 0.12}$ | Full method |
| Apply to all hidden expert layers | $0.42 \pm 0.07$ | Over-constrains shallow representation learning |
| Per-expert loss sum (no concatenation) | $0.40 \pm 0.08$ | Validates importance of cross-expert terms |
| Regulate $\mathbf{G}^{\mathrm{exp}}_{\mathrm{last}}$ instead of $\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}}$ | $0.43 \pm 0.09$ | Feature Gram gains are dominant |

### Key Findings

- **MoE requires plasticity intervention more than dense PPO**: Results show that while $r_e(\mathbf{K})$ drops in all architectures under CRL, MoE variants drop more sharply, echoing the intuition that gating sparsity amplifies representation collapse.
- **Cross-expert concatenation is a critical design**: Independent regularization per expert yields minimal gains (0.40), whereas joint regularization via concatenation reaches 0.54, proving that cross-expert correlation structures are the primary channel for plasticity loss.
- **$r_e(\mathbf{A}^{\mathrm{exp}}_{\mathrm{last}})$ correlates with $r_e(\mathbf{K})$ at 0.846 (Pearson)**: The effectiveness of the proxy is independently verified, going beyond a mere theoretical lower bound.
- **Different gain structures in MetaWorld vs. HumanoidBench**: The former benefits primarily in CRL (task-switching driven), while the latter benefits significantly even in single-task RL (where $10^7$ steps long horizons cause continuous distribution drift), suggesting plasticity loss is an intrinsic issue even without task switching.

## Highlights & Insights

- The primary contribution of this paper is providing a fuzzy phenomenon like "plasticity loss" with a mathematical definition and a provably effective, optimizable proxy. The chain from $r_e(\mathbf{K}) \to G^{\mathrm{GN}} \to$ block-diagonal $\to$ Kronecker proxy $\to$ Parseval penalty is rigorously formalized.
- "Gating-weighted concatenation across experts" is a sophisticated MoE-specific design. It bakes the sparsity of Top-$K$ routing directly into the Gram matrix, explicitly constraining experts to share a "distribute yet consistent" representation space rather than treating them as isolated modules.
- The 36% gain on single-task HumanoidBench tasks indicates that plasticity loss is not just a "continual learning" problem; distribution drift in long-horizon tasks is sufficient to trigger it. This suggests a need to re-evaluate training paradigms in long-horizon RL.
- The theoretical derivation holds for dense MoE and DS-MoE, allowing SPHERE to be generalized to recent "LLM-as-policy" settings.

## Limitations & Future Work

- K-FAC assumptions (block-diagonal and Kronecker independence) are verified empirically in the appendix but lack non-asymptotic error bounds; proxies may distort if experts are strongly coupled (e.g., shared experts).
- Experiments are limited to continuous control MoE policies; scaling to discrete actions or LLM-as-policy, where expert count and dimensions are orders of magnitude larger, may face memory/computation bottlenecks for feature Grams.
- $\lambda^e$ is a fixed hyperparameter; task-adaptive schedules were not explored. Strong spectral constraints might only be necessary in later CRL stages.
- Only the last layer is regularized based on the empirical observation that deep representations collapse most easily. It remains unclear if multiple layers should be regularized in deeper architectures.

## Related Work & Insights

- **vs. LayerNorm (Juliani & Ash 2024)**: While LN empirically mitigates plasticity loss by stabilizing forward values, it does not explicitly act on the NTK spectrum. SPHERE directly optimizes the plasticity metric with a provable direction.
- **vs. Parseval Regularization (Chung et al. 2024)**: Original PW regularizes weight matrices to be orthogonal in parameter space. SPHERE applies Parseval principles to expert feature Grams in representation space, adapted for cross-expert MoE structures.
- **vs. Spectral Normalization (Miyato et al. 2018)**: SN only controls the largest singular value; SPHERE maintains "full spectral uniformity," which is more direct for achieving an isotropic NTK spectrum.
- **vs. CBP (Dohare et al. 2024)**: CBP periodically re-initializes neurons (structural perturbation), which is complementary to the smooth gradient regularization provided by SPHERE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide an NTK-based formalization and optimizable proxy for MoE plasticity loss; elegant derivation chain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks (MetaWorld + HumanoidBench), two protocols (RL/CRL), 5 baselines, and 4 ablations; however, lacks LLM-MoE testing.
- Writing Quality: ⭐⭐⭐⭐ Math-heavy but clear derivation; Propositions are well-supported; smooth flow from motivation to theory and experiments.
- Value: ⭐⭐⭐⭐ Provides the first principled stabilization scheme for the emerging MoE-DRL field, with potential extensions to MoE LLM fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](dr_tulu_reinforcement_learning_with_evolving_rubrics_for_deep_research.md)
- [\[CVPR 2026\] Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning](../../CVPR2026/reinforcement_learning/cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](../../ICLR2026/reinforcement_learning/understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](../../ICLR2026/reinforcement_learning/spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)

</div>

<!-- RELATED:END -->
