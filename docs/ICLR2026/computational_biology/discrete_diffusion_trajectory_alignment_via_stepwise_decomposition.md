---
title: >-
  [Paper Note] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition
description: >-
  [ICLR2026][Computational Biology][discrete diffusion] This paper proposes SDPO (Stepwise Decomposition Preference Optimization), which decomposes the trajectory alignment problem of discrete diffusion models into stepwis…
tags:
  - "ICLR2026"
  - "Computational Biology"
  - "discrete diffusion"
  - "preference optimization"
  - "RLHF"
  - "trajectory alignment"
  - "stepwise decomposition"
date: 2026-05-08
content_hash: f7f258901ce902b1
---

# Discrete Diffusion Trajectory Alignment via Stepwise Decomposition

**Conference**: ICLR2026
**arXiv**: [2507.04832](https://arxiv.org/abs/2507.04832)  
**Code**: [hanjq17/discrete-diffusion-sdpo](https://github.com/hanjq17/discrete-diffusion-sdpo)  
**Area**: Medical Imaging
**Keywords**: discrete diffusion, preference optimization, RLHF, trajectory alignment, stepwise decomposition
**Authors**: Jiaqi Han, Austin Wang, Minkai Xu, Wenda Chu, Meihua Dang, Haotian Ye, Huayu Chen, Yisong Yue, Stefano Ermon (Stanford, Caltech, Tsinghua)

## TL;DR

This paper proposes SDPO (Stepwise Decomposition Preference Optimization), which decomposes the trajectory alignment problem of discrete diffusion models into stepwise posterior alignment subproblems, avoiding the difficulty of backpropagating gradients through the entire denoising chain. SDPO achieves significant improvements over existing methods across three tasks: DNA sequence design, protein inverse folding, and language modeling.

## Background & Motivation

Discrete diffusion models have demonstrated great potential in sequential data modeling, spanning domains such as DNA sequence design, protein inverse folding, and text generation. However, how to perform reward alignment on pretrained discrete diffusion models—analogous to RLHF for large language models—remains an unresolved core problem.

Primary challenges faced by existing methods:

1. **Gradient backpropagation is difficult due to discreteness**: The sampling chain of discrete diffusion models consists of sequence-level discrete random variables. Rewards are typically defined only on the final clean sequence $\mathbf{x}_0$, making gradient backpropagation through the entire denoising chain computationally expensive and unstable.
2. **Inaccurate likelihood computation**: When aligning the joint distribution $p_\theta(\mathbf{x}_{0:T})$ of the full trajectory, exact likelihood computation and reward evaluation are intractable, leading to suboptimal performance.
3. **High sampling cost for online RL**: The chain-based sampling of discrete diffusion makes online RL methods (e.g., DRAKES, diffu-GRPO) require expensive online sampling at every training step.

## Core Problem

How to design an efficient offline preference optimization method that enables exact likelihood and reward computation while being compatible with arbitrary reward functions, so as to effectively align pretrained discrete diffusion models?

## Method

### Stepwise Decomposition

Core idea: The alignment problem over the full diffusion trajectory $p_\theta(\mathbf{x}_{0:T})$ is decomposed into $T$ subproblems, each independently aligning the factorized posterior approximation $\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)$ at step $t$.

For each diffusion step $t$, the subproblem is:

$$\max_{\hat{p}_\theta} \mathbb{E}_{\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c})}[r(\mathbf{x}_0,\mathbf{c})] - \beta_t D_{\mathrm{KL}}[\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c}) \| \hat{p}_{\mathrm{ref}}(\mathbf{x}_0|\mathbf{x}_t,\mathbf{c})]$$

Advantages of this formulation:

- The posterior $\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)$ allows efficient and exact likelihood computation.
- Rewards are directly defined on the clean sequence $\mathbf{x}_0$, eliminating the need for biased estimation over intermediate variables.
- Compatible with arbitrary reward functions, not restricted to the Bradley-Terry model.

### Theoretical Equivalence (Theorem 4.1)

Key theorem: The joint distribution $p^*(\mathbf{x}_{0:T})$ induced by the optimal solutions $\{\hat{p}^*(\mathbf{x}_0|\mathbf{x}_t)\}_{t=1}^T$ of the per-step subproblems simultaneously constitutes the optimal solution to the original trajectory alignment objective, with the corresponding chain reward taking the form of a sum of stepwise rewards $\hat{r}(\mathbf{x}_{0:T})=\beta\sum_{t=1}^T r_t(\mathbf{x}_{t-1};\mathbf{x}_t)$.

### Generalized Stepwise Alignment (Distribution Matching)

Each per-step posterior is optimized via distribution matching, yielding a cross-entropy loss:

$$\mathcal{L}(\theta) = -\mathbb{E}_{t,\mathbf{c},\mathbf{x}_0,q(\mathbf{x}_t|\mathbf{x}_0)} \sum_{i=1}^N \left( \frac{\exp(r(\mathbf{x}_0^{(i)},\mathbf{c}))}{\sum_j \exp(r(\mathbf{x}_0^{(j)},\mathbf{c}))} \cdot \log \frac{\exp(\tilde{r}_\theta(\mathbf{x}_0^{(i)},\mathbf{x}_t^{(i)},\mathbf{c},\beta_t))}{\sum_j \exp(\tilde{r}_\theta(\mathbf{x}_0^{(j)},\mathbf{x}_t^{(j)},\mathbf{c},\beta_t))}\right)$$

where the implicit reward $\tilde{r}_\theta$ is based on the log-likelihood difference between the model and the reference model. $w(t)=1-\alpha_t$ is used to distribute the loss uniformly across tokens.

### Iterative Labeling

During training, new samples can be iteratively generated and labeled by a reward model, progressively improving the quality of training data and further enhancing performance.

## Key Experimental Results

### DNA Sequence Design

| Method | Pred-Activity ↑ | ATAC-Acc ↑ | 3-mer Corr ↑ | JASPAR Corr ↑ |
|------|:---:|:---:|:---:|:---:|
| DRAKES | 5.61 | 92.5% | 0.887 | 0.911 |
| diffu-GRPO | 5.86 | 33.0% | 0.783 | 0.903 |
| **SDPO** | **6.30** | **94.8%** | **0.900** | **0.936** |

- Predicted activity improves by **12.3%** over the strongest RL baseline DRAKES.
- High ATAC accuracy and natural sequence characteristics are simultaneously preserved.

### Protein Inverse Folding

| Method | Pred-ddG ↑ | %(ddG>0) ↑ | scRMSD ↓ | Success Rate ↑ |
|------|:---:|:---:|:---:|:---:|
| DRAKES | 1.095 | 86.4% | 0.918 | 78.6% |
| diffu-GRPO | 1.286 | 76.8% | 1.192 | 37.2% |
| **SDPO** | **1.400** | **87.1%** | 0.938 | 75.5% |

- Pred-ddG substantially outperforms all baselines; although diffu-GRPO achieves high ddG, its scRMSD degrades severely, indicating reward over-optimization.

### Language Modeling (LLaDA-8B-Instruct)

| Method | AlpacaEval LC ↑ | AlpacaEval WR ↑ | GSM8K ↑ | IFEval ↑ |
|------|:---:|:---:|:---:|:---:|
| Instruct | 10.6% | 6.8% | 78.6 | 52.9 |
| D2-DPO | 12.1% | 7.5% | 78.1 | 53.8 |
| **SDPO** | **14.2%** | **8.7%** | **81.2** | **55.1** |

- GSM8K improves from 78.6 to 81.2, surpassing LLaMA-3-8B post-RL training results.

### Training Efficiency

Average time per training step: DRAKES 6.02s, diffu-GRPO 1.51s, **SDPO only 0.77s** (no online sampling required).

## Highlights & Insights

1. **Theoretical elegance**: Trajectory alignment is decomposed into stepwise posterior alignment, with rigorous proof of equivalence between the two optimal solutions, yielding a theoretically sound framework.
2. **Strong generality**: Compatible with arbitrary reward functions, not restricted to simplified models such as Bradley-Terry; reduces to a special case of DPO when $N=2$ and the BT reward is used.
3. **Efficient offline training**: No online policy sampling is required; training speed is approximately 8× faster than DRAKES.
4. **Cross-domain validation**: Consistently outperforms baselines across three highly diverse domains: DNA design, protein engineering, and language modeling.
5. **Iterative labeling**: Sustained performance gains are achieved with far fewer additional annotations (15k for SDPO vs. 128k for DRAKES) while reaching higher rewards.

## Limitations & Future Work

1. **Success Rate is slightly lower than DRAKES on the protein task** (75.5% vs. 78.6%), indicating room for improvement along the scRMSD dimension.
2. **The factorized posterior approximation** ($\hat{p}_\theta(\mathbf{x}_0|\mathbf{x}_t)=\prod_i \hat{p}_\theta(\mathbf{x}_0^{(i)}|\mathbf{x}_t)$) ignores inter-token dependencies, which may introduce errors for long sequences.
3. **Language modeling experiments are limited in scale**: Validation is conducted only on LLaDA-8B and has not been extended to larger models or broader benchmarks.
4. **Bias in Monte Carlo estimation**: With finite $N$, the partition function estimate remains biased; results show performance saturating but not declining as $N$ increases from 25 to 100.
5. **Iterative labeling requires a callable reward model**, limiting applicability in settings where only preference data is available.

## Related Work & Insights

| Dimension | DPO / D2-DPO | DRAKES | diffu-GRPO | SDPO |
|------|:---:|:---:|:---:|:---:|
| Online/Offline | Offline | Online | Online | Offline |
| Reward Type | Bradley-Terry | Arbitrary | Arbitrary | Arbitrary |
| Optimization Granularity | Trajectory-level | Trajectory-level | Trajectory-level | **Stepwise** |
| Training Efficiency | High | Low | Medium | **High** |
| Likelihood Exactness | Approximate | Approximate | Approximate | **Exact** |
| Reward Over-optimization Risk | Low | Medium | **High** | Low |

- Compared to D2-DPO/VRPO: SDPO does not rely on the BT model assumption, supports arbitrary rewards, and achieves superior performance.
- Compared to DRAKES: SDPO is trained offline with ~8× higher efficiency and achieves 12.3% higher predicted activity on DNA tasks.
- Compared to diffu-GRPO: SDPO does not suffer from reward over-optimization (diffu-GRPO achieves only 37.2% Success Rate on the protein task).
- Compared to inference-time guidance (CG/SMC/TDS): SDPO optimizes at training time, incurring no additional overhead at sampling time.

### Inspirations and Connections

1. **Transferability of the stepwise decomposition idea**: The decomposition strategy is broadly applicable and may transfer to alignment of continuous diffusion models or even other multi-step decision optimization settings.
2. **A unified perspective with DPO**: SDPO reduces to DPO under $N=2$ with BT reward, providing a more general framework for applying DPO to diffusion models.
3. **Biological sequence design**: Significant improvements on DNA and protein tasks suggest the method is particularly well-suited for bioengineering applications with well-defined sequence-level rewards.
4. **Preference optimization for large language diffusion models**: Validation on LLaDA provides a viable pathway for post-training alignment of discrete diffusion language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Stepwise decomposition combined with an equivalence proof offers a fundamentally new perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three highly diverse domains with extensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical derivations are clear; experimental organization is systematic)
- Value: ⭐⭐⭐⭐⭐ (Establishes a new benchmark for discrete diffusion model alignment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [\[ICLR 2026\] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[ICML 2026\] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](../../ICML2026/computational_biology/td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)

</div>

<!-- RELATED:END -->
