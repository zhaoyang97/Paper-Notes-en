---
title: >-
  [Paper Note] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases
description: >-
  [ICML 2026][Others][small-vs-large gap] This paper systematically characterizes and explains the "small-vs-large gap" phenomenon, where repeating smaller datasets leads to faster convergence than using larger datasets. The authors prove that this acceleration cannot be explained by the CSQ-SQ gap, gradient variance reduction, or input distribution bias. By a
tags:
  - ICML 2026
  - Others
  - small-vs-large gap
date: 2026-05-08
content_hash: 2ccd9cfe282cffe6
---
# Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases

**Conference**: ICML 2026  
**arXiv**: [2605.20314](https://arxiv.org/abs/2605.20314)  
**Code**: TBD  
**Area**: Optimization / Feature Learning / Training Dynamics  
**Keywords**: small-vs-large gap, sampling bias, inter-layer norm, feature learning, repeated training

## TL;DR
This paper systematically characterizes and explains the "small-vs-large gap" phenomenon, where repeating smaller datasets leads to faster convergence than using larger datasets. The authors prove that this acceleration cannot be explained by the CSQ-SQ gap, gradient variance reduction, or input distribution bias. By analyzing a $2$-layer quadratic MLP on $2$-sparse parity, they derive a closed-form step bound $T = O((Nd)^{1/4} \log(d/\varepsilon))$. Through intervention experiments—including random labels, initialization scaling, and inter-layer learning rates—they verify the core mechanism: the $O(N^{-1/2})$ sampling bias inherent in small datasets accelerates first-layer feature learning by driving faster growth of the second-layer norm.

## Background & Motivation
**Background**: A core tenet of deep learning is "more data is better," supported by scaling laws and classical generalization theory. However, recent works (Charton & Kempe 2024; Zucchet et al. 2025; Kopiczko et al. 2026) have identified an anomaly: under a fixed compute budget (steps $\times$ batch size), repeatedly training on a smaller dataset can achieve better test performance than online training with fresh samples from a larger dataset. In tasks like sparse parity, this compute saving can reach two orders of magnitude. This phenomenon is termed the "small-vs-large gap."

**Limitations of Prior Work**: Existing explanations are insufficient. (1) Dandi et al. 2024 suggested that repeating batches upgrades SGD from CSQ to the stronger SQ algorithm, but this only applies to tasks where an SQ-CSQ gap exists (e.g., single-index models) and fails for discrete tasks like sparse parity or modular addition where SQ = CSQ. Furthermore, the gap persists in full-batch GD where data is always "repeated." (2) Gradient variance reduction (Kotha 2025) cannot explain the full-batch setting as GD has no stochastic variance. (3) The input distribution bias theory (Cornacchia et al. 2025) provides Fourier coefficients of $O(N^{-k/2})$, which vanish for sparsity $k=6$. Empirically, the gap remains even when removing input bias (forcing $\hat{\mathbb{E}}[x] = 0$).

**Key Challenge**: The phenomenon is universal across mini-batch/full-batch, SIM/parity/ICL/mod-add, and MLP/Transformer architectures, yet existing theories fail to cover at least one of these settings. A unified mechanism across settings is required.

**Goal**: (1) Systematically verify the gap across a wide matrix of tasks, architectures, and optimizers; (2) Rule out the three candidate explanations; (3) Propose a new mechanism with an analyzable model and closed-form bounds; (4) Design intervention experiments to validate the mechanism.

**Key Insight**: In a $2$-layer MLP learning parity, the first layer is the feature learning layer, while the second-layer norm $|a|$ directly controls the effective gradient of the first layer via $\nabla_w L$. Anything that causes $|a|$ to grow earlier will accelerate first-layer feature learning. The authors hypothesize that the "sampling bias" of small datasets is precisely such a force.

**Core Idea**: The essence of the small-vs-large gap is not "seeing less data" or "repetition," but rather that the variance of the empirical moment $\hat M = \frac{1}{N}\sum y x x^\top$ from the population moment in small datasets is $\Theta(N^{-1/2})$, which is significantly larger than $1/d$. This pushes the growth of the second-layer norm earlier in training, indirectly accelerating feature learning. This is a passively induced inter-layer growth imbalance, equivalent to an implicit inter-layer learning rate schedule.

## Method
The methodology consists of two parts: (a) establishing a step complexity theorem on an analyzable toy model; (b) designing intervention experiments using inter-layer norm growth as an observable signal to verify the mechanism.

### Overall Architecture
- **Task Suite**: Single-index models (SIM, Hermite link), $(d,k)$-sparse parity, in-context linear regression, and $(N,p)$-modular addition. Optimizers include mini-batch SGD and full-batch GD. Models include $2$-layer MLPs (ReLU, no residual) and $2$-layer Transformers (optional QK normalization).
- **Data Strategy**: Beyond standard single-set repetition, a $T$-phase training strategy is introduced (generalizing Charton & Kempe 2024), where phase $i$ trains on a subset $\mathcal{S}_i \subset \mathcal{S}_{i+1}$. The heuristic is to achieve non-trivial training performance quickly on a small subset before ensuring generalization on a larger set.
- **Analytical Model**: $f(x) = a \sigma(w^\top x) - 1$ with $\sigma(z) = \frac{1}{2}z^2$, correlation loss $\ell(y,y') = -yy'$, and projected updates: $a$ is clipped to $[-1, 1]$ and $w$ is normalized to the unit sphere at each step. For $2$-sparse parity, $w^\star$ is non-zero only in the first two dimensions.

### Key Designs

**1. Closed-form Step Bounds for 2-phase Training (Theorem 1): Quantifying Small-Data Acceleration**

To prove acceleration, the authors calculate the steps for a toy model. They prove that for $d \le N \le d^2$, $2$-phase training requires only $O((Nd)^{1/4} \log(d/\varepsilon))$ steps to converge $w$ to $\|\hat w - w^\star\|_2 \lesssim \sqrt{\varepsilon}$, which is much smaller than the $O(m^{1/2}\log(d/\varepsilon))$ required for population training with width $m \gg d^2$. In Phase 1, projected GD on a subset of size $N$ runs until $|a| \ge a_\star$. The key is that the gradient magnitude of $a$ is determined by $q^{(t)} = (w^{(t)})^\top \hat M w^{(t)}$. The anti-concentration of $\hat M$ gives $|q^{(t)}| = \Theta(N^{-1/2})$, which is much larger than the population gradient $\Theta(1/d)$. Thus, $a$ grows at a rate of $N^{-1/2}$, reaching $a_\star$ in $T_1 \lesssim a_\star \sqrt{N}/\eta$ steps. Phase 2 switches to population gradients for power iteration, with a convergence rate controlled by $\eta a_\star$, requiring $T_2 \lesssim \frac{2}{\eta a_\star}\log(d/\varepsilon)$. Optimizing for $a_\star$ yields the total rate of $(Nd)^{1/4}$. This theorem isolates the mechanism: $T_1$ is driven by sampling bias (independent of labels), while $T_2$ depends on the magnitude of $a_\star$.

**2. Random Label Verification (Corollary 2 + Experiments): Decoupling Sampling Bias from Task Signals**

If the acceleration stems from task signals or input distribution bias, training with random labels should yield no acceleration. Replacing Phase 1 of Theorem 1 with training on a small dataset using uniform random $\pm 1$ labels, the theory predicts $|a|$ still achieves an early growth rate of $\Theta(N^{-1/2})$, with step complexity $T = O(\sqrt{N}/(\eta\sqrt{d}) + \sqrt{d}\log(d/\varepsilon)/\eta)$. Experiments on MLP-parity, MLP-SIM, and Transformer-mod addition show that the random label curve (green) almost overlaps with the true-label small-set curve (yellow), and both are significantly faster than population training (blue). The observed $\|a\|_2 / \|W\|_F$ ratio also rises faster under small/random label sets, proving that "sampling bias $\to$ fast second-layer growth" is the critical path.

**3. Inter-layer Initialization and Learning Rate Interventions (Section 5.2): Eliminating the Gap**

If the mechanism is true, manually reproducing the imbalance should eliminate the gap on large datasets. The authors apply three interventions: increasing the initial scale of the second layer $|a^{(0)}|$; using a higher learning rate $\eta_a$ for the second layer; and observing if QK normalization in Transformers plays a similar role. Any of these interventions significantly reduces or eliminates the gap of large datasets relative to small ones. The theoretical basis is that Phase 2 convergence is proportional to $\eta a_\star$ (the relative inter-layer growth speed). This elevates the phenomenon from an empirical observation to a parameterizable optimization effect.

### Loss & Training
All MLP/Transformers use default PyTorch initialization ($W_{ij} \sim \text{Unif}[-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}}] $), SGD for MLPs, and AdamW for Transformers. Learning rates are swept independently for each setting. Performance is averaged over multiple seeds at a fixed compute budget (batch $\times$ steps).

## Key Experimental Results

### Main Results

| Task / Setting | Dataset Size Comparison | Observed Compute Saving | Note |
|:---|:---|:---|:---|
| $(20,6)$-sparse parity (mini-batch SGD, 2-layer Transformer) | Small set vs. Online | Yellow converges much earlier than blue | Fig.1, universal across tasks |
| $(20,6)$-sparse parity (full-batch GD, 2-layer MLP) | $N = 2^{14}$ vs. $N = 2^{20}$ | ~100$\times$ compute acceleration | Fig.2, refutes SQ-CSQ & variance hypotheses |
| SIM ($d=40$, full-batch GD) | Small set vs. Population | Faster at every step | Same as Fig.2 |
| ICL Linear Regression / Mod Addition | Multi-phase training | Significant acceleration | Fig.1, cross-architecture |

### Ablation Study

| Intervention | Key Metric | Conclusion |
|:---|:---|:---|
| Forching $\hat{\mathbb{E}}[x]=0$, $\hat{\mathbb{E}}[y]=0$ | Small-set remains fast | Input bias is not the primary cause |
| Injecting small-set bias into large set ($m \in \{4..12\}$) | Only matched at $m=5$ | Bias magnitude must be unlearnable to match |
| Phase 1 with random labels on small set | Acceleration matches true labels | Labels are irrelevant; sampling bias dominates |
| Scaling up 2nd layer init / Inter-layer $\eta_a$ | Gap reduced or eliminated | Directly verifies inter-layer growth mechanism |
| Transformer QK Norm toggle | Nuanced effect | Implicitly regulates inter-layer dynamics |

### Key Findings
- The gap persists under full-batch GD, providing the cleanest evidence against "stochastic-driven acceleration" hypotheses.
- The $\|a\|_2 / \|W\|_F$ ratio is an observable proxy for the mechanism: small data, random labels, and large second-layer initialization all correspond to faster ratio increases.
- Multi-phase training requires only an initial small subset; subsequent larger subsets ensure generalization, providing a template for training schedules.
- The optimal choice of $a_\star$ leads to $(Nd)^{1/4}$ complexity, suggesting that for reasoning tasks (inherently discrete/combinatorial), repeating small datasets may be more efficient than scaling data.

## Highlights & Insights
- **Link between Data and Optimization**: The "small data acceleration = implicit inter-layer learning rate" perspective unifies data strategies and optimizer strategies under the variable of "relative inter-layer growth speed."
- **Analyzable Toy Model**: Using $2$-sparse parity with quadratic activation allows for closed-form upper bounds for both phases, where theoretical predictions precisely match the observable proxy (norm ratio).
- **Utility of Random Labels**: Training on random labels can serve as an "inter-layer warm-up," suggesting that supposedly "meaningless" pre-training steps (e.g., noise batches) may have structural benefits.

## Limitations & Future Work
- The theory is restricted to $2$-sparse parity, $2$-layer quadratic MLPs, correlation loss, and projected updates. Extension to ReLU, deeper networks, and cross-entropy remains open.
- Experiments are focused on synthetic tasks. While citing LLM post-training observations as evidence, the authors did not systematically replicate these on full-scale LLM/ViT models.
- The mechanism focuses on $2$-layer imbalances; the role of relative growth in deeper networks and its interaction with LayerNorm/RMSNorm requires dedicated study.
- The risks of overfitting in over-parameterized or small models when repeating data were not deeply explored.

## Related Work & Insights
- **vs. Dandi et al. (2024) / Lee et al. (2025)**: Their SQ-CSQ theory explains SIM for batch SGD but is refuted by full-batch GD and discrete task counterexamples provided here.
- **vs. Kotha et al. (2025)**: While variance reduction explains mini-batch settings, the continued acceleration in full-batch GD proves variance is not the only factor.
- **vs. Cornacchia et al. (2025)**: Their $O(\eta^k)$ bias signal is significantly smaller than the $O(N^{-1/2})$ sampling bias; this was directly refuted by bias injection experiments.
- **vs. µP / Tensor Programs**: While µP explicitly controls inter-layer growth via parameterization, this work shows that data scale itself achieves a similar effect via sampling bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies the small-vs-large gap anomaly under one mechanism and refutes three existing theories.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive intervention matrix; however, lacks large-scale LLM verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical chain; the "refutation-confirmation" structure in Sections 4 and 5 is well-executed.
- **Value**: ⭐⭐⭐⭐ Provides a new intuition for training (small data + many epochs is not just a fallback) and inter-layer optimizer design.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](../../ACL2025/others/fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)

</div>

<!-- RELATED:END -->
