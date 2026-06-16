---
title: >-
  [Paper Note] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases
description: >-
  [ICML 2026][Others][small-vs-large gap] This paper systematically characterizes and explains the "small-vs-large gap" phenomenon—where repeating smaller datasets results in faster convergence than training on large datasets. The authors prove that this acceleration cannot be explained by existing theories such as CSQ-SQ gaps, gradient variance reduction, or
tags:
  - ICML 2026
  - Others
  - small-vs-large gap
date: 2026-05-08
content_hash: da8b9a46afcda431
---
# Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases

**Conference**: ICML 2026  
**arXiv**: [2605.20314](https://arxiv.org/abs/2605.20314)  
**Code**: TBD  
**Area**: Optimization / Feature Learning / Training Dynamics  
**Keywords**: small-vs-large gap, sampling bias, inter-layer norm, feature learning, repeated training

## TL;DR
This paper systematically characterizes and explains the "small-vs-large gap" phenomenon—where repeating smaller datasets results in faster convergence than training on large datasets. The authors prove that this acceleration cannot be explained by existing theories such as CSQ-SQ gaps, gradient variance reduction, or input distribution bias. Instead, using a 2-layer quadratic MLP on 2-sparse parity, they derive a closed-form step bound $T = O((Nd)^{1/4} \log(d/\varepsilon))$. Through intervention experiments including random labels, initialization scaling, and inter-layer learning rates, they verify that the acceleration is driven by the $O(N^{-1/2})$ sampling bias inherent in small datasets, which speeds up first-layer feature learning by accelerating the growth of the second-layer norm.

## Background & Motivation
**Background**: A mainstream tenet of deep learning is "the more data, the better," supported by scaling laws and classical generalization theory. However, recent works (Charton & Kempe 2024; Zucchet et al. 2025; Kopiczko et al. 2026) have identified an anomaly: under a fixed compute budget (steps × batch size), repeatedly training on a small dataset can achieve better test performance than online training on a large "fresh-sample" dataset. In tasks like sparse parity, this compute saving can reach two orders of magnitude. The authors refer to this phenomenon as the "small-vs-large gap."

**Limitations of Prior Work**: Existing explanations are insufficient. (1) Dandi et al. (2024) proposed that repeated batches upgrade SGD from CSQ to SQ algorithms, but this only applies to tasks where such a gap exists (e.g., single-index models). On discrete tasks like sparse parity or modular addition where SQ = CSQ, the theory fails. Furthermore, the gap persists in full-batch GD where data is "repeated" in both cases. (2) Gradient variance reduction (Kotha 2025) cannot explain the full-batch setting since GD has no stochastic variance. (3) Cornacchia et al. (2025) proposed "input distribution bias," but their Fourier coefficients are $O(N^{-k/2})$, which vanishes for sparsity $k=6$; moreover, empirical evidence shows that removing input bias (forcing $\hat{\mathbb{E}}[x] = 0$) does not eliminate the gap.

**Key Challenge**: The phenomenon is universal across configurations (mini-batch/full-batch, SIM/parity/ICL/mod-add, MLP/Transformer), but no existing theory explains all settings. A unified mechanism across these configurations is required.

**Goal**: (1) Systematically verify the gap across a broad matrix of tasks/architectures/optimizers; (2) Exclude previous candidate explanations; (3) Propose a new mechanism with an analyzable model and closed-form bounds; (4) Design intervention experiments to verify the mechanism.

**Key Insight**: In a 2-layer MLP learning parity, the first layer (input layer) is the feature learning layer, while the second-layer norm $|a|$ directly controls the effective gradient of the first layer by multiplying $\nabla_w L$. Any force that causes $|a|$ to grow earlier will accelerate first-layer feature learning. The authors hypothesize that the "sampling bias" of small datasets is exactly such a force.

**Core Idea**: The essence of the small-vs-large gap is not "seeing less data" or "repetition," but rather that the variance of the small dataset's empirical moment $\hat M = \frac{1}{N}\sum y x x^\top$ from the population moment is $\Theta(N^{-1/2})$. This is much larger than $1/d$, which pushes the second-layer norm faster in early training, indirectly accelerating first-layer feature learning. This constitutes a passively induced inter-layer growth imbalance, equivalent to an implicit inter-layer learning rate schedule.

## Method
The methodology consists of two components: (a) providing a step complexity theorem on an analyzable toy model; (b) designing intervention experiments using inter-layer norm growth as an observable signal to verify the mechanism.

### Overall Architecture
- **Task Set**: Single-index models (SIM, Hermite link), $(d,k)$-sparse parity, in-context linear regression, and $(N,p)$-modular addition. Optimizers include mini-batch SGD and full-batch GD. Models used are 2-layer MLPs (ReLU, no residual) and 2-layer Transformers (optional QK normalization).
- **Data Strategy**: In addition to standard single-set repetition, $T$-phase training is introduced (extending Charton & Kempe 2024), where phase $i$ trains on subset $\mathcal{S}_i \subset \mathcal{S}_{i+1}$. The heuristic is to achieve non-trivial training performance on small subsets early and use large subsets in the final stage for generalization.
- **Analysis Model**: $f(x) = a \sigma(w^\top x) - 1$ where $\sigma(z) = \frac{1}{2}z^2$ and correlation loss $\ell(y,y') = -yy'$. Update with projection: $a$ is clipped to $[-1, 1]$, and $w$ is normalized to the unit ball at each step. For 2-sparse parity, $w^\star$ is non-zero only in the first two dimensions.

### Key Designs

**1. Closed-form Bounds for 2-phase Training (Theorem 1): Quantifying Small Data Acceleration**

To prove acceleration exists, steps must be calculated for a toy model. The paper proves that for $d \le N \le d^2$, 2-phase training requires only $O((Nd)^{1/4} \log(d/\varepsilon))$ steps for $w$ to converge to $\|\hat w - w^\star\|_2 \lesssim \sqrt{\varepsilon}$. This is much smaller than the $O(m^{1/2}\log(d/\varepsilon))$ required for population training with width $m \gg d^2$. In Phase 1, projected GD on a subset of size $N$ runs until $|a| \ge a_\star$. The magnitude of the gradient of $a$ is determined by $q^{(t)} = (w^{(t)})^\top \hat M w^{(t)}$. The anti-concentration of $\hat M = \frac{1}{N}\sum y x x^\top$ yields $|q^{(t)}| = \Theta(N^{-1/2})$, which is much larger than the population gradient $\Theta(1/d)$. Thus $|a|$ grows at a rate of $N^{-1/2}$, reaching $a_\star$ in $T_1 \lesssim a_\star \sqrt{N}/\eta$ steps. Phase 2 switches to population gradients for power iteration on $M$, with convergence controlled by $\eta a_\star$, yielding $T_2 \lesssim \frac{2}{\eta a_\star}\log(d/\varepsilon)$. Optimizing for $a_\star$ gives the $(Nd)^{1/4}$ rate. The theorem decomposes the mechanism: $T_1$ is driven by sampling bias (independent of label signal), and $T_2$ depends solely on the magnitude of $a_\star$.

**2. Random Label Verification (Corollary 2 + Experiments): Separating Bias from Signal**

If acceleration stems from task signals or input distribution bias, training with random labels should not show acceleration. This is the cleanest differential experiment. Replacing Phase 1 of Theorem 1 with "training on a small dataset with uniformly sampled $\pm 1$ random labels," the theory predicts $|a|$ still grows at $\Theta(N^{-1/2})$, leading to $T = O(\sqrt{N}/(\eta\sqrt{d}) + \sqrt{d}\log(d/\varepsilon)/\eta)$. Experiments on MLP-parity, MLP-SIM, and Transformer-modular addition show that curves for random-label Phase 1 (green) nearly overlap with small-set true labels (yellow), and both are faster than large-set training (blue). The measured $\|a\|_2 / \|W\|_F$ ratio rises faster under small/random labels. This proves "sampling bias $\rightarrow$ rapid second-layer growth" is the key path, while the label signal is secondary.

**3. Inter-layer Initialization and LR Intervention (Section 5.2): Eliminating the Gap**

If the mechanism is true, artificially reproducing the "inter-layer growth imbalance" caused by sampling bias should eliminate the gap on large datasets. Three interventions are tested on MLPs and Transformers: increasing the second-layer initialization scale $|a^{(0)}|$; using a higher inter-layer learning rate $\eta_a$ for the second layer; and observing QK normalization in Transformers. Results show that any of these interventions significantly reduces or eliminates the gap of large datasets relative to small ones. This suggests that these engineering techniques are equivalent to "using small datasets" by aligning the relative inter-layer growth speed, elevating the phenomenon from observation to a parameter-controlled optimization effect.

### Training Strategy
All MLP/Transformer models use standard initialization ($W_{ij} \sim \text{Unif}[-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}}]$). SGD is used for MLPs and AdamW for Transformers, with independent LR sweeps for each setting. Performance is averaged over multiple random seeds at fixed compute = batch × steps to represent success probability.

## Key Experimental Results

### Main Results
| Task / Setup | Dataset Comparison | Observed Compute Saving | Description |
|--------------|--------------------|-------------------------|-------------|
| (20,6)-sparse parity (mini-batch SGD, 2-layer Transformer) | Small set vs. Online | Yellow converges earlier than Blue | Fig. 1, universal phenomenon |
| (20,6)-sparse parity (full-batch GD, 2-layer MLP) | $N = 2^{14}$ vs. $N = 2^{20}$ | ~100x compute acceleration | Fig. 2, refutes SQ-CSQ and variance hypothesis |
| SIM ($d=40$, full-batch GD) | Small set vs. Population | Faster at every step | Similar to Fig. 2 |
| ICL Linear Regression / Mod Addition (Transformer) | Multi-phase training | Significant acceleration | Fig. 1, cross-architecture |

### Ablation Study
| Intervention | Key Metric | Conclusion |
|--------------|------------|------------|
| Forcing $\hat{\mathbb{E}}[x]=0$, $\hat{\mathbb{E}}[y]=0$ | Small-set remains fast | Input bias is not the primary cause |
| Injecting small-set bias into large-set ($m \in \{4..12\}$) | Only matched at $m=5$ | Bias must be small enough to stay unlearned, matching Cornacchia's theory |
| Phase 1 with random labels on small set | Identical acceleration magnitude | Label signal irrelevant; sampling bias dominant |
| Scaling up 2nd layer init / Inter-layer $\eta_a$ | Gap significantly reduced/disappeared | Directly verifies inter-layer growth mechanism |
| Transformer QK Norm toggle | Nuanced effect | Implicitly regulates inter-layer dynamics |

### Key Findings
- The gap persists under full-batch GD, providing evidence against all "stochasticity-driven" acceleration hypotheses.
- The ratio $\|a\|_2 / \|W\|_F$ serves as an observable proxy for the mechanism: small data, random labels, and large second-layer initialization all correspond to a faster rise in this ratio.
- Multi-phase training requires small subsets only in the early stage; switching to large subsets later preserves generalization, offering a simple training schedule template.
- The optimal choice of $a_\star$ leads to $(Nd)^{1/4}$ complexity, suggesting that for reasoning tasks (inherently discrete/combinatorial), repeating small data might be more efficient than scaling data.

## Highlights & Insights
- "Small data acceleration = implicit inter-layer learning rate" provides a transferable perspective, unifying data strategies and optimizer strategies through "relative inter-layer growth speed."
- Using 2-sparse parity with quadratic activation as a toy model allows for calculating closed-form upper bounds for both phases, providing a paradigm for analyzing training dynamics.
- Training with random labels can act as "inter-layer warming," suggesting that "seemingly meaningless" warm-up steps (e.g., using noise batches or random labels) may have significant optimization utility.

## Limitations & Future Work
- The theory currently covers 2-sparse parity with 2-layer quadratic MLPs, correlation loss, and projected updates. Extension to ReLU, deeper networks, and cross-entropy remains open.
- Experiments focus on synthetic tasks (parity/SIM/ICL/mod-add). While leveraging LLM post-training observations from Kopiczko (2026), the authors have not systematically reproduced this on large-scale LLMs/ViTs.
- The mechanism centers on inter-layer imbalance in 2-layer networks; whether this remains critical for deeper networks or how it interacts with LayerNorm/RMSNorm/QK Norm requires further research.
- Risks of overfitting in overparameterized or small models when repeating small data were not explored deeply.

## Related Work & Insights
- **vs. Dandi et al. 2024 / Lee et al. 2025 (CSQ $\rightarrow$ SQ)**: They explained acceleration for batch SGD on SIM, but this is challenged by full-batch GD results on discrete tasks in this paper.
- **vs. Kotha et al. 2025 (Gradient Variance Reduction)**: This explains mini-batch effects, but the persistence of acceleration in full-batch GD proves variance is not the sole factor.
- **vs. Cornacchia et al. 2025 (Input Distribution Bias)**: They propose $O(\eta^k)$ signals, which are an order of magnitude smaller than the $O(N^{-1/2})$ sampling bias found here.
- **vs. Charton & Kempe 2024 / Kopiczko 2026**: These works provided heuristic schedules. This paper traces their effectiveness to a unified inter-layer growth mechanism.
- **vs. µP / Tensor Programs (Yang & Hu 2020)**: Those works control inter-layer growth via parameterization; this paper shows data scale performs a similar role via sampling bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies the "small-vs-large gap" anomaly into a single mechanism and refutes three previous theories.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid intervention design and task coverage, though lacks large-scale LLM validation by the authors.
- **Writing Quality**: ⭐⭐⭐⭐ Clear argumentation chain; Section 4 (falsification) and Section 5 (verification) form a strong structure.
- **Value**: ⭐⭐⭐⭐ Provides new training intuitions and a new dimension for thinking about inter-layer learning rates and initialization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Faster Path to Continual Learning](../../CVPR2026/others/a_faster_path_to_continual_learning.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](../../ACL2025/others/fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
