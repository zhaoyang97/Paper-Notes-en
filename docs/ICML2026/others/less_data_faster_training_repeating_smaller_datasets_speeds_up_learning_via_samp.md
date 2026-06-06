---
title: >-
  [Paper Note] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases
description: >-
  [ICML 2026][small-vs-large gap] This paper systematically characterizes and explains the "small-vs-large gap" phenomenon, where repeating smaller datasets leads to faster convergence than training on larger datasets. The…
tags:
  - "ICML 2026"
  - "small-vs-large gap"
  - "sampling bias"
  - "inter-layer norm"
  - "feature learning"
  - "multi-epoch training"
date: 2026-05-08
content_hash: cd97b958f478de0d
---

# Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases

**Conference**: ICML 2026  
**arXiv**: [2605.20314](https://arxiv.org/abs/2605.20314)  
**Code**: TBD  
**Area**: Optimization / Feature Learning / Training Dynamics  
**Keywords**: small-vs-large gap, sampling bias, inter-layer norm, feature learning, multi-epoch training

## TL;DR
This paper systematically characterizes and explains the "small-vs-large gap" phenomenon, where repeating smaller datasets leads to faster convergence than training on larger datasets. The authors prove that this acceleration cannot be explained by the CSQ-SQ gap, gradient variance reduction, or input distribution bias. By providing a closed-form step complexity bound $T = O((Nd)^{1/4} \log(d/\varepsilon))$ for a 2-layer MLP with quadratic activation on 2-sparse parity and through intervention experiments (random labels, initialization scaling, layer-wise learning rates), they demonstrate that the true driver is the $O(N^{-1/2})$ sampling bias inherent in small datasets, which accelerates first-layer feature learning by speeding up second-layer norm growth.

## Background & Motivation
**Background**: The mainstream doctrine in deep learning is "more data is better," supported by scaling laws and classical generalization theory. However, recent work (Charton & Kempe 2024; Zucchet et al. 2025; Kopiczko et al. 2026) has identified a counter-intuitive phenomenon: under a fixed compute budget (steps $\times$ batch size), repeatedly training on a small dataset can achieve better test performance than fresh-sample online training on a large dataset. This compute saving can reach two orders of magnitude on sparse parity tasks, a phenomenon termed the "small-vs-large gap."

**Limitations of Prior Work**: Existing explanations are insufficient. (1) Dandi et al. 2024 proposed that "repeated batches upgrade SGD from a CSQ algorithm to a stronger SQ algorithm," but this only applies to tasks with an SQ-CSQ lower bound gap (like single-index models) and fails for discrete tasks (sparse parity, modular addition) or full-batch GD where the gap persists. (2) Gradient variance reduction (Kotha 2025) cannot explain the full-batch setting where no stochastic variance exists. (3) The "input distribution bias" theory (Cornacchia et al. 2025) gives Fourier coefficients of $O(N^{-k/2})$, which are negligible for sparsity $k=6$; furthermore, the gap persists even after removing input bias ($\hat{\mathbb{E}}[x] = 0$).

**Key Challenge**: The phenomenon is universal (mini-batch/full-batch, SIM/parity/ICL/mod-add, MLP/Transformer), but existing theories fail to explain at least one of these settings. A unified mechanism across all settings is required.

**Goal**: (1) Systematically validate the gap across a broad matrix of tasks/architectures/optimizers; (2) Exclude three classes of candidate explanations; (3) Propose a new mechanism with an analyzable toy model and step complexity bounds; (4) Design intervention experiments to verify the mechanism.

**Key Insight**: In a 2-layer MLP learning parity, only the first layer (input layer) performs feature learning, while the second-layer norm $|a|$ directly controls the effective gradient of the first layer by multiplying into $\nabla_w L$. Any force that causes $|a|$ to rise early will accelerate first-layer feature learning. The authors hypothesize that the "sampling bias" of small datasets is exactly such a force.

**Core Idea**: The essence of the small-vs-large gap is not "seeing less data" or "repetition," but that the variance of empirical moments $\hat M = \frac{1}{N}\sum y x x^\top$ from population moments in small datasets is $\Theta(N^{-1/2})$, which is much larger than $1/d$. This pushes the second-layer norm faster in early training, indirectly accelerating first-layer feature learning—an induced inter-layer growth imbalance equivalent to an implicit layer-wise learning rate schedule.

## Method
The methodology consists of two parts: (a) providing a step complexity theorem on an analyzable toy model; (b) designing intervention experiments using "inter-layer norm growth" as an observable signal to verify the mechanism.

### Overall Architecture
- **Task Set**: Single-index models (SIM, Hermite link), $(d,k)$-sparse parity, in-context linear regression, and $(N,p)$-modular addition. Optimizers include mini-batch SGD and full-batch GD. Models use 2-layer MLPs (ReLU, no residual) and 2-layer Transformers (optional QK normalization).
- **Data Strategy**: Besides standard single-set repetition, $T$-phase training is used: phase $i$ trains on a subset $\mathcal{S}_i \subset \mathcal{S}_{i+1}$ to achieve rapid non-trivial training performance on small sets before generalizing on larger ones.
- **Analyzable Model**: $f(x) = a \sigma(w^\top x) - 1$ where $\sigma(z) = \frac{1}{2}z^2$, using correlation loss $\ell(y,y') = -yy'$ with projected updates ($a$ clipped to $[-1, 1]$, $w$ normalized to the unit sphere) on 2-sparse parity.

### Key Designs

1.  **Closed-form Bound for 2-phase Training (Theorem 1)**:
    - **Function**: Translates "small data acceleration" into a quantifiable step bound. Proves that for $d \le N \le d^2$, 2-phase training requires only $O((Nd)^{1/4} \log(d/\varepsilon))$ steps for $w$ to converge to $\|\hat w - w^\star\|_2 \lesssim \sqrt{\varepsilon}$, far fewer than the $O(m^{1/2}\log(d/\varepsilon))$ required for full population training (when width $m \gg d^2$).
    - **Mechanism**: In Phase 1, projected GD on a subset of size $N$ runs until $|a| \ge a_\star$. The gradient of $a$ is determined by $q^{(t)} = (w^{(t)})^\top \hat M w^{(t)}$. Anti-concentration of $\hat M = \frac{1}{N}\sum y x x^\top$ gives $|q^{(t)}| = \Theta(N^{-1/2})$, much larger than the population gradient $\Theta(1/d)$. Thus, $|a|$ grows linearly at $N^{-1/2}$ on small data, reaching $a_\star$ in $T_1 \lesssim a_\star \sqrt{N}/\eta$ steps. Phase 2 switches to the population gradient, performing power iteration on the true matrix $M$ with a convergence rate controlled by $\eta a_\star$.
    - **Design Motivation**: The two terms $T_1, T_2$ decouple the mechanism: $T_1$ is driven purely by sampling bias (mostly independent of label signal), while $T_2$ depends on the magnitude of $a_\star$. This implies any method that pushes $a$ early should yield equivalent acceleration.

2.  **Random Label Verification (Corollary 2 + Experiments)**:
    - **Function**: Separates "sampling bias driving second-layer growth" from "task signal driving first-layer learning."
    - **Mechanism**: Replacing Phase 1 with training on a small dataset with uniformly sampled $\pm 1$ random labels theoretically still yields an $O(N^{-1/2})$ growth rate for $|a|$. Experiments on MLPs and Transformers show that curves for random-label Phase 1 (green) nearly overlap with true-label small-set curves (yellow), both being significantly faster than large-set training (blue).
    - **Design Motivation**: This is the cleanest "difference experiment." If acceleration were from task signals or input distribution bias, random labels would not help. Their success proves that the "sampling bias $\to$ fast second-layer growth" path is the primary driver.

3.  **Inter-layer Initialization & Learning Rate Interventions (Section 5.2)**:
    - **Function**: Actively simulates the growth imbalance created by sampling bias to verify if reproducing this imbalance eliminates the gap.
    - **Mechanism**: Experimental interventions include: (i) increasing the initialization scale of the second layer; (ii) using layer-wise learning rates ($\eta_a > \eta_w$); (iii) observing the role of QK normalization in Transformers. Any of these interventions significantly reduces or eliminates the gap between large and small dataset training.
    - **Design Motivation**: Since the convergence rate depends on the relative growth speed between layers, any engineering means to balance this should be equivalent to using a smaller dataset.

### Training Strategy
All MLPs/Transformers use default PyTorch initialization ($W_{ij} \sim \text{Unif}[-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}}]$). MLPs use SGD, and Transformers use AdamW. Learning rates are swept independently for each setting. Performance is averaged over multiple random seeds at fixed compute = batch $\times$ steps.

## Key Experimental Results

### Main Results
| Task / Setting | Dataset Comparison | Observed Compute Saving | Note |
| :--- | :--- | :--- | :--- |
| (20,6)-sparse parity (Transformer) | Small set vs. Online | Yellow converges much earlier | Fig.1, universal phenomenon |
| (20,6)-sparse parity (MLP, Full-batch) | $N = 2^{14}$ vs. $2^{20}$ | ~100x acceleration | Fig.2, refutes SQ/Variance theories |
| SIM ($d=40$, Full-batch GD) | Small set vs. Population | Faster at every step | Fig.2 |
| ICL / Mod Addition (Transformer) | Multi-phase training | Significant acceleration | Fig.1, cross-architecture |

### Ablation Study
| Intervention | Key Metric | Conclusion |
| :--- | :--- | :--- |
| Forcing $\hat{\mathbb{E}}[x]=0$, $\hat{\mathbb{E}}[y]=0$ | Small-set still faster | Input bias is not the primary cause |
| Injecting small-set bias into large set | Only matches at specific $m$ | Bias magnitude must be tiny to match Cornacchia theory |
| Phase 1 with random labels | Acceleration matches true labels | Label signal is irrelevant; bias is key |
| Scaling up 2nd-layer init / $\eta_a$ | Gap significantly shrinks/vanishes | Directly validates growth mechanism |
| QK Norm in Transformers | Nuanced effect | Implicitly regulates inter-layer dynamics |

### Key Findings
- The gap persists under full-batch GD, providing the cleanest evidence against "stochasticity-driven" acceleration hypotheses.
- The ratio $\|a\|_2 / \|W\|_F$ is an observable proxy for the mechanism; small data, random labels, and large second-layer init all correspond to a faster rise in this ratio.
- Multi-phase training requires small subsets only in the early stages; later stages can use large subsets to ensure generalization.
- The $(Nd)^{1/4}$ complexity suggests that for reasoning tasks (inherently discrete/combinatorial), repeating small data may be more efficient than scaling data.

## Highlights & Insights
- The perspective of "Small data acceleration = Implicit layer-wise LR" is highly transferable, unifying data strategies and optimizer strategies under the fundamental variable of "relative inter-layer growth."
- Using 2-sparse parity with quadratic activation allows for closed-form bounds for both $T_1$ and $T_2$ that precisely match the observable proxy variable (norm ratios).
- Random label training provides a "layer-wise pre-warmup" equivalent to real pre-training, suggesting that seemingly "meaningless" warmup steps may be more significant than previously thought.

## Limitations & Future Work
- Theory covers a highly controlled setting (2-sparse parity, quadratic MLP, correlation loss, projected updates); generalization to ReLU, deep networks, or cross-entropy remains open.
- Experiments focus on synthetic tasks; while citing LLM post-training observations (Kopiczko 2026), the authors did not systematically replicate results on large-scale LLMs/ViTs.
- The mechanism centers on the imbalance between two layers; whether relative growth remains critical in deeper networks and its interaction with various Norm layers needs further study.
- The risk of overfitting when repeating small data in over-parameterized models was not deeply discussed.

## Related Work & Insights
- **vs. Dandi et al. 2024 (SQ Theory)**: They explained acceleration in SIM for batch SGD, but were contradicted here by full-batch GD results and discrete tasks.
- **vs. Kotha et al. 2025 (Variance Theory)**: Explained part of the mini-batch phenomenon, but the persistence of the gap in full-batch GD proves variance is not the only key.
- **vs. Cornacchia et al. 2025 (Input Bias)**: Provides a signal of $O(\eta^k)$, which is an order of magnitude smaller than the $O(N^{-1/2})$ sampling bias.
- **vs. $\mu$P / Tensor Programs**: Those works explicitly control inter-layer growth via parameterization; this paper shows that "data scale" itself acts on the same principle via sampling bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies scattered anomalies into a single mechanism while refuting three existing theories.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong intervention designs across tasks/optimizers, though lacking large-scale LLM validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain; the "falsification-verification" structure between Sections 4 and 5 is elegant.
- Value: ⭐⭐⭐⭐ Provides new training intuitions and suggests a new dimension for designing layer-wise learning rates and initializations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)
- [\[AAAI 2026\] A Phase Transition for Opinion Dynamics with Competing Biases](../../AAAI2026/others/a_phase_transition_for_opinion_dynamics_with_competing_biase.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)

</div>

<!-- RELATED:END -->
