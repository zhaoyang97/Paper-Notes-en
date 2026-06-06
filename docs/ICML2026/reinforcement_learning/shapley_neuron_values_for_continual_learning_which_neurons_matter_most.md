---
title: >-
  [Paper Note] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?
description: >-
  [ICML 2026][Reinforcement Learning][Continual Learning] The authors adapt Shapley values from cooperative game theory to the "filter" level of Convolutional Neural Networks (CNNs). By using a three-fold approximation—Mon…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Continual Learning"
  - "Shapley Values"
  - "Neuron Importance"
  - "Catastrophic Forgetting"
  - "Buffer-Free"
date: 2026-05-08
content_hash: aa93e19d27e6af23
---

# Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?

**Conference**: ICML 2026  
**arXiv**: [2605.15877](https://arxiv.org/abs/2605.15877)  
**Code**: GitHub (The paper marks a "GitHub Code" link, URL to be added)  
**Area**: Interpretability / Continual Learning / Shapley Value Neuron Attribution  
**Keywords**: Continual Learning, Shapley Values, Neuron Importance, Catastrophic Forgetting, Buffer-Free

## TL;DR
The authors adapt Shapley values from cooperative game theory to the "filter" level of Convolutional Neural Networks (CNNs). By using a three-fold approximation—Monte Carlo (MC), truncation, and Multi-Armed Bandit (MAB)—they estimate continuous importance rankings for each neuron. Freezing the Top-$r\%$ "expert" neurons while keeping others plastic enables Class-Incremental Learning (CIL) on ImageNet-1k to achieve a $+2.88\%$ accuracy gain over the leading buffer-free method, and a $+6.46\%$ gain in Task-Incremental Learning (TIL), all without storing samples or expanding the architecture.

## Background & Motivation

**Background**: The Continual Learning (CL) community primarily focuses on three categories: regularization-based (EWC, SI, LwF), replay-based (iCaRL, DER++, PODNet, GEM), and dynamic architecture-based (PNN, DyTox, MEMO). While replay-based methods show the strongest performance, they require storing samples (violating the strict "current task data only" definition and potentially infringing on GDPR). Dynamic architectures suffer from unbounded parameter growth, and regularization-based methods often collapse when tasks are highly heterogeneous.

**Limitations of Prior Work**: A core deficiency in existing buffer-free methods is the **lack of knowledge regarding which neurons are truly important**. Approaches like "Winning Subnetwork" (WSN) use binary $\{0,1\}$ scoring, which fails to distinguish between "marginal Top-$k$" and "absolute Top-$k$" neurons under tight capacity budgets. NFL+ gradually freezes parameters but lacks a principled measure of importance.

**Key Challenge**: The stability-plasticity trade-off becomes a binary choice of "which neurons to freeze and which to keep" under fixed capacity. Although modern over-parameterized networks have sufficient capacity, they lack a fair, continuous, and theoretically grounded attribution mechanism.

**Goal**: To construct a principled measure that assigns **continuous real-valued** importance to each neuron without storing samples or expanding architecture, ensuring that Top-$k$ selection remains stable and effective across various sparsity budgets $c$.

**Key Insight**: Noticing that the Shapley value in cooperative game theory is the **unique** allocation scheme satisfying the four axioms of Efficiency, Null Contribution, Symmetry, and Linearity. Mapping "neurons as players" and "model accuracy as payoff" inherits these fairness guarantees; the remaining challenge is the engineering problem of efficient estimation within an exponential subset space.

**Core Idea**: Redefine neuron importance using "Shapley Neuron Values," then employ MC + Truncation + MAB to reduce exponential complexity to a manageable level. By accumulating frozen masks task-by-task, the method allocates multi-task expert subnetworks within a fixed capacity.

## Method

### Overall Architecture
Consider a network with $L$ layers, where the $l$-th layer has $C_l$ filters. The total number of neurons is $N = \sum_{l=1}^L C_l$, forming the set $\mathcal{M} = \{m_i\}_{i=1}^N$. A sequence of tasks $\{T_t\}_{t=1}^T$ arrives sequentially, with access only to the current task data $\mathcal{D}_t$. For each task $t$:

1.  Train the "plastic" neurons (those outside the cumulative frozen mask $\mathbf{B}_{t-1} = \bigcup_{i<t} S_i$) for several epochs.
2.  Estimate the Shapley value $\hat{\phi}_i$ for each neuron using the validation set after training.
3.  Select the Top-$\lfloor c\cdot N\rfloor$ neurons to form the current task's binary mask $S_t$, and merge it into the cumulative frozen set $\mathbf{B}_t$.
4.  Save the task head $h_t$ and proceed to the next task.

The parameter update rule applies the frozen mask $\mathbf{M}_{t-1}$ directly to the gradient:

$\theta \leftarrow \theta - \eta\Big(\frac{\partial \mathcal{L}}{\partial \theta}\odot \mathbf{M}_{t-1}\Big)$

where $(\mathbf{M}_{t-1})_j = 0$ if and only if $\theta_j$ belongs to a neuron that has already been frozen.

### Key Designs

1.  **Axiomatic Definition of Shapley Neuron Value**:
    - **Function**: Distribute the model accuracy $V(\mathcal{M})$ uniquely and fairly among $N$ neurons, assigning each a continuous real-valued importance $\phi_i \in \mathbb{R}$ such that $\sum_i \phi_i = V(\mathcal{M})$.
    - **Mechanism**: When masking a neuron, its output is **not** set to zero but replaced with its **mean response**. This blocks the information flow while preserving the statistics for downstream layers, avoiding the cascading collapse caused by zeroing. Letting $V(\mathcal{S})$ be the accuracy when only subset $\mathcal{S}$ is retained and the rest are substituted, the Shapley value $\phi_i = \sum_{\mathcal{S}\subseteq \mathcal{M}\setminus\{i\}} \frac{|\mathcal{S}|!(|\mathcal{M}|-|\mathcal{S}|-1)!}{|\mathcal{M}|!}\bigl[V(\mathcal{S}\cup\{i\}) - V(\mathcal{S})\bigr]$ is the unique form satisfying the four axioms.
    - **Design Motivation**: Previous binary methods (like WSN) only answer whether a neuron is in the "Top-$k$." Under small budgets $c$ (e.g., $c=0.03$), fine-grained ranking is impossible. Continuous rankings naturally provide interpretable Top-$k$ selections across multiple $c$ values and align with axioms—e.g., Null Contribution automatically removes dead neurons that never contribute, and Symmetry ensures identical neurons are treated equally.

2.  **Three-fold Approximation Algorithm (MC + Truncation + MAB)**:
    - **Function**: Reduce the $O(N!)$ exact Shapley calculation to a complexity feasible for ResNet-18.
    - **Mechanism**:
        (i) **Monte Carlo**: Rewrite $\phi_i$ as $\phi_i = \mathbb{E}_{\pi\sim\Pi}\bigl[V(\mathcal{S}_i^{\pi}\cup\{i\}) - V(\mathcal{S}_i^{\pi})\bigr]$ and estimate marginal contributions by sampling random permutations.
        (ii) **Truncation**: Skip marginal calculations for a neuron when the permutation prefix $\mathcal{S}_i^{\pi}$ is too small and $V(\mathcal{S}_i^{\pi}) \le \tau$ (marginal utility is meaningless when the model is dysfunctional), saving nearly an order of magnitude in computation.
        (iii) **Multi-Armed Bandit**: Since the goal is only to reliably distinguish the Top-$k$, sampling continues only for neurons whose confidence intervals still cross the current $k$-th largest value. Samples are not wasted on neurons clearly inside or outside the Top-$k$. Formally, the algorithm converges when the set $\mathcal{A} \leftarrow \{i : |\hat{\phi}_i - \phi^{(k)}| < \delta_i\}$ is empty, where $\delta_i = z_\alpha \cdot \sigma_i/\sqrt{n_i}$.
    - **Design Motivation**: Pure MC is too slow for ResNet-18 (where $N$ is in the thousands). Truncation utilizes the physical intuition of "signal decay." MAB changes the objective from "estimating all $\phi_i$" to "finding Top-$k$," aligning the estimation target perfectly with the downstream decision target.

3.  **Cumulative Frozen Masks and Stable–Plastic Decoupling**:
    - **Function**: Hard-freeze neurons "claimed" by previous tasks and leave the rest for new tasks, explicitly separating Stable Neurons (old knowledge) and Plastic Neurons (new tasks).
    - **Mechanism**: Accumulate a binary mask $\mathbf{B}_{t-1} \in \{0,1\}^N$ at the neuron level, then expand it to the parameter level $\mathbf{M}_{t-1}$ (zeroing weights $\theta_j$ belonging to frozen neurons). During training, gradients are multiplied by $\odot\,\mathbf{M}_{t-1}$. The sparsity budget for each task $c \in (0,1)$ is typically set to $1/T$ or smaller.
    - **Design Motivation**: Compared to WSN's "soft mask + retraining" or regularization's "soft constraints," hard freezing offers a strong guarantee of zero backward transfer (BWT $\approx 0.00$), reducing catastrophic forgetting from "minimized" to "strictly zero."

### Loss & Training
Standard Cross-Entropy + Adam is used with He-initialized ResNet-18. CIFAR-100/Tiny-ImageNet are trained for 200 epochs, and ImageNet-1k for 100 epochs, all using early stopping. Hyperparameters are grid-searched on the first task's validation set and then **frozen** for all subsequent tasks (GTEP protocol).

## Key Experimental Results

### Main Results

| Dataset | Scenario | Tasks | SNV (**Ours**) | 2nd Place Buffer-Free | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ImageNet-1k | CIL | 10 | $\mathbf{41.30\%}$ | NFL+ $38.42\%$ | $+2.88$ |
| ImageNet-1k | CIL | 20 | $\mathbf{34.20\%}$ | NFL+ $31.50\%$ | $+2.70$ |
| ImageNet-1k | CIL | 50 | $\mathbf{25.60\%}$ | NFL+ $22.40\%$ | $+3.20$ |
| ImageNet-1k | TIL | 10 | $\mathbf{57.82\%}$ | NFL+ $51.36\%$ | $+6.46$ |
| ImageNet-1k | TIL | 20 | $\mathbf{50.45\%}$ | NFL+ $45.80\%$ | $+4.65$ |
| ImageNet-1k | TIL | 50 | $\mathbf{40.18\%}$ | NFL+ $37.20\%$ | $+2.98$ |

SNV even outperforms the replay-based method DyTox ($40.15\%$ with 20,000 exemplars) on CIL/10, proving that principled neuron selection can compensate for the absence of a replay buffer.

### Ablation Study

| Config (TIL, CIFAR-100, 10 tasks) | ACC | BWT | PS | Description |
| :--- | :--- | :--- | :--- | :--- |
| SNV, $c=0.03$ | $71.74$ | $0.00$ | $0.52$ | Leads WSN $(59.65)$ even under tight budget |
| SNV, $c=0.05$ | $74.52$ | $0.00$ | $0.54$ | |
| SNV, $c=0.1$ | $76.19$ | $0.00$ | $\mathbf{0.62}$ | Optimal PS |
| SNV, $c=0.3$ | $77.89$ | $0.00$ | $0.58$ | |
| SNV, $c=0.5$ | $\mathbf{79.76}$ | $0.00$ | $0.60$ | Optimal ACC |
| WSN, $c=0.5$ | $64.00$ | $0.00$ | $0.66$ | Binary scoring has a lower upper bound |
| NFL+ (no $c$) | $70.68$ | $-0.35$ | $\mathbf{0.65}$ | Non-zero BWT |
| EWC | $36.47$ | $-54.13$ | $0.37$ | Regularization collapses on 1000 classes |

### Key Findings
- **Zero backward transfer is a structural outcome**: All SNV configurations show BWT of strictly $0.00$ due to hard freezing. NFL+ and EWC show significant negative BWT, indicating their "soft" constraints fail to preserve old knowledge in long task sequences.
- **SNV advantage is greatest at small budgets**: At $c=0.03$, SNV ($71.74$) vs. WSN ($59.65$) confirms the intuition that continuous scoring yields more accurate rankings under tight budgets.
- **Pruning cliff reveals parameter efficiency**: SNV on CIFAR-100 can withstand $30\%$ pruning before performance collapses, whereas NFL+ collapses between $20\%$--$30\%$. EWC/LwF slide slowly from $0\%$--$80\%$, suggesting significant redundancy. The "cliff" in SNV indicates a healthy signal that every neuron is being effectively utilized.
- **Memory-based methods matched on ImageNet-1k**: While DyTox leads most buffer-free methods using 20k exemplars, SNV matches or surpasses it without any stored samples. The authors suggest that small-buffer methods should be classified as sequential learning rather than strict CL.

## Highlights & Insights
- **Mean Response instead of Zeroing**: Replacing neuron outputs with mean activations on the validation set prevents downstream distribution collapse, ensuring the marginal difference in $V(\mathcal{S})$ provides a true signal—a key detail for stable MC estimation.
- **MAB Alignment with Decision Goals**: Traditional neuron Shapley estimation seeks accuracy for all $\phi_i$; this work seeks accuracy only for the Top-$k$ threshold. Reformulating the statistical goal as best-arm identification is a transferable insight for any attribution scenario utilizing Top-$k$ results.
- **GTEP Hyperparameter Protocol**: Hyperparameters are tuned only on the first task, avoiding the common "hidden cheating" in CL where hyperparameters are grid-searched over the entire task sequence.
- **Honesty regarding the Buffer-Free vs. Memory-Based Comparison**: The authors explicitly argue that comparing fixed buffer sizes is often unfair and propose treating "small replay buffer" and "strict CL" as two distinct problems.

## Limitations & Future Work
- The Shapley estimation is limited to the "filter" granularity of CNNs. Moving to Transformer/Attention head granularity may cause marginal contribution noise to explode due to strong co-adaptation between heads.
- MC + MAB still incurs non-trivial overhead. Evaluating $\phi_i$ requires many forward passes after each task; the wall-clock overhead relative to training time was not disclosed.
- The fixed capacity assumption acts as both an upper and lower bound. As $T \to \infty$ and $c = 1/T \to 0$, budget pressure will lead to a deadlock of "no plastic neurons available."
- Experiments were restricted to vision CIL/TIL; evidence for NLP continual learning (e.g., sequence labeling) is currently lacking.

## Related Work & Insights
- **vs WSN (Winning SubNetwork)**: WSN uses binary $\{0,1\}$ selection; SNV uses continuous $\phi_i$. The $12\%$ lead under small $c$ proves continuous values are superior.
- **vs NFL+ (No Forgetting Learning)**: NFL+ is a progressive freezing heuristic without a principled importance measure. SNV wins across all 6 ImageNet-1k settings by $+2.7$ to $+6.5$.
- **vs EWC / SI / LwF**: Regularization methods using rough approximations like diagonal Fisher fail to capture complex parameter coupling in 1000-class tasks. SNV's subset-level marginals reflect true combinatorial contributions.
- **vs Neuron Shapley (Ghorbani et al.)**: This work inherits the MC+Truncation framework but adds MAB to prioritize Top-$k$ identification and is the first to apply Neuron Shapley for freezing in CL.
- **Insight**: By replacing the value function $V$ with a "fairness metric" or "robustness index," the SNV framework can be repurposed for fairness attribution or identifying adversarial vulnerabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ — Shapley + Neuron is not new, but combining MAB for Top-$k$ identification within a CL freezing pipeline is.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive main experiments across three datasets, scenarios, and task counts; deep analysis into pruning and BWT. Wall-clock time is missing.
- Writing Quality: ⭐⭐⭐⭐ — The logic from axioms to unique allocation to estimation is clean, though some minor variable definition inconsistencies exist.
- Value: ⭐⭐⭐⭐ — Provides a principled and interpretable base for buffer-free CL with a controllable budget.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Deployed Reinforcement Learning should be Continual](position_deployed_reinforcement_learning_should_be_continual.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [\[NeurIPS 2025\] Approximating Shapley Explanations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/approximating_shapley_explanations_in_reinforcement_learning.md)
- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](../../ACL2026/reinforcement_learning/savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
