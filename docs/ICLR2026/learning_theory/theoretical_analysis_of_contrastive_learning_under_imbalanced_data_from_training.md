---
title: >-
  [Paper Note] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution
description: >-
  [ICLR 2026][Contrastive Learning Theory][Contrastive Learning] This paper provides a training dynamics theory for contrastive learning under **imbalanced data**. Analyzing a "Transformer-MLP + sparse coding data model," the authors prove that neuron weights undergo a three-stage evolution where minority features are learned more weakly and with more interference due to low frequency. The study theoretically demonstrates that **magnitude pruning** amplifies gradient updates in…
tags:
  - "ICLR 2026"
  - "Contrastive Learning Theory"
  - "Representation Learning"
  - "Feature Learning"
  - "Contrastive Learning"
  - "Data Imbalance"
  - "Training Dynamics"
  - "Magnitude Pruning"
date: 2026-05-08
content_hash: 9c6e8c5612975a43
---

# Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DUXG9E8dEO](https://openreview.net/forum?id=DUXG9E8dEO)  
**Code**: TBD  
**Area**: Contrastive Learning Theory / Representation Learning / Feature Learning  
**Keywords**: Contrastive Learning, Data Imbalance, Training Dynamics, Feature Learning, Magnitude Pruning

## TL;DR
This paper provides a training dynamics theory for contrastive learning under **imbalanced data**. Analyzing a "Transformer-MLP + sparse coding data model," the authors prove that neuron weights undergo a three-stage evolution where minority features are learned more weakly and with more interference due to low frequency. The study theoretically demonstrates that **magnitude pruning** amplifies gradient updates in minority feature directions, thereby restoring representation quality compromised by imbalance (validated by linear probe experiments on CIFAR-LT / ImageNet-LT).

## Background & Motivation

**Background**: Contrastive learning (such as the InfoNCE paradigm behind SimCLR and CLIP) learns universal representations from unlabeled data by "pulling positive pairs together and pushing negative pairs apart." It has become a cornerstone for pre-training in vision, multi-modal, and vision-language models. Recent theoretical progress has explained "why data augmentation is necessary," "why it outperforms GANs," and "how it reduces downstream sample complexity."

**Limitations of Prior Work**: Real-world data is almost always long-tailed or imbalanced—majority classes dominate the composition of positive and negative pairs, leaving minority classes severely underrepresented. This lead to poor discriminative feature learning for minority classes and a decline in overall representation quality. While supervised learning mitigates this via re-weighting or re-sampling, these methods **depend on accurate class labels**, which are unavailable in self-supervised contrastive learning. Consequently, the community has turned to empirical methods like **pruning** to implicitly address long-tailed data.

**Key Challenge**: All these mitigation strategies—whether re-sampling or pruning—are **heuristic**. While improvements are observed in practice, the community lacks a clear understanding of **"exactly which mechanism imbalance uses to damage representations"** and why pruning is effective. A gap exists between theory and practice.

**Goal**: To bridge this gap by addressing three sub-problems: (1) How do neurons incrementally learn features under imbalanced contrastive learning? (2) Quantitatively, how does the low frequency of minority features degrade representation capability? (3) Why does magnitude pruning recover minority features at the dynamics level?

**Key Insight**: The authors adopt a **feature learning paradigm**, modeling data as a **sparse coding model** (where each token is a sparse linear combination of dictionary features + noise). They characterize majority/minority via feature occurrence frequency $\epsilon_j$. By **tracking the evolution of the inner product between neuron weights and each feature direction throughout training**, they can rigorously derive the causal chain from "frequency → learning rate → representation purity."

**Core Idea**: Using the "dynamics of neuron projections onto feature directions" as a metric, the authors prove that **low-frequency minority features are learned more weakly and with higher interference, supported by fewer specialist neurons**. They then prove that **magnitude pruning biasedly amplifies gradients in minority feature directions**, quantitatively compensating for the damage caused by imbalance.

## Method

### Overall Architecture

This paper is not a new algorithm but rather an **analytical framework + a re-interpreted pruning process**. The logical chain is: first, define a **tractable simplified model** (Transformer-MLP encoder + sparse coding data) trained with InfoNCE loss; then, **divide the training process into three stages**, providing evolution bounds for neuron weights in each stage; next, **quantitatively characterize how imbalance (frequency ratio $\epsilon_{\min}/\epsilon_{\max}$)** weakens representations across three dimensions; finally, incorporate magnitude pruning into the same framework to prove how it **modifies the update rate of minority feature directions** to recover the disadvantage.

Specific setup: The input sequence $X=[x^{(1)},\dots,x^{(L)}]$ passes through a **single-head self-attention** followed by an MLP with **Bipolar ReLU (BReLU)**, outputting an embedding $f_\theta(X)\in\mathbb{R}^m$. BReLU is defined as $\mathrm{BReLU}_b(s)=\mathrm{ReLU}(s-b)-\mathrm{ReLU}(-s-b)$, a symmetric activation with threshold $b$. The $i$-th hidden unit is:

$$h_i(X_n)=\sum_{r=1}^{L}\mathrm{BReLU}_{b_i^{(t)}}\big(\langle w_i^{(t)},\ \mathrm{Attention}(W_Q x_n^{(r)}, W_K X_n, W_V X_n)\rangle\big).$$

The training objective is the InfoNCE empirical risk with $\ell_2$ regularization:

$$\widehat{L}_{\mathrm{aug}}(f_\theta)=\frac{1}{K}\sum_{k=1}^{K}\ell\big(f_\theta,X_k,Y_k,N_k\big)+\frac{\lambda}{2}\|\theta\|_F^2,$$

where similarity is $\mathrm{sim}_{f_\theta}(X_n,Y_n)=\langle f_\theta(X_n),\ \mathrm{StopGrad}(f_\theta(Y_n))\rangle$, with StopGrad acting as an identity mapping during forward passes and blocking gradients during backward passes. The "microscope" for the analysis is the inner product $\langle w_i^{(t)}, M_j\rangle$—the projection of neuron $i$ onto the $j$-th feature direction $M_j$—as it evolves over training steps $t$.

### Key Designs

**1. Sparse Coding Data Model + Frequency Parameter $\epsilon_j$: Translating "Imbalance" into Tractable Frequency**

To theoretically discuss how "imbalance damages representations," a data model that quantifies "majority/minority" is required. The authors use a **sparse coding model** (Assumption 3.1): each token $x^{(\ell)}_n = M z^{(\ell)}_n + \xi^{(\ell)}_n$, where $M=[M_1,\dots,M_d]$ is a **column-orthogonal** dictionary matrix, $z$ is a sparse latent signal, and $\xi\sim\mathcal{N}(0,\sigma_\xi^2 I)$ is noise. Crucially, the noise variance $\sigma_\xi^2=\Theta(\sqrt{\log d}/d)$ allows the **noise magnitude to be comparable to or exceed the signal** (when $d_1\gg d$), meaning no linear mapping can directly recover the latent signal—making the problem "simple in form but difficult in essence," ideal for testing non-linear networks. Positive/negative pairs are formalized via signal support and sign consistency (Assumption 3.4).

Imbalance is injected via **Definition 3.1**: the activation probability of feature $j$ is $\Pr(z^{(i)}_j\neq 0)=\Theta(\epsilon_j \log\log d / d)$. $\epsilon_{\max}=\max_j\epsilon_j$ corresponds to **majority features**, and $\epsilon_{\min}=\min_j\epsilon_j$ to **minority features**. Thus, the "degree of imbalance" is compressed into a clean ratio $\epsilon_{\min}/\epsilon_{\max}$, upon which all subsequent quantitative conclusions depend.

**2. Three-Stage Training Dynamics: From "Random Growth" to "Specialization"**

This is the core mechanistic conclusion: while the algorithm appears to be uniform gradient descent, neuron weights actually evolve in **three stages** (Lemma 3.1, Lemma 3.2, Theorem 3.1).

*Stage 1 (Lemma 3.1)*: Neurons **grow in feature directions and are suppressed in non-feature directions**, with the growth rate determined by frequency. Formally,

$$|\langle w_i^{(t+1)},M_j\rangle|\ \ge\ |\langle w_i^{(t)},M_j\rangle|\Big(1-\eta\lambda+\epsilon_j\frac{\eta C_z\log\log d}{d}\Big)-\widetilde{O}\Big(\frac{\eta\|w_i^{(t)}\|^2}{\mathrm{poly}(d_1)}\Big),$$

while projections in non-feature directions $M_j^\perp$ only decay. The growth factor directly contains $\epsilon_j$: **high-frequency majority features grow quickly, while low-frequency minority features are difficult to capture early on.**

*Stage 2 (Lemma 3.2)*: Neurons differentiate into two types: **lucky neurons** $M^\star_j$ (aligned with a single feature direction) and **ordinary neurons** $M_j$ (aligned with a direction but more "mixed"). Lucky neurons significantly strengthen their alignment with $M_j$: $|\langle w^{(T_2)}_i,M_j\rangle|^2\ge 2\cdot\frac{\epsilon_j}{\epsilon_{\max}}\|w^{(T_1)}_i\|_2^2$, with their count $|M^\star_j|\ge m\cdot d^{-(\epsilon_{\max}/\epsilon_{\min})^2}$. Ordinary neurons are bounded by a constant multiple of the lucky neurons' feature components. The result is **purified learned features and continuous suppression of non-feature components.**

*Stage 3 (Theorem 3.1, Convergence)*: The training error converges to $o(1)$, and each neuron converges to:

$$w_i^{(t)}=\sum_{j\in N_i}\alpha_{i,j}M_j+\sum_{j\notin N_i}\alpha'_{i,j}M_j+\sum_{j\in[d_1]\setminus[d]}\beta_{i,j}M_j^\perp,$$

meaning they exhibit **strong alignment with a small cluster of primary features $N_i$, weak alignment with others, and nearly zero in non-feature directions**. Primary coefficients satisfy $\alpha_{i,j}\in[\frac{\epsilon_j}{\epsilon_{\max}}\frac{\tau}{\Xi_2},\ \frac{\epsilon_j}{\epsilon_{\max}}\tau]$. This convergence characterization is the basis for conclusions regarding "imbalance harm" and "pruning rescue."

**3. The Triple Harm of Imbalance: How Frequency Ratios Suppress Magnitude, Expand Mixing, and Reduce Specialist Neurons**

Applying Stage 3 conclusions, the authors quantitatively identify **three entangled pathways** through which imbalance damages representations (Remark on Theorem 3.1):

- **Weakened Minority Features**: The magnitude $\alpha_{i,j} \propto \epsilon_j/\epsilon_{\max}$ of a neuron on $N_i$ decreases as $\epsilon_j$ decreases.
- **Increased Neuron Mixing**: The size of the primary feature set $|N_i|=O(d^{1-(\epsilon_{\min}/\epsilon_{\max})^2})$ increases as $\epsilon_{\min}/\epsilon_{\max}$ decreases, meaning neurons are forced to **mix multiple features** rather than remaining pure.
- **Decreased Specialist Neurons**: For each feature $M_j$, the lower bound for pure specialist neurons is $\Omega(m\cdot d^{-(\epsilon_{\max}/\epsilon_{\min})^2})$, which drops sharply as the gap between $\epsilon_{\max}$ and $\epsilon_{\min}$ widens.

Since the success of contrastive learning **relies on lucky neurons (specialized in single pure features)**—as mixed neurons are only useful for a few downstream tasks—these three points imply that imbalance makes minority features weak, mixed, and poorly supported by specialists. This **forces the model to use more neurons/complex structures** to cover all features, at the cost of higher computation. Remark 3 of Theorem 3.1 further notes that pure and separable upstream feature directions allow simple linear probes to extract features; **higher neuron specialization leads to better linear separability and downstream generalization.**

**4. Magnitude Pruning Amplifies Minority Features: Rewriting Update Rates via "Prunability"**

Given that small-magnitude neurons are precisely those learning minority features, the authors provide a dynamical explanation for magnitude pruning (Algorithm 1 + Theorem 3.2). The algorithm uses **forward masking without backward masking**: each epoch, the $\alpha$ proportion of neurons with the lowest magnitudes are temporarily masked ($M^{(t)}$) during the forward pass to calculate loss, but **gradients are applied to the full parameter set**:

$$\theta^{(t+1)}\leftarrow(1-\eta\lambda)\theta^{(t)}-\eta\cdot g(\theta^{(t)}_t,M^{(t)}).$$

Mechanism (Theorem 3.2): When minority-feature neurons are pruned, the **positive logit of those samples decreases and the negative logit increases**, thereby amplifying the gradients of samples containing minority features. Quantitatively, the growth rate of lucky neurons aligned with minority direction $M_{j^\star}$ is boosted to $\Omega(\eta\epsilon_{j^\star}^2\alpha\,C_z\log\log d/d)$ (order $\alpha/d$), while non-minority directions only increase at order $\alpha/d^2$—a **$1/d$ factor difference**. Crucially, the convergence coefficient $\alpha_{i,j^\star}\in(\tau/\Xi_2,\ \tau)$ **no longer depends on the ratio $\epsilon_{\min}/\epsilon_{\max}$**. This theoretically explains the empirical observation: "pruning → amplified minority features → more specialist neurons → more robust and balanced representations."

### Loss & Training

Training utilizes InfoNCE with $\ell_2$ regularization (weight decay $\lambda$), temperature $\tau$, and inner product similarity with StopGrad. Pruning (Algorithm 1): MLP weights follow Gaussian initialization, attention weights are initialized as identity, and masks start at all ones. In each step, an $\alpha$ ratio of minimum-magnitude neurons is pruned; masked parameters are used for the forward pass, while the full parameter set is updated in the backward pass. Key scale parameters for analysis: Stage 1 duration $T_1=\Theta(d_1\log d/(\eta\log\log d))$, Stage 2 duration adds $\Theta(d\tau\log d/(\epsilon_{\max}\eta\log\log d))$, and convergence interval $T\in[T_3,T_4]=[d^{1.01}/\eta,\ d^{1.99}/\eta]$.

## Key Experimental Results

Experiments are "theory-validating" in nature: **linear probing** on CIFAR10-LT / CIFAR100-LT / ImageNet-LT, comparing vanilla contrastive learning (w/o pruning) with magnitude pruning (w/ pruning). Imbalance ratio $\rho$ = number of majority samples / number of minority samples. Metrics involve overall Accuracy and $\Delta_{20}$ (accuracy gap between top 20% head and bottom 20% tail classes).

### Main Results

| Dataset | $\rho$ | Acc (w/o Pruning) | Acc (w/ Pruning) | $\Delta_{20}$ (w/o) | $\Delta_{20}$ (w/ Pruning) |
|--------|------|------|------|------|------|
| CIFAR10-LT | 1 | 90.93 | **91.52** | 1.54 | **1.28** |
| CIFAR10-LT | 10 | 79.25 | **84.92** | 3.42 | **2.99** |
| CIFAR10-LT | 50 | 75.58 | **83.60** | 3.92 | **3.35** |
| CIFAR10-LT | 100 | 74.24 | **81.31** | 5.69 | **5.62** |
| CIFAR100-LT | 10 | 51.21 | **56.33** | 2.45 | **1.37** |
| CIFAR100-LT | 50 | 49.32 | **56.12** | 4.95 | **2.57** |
| CIFAR100-LT | 100 | 47.12 | **54.93** | 7.11 | **4.38** |
| ImageNet-LT | 256 | 63.21 | **65.12** | 8.47 | **7.21** |

### Key Findings (Pruning gains amplify as imbalance worsens)

| Phenomenon | Data | Explanation |
|------|------|------|
| Heavier imbalance, larger pruning gain | CIFAR10-LT $\rho{=}10$ gains +5.67, $\rho{=}50$ gains +8.02 | Consistent with theory: pruning specifically recovers minority features |
| Narrowed head-tail gap $\Delta_{20}$ | CIFAR100-LT $\rho{=}100$ from 7.11 → 4.38 | Pruning improves balance, not just global shift |
| Small gain in balanced scenarios | CIFAR10-LT $\rho{=}1$ only +0.59 | At $\rho{=}1$, there are fewer minority features to "rescue" |

### Key Findings
- **Pruning gain scales monotonically with $\rho$**: A larger $\rho$ means minority features are rarer and vanilla learning is poorer. Pruning biasedly amplifies these directions, leading to sharper gains—reflecting the "$\alpha/d$ vs $\alpha/d^2$ rate difference."
- **Beyond global accuracy, focus on gap reduction**: The decrease in $\Delta_{20}$ proves that pruning restores performance for tail classes rather than shifting the whole distribution, confirming the "increase in specialist neurons" theory.
- Synthetic experiments in the appendix directly validate the three-stage dynamics and neuron projection curves.

## Highlights & Insights
- **Condensing "imbalance" into a single frequency ratio $\epsilon_{\min}/\epsilon_{\max}$**: The entire set of harms (weak magnitude, mixing, reduced specialists) is analytically linked to this ratio, making the conclusion "pruning removes dependence on this ratio" highly elegant.
- **Providing a dynamical causal explanation for pruning**: Instead of just saying it works, the paper links small-magnitude neurons ↔ minority features ↔ forward pruning imbalance ↔ gradient amplification. This turns an empirical trick into theory.
- **The lucky/ordinary neuron dichotomy**: Attributing representation quality to the count of "neurons specialized in a single pure feature" provides a transferable perspective for analyzing other self-supervised or long-tail methods.
- **Moving towards modern architectures**: Analyzing Transformer-MLP (with self-attention) instead of traditional single-hidden-layer feedforward networks is a step forward for feature learning theory.

## Limitations & Future Work
- **Lack of precise characterization of pruning ratios**: The authors acknowledge they cannot yet provide a full characterization of how performance varies with pruning ratio $\alpha$ and specific pruning schemes, which remains future work.
- **Highly simplified architecture**: Single-head attention + single-layer MLP + BReLU + orthogonal dictionary + Gaussian noise is far from deep Transformers or real image distributions.
- **The theory-experiment bridge is somewhat loose**: The theory is built on sparse coding/frequency models, while experiments use CIFAR/ImageNet. The mapping "feature frequency $\approx$ class frequency" is an implicit assumption not fully proven.
- **Future directions**: Exploring mechanisms other than pruning that biasedly amplify minority gradients (e.g., frequency-adaptive temperature), and extending analysis to multi-head/multi-layer structures.

## Related Work & Insights
- **vs. Wen & Li (2021) / Sun et al. (2025)**: Previous works analyzed contrastive dynamics in single-layer FFNs. This paper introduces the Transformer architecture and **explicitly incorporates data imbalance**, providing a systematic analysis of how imbalance affects feature decoupling.
- **vs. Allen-Zhu & Li (feature purification) / Supervised Feature Learning**: Those works tie features to ground-truth labels. This paper tracks "neuron ↔ feature direction alignment" dynamics in a self-supervised setting.
- **vs. HaoChen et al. (2021) / Spectral Contrastive Theory**: Those use spectral clustering to explain *why* contrastive learning works. This paper focuses on *how* imbalance breaks it and how pruning fixes it.
- **vs. Jiang et al. (2021) / Qian et al. (2022) Empirical Pruning**: While those discovered that pruning improves underrepresented classes, **this paper provides the first dynamical theoretical explanation** ($\alpha/d$ vs $\alpha/d^2$ rate gap).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to characterize contrastive dynamics under Transformer encoders + imbalanced data with a theoretical basis for pruning.
- Experimental Thoroughness: ⭐⭐⭐ Validates trends on three long-tail benchmarks and synthetic data, though architecture scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear three-stage narrative; the frequency ratio provides a strong logical thread.
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for the "unlabeled long-tail + pruning" route, offering insights into self-supervised representation structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap](theoretical_modeling_of_large_language_model_self-improvement_training_dynamics_.md)
- [\[ICLR 2026\] Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data](fast_escape_slow_convergence_learning_dynamics_of_phase_retrieval_under_power-la.md)
- [\[ICLR 2026\] Convergence Analysis of Tsetlin Machines under Noise-Free and Noisy Training Conditions: From 2 Bits to k Bits](convergence_analysis_of_tsetlin_machines_under_noise-free_and_noisy_training_con.md)

</div>

<!-- RELATED:END -->
