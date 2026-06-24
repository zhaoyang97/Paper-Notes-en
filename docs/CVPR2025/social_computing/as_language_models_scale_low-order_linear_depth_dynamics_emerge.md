---
title: >-
  [Paper Note] As Language Models Scale, Low-order Linear Depth Dynamics Emerge
description: >-
  [CVPR 2025][Social Computing][Transformer Dynamics] By treating the depth dimension of Transformers as a discrete-time dynamical system, this paper finds that a linear state-space surrogate model of just 32 dimensions can predict inter-layer sensitivity curves with high precision (Spearman up to 0.99) within a given context. Surprisingly, **as the model scales, the low-order linear surrogate becomes even more accurate**—unveiling a new scaling law.
tags:
  - "CVPR 2025"
  - "Social Computing"
  - "Transformer Dynamics"
  - "Linear Surrogate Models"
  - "Activation Steering"
  - "System Identification"
  - "Scaling Law"
date: 2026-05-08
content_hash: 9ca589cb2d43e1fa
---

# As Language Models Scale, Low-order Linear Depth Dynamics Emerge

**Conference**: CVPR 2025  
**arXiv**: [2603.12541](https://arxiv.org/abs/2603.12541)  
**Code**: To be confirmed  
**Area**: Social Computing  
**Keywords**: Transformer Dynamics, Linear Surrogate Models, Activation Steering, System Identification, Scaling Law

## TL;DR
By treating the depth dimension of Transformers as a discrete-time dynamical system, this paper finds that a linear state-space surrogate model of just 32 dimensions can predict inter-layer sensitivity curves with high precision (Spearman up to 0.99) within a given context. Surprisingly, **as the model scales, the low-order linear surrogate becomes even more accurate**—unveiling a new scaling law.

## Background & Motivation
**Background**: Activation steering has become an important method for modifying the behavior of LLMs, controlling attributes like sentiment and topic by injecting contrastive activation vectors during forward passes. However, choosing which layer to inject them into and how much to inject still relies on layer-by-layer brute-force search or heuristic rules (such as injecting into the last layer).

**Limitations of Prior Work**: There is a lack of **computational models** for the dynamic response along the depth of Transformers. Existing works (such as the linear representation hypothesis and the linear latent state trajectories discovered by Aubry et al.) explain why linear directions can encode semantics, but fail to answer how perturbations propagate with depth after injection and how they ultimately affect the output.

**Key Challenge**: Transformers are high-dimensional non-linear systems, which intuitively should not have simple system descriptions. However, does a **local** low-dimensional linear approximation actually exist?

**Goal**: (1) Can a computable low-order linear surrogate be found to predict inter-layer sensitivity? (2) Does the quality of this surrogate scale with the model size? (3) Can the surrogate model guide the design of more efficient intervention strategies?

**Key Insight**: Drawing on **system identification** methods from control theory, depth is treated as discrete time, the hidden state of the last token as the system state, and activation steering as the control input to linearize and reduce the order of the local dynamics under a frozen context.

**Core Idea**: Model the Transformer depth dynamics as a low-order linear state-space system, discover that this approximation becomes more accurate as the model scales, and utilize this surrogate to design minimum-energy multi-layer intervention policies.

## Method

### Overall Architecture
Given a prompt $p$, the $L$ layers of the Transformer are treated as a discrete-time system. The system state $x_\ell(p)$ is defined as the hidden state vector of the last token at layer $\ell$ ($\in \mathbb{R}^H$). Freezing the representations of other tokens yields the **context-conditioned single-step mapping** $x_{\ell+1} = f_\ell(x_\ell; p)$.

Pipeline: Estimate the concept direction $v_\ell$ $\rightarrow$ perform local linearization to obtain the Jacobian $A_\ell(p)$ $\rightarrow$ project to a reduced $d=32$ dimension using reachability-driven Krylov bases $\rightarrow$ identify the linear state-space model (LLV surrogate) in the reduced space $\rightarrow$ predict the inter-layer gain curve $\rightarrow$ design a minimum-energy multi-layer intervention policy $\rightarrow$ validate on the full model.

### Key Designs

1. **Local linearization with frozen context**:

    - **Function**: Perform Jacobian linearization around the operating trajectory $\bar{x}_\ell(p)$ for a given prompt, yielding $\delta x_{\ell+1} \approx A_\ell(p) \delta x_\ell + A_\ell(p) v_\ell u_\ell$
    - **Mechanism**: Freeze non-last token rows and only perturb the last token state before running it through another Transformer block, isolating the single-token depth dynamics directly associated with steering
    - **Design Motivation**: The full attention matrix is too large to handle directly. By freezing the context, the dynamics simplify to a deterministic point-to-point mapping, allowing the estimation of Jacobian-vector products via central differences

2. **Concept-anchored Krylov dimension reduction bases**:

    - **Function**: Construct a projection basis $P_\ell \in \mathbb{R}^{H \times 32}$ where the first column is precisely the layer-wise concept direction $v_\ell$, and the remaining 31 columns are filled via Krylov reachability construction
    - **Mechanism**: Start from $A_\ell(p) v_\ell$, propagate forward along the average Jacobian, and orthogonalize, yielding "the subspace actually excited by the steering perturbation"
    - **Design Motivation**: Superior to random orthogonal complements—it prioritizes preserving directions reachable by the control input. Ablation experiments demonstrate that the Krylov basis systematically improves predictions compared to random complement bases

3. **Reduced-order LLV surrogate model**:

    - **Function**: Identify a linear dynamical system $r_{\ell+1} \approx \bar{A}_\ell(p) r_\ell + \bar{B}_\ell(p) u_\ell$ in the 32-dimensional reduced-order space
    - **Mechanism**: The reduced-order matrices are $\bar{A}_\ell = P_{\ell+1}^\top A_\ell P_\ell$ and $\bar{B}_\ell = P_{\ell+1}^\top A_\ell v_\ell$, which can predict the final sensitivity of single-layer interventions: $g_k^{pred} \approx C \Phi(k+1, L) \bar{B}_k$
    - **Design Motivation**: The 32-dimensional surrogate model can analytically solve for optimal interventions, whereas the 1280-dimensional (GPT-2-large) full system cannot

4. **Minimum-energy multi-layer control**:

    - **Function**: Given a target concept shift $\Delta y_{tar}$, analytically solve for the multi-layer allocation scheme $u^* = \frac{\Delta y_{tar}}{\|h\|_2^2} h$ that minimizes the required injected energy
    - **Design Motivation**: Heuristic strategies, such as uniform layer injection or last-layer-only injection, waste energy. The surrogate model identifies areas of high sensitivity and concentrates the injection there

### Loss & Training
**No training is involved**—the entire framework is analytical and diagnostic. It is evaluated on the GPT-2 family (GPT-2, GPT-2-medium, GPT-2-large) across 10 binary classification NLP tasks (sentiment, toxicity, irony, hate speech, etc.). The dataset is split into three disjoint sets: concept split (estimating concept directions, 400/class), operating split (identifying local dynamics, 200/class), and held-out split (evaluation only, 200/class). The reduced dimension is $d=32$ (1 concept direction + 31 Krylov bases), and the perturbation magnitude is $\epsilon=0.1$. Jacobian computations utilize forward-mode JVP or fall back to central differences.

## Key Experimental Results

### Main Results: Gain Curve Prediction Accuracy

| Model | Parameters | Reduced Dim | Mean Spearman↑ | Mean Pearson↑ |
|------|:------:|:---:|:---:|:---:|
| GPT-2 | 117M | 32 | 0.77 | 0.68 |
| GPT-2-medium | 345M | 32 | 0.81 | 0.74 |
| **GPT-2-large** | **774M** | **32** | **0.995** | **0.997** |

### Ablation Study: Minimum-Energy Intervention vs. Heuristics

| Intervention Strategy | Relative Energy (LLV-optimal=1.0) | Description |
|----------|:---:|------|
| **LLV-optimal** | **1.0** | Optimally designed by the surrogate model |
| Uniform-all | 2-5x | Uniform injection across all layers |
| Last-layer only | 10-100x | Injection in the last layer only |
| Random single-layer | 10-1000x | Randomly chosen single layer |

### Key Findings
- **Scaling Law of Identifiability**: At a fixed reduced dimension of 32, the linear surrogate becomes more accurate as the model size increases. GPT-2-large predicts the inter-layer gain curves across all 10 tasks almost perfectly (Spearman 0.99-1.00). This reveals a counter-intuitive scaling law where **local interpretability improves as model complexity increases**.
- **The optimal intervention layer is task-dependent**: Some tasks exhibit the highest gain in the last layer, while others have broad, high-gain plateaus in middle layers—indicating that a universal "last-layer injection" strategy is inherently suboptimal.
- **Krylov Basis vs. Random Basis**: Systematic improvements are observed, especially on difficult tasks, demonstrating that steering effects concentrate within low-dimensional reachable subspaces.
- **Direct Operational Value of Control**: The LLV-optimal strategy consistently achieves the lowest or tied-for-lowest energy when validated on the full model, saving 2-5x energy compared to uniform-all.
- **Robustness to Perturbation Magnitude**: High consistency in gain curve prediction is maintained within the range of $\epsilon \in [0.01, 0.5]$.
- **Saturation Effect of Reduced Dimension**: Accuracy increases when expanding from minimal dimensions to 32, and nearly saturates beyond 32—indicating that the steering-related dynamics are indeed compact and low-dimensional.

## Highlights & Insights
- **"The larger the model, the more locally linear it is" is the most profound discovery**. Conventionally, larger models are thought to be less interpretable; however, this paper demonstrates that local depth dynamics are actually more compressible. Intuitively, larger width and representation redundancy might stabilize local Jacobian responses, reducing estimation variance in projected dynamics. If this observation generalizes to other architectures and scales (e.g., Llama-65B), it will profoundly reshape our understanding of LLM interpretability.
- **The application of a control-theory perspective to LLMs** is highly elegant: system identification $\rightarrow$ reduced-order modeling $\rightarrow$ optimal control design $\rightarrow$ full-model validation. This serves as a textbook application of the "model-as-system" paradigm, demonstrating the immense potential of interdisciplinary approaches (control theory $\times$ deep learning).
- **A paradigm shift from "brute-force layer-wise search" to a "predictive system problem"**: Instead of testing every single layer to locate the optimal intervention point, the surrogate model provides the answer directly. This holds immediate practical value for alignment adjustments in real-world deployment.
- **The practicality of the minimum-energy intervention policy**: Saving 2-5x energy over uniform-all and 10-100x over last-layer-only means that the same behavioral modifications can be achieved with much smaller perturbations, minimizing side effects on the model's other capabilities.

## Limitations & Future Work
- **Validation is limited to the GPT-2 family** (up to 774M parameters); whether it generalizes to 7B/70B scale LLMs remains unknown. Generalization to scale would amplify its significance.
- **The frozen-context assumption** limits practical applicability—the surrogate model might fail in multi-turn dialogues or long contexts where the context cannot be frozen.
- **Only single-token states are considered**—multi-token interaction dynamics are not modeled, which might be inadequate for tasks requiring multi-token coordination.
- **Concept direction estimation relies on annotated data**—positive and negative samples are required to estimate layer-wise concept directions, making it not fully unsupervised. In unannotated scenarios, combining this with unsupervised concept discovery warrants further exploration.

## Related Work & Insights
- **vs. Activation Addition (Turner et al.)**: They demonstrated that linear directions can control behavior but did not explain how perturbations propagate across layers. This work bridges that gap by modeling the "propagation dynamics," elevating steering from empirical manipulation to a method backed by predictive theory.
- **vs. Linear Representation Hypothesis (Park et al.)**: While that hypothesis explains why concepts can be represented linearly (static representational geometry), this paper answers the dynamic question of how linear perturbations evolve along the depth, making them complementary.
- **vs. Aubry et al.**: They discovered Jacobian singular direction alignment and linear trajectories of latent states; this work builds on that by performing comprehensive system identification and uncovering the scaling law.
- **vs. Moon (Control Theory for Interpretability)**: Moon analyzed neural networks using controllability, observability, and Hankel singular values, but did not identify reduced-order surrogate models for prediction. This work closes the loop: identification $\rightarrow$ prediction $\rightarrow$ validation.
- This paper could serve as a theoretical foundation for the activation steering domain—marking a transition from "empirically effective" to "backed by system theory."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The control theory perspective + Scaling Law of Identifiability is a wholly new discovery with no precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 tasks + 3 scales + detailed ablations (Krylov vs. random, $\epsilon$ sweep, dimension sweep), though it lacks validation on modern large models (7B+).
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, with a seamless logic flow from motivation $\rightarrow$ formalization $\rightarrow$ prediction $\rightarrow$ validation, complemented by beautifully designed figures.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic theoretical framework for LLM interpretability, holds great practical value in reducing "operational complexity," and uncovering the scaling law carries profound theoretical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](../../ACL2025/social_computing/exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ACL 2026\] Splits! Flexible Sociocultural Linguistic Investigation at Scale](../../ACL2026/social_computing/splits_flexible_sociocultural_linguistic_investigation_at_scale.md)
- [\[NeurIPS 2025\] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](../../NeurIPS2025/social_computing/deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)

</div>

<!-- RELATED:END -->
