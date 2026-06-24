---
title: >-
  [Paper Note] On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning
description: >-
  [ICLR 2026][Optimization][Decentralized SGD] In decentralized training with extreme communication constraints and high data heterogeneity, the authors find that a "global merging" (averaging all node models) at the end of training brings global test performance close to Federated Learning levels. They theoretically prove for the first time that the global merging model of decentralized SGD can achieve the convergence rate of parallel SGD—the key lies in reinterpreting the int…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Decentralized SGD"
  - "Global Merging"
  - "Communication Scheduling"
  - "Data Heterogeneity"
  - "Convergence Analysis"
date: 2026-05-08
content_hash: f5f875c3d22af9c8
---

# On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=zrFnwRHuQo](https://openreview.net/forum?id=zrFnwRHuQo)  
**Code**: https://github.com/Raiden-Zhu/ICLR-2026-Grokking-in-Decentralized-Learning  
**Area**: Distributed Optimization / Decentralized Learning / Model Merging  
**Keywords**: Decentralized SGD, Global Merging, Communication Scheduling, Data Heterogeneity, Convergence Analysis

## TL;DR
In decentralized training with extreme communication constraints and high data heterogeneity, the authors find that a "global merging" (averaging all node models) at the end of training brings global test performance close to Federated Learning levels. They theoretically prove for the first time that the global merging model of decentralized SGD can achieve the convergence rate of parallel SGD—the key lies in reinterpreting the inter-node variance, previously considered "harmful noise," as a "constructive component" necessary for matching this rate.

## Background & Motivation

**Background**: Decentralized learning crowd-sources training tasks to geographically dispersed devices, relying on peer-to-peer communication between nodes to exchange model parameters. This is more scalable than solutions depending on central parameter servers. Its core constraint is bandwidth—communication between dispersed nodes is expensive and scarce, making the allocation of communication resources a fundamental problem in this field.

**Limitations of Prior Work**: Previous work on optimizing communication has almost exclusively focused on the "spatial dimension," i.e., designing communication topology graphs (which nodes connect to which). However, the "temporal dimension" of allocation—when and how frequently to synchronize—remains largely unstudied in fully decentralized settings. While temporal allocation has been explored in Federated Learning, that paradigm maintains a global model on a central server, leading to the intuition that "early, frequent communication is crucial for aligning local models," a heuristic that cannot be directly applied to decentralized settings without a central server.

**Key Challenge**: In heterogeneous environments where data is non-identically and independently distributed (non-IID) and $L(\cdot) \not\equiv L_k(\cdot)$, the goal is no longer optimizing a universal global model but ensuring local models can generalize to the global distribution. Intuitively, low communication combined with data heterogeneity leads to massive divergence between node models, making them "unmergeable" and resulting in poor performance.

**Goal**: To answer a specific question: how should the communication budget be allocated over time in decentralized learning? And to explain why such an allocation is effective.

**Key Insight**: The authors designed a series of experiments concentrating the communication budget into different time windows and unexpectedly observed that the later the communication is shifted in the training process, the better the final test performance; more extremely, even a single all-to-all communication (a single global merging) at the final step can dramatically improve performance.

**Core Idea**: Use a "single global merging" to replace "continuous frequent communication" and prove that inter-node variance is not pure noise but a constructive force that helps the merged model match the parallel SGD rate through "progressive sharpening."

## Method

This paper does not propose a new algorithm but rather "discovers a counter-intuitive phenomenon and provides the first convergence theory to explain it." The logic chain is: observe the benefits of "late communication + single merging" via controlled experiments, formalize the concept of "mergeability" and verify it throughout training via counterfactual experiments, and finally establish a convergence analysis proving that the global merging model can catch up with parallel SGD, thereby explaining "why communication should be placed in the final stages."

### Overall Architecture

Decentralized training follows a standard process (Algorithm 1): each node $k$ performs a local update on its local distribution $D_k$ as $\theta_k^{(t+1/2)} \leftarrow \text{Optimizer}(\theta_k^{(t)}, \xi_k^{(t)})$, followed by a gossip average with neighbors via a mixing matrix $W^{(t)}$ as $\theta_k^{(t+1)} \leftarrow \sum_{l} W_{k,l}^{(t)} \theta_l^{(t+1/2)}$. The authors build upon this framework with two components:

First, **Communication Scheduling Experiments**: Training is divided into several consecutive windows. All-to-all communication (AllReduce global synchronization) is enabled only within a selected window; otherwise, each node communicates with a random peer with a probability of 0.2 (low communication). By shifting the high-communication window, they observe its impact on final performance, finding that later is better, eventually pushing this to the limit: a single global merging at the final step.

Second, **Mechanism Analysis**: They define two metrics—global test accuracy and mergeability—using a "counterfactual global average model" to track whether local models are mergeable at various training stages. Convergence theory is then used to explain why limited but non-zero communication maintains mergeability and why communication should be concentrated in the late stages. This part consists of mechanical/theoretical analysis (matrix operations and convergence rate derivations) rather than a clear multi-stage pipeline.

### Key Designs

**1. Temporal Communication Allocation: Concentrating Budget in Late Training**

Addressing the scarcity of communication resources, the authors divide training into 10 or 20 equal windows, enabling AllReduce only in one window while maintaining low communication (probability 0.2 of connecting to a random peer) in others. By scanning through windows, they find a consistent trend: allocating the high-communication budget to later windows results in more significant final test accuracy improvements. This contradicts the mainstream consensus in Federated Learning that "early frequent communication is needed for alignment"—in decentralized non-IID settings, early local models are descending individually and forced alignment is less meaningful; instead, the final synchronization is most critical.

**2. Single Global Merging: Equaling Federated Learning with One Average**

Compressing the high-communication window to its limit—performing only one global merging (averaging all node parameters, equivalent to one AllReduce) at the very end of training. On CLIP ViT-B/32 and ResNet-18 with 32 nodes and high heterogeneity (Dirichlet $\alpha=0.1$), this single merging suffices to boost global test performance to levels near Federated Learning. The authors emphasize the **non-triviality** of this gain (Remark 2): although approximately 60 random peer exchanges occur under low communication, which might seem equivalent to multiple implicit global aggregations, the local models before merging still perform close to the "zero communication" baseline; the performance only surges after merging. This indicates that the effect of a single merging is not a simple accumulation of sparse gossip but a qualitative leap. Regarding communication costs, standard Ring-AllReduce is $O(2mPT)$, whereas this approach is $O(mRPT + 2mP)$ where $R \ll 2$, significantly reducing overhead.

**3. Mergeability: Limited but Non-zero Communication as a Switch**

The authors formalize "mergeability" to characterize this phenomenon. A set of local models $\{\theta_k\}$ is **globally mergeable** under the global risk $L(\cdot)$ if there exist weights $\{w_k\} \in [0,1]$ such that

$$L\!\left(\sum_{k\in V} w_k \theta_k\right) \le \sum_{k\in V} w_k L(\theta_k),$$

meaning the linearly interpolated model is no worse than the original local models—this property is non-trivial due to the non-convexity of $L$. To verify this, they compute a "counterfactual global average model" at every round (manual averaging without actual merging). They find that under low communication ($p=0.2$ to a random peer), the merged model curve consistently outperforms the local model curves, indicating mergeability throughout all stages. Conversely, in ablation studies with purely local training (zero communication), the counterfactual merged model accuracy is near zero, proving that mergeability is **not an inherent property** of local models. Crucially, the surge in performance upon merging demonstrates that extremely limited but non-zero communication acts as the switch for "mergeability." The authors note this as **mergeability without consensus**: local models do not converge to the same point but are guided into a "ring-like high-loss region" surrounding a central low-loss basin (Figure 1c).

**4. Convergence Theory: Reclassifying Node Variance from "Noise" to "Constructive Component"**

This theoretical core explains the observed phenomena. Previous analyses of decentralized SGD (DSGD) treated extra terms—gradient noise and inter-node parameter variance—as harmful terms to be controlled, leading to a penalty of $O\!\big(\tfrac{1-p}{p\varepsilon} + \tfrac{\sqrt{p}\,\sigma+\zeta}{p\varepsilon^{3/2}}\big)$ over parallel SGD. This paper adopts a new proof framework (Theorem 1) that interprets the variance partially as a constructive force, yielding the rate:

$$T = O\!\left(\frac{\sigma^2}{m\varepsilon^2} + \frac{1}{\varepsilon} + \frac{1}{\varepsilon}\sum_{t=0}^{T-1} U^{(t)}\right)\!\big(L(\theta^{(0)}) - L^\star\big),$$

where $U^{(t)}$ couples consensus distance $\Xi_t^2 = \mathrm{Tr}(\Gamma^{(t)})$ (the covariance of node variance) with high-order geometric terms. The authors introduce the **Progressive Sharpening Assumption** (Assumption 4): $\nabla L(\theta)^\top \nabla \mathrm{Tr}(\nabla^2 L(\theta)\Sigma) < 0$, suggesting that the optimizer moves toward regions of increased sharpness while decreasing loss. Under this assumption and $\eta > 1/L_2$, Proposition 2 proves $U^{(t)} < 0$—meaning node variance actually aids convergence. Intuitively (descent lemma, Eq. 9), the progressive sharpening term grows with $O(\Xi_t^2)$ while higher-order residuals are $O(\Xi_t^3)$; if $\Xi_t$ is controlled such that second-order gains outweigh third-order errors, the merged DSGD model matches or exceeds parallel SGD. When consensus error is zero (single node or perfect parallel SGD), $U^{(t)}\equiv 0$, and the theory reverts to the standard SGD rate.

The boundedness of $\Xi_t$ is determined by the connectivity parameter $p$: $\mathbb{E}[\Xi_t^2] \le O\!\big(\tfrac{1-p}{p^2}\big)$. Random communication graphs achieve $p = \Theta(1)$, maintaining efficient information mixing at low cost. Finally, Proposition 3 maps the condition to a relationship between $\Xi_t$ and the global gradient lower bound $\mu_t$ ($\|\nabla L(\bar\theta^{(t)})\| \ge \mu_t$): the constraint is loose early in training when $\mu_t$ is large, allowing for infrequent communication, but tightens late in training as $\mu_t$ decreases, necessitating more frequent communication.

### Loss & Training
No new loss functions are proposed. Standard SGD or AdamW optimizers are used; the theoretical analysis focuses on DSGD. Key hyperparameters include a communication probability $R = 0.2$, node count $m \in \{16, 32\}$, data heterogeneity Dirichlet $\alpha = 0.1$, and the learning rate must fall within the "oscillatory convergence" interval $\tfrac{1}{L_2} < \eta < \tfrac{2}{L_2}$ to ensure $U^{(t)} < 0$.

## Key Experimental Results

### Main Results
Under the Tiny ImageNet setting with 32 nodes, non-IID ($\alpha=0.1$), and a probability of 0.2 for connecting to a random peer per round, a global merging was performed at the end of training:

| Setting | Model | Before Merging (Local) | After Merging (Single Global) |
|------|------|-------------------|----------------------|
| Decentralized + Single Merging | CLIP ViT-B/32 | Near zero-comm baseline (Poor) | Massive boost, near FL |
| Decentralized + Single Merging | ResNet-18 (No Pre-train) | Near zero-comm baseline (Poor) | Massive boost, near FL |

Communication cost comparison ($P$ model size, $m$ number of nodes, $T$ rounds):

| Scheme | Total Communication Cost |
|------|-----------|
| Standard Ring-AllReduce | $O(2mPT)$ |
| Ours (Sparse gossip + single final merging) | $O(mRPT + 2mP)$, $R \ll 2$ |

Convergence rate comparison (non-IID, $m$ nodes):

| Algorithm | Rate |
|------|------|
| Parallel SGD | $O(\sigma^2/m\varepsilon^2 + 1/\varepsilon)$ |
| DSGD (Koloskova 2020) | $O(\sigma^2/m\varepsilon^2 + 1/p\varepsilon + \tfrac{\sqrt p\sigma+\zeta}{p}\varepsilon^{-3/2})$ |
| DSGD (Ours) | $O(\sigma^2/m\varepsilon^2 + 1/\varepsilon + \tfrac1\varepsilon\sum_t U^{(t)})$ with $U^{(t)}<0$ |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Low comm ($p>0$) + Final Merging | Counterfactual merge curve consistently outperforms local models | Local models mergeable at all stages |
| Zero comm (Purely local $p=0$) | Counterfactual merge accuracy ≈ 0 | Mergeability requires communication |
| Comm window placement | Later windows yield higher final accuracy | Verifies "late communication is critical" |
| Window size: 1/10 → 1/20 → Single step | Gains remain significant | A single merging is sufficient |

### Key Findings
- The gain from a single global merging is **non-trivial**: approximately 60 random peer exchanges do not improve local performance, yet a massive gap exists between models before and after merging, proving it is not a linear result of sparse gossip.
- "Limited but non-zero communication" is the switch for mergeability: at $p=0$, $\Xi_t$ diverges and models cannot be merged; as long as $p=\Theta(1)$, $\Xi_t$ is stabilized and mergeability is preserved.
- The empirical temporal observation (concentrating communication late) aligns perfectly with the theory ($\mu_t$ decreases late, tightening constraints and requiring more communication).

## Highlights & Insights
- **Reclassifying "harmful noise" as "constructive"**: The most elegant aspect is the theoretical inversion—shifting from suppressing $\Xi_t$ to using progressive sharpening to move $O(\Xi_t^2)$ to the favorable side of the convergence rate, proving decentralized merged models can match parallel SGD.
- **Minimalist yet counter-intuitive solution**: By changing nothing except adding a final average, high-heterogeneity decentralized training is rescued to near-FL levels, suggesting decentralized learning potential is severely undervalued.
- **Guidance for adaptive communication**: Proposition 3 links communication frequency $p$ with the real-time gradient $\mu_t$, providing a blueprint for adaptive communication scheduling based on training dynamics.
- **Insights for model merging**: Decentralized training may guide nodes into "connected basins of attraction," enabling simple permutation-free merging.

## Limitations & Future Work
- The theory relies on the Progressive Sharpening Assumption (Assumption 4) and a gradient lower bound $\mu_t > 0$, which are empirical observations in deep networks rather than strict guarantees.
- Global merging might be difficult to implement in some purely decentralized scenarios (requiring an AllReduce step); the authors suggest multi-round gossip as an approximation, but the resulting loss is not fully quantified.
- Experimental scale is limited to 16/32 nodes and CIFAR-100 / Tiny ImageNet; the effectiveness of "single merging" on LLM-scale decentralized pre-training remains to be verified.
- The temporal allocation is currently an empirical conclusion from offline scans; while a theoretical criterion is provided, an end-to-end adaptive scheduling algorithm has not yet been implemented.

## Related Work & Insights
- **vs D-PSGD (Lian et al. 2017)**: D-PSGD used final merging in IID settings without analyzing the performance gap; this work systematically studies this "recovery" in non-IID settings with theoretical grounding.
- **vs Periodic Global Averaging (Chen et al. 2021)**: They required frequent global communication (every $H=48$ steps); this work recovers performance with just one merging.
- **vs SCSP (Aketi et al. 2021)**: SCSP also uses final merging but employs gradient sparsification (top-k) and fixed topologies; this work utilizes topological sparsification (sparse gossip) and supports robust mergeability under many local steps (e.g., $H=100$).
- **vs FL Temporal Allocation (Wang et al. 2019)**: FL maintains a central global model and advocates for early frequent communication; this work reaches the opposite conclusion for decentralized settings—communication should be saved for the end.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Counter-intuitive phenomenon combined with a novel theory matching decentralized and parallel rates.
- Experimental Thoroughness: ⭐⭐⭐⭐ Consistent validation across datasets/architectures, though lacking large-scale model scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from phenomenon to definition to theory, though theoretical parts have a high entry barrier.
- Value: ⭐⭐⭐⭐⭐ Significantly reduces communication costs for decentralized training and opens new directions for model merging and adaptive scheduling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[ICML 2026\] SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation](../../ICML2026/optimization/symerge_from_non-interference_to_synergistic_merging_via_single-layer_adaptation.md)
- [\[AAAI 2026\] A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](../../AAAI2026/optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)
- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Single-Loop Byzantine-Resilient Federated Bilevel Optimization](single-loop_byzantine-resilient_federated_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
