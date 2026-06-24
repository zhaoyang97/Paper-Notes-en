---
title: >-
  [Paper Note] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity
description: >-
  [ICLR2026][Learning Theory][Loss of Plasticity] This paper redefines "Loss of Plasticity (LoP)" from a dynamical systems perspective as gradient trajectories being trapped within **invariant submanifolds** of the parameter space. It proves that frozen and cloned units form such "trap manifolds" and presents a counter-intuitive conclusion: the very low-rank compression mechanisms that promote generalization in static tasks are what drive the network into these plasticity-deple…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Continual Learning"
  - "Optimization Dynamics"
  - "Loss of Plasticity"
  - "Invariant Manifold"
  - "Gradient Dynamics"
  - "Effective Rank"
  - "Neural Collapse"
date: 2026-05-08
content_hash: 0f9c6a769fd5d3ce
---

# Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=g6kof5fSba](https://openreview.net/forum?id=g6kof5fSba)  
**Code**: https://github.com/ajoudaki/loss-of-plasticity  
**Area**: Learning Theory / Continual Learning / Optimization Dynamics  
**Keywords**: Loss of Plasticity, Invariant Manifold, Gradient Dynamics, Effective Rank, Neural Collapse

## TL;DR
This paper redefines "Loss of Plasticity (LoP)" from a dynamical systems perspective as gradient trajectories being trapped within **invariant submanifolds** of the parameter space. It proves that frozen and cloned units form such "trap manifolds" and presents a counter-intuitive conclusion: the very low-rank compression mechanisms that promote generalization in static tasks are what drive the network into these plasticity-depleted manifolds.

## Background & Motivation
**Background**: Deep networks perform exceptionally well under stationary data distributions. However, in non-stationary environments like continual learning or lifelong learning where distributions drift, they exhibit the phenomenon of being "unable to learn new things," known as Loss of Plasticity (LoP). This is distinct from catastrophic forgetting: forgetting erases old knowledge, while LoP is the **inability to acquire new knowledge**—performance on old tasks may remain intact, but the ability to learn new tasks drops sharply.

**Limitations of Prior Work**: Existing literature mostly stays at the level of **describing symptoms**: exploding weight norms, the emergence of "dead units" (saturated units), and the collapse of the effective rank of representations. Some works attribute LoP to degraded backpropagation mechanisms or the Neural Tangent Kernel (NTK) becoming low-rank. However, these are phenomenological—they characterize what a failed network looks like but fail to answer the **core mechanical question**: Why can't gradient descent recover? If LoP is merely a poor parameter configuration, why doesn't the gradient push the model back into useful regions, and why does the network fail to restore feature diversity under new task distributions?

**Key Challenge**: The authors argue that LoP is not a "degradation of statistics (rank, weight norm)" but a **topological trap**—once the optimization trajectory enters certain geometric structures, the gradient flow is strictly confined within them. A deeper contradiction exists: the forces that make a network generalize well on the current task (feature compression into low ranks, neural collapse) are precisely the forces that push it into these traps. "Excellent student behavior" on the current task effectively builds the walls for future tasks.

**Goal**: (1) Provide a **formal definition** of LoP based on dynamical systems that transcends mere symptoms; (2) Identify specific geometric mechanisms that create these traps and prove that gradient descent cannot escape them; (3) Explain **why these traps form spontaneously** during normal training; (4) Provide intervention methods to break the traps and restore plasticity.

**Key Insight**: Treating gradient descent as a trajectory in parameter space, the authors seek geometric structures that act as "sinks" using dynamical systems theory—specifically, invariant submanifolds that once entered, cannot be exited.

**Core Idea**: Formalize LoP as **LoP Manifolds**—invariant submanifolds in the parameter space where the loss gradient is everywhere tangent to the manifold. Consequently, once a gradient flow enters, it remains there forever unless external perturbations intervene.

## Method

### Overall Architecture
The entire paper follows a theoretical chain of "Definition $\to$ Mechanism $\to$ Cause $\to$ Remedy," rather than proposing a specific trainable model. The logic: first **define** LoP manifolds within a dynamical systems framework (invariant subspaces where gradients are always tangent); then **prove** the existence of two specific manifolds—the frozen unit manifold $\mathcal{M}_F$ and the cloned unit manifold $\mathcal{M}_C$—showing that standard gradient optimization cannot exit them; next, explain **where these manifolds come from** via a "rank gain vs. neural collapse" tension theorem, illustrating that low-rank compression naturally pushes networks toward these manifolds; finally, provide methods on **how to escape**—normalization for prevention, and noise/dropout/continual backprop to break symmetry.

Think of LoP as low-dimensional "reefs" $\mathcal{M}\subset\Theta$ embedded in the parameter space $\Theta$. In independent tasks, the trajectory explores the full space, but in continual learning, early tasks push the parameters onto the reef. Once grounded, the gradients of subsequent tasks become tangent to the reef, locking future dynamics into this restricted subspace and preventing the recovery of plasticity needed for new distributions.

### Key Designs

**1. LoP Manifold: Redefining "Loss of Plasticity" from Statistical Degradation to Topological Trap**

To address the pain point of "only describing symptoms without explaining irreversibility," the authors provide a formal definition (Def 2.1): A manifold $\mathcal{M}\subset\Theta$ induces LoP if and only if the loss gradient is tangent to the manifold at every point, i.e., $\nabla_\theta L(\theta)\in T_\theta\mathcal{M}$ for all $\theta\in\mathcal{M}$, where $T_\theta\mathcal{M}$ is the tangent space. This tangency condition ensures that the continuous-time gradient flow $\dot\theta=-\nabla_\theta L(\theta)$ starting in $\mathcal{M}$ stays in $\mathcal{M}$ forever. For general curved manifolds, discrete updates $\theta_{k+1}=\theta_k-\eta_k\nabla_\theta L$ might escape due to discretization steps; however, for **affine LoP manifolds**, the gradient flow and (stochastic) gradient descent are strictly invariant, making discrete steps inescapable.

The most critical concept is **functional LoP**: if the tangency condition holds **independently of the specific data distribution**, the trap is determined solely by the network architecture and gradient dynamics. **No change in tasks or distributions can save it**—this is the most lethal type in continual learning. This step translates the vague "network can't learn" into a provable, analyzable geometric object.

**2. Two Types of Trap Manifolds: Frozen Units $\mathcal{M}_F$ and Cloned Units $\mathcal{M}_C$, with Generalization to Equitable Partitions**

Definition alone is insufficient; existence in real networks must be proven. Theorem 2.1 identifies two types of affine LoP manifolds based on the network's computational DAG $G=(V,E)$:

- **Frozen Unit Manifold $\mathcal{M}_F$**: If certain units $v\in F$ are continuously saturated for all finite inputs (derivative $f'(z_v)=0$, e.g., large $\|\theta_{in(v)}\|$ in tanh or highly negative bias in ReLU), all parameter gradients flowing into these units are constant 0. These coordinates are frozen. Writing these linear constraints as $\theta_{in(v)}=\text{const}$, the resulting affine subspace $\mathcal{M}_F:=\{\theta:\theta_{in(v)}=\text{const}\ \forall v\in F\}$ satisfies $\nabla L(\theta)\in T_\theta\mathcal{M}_F$. Once GD/SGD enters, it cannot exit.

- **Cloned Unit Manifold $\mathcal{M}_C$**: Partitioning nodes into disjoint blocks $\{S_1,\dots,S_k\}$, requiring every pair of blocks $(S_i,S_j)$ to satisfy "equal row sum and equal column sum" constraints—$\sum_{v\in S_j}\theta_{uv}$ is equal for different $u,u'\in S_i$ within the block, and vice-versa for columns. When satisfied: (i) units in the same block have identical forward values for any input, (ii) backpropagation errors are identical, and (iii) gradients for every edge connecting the same pair of blocks are equal. This means gradients naturally satisfy the row/column sum constraints and are tangent to $\mathcal{M}_C$, keeping updates within the manifold.

A **key generalization** emphasized: previous analyses of singularities (Fukumizu & Amari) or invariant sets (Chen et al.) required cloned unit weights to be **strictly equal**. This paper proves that as long as **equitability (equal row/column sums)** is satisfied—a much looser condition—invariance holds. Individual weights can differ if specific input/output weight sums are conserved. This significantly broadens the class of structures identified as LoP manifolds. Theorem 2.2 provides a **modular version**: if each module locally satisfies "cloned inputs $\to$ cloned outputs (forward invariance), cloned backprop $\to$ cloned backprop (backward invariance), and updates preserve invariance (persistence)," and cloned partitions at interfaces are consistent, the whole network falls into a cloned manifold. This allows for "cloning certificates" in modern architectures like ResNet/ViT. The authors also note that SGD, Momentum, and Adam cannot escape if initialized when cloning occurs; weight decay is the only exception that might break certain symmetries.

**3. Rank-Plasticity Tension: The Forces for Generalization are the Forces Driving LoP**

Having proven the existence of traps, the paper addresses **why they form spontaneously**. Theorem 3.1 (Rank Gain and Decorrelation Potential) links "rank dynamics" to "plasticity." Using differentiable rank proxies like the Rényi-2 effective rank $\mathrm{er}_2(M)=(\mathrm{tr}\,M)^2/\|M\|_F^2$ or Shannon effective rank $\mathrm{er}(M)=\exp(H(\lambda(M)/\mathrm{tr}\,M))$, the authors show that for normalized non-linearities $\phi$, the rank gain satisfies:

$$\frac{\mathrm{er}_2(K_\phi(C))}{\mathrm{er}_2(C)}\ \ge\ 1+\gamma_\phi\frac{\Psi(C)}{\|C\|_F^2}.$$

Where $K_\phi(C)$ is the decorrelation kernel, $\gamma_\phi$ is the decorrelation strength, and $\Psi(C)$ is the decorrelation potential. If correlation coefficients fall in $(0,1)$, positive potential $\Psi(C)>0$ exists, which the non-linearity "consumes" to increase rank at a speed controlled by $\gamma_\phi\in[0,1]$. Tension arises here:

- **Cause of Frozen Units**: $\gamma_\phi$ is monotonic with saturation parameters—more negative ReLU bias or larger tanh gain increases $\gamma_\phi$. Thus, the "pursuit of feature diversity (high decorrelation)" creates pressure to push units into the vanishing derivative zone. **Training designed to recover rank simultaneously creates dead units.**
- **Cause of Cloned Units**: Neural collapse is a recognized training endpoint where intra-class variance vanishes and features collapse into a low-rank subspace. The theorem predicts that for rank to stabilize (ratio $\approx 1$) while retaining non-linearity ($\gamma_\phi>0$), $\Psi(C)\to 0$ must occur. This forces correlation coefficients to converge to $\{0,\pm 1\}$—i.e., orthogonal subspaces + cloned units, **exactly the geometry of LoP manifolds.**

The authors classify this as a pathology of the "Rich/Feature Learning" regime: the same mechanism that allows deep networks to learn efficient compressed representations eventually traps them in low-rank manifolds, damaging future adaptability (the "Lazy"/NTK regime avoids this because weights hardly move, maintaining high rank).

**4. Mitigation & Escape: Normalization for Prevention, Noise/Dropout/CBP to Break Symmetry**

Given the causes, the remedies are targeted, with **prevention and recovery following different logics**. Prevention relies on **Normalization** (BN/LN): by stabilizing pre-activation statistics, it keeps activations in the dynamic non-linear region, preventing drifts into saturation. Even with learnable affine parameters $(\gamma,\beta)$, they typically maintain pre-activations in "healthy" ranges. Normalization biases the representation's Gram matrix towards isometry (identity matrix), resisting the rank collapse associated with LoP. Experiments show it significantly maintains higher effective rank and suppresses dead/redundant units.

Recovery relies on **symmetry-breaking perturbations**: once a manifold is formed (large-scale freezing or cloning), "proactive" means like normalization are often insufficient to break established perfect clones. However, if LoP manifolds are unstable/saddle-like, perturbations provide escape directions. Noisy SGD adds Gaussian noise (proportional to gradient norm) to the gradient, making cloned unit backprop asymmetric and forcing parameters to differentiate. Dropout randomly zeros activations, causing different clones to activate under different masks, breaking both forward and backward symmetry. In MLP experiments, **single-step noise at only 0.01 of the relative gradient norm** was enough to initiate escape; however, in ViT, even after escaping, the trajectory did not move far. The authors warn that the effect of perturbations is **context-dependent**—dropout breaks symmetry in induced clones but might hinder the consolidation of new knowledge and exacerbate forgetting in continual learning bit-flipping tasks.

## Key Experimental Results

As a theoretical paper, experiments focus on numerical simulations to **validate theoretical predictions** (mostly line plots, no traditional SOTA comparison tables).

### Main Results: Symptoms Emerge Concurrently with Continual Learning
Continual training on 40 sequences of 5-class tasks derived from Tiny ImageNet (MLP/CNN/ResNet/ViT) showed that performance degradation occurs **synchronously** with LoP symptoms:

| Architecture | Dominant Symptom | Relation to Training Accuracy | Effective Rank |
| :--- | :--- | :--- | :--- |
| MLP | Ratio of Dead Units ↑ | Accuracy drops as dead units increase | Synchronous degradation |
| CNN | Dead / Cloned Units ↑ | Accuracy drop accompanied by unit degradation | Synchronous degradation |
| ResNet | Cloned Units ↑ | Accuracy degradation | Synchronous degradation |
| ViT | Ratio of Cloned Units ↑ (Most obvious) | Accuracy degradation | Synchronous degradation |

Conclusions align with Theorem 2.1/3.1: performance degradation is always **concurrent** with the emergence of dead/redundant units and a drop in representation diversity, with both paths (cloning, freezing) observed.

### Cloning & Escape Experiments (MLP)
A base model is trained, then "doubled" to create cloned blocks with identical activations, followed by further training:

| Optimization Setting | Cloning R² | Escape from Cloned Manifold |
| :--- | :--- | :--- |
| Clone + SGD | Stays ≈1 (Trapped) | No |
| Clone + Adam | Stays high (Trapped) | No (Validates Remark 2.4) |
| Clone + Noisy SGD | R² drops | Yes |
| Clone + Dropout | R² drops | Yes |

Cloning R² measures the proportion of variance in a block explained by the block mean (1 = perfect clone). Despite mechanical differences from SGD, Adam cannot escape the manifold, precisely confirming the theory.

### Intervention & Recovery

| Intervention | Observations | Mechanism |
| :--- | :--- | :--- |
| BN / LN (with/without affine) | Maintains higher effective rank throughout, suppresses dead/cloned units | Pre-activation normalization, Gram matrix biased toward isometry |
| SGD → CBP Switch (Bit-flipping, 5M samples, switch half-way) | Rank recovers, online training loss drops, plasticity regained | Continuous injection of randomness breaks symmetry |
| Model Scale ↑ | Gap between SGD and CBP widens | Scaling may exacerbate LoP symptoms |

### Key Findings
- **All gradient-based optimizers (SGD/Momentum/Adam) are trapped by cloned manifolds**; only weight decay might break symmetry—indicating traps stem from structural symmetry in gradient dynamics rather than optimizer preference.
- **Escape difficulty depends on architecture and manifold curvature**: MLPs escape with minimal noise, while ViTs fail to travel far after escaping, suggesting massive differences in normal curvature (stability) across manifolds.
- **Prevention $\neq$ Remedy**: Normalization prevents manifold formation but cannot break perfect clones once established; the latter requires noise/CBP. Dropout's role in continual learning can sometimes be a burden.

## Highlights & Insights
- **Translating "Inability to Learn" into Geometric Irreversibility**: Using the tangency condition ("gradients are everywhere tangent to the manifold"), the paper turns vague plasticity loss into a provable topological trap and distinguishes "data-independent" functional LoP—a brilliant conceptual leap.
- **Equitability Generalization is Crucial**: Relaxing cloning from "equal weights" to "equal row/column sums" means that numerous "approximately redundant but not identical" structures in reality are actually traps, explaining why LoP is so pervasive and stubborn.
- **Rank-Plasticity Tension is a Counter-intuitive Conclusion**: Low-rank compression and neural collapse, typically viewed as "generalization-friendly," are proven to be the forces driving networks into traps. This places static generalization and dynamic adaptability in opposition, providing clear guidance for continual learning—one must actively maintain/regenerate representation diversity.
- **Cross-domain Bridge**: By linking "unit cloning" in model compression with LoP in continual learning via the same theorems, it suggests that tools like CBP and noisy backprop could be used inversely for model expansion/escaping cloned states.

## Limitations & Future Work
- **Focus on Linear/Affine LoP Manifolds**: The authors explicitly state that the existence and occurrence of non-linear LoP manifolds remain open; discrete updates might escape curved manifolds, so theoretical guarantees only strictly apply to affine cases.
- **Missing Stability Analysis**: Whether a manifold is stable, unstable, or a saddle determines the ease of escape and its cost (unstable manifolds with flat normal curvature might require massive perturbations). The paper acknowledges that stability conditions for various manifolds and architecture/data biases toward specific types are not yet systematically characterized.
- **"Genuine Recovery" Unsolved**: After being rescued from a cloned state by noise/dropout, can a model explore the parameter space as effectively as a fresh initialization to find equally generalizable solutions? This relates to whether remedies are merely "treating symptoms."
- **Context-Dependent Interventions**: Dropout works in artificial clones but can be harmful in bit-flipping continual learning, indicating the lack of a universal plasticity maintenance strategy.

## Related Work & Insights
- **vs. Symptom-based Explanations (Nikishin / Sokar / Lyle et al.)**: Prior works attribute LoP to observable symptoms like weight explosion or dead units. This paper explains **why these symptoms are persistent and irreversible**—they are outward manifestations of LoP manifolds, not the root cause.
- **vs. Singularities / Invariant Set Analysis (Fukumizu & Amari 2000; Chen et al. 2023)**: Earlier work required strictly equal weights for cloned units to form singularities. This paper greatly generalizes this via equitability and explicitly joins "Loss of Plasticity" to this geometric lineage.
- **vs. Continual Backprop (Dohare et al. 2024)**: CBP is an empirical method of injecting randomness to maintain plasticity. This paper provides a dynamical systems explanation for "why noise allows escape" (breaking tangency constraints/symmetry) and positions CBP as a "remedy for established traps."
- **Inspiration**: Low-rank/simplicity bias is an advantage in static tasks but a liability in continual learning, suggesting that continual learning algorithms should explicitly incorporate "maintaining/regenerating representation rank" as an objective. Research lines in model compression and continual learning can share tools via the LoP manifold framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining LoP as topological traps on invariant manifolds and generalizing via equitability is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated theoretical predictions across MLP/CNN/ResNet/ViT, though primarily qualitative/plots without systematic quantitative comparison against existing mitigation methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear chain of Definition-Mechanism-Cause-Remedy; theorems are well-supported by intuitive explanations.
- Value: ⭐⭐⭐⭐⭐ Provides a provable mathematical characterization of the core obstacle in continual/lifelong learning and points toward "active representation diversity maintenance" as a design direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Correlated Reward Models: Statistical Barriers and Opportunities](learning_correlated_reward_models_statistical_barriers_and_opportunities.md)
- [\[ICLR 2026\] PAC-Bayes Bounds for Cumulative Loss in Continual Learning](pac-bayes_bounds_for_cumulative_loss_in_continual_learning.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)

</div>

<!-- RELATED:END -->
