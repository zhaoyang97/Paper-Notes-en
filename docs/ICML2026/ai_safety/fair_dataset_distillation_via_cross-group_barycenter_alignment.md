---
title: >-
  [Paper Note] Fair Dataset Distillation via Cross-Group Barycenter Alignment
description: >-
  [ICML 2026][AI Safety][Dataset Distillation] This paper reveals that Dataset Distillation (DD) amplifies biases in original data—rooted in the interaction between "subgroup size imbalance" and "subgroup representation se…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Dataset Distillation"
  - "Group Fairness"
  - "Subgroup Barycenter Alignment"
  - "EOD"
  - "Representation Alignment"
date: 2026-05-08
content_hash: abe519167a117405
---

# Fair Dataset Distillation via Cross-Group Barycenter Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.00185](https://arxiv.org/abs/2605.00185)  
**Code**: No public link  
**Area**: Dataset Distillation / Fair Machine Learning / AI Safety  
**Keywords**: Dataset Distillation, Group Fairness, Subgroup Barycenter Alignment, EOD, Representation Alignment

## TL;DR
This paper reveals that Dataset Distillation (DD) amplifies biases in original data—rooted in the interaction between "subgroup size imbalance" and "subgroup representation separation." It proposes COBRA: using the (group-size-independent) barycenter of subgroup representations as the distillation target, which simultaneously reduces EOD and improves accuracy across multiple DD frameworks.

## Background & Motivation
**Background**: Dataset distillation compresses thousands of real samples into a few synthetic images, allowing downstream models trained on the synthetic set to approach full-set performance. Mainstream methods (DC, IDC, DM, CAFE, MTT, etc.) typically select a "representation" $\phi(x;\theta)$ (gradients, embeddings, features, trajectories) for each class $y$ and align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to the $\Phi_{T_y}$ of the real set.

**Limitations of Prior Work**: When the training set contains subgroup structures based on protected attributes $A\in\mathcal{A}$ (gender, skin color, age, etc.) and subgroup representation patterns differ, simply aligning $\Phi_{S_y}$ to the "mean statistics of all samples" $\Phi_{T_y}=\sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ is dominated by majority subgroups. Consequently, minority subgroups are barely represented in the synthetic set. Downstream models trained on such synthetic sets show significant drops in conditional accuracy for minority subgroups, leading to increased Equalized Odds Difference (EOD).

**Key Challenge**: The authors emphasize a neglected fact: "Uniform subgroup sampling" (Uniform DD) cannot solve the problem because bias arises from the **product of two independent factors**: (i) subgroup size imbalance where $\pi_{a\mid y}$ deviates from uniform, and (ii) subgroup representation separation where $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$ is large. The presence of either factor leads to an imbalance in the residual $\Delta_{a\mid y}^*=\Phi_{T_{a\mid y}}-\Phi_{S_y}^*$.

**Goal**: (1) Formally decompose the two sources of bias amplification in DD and provide an upper bound; (2) Design a target with a tight upper bound that ensures "equidistant" synthetic representations for all subgroups; (3) Maintain compatibility with existing DD frameworks (DC, DM, CAFE, IDC) by only replacing the matching target.

**Key Insight**: Under MSE distance, the fixed point of standard DD is $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ (weighted average), which naturally favors majority groups. By using the **barycenter $m_y^* = \arg\min_m \sum_a d(\Phi_{T_{a\mid y}}, m)$**, which is independent of weights $\pi$, the distance gap from the synthetic set to each subgroup can be tightened.

**Core Idea**: Replace the "subgroup weighted mean" with the "cross-group barycenter" as the class-conditional distillation target. This removes the dependency on group size from the objective, preventing the majority groups from inflating the worst-case subgroup residual.

## Method

### Overall Architecture
COBRA is a two-step process: (1) Within each class $y$, calculate the class-conditional subgroup statistics $\Phi_{T_{a\mid y}}=\frac{1}{|T_{a\mid y}|}\sum_{x\in T_{a\mid y}}\phi(x;\theta_T)$ for real data based on subgroups $a\in\mathcal{A}$, then compute the barycenter $m_y^*$ between subgroups (using uniform weights $w_a=1/|\mathcal{A}|$ under a suitable distance $d$); (2) Align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to $m_y^*$ instead of the original $\Phi_{T_y}$. The loss becomes $\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*, \Phi_{S_y})$. This framework is agnostic to the specific form of $\phi$ (gradients/embeddings/features/trajectories), allowing it to be integrated into method like DC, DM, CAFE, and IDC.

### Key Designs

1.  **Formal Decomposition of Bias Mechanism**:
    - **Function**: Traces EOD degradation in DD to a provable upper bound, highlighting that group imbalance and representational separation must be considered jointly.
    - **Mechanism**: Under MSE distance, the fixed point via SGD is $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$. The residual is $\Delta_{a\mid y}^* = \sum_{a'\neq a}\pi_{a'\mid y}(\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}})$, thus $\|\Delta_{a\mid y}^*\|_2 \leq \sum_{a'\neq a}\pi_{a'\mid y}\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|_2$. Both factors are essential.
    - **Design Motivation**: Previous work (FairDD) attributed the problem solely to group imbalance. This paper uses dual-axis controlled experiments in Figure 2 to prove that varying separation with fixed imbalance, or varying imbalance with fixed separation, both independently increase EOD. Thus, a unified correction is necessary.

2.  **Cross-group Barycenter $m_y^*$ as the New Target**:
    - **Function**: Ensures the distillation target is as equidistant as possible to each subgroup, minimizing the maximum residual and geometrically dismantling the interaction terms in the upper bound.
    - **Mechanism**: For $d(u,v)=\|u-v\|_Q^2$ (positive definite $Q$), the inner optimization $m_y^* = \arg\min_m \sum_a \|\Phi_{T_{a\mid y}}-m\|_Q^2$ has the closed-form solution $m_y^* = \frac{1}{|\mathcal{A}|}\sum_a \Phi_{T_{a\mid y}}$—the subgroup-level uniform average, completely independent of $\pi_{a\mid y}$. This contrasts with the $\pi$-weighted average in vanilla DD.
    - **Design Motivation**: Uniform weights $w_a$ decouple the target from subgroup size; the barycenter minimizes the total distance to all subgroups, acting as a "most equitable center."

3.  **Theoretical Guarantee: Non-increasing Worst-case Residual**:
    - **Function**: Theorem 4.1 rigorously demonstrates that COBRA does not make the worst subgroup worse than vanilla DD.
    - **Mechanism**: Defining $s_y = m_y^\text{van}-m_y^*$ as the imbalance shift, if the worst subgroup $a^\dagger$ satisfies $\langle \Delta_{a^\dagger\mid y}^C, s_y\rangle_Q \leq 0$ (a mild geometric condition where the worst subgroup is oriented opposite to the imbalance shift), then $\max_a \|\Delta_{a\mid y}^C\|_Q \leq \max_a \|\Delta_{a\mid y}^V\|_Q$.
    - **Design Motivation**: Empirically, FairDD averages per-group loss but parameter updates can still drift. COBRA directly tightens the worst-case residual, a geometric quantity directly related to EOD, elevating "fairness" from loss averaging to representation alignment.

### Loss & Training
$\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*,\Phi_{S_y})$. Compared to vanilla DD, only the alignment target is changed. Since the barycenter has a closed-form solution under $\|u-v\|_Q^2$, no inner-loop optimization is required, maintaining efficiency comparable to vanilla DD. Other hyperparameters (IPC, architecture, initialization, distance $D$) follow the default settings of the backbone DD (DC/DM/CAFE/IDC).

## Key Experimental Results

### Main Results

| Dataset | Backbone | IPC | Vanilla EOD/Acc | FairDD EOD/Acc | COBRA EOD/Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CIFAR10-S | DM | 100 | 82.87 / 45.4 | 25.17 / 61.2 | **9.37 / 62.4** |
| CIFAR10-S | DC | 50 | 71.85 / 39.5 | 35.65 / 46.2 | **26.18 / 46.6** |
| C-MNIST (BG) | DM | 50 | 100.0 / 48.8 | — | **7.46 / 96.8** |
| BFFHQ (Real) | DM | 100 | 63.47 / 65.8 | — | **7.87 / 74.2** |
| Full baseline | — | — | EOD 48.96 / Acc 69.71 (CIFAR10-S) | — | — |

(Values from Table 1; COBRA consistently reduces EOD while improving/maintaining Acc across all IPCs and backbones.)

### Ablation Study

| Configuration | Key Result | Description |
| :--- | :--- | :--- |
| Choice of distance $d$ | MSE/cosine/MMD all work; MSE is most efficient | Robustness to $d$ |
| Backbone (DC/DM/CAFE/IDC) | Validated across 4 DD paradigms | Orthogonal and plug-and-play |
| Real vs Synthetic baseline | Vanilla DD has higher EOD than Full training | Specifically confirms DD amplifies original bias |
| Uniform Subgroup Distillation | Worse when subgroups are close in representation | Mirroring imbalance alone is insufficient |
| Varying imbalance/separation | Both curves independently increase EOD | Validates the "dual-factor interaction" source of bias |

### Key Findings
- **DD Amplifies Original Bias**: On CIFAR10-S, Vanilla DD EOD is much higher than Full training (e.g., 82.87 vs 48.96 at IPC=100), quantifying that distillation fails to preserve fairness and can exacerbate it.
- **Smaller IPC Leads to Greater Amplification**: Reducing IPC limits minority group capacity, strengthening spurious correlations and sharply increasing EOD.
- **Barycenter as a Geometric Fair Solution**: Worst-case residuals are directly linked to subgroup-level error variance in EOD. Fairness is addressed at the "representation geometric center" level rather than just "weighted loss averaging."
- **Cross-dataset Universality**: Significant gains are maintained from synthetic C-MNIST/CIFAR10-S to real UTKFace/BFFHQ, compatible with four backbone DD methods.

## Highlights & Insights
- Formally decomposes the DD fairness problem into a product of imbalance $\times$ separation and provides a dual-axis controlled experimental methodology, establishing a clear standard for future fair DD research.
- The concept of "barycenter as target" is borrowed from optimal transport/clustering but introduced to DD with almost zero computational overhead (closed-form under MSE), offering both academic elegance and engineering friendliness.
- The theoretical condition $\langle\Delta,s\rangle_Q \leq 0$ is geometrically intuitive, suggesting that the most disadvantaged subgroup should lie in the opposite direction of the imbalance shift.
- Can be immediately embedded into any representation-matching DD method (DC/DM/CAFE/IDC/MTT) with minimal adoption cost.

## Limitations & Future Work
- Assumes protected attributes $A$ are observable during training and subgroup labels are available; in practice, medical data often lacks subgroup labels due to privacy.
- If a subgroup's representation is "mean-unobtrusive but high-variance," a simple mean barycenter might hide distributional differences; distribution-level (Wasserstein) barycenters should be considered.
- Theorem 4.1 focuses on the worstcase and does not provide a tight bound for expected EOD improvement; the coupling with downstream model selection is not yet characterized.
- Only EOD is discussed; the impact on other fairness criteria like demographic parity or equal opportunity remains unexplored.
- Scalability to ImageNet-scale data with large IPC and compatibility with trajectory matching (MTT) require further validation.

## Related Work & Insights
- **vs FairDD (Zhou et al., 2025)**: They perform per-group loss averaging to fix imbalance; this paper proves that the separation factor alone also amplifies EOD, and COBRA consistently outperforms FairDD across datasets.
- **vs Standard DD (DC/DM/CAFE/IDC/MTT)**: Only changes the alignment target $m_y^*$ without modifying the backbone algorithm, proving fairness gains are orthogonal to representation matching paradigms.
- **vs Long-tail DD (Cui 2024, Lu 2024, Zhao 2025)**: Those works focus on class imbalance, while this paper addresses subgroup-level protected-attribute imbalance, which is a finer-grained fairness.
- **vs Barycenter in Fair ML (Gordaliza 2019, Charpentier 2023)**: They use OT barycenter preprocessing in the original data space; this paper uses barycenter distillation in the representation space, which is more lightweight and suitable for DD pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ Precision in characterizing the interaction of "imbalance $\times$ separation" and providing a geometric solution is a significant conceptual advancement for fair DD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 backbones $\times$ multiple IPCs $\times$ 5 datasets (synthetic + real) with systematic ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation of bias mechanisms; Theorem and Figures complement each other well.
- Value: ⭐⭐⭐⭐ Plug-and-play with low entry barriers; practical significance for fairness in high-stakes deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[CVPR 2026\] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](../../CVPR2026/ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)

</div>

<!-- RELATED:END -->
