---
title: >-
  [Paper Note] Fair Dataset Distillation via Cross-Group Barycenter Alignment
description: >-
  [ICML 2026][AI Safety][Dataset Distillation] This paper reveals that Dataset Distillation (DD) amplifies biases present in the original data—a phenomenon rooted in the interaction between "subgroup size imbalance" and "subgroup representation separation." It proposes COBRA: using the (group-size independent) barycenter of each subgroup's representation as the distillation target, which simultaneously reduces EOD and improves accuracy across multiple DD frameworks.
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Dataset Distillation"
  - "Group Fairness"
  - "Subgroup Barycenter Alignment"
  - "EOD"
  - "Representation Alignment"
date: 2026-05-08
content_hash: 9417dde8f963fd3b
---

# Fair Dataset Distillation via Cross-Group Barycenter Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.00185](https://arxiv.org/abs/2605.00185)  
**Code**: No public link  
**Area**: Dataset Distillation / Fair Machine Learning / AI Safety  
**Keywords**: Dataset Distillation, Group Fairness, Subgroup Barycenter Alignment, EOD, Representation Alignment

## TL;DR
This paper reveals that Dataset Distillation (DD) amplifies biases present in the original data—a phenomenon rooted in the interaction between "subgroup size imbalance" and "subgroup representation separation." It proposes COBRA: using the (group-size independent) barycenter of each subgroup's representation as the distillation target, which simultaneously reduces EOD and improves accuracy across multiple DD frameworks.

## Background & Motivation
**Background**: Dataset distillation compresses thousands of real samples into a few synthetic images, allowing downstream models trained on the synthetic set to approach full-set performance. The common paradigm of mainstream methods (DC/IDC/DM/CAFE/MTT, etc.) is to select a "representation" $\phi(x;\theta)$ (gradient, embedding, feature, or trajectory) for each class $y$ and align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set with the $\Phi_{T_y}$ of the real set.

**Limitations of Prior Work**: When the training set contains subgroup structures defined by protected attributes $A\in\mathcal{A}$ (e.g., gender, skin tone, age) and these subgroups exhibit different representation patterns, simply aligning $\Phi_{S_y}$ to the "average statistics of all samples" $\Phi_{T_y}=\sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ causes the process to be dominated by majority subgroups. Consequently, minority subgroups are nearly absent in the synthetic set. Downstream models trained on such synthetic sets suffer a significant drop in conditional accuracy for minority subgroups, leading to an increased Equalized Odds Difference (EOD).

**Key Challenge**: The authors emphasize a neglected fact—"Uniform DD" (subgroup-proportional sampling) cannot fully solve the problem because the bias stems from the **product of two independent factors**: (i) subgroup size imbalance where $\pi_{a\mid y}$ deviates from uniform, and (ii) large separation between subgroups in the representation space $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a' \mid y}}\|$. Either factor is sufficient to cause an imbalance in the residual $\Delta_{a\mid y}^*=\Phi_{T_{a\mid y}}-\Phi_{S_y}^*$.

**Goal**: (1) Formally decompose the two sources of bias amplification in DD and provide an upper bound; (2) Design a target with a tight upper bound that ensures "equidistant" synthetic representations for all subgroups; (3) Maintain compatibility with existing DD frameworks (DC/DM/CAFE/IDC) by only replacing the matching target.

**Key Insight**: Under MSE distance, the fixed point of standard DD is $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ (weighted average), which naturally favors large groups. By switching the target to a **barycenter $m_y^* = \arg\min_m \sum_a d(\Phi_{T_{a\mid y}}, m)$ that is independent of weights $\pi$**, the gap between the distances from the synthetic set to each subgroup can be tightened.

**Core Idea**: Replace the "subgroup weighted mean" with the "cross-group barycenter" as the class-conditional distillation target. This removes the dependence on group size directly from the objective, preventing the residual of the worst-off subgroup from being overwhelmed by majority groups.

## Method

### Overall Architecture
COBRA aims to resolve the amplification of original bias by dataset distillation. The mechanism involves only one change: replacing the distillation alignment target for each category from the "subgroup weighted mean" to the "subgroup barycenter." Specifically, it follows two steps: first, calculate the class-conditional subgroup statistics $\Phi_{T_{a\mid y}}=\frac{1}{|T_{a\mid y}|}\sum_{x\in T_{a\mid y}}\phi(x;\theta_T)$ for each subgroup $a\in\mathcal{A}$ within each class $y$; then, solve for the barycenter $m_y^*$ among these subgroup statistics to serve as the distillation target. Finally, align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to $m_y^*$ instead of the original full-sample mean $\Phi_{T_y}$. Since $\phi$ can be any representation (gradient, embedding, feature, or trajectory), this replacement is backbone-agnostic and can be integrated into mainstream DD methods like DC, DM, CAFE, and IDC.

### Key Designs

**1. Formal Decomposition of Bias Mechanism: Proving the Two Factors**

To fix the bias, one must first understand its source. The authors derive the SGD update for standard DD under MSE distance and obtain the fixed point for synthetic set statistics $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$—a mean weighted by subgroup size $\pi_{a\mid y}$, which naturally biases towards majority subgroups. The residual for each subgroup relative to this target is $\Delta_{a\mid y}^* = \sum_{a'\neq a}\pi_{a'\mid y}(\Phi_{T_{a\mid y}}-\Phi_{T_{a' \mid y}})$, which can be bounded as $\|\Delta_{a\mid y}^*\|_2 \leq \sum_{a'\neq a}\pi_{a'\mid y}\|\Phi_{T_{a\mid y}}-\Phi_{T_{a' \mid y}}\|_2$. This upper bound contains two factors: subgroup size imbalance $\pi_{a'\mid y}$ and representation separation $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a' \mid y}}\|$. They are multiplicative and both necessary. This corrects a key assumption in FairDD, which attributed problems solely to "subgroup imbalance." The dual-axis controlled experiments in Figure 2 prove that fixing imbalance while varying separation, or vice-versa, independently raises EOD; thus, fixing imbalance alone is insufficient.

**2. Cross-Group Barycenter $m_y^*$: A New Alignment Target Decoupled from Group Size**

Since the $\pi$-weighted mean is the source of bias, $\pi$ is removed from the objective. Let $d(u,v)=\|u-v\|_Q^2$ ($Q$ is positive definite). Solving the inner optimization $m_y^* = \arg\min_m \sum_a \|\Phi_{T_{a\mid y}}-m\|_Q^2$ yields the closed-form solution $m_y^* = \frac{1}{|\mathcal{A}|}\sum_a \Phi_{T_{a\mid y}}$—a subgroup-level uniform average. This contains no $\pi_{a\mid y}$, contrasting directly with the weighted average in vanilla DD. Choosing uniform weights $w_a=1/|\mathcal{A}|$ severs the dependence on subgroup size. Geometrically, this target minimizes the total distance to all subgroups, acting as the "fairest center" that balances distances to the synthetic representation and suppresses the maximum residual.

**3. Theoretical Guarantee: The Worst-Case Residual Is No Worse Than Vanilla**

Changing the target must not worsen results. Theorem 4.1 defines $s_y = m_y^\text{van}-m_y^*$ as the shift vector caused by imbalance. As long as the worst-off subgroup $a^\dagger$ satisfies $\langle \Delta_{a^\dagger\mid y}^C, s_y\rangle_Q \leq 0$ (meaning it resides in the opposite direction of the imbalance shift), then $\max_a \|\Delta_{a\mid y}^C\|_Q \leq \max_a \|\Delta_{a\mid y}^V\|_Q$—the worst-case residual of COBRA does not exceed that of vanilla. Geometrically, this condition is mild: the most disadvantaged subgroup naturally falls in the opposite direction of the weighted target. This elevates fairness from "loss averaging" (FairDD) to the "representation alignment" level by tightening the worst-case residual linked to subgroup-level error differences.

### Loss & Training
The final loss is $\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*,\Phi_{S_y})$, which only replaces the alignment target relative to vanilla DD. Since the barycenter has a closed-form solution under $\|u-v\|_Q^2$, no additional inner-loop iterations are required, maintaining efficiency close to vanilla. Hyperparameters such as IPC, architecture, initialization, and outer distance $D$ follow the defaults of the backbone DD (DC/DM/CAFE/IDC), while the inner distance $d$ can be swapped for cosine or MMD.

## Key Experimental Results

### Main Results

| Dataset | Backbone | IPC | Vanilla EOD/Acc | FairDD EOD/Acc | COBRA EOD/Acc |
|--------|---------|-----|-----------------|----------------|---------------|
| CIFAR10-S | DM | 100 | 82.87 / 45.4 | 25.17 / 61.2 | **9.37 / 62.4** |
| CIFAR10-S | DC | 50 | 71.85 / 39.5 | 35.65 / 46.2 | **26.18 / 46.6** |
| C-MNIST (BG) | DM | 50 | 100.0 / 48.8 | — | **7.46 / 96.8** |
| BFFHQ (Real) | DM | 100 | 63.47 / 65.8 | — | **7.87 / 74.2** |
| Full baseline | — | — | EOD 48.96 / Acc 69.71 (CIFAR10-S) | — | — |

(Values from Table 1; COBRA simultaneously reduces EOD and improves/maintains Acc across all IPCs and backbones)

### Ablation Study

| Configuration | Key Finding | Description |
|------|---------|------|
| Distance $d$ choice | MSE/cosine/MMD all work; MSE is most efficient | Framework is robust to $d$ |
| Backbone (DC/DM/CAFE/IDC) | Effective across 4 DD paradigms | Orthogonal and plug-and-play |
| Real vs. Synthetic baseline | Vanilla DD has higher EOD than Full training | Empirical evidence of bias amplification |
| Uniform DD | Sometimes worse when subgroups are close in representation space | Single factor correction (imbalance) is insufficient |
| Varying imbalance / separation | Both curves independently drive up EOD | Validates "two-factor interaction" as the source |

### Key Findings
- **DD Amplifies Original Bias**: On CIFAR10-S, Vanilla DD's EOD is much higher than Full training (e.g., 82.87 vs 48.96 at IPC=100), quantifying for the first time that distillation not only fails to preserve fairness but makes it worse.
- **Smaller IPC Leads to Greater Amplification**: As IPC decreases, the capacity for minority groups is insufficient, strengthening spurious correlations and causing EOD to surge.
- **Barycenter as a Geometric Solution for Fairness**: The worst-case residual is directly linked to subgroup-level error differences in EOD, elevating the fairness problem to the "representation geometric center" level.
- **Cross-Dataset Generality**: Consistent gains are maintained from synthetic C-MNIST/CIFAR10-S to real UTKFace/BFFHQ, compatible with 4 types of backbone DD methods.

## Highlights & Insights
- Formally decomposes the DD fairness problem into the product of imbalance × separation and provides a systematic dual-axis experimental paradigm.
- The idea of "barycenter as target" is borrowed from optimal transport/clustering but introduced to DD with almost zero computational overhead (closed-form solution under MSE).
- The theoretical guarantee $\langle\Delta,s\rangle_Q \leq 0$ is geometrically intuitive and explainable, indicating the model focuses on the most disadvantaged subgroups.
- Can be immediately integrated into any representation-matching DD (DC/DM/CAFE/IDC/MTT) with extremely low adoption cost.

## Limitations & Future Work
- Assumes protected attributes $A$ are observable during training—unlabeled subgroups are common in privacy-sensitive domains like medical data.
- When a subgroup has an "unremarkable mean but high variance" in representation space, a simple mean barycenter may mask distribution differences; distribution-level (Wasserstein) barycenters should be considered.
- Theorem 4.1 is a worst-case bound and does not directly provide a tight bound for expected EOD improvement; the coupling with downstream model selection is not yet characterized.
- Only discusses EOD; the impact on other fairness criteria like demographic parity or equal opportunity remains unexamined.
- Scalability to ImageNet-level data with large IPC and compatibility with trajectory matching (MTT) require further validation.

## Related Work & Insights
- **vs. FairDD (Zhou et al., 2025)**: FairDD uses per-group loss averaging to fix imbalance; this paper proves separation alone can also amplify EOD, and COBRA consistently outperforms FairDD.
- **vs. Standard DD (DC/DM/CAFE/IDC/MTT)**: Only modifies the alignment target $m_y^*$, proving fairness gains are orthogonal to the representation matching paradigm.
- **vs. Long-tail DD (Cui 2024, Lu 2024, Zhao 2025)**: These works focus on class imbalance, while this paper addresses subgroup-level protected-attribute imbalance, which is a finer-grained fairness.
- **vs. Barycenter Fair ML (Gordaliza 2019, Charpentier 2023)**: They perform OT barycenter preprocessing in raw data space; this paper performs barycenter distillation in representation space, which is more lightweight and suited for DD pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ Precisely characterizes the "imbalance × separation" dual-factor structure and provides a geometric solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 backbones, multiple IPCs, and 5 datasets (synthetic and real).
- Writing Quality: ⭐⭐⭐⭐ Clear derivation of bias mechanism; Theorem and figures complement each other well.
- Value: ⭐⭐⭐⭐ Plug-and-play with low barriers for adoption; significant for fairness in high-stakes deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICLR 2026\] Dataset Distillation for Memorized Data: Soft Labels can Leak Held-Out Teacher Knowledge](../../ICLR2026/ai_safety/dataset_distillation_for_memorized_data_soft_labels_can_leak_held-out_teacher_kn.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](ai_alignment_encompasses_competing_technical_priorities.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)

</div>

<!-- RELATED:END -->
