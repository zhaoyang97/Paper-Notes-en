---
title: >-
  [Paper Note] Fair Dataset Distillation via Cross-Group Barycenter Alignment
description: >-
  [ICML 2026][AI Safety][Data Distillation] This paper reveals that dataset distillation (DD) magnifies biases present in the original data—a phenomenon rooted in the interaction between "subgroup sample size imbalance" and "subgroup representation separation." The authors propose COBRA, which utilizes the (group-size-invariant) barycenter of subgroup representa
tags:
  - ICML 2026
  - AI Safety
  - Data Distillation
  - EOD
date: 2026-05-08
content_hash: ce422b411b4280b0
---
# Fair Dataset Distillation via Cross-Group Barycenter Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.00185](https://arxiv.org/abs/2605.00185)  
**Code**: No public link  
**Area**: Dataset Distillation / Fair Machine Learning / AI Safety  
**Keywords**: Dataset Distillation, Group Fairness, Subgroup Barycenter Alignment, EOD, Representation Alignment

## TL;DR
This paper reveals that dataset distillation (DD) magnifies biases present in the original data—a phenomenon rooted in the interaction between "subgroup sample size imbalance" and "subgroup representation separation." The authors propose COBRA, which utilizes the (group-size-invariant) barycenter of subgroup representations as the distillation target. This approach simultaneously reduces EOD and improves accuracy across multiple DD frameworks.

## Background & Motivation
**Background**: Dataset distillation compresses thousands of real samples into a few synthetic images, allowing downstream models trained on the synthetic set to approach full-training performance. Prevailing methods (DC/IDC/DM/CAFE/MTT, etc.) follow a common paradigm: selecting a "representation" $\phi(x;\theta)$ (gradients, embeddings, features, or trajectories) for each class $y$ and aligning the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to the $\Phi_{T_y}$ of the real set.

**Limitations of Prior Work**: When the training set contains a subgroup structure defined by protected attributes $A\in\mathcal{A}$ (gender, skin tone, age, etc.) and representation patterns differ between subgroups, simply aligning $\Phi_{S_y}$ to the "mean statistics of all samples" $\Phi_{T_y}=\sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ causes the process to be dominated by the majority subgroup. Consequently, minority subgroups are barely represented in the synthetic set. Downstream models trained on these sets suffer significant drops in conditional accuracy for minority groups, leading to larger equalized odds difference (EOD).

**Key Challenge**: The authors emphasize a neglected fact—"Uniform DD" (equal subgroup sampling) cannot fully solve the problem because bias arises from the **product of two independent factors**: (i) subgroup size imbalance $\pi_{a\mid y}$ deviating from uniform, and (ii) large separation between subgroups in the representation space $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$. Either factor is sufficient to cause an imbalance in the residual $\Delta_{a\mid y}^*=\Phi_{T_{a\mid y}}-\Phi_{S_y}^*$.

**Goal**: (1) Formally decompose the two sources of bias magnification in DD and provide an upper bound; (2) Design a target with a tight upper bound to ensure all subgroups obtain "equidistant" synthetic representations; (3) Maintain compatibility with existing DD frameworks (DC/DM/CAFE/IDC) by only replacing the matching target.

**Key Insight**: Under MSE distance, the fixed point of standard DD is $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ (weighted average), which naturally biases toward the majority group. By switching to a **barycenter $m_y^* = \arg\min_m \sum_a d(\Phi_{T_{a\mid y}}, m)$ that is independent of weights $\pi$** as the target, the distance gap between the synthetic set and each subgroup can be tightened.

**Core Idea**: Replace the "subgroup weighted mean" with a "cross-group barycenter" as the class-conditional distillation target. This removes the dependency on group size from the objective, ensuring the residual of the worst-performing subgroup is not overwhelmed by the majority group.

## Method

### Overall Architecture
COBRA addresses the magnification of original bias in dataset distillation by modifying a single component: changing the distillation alignment target for each class from a "subgroup weighted mean" to a "subgroup barycenter." The process involves two steps: first, calculating the class-conditional subgroup statistics $\Phi_{T_{a\mid y}}=\frac{1}{|T_{a\mid y}|}\sum_{x\in T_{a\mid y}}\phi(x;\theta_T)$ for each subgroup $a\in\mathcal{A}$ within class $y$; second, solving for the barycenter $m_y^*$ of these subgroup statistics to serve as the distillation target. The synthetic class-conditional statistics $\Phi_{S_y}$ are then aligned to $m_y^*$ instead of the original full-sample mean $\Phi_{T_y}$. Since $\phi$ can represent any form of representation (gradients, embeddings, features, or trajectories), this replacement is backbone-agnostic and can be integrated into mainstream DD methods like DC, DM, CAFE, and IDC.

### Key Designs

**1. Formal Decomposition of Bias Mechanism: Proving the Two Factors of Bias**

To fix bias, one must first understand its source. The authors derive the SGD update for standard DD under MSE distance, obtaining the fixed point for synthetic set statistics: $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$. This is a mean weighted by subgroup size $\pi_{a\mid y}$, naturally favoring majority subgroups. The residual for each subgroup relative to this target is $\Delta_{a\mid y}^* = \sum_{a'\neq a}\pi_{a'\mid y}(\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}})$, which can be bounded as $\|\Delta_{a\mid y}^*\|_2 \leq \sum_{a'\neq a}\pi_{a'\mid y}\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|_2$. This upper bound contains two multiplicative factors: subgroup imbalance $\pi_{a'\mid y}$ and subgroup separation in representation space $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$. This corrects a key assumption in FairDD, which attributed problems solely to "subgroup imbalance." Figure 2's dual-axis controlled experiments demonstrate that holding one factor constant while adjusting the other can independently drive up EOD, proving that addressing imbalance alone is insufficient.

**2. Cross-group Barycenter $m_y^*$: A New Alignment Target Decoupled from Group Size**

Since the $\pi$-weighted mean is the source of bias, COBRA removes $\pi$ from the objective. Let $d(u,v)=\|u-v\|_Q^2$ ($Q$ is positive definite); the inner optimization $m_y^* = \arg\min_m \sum_a \|\Phi_{T_{a\mid y}}-m\|_Q^2$ yields the closed-form solution $m_y^* = \frac{1}{|\mathcal{A}|}\sum_a \Phi_{T_{a\mid y}}$. This is a subgroup-level uniform average, entirely free of $\pi_{a\mid y}$, providing a direct contrast to vanilla DD. Choosing uniform weights $w_a=1/|\mathcal{A}|$ severs the dependency on subgroup size, while the geometric meaning of the barycenter minimizes the total distance to all subgroups. This acts as a "fair center," evening out the distances from the synthetic representation to every subgroup and suppressing the maximum residual.

**3. Theoretical Guarantees: The Worst-case Subgroup Residual is Not Worse Than Vanilla**

Changing the target must not worsen results for other groups. Theorem 4.1 defines $s_y = m_y^\text{van}-m_y^*$ as the shift vector caused by imbalance. As long as the worst-performing subgroup $a^\dagger$ satisfies $\langle \Delta_{a^\dagger\mid y}^C, s_y\rangle_Q \leq 0$ (meaning it aligns opposite to the imbalance shift), then $\max_a \|\Delta_{a\mid y}^C\|_Q \leq \max_a \|\Delta_{a\mid y}^V\|_Q$—the worst-case residual of COBRA does not exceed that of vanilla DD. This condition is geometrically mild: the most disadvantaged subgroup should naturally fall in the opposite direction of the weighted target. This elevates the fairness guarantee from "loss averaging" to the level of "representation alignment" by tightening the worst-case residual, which is directly linked to subgroup-level error variance in EOD.

### Loss & Training
The final loss is $\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*,\Phi_{S_y})$, which simply replaces the alignment target of vanilla DD. Since the barycenter has a closed-form solution under $\|u-v\|_Q^2$, the inner loop requires no additional iterations, maintaining efficiency comparable to vanilla methods. Hyperparameters such as IPC, network architecture, initialization, and the outer distance $D$ follow the default settings of the backbone DD method (DC/DM/CAFE/IDC). The inner distance $d$ can also be replaced with cosine or MMD for ablation.

## Key Experimental Results

### Main Results

| Dataset | Backbone | IPC | Vanilla EOD/Acc | FairDD EOD/Acc | COBRA EOD/Acc |
|--------|---------|-----|-----------------|----------------|---------------|
| CIFAR10-S | DM | 100 | 82.87 / 45.4 | 25.17 / 61.2 | **9.37 / 62.4** |
| CIFAR10-S | DC | 50 | 71.85 / 39.5 | 35.65 / 46.2 | **26.18 / 46.6** |
| C-MNIST (BG) | DM | 50 | 100.0 / 48.8 | — | **7.46 / 96.8** |
| BFFHQ (Real) | DM | 100 | 63.47 / 65.8 | — | **7.87 / 74.2** |
| Full baseline | — | — | EOD 48.96 / Acc 69.71 (CIFAR10-S) | — | — |

(Values from Table 1; COBRA consistently reduces EOD and improves or maintains Accuracy across all IPCs and backbones.)

### Ablation Study

| Configuration | Key Result | Description |
|------|---------|------|
| Distance $d$ Choice | MSE/cosine/MMD all work; MSE is most efficient | Framework is robust to $d$ |
| Backbone (DC/DM/CAFE/IDC) | Effective across 4 DD paradigms | Orthogonal and plug-and-play |
| Real vs. Synthetic Baseline | Vanilla DD has higher EOD than Full training | Empirical confirmation that DD magnifies bias |
| Uniform Subgroup Sampling | Worse when subgroups are close in representation | Fixing imbalance alone is insufficient |
| Varying imbalance / separation | Both curves independently increase EOD | Validates "two-factor interaction" as the source |

### Key Findings
- **DD Magnifies Original Bias**: On CIFAR10-S, Vanilla DD's EOD is much higher than Full training (e.g., 82.87 vs. 48.96 at IPC=100), clearly quantifying for the first time that distillation not only fails to preserve fairness but becomes more unfair than training on the full set.
- **Smaller IPC Leads to Greater Magnification**: As IPC decreases, majority group capacity becomes insufficient, strengthening spurious correlations and causing EOD to rise sharply.
- **Barycenter as a Geometric Solution for Fairness**: The worst-case residual directly relates to subgroup-level error variance in EOD, elevating the fairness problem from "weighted loss averaging" to "representation geometric centers."
- **Generalization Across Datasets**: Persistent gains are observed from synthetic C-MNIST/CIFAR10-S to real UTKFace/BFFHQ, maintaining compatibility with four different backbone DD methods.

## Highlights & Insights
- The rigorous decomposition of the fairness problem in DD into a product of "imbalance × separation," backed by dual-axis controlled experiments in Figure 2, provides a clear research methodology that can serve as a standard baseline for future fair DD work.
- The idea of "barycenter as target" is borrowed from optimal transport/clustering but introduces negligible computational overhead in DD (closed-form solution under MSE), offering both academic elegance and engineering friendliness.
- The theoretical guarantee condition $\langle\Delta,s\rangle_Q \leq 0$ is geometrically intuitive and interpretable, suggesting that the most disadvantaged subgroup should reside in the opposite direction of the imbalance shift.
- It can be immediately integrated into any representation-matching DD method (DC/DM/CAFE/IDC/MTT), with a very low adoption cost for the community.

## Limitations & Future Work
- Assumes that the protected attribute $A$ is observable during training and subgroup labels are available—a condition often not met in real-world scenarios like medical data due to privacy concerns.
- When a subgroup is "unobtrusive in mean but high in variance" in the representation space, a simple mean barycenter might mask distributional differences; distributional (Wasserstein) barycenters should be considered.
- Theorem 4.1 focuses on the worst-case and does not provide a tight bound for expected EOD improvement; the coupling with downstream model selection is not yet characterized.
- Only EOD is discussed as a fairness criterion; the impact on other criteria such as demographic parity or equal opportunity has not been explored.
- Scalability to ImageNet-level data with large IPC and compatibility with trajectory matching (MTT) require further validation.

## Related Work & Insights
- **vs. FairDD (Zhou et al., 2025)**: FairDD performs per-group loss averaging to fix imbalance; this paper proves that the separation factor alone can also magnify EOD, and COBRA consistently outperforms FairDD across all datasets/IPCs.
- **vs. Standard DD (DC/DM/CAFE/IDC/MTT)**: Only modifies the alignment target $m_y^*$ without altering the backbone algorithm, proving that fairness gains are orthogonal to the representation matching paradigm.
- **vs. Long-tail DD (Cui 2024, Lu 2024, Zhao 2025)**: Those works focus on class imbalance, whereas this paper focuses on subgroup-level protected-attribute imbalance, which is a more fine-grained aspect of fairness.
- **vs. Barycenter in Fair ML (Gordaliza 2019, Charpentier 2023)**: They perform OT barycenter preprocessing in the original data space; this paper performs barycenter distillation in the representation space, which is more lightweight and better suited for DD pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ Accurately characterizing the neglected "imbalance × separation" dual-factor structure and providing a geometric solution is a significant conceptual advancement in fair DD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 backbones × multiple IPCs × 5 datasets (synthetic + real) with systematic ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation of bias mechanisms, mutual validation between Theorems and Figures, and smooth argumentation.
- Value: ⭐⭐⭐⭐ Plug-and-play with low barriers to community adoption; practically significant for fairness in high-stakes deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)

</div>

<!-- RELATED:END -->
