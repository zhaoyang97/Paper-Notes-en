---
title: >-
  [Paper Note] Fair Dataset Distillation via Cross-Group Barycenter Alignment
description: >-
  [ICML 2026][AI Safety][Dataset Distillation] This work reveals that dataset distillation (DD) amplifies biases present in the original data—rooted in the interaction between "subgroup sample size imbalance" and "subgroup…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Dataset Distillation"
  - "Group Fairness"
  - "Subgroup Barycenter Alignment"
  - "EOD"
  - "Representation Alignment"
date: 2026-05-08
content_hash: 3fc8b7ce43f824cb
---

# Fair Dataset Distillation via Cross-Group Barycenter Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.00185](https://arxiv.org/abs/2605.00185)  
**Code**: No public link  
**Area**: Dataset Distillation / Fair Machine Learning / AI Safety  
**Keywords**: Dataset Distillation, Group Fairness, Subgroup Barycenter Alignment, EOD, Representation Alignment

## TL;DR
This work reveals that dataset distillation (DD) amplifies biases present in the original data—rooted in the interaction between "subgroup sample size imbalance" and "subgroup representational separation." The authors propose COBRA: using the barycenter of subgroup representations (independent of group size) as the distillation target, which simultaneously reduces EOD and improves accuracy across multiple DD frameworks.

## Background & Motivation
**Background**: Dataset distillation compresses thousands of real samples into a few synthetic images, enabling downstream models trained on the synthetic set to approach full-data performance. Mainstream methods (DC/IDC/DM/CAFE/MTT, etc.) share a common approach: for each class $y$, select a "representation" $\phi(x;\theta)$ (gradient, embedding, feature, trajectory), and align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to those of the real set $\Phi_{T_y}$.

**Limitations of Prior Work**: When the training set contains subgroups defined by protected attributes $A\in\mathcal{A}$ (gender, skin color, age, etc.), and subgroup representation patterns differ, simply aligning $\Phi_{S_y}$ to the "average statistics of all samples" $\Phi_{T_y}=\sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ is dominated by majority subgroups, causing minority subgroups to be nearly absent in the synthetic set. Downstream models trained on the synthetic set show significantly reduced conditional accuracy for minority subgroups, and EOD (equalized odds difference) increases.

**Key Challenge**: The authors repeatedly emphasize an overlooked fact—"uniform subgroup sampling" (Uniform DD) cannot fundamentally solve the problem, because the bias arises from the **product of two independent factors**: (i) subgroup size imbalance $\pi_{a\mid y}$ deviates from uniformity, and (ii) subgroups are well-separated in representation space $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$ is large. The presence of either is sufficient to cause the residual $\Delta_{a\mid y}^*=\Phi_{T_{a\mid y}}-\Phi_{S_y}^*$ to be unbalanced.

**Goal**: (1) Formally decompose the two sources of bias amplification in DD and provide an upper bound; (2) Design a target with a tight upper bound so that all subgroups receive "equidistant" synthetic representations; (3) Remain compatible with existing DD frameworks (DC/DM/CAFE/IDC), only replacing the matching target.

**Key Insight**: Under MSE distance, the fixed point of standard DD is $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ (weighted average), which naturally favors large groups; replacing it with a **barycenter independent of weights $\pi$** $m_y^* = \arg\min_m \sum_a d(\Phi_{T_{a\mid y}}, m)$ as the target tightens the distance gap from the synthetic set to each subgroup.

**Core Idea**: Replace the "subgroup weighted mean" with the "cross-group barycenter" as the class-conditional distillation target, directly removing dependence on group size from the target, so that the worst-case subgroup residual is not dominated by majority groups.

## Method

### Overall Architecture
COBRA is a two-step process: (1) For each class $y$, compute class-conditional subgroup statistics on real data for each subgroup $a\in\mathcal{A}$, $\Phi_{T_{a\mid y}}=\frac{1}{|T_{a\mid y}|}\sum_{x\in T_{a\mid y}}\phi(x;\theta_T)$, then compute the barycenter $m_y^*$ among subgroups (under a suitable distance $d$, with uniform weights $w_a=1/|\mathcal{A}|$); (2) Align the class-conditional statistics $\Phi_{S_y}$ of the synthetic set to $m_y^*$ instead of the original $\Phi_{T_y}$, with loss $\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*, \Phi_{S_y})$. The framework is agnostic to the specific form of $\phi$ (gradient/embedding/feature/trajectory), and can be directly embedded into DD methods such as DC, DM, CAFE, IDC.

### Key Designs

1. **Formal Decomposition of Bias Mechanism**:

    - Function: Traces EOD degradation in DD to a provable upper bound, explicitly showing that group imbalance and representational separation must be jointly considered.
    - Mechanism: Under MSE distance, SGD updates yield the fixed point $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$, with residual $\Delta_{a\mid y}^* = \sum_{a'\neq a}\pi_{a'\mid y}(\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}})$, so $\|\Delta_{a\mid y}^*\|_2 \leq \sum_{a'\neq a}\pi_{a'\mid y}\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|_2$, where both factors are necessary.
    - Design Motivation: Previous work (FairDD) attributed the problem solely to group imbalance; this work uses dual-axis controlled experiments (Figure 2) to show that fixing imbalance and varying separation, or vice versa, each independently increases EOD, so single-factor correction is insufficient—both must be controlled together.

2. **Cross-Group Barycenter $m_y^*$ as New Target**:

    - Function: Ensures the distillation target is equidistant to each subgroup, minimizing the maximum residual and geometrically breaking the interaction term in the upper bound.
    - Mechanism: With $d(u,v)=\|u-v\|_Q^2$ (positive definite $Q$), the inner optimization $m_y^* = \arg\min_m \sum_a \|\Phi_{T_{a\mid y}}-m\|_Q^2$ has a closed-form solution $m_y^* = \frac{1}{|\mathcal{A}|}\sum_a \Phi_{T_{a\mid y}}$—i.e., uniform mean at the subgroup level, completely independent of $\pi_{a\mid y}$; this contrasts with the $\pi$-weighted mean in vanilla DD.
    - Design Motivation: Choosing uniform weights $w_a$ eliminates dependence on subgroup size; the barycenter minimizes the total distance to all subgroups, serving as the "fairest center."

3. **Theoretical Guarantee: Worst-Case Residual Does Not Increase**:

    - Function: Theorem 4.1 rigorously shows that COBRA does not make the worst-case subgroup worse than vanilla.
    - Mechanism: Define $s_y = m_y^\text{van}-m_y^*$ as the imbalance shift; if the worst-case subgroup $a^\dagger$ satisfies $\langle \Delta_{a^\dagger\mid y}^C, s_y\rangle_Q \leq 0$ (i.e., the worst-case subgroup is in the opposite direction of the imbalance shift—a geometrically mild condition), then $\max_a \|\Delta_{a\mid y}^C\|_Q \leq \max_a \|\Delta_{a\mid y}^V\|_Q$.
    - Design Motivation: Empirically, FairDD averages per-group loss but parameter updates can still drift; COBRA directly tightens the worst-case residual, a geometric quantity directly related to EOD, elevating the fairness guarantee from loss averaging to representation alignment.

### Loss & Training
$\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*,\Phi_{S_y})$, differing from vanilla DD only in the alignment target. The barycenter under $\|u-v\|_Q^2$ has a closed-form solution, so the inner loop is no longer needed, making efficiency comparable to vanilla; other hyperparameters (IPC, network architecture, initialization, distance $D$) follow the default settings of the backbone DD (DC/DM/CAFE/IDC). Other distances (cosine, etc.) can be used for ablation.

## Key Experimental Results

### Main Results

| Dataset | Backbone | IPC | Vanilla EOD/Acc | FairDD EOD/Acc | COBRA EOD/Acc |
|---------|----------|-----|-----------------|----------------|---------------|
| CIFAR10-S | DM | 100 | 82.87 / 45.4 | 25.17 / 61.2 | **9.37 / 62.4** |
| CIFAR10-S | DC | 50 | 71.85 / 39.5 | 35.65 / 46.2 | **26.18 / 46.6** |
| C-MNIST (BG) | DM | 50 | 100.0 / 48.8 | — | **7.46 / 96.8** |
| BFFHQ (real) | DM | 100 | 63.47 / 65.8 | — | **7.87 / 74.2** |
| Full baseline | — | — | EOD 48.96 / Acc 69.71 (CIFAR10-S) | — | — |

(Values from Table 1; COBRA consistently reduces EOD and improves/maintains Acc across all IPC and backbones.)

### Ablation Study

| Setting | Key Result | Note |
|---------|------------|------|
| Choice of distance $d$ | MSE/cosine/MMD all work, MSE closed-form is most efficient | Framework is robust to $d$ |
| Backbone (DC/DM/CAFE/IDC) | Effective across 4 DD paradigms | Orthogonal to backbone, plug-and-play |
| Real vs synthetic baseline | Vanilla DD has higher EOD than full training → DD amplifies original bias | Empirically supports Section 1's claim |
| Uniform DD (equal subgroup sampling) | When subgroups are close in representation, can be worse | Single correction for imbalance is insufficient |
| Varying imbalance / separation | Each curve independently increases EOD | Validates "two-factor interaction" as bias source |

### Key Findings
- **DD amplifies original bias**: On CIFAR10-S, vanilla DD's EOD is much higher than full training (e.g., IPC=100: 82.87 vs Full 48.96), quantifying for the first time that "distillation not only fails to preserve fairness, but is even less fair than full training."
- **Smaller IPC amplifies bias more**: As IPC decreases, minority subgroup capacity is insufficient, spurious correlations are further reinforced, and EOD rises sharply.
- **Barycenter is a geometric solution for fairness**: The worst-case residual directly relates to subgroup-level error differences in EOD, elevating the fairness issue from "weighted loss averaging" to the level of "geometric center of representations."
- **Cross-dataset generality**: From synthetic C-MNIST/CIFAR10-S to real UTKFace/BFFHQ, significant gains are maintained, and compatibility with 4 backbone DD methods is demonstrated.

## Highlights & Insights
- Rigorously decomposes DD's fairness problem into the product of imbalance × separation, with dual-axis controlled experiments (Figure 2); the research methodology is clear and can serve as a standard baseline paradigm for future fair DD work.
- The idea of barycenter as target is borrowed from optimal transport/clustering, but its introduction to DD adds almost no computational overhead (closed-form under MSE), offering both academic elegance and engineering friendliness.
- The theoretical guarantee condition $\langle\Delta,s\rangle_Q \leq 0$ is geometrically mild and interpretable—it means the worst-case subgroup is in the opposite direction of the imbalance shift, aligning with the intuition that "the most disadvantaged subgroup should lie opposite to the weighted target."
- Can be immediately embedded into any representation-matching DD (DC/DM/CAFE/IDC/MTT), with extremely low adoption cost for the community.

## Limitations & Future Work
- Assumes protected attribute $A$ is observable during training and subgroup labels are available—in real-world scenarios such as medical data, subgroup annotation is often unavailable due to privacy.
- When a subgroup is "mean-inconspicuous but high-variance" in representation space, a simple mean barycenter may mask distributional differences; distribution-level (Wasserstein) barycenter should be considered.
- Theorem 4.1 is a worst-case result and does not directly provide a tight bound for expected EOD improvement; coupling with downstream model selection is not yet characterized.
- Only EOD as a fairness criterion is discussed; effects on demographic parity, equal opportunity, and other criteria remain untested.
- Scalability to ImageNet-scale data + large IPC, and compatibility with trajectory matching (MTT), require further validation.

## Related Work & Insights
- **vs FairDD (Zhou et al., 2025)**: They average per-group loss, correcting only imbalance; this work proves that the separation factor alone can also amplify EOD, and COBRA consistently outperforms FairDD across all datasets/IPC.
- **vs Standard DD (DC/DM/CAFE/IDC/MTT)**: Only the alignment target $m_y^*$ is changed, leaving the backbone algorithm untouched, demonstrating that fairness gains are orthogonal to the representation matching paradigm.
- **vs Long-tail DD (Cui 2024, Lu 2024, Zhao 2025)**: Those works focus on class imbalance, while this work addresses subgroup-level protected-attribute imbalance, a finer-grained fairness issue.
- **vs Barycenter Fair ML (Gordaliza 2019, Charpentier 2023)**: They perform OT barycenter preprocessing in the original data space; this work performs barycenter distillation in representation space, which is lighter and better suited to the DD pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ Precisely characterizes the previously overlooked "imbalance × separation interaction" and provides a geometric solution—a key conceptual advance for fair DD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 backbones × multiple IPC × 5 datasets (synthetic + real), with systematic ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation of bias mechanism, Theorem + Figure cross-validation, smooth argumentation.
- Value: ⭐⭐⭐⭐ Plug-and-play, low adoption threshold for the community, and practical significance for fairness in high-stakes deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](../../ACL2026/ai_safety/xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[CVPR 2026\] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](../../CVPR2026/ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)

</div>

<!-- RELATED:END -->
