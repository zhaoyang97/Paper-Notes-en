---
title: >-
  [Paper Note] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data
description: >-
  [ICLR 2026][Self-Supervised Learning][Paper Note] This paper provides a rigorous theoretical analysis of the robustness of different distributed self-supervised learning (D-SSL) frameworks under non-IID data. It proves that Masked Image Modeling (MIM) is inherently more resistant to heterogeneity than Contrastive Learning (CL), and that robustness increases with the a
tags:
  - ICLR 2026
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 3b37f20ef6a77089
---
# Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=c3yt5VSZPQ](https://openreview.net/forum?id=c3yt5VSZPQ)  
**Code**: https://github.com/xuanyuLawrence/FedMAR-DecMAR  
**Area**: Self-Supervised Learning / Distributed Learning  
**Keywords**: Distributed Self-Supervised Learning, Non-IID, Masked Image Modeling, Contrastive Learning, Federated Learning

## TL;DR
This paper provides a rigorous theoretical analysis of the robustness of different distributed self-supervised learning (D-SSL) frameworks under non-IID data. It proves that Masked Image Modeling (MIM) is inherently more resistant to heterogeneity than Contrastive Learning (CL), and that robustness increases with the average network connectivity (Federated Learning is at least as robust as Decentralized Learning). Based on these insights, the authors design MAR loss with local-global alignment regularization as a practical exemplar.

## Background & Motivation
**Background**: In real-world scenarios, massive amounts of unlabeled data are distributed across clients (e.g., decentralized surveillance cameras). Applying Self-Supervised Learning (SSL) to distributed settings results in Distributed Self-Supervised Learning (D-SSL). It bifurcates along two dimensions: SSL methods (Contrastive Learning, CL, represented by SimSiam; and Masked Image Modeling, MIM, represented by MAE) and distributed frameworks (Federated Learning, FL, relying on a central server; and Decentralized Learning, DecL, through direct peer-to-peer communication).

**Limitations of Prior Work**: The core challenge for D-SSL is the highly heterogeneous (non-IID) nature of client data, which leads to significant degradation in representation quality and downstream accuracy. While several "more robust" algorithms like FedU, Orchestra, and L-DAWA have been proposed, these works are primarily **algorithmic patches** that lack a theoretical understanding of the "heterogeneity problem" itself.

**Key Challenge**: Existing theoretical analyses (e.g., Wang et al.) only cover specific cases like "CL+FL." No systematic answers exist for two fundamental questions: (1) Facing the same non-IID data, is MIM or CL inherently more resistant? (2) How does robustness change when an algorithm (e.g., FedU designed for FL) is moved to a DecL framework without server coordination? In other words, **the relationship between network structure and non-IID robustness is largely unexplored**.

**Goal**: To decompose the broad question of "how robust different D-SSL frameworks are to data heterogeneity" into two provable sub-problems—the robustness difference between SSL paradigms (MIM vs. CL) and the impact of network connectivity (DecL vs. FL) on robustness.

**Key Insight**: The authors construct a **simplified yet formal mathematical model for non-IID data**. Under a linear embedding assumption, they explicitly characterize the local and global representations learned by each D-SSL variant and use a unified scalar to measure "how sensitive the representation is to data heterogeneity." By deriving closed-form upper and lower bounds for sensitivity in terms of $d$ (dimension), they clarify which framework is more robust.

**Core Idea**: Use the "span between the upper and lower bounds of the Representability Vector" as a robustness metric. This unifies the robustness differences between MIM/CL and DecL/FL into comparable mathematical conclusions. Subsequently, they demonstrate how theory guides algorithm design via an alignment regularization (MAR loss).

## Method

### Overall Architecture
The "Method" in this work is primarily a **theoretical analysis framework** rather than an engineering pipeline, consisting of four steps: "Modeling $\to$ Metric $\to$ Comparison $\to$ Implementation." First, a simplified non-IID data generation model is constructed: there are $2N$ global classes, and the local distribution $D_i$ of the $i$-th client is concentrated on two primary classes ($2i-1$ and $2i$), with a tiny fraction of samples from a rare class $h_i$—formalizing both local class visibility and class imbalance. Second, under the linear embedding $f_W(x)=Wx$, the authors write local objectives for CL (SimSiam form) and MIM (reconstruction alignment form) and define the **Representability Vector (RV)** to quantify the quality of the learned feature space. Third, they derive the RV bounds for each framework (Local / DecL Global / FL Global) and use the "span" of these bounds to define sensitivity $s$. Finally, they translate these insights into an algorithm: targeting the finding that "MIM is robust but local encoders still drift," they propose MAR loss to explicitly align local and global representations.

### Key Designs

**1. Representability Vector: Compressing Feature Space Quality into a Comparable Scalar**

To compare the resistance of different D-SSL frameworks to heterogeneity, a unified metric is required. The authors define the **Representability Vector (RV)**: let the linear embedding matrix be $W=[w_1,\dots,w_c]^\top$ and its row space be $R=\mathrm{row}(W)$. The RV is $r=[\,\|\Pi_R(e_1)\|_2^2,\dots,\|\Pi_R(e_d)\|_2^2\,]^\top$, where $\Pi_R(e_k)$ is the projection of the standard basis vector $e_k$ onto the feature space $R$. Intuitively, a good feature space should "accommodate" the basis directions on which data generation depends; thus, the projections of these basis vectors should be large and close to each other. The RV translates abstract "representation quality" into $d$ calculable scalars. Based on this, the authors define **sensitivity** $s=\max_{k}\bar r_k-\min_{k}\bar r_k$, which is the span of the RV across the first $c$ coordinates. Every RV has a shared upper bound of 1 and a specific lower bound; **the smaller the span between these bounds, the more uniform the representation is across directions, the less it is affected by non-IID data, and the more robust the D-SSL framework is.**

**2. MIM is Inherently More Robust than CL: Evidence from Randomness in Alignment Targets**

Using RV and sensitivity, the authors derive closed-form bounds for both local and global RVs for MIM and CL (Theorem 4.2, 4.3). Comparing their sensitivities yields the core conclusion (Theorem 4.4): as $d\to\infty$, $s_C > s_M$, meaning CL's sensitivity is strictly greater than MIM's. **Why is MIM more robust?** The authors provide a crucial intuition: CL performs feature alignment on "positive pairs generated from the same image through data augmentation." Although augmentations generally preserve labels, the output is a **completely different image**, which introduces additional randomness that is biased by the local label distribution. In contrast, MIM performs reconstruction alignment between a masked part $x_2$ and an unmasked part $x_1$ of the same image—both **retain parts of the original data**, leading to less randomness between alignment targets. Coupled with existing data heterogeneity, the local representations learned by CL exhibit higher randomness and heavier bias, resulting in a global representation that is less uniform than that of MIM.

**3. Connectivity Determines Robustness, and FL $\ge$ DecL: Integrating Network Structure into the Bounds**

The second sub-problem concerns the impact of network structure. The authors observe that the lower bound of the DecL global RV includes a factor of $(1-1/|\bar A|)$, where $|\bar A|=\frac1N\sum_i|A_i|$ represents the **average connectivity** of the network. This leads to Corollary 4.5: in fully decentralized (serverless) scenarios, the robustness of D-SSL to heterogeneous data **improves as the average connectivity $|\bar A|$ increases.** Furthermore, FL leverages a central server, which is equivalent to each client "indirectly connecting to everyone," mimicking a fully connected decentralized topology ($|A_i|=N$). Consequently, Theorem 4.6 states $s_{Dec}\ge s_{Fed}$, meaning **FL's sensitivity is no greater than DecL's, and Federated Learning is at least as robust as Decentralized Learning against heterogeneity.** The practical implication is direct: if data heterogeneity is the primary concern, FL should be preferred; however, if a trusted server is unavailable, one should strive to **increase the average connectivity between clients** (e.g., identifying under-connected clients and adding direct edges).

**4. MAR Loss: Fall-to-Practice with Adaptive MMD Alignment and Cosine Decay Weights**

While theory indicates MIM is more robust, it also reveals that MIM training dynamics are **dominated by the local covariance of each client**, causing local encoders to drift in different directions before aggregation. This inspires MAR loss, which adds an **explicit and dynamic** local-global alignment regularization to the MIM objective:

$$\mathcal{L}_{MAR}=\mathbb{E}_{x\sim D_i}\mathbb{E}_{x_1,x_2|x}\big[\|f_d(f_e(x_1))-x_2\|^2+\gamma_t^{(i)}\cdot\text{A-MMD}(z_i,\bar z)\big]$$

where $z_i=f_e(x_1)$ is the local masked representation and $\bar z$ is the global representation. The alignment term uses **Adaptive Maximum Mean Discrepancy (A-MMD)** to measure the distribution gap. Unlike traditional FL methods that use fixed-bandwidth MMD, A-MMD automatically determines the Gaussian kernel bandwidth from the data—$k(z,z')=\exp\!\big(-\frac{\|z-z'\|}{2(\mathrm{mean}_{a\neq b}\|z_a-z_b\|)^2}\big)$—scaling the kernel to the actual embedding distribution for stability across non-IID clients. The weight $\gamma_t^{(i)}$ follows a **cosine schedule**, smoothly decaying from $\gamma_{max}$ to $\gamma_{min}$: $\gamma_t^{(i)}=\gamma_{min}+(\gamma_{max}-\gamma_{min})\cdot\frac12\big(1+\cos\frac{\pi\,\omega_t^{(i)}}{\Omega}\big)$, where $\omega_t^{(i)}$ is the number of times client $i$ has been selected up to round $t$. This **applies strong alignment in the early stages when client divergence is highest and relaxes it later** to reduce overhead. MAR is compatible with both FL (FedMAR) and DecL (DecMAR).

## Key Experimental Results

Experiments were pre-trained on Mini-ImageNet (using Dirichlet distribution for label non-IID and independent augmentations for feature non-IID, with Erdős–Rényi models for DecL networks) and fine-tuned on CIFAR-10 / CIFAR-100 / ImageNet. Backbones include ResNet and ViT.

### Main Results: Heterogeneity Sensitivity of MIM vs. CL

The table below shows fine-tuning accuracy under IID and non-IID settings, with the drop from IID $\to$ non-IID in parentheses (smaller is more robust). **MAE (MIM) exhibits significantly smaller drops than SimSiam (CL)** across the same backbones, validating Theorem 4.4.

| Configuration | Dataset | IID | Label non-IID (Gain/Drop) | Feature non-IID (Gain/Drop) |
|------|--------|-----|---------------------|---------------------|
| SimSiam+CNN | CIFAR-10 | 86.03 | 84.33 (↓1.70) | 84.62 (↓1.41) |
| MAE+CNN | CIFAR-10 | 87.28 | 86.97 (↓0.31) | 86.17 (↓1.11) |
| SimSiam+ViT | CIFAR-100 | 48.60 | 43.49 (↓5.11) | 43.07 (↓5.53) |
| MAE+ViT | CIFAR-100 | 50.04 | 48.95 (↓1.09) | 49.60 (↓0.44) |

In the ViT + CIFAR-100 setting, where heterogeneity is most impactful, CL drops over 5 points while MIM only drops about 1 point, highlighting the stark difference.

### Connectivity and FL $\ge$ DecL Validation

Varying the average connectivity $|\bar A|$ (from 4 to 20) in a 20-client network on CIFAR-100: DecL accuracy increases monotonically with $|\bar A|$ (validating Corollary 4.5). Furthermore, across both uniform and general topologies, **the FL curve is never lower than DecL** (validating Theorem 4.6).

### Comparison with SOTA Federated Self-Supervised Methods

In a 100-client cross-device setting with high heterogeneity ($\alpha=0.1$), FedMAR is compared with several F-SSL SOTAs:

| Method | Backbone | CIFAR-10 | CIFAR-100 | ImageNet |
|------|------|----------|-----------|----------|
| Orchestra | ResNet-18 | 88.87 | 70.11 | 65.02 |
| FeatARC | ResNet-18 | 89.60 | 64.11 | 68.17 |
| LDAWA | ResNet-18 | 89.95 | 68.96 | 51.43 |
| **FedMAR** | ResNet-18 | **92.70** | **70.82** | 65.36 |
| **FedMAR** | Tiny-ViT | 90.03 | **71.28** | **75.99** |

Using ResNet-18, FedMAR outperforms all baselines on CIFAR-10/100 and matches them on ImageNet. With Tiny-ViT (comparable parameters/FLOPs), it leads across all benchmarks, particularly on ImageNet with a jump to 75.99, demonstrating the efficacy of MAR loss on Transformers.

### Key Findings
- **MIM's robustness advantage is structural**: Even when isolating the SSL paradigm, MIM is consistently more stable than CL under both types of non-IID data, with the gap widening as heterogeneity increases.
- **Connectivity is the critical "knob" for DecL**: DecL accuracy scales with average connectivity, and FL represents the most robust (fully connected) case—providing actionable advice for serverless scenarios (i.e., add direct connections).
- **Both MAR components are effective**: Ablations confirm that the adaptive alignment and cosine decay weights each contribute to the performance gain. Communicating only masked embeddings keeps overhead manageable and privacy intact.

## Highlights & Insights
- **Defining robustness as a provable scalar**: By using the span of RV bounds, the authors turn empirical comparisons (MIM/CL, DecL/FL) into rigorous mathematical orderings—this is the most elegant part of the paper.
- **Fundamental explanation for MIM's robustness**: The insight that CL aligns "another augmented image" (high randomness/bias) while MIM aligns "two parts of the same image" (shared information) is a profound observation that could be extended to other SSL pretext tasks.
- **A paradigm for theoretically guided algorithm design**: The corollary that MIM remains robust but still suffers from local drift directly motivated the MAR alignment term; this "prove then regulate" approach is highly reusable for other distributed training scenarios involving distribution shift.

## Limitations & Future Work
- **Reliance on highly simplified assumptions**: The use of linear embeddings, a formalized 2N-class non-IID model, and asymptotic bounds as $d\to\infty$ makes the proofs feasible but introduces a gap when applying the quantitative strength of these conclusions to complex non-linear deep networks.
- **Sensitivity as an asymptotic ordering**: The theorems provide ordinal relationships under $\lim_{d\to\infty}$ but do not quantify the exact performance gap for finite dimensions.
- **MAR as an exemplar rather than a final SOTA**: MAR is positioned as a demonstration of the theory. There is significant room for tuning the alignment terms and A-MMD bandwidth. Translating connectivity insights into actual topology-optimizing algorithms (auto-edge adding) remains future work.

## Related Work & Insights
- **vs. Wang et al. (D-SSL Theory)**: They proved SSL is more robust than supervised learning in distributed settings but only for "CL+FL." This paper extends the analysis to the full MIM/CL $\times$ DecL/FL matrix and explicitly captures the role of network connectivity.
- **vs. FedU / Orchestra / L-DAWA (D-SSL Algorithms)**: These focus on algorithmic improvements with theory used to "prove their specific method works." This paper takes the inverse approach, first building a universal understanding of framework robustness and then using MAR as a guiding example.
- **vs. Standard MMD Federated Alignment**: Previous works (e.g., Ma et al.) used fixed-bandwidth MMD. MAR's A-MMD adaptively selects kernel bandwidth, making it more robust to the varied embedding distributions of non-IID clients.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes a unified mathematical framework for the rigorous comparison of non-IID robustness in D-SSL, filling a theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive validation across datasets/backbones; however, many MAR ablations are relocated to the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with well-paired theorems and intuitive explanations, though heavy derivation is deferred to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides provable guiding principles for D-SSL algorithm design and network architecture selection with clear practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[ICLR 2026\] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods](understanding_the_learning_phases_in_self-supervised_learning_via_critical_perio.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[ICLR 2026\] Dual Perspectives on Non-Contrastive Self-Supervised Learning](dual_perspectives_on_non-contrastive_self-supervised_learning.md)
- [\[ICLR 2026\] Equivariant Splitting: Self-supervised learning from incomplete data](equivariant_splitting_self-supervised_learning_from_incomplete_data.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)

</div>

<!-- RELATED:END -->
