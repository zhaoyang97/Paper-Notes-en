---
title: >-
  [Paper Note] Continual Learning of Domain-Invariant Representations
description: >-
  [ICML 2026][continual learning] The authors explicitly inject "Domain-Invariant Representation Learning (DIRL)" into continual learning for the first time: using the replay buffer as a medium for multi-domain invariance…
tags:
  - "ICML 2026"
  - "continual learning"
  - "domain-invariant representation"
  - "replay buffer"
  - "VREX"
  - "Fishr / CORAL / MMD / ANDMask"
date: 2026-05-08
content_hash: 63ce05e6e6b214cc
---

# Continual Learning of Domain-Invariant Representations

**Conference**: ICML 2026  
**arXiv**: [2605.15775](https://arxiv.org/abs/2605.15775)  
**Code**: None  
**Area**: Continual Learning / Self-supervised Representation Learning / Domain Generalization  
**Keywords**: continual learning, domain-invariant representation, replay buffer, VREX, Fishr / CORAL / MMD / ANDMask

## TL;DR
The authors explicitly inject "Domain-Invariant Representation Learning (DIRL)" into continual learning for the first time: using the replay buffer as a medium for multi-domain invariance computation and domain-conditioned alignment. They propose five methods—⋆-CL-{VREX, Fishr, CORAL, MMD, ANDMask}—pushing target domain accuracy to SOTA across six datasets in vision, medicine, manufacturing, and ecology.

## Background & Motivation

**Background**: Mainstream continual learning (CL) methods are categorized into four types: optimization-based (AGEM, UPGD), regularization-based (EWC, SI, SNR), architecture-based (progressive nets), and replay-based (ER-ACE, FDR, LODE, STAR). Their common goal is the stability-plasticity trade-off: preventing forgetting on seen training domains while achieving good backward transfer (BWT).

**Limitations of Prior Work**: All existing methods optimize performance solely on "seen domains," causing models to easily learn domain-specific shortcuts (e.g., color, texture, hospital-level bias). This leads to failure when deployed in a completely new target domain. This is a specific manifestation of shortcut learning in CL scenarios—high in-domain accuracy but poor out-of-domain performance.

**Key Challenge**: Existing DIRL methods (VREX, Fishr, CORAL, MMD, ANDMask) rely on joint access to multiple domains to simultaneously optimize invariance constraints. Conversely, CL is sequential, and data from past domains is no longer visible. Simply storing a domain-level statistic $\Phi_{s'}$ as an "anchor" and matching the current batch to it (naïve extension) fails to reproduce the semantics of multi-domain joint optimization, resulting in limited gains.

**Goal**: (i) Learn true domain-invariant representations on sequential data streams; (ii) evaluate under a deployment-oriented protocol—sequential train → deploy → test on new target domains; (iii) balance multi-domain invariance and stability without exceeding classical CL buffer budgets.

**Key Insight**: The replay buffer in CL naturally serves as a medium where "multiple domains coexist." The authors move invariance computation to the replay batch (rather than just the current domain) and introduce an alignment loss to prevent replay representations from drifting as training progresses.

**Core Idea**: A triplet of "replay-augmented ERM" + "multi-domain invariance penalty on replay batches" + "domain-conditioned invariance alignment" is used to rewrite any DIRL invariant (risk, gradient, feature, kernel embedding, gradient-sign mask) into a CL-friendly version.

## Method

### Overall Architecture
Setting: The model $h=g_\omega\circ f_\theta$ is trained sequentially on domains $S=\{D_1,\dots,D_k\}$. Each domain allows access only to its own data and a small buffer $M$ ($|M_{s'}|\ll|D_{s'}|$), with evaluation on an unseen target domain $D^t$. The overall training objective is $\min_{\theta,\omega} L^{\text{replay}}_{\text{ERM}}(\theta,\omega)+\lambda P^{\text{replay}}_s(\theta,\omega)+\beta L^{\text{align}}(\theta,\omega)$, where ERM is performed on current ∪ replay data, the second term is the multi-domain invariance penalty, and the third is the domain-conditioned alignment term.

### Key Designs

1.  **Replay-augmented ERM + Domain-partitioned buffer**:
    - **Function**: Ensures "multi-domain coexistence" becomes a real condition for every training step rather than a post-hoc approximation.
    - **Mechanism**: The buffer is partitioned by domain as $M=\bigcup_{s'<s}M_{s'}$. Each sample is stored as $(x,y,z)$, where $z$ is auxiliary information at the time of insertion (e.g., logits $h(x;\theta_{s'},\omega_{s'})$ or features $f_{\theta_{s'}}(x)$). The ERM term is expanded to $L^{\text{replay}}_{\text{ERM}}=\mathbb{E}_{(x,y)\sim B}[L(h(x),y)]$, where $B=\bigcup_{e\le s}B_e$ includes both the current domain batch $B_s$ and all replay batches $B_{s'}$.
    - **Design Motivation**: Traditional CL replay only prevents forgetting; here, replay simultaneously serves to "provide evidence of multi-domain invariance" and "prevent forgetting," thereby approximating the joint-access assumption of DIRL.

2.  **Multi-domain Invariance Computation (Preplay)**:
    - **Function**: Defines a unified penalty operator "on replay + current batches" for each candidate invariant, bringing 5 DIRL methods into CL.
    - **Mechanism**: Each domain uses a statistic $\widehat\phi_{s'}=\phi(\theta,\omega;B_{s'})$, with a penalty $P^{\text{replay}}_s=\textsc{InvPenalty}(\{\widehat\phi_{s'}\}_{s'\le s})$. Five specific instances:
        - ⋆-CL-VREX: $\phi_{s'}=\widehat r_{s'}=\mathbb{E}_{B_{s'}}[L(h(x),y)]$, penalizing $\frac{1}{s}\sum_{s'\le s}(\widehat r_{s'}-\bar r)^2$, i.e., cross-domain risk variance.
        - ⋆-CL-Fishr: $\phi_{s'}=\widehat v_{s'}=\mathrm{Var}_{B_{s'}}(\nabla_\omega L)$, penalizing $\frac{1}{s}\sum\|\widehat v_{s'}-\bar v\|_2^2$ to match the classification head's gradient variance.
        - ⋆-CL-CORAL: $\phi_{s'}=(\widehat\mu_{s'},\widehat\Sigma_{s'})$ first/second-order moments of features, penalizing mean differences + Frobenius norm of covariance differences.
        - ⋆-CL-MMD: $\phi_{s'}=\widehat\mu^z_{s'}=\mathbb{E}_{B_{s'}}[z(f_\theta(x))]$, where $z$ denotes random Fourier features of an RBF kernel, penalizing mean embedding distances.
        - ⋆-CL-ANDMask: Uses domain-level gradients $g_{s'}=\nabla_{\theta,\omega}L^{\text{ERM}}(B_{s'})$, constructs a sign-agreement mask $m=\mathbb{I}(\frac{1}{s}|\sum_{s'}\mathrm{sgn}(g_{s'})|\ge\tau)$, and updates $\nabla\leftarrow m\odot\frac{1}{s}\sum_{s'}g_{s'}$.
    - **Design Motivation**: Computing invariance on "simultaneously visible multi-domain batches" restores the joint multi-domain semantics of the original DIRL; as long as the buffer can sample representative batches, it is far more accurate than using static priors.

3.  **Domain-conditioned invariance alignment ($L_{\text{align}}$)**:
    - **Function**: Offsets the degradation where representations of past domains in the replay buffer drift as the model continues training.
    - **Mechanism**: Invokes the prior $\Phi_{s'}$ at insertion time (calculated via Welford online mean at the end of domain $s'$) and aligns the current model's statistic on $B_{s'}$ back to it: $L^{\text{align}}=\sum_{s'<s}d(\widehat\phi_{s'}(\theta,\omega;B_{s'}),\Phi_{s'})$. The key difference from the naïve method is that the naïve approach matches the "current domain batch" to "past domain priors" (forcing the flattening of real cross-domain differences), whereas ours matches the "replayed past domain batch" back to its "own historical statistic," resembling knowledge distillation.
    - **Design Motivation**: The authors found that with only Preplay, replay sample representations are dragged away by new domain optimization, causing invariance learned in past steps to be "quietly forgotten." $L_{\text{align}}$ uses distillation-style anchors to preserve the historical identity of invariance.

### Loss & Training
Total objective: $L^{\text{replay}}_{\text{ERM}}+\lambda P^{\text{replay}}_s+\beta L^{\text{align}}$. For large image datasets, ImageNet pre-trained ResNet18 is used; RotatedMNIST uses a 4-layer CNN, and Covertype uses a 4-layer MLP. Buffer sizes are 1000 (small datasets) or 5000 (others), with $\lambda, \beta$ determined via dataset-level HP search. The upper bound is URM (offline DIRL with access to all source domains), and baselines include 13 SOTA CL methods + 3 CDA/CTTA (TENT, SHOT++, CoTTA).

## Key Experimental Results

### Main Results
Six datasets: RotatedMNIST, CIFAR10C, TinyImageNetC, WM811K (wafer manufacturing defects, macro F1), Covertype, Camelyon17 (medical). Average ± SE reported over 3 independent runs. ⋆-CL-CORAL / ⋆-CL-MMD / ⋆-CL-VREX ranked 1st / 2nd / 3rd on average.

| Dataset | Metric | Ours ⋆-CL-CORAL | Prev. SOTA | Gain |
|---------|--------|----------------|------------|------|
| RotatedMNIST | acc (%) | 72.8 | 68.7 (CoPE) | +4.1 |
| CIFAR10C | acc (%) | 68.5 | 69.5 (STAR) | -1.0 (CORAL 2nd, ⋆-CL-MMD 69.0) |
| TinyImageNetC | acc (%) | 25.0 | 29.0 (ER-ACE) | -4.0 (⋆-CL-Fishr 29.0 / ⋆-CL-VREX 26.3) |
| WM811K | Macro F1 (%) | 84.8 | 85.4 (ER-ACE) | -0.6 (⋆-CL-MMD 85.5 highest) |
| Covertype | acc (%) | 45.2 | 41.2 (SARL) | +4.0 |
| Camelyon17 | acc (%) | 91.7 | 91.0 (AGEM) | +0.7 |
| **Average** | acc/F1 (%) | **64.7** | 62.8 (ER-ACE) | **+1.9** |

Overall, ⋆-CL-CORAL 64.7 > ⋆-CL-VREX 63.4 > ⋆-CL-MMD 63.1 > ER-ACE 62.8 > STAR 62.1, leaving finetune (50.4) and SARL (54.0) over 10 pp behind. Improvements are ~6 pp over optimization-based, ~10 pp over regularization-based, and ~2 pp over replay-based methods, with an ~8.6 pp remaining gap to the URM upper bound.

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| Full ⋆-CL (Preplay + Lalign) | Avg 64.7 | Complete method |
| naïve-CL-{VREX,Fishr,CORAL,MMD,ANDMask} | Barely higher than finetune | Static prior Φ fails the multi-domain joint semantics |
| Without $L^{\text{align}}$ (β=0) | Performance drop | Indicates alignment is critical for generalization (not just stability) |
| Dynamically recalculating $\Phi_{s'}$ | Performance drop | Anchor failure; proves Lalign must use insertion-time priors |
| Buffer reduced to 50% / 25% | Still leads replay baseline by ~4 pp | Invariance constraints allow small buffers to hold up |
| CDA / CTTA Baselines (TENT/SHOT++/CoTTA) | Lags by up to 10 pp | Shows CL+DIRL provides a more fundamental advantage over test-time adaptation |

### Key Findings
- **Lalign is not just for preventing forgetting; it is key to generalization**: Contrasting the view of alignment as a stability tool, experiments show it supports OOD generalization—cross-domain accuracy drops significantly when disabled.
- **Each invariant has its strengths**: ⋆-CL-CORAL excels in low-data/strong statistical shift scenarios, ⋆-CL-Fishr is more stable under strong non-stationarity like pixel corruption (TinyImageNetC), and ⋆-CL-MMD nearly matches CORAL on distribution alignment tasks. ANDMask fails on TinyImageNetC (11.8%) due to overly sparse sign-agreement masks.
- **In-domain integrity, OOD surge**: All ⋆-CL methods outperform finetune/regularization baselines in-domain, suggesting learned invariant structures benefit source domains as well, validating the hypothesis that "DIRL need not sacrifice in-domain performance."
- **BWT is almost positive**: The backwards transfer of the ⋆-CL series is non-negative or positive, meaning learning new domains improves accuracy on old ones—a rare phenomenon in CL attributed to the "invariant structure being a shared causal mechanism."

## Highlights & Insights
- **Systematic embedding of DIRL into CL for the first time**: While previous DIRL assumed joint access, the authors approximate this via replay + multi-domain batches and show that naïve "static priors" cannot replicate joint optimization—a negative result included in the ablation.
- **Value of the deployment-oriented evaluation protocol**: Changing CL evaluation from "held-out old domains" to "completely unseen target domains" reveals that methods that "seem not to forget" actually fail to learn invariant structures. This protocol itself should be adopted by the CL community.
- **Transferable Lalign anchor design**: Using statistics from the "time of insertion" as anchors (rather than dynamic recalculation) is a lightweight distillation that could benefit other online scenarios requiring "historical identity" (Federated Learning, self-supervised pretraining).

## Limitations & Future Work
- **Gap to URM upper bound (~8.6 pp)**: On RotatedMNIST, URM 81.3 vs ⋆-CL-CORAL 72.8, suggesting replay-based multi-domain invariance is still far from the joint-DIRL ceiling; smarter sample selection or generative replay may be next.
- **Buffer dependency**: While leading at 25% buffer size, performance absolute values drop to 50–60%; buffer-free settings were not discussed.
- **Lacking selection guidelines for the 5 invariants**: The paper suggests "trying them all," without providing guidance based on data characteristics (e.g., type of inter-domain drift).
- **ANDMask failure on hard tasks**: Results on TinyImageNetC (11.8%) were worse than finetuning; the authors admit sign-agreement is too strict for heterogeneous domains and consider softening or adaptive thresholds for future work.

## Related Work & Insights
- **vs. Classic CL (EWC, SI, ER-ACE, STAR)**: Classic methods lack a "cross-domain invariance" term. This work proves that adding Preplay+Lalign improves performance by ~2 pp on average without changing the buffer budget.
- **vs. DIRL (VREX, Fishr, CORAL, MMD, ANDMask)**: This work "CL-izes" these five invariances, creating five ⋆-CL methods and using the negative naïve baseline to show that simply storing priors is insufficient.
- **vs. CDA/CTTA (TENT, SHOT++, CoTTA)**: CDA/CTTA assume unsupervised updates on the target domain at deployment. This setting is stricter (frozen after deployment) yet still leads by 10 pp, proving "learning invariance first" is more fundamental than "post-hoc adaptation."
- **vs. URM (Krishnamachari 2024)**: URM uses offline joint optimization of all source domains and serves as the upper bound; ⋆-CL-CORAL is the closest method under the current sequential setting.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically bridges DIRL and CL, solving the fundamental flaw of "naïve static priors" with the Preplay+Lalign structure.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets × 17 baselines × 3 runs + 5 ⋆-CL methods + naïve ablation + buffer reduction/target domain/CDA/CTTA/in-domain/BWT; rare coverage.
- Writing Quality: ⭐⭐⭐⭐ Table 1 aligns the five methods to a unified template clearly; Fig 1's protocol makes the motivation immediate. However, the ANDMask failure on TinyImageNetC lacks deep analysis.
- Value: ⭐⭐⭐⭐ Directly applicable to CL in medicine/manufacturing/driving where models must deploy to "new machines/hospitals"; the evaluation protocol could influence future CL research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](learning_permutation-invariant_macroscopic_dynamics.md)
- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/others/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[ICML 2026\] CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)
- [\[AAAI 2026\] ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](../../AAAI2026/others/parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)

</div>

<!-- RELATED:END -->
