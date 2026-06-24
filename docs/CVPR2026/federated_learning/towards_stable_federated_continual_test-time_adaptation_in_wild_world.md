---
title: >-
  [Paper Note] Towards Stable Federated Continual Test-Time Adaptation in Wild World
description: >-
  [CVPR 2026][Federated Learning][Federated Continual Test-Time Adaptation] This paper proposes **BPFedCTTA**, which unifies "Federated Continual Test-Time Adaptation (FedCTTA)" from a Bayesian perspective: treating the global model as a Gaussian prior, stabilizing the local adaptation of each unlabeled client through MAP estimation (BPA), and selectively fusing client updates via an uncertainty gate calculated from output entropy (UGSA). This approach achieves adaptation to ne…
tags:
  - "CVPR 2026"
  - "Federated Learning"
  - "Federated Continual Test-Time Adaptation"
  - "Bayesian Prior"
  - "MAP Estimation"
  - "Uncertainty Gating"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 3998ef5247f300a4
---

# Towards Stable Federated Continual Test-Time Adaptation in Wild World

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Towards_Stable_Federated_Continual_Test-Time_Adaptation_in_Wild_World_CVPR_2026_paper.html)  
**Code**: https://github.com/LiwenWang919/BPFedCTTA  
**Area**: Federated Learning / Test-Time Adaptation / Continual Learning  
**Keywords**: Federated Continual Test-Time Adaptation, Bayesian Prior, MAP Estimation, Uncertainty Gating, Catastrophic Forgetting

## TL;DR
This paper proposes **BPFedCTTA**, which unifies "Federated Continual Test-Time Adaptation (FedCTTA)" from a Bayesian perspective: treating the global model as a Gaussian prior, stabilizing the local adaptation of each unlabeled client through MAP estimation (BPA), and selectively fusing client updates via an uncertainty gate calculated from output entropy (UGSA). This approach achieves adaptation to new domains while preserving the global model and mitigating catastrophic forgetting in extreme heterogeneous scenarios where clients arrive sequentially with completely unrelated distributions.

## Background & Motivation

**Background**: Federated Learning (FL) enables collaborative training across decentralized data while protecting privacy, but deployed models often encounter test data with distributions different from the training data. Personalized Federated Learning (PFL) can mitigate client heterogeneity but almost always assumes labeled data at target clients. Test-Time Adaptation (TTA) enables unlabeled adaptation but is designed for "centralized, single-domain" models.

**Limitations of Prior Work**: Directly transplanting TTA into federated scenarios causes issues. The FL global model is an "average solution" aggregated from multiple heterogeneous sources, residing in a **flat loss basin**—this favors generalization but is extremely sensitive to noisy updates. Unsupervised TTA updates at a single client lack proper regularization, making it easy to push parameters out of this stable basin, leading to rapid overfitting to the current client and local model drift. although prior FedCTTA work ([40]) exists, it assumes **predefined clusters + synchronous updates** and only handles known spatial heterogeneity. It cannot handle the extreme spatio-temporal heterogeneity where clients arrive **asynchronously and sequentially with completely unrelated distributions** in real deployments, nor does it have mechanisms to constrain local adaptation stability or quantify update reliability.

**Key Challenge**: The authors condense the problem into two coupled challenges. **(C1) Unstable unlabeled local adaptation**—naive TTA without regularization pushes the model away from the flat basin. **(C2) Unsafe continual global evolution**—the server has no ground truth to verify the quality of client updates in unlabeled sequential scenarios. Blindly aggregating an overfitted or severely drifted update like FedAvg "poisons" the global model and causes forgetting of accumulated knowledge. Meanwhile, data replay commonly used in continual learning is infeasible under FL privacy constraints.

**Goal / Key Insight**: The authors address C1 and C2 through a **unified probabilistic perspective**—treating the global model as a continuously evolving **prior** and reformulating local adaptation at each client as **approximate posterior inference** guided by unlabeled data. Consequently, "regularized local adaptation" and "safe aggregation" are not two temporary patches but two ends of the same Bayesian framework.

**Core Idea**: Replace "unconstrained TTA + FedAvg" with "Global model as prior + MAP for local stability, and output entropy as reliability gating for global stability," modeling continual federated adaptation as a recursive Bayesian filtering process.

## Method

### Overall Architecture
BPFedCTTA is designed for the FedCTTA setting: the system first trains an initial global model $\theta_G^{(0)}$ using $N$ labeled source clients via standard FedAvg. After deployment, $K$ target clients $\{C_k\}$ with only unlabeled data streams arrive **sequentially and asynchronously** at times $t_1, \dots, t_K$, with unrelated distributions $P_k$ and a strict ban on sharing features or logits. The overall objective is a trade-off between "local stability" and "global evolution":

$$\min_{\{\theta'_k,\theta_G^{(k)}\}} \sum_{k=1}^{K}\Big[\underbrace{\tilde{\mathcal{R}}_k(\theta'_k)}_{\text{Local Stability}} + \lambda\underbrace{\big(\textstyle\sum_{s=1}^{N}\mathcal{R}_s(\theta_G^{(k)})\big)}_{\text{Global Evolution}}\Big]$$

The framework uses a **two-level structure**: the client-side uses **BPA** to perform MAP local adaptation treating the current global model as a prior, yielding the personalized model $\theta'_k$ and reporting its uncertainty; the server-side uses **UGSA** to fuse this update into the global model $\theta_G^{(k-1)}\to\theta_G^{(k)}$ according to an uncertainty gate, then distributes it to the next client, forming a continuously evolving system.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Source clients with labeled data<br/>FedAvg training"] --> B["Global Model θ_G<br/>(Flat basin · Prior)"]
    B -->|Distributed to sequential<br/>unlabeled clients C_k| C["Bayesian Prior-guided Adaptation BPA<br/>Global model as Gaussian prior · MAP estimation"]
    C --> D["Personalized Model θ'_k<br/>+ Prediction entropy uncertainty σ_k"]
    D --> E["Uncertainty-Gated Single-client Aggregation UGSA<br/>w_k=softmax(-βσ_k) Gated fusion"]
    E -->|θ_G←θ_G+γ·w_k·(θ'_k−θ_G)| B
    E --> F["Evolved Global Model<br/>Deployed to next client"]
```

### Key Designs

**1. Bayesian Prior-guided Adaptation (BPA): Treating the global model as a prior and utilizing MAP estimation to lock unlabeled local adaptation within a stable basin.**

This design targets C1—adapting a complex model with only unlabeled data $X_k$ is inherently **ill-posed**, making it prone to instability or overfitting when data is scarce or noisy. The authors model local adaptation as an approximate posterior $p(\theta|X_k,\theta_G^{(k-1)})$: an isotropic Gaussian prior $p(\theta|\theta_G^{(k-1)})=\mathcal{N}(\theta|\theta_G^{(k-1)},\Sigma_0)$ is defined around the current global model, expressing the belief that the optimal parameters should stay near the well-generalized global solution. The likelihood $p(X_k|\theta)\propto\exp(-\mathcal{L}(\theta;X_k))$ of unlabeled data is implicitly modeled by a TTA loss (encouraging high confidence and local consistency). Instead of seeking the full posterior, BPA takes the **MAP** (posterior mode). After expanding the log-prior, the objective becomes entropy minimization plus a quadratic regularization term:

$$\theta'_k=\arg\min_\theta\Big(\mathcal{L}_{\text{EM}}(\theta)+\tfrac{1}{2}(\theta-\theta_G^{(k-1)})^\top\Sigma_0^{-1}(\theta-\theta_G^{(k-1)})\Big)$$

This Bayesian-derived quadratic regularization term $\mathcal{L}_{\text{BPA}}$ anchors adaptation to high-probability regions of the global prior, effectively providing TTA with a "principled, data-driven" regularizer—exactly what naive TTA lacks. Remarkably, the prior precision (inverse covariance) **scales adaptively with model uncertainty**: $\Sigma_0^{-1}=\text{diag}\big(\frac{1}{\sigma_0^2}\mathbb{E}_{x\in X_k}[\mathcal{H}(p(y|x;\theta_G^{(k-1)}))]\big)$, where $\mathcal{H}$ is Shannon entropy. When the model is confident (low entropy), it tightens the prior to lock parameters; when the model is uncertain (high entropy), it loosens the prior to allow room for adaptation, thereby preventing drift while allowing learning on heterogeneous unlabeled streams.

**2. Uncertainty-Gated Single-client Aggregation (UGSA): Treating aggregation as a Bayesian belief update and using output entropy as a gate to block unreliable updates.**

This design targets C2—the server has no ground truth to judge if a client update should be trusted. The probabilistic nature of BPA provides the answer: since local adaptation is posterior inference, the **predictive entropy of the adapted model** can directly quantify the reliability of the update. UGSA views aggregation as a Bayesian belief update of the global parameters: given the previous global model $p(\theta_G^{(k-1)})=\mathcal{N}(\theta_G^{(k-1)},\Sigma_G^{(k-1)})$ and the client posterior $q_k(\theta)=\mathcal{N}(\theta'_k,\Sigma_k)$, the new global distribution is obtained by minimizing a weighted sum of two KL divergences (fitting the client posterior without deviating from the old global prior). The solution is the geometric mean in the density space $p(\theta_G^{(k)})\propto p(\theta_G^{(k-1)})^{1-w_k}q_k(\theta)^{w_k}$. Gating weights are provided by client uncertainty:

$$\sigma_k=\mathbb{E}_{x\in X_k}[\mathcal{H}(p(y|x;\theta'_k))],\qquad w_k=\frac{\exp(-\beta\sigma_k)}{\sum_j\exp(-\beta\sigma_j)}$$

$\beta$ controls the decay sharpness—contributions from confident (low entropy) clients are amplified, while those from high uncertainty clients are suppressed to avoid noisy drift. This is equivalent to **Federated Bayesian Filtering**: $p(\theta_G^{(k)}|X_{1:k})\propto q_k(\theta|X_k)^{w_k}\,p(\theta_G^{(k-1)}|X_{1:k-1})$, which is precision-weighted evidence accumulation, allowing the global model to recursively and safely absorb new knowledge under unlabeled heterogeneous conditions. Practically, a diagonal approximation $\Sigma_G$ is used for speed, resulting in a concise update:

$$\theta_G^{(k)}=\theta_G^{(k-1)}+\gamma\,w_k\,(\theta'_k-\theta_G^{(k-1)})$$

where $\gamma$ is the server learning rate. Compared to the equal-weight averaging of FedAvg, the key difference in UGSA is that "updates are filtered through the reliability gate $w_k$ before injection," which is the fundamental reason it can block poisoning updates and mitigate catastrophic forgetting.

## Key Experimental Results

### Main Results
Classification used ResNet-18 (CIFAR10-C / CIFAR100-C, corruption severity 5, 10 clients / 50 rounds), natural image segmentation used SegFormer-B5 (Cityscapes→ACDC 4 types of adverse weather), and medical segmentation used U-Net (Retinal Fundus / Prostate MRI, naturally domain-partitioned by medical centers). Comparisons include FL (FedAvg), PFL (FedProx/FedBN/FedGA), TTA/CTTA (Tent/CoTTA/BeCoTTA/TCA), and TTA+FL (ATP/FedTHE+/FedCTTA/TTA-FedDG).

| Benchmark | Metric | Ours | Sub-optimal Baseline | Description |
|-----------|--------|------|----------------------|-------------|
| CIFAR10-C | Avg Acc | **68.44** | 68.09 (TTA-FedDG) | Sequential corruption adaptation; lead is small |
| CIFAR100-C| Avg Acc | **67.58** | 65.58 (TTA-FedDG) | Advantage widens with more classes (+2.0) |
| Cityscapes→ACDC | mIoU(Seq1→Seq4) | 62.88→**65.19** | FedCTTA 61.66→63.28 | **Gradual rise** within the sequence; shows continual learning effectiveness |
| Med. Seg. | Avg Dice | 75.06~75.52| Trades wins with TTA-FedDG | Top 2 in 3 out of 4 sequences; most stable across sequences |

In the comparison group, standard TTA significantly degrades in sequential scenarios (CoTTA drops from 58.60% to 56.99% in Table 2; Tent collapses from 57.06% to 52.72%), PFL performs mediocrely due to the need for labeled data (FedBN 64.34/57.91), while the source model and FedAvg/local fine-tuning **consistently drop points** over time in segmentation (FedAvg 54.30→53.51), highlighting the stability of BPFedCTTA's mIoU rising along the sequence.

### Ablation Study
Component ablation (Table 4, incremental addition) and comparison of uncertainty measurement methods (Table 5):

| Configuration | CIFAR10-C | CIFAR100-C | Cityscapes Seq1→Seq4 | Description |
|---------------|-----------|------------|----------------------|-------------|
| FedAvg        | 65.49     | 59.50      | 54.30→53.51          | No adaptation; segmentation degrades over time |
| + Tent        | 64.72     | 57.51      | 57.06→52.72          | Naive TTA collapses in sequential scenarios |
| + BPA         | 67.15     | 62.89      | 59.82→58.30          | Local regularization only; +2.43/+5.38 over Tent |
| + Tent + UGSA | 66.83     | 61.45      | 58.53→58.12          | Aggregation gating only; limited mitigation of forgetting |
| **Full (BPA+UGSA)** | **67.74** | **64.57** | 62.88→**65.19** | Complementary; achieves gradual rise within the sequence |

| UGSA Uncertainty Metric | CIFAR10-C | Time per client(s) | Forgetting ∆% (Lower is better) |
|-------------------------|-----------|--------------------|-----------------------------------|
| Pred. Entropy (Default) | 68.44     | 4.49               | 4.21                              |
| Energy Score            | 68.62     | 8.83               | 3.95                              |
| Pred. Consistency       | 68.79     | 19.46              | 3.82                              |
| MC Dropout (T=10)       | 68.75     | 64.92              | 3.45                              |
| Ensemble (M=3)          | 68.65     | 20.13              | 3.54                              |

### Key Findings
- **BPA and UGSA are complementary and both essential**: BPA stabilizes local adaptation (+5.38% on CIFAR100-C vs. Tent), while UGSA ensures safe global merging. Neither works as well alone, and only the full version achieves the 62.88→65.19 gradual rise in segmentation.
- **Uncertainty measurement is a speed-accuracy trade-off**: Predictive entropy offers the best balance (4.49s). Multi-view methods (consistency / MC Dropout / ensemble) yield lower forgetting and slightly higher accuracy, but MC Dropout is too slow (64.9s/client) for practical use, hence entropy is the default choice.
- **Hyperparameter $\beta$ is critical**: When $\beta=0$ (UGSA gating off), stability collapses to 51.8%. $\beta=0.2$ yields a stability peak of 62.9% and optimal overall mIoU of 63.5%. A server learning rate of $\gamma=0.5$ is best; too small $\gamma$ suppresses plasticity, while too large $\gamma$ hurts stability.
- **Heterogeneity Resilience**: Under extreme Non-IID (Dirichlet $\alpha=0.05$), it reaches 64.5%/62.5%, which is 1.4%/1.8% higher than the strongest baseline. Degradation from $\alpha=10$ to $\alpha=0.05$ is only 5.3%/6.7%, significantly less than FedAvg's 9.0%/9.3%.

## Highlights & Insights
- **A unified Bayesian view for two coupled problems**: Unifying "local regularization" and "safe aggregation" into prior-posterior inference. BPA's quadratic regularization and UGSA's entropy gate are not temporary patches but natural outcomes of the same framework—this "mechanistic consistency" makes the method logical and explainable.
- **Reusing flat basin geometry as a prior**: The authors perceptively note that FL global models reside in flat basins sensitive to noise, so they directly use this basin as a Gaussian prior to anchor adaptation. This converts an "FL model property" into a "TTA regularizer."
- **Uncertainty drives both local and global**: Predictive entropy adaptively scales prior precision in BPA and serves as a gating weight in UGSA. A single metric bridges both ends, simplifying the engineering.
- **Minimal implementation form**: UGSA simplifies to a single line: $\theta_G\!\leftarrow\!\theta_G+\gamma w_k(\theta'_k-\theta_G)$. It is essentially FedAvg with reliability weighting, making it almost zero-cost to integrate into existing federated frameworks. This "theory-heavy, implementation-light" transition is excellent.

## Limitations & Future Work
- **Thin advantage on CIFAR10-C**: 68.44 vs. TTA-FedDG 68.09 is only +0.35, and it is outperformed by baselines on several corruption types (e.g., Snow/Frost), indicating limited relative gain in simple sequential scenarios.
- **Numerical inconsistencies**: The text claims 67.58% on CIFAR100-C, but the full model in Tables 4 and 5 shows only 64.57% (the original text's context regarding variants should be checked). 
- **Strong assumptions on Gaussian priors and diagonal covariance**: BPA uses an isotropic Gaussian prior and UGSA uses a diagonal $\Sigma_G$ approximation, ignoring correlations between parameters. Theoretical proofs for C1/C2 are in the appendix and not expanded in the main text.
- **Risk in relying on entropy as a reliability proxy**: On OOD samples, the model may be "confidently wrong" (low entropy but incorrect), in which case the gate would amplify bad updates. Ablations showing lower forgetting with multi-view uncertainty suggest entropy is not the most stable reliability metric.
- **Assumed independence of client distributions**: In reality, sequential clients often have temporal correlation. The method does not exploit this structure; modeling the relationship between adjacent clients might yield further gains.

## Related Work & Insights
- **vs. Naive TTA (Tent/CoTTA)**: These rely on output-level entropy minimization or pseudo-label self-training. Without regularization in the federated flat basin, they collapse in sequential scenarios (Tent 57.06→52.72). This paper anchors them with a global prior quadratic regularization, turning "unconstrained adaptation" into "MAP-constrained adaptation."
- **vs. Previous FedCTTA [40]**: Prior work assumes synchronous, predefined clusters, and known spatial shifts, and shares noisy sample logits (leakier privacy) with a focus on personalization. This paper targets asynchronous sequential arrivals with unrelated distributions and strictly prohibits feature/logit sharing, focusing on the safe global evolution.
- **vs. Personalized FL (FedBN/FedProx/FedGA)**: Most PFL assumes labeled target clients and focuses on the training phase. They cannot handle test-time domain shift and sequential arrival. This paper is entirely unlabeled and oriented toward post-deployment continual adaptation.
- **vs. TTA+FL (FedTHE+/TTA-FedDG)**: Also integrates TTA into FL, but this paper differentiates itself via "Bayesian unification + uncertainty-gating aggregation," showing wider margins in difficult scenarios (high class count/long sequences).

## Rating
- Novelty: ⭐⭐⭐⭐ Unified Bayesian perspective integrates local regularization and safe aggregation; FedCTTA setting is more realistic for deployment, though individual components (MAP regularization, entropy gating) have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers classification + natural/medical segmentation benchmarks, over ten baselines, Non-IID, and hyperparameter sensitivity. However, numerical inconsistencies in tables and thin advantages on CIFAR10-C are noted.
- Writing Quality: ⭐⭐⭐⭐ Motivation-Challenge-Method logic is clear; framework is self-consistent and readable. Minor data mismatches between text and tables slightly affect credibility.
- Value: ⭐⭐⭐⭐ Addresses real-world federated deployment issues (privacy sensitivity, asynchronous arrival) with a minimal implementation (single-line gated aggregation). Highly relevant for industrial continual adaptation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning](hilora_hierarchical_low-rank_adaptation_for_personalized_federated_learning.md)
- [\[CVPR 2026\] From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity](from_selection_to_scheduling_federated_geometry-aware_correction_makes_exemplar_.md)
- [\[CVPR 2026\] FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization](fedadamom_adaptive_momentum_for_improved_generalization_in_federated_optimizatio.md)
- [\[CVPR 2026\] Domain Sensitive Federated Learning with Fisher-Informed Pruning](domain_sensitive_federated_learning_with_fisher-informed_pruning.md)
- [\[CVPR 2026\] Personalized Federated Training of Diffusion Models with Privacy Guarantees](personalized_federated_training_of_diffusion_models_with_privacy_guarantees.md)

</div>

<!-- RELATED:END -->
