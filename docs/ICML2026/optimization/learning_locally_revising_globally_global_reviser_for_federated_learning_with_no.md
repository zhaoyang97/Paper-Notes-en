---
title: >-
  [Paper Note] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels
description: >-
  [ICML 2026][Optimization][Federated Learning] This paper observes a "delayed memory" phenomenon where the global model in Federated Learning (FL) memorizes noisy labels significantly slower than in centralized training (…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Federated Learning"
  - "Label Noise"
  - "EMA Distillation"
  - "GMM Sample Selection"
  - "Privacy Protection"
date: 2026-05-08
content_hash: e4b7e7df6b1de67d
---

# Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels

**Conference**: ICML 2026  
**arXiv**: [2412.00452](https://arxiv.org/abs/2412.00452)  
**Code**: https://github.com/cs-yuxintian/FedGR-ICML26 (Available)  
**Area**: Federated Learning / Learning with Noisy Labels / Optimization  
**Keywords**: Federated Learning, Label Noise, EMA Distillation, GMM Sample Selection, Privacy Protection  

## TL;DR
This paper observes a "delayed memory" phenomenon where the global model in Federated Learning (FL) memorizes noisy labels significantly slower than in centralized training (memorization rate $\le30\%$ on CIFAR-10). Based on this, the authors propose FedGR, which utilizes a server-side GMM to jointly filter samples and estimate per-client noise ratios using aggregated loss proxies. It periodically "revises" the local EMA teacher using global parameters for distillation and incorporates global-local representation consistency regularization. These three synergistic modules achieve significant gains over 8 SOTA baselines under the dual heterogeneity (label noise $\times$ non-IID) setting on CIFAR-10/100 and Clothing1M.

## Background & Motivation

**Background**: Federated Learning (FL) combines model aggregation with data locality. FedAvg has become the de facto standard for privacy-sensitive scenarios like medical imaging and recommendation systems. Meanwhile, Centralized Learning with Noisy Labels (C-LNL) has mature solutions like Co-teaching and DivideMix, which primarily exploit the "memorization effect"—where networks learn clean samples before overfitting to noise—for sample selection.

**Limitations of Prior Work**: Directly applying C-LNL methods to FL faces two types of heterogeneity: (1) Huge variances in noise **types** (symmetric/asymmetric/mixed) and **ratios** across clients; (2) Data distribution Non-IID leading to class imbalance. The superposition of these factors causes client-independent sample selection or dual-network mechanisms (e.g., Co-teaching / DivideMix) to frequently fail in "clean-rate estimation," while consensus methods relying on shared statistical features often violate privacy boundaries.

**Key Challenge**: Independent client filtering leads to insufficient samples and jittery noise estimation. Sharing statistics between clients risks leaking distribution information. Simultaneously, local models are easily corrupted by high-noise clients, whereas the global model, though robust, struggles to fit specific local distributions.

**Goal**: Solve both "noise estimation" and "local training regularization" without leaking any information related to the joint distribution of $(\mathbf{x}, \mathbf{y})$.

**Key Insight**: The authors empirically find that the global model in FL memorizes noisy labels much slower than centralized models (under CIFAR-10 Sym noise, centralized models eventually memorize $\ge80\%$ of noisy labels, while the FL global model stays $\le30\%$). Furthermore, its test accuracy does not collapse after the "noisy peak" as centralized models do. They name this phenomenon "Intrinsic Label-Noise Robustness of FL" and leverage the global model as a trusted "Reviser."

**Core Idea**: Treat the "intrinsic delayed memory of the global model" as a free lunch for privacy protection. The server performs GMM filtering using only per-sample loss proxies (independent of data distribution) and sends the results back to clients. Meanwhile, the local EMA teacher is periodically "revised" by global parameters to prevent noise accumulation.

## Method

### Overall Architecture
FedGR adds three modules to the standard FedAvg loop, designed for local training of client $k$ at round $t$. The total loss is $\mathcal{L}_k = \mathcal{L}_k^{SR} + \lambda_{\mathcal{B}} \mathcal{B}_k + \lambda_{\mathcal{R}} \mathcal{R}_k$. The workflow consists of: (1) Clients use the global model $\mathbf{w}_g^{t-1}$ to calculate a moving average loss proxy $\bar{\ell}_i^t$ for each sample and upload them to the server; (2) The server fits a two-component GMM to proxies from all clients, partitions samples into clean/noisy subsets based on the "clean posterior probability" $q_{i,k}$, estimates per-client noise rates $r_k$, and sends the results back; (3) Clients refine labels hierarchically based on $r_k$ (Low noise + clean subset $\rightarrow$ keep; low noise + noisy subset $\rightarrow$ soft labels $q_{i,k}\hat{y}_i + (1-q_{i,k})y^{pse}_i$; high noise $r_k \ge \beta \rightarrow$ use pseudo-labels $y^{pse}_i$ generated by FixMatch on the global model); (4) A local EMA teacher is maintained and "revised" at the start of each round with global parameters $\mathbf{w}_{k,ema}^{t,0} = \gamma_g \mathbf{w}_{k,ema}^{t-1,m_k} + (1-\gamma_g)\mathbf{w}_g^{t-1}$, followed by standard EMA updates during local steps for distillation; (5) A global-local representation consistency regularizer $\mathcal{R}_k$ constrains the local backbone to stay near the global representation space.

### Key Designs

1.  **Federated Sieving + Label Refining (Server-side GMM Filtering + Hierarchical Refinement)**:
    -   **Function**: Centralizes the process of "identifying mislabeled samples + estimating per-client noise rates" at the server using only distribution-agnostic loss statistics.
    -   **Mechanism**: Client $k$ maintains a loss observation set $L_i^t = \{\ell_{i,p}\}_{p=1}^{T_k}$ for each sample, where $\ell_{i,T_k} = \mathcal{H}(\mathbf{p}_i^g, \hat{y}_i)$ is the cross-entropy calculated using the global model $\mathbf{w}_g^{t-1}$. The mean $\bar{\ell}_i^t = \frac{1}{T_k}\sum_p \ell_{i,p}$ is uploaded as the data proxy. The server fits a two-component GMM to all proxies from selected clients. The clean posterior $q_{i,k}$ yields the clean/noisy partition and per-client $r_k$. Refined labels $\tilde{y}_i$ are generated in three tiers: if $r_k < \beta$ and clean, original label is kept; if $r_k < \beta$ but noisy, a soft fusion $q_{i,k}\hat{y}_i + (1-q_{i,k})y^{pse}_i$ is used; if $r_k \ge \beta$, pseudo-labels $y^{pse}_i$ (FixMatch via global model) are used. Standard CE warm-up is applied for $\alpha$ rounds.
    -   **Design Motivation**: Noise distribution is high-variance at the client scale (10–50 samples cannot fit clear bimodal distributions). Aggregating proxies at the server allows robust modeling of the "noise vs. clean" peaks. Since only loss values are transmitted (independent of the joint distribution of $\mathbf{x}, \mathbf{y}$), it avoids privacy risks associated with sharing class frequencies or prototypes (e.g., FedCorr, FedNoRo).

2.  **Globally Revised EMA Distillation (Periodic Global "Washing" of Local EMA Teacher)**:
    -   **Function**: Generates logits from a noise-robust teacher for distillation to the local student while preventing local EMA corruption in high-noise clients.
    -   **Mechanism**: Each client maintains a local EMA model $\mathbf{w}_{k,ema}^{t,m_k}$. At the start of a round ($m_k=0$), it is "revised" via $\mathbf{w}_{k,ema}^{t,0} = \gamma_g \mathbf{w}_{k,ema}^{t-1,m_k} + (1-\gamma_g)\mathbf{w}_g^{t-1}$. During local steps ($m_k \ge 1$), it follows standard EMA: $\mathbf{w}_{k,ema}^{t,m_k} = \gamma_l \mathbf{w}_{k,ema}^{t,m_k-1} + (1-\gamma_l)\mathbf{w}_k^{t,m_k}$. Distillation uses the "revised" EMA's weak-augmentation logits $\mathbf{p}_i^{le,w}$ as the teacher: $\mathcal{B}_k = \mathbb{E}_{\hat{\mathcal{D}}_k}[KL(\mathbf{p}_i^{le,w}/\tau,\ \mathbf{p}_i^{l,s}/\tau)]$.
    -   **Design Motivation**: Online EMA accumulates error signals in high-noise clients. Periodically "overwriting" with global parameters combines "EMA smoothing" with "global robustness." Calculating the teacher once at $m_k=0$ reduces distillation computing cost from $O(\text{steps})$ to $O(1)$. Ablation (Table 4) shows accuracy drops from 63.64 to 51.07 (−12.6 points) in the Non-IID Sym 1.0 setting without this module.

3.  **Global Representation Regularization**:
    -   **Function**: Prevents the local backbone from drifting away from the global feature space, serving as a fail-safe when the EMA teacher itself begins to accumulate error.
    -   **Mechanism**: Constrains features of the local backbone $f(\cdot; \mathbf{w}_{k,f}^t)$ under weak augmentation to align with those of the global backbone $f(\cdot; \mathbf{w}_{g,f}^{t-1})$ (using Cosine or L2 consistency, weighted by $\lambda_{\mathcal{R}}$).
    -   **Design Motivation**: Relying solely on EMA distillation risks the teacher being misled by refined labels. Representation consistency is label-independent, constraining the model in feature space.

### Loss & Training
Total loss: $\mathcal{L}_k = \mathcal{L}_k^{SR} + \lambda_{\mathcal{B}} \mathcal{B}_k + \lambda_{\mathcal{R}} \mathcal{R}_k$. Hyperparameters: $\lambda_{\mathcal{B}}=1.0, \lambda_{\mathcal{R}}=0.1$ (CIFAR-10) or $0.2$ (others). Optimizer: SGD with constant learning rate. Local epochs: $10$ (CIFAR-10/100), $2$ (Clothing1M). Backbones: ResNet-18/34/pretrained ResNet-50. Clients: 100 for CIFAR, 500 for Clothing1M. Non-IID: Dirichlet $\alpha=0.3$. Warm-up: Random sampling without replacement for $\alpha$ rounds to ensure all clients are seen. Augmentation: RandAugment (strong) and FedCorr-style (weak). Evaluation: Mean accuracy over the final 10 rounds.

## Key Experimental Results

### Main Results
On CIFAR-10, controlling noise client ratio $\phi$ and noise rate range $\mathcal{U}(\rho_{\min}, \rho_{\max})$. Selected extreme settings "Sym 1.0/$\mathcal{U}(0.5,1.0)$" and "Mixed 1.0/$\mathcal{U}(0.2,0.4)$" are shown below:

| Method | IID Sym $\phi=1.0$ | IID Mixed $\phi=1.0$ | Non-IID Sym $\phi=1.0$ | Non-IID Mixed $\phi=1.0$ |
| :--- | :--- | :--- | :--- | :--- |
| FedAvg | 23.89 | 70.66 | 17.32 | 51.92 |
| FedProx | 23.02 | 64.44 | 16.69 | 49.77 |
| FL-Coteaching | 47.28 | 83.99 | 33.49 | 72.42 |
| FL-DivideMix | 68.47 | 85.19 | 38.35 | 68.86 |
| FedCorr (CVPR22) | 55.12 | 84.15 | 29.42 | 83.33 |
| FedNoRo (IJCAI23) | 33.98 | 71.07 | 18.60 | 57.09 |
| **FedGR (Ours)** | **83.91** | **93.13** | **63.64** | **86.50** |

Avg Accuracy on CIFAR-10: FedGR reaches 91.07, while the runner-up FL-DivideMix is at 81.11. In the most extreme Non-IID Sym $\phi=1.0$ setting, FedGR provides a +34.2 point gain over FedCorr.

### Ablation Study (CIFAR-10, Table 4)

| Configuration | IID Sym 1.0 | IID Mixed 1.0 | Non-IID Sym 1.0 | Non-IID Mixed 1.0 | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Full FedGR | 83.91 | 92.27 | 63.64 | 84.65 | Full Model |
| w/o FS | 54.59 | 91.71 | 45.48 | 84.01 | Remove Fed. Sieving: -29.3 on IID Sym |
| w/o LR | 75.23 | 90.46 | 59.48 | 83.21 | Remove Label Refining: -1 to -8 drop |
| w/o $\mathcal{R}_k$ | 81.49 | 91.84 | 58.23 | 82.70 | Remove Repr. Reg: -5.4 on Non-IID Sym |
| w/o $\mathcal{B}_k$ | 78.14 | 91.24 | 51.07 | 79.44 | Remove EMA Distill: -12.6 on Non-IID Sym |

### Key Findings
-   **Federated Sieving (FS) is the bottleneck**: Removing FS leads to a 29.3 point crash in IID Sym, far exceeding the impact of any other module. This confirms that server-side proxy aggregation is qualitatively different from independent client estimation.
-   **EMA Distillation ($\mathcal{B}_k$) is vital for dual heterogeneity**: Removing it drops accuracy by 12.6 points in Non-IID Sym 1.0, validating its design against noise pollution in class-imbalanced local training.
-   **Anomalous performance exceeding clean baseline**: In the Mixed $\phi=0.6$ setting, FedGR slightly outperforms FedAvg trained on clean data. This is attributed to the side effect of extra regularization, though the main value remains noise robustness.
-   **Sieving Accuracy**: Figure 3 shows the Pearson correlation between estimated noise $\{r_k\}$ and ground truth is $>0.9$, significantly higher than FedCorr/FedFixer.

## Highlights & Insights
-   **Systematizing global model robustness**: While previous FL works treat the global model as a "final product," this paper treats it as an "online implicit regularizer and trusted proxy." This perspective can be extended to Federated Domain Adaptation or Continual Learning.
-   **Privacy-friendly loss proxies**: Using per-sample moving average loss instead of prototypes or frequencies is a clean, reusable trick for scenarios requiring sample relative difficulty estimation without semantic leakage.
-   **EMA Revision Mechanism**: Explicitly stacking "long-term temporal smoothing" (EMA) with "population average" (global aggregation) while locking distillation to $m_k=0$ is both theoretically grounded and computationally efficient.

## Limitations & Future Work
-   Assumes loss proxies can form a bimodal distribution at the population level within $\alpha$ rounds. If the vast majority of clients have identical high noise, GMM may collapse (no fallback for 100% noise scenarios is mentioned).
-   Introduces 4 new hyperparameters ($\alpha, \beta, \gamma_g, \lambda_{\mathcal{R}}$). Sensitivity to $\gamma_g$ is high, requiring manual tuning per dataset.
-   The storage and bandwidth overhead for per-sample loss proxies on the server, though described as "moderate," remains a significant increment over vanilla FedAvg, with scalability to $>10\text{k}$ clients untested.
-   Privacy guarantees are intuitive; no formal Differential Privacy (DP) analysis is provided to verify immunity against membership inference attacks.

## Related Work & Insights
-   **vs. FedCorr (CVPR22)**: FedCorr also uses server-side correction but relies on more signals (model parameters + ratio statistics). FedGR outperforms it by +34 points on Non-IID Sym 1.0 while only transmitting loss proxies.
-   **vs. FedNoRo / FedDiv / FedFixer**: These methods perform screening or detection at the client level, failing under dual heterogeneity. FedGR's shift to the server with representation alignment proves significantly more robust.
-   **vs. DivideMix (Centralized)**: DivideMix uses dual networks for stable clean-rate estimation. FedGR adapts this GMM-based posterior logic to FL by replacing dual networks with a "Global-Local" + EMA distillation framework, fitting communication and resource constraints.

## Rating
-   Novelty: ⭐⭐⭐⭐ Combination of "global model delayed memory" and server-side GMM joint filtering is novel.
-   Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 8 baselines × multiple noise/distribution combos, though lacks extreme $100\%$ noise or $10\text{k}+$ client scenarios.
-   Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-mapped formulas and workflows.
-   Value: ⭐⭐⭐⭐ A strong, reproducible baseline for privacy-sensitive F-LNL with significant gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](../../ICLR2026/optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)

</div>

<!-- RELATED:END -->
