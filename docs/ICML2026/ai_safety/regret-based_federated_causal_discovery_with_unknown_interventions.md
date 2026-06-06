---
title: >-
  [Paper Note] Regret-Based Federated Causal Discovery with Unknown Interventions
description: >-
  [ICML 2026][AI Safety][Causal Discovery] Ours proposes I-PERI: In a federated setting where client intervention targets are completely unknown and only scalar regret values can be shared…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Causal Discovery"
  - "Federated Learning"
  - "Unknown Interventions"
  - "Φ-Markov Equivalence Classes"
  - "Regret"
  - "Differential Privacy"
date: 2026-05-08
content_hash: f6dc19c42f7526c8
---

# Regret-Based Federated Causal Discovery with Unknown Interventions

**Conference**: ICML 2026  
**arXiv**: [2512.23626](https://arxiv.org/abs/2512.23626)  
**Code**: https://github.com/CIPHOD/pyCIPHOD (Available)  
**Area**: Causal Inference / Federated Learning / Differential Privacy  
**Keywords**: Causal Discovery, Federated Learning, Unknown Interventions, Φ-Markov Equivalence Classes, Regret, Differential Privacy

## TL;DR
Ours proposes I-PERI: In a federated setting where client intervention targets are completely unknown and only scalar regret values can be shared, a two-stage process using "directed-consensus masking + undirected-consensus masking" is employed to recover a new equivalence class, Φ-MEC. This class is tighter than the observational MEC but looser than the I-MEC, providing $\epsilon$-differential privacy guarantees via Laplace noise.

## Background & Motivation

**Background**: The mainstream goal of causal discovery is to recover a CPDAG as a representative of the Markov Equivalence Class (MEC) of the underlying causal DAG. When data is naturally distributed across hospitals/institutions and cannot be centralized, Federated Causal Discovery (FCD) moves this task to a "central server + multiple clients" architecture, with PERI, FedDAG, FedCDH, and NOTEARS-ADMM being representative methods.

**Limitations of Prior Work**: Almost all FCD methods assume that **all clients share the same causal model and no interventions exist**. However, in real-world scenarios, treatment protocols, diagnostic standards, and enrollment policies at different hospitals constitute client-level **structural interventions**—they remove certain incoming edges in the causal graph, causing real structural differences between client CPDAGs. Treating this heterogeneity as noise results in regret-based methods like PERI **failing to converge to the true CPDAG**.

**Key Challenge**: (1) Existing "causal discovery with interventions" (e.g., Hauser & Bühlmann, Yang et al.) assumes **known intervention targets**, while leaking intervention targets in a federated scenario violates privacy; (2) Existing "unknown intervention + multi-environment" works (Jaber et al., Squires et al.) assume data can be centralized for direct comparison, which is unfeasible in federated settings. **What is the tightest identifiable equivalence class under the coexistence of unknown interventions, strict federated constraints, and differential privacy? This remains unanswered.**

**Goal**: (i) Formalize the identifiable equivalence class in the "client-level unknown general intervention + Federated + DP" setting; (ii) Provide an algorithm that only exchanges scalar regrets without leaking client graphs; (iii) Prove convergence and differential privacy.

**Key Insight**: The authors observe that while interventions make client graphs sparser by **removing edges**, they **generate new v-structures** when the intervention acts on a parent node of a *shielded collider*. This means local CPDAGs of clients actually expose certain edge directions that cannot be oriented using observational data alone. By treating "loss from missing edges" and "information from new orientations" separately, the algorithm can avoid being misled by interventions while utilizing the directional information they provide.

**Core Idea**: PERI's single regret is split into two stages—the first stage uses **directed-consensus masking** to penalize only edges that exist in the client but not the server to recover the common CPDAG; the second stage uses **undirected-consensus masking** to feed orientation information gained by clients due to interventions back into the server CPDAG, ultimately converging to a new equivalence class, the Φ-CPDAG.

## Method

### Overall Architecture

Consider $K$ clients, each holding a dataset $\mathbb{D}^k$ and unknown intervention targets $\Phi^k \subseteq \mathbb{V}$ (assuming at least one client is purely observational, i.e., $\exists k: \Phi^k = \emptyset$). Clients locally estimate the CPDAG of the *mutilated DAG* $\mathcal{C}(G_{\Phi^k})$ using PC or GES but **do not upload it**; they only upload the scalar regret between the server's candidate graph and the local graph.

The I-PERI pipeline:

1.  **Local CPDAG Estimation**: Each client independently runs PC/GES for $\mathcal{C}(G_{\Phi^k})$.
2.  **First Stage (Federated CPDAG)**: The server holds a candidate graph $H$ and searches via GES edge addition/deletion steps; each step calculates regret $R_k(H)$ based on "directed-consensus masking," selecting $\arg\min_H \max_k R_k(H)$ to converge to the CPDAG of the common underlying DAG $\mathcal{C}(G)$.
3.  **Second Stage (Orientation Refinement)**: On the CPDAG output from the first stage, directions are enumerated for **undirected edges**; this time, regret is calculated using "undirected-consensus masking," penalizing edges that are "unoriented by the server but oriented by the client due to intervention" $\Rightarrow$ forcing the server to follow the client’s direction to obtain the Φ-CPDAG.
4.  **Differential Privacy**: Each $R_k$ is augmented with Laplace noise (scale $\lambda = Q/\epsilon$, where $Q$ is the upper bound of regret sensitivity) before uploading to achieve $\epsilon$-DP.

Input = Local CPDAGs of clients (not leaked) + regret communication; Output = Server-side Φ-CPDAG, which orients more edges than the observational CPDAG but retains more undirected edges than the $\mathcal{I}$-CPDAG.

### Key Designs

1.  **Directed-consensus masking — First Stage**:
    *   **Function**: Defines an operation to synthesize a new graph $\mu(H, \mathcal{C}(G_{\Phi^k}))$ from the "server candidate graph $H$" and "client CPDAG $\mathcal{C}(G_{\Phi^k})$", then calculates regret on this masked graph: $R_k(H) = L(\mu(H, \mathcal{C}(G_{\Phi^k})), \mathbb{D}^k) - L(\mathcal{C}(G_{\Phi^k}), \mathbb{D}^k)$.
    *   **Mechanism**: The masking rules are threefold: (i) common edges are kept; (ii) edges absent in either side **remain absent** in the mask; (iii) if one side is directed and the other is undirected, the **directed** orientation is prioritized. **Design Motivation**: Missing edges in clients due to interventions are **not** counted toward server loss (avoiding erroneous penalties for sparsification that PERI cannot handle), while edges present in clients but missing in the server **are** penalized (ensuring convergence to a CPDAG containing all common structures).
    *   **Design Motivation**: The original PERI regret $L(H,\mathbb{D}^k) - L(\mathcal{C}(G),\mathbb{D}^k)$ assumes all clients share $\mathcal{C}(G)$; with interventions, $\mathcal{C}(G_{\Phi^k}) \ne \mathcal{C}(G)$, preventing regret from reaching zero. Masking changes the semantics of "missing edges" from "errors" to "exemptions," allowing Theorem 3.1 to provide asymptotic guarantees $\hat{G} \to \mathcal{C}(G)$ as $n^k \to \infty$.

2.  **Undirected-consensus masking and Φ-CPDAG — Second Stage**:
    *   **Function**: A second regret search is performed on the first-stage CPDAG, this time replacing $\mu$ with $\nu(H, \mathcal{C}(G_{\Phi^k}))$; the output is a Φ-CPDAG with more oriented edges than the observational CPDAG.
    *   **Mechanism**: The rules for $\nu$ differ from $\mu$ in only one aspect: if one side is directed and the other is undirected, the **undirected** orientation is prioritized ("undirected priority," the reverse of stage one). Intuition: The server treats "new v-structure orientations derived from client interventions" as authoritative, forcing its corresponding undirected edges into those directions; missing edges in clients remain exempt.
    *   **Design Motivation**: Without knowing the specific intervention targets $\Phi^k$, one can only identify the MEC using observational CPDAGs; known targets allow identification of the tighter $\mathcal{I}$-MEC. **Φ-MEC** sits between the two—it **does not require** knowledge of intervention targets but **utilizes** new v-structures induced by interventions. It is the **tightest** identifiable equivalence class under the "Federated + Unknown Intervention + DP" constraint (Theorem 3.3).

3.  **$\epsilon$-Differential Privacy Mechanism based on Regret Sensitivity**:
    *   **Function**: Provides formal $\epsilon$-differential privacy guarantees for I-PERI by adding i.i.d. Laplace noise to local regrets before uploading.
    *   **Mechanism**: Lemma 3.1 proves that when the score function $L$ is partially differentiable w.r.t. parameters $\theta$, with $\|\theta\| \le M$ and $P_k(x;\theta) \ge r$, the difference in regret induced by two datasets differing by one record is bounded by $\max_k |\hat{R}_k(G) - \hat{R}'_k(G)| \le (2M+1)\log r^2 + \mathcal{O}(\log n / n)$. By adding Laplace noise with scale $\lambda = Q / \epsilon$, where $Q$ is this bound, $\epsilon$-DP is achieved via the Laplace mechanism (Proposition 3.1).
    *   **Design Motivation**: FCD literature often neglects privacy; sharing "local graphs / model parameters" exceeds DP-acceptable limits. Sharing only scalar regrets with Laplace noise provides "information-theoretic" DP without relying on encryption, aligning with the real-world requirements of federated causal discovery.

### Loss & Training

*   The score function $L$ is BIC, satisfying "consistency and decomposability" to ensure GES-style search convergence.
*   Stage 1 optimization: $\hat{G} = \arg\min_{H \in \mathcal{C}(\mathbb{G})} \max_k R_k^{\mu}(H)$. Stage 2 restricts the search space to partially oriented graphs derived from the first-stage CPDAG, targeting $\arg\min \max_k R_k^{\nu}$.
*   Assumption 2.1: At least one client holds purely observational data ($\Phi^k = \emptyset$) to "anchor" the public DAG; this is a much weaker requirement than "knowing intervention targets."

## Key Experimental Results

### Main Results

Linear synthetic data; DAG generated by Erdős-Rényi (expected edges = $p$); client data generated by linear SEM + additive Gaussian noise $V_i = \sum_{V_j \in Pa^G_i} w_{ji} V_j + N_i$. Each client contains a **single structural intervention** (except one observational client), biased toward "inducing new v-structures on shielded colliders." Metrics: SHD (lower is better), F1 (higher is better). Average of 10 random seeds.

| Nodes $p$ | Metric | I-PERI | PERI | NOTEARS-ADMM | FedDAG | FedCDH |
|-----------|--------|--------|------|--------------|--------|--------|
| 3         | SHD    | **1.53 ± 1.16** | 3.16 | 1.64         | 3.01   | 2.27   |
| 4         | SHD    | **2.87 ± 1.88** | 4.43 | 2.99         | 3.46   | 4.83   |
| 8         | SHD    | **4.44 ± 3.04** | 8.40 | 8.44         | 6.68   | 14.86  |
| 10        | SHD    | 9.85            | 11.75| 13.70        | **9.04**| 25.97  |
| 20        | SHD    | **27.8 ± 4.79** | 30.0 | 29.45        | 30.74  | 61.74  |
| 8         | F1     | **0.74**        | 0.64 | 0.46         | 0.72   | 0.44   |

Ours achieved the best SHD in 4 out of 5 variable scales; F1 advantages are particularly significant in small graphs. Figure 7 shows I-PERI is **several orders of magnitude faster** than all baselines on a symlog time axis.

### Ablation Study

| Configuration | Key Findings |
| :--- | :--- |
| I-PERI Full Two-Stage | SHD 4.44 ($p=8$). |
| Removing Stage 2 (Modified PERI, $\mu$-mask only) | SHD 8.40. Degenerates to recovering only the observational CPDAG; intervention-derived directions are lost, doubling the error rate. |
| Client-side GES instead of PC | Identical trends; I-PERI still outperforms all baselines. Shows robustness to local discovery algorithms. |
| Heterogeneous Sample Size | I-PERI maintains stable leadership; robust under federated heterogeneity. |
| Non-linear Data | I-PERI remains effective, proving the method does not rely on linear SEM. |

### Key Findings

*   **Interventions can be "utilized" rather than "tolerated"**: Removing Stage 2 doubles SHD, indicating that the "client v-structure → server orientation" contribution is the source of the performance leap.
*   **Quality of client CPDAG is the upper bound**: The server follows the client's oriented edges; thus, errors in local graphs propogate.
*   **Extremely low computational overhead**: I-PERI is significantly faster than NOTEARS-ADMM / FedDAG because it avoids joint optimization of the full graph.
*   **DP is "free"**: Since only scalar regrets are exchanged, adding Laplace noise does not change the communication structure.

## Highlights & Insights

*   **Φ-MEC as a new insight**: Expands the identifiability hierarchy from "MEC $\subset \mathcal{I}$-MEC" to "MEC $\subset$ Φ-MEC $\subset \mathcal{I}$-MEC," characterizing the tightest achievable upper bound under "unknown intervention + privacy" constraints.
*   **Symmetry of two-stage masking**: Stage 1 uses "directed priority + missing edge exemption" to avoid errors; Stage 2 uses "undirected priority + missing edge exemption" to adopt client directions.
*   **Mechanism**: Modelling client heterogeneity as "interventions" rather than "noise" is a transferable insight for federated graph learning or RL.
*   **Theory and Privacy**: Provides Lemma 3.1 sensitivity bounds and Proposition 3.1 $\epsilon$-DP proof while correcting errors in the original PERI paper.

## Limitations & Future Work

*   **Dependence on local CPDAG accuracy**: Errors in the local graph are amplified at the server; performance under small samples or high noise is not fully explored.
*   **Constraint of Assumption 2.1**: Requires at least one purely observational client; if all clients are intervened, convergence guarantees for Φ-MEC may fail.
*   **Assumptions**: Still assumes causal sufficiency, faithfulness, and no selection bias. Extending to latent variable settings is future work.
*   **Structural Intervention Focus**: Does not provide extra Gain over PERI for purely parametric interventions (changing only conditional distributions).
*   **DP-utility Trade-off**: Lacks empirical curves showing SHD changes across different $\epsilon$ values.

## Related Work & Insights

*   **vs PERI (Mian et al., 2023)**: I-PERI generalizes to "unknown client interventions" and fixes a sensitivity proof error in the original paper, adding a new mechanism in Stage 2.
*   **vs ℐ-MEC (Hauser & Bühlmann, 2012)**: ℐ-CPDAG is tighter but requires knowing intervention targets, which violates privacy in federated settings.
*   **vs Multi-environment unknown interventions (Jaber et al., 2020)**: These assume centralized data or direct comparison of intervention regularities; I-PERI operates under strict scalar regret communication.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Φ-MEC is a highly original definition for the "unknown intervention + federated + DP" setting with solid convergence proofs.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers various variable scales, sample sizes, and non-linearities, though lacks real-world medical data evaluation.
*   Writing Quality: ⭐⭐⭐⭐ High readability with tightly integrated definitions and theorems; notation is dense.
*   Value: ⭐⭐⭐⭐ Provides a practical baseline that utilizes intervention heterogeneity while maintaining privacy guarantees for multi-center studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](../../NeurIPS2025/ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICML 2025\] Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret](../../ICML2025/ai_safety/connecting_thompson_sampling_and_ucb_towards_more_efficient_trade-offs_between_p.md)
- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](../../NeurIPS2025/ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)

</div>

<!-- RELATED:END -->
