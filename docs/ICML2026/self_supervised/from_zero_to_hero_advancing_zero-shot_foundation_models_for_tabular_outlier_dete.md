---
title: >-
  [Paper Note] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection
description: >-
  [ICML 2026][Self-Supervised Learning][Zero-shot Outlier Detection] This paper proposes OutFormer, a tabular Prior-Fitted Network (PFN) pretrained on a mixture of three synthetic priors (GMM/SCM/Copula) and stabilized via…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Zero-shot Outlier Detection"
  - "Prior-Fitted Networks"
  - "Synthetic Prior Mixture"
  - "Self-Evolving Curriculum"
  - "Multi-Armed Bandit"
date: 2026-05-08
content_hash: d222c0e661b9b09a
---

# From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection

**Conference**: ICML 2026  
**arXiv**: [2602.03018](https://arxiv.org/abs/2602.03018)  
**Code**: https://github.com/psorus/Outformer  
**Area**: Tabular Foundation Models / Outlier Detection / In-Context Learning  
**Keywords**: Zero-shot Outlier Detection, Prior-Fitted Networks, Synthetic Prior Mixture, Self-Evolving Curriculum, Multi-Armed Bandit

## TL;DR
This paper proposes OutFormer, a tabular Prior-Fitted Network (PFN) pretrained on a mixture of three synthetic priors (GMM/SCM/Copula) and stabilized via a multi-armed bandit self-evolving curriculum. It achieves zero-shot tabular outlier detection by consuming training data as in-context information and providing labels in a single forward pass, reaching SOTA rankings on ADBench and two new 1500+ dataset benchmarks with inference latency close to shallow models.

## Background & Motivation

**Background**: Tabular outlier detection (OD) has long been hindered by the "model selection + hyperparameter selection" bottleneck. In practice, labeled outliers are almost non-existent. Shallow methods (kNN/LOF/IForest) require testing numerous hyperparameters, while deep methods (DeepSVDD/ICL/DDPM) require training for every new table. The Prior-Fitted Network (PFN) route, pioneered by TabPFN, demonstrated that a Transformer pretrained on massive synthetic data can perform downstream predictions by taking the training set as in-context data in a single forward pass, completely bypassing training and tuning. This approach was first applied to OD by FoMo-0D (Shen et al. 2025).

**Limitations of Prior Work**: Although FoMo-0D ranked second on ADBench, it suffers from several issues: (1) it uses only GMM priors, failing to model non-Gaussian marginal distributions, causal structures, and long-tail dependencies in real tables; (2) attempts to use richer priors resulted in worse performance than GMM-only (Table 5 shows mixed-prior 0.898 < GMM-only 0.920 on ADBench) due to disparate gradient scales where GMM signals are suppressed; (3) ADBench contains only 57 tables, lacking statistical significance.

**Key Challenge**: To make PFNs truly "universal" for OD, one needs (a) a richer set of synthetic priors to cover real-world table generation mechanisms; however, (b) naive mixed training leads to performance degradation due to task difficulty imbalances among priors. Therefore, the conflict between "diversity" and "trainability" must be resolved; simply stacking priors is insufficient.

**Goal**: Construct a mixture of synthetic priors covering multiple inlier/outlier archetypes and design a training curriculum that does not require manual difficulty ordering. This allows the model to learn across all priors, achieving SOTA results across ADBench/OddBench/OvRBench (totaling 1500+ datasets) while maintaining inference latency at the level of shallow methods.

**Key Insight**: The authors observe that this is essentially a "non-stationary multi-task learning" problem. Different combinations of (prior, dimensionality) serve as "arms" in a multi-armed bandit (MAB), where the difficulty of each arm evolves during training. This naturally fits the MAB framework. Furthermore, by selecting a reward that distinguishes between "too hard," "too easy," and "learnable," the model can autonomously select the most valuable tasks without human-defined curricula.

**Core Idea**: A three-part solution: (1) Mixed Priors: Combining GMM (multi-modal), SCM (causal), and Copula (arbitrary marginals + dependencies) with matched outlier archetypes; (2) Self-Evolving Curriculum (SEC): Treating each (prior, dim-bin) as an MAB arm, using the variance of point-level losses within a batch as the reward to automatically avoid "all-wrong/all-right/highly uncertain" batches; (3) Inference Ensemble: Multi-pass forward passes with random sampling of dimensions and samples to bypass context length limits.

## Method

### Overall Architecture
OutFormer is a 10-layer Transformer (512 hidden, 8 heads, 45.1M params). During training, at each step, an MAB selects a task category $c=(\text{prior}, \text{dim-bin})$ based on current weights. A synthetic dataset $(\mathcal{D}_{\text{train}},\mathcal{D}_{\text{test}})$ is sampled online from the corresponding prior. The model takes $\mathcal{D}_{\text{train}}$ as context and $\mathcal{D}_{\text{test}}$ as queries, outputting outlier probabilities via cross-attention. The loss is the cross-entropy on masked labels. Point-level losses are fed back to the MAB as rewards for weight updates, and a pace scheduler backpropagates only those points below a dynamic threshold. During inference, the model is frozen; for a new real table, all training samples are used as context, and queries are processed in a single pass averaged over 50 random dimension/sample sub-ensembles.

### Key Designs

1.  **Mixed Synthetic Priors (GMM + SCM + Copula)**:
    - **Function**: Covers three major generation mechanisms—multi-modality, causal structure, and arbitrary marginals/dependencies—with corresponding outlier archetypes.
    - **Mechanism**: GMM models multi-modal inliers and generates contextual subspace outliers. SCM uses an MLP with random edges as a causal graph, sampling inliers via the structural equation $X_j = f_j(X_{Pa(X_j;G)},\epsilon_j)$, with "measurement outliers" (large variance injection) and "structural outliers" (mechanism changes). Copula uses Sklar’s theorem $F(x_1,\dots,x_d)=C(F_1(x_1),\dots,F_d(x_d))$ to decouple marginals (sampled from Gaussian/Beta/Exp/etc.) and dependencies, generating "probabilistic" and "dependence" outliers.
    - **Design Motivation**: Figure 4 (Table 5) shows that these priors are complementary; training on Copula and testing on SCM yields only 0.76 AUROC. Furthermore, traditional methods like kNN/IForest cannot reach the diagonal optimal performance on these priors, proving they are non-trivial.

2.  **Self-Evolving Curriculum (SEC) via Multi-Armed Bandit**:
    - **Function**: Automatically allocates sampling weights across $P\cdot K$ (prior, dim-bin) categories, focusing training on "learnable" arms that are neither too easy nor too difficult.
    - **Mechanism**: Each category is an MAB arm. The reward $r(c)$ is the variance of point-level centered cross-entropy losses within the batch: $r(c)=\tfrac{1}{n_c}(l_i-\text{mean}(\{l_1,\ldots,l_{n_c}\}))^2$. This variance is maximized when a batch is half "high-confidence correct" and half "high-confidence wrong," and zero when all points have 0.5 probability. A pace scheduler further filters points based on a time-varying loss threshold.
    - **Design Motivation**: Naive mixed training performed worse than GMM-only. The authors identified that large gradients from other priors overwhelmed GMM signals. SEC uses batch loss variance to de-prioritize "data uncertainty (all 0.5)" and "too hard/too easy" tasks, correctly capturing the "learnable" signal and improving performance across all datasets.

3.  **Inference-time Multi-view Ensemble**:
    - **Function**: Enables zero-shot inference for large datasets exceeding Transformer context limits ($n>5K, d>100$) and provides ensemble gains for small datasets.
    - **Mechanism**: Performs 50 forward passes per test sample using (subsample context, feature bagging) and averages the scores. Subsampling selects different inlier subsets, and feature bagging selects different dimension subsets to reduce variance.
    - **Design Motivation**: Transformers have $O(n^2)$ complexity, limiting context size. Ensembling is a classic OD performance booster, and since PFN ensembles only require forward passes without retraining, the cost is significantly lower than traditional ensembling.

### Loss & Training
The pretraining objective is to minimize the binary cross-entropy on synthetic queries: $\mathcal{L}=\mathbb{E}_{(\mathbf{x},y,\mathcal{D}_{\text{train}})\sim p(\mathcal{D})}[-\log q_\theta(y\mid \mathbf{x}, \mathcal{D}_{\text{train}})]$, which approximates the Posterior Predictive Distribution (PPD). Training involves 1500 batches, each containing 1K synthetic datasets with up to 5K context points, completed on 4x A6000 GPUs.

## Key Experimental Results

### Main Results (ADBench, 57 datasets)

| Model | Avg. Rank ↓ | ELO ↑ | Winrate ↑ | rAUC ↑ | $C_\Delta$ ↓ | vs OutFormer p-val. |
|------|-------------|-------|-----------|--------|--------------|---------------------|
| DTE-NP (Prev. SOTA) | 5.12 | 1043 | 0.61 | 0.939 | 0.39 | 0.06 |
| kNN | 5.05 | 1001 | 0.61 | 0.938 | 0.36 | 0.06 |
| LOF | 6.04 | 961 | 0.53 | 0.913 | 0.43 | 0.00 |
| IForest | 8.46 | 794 | 0.32 | 0.879 | 0.52 | 0.00 |
| DDPM | 7.12 | 943 | 0.43 | 0.904 | 0.48 | 0.00 |
| DeepSVDD | 9.81 | 788 | 0.20 | 0.796 | 0.63 | 0.00 |
| TabPFN-OD | 4.74 | 1227 | 0.65 | 0.945 | 0.34 | 0.12 |
| FoMo-0D | 6.00 | 1084 | 0.54 | 0.928 | 0.41 | 0.01 |
| **OutFormer** | **4.02** | **1235** | **0.71** | **0.956** | **0.32** | – |

On the combined benchmark of 1500+ datasets (ADBench+OddBench+OvRBench), OutFormer achieves the top rank (~5.0 Avg. Rank). AUPRC results show statistically significant superiority ($p\le 0.00$) over all baselines.

### Ablation Study (SEC × Prior Combinations, Table 5)

| Configuration | GMM only (test) | Mixed (test) | ADBench (real) |
|------|-----------------|--------------|----------------|
| GMM-only train | 0.941 | 0.935 | 0.920 |
| Mixed (w/o SEC) | 0.873 | 0.937 | 0.898 |
| Mixed (w/ SEC) | **0.930** | **0.968** | **0.926** |

Naive mixing dropped ADBench performance (0.898) and GMM test performance (0.873). Adding SEC restored GMM performance to 0.930 and boosted Mixed and ADBench results, proving SEC unlocks synergistic potential between priors.

### Key Findings
- The three priors are truly complementary (Table 4); training on one and testing on another shows drops up to ~25 AUROC points. This confirms real tables manifest diverse generation mechanisms.
- SEC rescues GMM signals: Naive mixed training sees higher GMM loss than GMM-only training because larger gradients from other priors dominate. SEC uses batch-loss-variance reward to lift the weights of "learnable" GMM tasks.
- Inference latency for OutFormer is within the same order of magnitude as shallow methods (q10–q90) and 1–2 orders of magnitude faster than deep methods like DDPM/DeepSVDD which require training.

## Highlights & Insights
- The "complementarity of three priors" is an engineering-valuable conclusion. GMM, SCM, and Copula capture the primary mechanisms of tabular data and can serve as a "standard recipe" for future tabular PFNs.
- Treating mixed-task pretraining as an MAB problem is a elegant transfer from LLM curriculum learning. Using batch loss variance as a reward distinguishes "intrinsic uncertainty" from "learnable uncertainty," preventing the curriculum from being stalled by noise.
- The "Zero-shot + Shallow-level speed" paradigm is a game-changer. OutFormer enables "plug-and-play" OD APIs, eliminating the need for per-user AutoML or manual tuning.

## Limitations & Future Work
- Context length remains a hard constraint ($n>5K, d>100$). Larger tables require ensembling, which increases overhead 50x and may lose information on highly correlated features during bagging.
- Prior design is still manual; GMM/SCM/Copula represent the authors' choice rather than a proven "optimal set." Performance might drop on distributions outside these three (e.g., time-series).
- The MAB reward robustness on extremely imbalanced batches and the sensitivity of the temperature hyperparameter require further investigation.
- Training on binary labels precludes outlier score calibration or fine-grained anomaly categorization.

## Related Work & Insights
- **vs FoMo-0D (2025)**: Ours expands priors from 1 to 3 types with 5 archetypes and uses SEC to solve the degradation caused by prior mixing, significantly increasing ELO from 1084 to 1235.
- **vs TabPFN-OD**: OutFormer outperforms an adapted version of TabPFN (using feature prediction error as scores), highlighting the necessity of task-specific OD priors and curricula.
- **vs DTE-NP (2024)**: DTE-NP is a per-dataset training SOTA. Ours reduces Avg. Rank from 5.12 to 4.02 via zero-shot forward pass.
- **vs Unsupervised model selection**: These methods learn surrogates to select models. Ours effectively internalizes "model selection" into the ICL reasoning process through scale-driven pretraining.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of mixed priors and MAB curriculum is relatively new in the PFN context.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1500+ datasets, 11 baselines, and rigorous statistical tests.
- Writing Quality: ⭐⭐⭐⭐ Clear explanations, though some critical ablations are relegated to the Appendix.
- Value: ⭐⭐⭐⭐⭐ Provides an open-source, plug-and-play OD model that solves the model selection problem for industry applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[ICCV 2025\] CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)

</div>

<!-- RELATED:END -->
