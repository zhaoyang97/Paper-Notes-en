---
title: >-
  [Paper Note] Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice
description: >-
  [ICLR 2026][Pretraining][Paper Note] This paper points out a fatal flaw in the practice widely relied upon by frontier teams—comparing data recipes using small proxy models with fixed hyperparameters. Dataset rankings can be flipped by minor changes in the learning rate. The authors propose training proxy models with an extremely small learning rate ($10^
tags:
  - ICLR 2026
  - Pretraining
date: 2026-05-08
content_hash: 2abe52e3ca1b618f
---
# Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=2FZC0c06jP](https://openreview.net/forum?id=2FZC0c06jP)  
**Code**: TBD  
**Area**: LLM Pre-training / Data Curation  
**Keywords**: Proxy models, data recipe ablation, learning rate, hyperparameter sensitivity, transferability

## TL;DR
This paper points out a fatal flaw in the practice widely relied upon by frontier teams—comparing data recipes using small proxy models with fixed hyperparameters. Dataset rankings can be flipped by minor changes in the learning rate. The authors propose training proxy models with an extremely small learning rate ($10^{-5}\sim10^{-6}$) as a simple patch, which improves the Spearman correlation of rankings from a proxy (GPT2-125M) to a target model (Pythia-1B) from $<0.75$ to $>0.95$ across 23 data recipes.

## Background & Motivation
**Background**: High-quality data has become the primary driver for modern LLM progress. However, there is little theory or intuition regarding "which data recipe is better," and in practice, evaluation depends on actual training. Since training a full-scale model for every candidate recipe is prohibitively expensive, the industry commonly employs "proxy models"—smaller models trained on candidate datasets to rank them by validation loss, with the winning recipe recommended to the large-scale training team. This practice supports data decisions for numerous well-known open-source datasets and models (e.g., DCLM, RefinedWeb, LLaMA).

**Limitations of Prior Work**: For "fair comparison," almost all data-centric research and benchmarks train proxy models using the **same set of fixed hyperparameters across all candidate recipes**. However, the authors discover a subtle but fatal issue with this fixed-configuration protocol: dataset ranking conclusions can **flip** with slight adjustments to training hyperparameters, particularly the learning rate (LR). The paper compares DCLM with its aggressively de-duplicated variant, DCLM-dedup-GS: at lower LRs, DCLM is superior in both validation loss and downstream benchmarks, but with a slight increase in LR, the conclusion reverses.

**Key Challenge**: The fundamental reason is that the **optimal training configuration is data-dependent**. Data distribution and training hyperparameters are strongly coupled; every data recipe naturally requires its own optimal configuration. The fixed-hyperparameter protocol not only overfits conclusions to a specific setting but, more seriously, disconnects from the actual LLM development pipeline: data teams select recipes using fixed hyperparameters, while model training teams **tune hyperparameters specifically for that dataset** once they receive it (e.g., GPT-3 determines batch size based on gradient noise scale, and LR/optimizers are adjusted per data). The evaluation criteria used in these two stages are inconsistent.

**Goal**: (1) Redefine the evaluation objective for "high-quality datasets" to align with actual development workflows; (2) Provide a directly applicable patch for existing proxy model practices without significantly increasing costs, ensuring that conclusions from small-scale experiments reliably transfer to tuned large-scale models.

**Key Insight**: The authors start from an overlooked fact: the true value of a data recipe should be the **attainable performance under its own optimal hyperparameters**, rather than performance under a preset, potentially sub-optimal hyperparameter. Following this goal, the authors use a Taylor expansion of a single-step gradient update to analyze "why LR flips rankings" and infer that "reducing LR to a sufficiently small value" can eliminate such flips.

**Core Idea**: Change the objective of data recipe ablation to "optimal under respective tuned hyperparameters" and approximate this goal using a simple patch: **training proxy models with a tiny learning rate**. A small LR ensures that the ranking is determined solely by the distributional similarity between the data and the validation set (first-order gradient alignment), thereby maintaining stability across different hyperparameters and model scales.

## Method

### Overall Architecture
The paper does not propose a new model or algorithm but rather **redefines the evaluation objective and patches existing protocols**. The logic follows a diagnostic-attribution-repair chain.

Step 1: **Redefine the objective**. The authors formalize data recipe ablation as finding the dataset among candidates $\mathcal{D}=\{D_1,\dots,D_n\}$ that minimizes validation loss under its respective optimal hyperparameters:
$$D_{i^*} := \arg\min_{i\in[n]} \min_{\lambda\in\Lambda} \ell_{\text{val}}(\theta(D_i;\lambda))$$
where $\theta(D;\lambda)$ represents a model trained on dataset $D$ with hyperparameter configuration $\lambda$, and $\Lambda$ is the feasible hyperparameter space constrained by compute budget. The key change is the inner $\min_\lambda$—each dataset is evaluated by its own optimal hyperparameters rather than sharing a fixed $\lambda_0$.

Step 2: **Diagnose vulnerability**. For a proxy model to succeed under this new objective, the optimal recipe selected at small scale must remain optimal when (i) the model is scaled to the target size and (ii) the training team tunes hyperparameters. Current fixed-hyperparameter practices fail even the first requirement—the authors experimentally demonstrate that minor LR changes flip rankings and explain why using single-step gradient analysis.

Step 3: **Provide the patch**. Based on the diagnosis, the authors propose training the proxy model with a tiny learning rate and ranking datasets by the resulting validation loss. They provide theoretical proof using random feature models and practical rules for selecting "tiny" LRs.

### Key Designs

**1. Data-Dependent Optimal Hyperparameter Objective: From "Fair Comparison" to "Respective Optimum"**

Addressing the pain point that "fixed-hyperparameter protocols are disconnected from real workflows," the authors rewrite the evaluation objective from $\min_i \ell_{\text{val}}(\theta(D_i;\lambda_0))$ (shared fixed $\lambda_0$) to $\min_i \min_\lambda \ell_{\text{val}}(\theta(D_i;\lambda))$ (individual tuning per dataset). This change corrects the evaluation paradigm of the field: the value of a data recipe should be measured by its **optimal reachable performance**. This is not pedantry—strong interaction between data and hyperparameters is well-documented, and real pipelines like GPT-3, PaLM, and Falcon tune hyperparameters per data. Only data-centric benchmarks remained stuck with fixed hyperparameters. The new objective creates a definition of "success" for small-scale experiments that is consistent with production environments.

**2. Diagnosis of Proxy Model Hyperparameter Vulnerability: The Curse of High-Order Effects**

To understand why fixed hyperparameters fail, the authors scanned multiple key hyperparameters (batch size, weight decay, token/parameter ratio) and found that dataset rankings are most sensitive to LR. The intuitive explanation comes from a Taylor expansion of the validation loss change after a single gradient update:
$$\Delta\ell_{\text{val}}(\theta) \approx -\eta\,\nabla\ell_{\text{val}}(\theta)\cdot\nabla\ell(\theta) + \frac{\eta^2}{2}\,\nabla\ell(\theta)^T H_{\ell_{\text{val}}}(\theta)\,\nabla\ell(\theta)$$
When the learning rate $\eta$ is small, the ranking is primarily determined by the **first-order gradient alignment term** $\nabla\ell_{\text{val}}\cdot\nabla\ell_i$, which measures the distributional similarity between the training and validation sets from the network's perspective. However, as $\eta$ increases to moderate levels, the **second-order curvature term** containing the Hessian $H_{\ell_{\text{val}}}$ becomes significant. Two datasets with better first-order alignment can be overtaken due to differences in second-order terms, leading to ranking flips. This is what the authors call the "curse of high-order effects"—conclusions from fixed-hyperparameter experiments overfit to that specific LR and almost inevitably become disordered when scaled to larger models or wider hyperparameter scans.

**3. Tiny Learning Rate Patch: Locking Rankings with First-Order Alignment**

To address the root cause, the repair is direct: suppress the proxy model's LR to a tiny value and rank datasets by the resulting validation loss. This is supported by two empirical findings. **Finding I** (within the same model): At a tiny LR, the loss is strongly correlated with the optimal loss after full hyperparameter tuning. Formally, for most dataset pairs, $\text{sign}(\Delta\ell_{\text{val}}(\theta_{\text{proxy}},\eta_0)) = \text{sign}(\Delta\ell_{\text{val-opt}}(\theta_{\text{proxy}}))$. The intuition is that as $\eta\to 0$, updates are dominated by the first-order alignment term, and validation loss at an infinitesimal LR characterizes the irreducible gap between training and validation distributions. **Finding II** (across scales): When both proxy and target models use tiny LRs, dataset rankings are almost perfectly preserved across model scales because tiny LRs suppress higher-order interactions that disturb comparisons, and the relative ranking of first-order alignment is stable across architectures. Combining these yields the core condition: $\text{sign}(\Delta\ell_{\text{val}}(\theta_{\text{proxy}},\eta_0)) = \text{sign}(\Delta\ell_{\text{val-opt}}(\theta_{\text{tgt}}))$—the tiny LR ranking of the proxy equals the tuned ranking of the large model.

**4. Theoretical Guarantees on Random Feature Models and "Tiny" Selection Rules**

To ensure the patch is more than an empirical observation, the authors provide functional proof on Random Feature Models (chosen because they approximate training dynamics across LRs and scales and relate closely to NTK). **Theorem 1 (Informal)**: Given candidate distributions $D_A, D_B$, if the Random Feature Model width exceeds a threshold, the relative ranking of $\ell_{\text{val}}(\theta(D_A;\eta))$ and $\ell_{\text{val}}(\theta(D_B;\eta))$ after training with a sufficiently small LR equals the ranking of their validation losses in the infinite-width limit with high probability. The proof decomposes the change in validation loss under SGD into a **deterministic drift term** capturing true data quality gaps and a **variance term** from stochastic updates. When the model is wide and the LR is small, drift dominates variance, and the sign of the observed loss difference matches the sign of the infinite-width optimal gap. This is the mathematical formalization of "small LR suppressing high-order effects." The infinite-width limit is equivalent to kernel regression, giving the optimal validation loss within a function class.

Regarding "how small is tiny," theoretically $\eta_{\text{tiny}}$ must be much smaller than $1/\lambda_{\max}$ ($\lambda_{\max}$ is the largest eigenvalue of the validation loss Hessian). Practically, the authors suggest a simple rule of thumb: use an LR 1–2 orders of magnitude smaller than standard. For LLM pre-training, this usually falls between $10^{-5}\sim10^{-6}$, a range that ensures transferability while staying safely above the lower bound for numerical precision issues.

## Key Experimental Results

### Main Results
The authors evaluate proxy-to-target ranking transferability across three model families (GPT2, Pythia, OPT, 70M–1B) and 23 data recipes. Validation sets use domain losses from The Pile; downstream tasks include five benchmarks: HellaSwag, Winogrande, OpenBookQA, ARC-Easy, and CommonsenseQA. Target models undergo dataset-specific hyperparameter tuning, with the token/parameter ratio set to 20 per Chinchilla. Over 20,000 model trainings were conducted.

| Configuration | Proxy→Target Spearman Correlation | Description |
| :--- | :--- | :--- |
| Standard LR (GPT2-Small $3\times10^{-4}$) | $<0.75$ (near 0 or negative for many domains) | Fixed hyperparameter practice; severe ranking inconsistency |
| Tiny LR ($<1\times10^{-4}$) | $>0.92$ (across all three architectures) | Massive improvement once LR drops below threshold |
| GPT2-125M → Pythia-1B ($\eta=10^{-5}$) | $>0.95$ (253 dataset pairs) | Near-perfect transfer |

The 23 recipes cover four data curation dimensions: (1) Domain composition (10 Pile mix variants); (2) Corpus comparison (C4 / DCLM-baseline / RefinedWeb); (3) Score-based filtering (RedPajama-V2 variants); (4) De-duplication (4 DCLM variants with different stringency).

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Standard LR, small k | Top-k decision regret $>0.25$ val loss degradation | Selected top-k recipes perform poorly on large models |
| Tiny LR, small k | Top-k gap approaches 0 | Reliably selects optimal datasets for large models |
| DCLM vs DCLM-dedup-GS (Low LR) | DCLM is superior | Wins in loss and downstream tasks |
| DCLM vs DCLM-dedup-GS (Higher LR) | Ranking flip | Conclusion reverses with only minor LR adjustment |

### Key Findings
- Dataset rankings are far more sensitive to **learning rate** than to batch size, weight decay, or token/parameter ratio, justifying the focus on LR.
- Standard LR proxies exhibit "catastrophic failure" on multiple Pile domains—correlation near zero or negative—indicating that proxy unreliability is systemic.
- The effective range for tiny LRs ($10^{-5}\sim10^{-6}$) is consistent across model scales and yields stable results.
- The authors specifically mention the DataDecide benchmark from AI2: because it uses fixed training configurations for all datasets, its conclusions may overfit to those settings. Consequently, the authors built their own evaluation protocol where every recipe is tuned.

## Highlights & Insights
- **Redefining "Fairness"**: The industry default is "using the same hyperparameters for all datasets is fair." The authors point out this is precisely what disconnects from the real pipeline. True fairness is "comparing each at its respective optimum"—this perspective shift is the true "Aha!" moment of the paper.
- **Diagnosis and Repair from the Same Source**: The Taylor expansion of a single-step gradient update explains both "why rankings flip" (second-order curvature) and "how to fix it" (small LR makes first-order terms dominant). Diagnosis and patch are derived from the same analysis, creating a complete logical loop.
- **Cheap and Actionable Patch**: No expensive hyperparameter scans are needed; just reducing the proxy LR by 1–2 orders of magnitude is a true drop-in patch that data teams can use immediately.
- **Clear Theoretical Anchor**: Using the infinite-width limit of Random Feature Models to anchor "tiny LR proxies" and "tuned large models" to the same ranking provides provable support for empirical findings.

## Limitations & Future Work
- **Limited to Single-Epoch Pre-training**: The work only covers single-epoch LLM pre-training without repeating samples. How to handle sub-sampling and cross-epoch repetition patterns in multi-epoch training or curriculum learning remains an open problem.
- **Patch is a Stopgap, Not the Ultimate Solution**: The root cause is the strong coupling of data and training configuration. The authors admit that in the long run, data and hyperparameters should be **jointly optimized** (e.g., using gradient-based hyperparameter optimization); tiny LRs are just a practical patch.
- **Theory Limited to Random Feature Models**: Theorems hold under the RFM/NTK framework; for real deep networks, these are "principled GUIDance" rather than strict guarantees, and Taylor expansion is only a single-step analysis.
- **Target Model Scale Capped at 1B**: Due to compute constraints (tuning each recipe required 20,000 runs), the largest target model is 1B. Whether this extrapolates to frontier scales remains to be verified.

## Related Work & Insights
- **Vs. DataDecide (Magnusson et al., 2025)**: AI2 provides pre-trained models up to 1B for efficient proxy transferability evaluation, but uses **fixed training configurations**. This paper argues that this overfits conclusions, and by building a tuned protocol, it reaches different transferability conclusions.
- **Vs. Classic Selection via Proxy (Coleman et al., 2020)**: Prior work assumed proxy rankings under fixed hyperparameters were reliable. This paper is the first to systematically reveal the vulnerability of this assumption in LLM pre-training data ablation.
- **Vs. Gradient Alignment-based Data Selection (Wang et al., 2024; Fan et al., 2024)**: These works measure data value via training-validation gradient alignment. This paper connects to them by showing that validation loss at tiny LRs is dominated by this first-order alignment term.
- **Vs. Random Feature / Neural Scaling Law Theory (Bordelon et al., 2024; Lin et al., 2024)**: By using RFMs as a proxy for deep network scale dynamics and proving that tiny LR training maintains ranking consistency with infinite-width optima, the paper combines data-centric problems with scaling law theoretical tools.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reveals systemic flaws in the "fixed hyperparameter" assumption; the perspective shift is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model families, 70M–1B, 23 recipes, four data curation dimensions, 20,000 training runs; provides solid evidence.
- Writing Quality: ⭐⭐⭐⭐ Diagnosis-attribution-repair logic is clear; theoretical and empirical evidence complement each other.
- Value: ⭐⭐⭐⭐⭐ Provides a cheap, actionable patch for frontier data teams, directly impacting the reliability of data recipe decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods](rethinking_data_curation_in_llm_training_online_reweighting_offers_better_genera.md)
- [\[ICLR 2026\] Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Fine-Tuning](task-aware_data_selection_via_proxy-label_enhanced_distribution_matching_for_llm.md)
- [\[ICLR 2026\] DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks](duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICLR 2026\] SPICE: Submodular Penalized Information–Conflict Selection for Efficient Large Language Model Training](spice_submodular_penalized_informationconflict_selection_for_efficient_large_lan.md)

</div>

<!-- RELATED:END -->
