---
title: >-
  [Paper Note] Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework
description: >-
  [ICML 2026][Multimodal VLM][Tabular Synthesis] This paper proposes the H-TDBU framework, which utilizes LLMs or manually written rules in a top-down path to generate a "logical skeleton" $\mathcal{S}$…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Tabular Synthesis"
  - "Top-Down/Bottom-Up"
  - "Rule Constraints"
  - "LLM as Rule Generator"
  - "XGBoost Conditional Generation"
date: 2026-05-08
content_hash: 8c23f0a63e6be3f0
---

# Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework

**Conference**: ICML 2026  
**arXiv**: [2605.28198](https://arxiv.org/abs/2605.28198)  
**Code**: None  
**Area**: Synthetic Tabular Data Generation / Multi-modal Financial Data / Weak Multi-modal Alignment / Controllable Generation  
**Keywords**: Tabular Synthesis, Top-Down/Bottom-Up, Rule Constraints, LLM as Rule Generator, XGBoost Conditional Generation

## TL;DR
This paper proposes the H-TDBU framework, which utilizes LLMs or manually written rules in a top-down path to generate a "logical skeleton" $\mathcal{S}$, while employing lightweight bottom-up generators such as RandomForest, XGBoost, or CTGAN to learn "statistical textures" $z$. These components are integrated via a conditional generator $G(z\in\mathcal{Z}\mid\mathcal{S})$ and iteratively refined through a TSTR + XModal feedback loop. On weak multi-modal financial benchmarks, the TSTR AUROC outperforms pure neural network baselines while maintaining cross-modal consistency.

## Background & Motivation

**Background**: Synthetic tabular data generation is a primary solution for mitigating data scarcity, privacy constraints, and multi-source heterogeneity. Currently, two paradigms coexist: pure generative models like CTGAN, TVAE, TabDDPM, and STaSy that directly learn real data distributions for sampling; and LLM-based approaches like GReaT, REaLTabFormer, and TabuLa that serialize table rows into strings for in-context generation.

**Limitations of Prior Work**: Pure generative models are prone to overfitting and mode collapse in data-scarce scenarios such as finance and healthcare, often smoothing out long-tail events like fraud detection. Conversely, LLM-based routes struggle with structural consistency and functional dependencies due to autoregressive token decoding, frequently producing "plausible but logically invalid" samples. They also show weakness in modeling numerical-categorical interactions and long-range feature dependencies. Recent research also suggests that recursive training on synthetic data leads to model collapse (Shumailov et al. 2024 Nature).

**Key Challenge**: Synthetic tabular data essentially requires two capabilities: **logical controllability** (satisfying business rules, cross-modal alignment, and rare event coverage) and **statistical authenticity** (marginal distributions, feature correlations, and indistinguishability from real data). Compressing both into a single end-to-end model often fails to optimize either: logic is prioritized via expensive LLM reasoning (e.g., Davidson et al. 2026), or statistics are prioritized at the expense of controllability.

**Goal**: (1) Decouple "logical structure" and "statistical texture" at the framework level; (2) Use LLMs only for rule generation rather than row-by-row sampling to minimize expensive API calls; (3) Demonstrate that controllability and downstream utility can be simultaneously preserved on weak multi-modal (tabular + text sentiment) financial benchmarks.

**Key Insight**: The authors observe that LLMs excel not at "generating 12,000 table rows," but at "writing a JSON of alignment rules" after reading the data schema. Thus, the LLM can serve as a *rule provider* instead of a *data provider*, delegating the heavy lifting of per-row generation to cost-effective tree-based models.

**Core Idea**: A hybrid top-down/bottom-up framework reconciles LLM/manual rules $\mathcal{S}$ with a latent space $z$ learned by lightweight generators via $G(z\mid\mathcal{S})$. A dual-metric feedback loop (TSTR + XModal) automatically detects whether to retrain the bottom-up model or adjust top-down constraints.

## Method

### Overall Architecture

H-TDBU consists of a three-stage pipeline. The **Input** is the real data $\mathcal{D}_{\text{real}}$ and a manual description or LLM prompt regarding the schema. The **Output** is $\mathcal{D}_{\text{syn}}$, a synthetic table that adheres to business rules and statistically approximates the real distribution.

Mechanism:

1.  **Top-Down Path** — Human or LLM produces a structural template $\mathcal{S}$ after reading data summaries, essentially a JSON describing "factor $f_i$ → samplable attributes." For instance, $\mathcal{S}$ might dictate that "`target=1` must be paired with positive financial text." This step provides the logical skeleton without concerning realism.
2.  **Bottom-Up Path** — An ensemble (RandomForest/XGBoost) or generative model (CTGAN/TVAE/Gaussian copula) fits a latent space $\mathcal{Z}$ from $\mathcal{D}_{\text{real}}$, capturing complex inter-column correlations. This step focuses on "looking real" without adhering to business logic.
3.  **Synthesis & Reconciliation** — The skeleton $\mathcal{S}$ and latent noise $z\in\mathcal{Z}$ are fed to the conditional generator $X_{\text{Syn}}:=G(z\in\mathcal{Z}\mid\mathcal{S})$, where every generated row is constrained by $\mathcal{S}$. Post-generation, TSTR evaluates downstream utility while XModal assesses cross-modal alignment. If XModal fails, *Retrain Model* is triggered for the bottom-up path; if rule violations are high, *Adjust Constraints* is triggered for the top-down path.

### Key Designs

1.  **LLM as rule provider rather than data provider**:
    *   **Function**: The LLM outputs a one-time structural template $\mathcal{S}$ (JSON format). Subsequent 12,000 rows of synthetic data are generated by tree-based models independently of the LLM.
    *   **Mechanism**: Data summaries of Bank Marketing target distributions and FinancialPhraseBank sentiment labels are fed into Gemini 3.1 Pro to generate a rule-provider JSON. This JSON strictly maps `target=1` to positive text. In contrast, manual rules allow `target=1` to pair with either positive or neutral text. The framework keeps the bottom-up and evaluation modules identical across rules, isolating the LLM's contribution.
    *   **Design Motivation**: To address the costs and reasoning sensitivity of fully LLM-driven solutions (like Davidson et al. 2026). Reducing the LLM's role to rule generation drops the cost from $O(N_{\text{rows}})$ to $O(1)$.

2.  **Conditional decoupling of Top-down skeleton and Bottom-up texture**:
    *   **Function**: Uses $G(z\in\mathcal{Z}\mid\mathcal{S})$ to separate "logical controllability" and "statistical authenticity" into independently optimizable modules.
    *   **Mechanism**: A bottom-up generator learns unconstrained latent representations $z$ from real data (e.g., XGBoost conditional sampling). During sampling, $\mathcal{S}$ acts as a hard/soft constraint to prune $z$, ensuring $X_{\text{Syn}}$ maintains learned correlations while obeying $\mathcal{S}$. This avoids optimization difficulties inherent in embedding rules directly into GAN/Diffusion losses.
    *   **Design Motivation**: Decoupling allows the reuse of lightweight generators without retraining for every new rule set and enables switching rules without changing the model.

3.  **TSTR + XModal dual-metric feedback loop**:
    *   **Function**: Automatically determines if synthesis failure originates from "weak statistical learning" or "incorrect logical constraints," triggering *Retrain Model* or *Adjust Constraints* respectively.
    *   **Mechanism**: A logistic regression is trained on $\mathcal{D}_{\text{syn}}$ and evaluated on the real test set to obtain TSTR metrics. Simultaneously, XModal measures the Total Variation Distance (TVD) between the real and synthetic joint distributions of "table target × text sentiment."
    *   **Design Motivation**: Traditional TSTR cannot distinguish between modeling failure and rule specification failure. Cross-modal metrics allow the authors to observe that stricter rules can improve TSTR (due to clearer decision boundaries) even if XModal varies.

### Loss & Training

The bottom-up path utilizes the native training schemes of each generator: RandomForest (30 trees, 5,000 sample limit, 12 conditioning columns), XGBoost (80 estimators, max depth 6, 8,000 sample limit, 12 conditioning columns), and CTGAN/TVAE (SDV defaults, 50 epochs). 12,000 rows are generated per configuration across three seeds (42/123/2024). The top-down path requires no training, only the generation of JSON rules.

## Key Experimental Results

### Main Results

Weak multi-modal financial benchmark (Bank Marketing table + FinancialPhraseBank sentiment), 12,000 rows generated:

| Dataset | Method | TSTR Acc ↑ | F1 ↑ | AUROC ↑ | XModal ↓ |
|--------|------|-----------|------|---------|----------|
| Manual | Independent | 0.8830 | 0.0000 | 0.4905 | 0.1094 |
| Manual | Gaussian copula | 0.8949 | 0.2034 | 0.7738 | **0.0485** |
| Manual | **RandomForest** | **0.9281** | **0.6122** | 0.9188 | 0.0555 |
| Manual | XGBoost | 0.9139 | 0.5621 | **0.9190** | 0.1127 |
| Manual | CTGAN | 0.8992 | 0.2694 | 0.8358 | 0.0646 |
| Manual | TVAE | 0.8622 | 0.5581 | 0.8827 | 0.0533 |
| Gemini | Independent | 0.8830 | 0.0000 | 0.5359 | 0.3320 |
| Gemini | Gaussian copula | 0.9423 | 0.6749 | 0.9968 | 0.1605 |
| Gemini | **RandomForest** | **0.9971** | **0.9878** | **0.9998** | 0.1437 |
| Gemini | XGBoost | 0.9746 | 0.9003 | 0.9948 | 0.3234 |
| Gemini | CTGAN | 0.9574 | 0.7863 | 0.9903 | 0.1225 |
| Gemini | TVAE | 0.9881 | 0.9476 | 0.9885 | **0.0193** |

Analysis: Switching to the stricter Gemini-generated rules improved TSTR for **all** generators (e.g., RandomForest AUROC 0.9188 → 0.9998), as the top-down path clarified the decision boundaries.

### Ablation Study

XGBoost ablation (varying training rows and conditioning columns):

| Benchmark | Optimal Config | Acc ↑ | F1 ↑ | AUROC ↑ | Note |
|-----------|----------|-------|------|---------|------|
| Manual | 12 cols | 0.9139 | 0.5621 | 0.9190 | Loose rules require more context to complete |
| Gemini | 4 cols | 0.9925 | 0.9690 | 0.9999 | Strict rules allow easy separation with few columns |

### Key Findings

*   **Stricter rules simplify generation**: The Gemini rules locked `target=1` to positive sentiment, boosting TSTR to ~0.99; top-down separability directly dictates bottom-up difficulty.
*   **Utility and fidelity are distinct**: On the manual benchmark, RandomForest achieved high AUROC but worse XModal than Gaussian copula, necessitating dual-metric evaluation.
*   **Lightweight tree-based methods are competitive**: RandomForest matched or exceeded CTGAN/TVAE while training in a fraction of the time, validating the cost-effectiveness of the rule-based approach.
*   **Conditioning column adaptive scaling**: Manual rules favored 12 columns while Gemini rules favored 4, aligning with the intuition that strict rules push information to the constraint side.

## Highlights & Insights

*   **Downgrades LLM from "per-row generator" to "one-shot rule provider"**, drastically reducing costs by several orders of magnitude compared to multi-stage reasoning solutions while retaining semantic priors.
*   **Clean controllability experimental design**: By fixing the bottom-up pipeline and only varying JSON rules, the authors quantify the causal contribution of "rule providers," a methodology applicable to other LLM-as-X workflows.
*   **Dual-metric feedback loop**: Maps "modeling failure" and "rule failure" to specific corrective actions (*Retrain* vs. *Adjust*), transforming ad-hoc debugging into a structured process.

## Limitations & Future Work

*   The feedback loop (thresholds for XModal, triggers for retraining) lacks an automated algorithmic description, acting more as an engineering blueprint than a reproducible protocol.
*   The benchmarks are relatively weak; the multi-modal experiment uses simple sentiment labels rather than complex financial reports, and common real-world complexities like time-series or multi-table relationships are not covered.
*   "Controllability" is limited to simple binary alignment; complex multi-constraint or functional dependencies (e.g., age → income ceiling → credit limit) were not tested, nor were hard constraint violation rates reported.
*   LLM rule robustness was not analyzed regarding different models, prompts, or summary granularity.
*   Direct head-to-head comparisons with reasoning-driven works like Davidson et al. (2026) are missing.

## Related Work & Insights

*   **vs. CTGAN / TVAE / TabDDPM / STaSy**: Pure generative models learn $p(\text{table})$ directly. Ours splits this into $p(\text{table}\mid\mathcal{S})$ and $p(\mathcal{S})$, delegating them to cheap models and LLMs. Ours excels in controllability but is redundant in purely exploratory scenarios.
*   **vs. GReaT / REaLTabFormer / TabuLa**: These generate text rows via LLMs. Ours only uses LLMs for meta-rules, reducing cost and enabling explicit logic, though potentially losing some of the LLM's inherent semantic depth.
*   **vs. Davidson et al. 2026 / Umesh et al. 2026 (reasoning-driven synthesis)**: These rely on multi-step reasoning for consistency. Ours amortizes this reasoning into a one-time rule creation, trading flexible dynamic adjustment for massive cost savings.
*   **vs. Shumailov et al. 2024 (Nature, model collapse)**: Ours provides a solution to the recursive training collapse by using the top-down path to force rare events into the dataset.

## Rating
*   Novelty: ⭐⭐⭐ The decoupling of LLM rule-writing and cheap data generation is insightful, though the individual components are established tools.
*   Experimental Thoroughness: ⭐⭐ Limited benchmarks and missing comparisons with reasoning-driven baselines. Reliability metrics for the feedback loop and LLM rules are absent.
*   Writing Quality: ⭐⭐⭐ Clear logic and clean visualizations, though some implementation details are abstract.
*   Value: ⭐⭐⭐⭐ Provides a pragmatic answer to the role of LLMs in tabular synthesis, offering a cost-control strategy relevant to other LLM-augmented pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data](left-right_symmetry_breaking_in_clip-style_vision-language_models_trained_on_syn.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](../../ACL2026/multimodal_vlm/slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ICML 2026\] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)

</div>

<!-- RELATED:END -->
