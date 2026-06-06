---
title: >-
  [Paper Note] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training
description: >-
  [ICML 2026][Model Compression][Data mixing] To search for optimal data mixing proportions in LLM pre-training without being constrained by the prohibitive cost of proxy experiments…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Data mixing"
  - "model merging"
  - "pre-training"
  - "proxy models"
  - "proportion search"
date: 2026-05-08
content_hash: 9a41d7b46eb9d405
---

# Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training

**Conference**: ICML 2026  
**arXiv**: [2602.00747](https://arxiv.org/abs/2602.00747)  
**Code**: https://github.com/Lucius-lsr/DeMix  
**Area**: Model Compression / LLM Pre-training / Model Merging  
**Keywords**: Data mixing, model merging, pre-training, proxy models, proportion search

## TL;DR
To search for optimal data mixing proportions in LLM pre-training without being constrained by the prohibitive cost of proxy experiments, this paper proposes DeMix. It trains $N$ component models once (each corresponding to a candidate subset). Subsequently, any candidate proportion $\{\alpha_i\}$ is treated as a "training-free" proxy via weighted merging $\sum_i \alpha_i \Theta_i$. LightGBM is then used for iterative regression on the simplex to select the optimal recipe. Ultimately, DeMix achieves superior downstream scores using approximately $6\times$ less compute than RegMix/CLIMB and releases the 22T-token DeMix Corpora.

## Background & Motivation

**Background**: The "mixing ratio" of LLM pre-training data crucially affects final capabilities; the proportions of general corpus, math, and code directly determine performance on benchmarks like GSM8K, HumanEval, and HellaSwag. The mainstream approach involves medium-scale proxy experiments (e.g., training $N$ candidate ratios using 8B models and 100B tokens), which is accurate but extremely expensive.

**Limitations of Prior Work**: Automated search methods (RegMix / CLIMB / DoReMi) utilize tiny-scale proxies (small models + low budget) with hundreds of training runs followed by regression. However, the gap between tiny proxies and target scales is too large, and their predictions for complex tasks like math and code have been repeatedly shown to be unreliable. Increasing the proxy budget to improve reliability linearly escalates costs, defeating the original purpose of saving resources.

**Key Challenge**: The search space is a continuous simplex, and evaluating each candidate typically requires a full training run. Consequently, the "number of proxy models" and "fidelity of a single proxy" are tied to the same compute budget, forcing a trade-off between the two.

**Goal**: Under a fixed total budget, simultaneously achieve (i) a large number of proxy samples, (ii) sufficient fidelity for each proxy, and (iii) lower end-to-end compute costs than existing methods.

**Key Insight**: Inspired by the "additivity" observed in task arithmetic and model merging—where $\Delta(D_i\cup D_j)\approx \Delta(D_i)+\Delta(D_j)$ holds when parameter shift $\delta\ll 1$—once component models for each subset are trained, the "mixed model" corresponding to proportions $\{\alpha_i\}$ can be synthesized directly via $\sum_i \alpha_i \Theta_i$, eliminating the need for retraining.

**Core Idea**: Decouple "searching" from "training." Training occurs only once for the $N$ component models (one-time cost). In the search phase, any $\{\alpha_i\}$ is merely a weighted matrix sum followed by benchmark inference. By decoupling proxy quantity from training compute, the number of proxies can be scaled to the $10^5$ level.

## Method

### Overall Architecture
The DeMix pipeline consists of four steps: (1) Data preprocessing: deduplicating, filtering (via PPL/FastText), and splitting the raw large-scale corpus into $N$ candidate subsets; (2) Component model preparation: all $N$ components share a base model $\Theta_{\text{base}}$ pre-trained on 50B tokens of general data, followed by continued training on small datasets where "domain data + general data" are mixed at a fixed ratio $\beta=0.5$, yielding $\Theta_i = \Theta_{\text{base}} + \Delta(D_i)$; (3) Model merging proxy: any candidate mixing ratio $\{\alpha_i^j\}$ is used to synthesize a proxy model via $M_{\text{mix}}^j = \sum_{i=1}^{N}\alpha_i^j \Theta_i$, followed by direct benchmark inference without training; (4) Iterative ratio prediction: LightGBM is used in a loop of sampling, scoring, and re-sampling to concentrate the distribution toward high-scoring regions. Finally, the average of the top candidates is used as the pre-training recipe for the target 1.7B / 8B models.

### Key Designs

1.  **Model Merging as Proxy (Core Theory)**:
    - **Function**: Approximates the "model trained on a mixed dataset" as the "weighted merge of models trained on individual datasets," removing the training cost for each candidate ratio.
    - **Mechanism**: Defines a training operator $\mathcal{T}(D,\Theta_{\text{base}})$ and weight increment $\Delta(D) = \mathcal{T}(D,\Theta_{\text{base}}) - \Theta_{\text{base}}$. When $\delta = \frac{\sum|\Delta(D)|}{\sum|\mathcal{T}(D,\Theta_{\text{base}})| + \sum|\Theta_{\text{base}}|} \ll 1$ (measured at ~10%), $\Delta(D_i\cup D_j)\approx \Delta(D_i)+\Delta(D_j)$. Extending to arbitrary weights gives $\Theta_{\text{mix}}\approx \sum_i \alpha_i \Theta_i$. Proxy models are synthesized as $M_{\text{mix}}^j = \sum_i \alpha_i^j \Theta_i$ and evaluated via benchmarking. The compute for a single proxy is equivalent to the training cost of 0.01B tokens, which is $200\times$ cheaper than a 2B-token training proxy.
    - **Design Motivation**: Reduces proxy generation from $\mathcal{O}(\text{train cost})$ to $\mathcal{O}(\text{inference cost})$ within the small-$\delta$ regime, allowing "proxy quantity" and "proxy fidelity" to no longer share the same budget. This avoids both the distortion of tiny-scale proxies and the overhead of large-scale training proxies.

2.  **Shared Base + Fixed $\beta$ Mixing Component Training Protocol**:
    - **Function**: Ensures that $N$ components remain within a geometric neighborhood where parameter shifts are additive during weighted merging, while retaining sufficient general language capability to reduce merging drift.
    - **Mechanism**: All components start from the same $\Theta_{\text{base}}$ (trained on 50B general tokens). Their respective domain training data is not pure domain corpus but a mix of "domain data + general data" at $\beta=0.5$. This pulls each component back toward the shared "general language" manifold to minimize $\delta$. Training uses a batch size of 512, sequence length of 8192, and an initial lr of 3e-4 with a cosine schedule (dropping to 20%).
    - **Design Motivation**: Pure domain training would cause components to drift too far, violating the $\delta \ll 1$ assumption and causing merge distortion. Mixing general data acts as a "tether" around the shared base, keeping components close in parameter space so that weighted averaging remains a reliable approximation of mixed training.

3.  **Iterative LightGBM Regression + Simplex Resampling**:
    - **Function**: Approaches the optimal mixing ratio on the continuous simplex through a cycle of "sampling—merging—scoring—predictor training—resampling," converting proxy evaluation into a black-box regression optimization.
    - **Mechanism**: The scoring metric is the average rank (rather than absolute score) of the proxy model across general, code, and math benchmarks, ensuring robustness to scale mismatches. The process starts by uniformly sampling a large batch of $\{\alpha_i^j\}$ on the simplex, synthesizing $M_{\text{mix}}^j$, and obtaining ranks $r^j$. LightGBM is trained on (mixture, rank) pairs (lr=0.02, 300 rounds). The predictor then scores a massive set of newly sampled ratios, and the top candidates proceed to the next round of resampling. After three iterations with 64/32/16 = 112 total proxies, the top-128 candidates are averaged for the final pre-training ratio.
    - **Design Motivation**: Merged proxies are cheap enough for "batch evaluation," but the simplex dimension remains high ($N \ge 7$). Regression models extrapolate from "few ground-truth scores" to "massive unscored points." Using rank avoids regression noise from different benchmark numerical ranges. Iterative resampling concentrates the search on high-scoring neighborhoods.

### Loss & Training
DeMix itself uses no special loss functions—both components and final models are trained with standard next-token prediction loss. The only non-trivial "training strategy" is the independent pre-training of the base on 50B general tokens, controlling domain shift with $\beta=0.5$ for components, and training target 1.7B / 8B models on 50B tokens using the searched ratio.

## Key Experimental Results

### Main Results

Proxy fidelity (Spearman $\rho$ vs. 96 reference models trained on 50B tokens; total budget in B tokens; Macro Avg across categories):

| Method | Total Budget (B) | Proxies / Budget per Proxy (B) | $\rho$ Macro | Top 25% $\rho$ Macro | Capability Recovery Macro |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Trained Proxy (RegMix/CLIMB) | 224 | 112 / 2 | 0.53 | 0.20 | 0.77 |
| Trained Proxy | 1344 | 112 / 12 | 0.82 | 0.57 | 0.87 |
| **DeMix** (Ours) | 15 | 112 / 0.01 | 0.55 | 0.27 | 0.76 |
| **DeMix** | 71 | 112 / 0.01 (10×7 components) | 0.60 | 0.41 | 0.80 |
| **DeMix** | 211 | 112 / 0.01 (30×7 components) | 0.81 | 0.59 | 0.83 |
| **DeMix** | 351 | 112 / 0.01 (50×7 components) | 0.80 | 0.50 | 0.85 |

DeMix matches the correlation of 1344B training proxies ($\rho \approx 0.81$ vs $0.82$) with only ~211B budget, a $6\times$ compute advantage.

Downstream performance of final mixing ratios (macro avg rank, lower is better, relative to 96 references):

| Method | Total Budget (B) | General Avg | Code Avg | Math Avg | Macro Avg Rank ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Uniform | – | 59.01 | 18.34 | 9.62 | 36.67 |
| RegMix (448B) | 448 | 59.18 | 20.09 | 11.63 | 28.00 |
| CLIMB (448B) | 448 | 58.74 | 21.10 | 16.07 | 27.67 |
| **DeMix** (211B) | 211 | – | – | – | **Best** |

### Ablation Study

| Configuration | Observation | Interpretation |
| :--- | :--- | :--- |
| Component Count $N$: 7 → 35 | $\rho$ and downstream scores improve monotonically with $N$ | More bases cover finer domains, leading to higher merge fidelity. |
| Component Mixing Ratio $\beta$: 0 → 1 | $\beta=0$ (pure domain) leads to large $\delta$ and merge distortion; $\beta \to 1$ yields no differentiation; $\beta=0.5$ is optimal. | Validates the necessity of "domain + general" mixing for small-$\delta$ geometry. |
| Budget vs. Proxy Count | Spending budget on "more cheap proxies" outperforms "few expensive proxies." | Quantity over Quality: Sufficient proxies + regression extrapolation beats sparse precision training. |
| Top 25% Spearman | RegMix has low correlation at the head (0.20), while DeMix-211B reaches 0.59. | More accurate top-tier ranking directly determines the quality of the selected optimal ratio. |

### Key Findings
- The ceiling of DeMix is determined by the validity of the small-$\delta$ assumption rather than individual proxy accuracy. If components drift too far from the base, the additive approximation breaks down, and proxy-training correlation collapses.
- Using "ranking" instead of "absolute scores" as the regression target is robust across benchmark scales; this is the practical reason LightGBM can stably extrapolate from 112 proxies to millions of simplex points.
- DeMix shows the greatest relative advantage in Math/Code tasks, where RegMix/CLIMB tiny-scale proxies are most prone to distortion. High-fidelity proxies significantly impact the final recipe for difficult tasks.
- The 22T-token DeMix Corpora is open-sourced alongside the paper, amortizing the expensive cost of finding good ratios for the entire community.

## Highlights & Insights
- Upgrades model merging from a "deployment-time capability blending" toy into "pre-training infrastructure" for ratio selection. The elegance lies in the fact that the merged model doesn't need to be fully functional; it only needs to provide an ordered benchmark score, substantially lowering the quality requirement for merging.
- The design of using $\beta$ to control component drift can be transferred to any scenario involving "weighted merging to approximate mixed training" (multi-task fine-tuning, domain adaptation, SFT data mixing). it provides a tunable "knob" for the small-$\delta$ assumption.
- By cutting proxy evaluation cost from $\mathcal{O}(\text{train})$ to $\mathcal{O}(\text{infer})$, the impact of search space dimensionality (number of candidate subsets $N$) on the total budget changes from multiplicative to additive, encouraging finer domain partitioning.

## Limitations & Future Work
- The $\delta \ll 1$ assumption was measured at ~10% in the paper, but its validity for larger models, longer training, or more aggressive domain gaps is not fully verified. If $\delta$ grows, merging correlation will rapidly degrade.
- The training cost for component models (e.g., 30×7 ≈ 211B tokens) grows linearly with $N$, creating a significant "upfront cost" for ultra-fine-grained (hundreds of subsets) partitioning.
- Experiments focused on Qwen3-1.7B and 8B scales with final validation on 50B tokens. Whether these optimal ratios hold for 70B+ models and trillion-token training remains unanswered.
- Relying on "benchmark ranks" introduces coverage bias; if the benchmark set changes, the optimal recipe might drift, requiring a new search rather than simple data swapping.

## Related Work & Insights
- **vs. RegMix / CLIMB**: They rely on training hundreds of real proxies with small budgets. DeMix replaces training with model merging, reducing proxy generation costs to inference levels. DeMix lowers the budget by $6\times$ while achieving significantly higher top-tier correlation.
- **vs. DoReMi / Rho Loss**: These rely on evaluation loss rather than real downstream rankings, making generalization to math/code difficult. DeMix aligns directly with the final goal by using benchmark rankings.
- **vs. Task Arithmetic / TIES-Merging**: DeMix borrows the additivity assumption but relaxes the performance requirement. It only cares about "score ordering" rather than "model utility," lowering the engineering bar.
- **vs. Classical Scaling Laws**: Scaling laws provide "model size vs. data volume" trends but lack guidance on internal data proportions. DeMix uses model merging to address the data dimension choice instead of extrapolation.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] GradPower: Powering Gradients for Faster Language Model Pre-Training](gradpower_powering_gradients_for_faster_language_model_pre-training.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](../../ICLR2026/model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)

</div>

<!-- RELATED:END -->
