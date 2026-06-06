---
title: >-
  [Paper Note] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning
description: >-
  [ICML 2026][Image Generation][Training Data Attribution] GUDA reformulates "group-wise training data attribution" (TDA) as a counterfactual question: "How much would the model's log-likelihood for a sample drop if a spec…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Training Data Attribution"
  - "Counterfactual"
  - "Group-wise Attribution"
  - "Machine Unlearning"
  - "Diffusion Models"
date: 2026-05-08
content_hash: c7c45185c9a533d5
---

# GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning

**Conference**: ICML 2026  
**arXiv**: [2601.22651](https://arxiv.org/abs/2601.22651)  
**Code**: https://github.com/sony/guda (Available)  
**Area**: Interpretability / Training Data Attribution / Diffusion Models / Machine Unlearning  
**Keywords**: Training Data Attribution, Counterfactual, Group-wise Attribution, Machine Unlearning, Diffusion Models

## TL;DR
GUDA reformulates "group-wise training data attribution" (TDA) as a counterfactual question: "How much would the model's log-likelihood for a sample drop if a specific group were absent during training?" It utilizes machine unlearning to "erase a group" from the full model to approximate the counterfactual model obtained via Leave-One-Group-Out (LOGO) retraining. Using the difference in ELBO as an attribution score, GUDA proves more accurate than CLIP similarity and instance-level gradient attribution on CIFAR-10 and Stable Diffusion artistic style attribution, while being approximately 100x faster than LOGO retraining.

## Background & Motivation

**Background**: Current training data attribution (TDA) for diffusion models mostly remains at the instance level—methods such as Influence Functions, TracIn, TRAK, D-TRAK, and DAS answer "which specific training sample contributed most to this generation." However, in real-world scenarios like copyright assessment, fair compensation for data contributors, and style auditing, group-level answers are required: which artist or which object category contributed to this generated image.

**Limitations of Prior Work**: Simply summing instance-level scores does not equate to group-level attribution for two reasons: (1) Poor scalability, as the cost of per-sample evaluation explodes with dataset size; (2) Non-linearity, where groups interact through shared representations, meaning the collective effect is not merely the sum of individual samples. Existing similarity methods (e.g., CLIP cosine distance) only measure visual similarity in embedding space and do not reflect the causal effect of "how the model would change if this group were removed during training." Furthermore, using instance-level unlearning signals as attribution (e.g., Wang et al. 2024) is shown to be nearly random at the group level.

**Key Challenge**: The gold standard for group-level attribution is LOGO retraining—training a counterfactual model $\theta^{\mathrm{logo}}_{-k}$ from scratch for each group $k$ by omitting $\mathcal{D}_k$, and comparing its explanatory power on a sample against the full model $\theta^{\mathrm{full}}$. While theoretically sound, training $N+1$ models is infeasible as the number of groups grows (e.g., LOGO took 207 hours on CIFAR-10 in this study).

**Goal**: To find a scalable solution that approximates $\theta^{\mathrm{logo}}_{-k}$ without retraining, while maintaining the clear estimation target of a "counterfactual model" and enabling verification of approximation quality.

**Key Insight**: The question "what would happen if group $k$ did not exist" is essentially a data removal query, which is the domain of machine unlearning. Thus, one can start from the full model $\theta^{\mathrm{full}}$ and perform lightweight unlearning fine-tuning on that group to obtain a counterfactual approximation $\theta^{\mathrm{ul}}_{-k}$.

**Core Idea**: A LOGOA counterfactual estimator is defined as the ELBO difference between the full model and the LOGO counterfactual model on a sample. An "unlearning operator" targeting LOGO (using ReTrack redirection for the unconditional setting and style anchor redirection for the conditional setting) is used to approximate the counterfactual model. This replaces "$N+1$ retrainings" with "one full training + one unlearning fine-tuning per group."

## Method

### Overall Architecture

The end-to-end pipeline of GUDA consists of two stages:

1.  **Counterfactual Model Pre-construction**: A shared full model $\theta^{\mathrm{full}}$ is trained, and then for each group $k=1,\ldots,N$, an unlearning operator $\mathcal{U}$ is applied starting from $\theta^{\mathrm{full}}$ to produce an approximate counterfactual model $\theta^{\mathrm{ul}}_{-k} = \mathcal{U}(\theta^{\mathrm{full}}; \mathcal{D}_k, \mathcal{D}_{-k})$. This stage is performed once and can be reused offline.
2.  **Query-time Scoring**: Given a generated sample $(x_0, c)$, the ELBO is calculated under $\theta^{\mathrm{full}}$ and $\theta^{\mathrm{ul}}_{-k}$ for each group. The difference yields the score: $\mathrm{GUDA}_k(x_0, c) = \mathrm{ELBO}(x_0|c; \theta^{\mathrm{full}}) - \mathrm{ELBO}(x_0|c; \theta^{\mathrm{ul}}_{-k})$. A positive score indicates better explanatory power by the full model, suggesting group $k$ contributed to the generation.

The choice of ELBO is a key engineering compromise: while log-likelihood can be calculated via probability-flow ODE, running $N$ counterfactual models for numerous queries is too expensive. As a lower bound, ELBO is highly correlated with log-likelihood (empirically, $\Delta \log p$ estimated via ODE on CIFAR-10 strongly correlates with $\Delta\mathrm{ELBO}$), particularly for identifying top-contributing groups.

The unified unlearning loss structure is $\mathcal{L}_{\text{unlearn}} = \mathcal{L}_{\text{forget}} + \lambda_{\text{pres}} \mathcal{L}_{\text{preserve}}$, where the preservation term $\mathcal{L}_{\text{preserve}} = \mathbb{E}_{(x,c) \sim \mathcal{D}_{-k}, t, \varepsilon}[\|\epsilon_\theta(x_t, t, c) - \epsilon_{\theta^{\mathrm{full}}}(x_t, t, c)\|_2^2]$ uses the frozen full model for score matching to prevent catastrophic forgetting. The forget term $\mathcal{L}_{\text{forget}}$ differs between the unconditional (Guda-U) and conditional (Guda-C) settings.

### Key Designs

1.  **LOGOA Counterfactual Estimator + ELBO Proxy**:
    *   **Function**: Explicitly defines "group-level attribution" as a quantifiable counterfactual scalar, decoupled from specific algorithms.
    *   **Mechanism**: For a generated sample $(x_0, c)$ and group $k$, it defines $\mathrm{LOGOA}_k(x_0, c) = \mathrm{ELBO}(x_0|c; \theta^{\mathrm{full}}) - \mathrm{ELBO}(x_0|c; \theta^{\mathrm{logo}}_{-k})$. While $\log p_\theta$ is ideal, its estimation in diffusion models requires expensive probability-flow ODEs, so it is replaced by ELBO, a tractable lower bound; a larger ELBO represents a higher (lower-bound) likelihood assigned by the model.
    *   **Design Motivation**: Clarifying the "oracle" provides a direct verification target for the quality of unlearning methods. This explains why the instance-level approach by Wang et al. (2024) failed at group attribution—their estimator did not point toward $\theta^{\mathrm{logo}}_{-k}$.

2.  **ReTrack Redirection Unlearning for Guda-U (Unconditional)**:
    *   **Function**: Constructs an unlearning loss in unconditional diffusion whose expectation is equivalent to "training only on the retain set," thereby approximating $\theta^{\mathrm{logo}}_{-k}$ rather than vaguely "erasing group $k$."
    *   **Mechanism**: After adding noise to $x_0^{(f)} \in \mathcal{D}_k$ to get $x_t$, the model's denoising target is changed from the original $\varepsilon$ to an importance-weighted target on the retain set $\bar{\varepsilon}_t(x_t) = \sum_{x_0^{(r)} \in \mathcal{D}_{-k}} w_t(x_t; x_0^{(r)}) (x_t - \sqrt{\bar{\alpha}_t} x_0^{(r)})/\sigma_t$, where $w_t \propto q_t(x_t|x_0^{(r)})$. In practice, only the $K$ nearest neighbors are used. The full ReTrack also includes a density ratio correction to make the objective equal to the retain-only training objective in expectation; the practical version omits the density ratio for efficiency.
    *   **Design Motivation**: Compared to general unlearning losses (like ESD) that only care about preventing the model from generating content from group $k$, ReTrack's goal explicitly aligns with the LOGO counterfactual. This choice is critical—experiments show GUDA+ReTrack outperforms GUDA+ESD (72.7% vs 61.9% in Top-1 accuracy).

3.  **Weighted Style Selection Anchors (WSS) for Guda-C (Conditional T2I)**:
    *   **Function**: Extends ReTrack to text-to-image models like Stable Diffusion, addressing the difficulty where prompts containing style $k$ become "out-of-support" under retain-only training.
    *   **Mechanism**: For a forget sample $(x_f, c_f) \sim \mathcal{D}_k$, an anchor condition $c_a$ is built by keeping the content description (object/scene) of $c_f$ and replacing the style description with a style $s$ sampled from retain styles $\mathcal{S}_{\text{retain}}$ weighted by CLIP similarity. The unlearning loss $\mathcal{L}_{\text{forget}}^{(C)} = \mathbb{E}[\|\epsilon_\theta(x_t, t, c_f) - \epsilon_{\theta^{\mathrm{full}}}(x_t, t, c_a)\|_2^2]$ aligns the unlearning model's prediction under the forget condition with the full model's prediction under the retain anchor.
    *   **Design Motivation**: Naively applying ReTrack fails because, in conditional settings, the forget condition itself is absent from the training support of a LOGO model. WSS redirects the forget condition into the retain prompt distribution while maintaining content consistency with the current noisy latent $x_t$, making score matching meaningful.

### Loss & Training

The total loss is $\mathcal{L}_{\text{unlearn}} = \mathcal{L}_{\text{forget}} + \lambda_{\text{pres}} \mathcal{L}_{\text{preserve}}$. $\mathcal{L}_{\text{forget}}^{(U)}$ for Guda-U uses the ReTrack importance weighting (Eq. 8), while $\mathcal{L}_{\text{forget}}^{(C)}$ for Guda-C uses WSS anchor redirection (Eq. 9). The preservation term consistently uses score-matching distillation to the retain set. On CIFAR-10, each counterfactual is run for 20 epochs (compared to 2,400 for LOGO). On UnlearnCanvas, both LOGO and unlearning are performed as fine-tuning from a shared SD 1.5 checkpoint.

## Key Experimental Results

### Main Results

CIFAR-10 (10 classes, 2,048 queries, group-wise attribution comparison):

| Method | Top-1 ↑ | NDCG@3 ↑ | Spearman ↑ | Total Time |
| :--- | :--- | :--- | :--- | :--- |
| GUDA (ReTrack) | **0.727** | **0.677** | 0.265 | 2:02 |
| GUDA w/ ESD | 0.619 | 0.634 | 0.241 | 1:33 |
| CLIPA (Similarity) | 0.662 | 0.646 | 0.246 | <1s |
| DAS (Inst.-level Grad) | 0.716 | 0.675 | **0.267** | 35:24 |
| D-TRAK | 0.609 | 0.639 | 0.258 | 30:30 |
| TRAK | 0.118 | 0.317 | 0.030 | 30:58 |
| LOGOA (oracle) | — | — | — | 207:47 |

UnlearnCanvas (Stable Diffusion 1.5, 60 styles trained, 16 evaluated, 320 queries):

| Method | Top-1 ↑ | NDCG@3 ↑ | RBO ↑ | Spearman ↑ | Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GUDA (Ours) | **0.456** | **0.734** | **0.446** | **0.239** | 8:54 |
| CLIPA | 0.338 | 0.672 | 0.393 | 0.117 | <1s |
| Wang et al. (2024, Inst.-level Unlearn) | 0.047 | 0.588 | 0.355 | 0.147 | 158:33 |
| LOGOA (oracle) | — | — | — | — | 46:08 |

### Ablation Study

| Configuration | Top-1 | Description |
| :--- | :--- | :--- |
| GUDA + ReTrack forget | 0.727 | Full method |
| GUDA + ESD forget | 0.619 | Same framework but different unlearning loss; Top-1 drops 10.8 points, proving LOGO-aligned loss is key |
| CLIPA | 0.662 | Pure similarity, no counterfactual consideration |
| Wang et al. (Inst.-level Unlearn) | 0.047 | Unlearning the generated image itself as an instance signal → nearly random group-level results |

### Key Findings

*   **Choice of unlearning operator determines attribution quality**: Within the same GUDA framework, switching from ReTrack to ESD drops Top-1 from 72.7% to 61.9%, indicating that future improvements in LOGO-aligned unlearning can directly enhance attribution precision.
*   **Semantic similarity ≠ Counterfactual influence**: While CLIPA's 66.2% Top-1 on CIFAR-10 is decent due to distinct classes, it still trails GUDA by 6.5 points. On UnlearnCanvas, the gap is wider (33.8% vs 45.6%), as similarity fails to capture the actual impact of removing a style during training.
*   **Instance-level methods do not aggregate linearly**: TRAK on CIFAR-10 yields 11.8% Top-1 (near random 10%), and Wang et al. on SD yields 4.7%. Aggregating instance graduates or loss rankings fails to capture group interactions; thus, counterfactual models must be modeled directly.
*   **Overwhelming efficiency advantage**: On CIFAR-10, GUDA is ~100x faster than LOGO oracle (2h vs 207h) because unlearning requires only 20 epochs compared to 2,400 for retraining. Querying is also ~7x faster than the gradient method DAS (1.6s vs 11.6s per image).
*   **Head metrics are more reliable than Spearman**: Wang et al. showed a higher Spearman (0.147) than CLIPA (0.117) but had random Top-1 performance. Spearman can be skewed by consistency in low-ranking groups, so Top-k / NDCG@k / MRR should be the primary metrics for group attribution.

## Highlights & Insights

*   **Bridges "Group Attribution" and "Machine Unlearning"**: The counterfactual model needed for group attribution is exactly the object constructed by unlearning algorithms. This allows progress in one field to translate directly to the other.
*   **Oracle-first design**: Unlike many works that propose an arbitrary scoring formula, GUDA defines the LOGOA oracle first and uses LOGO retraining as ground truth, reducing the problem to how closely an unlearning operator can approximate LOGO.
*   **Transferability of WSS**: The WSS anchor construction method can be migrated to any counterfactual problem where "deleting a condition also deletes its distribution." It provides a general pattern for maintaining content consistency while shifting attributes.
*   **Offline pre-computation + Online constant-time query**: This structure is ideal for "Style Attribution as a Service"—once counters are computed for each style, new generated images only require ELBO evaluation.

## Limitations & Future Work

*   **Non-overlapping group assumption**: Current theory and experiments assume $\{\mathcal{D}_k\}$ is a partition where each sample belongs to one group. Real-world styles and subjects often overlap; extending this to overlapping groups is a goal for future work.
*   **ELBO vs. log-likelihood**: The strong correlation between $\Delta\mathrm{ELBO}$ and $\Delta\log p$ is empirically verified only on CIFAR-10. Theoretically, the asymmetry of the KL gap could distort rankings of lower-tier groups.
*   **Not certified removal**: GUDA's unlearning provides a computational approximation for counterfactual modeling, not an information-theoretic or cryptographic guarantee of data removal.
*   **Large-scale evaluation constraints**: Despite training on 60 styles in UnlearnCanvas, evaluation was limited to 16, and the LOGO oracle used fine-tuning rather than full retraining. LOGO verification on the scale of LAION remains an open problem.

## Related Work & Insights

*   **vs Influence Function / TRAK / TracIn**: These focus on instance-level TDA. GUDA demonstrates that aggregating these scores fails to capture non-linear interactions within groups.
*   **vs Data Shapley / CS-Shapley**: These focus on data valuation from a game theory perspective. GUDA is more "model-centric," focusing on constructing the counterfactual model $\theta^{\mathrm{logo}}_{-k}$.
*   **vs Wang et al. (2024)**: While both use the "unlearning" label, Wang et al. unlearn the target image itself to get loss-based rankings, whereas GUDA unlearns the entire group to approximate the counterfactual model.
*   **vs ESD / ReTrack**: GUDA treats these as LOGO approximation tools and demonstrates that ReTrack is superior for attribution because it aligns with retain-only training in expectation.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ The bridge between counterfactual attribution and machine unlearning is elegant and redefines the problem space.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Solid verification with LOGO oracle on CIFAR-10 and SD 1.5; however, scale is still limited by the cost of the oracle.
*   Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear narrative flow, from defining the oracle to approximations and empirical verification.
*   Value: ⭐⭐⭐⭐⭐ A highly practical tool for AI copyright and auditing with an extensible framework for future unlearning operators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](../../NeurIPS2025/image_generation/large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)
- [\[ICML 2026\] Unlearning in Diffusion Models: A Unified Framework Based on KL Divergence and Likelihood Constraints](unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICML 2026\] Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)

</div>

<!-- RELATED:END -->
