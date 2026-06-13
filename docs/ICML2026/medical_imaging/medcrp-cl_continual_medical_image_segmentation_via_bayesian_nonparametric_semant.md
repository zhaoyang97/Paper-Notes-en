---
title: >-
  [Paper Note] MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery
description: >-
  [ICML 2026][Medical Imaging][Continual Learning] The Chinese Restaurant Process (CRP) is utilized for online Bayesian nonparametric clustering of clinical text prompts to automatically discover "semantic modalities." Ind…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Continual Learning"
  - "Medical Image Segmentation"
  - "Chinese Restaurant Process"
  - "LoRA"
  - "EWC"
date: 2026-05-08
content_hash: 7434a680b4221cb9
---

# MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery

**Conference**: ICML 2026  
**arXiv**: [2605.20297](https://arxiv.org/abs/2605.20297)  
**Code**: https://github.com/zygao930/MedCRP-CL (Available)  
**Area**: Medical Image Segmentation / Continual Learning  
**Keywords**: Continual Learning, Medical Image Segmentation, Chinese Restaurant Process, LoRA, EWC

## TL;DR
The Chinese Restaurant Process (CRP) is utilized for online Bayesian nonparametric clustering of clinical text prompts to automatically discover "semantic modalities." Individual LoRA adapters are assigned to each semantic modality and combined with intra-modality EWC. Across 16 medical segmentation tasks, this approach achieves a 73.3% Dice score and a 4.1% forgetting rate, using only 1/6 of the parameters required by MoE baselines.

## Background & Motivation

**Background**: When medical image segmentation models are deployed clinically, they must continuously incorporate data from new institutions, modalities, and diseases. This is inherently a Continual Learning (CL) scenario. Existing solutions generally fall into two categories: regularization methods like EWC, which apply uniform Fisher constraints to all tasks, and expert routing methods like MoE-Adapters, which require a pre-specified number of experts (e.g., $K=16$).

**Limitations of Prior Work**: Uniform regularization on heterogeneous tasks leads to a severe "compromise"—for example, chest X-rays and colonoscopy images should not share parameters; forcing constraints between them exacerbates catastrophic forgetting. Conversely, MoE models with a preset number of experts cannot predict future task diversity and consume significant parameters (51.9M). Furthermore, medical scenarios often prohibit replaying historical patient data due to HIPAA/GDPR, making conventional replay buffers unusable.

**Key Challenge**: The dilemma between parameter sharing and parameter isolation. Coarse sharing causes interference between dissimilar tasks, while hard isolation prevents beneficial transfer between similar tasks. To solve this, one must determine: "Which tasks should share, and which should be isolated?"

**Goal**: Discover task structures online and perform structure-aware continual learning without pre-specifying the number of clusters, accessing future tasks, or storing raw patient data.

**Key Insight**: Physical imaging modalities (e.g., "Ultrasound," "X-ray") are too coarse—cardiac and breast ultrasounds share physical principles but have entirely different anatomical structures and pathological patterns. Image-level clustering is unstable in high-dimensional spaces. **Clinical text prompts** naturally encode combinations of "anatomical region + pathological context," serving as more suitable task grouping signals.

**Core Idea**: Utilize CRP for Bayesian nonparametric clustering in the CLIP prompt embedding space to automatically discover "semantic modalities" (finer than physical modalities). Assign independent LoRA adapters and intra-modality EWC to each semantic modality to achieve "strict cross-modality isolation and intra-modality transfer/sharing."

## Method

### Overall Architecture
The input is a sequence of tasks $\mathcal{T}=\{T_1,\dots,T_N\}$ arriving over time, where each task contains (image, segmentation mask, clinical text prompt). Based on a frozen CLIPSeg backbone, the following steps are executed for each new task:

1.  **Semantic Modality Discovery**: A frozen CLIP text encoder extracts the prompt embedding $e_t$. The posterior is calculated using a CRP prior and an adaptive Gaussian likelihood, with MAP used to decide whether to join an existing modality $k$ or create a new one.
2.  **Modality-Specific Training**: The LoRA pair $(A_k, B_k)$ corresponding to the modality is activated and trained using Dice + CE loss. If the modality already contains previous tasks, an intra-modality EWC regularization term $\Omega_k$ is added.
3.  **Statistics Update**: Online updates are performed for the modality center $\mu_k$, intra/inter-modality similarity distributions, and the modality's Fisher information $\bar F_k$. Current parameters are stored as the modality anchor.

### Key Designs

1.  **Bayesian Modality Assignment with CRP Prior + Adaptive Likelihood**:
    *   **Function**: Online decision on whether a new task joins an existing semantic modality or starts a new one without pre-setting $K$.
    *   **Mechanism**: The CRP prior provides $P(z_t=k)\propto n_k/(t-1+\alpha)$ and $P(z_t=\text{new})\propto \alpha/(t-1+\alpha)$. The likelihood models intra/inter-modality similarities as two Gaussians $\mathcal{N}(\mu_{\text{intra}},\sigma^2_{\text{intra}})$ and $\mathcal{N}(\mu_{\text{inter}},\sigma^2_{\text{inter}})$, updated online via Welford’s algorithm. The log-likelihood ratio is $\ell(s)=\frac{(s-\mu_{\text{inter}})^2}{2\sigma^2_{\text{inter}}}-\frac{(s-\mu_{\text{intra}})^2}{2\sigma^2_{\text{intra}}}+\log\frac{\sigma_{\text{inter}}}{\sigma_{\text{intra}}}$. If samples are insufficient (cold start), it degrades to a logit-based likelihood. Final MAP decision: $z_t=\arg\max_k \log P(z_t=k\mid z_{1:t-1},e_t)$.
    *   **Design Motivation**: CRP allows $K$ to be data-driven rather than a hyperparameter. Adaptive Gaussians transform the "similarity threshold" from a manual hyperparameter into a learnable quantity, avoiding threshold drift. The negative term ensures that tasks dissimilar to all existing modalities are forced to start a new "table."

2.  **Modality-Specific LoRA Adapters**:
    *   **Function**: Maintain independent low-rank adapters for each semantic modality to achieve absolute cross-modality isolation and intra-modality sharing.
    *   **Mechanism**: LoRA is applied to the Q/K/V/O projections of CLIPSeg. The weights for modality $k$ are $W_k=W_0+\frac{\alpha_{\text{LoRA}}}{r}B_k A_k$ with rank=8 and $\alpha_{\text{LoRA}}=16$. A new $(A_k, B_k)$ is allocated whenever CRP discovers a new modality.
    *   **Design Motivation**: Freezing the backbone eliminates the source of catastrophic forgetting. Modality-based branching naturally prevents negative transfer between distinct types like X-rays and endoscopy. Reusing LoRA for the same modality enables knowledge transfer, a key difference from "one-expert-per-task" MoE models.

3.  **Intra-Modality Elastic Weight Consolidation (Intra-Modality EWC)**:
    *   **Function**: Prevent subsequent tasks from overwriting critical parameters learned by previous tasks within the same semantic modality.
    *   **Mechanism**: After training task $t$, Fisher information $F_k^{(t)}=\mathbb{E}[\nabla_{\theta_k}\log p(y\mid x;\theta_k)^{\otimes 2}]$ is estimated and merged via exponential moving average: $\bar F_k\leftarrow \frac{n_k-1}{n_k}\bar F_k+\frac{1}{n_k}F_k^{(t)}$. Subsequent tasks incorporate $\Omega_k(\theta_k)=\sum_i \bar F_{k,i}(\theta_{k,i}-\theta_{k,i}^*)^2$, where $\Omega_k=0$ only for the first task in a modality.
    *   **Design Motivation**: Classic EWC fails on heterogeneous task sequences because cross-task Fisher information conflicts. This method restricts constraints to modalities deemed similar by CRP, avoiding such conflicts. The replay-free nature (storing only statistics) complies with medical privacy.

### Loss & Training
The training objective is $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{Dice}+\mathbf{1}_{[n_{z_t}>1]}\cdot\Omega_{z_t}(\theta_{z_t})$. Dice loss addresses medical image class imbalance; EWC is enabled only when a modality contains multiple tasks. Optimizer: AdamW, lr=$1\times 10^{-3}$, weight decay=$8\times 10^{-5}$, max 60 epochs per task with patience=8. CRP concentration $\alpha=5$, EWC coefficient $\lambda=5000$. Fisher estimated with 200 samples. Images are resized to $352\times 352$.

## Key Experimental Results

### Main Results
16 medical segmentation tasks (5 colonoscopy + 1 dermoscopy + 3 ultrasound + 7 chest X-ray) were compared with 5 CL baselines under mixed task ordering:

| Method | Dice (%) ↑ | Forgetting (%) ↓ | Params (M) | GPU (GB) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Individual (Upper Bound) | 77.9 | – | 19.8 | 12.4 | Independent model per task |
| Sequential | 48.0 ± 7.1 | 28.3 ± 7.7 | 1.2 | 5.8 | Naive Fine-tuning |
| EWC | 56.8 ± 3.7 | 11.3 ± 3.5 | 1.2 | 5.8 | Unified Regularization |
| RAPF | 58.4 ± 1.7 | 7.2 ± 2.6 | 0.9 | 5.6 | Adapter Fusion |
| CL-LoRA | 60.7 ± 2.0 | 9.7 ± 1.4 | 0.05 | 5.7 | LoRA + KD |
| MoE-Adapters | 65.3 ± 3.4 | 7.1 ± 3.2 | 51.9 | 13.3 | K=16 Experts |
| **Ours (MedCRP-CL)** | **73.3 ± 1.0** | **4.1 ± 0.8** | **8.6** | 12.4 | CRP+LoRA+EWC |

Compared to MoE-Adapters, Dice increased by 8.0%, and forgetting decreased by 3.0%, using 1/6 of the parameters. Performance on early tasks (e.g., CAMUS 82.3 vs MoE 42.4) remained stable.

### Ablation Study

Module Ablation:

| Configuration | CRP | LoRA | EWC | Dice (%) | Forgetting (%) |
| :--- | :---: | :---: | :---: | :--- | :--- |
| Full Model | ✓ | ✓ | ✓ | 73.33 | 4.09 |
| w/o EWC | ✓ | ✓ | × | 71.92 | 5.41 |
| w/o CRP | × | ✓ | ✓ | 57.59 | 15.55 |
| Single LoRA | × | ✓ | × | 46.94 | 27.34 |
| w/o LoRA | ✓ | × | ✓ | 45.39 | 0.03 |

Modality Discovery Strategy Comparison:

| Modality Assignment | K | Dice (%) | Forgetting (%) |
| :--- | :--- | :--- | :--- |
| Physical Imaging Type | 4 | 65.75 | 9.23 |
| CRP Discovery (Ours) | 5 | 73.33 | 4.09 |

### Key Findings
*   **CRP is the Foundation**: Removing CRP caused the forgetting rate to spike from 4.09% to 15.55% and Dice to drop from 73.33% to 57.59%. Removing LoRA resulted in nearly zero forgetting but only 45.39% Dice—indicating that both "routing discovery" and "parameter capacity" are essential.
*   **Semantic Modality $\neq$ Physical Modality**: CRP discovered $K=5$ instead of $K=4$. The key difference was splitting cardiac ultrasound (CAMUS) and breast ultrasound (BUSI) into different modalities. Despite high visual similarity (0.95+), their text similarity was low (~0.45), providing the correct grouping signal.
*   **Structural Robustness**: Using 10 different text encoders (BiomedCLIP, OpenCLIP, etc.) yielded the same $K=5$ and partitions at $\alpha=5$. Stability remained even with clinical noise like typos or abbreviations.
*   **Order Robustness**: Dice and forgetting rates remained stable across grouped, interleaved, mixed, and reversed task sequences.

## Highlights & Insights
*   **Task Routing via Text instead of Images**: This is a neglected direction in medical CL. Textual clinical prompts are short, discriminative, and naturally generated by doctors, avoiding high-dimensional instability and cross-center acquisition bias.
*   **Adaptive Gaussian Likelihood**: Modeling the decision of "same or new modality" as an online likelihood ratio avoids the sensitivity of distance thresholds seen in DP-Means style methods.
*   **Synergy between CRP and LoRA**: CRP provides discrete decisions for expansion, while LoRA provides low-cost parameter instantiation. This combines dynamic expansion with parameter efficiency more effectively than fixed $K$ MoE models.
*   **Replay-free Significance**: Storing only LoRA weights, Fisher EMA, and center vectors without patient data directly satisfies HIPAA/GDPR, which is crucial for real-world deployment.

## Limitations & Future Work
*   **Dependency on Contrastive Training**: The method applies mainly to contrastive-trained text encoders; it may underperform with generative-style encoders like SigLIP.
*   **Prompt Quality Dependency**: While robust to controlled noise, real-world clinical reports are highly variable. Real-world stability across long-term multi-center experiments is yet to be proven.
*   **2D Limitation**: Evaluation was conducted on 2D tasks. The stability of semantic modalities derived from slice-level prompts in 3D CT/MRI volumes requires further research.
*   **Modality Interpretability**: The semantic meaning of the $K=5$ modalities requires post-hoc t-SNE interpretation. As the number of tasks scales to hundreds, whether $K$ remains manageable is an open question. Hierarchical CRP (hCRP) could be a solution.

## Related Work & Insights
*   **vs EWC / RAPF**: Unlike unified Fisher constraints, this work applies EWC only within modalities, fundamentally avoiding conflicts between heterogeneous tasks.
*   **vs CL-LoRA**: CL-LoRA lacks task structure discovery and forces all tasks into one set of adapters, resulting in lower Dice. This work gains 12+ points via modality-level LoRA.
*   **vs MoE-Adapters**: MoE-Adapters reuse fixed experts for every task, leading to parameter explosion and an inability to adapt to new modalities. MedCRP-CL outperforms it with dynamic $K$ and 1/6 of the parameters.

## Rating
*   Novelty: ⭐⭐⭐⭐ Introducing CRP and prompt-based modality discovery to medical CL is a fresh direction.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive testing across 16 tasks, multiple orders, and various encoders.
*   Writing Quality: ⭐⭐⭐⭐ Clear logical chain from motivation to method and ablation.
*   Value: ⭐⭐⭐⭐ Replay-free, automatic $K$ discovery, and high efficiency provide direct value for clinical CL deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[ICML 2026\] Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](../../AAAI2026/medical_imaging/bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)
- [\[ICML 2026\] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)

</div>

<!-- RELATED:END -->
