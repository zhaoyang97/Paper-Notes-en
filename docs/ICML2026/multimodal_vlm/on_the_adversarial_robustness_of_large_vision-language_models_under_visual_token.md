---
title: >-
  [Paper Note] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression
description: >-
  [ICML 2026][Multimodal VLM][Visual Token Compression] This paper presents the first systematic study of the adversarial robustness of Large Vision-Language Models (LVLMs) under visual token compression. It identifies an…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Token Compression"
  - "Adversarial Attack"
  - "Robustness Evaluation"
  - "LVLM"
  - "Encoder Attack"
date: 2026-05-08
content_hash: 6d2e330b8fd8d655
---

# On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression

**Conference**: ICML 2026  
**arXiv**: [2601.21531](https://arxiv.org/abs/2601.21531)  
**Code**: https://github.com/XinweiZhang1998/CAGE (Available)  
**Area**: Multimodal VLM / AI Safety / Adversarial Robustness  
**Keywords**: Visual Token Compression, Adversarial Attack, Robustness Evaluation, LVLM, Encoder Attack

## TL;DR
This paper presents the first systematic study of the adversarial robustness of Large Vision-Language Models (LVLMs) under visual token compression. It identifies an "optimization-inference space mismatch" in existing encoder attacks and proposes the CAGE attack, which utilizes Expected Feature Disturbance (EFD) and Ranking-Disturbance Alignment (RDA) to significantly reduce the robust accuracy of compressed LVLMs under unknown compression mechanisms and token budgets.

## Background & Motivation

**Background**: Mainstream LVLMs such as LLaVA-NeXT and InternVL process hundreds to thousands of visual tokens per image, leading to extremely high deployment costs. Consequently, "plug-and-play" visual token compression methods like VisionZip, VisPruner, DivPrune, FlowCut, and PruMerge have become standard for deployment. These methods use attention scores to select the Top-K most informative tokens and optionally merge secondary tokens, reducing the visual sequence length from $N=576$ to $K \ll N$ to achieve significant speedups with minimal performance degradation.

**Limitations of Prior Work**: While compressed LVLMs are increasingly deployed in safety-critical scenarios like autonomous driving and robotics, there has been almost no academic evaluation of their adversarial robustness "after compression." Existing evaluations typically follow encoder attack paradigms (e.g., VEAttack) that optimize perturbations in the full $N$-token representation space and apply them to the compressed model, which may lead to severely distorted results.

**Key Challenge**: Perturbations are optimized on "full token representations," but inference only processes "compressed representations." Two specific failure paths exist: (i) **Budget Dilution**: A significant portion of the optimization signal is assigned to tokens that are pruned and do not participate in inference. (ii) **Dependency Rupture**: Compression prunes context/background tokens, breaking the cross-token interactions that the attack relies on during global optimization. Combined, these factors lead to a significant overestimation of robust accuracy.

**Goal**: (1) Establish the "robustness of compressed LVLMs" as an independent research problem. (2) Design an attack that aligns with compression bottlenecks under grey-box conditions where the deployment budget $K_{\text{model}}$ and the specific compression mechanism $\mathcal{C}$ are unknown.

**Key Insight**: The authors observe two complementary phenomena: ① After sorting tokens by attention, the cosine shift of perturbed features decreases monotonically as $K$ increases, indicating that perturbations naturally concentrate on high-importance tokens that are likely to "survive." ② The attack is strongest when $K_{\text{attack}} = K_{\text{model}}$ (e.g., in a 16-token deployment, the robust accuracy of a full-token attack is 49.7%, which drops to 44.4% when aligned to 16 tokens).

**Core Idea**: Instead of distributing perturbations across all tokens, use a probabilistic framework to concentrate perturbation energy on tokens that "survive across multiple possible budgets," while actively pushing the attention scores of these tokens higher to ensure they are actually selected.

## Method

### Overall Architecture
CAGE maintains the grey-box premise of encoder attacks (white-box access to the visual encoder $\mathcal{E}$, black-box access to the compression module $\mathcal{C}$ and LLM $\mathcal{F}$, with $K_{\text{model}}$ unknown). A single PGD iteration in the optimization pipeline is as follows: (1) The input image $\mathbf{x}+\boldsymbol{\delta}$ passes through the encoder to obtain perturbed features $\mathbf{H}'$ and perturbed attention scores $s_i^{\mathrm{adv}}$. (2) Tokens are ranked as $r_i$ based on $s_i^{\mathrm{adv}}$, and the survival probability $\pi_i$ for each token is calculated according to a prior distribution $P(K_{\text{model}})$. (3) The cosine distance $d_i$ is weighted by $\pi_i$ to obtain the EFD loss. (4) $d_i$ and $s_i^{\mathrm{adv}}$ are converted into distributions via softmax to maximize the RDA alignment term. (5) Joint backpropagation updates $\boldsymbol{\delta}$, followed by projection onto the $\ell_\infty$ ball $\|\boldsymbol{\delta}\|_\infty \le \epsilon$. The attack target is always the encoder, independent of text prompts and without assuming knowledge of $K_{\text{model}}$ or $\mathcal{C}$.

### Key Designs

1.  **Expected Feature Disturbance (EFD) (Aligning with Unknown Token Budgets)**:
    - **Function**: Concentrates perturbation energy on tokens likely to survive across various budgets to avoid budget dilution and dependency rupture.
    - **Mechanism**: Treats the deployment budget $K_{\text{model}}$ as an unknown discrete random variable with a prior $K_{\text{model}} \sim \mathcal{U}[K_{\min}, K_{\max}]$. The survival probability of token $i$ is defined as $\pi_i = P(K_{\text{model}} > r_i)$, forming a soft mask that decays with rank (1 for high rank, gradually decreasing, 0 for low rank). Perturbation intensity is measured by cosine distance $d_i = 1 - \mathcal{S}(\mathbf{z}_i^{\mathrm{adv}}, \mathbf{z}_i^{\mathrm{cln}})$, and the loss is $\mathcal{L}_{\text{EFD}} = \sum_i \pi_i d_i / \sum_i \pi_i$.
    - **Design Motivation**: Direct weighting by attention scores $s_i$ concentrates too heavily on a few top tokens (due to sharp softmax), leaving middle-rank tokens with almost no gradient. Since middle-rank tokens have a non-trivial probability of being selected under moderate/unknown budgets, $\pi_i$ derived from cross-budget integration serves as a weights that are truly "aligned with the compression bottleneck."

2.  **Ranking-Disturbance Alignment (RDA) (Ensuring Perturbed Tokens are Selected)**:
    - **Function**: Actively pushes high-perturbation tokens into the top attention ranks to increase the probability that they pass through the compression bottleneck and affect LLM input.
    - **Mechanism**: Softmaxes $d_i$ and $s_i^{\mathrm{adv}}$ into distributions $p_i^{(d)}$ and $p_i^{(s)}$, respectively, to maximize $\mathcal{L}_{\text{RDA}} = \sum_i p_i^{(d)} \log p_i^{(s)}$ (fitting the selection distribution to the perturbation distribution). A stop-gradient is applied to $p^{(d)}$ during optimization to avoid degradation from simultaneous shifting.
    - **Design Motivation**: Theoretically, the gradient of $\mathcal{L}_{\text{EFD}}$ decomposes into $\sum_i \pi_i \nabla d_i$ (perturbing already selected tokens) and $\sum_i d_i \nabla \pi_i$ (pushing high-perturbation tokens up the rank). However, the second term has sparse and ill-conditioned gradients at switching points due to the piecewise constant nature of Top-K selection. RDA explicitly recovers this "buried" gradient path through differentiable distribution matching.

3.  **Joint Optimization and Encoder Attack Form**:
    - **Function**: Fuses both objectives under the PGD framework while remaining decoupled from specific compression mechanisms.
    - **Mechanism**: Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{EFD}} + \lambda \cdot \mathcal{L}_{\text{RDA}}$ subject to $\|\boldsymbol{\delta}\|_\infty \le \epsilon$. EFD creates the "payload" within the "survival set," while RDA handles the "delivery" of high-perturbation tokens into the survival set.
    - **Design Motivation**: Nearly all mainstream compression methods start with Top-K selection (either pure selection like DivPrune or selection followed by merging like VisionZip). Targeting this step provides a unified interface that naturally covers diverse downstream merging mechanisms. The joint terms give the attack control over both "who is selected" and "how much they are perturbed."

### Loss & Training
The attack remains an encoder attack (no text or LLM forward pass required) using PGD iterations + $\ell_\infty$ projection. The survival probability $\pi_i$ is recalculated at each step using current adversarial attention scores. A stop-gradient is applied to $p^{(d)}$, and $\lambda$ and $\epsilon$ are the primary hyperparameters. The prior distribution is chosen as a uniform distribution $[K_{\min}, K_{\max}]$, covering a reasonable range without needing to guess the exact deployment budget.

## Key Experimental Results

### Main Results
Evaluation is conducted on LLaVA against 5 representative compression methods (VisionZip / VisPruner / DivPrune / FlowCut / PruMerge) across 3 datasets (GQA / TextVQA / VQA-v2). The table below compares average robust accuracy at $K_{\text{model}}=192$ (lower is stronger).

| Dataset | Clean | Robust (Base Baseline) | Robust (CAGE) | Robust Accuracy Gain |
| :--- | :--- | :--- | :--- | :--- |
| GQA | 56.1 | 40.8 | 36.2 | ↓11.3% |
| TextVQA | 56.7 | 33.5 | 23.4 | ↓30.1% |
| VQA-v2 | 73.4 | 55.4 | 47.2 | ↓14.8% |
| Upper Bound (576 Tokens, No Compression) | 60.3 / 57.5 / 74.5 | 42.3 / 34.7 / 55.8 | 39.4 / 26.5 / 49.4 | ↓6.9 / 23.6 / 11.4% |

Observations: (1) CAGE consistently drives robust accuracy lower than the baseline across all compression methods, with the largest impact on TextVQA (an OCR-VQA task sensitive to token loss due to its reliance on fine-grained visual evidence). (2) Even in the "upper bound" setting without compression, CAGE outperforms the baseline, suggesting that the EFD/RDA mechanism of "concentration + attention boosting" is inherently a stronger encoder attack.

### Ablation Study
| Configuration | Robust Accuracy at $K_{\text{model}}=16$ (%, ↓) | Conclusion |
| :--- | :--- | :--- |
| $K_{\text{attack}}=576$ (Full token, VEAttack default) | 49.7 | Weak due to space mismatch |
| $K_{\text{attack}}=192$ | 45.3 | Strengthens as it nears budget |
| $K_{\text{attack}}=64$ | 44.7 | Further alignment |
| $K_{\text{attack}}=16$ (Exact alignment) | 44.4 | Strongest, but requires known $K_{\text{model}}$ |
| CAGE (EFD only) | Between the rows above | Concentration alone is insufficient |
| CAGE (EFD + RDA, Full) | Outperforms fixed $K_{\text{attack}}$ across budgets | More robust across budgets |

### Key Findings
-   **Compression inherently "retains" heavily perturbed tokens**: The monotonic decrease of cosine shift under attention ranking implies that perturbations naturally concentrate on Top tokens. Since compression retains these tokens, it does not provide "automatic immunity" but rather preserves the "dirtiest" evidence.
-   **Budget Alignment Effect**: Under a fixed attack budget, the strongest attack occurs when the budget aligns with $K_{\text{model}}$, explaining why traditional full-token attacks overestimate robustness.
-   **Preliminary Defense**: Potential defense strategies provide some mitigation but are insufficient to close the vulnerability, highlighting the need for safety-aware compression research.

## Highlights & Insights
-   **Centers "Compression" in LVLM Safety Evaluation**: Reframes an engineering acceleration trick as a safety attack surface. The method is elegant, requiring only a prior and a KL-style alignment term with low migration costs.
-   **Probabilistic Handling of Unknown Budgets**: Uses a uniform prior to marginalize out $K_{\text{model}}$, allowing attacks without prior knowledge of deployment configurations. This is highly practical for grey-box environments and transferable to any "invisible inference path with Top-K selection" (e.g., MoE routing or sparse attention).
-   **Gradient Analysis Driven RDA**: Identifies that the "distribution boosting term" fails due to discrete Top-K selection and recovers it via differentiable KL matching. This methodology can be applied to many modules containing Top-K operations.

## Limitations & Future Work
-   Evaluations are limited to LLaVA and 5 plug-and-play methods, excluding compression introduced during training (e.g., Token Merging), cross-modal compression, or multi-image agent settings which are critical for deployment.
-   The attack remains an encoder-based attack and has not been compared against end-to-end LLM backpropagation attacks; being prompt-agnostic also means it cannot utilize task-specific clues.
-   Defensive research is thin. Future work should explore "compression-aware" defenses like randomized $K$, attention reshuffling, or compression-robust training.
-   The uniform prior is an engineering simplification. If deployment budgets are skewed, EFD weights may be sub-optimal; online prior adjustment based on success feedback could be considered.

## Related Work & Insights
-   **vs. VEAttack (Mei et al., 2026)**: VEAttack maximizes cosine shift on full tokens, which this paper shows systematically underestimtes attack strength in compressed LVLMs. CAGE strengthens this significantly via $\pi_i$ weighting and RDA alignment.
-   **vs. Cui et al., 2024 / Wang et al., 2024c, etc.**: While using similar grey-box encoder premises, this work explicitly models "inference-time compression bottlenecks" into the optimization objective.
-   **vs. Compression Methods (VisionZip, etc.)**: These focus on "performance-efficiency" trade-offs. This paper treats them as attack targets, revealing that more aggressive compression leads to more severe robustness evaluation distortion.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ First work to systematically reveal the mismatch between token compression and robustness evaluation.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 methods × 3 datasets × multiple budgets with stable conclusions, though limited to one LVLM backbone.
-   Writing Quality: ⭐⭐⭐⭐ Clear narrative with two key insights and gradient derivations for RDA; high information density in figures.
-   Value: ⭐⭐⭐⭐⭐ Explicitly concludes that compressed LVLM deployment requires safety re-evaluation, impacting industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[ICML 2026\] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](../../CVPR2026/multimodal_vlm/agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](../../CVPR2026/multimodal_vlm/agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICCV 2025\] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](../../ICCV2025/multimodal_vlm/dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)

</div>

<!-- RELATED:END -->
