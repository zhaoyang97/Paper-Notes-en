---
title: >-
  [Paper Note] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models
description: >-
  [ICML 2026][LLM Safety][CLIP] A-TPT utilizes Gradient Attention Rollout reinforced against adversarial perturbations to extract "semantic anchors" from the CLIP vision encoder. This attention map guides spatially non-uni…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "CLIP"
  - "Test-time prompt tuning"
  - "Adversarial robustness"
  - "Attention rollout"
  - "Fine-grained classification"
date: 2026-05-08
content_hash: f58c08de596ebc42
---

# Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.19956](https://arxiv.org/abs/2605.19956)  
**Code**: https://github.com/SEU-VIPGroup/A-TPT (Yes)  
**Area**: Multimodal VLM / Adversarial Robustness / Test-Time Adaptation  
**Keywords**: CLIP, Test-time prompt tuning, Adversarial robustness, Attention rollout, Fine-grained classification

## TL;DR
A-TPT utilizes Gradient Attention Rollout reinforced against adversarial perturbations to extract "semantic anchors" from the CLIP vision encoder. This attention map guides spatially non-uniform multi-view augmentation and weighted ensemble based on anisotropic Total Variation for prompt tuning. It simultaneously improves adversarial and clean accuracy across 9 datasets in fine-grained scenarios.

## Background & Motivation
**Background**: VLMs like CLIP exhibit strong performance on zero-shot downstream tasks, but adversarial perturbations (FGSM/PGD, Co-Attack, etc.) cause inference quality to collapse. Defense strategies follow two main tracks: training-time adaptation (VPT, FAP, SLADE, etc.), which is effective but requires expensive labeled adversarial data, and test-time adaptation (TPT, C-TPT, DiffTPT, MTA, AOM, TAPT, R-TPT), which is more efficient but primarily designed for natural distribution shifts, offering limited robustness against "feature space distortion" caused by real adversarial attacks.

**Limitations of Prior Work**: Current adversarial test-time methods (MTA/AOM/TAPT/R-TPT) are mostly based on multi-view augmentation combined with entropy/alignment objectives. These augmentations often involve random region-editing, which tends to discard discriminative regions (e.g., bird heads, car logos, wing shapes) in fine-grained classification, further losing fragile category-distinguishing signals.

**Key Challenge**: To achieve stability, discriminative semantic parts must be preserved. However, preserving these parts requires reliable semantic recognition signals. Existing semantics-preserving augmentations (FN-NET, NAS, Pu et al.) either learn in feature space or use logits as self-supervised labels. Under adversarial perturbations: (a) feature vectors are pushed across decision boundaries (illustrated via cosine similarity in Figure 1a of the paper); (b) true labels are often excluded from Top-K predictions (Figure 1b). Both paths fail in adversarial scenarios, and such "semantic recognition" is typically coupled with the training phase, making it unsuitable for test-time deployment.

**Goal**: To construct a test-time method—without introducing additional training or relying on external models—that can identify intact semantic parts under adversarial perturbations and use them as anchors to guide augmentation and ensemble.

**Key Insight**: The authors observe that attention maps reside in the "image space," making them harder to flip entirely via pixel-level $\ell_\infty$ perturbations compared to feature vectors. By making the gradient signal of GAR (Gradient Attention Rollout) itself less sensitive to perturbations, a relatively robust "annotation" of key parts can be obtained.

**Core Idea**: Use "adversarially reinforced attention" as a semantic anchor across three stages: guiding the spatial distribution of augmentation intensity, determining reliability weights for multi-view ensemble, and performing prompt tuning only on credible views.

## Method

### Overall Architecture
A-TPT is built entirely on frozen CLIP (ViT-B/16, ViT-B/32, RN50). For a single test sample $x_0$, the pipeline consists of: (1) **Attention Refinement**: Extracting a CLS-to-patch attention map $\mathbf{A}(x)\in\mathbb{R}^{H\times W}$ using an improved GAR that is stable under PGD perturbations; (2) **Attention-Guided Multi-View Augmentation**: Generating basic views $\{b_i\}$ via Random-Flip + Center-Crop and aggressive views $\{\tilde{x}_i\}$ via AugMix, then performing spatially non-uniform mixing where high-attention regions protect the original image and low-attention regions allow for diverse perturbations; (3) **TV-Based Ensemble**: Filtering views by entropy and calculating reliability weights $w_i$ based on the anisotropic Total Variation (TV) of the attention maps. These weights are used for the prompt tuning loss and final logit aggregation. Only a learnable prompt $P$ is updated for 1 step using Adam (lr=0.005), while encoders remain frozen.

### Key Designs

1.  **Token-gradient reinforced attention rollout (Attention Refinement)**:
    -   **Function**: Repairs the adversarial vulnerability of GAR to output CLS-to-patch attention maps that are stable under PGD, serving as semantic anchors.
    -   **Mechanism**: Original GAR uses $\hat{\mathbf{A}}^{(b)}=\mathbf{I}+E_h(\nabla_{\mathbf{A}^{(b)}}S\odot\mathbf{A}^{(b)})^+$ at layer $b$, where $\nabla_{\mathbf{A}^{(b)}}S$ is a per-attention-edge second-order sensitivity that becomes scattered under perturbation. The authors replace this with token-dimensional inner product weights $\mathbf{W}^{(b)}(x)=\mathcal{N}([\langle\mathbf{T}^{(b)}(x),\nabla_{\mathbf{T}^{(b)}(x)}S(x)\rangle_d]_+)$ (inner product along the embedding dimension, ReLU, and $\ell_1$ normalization), followed by column scaling: $\hat{\mathbf{A}}^{(b)}=\mathbf{I}+E_h(\mathbf{A}^{(b)}\,\mathrm{diag}(\mathbf{W}^{(b)}))^+$. A stability trick is added by averaging and stabilizing only the last two layers: $\hat{\mathbf{A}}_\text{avg}=(\hat{\mathbf{A}}^{(B-1)}+\hat{\mathbf{A}}^{(B)})/2$, $\hat{\mathbf{A}}=\hat{\mathbf{A}}^{(B)}\hat{\mathbf{A}}_\text{avg}$, removing shallow-layer noise.
    -   **Design Motivation**: Token-level gradients are first-order quantities aggregated across the embedding dimension; perturbations injected into tokens are largely averaged out during aggregation. Conversely, original attention-level gradients are second-order quantities multiplied by the attention itself, making them exponentially sensitive to noise. This step is the prerequisite for the entire method.

2.  **Spatially non-uniform attention-guided multi-view augmentation (Attention-Guided Multi-View Augmentation)**:
    -   **Function**: Maintains discriminative fine-grained parts while creating sufficient diversity to provide "clean yet diverse" inputs for prompt tuning.
    -   **Mechanism**: A ratio $r$ is used to mask the top $\lceil rHW\rceil$ attention positions as $M_\text{high}$, with the remainder as $M_\text{low}=1-M_\text{high}$. Mixing intensity is defined as $\lambda(r)=M_\text{high}\,m_\text{high}+M_\text{low}\,m_\text{low}$ (where $m_\text{high}<m_\text{low}$). Spatial pixel-wise mixing is performed: $x_i=(1-\lambda)\odot b_i+\lambda\odot\tilde{x}_i$. Discriminative areas largely preserve $b_i$, while background areas are heavily augmented.
    -   **Design Motivation**: Previous methods like R-TPT/TAPT apply AugMix full-frame, treating all pixels equally. This destroys discriminative parts in fine-grained tasks. Since informative signals under attack come from discriminative regions, they must be prioritized.

3.  **Anisotropic Total Variation based reliability ensemble (TV-Based Ensemble)**:
    -   **Function**: Assigns a scalar weight $w_i$ to each low-entropy candidate view to filter out "pseudo-good" views dominated by adversarial noise or background.
    -   **Mechanism**: Anisotropic TV is calculated for each view's attention $\mathbf{A}(x_i)$: $\mathrm{TV}(\mathbf{A}(x_i))=\sum_{u,v}|A_{u+1,v}-A_{u,v}|+\sum_{u,v}|A_{u,v+1}-A_{u,v}|$. Reliability weights are computed via softmax of the negative exponential: $w_i=\exp(-\mathrm{TV}(\mathbf{A}(x_i)))/\sum_{j\in\mathcal{B}}\exp(-\mathrm{TV}(\mathbf{A}(x_j)))$. Final prediction: $\hat{c}=\arg\max_c\sum_{i\in\mathcal{B}}w_i p_c(x_i)$.
    -   **Design Motivation**: Empirical observation suggests that a "good" view shows high, contiguous attention responses in discriminative regions (low TV), whereas adversarial high-frequency artifacts or background-dominated views result in fragmented attention or isolated peaks (high TV). TV characterizes "attention spatial consistency" better than entropy alone.

### Loss & Training
Prompt tuning follows the TPT entropy minimization framework: $\mathcal{L}_H(P)=-\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\sum_c p_c(x_i)\log p_c(x_i)$, where $\mathcal{B}$ consists of views selected after augmentation and filtering. Adam optimizer is used with weight decay for $T=1$ step and lr $=0.005$. Adversarial samples are generated via PGD: $\varepsilon=4/255$, 100 steps for ViT; $\varepsilon=1/255$, 1 step for ResNet50. Training is executed on 8 RTX-4090 GPUs. CLIP backbones and augmentation networks are not trained.

## Key Experimental Results

### Main Results
Evaluation on 8 fine-grained/general datasets + ImageNet-OOD, compared against TPT-Ensemble, MTA, R-TPT, and TTC.

| Dataset (Adv. acc., ViT-B/16) | CLIP | TPT-Ens | MTA | R-TPT | **A-TPT** | Gain (vs R-TPT) |
|---|---|---|---|---|---|---|
| OxfordPets | 0.0 | 51.2 | 51.8 | 60.2 | **70.5** | +10.3 |
| Caltech101 | 0.0 | 74.7 | 72.1 | 82.0 | **85.6** | +3.6 |
| StanfordCars | 0.0 | 26.0 | 18.5 | 34.7 | **39.2** | +4.5 |
| DTD | 0.0 | 25.1 | 16.2 | 32.8 | **37.8** | +5.0 |
| UCF101 | 0.0 | 30.6 | 27.5 | 43.2 | **51.7** | +8.5 |
| EuroSAT | 0.0 | 2.2 | 1.2 | 8.5 | **13.1** | +4.6 |
| Flower102 | 0.0 | 36.3 | 27.9 | 44.6 | **52.6** | +8.0 |
| FGVC-Aircraft | 0.0 | 8.7 | 4.3 | 13.2 | **15.1** | +1.9 |
| **Average** | **0.0** | 31.9 | 27.4 | 39.9 | **45.7** | **+5.8** |

A-TPT also achieves the best average clean accuracy (63.0 for ViT-B/16 vs. 61.1 for R-TPT), indicating that reinforced attention does not compromise clean performance. On ResNet50 (ImageNet-OOD), A-TPT remains superior (clean 48.0, adv 35.8 vs. R-TPT 47.1/35.4).

### Ablation Study
Ablation of modules in Sec 4.4:

| Configuration | Avg Adv. acc. (ViT-B/16, 8 datasets) | Description |
|------|----|------|
| Full A-TPT | 45.7 | All three modules included |
| w/o Token-grad refinement (using original GAR) | Significant Drop | Attention scattered by PGD, mask becomes unstable. |
| w/o Attention-guided augmentation (full AugMix) | Notable Drop | Significant drop in fine-grained tasks (Pets/Flowers). |
| w/o TV-based ensemble (uniform averaging) | Slight Drop | TV mainly filters "low entropy but misaligned" views. |
| w/o GAR last two layers averaging | Marginal Drop | Shallow noise leaks into rollout without stabilization. |

### Key Findings
-   **Superiority over R-TPT**: Gains of 5–10% on tasks with highly localized discriminative regions (Pets, UCF101, Flowers) validate that "discriminative region protection" is a critical gap in current full-frame augmentation methods.
-   **CLIP Zero-Shot Baseline**: CLIP zero-shot accuracy is 0% under PGD $\varepsilon=4/255$. TTA recovers this to 30-40%, and A-TPT pushes it further to 45.7%, approaching training-time method performance.
-   **Clean + Adversarial Win-Win**: Unlike MTA, which sacrifices adversarial robustness for clean accuracy in some cases, A-TPT maintains both because its anchors reside in the image space, avoiding feature space distortion.

## Highlights & Insights
-   **Reframing Robustness**: Resolving "test-time robustness" by improving "attention robustness" is a clean perspective shift. Instead of fighting in a contaminated feature space, A-TPT anchors the discriminative parts in the image space.
-   **Token-level vs. Attention-level Gradients**: Using first-order token-level gradients to replace second-order attention gradients is a reusable trick for robust explainability in ViTs.
-   **TV as Reliability Metric**: TV provides a zero-cost dimension to measure "attention spatial consistency," which is more nuanced than simple logit entropy.

## Limitations & Future Work
-   **Dependency**: The method relies on a "good enough" initial attention map. Performance gains are smaller on texture-dominant tasks like EuroSAT (+4.6) where discriminative regions are dispersed.
-   **Observations**: (1) Main experiments focus on PGD; cross-modal attacks like VLATTACK are not detailed in the main table. (2) Token-gradient refinement requires an extra backward pass per test image, increasing latency compared to R-TPT. (3) Sensitivity of hyperparameters $r$, $m_\text{high}$, and $m_\text{low}$ was not extensively scanned in the main text.

## Related Work & Insights
-   **vs R-TPT**: A-TPT addresses the "discriminative region protection" gap in R-TPT's global AugMix approach.
-   **vs MTA**: MTA assumes adversarial feature clusters remain valid, which is disproven by A-TPT's findings. A-TPT bypasses feature distortion by operating in the attention space.
-   **vs C-TPT / DiffTPT**: These focus on natural shifts. A-TPT shows that attention anchors, rather than diffusion models, can provide strong adversarial defense at test time.

## Rating
- Novelty: ⭐⭐⭐⭐ (Attention-as-anchor perspective is clean; tricks are reusable.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers clean/adv, OOD, and two backbones; lacks diverse attack types in main results.)
- Writing Quality: ⭐⭐⭐⭐ (Motivations are well-illustrated by Figure 1.)
- Value: ⭐⭐⭐⭐ (A practical, training-free solution for VLM deployment.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](../../CVPR2025/llm_safety/tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)

</div>

<!-- RELATED:END -->
