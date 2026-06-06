---
title: >-
  [Paper Note] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection
description: >-
  [ICML 2026][Image Generation][AIGI Detection] The paper addresses the issue where catastrophic forgetting destroys transferable priors during CLIP fine-tuning for AI-generated image detection. It proposes DGS-Net: decomp…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AIGI Detection"
  - "CLIP LoRA"
  - "Catastrophic Forgetting"
  - "Orthogonal Gradient Projection"
  - "Distillation Alignment"
date: 2026-05-08
content_hash: 0a85e03686a714d5
---

# DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection

**Conference**: ICML 2026  
**arXiv**: [2511.13108](https://arxiv.org/abs/2511.13108)  
**Code**: https://horizontel.github.io/DGS-Net/  
**Area**: AI-Generated Image Detection / CLIP Fine-tuning / Gradient Manipulation  
**Keywords**: AIGI Detection, CLIP LoRA, Catastrophic Forgetting, Orthogonal Gradient Projection, Distillation Alignment

## TL;DR
The paper addresses the issue where catastrophic forgetting destroys transferable priors during CLIP fine-tuning for AI-generated image detection. It proposes DGS-Net: decomposing the classification loss gradient into a harmful positive component $g^+$ and a beneficial negative component $g^-$ by coordinates. The image gradients of the training network are first orthogonally projected into the complement space of the frozen CLIP **text gradient's harmful direction** (Orthogonal Suppression, removing task-irrelevant semantics), then aligned with the frozen CLIP **image gradient's beneficial direction** (Prior Alignment, preserving pre-trained priors). This achieves an average detection accuracy 6.6% higher than SOTA across 50 generative models.

## Background & Motivation
**Background**: Large-scale multi-modal pre-trained models like CLIP provide competitive "open-set" general features for AI-generated image (AIGI) detection. UnivFD achieves decent generalization results on various generators by freezing CLIP and training a linear head. Subsequent works (C2P-CLIP, Effort, NS-Net, etc.) use LoRA fine-tuning to inject forgery-specific features.

**Limitations of Prior Work**: Through t-SNE visualization of four datasets (ProGAN / R3GAN / SDXL / SimSwap) in Fig. 1, the authors find that: (1) the frozen CLIP geometric structure is intact but real/fake images are inseparable; (2) LoRA fine-tuning separates real/fake images but collapses the original CLIP geometric manifold, leading to severe performance drops in cross-generator generalization. In other words, "fine-tuning" is a double-edged sword that extracts detection signals while destroying transferable priors.

**Key Challenge**: Only **a portion** of pre-trained knowledge is useful for detection (directions related to forgery artifacts), while others (related to semantic content) are irrelevant or even interference. Traditional feature distillation performs global alignment, dragging both parts together. Consequently, it fails to preserve truly useful priors while retaining a heavy burden of task-irrelevant semantics.

**Goal**: (1) Ensure the update direction of the training network stays within a "task-harmless" subspace; (2) selectively recover "task-beneficial" pre-trained priors via distillation; (3) avoid the imprecise nature of global feature alignment.

**Key Insight**: The authors interpret the direction using a first-order Taylor expansion—for a classification loss $\mathcal{L}(u, y)$, the **positive component** $g^+=[\nabla_u \mathcal{L}]_+$ of the gradient $\nabla_u \mathcal{L}$ represents coordinates where increasing features increases the loss (i.e., **harmful directions**). The **negative component** $g^- = [\nabla_u \mathcal{L}]_-$ represents coordinates where increasing features decreases the loss (i.e., **beneficial directions**). This coordinate-level decomposition provides a scale for "knowledge value." Fig. 3 shows a baseline experiment: a classifier trained only on BLIP-generated text descriptions achieves ~60% accuracy, indicating that semantic information is partially correlated with real/fake labels but is mostly noise. This provides empirical evidence that the positive component of the text gradient represents the direction of semantic interference.

**Core Idea**: Unify "knowledge preservation/suppression" through surgery in the **gradient space**. The harmful direction of the text gradient indicates what should be suppressed, while the beneficial direction of the image teacher gradient indicates what should be reinforced. The former is removed from the training gradient via orthogonal projection, and the latter is injected into the descent direction via a distillation loss.

## Method

### Overall Architecture
DGS-Net involves three parallel forward branches: the trainable network (CLIP image encoder + LoRA, denoted as student $E_{\text{img}}(\cdot;\theta)$), a frozen text encoder (CLIP $E_{\text{text}}$), and a frozen image teacher (pre-finetune CLIP $E_{\text{img}}^T$). Each branch uses an independent linear head $h$ to calculate BCEWithLogits losses $\mathcal{L}_{\text{img}}, \mathcal{L}_{\text{text}}, \mathcal{L}_{\text{img}}^T$. At the **feature level**, gradients are extracted: $g_{\text{task}}=\nabla_f \mathcal{L}_{\text{img}}, g_{\text{text}}=\nabla_t \mathcal{L}_{\text{text}}, g_{\text{img}}=\nabla_{f^T}\mathcal{L}_{\text{img}}^T$. Two main components follow: Orthogonal Suppression prunes $g_{\text{task}}$ using $g_{\text{text}}^+$, and Prior Alignment adds an extra distillation signal using $g_{\text{img}}^-$. Finally, LoRA parameters $\theta$ are updated via backpropagation. Text prompts are automatically generated from images using BLIP.

### Key Designs

1.  **Gradient Positive/Negative Decomposition (Preliminaries)**:
    *   **Function**: Splits the feature-level gradient of any classification loss into two complementary sets of directions—"to be suppressed" and "to be reinforced"—providing the basis for subsequent components.
    *   **Mechanism**: Based on the first-order expansion of $\mathcal{L}$ at feature $u$: $\mathcal{L}(u+\varepsilon e, y) \approx \mathcal{L}(u, y) + \varepsilon\langle \nabla_u \mathcal{L}, e\rangle$, a perturbation along the unit direction $e_j$ increases $\mathcal{L}$ if and only if $\partial \mathcal{L}/\partial u_j > 0$. Accordingly, $g^+ \triangleq [\nabla_u \mathcal{L}]_+$ and $g^- \triangleq [\nabla_u \mathcal{L}]_-$ are defined (element-wise positive/negative parts). $g^+$ spans a "local suppression half-space," while $g^-$ spans a "local encouragement half-space." This **coordinate-level** determination allows identifying harmful/beneficial dimensions as bit-wise labels.
    *   **Design Motivation**: Traditional distillation only considers the "magnitude of difference" without direction; orthogonal projection methods (like PCGrad) only address directional conflicts of whole vectors. This paper contributes by discovering that positive/negative gradient components correspond to feature directions of different values, upgrading "value-aware gradient surgery" to coordinate granularity.

2.  **Orthogonal Suppression**:
    *   **Function**: Orthogonally projects the training image encoder gradient $g_{\text{task}}$ onto the **orthogonal complement** of the harmful text gradient direction $g_{\text{text}}^+$, ensuring the image encoder does not move along task-irrelevant semantic dimensions.
    *   **Mechanism**: The frozen text encoder computes $g_{\text{text}}$, and its positive component $g_{\text{text}}^+$ is taken as the "local loss-increasing direction caused by semantic dimensions." Since CLIP vision-text features are well-aligned, the text gradient serves as a proxy for the "semantic subspace" within the image gradient. Then, $g_{\text{task}}$ is projected: $\tilde{g}_{\text{task}} = g_{\text{task}} - \langle g_{\text{task}}, \hat{g}_{\text{text}}^+\rangle \hat{g}_{\text{text}}^+$ (where $\hat{g}_{\text{text}}^+$ is the normalized version). The experiment showing ~60% accuracy for BLIP-only text classification confirms that while semantics have weak correlation with real/fake labels, they hinder cross-generator generalization and should be treated as "interference directions."
    *   **Design Motivation**: Prior methods either retained all semantic information (UnivFD) or replaced it entirely (LoRA), failing to recognize that forgery artifacts and semantic content belong to different subspaces. Using the text gradient's harmful direction as an explicit marker to strip away semantics acts as a "semantic filter," restricting training to directions that are semantically irrelevant but beneficial for classification.

3.  **Prior Alignment**:
    *   **Function**: Extracts **beneficial** gradient components $g_{\text{img}}^-$ from the frozen CLIP image encoder as a lightweight distillation signal to re-inject pre-trained directions that aid in real/fake distinction into the student network.
    *   **Mechanism**: The frozen teacher $E_{\text{img}}^T$ performs a forward pass on the same image-label pair to compute $g_{\text{img}}$. The negative component $g_{\text{img}}^-$ is extracted. This is used as a distillation target for lightweight alignment in the gradient space, biasing the student's update direction towards the feature regions represented by $g_{\text{img}}^-$. This informs the student that these pre-existing directions are beneficial for the task and should not be discarded during fine-tuning.
    *   **Design Motivation**: Unlike traditional feature distillation $\|f - f^T\|^2$, only a **beneficial subset** is aligned, corresponding to "selective prior preservation." This prevents the retention of task-irrelevant semantics that cause geometric collapse. The two components work together: Orthogonal Suppression cuts irrelevant dimensions, and Prior Alignment pulls back useful ones, achieving "prior preservation + interference suppression" in the gradient space.

### Loss & Training
The student backbone is a CLIP image encoder with LoRA. Captions generated by BLIP are fed to the frozen text encoder. All three branches use BCEWithLogits. During backpropagation, the gradients are modified via the two-step surgery before being passed to the LoRA parameters. The teacher encoder is a pre-finetune copy of CLIP used only for forward passes to provide $g_{\text{img}}^-$.

## Key Experimental Results

### Main Results
Cross-model detection accuracy on AIGCDetectBench (Partial excerpt, mAcc = average of Real + 17 generators):

| Method | Real | ProGAN | StyleGAN2 | SD v1.4 | ADM | GLIDE | Midjourney | DALLE2 | mAcc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN-Spot | 99.0 | 95.3 | 22.0 | 55.9 | 1.8 | 4.8 | 5.2 | 4.5 | 29.0 |
| UnivFD (Frozen CLIP) | 92.3 | 98.9 | 48.7 | 96.3 | 12.7 | 75.6 | 61.2 | 62.3 | 72.7 |
| FreqNet | 89.9 | 99.4 | 67.5 | 99.9 | 37.7 | 78.9 | 80.8 | 88.8 | 71.7 |
| NPR | 99.3 | 98.9 | 58.7 | 100.0 | 26.5 | 69.2 | 71.0 | 89.8 | 53.1 |

The authors claim in the text that the average detection accuracy across 50 generative models is **6.6%** higher than SOTA. t-SNE (Fig. 1) shows that DGS-Net achieves real/fake separability while maintaining CLIP's original geometric manifold—something LoRA fine-tuning fails to do.

### Ablation Study
Descriptions from the abstract and Section 4:

| Configuration | Description |
| :--- | :--- |
| Full DGS-Net | Orthogonal Suppression + Prior Alignment both enabled |
| w/o Orthogonal Suppression | No removal of harmful text directions → Cross-generator generalization drops |
| w/o Prior Alignment | No injection of beneficial teacher directions → CLIP prior lost, geometric collapse |
| Global feature distill (Traditional) | Distilling entire features → Retains task-irrelevant semantics, inferior to selective alignment |

### Key Findings
*   **"Positive/Negative Gradient Components = Different Values" is the crucial insight**: The fact that BLIP text only predicts 60% accuracy shows semantics have weak label correlation. Projecting out the positive part of the text gradient corresponds to removing these correlated but non-generalizable cues.
*   **CLIP's embedded geometry is critical for cross-generator generalization**: Methods that fail to preserve manifold geometry on t-SNE (like direct LoRA fine-tuning) inevitably lose performance on new generators. DGS-Net protects this geometry using selective prior preservation.
*   **Text gradient is a free proxy for the image semantic subspace**: Using BLIP + CLIP text encoder allows obtaining "harmful directions of semantic dimensions" without extra training or annotation, providing significant gains with low engineering overhead.
*   **Improvements across different generators and forgery families (GAN/Diffusion/Deepfake)**: This suggests that "prior preservation + task-irrelevant suppression" is a universal mechanism orthogonal to specific artifact types.

## Highlights & Insights
*   **Moving distillation from feature space to gradient space**: Traditional distillation aligns $f$ with $f^T$; this work aligns the "descent direction." Gradients are closer to "how knowledge is used" than features, offering higher selectivity and less geometric damage. This paradigm can be extended to any downstream adaptation task requiring partial prior preservation.
*   **Coordinate-level positive/negative split provides a new scale for gradient surgery**: Compared to "whole-vector directional conflict" routes like PCGrad/GradVac, DGS-Net's "component-level sign filtering" is finer-grained and better suited for scenarios with separable knowledge subspaces (like CLIP multi-modality).
*   **Cross-modal gradients as regularization signals**: Pruning image gradients with text gradients essentially repurposes CLIP's vision-text alignment as a free regularization prior, requiring no additional annotation.
*   **Effective interpretability visualization**: The t-SNE results demonstrate that "prior manifold preservation vs. real/fake separability" is usually a trade-off, which DGS-Net successfully navigates. The BLIP-only-60% experiment convincingly explains why text serves as interference.

## Limitations & Future Work
*   The correspondence between text and image gradients relies on the multi-modal alignment quality of CLIP; for backbones with weak alignment (like pure vision SSL), using text gradients as a proxy may fail.
*   The quality of BLIP-generated captions directly impacts the reliability of $g_{\text{text}}$; directions may be inaccurate for images with vague or incorrect captions (e.g., abstract art, deepfake faces).
*   Using "coordinate signs" for splitting is a simplification of first-order expansion; in practice, dimensions are highly correlated, and sign determination may be noisy in coupled dimensions.
*   The training cost increases due to three forward passes and two extra gradient calculations; the paper does not quantify training wall-time or memory overhead.
*   Experiments focus on classification accuracy; robustness against various attacks (compression, perturbation, adversarial examples) is not systematically reported.

## Related Work & Insights
*   **vs. UnivFD**: UnivFD freezes CLIP and learns only a linear head, preserving priors but lacking separability. DGS-Net fine-tunes with LoRA while preserving geometry via selective distillation, achieving the best of both worlds.
*   **vs. LoRA fine-tuning (C2P-CLIP, Effort, NS-Net)**: Pure LoRA drops in generalization because it destroys the CLIP manifold. DGS-Net preserves beneficial priors and suppresses irrelevant semantics through gradient surgery.
*   **vs. PCGrad / GradVac (multi-task gradient surgery)**: Those methods use projections on whole vectors to resolve inter-task conflicts; DGS-Net uses component-level sign filtering for scenarios requiring protection of multiple types of knowledge within a single task.
*   **vs. Feature-level KD (FitNet, Hinton KD)**: Traditional KD aligns features globally; DGS-Net aligns only beneficial directions, avoiding the negative impact of irrelevant knowledge.
*   **Insight**: The idea of "using auxiliary modal gradients as proxies to prune interference subspaces in the primary modality" can be transferred to medical image classification (using text reports to remove background texture effects), video detection (using audio gradients to filter visual redundancy), and cross-domain ReID tasks.

## Rating
*   Novelty: ⭐⭐⭐⭐ Combining "positive/negative gradient components = value difference" and "cross-modal gradient regularization" into a new selective distillation paradigm is a refined approach.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Tested on 50 generators with multiple SOTA comparisons and t-SNE interpretability; main tables cover GAN, Diffusion, and Deepfake families.
*   Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation to preliminaries to method; good synergy between formulas and figures.
*   Value: ⭐⭐⭐⭐ The +6.6% mAcc gain is directly valuable for AIGI detection; the selective gradient distillation framework is transferable to other fine-tuning tasks requiring prior preservation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[ICLR 2026\] Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function](../../ICLR2026/image_generation/diffusion_fine-tuning_via_reparameterized_policy_gradient_of_the_soft_q-function.md)
- [\[ICML 2026\] Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)
- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/image_generation/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)

</div>

<!-- RELATED:END -->
