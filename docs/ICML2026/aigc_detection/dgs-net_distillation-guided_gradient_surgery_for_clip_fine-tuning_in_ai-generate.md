---
title: >-
  [Paper Note] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection
description: >-
  [ICML 2026][AIGC Detection][AIGI detection] This paper addresses the issue of "catastrophic forgetting of transferable priors when fine-tuning CLIP for AI-generated image detection" by proposing DGS-Net: the gradient of…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "AIGI detection"
  - "CLIP LoRA"
  - "catastrophic forgetting"
  - "gradient orthogonal projection"
  - "distillation alignment"
date: 2026-05-08
content_hash: 38209dbb7384687a
---

# DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection

**Conference**: ICML 2026  
**arXiv**: [2511.13108](https://arxiv.org/abs/2511.13108)  
**Code**: https://horizontel.github.io/DGS-Net/  
**Area**: AI-generated image detection / CLIP fine-tuning / gradient manipulation  
**Keywords**: AIGI detection, CLIP LoRA, catastrophic forgetting, gradient orthogonal projection, distillation alignment

## TL;DR
This paper addresses the issue of "catastrophic forgetting of transferable priors when fine-tuning CLIP for AI-generated image detection" by proposing DGS-Net: the gradient of the classification loss is decomposed by coordinate into harmful positive components $g^+$ and beneficial negative components $g^-$. The image gradient of the training network is first orthogonally projected onto the complement space of the harmful direction of the frozen CLIP **text gradient** (Orthogonal Suppression, removing task-irrelevant semantics), and then further aligned to the beneficial direction of the frozen CLIP **image gradient** (Prior Alignment, preserving pre-trained priors). As a result, the average detection accuracy across 50 generative models surpasses SOTA by 6.6%.

## Background & Motivation
**Background**: Large-scale multimodal pre-trained models like CLIP provide highly competitive "open-set" general features for AI-generated image (AIGI) detection. UnivFD achieves strong generalization on many generators by simply freezing CLIP and training a linear head, while subsequent works (C2P-CLIP, Effort, NS-Net, etc.) use LoRA fine-tuning to inject forgery-specific features.

**Limitations of Prior Work**: The authors construct four datasets (ProGAN / R3GAN / SDXL / SimSwap) and perform t-SNE visualization (Fig. 1), revealing: (1) Frozen CLIP preserves geometric structure but cannot separate real/fake; (2) LoRA fine-tuning separates real/fake but collapses the original CLIP manifold, severely harming cross-generator generalization. In other words, "fine-tuning" is a double-edged sword for this task—while it provides detection signals, it destroys transferable priors.

**Key Challenge**: Only **part** of the pre-trained knowledge is useful for detection (directions related to forgery artifacts), while the rest (semantic content) is irrelevant or even distracting. Traditional feature distillation uses global alignment, which retains both parts, resulting in "failing to preserve truly useful priors and retaining a large amount of task-irrelevant semantic baggage."

**Goal**: (1) Ensure the update direction of the training network only moves within the "task-benign" subspace; (2) Selectively pull back "task-beneficial" pre-trained priors via distillation; (3) Avoid global feature alignment.

**Key Insight**: The authors leverage a first-order Taylor expansion perspective—given a classification loss $\mathcal{L}(u, y)$, the **positive component** of the gradient $g^+=[\nabla_u \mathcal{L}]_+$ indicates "increasing features along these coordinates increases loss," i.e., **harmful directions**; the **negative component** $g^- = [\nabla_u \mathcal{L}]_-$ means "increasing features along these coordinates decreases loss," i.e., **beneficial directions**. This coordinate-wise decomposition provides a metric for "knowledge value." Fig. 3 presents a contrast experiment: using BLIP-generated text descriptions to train a classifier yields about 60% accuracy, indicating semantic information is only partially related to real/fake labels, with most being noise—empirically supporting that "the positive component of the text gradient represents semantic interference directions."

**Core Idea**: Unify "knowledge retention/suppression" in **gradient space**—the harmful direction of the text gradient indicates what should be suppressed, while the beneficial direction of the image teacher gradient indicates what should be enhanced; the former is removed from the training gradient via orthogonal projection, and the latter is injected into the descent direction via a distillation loss.

## Method

### Overall Architecture
DGS-Net processes three parallel branches during training: the trainable network (CLIP image encoder + LoRA, called the student $E_{\text{img}}(\cdot;\theta)$), the text encoder (CLIP $E_{\text{text}}$, frozen), and the image teacher (pre-finetune CLIP $E_{\text{img}}^T$, frozen). Each branch uses an independent linear head $h$ to compute BCEWithLogits losses $\mathcal{L}_{\text{img}}, \mathcal{L}_{\text{text}}, \mathcal{L}_{\text{img}}^T$; then, gradients are taken at the **feature** level: $g_{\text{task}}=\nabla_f \mathcal{L}_{\text{img}}, g_{\text{text}}=\nabla_t \mathcal{L}_{\text{text}}, g_{\text{img}}=\nabla_{f^T}\mathcal{L}_{\text{img}}^T$. The two main components are Orthogonal Suppression, which prunes $g_{\text{task}}$ using $g_{\text{text}}^+$, and Prior Alignment, which adds an extra distillation signal using $g_{\text{img}}^-$. Finally, LoRA parameters $\theta$ are updated via backpropagation. BLIP is used to automatically generate prompts for the text side.

### Key Designs

1. **Gradient Positive-Negative Decomposition (Preliminaries)**:

    - **Function**: Decomposes the feature-level gradient of any classification loss by coordinate sign into two complementary directions: "to be suppressed" and "to be enhanced," providing the basis for the two subsequent components.
    - **Mechanism**: For loss $\mathcal{L}$ at feature $u$, the first-order expansion $\mathcal{L}(u+\varepsilon e, y) \approx \mathcal{L}(u, y) + \varepsilon\langle \nabla_u \mathcal{L}, e\rangle$ shows that perturbing along unit direction $e_j$ increases $\mathcal{L}$ iff $\partial \mathcal{L}/\partial u_j > 0$. Thus, define $g^+ \triangleq [\nabla_u \mathcal{L}]_+, g^- \triangleq [\nabla_u \mathcal{L}]_-$ (element-wise positive/negative parts). $g^+$ spans a "locally suppressive half-space," $g^-$ a "locally encouraging half-space." This **coordinate-level** rather than global direction-level determination allows per-dimension labeling of harmful/beneficial features.
    - **Design Motivation**: Traditional distillation considers only "magnitude of difference" regardless of direction; orthogonal projection methods (e.g., PCGrad) consider only whole-vector conflicts. The contribution here is recognizing that "positive/negative gradient components" correspond to "feature directions of different value," enabling value-aware gradient surgery at coordinate granularity.

2. **Orthogonal Suppression**:

    - **Function**: Orthogonally projects the training image encoder gradient $g_{\text{task}}$ onto the orthogonal complement of the harmful direction $g_{\text{text}}^+$ of the text gradient, preventing the image encoder from moving along task-irrelevant semantic dimensions.
    - **Mechanism**: The frozen text encoder computes $g_{\text{text}}$, and its positive component $g_{\text{text}}^+$ is taken as the "local loss-increasing direction due to semantic dimensions." Since CLIP aligns visual and textual features well, the text gradient serves as a proxy for the semantic subspace in the image gradient. $g_{\text{task}}$ is projected onto the orthogonal complement of $\{g_{\text{text}}^+\}$: $\tilde{g}_{\text{task}} = g_{\text{task}} - \langle g_{\text{task}}, \hat{g}_{\text{text}}^+\rangle \hat{g}_{\text{text}}^+$ (where $\hat{g}_{\text{text}}^+$ is normalized). The "BLIP text classifier ~60%" experiment in Fig. 3 supports this—semantics and real/fake are weakly correlated, but relying on them as the main cue harms cross-generator generalization, so they should be removed as "interference directions."
    - **Design Motivation**: Previous approaches either retain all semantic information (e.g., UnivFD) or replace it entirely (e.g., direct LoRA), without recognizing that forgery artifacts and semantic content are fundamentally different subspaces. Using the harmful direction of the text gradient as an explicit marker to strip semantics acts as a "semantic filter," ensuring training only proceeds in directions "unrelated to semantics but reducing classification loss," which is more conducive to cross-generator generalization.

3. **Prior Alignment**:

    - **Function**: Extracts the **beneficial** gradient component $g_{\text{img}}^-$ from the frozen CLIP image encoder as a lightweight distillation signal, injecting directions from pre-training that are "helpful for real/fake discrimination" back into the student network.
    - **Mechanism**: The frozen teacher $E_{\text{img}}^T$ computes $g_{\text{img}}$ on the same image-label pairs; its negative component $g_{\text{img}}^-$ (by definition, "directions where positive perturbation reduces loss") is used as the distillation target, aligning the student's update direction in gradient space toward the feature region represented by $g_{\text{img}}^-$. This is akin to the teacher telling the student, "these directions are beneficial from pre-training, don't wash them out during fine-tuning."
    - **Design Motivation**: Unlike traditional feature distillation, which globally aligns $\|f - f^T\|^2$, this approach aligns only the **beneficial subset**, corresponding to "selective prior preservation." This avoids dragging along task-irrelevant semantic parts from pre-training (the root cause of geometric collapse). The two components work together—Orthogonal Suppression removes irrelevant dimensions, Prior Alignment restores useful ones, achieving "prior retention + interference suppression" in gradient space.

### Loss & Training
The student backbone is a CLIP image encoder with LoRA injection; BLIP automatically generates captions for the frozen text encoder. All three branches use BCEWithLogits, and during backpropagation, gradients are modified as described above before being passed to the LoRA parameters. The teacher encoder is a pre-fine-tune CLIP copy, used only for forward passes to provide $g_{\text{img}}^-$.

## Key Experimental Results

### Main Results
AIGCDetectBench cross-model detection accuracy (partial excerpt, mAcc = average of real + 17 generators):

| Method | Real | ProGAN | StyleGAN2 | SD v1.4 | ADM | GLIDE | Midjourney | DALLE2 | mAcc |
|--------|------|--------|-----------|---------|------|-------|------------|--------|------|
| CNN-Spot | 99.0 | 95.3 | 22.0 | 55.9 | 1.8 | 4.8 | 5.2 | 4.5 | 29.0 |
| UnivFD (frozen CLIP) | 92.3 | 98.9 | 48.7 | 96.3 | 12.7 | 75.6 | 61.2 | 62.3 | 72.7 |
| FreqNet | 89.9 | 99.4 | 67.5 | 99.9 | 37.7 | 78.9 | 80.8 | 88.8 | 71.7 |
| NPR | 99.3 | 98.9 | 58.7 | 100.0 | 26.5 | 69.2 | 71.0 | 89.8 | 53.1 |

The authors state in the text: the average detection accuracy across 50 generative models surpasses SOTA by **6.6%**; t-SNE (Fig. 1) shows DGS-Net achieves both real/fake separability and preservation of the original CLIP manifold—something LoRA fine-tuning alone cannot do.

### Ablation Study
Paper abstract + Section 4 description (detailed tables are in the latter part of the paper, not included in the cached excerpt):

| Configuration | Description |
|---------------|-------------|
| Full DGS-Net | Both Orthogonal Suppression and Prior Alignment enabled |
| w/o Orthogonal Suppression | No removal of harmful text direction → cross-generator generalization drops |
| w/o Prior Alignment | No injection of beneficial image teacher direction → CLIP prior lost, geometric collapse |
| Global feature distill (traditional) | Distills the entire feature → drags along task-irrelevant semantics, less effective than selective |

### Key Findings
- **"Positive/negative gradient components = different value" is a key insight**: BLIP text can only predict 60% real/fake, indicating semantics are weakly correlated with labels; orthogonally projecting out the positive part of the text gradient effectively removes "these partially relevant but non-generalizable cues."
- **CLIP's intrinsic geometry is crucial for cross-generator generalization**: Methods that fail to preserve manifold geometry (e.g., direct LoRA fine-tuning) inevitably lose performance on new generators; DGS-Net uses selective prior to protect this geometry.
- **Text gradient is a free proxy for the image semantic subspace**: BLIP + CLIP text encoder can provide "harmful semantic directions" without extra training or annotation, with low engineering cost and clear effect.
- **Performance improves across different generators and forgery families (GAN/Diffusion/Deepfake)**: Indicates that "prior retention + task-irrelevant suppression" is a general mechanism orthogonal to specific artifact types.

## Highlights & Insights
- **Moves distillation from feature space to gradient space**: Traditional distillation aligns $f$ and $f^T$; this work aligns "descent directions." Gradients are closer to "what knowledge is being used," enabling more selective retention and less geometric damage—this paradigm can be extended to any downstream adaptation task requiring "partial prior retention."
- **Coordinate-level positive/negative splitting provides new granularity for gradient surgery**: Compared to PCGrad/GradVac's "whole-vector conflict" approach, DGS-Net's "component-wise sign filtering" is finer-grained and better suited for "knowledge-separable subspace" scenarios (e.g., CLIP multimodality).
- **Cross-modal gradients as regularization signals**: Using text gradients to prune image gradients essentially repurposes "CLIP text-visual alignment" as a free regularization prior, requiring no extra annotation.
- **Strong interpretability via visualization**: The three t-SNE plots directly illustrate the usual trade-off between "prior manifold vs. real/fake separability," with DGS-Net achieving both; the BLIP-only-60% contrast experiment convincingly explains "why text is interference."

## Limitations & Future Work
- The correspondence between text and image gradients depends on the multimodal alignment quality of CLIP; if replaced with a weakly aligned backbone (e.g., pure visual SSL), using text gradients as a proxy may fail.
- The quality of BLIP-generated captions directly affects the reliability of $g_{\text{text}}$; for images with vague or incorrect captions (e.g., abstract art, deepfake faces), the text gradient direction may be inaccurate.
- Using "coordinate sign" for positive/negative splitting is a simplification of the local first-order expansion; in practice, feature space dimensions are highly correlated, and sign determination may be noisy in highly coupled dimensions.
- Three forward passes plus two extra gradient computations increase training cost; the paper does not quantify wall-time or memory overhead.
- Experiments focus mainly on classification accuracy, without systematic reporting of robustness to various attacks (compression, perturbation, adversarial examples).

## Related Work & Insights
- **vs UnivFD**: UnivFD freezes CLIP and trains only a linear head, preserving priors but failing to separate real/fake; DGS-Net fine-tunes LoRA while using selective distillation to preserve geometry, achieving both advantages.
- **vs LoRA fine-tune (C2P-CLIP, Effort, NS-Net)**: Pure LoRA fine-tuning loses generalization due to manifold collapse; DGS-Net preserves beneficial priors and suppresses irrelevant semantics via gradient surgery.
- **vs PCGrad / GradVac (multi-task gradient surgery)**: Those methods project at the whole-vector level to resolve task conflicts; DGS-Net filters by component sign, targeting the "single-task but multi-knowledge protection" setting.
- **vs Feature-level KD (FitNet, Hinton KD)**: Traditional KD globally aligns features; DGS-Net aligns only beneficial directions, avoiding negative transfer from irrelevant knowledge.
- **Insights**: The idea of "using auxiliary modality gradients as proxies to remove interference subspaces in the main modality" can be transferred to—medical image classification (using text report gradients to remove background texture effects), video detection (using audio gradients to filter visual redundancy), cross-domain ReID, and other multimodal adaptation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines "gradient positive/negative components = value difference" and "cross-modal gradients as regularization" into a new selective distillation paradigm; elegant approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ 50 generators + multiple SOTA comparisons + t-SNE interpretability; not all ablation data included in the cache, but the main table covers GAN/Diffusion/Deepfake families.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation → preliminaries → method; formulas and figures (Fig. 1 t-SNE + Fig. 3 BLIP experiment + Fig. 2 framework) are well integrated.
- Value: ⭐⭐⭐⭐ +6.6% mAcc provides direct value for AIGI detection; the selective gradient distillation framework can be transferred to other fine-tuning tasks requiring pre-trained prior retention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](../../CVPR2026/aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)
- [\[ICML 2026\] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)
- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](../../ACL2026/aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)

</div>

<!-- RELATED:END -->
