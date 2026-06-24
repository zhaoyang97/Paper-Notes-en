---
title: >-
  [Paper Note] Common Inpainted Objects In-N-Out of Context
description: >-
  [CVPR 2026][AIGC Detection][Contextual Reasoning] The authors systematically replace objects in COCO images using Stable Diffusion inpainting to generate 97k "in-context / out-of-context" images of the exact same object. They then employ three 72B multimodal large models for consensus annotation of three-dimensional contextual labels ("location / size / co-occurrence"), building COinCO—the first inpainted forgery dataset with complete contextual annotations. Finally…
tags:
  - "CVPR 2026"
  - "AIGC Detection"
  - "Contextual Reasoning"
  - "Diffusion Inpainting"
  - "Forgery Detection"
  - "Dataset"
  - "COCO"
date: 2026-05-08
content_hash: 9bd75abe45c39141
---

# Common Inpainted Objects In-N-Out of Context

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Common_Inpainted_Objects_In-N-Out_of_Context_CVPR_2026_paper.html)  
**Code**: https://co-in-co.github.io/  
**Area**: AIGC Detection / Image Forensics / Contextual Reasoning  
**Keywords**: Contextual Reasoning, Diffusion Inpainting, Forgery Detection, Dataset, COCO

## TL;DR
The authors systematically replace objects in COCO images using Stable Diffusion inpainting to generate 97k "in-context / out-of-context" images of the exact same object. They then employ three 72B multimodal large models for consensus annotation of three-dimensional contextual labels ("location / size / co-occurrence"), building COinCO—the first inpainted forgery dataset with complete contextual annotations. Finally, they demonstrate three downstream tasks: fine-grained contextual classification, inferring objects from context, and training-free enhancement of SOTA forgery localization.

## Background & Motivation
**Background**：Context is the core of human visual understanding—a horse on a meadow is natural, but a mini-zebra next to a person on a beach is immediately "out of place." This judgment of "whether an object belongs to this scene" provides a complementary cue for identifying manipulated or forged content. While existing datasets in the COCO series (COCO-Stuff, LVIS, RefCOCO, etc.) continuously expand objects, scenes, and text alignment, none of them **manipulate the contextual relationships between objects and scenes**.

**Limitations of Prior Work**：The greatest obstacle to training models that can detect "contextual violations" is that **anomalous scenes are extremely rare in the real world**. Objects in common visual datasets almost always reside in their expected places, lacking "out-of-context" samples. Existing diffusion-inpainted forgery datasets (e.g., COCOGlide, TGIF) are either very small (COCOGlide contains only 512 images) or restrict the replaced object to **the same category as the original object**—which does not change the context, resulting in no out-of-context objects and no contextual labels.

**Key Challenge**：Models need a large number of "out-of-context" samples to learn contextual reasoning, but such samples are naturally rare and cannot be acquired through standard collection.

**Goal**：(1) Create a **controllably generated**, large-scale dataset containing both in-context and out-of-context objects with fine-grained contextual annotations; (2) Demonstrate that this dataset can drive downstream tasks centered around "context."

**Key Insight**：Instead of searching for rare samples in the wild, it is better to **precisely replace an object in a real scene**—retaining the overall scene structure while injecting controlled changes to the "object-scene" relationship. COCO happens to provide 80 everyday categories, reliable instance masks, and rich annotations, making it an ideal foundation for context manipulation.

**Core Idea**：Use diffusion inpainting to replace a random object in a COCO image with a random category, and then use LLMs to perform consensus labeling based on three criteria ("location / size / co-occurrence"), thereby **replacing manual collection with generation** to produce large-scale annotated in-context and out-of-context samples.

## Method

### Overall Architecture
The core of COinCO is a pipeline consisting of "dataset construction + three downstream tasks." On the construction side: starting from COCO images, a random object in each image is selected and replaced with a random COCO category using Stable Diffusion (inpainting). An object detector then verifies that "the new object is indeed drawn within the mask area." Finally, three 72B-level multimodal models independently evaluate and reach a consensus based on three criteria—location, size, and co-occurrence—tagging each inpainted object with "in-context/out-of-context" labels and generating step-by-step reasoning texts. On the task side: this data is used to demonstrate three applications—distilling the 72B teacher into deployable 3B context classifiers, training a model to "infer expected objects from context," and injecting contextual signals into SOTA forgery localizers in a training-free manner.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["COCO Images & Annotations"] --> B["Diffusion Inpainting Replacement<br/>Random Object → Random Category"]
    B --> C{"YOLOv8x Verification<br/>New Object inside mask?"}
    C -->|Failure (Retry max 2 rounds)| B
    C -->|Success| D["Three-Model Consensus Labeling<br/>Location/Size/Co-occurrence → In/Out"]
    D --> E["COinCO Dataset<br/>97,722 Images / 73,929 Labeled"]
    E --> F["Task 1: Distilled Context Classification<br/>72B → 3×3B Single-criterion Students"]
    E --> G["Task 2: Objects-from-Context<br/>Instance-level + Clique-level"]
    E --> H["Task 3: Context-augmented Forgery Localization<br/>Training-free boost for suspicious scores"]
```

### Key Designs

**1. Diffusion Inpainting for Controllable Out-of-Context Samples: Select Object → Dilate Mask → Randomly Swap Category**

Since out-of-context samples cannot be collected easily, the authors artificially create them by using Stable Diffusion to "precisely replace an object" in a real scene. For each COCO image, a labeled object is randomly selected, its instance mask is slightly dilated, and the bounding box is used as the inpainting region. The dilation and box expansion are designed to **eliminate visual remnants of the original object**, preventing conflict with the new object's shape. The prompt for replacement is randomly drawn from the 80 COCO categories, leading to category mismatches that naturally create violations in size, location, or co-occurrence. To address the issue where vanilla inpainting pipelines fail to render small objects well, the authors crop around the enlarged inpainting region, perform inpainting, rescale back to the original size, and seamlessly blend the inpainted patch using alpha blending. The authors also observe a counterintuitive phenomenon: stronger inpainting models like Flux and Qwen-Image tend to generate realistic, "in-context" objects, sacrificing the ability to produce out-of-context scenarios while showing poorer spatial control over the inpainting area. Thus, stronger models aiming for high realism are not necessarily better for this task.

**2. Object Detection for "Inpainting Success" Validation + Three Rounds of Retries**

Inpainting often fails (e.g., the target object is not drawn, or is distorted) and must be filtered out automatically. The authors delegate this step to an object detector: if the detector finds the target category inside the mask area, the inpainting is successfully validated. Selecting the detector involves a trade-off: GroundingDINO has the highest precision (96.14%) but only 38.66% recall, which would falsely discard many successful cases; YOLOv8x achieves the best balance with an F1 score of 78.75% (93.78% precision / 67.86% recall), and is chosen as the validator. Failed cases undergo up to two additional rounds of inpainting and verification. If all three rounds fail, the image is discarded, resulting in 97,722 successful images. The authors note that false detections often correspond to low-quality inpaintings ("the object is present but highly distorted") rather than complete rendering failures.

**3. Three-Model Consensus + Three-Criteria Contextual Annotation: Location / Size / Co-occurrence**

Since relying on a single large model to assign "in/out of context" labels is unreliable, the authors use **multi-model consensus**. Three SOTA multimodal large models (Molmo-72B, Qwen2.5-VL-72B, and InternVL3.5-38B) independently perform reasoning on each inpainted image across three criteria based on a structured prompt: ① Location—whether the spatial placement of the object is reasonable for the scene layout; ② Size—whether the scale of the object matches the scene geometry; ③ Co-occurrence—whether its presence alongside existing objects makes sense. **A violation of any single criterion labels the image as out-of-context**, and all three must be satisfied to be classified as in-context. Each model first outputs a per-criterion analysis and then a final binary decision. **Only images where all three models share the exact same final decision are kept** as labeled data, yielding 73,929 images (64,879 out-of-context and 9,050 in-context), while releasing the per-criterion reasoning text. Human evaluation of 1,000 randomly selected samples shows a 95.10% agreement with the consensus labels. Among these, Qwen2.5-VL-72B achieves the highest accuracy (87.80%) on samples with criterion disagreement and its reasoning quality is preferred in 94.70% of cases, making it the primary annotator while the other two serve as cross-verifiers.

**4. Single-Criterion Distillation: 72B Teacher → 3 × 3B Students (One for each of Location / Size / Co-occurrence)**

Although the 72B teacher provides accurate classifications, it is too large and slow for real-world deployment. The authors use knowledge distillation to compress it into a practical size, **splitting it into three separate Qwen2.5-VL-3B student models by criterion**—one for location, one for size, and one for co-occurrence. This achieves a 24× reduction in model size. Each student learns not only the final decision (in/out) but also the **reasoning process for its designated criterion**, preserving interpretable natural language explanations. This modular design of breaking down complex contextual understanding into focused, single-criterion reasoning allows each criterion to be optimized independently. Training is conducted on 24,000 balanced samples from the 73,929 consensus images (in-context:out-of-context = 1:1, with 20k train / 2k val / 2k test), using reasoning responses from Qwen2.5-VL-72B as the ground truth.

**5. Objects-from-Context: VAE latent + MLP, Instance-level + Clique-level**

This is a new task that asks in reverse: "**What object is suitable for this context?**" to evaluate the model's contextual understanding of real images. The model takes two inputs: the inpainted image and a binary mask indicating the target region. It extracts their latents using Stable Diffusion's VAE encoder and feeds them to an MLP to predict over the 80 COCO categories. During training, the dilated bounding box of the original object serves as the mask, and the original category serves as the label. Prediction occurs at two levels: **instance-level** directly predicts the specific category among the 80 classes, while **clique-level** maps the predicted category to 12 COCO superclasses (e.g., animals, appliances, electronics, food, furniture, sports, vehicles) for evaluation. The model supports arbitrary mask sizes and positions, enabling applications such as "inferring the original object before tampering" (forensics) and "recommending suitable objects based on spatial location" (intelligent editing).

**6. Context-augmented Forgery Localization: Training-free Boost in Out-of-Context Regions**

Finally, context is leveraged as a complementary signal for forgery localization without fine-tuning the base models. The method is straightforward: for each pixel, if it falls within the mask region of an object identified as out-of-context ($M_{OC}$), the original forgery score $P(x,y)$ produced by the localizer is multiplied by an enhancement factor $\gamma$ and clipped to 1.0, and otherwise remains unchanged:

$$P'(x, y) = \begin{cases} \min(P(x, y) \times \gamma,\ 1.0), & (x, y) \in M_{OC} \\ P(x, y), & \text{otherwise} \end{cases}$$

In the experiments, $\gamma = 5$. Two settings are evaluated: "oracle" uses the ground-truth out-of-context mask (representing the performance upper bound); the "practical" setting assumes the fake object is unknown and uses Molmo-7B to identify suspicious objects based only on size and location (**deliberately excluding the co-occurrence criterion to prevent false positives**; for instance, if an image contains only an apple and a newly inserted traffic light, co-occurrence analysis might flag both as suspicious). The union of masks for multiple suspicious objects forms $M_{OC}$. This method is robust because even if the LLM falsely labels a real object as out-of-context, the enhancement merely **amplifies the base model's existing predictions** rather than manufacturing false positives out of nothing, as context detection and forgery localization are inherently complementary.

## Key Experimental Results

### Main Results

The contextual reasoning accuracy (%) after single-criterion distillation, validated on both the COinCO test set and the unmodified original COCO images—the latter proves that the model learns true semantics/context rather than inpainting artifacts:

| Criterion | COinCO Pre-distill | COinCO Post-distill | $\Delta$ | Original COCO Pre | Original COCO Post | $\Delta$ |
|------|--------------|--------------|------|------------|------------|------|
| Size | 55.4 | 65.9 | +10.5 | 52.6 | 83.2 | +30.6 |
| Location | 67.6 | 75.4 | +7.8 | 47.6 | 91.4 | +43.8 |
| Co-occurrence | 75.5 | 79.9 | +4.4 | 89.0 | 87.0 | −2.0 |
| Average | 66.2 | 73.7 | +7.6 | 63.1 | 87.2 | +24.1 |

Performance on the "Objects-from-Context" task (evaluated on 2,402 inpainted images from the COCO 2017 validation set), significantly outperforming random and co-occurrence baselines:

| Method | Instance Top-1 | Instance Top-3 | Instance Top-5 | Clique Top-1 | Clique Top-3 | Clique Top-5 |
|------|-----------|-----------|-----------|-------------|-------------|-------------|
| Random | 1.25 | 3.75 | 6.25 | 8.33 | 25.00 | 41.67 |
| Co-occurrence [40] | 1.54 | 4.70 | 7.29 | 9.37 | 30.72 | 52.91 |
| Ours | **16.32** | **31.89** | **42.80** | **35.10** | **61.41** | **78.31** |

Performance of SOTA forgery localizers on COinCO (the forgery localization GT is the entire bounding box of the original object):

| Method | Acc | F1 | AUC | AP |
|------|-----|-----|-----|-----|
| ManTraNet [60] | 85.4 | 30.7 | 84.9 | 50.7 |
| Trufor [20] | 89.7 | 55.9 | 93.4 | 73.6 |
| PSCC-Net [39] | 89.5 | 46.8 | 95.4 | 79.5 |
| CAT-Net [32] | **92.7** | **76.5** | **97.4** | **90.3** |

### Key Findings
- **The performance gains on the original COCO dataset after distillation are far greater than on the COinCO test set**: +43.8 for location, +30.6 for size (avg +37.2). This indicates that the 3B student learned **generalizable context principles transferable to real images** rather than merely memorizing dataset-specific inpainting artifacts. The co-occurrence criterion already reached 89.0% on the original COCO before distillation and slightly declined to 87.0% after, showing it had already captured natural co-occurrence relationships and needed no further tuning.
- **Acc and AUC are generally >84% in forgery localization but are biased**: Since the inpainted regions are on average very small and most pixels in the image are pristine, these metrics are naturally high. **F1 and AP are more reflective of the actual localization accuracy**, where gaps between models become apparent (e.g., CAT-Net F1 76.5 vs. ManTraNet 30.7).
- **Contextual enhancement improves all SOTA localizers in a completely training-free manner**. The oracle setting establishes the performance ceiling, while the practical setting using Molmo-7B based solely on size/location consistently shows effectiveness.

## Highlights & Insights
- **The paradigm of "using generation to construct scarce samples" is highly elegant**: Instead of failing to collect out-of-context samples, the approach uses diffusion inpainting to precisely replace an object in real-world scenes, preserving scene structure while injecting controlled anomalies. This concept of "controllably manipulating real data" is highly transferable to any task plagued by scarce anomalous or long-tail samples.
- **Multi-model consensus combined with per-criterion reasoning** is significantly more reliable than single-model annotating (achieving 95.10% human agreement). Crucially, the dataset releases not just binary labels but also the **per-criterion textual explanations** from the three models, which fuels the subsequent distillation of interpretable student models.
- **Deconstructing and distilling contextual understanding into separate criteria is highly clever**: Training three individual 3B student models for location, size, and co-occurrence reduces parameters 24× while maintaining interpretability and allowing independent iteration per criterion—effectively breaking down complex semantic reasoning into focused sub-tasks.
- **The "non-decreasing" nature of context-augmented forgery localization**: Since context and localization are complementary, even if the large model misclassifies a real object as out-of-context, multiplying by $\gamma$ only amplifies existing predictions from the base localizer, preventing the creation of false positives. This makes a highly simplistic post-processing step remarkably robust.
- **Intentionally excluding the co-occurrence criterion in the practical setting to avoid false positives** is a simple yet pragmatic engineering decision: when few objects are present, co-occurrence analysis can easily flag common, normal objects as out-of-context.

## Limitations & Future Work
- **Limitation to the COCO 80-category closed set**: The replaced objects and contextual criteria are anchored on the pre-defined COCO categories. In open-vocabulary or rare-object scenarios, contextual expectations are inherently more ambiguous, which the authors identify as future work.
- **Severe imbalance between in-context and out-of-context labels**: Among the 73,929 labeled images, 64,879 are out-of-context while only 9,050 are in-context (random category swapping naturally tends to produce violations). Although 1:1 balanced sampling is performed during distillation, the diversity of the "in-context" samples may be limited.
- **Dependence on weaker inpainting models**: The authors find that stronger models like Flux and Qwen-Image exhibit a bias towards producing in-context objects and demonstrate worse spatial control. As a result, the dataset's characteristics are somewhat tied to specific Stable Diffusion inpainting behaviors, meaning artifact distributions will vary across different image generators.
- **Forgery localization GT uses the entire bounding box of the original object**: This includes pristine background pixels within the box as part of the forged region, potentially biasing the computed metrics for fine-grained localization.
- **Context-augmentation is a post-processing heuristic with a manually-tuned hyperparameter $\gamma$ (set to 5)**, whose optimal value may vary across different localizers and scenes.

## Related Work & Insights
- **vs. COCOGlide [20] / TGIF [41]**: These also perform diffusion inpainting based on COCO, but are limited in data volume (e.g., COCOGlide contains only 512 images), restrict replacements to the same category as the original object (leaving context unchanged and generating no out-of-context samples), and lacks contextual labels. COinCO features 97,722 images, cross-category replacements, and three-criteria contextual annotations.
- **vs. COCO Series Extensions (COCO-Stuff / LVIS / RefCOCO etc.)**: These focus on extending annotations, categories, or text alignment, but **do not manipulate contextual relations**. COinCO is the first to blend in-context and out-of-context objects on a large scale.
- **vs. Traditional Contextual Reasoning (Biederman's support/probability/size principles, Graph Contextual Reasoning Network [1])**: Prior works leveraged co-occurrence, spatial support relationships, or graph neural networks. COinCO is the first to carry out contextual reasoning using multimodal LLMs and apply it directly to context-aware forgery detection.
- **vs. SOTA Forgery Localizers (PSCC-Net / CAT-Net / ManTra-Net / TruFor)**: These employ spatial-channel correlation, JPEG compression artifacts, manipulation traces, or RGB+noise fingerprints for localization. Rather than replacing them, COinCO leverages context signals to **augment their predictions in out-of-context regions without fine-tuning**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First inpainting forgery dataset with fine-grained contextual annotations; the combination of "generating scarce out-of-context samples" and "multi-model consensus on three-criteria annotations" is highly solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative results across three downstream tasks, validation on 1,000 human-evaluated images, and transfer validation on original COCO are provided, though analyses on $\gamma$ and design choices for the inpainting models are deferred to the supplementary material.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline, thoroughly articulated motivations, and easy-to-follow explanations aligned with visual aids.
- Value: ⭐⭐⭐⭐⭐ The dataset, tasks, and interpretable distilled models are directly reusable by the image forensics and contextual reasoning communities, with code and data publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries](../../ACL2026/aigc_detection/frame_in_frame_out_measuring_framing_bias_in_llm-generated_news_summaries.md)
- [\[CVPR 2026\] Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection](enabling_supervised_learning_of_generative_signatures_for_generalized_ai-generat.md)
- [\[CVPR 2026\] PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection](ppm-clip_probabilistic_prompt_modeling_for_generalizable_ai-generated_image_dete.md)
- [\[CVPR 2026\] Learning Forgery-Aware Lip Representations Without Forgery Priors](learning_forgery-aware_lip_representations_without_forgery_priors.md)
- [\[CVPR 2026\] Inconsistency-aware Multimodal Schrodinger Bridge for Deepfake Localization](inconsistency-aware_multimodal_schrodinger_bridge_for_deepfake_localization.md)

</div>

<!-- RELATED:END -->
