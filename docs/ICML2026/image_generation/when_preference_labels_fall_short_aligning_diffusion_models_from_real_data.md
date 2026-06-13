---
title: >-
  [Paper Note] When Preference Labels Fall Short: Aligning Diffusion Models from Real Data
description: >-
  [ICML2026][Image Generation][Diffusion Models] This paper posits that preference labels consisting of generated images tend to guide models toward "relatively better but still flawed" samples. It proposes automatically c…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Preference Alignment"
  - "Real Images"
  - "Diffusion-DPO"
  - "Data Curation"
date: 2026-05-08
content_hash: f391fefae6c38e61
---

# When Preference Labels Fall Short: Aligning Diffusion Models from Real Data

**Conference**: ICML2026  
**arXiv**: [2605.19839](https://arxiv.org/abs/2605.19839)  
**Code**: https://cwyxx.github.io/RealAlign  
**Area**: image_generation  
**Keywords**: Diffusion Models, Preference Alignment, Real Images, Diffusion-DPO, Data Curation

## TL;DR
This paper posits that preference labels consisting of generated images tend to guide models toward "relatively better but still flawed" samples. It proposes automatically constructing preference signals using real images and their controllable degraded versions. By using only 512 sample pairs, it aligns SD-1.5 and SD-3.5-M, achieving results that are comparable to or complement Diffusion-DPO / FlowGRPO.

## Background & Motivation
**Background**: Text-to-image diffusion models typically learn likelihood through large-scale image-text pairs first, followed by a preference alignment phase to make outputs more consistent with human aesthetics, realism, and prompts. Methods such as Diffusion-DPO and FlowGRPO, which introduce human preferences, reward models, or pairwise comparisons into diffusion post-training, have become a primary route for improving generation quality.

**Limitations of Prior Work**: Mainstream preference datasets are often sampled from one or more generative models, with humans labeling which image is better. This preference is relative: the selected image might simply have fewer flaws than the other while still possessing local artifacts, unnatural colors, or stylistic biases. If a model optimizes this signal over the long term, it may learn the "generator bias" present in the preference data rather than approaching truly high-quality images.

**Key Challenge**: Preference alignment aims to teach the model what constitutes a good image, but binary labels between two generated images do not sufficiently define an absolute reference for "good." When both images are poor, preference labels only inform the model which one is less worse, making it difficult to provide stable anchors for realism, structural consistency, and semantic coherence.

**Goal**: Ours aims to reduce reliance on manual preference annotations by automatically constructing preference supervision from existing real images, validating whether such supervision can independently align diffusion models or serve as a complementary post-training stage for existing preference methods.

**Key Insight**: Real images from photography platforms or high-quality datasets naturally contain human choices, composition, semantics, and realism. The paper treats real images as positive samples and generates negative samples through locally controllable degradation, deriving preference signals from the differences between real images and their flawed versions.

**Core Idea**: Instead of having humans compare two generated images, real images are treated as "implicit preference references." The diffusion model is first pulled toward the real image distribution, followed by DPO-style pairwise learning using real images and locally degraded images.

## Method
The proposed method is a data-centric diffusion model alignment framework. It does not invent new diffusion samplers or replace the DPO objective but re-designs the source of preference signals: starting from professional real photos in HPDv3, it applies filtering, salient region perturbation, and two-stage training to transform the visual preferences implicit in real data into optimizable supervision.

### Overall Architecture
First, professional photos from HPDv3 are selected as candidate positive samples, filtered by colorfulness to remove visually flat or low-contrast images. Next, U2-Net is used to identify salient regions for each real image, which are then redrawn using prompt-conditioned SD v1.5 Inpainting. Since the inpainting model introduces flaws in local texture, structure, or semantics, the original and redrawn images form a preference pair where the "real reference is superior to the degraded version." Finally, training is conducted in two steps: Stage 1 utilizes a distribution-level objective similar to Diffusion-DRO to bring the model closer to the real image distribution; Stage 2 performs fine-grained preference learning between real and degraded images using Diffusion-DPO, starting from the Stage 1 model.

This workflow is characterized by supervision signals derived entirely from real data and automatic perturbations, requiring no manual comparison labels. It is also tested as a plug-in post-training step after Diffusion-DPO or FlowGRPO to verify that real-data supervision complements traditional preference supervision.

### Key Designs
1. **Real Images as Implicit Preference References**:
	- **Function**: Provides a more stable quality anchor than preference labels of generated samples.
	- **Mechanism**: Professional photos are selected from HPDv3 and filtered by color richness to retain samples with higher visual quality and clearer composition. These images are treated as representatives of preferred regions rather than ordinary unlabeled data.
	- **Design Motivation**: Generated image preference pairs only provide relative rankings, whereas real images are closer to the absolute target distribution humans want models to generate, reducing the inheritance of generator artifacts and stylistic biases.

2. **Salient Region Degradation for Negative Sample Construction**:
	- **Function**: Automatically constructs contrastive samples that correspond closely to real images but are inferior in preference-related dimensions.
	- **Mechanism**: Salient regions are extracted using U2-Net and redrawn with SD v1.5 Inpainting based on prompts. Samples with a PickScore difference of less than 0.02 are discarded to ensure a clear distinction between positive and negative pairs.
	- **Design Motivation**: If negative samples differ too much from positive ones, the model may learn irrelevant variations; if the difference is too small, supervision is too weak. Local degradation concentrates differences on generation quality dimensions such as texture, structure, and semantic consistency.

3. **Distribution Warm-up + Pairwise Preference Learning**:
	- **Function**: Mitigates the distance between the real image distribution and the current generation distribution before fine-grained alignment.
	- **Mechanism**: Stage 1 uses a Diffusion-DRO style objective to compare real images with model-generated samples, moving the model toward the real distribution. Stage 2 uses Diffusion-DPO to increase the likelihood of real images and decrease the likelihood of degraded images, employing a KL constraint to avoid distribution drift.
	- **Design Motivation**: Directly applying real-degraded image pairs to DPO has limited effectiveness because the model's starting point is too far from the real image distribution; an initial distribution warm-up stabilizes subsequent pairwise learning.

### Loss & Training
Training utilizes LoRA. SD-1.5 uses rank 4 and scaling 4; SD-3.5-M uses rank 32 and scaling 64. Stage 1 uses the Diffusion-DRO objective to train a reward/policy model to distinguish real images from current policy-generated images, using a margin to avoid over-optimizing correctly ranked samples. For SD-1.5, the learning rate is $1e^{-4}$ for 1600 steps; for SD-3.5-M, the learning rate is $2e^{-4}$ for 3200 steps. Stage 2 uses the Diffusion-DPO objective, where the positive sample is the real image $x_0^w$ and the negative sample is the degraded image $x_0^l$. The learning rate is $2.56e^{-6}$, with 1000 steps for SD-1.5 and 500 steps for SD-3.5-M.

## Key Experimental Results

### Main Results
The model is trained on SD-1.5 and SD-3.5-M and evaluated on Pick-a-Pic v2, DrawBench, and Parti-Prompts. Metrics include PickScore, ImageReward, UnifiedReward, HPSv3, DeQA, and LAION aesthetic score.

| Model / Dataset | Method | Training Data | PickScore | ImageReward | HPSv3 | DeQA | Aes |
|---------------|------|----------|-----------|-------------|-------|------|-----|
| SD-1.5 / Pick-a-Pic v2 | Base | None | 20.65 | 0.16 | 5.98 | 3.70 | 5.48 |
| SD-1.5 / Pick-a-Pic v2 | Diffusion-DPO | 851k pairs | 21.03 | 0.33 | 6.80 | 3.78 | 5.59 |
| SD-1.5 / Pick-a-Pic v2 | Ours | 512 pairs | 21.04 | 0.38 | 7.33 | 3.96 | 5.64 |
| SD-3.5-M / DrawBench | Base | None | 22.42 | 0.79 | 10.03 | 4.09 | 5.44 |
| SD-3.5-M / DrawBench | Diffusion-DPO | 851k pairs | 22.70 | 0.97 | 10.79 | 3.96 | 5.44 |
| SD-3.5-M / DrawBench | Ours | 512 pairs | 22.80 | 1.08 | 12.77 | 4.26 | 5.55 |
| SD-3.5-M / Parti-Prompts | Base | None | 22.54 | 1.11 | 8.97 | 4.00 | 5.60 |
| SD-3.5-M / Parti-Prompts | Ours | 512 pairs | 22.90 | 1.27 | 10.66 | 4.20 | 5.73 |

### Ablation Study
The two-stage strategy is the most critical component ablation. Results indicate that Stage 2 alone yields minimal gains, Stage 1 alone provides significant improvement, and the combination of both stages is optimal.

| Stage 1 Warm-up | Stage 2 Learning | PickScore | HPSv3 | DeQA | Aes | Notes |
|------------------|------------------|-----------|-------|------|-----|------|
| ✗ | ✗ | 20.65 | 5.98 | 3.70 | 5.48 | SD-1.5 base |
| ✗ | ✓ | 20.74 | 6.11 | 3.93 | 5.52 | Limited gain using DPO on constructed pairs directly |
| ✓ | ✗ | 20.87 | 6.83 | 3.75 | 5.57 | High contribution from real distribution warm-up |
| ✓ | ✓ | 21.04 | 7.33 | 3.96 | 5.64 | Full method is best |

| Generalization Analysis | Result | Explanation |
|----------|------|------|
| Non-realistic Anime, vs SD-3.5-M | 73.33% win rate | Real image signals improve composition/semantics beyond just photo-realism |
| Non-realistic Concept-Art, vs Diffusion-DPO | 66.67% win rate | Remains complementary to existing preference methods |
| DPG-Bench SD-1.5 | Base 62.84, Ours 64.38 | Improvement in dense prompt following |
| DPG-Bench SD-3.5-M | Base 83.40, Ours 85.43 | Equally effective on larger models |

### Key Findings
- Using only 512 pairs of preference samples constructed from real data, SD-1.5 achieves metrics comparable to or better than Diffusion-DPO trained on 851k pairs, particularly with HPSv3 improving from 5.98 to 7.33.
- On SD-3.5-M, real-data preference signals significantly boost HPSv3 from 10.03 to 12.77, demonstrating that the method is not only effective for small models.
- Real-data supervision can be applied after Diffusion-DPO or FlowGRPO for further improvement, showing it addresses the data source dimension rather than competing with specific optimization algorithms.
- Increasing data volume from 256 to 512 yields clear gains, but benefits diminish thereafter, highlighting that sample quality and curation are more critical than pure scaling.

## Highlights & Insights
- The paper articulates the "shortage of preference labels" concretely: the issue is not necessarily the DPO objective but that the positive samples in generated pairs may themselves contain artifacts and stylistic biases.
- Real images are not treated as ordinary supervised data but are constructed into comparable preference pairs through local degradation, focusing the learning signal on interpretable quality differences.
- The two-stage training design is simple yet logical: addressing the distance between real and generative distributions before fine-grained DPO; ablation results support this sequence.
- This paper provides a data curation perspective for image generation alignment: rather than infinitely expanding manual preferences, one should first ask if positive and negative samples truly express the visual standards the model is intended to learn.

## Limitations & Future Work
- Positive samples mainly come from professional photos. Although experiments show transferability to Anime and Concept-Art, further validation is needed for non-photographic domains like abstract art, medical imaging, or blueprints.
- Negative samples are generated by an inpainting model, limiting degradation types to the model's capabilities; if negative sample flaws are uniform, the model might learn specific perturbation patterns.
- Automated metrics remain the primary quantitative basis; the user study involved only 18 participants and 60 prompts, which is relatively small in scale.
- The method currently relies on image-caption pairs and salient region detection; future work could explore real-data preference construction in video diffusion, 3D generation, or multimodal editing.

## Related Work & Insights
- **vs Diffusion-DPO**: Diffusion-DPO relies on large-scale manual preference pairs. Ours retains the DPO optimization form but replaces the source of preference pairs with real and degraded images, reducing annotation costs.
- **vs FlowGRPO**: FlowGRPO improves alignment through reward models and group optimization. Ours points out that reward models also inherit biases from generated samples, offering real-data supervision as a post-training patch.
- **vs ImageReward / PickScore Rewards**: Reward models provide differentiable preference signals but may induce stylistic homogenization; supervision constructed from real images emphasizes realism, structure, and semantic consistency.
- **Insight**: The "positive sample quality" of alignment data may be more important than the quantity of preference labels. In future RLHF/RLAIF for generative models, real data or high-quality expert data should be considered for the role of anchors.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The idea of constructing preference signals from real images is intuitive yet effective; the key contribution lies in data curation and the two-stage training combination.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two SD models, multiple metrics, user studies, and multiple ablations; real-world user testing scale could be further expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and experimental narrative are clear; the method is not over-complicated. Some formulas follow Diffusion-DRO/DPO, requiring a background in diffusion alignment.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for practical image generation alignment, especially in scenarios lacking manual preference labels but possessing high-quality real images.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/image_generation/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/image_generation/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

</div>

<!-- RELATED:END -->
