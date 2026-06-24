---
title: >-
  [Paper Note] VisualOverload: Probing Visual Understanding of VLMs in Really Dense Scenes
description: >-
  [CVPR 2026][Multimodal VLM][VQA benchmark] This paper constructs VisualOverload using 150 public domain paintings with ultra 4K resolution and highly dense human activities. It is a VQA benchmark featuring 2,720 human-annotated QA pairs with private ground truth, specifically designed to test foundational perception (Activity/Attribute/Counting/OCR/Reasoning/Scene Classification) of VLMs in "visual overload" scenarios. Experimental results across 37 models show that even the…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "VQA benchmark"
  - "Dense scenes"
  - "Fine-grained perception"
  - "Counting/OCR"
  - "Logical consistency"
date: 2026-05-08
content_hash: 544754bb2aa53410
---

# VisualOverload: Probing Visual Understanding of VLMs in Really Dense Scenes

**Conference**: CVPR 2026  
**arXiv**: [2509.25339](https://arxiv.org/abs/2509.25339)  
**Code**: https://paulgavrikov.github.io/visualoverload (Evaluation server + Private ground truth)  
**Area**: Multimodal VLM  
**Keywords**: VQA benchmark, Dense scenes, Fine-grained perception, Counting/OCR, Logical consistency

## TL;DR
This paper constructs VisualOverload using 150 public domain paintings with ultra 4K resolution and highly dense human activities. It is a VQA benchmark featuring 2,720 human-annotated QA pairs with private ground truth, specifically designed to test foundational perception (Activity/Attribute/Counting/OCR/Reasoning/Scene Classification) of VLMs in "visual overload" scenarios. Experimental results across 37 models show that even the strongest model, o3, achieves only 19.6% accuracy on the hardest subset, suggesting that the notion that "foundational visual understanding has been solved" is an illusion.

## Background & Motivation
**Background**: VQA has long served as a universal metric for measuring the image understanding capabilities of VLMs. Recently, SOTA models have approached saturation on classic benchmarks like SeedBench and MMVet, creating an impression that "foundational visual understanding is solved." Consequently, the community has shifted its evaluation focus toward domain-specific expertise (e.g., expert-level reasoning in MMMU).

**Limitations of Prior Work**: However, these scores overestimate the actual perceptual capabilities of models. Mainstream benchmarks either rely on low-resolution images focusing on global scene understanding of the foreground, or even when using high resolution, they merely perform "needle-in-a-haystack" retrieval of isolated small targets. Neither approach forces models to digest the complexity of a "scene simultaneously packed with information," thus failing to detect weaknesses exposed in real-world safety-critical applications that rely on fine-grained perception.

**Key Challenge**: The authors identify the bottleneck in **visual representation and multimodal alignment**. The primary role of a vision encoder is to compress spatial information into a semantic space. Empirical risk minimization only encourages the preservation of features that are "common during training and useful for the task," which naturally sets an upper bound on fine-grained perception. When a scene becomes so dense that its information content exceeds the "bandwidth" of the encoder (extreme examples being random noise), the model inevitably loses details.

**Goal**: To create a benchmark specifically for "dense + high-resolution" scenarios using foundational visual tasks that require no expert knowledge to pressure-test VLMs, and to quantify the extent and manner in which models fail under such pressure.

**Key Insight**: Utilize **public domain paintings with high densities of people, actions, and sub-plots** as a natural source of "visual overload." These artworks were originally created to be visually overwhelming, requiring prolonged gaze to capture all details. Since they are all in the public domain (artists deceased > 100 years), they are "fresh" (unlikely to have leaked into pre-training corpora) and free of copyright risks.

**Core Idea**: Instead of testing new knowledge or difficult question types, the benchmark uses foundational visual tasks that everyone considers "simple," but places them in scenes so dense they make models "gasp for air," revealing the true limitations of SOTA VLMs.

## Method

### Overall Architecture
This is a pure benchmark paper without training a new model. The core output is the dataset itself and an anti-cheating evaluation protocol. The pipeline involves: collecting 150 public domain paintings from Google Arts & Culture → uniformly downsampling to 4K → six annotators manually creating questions across six task categories (approx. 18 questions per image, including multiple-choice/binary/free-form formats, with binary questions designed as "logically opposite" pairs) → running 37 VLMs for quality control (filtering language bias where models answer correctly without the image) → splitting items into easy/medium/hard tiers based on the accuracy rates of all models → keeping ground truth private for evaluation server scoring only.

### Key Designs

**1. Visual Overload Source: Saturating "Information Bandwidth" with Dense Paintings**

Addressing the limitation that existing benchmarks fail to test collapses in dense scenes, the authors deliberately selected paintings filled with people, actions, and simultaneous narrative branches against extremely detailed backgrounds. These images are naturally ultra 4K (mostly $\ge 3840 \times 2160$). They were uniformly downsampled to the pixel count closest to 4K while maintaining aspect ratios (28 images originally below 4K were kept as-is but remained above Full HD). The fundamental difference from "needle-in-a-haystack" benchmarks is that the latter only tests retrieval of one isolated detail, while VisualOverload requires the model to simultaneously digest the richness of the entire painting—directly targeting the hypothesis of the "compression-induced information loss" upper bound of encoders. Public domain status also solves leakage (fresh data) and copyright issues.

**2. Six Foundational Tasks + Three Answer Formats: Testing Perception, Not Knowledge**

The questions cover six foundational visual tasks: Activity Recognition (N=150), Attribute Recognition (N=149, primarily color with spatial constraints like "the color of the leftmost ship's flag"), Counting (N=559), OCR (N=118, including English/Latin/Chinese/Dutch/Greek), Reasoning (N=356, requiring chain-of-thought, e.g., "Must one cross water to reach the two windmills on the right?"), and Scene Classification (N=1388, shallow understanding expected to be easy). Formats include multiple-choice (4-way or binary yes/no) and free-form; Counting and OCR use free-form (no options) to increase difficulty. All questions mandate that "the answer must be derived from the image, not linguistic priors," and only ask about things directly observable or reasonably inferable, excluding subjective interpretations or expert knowledge. This cleanly decouples "foundational perception" from "art history expertise."

**3. Logically Opposite Paired Questions: Low-Cost Detection of Shortcuts and Inconsistency**

Every binary yes/no question is paired with a logically opposite question (e.g., "Is it daytime?" paired with "Is it nighttime?"). This serves two purposes: first, it lowers the random guess baseline for binary questions (scored in pairs, both must be correct to earn a point, i.e., pair-wise accuracy); second, it provides a direct signal for **logical consistency**. A strong model, even if wrong, should remain logically consistent on opposite questions (saying "yes" to daytime should imply "no" to nighttime). This quantifies whether the model is truly reasoning based on the image or just guessing via shortcuts.

**4. Private Ground Truth + Blind Test Quality Control: Closing Leakage and Bias Loopholes**

To prevent the benchmark from being ingested by future VLM training data, ground truths are strictly private. Only images and questions are public, with scoring handled by an evaluation server using rate-limiting (to prevent ground truth extraction attacks). No development set is provided as tasks require no finetuning. Regarding language bias: the three strongest open-source models (InternVL3-38B, Qwen2.5-VL 32B, LLaVA-OV 72B) performed **blind tests** (ablating the image) to identify questions solvable by text alone. Gemini 2.5 Pro then detected language bias in each question to filter serious offenders (e.g., the correct answer is an obvious outlier or implied by context). After quality control, blind performance for most tasks dropped to near-random; Attribute and Counting remained slightly higher due to statistical patterns in ground truth distributions (e.g., small numbers being more common), which are irreducible priors rather than generalizable shortcuts.

**5. Difficulty Tiers Based on Model Accuracy: Objective Definition of "Hard"**

Difficulty is not judged subjectively by humans but by the **accuracy rate** of all 37 models for each question: accuracy in $[0,20]$ is Hard, $(20,90)$ is Medium, and $[90,100]$ is Easy. Thus, tiers are calibrated by the "actual performance of machines." The Hard tier naturally consists of questions that "almost no model gets right," pushing even the strongest models to their limits (o3 scores only 19.6% in the Hard tier).

## Key Experimental Results

### Main Results
Evaluation of 37 VLMs (open-source 450M–109B across small/medium/large parameters + specialized high-res models + 4 closed-source frontier models), using greedy decoding (or sampling where greedy fails, e.g., Llama 4). Accuracy is defined as an exact match with ground truth; Counting additionally reports RMSE, and OCR reports normalized Levenshtein edit distance.

| Model | Params (B) | Count | OCR | Reason | Scene | Hard | Total |
|-------|------------|-------|-----|--------|-------|------|-------|
| Random Baseline | - | 0.0 | 0.0 | 25.0 | 25.0 | 3.7 | 16.0 |
| Consistent Guess Baseline | - | 0.0 | 0.0 | 42.5 | 50.0 | 4.7 | 27.2 |
| o3 (Best Closed) | – | 36.7 | 61.0 | 75.1 | 94.7 | **19.6** | **69.5** |
| o4-mini | – | 38.3 | 62.7 | 67.8 | 93.7 | 17.2 | 69.1 |
| Gemini 2.0 Flash | – | **41.7** | 57.6 | 56.6 | 92.1 | 19.1 | 68.1 |
| InternVL3 38B (Best Open) | 38 | 35.4 | 45.8 | 69.8 | 92.2 | 7.2 | 67.6 |
| InternVL3 8B | 8 | 32.2 | 42.4 | 59.0 | 93.4 | 7.9 | 63.9 |
| Qwen2.5-VL 72B | 72 | 35.1 | 72.9 | 53.2 | 90.5 | 13.4 | 65.7 |
| LLaVA 1.5 7B | 7 | 13.2 | 3.4 | 39.5 | 43.2 | 1.9 | 30.8 |

Key Findings:

- **Counting is a Universal Failure**: Even the best, Gemini 2.0 Flash, reaches only 41.7%, while most models fall between 13–35%. Models cannot even accurately count animals in a scene.
- **Large Divergence in OCR**: The best o4-mini reaches 62.7%, but the LLaVA series drops to single digits (3.4%). Qwen2.5-VL 72B is unexpectedly strong at OCR (72.9%).
- **Reasoning Approaches Guessing Baseline**: Almost all models show minimal improvement over the "Consistent Guess Baseline" (42.5%) in reasoning; some small models perform worse. The only positive outlier is o3 (75.1%), attributed to its reasoning mode.
- **Scene Classification is the Only "Solved" Task**: 8B parameters are sufficient for 93.4%, as it only requires shallow global understanding—confirming that existing benchmarks overestimate capabilities.
- **Counter-intuitive Scaling**: Performance does not always improve with parameter size; the largest versions of InternVL3 and PaliGemma 2 actually show performance drops. Specialized high-res models (VILA HD, ILM-XC2-4KHD) are significantly weaker than standard models of the same size because modern VLMs (e.g., using AnyRes) already support high resolution, making performance more dependent on the backbone and training rather than "high-res specific design."

### Key Findings
- **Most Significant Performance Drops in Counting and Fine-grained OCR**: These tasks cannot be faked with a "global impression"; they require actual element-by-element processing, exposing the fine-grained perception upper bound of encoders.
- **Scene Classification $\neq$ Understanding**: Bright performance in shallow global tasks versus collapse in fine-grained tasks on the same image validates the hypothesis that "existing benchmarks overestimate capabilities."
- **Logical Inconsistency Reveals Shortcut Reliance**: Paired questions show accuracy for hard problems can fall below random/consistent guess baselines, suggesting models rely on shortcuts rather than robust reasoning.

## Ablation Study
While it lacks traditional model ablations, the paper includes several analyses validating the benchmark design:

| Analysis | Configuration | Conclusion |
|----------|---------------|------------|
| Blind Test QC | Ablate image, text only | Most tasks dropped to near-random performance, proving the "need to look at the image." Residual attribute/counting performance stems from unavoidable distributional priors. |
| Counting Tolerance | 10% / 50% / 100% tolerance | A 10% tolerance only recovered 1.6% accuracy, indicating errors are "orders of magnitude" away, not off-by-one. Large tolerances improve scores but are meaningless for real applications. |
| Resolution Ablation | VisualOverload at different res | Confirms that scene density is the true bottleneck, not just raw resolution. |

## Highlights & Insights
- **Smart use of paintings as "stress test" images**: Dense paintings are natural "overload" scenes designed to overwhelm the eye. Being public domain solves copyright, freshness, and source issues simultaneously.
- **Logically opposite pairs provide high signal at low cost**: One question pair serves dual purposes (lowering random baseline + measuring consistency), quantifying whether the model actually "sees" or just "shortcuts."
- **Difficulty tiers calibrated by model accuracy**: This avoids subjective human bias, ensuring the "Hard" tier is empirically difficult for machines, which is more persuasive than manual labeling.
- **The decoupling of Scene vs. Fine-grained tasks is the "Aha!" moment**: On the same image, an 8B model achieves 93% on scene classification but <42% on counting, proving that "foundational visual understanding is solved" is a mirage created by evaluation bias.
- **Specialized high-res models being worse** is a counter-intuitive insight, suggesting that high-res capability now depends more on backbone/training than specific architectures.

## Limitations & Future Work
- **Domain Specificity**: The authors acknowledge that since data consists of artworks, conclusions might not directly transfer to natural images, as art often emphasizes abstraction and style. They view this as a feature (foundational models should be robust like humans across representations), but it remains a caveat.
- **Task Comparability**: Absolute scores across tasks are not directly comparable due to varying difficulty and random baselines (e.g., Counting/OCR 0% vs. Multi-choice 25%).
- **Black-box Analysis**: Error analysis relies on aggregate statistics of all models to protect private ground truths, making it impossible to perform fine-grained per-item attribution for a single model.
- **Future Directions**: Transferring the "dense scene stress test" protocol to natural images, documents, or video; or designing training objectives to specifically reinforce fine-grained perception (e.g., local patch reconstruction/counting pre-training).

## Related Work & Insights
- **vs. Classic VQA (VQA, SeedBench, MMVet)**: These rely on low-res images for global understanding and often use auto-generated questions prone to bias and saturation. This paper uses high-res dense paintings + manual annotation + private truth to make "solved" tasks difficult again, revealing the actual weaknesses behind high scores.
- **vs. High-res "Needle-in-a-haystack" (VILA HD variants)**: Those benchmarks retrieve one isolated detail. VisualOverload requires digesting the whole image's complexity, targeting the encoder's bandwidth limits.
- **vs. MMMU etc.**: MMMU moves toward "expert knowledge + reasoning." This paper does the opposite, focusing on foundational perception without expert knowledge, proving that even this layer is far from solved.
- **vs. Art VQA (VISCOUNTH)**: Those require cultural/art history expertise. This paper explicitly excludes expert knowledge, using paintings only as a "dense information source" for pure visual understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ Clever image source selection and logical pair design, though fundamentally a benchmark paper.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of 37 models + multi-dimensional error analysis + rigorous QC (blind tests/tolerance).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, strong conclusions, and high information density.
- Value: ⭐⭐⭐⭐⭐ Debunks the "Foundational Visual Solved" illusion; private ground truth and server-side evaluation make it a durable resource for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions](do_vlms_perceive_or_recall_probing_visual_perception_vs_memory_with_classic_visu.md)
- [\[ICLR 2026\] HumanPCR: Probing MLLM Capabilities in Diverse Human-Centric Scenes](../../ICLR2026/multimodal_vlm/humanpcr_probing_mllm_capabilities_in_diverse_human-centric_scenes.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[CVPR 2026\] Structural Graph Probing of Vision-Language Models](structural_graph_probing_of_vision-language_models.md)
- [\[CVPR 2026\] EgoAVU: Egocentric Audio-Visual Understanding](egoavu_egocentric_audio-visual_understanding.md)

</div>

<!-- RELATED:END -->
