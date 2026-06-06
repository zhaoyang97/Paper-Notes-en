---
title: >-
  [Paper Note] Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination
description: >-
  [ICML2026][Multimodal VLM][Visual Re-examination] This paper introduces VisualSwap and VS-Bench to examine true visual re-examination capabilities by replacing the image after a VLM claims to "take another look." The stu…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Visual Re-examination"
  - "Multimodal Reasoning"
  - "Self-Reflection"
  - "Attention Analysis"
  - "Evaluation Benchmark"
date: 2026-05-08
content_hash: 1290478168694857
---

# Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination

**Conference**: ICML2026  
**arXiv**: [2605.15864](https://arxiv.org/abs/2605.15864)  
**Code**: https://visualswap.github.io/  
**Area**: Multimodal VLM  
**Keywords**: Visual Re-examination, Multimodal Reasoning, Self-Reflection, Attention Analysis, Evaluation Benchmark  

## TL;DR
This paper introduces VisualSwap and VS-Bench to examine true visual re-examination capabilities by replacing the image after a VLM claims to "take another look." The study finds that current reasoning VLMs often follow the inertia of previous text generation; only explicit multi-turn user instructions or enhanced visual attention significantly restore grounding.

## Background & Motivation
**Background**: Multimodal large models are now capable of generating long reasoning chains for tasks such as mathematical charts, geometric problems, and professional visual question answering. New-generation reasoning VLMs also produce self-reflective sentences like "let me double-check the image" within their chain-of-thought, which superficially suggests active verification of visual evidence.

**Limitations of Prior Work**: Whether these self-reflective sentences trigger actual visual re-reading or are merely reasoning heuristics learned by the language model has not been systematically measured. Standard VQA accuracy only tests if a model can understand an image in one go, failing to distinguish between "actual image re-examination" and "appending a checking mantra to an existing reasoning trajectory."

**Key Challenge**: The longer the reasoning chain of a VLM, the stronger the textual context becomes. If visual tokens do not receive sufficient renewed attention during subsequent generation, the model may trust its own generated text over the current image. In other words, the linguistic form of self-reflection and the actual execution of visual re-examination may be decoupled.

**Goal**: The authors aim to transform this into an observable diagnostic task: first have the model generate reasoning based on an original image, then swap it for a visually similar image with a different answer at the moment of "checking the image," observing whether the model detects the conflict, corrects its reasoning, and provides the answer corresponding to the new image.

**Key Insight**: Image replacement serves as a clean intervention. An ideal model, provided it truly re-reads the current visual input, should break away from the old reasoning and pivot to the new answer. If the model fails to look at the image, it will continue to recite details from the original image.

**Core Idea**: Use controlled image swapping to transform "the model saying it is looking" into a verifiable behavior, directly exposing textual inertia and attention failures during spontaneous visual re-examination in VLMs.

## Method
This paper does not propose a new model but rather a diagnostic framework, a specifically constructed benchmark, and a series of mechanistic analyses. It addresses a specific question: when self-reflection triggers appearing in the generation process of a VLM, does the model actually re-read the image tokens?

### Overall Architecture
The framework consists of three layers.

The first layer is the VisualSwap evaluation protocol. Each sample contains an original image $I_a$, a replacement image $I_b$, and the same question $Q$. The two images are similar in layout, style, and semantic scene but differ in key details, resulting in answers $A_a$ and $A_b$ respectively.

The second layer is the VS-Bench dataset. The authors selected questions requiring fine-grained visual understanding from MathVista, MathVerse, MathVision, and MMMU-Pro, constructing 200 image pairs per source for a total of 800 pairs. Construction ensures the question remains natural for both images, the images are globally similar, and key visual details are sufficient to change the answer.

The third layer is behavioral and mechanistic analysis. The authors compare standard single-turn responses, self-reflection probes within the same assistant turn, and multi-turn explicit user requests for re-examination. They use visual token attention, context length, prompt paraphrasing, natural trigger points, and attention amplification to explain failure modes.

In standard reasoning, the model answers directly based on $I_b$ and question $Q$ to obtain Base Accuracy, representing its inherent capability to solve the new image.

In the Probe setting, the model first sees $I_a$ and generates an initial reasoning $R_a$. Subsequently, the image is swapped to $I_b$, and a reflection prompt $P$ is appended within the same assistant response to let the model continue generating $R_b$. If the model truly re-reads the image, it should output $A_b$.

In the Multi-turn setting, $R_a$ is treated as the output of a previous assistant turn. The user then explicitly states "re-check the image" in a new turn. This shares the same new image and history as the Probe but introduces a clear user turn boundary to test if external instructions can interrupt textual inertia.

### Key Designs
1. **VisualSwap Two-stage Replacement Protocol**:
	- Function: Converts "visual re-examination" into a measurable counterfactual test rather than just observing reflection sentences.
	- Core Idea: Stage one lets the model generate reasoning chain $R_a$ based on $I_a$; stage one re-prefills the full context but replaces the image with $I_b$ before continuing generation. Performance degradation is defined as $\Delta=Acc_{base}-Acc_{probe}$.
	- Design Motivation: It is hard to judge if a model looks at the image by a single answer. By placing old reasoning and the new image in conflict, the model can only escape linguistic inertia by truly re-binding visual evidence.

2. **VS-Bench Similar but Opposite-Answer Image Pairs**:
	- Function: Provides 800 samples specifically for visual re-examination across geometry, function graphs, charts, synthetic scenes, and professional multimodal problems.
	- Core Idea: Questions and target images are taken from four strong visual-dependency benchmarks. A second image is generated via human-in-the-loop annotation and Nano Banana Pro, ensuring identical questions and global similarity but differing key details. Quality is verified via CLIP, SSIM, LPIPS, and human discrimination tests.
	- Design Motivation: If images differ too much, the model might detect the swap via anomaly detection; if the question no longer applies, errors stem from data flaws. VS-Bench maintains visual proximity to focus on fine-grained re-observation capabilities.

3. **Attention and Multi-turn Comparative Analysis**:
	- Function: Distinguishes between "lack of capability to read new images" and "capability exists but visual attention is not spontaneously invoked."
	- Core Idea: The authors define a visual attention score $S_{vis}^{(l)}(t)$, calculating the average attention of the current token to image tokens, comparing changes 100 tokens before and after the intervention in Probe vs. Multi-turn settings. They also use 2x visual attention amplification as a training-free intervention.
	- Design Motivation: While the main experiment shows the phenomenon, attention analysis explains the "why." Multi-turn almost restores the baseline and attention amplification improves the probe, suggesting failure is a lapse in autonomous attention control rather than a loss of understanding capability.

### Loss & Training
No new models were trained; the focus is on evaluation and diagnosis. Experiments use official chat templates and default generation configurations, with inference temperature set to 0.1 for both base and probe. VLMEvalKit is used for standardized evaluation and answer extraction.

The Probe is implemented not via hidden state or KV cache injection, but by re-prefilling the entire context with the replaced image. Thus, the model technically possesses the visual representation of $I_b$. This detail is crucial as it excludes the explanation that the model "cannot see the new image."

The attention amplification experiment does not update parameters. It simply multiplies the attention weights assigned to image tokens by 2 during the generation of $R_b$. It acts as a causal probe to verify if insufficient visual attention is the cause of failure.

## Key Experimental Results

### Main Results
The main experiment covers 15 VLMs, including instruct/thinking variants of Qwen3-VL, Qwen2.5-VL, OpenVLThinker, VL-Rethinker, Kimi-VL, and ERNIE-4.5-VL. The most significant observation is that all models show substantial drops under VisualSwap; thinking versions typically drop more severely, and scaling up the model size does not naturally solve this issue.

| Model | Variant | Avg Base | Avg Probe | Degradation $\Delta$ |
|------|------|-----------|-------------|---------------|
| Qwen3-VL-8B | Instruct | 69.1 | 46.6 | 22.5 |
| Qwen3-VL-8B | Thinking | 76.0 | 36.6 | 39.4 |
| Qwen3-VL-32B | Instruct | 79.6 | 61.8 | 17.9 |
| Qwen3-VL-32B | Thinking | 84.9 | 36.6 | 48.3 |
| Qwen3-VL-235B-A22B | Instruct | 81.1 | 61.3 | 19.9 |
| Qwen3-VL-235B-A22B | Thinking | 88.8 | 34.1 | 54.6 |
| ERNIE-4.5-VL-28B-A3B | Instruct | 63.3 | 29.0 | 34.3 |
| ERNIE-4.5-VL-28B-A3B | Thinking | 79.9 | 19.6 | 60.3 |
| Kimi-VL-A3B | Thinking | 69.8 | 27.4 | 42.4 |

The most striking finding is the atypical failure of thinking models. Qwen3-VL-235B-A22B-Thinking achieves 88.8% in standard single-turn tests but drops to 34.1% after the swap; the degradation for ERNIE-Thinking reaches 60.3 percentage points. While long reasoning improves standard task performance, it also reinforces textual trajectories.

### Ablation Study
The authors conducted multiple analyses to rule out explanations like "the new image is harder," "the prompt is suboptimal," or "context length leads to complete incapacity."

| Analysis Setup | Key Metric | Description |
|----------|----------|------|
| $I_a$ vs $I_b$ Independent Inference | Avg difference mostly between 0.1 to 6.3 pts | New images are not inherently harder; probe drops result from old reasoning context interference. |
| Multi-turn User Re-check | Qwen3-VL-235B-Thinking recovers from 34.1 to 85.4 | External user turns re-activate visual grounding, showing capability remains. |
| Context Length (0% to 100%) | Qwen3-VL-235B-Thinking drops from 88.8 to 34.1 | More $R_a$ retention results in stronger textual inertia, most evident in thinking models. |
| 10 Reflection Prompt Paraphrases | Small std dev (approx. 36.5 ± 4.9 for 235B Thinking) | Failure is not caused by specific prompt wording. |
| Natural Trigger Point Swapping | Qwen3-VL-8B-Instruct drops from 46.6 to 34.9 | Swapping at model-generated positions like "wait" results in equal or worse failures. |
| 2x Visual Attention Amplification | Qwen3-VL-8B-Thinking rises from 36.6 to 54.8 | Directly enhancing image token attention mitigates failure, supporting the attention-deficit explanation. |

### Key Findings
- VS-Bench image pairs represent a "distinguishable but conducive to laziness" difficulty: average CLIP similarity 0.95, SSIM 0.86, LPIPS 0.14. In human discrimination tests, 5 volunteers could identify answer-critical differences for all sampled pairs.
- There is a fundamental difference between explicit multi-turn user instructions and self-reflective sentences within the same assistant turn. The former significantly increases visual attention given the same visual input and history, while the latter mostly continues the linguistic trajectory.
- Attention analysis shows that the visual attention increase in Probe is minimal. For example, the visual attention increment for Qwen3-VL-235B-Thinking in a middle layer is ~1.07, compared to 2.21 in Multi-turn. Qwen3-VL-8B-Thinking shows a similar 2x difference.
- Closed-source APIs do not support the mid-response insertion required for the main probe nor do they expose attention; thus, the main experiment focuses on open-source models. Gemini 1.5 Flash Preview effectively only verifies the Multi-turn recovery phenomenon.

## Highlights & Insights
- The evaluation design is highly penetrating: rather than asking "can the model say it's checking," it creates a conflict between old text and new images. This transforms the abstract concept of "authenticity of self-reflection" into a measurable behavioral test.
- The fact that thinking models perform worse is highly insightful. Long CoT is often assumed to be more reliable, but in multimodal tasks, long text can become a strong prior, making it harder for the model to return to visual evidence.
- Multi-turn recovery suggests the problem is not a lack of capability to read images. Models can read new images but spontaneous self-generation lacks a strong control signal; user turn boundaries may correspond more frequently to "requirement to re-respond to external input" in training distributions.
- Attention amplification provides an actionable direction: future multimodal RL or SFT should not only reward final answers and logic but also explicitly reward the re-engagement of visual tokens during reflection.
- This work warns evaluators that model outputs like "I have re-checked the image" should not be taken as reliable evidence. Especially in medical diagnosis, autonomous driving, or chart auditing, linguistic caution may mask perceptual stagnation.

## Limitations & Future Work
- The main probe requires inserting prompts mid-response, which many closed-source APIs do not support. Whether closed-source models suffer from similar spontaneous re-examination failures remains to be verified through other interfaces or experiments.
- VS-Bench pairs are constructed via humans and generation tools. Although similarity and human discriminability are verified, the scale is limited to 800 pairs, and task types lean toward visual math and charts. Real-world open scenes, videos, and multi-image dialogues are not yet covered.
- Attention scores serve as mechanistic evidence. While higher visual attention usually correlates with better grounding, it is not a complete causal explanation. Future work could combine activation patching or visual token-level interventions.
- Attention amplification is a proof-of-concept rather than a deployment solution. Simply doubling the weight of image tokens might introduce noise or over-reliance in other tasks; a more refined dynamic trigger mechanism is needed.
- This paper does not train a new model that truly re-examines autonomously. A natural next step is constructing multi-turn visual verification data, incorporating "detecting conflict between old reasoning and new images" into SFT/RL, using visual attention intensity as an auxiliary reward.

## Related Work & Insights
- **vs. Traditional VQA / MMMU / MathVista**: Traditional benchmarks measure single-input performance; this work measures the ability to re-bind images when reasoning context conflicts with new visual evidence, focusing on "multi-turn reliability."
- **vs. Textual Self-Reflection (Self-Refine / Reflexion)**: Textual reflection checks logic and linguistic output; VLM self-reflection must re-visit visual evidence. This paper shows that merely porting textual reflection paradigms to multimodal reasoning misses grounding execution issues.
- **vs. Reasoning VLMs (OpenVLThinker / VL-Rethinker)**: These models enhance long reasoning and self-correction forms, but VisualSwap reveals a side effect: longer reasoning may anchor the model to its own historical text.
- **vs. Hallucination Detection**: While much research focuses on non-existent entities, this work investigates whether models actually "look" when they claim to verify. This is crucial for interpretability and auditing in high-stakes applications.
- **Insight**: When building VLM agents or visual reasoning systems, "visual re-examination" should be treated as an explicit external step rather than relying on internal CoT reminders. Engineering solutions could include forcing a new user turn, re-encoding the image, or identifying different regions before the final decision.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses image replacement to verify the authenticity of self-reflection; the problem is well-defined and the diagnostic signal is direct.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 15 models, 4 data sources, main experiments, and multiple mechanistic analyses; the chain of evidence is complete.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main line of argument; VisualSwap and Multi-turn comparisons are well-explained, though tables and appendices are dense.
- Value: ⭐⭐⭐⭐⭐ Extremely valuable for evaluating the reliability of reasoning VLMs, particularly in reminding the community not to mistake "the model says it checked" for actual visual grounding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)
- [\[ICLR 2026\] ICYM2I: The Illusion of Multimodal Informativeness under Missingness](../../ICLR2026/multimodal_vlm/icym2i_the_illusion_of_multimodal_informativeness_under_missingness.md)
- [\[ICCV 2025\] Generalizable Object Re-Identification via Visual In-Context Prompting](../../ICCV2025/multimodal_vlm/generalizable_object_re-identification_via_visual_in-context_prompting.md)
- [\[CVPR 2026\] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](../../CVPR2026/multimodal_vlm/seeing_through_touch_tactile_localization.md)

</div>

<!-- RELATED:END -->
