---
title: >-
  [Paper Note] UrbanFeel：A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective
description: >-
  [ICLR 2026][Multimodal VLM][Paper Note] UrbanFeel establishes a multimodal large model evaluation benchmark for urban street views. Using 11 tasks and 14.3K visual question-answering samples, it simultaneously examines static scene recognition, long-term temporal change understanding, and subjective perception consistency across dimensions like safety, aesth
tags:
  - ICLR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: c3eff4dd2f94b7ba
---
# UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=OtLC2JNGZf](https://openreview.net/forum?id=OtLC2JNGZf)  
**Code**: https://github.com/Hejun0915/UrbanFeel  
**Area**: Multimodal VLM  
**Keywords**: Urban street view understanding, Multimodal benchmark, Temporal changes, Human perception, Subjective environmental assessment

## TL;DR
UrbanFeel establishes a multimodal large model evaluation benchmark for urban street views. Using 11 tasks and 14.3K visual question-answering samples, it simultaneously examines static scene recognition, long-term temporal change understanding, and subjective perception consistency across dimensions like safety, aesthetics, wealth, and liveliness. The study finds that while current MLLMs approach human levels in single-frame subjective judgment, they significantly lag in cross-temporal sorting and urban evolution reasoning.

## Background & Motivation
**Background**: Urban analysis has long relied on remote sensing, street view imagery, and traditional vision models to understand the built environment, such as identifying roads, buildings, and greenery, or estimating urban visual quality. With the advancement of MLLMs like GPT-4o, Gemini, Qwen2.5-VL, and InternVL, researchers expect these models to directly interpret street views for urban planning, renewal monitoring, and human-centered governance.

**Limitations of Prior Work**: Existing urban multimodal benchmarks mostly focus on static, objective, single-time-point tasks, such as geolocation, infrastructure identification, navigation, or functional classification. While these test whether a model "sees" roads and buildings, they fail to address two critical questions in urban research: first, what changes occur in a neighborhood over a decade; second, whether these physical changes alter human perceptions of safety, aesthetics, wealth, and liveliness.

**Key Challenge**: A city is not a static photograph but a complex scene with long-term evolutionary trajectories and subjective human experiences. Although street views are closer to the human perspective than satellite imagery, high performance in single-frame recognition does not imply an understanding of temporal processes like "renovation, decay, expansion, or greening," nor does it guarantee that subjective judgments align with human feelings.

**Goal**: The authors aim to establish an evaluation framework closer to the human urban perception process. The tasks are decomposed into three levels: first, identifying static elements and viewpoint consistency; second, understanding physical changes at the same location across different years; and third, providing subjective environmental perceptions and localized visual evidence consistent with human judgment.

**Key Insight**: Street view imagery is chosen as the core visual source because it captures the pedestrian perspective—details directly related to daily experience such as building facades, road maintenance, greenery, shops, and human activity. By collecting single-perspective and panoramic street views spanning 2007 to 2024, the benchmark tests spatial perspective, temporal evolution, and subjective perception simultaneously.

**Core Idea**: The core innovation of UrbanFeel is integrating "urban physical change" and "human subjective perception" into a single multimodal evaluation benchmark. It uses temporal street view samples and manual perception annotations to verify whether MLLMs truly possess human-centered urban understanding capabilities.

## Method
### Overall Architecture
UrbanFeel proposes a benchmark construction and evaluation pipeline rather than a new MLLM architecture. The authors collect single-perspective and panoramic street views from 11 representative global cities. Through spatio-temporal clustering, quality filtering, and manual annotation, they form multi-view, multi-year samples for the same locations. 11 sub-tasks are designed around static scenes, temporal changes, and subjective perception. Finally, 20 closed-source and open-source MLLMs are evaluated using a unified protocol.

The key to this pipeline is that the data consists of "comparable images of the same urban location across different years and perspectives." Questions evolve from objective identification to explanations of change and subjective judgments aligned with human feelings. Thus, UrbanFeel is positioned as a human-centered urban understanding benchmark rather than a simple urban VQA dataset.

### Key Designs
**1. Three-tier Cognitive Task System: From Seeing street views to Understanding Urban Evolution and Human Feelings**

UrbanFeel decomposes urban understanding into three progressive dimensions. The first tier is static scene perception, examining whether models identify dominant elements and judge if single-perspective and panoramic images match the same location. This tier remains close to traditional vision understanding but adds street view consistency to prevent models from relying solely on object labels.

The second tier is temporal change understanding, focusing on urban evolution across years. Models must identify structural changes, judge change types, and even rank multiple images by development stage. The challenge is distinguishing long-term changes (e.g., facade renovation, road maintenance) from temporary factors (e.g., vehicles, weather, camera angles).

The third tier is subjective environmental perception, covering four dimensions: beautiful, safe, wealthy, and lively. Models are required to provide judgments and correspond them to local visual evidence (e.g., greenery, cleanliness, lighting, foot traffic). This pushes MLLMs beyond "answering based on the picture" to "perceiving the environment like a human."

**2. Multi-view and Multi-temporal Street View Construction: Historical Trajectories for Urban Samples**

UrbanFeel collects over 4,000 images from Mapillary and Google Street View API, covering 11 cities including Paris, Washington, D.C., Kuala Lumpur, and Tolyatti to avoid regional bias.

To ensure temporal comparability, images are organized spatio-temporally. Panoramas are rotated to true north to reduce viewpoint noise. Clustering is performed based on coordinates and timestamps using a 50-meter threshold. Low-quality samples (indoor, blurry, obstructed) are filtered using OneFormer for sky-ratio analysis followed by manual cleaning.

This approach addresses a fundamental issue: if samples are not historical sequences of the same location, model performance is uninterpretable. UrbanFeel transforms samples into comparable objects through spatial clustering and orientation standardization.

**3. Hybrid Question Generation and Manual Evidence Annotation: Scale, Interpretability, and Subjective Consistency**

UrbanFeel contains 14.3K visual questions (11.0K test, 3.3K validation). Formats include binary judgment, multiple-choice, ranking, and open-ended Q&A. Different tasks use different generation methods: metadata and rules generate matching and prediction tasks; change type identification uses semantic segmentation initialization followed by manual correction; subjective tasks are entirely authored by human annotators.

Manual annotation for subjective tasks involves more than simple "good/bad" labels. Annotators with geographic backgrounds judge the four dimensions based on guidelines and mark local visual evidence (e.g., lighting for safety, modern architecture for wealth). These annotations are treated as reference labels representing the consensus of the annotator group.

**4. Unified Evaluation Protocol and Quality Control: Mapping Free Output to Reproducible Metrics**

To handle verbose MLLM outputs, UrbanFeel uses a hybrid evaluation strategy. For judgment, choice, and ranking questions, it uses strict label parsing and exact matching. If a model provides long explanations without clean labels, a lightweight LLM extracts labels without altering semantics. For open-ended Q&A, Sentence-BERT measures semantic similarity with a threshold of $\tau=0.6$.

The overall accuracy is defined as $ACC=N_{correct}/N_{total}$. A human baseline was established using two non-overlapping groups of 10 participants with geographic backgrounds to minimize bias. Quality control includes forced output formats during inference and manual review of ambiguous samples.

## Key Experimental Results
### Main Results
20 MLLMs were evaluated in a zero-shot setting, including closed-source models (GPT-4o, Gemini-2.5-Pro) and open-source models (DeepSeek-VL2, Qwen2.5-VL, etc.). Gemini-2.5-Pro achieved the highest overall score of 65.9, close to the human score of 67.4.

| Model / Baseline | Overall Score | TSR (Temporal Sorting) | GP (Global Perception) | PCR (Panorama Change Recog.) | Main Observation |
|--------|------|------|------|------|------|
| Human | 67.4 | 70.0 | 66.6 | 21.2 | Humans are strongest at long-term sorting but insensitive to tiny pixel changes in panoramas |
| Gemini-2.5-Pro | 65.9 | 52.1 | 67.7 | 36.5 | Closest to humans; subjective perception matches humans, but TSR lags by 17.9 points |
| GPT-4o | 59.4 | 38.9 | 60.2 | 40.5 | Strong in static/local tasks; significantly deficient in temporal ranking |
| Qwen2.5-VL-72B | 60.2 | 26.0 | 65.7 | 40.9 | Strongest open-source model; significantly outperforms humans in PCR |
| Random | 31.5 | 3.9 | 51.0 | 19.5 | Much lower than models; indicates benchmark is not solvable through prior bias |

The core finding is the uneven capability distribution. Models can excel at static recognition or pixel-level change detection (outperforming humans), but they collapse when required to rank multiple street views by development stage. The 17.9-point gap between Gemini and humans in TSR indicates that "long-term urban evolution understanding" remains an unstable capability for MLLMs.

### Ablation Study
The TSR prompt ablation compared Direct Sorting, General CoT, and Re-Thinking strategies. Findings showed that explicit reasoning does not always improve temporal understanding; strong models often degrade due to over-interpretation.

| Model | Direct Sorting | General CoT | Re-Thinking | Description |
|------|------|------|------|------|
| Phi-4 | 5.5 | 4.1 | 7.3 | Small models benefit slightly from re-thinking, but absolute performance is low |
| InternVL3-8B | 4.6 | 3.2 | 5.0 | Near random; struggles with multi-image temporal relations |
| Qwen2.5-VL-72B | 32.0 | 24.7 | 29.7 | CoT actually reduces accuracy |
| GPT-4o | 46.1 | 38.4 | 37.0 | Explicit reasoning caused a drop from 46.1 to 37.0 |
| Gemini-2.5-Pro | 52.5 | 52.1 | 52.1 | Direct sorting is slightly better; reasoning prompts offer no gain |
| o3 | 60.3 | 59.8 | 60.7 | Minor differences; still below human 70.0 |

Authors attribute this to "over-reasoning," where models amplify temporal noise (vegetation, traffic) as evidence of development while ignoring stable structural cues. Urban evolution is not monotonic; it involves complex patterns of demolition, decay, and redevelopment.

### Key Findings
- The difficulty of UrbanFeel lies in organizing evidence across perspectives and years rather than simple object detection. Most models perform well on TCR but fail TSR, marking temporal reasoning as a bottleneck.
- Subjective perception shows a "local human-level" alignment. Models match humans in safety, aesthetics, and liveliness, but lag by 10.1% in the wealth dimension, likely because wealth perception is highly dependent on cultural and socio-economic context.
- Panoramas are not necessarily easier. Geometric distortion and information density increase the burden on MLLMs; single-perspective images average 11.7 percentage points higher accuracy.
- Urban identity prompts affect subjective judgment. Changing the city's name (e.g., Paris vs. Cape Town) alters model results, exposing potential geographic biases or semantic priors.
- Humans and models have different blind spots. Humans excel at placing changes into a logical timeline, while models are more sensitive to fine-grained pixel differences in panoramas (PCR).

## Highlights & Insights
- UrbanFeel moves urban benchmarks from objective spatial recognition to human perception alignment. It asks not just "is there a road," but "has this street changed, and how does it affect human feelings."
- The three-tier task system is ideal for diagnosing MLLM boundaries, clearly showing whether a model fails at recognition, reasoning, or alignment.
- Requiring localized visual evidence for subjective tasks is a transferable design for trustworthy AI in medical, remote sensing, and robotics applications.
- The negative results for CoT remind us that longer natural language reasoning does not equate to better visual evidence integration in temporal tasks; direct sorting often reduces hallucinated narratives.
- Urban identity intervention experiments extend the benchmark from "capability evaluation" to "bias diagnosis," highlighting risks for urban governance.

## Limitations & Future Work
- While covering 11 cities, representation in the Global South remains limited. Subjective annotations are culturally sensitive, and geographic imbalance may affect the generalization of conclusions.
- Subjective labels represent a specific group of annotators. Future work should involve local residents and cross-cultural consistency analysis to better interpret "human perception."
- The benchmark focuses on visible changes but lacks structured socio-economic or policy labels (e.g., why a street was renovated), limiting deeper causal reasoning.
- Tasks are currently atomic perception/reasoning. Future iterations could combine tasks into multi-turn workflows, such as identifying a change, explaining the perceptual impact, and proposing renewal suggestions.
- Due to compliance, Google Street View IDs are shared instead of raw images. This ensures reproducibility but requires users to have their own API access.

## Related Work & Insights
- **vs CityBench / CityGPT / USTBench**: These focus on static/objective infrastructure and navigation. UrbanFeel introduces multi-year sequences and subjective perception to address long-term evolution and human experience.
- **vs UrBench**: While UrBench emphasizes spatial reasoning, UrbanFeel adds historical data and subjective perception to understand how changes affect people.
- **vs Place Pulse / Traditional Perception Models**: Place Pulse uses crowdsourced 1v1 comparisons for scoring. UrbanFeel evaluates MLLMs by requiring them to explain judgments with local evidence, focusing on interpretability.
- **vs Visual Chronicles / Street View Change Detection**: UrbanFeel provides a controlled evaluation with unified metrics to systematically compare MLLM differences in change understanding.
- **Related Work & Insights**: For smart city applications, MLLMs must resist viewpoint noise and avoid mistaking temporary factors for trends, while being monitored for geographic and cultural biases.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines temporal change with subjective perception in one benchmark; significant data and task design contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 20 models with task-level analysis, viewpoint analysis, identity intervention, and prompt ablation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main line and motivation; minor complexity due to many task acronyms.
- Value: ⭐⭐⭐⭐⭐ Directly useful for urban AI, street view understanding, and bias diagnosis; sets a foundation for human-centered urban MLLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] HumanPCR: Probing MLLM Capabilities in Diverse Human-Centric Scenes](humanpcr_probing_mllm_capabilities_in_diverse_human-centric_scenes.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](../../CVPR2026/multimodal_vlm/posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[ICLR 2026\] MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](mme-unify_a_comprehensive_benchmark_for_unified_multimodal_understanding_and_gen.md)
- [\[ICLR 2026\] Human-MME: A Holistic Evaluation Benchmark for Human-Centric Multimodal Large Language Models](human-mme_a_holistic_evaluation_benchmark_for_human-centric_multimodal_large_lan.md)

</div>

<!-- RELATED:END -->
