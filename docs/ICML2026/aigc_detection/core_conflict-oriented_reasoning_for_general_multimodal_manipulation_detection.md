---
title: >-
  [Paper Note] CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection
description: >-
  [ICML 2026][AIGC Detection][Conflict Reasoning] The authors redefine "multimodal fake news detection" as a task of "explicitly capturing conflicts between modalities or with world knowledge." They construct CAC…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "Conflict Reasoning"
  - "Multimodal Forgery Detection"
  - "MLLM Fine-tuning"
  - "Conceptual Boundaries"
  - "Few-shot Generalization"
date: 2026-05-08
content_hash: 1476c051f8aee3c2
---

# CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection

**Conference**: ICML 2026  
**arXiv**: [2606.03066](https://arxiv.org/abs/2606.03066)  
**Code**: https://github.com/shen8424/CORE  
**Area**: AIGC Detection / Multimodal Misinformation / MLLM Reasoning  
**Keywords**: Conflict Reasoning, Multimodal Forgery Detection, MLLM Fine-tuning, Conceptual Boundaries, Few-shot Generalization

## TL;DR
The authors redefine "multimodal fake news detection" as a task of "explicitly capturing conflicts between modalities or with world knowledge." They construct CAC, a 14k corpus with fine-grained conflict annotations, and propose the CORE framework. By using Conflict-Perception Training (CPT) to reshape the conceptual boundaries of MLLMs, the approach significantly outperforms specialized SOTA models on four datasets (DGM4, MDSM, MMFakeBench, NewsCLIPpings) using only 100–750 samples.

## Background & Motivation
**Background**: Current mainstream multimodal fake news detection mostly involves designing specialized models and training paradigms for specific forgery types (face attribute manipulation, full image replacement, text fabrication, entity substitution). For example, HAMMER and ASAP utilize contrastive learning with fine-grained localization, RamDG leverages external knowledge bases for celebrity news, while SNIFFER, FKA-Owl, MMD-Agent, and AMD employ specialized two-stage fine-tuning or multi-step reasoning on MLLMs.

**Limitations of Prior Work**: These solutions are tightly coupled with specific forgery patterns. Since generative models iterate faster than data collection and model retraining, performance drops sharply when encountering new types of manipulation not seen in the training set. The authors refer to this as "data dependency + structural rigidity."

**Key Challenge**: Human recognition of fake news does not rely on having "seen this type before" but rather on "activating world knowledge → identifying internal contradictions." For instance, "Trump wins a football award" violates the common knowledge that "Trump = politician." Existing discriminative models fail to explicitly model such conflicts, instead implicitly fitting pixel-level artifacts.

**Goal**: To decouple general detection capability into two components: (i) possessing world knowledge, and (ii) placing concepts clearly within the representation space to make conflicts visible.

**Key Insight**: The authors first performed two confirmatory experiments. Table 1 shows that MLLMs achieve 96 ACC on a 200-question world knowledge benchmark, while CLIP/ALBEF models only reach 41, indicating that knowledge is already present in MLLMs. However, t-SNE and linear classification show that Qwen2.5VL-3B distinguishes "US President vs. Football Award" with only 61 ACC and "US President vs. UK Prime Minister" with 53 ACC—the conceptual boundaries are blurred. Therefore, the bottleneck is not a lack of knowledge, but a lack of clear conceptual boundaries.

**Core Idea**: Use a corpus annotated with "conflict factors + conflict sources" to explicitly supervise the MLLM in pushing conflicting concepts apart in the representation space. This shifts the focus from "identifying specific forgery patterns" to "identifying the shared conflict structures behind all forgeries," thereby achieving few-shot/zero-shot generalization to unseen forgery types.

## Method
CORE consists of three steps: first, constructing the CAC corpus; second, performing Modality Bridging Pre-Training (MBPT) to train a Cross-Modal Aligner; and finally, conducting Conflict-Perception Training (CPT) to reshape the conceptual geometry of the MLLM. The model is eventually deployed via rapid adaptation for few-shot or zero-shot scenarios on new forgery types.

### Overall Architecture
- Input: A pair of $(I, T)$ multimodal news (image + text).
- Backbone: MLLM (instantiated with Qwen2.5VL-3B and Gemma3-4B).
- Key Middleware: A lightweight Cross-Modal Aligner that inherits the alignment learned during MBPT to fully utilize the fine-grained conflict annotations in CAC.
- Supervision: During the CPT stage, the model is provided with "is_fake" labels, "conflict factors $C_1, C_2$," and "conflict sources $S_1, S_2 \in \{\text{text}, \text{image}, \text{world knowledge}\}$."
- Output: Forgery judgment + natural language conflict explanation (factor / source).
- Deployment: In the rapid adaptation phase, only 100–750 samples are needed for fine-tuning on new forgery types, or even direct zero-shot transfer.

### Key Designs

1.  **Conflict Attribution Corpus (CAC)**:
    - **Function**: Provides explicit supervision for conflict-perception training, decomposing abstract "fake news" labels into localizable "conflict factor + conflict source" pairs.
    - **Mechanism**: 100k $(I, T)$ pairs from the SAMM dataset (which includes manipulation regions and type priors) are used as a base. Background knowledge for corresponding entities is retrieved via the Google Search API. Images, text, forgery priors, and background knowledge are fed into an MLLM randomly sampled from $\{\text{GPT-4o, Gemini2.5-Pro, Qwen3-VL-Plus}\}$ to generate natural language conflict explanations, which are cross-verified by two other MLLMs. Validated samples are distilled into structured $\langle C_1, C_2, S_1, S_2 \rangle$ by an MLLM and reviewed by the others. The final 14k samples have a balanced distribution of conflict sources: 29.98% text / 36.86% image / 33.16% world knowledge.
    - **Design Motivation**: To avoid bias from a single MLLM contaminating labels. The three-way conflict source classification forces the coverage of both cross-modal and external knowledge conflicts, preventing the model from narrowing "conflict" down to simple image-text inconsistency.

2.  **Modality Bridging Pre-Training (MBPT) + Cross-Modal Aligner**:
    - **Function**: Aligns the frozen visual encoder output with the language space, paving the way for CPT to reshape conceptual boundaries.
    - **Mechanism**: A lightweight aligner $f_\phi$ is trained to project visual tokens into a space consumable by the language side, ensuring that subsequent CPT does not have to bypass the modality gap when reshaping concepts. This step follows the same training recipe as MLLM adapters but aims only for "alignment" without introducing conflict supervision.
    - **Design Motivation**: The authors found that direct CPT without prior alignment leads to inconsistent decoding of cross-modal conflict annotations (e.g., "Trump in image vs. football award in text"), as supervision signals are diluted by the modality gap.

3.  **Conflict-Perception Training (CPT)**:
    - **Function**: Translates CAC conflict factor/source supervision into "conceptual boundary reshaping," making multimodal representations more separable along the direction of semantic conflict.
    - **Mechanism**: The MLLM autoregressive loss is used to fit both "forgery detection + conflict explanation generation," with a contrastive regularization term added to the representation layer. Concepts from the same conflict source (e.g., both belonging to "world knowledge conflict") are compared, while codes from different sources are pushed apart. The optimization objective is defined as $$\mathcal{L}_{\text{CPT}} = \mathcal{L}_{\text{LM}} + \lambda \, \mathcal{L}_{\text{contrast}}$$, where $\mathcal{L}_{\text{contrast}}$ aligns $(C_1, C_2)$ pairs provided by CAC, forcing conflicting concepts to form clear clusters in the representation space. Post-training t-SNE (Fig. 2b) shows that initially overlapping "President vs. Football Award" concepts become nearly linearly separable.
    - **Design Motivation**: MLLM knowledge resides in parameters, but the models lack "knowledge of what they know." Direct discriminative heads encourage models to take shortcuts like "finding pixel textures." Explicitly supervising conflict factors/sources forces the model to articulate the observed contradiction and its source before making a "fake" judgment, thus grounding the decision path in conflict reasoning.

### Loss & Training
The total objective of CORE is denoted as $\mathcal{L}_{\text{CORE}} = \mathcal{L}_{\text{LM}}(\hat{y}, y) + \lambda_1 \, \mathcal{L}_{\text{factor}}(\hat{C}, C) + \lambda_2 \, \mathcal{L}_{\text{source}}(\hat{S}, S) + \lambda_3 \, \mathcal{L}_{\text{contrast}}$. During the rapid adaptation phase, the backbone is frozen, and only small LoRA-style adapters are fine-tuned on few-shot samples. In zero-shot scenarios, the CPT-trained checkpoint is used directly for inference, leveraging conflict reasoning to generalize to out-of-domain data like DGM4 and MMFakeBench.

## Key Experimental Results

### Main Results
On four datasets (DGM4, MDSM, MMFakeBench, NewsCLIPpings), using 100–750 samples (100–350 for MMFakeBench) for adaptation, CORE was compared against specialized solutions and 235B-scale models like Qwen3VL and 27B Gemma3.

| Dataset | Samples | CORE$_\text{Qwen}$ | Best Previous Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| DGM4 | 750 | 65.4 | 60.8 (Gemma3-4B) | +4.6 |
| MDSM | 750 | 74.5 | 63.0 (Gemma3-4B) | +11.5 |
| MMFakeBench | 350 | 79.4 | 68.3 (HAMMER++) | +11.1 |
| NewsCLIPpings | 750 | 71.0 | 61.4 (Gemma3-4B) | +9.6 |
| DGM4 | 100 | 59.7 | 51.6 (Qwen3VL-235B) | +8.1 |
| MMFakeBench | 100 | 73.5 | 61.1 (Gemma3-4B) | +12.4 |

CORE$_\text{Gemma}$ further reached 82.0 on MDSM with 750 samples. Notably, 235B-level zero-shot MLLMs (Qwen3VL-235B, Gemma3-27B, LLaMA3.2-90B, SeedVL-1.5) were significantly outperformed by the 3B/4B CORE across all datasets—demonstrating that conceptual boundaries, rather than parameter count, are the primary bottleneck.

### Ablation Study
The key ablation results are summarized below:

| Configuration | MMFakeBench-350 ACC | Description |
| :--- | :--- | :--- |
| CORE$_\text{Qwen}$ Full | 79.4 | Full CAC + MBPT + CPT |
| w/o CPT | ~66.5 | Degenerates to Qwen2.5VL with alignment only; similar to baseline |
| w/o source supervision | Moderate decrease | Model fails to distinguish conflict source modality; cross-modal conflicts suffer |
| w/o factor supervision | Moderate decrease | Model only outputs "fake" labels; unstable generalization to new forgeries |
| w/o MBPT | Significant decrease | Cross-modal alignment missing, diluting CPT supervision signals |

### Key Findings
- **Conceptual boundaries are more important than parameters**: The 3B CORE outperforms 235B zero-shot MLLMs, suggesting the bottleneck in fake news detection is having "clear boundaries for what one knows" rather than "how much one knows."
- **Source supervision is key to generalization**: Removing source supervision leads to the sharpest drop in cross-modal conflict samples (e.g., entity swapping in NewsCLIPpings), showing that modeling "where conflicts come from" forces the model to learn modality alignment priors.
- **Maximum advantage at minimum sample sizes**: With 100 samples on MMFakeBench, CORE outperforms the strongest baseline by +12.4, proving that conflict reasoning is a true "prior" rather than another overfitting shortcut.
- **t-SNE Visualization**: Before training, "US President vs. Football Award" concepts are fully overlapped (linear separability only 61%); after CORE training, the two clusters are clearly separated, qualitatively confirming the conceptual geometry reshaping of CPT.

## Highlights & Insights
- **Translating "Forgery Detection" to "Conflict Reasoning"**: This is a paradigm-level shift, similar to the "Chain-of-Thought" moment in DeepFake detection. Instead of chasing pixel artifacts, the model pursues semantic/common-sense contradictions, naturally gaining robustness against unseen generators.
- **Using an Ensembe of MLLMs for CAC Generation + Review**: This "expert pool + cross-validation" pipeline can be directly applied to other "hard-to-label fine-grained reasoning corpora," such as medical diagnostic explanations or legal evidence conflicts.
- **Clear Separation of Conceptual Boundaries vs. Knowledge**: The combination of Table 1 and t-SNE experiments quantifies the "MLLM knows but cannot articulate" pain point, providing a reusable diagnostic protocol for future MLLM fine-tuning research.
- **View-Agnostic Few-Shot Setting**: The ability to perform stably across four vastly different datasets with only 100 samples suggests that CPT learns a task-agnostic "conflict-perception prior," where rapid adaptation only needs to calibrate the threshold for "what counts as a conflict."

## Limitations & Future Work
- The 14k CAC is still relatively small and depends on SAMM's manipulation priors; coverage for fully synthetic samples (e.g., end-to-end diffusion-generated image/text) is unknown. CAC itself is generated by MLLMs and might inherit world knowledge biases from GPT-4o/Gemini.
- Adversarial robustness was not reported—attackers could potentially weaken CORE by explicitly erasing "common-sense contradictions" during generation (e.g., changing "Trump + Football" to "Football Player + Football Award").
- Only 3B/4B MLLMs were evaluated; whether conflict reasoning continues to scale with backbone size or how it interacts with explicit CoT reasoning (like R1) remains unverified.
- Conflict sources are limited to three categories (Text/Image/World Knowledge); multimodal extensions like audio, video, or temporal consistency would require a redesigned annotation schema.

## Related Work & Insights
- **vs. HAMMER / HAMMER++**: While they use contrastive learning + specialized localization modules for image-text inconsistency, CORE replaces specialized heads with MLLM language outputs for conflict explanations. CORE generalizes from single conflict types to general conflicts and significantly outperforms them in few-shot settings (DGM4-750 +7.5 ACC).
- **vs. FKA-Owl**: FKA-Owl uses external knowledge bases to target "common-sense fallacies." CORE leverages the inherent knowledge of the MLLM and simply reshapes boundaries via CPT, achieving similar capabilities without retrieval dependencies and outperforming FKA-Owl by 10+ points consistently.
- **vs. MMD-Agent / AMD**: These models stack strong priors like "multi-step reasoning + regional coordinates + manipulation types." CORE maintains only the most abstract layer of supervision (conflict factors/sources), resulting in better generalization. This suggests "less is more" for MLLMs—task-specific structures may actually limit backbone transferability.
- **vs. Zero-shot Large MLLMs (235B class)**: Qwen3VL-235B and Gemma3-27B stagnate around 50 ACC on DGM4, while CORE-3B reaches 65+, suggesting that "fake news detection capability" in MLLMs is bottlenecked by conceptual boundaries rather than parameter capacity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Using explicit conflict supervision to reshape MLLM conceptual boundaries" is a rare and effective paradigm shift for this task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across four datasets, multiple sample sizes, and multiple backbones is strong; however, adversarial and fully generative sample coverage is slightly lacking.
- Writing Quality: ⭐⭐⭐⭐ The three-part motivation (Knowledge vs. Boundaries) is very clear, though sub-heading hierarchy within the Method section is slightly disorganized.
- Value: ⭐⭐⭐⭐⭐ Dual output of a new paradigm and a high-quality dataset (CAC) makes this work a strong candidate for a sub-community benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](../../NeurIPS2025/aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[AAAI 2026\] ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models](../../AAAI2026/aigc_detection/actishade_activating_overshadowed_knowledge_to_guide_multi-h.md)
- [\[NeurIPS 2025\] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](../../NeurIPS2025/aigc_detection/reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)

</div>

<!-- RELATED:END -->
