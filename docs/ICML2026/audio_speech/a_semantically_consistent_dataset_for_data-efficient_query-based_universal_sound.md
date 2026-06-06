---
title: >-
  [Paper Note] A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation
description: >-
  [ICML 2026][Audio & Speech][Universal Sound Separation] This paper introduces Hive, a universal sound separation dataset constructed through single-event purification and semantically consistent mixing. Using approximate…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Universal Sound Separation"
  - "Audio Dataset"
  - "Semantically Consistent Mixing"
  - "Data-Efficient"
  - "Single-Event Mining"
date: 2026-05-08
content_hash: c9859359232a79bd
---

# A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation

**Conference**: ICML 2026  
**arXiv**: [2601.22599](https://arxiv.org/abs/2601.22599)  
**Code**: https://cslikai.cn/Hive  
**Area**: Audio & Speech / Universal Sound Separation  
**Keywords**: Universal Sound Separation, Audio Dataset, Semantically Consistent Mixing, Data-Efficient, Single-Event Mining  

## TL;DR
This paper introduces Hive, a universal sound separation dataset constructed through single-event purification and semantically consistent mixing. Using approximately 2.4k hours of high-purity source audio, it enables systems like AudioSep and FlowSep to approach or even surpass the performance of models trained on million-hour datasets across multiple separation metrics.

## Background & Motivation
**Background**: Query-based Universal Sound Separation (USS) aims to separate any target sound from complex mixtures based on text, audio, or visual prompts. Existing approaches are generally categorized into two types: discriminative methods like AudioSep, which directly estimate the target signal, and generative methods like FlowSep and SAM-Audio, which generate source signals via distribution modeling or unified prompting interfaces.

**Limitations of Prior Work**: Many existing methods rely on large-scale in-the-wild audio such as AudioSet and VGGSound. Despite their scale, these datasets often contain only weak labels; for instance, a "rain" clip might be accompanied by wind, traffic, or speech. Under such supervision, models easily learn to treat co-occurring backgrounds as part of the target category, resulting in residual interference or the generation of irrelevant background textures in the separated output.

**Key Challenge**: Universal sound separation requires both broad category coverage and clean, localizable supervisory signals. Simply increasing data and model scale can mitigate some issues but also amplifies weak-label noise and co-occurrence biases, leading to increasingly high training costs.

**Goal**: The authors address a data-centric question: Is it possible to train competitive USS models with significantly less data by purifying source audio into high-purity single events and synthesizing mixtures in a semantically reasonable manner?

**Key Insight**: Instead of proposing a new separation network, the authors identify the bottleneck in the data generation process itself. They decouple data quality into two independent axes: "source event purity" and "mixture plausibility," which are controlled via multimodal model-assisted cleaning and a semantic compatibility matrix, respectively.

**Core Idea**: Replace random concatenation of samples from weakly-labeled in-the-wild audio with high-purity single-event mining and semantically consistent mixing.

## Method
The methodology of Hive focuses on an offline data construction pipeline: extracting candidate segments from multiple public audio libraries, aligning them to a taxonomy optimized for separation tasks, and synthesizing multi-source mixtures based on semantic compatibility. The objective is to make every supervisory sample more "trustworthy" rather than simply "larger."

### Overall Architecture
The input consists of in-the-wild audio from 12 public sources, including AudioSet, VGGSound, FreeSound, and BBC Sound Effects. The output comprises two layers: approximately 0.9M high-purity single-event clips (totaling ~2,442 hours) and 19.6M synthetic training/validation/test mixtures (totaling ~22.4k hours).

The pipeline consists of three steps. First, ontology reconstruction: compressing 474 AudioSet leaf nodes into 283 more separable event categories, removing environmental or format tags like "Inside" or "MP4." Second, single-event semantic-acoustic alignment: combining metadata filtering, multi-event detection, and coarse-to-fine classification to ensure each segment contains only one explicit foreground event. Third, standardization: unifying different sources to $44.1$ kHz and using super-resolution models to restore high-frequency details in low-sample-rate audio.

During synthesis, the paper avoids random mixing by constructing a binary semantic compatibility matrix between event categories. For each mixture, an anchor event is selected, and compatible sources (2 to 5 in total) are added iteratively. Source segments are normalized by length, loudness, and SNR before being combined via an additive mixing model.

### Key Designs
1.  **Separation-Oriented Ontology Reconstruction**:
    - **Function**: Transforms original weakly-labeled categories into a label space suitable for "separable foreground events."
    - **Mechanism**: Merges synonymous or acoustically overlapping labels while removing abstract environments, file formats, and non-localizable attributes. For example, action descriptions are merged into entity categories, and fine-grained biological sounds with weak acoustic distinctions are rolled up to parent classes.
    - **Design Motivation**: USS requires target categories to be mutually exclusive and acoustically distinguishable. Ambiguous labels lead to fuzzy supervision regardless of model capacity.

2.  **Single-Event Semantic-Acoustic Alignment**:
    - **Function**: Filters truly single-event segments from in-the-wild audio and provides precise leaf-node labels.
    - **Mechanism**: Multi-label samples are discarded, followed by zero-shot binary classification using Qwen3-Omni to filter unlabeled co-occurrences or transient interference. An audio-tagging model then predicts coarse parent classes, while Qwen3-Omni performs fine-grained classification to leaf nodes.
    - **Design Motivation**: Relying solely on original dataset labels propagates weak-label noise. The coarse-to-fine combination allows discriminative models to provide robust filtering while multimodal models handle semantic refinement, reducing tail-category mismatches.

3.  **Semantically Consistent Mixing Protocol**:
    - **Function**: Ensures synthesized multi-source scenes are complex yet plausible.
    - **Mechanism**: A compatibility matrix $M \in \{0,1\}^{N \times N}$ is constructed. A new event is added only if it is pairwise compatible with all existing events in the mixture. The mixing formula stacks the target source with interference sources sampled at an SNR range of $[-5,5]$ dB.
    - **Design Motivation**: Random mixing creates unnatural contextual priors even with clean source audio. Semantic constraints guide the model to learn realistic, co-occurring complex scenes rather than being confused by nonsensical combinations.

### Loss & Training
The primary contribution is the dataset and its construction protocol. Separation models follow the original architectures and hyperparameters of AudioSep and FlowSep. AudioSep uses AdamW, batch size 64, and an initial learning rate of $10^{-3}$ with plateau decay. FlowSep uses a fixed learning rate of $5 \times 10^{-5}$. Both models are trained on Hive for approximately 3M steps, with all evaluation outputs resampled to $44.1$ kHz.

## Key Experimental Results

### Main Results
Hive's evaluation is two-fold: verifying the difficulty of high-density semantically consistent mixing on its own test set and validating out-of-distribution (OOD) generalization on third-party benchmarks.

| Dataset / Scenario | Metric | Ours | Key Comparison | Gain / Conclusion |
|--------------------|--------|------|----------------|-------------------|
| Hive test | AudioSep(Hive) SDR / SI-SDR | 5.67 / 5.02 | AudioSep Original 2.37 / 1.58 | High-purity small-scale data significantly outperforms large-scale weakly labeled training. |
| Hive test | AudioSep(Hive) MUSHRA | 68.4 | SAM-Audio 62.6, AudioSep Original 60.9 | Perceptual quality approaches or exceeds million-hour baselines. |
| Hive test | FlowSep(Hive) MUSHRA | 61.8 | FlowSep Original 54.7 | Generative separation also benefits from Hive. |
| USS-Bench | AudioSep(Hive) SDR / OQ | 2.29 / 3.56 | AudioSep Original -1.86 / 2.97 | De-cooccurrence supervision is more effective in OOD scenarios. |
| MUSDB18-HQ | AudioSep(Hive) SDR | 1.36 | AudioSep Original -1.01 | Generalization gains are observed even in music separation. |
| VGGClean_eval | FlowSep(Hive) OQ | 3.18 | FlowSep Original 2.99 | Reference-free quality improvement proves it is not just overfitting Hive. |

### Ablation Study
| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Consistent Mix AudioSep | SDR 4.12, SI-SDR 3.37, CLAP-T 0.29 | Trained on 175k samples using the compatibility matrix. |
| Random Mix AudioSep | SDR 3.12, SI-SDR 2.35, CLAP-T 0.24 | Same source audio without semantic compatibility; SDR drops by 1.0 dB. |
| Consistent Mix FlowSep | LPAPS 4.24, CLAP-T 0.17, OQ 2.79 | Generative models also benefit from plausible mixing. |
| Random Mix FlowSep | LPAPS 4.35, CLAP-T 0.13, OQ 2.64 | Perceptual and semantic metrics both decline. |
| AudioSep Original shortcut gap | co-occ. 1.65 vs decorr. 3.06, gap -1.41 dB | Original training relies more on co-occurrence shortcuts. |
| AudioSep(Hive) shortcut gap | co-occ. 5.48 vs decorr. 5.87, gap -0.39 dB | Hive significantly reduces reliance on interference co-occurrence. |

### Key Findings
- Source purity and semantic consistency are complementary: purifying source audio is helpful, but reasonable mixing further improves separation, perceptual, and semantic metrics.
- Hive demonstrates remarkable data efficiency: models trained on ~2.4k hours of source audio approach or exceed systems trained on 14.1k or even millions of hours.
- Training scale still matters given clean supervision; increasing samples from 175k to 17.5M continues to improve AudioSep's SDR by 1.55 dB, suggesting Hive does not saturate quickly.

## Highlights & Insights
- Accurate problem diagnosis: The authors do not attribute residual interference solely to network architecture, but systematically investigate weak labels, co-occurrence, and random mixing, suggesting the USS bottleneck likely lies in the supervision itself.
- Semantically consistent mixing is a reusable dataset trick: This approach could be transferred to audio-visual tasks, video source separation, or event detection. Constructing category compatibility matrices helps reduce bias introduced by nonsensical negative samples.
- Shortcut paired evaluation is highly valuable: Comparing fixed targets while varying the statistical co-occurrence of interference proves whether a model relies on background shortcuts more effectively than average SDR alone.

## Limitations & Future Work
- Hive primarily consists of synthetic mixtures and lacks room impulse responses (RIRs), realistic recording spatial structures, and device noise, which may lead to domain gaps in real-world deployments.
- The cleaning and compatibility matrix depend on multimodal large models like Qwen3-Omni, which may inherit category biases, particularly for tail classes or ambiguous sounds.
- The study focuses on AudioSep/FlowSep without systematic exploration of the scaling laws for larger or newer unified audio foundation models on Hive.
- Future work could incorporate real RIR augmentation, tail-class-aware sampling, LLM relabel bias audits, and naturally recorded high-density USS benchmarks.

## Related Work & Insights
- **vs AudioSep / CLIPSep**: While these methods focus on model architectures and large-scale weakly-labeled data, Hive emphasizes purified single-event supervision, offering higher data efficiency at the cost of a more complex construction pipeline.
- **vs SAM-Audio**: SAM-Audio represents million-hour scale unified audio models. Hive shows that high-purity small-scale data can narrow the quality gap, though it does not yet replace the multimodal prompting capabilities of larger models.
- **vs Scaper / FUSS**: Scaper and FUSS act as controlled mixing tools/datasets. Hive differs by actively addressing upstream source purification and semantic compatibility for in-the-wild sources.
- **Insight**: For many foundation model tasks, "high information density supervision" may be more cost-effective than blindly expanding weakly-labeled data. Similar purification-synthesis protocols could be applied to video events, robotic multimodal perception, or medical datasets.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The innovation primarily lies in the combination of purification and semantic mixing protocols rather than model architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Hive, third-party benchmarks, semantic consistency, shortcuts, and scaling.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative and rich tables, though the large number of appendices and metrics requires careful filtering.
- Value: ⭐⭐⭐⭐⭐ Highly practical for both universal sound separation and audio foundation model training, particularly for resource-constrained teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ICML 2026\] Multimodal Fusion via Self-Consistent Task-Gradient Fields](multimodal_fusion_via_self-consistent_task-gradient_fields.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ACL 2026\] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](../../ACL2026/audio_speech/data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](../../ICLR2026/audio_speech/efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)

</div>

<!-- RELATED:END -->
