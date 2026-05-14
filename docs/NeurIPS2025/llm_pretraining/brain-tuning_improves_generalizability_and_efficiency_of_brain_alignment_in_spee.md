---
title: >-
  [Paper Note] Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models
description: >-
  [NEURIPS2025][LLM Pretraining][brain-tuning] This paper proposes Multi-brain-tuning, a method that jointly fine-tunes pretrained speech models on fMRI data from multiple participants…
tags:
  - "NEURIPS2025"
  - "LLM Pretraining"
  - "brain-tuning"
  - "fMRI"
  - "speech model"
  - "brain alignment"
  - "multi-participant"
  - "LoRA"
date: 2026-05-08
content_hash: b875c3a5608454de
---

# Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models

**Conference**: NEURIPS2025  
**arXiv**: [2510.21520](https://arxiv.org/abs/2510.21520)  
**Code**: [bridge-ai-neuro/multi-brain-tuning](https://github.com/bridge-ai-neuro/multi-brain-tuning)  
**Area**: LLM Pretraining  
**Keywords**: brain-tuning, fMRI, speech model, brain alignment, multi-participant, LoRA  

## TL;DR
This paper proposes Multi-brain-tuning, a method that jointly fine-tunes pretrained speech models on fMRI data from multiple participants, reducing the data required for brain alignment by 5×, improving alignment by up to 50%, and generalizing to unseen participants and datasets.

## Background & Motivation
Pretrained language models (LMs) have demonstrated strong performance in predicting brain activity (e.g., fMRI signals) during natural language processing, positioning them as promising tools for studying neural language processing. However, existing brain alignment methods face two major bottlenecks:

1. **Low data efficiency**: Large amounts of fMRI data are required per new participant to reliably estimate model–brain alignment.
2. **Strong participant dependence**: Models are trained independently per participant, precluding cross-participant generalization and group-level analyses.

Even recent methods such as brain-tuning and BrainWavLM incorporate brain data into model training, they remain participant-specific and lack scalability. This paper directly targets these bottlenecks by proposing a scalable multi-participant joint fine-tuning framework.

## Core Problem
How to design a brain-tuning method that: (1) substantially reduces the amount of fMRI data required for new participants; (2) generalizes across participants rather than being participant-specific; and (3) does not degrade model performance on downstream semantic tasks?

## Method

### Pretrained Speech Models
Two mainstream self-supervised speech Transformer families serve as backbones:

- **Wav2Vec2.0**: ~90M parameters, 12 Transformer layers, embedding dimension 768.
- **HuBERT**: Comparable architecture and parameter count to Wav2Vec2.0.

Both are pretrained on ~960 hours of audio with no overlap with the fMRI datasets.

### Datasets
- **Moth Radio Hour** (primary training/evaluation): fMRI recordings from 8 participants listening to autobiographical stories; 3 participants have ~16.1 hours (84 stories) and the remaining have ~6.4 hours (27 stories), TR=2.0s.
- **Narratives** (cross-dataset generalization): 16 participants listening to a 56-minute fictional short story, TR=1.5s.

### Spatial Alignment
Anatomical variability across participants is the core challenge for joint training. Each participant's data is projected onto a common cortical surface using FreeSurfer v7, and auditory regions (A1–A4) and late-language ROIs (e.g., bilateral inferior frontal gyrus, angular gyrus, anterior/posterior temporal lobe) are defined using the Glasser et al. parcellation atlas, yielding approximately 30K voxels per participant.

### Multi-brain-tuning Pipeline
1. **Data preparation**: Audio is segmented into 2s clips, with 8s of context prepended to compensate for hemodynamic delay, forming (10s audio, 1 fMRI TR) paired samples.
2. **Model architecture**: An average pooling layer and a **unified projection head** are appended on top of the speech model.
3. **Training strategy**: For a batch of stimuli $S$, the L₂ loss is computed and backpropagated for each participant $P_i$'s fMRI response sequentially; stimuli serve as anchors without requiring all participants to share identical stimulus sets.
4. **LoRA fine-tuning**: LoRA with rank=8 is applied (accounting for only 0.625% of total parameters); the feature extractor is frozen and only LoRA parameters and the projection head are updated.
5. **Training setup**: batch size 128, learning rate $1 \times 10^{-4}$ (10% warmup + linear decay), 30 epochs, ~6 hours on 2× NVIDIA A40 GPUs.

### Key Findings on Design Choices
- A unified projection head outperforms participant-specific projection heads and shared response modeling (SRM).
- Computing the loss independently per participant outperforms averaging fMRI responses or averaging losses, as it preserves individual-level signal.
- L₂ loss scales better with increasing data than Correlation loss or Cosine+L₂ loss.

### Baselines
- **Single-brain-tuned**: Fine-tuned on a single participant's data with the same architecture and settings.
- **LLM-tuned**: Fine-tuned using LLaMA2-7B representations as surrogate brain responses.
- **Stimulus-tuned**: Continued self-supervised pretraining on the stimulus audio.

## Key Experimental Results

### Brain Alignment Efficiency
- The Multi-brain-tuned model achieves the best brain alignment of the pretrained model using full encoding data with only **1/5 of the encoding data**, while Single-brain-tuned requires approximately 2/5.
- With full encoding data, brain alignment improves by up to **50%** over the pretrained baseline.
- This advantage holds consistently for both trained and unseen participants, and across both Wav2Vec2.0 and HuBERT model families.

### Generalization
- As fine-tuning data increases, Multi-brain-tuned performance on unseen participants continues to improve, while Single-brain-tuned saturates at approximately 6 hours.
- Brain map visualizations show that improvements are broadly distributed across frontal and parietal regions.
- Cross-dataset evaluation (Moth→Narratives): the improvement from Multi-brain-tuned approaches that of models trained directly on Narratives data.

### Downstream Performance
- On Phoneme Prediction and Phonetic Sentence Type Prediction, brain-tuned models **never underperform** the pretrained model, ruling out catastrophic forgetting.
- Multi-brain-tuned eventually matches LLM-tuned baseline performance as training data increases.

### Ablation Study
- Gains plateau beyond LoRA rank 8; full model fine-tuning does not outperform rank-8 LoRA.
- With sufficient data, L₂ loss substantially outperforms Correlation loss and Cosine+L₂ loss; however, at small data scales (≤6h), Correlation loss is marginally superior.

## Highlights & Insights
1. **Simple yet effective design**: A unified projection head combined with LoRA rank-8 achieves cross-participant generalization without participant-specific parameters.
2. **Bidirectional benefits**: Brain-tuning simultaneously improves brain alignment and downstream semantic task performance, demonstrating mutual value for neuroscience and AI.
3. **Practical 5× data reduction**: The method substantially lowers fMRI data requirements for new participants, facilitating group-level cognitive studies.
4. **Strong cross-dataset generalization**: Significant improvements are observed on the entirely distinct Narratives dataset.
5. **Systematic ablations and baselines**: Comprehensive evaluations cover training objectives, LoRA rank, and data scaling across multiple dimensions.

## Limitations & Future Work
1. The study focuses exclusively on language-related brain regions, without extending to non-language areas or task-specific functional regions.
2. Experiments are limited to English, constrained by the language coverage of large-scale public fMRI datasets.
3. The design of the training loss remains open to exploration; the superior performance of Correlation loss in low-data regimes suggests potential for a better hybrid loss.
4. Spatial alignment relies on FreeSurfer and the Glasser atlas, which may be inflexible for atypical brain structures.
5. Although a scaling trend with participant count is observed, the current validation is limited to 3 training participants and 5 evaluation participants.

## Related Work & Insights

| Method | Multi-participant | Generalizes to New Participants | Leverages Pretrained Model | Speech Domain |
|--------|------------------|---------------------------------|---------------------------|---------------|
| Brain-tuning (Moussa et al., 2025) | ✗ | Limited | ✓ | ✓ |
| BrainWavLM (Vattikonda et al., 2025) | ✗ | Limited | ✓ (LoRA) | ✓ |
| Hyperalignment (Haxby et al., 2020) | ✓ | ✓ | ✗ | ✗ |
| Brain decoding (Défossez et al., 2023) | ✓ | Limited | ✗ | ✓ |
| **Ours (Multi-brain-tuning)** | **✓** | **✓** | **✓ (LoRA)** | **✓** |

The key distinction of this work is that, building upon pretrained speech models, it achieves cross-participant generalization and improved brain alignment simultaneously through joint multi-participant training with a unified projection head, without introducing participant-specific parameters.

The unified projection head combined with anchor-stimulus training is transferable to other multi-subject/multi-modal alignment scenarios (e.g., multi-patient medical imaging, multi-user BCI). The sufficiency of LoRA rank-8 echoes general findings in parameter-efficient fine-tuning, suggesting that the learnable signal dimensions in brain activity are limited. The superiority of L₂ loss over correlation-based losses under sufficient data may generalize to other noisy signal regression tasks. The bidirectional benefit—where brain data improves model semantic representations—provides direct empirical evidence for the research direction of enhancing AI with cognitive signals.

## Rating
- Novelty: ⭐⭐⭐⭐ (First systematic exploration of multi-participant joint brain-tuning)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two model families, multiple baselines, thorough ablations, cross-dataset validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, coherent logic)
- Value: ⭐⭐⭐⭐ (Substantial contribution to the intersection of cognitive neuroscience and speech AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/llm_pretraining/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
