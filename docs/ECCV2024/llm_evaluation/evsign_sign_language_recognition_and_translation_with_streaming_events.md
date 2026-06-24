---
title: >-
  [Paper Note] EvSign: Sign Language Recognition and Translation with Streaming Events
description: >-
  [ECCV 2024][LLM Evaluation][Sign Language Recognition] This work constructs the first event-camera benchmark dataset, EvSign, for Continuous Sign Language Recognition (CSLR) and Sign Language Translation (SLT) tasks, and proposes an efficient sparse Transformer-based framework that achieves comparable or superior performance to SOTA RGB methods using only 0.34% FLOPs and 44.2% of the parameters.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Sign Language Recognition"
  - "Sign Language Translation"
  - "Event Camera"
  - "Sparse Convolution"
  - "Temporal Modeling"
date: 2026-05-08
content_hash: f474a5921eb8e443
---

# EvSign: Sign Language Recognition and Translation with Streaming Events

**Conference**: ECCV 2024  
**arXiv**: [2407.12593](https://arxiv.org/abs/2407.12593)  
**Code**: [Project Page](https://zhang-pengyu.github.io/EVSign)  
**Area**: LLM Evaluation  
**Keywords**: Sign Language Recognition, Sign Language Translation, Event Camera, Sparse Convolution, Temporal Modeling

## TL;DR

This work constructs the first event-camera benchmark dataset, EvSign, for Continuous Sign Language Recognition (CSLR) and Sign Language Translation (SLT) tasks, and proposes an efficient sparse Transformer-based framework that achieves comparable or superior performance to SOTA RGB methods using only 0.34% FLOPs and 44.2% of the parameters.

## Background & Motivation

Sign language is an essential communication tool for the hearing-impaired community, and video-based sign language recognition and translation are crucial research directions. Existing methods face the following challenges:

**Inherent limitations of RGB sensors**: Rapid hand movements cause motion blur, clothing textures introduce background interference, and information degrades under extreme lighting conditions.

**Natural advantages of event cameras**: Asynchronously capture luminance changes with extremely high temporal resolution (1MHz vs. RGB 120Hz), high dynamic range, and low latency—making them inherently suitable for capturing dynamic hand movements.

**Limitations of existing event-based sign language datasets**:
   - They only support Isolated Sign Language Recognition (ISLR), not continuous recognition and translation.
   - The vocabulary size is extremely small (SL-Animals-DVS has only 19 words, EvASL only 56 words).
   - The sensor resolution is low (128×128).

**Existing methods do not fully exploit event characteristics**: Directly applying networks designed for RGB (such as AlexNet or ResNet) to process event data, thereby ignoring the sparsity of event data.

## Method

### Overall Architecture

The contributions of this work are twofold:

**A. EvSign Dataset**: A large-scale Chinese sign language event-based benchmark.  
**B. Efficient Transformer Framework**: An SLR+SLT model specifically designed for the characteristics of event data.

Overall pipeline: Event stream $\rightarrow$ Voxel grid representation $\rightarrow$ Sparse convolutional backbone $\rightarrow$ Local token fusion $\rightarrow$ Gloss-aware temporal aggregation $\rightarrow$ Recognition head / Translation head.

### Key Designs

#### 1. EvSign Dataset

**Acquisition equipment**: iniVation DVXplorer-S-Duo stereo camera (event stream at 640×480, RGB at 480×320 @ 25FPS).

**Corpus source**: Everyday life scenarios (shopping, education, medical, tourism, social interaction). Sign language vocabulary is derived from the Chinese National Sign Language Dictionary and CSL-Daily.

**Data scale**:
- 6,773 event stream videos (Train 5,570 / Dev 553 / Test 650)
- 1,387 sign language glosses, 1,947 Chinese words
- 9 professional deaf volunteers
- Total duration of approximately 8.5 hours

**Annotation pipeline**: A two-step annotation process—annotators first identify sign language glosses in the RGB video, and the authors then verify the consistency to ensure each sign language corresponds to a unique gloss label.

Advantages over existing datasets: It is the first event-based dataset supporting CSLR and SLT, featuring a vocabulary size that vastly exceeds similar datasets (vs. SL-Animals with 19 words, EvASL with 56 words), with a higher resolution (640×480 vs. 128×128).

#### 2. Sparse Backbone (SConv)

Event data is naturally sparse (only encoding kinetic regions). Therefore, a ResNet18-structured sparse convolutional network is adopted to process voxel grid representations:
- Fully exploits data sparsity to significantly reduce computational costs.
- Preserves feature-level sparsity better than conventional convolutions, producing sharper boundaries.
- Outputs a set of visual tokens $\mathbf{O}^v \in \mathbb{R}^{P \times C}$.

#### 3. Local Token Fusion (LTF)

Prior to long-video temporal modeling, local motion information is aggregated and the number of tokens is reduced:
- A two-layer structure, where each layer consists of Window Multi-Head Self-Attention (W-MSA) + Max Pooling.
- Window size $I$, downsampling ratio $\gamma=4$.
- Formula: $\mathbf{O}^f = \text{MaxPool}(\text{W-MSA}(\tilde{\mathbf{O}}^v) + \tilde{\mathbf{O}}^v)$.
- Produces fused tokens $\mathbf{O}^f \in \mathbb{R}^{L \times C}$, where $L = P/\gamma$.

#### 4. Gloss-Aware Temporal Aggregation (GATA)

The core temporal modeling module decouples temporal information into **intra-gloss** and **inter-gloss** levels:

**Gloss-Aware Masked Attention (GAMA) — intra-gloss aggregation**:
- Uses cross-attention to aggregate information from visual tokens $\mathbf{O}^v$ to fused tokens $\mathbf{O}^f$.
- Key innovation: Introduces a gloss-aware mask $\mathbf{M} = \mathcal{N}(\rho) \odot \mathcal{N}(\delta)$.
    - $\rho$: Feature space similarity—tokens of the same category have highly correlated representations.
    - $\delta$: Spatiotemporal constraint—an RBF kernel measures pseudotimestamp distance to prevent erroneous aggregation between identical glosses at different positions.
- Formula: $\text{GAMA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}} \odot \mathbf{M})\mathbf{V}$.

**Inter-Gloss Temporal Aggregation (IGTA)**:
- Standard multi-head self-attention scales global motion coherence.
- Learns temporal dependencies across different glosses.

#### 5. Task Heads

- **Recognition Head (RH)**: Fully connected layer + softmax, outputs gloss sequence probabilities, supervised by CTC loss.
- **Translation Head (TH)**: Autoregressive Transformer decoder, translates gloss-aware tokens into spoken sentences, supervised by cross-entropy loss.

### Loss & Training

**SLR loss**: $\mathcal{L}_{SLR} = \lambda_{inter}\mathcal{L}_{inter} + \lambda_{final}\mathcal{L}_{final}$ (intermediate + final CTC losses, both weights are 1)

**SLT loss**: $\mathcal{L}_{SLT} = \mathcal{L}_{SLR} + \lambda_{ce}\mathcal{L}_{ce}$ (incorporates an additional cross-entropy loss for translation)

Training details: Adam optimizer, cosine annealing, initial learning rate of 3e-5, batch size of 2, 200 epochs on a single RTX 3090.

## Key Experimental Results

### Main Results — Sign Language Recognition (WER↓)

| Method | Modality | PHOENIX14T Dev | PHOENIX14T Test | EvSign Dev | EvSign Test | FLOPs | Params |
|------|------|---------------|----------------|------------|------------|-------|--------|
| VAC | RGB | 20.17 | 21.60 | 32.08 | 30.43 | 228.87G | 31.64M |
| CorrNet | RGB | 18.90 | 20.50 | 32.37 | 32.04 | 234.59G | 32.04M |
| CorrNet | EV | 24.57 | 24.55 | 29.98 | 29.95 | 244.63G | 32.05M |
| **Ours** | **EV** | 23.89 | 24.03 | **29.19** | **28.69** | **0.84G** | **14.19M** |

On EvSign, event-based methods comprehensively outperform RGB-based methods; the proposed method achieves the lowest WER using only 0.34% of the FLOPs.

### SLT Results (EvSign Dataset, Selected Metrics)

| Method | Modality | Dev ROUGE-L↑ | Test ROUGE-L↑ | Dev BLEU-4↑ | Test BLEU-4↑ |
|------|------|-------------|-------------|------------|------------|
| SLT | RGB | - | - | - | - |
| CorrNet+TH | EV | - | - | - | - |
| **Ours** | **EV** | Best | Best | Competitive | Best |

On synthesized PHOENIX14T, the SLT results of the event-based method achieve ROUGE gains of 1.06%/0.89% (dev/test) over SLT (RGB).

### Ablation Study

The design effectiveness is validated by removing various modules:
- The sparse backbone is the key to computational efficiency (reducing computation by two orders of magnitude).
- LTF successfully reduces token sequence length while retaining local motion information.
- Both the feature similarity mask and temporal distance constraint in GAMA contribute to performance.
- Inter-gloss self-attention complements global temporal modeling.

### Key Findings

1. **Event cameras comprehensively outperform RGB on real-world data**: All methods utilizing event streams achieve a lower WER than RGB on the EvSign dataset (e.g., CorrNet: 29.95% vs. 32.04%).
2. **Event advantages are less pronounced on synthesized data**: Because of the poor quality of original PHOENIX14T videos (blur, low frame rate), synthesized event streams are of limited quality.
3. **Extreme efficiency**: The proposed method requires only 0.84G FLOPs per video (vs. CorrNet's 244.63G), a 290x efficiency improvement.
4. **Parameter savings**: Only 14.19M parameters (44.2% of CorrNet), making it highly suitable for edge deployment.
5. **Privacy advantages of event cameras**: They only capture motion details without recording static facial features, naturally preserving user privacy.

## Highlights & Insights

1. **First CSLR+SLT Event Benchmark**: Fills the gap in event-based sign language research from ISLR to continuous recognition and translation.
2. **Dual Exploitation of Sparsity**: Event data is naturally sparse $\rightarrow$ processed via sparse convolutions $\rightarrow$ downsampled via local fusion $\rightarrow$ leading to extremely low computational complexity.
3. **Elegant Design of Gloss-Aware Mask**: Simultaneously accounts for feature similarity (which tokens belong to the same gloss) and temporal distance (preventing erroneous long-range aggregation of identical glosses), which is more reasonable than simple global attention.
4. **Practical Value**: The high temporal resolution, low latency, and built-in privacy protection of event cameras make them highly suitable for wearable sign language translation devices.

## Limitations & Future Work

- EvSign only contains Chinese Sign Language, and generalization to other sign language systems requires further validation.
- The scale of 9 signers is relatively small, with limited coverage of individual variability.
- The method still relies on gloss as an intermediate representation (Sign2Gloss2Text), leaving the potential of end-to-end gloss-free schemes on event data unexplored.
- Voxel grid representation partially discards the asynchronous nature of events; future work could investigate directly processing raw event streams.
- The quality of synthesized events is constrained by the quality of source RGB videos, necessitating more authentic, high-quality event data.

## Related Work & Insights

- **CorrNet**: Current SOTA for RGB sign language recognition, constructing discriminative spatial representations through a correlation network.
- **VAC**: Visual alignment constraint method; this work extends its translation head to the event modality.
- **SL-Animals-DVS / EvASL**: Prior event-based sign language datasets, which are small in scale, limited to ISLR, and low in resolution.
- **PixelCNN/VQ-VAE event sampling**: Shi et al. designed an event sampling strategy but still used CNNs for processing, failing to exploit sparsity.
- **SEN/TLP**: Recent RGB sign language recognition methods, which are comprehensively outperformed by this work in the event domain.

## Rating

- Novelty: ⭐⭐⭐⭐ — First CSLR+SLT event-based benchmark + tailored framework design.
- Technical Depth: ⭐⭐⭐⭐ — Coherent design of the sparse backbone and GATA module.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Extensive experiments on both synthetic and real-world data, dual tasks (SLR+SLT), and detailed efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and intuitive diagrams, though SLT experimental details could be slightly more comprehensive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SwiLTra-Bench: The Swiss Legal Translation Benchmark](../../ACL2025/llm_evaluation/swiltra-bench_the_swiss_legal_translation_benchmark.md)
- [\[ICCV 2025\] SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](../../ICCV2025/llm_evaluation/svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ICCV 2025\] StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](../../ICCV2025/llm_evaluation/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](../../ACL2026/llm_evaluation/beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)

</div>

<!-- RELATED:END -->
