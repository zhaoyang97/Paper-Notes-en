---
title: >-
  [Paper Note] DisTime: Distribution-based Time Representation for Video Large Language Models
description: >-
  [ICCV 2025][Video Understanding][Video-LLM] This paper proposes DisTime, a framework that enables continuous time representation in Video-LLMs via a single learnable time token and a distribution-based time decoder. Comp…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video-LLM"
  - "time representation"
  - "distribution-based decoding"
  - "temporal grounding"
  - "time-sensitive datasets"
date: 2026-05-08
content_hash: 582d2f7db396732c
---

# DisTime: Distribution-based Time Representation for Video Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2505.24329](https://arxiv.org/abs/2505.24329)
**Code**: [GitHub](https://github.com/josephzpng/DisTime)
**Area**: Video Understanding / Temporal Grounding
**Keywords**: Video-LLM, time representation, distribution-based decoding, temporal grounding, time-sensitive datasets

## TL;DR

This paper proposes DisTime, a framework that enables continuous time representation in Video-LLMs via a single learnable time token and a distribution-based time decoder. Complemented by the large-scale automatically annotated dataset InternVid-TG (1.25M events), DisTime achieves state-of-the-art performance on three categories of time-sensitive tasks: moment retrieval, dense video captioning, and grounded VQA.

## Background & Motivation

Current Video-LLMs perform well on general video understanding but exhibit fundamental limitations in **precise temporal grounding**. Existing time representation schemes each carry inherent drawbacks:

**Text-modal discretization** (e.g., VTimeLLM, TimeMarker): Time is expressed as text-form numerals, but time and numeric values share a decision boundary, increasing classification confusion.

**Multi-token discretization** (e.g., Momentor, VTG-LLM): A large set of dedicated time tokens is introduced, but the long-tail distribution of training data leaves some tokens under-trained, and temporal continuity is not modeled.

**Dedicated temporal heads** (e.g., InternVideo2.5): Time-aware modules with substantial parameter overhead (e.g., CG-DETR) require secondary visual input and incur high computational cost.

Furthermore, existing time-sensitive datasets suffer from **temporal granularity constraints**—VTimeLLM relies on shot boundaries, InternVid-MR uses fixed 2-second windows, and Momentor depends on shot consistency—all of which are too coarse to accurately capture event temporal boundaries.

## Method

### Overall Architecture

DisTime consists of five core components: a visual encoder with projector, a text encoder, an LLM, a time decoder $\Phi_{\text{time-dec}}$, and a time encoder $\Phi_{\text{time-enc}}$. Sampled video frames are encoded visually and interleaved with time tokens corresponding to their timestamps, then fed into the LLM together with user instructions. When the LLM generates a `<TIME_STAMP>` token, its hidden state is passed to the time decoder to produce a continuous timestamp.

### Key Designs

1. **Distribution-based Time Token**:
   A single learnable token `<TIME_STAMP>` represents continuous time, explicitly separated from text numeral tokens. The core innovation is that **absolute time values are not directly regressed**; instead, the token is first transformed into a time distribution, and the timestamp is obtained via weighted summation.
   - The normalized time axis $[0,1]$ is divided into $reg_{max}+1$ discrete anchors.
   - An MLP followed by softmax maps the hidden state of `<TIME_STAMP>` to a distribution vector $\mathbf{e} \in \mathbb{R}^{2 \times (reg_{max}+1)}$.
   - The continuous timestamp is obtained by anchor-weighted summation: $st = \sum_{i=0}^{reg_{max}} \mathbf{e}_{st}^{(i)} \cdot a_i$, where $a_i = i/reg_{max}$.

   The advantage of distribution-based decoding lies in **modeling the inherent ambiguity of event boundaries**—for instance, does the onset of "a person drinking water" include the act of picking up the cup? Such annotation ambiguity makes direct regression prone to precision errors.

2. **Time Encoder**:
   The inverse of the decoder, encoding continuous timestamps back into time tokens processable by the LLM. A timestamp is first projected into a Gaussian-regularized distribution $p_{st} \sim \mathcal{N}(st, \delta^2)$, discretized, and then mapped to the LLM token space via an MLP:
   $$\tau = \text{MLP}([\hat{\mathbf{e}}_{st}, \hat{\mathbf{e}}_{et}])$$
   The encoder is extremely lightweight, accounting for only 0.36% of the parameters of InternVL2.5-1B.

3. **Iterative Time Refinement**:
   During autoregressive generation, upon encountering a `<TIME_STAMP>` token, the hidden state is decoded into a timestamp → re-encoded into a time token → substituted back for subsequent steps. This re-encoding converts the ambiguous time distribution into a standardized Gaussian representation, ensuring distributional alignment across time tokens and enhancing the LLM's temporal consistency.

4. **InternVid-TG Dataset Construction**:
   A four-step annotation paradigm is proposed:
   - **Event Extraction**: GPT-4o identifies video events from 1fps image sequences (~7 events per video).
   - **Event Localization**: Three specialized models (UniMD, Mr.Blip, TFVTG) independently localize event boundaries.
   - **Score-based Integration**: InternVideo2 computes video-text cosine similarity, selecting the highest-scoring model's localization result for each event.
   - **Instruction Generation**: Five dialogue templates convert annotations into single-turn training dialogues.

   The resulting dataset covers 179K videos with 1.25M event annotations, exceeding ActivityNet-Captions by 55×.

### Loss & Training

Three loss functions are jointly optimized with equal weight of 1:
- $\mathcal{L}_{ntp}$: Standard next-token prediction loss.
- $\mathcal{L}_{reg}$: 1D-IoU regression loss, directly optimizing temporal interval overlap.
- $\mathcal{L}_{dist}$: Distribution Focal Loss for learning the time distribution.

Training strategy: the visual backbone and intermediate layers are frozen; LoRA is applied to fine-tune the LLM; token embeddings, the LLM head, and the time encoder/decoder are trained in full.

## Key Experimental Results

### Main Results

| Model | Scale | Charades-STA R@1(IoU=0.3) | R@1(IoU=0.5) | ANet R@1(IoU=0.3) | R@1(IoU=0.5) |
|-------|-------|--------------------------|-------------|-------------------|-------------|
| VTimeLLM | 13B | 55.3 | 34.3 | 44.8 | 29.5 |
| TimeMarker | 8B | 73.5 | 51.9 | 67.4 | 50.7 |
| InternVL2.5 (baseline) | 1B | 3.1 | 1.5 | 5.3 | 2.9 |
| **DisTime-InternVL** | **1B** | **78.1** | **56.3** | **67.1** | **45.4** |
| **DisTime-InternVL** | **8B** | **81.0** | **60.3** | **72.9** | **53.2** |
| Mr.BLIP (specialist) | 3B | — | 69.3 | — | 53.9 |

### Ablation Study

| Direct | Dist. | Re-Enc. | Charades R@1(0.5) | R@1(0.7) | YouCook2 F1 |
|--------|-------|---------|-------------------|---------|-------------|
| ✓ | | | 51.9 | 24.9 | 2.2 |
| | ✓ | | 53.5 | 26.7 | 16.3 |
| | ✓ | ✓ | **56.3** | **29.7** | **20.5** |

| Training Data | Charades R@1(0.3) | QVHighlights R@1(0.3) |
|--------------|-------------------|----------------------|
| Baseline | 77.4 | 38.7 |
| + VTimeLLM data | 76.2 | 51.0 |
| + Momentor data | 76.6 | 39.7 |
| + InternVid-TG | **78.1** | **54.1** |

### Key Findings

- **From 3.1% to 78.1%**: DisTime improves InternVL2.5-1B's R@1(IoU=0.3) on Charades-STA by 25×, demonstrating the decisive influence of time representation design on LLM temporal awareness.
- **Distribution-based decoding substantially outperforms direct regression**: YouCook2 F1 improves from 2.2% to 16.3%, with iterative time re-encoding further pushing it to 20.5%.
- **InternVid-TG data quality surpasses the larger Momentor dataset**: Momentor contains 1.46M events yet causes a performance drop on Charades after training, indicating that annotation quality matters more than scale.
- **Zero-shot results on Charades-STA surpass all specialist models and Video-LLMs** (R@1(0.3) = 81.0%).
- The method is plug-and-play, applicable to both InternVL2.5 and LLaVA-OneVision architectures.

## Highlights & Insights

- **Effectiveness of minimalist design**: A single additional token combined with an extremely lightweight MLP decoder (0.36% of total parameters) is sufficient to endow an LLM with precise temporal awareness.
- **Distribution vs. point estimation**: Event boundaries are inherently ambiguous; modeling them with distributions is physically more meaningful than point regression—an insight worth generalizing across domains.
- **Novel data annotation paradigm**: LLM-based event extraction + specialist model localization + similarity-score-based integration assigns each step to the most capable tool, yielding more reliable annotations than end-to-end approaches.

## Limitations & Future Work

- InternVL2.5 samples only 16 frames, which may be insufficient for tasks requiring fine-grained temporal understanding such as ANet-Captions.
- Autoregressive generation of time tokens increases inference latency.
- The annotation quality of InternVid-TG remains bounded by the capabilities of the three grounding models.
- The current design only supports input token sequences that preserve temporal alignment and is incompatible with models that perform global temporal aggregation (e.g., LinVT).

## Related Work & Insights

- Distribution Focal Loss (DFL) was originally proposed for bounding box regression in object detection; this work successfully transfers it to temporal grounding, representing a productive cross-domain adaptation.
- Compared to TimeMarker: the latter incorporates frame timestamps into multimodal inputs but still represents time using text tokens, whereas DisTime fully decouples the representation spaces for time and numerals.
- Insight: **The design of time representation** may be the primary bottleneck for temporal understanding in Video-LLMs, rather than model scale or training data volume.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The distribution-based time representation is elegant and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers MR, DVC, and Grounded-VQA tasks alongside general QA, with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear method descriptions.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical weakness of Video-LLMs; the dataset contribution is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

</div>

<!-- RELATED:END -->
