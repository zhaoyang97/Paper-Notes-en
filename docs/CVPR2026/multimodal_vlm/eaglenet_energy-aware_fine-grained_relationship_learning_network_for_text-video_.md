---
title: >-
  [Paper Note] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval
description: >-
  [CVPR 2026][Multimodal VLM][Text-video retrieval] EagleNet constructs a text-frame relational graph and employs a relational graph attention network to learn fine-grained text-frame and frame-frame relationships, generating enhanced text embeddings enriched with video contextual information. An energy-based matching mechanism is further introduced to capture the distribution of ground-truth text-video pairs. The method achieves state-of-the-art performance on four benchmark datasets.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Text-video retrieval
  - graph attention network
  - energy-based model
  - fine-grained relationship learning
  - cross-modal alignment
date: 2026-05-08
content_hash: 12399bcf405875bd
---

# EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.25267](https://arxiv.org/abs/2603.25267)
**Code**: [https://github.com/draym28/EagleNet](https://github.com/draym28/EagleNet)
**Area**: Multimodal VLM / Video Understanding
**Keywords**: Text-video retrieval, graph attention network, energy-based model, fine-grained relationship learning, cross-modal alignment

## TL;DR
EagleNet constructs a text-frame relational graph and employs a relational graph attention network to learn fine-grained text-frame and frame-frame relationships, generating enhanced text embeddings enriched with video contextual information. An energy-based matching mechanism is further introduced to capture the distribution of ground-truth text-video pairs. The method achieves state-of-the-art performance on four benchmark datasets.

## Background & Motivation

1. **Background**: Mainstream methods in text-video retrieval (TVR) are predominantly built upon CLIP pre-trained models, focusing on learning high-quality video representations or improving cross-modal alignment strategies. A minority of recent works have begun to address the problem of insufficient text expressiveness, as short video descriptions often fail to fully reflect the rich semantics of the corresponding video.

2. **Limitations of Prior Work**:
    - Methods such as TMASS and TV-ProxyNet attempt to expand textual semantics via sampling or proxy-based strategies, but only consider text-frame or text-video interactions.
    - Frame-frame relations within the video are completely ignored.
    - Consequently, the augmented text embeddings fail to capture contextual information across frames, resulting in a representation gap between text and video.

3. **Key Challenge**: Textual semantic expansion requires simultaneous understanding of "what each frame conveys" (text-frame interaction) and "how frames relate to one another" (frame-frame relations). Existing methods address only the former while neglecting the latter, despite the fact that frame-frame relations are critical for capturing global and temporal video semantics.

4. **Goal**:
    - How to generate enhanced text embeddings that jointly incorporate text-frame interactions and frame contextual information?
    - How to improve cross-modal matching at a fine-grained level to more accurately capture the distribution of ground-truth text-video pairs?

5. **Key Insight**: Text candidates and video frames are treated as graph nodes, with three types of edges modeled (text-text, text-frame, frame-frame). A relational graph attention network is used to learn all relational weights, which are then aggregated into enhanced text embeddings.

6. **Core Idea**: A text-frame relational graph is constructed to learn fine-grained text-frame and frame-frame interactions, and an energy-based matching mechanism is employed to capture the distribution of ground-truth pairs, thereby producing video-context-aware enhanced text embeddings.

## Method

### Overall Architecture
EagleNet adopts CLIP as its backbone and comprises two core modules: (1) **Fine-Grained Relationship Learning (FRL)**, which constructs a text-frame relational graph and applies a relational graph attention network to learn fine-grained relationships, producing context-aware enhanced text embeddings; and (2) **Energy-Aware Matching (EAM)**, which employs an energy-based model to capture fine-grained text-frame interaction energies, facilitating accurate modeling of the ground-truth text-video pair distribution. A sigmoid loss replaces the conventional softmax contrastive loss for more stable cross-modal alignment.

### Key Designs

1. **Fine-Grained Relationship Learning (FRL)**:
    - **Function**: Generate enhanced text embeddings that incorporate frame contextual information.
    - **Mechanism**: A stochastic text modeling strategy is first used to sample $S=20$ text candidates $\{\mathbf{t}_i^{sto}\}$. These, together with the original text embedding and $M$ frame embeddings (with temporal positional encodings), form the node matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, where $n = 1 + S + M$. A Relational Graph Attention Network (RGAT) then learns attention weights for three relation types (text-text, frame-frame, text-frame). For each relation type $r$ and node pair $(i,j)$, RGAT computes edge weights as $e_{ij}^{r,h} = \psi^r([\mathbf{W}^{r,h}\mathbf{h}_i \| \mathbf{W}^{r,h}\mathbf{h}_j])$, followed by LeakyReLU and softmax to obtain attention scores. Text-frame edge weights are extracted and averaged to produce weighted aggregations over text nodes, yielding the enhanced text embedding $\mathbf{t}^{gen} = \sum_i w_i \mathbf{X}_i$.
    - **Design Motivation**: Unlike methods such as TMASS that consider only text-frame interactions, FRL explicitly incorporates frame-frame relations, enabling the text embedding to capture inter-frame contextual dependencies and effectively suppressing redundant information and noise.

2. **Energy-Aware Matching (EAM)**:
    - **Function**: Enhance text-frame relationship learning at a fine-grained level and precisely model the distribution of ground-truth text-video pairs.
    - **Mechanism**: A Boltzmann distribution $p_\theta(\mathbf{t}, \mathbf{F}) = \frac{\exp(-E_\theta(\mathbf{t}, \mathbf{F}))}{Z_\theta}$ is used to model the joint distribution of text-video pairs. The text-video energy is defined as the average of text-frame energies: $E_\theta(\mathbf{t}, \mathbf{F}) = \frac{1}{M}\sum_i^M E_\theta(\mathbf{t}, \mathbf{f}_i)$, fully leveraging fine-grained interactions. The energy function can be instantiated as negative cosine similarity, bilinear scoring, or an MLP. The model is trained via negative log-likelihood loss, with $K=20$ steps of MCMC Langevin dynamics used to generate negative text-video pairs. EAM is used only during training and introduces no additional inference cost.
    - **Design Motivation**: Global contrastive loss aligns text and video only at the holistic level. EAM precisely captures detailed text-frame interaction patterns at a fine-grained level through the energy-based formulation.

3. **Sigmoid Loss in Place of Softmax Loss**:
    - **Function**: Provide more effective cross-modal alignment and more stable training.
    - **Mechanism**: $\mathcal{L}_{sig} = -\frac{1}{B}\sum_i\sum_j \log\frac{1}{1 + e^{\mathbb{I}_{ij}(\tau \cdot s(\mathbf{t}_i, \mathbf{v}_j) + b)}}$, where $\mathbb{I}_{ij}$ is the positive/negative pair indicator and $\tau$, $b$ are learnable parameters.
    - **Design Motivation**: Softmax loss normalizes across both dimensions of the batch similarity matrix and is sensitive to the choice of negatives and batch size. Sigmoid loss treats each pair independently, making it naturally suited to the multi-match scenario in TVR where a single text query may semantically correspond to multiple videos.

### Loss & Training
The overall training objective is: $\mathcal{L}_{total} = \mathcal{L}_{sig}(\mathbf{t}^{gen}, \mathbf{v}) + \lambda_{sup}\mathcal{L}_{sig}(\mathbf{t}^{sup}, \mathbf{v}) + \lambda_{eam}\mathcal{L}_{eam}$

where $\lambda_{sup} = 0.8$ and $\lambda_{eam} = 1.0$. The model is initialized with CLIP ViT-B/32 or ViT-B/16. The learning rate is $10^{-7}$ for CLIP modules and $10^{-4}$ for non-CLIP modules, with a batch size of 64 and training for 5 epochs.

## Key Experimental Results

### Main Results — MSRVTT (ViT-B/16)

| Method | T2V R@1↑ | T2V R@5↑ | T2V R@10↑ | V2T R@1↑ | Rsum↑ |
|--------|----------|----------|-----------|----------|-------|
| CLIP4Clip | 45.2 | 72.2 | 81.4 | 42.9 | 393.2 |
| XPool | 49.2 | 73.9 | 82.6 | 48.0 | 411.5 |
| GLSCL | 49.9 | 76.3 | 84.1 | 48.3 | 419.0 |
| Video-ColBERT | 50.0 | 76.3 | 84.3 | 47.9 | 417.8 |
| **EagleNet** | **51.0** | **76.2** | **85.6** | **49.2** | **425.7** |

### Main Results — DiDeMo & MSVD (ViT-B/16)

| Method | DiDeMo R@1↑ | MSVD R@1↑ | VATEX R@1↑ | Rsum↑ |
|--------|-------------|-----------|------------|-------|
| TV-ProxyNet | 47.9 | 49.7 | 64.0 | 676.6 |
| TempMe | 50.2 | - | - | - |
| **EagleNet** | **51.5** | **50.9** | 63.6 | **687.7** |

### Ablation Study

| Configuration | MSRVTT R@1↑ | DiDeMo R@1↑ | Avg. R@1↑ |
|---------------|-------------|-------------|-----------|
| Baseline (TMASS) | 48.5 | 42.1 | 45.3 |
| + FRL | 48.8 | 47.9 | 48.4 |
| + EAM | 49.0 | 43.4 | 46.2 |
| + FRL + EAM | 50.5 | 49.2 | 49.9 |
| + Sigmoid Loss | 47.8 | 43.9 | 45.9 |
| + FRL + EAM + Sigmoid (Full) | **51.0** | **51.5** | **51.3** |

### Key Findings
- **FRL yields the largest gains on DiDeMo**: Adding FRL alone improves DiDeMo R@1 from 42.1 to 47.9 (+5.8), demonstrating that inter-frame relation modeling is particularly important for longer videos.
- **Strong complementarity among the three components**: Each component alone provides limited improvement, but their combination yields +2.5% on MSRVTT R@1 and +9.4% on DiDeMo R@1.
- **Energy function selection**: Bilinear and MLP energy functions achieve comparable and superior performance to cosine similarity, indicating that learnable parameters facilitate more accurate text-frame energy modeling.
- **Average pooling for frame energy aggregation is optimal**: It outperforms max pooling, min pooling, and direct video-level energy $E_\theta(\mathbf{t}, \mathbf{v})$.

## Highlights & Insights
- **Frame-frame relation modeling for textual semantic expansion**: This is an insightful observation — augmenting text semantics should account not only for "text-to-frame correspondences" but also for "inter-frame contextual relations," the latter enabling the text embedding to capture global and temporal video semantics.
- **First application of EBMs to TVR**: Energy-based models are naturally suited for fine-grained matching, with MCMC sampling used to generate negative pairs during training. As EAM is training-only, it introduces no additional inference overhead.
- **Correction of data leakage in the TMASS codebase**: This rigorous experimental practice is commendable; multiple baseline methods were re-implemented to ensure fair comparison.

## Limitations & Future Work
- The RGAT design is relatively straightforward; more advanced graph Transformer architectures could be explored.
- The text candidate sampling strategy (random Gaussian sampling) is coarse; semantics-guided directed sampling warrants investigation.
- The $K=20$ MCMC sampling steps affect training efficiency; more computationally efficient sampling strategies should be explored.
- Evaluation is primarily conducted on short-video datasets; performance on long-video scenarios remains to be verified.

## Related Work & Insights
- **vs. TMASS**: TMASS determines the radius for stochastic text sampling solely based on text-video similarity, ignoring inter-frame relations. EagleNet explicitly models frame-frame relations through a relational graph construction.
- **vs. TV-ProxyNet**: TV-ProxyNet transforms text into video-aware proxies via video-conditioned directors but similarly neglects inter-frame context. EagleNet jointly models text-frame and frame-frame relations during relational learning.
- **vs. Video-ColBERT**: Both methods employ sigmoid loss, but EagleNet additionally introduces FRL and EAM for deeper structural relationship learning and fine-grained energy-based matching.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Combining relational graph learning and energy-based models in the TVR setting is a novel and well-motivated contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation spans four datasets, two CLIP backbones, comprehensive ablations, and analysis of multiple design variants.
- **Writing Quality**: ⭐⭐⭐⭐ — Methodology is clearly described, though the density of equations requires careful reading.
- **Value**: ⭐⭐⭐⭐ — Achieves consistent state-of-the-art improvements in the highly competitive TVR field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[AAAI 2026\] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning](../../AAAI2026/multimodal_vlm/heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)

</div>

<!-- RELATED:END -->
