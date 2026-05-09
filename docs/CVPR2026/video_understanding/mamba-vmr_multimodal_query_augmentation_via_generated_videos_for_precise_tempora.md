---
title: >-
  [Paper Note] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding
description: >-
  [CVPR 2026][Video Understanding][Video Moment Retrieval] A two-stage video moment retrieval framework is proposed: the first stage employs LLM-guided caption matching and generates auxiliary short videos as temporal priors; the second stage uses a multimodal-controlled Mamba network to efficiently fuse generated priors with long sequences, achieving state-of-the-art performance on TVR (R@1/IoU=0.5: 45.20%) while reducing computational overhead.
tags:
  - CVPR 2026
  - Video Understanding
  - Video Moment Retrieval
  - Multimodal Query Augmentation
  - Generated Video Prior
  - Mamba
  - Temporal Grounding
date: 2026-05-08
content_hash: 2e4380b90430fc3b
---

# Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding

**Conference**: CVPR 2026
**arXiv**: [2603.22121](https://arxiv.org/abs/2603.22121)
**Code**: [https://github.com/YunzhuoSun/Manba-VMR](https://github.com/YunzhuoSun/Manba-VMR)
**Area**: Video Understanding / Multimodal VLM
**Keywords**: Video Moment Retrieval, Multimodal Query Augmentation, Generated Video Prior, Mamba, Temporal Grounding

## TL;DR
A two-stage video moment retrieval framework is proposed: the first stage employs LLM-guided caption matching and generates auxiliary short videos as temporal priors; the second stage uses a multimodal-controlled Mamba network to efficiently fuse generated priors with long sequences, achieving state-of-the-art performance on TVR (R@1/IoU=0.5: 45.20%) while reducing computational overhead.

## Background & Motivation

1. **State of the Field**: Video Moment Retrieval (VMR) aims to localize temporal segments in untrimmed videos that semantically correspond to a text query. Existing methods primarily rely on natural language queries (NLQ) or static image augmentation (e.g., ICQ using DALL-E-generated images), and employ Transformer architectures for cross-modal fusion.

2. **Limitations of Prior Work**: Pure text queries tend to introduce temporal ambiguity when handling complex queries with multiple verbs. For example, "Adams walks into the room and hands coffee to Park" requires understanding the sequential relationship between "walking" and "handing," yet text descriptions lack dynamic cues. Static image augmentation improves semantic expressiveness but fails to convey dynamic motion information—generated images omit the temporal progression of actions (e.g., entering the room → approaching → reaching out to hand coffee), leading to grounding errors.

3. **Root Cause**: Multi-verb queries require explicit temporal dynamic cues (motion cues), which neither text nor static images can provide. Furthermore, incorporating generated videos extends the input sequence, making the quadratic complexity of Transformers a critical bottleneck.

4. **Paper Goals**: (a) How to generate auxiliary information with rich temporal dynamics for queries? (b) How to efficiently fuse generated priors with long video sequences?

5. **Starting Point**: A text-to-video diffusion model (CogVideoX) is leveraged to generate short videos as temporal priors, capturing implicit motion information; Mamba (SSM) replaces Transformers to enable linear-time complexity for long-sequence modeling.

6. **Core Idea**: Dynamic videos rather than static images are generated as temporal priors for query augmentation, and a multimodal-controlled Mamba network efficiently fuses text, generated videos, and target videos for precise temporal grounding.

## Method

### Overall Architecture
The framework consists of two stages. **Stage 1**: LLM-guided caption matching → query decomposition into verb-centric sub-events → short video generation by fusing query and captions. **Stage 2**: Text embeddings, generated video embeddings, and target video embeddings are fed into a multimodal-controlled Mamba network → contextual features → linear head predicts start/end timestamps → NMS refinement.

### Key Designs

1. **LLM-Guided Caption Matching and Query Processing**:

    - **Function**: Decomposes complex queries into verb-centric sub-events and matches relevant textual cues from video captions.
    - **Mechanism**: LLaMA-3.1 decomposes a query into sub-events by verb, e.g., "walks into the room and hands coffee" → "opens the door and walks into the room," "approaches Park while holding coffee," "reaches out to hand the coffee." Implicit intermediate actions (e.g., "opening the door") are also supplemented. Each caption sentence is then evaluated for relevance to each sub-query: $r_j = \max_i \sigma(\text{LLM}(q_i, s_j))$, and the top-$k$ captions above threshold $\theta$ form the refined subset $S'$.
    - **Design Motivation**: Query decomposition breaks high-level abstract descriptions into fine-grained action sequences, while caption matching introduces dialogue cues to compensate for query ambiguity. Together they provide rich temporal context for subsequent video generation.

2. **Temporal Prior Generation**:

    - **Function**: Generates short videos by fusing queries and captions to capture implicit dynamic motion information.
    - **Mechanism**: An LLM merges the query $q$ and matched captions $S'$ into a coherent narrative prompt $p = q \oplus \text{LLM}(\{s\}_{s \in S'})$, which is fed into CogVideoX to generate a 6-second auxiliary short video $v_g \sim \mathcal{D}(p, \theta)$. The generated video is substantially shorter than the target video ($L_g \ll L_o$), providing a dynamic "preview" of the target event.
    - **Design Motivation**: Static images (e.g., those generated by DALL-E in ICQ) supplement semantics but cannot express motion sequences. Generated videos inherently contain temporal dynamics, addressing the fundamental limitation of static augmentation for temporal grounding. Experiments confirm that CogVideoX outperforms Stable Video Diffusion due to its superior motion fidelity.

3. **Multimodal-Controlled Mamba Network**:

    - **Function**: Efficiently fuses multimodal information from text, generated video priors, and target videos to produce contextual features for timestamp prediction.
    - **Mechanism**: Target video embeddings $e_o$ augmented with GCN-extracted relational embeddings $r_o$ form the input sequence $x = e_o + r_o$. The core is a bidirectional SSM with state transition $h_t = Ah_{t-1} + Bx_t,\ y_t = Ch_t$. The key innovation is **video-guided gating**: $g_t = \sigma(W_g[e_q;\ \text{pooled}(e_g)]_t)$, which dynamically modulates state transitions as $h_t' = g_t \odot (Ah_{t-1} + Bx_t)$, where $e_q$ denotes text embeddings and $\text{pooled}(e_g)$ denotes mean-pooled embeddings of the generated video. The gating mechanism focuses Mamba on video segments aligned with the motion prior while filtering out irrelevant noise.
    - **Design Motivation**: Transformer's quadratic complexity causes memory explosion on long sequences (OOM beyond length 700), whereas Mamba's linear complexity is well-suited for untrimmed long videos. The video-guided gate injects motion information from the generated prior into state propagation, which is more efficient than naive concatenation.

### Loss & Training
The total loss comprises three components: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{bound}} + \lambda_2 \mathcal{L}_{\text{rel}} + \lambda_3 \mathcal{L}_{\text{cont}}$. The boundary loss $\mathcal{L}_{\text{bound}}$ is BCE over start/end positions; the relevance loss $\mathcal{L}_{\text{rel}}$ is clip-level BCE scoring; the contrastive loss $\mathcal{L}_{\text{cont}}$ uses InfoNCE to maximize similarity between generated videos and positive clips. Weights are set to $\lambda_1=1,\ \lambda_2=0.5,\ \lambda_3=0.1$. Training uses the AdamW optimizer for 20 epochs on 4× RTX 4090 GPUs.

## Key Experimental Results

### Main Results
Comparison on TVR dataset:

| Method | R@1/IoU=0.5 | R@10/IoU=0.5 | R@1/IoU=0.7 | R@10/IoU=0.7 |
|--------|-------------|--------------|-------------|--------------|
| HERO | 33.86 | 58.69 | 10.15 | 34.00 |
| SgLFT | 42.51 | 72.41 | 21.03 | 54.62 |
| ICQ | 44.13 | 75.27 | 24.08 | 59.23 |
| **Ours** | **45.20** | **76.09** | **25.10** | **60.87** |

Improvements are also observed on ActivityNet: R@100/IoU=0.5 increases from ICQ's 81.20 to 83.59.

### Ablation Study

| Configuration | R1/0.5 | R1/0.7 | Note |
|---------------|--------|--------|------|
| Full model | 45.20 | 25.10 | Complete method |
| w/o LLM module | 40.15 | 21.45 | Remove caption matching + query decomposition, −5.05 |
| w/o video prior | 38.76 | 20.08 | Replace with static images, −6.44 |
| w/o video gating | 41.23 | 22.34 | Standard SSM without gating, −3.97 |
| Transformer replacing Mamba | 37.89 | 19.56 | −7.31, OOM on long sequences |
| w/o contrastive loss | 41.08 | 21.56 | Degraded multimodal fusion |

### Key Findings
- Temporal prior generation contributes the most (−6.44 when removed), demonstrating that dynamic motion cues are critical for temporal grounding, far surpassing static image augmentation.
- Multi-verb query analysis: on queries with 4+ verbs, the proposed method achieves 35.9% vs. ICQ's 16.8% vs. SgLFT's 16.3%, an improvement of ~19%, confirming that generated videos are especially effective for modeling complex temporal relationships.
- Mamba exhibits linear memory growth vs. Transformer OOM at sequence length 700, validating the design choice of Mamba.
- CogVideoX > Stable Video Diffusion > DALL-E static > no prior; better video generation models yield more precise temporal grounding.

## Highlights & Insights
- **Generated videos as query augmentation instead of generated images**: This represents a paradigm shift—from static semantic supplementation to dynamic temporal supplementation. The idea is transferable to other video tasks requiring temporal understanding (e.g., video QA, action prediction).
- **LLM-based query decomposition to supplement implicit actions**: The reasoning capability of LLMs is exploited to complete unstated intermediate steps in queries (e.g., "opening the door"), serving as a cheap yet effective form of data augmentation.
- **Video-guided gating integrated into Mamba**: The generated prior guides SSM state propagation through gating, achieving more precise and efficient fusion than simple concatenation.

## Limitations & Future Work
- Video generation quality is a bottleneck—irrelevant motion generated by CogVideoX may introduce noise.
- Video generation requires precomputation (6 seconds/clip); offline mode is acceptable but precludes real-time applications.
- Evaluation is limited to TVR and ActivityNet; comprehensive comparisons on other VMR benchmarks (e.g., Charades-STA) are absent.
- Caption dependency: without captions on ActivityNet, the method degrades to pure query-based video generation, underperforming the caption-enriched TVR setting, indicating that caption fusion is important but not universally available.

## Related Work & Insights
- **vs. ICQ**: ICQ augments queries with static images generated by DALL-E; this work uses CogVideoX-generated dynamic videos. The performance gap is particularly pronounced on multi-verb queries (35.9% vs. 16.8%).
- **vs. SgLFT**: SgLFT uses semantically guided Transformers to fuse captions but lacks motion priors and is constrained by quadratic complexity.
- **vs. Motion Mamba**: This work extends Motion Mamba's text-controlled selection mechanism by adding video-guided gating for multimodal control.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using generated videos as temporal priors is a novel direction in VMR; LLM-based caption matching and Mamba fusion also exhibit clear innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation study is comprehensive and multi-verb analysis is convincing, though the number of evaluation datasets is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Insightful for the VMR community; the generated video prior paradigm has potential to extend to broader video understanding tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

<!-- RELATED:END -->
