---
title: >-
  [Paper Note] StreamReady: Learning What to Answer and When in Long Streaming Videos
description: >-
  [CVPR 2026][Video Understanding][Streaming video understanding] This paper introduces a readiness-aware paradigm for streaming video understanding. By incorporating a learnable `&lt;RDY>` token and proposing the Answer Read…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Streaming video understanding"
  - "answer readiness"
  - "temporal reasoning"
  - "multimodal large language models"
  - "proactive question answering"
date: 2026-05-08
content_hash: b7b987cca445dea8
---

# StreamReady: Learning What to Answer and When in Long Streaming Videos

**Conference**: CVPR 2026
**arXiv**: [2603.08620](https://arxiv.org/abs/2603.08620)
**Code**: [Project Page](https://sacrcv.github.io/StreamReady-website/)
**Area**: Video Understanding
**Keywords**: Streaming video understanding, answer readiness, temporal reasoning, multimodal large language models, proactive question answering

## TL;DR

This paper introduces a readiness-aware paradigm for streaming video understanding. By incorporating a learnable `<RDY>` token and proposing the Answer Readiness Score (ARS) metric, the model is trained not only to produce correct answers but also to respond at the appropriate moment when sufficient evidence has appeared. The approach achieves state-of-the-art results on 9 streaming and offline video benchmarks.

## Background & Motivation

**Urgent need for streaming long-video understanding**: Real-world applications such as surveillance, sports analytics, robotics, and assistive systems require models to reason in real time as video frames arrive sequentially, rather than processing complete videos offline.

**Existing methods focus on *what* to answer while ignoring *when* to answer**: Most streaming video models evaluate only answer correctness, entirely overlooking response timing — answering too early implies speculation, while answering too late compromises real-time utility.

**Challenges in proactive reasoning scenarios**: In proactive settings, questions are posed before the relevant evidence appears; the model must continuously observe the video stream and determine when sufficient visual evidence has accumulated to warrant a response.

**Lack of temporally annotated evaluation benchmarks**: Existing streaming benchmarks lack annotations of evidence time windows, making it impossible to systematically assess whether a model's response timing is appropriate.

**Limitations of existing delayed-answer strategies**: Auxiliary MLLM-based delay (StreamBridge) and prompt-based delay strategies suffer from non-determinism or additional computational overhead, and lack deep coupling with the reasoning module.

**Gap in fine-grained temporal evaluation metrics**: A metric is needed that simultaneously measures correctness and timing, applying severe penalties for premature responses (speculation) while tolerating mild delays.

## Method

### Overall Architecture

StreamReady uses Qwen-2-VL (7B) as the backbone LLM and initializes a dual-branch Q-Former with pretrained weights from HierarQ. It constructs a three-stage pipeline of **hierarchical memory → query-aware reasoning → readiness gating**:

1. **Memory Storage**: Streaming frames are stored in a Visual Memory Tree (three-level structure: raw frame buffer $\mathcal{M}_{V1}$, EMA clustering centroids $\mathcal{M}_{V2}$, and abstract prototypes $\mathcal{M}_{V3}$) plus a Contextual Memory Bank $\mathcal{M}_C$ (semantic memory of historical QA pairs, storing question embeddings and answer representations). The three levels range from fine to coarse, yielding a compact yet information-rich video representation.
2. **Query-Aware Reasoning**: Upon receiving a question, the short-term branch $Q_s$ directly extracts local evidence from $\mathcal{M}_{V1}$ to obtain $z_s$; the long-term branch $Q_\ell$ performs coarse-to-fine retrieval over $\mathcal{M}_{V2}$/$\mathcal{M}_{V3}$ to locate and extract distant evidence, which is then fused cross-scale with $z_s$ to obtain $z_\ell$; a contextual reasoning step further integrates historical QA semantics into $z_\ell$.
3. **Readiness Mechanism**: A `<RDY>` token is appended to the long-term reasoning representation $z_\ell$, and a lightweight Readiness Head (2-layer MLP) outputs a readiness score $R_{\text{pred}} \in [0,1]$. During inference, LLM generation is triggered only when $R_{\text{pred}}$ exceeds the threshold 0.35; otherwise, the model continues observing subsequent frames. After answering, the fused representation is stored as $a_i$ in $\mathcal{M}_C$.

### Key Designs

- **Three-level compression in the Visual Memory Tree**: $\mathcal{M}_{V1}$ is a FIFO raw frame buffer that retains recent raw frame embeddings for short-term detail perception. $\mathcal{M}_{V2}$ dynamically maintains a centroid set via EMA clustering; when $\mathcal{M}_{V1}$ is full, evicted frames update centroids according to Equation (1), with threshold $\tau_t$ adaptively adjusted based on scene stability (tightened for stable scenes to promote merging, relaxed for novel scenes to allow new centroids). $\mathcal{M}_{V3}$ further abstracts into a coarse-grained prototype set when $\mathcal{M}_{V2}$ reaches capacity or experiences distributional drift, with periodic mini-K-means realignment to maintain consistency.
- **Coarse-to-fine query retrieval**: Top-K selection over $\mathcal{M}_{V3}$ prototypes first identifies relevant temporal segments (softmax normalization for stable routing), then Top-$m$ extraction over corresponding $\mathcal{M}_{V2}$ centroids retrieves fine-grained evidence (unnormalized to preserve sharp ranking). This is analogous to episodic recall — prototypes provide coarse temporal anchors, centroids provide detail.
- **Contextual Memory Bank**: Stores question embeddings $q_i$ and answer representations $a_i$ from historical QA pairs. Soft-gated matching between the current question and historical questions selects the most relevant entries, which are fused into the long-term visual features $z_\ell$ via lightweight cross-attention, supporting semantic continuity across multi-turn reasoning.
- **Co-evolution of `<RDY>` token and reasoning module**: The `<RDY>` token is embedded directly within the learned representations of the long-term reasoning branch $Q_\ell$ and co-evolves with query-aligned evidence, naturally sensing the state transition from "weak alignment, low confidence" to "strong alignment, answerable." Ablation studies confirm this placement (ARS 0.68) substantially outperforms placement in the short-term branch (0.31) or at the input of the long-term branch (0.54).
- **Weakly supervised readiness learning**: During training, pseudo-positive/negative regions are constructed using the temporal similarity between $z_\ell$ and $\mathcal{M}_{V2}$, and readiness is learned via contrastive loss $\mathcal{L}_{ctr}$ without requiring ground-truth evidence timestamp annotations. The $\mathcal{M}_{V2}$ level is chosen for its optimal balance between detail and compactness.
- **Gradient isolation design**: $\mathcal{L}_{rdy}$ updates only the Readiness Head and the `<RDY>` token, with no gradient propagation to the reasoning module, allowing "what to answer" and "when to answer" to be optimized independently and avoiding interference between the two objectives.

### Loss & Training

- **Contrastive readiness loss**: $\mathcal{L}_{ctr} = -\log \sigma(R_{pred}(t^+) - R_{pred}(t^-))$, where $t^+ \in P$ (pseudo-positive region) and $t^- \in N$ (pseudo-negative region).
- **Temporal smoothness regularization**: $\mathcal{L}_{rdy} = \mathcal{L}_{ctr} + \lambda_{reg} \|\nabla_t R_{pred}(t)\|_1$, with $L_1$ regularization suppressing noisy fluctuations in the readiness signal.
- **ARS evaluation metric**: $\text{ARS} = \frac{1}{N}\sum_{i}(\text{EP}_i \cdot \text{LP}_i)$, where Early Penalty applies a sharp sigmoid penalty for premature responses ($\gamma_e=6$) and Late Penalty applies a mild decay for delayed responses ($\gamma_\ell=1$). Effective accuracy is defined as $\text{Acc}_e = \text{Acc} \times \text{ARS}$.

## Key Experimental Results

### Main Results

**ProReady-QA readiness-aware evaluation** (Table 2):

| Method | Size | Avg Acc. | Avg ARS | Acc_e |
|--------|------|----------|---------|-------|
| Qwen-2-VL (baseline) | 7B | 41.4 | 0.34 | 0.20 |
| HierarQ | 7B | 46.0 | 0.40 | 0.27 |
| StreamBridge | 7B | 53.1 | 0.60 | 0.42 |
| InfiniPot-V | 7B | 52.0 | 0.47 | 0.36 |
| **StreamReady** | **7B** | **56.4** | **0.69** | **0.53** |

StreamReady outperforms the best competitor, StreamBridge, by ~3% in accuracy and ~9% in ARS, with an 11-percentage-point improvement in effective accuracy. The largest ARS gains appear on REC (+0.25), GSD (+0.11), and CTD (+0.10) tasks, demonstrating that the readiness mechanism is especially effective for tasks requiring the model to wait for evidence.

**Generalization on streaming benchmarks** (Table 3):

- StreamingBench average 63.4 (vs. ViSpeak 58.6); proactive subset 48.2 (vs. ViSpeak 43.9)
- OVOBench average 68.2 (vs. StreamBridge 62.6); proactive subset 63.7 (vs. ViSpeak 61.6)
- VStream-QA RE/RM of 64.8/57.2, both best in class

**Offline long-video benchmarks** (Table 4):

- VideoMME 65.8, MLVU 71.3, MVBench 71.8, EgoSchema 70.4
- Comprehensively outperforms StreamBridge (64.4/69.6/64.4/66.9) and Flash-VStream (61.2/66.3/65.4/68.2)
- The readiness mechanism and contextual reasoning are disabled during offline evaluation; the memory hierarchy and query reasoning alone are sufficient to achieve competitive performance

### Ablation Study

| Configuration | REC Acc/ARS | GSD Acc/ARS | CTD Acc/ARS |
|---------------|-------------|-------------|-------------|
| Baseline (Qwen-2-VL) | 20.7/0.31 | 35.1/0.52 | 30.3/0.28 |
| + Memory + QA Reasoning | 39.4/0.48 | 60.9/0.53 | 43.6/0.39 |
| + Readiness Mechanism | **39.6/0.68** | **61.2/0.68** | **43.5/0.59** |

- The memory and reasoning modules primarily improve accuracy (+19 on REC); the readiness mechanism then substantially improves ARS (+0.20 on REC/CTD).
- The `<RDY>` token + MLP Head achieves performance comparable to a Transformer Head at lower computational cost.
- Placing `<RDY>` within the learned representation of the long-term reasoning branch yields the best results (ARS 0.68), versus only 0.31 when placed in the short-term branch.

### Key Findings

- Relying solely on basic reasoning modules (e.g., HierarQ's Q-Former) improves accuracy but offers little benefit for response timing; the readiness mechanism must be paired with a strong reasoning module to simultaneously improve both Acc and ARS.
- Using an auxiliary MLLM for readiness judgment (as in StreamBridge) achieves only ARS 0.60, falling short of the `<RDY>` token deeply coupled with the reasoning module (0.68).
- StreamReady maintains constant latency and memory as video length increases, owing to fixed-size centroid/prototype memory, whereas Qwen-2-VL runs out of memory on long videos.

## Highlights & Insights

- **First formal treatment of readiness-aware streaming video understanding**: Response timing is incorporated into evaluation, and the asymmetrically penalized ARS metric is introduced, filling a gap in the field.
- **Lightweight and elegant readiness mechanism**: A single `<RDY>` token plus an MLP Head, requiring no auxiliary model or heuristic rules, with zero additional inference overhead.
- **Weakly supervised temporal learning**: Requires no ground-truth evidence timestamp annotations; pseudo-supervision is automatically constructed from the similarity between reasoning representations and memory.
- **Complete benchmark contribution**: ProReady-QA provides 5 proactive task types, 5K QA pairs, 30–60 minute long videos, annotated evidence time windows, and supports both local and global multi-turn dependencies.
- **Broad generalization**: Achieves comprehensive state-of-the-art results across 9 benchmarks (streaming + offline), demonstrating the generality of the design.

## Limitations & Future Work

- ProReady-QA contains only 32 videos (10 Ego-4D + 22 MovieNet), limiting scale and diversity of video types, and may not cover all real-world streaming scenarios.
- Readiness learning relies on pseudo-supervision (similarity between $z_\ell$ and $\mathcal{M}_{V2}$); in scenarios where evidence is highly dispersed or ambiguous, pseudo-positive/negative regions may be inaccurate, leading to erroneous readiness judgments.
- The readiness threshold of 0.35 is a fixed hyperparameter; different tasks and video types may require tuning, and the paper does not explore adaptive thresholds or confidence-based dynamic strategies.
- The three-level memory tree's K-means/EMA updates introduce multiple hyperparameters ($\alpha$, $\tau_t$, $J$, $U$, $K$, $m$), imposing a non-trivial tuning burden; the choice of EMA decay factor significantly affects memory quality.
- Validation is conducted only at the 7B scale; performance at larger (e.g., 70B) or smaller (e.g., 2B) scales remains unexplored, and scalability has yet to be verified.
- The asymmetric penalty parameters of the ARS metric ($\gamma_e=6$, $\gamma_\ell=1$) are shown to be robust experimentally, but optimal penalty weights may differ across application domains (e.g., security surveillance vs. sports analysis).

## Related Work & Insights

- **Offline long-video understanding**: HierarQ, LLaVA-Video, LongVU, and similar methods employ memory or query-conditioned storage but require complete videos and memory reconstruction, making them unsuitable for streaming. StreamReady draws on the idea of query-aware conditioning but adapts it for streaming without memory resets.
- **Streaming video understanding**: StreamBridge uses an auxiliary MLLM to delay responses for proactive behavior; Flash-VStream, StreamForest, and InfiniPot-V focus on online memory management and retrieval efficiency; ViSpeak explores voice-interaction scenarios. None of these methods provide explicit timing control. StreamReady's readiness mechanism can serve as a complementary extension to these approaches.
- **Streaming benchmarks**: ODVBench, StreamBench, and OVBench support only past-dependent QA; StreamingBench, OVOBench, and Omni-MMI introduce proactive scenarios but are limited to short videos and local context. ProReady-QA is the first to provide evidence time window annotations, global multi-turn dependencies, and five proactive reasoning task types on long videos.
- **Response timing control**: Existing strategies include prompt-based delay ("Answer whenever you are ready") and auxiliary MLLM readiness judgment. StreamReady proposes a new paradigm of embedding readiness judgment within the reasoning module, eliminating external dependencies and non-determinism.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to formalize the response timing problem and propose a complete metric + method + benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 9 benchmarks with detailed ablations covering every component and design choice
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-formulated equations, and rich figures and tables
- Value: ⭐⭐⭐⭐⭐ — Defines a new problem, metric, method, and benchmark; makes a significant contribution to the field of streaming video understanding

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](color_when_it_counts_grayscale-guided_online_triggering_for_always-on_streaming_.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)

</div>

<!-- RELATED:END -->
