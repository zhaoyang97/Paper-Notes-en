---
title: >-
  [Paper Note] Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs
description: >-
  [ICLR 2026][Video Understanding][VideoLLM] This work presents the first systematic reverse-engineering of temporal reasoning in VideoLLMs using mechanistic interpretability tools (Attention Knockout + Logit Lens)…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "VideoLLM"
  - "Information Flow Analysis"
  - "Mechanistic Interpretability"
  - "Attention Pruning"
  - "Temporal Reasoning"
date: 2026-05-08
content_hash: 12053b33c27a7f40
---

# Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

**Conference**: ICLR 2026
**arXiv**: [2510.13251](https://arxiv.org/abs/2510.13251)
**Code**: [Project Page](https://map-the-flow.github.io)
**Area**: Video Understanding / Explainable AI
**Keywords**: VideoLLM, Information Flow Analysis, Mechanistic Interpretability, Attention Pruning, Temporal Reasoning

## TL;DR

This work presents the first systematic reverse-engineering of temporal reasoning in VideoLLMs using mechanistic interpretability tools (Attention Knockout + Logit Lens), uncovering a three-stage information flow blueprint—"early-to-mid-layer cross-frame interaction → mid-layer video-language integration → mid-to-late-layer answer generation"—and demonstrating that retaining only 42% of attention edges preserves VideoQA performance with negligible degradation.

## Background & Motivation

**Background**: The standard paradigm for VideoLLMs patchifies video frames into token sequences via a visual encoder, concatenates them with text tokens, and feeds the combined sequence into a causal-attention LLM for autoregressive generation. Prior research has largely focused on the "external design" of such models—scaling video instruction-tuning datasets, keyframe selection strategies, and video token compression methods. However, almost no systematic investigation has been conducted into *how* models internally extract temporal information from flattened frame-token sequences, or *where* semantic integration between video and language occurs.

**Limitations of Prior Work**: Interpretability studies on image-based MLLMs (e.g., Neo 2025) have identified structured behavioral patterns, but whether these findings generalize to the video domain remains entirely unknown. Video fundamentally differs from images: VideoQA requires aggregating information across multiple frames along the temporal dimension. Three core questions remain unaddressed: (1) How do VideoLLMs encode temporal order from a flattened frame-token sequence? (2) How do temporal concepts (e.g., "before," "after") propagate from video tokens to text tokens? (3) At which layer is the model "ready" to generate the correct answer?

**Key Challenge**: After patchification, video frames are reduced to a one-dimensional token sequence, with temporal structure implicitly encoded in positional indices. The model must rediscover and exploit these temporal relationships through some internal mechanism. Yet existing research focuses solely on performance improvement, leaving the question of "what happens inside this black box" entirely unanswered—a gap that impedes principled architectural improvements and inference acceleration.

**Goal**: To provide a complete blueprint of temporal reasoning in VideoLLMs: where information is extracted, in which layers it is integrated, and at what stage the answer is prepared. The work further validates whether the identified critical pathways sufficiently represent the model's reasoning process.

**Key Insight**: The authors adopt a mechanistic interpretability perspective, employing causal intervention tools (Attention Knockout to sever specific attention edges and measure the effect) and probing tools (Logit Lens to project intermediate hidden states into the vocabulary space), decomposing the VideoLLM's reasoning process into verifiable stages.

**Core Idea**: Attention Knockout and Logit Lens are used to reverse-engineer the attention pathways of VideoLLMs, revealing that temporal reasoning follows a consistent three-stage pattern—cross-frame interaction → temporal keyword alignment → answer generation—with the majority of attention edges being redundant.

## Method

### Overall Architecture

This paper presents an analysis study rather than a novel model. The overall pipeline is: (1) run a VideoLLM on VideoQA tasks → (2) apply Attention Knockout to systematically sever different categories of attention edges (cross-frame, video→question, question→last token, etc.) and observe changes in prediction probability → (3) apply Logit Lens to probe the semantic content of video token hidden states at each layer → (4) synthesize findings into a three-stage information flow blueprint → (5) validate: retain only the critical pathways, prune the remaining attention edges, and evaluate on benchmarks.

The primary experimental backbone is LLaVA-NeXT-7B fine-tuned on the VideoChat2-IT dataset for 3 epochs (LLaVA-NeXT-7B-Video-FT), using 8-frame sampling with 144 tokens per frame. Analysis covers 5 temporal reasoning task types (Action Antonym, Action Sequence, Scene Transition, Moving Direction, Object Count), all drawn from the TVBench benchmark.

### Key Designs

1. **Attention Knockout — Causal Tracing of Attention Contributions**

   - *Function*: Selectively severs attention connections between specific token pairs (by setting the corresponding attention mask entries to $-\infty$), then measures the resulting impact on model prediction probabilities.
   - *Mechanism*: For each layer $l$, attention edges of interest (e.g., cross-frame video-to-video attention) are severed using a window of size $k=9$ centered at that layer. The relative probability change is computed as $((p_{\text{knockout}} - p_{\text{base}})/p_{\text{base}}) \times 100$. The window size of 9 is chosen because narrower windows allow information to bypass the intervention via residual connections.
   - *Design Motivation*: Compared to observational analysis (e.g., directly inspecting attention weights), causal intervention accurately quantifies "how would model behavior change if this information pathway were absent"—the standard method in mechanistic interpretability research, originating from Geva et al. 2023.

2. **Logit Lens — Probing Semantic Content of Intermediate Layers**

   - *Function*: Projects intermediate hidden states through the language model head to obtain logits, revealing what each video token "looks like" as a vocabulary item at each layer.
   - *Mechanism*: The LM head is applied to video token representations at each layer; the frequency and positional distribution of spatial keywords (objects, colors) and temporal keywords (before/after/first, etc.) are tracked. Analysis is performed using LLaVA-NeXT-13B-Video-FT on the Action Sequence task.
   - *Design Motivation*: Attention Knockout identifies *which* pathways are important, but not *what information* flows through them. Logit Lens fills this gap by revealing when and where temporal concepts emerge within video tokens.

3. **Information Flow Stage Decomposition and Validation**

   - *Function*: Attention pathways are categorized into 6 types (cross-frame video-video, video→question, video→last, question→last, last→last, question→video), and the contribution of each type across layers is analyzed.
   - *Mechanism*: Each pathway type is knocked out layer by layer, and layer-versus-probability-change curves are plotted. Once pathway roles are established, only pathways in critical layer ranges are retained for each type (e.g., L6–15 for cross-frame interaction, L6–20 for video→question, L16–25 for question→last), while all other pathways are disabled.
   - *Design Motivation*: This closes the loop from discovery to validation—critical pathways are first identified analytically, then an end-to-end experiment retaining only those pathways confirms the correctness of the analysis.

### Four Core Findings

**Finding 1: VideoQA Fine-Tuning Induces Cross-Frame Interaction in Early-to-Mid Layers.** Comparing LLaVA-NeXT-7B (image-only training) with LLaVA-NeXT-7B-Video-FT (after video fine-tuning), knocking out cross-frame attention in early-to-mid layers causes a large performance drop in the video-fine-tuned model but almost no effect in the image model. This demonstrates that video fine-tuning specifically establishes cross-frame interaction capability in early-to-mid layers—a new ability absent from image pretraining. Completely disabling cross-frame attention in the first half of layers (L1–16) leads to accuracy drops of 18%–60.8%, and the model produces semantically opposite answers.

**Finding 2: Emergence and Selective Propagation of Temporal Concepts in Video Tokens.** Logit Lens reveals that spatial concepts (objects, colors) appear in foreground tokens at very early layers, whereas temporal concepts (before/after, etc.) emerge in mid layers and occupy positions *outside* the foreground tokens—the model first stabilizes spatial representations in foreground tokens, then encodes temporal dynamics in remaining token positions. When cross-frame attention is intact, temporal keywords in the question (e.g., "begins," "ends") precisely attend to semantically corresponding temporal segments in the video; disabling cross-frame attention degrades this to proximity-based positional attention.

**Finding 3: Answer Generation Initiates in Mid-to-Late Layers.** Tracking the probability of the correct answer at the last token position across layers, the probability rises sharply after approximately layer 20 (mid-layer), coinciding precisely with the completion of video→question information flow. The correct option rapidly dominates over alternatives, with no prolonged competition between options.

**Finding 4: Critical Information Pathways Constitute Only 42% of Total Attention Edges.** Retaining critical pathways while disabling the remaining 58% of attention edges results in negligible performance degradation on both TVBench and TOMATO benchmarks. Randomly pruning the same proportion causes performance collapse.

## Key Experimental Results

### Effective Pathway Pruning — Multi-Model Validation

| Model | Video Tokens | Attention Edge Retention | TVBench | TOMATO | vs Full Causal |
|-------|-------------|------------------------|---------|--------|----------------|
| LLaVA-NeXT-7B-Video-FT | 8×12×12 | **42%** (10.8M/25.7M) | 51.2 | 29.2 | −0.3 / −1.0 |
| LLaVA-NeXT-7B-Video-FT (random pruning) | same | 42% | 40.1 | 23.1 | −11.4 / −7.1 |
| LLaVA-NeXT-13B-Video-FT | 8×12×12 | **37%** (14.3M/32.2M) | 54.6 | 27.4 | −0.5 / +0.2 |
| Mini-InternVL-4B-Video-FT | 8×16×16 | **40%** (29.6M/74.6M) | 56.0 | 31.2 | 0.0 / −1.0 |
| VideoLLaMA3-7B | 8×12×12 | **58%** (11.4M/19.9M) | 57.2 | 28.7 | +2.0 / +0.7 |

Effective pathway pruning consistently works across 4 models of different architectures and scales. VideoLLaMA3 even surpasses the baseline after pruning, suggesting that some attention edges introduce disruptive noise.

### Cross-Frame Attention Ablation — By Task

| Task | Accuracy Drop After Disabling Early-Layer Cross-Frame Attention | Typical Error |
|------|---------------------------------------------------------------|---------------|
| Action Antonym | −24.1% | "stand up" → "sit on chair" (semantically opposite) |
| Action Sequence | −20.2% | "open bag" → "put bag in microwave" (completely wrong order) |
| Scene Transition | −18.0% | "bedroom→street" → "street→different location" (direction reversed) |
| Moving Direction | −44.8% | "move right" → "move left" (direction opposite) |
| Object Count | −60.8% | "zero moving objects" → "three" (completely wrong count) |

This table clearly demonstrates the indispensability of cross-frame attention—disabling it causes the model to produce semantically opposite answers rather than merely expressing uncertainty, indicating that without cross-frame interaction the model reverts to static biases.

### Key Findings

- **Unique Contribution of VideoQA Fine-Tuning**: By comparing same-architecture ImageLLMs and VideoLLMs, the study confirms that cross-frame attention interaction is a capability exclusively acquired through video fine-tuning. The cross-frame interaction patterns in early-to-mid layers appear only after video fine-tuning, answering the fundamental question of "what does video fine-tuning actually teach."
- **Emergent Nature of Temporal Concepts**: Temporal concepts in video tokens are not directly produced by the visual encoder but emerge spontaneously in the middle layers of the LLM. Spatial concepts stabilize first (in foreground tokens), and temporal concepts emerge later (in remaining tokens), with the two categories occupying non-overlapping positions in the token space.
- **Temporal Keywords as "Information Checkpoints"**: Temporal keywords in the question (action words and temporal adverbs in answer options) serve as information integration checkpoints. The pathway by which video information reaches these checkpoints differs across tasks: simpler tasks use direct paths (video→option), while tasks requiring target identification use indirect paths (video→non-option question token→option).
- **Mechanistic Diagnosis of Failure Cases**: Analysis of incorrectly predicted samples reveals that the cross-modal integration pathway patterns (mid-to-late layers) are consistent with correct samples, indicating that failure originates in the earlier video representation stage—either spurious cross-frame attention corrupts the representation (Case 1), or the model reverts to static scene biases (Case 2).

## Highlights & Insights

- **Complete Three-Stage Reasoning Blueprint**: The black-box reasoning process of VideoLLMs is decomposed into three verifiable and actionable stages. This is not merely descriptive analysis—the end-to-end validation of "retain only critical pathways" closes the analysis→hypothesis→validation loop, strengthening the credibility of the findings.
- **58% of Attention Edges Can Be Safely Pruned**: This finding has direct practical implications—it enables more efficient VideoLLM inference pipelines. Unlike heuristic token compression methods, this paper explains from a mechanistic perspective *why* certain token interactions are redundant, providing a theoretical basis for attention sparsification.
- **The Discovery of "Emerging" Temporal Concepts Is Particularly Compelling**: After video frames are processed by the spatial encoder, the tokens themselves do not directly encode temporal semantics. Yet temporal concepts spontaneously emerge in non-foreground token positions at intermediate layers during LLM processing. This suggests that the LLM's self-attention mechanism possesses the capacity to "invent" temporal semantics from positionally encoded sequences.
- **Two Distinct Failure Mechanisms**: The distinction between Case 1 (spurious cross-frame attention) and Case 2 (static bias reversion) provides targeted directions for improvement—the former requires improving cross-frame interaction quality, while the latter requires reducing static scene bias in training data.

## Limitations & Future Work

- **Limited Task Coverage**: Primary analysis is conducted on TVBench (multiple-choice QA). Although the appendix supplements with open-ended QA and long videos, the information flow patterns may differ fundamentally for generative tasks such as video captioning and summarization.
- **Model Scale Constraints**: The largest model analyzed contains 13B parameters. Whether models at the 70B+ scale follow the same three-stage pattern remains unknown; deeper networks may exhibit more complex information routing.
- **Layer Window Granularity in Attention Knockout**: A layer window of $k=9$ is used to prevent residual bypass, but this limits the layer-level resolution of the analysis. Finer-grained causal interventions (e.g., single-layer + MLP analysis) may reveal more precise mechanisms.
- **Manual Specification of Pruning Layer Ranges**: The layer ranges for effective pathways (e.g., L6–15 for cross-frame interaction) are determined manually based on observed patterns. Adaptively learning the optimal pathway range for each sample could further improve pruning efficiency.
- **Static Analysis Without Dynamic Adaptation**: Current analysis captures dataset-level statistical patterns, but the optimal pathway for each individual video sample may differ. Developing sample-adaptive pathway selection is a potentially valuable research direction.

## Related Work & Insights

- **vs. Image MLLM Interpretability (Neo 2025, Zhang 2025c)**: Prior work identified structured visual-language information flow patterns in image-based MLLMs. This paper extends the analytical paradigm to the video domain and discovers a fundamentally new capability—cross-frame temporal interaction. The key contribution is demonstrating that video fine-tuning introduces new computational pathways absent from image pretraining.
- **vs. Token Compression Methods (FastV, LLaVA-PruMerge, etc.)**: These methods heuristically remove video tokens from an efficiency standpoint. This paper explains from a mechanistic perspective why certain token interactions can be safely removed—they lie outside the critical information pathways. The two approaches are complementary: the findings presented here can guide more principled token/attention compression strategies.
- **vs. Early Exit Strategies (Elbayad 2020, Schuster 2022)**: The finding that answers are prepared after mid-layer processing directly motivates early exit—a large fraction of computation in later layers is redundant. Unlike conventional confidence-based early exit, this paper provides a structural early exit rationale grounded in the completion of information flow.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First complete mechanistic analysis of temporal reasoning in VideoLLMs, filling a critical gap in video interpretability research.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across 5 task types and 4 models with a complete analysis→hypothesis→validation loop; task types are primarily limited to multiple-choice QA.
- Writing Quality: ⭐⭐⭐⭐⭐ — Research questions are clearly decomposed, figures are intuitive (especially the flow blueprint in Fig. 1), and findings are presented with strong narrative progression.
- Value: ⭐⭐⭐⭐ — Directly actionable for VideoLLM architectural design, attention sparsification, and inference acceleration; failure mode analysis identifies clear improvement directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[AAAI 2026\] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](../../AAAI2026/video_understanding/reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](../../AAAI2026/video_understanding/causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)

</div>

<!-- RELATED:END -->
