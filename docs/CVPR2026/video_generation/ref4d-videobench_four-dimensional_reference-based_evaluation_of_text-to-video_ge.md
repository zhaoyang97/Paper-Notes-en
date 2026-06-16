---
title: >-
  [Paper Note] Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models
description: >-
  [CVPR 2026][Video Generation][Paper Note] To address the issue where sample-level failures in existing text-to-video (T2V) evaluations cannot be attributed due to "no-reference, prompt-only" paradigms, this paper proposes Ref4D-VideoBench. Using 600 real reference videos as structured spatio-temporal evidence, it designs 12 interpretable atomic metrics across
tags:
  - CVPR 2026
  - Video Generation
date: 2026-05-08
content_hash: 828bfd7bf61162f2
---
# Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html)  
**Code**: https://github.com/TAILab-W/Ref4D-VideoBench  
**Area**: Video Generation Evaluation / Benchmark  
**Keywords**: Text-to-video evaluation, Reference video, Multi-dimensional benchmark, Event graph alignment, World knowledge consistency

## TL;DR
To address the issue where sample-level failures in existing text-to-video (T2V) evaluations cannot be attributed due to "no-reference, prompt-only" paradigms, this paper proposes Ref4D-VideoBench. Using 600 real reference videos as structured spatio-temporal evidence, it designs 12 interpretable atomic metrics across four dimensions: semantic alignment, motion consistency, event temporal order, and world knowledge. It achieves significantly higher correlation with human scores across 8 T2V models compared to no-reference baselines (e.g., world knowledge SRCC 0.847 vs. baseline ≤0.42).

## Background & Motivation
**Background**: T2V generation (Sora2, HunyuanVideo, CogVideoX, etc.) has progressed rapidly and is viewed as a potential "world simulator." Currently, the dominant evaluation paradigm is *no-reference*—providing only a text prompt for automatic metrics or MLLM judges to score, represented by VBench, EvalCrafter, and T2VScore.

**Limitations of Prior Work**: The authors point out two critical flaws in the no-reference paradigm. First, the **lack of sample-level reference standards**: when generated videos exhibit object hallucinations, motion artifacts, or violations of common sense/safety, prompts alone cannot specifically locate "what went wrong and why," failing to provide accountable or interpretable judgments. Second, there is an **increasing reliance on opaque MLLM judge pipelines**, which are unreliable due to inherent biases, hallucinations, and inconsistent judgments.

**Key Challenge**: The complexity of T2V generation (fine-grained semantics, long-range temporal dependencies, physical common sense) exceeds the "verification granularity" provided by existing evaluation protocols. Many failure cases **cannot be verified by prompts alone**—the prompt specifies "what should be there" but not "how it moves, how events unfold, or which physical laws are obeyed."

**Goal**: Construct a T2V evaluation benchmark capable of providing fine-grained, auditable, and diagnostic judgments at the sample level.

**Key Insight**: In scenarios with clear expectations like "controlled generation," **reference videos naturally provide rich and unambiguous spatio-temporal evidence**—they reify entities, attributes, motion, events, and physical constraints. Thus, evaluation can be upgraded from "text alignment" to "itemized verification against a real video."

**Core Idea**: Replace pure text prompts with reference videos as the evidence source (reference-based rather than no-reference), decomposing the "consistency check" for each sample into interpretable atomic metrics across four dimensions, ensuring scores both align with humans and explain "why this score was given."

## Method

### Overall Architecture
Ref4D-VideoBench consists of two parts: an **evidence-bounded dataset** and a **four-dimensional structured evaluation framework**.

Dataset side (construction pipeline): Retrieve ~2500 candidate short videos from YouTube across nine themes → Use DOVER for quality assessment to retain the top 40% → Cut shots and control duration (≤20s, typically ~10s, 1–3 semantically coherent shots) → Use MiniCPM-V-4.5 to extract global scene elements, DDM-Net to detect event boundaries, and VideoLLaMA3-7B to generate event text descriptions, integrating them into structured spatio-temporal semantic evidence → Conditioned on this evidence, have an MLLM generate a descriptive prompt that "must cover key objects/attributes/events" → Manual verification to remove hallucinations. This results in **600 reference videos + paired prompts + JSON annotations**.

Evaluation side (inference time): Given a reference video $V_{ref}$ and a generated video $V_{gen}$, the framework extracts four types of evidence from the reference side (semantic entities-attributes, event intervals, fore/background motion trajectories, world knowledge rule base), calculates atomic metrics for each dimension, and aggregates them into dimensional scores. For the semantic, motion, and event dimensions, the final score uses a **dimension-specific linear aggregator**:

$$S^{(d)}(x) = {w^{(d)}}^{\top} f^{(d)}(x) + b^{(d)}$$

where $f^{(d)}(x)$ is the atomic metric vector for dimension $d$ ($d\in\{\text{semantic, motion, event}\}$). Data is split into train/test by sample ID once (to avoid leakage within the same scene), and the aggregator is fitted on the training set using least squares to the z-score normalized human Mean Opinion Score (MOS). The world knowledge dimension does not require a trained aggregator and is calculated across all samples directly.

### Key Designs

**1. Basic Semantic Alignment: Verifying entities and attributes under soft matching with hallucination penalties**

This dimension answers whether the generated video faithfully reproduces the entities and key attributes from the reference video without misassignment or fabrication. The approach extracts entity sets $R$ and $G$ from $V_{ref}$ and $V_{gen}$, encodes entity names and attribute values using a text encoder, calculates similarity $w(r,g)\in[0,1]$ for every pair $(r,g)$, and uses the Hungarian algorithm for one-to-one maximum weight bipartite matching to obtain $\mathcal{M}_{semantic}$. Based on this, three atomic metrics are defined: **CatCov (Category Coverage)** measures entity recall by taking $\text{cov}(r)=\max_{(r,g)\in\mathcal{M}_{semantic}} w(r,g)$ for each reference entity; **AIC (Attribute Integrity & Consistency)** calculates attribute coverage and "misbinding rate" (where an attribute value resembles other reference entities more than the current $r$) for each matched pair, defined as $S_{AIC}=\text{Coverage}\cdot(1-\text{Misbind})$; **Hallucination Penalty** treats unmatched entities/attributes in $G$ as hallucinations, penalizing based on their maximum similarity to the reference side, $S_{Hal}=1-\text{HallRate}$. These are concatenated into a semantic feature vector for the aggregator. Soft matching allows for semantic overlap (e.g., "man in white" vs "man wearing white") while catching hard errors like "extra antelope generated."

**2. Motion Consistency: Distribution comparison on relative fore/background motion with explicit "freeze/degradation" detection**

This dimension judges whether the generated video reproduces the significant motion patterns of the reference video while avoiding screen freezing or jittery degradation. The key technique is **focusing on foreground motion relative to the background**: extract fore/background masks and sparse point tracking for both videos to get average velocities $v^{fg}(t)$ and $v^{bg}(t)$, defining relative motion $\mathbf{r}(t)=\mathbf{v}^{fg}(t)-\mathbf{v}^{bg}(t)$. This eliminates global camera motion to focus on object dynamics. From the distribution of $\mathbf{r}(t)$, metrics for direction difference $D_{dir}$, magnitude difference $D_{mag}$, and smoothness/jerk difference $D_{smo}$ are derived, mapped to $[0,1]$ via $S_k=\exp(-\lambda_k D_k)$ ($\lambda_k=1$). Two degradation metrics are added: **RF (Repeated Frame ratio)** uses inter-frame similarity to measure highly repetitive frames in $V_{gen}$, and **LS (Low Speed ratio)** measures the proportion of time steps where the relative motion magnitude of $V_{gen}$ is below the 40th percentile of the reference. $\{S_{dir},S_{mag},S_{smo},\text{RF},\text{LS}\}$ form the motion feature vector. "Frozen screens" are modeled separately because a common T2V failure is near-stasis, which pure motion difference metrics might miss.

**3. Event Temporal Consistency: Using event graphs to compare types, temporal relations, and omissions/redundancy**

This dimension verifies if the generated video preserves the event-level content and temporal structure of the reference. Both videos are segmented into $N_{ref}$ and $N_{gen}$ event intervals, each with a text description encoded into embeddings $t_i^{ref}$ and $t_j^{gen}$. Similarity $Sim_{sem}(i,j)$ and relative Intersection over Union $rIoU(i,j)$ are calculated for candidate pairs; those meeting thresholds are bipartite matched to form $\mathcal{M}_{event}$. Three complementary metrics are used: **EGA (Event Graph Alignment)** measures local alignment in the joint "semantic $\times$ temporal" space: $q_{ij}=w_1\text{Sim}_{sem}(i,j)+w_2\text{rIoU}(i,j)$, using duration-based weights $\omega_i$ for the weighted average $S_{EGA}=\frac{\sum \omega_i q_{ij}}{\sum \omega_i}$. **ERel (Event Relationship consistency)** examines the relationship between any reference event pair $(i,k)$ using Allen interval relations (e.g., before, overlaps, during) and compares it to the matched generated pair; an affinity matrix maps these to $u_{ik}\in[0,1]$, where $S_{ERel}=\frac{1}{|\mathcal{P}|}\sum_{(i,k)\in\mathcal{P}} u_{ik}$. **ECR (Event Coverage & Redundancy)** balances coverage $C_{ref}$ and hallucinated event ratio $H_{gen}$ via a harmonic mean: $S_{ECR}=\frac{2 C_{ref}(1-H_{gen})}{C_{ref}+(1-H_{gen})+\epsilon}$. Compared to simple frame-level order scores, the event graph decomposes alignment, relationships, and omissions/additions for better diagnostic power.

**4. World Knowledge Consistency: Constructing per-video rule banks → VQA → Weighted scoring**

This dimension evaluates whether the generated video obeys physical, causal, and appearance constraints implied by the reference video. Rather than matching against a global knowledge base, the authors **construct a video-specific question bank** to anchor judgments to the specific scene. The process starts from a project-specific signal dictionary $\Sigma_{custom}$ (physics/causality/appearance with base weights $W(s)$), derives rules based on $V_{ref}$'s semantic and event evidence, normalizes their scope and polarity, and ranks them by significance. A video MLLM then rewrites rules into concise VQA entries (short answer + Yes/No assertions). A lightweight MLLM filters these to retain entries with quality $\geq 80$, resulting in the per-video question bank $B^+$. During evaluation, the video MLLM answers entries applicable to $V_{gen}$. Each entry yields a consistency score $\tilde{c}_q\in[0,1]$ modulated by evidence and safety-critical coverage. The total world knowledge score is a weighted average based on importance:

$$S_{world}=\frac{\sum_{q\in B^+}\alpha_q \tilde{c}_q}{\sum_{q\in B^+}\alpha_q},\quad \alpha_q\propto w_{type}(q)\sum_{s\in S(q)} W(s)\, d(q)$$

where weights $\alpha_q$ are determined by question type $w_{type}(q)$, total required signal weight, and difficulty $d(q)$. Compared to fixed global questionnaires, per-video banks are more relevant to the scene, reducing irrelevant judgments and increasing diagnostic specificity.

## Key Experimental Results

### Main Results: Sample-level Correlation with Human Scores (Tab. 1)
Evaluation of 8 T2V models (Sora2, Kling-v1, CogVideoX-5B/Fun-5B, JiMeng, VideoCrafter2, ViduQ2, Wan2.1). Humans provided MOS on a 1–5 Likert scale ( $\geq 3$ people per video, z-normalized). Metrics used are SRCC/PLCC/KRCC.

| Dimension | Strongest Baseline (Method) | Baseline SRCC | Ours SRCC | Ours PLCC | Ours KRCC |
|------|------------------|-----------|-----------|-----------|-----------|
| Semantic | Q-Align (quality) | 0.317 | **0.822** | 0.828 | 0.635 |
| Motion | Q-Align (quality) | 0.358 | **0.659** | 0.669 | 0.480 |
| Event | UMTScore | 0.220 | **0.755** | 0.773 | 0.626 |
| World Knowledge | Q-Align (quality) | 0.391 | **0.847** | 0.822 | 0.719 |

**Key Findings**: All no-reference baselines fail to exceed an SRCC of **0.42** in any dimension, while this paper's dimensions consistently fall between **0.48–0.847**. Generic similarity/quality metrics like CLIPScore/BLIPScore are weak (SRCC 0.03–0.30), indicating they only capture coarse trends.

### Main Results: Model Benchmarking (Tab. 2, scores mapped to [0,100])
| Type | Model | Semantic↑ | Motion↑ | Event↑ | World Knowledge↑ |
|------|------|-------|-------|-------|-----------|
| Closed | JiMeng | 61.64 | 63.87 | **58.89** | **75.05** |
| Closed | Sora2 | 59.39 | **64.26** | 58.32 | 72.21 |
| Closed | ViduQ2 | **62.84** | 58.87 | 56.89 | 71.54 |
| Open | CogVideoX-5B | 49.79 | 51.70 | 51.44 | 54.97 |

The framework effectively decouples capabilities: ViduQ2 leads in semantics, Sora2 in motion, and JiMeng in event/world knowledge. Closed-source models generally outperform open-source ones.

### Ablation Study
**Atomic Metrics vs. Human Scores (Tab. 3)**: Some individual atomic metrics show decent correlation (e.g., Semantic CatCov SRCC 0.734, Event ECR 0.718), but the learned aggregator further boosts performance while maintaining an interpretable, small feature space.

**World Knowledge: Rule Banks & MLLM Choice (Tab. 4)**: Removing rule/question banks (w/o bank) causes correlation to plummet—MiniCPM-V-4.5 drops from 0.847 to 0.413, proving that **per-video question banks are the primary driver** of performance in the world knowledge dimension, rather than simply having a stronger MLLM.

## Highlights & Insights
- **"Reference Video as Evidence" transforms evaluation into verification**: It moves from "guessing if points should be deducted based on a prompt" to "verifying against real video entities/events/rules," making failures locatable and explainable.
- **Event Graph + Allen Interval Relations**: Refining temporal evaluation from "correct order" to "correct before/overlap/during relations" between events provides much stronger diagnostic insight than frame-level order scores.
- **Per-Video Question Banks**: Anchoring world knowledge checks to the specific "scenario" avoids irrelevant common-sense questions. Ablation shows this is the main performance driver.
- **Foreground Relative to Background Motion**: This simple normalization technique effectively isolates object dynamics from camera motion during evaluation.

## Limitations & Future Work
- The authors acknowledge that Ref4D-VideoBench is oriented toward **reference-based, controlled** generation; it is not directly applicable to open-ended generation without clear expectations.
- Metric definitions rely on the assumptions of underlying modules (segmentation, tracking, event detection); errors in these modules propagate to scores, as seen in the slightly lower motion correlation.
- The dataset (600 segments) focuses on realistic everyday videos; representation for abstract or highly stylized generation remains to be explored.

## Related Work & Insights
- **vs VBench / EvalCrafter / T2VScore**: While these use fine-grained dimensions, they remain no-reference and lack sample-level verifiable anchors. Ref4D-VideoBench provides explicit evidence, significantly improving correlation (SRCC ≤0.42 vs. up to 0.847).
- **vs MLLM-as-judge**: Direct use of opaque MLLMs as judges introduces bias and hallucinations. This paper constrains MLLMs to structured roles (extracting evidence/answering VQA), with final scores derived from transparent atomic metrics + linear aggregation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematizing "reference videos as structured spatio-temporal evidence" into 12 metrics across four dimensions is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation of 8 models across multiple metrics and dimensions, including extensive ablation; however, the dataset size (600) is modest.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete formulas for metrics, though some details (safety coverage) are brief.
- Value: ⭐⭐⭐⭐⭐ Provides an auditable, diagnostic evaluation for T2V; structured evidence can also support reward design and safety audits.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Thinking with Frames: Generative Video Distortion Evaluation via Frame Reward Model](thinking_with_frames_generative_video_distortion_evaluation_via_frame_reward_mod.md)
- [\[CVPR 2026\] Inference-time Physics Alignment of Video Generative Models with Latent World Models](inference-time_physics_alignment_of_video_generative_models_with_latent_world_mo.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[CVPR 2026\] TempoControl: Temporal Attention Guidance for Text-to-Video Models](tempocontrol_temporal_attention_guidance_for_text-to-video_models.md)
- [\[CVPR 2025\] NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification](../../CVPR2025/video_generation/neuro-symbolic_evaluation_of_text-to-video_models_using_formal_verification.md)

</div>

<!-- RELATED:END -->
