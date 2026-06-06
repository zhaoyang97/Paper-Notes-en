---
title: >-
  [Paper Note] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models
description: >-
  [CVPR 2026][Video Generation][Text-to-video safety] This paper proposes TEAR, the first automated red-teaming framework targeting temporal-dimension vulnerabilities in text-to-video (T2V) models. Through a two-stage opti…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Text-to-video safety"
  - "automated red-teaming"
  - "temporal awareness"
  - "adversarial prompt generation"
  - "AI safety"
date: 2026-05-08
content_hash: 87e0c82f3f5647fc
---

# TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models

**Conference**: CVPR 2026
**arXiv**: [2511.21145](https://arxiv.org/abs/2511.21145)  
**Code**: None  
**Area**: Video Generation
**Keywords**: Text-to-video safety, automated red-teaming, temporal awareness, adversarial prompt generation, AI safety

## TL;DR
This paper proposes TEAR, the first automated red-teaming framework targeting temporal-dimension vulnerabilities in text-to-video (T2V) models. Through a two-stage optimized temporal-aware test generator and an iterative refinement model, TEAR generates textually benign prompts that exploit temporal dynamics to elicit harmful videos, achieving attack success rates (ASR) exceeding 80% on both open-source and commercial T2V models.

## Background & Motivation
T2V models (e.g., Veo, Hailuo, Wan) can now produce high-quality, temporally coherent videos, yet they are also susceptible to generating harmful content, making safety evaluation critically important.

**Key Challenge**: Existing red-teaming methods primarily target static image and text generation, and fail to capture **temporal dynamic safety risks** unique to video generation. Harmfulness in video can be absent from any single frame yet emerge from the temporal composition of frame sequences — for example, individually describing "a person picking up a knife" and "another person falling" are both benign, but their temporal combination can constitute a violent scene.

**Limitations of Prior Work**:
1. LLM red-teaming methods (e.g., CuriDial, FLIRT) focus on textual adversarial attacks and entirely overlook the temporal dimension of video.
2. Image red-teaming methods (e.g., ART, Groot) treat video as sequences of independent frames, failing to assess new risks arising from temporal composition.
3. T2VSafetyBench is the first T2V safety benchmark, but relies solely on static harmful prompts, resulting in limited ASR.
4. Incorporating temporal information substantially expands the search space, introducing new technical challenges.

**Core Idea**: Red-team prompt generation is formulated as an MDP. A two-stage optimization trains an LLM-based generator — first initialized on constructed data, then refined via online preference learning with rewards combining prompt safety and video temporal consistency — accompanied by an iterative refinement model that progressively improves attack efficacy.

## Method

### Overall Architecture
TEAR comprises three components:
1. **Temporal-aware Test Generator** — the core component, an LLM-based model trained to produce temporally restructured adversarial prompts from seed prompts.
2. **Refinement Model** — a multimodal LLM (Qwen-3-VL) that iteratively improves prompts based on judge feedback.
3. **Target T2V Model** — the video generation model under evaluation.

Red-teaming objective: find a prompt set $\mathcal{P}_v^*$ such that $\Phi_P(p) = 0$ (text judge deems the prompt safe) and $\Phi_V(\mathcal{M}(p)) = 1$ (the generated video is judged harmful).

### Key Designs

1. **Regularized Dataset Construction (Stage 1 Data Preparation)**:

    - **Function**: Construct training data from meta harmful prompts.
    - **Design Motivation**: High-quality prompt pairs that are textually benign yet video-harmful are required to initialize the generator.
    - **Mechanism**: For each harmful seed prompt $p_s$, an LLM performs temporal rewriting under three rules:
        - **Temporal Deconstruction**: Decompose the harmful instruction into discrete, temporally ordered static event descriptions.
        - **Sequential Enforcement**: Insert temporal connectives ("first," "two seconds later") to ensure strict chronological progression.
        - **Temporal-Spatial Synthesis**: Ensure that harmfulness is absent from any individual description and emerges only from temporal composition.
    - **Data Selection**: Only samples satisfying $\Phi_P(p_t)=0 \wedge \Phi_V(\mathcal{M}(p_t))=1$ are retained.

2. **Initial Generator Training (Stage 2)**:

    - **Function**: Initialize the base LLM on the constructed dataset.
    - **Design Motivation**: Equip the generator with a coarse distribution of the dataset so that it can produce initial adversarial prompts.
    - **Mechanism**: Autoregressive NLL loss $\mathcal{L}_{Ini} = -\mathbb{E}_{(p_s,p_t)\sim \mathbf{D}_p} \log p(p_t|p_s, I)$.
    - Built on Llama-3 with LoRA fine-tuning.

3. **Temporal-aware Online Preference Learning (Stage 3, Core)**:

    - **Function**: Further optimize the generator to produce textually safe prompts that nonetheless carry harmful semantics at the video level.
    - **Design Motivation**: Initial training only captures a coarse distribution; interaction with the actual T2V model is required for fine-grained alignment.
    - **Prompt Space Optimization** — reward function $\mathbf{R}_{pmt}$:
        - Safety term $(1 - \mathbf{g}_t(p_t))$: encourages prompts to pass a hate-speech detector.
        - Pattern alignment term $\frac{\mathbf{g}_r(p_t)+1}{2}$: encourages prompts to align (via cosine similarity) with the embedding prototype of pre-constructed temporal-style samples.
        - $\mathbf{R}_{pmt} = \alpha_1 \cdot (1 - \mathbf{g}_t(p_t)) + \alpha_2 \cdot \frac{\mathbf{g}_r(p_t)+1}{2}$
    - **Temporal-Spatial Consistency** — reward function $\mathbf{R}_{con}$:
        - The generated video is decomposed into a frame sequence; a video encoder extracts temporal features.
        - Global consistency $\mathbf{g}_{gc}$: measures alignment between the harmful semantics of the seed prompt and the global temporal content of the generated video.
        - Internal consistency $\mathbf{g}_{ic}$: measures the temporal coherence within the video itself (generation quality).
        - $\mathbf{R}_{con} = \min(\beta, \frac{\mathbf{g}_{gc} - \gamma_1}{\theta_1} + \frac{\mathbf{g}_{ic} - \gamma_2}{\theta_2})$
    - A PPO paradigm maximizes the total reward with a KL penalty to prevent over-optimization:
    - $\zeta = \mathbb{E}[\mathbf{R}_{pmt}(p_t) + \mathbf{R}_{con}(p_s, p_t) - \lambda \log \frac{G_\delta(p_t|p_s)}{G_{initial}(p_t|p_s)}]$

4. **Test Case Refinement (Iterative)**:

    - **Function**: Iteratively optimize the initial prompts produced by the generator.
    - **Design Motivation**: Generator outputs serve only as initialization; feedback-driven refinement is needed to further enhance stealth and effectiveness.
    - **Mechanism**: The refinement model (Qwen-3-VL) receives the prompt, the generated video, and feedback from $\Phi_P$ and $\Phi_V$ (including scores, explanations, and suggestions), and outputs a revised prompt $p_{t+1}$, forming a closed-loop iterative process.

### Loss & Training
- Stage 2: NLL loss initialization, 4,000 steps, batch size 8, LR $1.0 \times 10^{-5}$.
- Stage 3: PPO online RL, AdamW optimizer, LR $1.0 \times 10^{-6}$, cosine scheduler.
- Generation uses beam search with $b=16$ and a 100-token limit.

## Key Experimental Results

### Main Results — ASR on Open-Source Models

| Method | Hunyuan-Video ASR | Wan 2.2 ASR | Prompt Safety Pass Rate |
|--------|-------------------|-------------|-------------------------|
| Naive | 2.6% | 2.3% | ~98% |
| T2VSafetyBench | 40.8% | 37.2% | ~52% |
| UVD | 29.0% | 31.0% | ~90% |
| FLIRT | 57.2% | 56.4% | ~51% |
| ART | 52.6% | 49.7% | ~92% |
| **TEAR** | **82.3%** | **80.5%** | **~96%** |

### ASR on Commercial Models

| Model | ASR (Most Categories) | API/NSFW Pass Rate |
|-------|-----------------------|--------------------|
| Veo-3.1 | ≥85% | ~98% |
| Hailuo-2.3 | ≥85% | ~98% |
| Ray-2 | Slightly lower | ~98% |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| Seed-free generation | Hunyuan 79.2%, Wan 76.9% (still substantially outperforms FLIRT ~55%) |
| Effect of iterative refinement | ASR improves from 57–71% (direct generation) to 83–95% (8 refinement rounds) |
| Prompt diversity | 1-AvgSelfBLEU: 0.71–0.76; 1-Cossim: 0.69–0.73 |
| Cross-model transferability | Average ASR of 76.4% across 20 source–target pairs, majority exceeding 70% |

### Key Findings
- TEAR's ASR (82.3%) far surpasses the previous best FLIRT (57.2%), representing a **gain of 25 percentage points**.
- Safety filters in commercial T2V services are nearly ineffective against temporal composition attacks (pass rate ~98% yet ASR ≥85%).
- Cross-model transferability is remarkably strong (average 76.4%), indicating that T2V models share fundamental temporal security vulnerabilities.
- Iterative refinement is critical: ASR improves most rapidly in the first 3 rounds and converges after 8 rounds.
- The pornography category is the most difficult to attack (lowest ASR), likely due to more stringent safety filtering for this content type.

## Highlights & Insights
- **First systematic exposure of temporal-dimension security vulnerabilities in T2V models** — single frames are harmless, yet temporal composition is harmful, a severely overlooked risk.
- The three temporal rewriting rules (deconstruction, sequential enforcement, and synthesis) elegantly define the semantic space of temporal attacks.
- The dual-reward design (prompt safety + video temporal consistency) adeptly balances stealth and effectiveness.
- The safety failures of commercial models are alarming: prompt-level filters pass nearly all adversarial inputs, yet the generated videos are highly dangerous.

## Limitations & Future Work
- The method primarily adopts an attacker's perspective; corresponding defense strategies are not discussed in depth.
- The video judge $\Phi_V$ relies on the GPT-4o API, whose evaluation reliability is itself subject to uncertainty.
- The six harmful category taxonomy may be incomplete, leaving out temporal scenarios such as misinformation and privacy violations.
- The online RL stage requires a large number of calls to the target T2V model for video generation, incurring substantial computational cost.
- The paper includes examples of harmful content; although warnings are provided, the ethical discussion could be more thorough.

## Related Work & Insights
- ART and FLIRT are representative T2I red-teaming methods; this paper adapts them to the T2V setting but demonstrates that they neglect the temporal dimension.
- T2VSafetyBench is the first T2V safety benchmark, yet it relies solely on static harmful prompts.
- Implication for AI safety: safety evaluation of multimodal models must account for temporal and compositional effects across modalities — single-modality safety does not imply multimodal safety.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic definition and exploitation of temporal security vulnerabilities in T2V models; the problem formulation itself constitutes a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five T2V models (including 3 commercial), 4 baselines, 6 harmful content categories, and multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, well-structured methodology, and comprehensive experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Foundational contribution to the T2V safety field with direct implications for improving the safety of commercial models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models](when_numbers_speak_aligning_textual_numerals_and_visual_instances_in_text-to-vid.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)

</div>

<!-- RELATED:END -->
