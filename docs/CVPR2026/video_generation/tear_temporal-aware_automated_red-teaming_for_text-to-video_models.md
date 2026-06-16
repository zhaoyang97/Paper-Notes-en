---
title: >-
  [Paper Note] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models
description: >-
  [CVPR 2026][Video Generation][Paper Note] The study proposes TEAR, the first automated red-teaming framework targeting temporal-dimensional vulnerabilities in T2V models. By utilizing a two-stage optimized temporal-aware test generator and an iterative refinement model, it generates prompts that are textually harmless but trigger harmful videos through tempora
tags:
  - CVPR 2026
  - Video Generation
date: 2026-05-08
content_hash: d7df91c8fbd816f4
---
# TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models

**Conference**: CVPR 2026  
**arXiv**: [2511.21145](https://arxiv.org/abs/2511.21145)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Text-to-Video Safety, Automated Red-teaming, Temporal-aware, Adversarial Prompt Generation, AI Safety

## TL;DR
The study proposes TEAR, the first automated red-teaming framework targeting temporal-dimensional vulnerabilities in T2V models. By utilizing a two-stage optimized temporal-aware test generator and an iterative refinement model, it generates prompts that are textually harmless but trigger harmful videos through temporal dynamics, achieving 80%+ attack success rates on both open-source and commercial T2V models.

## Background & Motivation
Text-to-Video (T2V) models (e.g., Veo, Hailuo, Wan) can generate high-quality, temporally coherent videos, but they can also be triggered to produce harmful content, making safety assessment critical.

**Key Challenge**: Existing red-teaming methods primarily target static images and text generation, failing to capture the unique **temporal dynamic safety risks** in video generation. The harmfulness of a video may not exist in any single frame but emerges from the temporal combination of frame sequences—for instance, describing "a person picking up a knife" and "another person falling down" is individually harmless, but their temporal connection can constitute a violent scene.

**Limitations of Prior Work**:
1. LLM red-teaming methods (e.g., CuriDial, FLIRT) focus on textual adversarial attacks, completely ignoring the video temporal dimension.
2. Image red-teaming methods (e.g., ART, Groot) treat video as a sequence of independent frames, failing to evaluate risks arising from temporal combinations.
3. T2VSafetyBench, the first T2V safety benchmark, only uses static harmful prompts, resulting in limited attack success rates.
4. Introducing temporal information significantly expands the search space, posing new technical challenges.

**Core Idea**: Red-teaming prompt generation is modeled as an MDP. An LLM generator is optimized in two stages: initialization on structured data, followed by refinement through online preference learning that combines prompt safety rewards and video temporal consistency rewards, complemented by an iterative refinement model to continuously enhance attack effectiveness.

## Method

### Overall Architecture
TEAR consists of three components:
1. **Temporal-aware Test Generator** — The core component, based on a trained LLM, generates temporally restructured adversarial prompts from seed prompts.
2. **Refiner Model** — Based on a multimodal LLM (Qwen-3-VL), it iteratively improves prompts based on judgment feedback.
3. **Target T2V Model** — The video generation model under test.

Red-teaming goal: Discover a prompt set $\mathcal{P}_v^*$ such that $\Phi_P(p) = 0$ (text judged safe) and $\Phi_V(\mathcal{M}(p)) = 1$ (generated video judged harmful).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Harmful Seed Prompt"] --> B["Structured Dataset Construction<br/>Temporal Deconstruction → Sequential Enforcement → Spatio-temporal Synthesis<br/>Retain samples with safe text and harmful video"]
    B --> C["Initial Generator Training<br/>Llama-3 + LoRA, NLL Imitation Learning"]
    C --> D["Temporal-aware Online Preference Learning<br/>PPO Dual Rewards: Prompt Safety + Video Temporal Consistency"]
    D --> E["Generate Adversarial Prompts"]
    E --> F["Target T2V Model Generates Videos"]
    F -->|"Text not safe or video not harmful, with feedback"| G["Test Case Refinement<br/>Qwen-3-VL modifies prompt based on scores and advice"]
    G --> E
    F -->|"Attack Successful"| H["Adversarial prompt set: textually harmless but triggers harmful videos"]
```

### Key Designs

**1. Structured Dataset Construction: Transforming the "Safe Text, Harmful Video" paradigm into learnable samples**

The premise of the attack is to let the generator see enough prompt pairs where the text is harmless but the sequence depicts violent scenes. Since such samples do not exist in current datasets, the authors use an LLM to rewrite each harmful seed prompt $p_s$ into an adversarial version $p_t$ based on three rules: **Temporal Deconstruction** splits a harmful instruction into several temporally separated, individually static harmless event descriptions; **Sequential Enforcement** inserts time connectors like "first" or "two seconds later" to fix these segments onto a strict timeline; **Spatio-temporal Synthesis** ensures that the harmful semantics do not fall into any single sentence but only emerge when the frames are combined. For example, the seed "a person stabs another with a knife" is rewritten as "First, a person picks up a knife in the kitchen; two seconds later, another person slowly collapses holding their abdomen"—each sentence passes text safety checks, but the video sequence restores the violent act. Only samples satisfying $\Phi_P(p_t)=0 \wedge \Phi_V(\mathcal{M}(p_t))=1$ are retained as training data.

**2. Initial Generator Training: Using imitation learning to instill rewriting capabilities into the LLM**

With high-quality prompt pairs, the first step is not direct reinforcement learning but rather letting the generator learn the basic "pattern" of rewriting. The authors add LoRA to a Llama-3 base and fit the data distribution using an autoregressive NLL loss:

$$\mathcal{L}_{Ini} = -\mathbb{E}_{(p_s,p_t)\sim \mathbf{D}_p} \log p(p_t|p_s, I)$$

This step ensures the generator learns the rough distribution of the seed data and can stably output correctly formatted temporal prompts, providing a solid starting point for subsequent online optimization.

**3. Temporal-aware Online Preference Learning: Forcing effective attacks using prompt safety + video temporal dual rewards**

Imitation learning only learns prompts that "look" correct, but whether they effectively bypass text filters and trigger harmful videos requires interaction with actual T2V models. This is modeled as an MDP, with PPO optimizing two rewards. The prompt space reward $\mathbf{R}_{pmt}$ includes a safety term $(1-\mathbf{g}_t(p_t))$ to encourage passing hate speech detectors and a pattern alignment term $\frac{\mathbf{g}_r(p_t)+1}{2}$ to maintain alignment with pre-constructed temporal style prototypes (cosine similarity):

$$\mathbf{R}_{pmt} = \alpha_1 \cdot (1 - \mathbf{g}_t(p_t)) + \alpha_2 \cdot \frac{\mathbf{g}_r(p_t)+1}{2}$$

The temporal space consistency reward $\mathbf{R}_{con}$ measures whether generated videos are split into frame sequences with temporal features: global consistency $\mathbf{g}_{gc}$ measures if seed harmful semantics are realized, and internal consistency $\mathbf{g}_{ic}$ measures video coherence (to avoid sacrificing quality for the attack):

$$\mathbf{R}_{con} = \min\Big(\beta, \frac{\mathbf{g}_{gc} - \gamma_1}{\theta_1} + \frac{\mathbf{g}_{ic} - \gamma_2}{\theta_2}\Big)$$

PPO maximizes the sum of both rewards with a KL penalty to prevent over-optimization:

$$\zeta = \mathbb{E}\Big[\mathbf{R}_{pmt}(p_t) + \mathbf{R}_{con}(p_s, p_t) - \lambda \log \frac{G_\delta(p_t|p_s)}{G_{initial}(p_t|p_s)}\Big]$$

**4. Test Case Refinement: A closed-loop for iterative convergence**

The prompts output by the generator are often incomplete. An iterative refiner based on Qwen-3-VL creates a closed loop: it takes the current prompt, the generated video, and the scores/explanations from $\Phi_P$ and $\Phi_V$ to produce the next version $p_{t+1}$. This process incrementally raises the ASR (from 57-71% in direct generation to 83-95% after 8 rounds in experiments).

### Loss & Training
- Stage 2: NLL loss initialization, 4000 steps, batch 8, LR $1.0 \times 10^{-5}$
- Stage 3: PPO online RL, AdamW, LR $1.0 \times 10^{-6}$, cosine scheduler
- Generation uses beam search, $b=16$, 100 token limit

## Key Experimental Results

### Main Results — ASR on Open-source Models

| Method | Hunyuan-Video ASR | Wan 2.2 ASR | Prompt Safety Pass Rate |
|------|-------------------|-------------|----------------|
| Naive | 2.6% | 2.3% | ~98% |
| T2VSafetyBench | 40.8% | 37.2% | ~52% |
| UVD | 29.0% | 31.0% | ~90% |
| FLIRT | 57.2% | 56.4% | ~51% |
| ART | 52.6% | 49.7% | ~92% |
| **TEAR** | **82.3%** | **80.5%** | **~96%** |

### Commercial Model ASR

| Model | ASR (Major Categories) | API/NSFW Pass Rate |
|------|---------------|----------------|
| Veo-3.1 | ≥85% | ~98% |
| Hailuo-2.3 | ≥85% | ~98% |
| Ray-2 | Slightly lower | ~98% |

### Ablation Study

| Analysis Dimension | Result |
|---------|------|
| Seed-free Generation | Hunyuan 79.2%, Wan 76.9% (Significantly leads FLIRT ~55%) |
| Iterative Refinement Effect | ASR improves from 57-71% (direct) to 83-95% (8 rounds) |
| Prompt Diversity | 1-AvgSelfBLEU: 0.71-0.76, 1-Cossim: 0.69-0.73 |
| Cross-model Transferability | Avg ASR 76.4% across 20 source-target pairs, mostly >70% |

### Key Findings
- TEAR's ASR (82.3%) far exceeds the previous best FLIRT (57.2%), a **25 percentage point gain**.
- Safety filters of commercial T2V services are nearly ineffective against temporal combination attacks (Pass rate ~98% but ASR ≥85%).
- High cross-model transferability (76.4% average) suggests T2V models share fundamental temporal safety vulnerabilities.
- Iterative refinement is critical: ASR rises fastest in the first 3 rounds and converges after 8.
- Pornography is the most difficult category to attack (lowest ASR), likely due to stricter safety filtering for this category.

## Highlights & Insights
- **First revelation of temporal-dimensional safety vulnerabilities in T2V models**: Individual frames are harmless, but temporal combinations are harmful—a severely overlooked risk.
- Three temporal rewriting rules (deconstruction, sequential enforcement, synthesis) elegantly define the semantic space of temporal attacks.
- The dual reward design (prompt safety + video temporal consistency) effectively balances stealth and effectiveness.
- The failure of commercial models is alarming: prompt filters allow nearly all inputs, yet generated videos are highly dangerous.

## Limitations & Future Work
- The method focuses on the attacker's perspective; corresponding defense strategies are not discussed in depth.
- The video judgment system $\Phi_V$ relies on the GPT-4o API, whose evaluation reliability contains uncertainty.
- The categorization into 6 harmful types may be incomplete; scenarios like misinformation or privacy violations are not covered.
- The online RL stage requires frequent calls to target T2V models, resulting in high computational costs.

## Related Work & Insights
- ART and FLIRT represent T2I red-teaming, which this work adapts to T2V while proving they ignore the temporal dimension.
- T2VSafetyBench was the first T2V safety benchmark but only used static harmful prompts.
- Insight for AI Safety: Safety assessment of multimodal models must consider inter-modal temporal/compositional effects; single-model safety does not guarantee multimodal safety.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically defines and attacks temporal safety vulnerabilities for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 T2V models (3 commercial), 4 baselines, 6 harmful categories.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and methodology.
- Value: ⭐⭐⭐⭐⭐ Foundational contribution to T2V safety, directly impacting commercial safety improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TempoControl: Temporal Attention Guidance for Text-to-Video Models](tempocontrol_temporal_attention_guidance_for_text-to-video_models.md)
- [\[CVPR 2026\] PoseAnything: General Pose-guided Video Generation with Part-aware Temporal Coherence](poseanything_general_pose-guided_video_generation_with_part-aware_temporal_coher.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models](ref4d-videobench_four-dimensional_reference-based_evaluation_of_text-to-video_ge.md)

</div>

<!-- RELATED:END -->
