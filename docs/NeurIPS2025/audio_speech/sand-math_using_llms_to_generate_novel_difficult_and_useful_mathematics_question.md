---
title: >-
  [Paper Note] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers
description: >-
  [NeurIPS 2025][Audio & Speech][Mathematical Reasoning] This paper proposes SAND-Math, a fully automated synthetic mathematics question generation pipeline that requires no seed dataset. By employing Difficulty Hiking to systematically increase problem difficulty, augmenting the LIMO baseline with as few as 500 problems yields a 4.39pp improvement on AIME25.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Mathematical Reasoning
  - Synthetic Data
  - Difficulty Hiking
  - Data Quality
  - Post-Training
date: 2026-05-08
content_hash: 0d1a5babbc92f270
---

# SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers

**Conference**: NeurIPS 2025
**arXiv**: [2507.20527](https://arxiv.org/abs/2507.20527)
**Code**: [HuggingFace Dataset](https://huggingface.co/datasets/amd/SAND-MATH)
**Area**: Audio/Speech (LLM Mathematical Reasoning)
**Keywords**: Mathematical Reasoning, Synthetic Data, Difficulty Hiking, Data Quality, Post-Training

## TL;DR

This paper proposes SAND-Math, a fully automated synthetic mathematics question generation pipeline that requires no seed dataset. By employing Difficulty Hiking to systematically increase problem difficulty, augmenting the LIMO baseline with as few as 500 problems yields a 4.39pp improvement on AIME25.

## Background & Motivation

**Background**: Frontier reasoning models such as DeepSeek-R1, o3, and Gemini 2.5 Pro achieve remarkable performance on mathematical benchmarks, yet their training data and methodologies remain undisclosed. Research from LIMO and S1 suggests that reasoning capability depends more on data quality—particularly the inclusion of high-difficulty problems—than on scale.

**Limitations of Prior Work**: (a) Datasets such as NuminaMath and OpenR1 rely on manually curated competition problems, making them labor-intensive and limited in scale; (b) synthetic methods including KPDDS, MetaMathQA, and WizardMath primarily remix existing GSM8K/MATH training sets, making it difficult to surpass the difficulty ceiling of the seed data; (c) MATH2 requires human expert involvement and thus cannot scale.

**Key Challenge**: The supply of high-quality, high-difficulty mathematical training data is severely insufficient, and existing synthetic methods are constrained by the difficulty ceiling of their seed data.

**Goal**: To construct a fully automated pipeline that generates high-difficulty mathematics problems from scratch while ensuring correctness, novelty, and progressive difficulty increase.

**Key Insight**: Leveraging the "metacognitive" capacity of state-of-the-art LLMs—their implicit ability to model the characteristics of high-difficulty mathematical problems and generate new instances thereof.

**Core Idea**: LLMs themselves can generate high-difficulty math problems from minimal prompts, with difficulty further elevated through Difficulty Hiking via cross-domain concept fusion.

## Method

### Overall Architecture

A five-stage pipeline: generation → correctness filtering → deduplication and decontamination → difficulty filtering and scoring → novelty filtering, with an optional Difficulty Hiking stage.

### Key Designs

#### 1. Problem and Solution Generation
- A teacher model $\mathcal{M}_{\text{teacher}}$ (DeepSeek-R1) directly generates problem $q_i$ and $k=2$ independent solutions from an empirically optimized prompt.
- An initial set of $\mathcal{D}_1 = 23{,}437$ problems is obtained.

#### 2. Self-Consistency Filtering
- Only problems whose $k$ solutions agree are retained: $a'_{i1} = a'_{i2} = \cdots = a'_{ik}$
- 17,578 problems (~74%) are preserved.

#### 3. Deduplication + Decontamination
- **Semantic deduplication**: semhash framework with a similarity threshold of 0.99; 1,293 entries (7.3%) removed.
- **Decontamination**: a retrieval model identifies top-5 candidates, followed by semantic verification via a judge model; only 4 problems are removed.
- 16,281 problems remain.

#### 4. Difficulty Filtering + Scoring
- **Performance-based filtering**: only problems that the solver model $\mathcal{M}_{\text{solver}}$ (Qwen2.5-32B) answers incorrectly are retained → 9,211 problems (56.6%).
- **Difficulty scoring**: judge model $\mathcal{M}_{\text{judge}}$ (Llama-3.3-70B) assigns scores on a 1–10 scale (calibrated against AoPS AIME problems).

#### 5. Novelty Filtering
- Web search combined with semantic similarity (gte-Qwen2-7B embeddings) at threshold $\tau=0.85$.
- 4% of problems are removed, yielding a final set of **8,842 SAND-Math problems**.

#### 6. Difficulty Hiking (Core Contribution)
- The teacher model is re-prompted to rewrite problems; inputs include the original problem, its difficulty score, and a mandatory fusion of same-branch theorems with cross-domain concepts.
- A single iteration raises the mean difficulty from 5.02 → 5.98.
- Mid-to-low difficulty problems (4.0–5.0) are transformed into high-difficulty problems (6.0–8.0).

### Loss & Training

- Student model: Qwen2.5-32B-Instruct
- Full-parameter SFT using the LLaMA-Factory framework
- Learning rate 5e-6, 10 epochs, cosine scheduler
- DeepSpeed ZeRO-3, 8× AMD MI300X GPUs
- Evaluation: pass@1 ($n=16$, temp=0.7) for AIME/AMC; greedy decoding for MATH500

## Key Experimental Results

### Main Results — Augmentation Performance Comparison

| Training Data | # Samples | AIME25 | AIME24 | AMC | MATH500 | Avg |
|---|---|---|---|---|---|---|
| LIMO Baseline | 817 | 44.50 | 56.30 | 91.41 | 93.80 | 71.50 |
| LIMO + SAND-Math | 817+500 | **48.89** | **57.92** | **92.50** | **94.00** | **73.32** |
| LIMO + openr1_math | 817+500 | 47.71 | 56.04 | 92.50 | 93.80 | 72.51 |
| LIMO + MetamathQA | 817+500 | 31.04 | 46.25 | 47.24 | 56.40 | 45.23 |
| LIMO + OpenMathInstruct | 817+500 | 18.13 | 38.96 | 64.53 | 72.40 | 48.50 |

SAND-Math outperforms the second-best synthetic dataset (MetamathQA) by **17.85pp** on AIME25.

### Ablation Study — Effect of Difficulty Hiking

| Dataset | AIME25 | AIME24 | AMC24 | MATH500 | Avg |
|---|---|---|---|---|---|
| LIMO + Base (1500) | 46.38 | 59.09 | 92.71 | 93.6 | 72.94 |
| LIMO + DH (1500) | **49.23** | **60.55** | **93.17** | **94.6** | **74.39** |
| LIMO + DH_w_LF (1500) | 49.23 | 60.83 | 93.28 | 93.0 | 74.08 |

Difficulty Hiking improves the average score from 72.94 → 74.39 (+1.45pp); longer reasoning chains also contribute positively.

### Key Findings

1. SAND-Math fine-tuned in isolation (69.10) already approaches the human-curated openr1_math (70.27).
2. In the augmentation setting, SAND-Math outperforms all datasets (+0.81pp over openr1_math).
3. Difficulty scores are concentrated in the 6–8 range, substantially higher than other synthetic datasets (3–5).
4. The overall pipeline yield rate is ~35% (from 23K → 8.8K problems).

## Highlights & Insights

- **No seed data required**: Problems are generated entirely from scratch without relying on existing training sets such as GSM8K or MATH.
- **Elegant design of Difficulty Hiking**: The approach leverages the metacognitive capacity of LLMs to systematically increase problem complexity through cross-domain concept fusion.
- **High sample efficiency**: As few as 500 SAND-Math problems suffice to significantly augment a strong baseline.
- **Comprehensive quality assurance pipeline**: Self-consistency → deduplication → decontamination → difficulty filtering → novelty filtering, with each step quantitatively validated.

## Limitations & Future Work

1. The ceiling on output quality is bounded by the teacher model's capacity (DeepSeek-R1 yields 41.9%; GPT-OSS 120B achieves 74.0%).
2. Only 500 samples are used for proof-of-concept validation; large-scale post-training remains to be explored.
3. Difficulty Hiking is demonstrated for only a single iteration; the effect of multiple iterations is unknown.
4. The approach is limited to the mathematical domain; generalization to scientific or code reasoning tasks warrants further investigation.

## Related Work & Insights

- **LIMO** (Ye et al., 2025): A "less is more" philosophy for reasoning data.
- **S1** (Muennighoff et al., 2025): Simple test-time scaling.
- **MetaMathQA** (Yu et al., 2023): Bootstrapping math problems from seed data.
- **Insights**: High quality outweighs large scale; metacognitive capabilities can be systematically exploited for data generation.

## Rating

⭐⭐⭐⭐ (4/5)
The method is concise and effective, and the Difficulty Hiking concept is novel. However, the experimental scale is limited (validated with only 500 samples), and the approach exhibits strong dependence on the teacher model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](../../ICLR2026/audio_speech/the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)

</div>

<!-- RELATED:END -->
