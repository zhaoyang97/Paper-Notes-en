---
title: >-
  [Paper Note] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency
description: >-
  [ICCV 2025][Multimodal VLM][Multimodal large language model safety] This paper identifies a **Shuffle Inconsistency** between the comprehension capability and the safety capability of multimodal large language models (ML…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Multimodal large language model safety"
  - "jailbreak attack"
  - "Shuffle Inconsistency"
  - "black-box optimization"
  - "safety alignment"
date: 2026-05-08
content_hash: 69f30345b8ba0db6
---

# Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency

**Conference**: ICCV 2025
**arXiv**: [2501.04931](https://arxiv.org/abs/2501.04931)  
**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei (Beihang University)
**Area**: Multimodal VLM
**Keywords**: Multimodal large language model safety, jailbreak attack, Shuffle Inconsistency, black-box optimization, safety alignment

## TL;DR

This paper identifies a **Shuffle Inconsistency** between the comprehension capability and the safety capability of multimodal large language models (MLLMs)—models can understand shuffled harmful instructions, yet their safety mechanisms fail to defend against them. Building on this finding, the authors propose SI-Attack, a query-based black-box jailbreak method that achieves substantially higher attack success rates on both open-source and closed-source commercial models.

## Background & Motivation

Multimodal large language models (e.g., GPT-4o, Claude-3.5-Sonnet) have made remarkable progress in commercial applications, yet they remain vulnerable to security exploits. Jailbreak attacks, used as a red-teaming tool, aim to bypass model safety mechanisms and expose latent risks.

Key limitations of existing jailbreak methods:

**High complexity**: Most methods rely on elaborate adversarial optimization (e.g., adversarial perturbation injection) or carefully engineered multimodal prompts.

**Poor effectiveness against closed-source models**: Commercial closed-source models typically employ additional outer safety guardrails that detect harmful intent and intercept attack instructions, limiting the success rate of existing approaches.

**Lack of exploitation of capability gaps**: Prior work has largely overlooked differences between a model's comprehension capability and its safety capability.

Core finding of this paper: Prior studies have shown that MLLMs retain strong comprehension of shuffled text and images (e.g., in text/image retrieval tasks). The authors thereby raise two key questions:

- **Comprehension dimension**: Can MLLMs understand shuffled harmful text and images?
- **Safety dimension**: Can MLLMs' defense mechanisms resist shuffled harmful instructions?

Experiments reveal a surprising answer: **MLLMs can understand shuffled harmful instructions, but their safety mechanisms cannot effectively defend against them.** This inconsistency between comprehension capability and safety capability is termed "Shuffle Inconsistency."

## Method

### Overall Architecture

The core idea of SI-Attack is to exploit the Shuffle Inconsistency vulnerability by shuffling both the text and images of harmful instructions to bypass safety mechanisms, and to employ query-based black-box optimization to overcome the random instability of shuffling, selecting the most effective shuffled combinations.

Overall pipeline:
1. Split the harmful input text at the word level and randomly shuffle it.
2. Split the harmful input image at the patch level and randomly shuffle it.
3. Feed the shuffled text–image pair to the target MLLM and obtain its response.
4. Use a toxicity judge model to evaluate the harmfulness of the response.
5. If the toxicity score meets the threshold, the attack succeeds; otherwise, repeat shuffling until the maximum number of iterations is reached.

### Key Designs

**1. Text Shuffle**

The harmful text $T = [w_1, w_2, \ldots, w_n]$ is randomly shuffled at the word level:

$$T' = \text{Shuffle}_w(T)$$

Several shuffling strategies are compared: no shuffle, shuffling nouns and adjectives only, trigram shuffle, intra-trigram shuffle, BPE token-level shuffle, and full word shuffle. Results show that **full word shuffle performs best** (ASR 80.41%), as it most aggressively disrupts the pattern-matching capacity of safety mechanisms while still preserving semantic comprehensibility for the model.

**2. Image Shuffle**

The harmful image is divided into $m$ patches and randomly shuffled:

$$I' = \text{Shuffle}_p(I), \quad I = [p_1, p_2, \ldots, p_m]$$

Different patch counts are compared: 1, 4, 9, 16, 25, and 64. Results show that **4 patches performs best** (ASR 80.41%). Too many patches impair model comprehension, while too few fail to effectively bypass safety mechanisms.

**3. Query-Based Black-Box Optimization**

Naive random shuffling is unstable, and not every shuffled result successfully circumvents defenses. An iterative optimization mechanism is therefore introduced:

- ChatGPT-3.5 is used as the toxicity judge model $\mathcal{J}$.
- Toxicity scores range from 1 to 5 (1 = safe, 5 = highly harmful).
- Attack success threshold: $S_\tau = 4$.
- Maximum query iterations: 10.
- At each iteration, the text and image are re-shuffled randomly; toxicity is evaluated and the loop terminates if the threshold is met.

$$\text{ASR} = \frac{\text{sum}\{\mathcal{J}(I, y) \geq S_\tau\}}{N_{\text{total}}}$$

### Loss & Training

This method does not involve conventional gradient-based loss optimization. Instead, it relies on **query-feedback-driven black-box optimization**. The optimization objective is to maximize the toxicity score of the target model's response, using feedback to select the most effective shuffled combinations. At each iteration: if the toxicity score of the current shuffled input's response $\geq S_\tau$, the attack succeeds; otherwise, the next random shuffle is attempted.

## Key Experimental Results

### Main Results: Three Benchmark Evaluations

Evaluated models include 4 open-source models (LLaVA-NEXT, MiniGPT-4, InternVL-2, VLGuard) and 4 closed-source models (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Qwen-VL-Max).

**MM-safetybench (with typography) Attack Success Rate (ASR%)**:

| Model | Baseline Attack | SI-Attack | Gain |
|-------|----------------|-----------|------|
| LLaVA-NEXT | 43.99% | 62.68% | +18.69% |
| MiniGPT-4 | 27.20% | 62.44% | +35.24% |
| InternVL-2 | 40.30% | 71.01% | +30.71% |
| VLGuard | 9.52% | 40.77% | +31.25% |
| GPT-4o | 20.77% | 68.57% | +47.80% |
| Claude-3.5-Sonnet | 7.50% | 47.20% | +39.70% |
| Gemini-1.5-Pro | 21.07% | 71.25% | +50.18% |
| Qwen-VL-Max | 33.04% | 68.63% | +35.59% |

**SafeBench (Figstep) Attack Success Rate (ASR%)**:

| Model | Figstep | SI-Attack | Gain |
|-------|---------|-----------|------|
| LLaVA-NEXT | 44.40% | 74.00% | +29.60% |
| InternVL-2 | 38.60% | 82.60% | +44.00% |
| GPT-4o | 11.80% | 59.20% | +47.40% |
| Claude-3.5-Sonnet | 29.40% | 48.60% | +19.20% |
| Gemini-1.5-Pro | 50.60% | 80.20% | +29.60% |

### Ablation Study

**Contribution of image and text shuffle components** (GPT-4o, MM-safetybench 01-IA subset):

| Setting | Toxic Score | ASR(%) |
|---------|------------|--------|
| Original image & text | 1.64 | 13.40% |
| Image shuffle only | 2.51 | 35.05% |
| Text shuffle only | 3.69 | 67.01% |
| Both shuffled | 3.96 | 80.41% |

**Necessity of query optimization**:

| Setting | Toxic Score | ASR(%) |
|---------|------------|--------|
| Original input | 1.64 | 13.40% |
| Random shuffle (no optimization) | 2.65 | 28.87% |
| Optimized shuffle | 3.96 | 80.41% |

**Effect of maximum iteration count**:

| Iterations | Toxic Score | ASR(%) |
|-----------|------------|--------|
| 1 | 2.65 | 28.87% |
| 5 | 3.75 | 69.07% |
| 10 | 3.96 | 80.41% |
| 20 | 4.01 | 81.44% |

### Key Findings

1. **Text shuffle is more effective than image shuffle**: Text-only shuffle ASR (67.01%) far exceeds image-only shuffle (35.05%), indicating that safety vulnerabilities on the text side are more severe in MLLMs.
2. **Query optimization is critical**: ASR improves from 28.87% to 80.41% after optimization, demonstrating that not all random shuffles are effective and selection is necessary.
3. **Convergence at 10 iterations**: 20 iterations yield only marginal improvement (81.44% vs. 80.41%).
4. **Effective across model scales**: InternVL-2 variants at 4B/8B/26B all achieve approximately 70% ASR, indicating the method is scale-agnostic.
5. **Robust against PPL-based detectors**: SI-Attack maintains 71.13% ASR even under perplexity-based defense.

## Highlights & Insights

1. **Discovery of a fundamental security vulnerability**: Shuffle Inconsistency reveals a deep disconnect between the comprehension capability and safety capability of MLLMs—safety alignment training fails to cover the space of shuffled harmful instructions, a finding with significant implications for AI safety research.
2. **Exceptional simplicity**: The method requires no adversarial perturbation optimization, no white-box model access, and no elaborately crafted prompts; high attack effectiveness is achieved through random shuffling combined with query optimization alone.
3. **Strong performance against closed-source commercial models**: ASR on GPT-4o increases from 20.77% to 68.57%, and on Gemini-1.5-Pro from 21.07% to 71.25%, demonstrating that the outer safety guardrails of commercial models are also susceptible to this vulnerability.
4. **PCA visualization provides mechanistic explanation**: Visualization of hidden states in open-source models clearly shows that models form distinct internal representations for original and shuffled inputs, confirming that safety alignment training does not cover this distribution.
5. **Philosophical insight—strength as weakness**: When safety capability fails to keep pace with strong comprehension capability, comprehension itself becomes an exploitable weakness.

## Limitations & Future Work

1. **Relatively simple method, straightforward to defend against**: Once this vulnerability is identified, defenders should be able to mitigate it by incorporating shuffled samples into safety alignment training.
2. **Reliance on an external toxicity judge**: The method requires ChatGPT-3.5 for toxicity evaluation, increasing cost and dependence on an external API.
3. **Fixed shuffling strategies**: Only word-level and patch-level shuffling are explored; more flexible granularities (sentence-level, character-level, semantic-level) remain underexplored.
4. **Evaluation limitations**: Toxicity scores are automatically assessed by ChatGPT-3.5, which may deviate from human judgment.
5. **Lack of evaluation against recent defenses**: Performance under newer defense mechanisms such as adversarial training and input preprocessing is unknown.
6. **Ethical risk**: The paper releases a complete attack methodology, which may be subject to malicious exploitation.

## Related Work & Insights

**Jailbreak attack methods**:
- **FigStep** [Gong et al., 2023]: Embeds harmful text in typographic images to exploit OCR capabilities for jailbreaking.
- **MM-safetybench** [Liu et al., 2023]: Generates query-relevant images paired with typography to conduct attacks.
- **HADES** [Li et al., 2024]: Conceals and amplifies harmful intent through carefully crafted images.
- Compared to the above, SI-Attack is simpler and more effective against closed-source models.

**Defense methods**:
- **LLaMA Guard**: Fine-tunes LLaMA to detect harmful intent.
- **VLGuard**: Constructs a visual-language safety instruction dataset to fine-tune MLLMs.
- This paper finds that VLGuard's safety fine-tuning also fails to resist SI-Attack.

**Insights**:
1. Safety alignment training must cover a broader input distribution, including diverse variants and perturbation forms.
2. Comprehension and safety capabilities should be trained jointly to ensure mutual consistency.
3. Future defenses may consider input canonicalization at inference time—restoring shuffled inputs to standard form before safety checking.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | ⭐⭐⭐⭐ | Identifies the novel Shuffle Inconsistency vulnerability from a unique perspective |
| Technical Depth | ⭐⭐⭐ | Method is simple, but analysis is thorough (PCA visualization, multi-dimensional ablation) |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Three benchmarks, 8 models, detailed ablation, adaptive attack experiments—highly comprehensive |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, compelling motivation, rich figures and tables |
| Value | ⭐⭐⭐⭐ | Important warning signal for AI safety research; method is simple and easily reproducible |
| **Overall** | **⭐⭐⭐⭐** | An outstanding security analysis and red-teaming work with insightful findings and rigorous experiments |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)

</div>

<!-- RELATED:END -->
