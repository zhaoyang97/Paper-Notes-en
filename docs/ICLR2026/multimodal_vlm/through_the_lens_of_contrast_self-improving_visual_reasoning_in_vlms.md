---
title: >-
  [Paper Note] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs
description: >-
  [ICLR 2026][Multimodal VLM][visual reasoning] This paper proposes VC-STaR (Visual Contrastive Self-Taught Reasoner), motivated by the observation that *VLMs perceive visual content more accurately when comparing two similar images*. A contrastive self-improvement framework is designed: contrastive VQA pairs are constructed to elicit more faithful visual analysis from the model, and an LLM integrates this contrastive analysis into reasoning chains, yielding the high-quality visual reasoning dataset VisCoR-55K. Fine-tuning on this dataset achieves +5.7% on MMVP and +3.2% on Hallusion.
tags:
  - ICLR 2026
  - Multimodal VLM
  - visual reasoning
  - self-improving
  - visual contrast
  - hallucination mitigation
  - contrastive VQA pairs
date: 2026-05-08
content_hash: bace7d36452543d5
---

# Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs

**Conference**: ICLR 2026
**arXiv**: [2603.02556](https://arxiv.org/abs/2603.02556)
**Code**: [https://github.com/zhiyupan42/VC-STaR](https://github.com/zhiyupan42/VC-STaR)
**Area**: Multimodal VLM / Visual Reasoning
**Keywords**: visual reasoning, self-improving, visual contrast, hallucination mitigation, contrastive VQA pairs

## TL;DR
This paper proposes VC-STaR (Visual Contrastive Self-Taught Reasoner), motivated by the observation that *VLMs perceive visual content more accurately when comparing two similar images*. A contrastive self-improvement framework is designed: contrastive VQA pairs are constructed to elicit more faithful visual analysis from the model, and an LLM integrates this contrastive analysis into reasoning chains, yielding the high-quality visual reasoning dataset VisCoR-55K. Fine-tuning on this dataset achieves +5.7% on MMVP and +3.2% on Hallusion.

## Background & Motivation

**Background**: Vision-Language Models (VLMs), as extensions of large language models, have demonstrated strong multimodal reasoning capabilities. In the text-only domain, self-improvement methods (e.g., STaR, Self-Refine) — which enhance reasoning by having models iteratively refine their own reasoning chains — have proven effective and scalable paradigms for reasoning improvement.

**Limitations of Prior Work**: Directly transferring text-domain self-improvement methods to VLMs faces a fundamental challenge — visual hallucination. VLM-generated reasoning chains frequently contain hallucinations (describing non-existent content or misinterpreting visual information), whereas existing text-centric self-improvement frameworks focus solely on textual coherence and final answer correctness, offering no mechanism to verify or correct visual hallucinations embedded in reasoning. Worse, such methods may degrade into *speculative reasoning*, allowing textual priors to override genuine visual evidence.

**Key Challenge**: Self-improvement requires high-quality reasoning chains as training data, yet VLM-generated chains are contaminated by visual hallucinations — forming a "garbage in, garbage out" vicious cycle. The core problem is: how can visual hallucinations in VLM reasoning chains be corrected to enable high-quality visual reasoning data generation?

**Goal**: (1) Design a reliable visual hallucination correction mechanism to enable VLM self-improvement; (2) Construct a large-scale, high-quality visual reasoning dataset; (3) Significantly improve VLM visual reasoning through fine-tuning.

**Key Insight**: The authors identify an intriguing phenomenon — *VLMs see more accurately when contrasting*. When presented with a contrastive VQA pair (two visually similar images with different answers paired with semantically related questions), VLMs capture fine-grained visual cues more precisely, thereby correcting original hallucinations. Statistical analysis shows that the contrastive setting not only corrects more errors but also introduces fewer new ones.

**Core Idea**: Leverage the inherent contrastive capability of VLMs to correct visual hallucinations in their own reasoning chains, enabling self-guided improvement of visual reasoning.

## Method

### Overall Architecture
VC-STaR consists of two main pipelines: (1) **Contrastive VQA Pair Curation** — for each sample drawn from diverse VQA datasets, a contrastive counterpart is retrieved that is visually similar and semantically related in question; (2) **Contrasting and Rethinking** — a three-step process generates high-quality reasoning chains: the VLM first produces an initial reasoning chain for a single image (*thinking*), then performs contrastive analysis over both images (*contrasting*), and finally an LLM refines the initial chain based on the contrastive analysis (*rethinking*). The resulting VisCoR-55K dataset is used for supervised fine-tuning. No contrastive process is required at inference time; standard VLM inference is adopted.

### Key Designs

1. **Contrastive VQA Pair Curation**:

    - **Function**: For each VQA sample, retrieve a contrastive counterpart satisfying specific conditions to elicit more accurate visual perception through comparison.
    - **Mechanism**: A three-stage pipeline. **Data Collection**: Samples are collected from 21 VQA datasets spanning five categories (reasoning, chart, math, general, OCR) to ensure diversity. **Contrastive Sample Retrieval**: GTE text embeddings compute question similarity; an ID-based visual metric learning model computes image similarity. Sample $j$ is selected as the contrastive counterpart of sample $i$ when $\gamma(e_i^v, e_j^v) < \phi_v$ and $\gamma(e_i^q, e_j^q) < \phi_q$ are simultaneously satisfied. **Difficulty Sampling**: Samples are categorized into three difficulty levels — easy (VLM answers correctly without contrast), medium (VLM initially fails but succeeds with contrastive prompting), and hard (contrast cannot correct the error); only medium-difficulty samples are retained.
    - **Design Motivation**: Contrastive pairs must satisfy three key properties: questions must be semantically similar (providing a semantic anchor), images must be visually similar yet non-trivial (forcing fine-grained discrimination), and questions must require reasoning (rather than simple factual lookup). Retaining only medium-difficulty samples is justified because easy samples require no deep reasoning (potentially encouraging over-thinking), while hard samples cannot be corrected even with contrast (quality cannot be guaranteed).

2. **Contrasting and Rethinking**:

    - **Function**: Leverage contrastive VQA pairs to revise hallucination-prone reasoning chains into more faithful versions.
    - **Mechanism**: A three-step pipeline. **Thinking**: Given target VQA sample $(v_i, q_i, a_i)$ and the correct answer as a prompt, the VLM generates an initial reasoning chain $r_i = f(v_i, q_i, a_i | \theta, \delta^t)$. **Contrasting**: The VLM simultaneously observes the target sample and the contrastive sample $(\hat{v_i}, \hat{q_i}, \hat{a_i})$, generating a contrastive analysis $c_i$ — summarizing common patterns when answers agree, or analyzing fine-grained differences when answers differ. **Rethinking**: An external LLM $\psi$ (Qwen2.5-72B) refines the initial reasoning $r_i$ using the contrastive analysis $c_i$, producing a more faithful reasoning chain $\tilde{r_i} = f(r_i, c_i | \psi, \delta^r)$. A post-processing step based on text matching filters out incorrect reasoning chains.
    - **Design Motivation**: As shown by statistics in Figure 1, the contrastive setting corrects more hallucinations than answer-hint-only prompting. Using an external LLM for the Rethinking step transfers fine-grained visual information gained from dual-image contrast into single-image reasoning chains, allowing fine-tuned models to benefit during standard single-image inference.

3. **VisCoR-55K Dataset Construction**:

    - **Function**: Produce a high-quality visual reasoning fine-tuning dataset.
    - **Mechanism**: The three-step pipeline is applied to medium-difficulty contrastive VQA pairs, generating corrected reasoning chains. After filtering, approximately 55K high-quality visual reasoning samples are obtained, spanning five domains: general VQA, reasoning, math, chart/figure, and OCR. Full-parameter SFT is performed using the LLaMA-factory framework (3 epochs, learning rate $1e-5$, batch size 256) with the visual encoder frozen.
    - **Design Motivation**: Multi-domain coverage ensures generalization; quality filtering guarantees training data reliability; full-parameter fine-tuning maximizes knowledge absorption.

### Loss & Training
Standard supervised fine-tuning (SFT) loss is used. Training runs for 3 epochs on VisCoR-55K with learning rate $1e-5$, batch size 256, and the visual encoder frozen. No contrastive pipeline is required at inference time; standard VLM inference is followed.

## Key Experimental Results

### Main Results

Base model: Qwen2.5VL-7B. Compared against self-improvement baselines and models trained on off-the-shelf visual reasoning datasets:

| Method | MMVP | Hallusion | MathVista | MathVision | MMStar | MME-RW | Avg. |
|--------|------|-----------|-----------|------------|--------|--------|------|
| Base Model | 70.0 | 53.1 | 68.4 | 24.0 | 61.8 | 55.9 | 55.5 |
| STaR (self-improve) | 73.0(+3.0) | 55.9(+2.8) | 66.9(-1.5) | 19.8(-4.2) | 58.9(-2.9) | 58.1(+2.2) | 55.4 |
| Feedback (self-improve) | 75.0(+5.0) | 53.4(+0.3) | 68.8(+0.4) | 22.1(-1.9) | 63.2(+1.4) | 56.0(+0.1) | 56.4 |
| LLaVA-CoT (dataset) | 71.7(+1.7) | 50.3(-2.8) | 68.4(+0.0) | 24.4(+0.4) | 63.1(+1.3) | 59.3(+3.4) | 56.2 |
| R1-Onevision (dataset) | 68.0(-2.0) | 55.8(+2.7) | 68.2(-0.2) | 25.4(+1.4) | 53.2(-8.6) | 46.3(-9.6) | 52.8 |
| LPT (dataset) | 74.0(+4.0) | 53.4(+0.3) | 69.2(+0.8) | 24.2(+0.2) | 64.3(+2.5) | 56.1(+0.2) | 56.9 |
| **VC-STaR (Ours)** | **75.7(+5.7)** | **56.3(+3.2)** | **69.7(+1.3)** | **25.3(+1.3)** | 62.4(+0.6) | **59.3(+3.4)** | **58.1** |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|------------|-------|
| Positive pairs only (same answer) | GQA Total: 50.6(+5.2) | Positive contrast is effective but insufficient |
| Negative pairs only (different answer) | GQA Total: 53.7(+8.3) | Negative contrast is more effective |
| Positive + Negative pairs | GQA Total: **54.7(+9.3)** | Complementary; combination is optimal |
| +20K easy samples | Hallusion: 52.2(-4.1) | Easy samples are harmful (over-thinking) |
| +40K easy samples | Hallusion: 55.7(-0.6), MMStar: 59.5(-2.9) | More easy samples cause larger degradation |
| Qwen2.5VL-33B + VC-STaR | Hallusion: 53.2(+6.3), MathVision: 21.9(+3.5) | Larger models benefit equally |
| InternVL2.5-8B + VC-STaR | Hallusion: 55.4(+7.2), MathVision: 23.4(+2.1) | Generalizes across model families |

### Key Findings
- **VC-STaR is the only method that achieves consistent positive gains across all benchmarks**: Other self-improvement methods (STaR, Feedback) improve hallucination benchmarks at the cost of mathematical ability, whereas VC-STaR yields gains across hallucination, math, and general benchmarks simultaneously, with an average improvement of 2.6%.
- **Text-only reasoning chains are ineffective**: Fine-tuning with text-only reasoning chains (Virgo) leads to severe degradation (MME-RW -26.5%), strongly demonstrating the indispensability of the visual modality in visual reasoning.
- **Negative contrastive pairs are more effective than positive ones**: Negative pairs (different answers) improve GQA by 8.3%, substantially outperforming positive pairs (5.2%), as differing answers produce stronger semantic contrast.
- **Easy samples are harmful**: Incorporating easy samples degrades performance, likely because simple questions do not require deep reasoning, causing the model to learn an undesirable over-thinking pattern.
- **The method is model-agnostic**: Consistent effectiveness is demonstrated on Qwen2.5VL-33B and InternVL2.5-8B, with Hallusion improvements of 6.3% and 7.2% respectively.

## Highlights & Insights
- **The insight that "contrast enables VLMs to see more accurately" is remarkably elegant**: This finding reveals a previously overlooked capability of VLMs — while hallucinations arise when inspecting a single image, a much more accurate visual perception emerges when comparing two images. This essentially exploits the model's comparative reasoning ability to correct deficiencies in its direct reasoning.
- **The three-step pipeline is coherently designed**: The Thinking → Contrasting → Rethinking flow elegantly transfers fine-grained visual information obtained through dual-image contrast into single-image reasoning capability — no contrast is needed at inference, yet inference quality is enriched by it.
- **The difficulty sampling design embodies a principled philosophy**: The strategy of retaining only medium-difficulty samples is instructive — trivially easy samples yield no reasoning value (encouraging over-thinking), while intractably hard samples cannot be rescued even with contrast (uncontrollable quality); only appropriately challenging samples produce the most informative training signal.

## Limitations & Future Work
- **High computational cost of contrastive pair construction**: Computing embeddings over large-scale datasets and retrieving contrastive samples makes the data construction pipeline non-trivial in terms of overhead.
- **Rethinking relies on an external LLM**: Employing Qwen2.5-72B for reasoning refinement increases resource requirements and introduces an additional dependency.
- **Comprehensive evaluation is limited to 7B-scale models**: Although preliminary validation on 33B and 8B models is provided, coverage of larger-scale or more diverse model families remains limited.
- **Potential domain distribution bias in VisCoR-55K**: The dataset composition may be skewed toward certain task types, potentially affecting generalization.
- Future work could explore online self-improvement without contrastive pairs, or design more efficient contrastive pair construction pipelines.

## Related Work & Insights
- **vs. STaR**: STaR regenerates reasoning chains using answer hints but cannot correct visual hallucinations. VC-STaR's contrastive mechanism directly addresses hallucinations, achieving +3.2% on Hallusion compared to STaR's +2.8%.
- **vs. LLaVA-CoT**: LLaVA-CoT uses GPT-4o to populate hand-crafted templates for reasoning data generation, but template-based approaches struggle to generalize across tasks. VC-STaR requires no manual templates, automatically generating more diverse reasoning chains through the contrastive mechanism.
- **vs. R1-Onevision**: R1-OV generates reasoning chains from image captions via DeepSeek-R1, but textual descriptions inevitably lose visual information. VC-STaR's visually native approach operates directly on images, preserving complete visual information.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The insight that "contrast enables VLMs to see more accurately" is original and profound; the contrastive self-improvement paradigm opens a new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six benchmarks cover hallucination, math, and general capabilities; ablation studies are comprehensive (contrast type, difficulty sampling, cross-model generalization).
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured with well-designed figures, though some technical sections are densely packed.
- **Value**: ⭐⭐⭐⭐⭐ An effective visual reasoning self-improvement paradigm is established; both the dataset and the method are poised for broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)
- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)

</div>

<!-- RELATED:END -->
