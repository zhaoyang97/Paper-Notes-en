---
title: >-
  [Paper Note] Improving Large Vision and Language Models by Learning from a Panel of Peers
description: >-
  [ICCV2025][Self-Supervised Learning][Self-improvement] This paper proposes the Panel-of-Peers (PoP) learning framework, in which multiple LVLMs of comparable capability mutually generate candidate responses and score each other to construct preference data. Combined with iterative self-improvement via SimPO, PoP raises the average score across 15 benchmarks from 48% to 57% without any human-annotated data.
tags:
  - "ICCV2025"
  - "Self-Supervised Learning"
  - "Self-improvement"
  - "Preference Alignment"
  - "Peer Learning"
  - "LVLM"
  - "Reward Modeling"
date: 2026-05-08
content_hash: 9d672d759c0f381b
---

# Improving Large Vision and Language Models by Learning from a Panel of Peers

**Conference**: ICCV2025
**arXiv**: [2509.01610](https://arxiv.org/abs/2509.01610)  
**Code**: -  
**Area**: Self-Supervised Learning
**Keywords**: Self-improvement, Preference Alignment, Peer Learning, LVLM, Reward Modeling

## TL;DR

This paper proposes the Panel-of-Peers (PoP) learning framework, in which multiple LVLMs of comparable capability mutually generate candidate responses and score each other to construct preference data. Combined with iterative self-improvement via SimPO, PoP raises the average score across 15 benchmarks from 48% to 57% without any human-annotated data.

## Background & Motivation

Large vision-language models (LVLMs) have demonstrated strong performance across diverse tasks, yet their further improvement remains challenging:

**High annotation cost**: Conventional alignment methods rely on human preference data, which is expensive to collect and difficult to scale.

**Limited quality of machine-generated data**: Distilling from foundation models such as GPT-4V imposes a performance ceiling determined by the teacher model.

**Hallucination in self-supervised preference data**: Data generated and evaluated by the model itself is prone to introducing hallucinations.

The authors draw inspiration from **collaborative classroom learning**: students first acquire foundational knowledge and then deepen their understanding through exercises and peer discussion. Analogously, a group of LVLMs with comparable performance ("peers") can learn from one another by mutually generating and evaluating responses. Unlike conventional teacher–student distillation, the core of PoP is **mutual feedback among peers**, where each model acts simultaneously as a generator and an evaluator.

## Method

### Overall Architecture

The PoP pipeline alternates between two stages:

1. **Candidate response generation**: Each model in the panel generates candidate responses to the same set of image–question pairs.
2. **Data creation and fine-tuning**: Models score each other's outputs → preference datasets are constructed → models are fine-tuned with a preference optimization algorithm.

These two stages alternate to form an iterative self-improvement loop.

### Reward Modeling

Each model $\pi_i$ in the panel acts as a judge and scores the output $\mathbf{y}_j$ produced by another model $\pi_j$ along five dimensions: helpfulness, correctness, level of detail, coherence, and complexity. Each dimension is rated on a 1–5 Likert scale and normalized to $[0,1]$.

The ensemble reward aggregates scores from all judges via mean voting:

$$R_\mu(\mathbf{y}_j) = \frac{1}{M} \sum_{i=1}^{M} R_i(\mathbf{y}_j)$$

A key design choice is that scoring relies solely on each model's internal knowledge, requiring no reference answers.

### Candidate Response Generation

Each model generates multiple candidate responses via Best-of-N sampling and selects the best:

$$\mathbf{y}^* = \arg\max_{\mathbf{y}^{(n)} \in Y_N} R(\mathbf{y}^{(n)})$$

The standard variant (PoP) samples 15 candidates per model; the simplified variant (st-PoP) samples only 1. Rejection sampling is then applied, retaining only positive samples with a reward score $\geq 0.85$ and maintaining a reward margin of 0.75 between positive and negative examples.

### Preference Data Construction and Iterative Training

The responses with the highest and lowest cumulative rewards are selected as the preferred and dispreferred responses, respectively. SimPO is adopted as the preference optimization objective:

$$\mathcal{L}_{\text{SimPO-PoP}} = -\mathbb{E}_{\mathcal{D}_t}\left[\log\sigma\left(\frac{\beta}{|\mathbf{y}^{(n)}|}\pi_{\theta_t}(\mathbf{y}^{(n)}) - \frac{\beta}{|\mathbf{y}^{(1)}|}\pi_{\theta_t}(\mathbf{y}^{(1)}) - \gamma\right)\right]$$

SimPO is chosen because it requires no reference model, its implicit reward aligns directly with generation metrics, and the margin $\gamma$ helps separate preferred from dispreferred responses.

**Iterative mechanism**: After each iteration, the panel is re-initialized with the checkpoints from the previous round, and the generate–score–fine-tune cycle is repeated for a total of 3 iterations.

### Experimental Setup

- **Panel members**: LLaVA-1.5 architecture with Mistral-7B / Llama3-8B / Vicuna-7B backbones.
- **Training data**: 1M images randomly sampled from Cambrian-7M (no GT answers); approximately 300K preference samples are produced per iteration.
- **Evaluation**: 15 benchmarks covering General VQA, Knowledge, Chart & OCR, Hallucination, and Vision-Centric tasks.

## Key Experimental Results

### Main Results: Comparison with Preference Optimization Methods (Based on LLaVA-1.5-Vicuna-7B)

| Method | Extra Data | MMB | SEED-B | MM-Vet | SciQA | POPE |
|--------|-----------|-----|--------|--------|-------|------|
| LLaVA-1.5-7B | - | 64.3 | 58.6 | 30.5 | 66.8 | 85.9 |
| +RLHF | 10k | 63.4 | 58.1 | 31.1 | 65.8 | 81.5 |
| +CSR | 17k | 65.4 | 60.3 | 33.9 | 70.7 | 87.0 |
| +SeVa | 8k | 65.6 | 65.8 | 37.2 | 67.5 | 86.7 |
| +STIC | 6k | 65.3 | 66.2 | 32.6 | 67.4 | 85.8 |
| **+PoP-iter1** | **300k** | **68.7** | **67.9** | 34.1 | 71.2 | 87.0 |
| **+PoP-iter3** | **900k** | **72.5** | **68.8** | 35.0 | **86.4** | 87.0 |

PoP-iter3 achieves a 19.6 absolute percentage point improvement on SciQA (from 66.8% to 86.4%) and an 8.2 point improvement on MMB.

### Iterative Progress across 15 Benchmarks

| Iteration | PoP-Mistral | PoP-Vicuna | PoP-LLaMA3 |
|-----------|------------|------------|------------|
| Iter 0 | 47.7 | 48.0 | 48.7 |
| Iter 1 | 54.3 | 53.1 | 55.6 |
| Iter 2 | 55.7 | 55.9 | 57.3 |
| Iter 3 | 56.4 | 56.7 | **58.2** |

The LLaMA3 member improves from 48.7% to 58.2% (+9.5 absolute points), demonstrating the effectiveness of iterative self-improvement. Performance saturates after 3 iterations.

### Ablation Study

- **Absolute vs. relative scoring**: Absolute scoring (each model scores independently) outperforms relative scoring (all responses compared simultaneously), as the latter causes context loss due to excessively long prompts.
- **SFT vs. SimPO**: The two are comparable under the multi-sample PoP setting; however, SimPO shows a clear advantage in reducing hallucinations and improving vision-centric tasks.
- **Peer learning of new skills**: An "OCR-Dumb" model lacking OCR capability successfully acquires that skill from OCR-capable peers via PoP, validating cross-model knowledge transfer.
- **PoP vs. direct SFT on 900K GT data**: PoP-Vicuna reaches 57.0, whereas SFT on 900K ground-truth answers reaches only 54.0, indicating that peer-feedback synthetic data outperforms static annotated data.

## Highlights & Insights

1. **Self-improvement without human annotation**: The entire pipeline uses no ground-truth answers, relying solely on images and questions; models learn from mutual evaluation.
2. **Peer feedback surpasses GT data**: Training on 900K PoP synthetic samples (57.0) outperforms SFT on an equivalent amount of GT data (54.0)—a counterintuitive yet significant finding.
3. **Cross-model knowledge transfer**: PoP enables models lacking specific capabilities to acquire them from peers (e.g., OCR), demonstrating the potential of collaborative learning.
4. **Generality of the approach**: The method imposes no constraints on panel size or model capacity and can be seamlessly extended to future frontier models.

## Limitations & Future Work

- High computational cost: preference data collection takes approximately 80 hours, and fine-tuning each model requires approximately 10 hours (8×A100 80GB).
- At least 3 baseline models are required to form the panel, increasing initial investment.
- Each iteration demands large-scale data sampling (generating 900K preference pairs from 1M images), and performance saturates after 3 iterations.
- The design of evaluation prompts has a substantial impact on final performance, yet a systematic methodology for prompt selection is lacking.

## Related Work & Insights

- **LVLM alignment**: LLaVA-RLHF (10K human interactions), POVID/SeVa (injecting errors to simulate hallucinations), CSR (CLIPScore-based ranking), STIC (augmentation + DPO).
- **Self-improving LLMs**: Self-Rewarding (LLM serves simultaneously as reward model and generator), SPIN (two-player game framework).
- **Model-as-judge**: PoLL (a panel of weaker models can produce human-aligned scores comparable to stronger models), Prometheus-Vision.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Quality | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)
- [\[CVPR 2026\] Quantized Residuals to Continuous Prompts for Few-Shot Class Incremental Learning in Vision-Language Models](../../CVPR2026/self_supervised/quantized_residuals_to_continuous_prompts_for_few-shot_class_incremental_learning.md)

</div>

<!-- RELATED:END -->
