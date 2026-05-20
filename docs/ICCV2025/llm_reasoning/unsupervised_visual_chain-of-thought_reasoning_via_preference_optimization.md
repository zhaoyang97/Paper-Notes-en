---
title: >-
  [Paper Note] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization
description: >-
  [ICCV 2025][LLM Reasoning][Visual CoT] This paper proposes UV-CoT, a framework that enables image-level chain-of-thought (Visual CoT) reasoning without any manual bounding box annotations…
tags:
  - "ICCV 2025"
  - "LLM Reasoning"
  - "Visual CoT"
  - "Preference Optimization"
  - "Unsupervised Learning"
  - "Multimodal Reasoning"
  - "Bounding Box"
date: 2026-05-08
content_hash: 65305fef1660f09c
---

# Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization

**Conference**: ICCV 2025
**arXiv**: [2504.18397](https://arxiv.org/abs/2504.18397)  
**Code**: [https://github.com/kesenzhao/UV-CoT](https://github.com/kesenzhao/UV-CoT)  
**Area**: LLM Reasoning / Multimodal
**Keywords**: Visual CoT, Preference Optimization, Unsupervised Learning, Multimodal Reasoning, Bounding Box

## TL;DR

This paper proposes UV-CoT, a framework that enables image-level chain-of-thought (Visual CoT) reasoning without any manual bounding box annotations, by automatically constructing preference data and introducing an improved Score-DPO loss. UV-CoT surpasses the supervised Visual-CoT method on 6 benchmarks.

## Background & Motivation

CoT reasoning has significantly improved the interpretability and problem-solving capabilities of MLLMs. However, existing approaches focus on textual CoT and cannot dynamically adjust attention to different spatial regions of the input image. The only Visual CoT work (Visual-CoT) introduces image-level reasoning but suffers from two critical limitations: (1) it relies on large-scale manually annotated bounding box data, which is costly and difficult to scale; and (2) it learns only from positive samples via SFT, limiting generalization.

The core motivation of UV-CoT is: can a model learn the ability to "decide where to look before reasoning" without any manual annotations? The key insight is that while it is difficult to prompt MLLMs to directly generate precise coordinates, it is much easier to rank multiple candidate regions. This reformulates the hard coordinate regression problem into a more tractable preference comparison problem.

## Method

### Overall Architecture

During inference, UV-CoT simulates the human perception process: given an original image and a question, the model is guided via CoT prompting to generate bounding box coordinates of key regions, which are then cropped by a visual sampler. Visual tokens from both the original and cropped images are combined to produce a more accurate answer. Training consists of two core stages: automatic preference data construction and preference optimization via Score-DPO.

### Key Designs

1. **Automatic Preference Data Generation Pipeline (Algorithm 1)**:

    - **Response Generation**: Given an image-question pair $x$, the target model $f_{\text{tar}}$ (LLaVA-1.5-7B) generates $n$ diverse candidate bounding boxes and corresponding responses $\{y_t^i\}_{i=1}^n$ via template prompting and stochastic decoding.
    - **Response Evaluation**: An evaluator model $f_{\text{eval}}$ (OmniLMM-12B) scores each response. A key innovation is the introduction of cumulative evaluation: $s^i = s_{\text{cur}}^i + \gamma s_{\text{nxt}}^i$, where $s_{\text{nxt}}^i$ measures the impact of the current region on subsequent reasoning steps, and $\gamma$ is a hyperparameter.
    - **Pair Construction**: $k$ preference pairs (preferred vs. dis-preferred) are randomly sampled from the $n$ candidates, each containing a full reasoning chain and corresponding scores $\{y_w, s_w, y_l, s_l\}$.
    - **Response Selection**: The highest-scoring response is retained as context for the next reasoning step, forming an "optimal chain."

2. **Score-DPO (sDPO) Loss**: Standard DPO only ranks preference data without quantifying preference intensity. UV-CoT introduces sDPO, which incorporates score margins:
    $$\mathcal{L}_{\text{sDPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - (g(s_w) - g(s_l))\right)\right]$$
   where $g(\cdot)$ is a monotonically increasing function that maps preference scores to the logit space of the DPO objective. Derived from the Gumbel distribution, $\Delta_r = g(s_w) - g(s_l)$ quantifies the degree of difference between preference pairs, enabling the model to not only distinguish preference order but also optimize the magnitude of preference differences.

3. **Iterative Learning Strategy (Algorithm 2)**: To avoid the distribution mismatch between static preference data and the evolving model in standard DPO, the training query set $\mathcal{X}$ is divided into $m$ subsets and training proceeds for $m$ iterations. At each iteration, the current model $f_{\text{tar}}^i$ generates new preference data $\mathcal{D}_i$ on subset $\mathcal{X}_i$, and training yields $f_{\text{tar}}^{i+1}$. This ensures preference data remains aligned with the model's current capability. In practice, 4 iterations are used, yielding 249K preference pairs in total—fewer than the 376K annotated samples used by Visual-CoT.

### Loss & Training

AdamW optimizer is used, with 4 training epochs per iteration, learning rate $5\times10^{-7}$, $\beta=0.1$, and batch size 8. Data generation takes 80 hours and training takes 60 hours, both on 8×A100 40GB GPUs. The target model is LLaVA-1.5-7B and the evaluator is OmniLMM-12B.

## Key Experimental Results

### Main Results

| Model | DocVQA | TextVQA | GQA | VSR | Avg |
|-------|--------|---------|-----|-----|-----|
| LLaVA-1.5-7B | 0.198 | 0.507 | 0.480 | 0.504 | 0.393 |
| OmniLMM-12B (Evaluator) | 0.254 | 0.578 | 0.509 | 0.523 | 0.443 |
| Visual-CoT-7B (100% annotation) | 0.294 | 0.673 | 0.546 | 0.532 | 0.482 |
| **UV-CoT (0% annotation)** | 0.265 | **0.686** | 0.536 | **0.548** | 0.473 |
| **UV-CoT (10% annotation)** | 0.283 | **0.711** | **0.568** | **0.553** | **0.494** |

| Zero-shot Dataset | Visual-CoT | UV-CoT | UV-CoT* | Description |
|------------------|-----------|--------|---------|-------------|
| DUDE | 0.206 | 0.241 | 0.253 | Document Understanding |
| Visual7w | 0.397 | 0.432 | 0.455 | General VQA |
| V*Bench OCR | 0.593 | **0.677** | - | High-res OCR |
| V*Bench Avg | 0.347 | **0.402** | - | High-res Reasoning |

### Ablation Study

| Configuration | Avg Accuracy | Description |
|---------------|-------------|-------------|
| UV-CoT (10% annotation) | **0.494** | Full model |
| w/o UV-CoT reasoning | 0.417 | Remove CoT, answer directly, −7.7% |
| UV-CoT w/ GT BBox | 0.618 | Upper bound reference, +12.4% |
| w/ standard DPO | 0.475 | −1.9%, cannot quantify preference intensity |
| w/o iterative learning | 0.459 | −3.5%, static data distribution mismatch |
| w/o $\gamma$ (no next-step influence) | 0.406 | −8.8%, MLLM cannot directly assess BBox quality |

### Key Findings

- UV-CoT outperforms its evaluator OmniLMM-12B by an average of 5.1%, indicating that this is not simple model distillation.
- UV-CoT with only 10% annotation surpasses Visual-CoT trained with 100% annotation (0.494 vs. 0.482).
- On V*Bench high-resolution image reasoning, Visual CoT methods show the most prominent advantage over non-CoT baselines (>50% OCR improvement), and UV-CoT further outperforms Visual-CoT by 5.5%.
- Ablation of $\gamma$ demonstrates that accounting for the region's impact on subsequent reasoning steps is critical (−8.8%), as MLLMs cannot reliably evaluate bounding box quality directly.

## Highlights & Insights

- The paper elegantly reformulates Visual CoT as a preference ranking problem, circumventing the bottleneck of imprecise coordinate generation by MLLMs.
- The mathematical derivation of the sDPO loss is grounded in the Gumbel distribution, making it theoretically better suited for modeling continuous preference differences than standard DPO.
- The combination of iterative learning and automatic data generation forms a self-improvement loop, reflecting the spirit of online learning.
- The design of incorporating current and next-step influence during evaluation (the $\gamma$ parameter) is conceptually analogous to temporal difference methods in reinforcement learning.

## Limitations & Future Work

- The current approach generates only one bounding box per step; the chain length and branching factor of multi-step reasoning remain limited.
- Data generation (80 hours) and training (60 hours) are computationally intensive and could be further optimized.
- A notable performance gap relative to GT BBox remains on DocVQA and InfographicsVQA, indicating room for improvement in precise localization.
- Validation is limited to the 7B scale; effectiveness on larger models remains to be verified.

## Related Work & Insights

- UV-CoT is complementary to RLHF/DPO-based textual CoT optimization, focusing specifically on image-level spatial decision-making.
- The automatic preference data generation pipeline is generalizable to other vision tasks requiring spatial reasoning.
- The margin design in sDPO can be applied to other scenarios that require quantifying preference intensity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First successful attempt at unsupervised Visual CoT, with a rigorous theoretical derivation of sDPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10 datasets across multiple tasks; comprehensive ablations and zero-shot validation are convincing.
- Writing Quality: ⭐⭐⭐⭐ Algorithm descriptions are clear and figures are informative.
- Value: ⭐⭐⭐⭐⭐ Surpasses supervised methods without any annotation; excellent practicality and scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_reasoning/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)

</div>

<!-- RELATED:END -->
