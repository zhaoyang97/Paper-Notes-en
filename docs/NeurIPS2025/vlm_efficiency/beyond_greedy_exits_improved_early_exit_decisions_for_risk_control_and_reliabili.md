---
title: >-
  [Paper Note] Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability
description: >-
  [NeurIPS 2025][Multimodal Efficiency][Early exit mechanism] UAT (Unsupervised Adaptive Thresholding) designs a reliability function for early-exit DNNs to assess the quality of intermediate layer outputs, and employs a multi-armed bandit (MAB) algorithm to dynamically learn optimal exit thresholds at inference time, achieving 1.7–2.1× speedup with less than 2% performance degradation while remaining robust to distribution shift.
tags:
  - "NeurIPS 2025"
  - "Multimodal Efficiency"
  - "Early exit mechanism"
  - "risk control"
  - "multi-armed bandit"
  - "reliability function"
  - "adaptive thresholding"
date: 2026-05-08
content_hash: d6c7db6f5077c90c
---

# Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability

**Conference**: NeurIPS 2025
**arXiv**: [2509.23666](https://arxiv.org/abs/2509.23666)  
**Code**: To be confirmed  
**Area**: Multimodal VLM
**Keywords**: Early exit mechanism, risk control, multi-armed bandit, reliability function, adaptive thresholding

## TL;DR
UAT (Unsupervised Adaptive Thresholding) designs a reliability function for early-exit DNNs to assess the quality of intermediate layer outputs, and employs a multi-armed bandit (MAB) algorithm to dynamically learn optimal exit thresholds at inference time, achieving 1.7–2.1× speedup with less than 2% performance degradation while remaining robust to distribution shift.

## Background & Motivation

**Background**: Early exit mechanisms attach classification heads to intermediate layers of DNNs, allowing simple samples to exit early and thus accelerate inference. Existing methods use fixed thresholds (based on confidence or entropy) to determine whether to exit.

**Limitations of Prior Work**: (a) Models may be overconfident at shallow layers, leading to erroneous exits; (b) fixed thresholds cannot adapt to shifts in data distribution—performance degrades sharply under distribution shift; (c) existing methods (e.g., BEEM) are restricted to classification tasks and do not support generative tasks.

**Key Challenge**: The trade-off between the benefit of early exit (speed) and its risk (quality degradation) requires careful balance. Fixed thresholds apply a one-size-fits-all policy across all inputs, failing to reflect the actual reliability of different layers and different samples.

**Goal**: (a) Train a reliability function to assess whether the output of each layer is trustworthy; (b) dynamically adjust exit thresholds at inference time to adapt to distribution shifts.

**Key Insight**: The problem of selecting the exit layer is formulated as a multi-armed bandit problem—each layer constitutes an "arm," and the reward combines reliability (quality) with computational cost (layer depth). A UCB algorithm is used to online-learn the optimal exit policy.

**Core Idea**: Training a reliability function to evaluate intermediate layer output quality + using a multi-armed bandit to dynamically optimize exit thresholds at inference time = adaptive early exit.

## Method

### Overall Architecture
Two phases: **Offline phase**—a reliability function $g(p_i, i)$ is learned jointly during early-exit DNN training, where $p_i$ is the output distribution of the $i$-th layer and $i$ is the layer index. **Online phase**—at inference time, a UCB multi-armed bandit learns the optimal exit threshold $\tau^*$ by balancing reliability and computational cost.

### Key Designs

1. **Reliability Function Training**:

    - Function: Learns a function $g(p_i(\cdot|x), i)$ that evaluates whether the prediction of layer $i$ for input $x$ is reliable.
    - Mechanism: The training loss is modified to $\mathcal{L}_i = \mathcal{L}_{CE}(p_i, y)(1 + g(p_i, i)) + \Phi(c - \phi(g))$. The first term increases the loss when reliability is high (forcing $g$ to output low values when the prediction is correct and high values when incorrect); the second term is a constraint that controls the output range of $g$.
    - Design Motivation: Standard confidence (softmax maximum) is not reflective of true reliability—a model may exhibit high confidence but be incorrect at shallow layers. The reliability function explicitly learns the relationship between prediction quality and layer depth.

2. **Multi-Armed Bandit (MAB) Exit Strategy**:

    - Function: Online learning of the optimal exit threshold at inference time.
    - Mechanism: The reward is defined as $r(\tau) = C_\tau^i \cdot (1 - C_g^i) - \psi(i)$, where $C_\tau^i$ is the threshold-based confidence, $C_g^i = 1 - g(\cdot)$ is the unreliability score, and $\psi(i) = \lambda \cdot i$ is a depth penalty. A UCB algorithm balances exploration (trying different thresholds) and exploitation (using the known best threshold).
    - Design Motivation: Fixed thresholds cannot adapt to distribution shift. The MAB continuously learns during inference and automatically adjusts to the optimal operating point. UCB provides theoretical regret bound guarantees.

3. **Generative Task Support**:

    - Function: Extends the early exit mechanism to sequence generation (summarization, image captioning).
    - Mechanism: For encoder-decoder models, reliability is assessed at intermediate encoder layers to decide whether to skip subsequent encoder layers and proceed directly to the decoder.
    - Design Motivation: Existing early exit methods (e.g., BEEM) support only classification tasks. UAT is the first early exit framework that simultaneously supports both classification and generative tasks.

### Loss & Training
- Training loss = weighted cross-entropy loss across all layers + reliability function regularization
- Reliability function: simple linear architecture $g(p, i) = w^\top [p; i] + b$
- MAB hyperparameter: the threshold candidate set is discretized as $\{0.1, 0.2, \ldots, 0.9\}$

## Key Experimental Results

### Main Results

| Task | Model | Method | Performance | Speedup |
|------|-------|--------|-------------|---------|
| SST-2 (Classification) | BERT-large | BEEM | 95.4% | 1.60× |
| | | **UAT** | **95.8%** | **1.85×** |
| SamSum (Summarization) | T5-large | Full model | 48.82 RL | 1.00× |
| | | **UAT** | 48.09 RL | **2.10×** |
| COCO (Captioning) | BLIP-2 | Full model | 44.0 B4 | 1.00× |
| | | **UAT** | **43.3 B4** | **1.72×** |

### Distribution Robustness Experiments

| Dataset | Method | In-domain | Out-domain |
|---------|--------|-----------|------------|
| NoCaps | Full model | 124.5 | 124.5 |
| | Fixed threshold | 121.8 | 118.2 |
| | **UAT** | **123.1** | **122.7** |

### Key Findings
- UAT outperforms BEEM on classification tasks (95.8% vs. 95.4%) while achieving a higher speedup (1.85× vs. 1.60×).
- On generative tasks, a 2.10× speedup incurs only 0.73% ROUGE-L degradation (summarization) and 1.6% BLEU-4 degradation (image captioning).
- Under distribution shift, UAT significantly outperforms fixed-threshold methods—out-domain performance degradation is minimal (122.7 vs. 118.2).
- The MAB converges to a near-optimal threshold after approximately 100 samples.
- The reliability function better distinguishes correct from incorrect predictions than raw confidence, especially at shallow layers.

## Highlights & Insights
- **Elegant use of MAB for online threshold learning**: Formulating discrete threshold selection as a bandit problem provides theoretical regret guarantees and converges rapidly in practice. This framework is transferable to other scenarios requiring online hyperparameter adaptation.
- **Reliability function vs. confidence**: The finding that model confidence does not equate to reliability (especially at shallow layers) motivates the explicit learning of a reliability criterion as a superior exit signal.
- **First adaptive early exit framework supporting generative tasks**: Fills a gap in the early exit literature for NLG and vision-language tasks.

## Limitations & Future Work
- The reliability function requires ground truth labels for training (offline phase).
- The linear architecture of the reliability function has limited expressiveness; more complex tasks may require a stronger evaluator.
- The exploration phase of the MAB may produce suboptimal exits for the first few dozen samples.
- The loss is assumed to be bounded in $[0, 1]$.

## Related Work & Insights
- **vs. BEEM**: BEEM supports only classification and uses fixed thresholds; UAT supports generative tasks with adaptive thresholds.
- **vs. Patience/Entropy-based methods**: Traditional methods require manual threshold tuning; UAT learns thresholds automatically online.
- **vs. SnapKV/LazyLLM (token pruning)**: These represent complementary lines of work—early exit skips layers, whereas token pruning reduces sequence length; the two approaches are mutually compatible.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using MAB for early exit threshold learning is original; the design of the reliability function reflects genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three task types (classification, summarization, captioning) including distribution shift evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the methodological pipeline is well-presented.
- Value: ⭐⭐⭐⭐ Provides a more robust online adaptive solution for early exit mechanisms.

### Additional Technical Details
- Offline phase loss weighting: $\mathcal{L} = \sum_{i=1}^L i \cdot \mathcal{L}_i / \sum_i i$, assigning greater weight to deeper layers to reflect higher computational cost.
- The setting $\lambda = \epsilon/L$ from Theorem 4.1 is used across all experiments without additional tuning.
- The computational overhead of UAT's exploration phase (first $|\Omega|$ samples) is negligible (verified in Appendix B.8).
- Under domain transfer (SST-2→IMDB), UAT maintains stable performance compared to fixed-threshold methods.
- Training the reliability function $g$ does not affect the original model's predictive performance (verified in Appendix B.9).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](../../ICLR2026/vlm_efficiency/hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[ICML 2026\] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](../../ICML2026/vlm_efficiency/less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)
- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)

</div>

<!-- RELATED:END -->
