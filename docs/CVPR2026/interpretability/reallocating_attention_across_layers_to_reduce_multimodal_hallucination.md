---
title: >-
  [Paper Note] Reallocating Attention Across Layers to Reduce Multimodal Hallucination
description: >-
  [CVPR 2026][Interpretability][multimodal hallucination] A lightweight, training-free plugin method is proposed to mitigate hallucination in Multimodal Large Reasoning Models (MLRMs) by identifying perceptual and reasonin…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "multimodal hallucination"
  - "attention heads"
  - "perception-reasoning layer stratification"
  - "training-free plugin"
  - "attention reallocation"
date: 2026-05-08
content_hash: 6965580fc294f2c0
---

# Reallocating Attention Across Layers to Reduce Multimodal Hallucination

**Conference**: CVPR 2026
**arXiv**: [2510.10285](https://arxiv.org/abs/2510.10285)  
**Code**: None  
**Area**: Interpretability
**Keywords**: multimodal hallucination, attention heads, perception-reasoning layer stratification, training-free plugin, attention reallocation

## TL;DR

A lightweight, training-free plugin method is proposed to mitigate hallucination in Multimodal Large Reasoning Models (MLRMs) by identifying perceptual and reasoning attention heads and applying Class-Conditioned Rescaling to rebalance cross-layer attention distribution. The method achieves an average improvement of 4.2% across 5 benchmarks with negligible additional inference overhead.

## Background & Motivation

**Escalating multimodal hallucination**: MLRMs frequently generate conclusions that contradict visual evidence or are inconsistent with their own reasoning chains, severely undermining model reliability and deployability.

**Biased assumptions in prior work**: Mainstream approaches (e.g., stronger supervision, fine-grained alignment, external visual priors) presuppose that hallucination primarily stems from insufficient utilization of visual information, overlooking the distributional imbalance between perception and reasoning within the model.

**Interpretability research reveals layer-wise mechanisms**: Prior work has identified a staged structure in Transformer attention—shallow layers rely on visual signals for evidence extraction, while deeper layers shift toward text-based symbolic reasoning—suggesting that hallucination may arise from cross-layer functional dysregulation.

**Two complementary failure modes**: "Perceptual Bias" in shallow layers causes visual attention to scatter and dilutes critical evidence; "Reasoning Drift" in deep layers causes the reasoning chain to deviate from intermediate step premises.

**Manual inspection of 200 cases**: Approximately 16% of hallucinations originate from perceptual bias, 20% from reasoning drift, and 10% from co-occurrence of both, indicating a multi-stage compound problem.

**Latent potential within models**: Models may already contain attention heads with perceptual or reasoning specialization, but these heads do not play a dominant role in their current state and require explicit identification and amplification.

## Method

### Overall Architecture

The method consists of two steps, serving as a plug-and-play lightweight plugin:

1. **Functional Head Identification**: Attention heads are classified as perceptual or reasoning heads using the modality allocation ratio of attention weights and layer depth information.
2. **Class-Conditioned Rescaling**: Multiplicative gains are applied to the identified functional heads to amplify their contributions, counteracting perceptual bias and reasoning drift without altering the underlying attention computation mechanism.

The entire process requires no retraining and no architectural modification; it applies only a minor scaling operation prior to the attention output projection.

### Key Designs

**Visual Attention Ratio**: For each layer $\ell$ and head $h$, the visual attention allocation ratio is computed as $S_v^{(\ell)}(h) = \sum_{j \in \mathcal{T}_v} a_{i^* j}^{(h,\ell)}$, measuring the degree to which the head attends to visual tokens. This is complementary to the textual ratio: $S_v + S_t = 1$.

**Functional Head Classification Rules**: Two ratio thresholds $\tau_{\text{perc}}$ and $\tau_{\text{reas}}$ (where $\tau_{\text{reas}} < \tau_{\text{perc}}$) are introduced, along with layer boundaries $\ell_{\text{perc}}$ (upper bound for perceptual layers) and $\ell_{\text{reas}}$ (lower bound for reasoning layers):

- Perceptual heads: $\mathcal{H}_{\text{perc}}^{(\ell)} = \{h : \ell \le \ell_{\text{perc}} \wedge S_v^{(\ell)}(h) \ge \tau_{\text{perc}}\}$
- Reasoning heads: $\mathcal{H}_{\text{reas}}^{(\ell)} = \{h : \ell \ge \ell_{\text{reas}} \wedge S_v^{(\ell)}(h) \le \tau_{\text{reas}}\}$

**Rescaling Strategy**: Global gains $g_{\text{perc}} \ge 1$ and $g_{\text{reas}} \ge 1$ are assigned to perceptual and reasoning heads respectively, while all other heads remain unchanged (gain of 1). The scaling is applied after per-head output computation and before the output projection:

$$Y_{\text{out}}^{(\ell)} = \text{Concat}(g^{(1,\ell)} O^{(1,\ell)}, \ldots, g^{(H,\ell)} O^{(H,\ell)}) W_O^{(\ell)}$$

**Default Hyperparameters (Ocean-R1)**: $\ell_{\text{perc}}=7$, $\ell_{\text{reas}}=3$, $g_{\text{reas}}=1.30$, $\tau_{\text{reas}}=0.01$, $g_{\text{perc}}=1.16$, $\tau_{\text{perc}}=0.22$.

### Loss & Training

The method requires no training and involves no optimization objective. Its theoretical goal is to minimize hallucination intensity:

$$\mathcal{I} = \lambda_1 \mathbb{E}_{\ell \in \mathcal{L}_{\text{perc}}}[\mathcal{E}_{\text{perc}}^{(\ell)}] + \lambda_2 \mathbb{E}_{\ell \in \mathcal{L}_{\text{reas}}}[\mathcal{E}_{\text{reas}}^{(\ell)}]$$

where perceptual bias $\mathcal{E}_{\text{perc}}$ and reasoning drift $\mathcal{E}_{\text{reas}}$ measure the L2 deviation of shallow-layer visual features and deep-layer reasoning representations from their respective ideal targets.

## Key Experimental Results

Evaluation is conducted on 3 MLRMs (Kimi-VL A3B-Thinking, Ocean-R1 7B-Instruct, R1-Onevision 7B) across 5 benchmarks.

### Main Results (Table 1, compared with Vanilla and 3 baselines)

| Method | MathVista | MathVision | HallusionBench Acc | MMStar Acc | SEED-Bench Acc |
|--------|-----------|------------|---------------------|------------|----------------|
| Kimi-VL Vanilla | 63.48 | 56.24 | 64.76 | 59.76 | 66.26 |
| Kimi-VL + VCD | 63.51 | 56.14 | 65.68 | 59.48 | 66.52 |
| Kimi-VL + AGLA | 67.32 | 58.88 | 61.36 | 63.76 | 69.27 |
| **Kimi-VL + Ours** | **69.78** | **60.54** | **68.19** | **66.49** | **69.74** |
| Ocean-R1 Vanilla | 54.58 | 20.05 | 49.41 | 45.24 | 59.76 |
| **Ocean-R1 + Ours** | **59.32** | **26.01** | **53.64** | **50.77** | **66.51** |
| R1-OneVision Vanilla | 59.92 | 33.54 | 58.26 | 56.26 | 68.48 |
| **R1-OneVision + Ours** | **60.09** | **39.12** | **60.77** | **58.02** | **69.52** |

### Ablation Study

- **w/o Reasoning Rescaling**: Visual tasks benefit more, while mathematical reasoning tasks show limited improvement.
- **w/o Perceptual Rescaling**: Mathematical reasoning improves more noticeably, but visual task performance declines.
- **Non-additive effects**: On R1-OneVision, MathVision performance drops by $-3.91\%$ when enhancing either head type alone, yet rises by $+5.58\%$ when both are enhanced simultaneously, demonstrating that perceptual and reasoning heads require joint optimization.
- **Model heterogeneity**: Different architectures exhibit varying dependencies on perceptual/reasoning heads; for instance, Kimi-VL achieves $+6.71\%$ on MMStar by enhancing only perceptual heads, whereas the same configuration yields $-1.51\%$ for Ocean-R1.

### Key Findings

1. **Approximately 95% of tasks achieve optimal results**, with an average improvement of 4.2% and up to 7% on difficult tasks.
2. **Exceptional efficiency**: Only approximately 2 seconds of additional inference time (roughly 9% of baseline latency), far less than the 1.2×–6.6× overhead introduced by VCD/CGD/AGLA.
3. **Layer boundaries exhibit task-dependent "band" regions** rather than a single split point; visual tasks favor shallower boundaries while reasoning tasks favor deeper ones.
4. **Optimal gain $g \approx 1.14$**: Excessively large gains lead to performance degradation; reasoning gain $g_{\text{reas}}$ is more stable, while perceptual gain $g_{\text{perc}}$ exhibits greater variance.
5. **Sparse intervention is most effective**: Performance peaks when approximately 6.4% of heads (50–150 heads) are selected, and degrades when the selection ratio rises to 18%.
6. **Intermediate layers (layers 10–17) form a transition zone**: In this range, perceptual and reasoning functions are highly intertwined, and enhancing either direction alone is ineffective, corroborating the layer-stratification hypothesis.
7. **Limited cross-model transferability**: Optimal hyperparameter configurations differ substantially across architectures, and different MLRMs exhibit distinct dependency patterns on perceptual/reasoning heads.

## Highlights & Insights

- **Plug-and-play**: No training or architectural modification is required; the method operates directly on attention head outputs during inference, offering strong practical utility.
- **Strong theory–experiment consistency**: The method design directly corresponds to the theoretical objectives derived from the formalization of perceptual bias and reasoning drift, with experiments comprehensively validating each component's contribution.
- **Minimal overhead**: Compared to other inference-time methods (VCD 1.2×, CGD 6.6×), this method adds only approximately 1% computational cost, making it essentially free.
- **Interpretability perspective**: A novel viewpoint is provided for understanding and regulating multimodal reasoning reliability through the lens of cross-layer functional dynamics.

## Limitations & Future Work

- Layer boundaries and thresholds require tuning for different models and tasks, with a large search space (150+ boundary configurations, 24+ rescaling strategies) and no automated selection mechanism.
- Intervention is limited to the inference stage and cannot correct systematic biases introduced during training.
- Experiments cover only 7B-scale models; effectiveness and scalability on larger models (e.g., 70B+) remain unverified.
- Although $\ell_{\text{perc}}$ and $\ell_{\text{reas}}$ may overlap or leave gaps, how to automatically determine the optimal intervals remains an open problem.
- Gains are global constants (layer-agnostic); the potential benefits of layer-wise adaptive gains or sample-wise dynamic gains are unexplored.
- Identification of perceptual/reasoning heads relies on the attention distribution at a single query position $i^*$; aggregation strategies (e.g., multi-position averaging) may yield greater robustness.
- Performance on generative tasks (e.g., image captioning) rather than discriminative benchmarks is not analyzed.

## Related Work & Insights

- **Contrastive decoding methods**: VCD suppresses hallucination by contrasting original and counterfactual image views; CGD uses CLIP to guide decoding; AGLA fuses global and local views to strengthen visual grounding. While effective, these methods introduce significant inference overhead (1.2×–6.6×).
- **Alignment and preference learning**: Fine-tuning improves cross-modal alignment quality but requires additional training data and computational resources, making it unsuitable for black-box deployment scenarios.
- **Multimodal interpretability**: Prior work has revealed that visual attention concentration in attention heads is higher in shallow-to-middle layers (e.g., visual head analysis); this paper extends such findings by translating interpretability insights into actionable intervention schemes.
- **Attention head pruning and analysis**: Prior work found that pruning heads with high visual allocation disproportionately affects visual tasks; this paper takes the opposite approach, leveraging functional heads through "amplification" rather than "pruning."
- **Chain-of-thought reasoning**: CoT-based methods improve multimodal reasoning via prompting or annotated reasoning chains, but rely on hand-crafted prompts or heavy supervision; the proposed method is complementary to such approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ — A new paradigm of functional head identification and rescaling, approaching hallucination from the perspective of perception-reasoning layer stratification
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 models × 5 benchmarks × 4 baselines, with large-scale hyperparameter search and ablation studies
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical analysis, complete mathematical derivations, and rich figures and tables
- Value: ⭐⭐⭐⭐ — Plug-and-play with near-zero overhead, offering high engineering deployment value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[ICML 2026\] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](../../ICML2026/interpretability/finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)
- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](../../AAAI2026/interpretability/llm_circuit_analyses_consistent_across_training_and_scale.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)

</div>

<!-- RELATED:END -->
