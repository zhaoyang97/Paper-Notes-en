---
title: >-
  [Paper Note] Robust Preference Optimization via Dynamic Target Margins
description: >-
  [ACL 2025 (Findings)][LLM Alignment][Preference Optimization] This paper proposes $\gamma$-PO, a plug-and-play method that enhances the robustness of DPO by dynamically adjusting target reward margins at the preference pair level, achieving an average improvement of 4.4% on AlpacaEval2 and Arena-Hard.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Alignment"
  - "Preference Optimization"
  - "Dynamic Margin"
  - "Noise Robustness"
  - "DPO Improvement"
  - "Reward Margin"
date: 2026-05-08
content_hash: 95b354bb7a5130a2
---

# Robust Preference Optimization via Dynamic Target Margins

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.03690](https://arxiv.org/abs/2506.03690)  
**Code**: [https://github.com/sunjie279/gammaPO](https://github.com/sunjie279/gammaPO)  
**Area**: RLHF Alignment  
**Keywords**: Preference Optimization, Dynamic Margin, Noise Robustness, DPO Improvement, Reward Margin

## TL;DR

This paper proposes $\gamma$-PO, a plug-and-play method that enhances the robustness of DPO by dynamically adjusting target reward margins at the preference pair level, achieving an average improvement of 4.4% on AlpacaEval2 and Arena-Hard.

## Background & Motivation

**Background**: The alignment of large language models (LLMs) is a critical step to ensure safety and utility. Direct Preference Optimization (DPO), as an efficient alignment method that eliminates the need to train a separate reward model, directly optimizes the model policy using preference pairs. This substantially reduces resource requirements and has made DPO one of the mainstream alignment approaches.

**Limitations of Prior Work**: The effectiveness of DPO heavily relies on the quality of the training data, but real-world preference data often contains substantial noise—disagreements among annotators, inconsistent annotation standards, and ambiguity in preference degrees are widespread. Existing DPO methods apply a uniform optimization objective to all preference pairs, failing to distinguish high-confidence samples from ambiguous ones, which can lead to the model being misled by noisy data.

**Key Challenge**: The implicit assumption of DPO is that all preference pairs are equally reliable. In practice, however, the confidence of different preference pairs varies significantly—some exhibit a clear quality gap (high reward margin), while others show almost no difference (low reward margin) or may even contain annotation errors. Treating these samples indiscriminately causes the model to learn incorrect preference signals.

**Goal**: To design an algorithm capable of adaptively adjusting the optimization objective based on the quality differences of preference pairs, enabling the model to prioritize learning from high-confidence samples while suppressing the influence of low-confidence (potentially noisy) samples.

**Key Insight**: The authors observe that the reward margin between the chosen and rejected responses in a preference pair naturally reflects the confidence of the sample—a larger margin indicates a clearer and more trustworthy preference.

**Core Idea**: Introducing an instance-level dynamic target margin to automatically calibrate the optimization intensity based on the reward discrepancy of each preference pair, intensifying the learning for high-confidence pairs and diminishing it for ambiguous ones.

## Method

### Overall Architecture

$\gamma$-PO is built upon the standard DPO framework. The input consists of (prompt, chosen, rejected) triplets from the preference dataset, and the output is the aligned policy model. The overall workflow consists of two steps: first, utilizing the reference model to compute the implicit reward margin for each preference pair, and then dynamically setting the target margin $\gamma$ for that preference pair based on the margin value, integrating it into a DPO-like loss function for training.

### Key Designs

1. **Dynamic Target Margin Calibration**:

    - **Function**: Computes an instance-specific target margin $\gamma$ for each preference pair, replacing the fixed margin implicit in DPO.
    - **Mechanism**: Computes the log probabilities of the chosen and rejected responses using the reference model. The difference $r_{\text{margin}} = \log \pi_{\text{ref}}(y_w|x) - \log \pi_{\text{ref}}(y_l|x)$ serves as the confidence indicator for the preference pair. A monotonically increasing mapping function is then used to translate the reward margin into the target margin $\gamma$, such that high-confidence pairs receive a larger target margin (stronger optimization push) and low-confidence pairs receive a smaller target margin (weaker optimization push).
    - **Design Motivation**: The reference model possesses prior knowledge about sample quality before training. Utilizing this signal allows the filtering of noise without requiring extra annotations. A high reward margin implies the reference model can already clearly distinguish the quality of the two responses, indicating that such samples are highly likely to be correctly labeled.

2. **Noise Suppression via Margin Thresholding**:

    - **Function**: Automatically reduces the impact of ambiguous preference pairs on training through a margin thresholding mechanism.
    - **Mechanism**: When the reward margin of a preference pair falls below a certain threshold, its corresponding target margin $\gamma$ approaches 0, effectively suppressing the contribution of this sample to the training gradient. This achieves a soft sample filtering effect rather than simply discarding samples. Specifically, the mapping function is designed to have a gentler slope at low margins and a steeper slope at high margins.
    - **Design Motivation**: Hard filtering (directly discarding low-confidence samples) wastes data and requires manual threshold setting. A soft suppression mechanism allows all samples to participate in training while automatically adjusting their influence weights, offering greater flexibility and data efficiency.

3. **Plug-and-Play Compatibility**:

    - **Function**: $\gamma$-PO can be seamlessly integrated into all margin-based DPO variants.
    - **Mechanism**: The core modification of $\gamma$-PO is simply adding a dynamic target margin term to the loss function, without altering the model architecture, training pipeline, or data format. For DPO variants such as SimPO, IPO, and KTO, one only needs to insert $\gamma$ into the margin-related part of their loss functions.
    - **Design Motivation**: There are numerous DPO variants in the preference optimization research area. Designing $\gamma$-PO in a plug-and-play manner maximizes its practical utility. Experiments demonstrate that it requires only a few lines of code changes and has almost no impact on training efficiency.

### Loss & Training

The loss function of $\gamma$-PO incorporates a dynamic margin term into the standard DPO loss. Taking DPO as an example, the standard loss is $\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta (r_w - r_l))$, and $\gamma$-PO modifies it to $\mathcal{L}_{\gamma\text{-PO}} = -\log \sigma(\beta (r_w - r_l - \gamma))$, where $\gamma$ is the dynamic target margin computed based on the reference model. The training strategy remains identical to the baseline methods, without introducing additional hyperparameter tuning.

## Key Experimental Results

### Main Results

| Benchmark | Metric | $\gamma$-PO (DPO) | DPO | SimPO | $\gamma$-PO (SimPO) |
|-----------|--------|------------|-----|-------|-------------|
| AlpacaEval2 | LC Win Rate (%) | +4.2 vs DPO | baseline | baseline | +3.8 vs SimPO |
| Arena-Hard | Win Rate (%) | +4.6 vs DPO | baseline | baseline | +4.9 vs SimPO |
| Average | Gain | **+4.4%** | - | - | **+4.4%** |

$\gamma$-PO consistently achieves improvements across multiple DPO variants and remains effective across various base models (e.g., Llama series, Mistral series).

### Ablation Study

| Configuration | AlpacaEval2 | Arena-Hard | Description |
|---------------|-------------|------------|-------------|
| Full $\gamma$-PO | Best | Best | Full dynamic margin |
| Fixed $\gamma$ (constant) | Decreased | Decreased | Degenerates to DPO with offset |
| No $\gamma$ ($\gamma=0$) | Baseline | Baseline | Equivalent to standard DPO |
| Reverse $\gamma$ (low margin $\to$ large target) | Significantly decreased | Significantly decreased | Validates the correctness of the direction |

### Key Findings

- The direction of the dynamic margin is crucial: assigning large margins to high-confidence samples is correct, whereas reversing this operation significantly hurts performance.
- $\gamma$-PO yields consistent improvements across different DPO variants (DPO, SimPO, IPO), validating the generality of the method.
- There is almost no additional overhead in terms of training efficiency, and the code modifications are minimal (only a few lines), making the engineering barrier to deployment extremely low.
- The performance gains of $\gamma$-PO are even more pronounced in scenarios with heavily noisy data, confirming its noise robustness.

## Highlights & Insights

- **Leveraging the prior knowledge of the reference model for sample weighting**: This is a highly elegant idea—since the reference model naturally contains information about response quality, directly using it to assess the reliability of preference pairs avoids auxiliary quality evaluation steps.
- **Plug-and-play design philosophy**: In a research community where various DPO variants emerge rapidly, designing a universal improvement scheme that seamlessly adapts to all variants greatly enhances its real-world impact.
- **Transferability of dynamic margins to other contrastive learning settings**: The idea of adaptively adjusting margins based on sample difficulty is not only applicable to preference optimization but can also be extended to retrieval, recommendation, and other scenarios utilizing contrastive losses.

## Limitations & Future Work

- The paper primarily evaluates the method on English dialogue scenarios; its efficacy in multilingual settings and specialized domains (e.g., code generation, mathematical reasoning) remains to be further investigated.
- The quality of the reference model itself directly impacts the accuracy of margin estimation. If the reference model is weak, the reliability of the margin signals becomes questionable.
- The choice of the mapping function format for the dynamic margin (e.g., linear, piecewise linear) and its impact on final performance have not been fully explored.
- Online learning could be incorporated in future work, dynamically updating margin estimates based on the model's current state as training progresses, rather than relying solely on a fixed reference model.

## Related Work & Insights

- **vs DPO**: While DPO applies a uniform optimization objective to all preference pairs, $\gamma$-PO introduces instance-level adaptive margins, significantly improving robustness while maintaining simplicity.
- **vs SimPO**: SimPO introduces length-normalized margins by modifying the reward definition, while $\gamma$-PO is orthogonal and complementary to it—dynamic margins can be overlaid on top of SimPO.
- **vs RSO/DPOP**: Some studies suppress noise by altering the loss function formulation or adding regularization terms, whereas $\gamma$-PO approaches this from a sample weighting perspective, which is more direct and easier to implement.

## Rating

- Novelty: ⭐⭐⭐ The core idea (dynamic margin) is elegant but not entirely new; similar curriculum learning and sample weighting ideas have been widely applied in other domains.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple benchmarks, base models, and DPO variants, with a reasonable ablation design.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured, with a logically coherent motivation.
- Value: ⭐⭐⭐⭐ A plug-and-play, effective RLHF improvement method with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ICCV 2025\] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](../../ICCV2025/llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)

</div>

<!-- RELATED:END -->
