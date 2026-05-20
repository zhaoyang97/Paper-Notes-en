---
title: >-
  [Paper Note] Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs
description: >-
  [NeurIPS 2025][Model Compression][Benford's Law] This paper demonstrates that numerical hallucinations in LLMs originate from the Benford's Law-conforming digit frequency distribution in pretraining corpora—where digit 1…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Benford's Law"
  - "digit bias"
  - "numerical hallucination"
  - "FFN neurons"
  - "selective pruning"
date: 2026-05-08
content_hash: 723485f72ee8bb7a
---

# Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.01734](https://arxiv.org/abs/2506.01734)  
**Code**: [https://github.com/shamy28/Benford-Curse](https://github.com/shamy28/Benford-Curse)  
**Area**: Model Compression
**Keywords**: Benford's Law, digit bias, numerical hallucination, FFN neurons, selective pruning

## TL;DR
This paper demonstrates that numerical hallucinations in LLMs originate from the Benford's Law-conforming digit frequency distribution in pretraining corpora—where digit 1 appears with ~30% probability while digit 9 appears with only ~5%—and that this bias is internalized by specific "digit-selective neurons" in the later FFN layers. A Digit Selectivity Coefficient (DSC) is proposed to localize biased neurons, and pruning 0.01% of neurons corrects 1.36–3.49% of erroneous predictions.

## Background & Motivation

**Background**: LLMs frequently fail at basic numerical reasoning tasks (arithmetic, sequence prediction), systematically generating digits biased toward smaller values. This "numerical hallucination" severely limits LLM applicability in domains requiring precise numerical outputs, such as finance and scientific computing.

**Limitations of Prior Work**: Previous research has focused on reasoning chain errors or tokenization issues without tracing the problem back to the digit distribution bias inherent in the training data itself. Methods such as Chain-of-Thought improve reasoning but cannot correct the underlying digit generation bias.

**Key Challenge**: Real-world data naturally conforms to Benford's Law (the leading digit 1 appears with ~30.1% probability), whereas numerical reasoning tasks require a uniform digit generation capability. The statistical properties of pretraining data are internalized by the model as a systematic bias.

**Goal**: (a) Demonstrate that LLM numerical bias genuinely originates from the Benford distribution in training data; (b) localize the specific neurons responsible for the bias; (c) explore lightweight correction methods.

**Key Insight**: The Logit Lens technique is employed to trace the evolution of digit preferences across Transformer layers, revealing that the bias concentrates in FFN modules of later layers rather than in attention modules. The DSC metric is used to precisely identify biased neurons.

**Core Idea**: Benford-distributed training data → digit-selective neurons in later FFN layers → systematic numerical hallucination → DSC localization + targeted pruning correction.

## Method

### Overall Architecture
Analyze digit distribution in pretraining corpora (verify Benford's Law) → construct a uniformly distributed digit bias benchmark (7 tasks, >1,000 examples/task) → trace per-layer digit preference trajectories via Logit Lens → propose DSC to quantify neuron digit selectivity → Spearman correlation analysis of FFN vs. attention → prune the top 0.01% FFN neurons most biased toward digit 1.

### Key Designs

1. **Digit Bias Benchmark**:

    - **Function**: Construct a numerical reasoning test set with uniformly distributed answers to isolate the model's generation bias.
    - **Mechanism**: Seven tasks (addition/subtraction, multiplication, division, expression evaluation, integer roots, linear equations, sequence summation), each with >1,000 examples where each digit (0–9) appears with ~10% probability in the answers. The deviation of the model's generated digit frequency from the uniform distribution is measured.
    - **Design Motivation**: If the ground-truth answers are unbiased while the model's outputs skew toward smaller digits, the bias must originate from the model itself. Analysis of "leading erroneous digits" further reveals that error positions also follow a Benford distribution.

2. **Digit Selectivity Coefficient (DSC) + Layer-wise Localization**:

    - **Function**: Quantify the preference of each FFN neuron for specific digits.
    - **Mechanism**: $\text{DSC}_i = S / \text{rank}(i)$, where $S$ is the sum of ranks across all digit tokens. Logit Lens analysis reveals that digit bias emerges sharply in layers 20–27 (near the final layers), with virtually no bias in early layers. Spearman correlation analysis shows that the DSC of FFN outputs is highly correlated with the DSC of the residual stream in later layers ($r=0.949$), whereas the attention module exhibits much weaker correlation.
    - **Design Motivation**: Precisely identifying "which layers, which modules, and which neurons" encode digit bias provides the foundation for targeted correction.

3. **Targeted Neuron Pruning**:

    - **Function**: Correct numerical hallucinations by removing neurons most biased toward digit 1.
    - **Mechanism**: FFN neurons are ranked by DSC, the top 0.01% most biased toward digit 1 are selected, and their weights are set to zero. Validated on LLaMA2-7B, Mistral-7B, and Qwen2.5-7B.
    - **Design Motivation**: Pruning an extremely small fraction of neurons (0.01%) minimizes damage to other model capabilities. Experiments show corrections of 1.36–3.49% in erroneous predictions, with the frequency of digit 1 reduced from 16.26% to 11.17%.

### Loss & Training
- This is a purely analytical study with inference-time intervention; no training is performed.
- Benford distribution is verified on the Olmo-Mix-1124 pretraining corpus.
- Results are validated across four model families (LLaMA2, Mistral, Qwen2.5, Gemma2).

## Key Experimental Results

### Main Results

| Model | Digit 1 Freq. (Original) | Digit 1 Freq. (After Pruning) | Correction Rate (Evaluate) | Correction Rate (GSM8k) |
|-------|--------------------------|-------------------------------|---------------------------|------------------------|
| LLaMA2-7B | 16.26% | 11.17% | 1.36% | 2.35% |
| Mistral-7B | 15.63% | 11.85% | 1.22% | — |
| Qwen2.5-7B | 16.45% | 14.72% | 3.49% | 2.12% |

(Target uniform distribution: 10% per digit)

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Layer at which bias emerges | Layers 20–27 (near the end); virtually no bias in early layers |
| FFN vs. attention | DSC correlation between FFN output and residual stream: $r=0.949$; attention module correlation is substantially weaker |
| Pearson correlation (corpus frequency vs. neuron preference) | $r=0.949$ (very strong positive correlation) |
| Leading erroneous digit distribution | Closely follows Benford's Law (more skewed than generated digits overall) |

### Key Findings
- Digit frequency in pretraining corpora closely conforms to Benford's Law (digit 1 ~30%, digit 9 ~5%).
- This distribution is internalized by specific neurons in later FFN layers—the number of neurons with the highest selectivity for digit 1 far exceeds those for digit 9.
- The model's "leading erroneous digit" is more biased than its overall output, indicating that the bias is strongest at decision boundaries.
- Pruning only 0.01% of neurons significantly reduces the over-generation of digit 1.
- A consistent pattern is observed across four distinct model families, suggesting this is a universal phenomenon.

## Highlights & Insights
- **Highly novel perspective via Benford's Law**: This is the first work to connect a classical statistical law to numerical hallucination in LLMs, providing a data-driven causal explanation (though the authors prudently refrain from claiming strict causality).
- **Impressive precision in layer-wise localization**: The bias is not uniformly distributed across the entire model but is concentrated in a very small fraction of neurons in later FFN layers—0.01% pruning yields a notable correction effect.
- **The Benford analysis of "leading erroneous digits" is particularly insightful**: It implies that the model tends to guess smaller digits when uncertain, suggesting an implicit frequency-based prior.

## Limitations & Future Work
- Experiments are conducted only on 7–9B models; whether the same phenomenon holds for larger models or MoE architectures remains unknown.
- Only correlation, not causality, is established—retraining on data with controlled digit distributions would be required to confirm a causal relationship.
- The pruning method is coarse-grained and may inadvertently harm correct predictions; the correction rate is only 1.36–3.49%.
- The analysis primarily targets models with single-digit tokenizers; models with multi-digit tokenizers (e.g., GPT-4) may behave differently.

## Related Work & Insights
- **vs. GoT/CoT and other reasoning enhancements**: Reasoning chains improve logic but do not correct the underlying digit bias; this paper directly addresses the root cause of the bias.
- **vs. numerical embedding methods**: Methods such as NumericalEncoding address token representations, whereas this paper focuses on the training data distribution and neuron-level preferences.
- **vs. tokenizer numerical representation research**: Approaches like xVal improve digit tokenization but do not address the digit distribution bias in training data.
- **vs. debiasing in NLP**: Text debiasing methods target social biases; digit debiasing in this paper constitutes an entirely new dimension.
- **Transferable techniques**: The DSC metric and conditional pruning approach can be generalized to analyze other types of generation bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Entirely novel Benford's Law perspective with deep layer-wise localization analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ Four model families + seven tasks + layer-wise analysis + pruning validation
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative progresses systematically from observation → mechanism → intervention, making it highly compelling
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental cause of numerical hallucination in LLMs with direct implications for model improvement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[NeurIPS 2025\] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs](matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)
- [\[NeurIPS 2025\] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)
- [\[ICLR 2026\] LLM DNA: Tracing Model Evolution via Functional Representations](../../ICLR2026/model_compression/llm_dna_tracing_model_evolution_via_functional_representations.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)

</div>

<!-- RELATED:END -->
