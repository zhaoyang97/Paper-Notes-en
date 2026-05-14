---
title: >-
  [Paper Note] Disentangling Latent Shifts of In-Context Learning with Weak Supervision
description: >-
  [NeurIPS 2025][Model Compression][In-Context Learning] WILDA treats ICL as a weak supervision signal and encodes demonstration-induced latent shifts into lightweight LoRA adapters via a teacher-student framework…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "In-Context Learning"
  - "Weak Supervision"
  - "Adapter Arithmetic"
  - "Latent Shift Disentanglement"
  - "LoRA"
date: 2026-05-08
content_hash: 16bbaab823e309fb
---

# Disentangling Latent Shifts of In-Context Learning with Weak Supervision

**Conference**: NeurIPS 2025
**arXiv**: [2410.01508](https://arxiv.org/abs/2410.01508)
**Code**: [github.com/josipjukic/wilda](https://github.com/josipjukic/wilda)
**Area**: Model Compression / LLM Efficiency
**Keywords**: In-Context Learning, Weak Supervision, Adapter Arithmetic, Latent Shift Disentanglement, LoRA

## TL;DR
WILDA treats ICL as a weak supervision signal and encodes demonstration-induced latent shifts into lightweight LoRA adapters via a teacher-student framework, enabling efficient inference without repeated prompting. The student surpasses the teacher through pseudo-label correction and coverage extension, demonstrating weak-to-strong generalization.

## Background & Motivation

**Background**: In-Context Learning (ICL) enables LLMs to perform few-shot learning from a small number of labeled demonstrations in the prompt without any parameter updates, making it a core adaptation mechanism in low-resource settings.

**Limitations of Prior Work**: (a) ICL is highly sensitive to the selection and ordering of demonstrations, leading to unstable predictions; (b) multi-shot demonstrations require long contexts, imposing inference inefficiency and context window constraints; (c) performance degrades when the number of demonstrations exceeds a threshold — ICL scales poorly.

**Key Challenge**: Existing disentanglement methods (ICV, Batch-ICL) directly manipulate attention heads or hidden states to isolate demonstration effects, but rely on linear attention approximations and neglect critical architectural components such as FFN layers, activation functions, and residual connections.

**Goal**: How can the effect of ICL demonstrations be "parameterized" into a reusable compact representation without modifying the model's internal states?

**Key Insight**: A functional perspective on ICL — focusing on final model outputs rather than intermediate states. ICL outputs themselves fully embody the effect of demonstrations and can serve as weak supervision signals.

**Core Idea**: Train lightweight adapters using ICL predictions as pseudo-labels, encoding demonstration-induced latent shifts as reusable parameters.

## Method

### Overall Architecture
WILDA adopts a teacher-student setup: the teacher is a standard ICL model (processing demonstrations + query), while the student shares the same architecture but is equipped with LoRA adapters (processing queries only). The student learns by minimizing the cross-entropy loss against the teacher's output distribution.

### Key Designs

1. **ICL as Weak Supervision**:

    - Function: Uses the full probability distribution from ICL (rather than hard labels) as the teacher signal.
    - Mechanism: The loss is $\sum_{x_q \in \mathcal{D}_{\text{unlab}}} \ell_{\text{CE}}(\mathbf{f}_{\text{teacher}}([\mathbf{X}_d^*; x_q]), \mathbf{f}_{\text{student}}(x_q))$, where $\mathcal{D}_{\text{unlab}}$ is an unlabeled dataset (as few as 100 samples suffice).
    - Design Motivation: Rather than directly manipulating attention heads or hidden states, this approach captures the complete influence of demonstrations from the model's output — incorporating the combined effect of attention, FFN, residual connections, and all other components.

2. **Adapter-Parameterized Latent Shifts**:

    - Function: Encodes demonstration-induced effects as LoRA weights $\mathbf{W}_{\text{ICL}}$.
    - Mechanism: Model parameters are decomposed as $\mathbf{W}_{\text{ZS}} \oplus \mathbf{W}_{\text{ICL}}$, where $\mathbf{W}_{\text{ZS}}$ is the zero-shot base and $\mathbf{W}_{\text{ICL}}$ is the ICL shift captured by the adapter. The final hidden state satisfies $\mathbf{h}_{\text{LLM}}(x_q | \mathbf{W}_{\text{ZS}} \oplus \mathbf{W}_{\text{ICL}}) = \mathbf{h}_{\text{LLM}}(x_q | \mathbf{W}_{\text{ZS}}) + \Delta \mathbf{h}_{\text{ICL}}$.
    - Design Motivation: The adapter accounts for only 0.1–0.3% of parameters, requires no demonstrations at inference time, and can be combined with new demonstrations.

3. **Adapter Arithmetic**:

    - Function: Trains independent adapters on multiple demonstration subsets and merges them via parameter summation.
    - Mechanism: A large demonstration pool is split into 2/4/8 subsets of 16 demonstrations each; separate adapters are trained and directly summed: $\mathbf{W}_{\text{ICL}}^{\text{merged}} = \sum_k \mathbf{W}_{\text{ICL}}^{(k)}$.
    - Design Motivation: Overcomes context window limitations, enabling the model to effectively leverage demonstration sets far exceeding the window length.

4. **Three Training Variants**:

    - wilda-f (fixed): Demonstration set remains fixed throughout training.
    - wilda-s (shuffle): Demonstrations are shuffled every epoch → mitigates order sensitivity.
    - wilda-r (resample): Demonstrations are resampled from a larger pool every epoch.

### Loss & Training
Cross-entropy loss aligns the student with the teacher's full probability distribution. Training runs for 10 epochs using LoRA adapters (only adapter parameters are updated). The same LLM instance alternates between teacher (adapter disabled) and student (adapter enabled) roles during training.

## Key Experimental Results

### Main Results (16-shot, 100 unlabeled samples, Llama 3 8B)

| Dataset | 0-shot | n-shot ICL | Batch-ICL | wilda-s | Gain (vs ICL) |
|--------|--------|-----------|-----------|---------|--------------|
| RTE | 62.3 | 75.1 | 77.8 | **86.0** | +10.9 |
| SST | 79.1 | 93.5 | 94.1 | **96.1** | +2.6 |
| QNLI | 64.3 | 77.0 | 78.0 | **81.4** | +4.4 |
| MNLI | 59.9 | 68.0 | 70.9 | **73.1** | +5.1 |
| CoLA | 44.6 | 58.5 | 59.8 | **64.3** | +5.8 |
| MRPC | 63.6 | 74.0 | 75.2 | **77.7** | +3.7 |
| QQP | 61.1 | 70.0 | 72.5 | **73.1** | +3.1 |
| Math (MMLU) | 31.5 | 43.5 | 36.2 | **49.5** | +6.0 |
| Misc (MMLU) | 62.5 | 84.0 | 81.0 | **88.0** | +4.0 |

The standard deviation of wilda-s is substantially lower than that of ICL (e.g., RTE: 0.6 vs. 6.5), indicating a significant improvement in stability.

### Ablation Study: Adapter Arithmetic (Llama 3 8B, Knowledge Fusion)

| Demo Combination | Method | RTE | SST | MMLU-Math | MMLU-Misc |
|----------|------|-----|-----|-----------|-----------|
| 2×16 | Batch-ICL | 80.2 | 95.3 | 43.5 | 83.0 |
| 2×16 | wilda-s | **87.1** | **96.4** | **51.5** | **89.5** |
| 4×16 | Batch-ICL | 84.4 | 96.4 | 45.5 | 84.5 |
| 4×16 | wilda-s | **88.4** | **97.5** | **53.5** | **91.0** |
| 8×16 | wilda-s | **92.8** | — | — | — |

Performance of wilda-s continues to improve as the number of subsets increases, demonstrating strong scalability.

### Key Findings
- **Weak-to-Strong Generalization**: The student consistently surpassing the teacher (ICL) is a robust phenomenon, achieved through two mechanisms: pseudo-label correction (correcting teacher errors) and coverage extension (generalizing to samples unseen by the teacher).
- wilda-s (shuffle) achieves the best overall performance — shuffling demonstrations effectively mitigates ICL's positional biases (primacy/recency effects).
- Only 100 unlabeled samples are required for effective adapter training, demonstrating high data efficiency.
- Strong OOD generalization: WILDA significantly outperforms ICL in cross-dataset transfer scenarios (e.g., QNLI→RTE).

## Highlights & Insights
- **Reframing ICL as Weak Supervision**: Rather than directly manipulating attention mechanisms, WILDA captures the full effect of ICL from the output side. This "black-box" perspective is more complete and more applicable to complex architectures.
- **Adapter Arithmetic for Extended-Context ICL**: By splitting, training, and merging adapters, WILDA bypasses context window limitations. The combined effect of 8×16=128 demonstrations substantially outperforms direct 128-shot ICL.
- **Extreme Parameter Efficiency**: LoRA accounts for only 0.1–0.3% of parameters; training requires 100 unlabeled samples and 10 epochs — minimal cost with substantial gains.

## Limitations & Future Work
- Adapter training still incurs some computational overhead (though lightweight), making it less flexible than direct ICL in pure inference settings.
- Pseudo-label quality depends on the teacher ICL's base capability — the approach may fail on tasks where ICL itself performs poorly.
- Validation is limited to classification tasks; effectiveness on generation tasks (e.g., summarization, translation) remains unknown.
- Simple parameter summation in adapter arithmetic may not be the optimal merging strategy; weighted or learned merging coefficients warrant exploration.
- Cross-model transferability of adapters is not discussed.

## Related Work & Insights
- **vs. ICV (In-Context Vectors)**: ICV extracts demonstration representations from hidden states and relies on linear attention approximations; WILDA learns from model outputs without assuming a specific attention mechanism.
- **vs. Batch-ICL**: Batch-ICL aggregates meta-gradients from multiple one-shot passes and still operates on internal states; WILDA's parameterized adapters offer greater flexibility.
- **vs. PBFT (Pattern-Based Fine-Tuning)**: PBFT fine-tunes on labeled data, whereas WILDA uses only unlabeled data with ICL pseudo-labels, better suiting few-shot scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The ICL-as-weak-supervision perspective is novel; adapter arithmetic is practically useful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 7 GLUE tasks + 2 MMLU subsets + 3 models, with OOD, stability, and fusion analyses.
- Writing Quality: ⭐⭐⭐⭐ Theoretical motivation is clear and experiments are comprehensive, though some content is redundant.
- Value: ⭐⭐⭐⭐ Provides an efficient and stable alternative to ICL; the adapter arithmetic approach has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Order-Level Attention Similarity Across Language Models: A Latent Commonality](order-level_attention_similarity_across_language_models_a_latent_commonality.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
