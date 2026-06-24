---
title: >-
  [Paper Note] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs
description: >-
  [ICML2025][LLM Safety][fine-tuning analysis] This paper proposes the Tuning Contribution (TuCo) metric, which precisely decomposes the forward pass of a fine-tuned LLM into a Pre-Training Component (PTC) and a Fine-Tuning Component (FTC). This enables the first instance-level (per-prompt) quantitative analysis of fine-tuning's contribution during inference and reveals that jailbreak attacks bypass safety guardrails by weakening the magnitude of the FTC.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "fine-tuning analysis"
  - "interpretability"
  - "jailbreak attacks"
  - "residual decomposition"
  - "internal Transformer representations"
date: 2026-05-08
content_hash: ff0e8ec9412ba076
---

# TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs

**Conference**: ICML2025  
**arXiv**: [2506.23423](https://arxiv.org/abs/2506.23423)  
**Code**: [github.com/FelipeNuti/tuning-contribution](https://github.com/FelipeNuti/tuning-contribution)  
**Area**: LLM Analysis / AI Safety  
**Keywords**: fine-tuning analysis, interpretability, jailbreak attacks, residual decomposition, internal Transformer representations

## TL;DR

This paper proposes the Tuning Contribution (TuCo) metric, which precisely decomposes the forward pass of a fine-tuned LLM into a Pre-Training Component (PTC) and a Fine-Tuning Component (FTC). This enables the first instance-level (per-prompt) quantitative analysis of fine-tuning's contribution during inference and reveals that jailbreak attacks bypass safety guardrails by weakening the magnitude of the FTC.

## Background & Motivation

- **Lack of fine-grained measurements for fine-tuning efficacy**: Existing studies analyze the impact of fine-tuning on LLMs solely at the dataset level (e.g., benchmark performance, mechanistic interpretability), lacking quantitative methods targeted at individual prompt outputs.
- **Mechanistic hypotheses of jailbreak attacks lack quantitative verification**: Wei et al. (2024) and Kotha et al. proposed that jailbreak attacks exploit the "competition" between pre-training and fine-tuning objectives, but this hypothesis has never been directly formalized or measured.
- **Hidden states vs. final outputs**: Fine-tuning can significantly alter intermediate hidden states without affecting the final token predictions. Therefore, the entire forward pass needs to be examined rather than solely comparing the final outputs.

## Method

### 1. Exact Decomposition: PTC and FTC

For a Transformer with an $L$-layer residual structure, given a fine-tuned model $\mathcal{T}^{\text{FT}}_\Theta$ and its corresponding pre-trained model $\mathcal{T}^{\text{PT}}_\phi$, the update at each layer can be decomposed as:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \underbrace{f^{\text{PT}}_\phi(\mathbf{x}_l, l)}_{\text{PTC}_l} + \underbrace{f^{\text{FT}}_\Theta(\mathbf{x}_l, l) - f^{\text{PT}}_\phi(\mathbf{x}_l, l)}_{\text{FTC}_l}$$

- **PTC (Pre-Training Component)**: The output of the corresponding layer in the pre-trained model, representing the computational circuits formed during pre-training.
- **FTC (Fine-Tuning Component)**: The difference between the outputs of the fine-tuned and pre-trained models at the same layer given the same input, representing the new computational circuits introduced by fine-tuning.

This decomposition holds **exactly** for all residual-stream Transformers without assuming pre-existing knowledge of specific circuit decompositions.

### 2. Definition of TuCo

Accumulating the FTC and PTC of the last token across all layers:

$$I^{\text{FTC}} = \sum_{l=0}^{L-1} \text{FTC}_l[-1], \quad I^{\text{PTC}} = \sum_{l=0}^{L-1} \text{PTC}_l[-1]$$

Tuning Contribution is defined as:

$$\text{TuCo}(\mathbf{x}) = \frac{\|I^{\text{FTC}}\|}{\|I^{\text{PTC}}\| + \|I^{\text{FTC}}\|}$$

The value of TuCo ranges within $[0, 1]$, where a higher value indicates a greater impact of fine-tuning on the response to the corresponding prompt.

### 3. Grönwall Theoretical Bound

The paper proves a discrete Grönwall bound: when the PTC is bounded and Lipschitz,

$$\|\mathbf{x}^{\text{FT}}_L - \mathbf{x}^{\text{PT}}_L\|_1 \leq L \|\text{PTC}\|_{\sup} (1 + \|\text{PTC}\|_{\text{Lip}})^L \frac{\beta}{1-\beta}$$

where $\beta = \max_l \frac{\|\overline{\text{FTC}}_l\|_1}{\|\overline{\text{PTC}}_l\|_1 + \|\overline{\text{FTC}}_l\|_1}$, theoretically guaranteeing that the output of the fine-tuned model is close to the pre-trained model when the FTC is small.

### 4. FTC α-Scaling

Regulating the magnitude of the fine-tuning component via a scaling factor $\alpha$:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{PTC}(\mathbf{x}_l, l) + \alpha \cdot \text{FTC}(\mathbf{x}_l, l)$$

$\alpha=1$ recovers the fine-tuned model, while $\alpha=0$ approximates the behavior of the pre-trained model.

## Key Experimental Results

**Model Coverage**: Llama 2 (7B/13B), Llama 3 (8B), Gemma 7B, Vicuna v1.5 (7B/13B), Mistral (V0.1/V0.2 7B), and Zephyr Gemma 7B, totaling 9 open-source models.

### Experiment 1: Controlling Model Behavior via FTC α-Scaling

| Experiment | Metric | Results |
|------|------|------|
| MMLU (57 tasks) | Accuracy improvement under optimal $\alpha$ | 1.03%–2.69% (significant in 71% of tasks) |
| MWE Behavioral Evaluation | Maximizing behavioral consistency | +1.55%–5.18% (significant across all models) |
| MWE Behavioral Evaluation | Minimizing behavioral consistency | -2.80% to -25.24% |
| Christian Belief Alignment (Llama2 13B) | $\alpha=1.25$ vs $\alpha=1.0$ | +24% alignment rate |

### Experiment 2: TuCo Discriminability between Web Text vs. Chat Inputs

| Model | AUC (OpenWebText vs. HH-RLHF) |
|------|------|
| Llama 2 7B/13B | 1.00 |
| Vicuna 7B/13B | 0.99 |
| Gemma 7B | 0.93 |
| Llama 3 8B | 1.00 |

### Experiment 3: Jailbreak Attacks Reduce TuCo

| Attack Type | Model | AUC (Attacked vs. Unattacked) |
|------|------|------|
| GCG Gradient Attack | Llama 2 7B | 1.00 |
| GCG Gradient Attack | Llama 2 13B | 0.80 |
| Congruent Prompting (En vs. Ml/Sw) | Llama 2 13B | 1.00 |
| Many-Shot | All models | TuCo monotonically decreases with the number of shots |

### Experiment 4: Lower TuCo for Successful Jailbreaks

| Model | Successful Jailbreak AUC | Vanilla Jailbreak Rate | GCG Jailbreak Rate |
|------|------|------|------|
| Llama 2 13B | **0.87** | 0.19% | 1.1% |
| Llama 2 7B | 0.83 | 0.19% | 16.36% |
| Gemma 7B | 0.94 | 6.92% | 7.42% |
| Vicuna 7B | 0.87 | 29.23% | 85.13% |

### Experiment 5: Discrepancy between TuCo and OutputCo

OutputCo only compares the final hidden states, whereas TuCo inspects the entire forward pass. In experiments where numerous refusal exemplars are followed by a harmless query, OutputCo decreases as the number of exemplars increases (indicating the model quickly learns to refuse), while TuCo conversely increases (reflecting enhanced activity of internal fine-tuning circuits). This demonstrates that the two metrics capture distinct information.

## Highlights & Insights

1. **Theoretical Rigor**: Starting from a formal definition of Generalized Components, the authors prove that any fine-tuned Transformer can be exactly decomposed into PTC + FTC, requiring no prior assumptions about circuit structures.
2. **First Prompt-Level Fine-Tuning Contribution Metric**: Computable at inference time and applicable to billion-parameter scale models, with a computational overhead of approximately one additional forward pass.
3. **Quantitative Evidence of Jailbreak Mechanisms**: Three mainstream attacks (GCG, congruent prompting, and Many-Shot) all significantly reduce TuCo, and **TuCo decreases even further when the attack succeeds** (AUC up to 0.87). This directly quantifies the hypothesis that "jailbreaking equals weakening the fine-tuning effect."
4. **The Ranking of TuCo in Low-Resource Languages Aligns Perfectly with Web Corpus Shares**: English > Japanese > Hungarian > Swahili/Malayalam, revealing a direct relationship between fine-tuning coverage and the training data distribution.
5. **FTC α-Scaling practically modulates model behavior**, achieving a 1–3% performance gain on MMLU, though the authors emphasize this is for verification rather than the ultimate goal.

## Limitations & Future Work

1. **Requires Simultaneous Access to Both Pre-Trained and Fine-Tuned Models**: This is inapplicable to closed-source models (e.g., GPT-4, Claude), which limits real-world deployment scenarios.
2. **Computational Overhead**: Conducting forward passes on two models simultaneously poses a burden for large-scale deployment.
3. **TuCo Is Not an Attack Detection Tool**: Despite the high AUC, the authors explicitly state that TuCo is an analytical tool rather than a defense mechanism. Using it directly for real-time detection might be vulnerable to adversarial bypass.
4. **Limited Model Scale**: Evaluations were only conducted on models up to 13B parameters; its applicability to larger models (70B+) or MoE architectures remains unverified.
5. **Applicability to PEFT Methods (e.g., LoRA)**: The paper does not specifically discuss the characteristics of FTC under parameter-efficient fine-tuning methods like LoRA/QLoRA.
6. **Causality vs. Correlation**: Although decreased TuCo is highly correlated with successful jailbreaks, a causal relationship has not yet been established.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first metric to quantify fine-tuning contributions on a per-prompt basis, complete with thorough theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 9 models × 3 attack types × multiple evaluation tasks; the ablation is comprehensive but the model scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logic chain flows clearly and rigorously from theoretical motivation to formal definition, and finally to experimental validation.
- Value: ⭐⭐⭐⭐ — Provides a new dimension for LLM safety and interpretability, though dependency on the availability of the pre-trained model limits its practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](../../ACL2025/llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[ICML 2025\] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)
- [\[ACL 2025\] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](../../ACL2025/llm_safety/from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)
- [\[ICML 2025\] Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)

</div>

<!-- RELATED:END -->
