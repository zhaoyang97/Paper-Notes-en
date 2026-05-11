---
title: >-
  [Paper Note] ASIDE: Architectural Separation of Instructions and Data in Language Models
description: >-
  [ICLR 2026][LLM Evaluation][instruction-data separation] This paper proposes ASIDE, an architectural modification that distinguishes instructions from data at the token embedding level via orthogonal rotation. Requiring…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "instruction-data separation"
  - "prompt injection"
  - "orthogonal rotation"
  - "token embedding"
  - "architectural safety"
date: 2026-05-08
content_hash: 5c883613f2a07eba
---

# ASIDE: Architectural Separation of Instructions and Data in Language Models

**Conference**: ICLR 2026
**arXiv**: [2503.10566](https://arxiv.org/abs/2503.10566)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: instruction-data separation, prompt injection, orthogonal rotation, token embedding, architectural safety

## TL;DR
This paper proposes ASIDE, an architectural modification that distinguishes instructions from data at the token embedding level via orthogonal rotation. Requiring only changes to the forward pass and training on standard instruction fine-tuning data, ASIDE significantly improves instruction-data separation and robustness against prompt injection without any dedicated safety training.

## Background & Motivation
**Background**: LLMs are widely integrated into software systems such as email clients and agent pipelines, where inputs naturally fall into two categories—instructions (to be executed) and data (to be processed but not executed). However, current LLM architectures apply identical embeddings to both, making it impossible for the model to distinguish them internally.

**Limitations of Prior Work**: The absence of instruction-data separation is the root cause of successful prompt injection attacks (both indirect and direct). Existing defenses either rely on prompt engineering or special delimiters (which are easily bypassed) or on adversarial training (which targets only specific attack patterns), and neither addresses the problem fundamentally.

**Key Challenge**: In conventional LLMs, a token carries identical embeddings regardless of whether it appears in an instruction or in data. The model must infer the functional role of each token from context alone—a task that is extremely difficult to perform reliably in deep networks.

**Goal**: How can a model distinguish instruction tokens from data tokens from the very first layer, without adding parameters or repretraining?

**Key Insight**: Token embeddings typically exhibit low-rank structure; instructions and data can share the same high-dimensional space while residing in different linear subspaces. Orthogonal rotation can create this separation without altering embedding norms or inner-product structure.

**Core Idea**: Apply a fixed $\frac{\pi}{2}$ orthogonal rotation to the embeddings of data tokens, enabling the model to distinguish instructions from data via embeddings from the first layer onward.

## Method

### Overall Architecture
ASIDE modifies only the forward pass of the LLM's embedding layer. For an input token $x$, the embedding is $E[I_x, \cdot]$ (the original embedding) if the token belongs to an instruction, and $R(E[I_x, \cdot])$ if it belongs to data, where $R \in \mathbb{R}^{d \times d}$ is a fixed orthogonal rotation matrix. The model is then fine-tuned with standard SFT on the Alpaca-clean dataset.

### Key Designs

1. **Isoclinic Orthogonal Rotation**:

    - **Function**: Rotates data token embeddings into an orthogonal subspace.
    - **Mechanism**: The embedding dimensions are grouped into pairs, and each pair is transformed by the $\frac{\pi}{2}$ rotation matrix $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. This is a fixed, non-learnable transformation.
    - **Design Motivation**: Orthogonal rotation preserves vector norms and relative angles (introducing no information loss) while creating two fully orthogonal subspaces. This is more effective than the learnable offset vectors used in ISE (Wu et al., 2024), which are gradually absorbed by the model in deeper layers and lose their discriminative power. Rotation maintains permanent geometric separability.
    - **Zero additional parameters**: The rotation matrix is fixed and introduces no trainable parameters.

2. **Functional Role Annotation**:

    - **Function**: Marks each token at deployment time as either instruction or data.
    - **Mechanism**: Leverages role information already available in the system design (e.g., email body is always data; system prompt is always instruction), requiring no inference by the model.
    - **Limitation**: Requires that the deployment scenario can provide token-level role annotations; not applicable to general-purpose chat settings where the boundary between instruction and data is ambiguous.

3. **Backward-Compatible Integration Pipeline**:

    - **Function**: Integrates ASIDE into a pretrained model.
    - **Steps**: (1) Modify the forward pass to incorporate the rotation logic; (2) Fine-tune for 3 epochs on standard SFT data (no safety-specific data).

### Loss & Training
Standard SFT with no adversarial training and no safety-specific objective. Training is performed on the Alpaca-clean-gpt4-turbo dataset (51.8k samples) with learning rate in $[1 \times 10^{-6}, 2 \times 10^{-5}]$, batch size 64–256, and warm-up ratio in $[0, 0.1]$.

## Key Experimental Results

### Main Results: Instruction-Data Separation (SEP Score)

| Model | Vanilla | ISE | ASIDE | Gain (vs. Vanilla) |
|-------|---------|-----|-------|--------------------|
| Llama 2 7B | ~55% | ~52% | ~67% | +12.3 pp |
| Llama 3.1 8B | ~50% | ~53% | ~70% | +20 pp |
| Qwen 2.5 7B | ~57% | ~57% | ~75% | +18 pp |
| Qwen 3 8B | ~31% | ~20% | ~65% | +34 pp |
| Mistral 7B | ~28% | ~50% | ~72% | +44.1 pp |

ASIDE's utility (AlpacaEval, SEP Utility) remains on par with the Vanilla baseline.

### Prompt Injection Robustness (ASR↓)

| Model | Attack Type | Vanilla | ASIDE | Reduction |
|-------|-------------|---------|-------|-----------|
| Llama 3.1 8B | BIPIA-text | 13.6% | 4.1% | −9.5 pp |
| Llama 3.1 8B | BIPIA-code | 22.8% | 9.2% | −13.6 pp |
| Llama 3.1 8B | StruQ-ID | 43.3% | 41.3% | −2.0 pp |
| Qwen 2.5 7B | BIPIA-text | 18.3% | 14.5% | −3.8 pp |
| Qwen 3 8B | BIPIA-text | 10.2% | 2.8% | −7.4 pp |
| Qwen 3 8B | StruQ-ID | 47.0% | 8.1% | −38.9 pp |
| Mistral 7B | BIPIA-text | 11.1% | 0.5% | −10.6 pp |
| Mistral 7B | StruQ-ID | 33.4% | 9.6% | −23.8 pp |

### Key Findings
- ASIDE consistently improves the SEP score across all models by 12–44 pp while maintaining utility.
- Indirect prompt injection ASR is reduced by approximately 10–40 pp on average, with particularly pronounced improvements on Mistral and Qwen3.
- ISE shows no statistically significant improvement over the Vanilla baseline on most models and is sometimes worse, indicating that learnable offset vectors are insufficient to maintain separation in deeper layers.
- Linear probing analysis reveals that ASIDE achieves 100% linear separability from layer 0 (immediately after embedding), whereas the Vanilla model only gradually becomes separable at layers 5–10.
- Concept activation analysis shows that ASIDE effectively suppresses spurious activation of "instruction concepts" within data regions.

## Highlights & Insights
- **Solving security problems architecturally**: By analogy to data execution prevention (DEP) in computer security, ASIDE enforces a hardware-level distinction between executable and non-executable memory. This work is the first to successfully transfer this principle to LLMs.
- **Zero-cost security gains**: Significant security improvements are achieved without adversarial training, safety datasets, or additional parameters—relying solely on a fixed rotation and standard SFT. This finding is particularly compelling.
- **Design insight: rotation vs. offset**: ISE creates separation in embedding space via learnable offsets, but these offsets are gradually "absorbed" by the model as depth increases. Orthogonal rotation is geometrically more durable because it creates orthogonal subspaces rather than merely offset regions within the same subspace.

## Limitations & Future Work
- Deployment requires knowledge of each token's functional role (instruction vs. data), which restricts applicability. In general-purpose chat assistants, user inputs can simultaneously serve as instructions and contain data, making role boundaries ambiguous.
- Experiments are not conducted on safety-fine-tuned Instruct models (intentionally, but Instruct models are standard in practice); the combined effect of ASIDE and safety tuning remains unknown.
- Improvement on StruQ-OOD attacks is limited, indicating that out-of-distribution injection remains a challenge.
- Whether the fixed $\frac{\pi}{2}$ rotation is optimal is not investigated; other rotation angles or learnable rotations are not explored.

## Related Work & Insights
- **vs. ISE (Wu et al., 2024)**: ISE uses learnable offset vectors to differentiate roles but loses separation in deeper layers, performing comparably to or worse than the Vanilla baseline on safety metrics. ASIDE achieves more durable separation via orthogonal rotation.
- **vs. Prompt Engineering**: Delimiters and special tokens perform separation at the input level and can be forged by adversaries; ASIDE operates at the embedding level, where rotation cannot be manipulated through text.
- **vs. Adversarial Training**: Adversarial training defends only against observed attack patterns; ASIDE provides a structural, attack-agnostic defense.
- The method can be directly extended to multi-level instruction hierarchies (e.g., system > user > tool) by defining additional orthogonal transformations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work to achieve instruction-data separation at the architectural level; the concept is clear and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 models × 8 safety benchmarks + separation evaluation + interpretability analysis, though Instruct models are not tested.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is grounded in classical principles from computer security; the writing is fluent and logically rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Introduces a novel safety enhancement pathway that is both practical and theoretically well-founded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](../../AAAI2026/llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[CVPR 2026\] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](../../CVPR2026/llm_evaluation/reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)

</div>

<!-- RELATED:END -->
