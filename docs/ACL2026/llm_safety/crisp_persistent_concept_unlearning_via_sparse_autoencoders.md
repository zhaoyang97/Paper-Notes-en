---
title: >-
  [Paper Note] CRISP: Persistent Concept Unlearning via Sparse Autoencoders
description: >-
  [ACL 2026][LLM Safety][SAE] Addressing the issue where SAE-based unlearning mostly relies on inference-time interventions while parameters still retain sensitive knowledge…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "SAE"
  - "Persistent Unlearning"
  - "WMDP"
  - "Contrastive Feature Selection"
  - "LoRA"
  - "Concept Suppression"
date: 2026-05-08
content_hash: 421722db4d9a1d3f
---

# CRISP: Persistent Concept Unlearning via Sparse Autoencoders

**Conference**: ACL 2026  
**arXiv**: [2508.13650](https://arxiv.org/abs/2508.13650)  
**Code**: https://github.com/technion-cs-nlp/CRISP  
**Area**: LLM Safety / Unlearning / Interpretability / SAE  
**Keywords**: SAE, Persistent Unlearning, WMDP, Contrastive Feature Selection, LoRA, Concept Suppression

## TL;DR
Addressing the issue where SAE-based unlearning mostly relies on inference-time interventions while parameters still retain sensitive knowledge, CRISP automatically identifies SAE features that are "strongly activated only on target" by comparing target/retain corpora. It then uses LoRA with a tripartite loss (unlearn + retain + coherence) to "solder" the activation of these features to zero. This achieves a new Pareto frontier on the unlearn-retain-fluency axes in WMDP-Bio/Cyber, outperforming ELM by 27–34 points and RMU by 5–8 points.

## Background & Motivation

**Background**: After deployment, LLMs often require the removal of hazardous knowledge (biological weapons/privacy/copyright). Mainstream unlearning follows two paths: (1) Direct parameter editing (RMU, ELM), which rewrites entire hidden states using random directions or self-classification; (2) SAE inference intervention, which clamps SAE feature activations corresponding to target concepts to minimum values.

**Limitations of Prior Work**: (1) Parameter editing causes collateral damage—deleting "how to enhance virus infectivity" also disrupts normal knowledge like "how the immune system fights viruses"; it also causes fluency to collapse (repetition or off-topic) on target concepts. (2) SAE inference intervention only modifies activations during inference; the hazardous knowledge in the parameters remains untouched and can be recovered if an attacker bypasses the hook in open-source scenarios.

**Key Challenge**: "Precision" (the monosemantic advantage of SAE) and "Persistence" (parameter-level editing) are currently decoupled: precise SAE methods are not persistent, and persistent methods are not precise enough.

**Goal**: Integrate the fine-grained interpretability of SAEs into model parameters to achieve: (a) Persistence (safe for open-source scenarios); (b) Precision (no spillover to neighboring benign concepts); (c) Fluency (retaining the ability to write coherent, factually consistent neutral content around target concepts).

**Key Insight**: Since SAE features have already decoupled concepts, why not "first identify target-specific features and then train the model to suppress these features itself"? Treat the SAE as a "concept compass," but solidify the compass direction into weights using LoRA.

**Core Idea**: CRISP = **Contrastive frequency/ratio feature selection** + **LoRA fine-tuning for self-suppression**, upgrading "inference-time clamping" to "training-time solidification."

## Method

### Overall Architecture

A two-stage pipeline that requires no modification to the SAE itself:

1. **Phase 1 — Feature Selection**: Pass $\mathcal{D}_{\text{target}}$ (forget corpus) and $\mathcal{D}_{\text{retain}}$ (retain corpus) through the model + pre-trained SAE to record token-level activations. Filter and select $\mathcal{F}_{\text{salient}}$ using a dual metric of "activation frequency difference + relative activation ratio."
2. **Phase 2 — Model Optimization**: Fine-tune the original model $M$ using LoRA. The goal is to suppress $\mathcal{F}_{\text{salient}}$ activations when encountering $\mathcal{D}_{\text{target}}$, while maintaining hidden states identical to the original model $M_0$ on $\mathcal{D}_{\text{retain}}$.

Operations are targeted at mid-layers (Layer 24 for Llama-3.1-8B, Layer 14 for Gemma-2-2B), where SAE feature decoupling is highest.

### Key Designs

1. **Automated Contrastive Salient Feature Selection**:

    - **Function**: Precisely locate a small number of features encoding "only target concepts" from hundreds of thousands of SAE features to avoid collateral damage.
    - **Mechanism**: Define two metrics—activation frequency difference $\Delta\phi(f_i)=\phi(f_i,\mathcal{D}_{\text{target}})-\phi(f_i,\mathcal{D}_{\text{retain}})$, selecting top-$k$ by frequency difference; and relative activation ratio $\rho(f_i)=A(f_i,\mathcal{D}_{\text{target}})/(A(f_i,\mathcal{D}_{\text{retain}})+\epsilon)$, followed by a secondary filter with threshold $\tau$. Finally, $\mathcal{F}_{\text{salient}}=\{f_i\in\mathcal{F}_{\text{freq}}\mid\rho(f_i)\ge\tau\}$.
    - **Design Motivation**: Frequency difference alone might misidentify shared features that are common to both but slightly more frequent in the target. Ratio alone might pick marginal features that almost only appear in the target but have negligible total activation. The intersection of both metrics yields truly salient features.

2. **Tripartite Loss + LoRA Persistence**:

    - **Function**: Write feature suppression into weights without destroying the original model structure or benign representations.
    - **Mechanism**: **Unlearn loss** $\mathcal{L}_{\text{unlearn}}=\mathbb{E}_{t\sim\mathcal{D}_{\text{target}}}\mathbb{E}_{f_i\sim\mathcal{F}_{\text{salient}}}[a_i^{(t)}+\lambda c_t]$ directly minimizes salient feature activation on target tokens; **Retain loss** $\mathcal{L}_{\text{retain}}=\mathbb{E}_{t\sim\mathcal{D}_{\text{retain}}}\|h_M^{(t)}-h_{M_0}^{(t)}\|_2^2$ keeps hidden states on the retain set consistent with the original model; **Coherence loss** uses 20 area-neutral sentences generated by Claude at the final layer to preserve fluency near target concepts. Total loss $\mathcal{L}=\alpha\mathcal{L}_{\text{unlearn}}+\beta\mathcal{L}_{\text{retain}}+\gamma\mathcal{L}_{\text{coherence}}$, updating only LoRA parameters.
    - **Design Motivation**: Using unlearn loss alone would "over-suppress" and crush retain performance. The tripartite joint loss creates a balance: "suppress target / preserve retain / ensure fluency." LoRA makes the edits reversible and parameter-efficient.

3. **Multi-layer Joint Intervention + Mid-layer Localization**:

    - **Function**: Allow unlearning to span multiple layers rather than a single layer hook.
    - **Mechanism**: Features are suppressed simultaneously across a pre-selected group of layers (near Layer 24 for Llama, Layer 14 for Gemma). Losses are calculated independently per layer and then averaged. Selection is based on the higher interpretability and finer concept granularity of SAE features in later layers on Neuronpedia.
    - **Design Motivation**: Single-layer suppression might be "compensated" by downstream layers. Mid-to-late layers are knowledge abstraction layers, ideal for concept-level rather than literal-level editing.

### Loss & Training
LoRA adapters (ranks detailed in Appendix) are used. A sweep of 200 hyperparameter sets per method is conducted, selecting the optimal configuration based on an aggregate score of unlearn + retain + MMLU on the validation set. MCQ pairs for validation/test are split 50/50. Overall = HM(100-U, R, M, F·50, C·50), where the Harmonic Mean (HM) penalizes any single-dimension weakness.

## Key Experimental Results

### Main Results (WMDP Bio / Cyber, 5 Dimensions + Overall HM)

| Model / Set | Method | Overall ↑ | Unlearn↓ | Retain↑ | MMLU↑ | Fluency↑ | Concept↑ |
|---|---|---|---|---|---|---|---|
| Bio / Llama-3.1-8B | Original | 56.60 | 68.29 | 76.81 | 61.15 | 1.24 | 1.77 |
| Bio / Llama-3.1-8B | ELM | 33.93 | 41.44 | 62.17 | 55.31 | 0.25 | 1.24 |
| Bio / Llama-3.1-8B | RMU | 52.51 | 34.54 | 67.75 | 59.50 | 0.56 | 1.58 |
| Bio / Llama-3.1-8B | **CRISP** | **60.10** | **30.93** | **74.13** | **60.28** | **0.77** | 1.58 |
| Bio / Gemma-2-2B | Original | 54.37 | 55.26 | 55.27 | 46.30 | 1.07 | 1.78 |
| Bio / Gemma-2-2B | ELM | 22.13 | 27.80 | 40.54 | 35.80 | 0.14 | 1.20 |
| Bio / Gemma-2-2B | RMU | 51.91 | **27.79** | 48.77 | 42.77 | 0.76 | 1.63 |
| Bio / Gemma-2-2B | **CRISP** | **56.70** | 29.67 | **54.45** | **46.33** | **0.92** | 1.63 |
| Cyber / Llama-3.1-8B | Original | 61.32 | 40.95 | 54.00 | 61.15 | 1.27 | 1.43 |
| Cyber / Llama-3.1-8B | ELM | 58.91 | 30.78 | 53.00 | 58.56 | 0.99 | 1.40 |
| Cyber / Llama-3.1-8B | RMU | 52.47 | 33.70 | 55.00 | 61.15 | 0.68 | 1.23 |
| Cyber / Llama-3.1-8B | **CRISP** | **61.74** | **29.38** | 53.00 | 58.86 | **1.14** | **1.49** |
| Cyber / Gemma-2-2B | Original | 52.57 | 33.90 | 39.00 | 46.30 | 1.05 | 1.46 |
| Cyber / Gemma-2-2B | ELM | 43.33 | 28.87 | 29.00 | 38.71 | 0.76 | 1.36 |
| Cyber / Gemma-2-2B | RMU | 44.79 | 28.67 | 36.00 | 44.79 | 0.64 | 1.23 |
| Cyber / Gemma-2-2B | **CRISP** | **49.02** | **27.26** | **38.00** | **46.26** | **0.81** | 1.28 |

CRISP takes first place in Overall score across all 4 (model, dataset) settings; Bio-Llama is +26.17/+7.59 relative to ELM/RMU, and Bio-Gemma is +34.57/+4.79. The gap narrows in Cyber but remains leading, suggesting unlearning is more challenging in domains like "Cybersecurity" where content is more dispersed.

### Ablation Study (Overall change after removing key designs, qualitative summary from §5–§6)

| Configuration | Bio-Llama Overall | Description |
|------|-------------------|------|
| Full CRISP | 60.10 | unlearn + retain + coherence + dual index feature selection |
| w/o Coherence loss | ↓ (fluency drops toward RMU's 0.56) | Generations near target concepts begin to repeat/veer off-topic |
| w/o Retain loss | ↓ (retain acc drops toward ELM's 62.17) | Benign knowledge is suppressed by association |
| w/o $\rho$ ratio filter (only $\Delta\phi$) | ↓ | Shared features are misidentified, leading to significant retain drop |
| Inference-time clamp (no LoRA training) | Non-persistent | Attacker can recover knowledge by bypassing the hook; weights still contain knowledge |

### Key Findings
- **Pareto Dominance**: Scatter plots of 200 hyperparameter sets show that almost all CRISP configurations are near the ideal point (random unlearn + no retain loss), followed by RMU, with ELM the furthest. This indicates CRISP is not only better at its peak but also more stable across its hyperparameter surface.
- **Significant Fluency Advantage**: On Bio-Gemma, ELM's fluency score is only 0.14 (garbled text); CRISP's 0.92 is close to the Original's 1.07. This suggests that the precision of "who to suppress" is more critical than the intensity of "how much to suppress."
- **Semantic Interpretability of Concept Separation**: Analysis of features selected at Llama L24 / Gemma L14 shows target features cluster around viruses/transmission/biothreat vectors, whereas benign features focus on anatomy/research methods. Shared features are mostly formatting tokens. Two features on Gemma were mislabeled by Neuronpedia as "flowers/finance" but actually activated on virus replication, indicating CRISP's selection criteria can identify true relevance even when metadata is flawed.
- **Cross-model Consistency**: The method exhibits similar distribution patterns across Llama and Gemma, suggesting the contrastive feature selection does not heavily depend on the specific training method of the SAE.

## Highlights & Insights
- **Translating "Interpretability Tools" into "Training Signals"**: Previously, SAEs were mainly used for probing/steering. Ours transforms "the SAE tells us this feature is concept X" directly into the differentiable objective of "suppress feature activation during training," marking the first time interpretability is persistently written into parameters.
- **Contrastive Dual Metrics (Frequency Difference + Intensity Ratio)**: This is a low-cost yet universal concept localization trick, transferable to any "delete/enhance a concept" scenario beyond unlearning (e.g., steering, debiasing, style control).
- **Tripartite Balance of "Suppress / Preserve / Fluency"**: Explicitly decoupling previously bundled unlearning goals into three axes allows hyperparameter sweeps to directly reach the Pareto frontier, providing a clean methodology.
- **Threat Model Awareness**: The paper explicitly clarifies that "inference intervention does not count as unlearning in open-source scenarios," which is more honest than many RLHF/safety papers and clearly defines the true boundaries of safety.

## Limitations & Future Work
- **Dependence on Pre-trained SAE Quality**: When a target concept is dispersed across multiple polysemantic features, contrastive metrics may fail; the 4008/11127 mislabeling on Gemma is a sign. Higher quality, finer-grained SAEs or online SAE fine-tuning are needed.
- **Limited Validation Domains**: Only tested on WMDP (Bio/Cyber) + Harry Potter; copyright, multimodal, long-context/RAG, and hazardous knowledge in dialogue history remain untested. Effect on "instruction-tuned aligned models" is uncertain, as base model performance may not generalize.
- **No Formal Unlearning Guarantees**: The authors acknowledge that residual knowledge may exist in a distributed manner and did not perform robust evaluations against adversarial extraction (e.g., fine-tune attacks, logit-leakage attacks). The next step is clearly to include adversarial finetuning recovery experiments.
- **High Cost of 200-set Hyperparameter Sweep**: Since the HM score is sensitive to any single-axis weakness, a broad sweep is required to find the balance point, leading to non-trivial costs when applying to new domains.

## Related Work & Insights
- **vs RMU (Li et al. 2024)**: RMU pushes target hidden states toward random directions, which is a coarse-grained full-state perturbation; CRISP only suppresses specific SAE-selected directions, leading to 6+ points higher retain and double the fluency.
- **vs ELM (Gandikota et al. 2024)**: ELM uses self-classification + LoRA to modify early layers to align target representations with "benign stand-ins," which causes off-topic or garbled text. CRISP performs feature-level suppression in mid-layers, avoiding global representation shift.
- **vs Farrell et al. 2024 (SAE clamp)**: Use inference-time clamping; CRISP converts the same feature selection target into a LoRA training target for persistence.
- **vs PISCES (Gur-Arieh et al. 2025)**: Also uses SAE for persistent unlearning but requires manual feature selection and only edits FFN $W_2$; CRISP automates selection and modifies the full attention/FFN stack via LoRA, making it more scalable.
- **Insight**: The paradigm of "compositional interpretability tool (SAE / circuits / DAS) → differentiable training loss" can be extended to model editing, debiasing, refusal alignment, and style removal. CRISP serves as a highly complete template.

## Rating
- Novelty: ⭐⭐⭐⭐ Persistence of inference-time SAE clamp is a direct but significant step. The combination of dual metrics and tripartite loss is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 models × 2 datasets × 200 hyperparameter sweeps + Pareto plots + semantic analysis; half a star deducted for the lack of adversarial robustness and instruction model testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from motivation to analysis. Clear HM score formula and threat model discussions. Table 2 provides a compelling qualitative comparison.
- Value: ⭐⭐⭐⭐ A new SOTA in persistent unlearning for LLM safety, with a transferable methodology for steering/debiasing. A representative work for SAE application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/llm_safety/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](../../NeurIPS2025/llm_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](cap_controllable_alignment_prompting_for_unlearning_in_llms.md)
- [\[ACL 2026\] SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation](sage_sparse_adaptive_guidance_for_dependency-aware_tabular_data_generation.md)

</div>

<!-- RELATED:END -->
