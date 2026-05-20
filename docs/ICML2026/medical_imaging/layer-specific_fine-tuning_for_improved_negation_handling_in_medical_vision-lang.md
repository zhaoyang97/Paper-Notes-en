---
title: >-
  [Paper Note] Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models
description: >-
  [ICML 2026][Medical Imaging][Medical CLIP] NAST uses causal tracing to compute the causal contribution (CTE) of each layer in the CLIP text encoder for negation understanding…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Medical CLIP"
  - "Negation Understanding"
  - "Causal Tracing"
  - "Layered Fine-Tuning"
  - "LoRA"
date: 2026-05-08
content_hash: d901fa61ad4c3220
---

# Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.12498](https://arxiv.org/abs/2602.12498)  
**Code**: https://github.com/healthylaife/NAST  
**Area**: Multimodal VLM / Medical Imaging / Interpretability-Guided Training  
**Keywords**: Medical CLIP, Negation Understanding, Causal Tracing, Layered Fine-Tuning, LoRA

## TL;DR
NAST uses causal tracing to compute the causal contribution (CTE) of each layer in the CLIP text encoder for negation understanding, then applies these CTEs for layer-wise gradient scaling in LoRA fine-tuning. This significantly enhances the semantic sensitivity of medical VLMs in distinguishing "presence/absence of symptoms," reducing the affirmative-negation accuracy gap from 21.6% to 4.2%.

## Background & Motivation
**Background**: Medical VLMs such as MedCLIP, BioMedCLIP, and BioViL-T have shown remarkable performance in image-report alignment and zero-shot diagnosis, and have been explored for automatic report generation, retrieval, and decision support.

**Limitations of Prior Work**: Negation is ubiquitous in radiology reports—e.g., "no pneumothorax," "no pleural effusion," "no consolidation in the right lower lobe." Negation is not just "absence of an object," but often applies to attributes ("no large effusion," "not right lower lobe consolidation"). However, medical VLMs are mainly pre-trained on affirmative descriptions, treating negation as a blind spot: using controlled "affirmative vs. negation semantically equivalent sentences" (e.g., "normal heart size" vs. "no cardiomegaly"), it is found that all mainstream medical VLMs systematically prefer affirmative sentences, with significantly worse negation understanding.

**Key Challenge**: Simply adding negation samples for fine-tuning (as in NegCLIP, ConCLIP, NegBench) only marginally alleviates the issue, because negation signals are not evenly distributed across model layers—they are likely concentrated in a few layers of the text encoder. Uniform parameter tuning is inefficient and may degrade other capabilities.

**Goal**: (i) Provide a polarity-controlled diagnostic benchmark to distinguish "poor negation understanding" from "poor adjective understanding"; (ii) Provide a fine-tuning dataset that injects "negation knowledge" at the attribute level (existence/location/severity) into medical VLMs; (iii) Use causal interpretability tools to identify "which layers handle negation," and selectively fine-tune them to improve negation capability while preserving non-negation abilities.

**Key Insight**: Transfer mechanistic interpretability tools (causal tracing, Meng et al.) from LLMs to the CLIP text encoder, making "which layer, which token is sensitive to negation" computable as CTE scores, and directly feeding them to the optimizer for layer-wise gradient scaling.

**Core Idea**: Use causal tracing to compute CTE → normalize as layer weights $\alpha_\ell$ → during LoRA fine-tuning, scale each layer's gradient by $\alpha_\ell^\beta$, focusing training resources on the layers truly responsible for negation.

## Method

### Overall Architecture
NAST consists of three components: (i) MedNega-CXR diagnostic benchmark—LLM-generated affirmative-negation MCQ pairs based on MIMIC-CXR, reviewed by two radiologists; (ii) Contextual negation fine-tuning dataset—using CAD annotations, each structured fact $(\text{condition}, \text{existence}, \text{location}, \text{severity})$ is perturbed by changing only one attribute, yielding about one million image-text pairs; (iii) NAST algorithm—first, causal tracing computes CTE for each layer and position in the text encoder, then layer-wise weighted gradient updates are used for LoRA fine-tuning, targeting a weighted sum of contrastive loss and claim-ranking loss.

### Key Designs

1. **MedNega-CXR Diagnostic Benchmark (polarity-controlled MCQ pairs)**:

    - **Function**: Directly compares pairs of descriptions that are semantically equivalent except for polarity, isolating negation understanding from other confounding abilities.
    - **Mechanism**: Selects studies from MIMIC-CXR/CheXpert with ≥2 positives and ≥3 negatives, and collaborates with radiologists to find affirmative equivalents for each negative condition ("no cardiomegaly" ↔ "normal heart size"). Three-step process: construct contrastive label arrangements (hard negatives) → LLM generates explicit negation MCQs → another LLM replaces negation phrases with affirmative equivalents while preserving structure. Yields 6,965 MCQ pairs differing only in polarity.
    - **Design Motivation**: Medicine offers a unique advantage—"no pneumonia" can be equivalently expressed as "lungs are well aerated," enabling clean contrastive pairs; in general domains, "no car" lacks a single affirmative equivalent. This controlled contrast ensures the evaluation truly tests negation understanding, not adjective understanding or visual perception.

2. **CAD-Based Attribute-Level Negation Fine-Tuning Dataset**:

    - **Function**: Ensures fine-tuning supervision covers clinically realistic forms of negation—existence, location, and severity counterfactuals.
    - **Mechanism**: For each real fact $(c, e, l, s)$, generate counterfactuals by changing only one attribute (present↔absent, left↔right, small↔large, etc.), and convert to natural language using radiology-style templates. Two supervision formats: (a) claim-based contrast set—one correct claim plus multiple hard negatives; (b) single negation captions for auxiliary contrastive training.
    - **Design Motivation**: Existing negation datasets (CC-Neg, NegBench) mainly address object presence, lacking the attribute-level negation crucial in medicine. This work uses structured annotations and controlled perturbations to generate 1M pairs, sufficient in both scale and specificity.

3. **CTE-Weighted Layered LoRA Fine-Tuning**:

    - **Function**: Directly translates interpretability-derived "which layers handle negation" into "which layers are updated more."
    - **Mechanism**: (i) Use causal tracing as a causal probe on the CLIP text encoder—pair (correct caption, foil caption) of equal length, record foil forward hidden states, then during correct caption forward pass, replace the $\ell$-th layer, $p$-th token with the foil's hidden state to obtain $S^{\ell,p}$; CTE $(\ell, p) = (S^{\text{corr}} - S^{\ell,p}) / (S^{\text{corr}} - S^{\text{foil}})$. Results show negation signals concentrate in layers 1-4, peaking at layer 2. (ii) Aggregate token-level CTEs per layer to get $\mathrm{CTE}_\ell$, min-max normalize to $\alpha_\ell \in [0,1]$. (iii) During LoRA fine-tuning, scale gradients as $\tilde{g}_\ell = \alpha_\ell^\beta \cdot g_\ell$, with $\beta$ controlling concentration; total loss $\mathcal{L}_{\text{total}} = \lambda \mathcal{L}_{\text{CLIP}} + (1-\lambda) \mathcal{L}_{\text{claim}}$.
    - **Design Motivation**: Uniform LoRA fine-tuning updates all layers, consuming pre-trained capabilities and diluting negation signal learning; focusing updates on "layers truly responsible for negation" is a more efficient and safer injection method. Using $\alpha_\ell^\beta$ instead of directly multiplying learning rates preserves a global learning rate for training stability.

### Loss & Training
$\mathcal{L}_{\text{CLIP}}$ is the standard CLIP symmetric contrastive loss (applied to batches with explicit negation captions); $\mathcal{L}_{\text{claim}} = \frac{1}{M}\sum_i \log \frac{\exp(\ell_{i, c_i})}{\sum_j \exp(\ell_{i, j})}$ is the claim-ranking loss (encouraging the correct claim to have higher similarity than hard negatives). The optimizer is AdamW, with fixed learning rate, trained on a single RTX 4070. $\lambda$ and $\beta$ are key hyperparameters.

## Key Experimental Results

### Main Results
Contextual negation task (Table 1, unit: %):

| Model | R@1↑ | R@5↑ | Claim Acc.↑ |
|------|------|------|-------------|
| CLIP | 23.5 | 34.7 | 24.6 |
| NegCLIP | 36.2 | 52.4 | 41.3 |
| ConCLIP | 39.7 | 55.8 | 44.9 |
| NegBench | 43.1 | 59.2 | 48.7 |
| **NAST (Ours)** | **49.5** | **65.7** | **55.6** |

Negation-focused baselines improve incrementally, but NAST further increases claim accuracy by 6.9 points over the strongest baseline.

### Ablation Study
Affirmative-negation gap (Table 3, lower is better) + update distribution (Table 4):

| Model | Affirm – Negation Gap (Claim Acc., %) |
|------|--------------------------------------|
| CLIP | 21.6 |
| NegCLIP | 12.8 |
| ConCLIP | 10.7 |
| NegBench | 10.2 |
| **NAST** | **4.2** |

| Method | Top-3 Layers Share of Updates | Top-5 Layers Share of Updates |
|------|------|------|
| Uniform FT | 28.4% | 41.7% |
| **NAST (CTE-weighted)** | **52.6%** | **69.3%** |

CTE weighting indeed concentrates updates on the top negation-sensitive layers, corresponding to gains in claim accuracy.

### Key Findings
- **Layer-localized negation processing**: CTEs concentrate in layers 1-4, peaking at layer 2; this aligns with LLM literature where early layers handle syntactic function words and deeper layers handle semantics.
- NAST's improvement **mainly comes from increased negation accuracy** rather than decreased affirmative accuracy—affirmative performance even slightly improves (Table 2), indicating CTE guidance does not harm general alignment.
- The finding that "a few layers handle a few functions" suggests that generic all-layer LoRA fine-tuning is wasteful, and interpretability-guided sparse fine-tuning could be the next generation of parameter-efficient adaptation.

## Highlights & Insights
- "Compute scores via causal tracing → feed scores to optimizer as layer weights" is a paradigm shift from **diagnosis** to **prescription** in mechanistic interpretability—future medical/general VLMs can directly adopt this framework.
- MedNega-CXR fully leverages the unique convenience of "affirmative equivalence" in medical contexts: it is difficult to construct clean polarity contrasts in general domains, but medicine provides a unique experimental platform for interpretability research.
- Without touching the backbone and only weighting LoRA, the gap is reduced from 21.6 to 4.2; this shows that medical VLMs' negation handling is just one step away (a few key layers), not requiring full retraining.

## Limitations & Future Work
- CTE is computed on a manually curated "severe edema vs no edema" contrast set; its transferability to other clinical scenarios (rare diseases, ambiguous expressions) is unverified.
- Both causal tracing and LoRA are applied only on the text encoder side, not touching the vision encoder or cross-modal projection; if the vision side also has polarity-sensitive bias, this method does not address it.
- Evaluation is limited to MIMIC-CXR style reports and CheXpert ontology; for other modalities such as CT, MRI, pathology images, and non-English clinical texts, CTE needs to be recomputed and validated.

## Related Work & Insights
- **vs NegCLIP / ConCLIP / NegBench**: These rely on "adding negation samples + contrastive loss," while this work adds "layer-targeted optimization" on top.
- **vs Causal Tracing for LLM (Meng et al.)**: Transfers ROME-style causal tracing from LLM knowledge localization to negation handling in the CLIP text encoder, and for the first time uses tracing results as optimizer input.
- **vs Layer-wise Adaptive LR (LARS, LAMB)**: Those methods automatically adjust each layer's LR by gradient norm; this work adjusts by causal contribution, making it "semantics-aware."

## Rating
- Novelty: ⭐⭐⭐⭐ First to convert causal tracing into layer-wise training rules, with a clear methodological path.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines + multi-task + update distribution ablation, comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐ Concise flow from problem diagnosis to data, method, and evaluation.
- Value: ⭐⭐⭐⭐ Negation understanding in medical safety scenarios is a real pain point, and CTE weighting is widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/multimodal_vlm/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](../../AAAI2026/multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](../../CVPR2025/multimodal_vlm/vision-language_models_do_not_understand_negation.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs](../../ACL2025/multimodal_vlm/trimllm_layer_dropping.md)

</div>

<!-- RELATED:END -->
