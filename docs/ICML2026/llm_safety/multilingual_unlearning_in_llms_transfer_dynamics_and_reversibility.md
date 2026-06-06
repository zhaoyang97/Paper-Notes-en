---
title: >-
  [Paper Note] Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility
description: >-
  [ICML 2026][LLM Safety][LLM Unlearning] This paper extends the TOFU unlearning benchmark to five languages to systematically study "cross-lingual unlearning transfer." It finds that unlearning strength varies with the ki…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "LLM Unlearning"
  - "Cross-lingual Transfer"
  - "Representation Space"
  - "Steering Vectors"
  - "Reversible Unlearning"
date: 2026-05-08
content_hash: 3f0a3ad5abce806c
---

# Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility

**Conference**: ICML 2026  
**arXiv**: [2606.03291](https://arxiv.org/abs/2606.03291)  
**Code**: https://github.com/MLCY1/multilingual-unlearning-in-llms  
**Area**: LLM Safety / Privacy / Multilingual LLMs  
**Keywords**: LLM Unlearning, Cross-lingual Transfer, Representation Space, Steering Vectors, Reversible Unlearning

## TL;DR
This paper extends the TOFU unlearning benchmark to five languages to systematically study "cross-lingual unlearning transfer." It finds that unlearning strength varies with the kinship of language families and writing systems. Furthermore, unlearning primarily affects language-specific decoding layers at the end of the model while leaving the shared semantic space in the early-to-mid sections nearly untouched. Consequently, an inference-time steering vector can recover 50% of forgotten knowledge in Qwen and 90% in Gemma, suggesting that existing LLM unlearning is essentially "surface inhibition" rather than true erasure.

## Background & Motivation

**Background**: The vast amounts of data absorbed during LLM training may contain sensitive facts. Coupled with GDPR "Right to be Forgotten" requirements, this has spurred research into "LLM unlearning"—erasing specific knowledge without retraining. Prevailing methods (GA, NPO, DPO-style) apply modification objectives to fine-tuned models to discourage them from revealing target content in the "forget" set.

**Limitations of Prior Work**: (1) Existing evaluations are almost exclusively conducted in English, leaving the extent of "unlearning transfer" in multilingual scenarios uncharacterized—despite sensitive facts often appearing repeatedly across multiple languages in real-world deployments. (2) Even in monolingual settings, some works suggest unlearning acts as an "inhibition signal," but they lack mechanism localization (which layers?) and evidence of reversibility without relearning.

**Key Challenge**: If multilingual unlearning only modifies "language-specific decoding layers," knowledge remains intact in the shared semantic space, allowing attackers to retrieve it via cross-lingual queries or reverse steering during inference. If it truly alters the "cross-lingual conceptual space," the safety guarantee is much stronger. These two scenarios entail vastly different deployment risks, yet prior work has failed to distinguish between them.

**Goal**: (i) Systematically characterize cross-lingual unlearning transfer patterns across language families, writing systems, and pre-training coverage; (ii) Locate unlearning actions via mechanistic interpretability; (iii) Verify reversibility using a simple inference-time steering vector and measure its cross-lingual transferability.

**Key Insight**: The authors translate the TOFU dataset (20 QA pairs for each of 200 fictitious authors) into five languages (EN/CH/DE/RU/TU), controlling for three axes: shared language family vs. shared writing system vs. neither. By fine-tuning, unlearning, and querying across different languages, they generate a $5 \times 5 \times 5$ transfer matrix and use NLI rather than lexical overlap to evaluate semantic equivalence.

**Core Idea**: The difference in latent representations between the fine-tuned and unlearned models for the same prompt is treated as the "unlearning direction" (steering vector), which is then injected as a weighted reverse perturbation during the forward pass. If this direction is a "language-agnostic inhibition direction," it should recover knowledge in any language—a hypothesis this paper validates.

## Method

### Overall Architecture

The experimental workflow consists of four steps: (1) LoRA fine-tuning on a specific $\mathcal{L}_{FT}$ using multilingual TOFU data to obtain $f_{\text{ft}}$; (2) Applying a DPO-style unlearning objective $J_{UN}$ on $\mathcal{L}_{\text{unl}}$ to erase 1% of the "forget" authors, obtaining $f_{\text{un}}$; (3) Evaluating forget/retain accuracy on $\mathcal{L}_Q$ via NLI to generate the transfer matrix; (4) Extracting latent representations for cosine similarity analysis and constructing steering vectors for reversibility experiments.

Experiments are conducted on Qwen2.5-7B and Gemma2-9B. The unlearning objective is $\arg\min_\theta \frac{1}{|\mathcal{L}_{\text{unl}}|} \sum_{\ell} (\mathbb{E}_{D_\ell^{\text{forget}}} J_{\text{forget}} + \lambda \mathbb{E}_{D_\ell^{\text{retain}}} J_{\text{retain}})$, where $J_{\text{forget}}$ uses DPO to prefer "I don't know" over the "ground truth."

### Key Designs

1. **Three-Axis Controlled Multilingual Unlearning Transfer Matrix**:
    - **Function**: Decomposes the open question of how linguistic relations affect unlearning transfer into three observable dimensions.
    - **Mechanism**: Selects five languages covering four combinations: "Same Family + Same Script" (EN/DE), "Same Family + Different Script" (EN/RU), "Different Family + Same Script" (EN/TU), and "Neither Shared" (EN/CH). For each $(\mathcal{L}_{FT}, \mathcal{L}_{\text{unl}}, \mathcal{L}_Q)$ triplet, the NLI score change relative to unlearning is reported.
    - **Design Motivation**: Previous work only measured English, failing to distinguish between script effects, family effects, and pre-training coverage effects. This $5 \times 5 \times 5$ matrix allows all three effects to be observed independently.

2. **Cross-lingual Prompting as an "Output-side" Diagnostic**:
    - **Function**: Determines whether a model "does not know the answer" or "knows it but cannot decode it."
    - **Mechanism**: Queries are in language $q$, but the model is required to answer in the fine-tuning language $\ell$. Performance gains $\Delta_{\ell \leftarrow q}$ are recorded. If $\Delta > 0$ is large, knowledge is intact in the shared space but bound during decoding. The correlation between $\Delta_{\ell \leftarrow q}$ and the transfer matrix (Pearson $r=0.50$, Spearman $\rho=0.60$) directly links the "shared semantic space" to "transfer intensity."
    - **Design Motivation**: Since decoding binding is only indirectly observable, "swapping query/response languages" opens the bottleneck to prove that unlearning damage propagates through the shared space to downstream decoding layers.

3. **Unlearning Direction = Representation Difference; Inference-time Steering to Verify Reversibility**:
    - **Function**: Upgrades the "unlearning as inhibition" hypothesis to mechanistic evidence and quantitatively restores forgotten knowledge.
    - **Mechanism**: For a "forget" question, the latent state difference $d^{(l)} = h_{\text{ft}}^{(l)} - h_{\text{un}}^{(l)}$ is taken at the final token of layer $l$. A weighted perturbation is then injected along this direction during $f_{\text{un}}$'s forward pass. If knowledge is recovered across languages using a single direction, unlearning is "language-agnostic surface inhibition." Layer-wise cosine similarity shows $f_{\text{un}}$ and $f_{\text{ft}}$ overlap in early-to-mid layers, with divergence concentrated in the final decoding layers.
    - **Design Motivation**: Unlike previous evidence of "reversibility" that relied on brief relearning (requiring data) or prefix induction (requiring the answer), this single-direction inference-time steering requires neither data nor the answer and transfers across languages. This constitutes the strongest counter-evidence against "true erasure" in LLM unlearning.

### Evaluation Metrics

A multilingual NLI model (xlm-roberta-large-xnli) is used to determine if the generated answer $\hat y$ and the ground truth $y$ entail each other, avoiding the inaccuracies of lexical overlap across languages. Native speakers validated NLI reliability on 50 samples.

## Key Experimental Results

### Main Results: Cross-lingual Unlearning Transfer (Qwen2.5-7B)

| FT \ Unlearn | EN Query | CH Query | DE Query | RU Query | TU Query |
|--------------|---------|---------|---------|---------|---------|
| EN / EN | **-90** | -4 | -7 | -9 | -4 |
| EN / CH | -7 | **-8** | +1 | -5 | -3 |
| EN / DE | **-17** | -6 | **-4** | -5 | -4 |
| DE / EN | -13 | -4 | **-41** | -7 | 0 |
| TU / EN | -10 | -2 | -1 | -6 | **-55** |
| CH / TU | -1 | **+6** | -4 | -4 | 0 |

Numbers represent the absolute decrease in NLI scores relative to the fine-tuned base (more negative = stronger unlearning). Observations: (1) Transfer is strongest within the same family and script (EN→DE, EN→EN); (2) Unlearning high-coverage languages (EN/CH) results in stronger transfer than low-coverage ones (DE/RU/TU); (3) Unlearning in weak languages can still backward-impact strong languages (TU/EN cell -55).

### Cross-lingual Prompting Gain $\Delta_{\ell \leftarrow q}$

| FT \ Query | EN | CH | DE | RU | TU |
|------------|----|----|----|----|----|
| EN | — | +29 | +61 | +30 | +27 |
| CH | +11 | — | +10 | +12 | +12 |
| DE | +33 | +22 | — | +5 | +18 |
| RU | +20 | +8 | +15 | — | +7 |
| TU | +33 | +11 | +22 | +17 | — |

Significant positive gains prove that knowledge remains intact in the shared semantic space, with the failure being language-specific decoding. Correlations with the transfer matrix are significant (Pearson $r=0.50$, Spearman $\rho=0.60$, $p<0.05$).

### Reversibility: Knowledge Recovery via a Single Steering Direction

| Model | Recovery Rate (Forget NLI Rebound) | Cross-lingual Transfer? | Forget Data Needed? |
|------|---|---|---|
| Qwen2.5-7B | $\approx 50\%$ | Yes | **No** |
| Gemma2-9B | $\approx 90\%$ | Yes | No |

### Key Findings

- **Dual Impact of Language Family and Script**: Controlling for script, EN→RU (same family, different script) is stronger than EN→CH (neither shared); controlling for family, EN→TU (same script, different family) is stronger than EN→CH. Both axes contribute independently.
- **Asymmetric Transfer**: High-coverage languages (EN, CH) are more powerful unlearning sources, while low-coverage ones (DE/RU/TU) are weaker—consistent with the hypothesis that models reason in a shared space anchored by dominant languages.
- **"I Don't Know" Still Transfers**: For a TU fine-tuned model, an EN query base yields only 11% NLI, but after unlearning in EN, the TU query NLI drops by 55%—validating the shared concept space hypothesis.
- **Layer Localization**: Cosine similarity between $f_{\text{un}}$ and $f_{\text{ft}}$ is nearly identical in early-to-mid layers and only diverges significantly in the final few layers—damage from unlearning is concentrated in the "concept-to-language-specific-output" step.
- **Reversibility**: A 90% recovery rate on Gemma implies unlearning is almost "cosmetic"; a 50% rate on Qwen still poses a substantial safety risk.

## Highlights & Insights

- **First Systematic Multilingual Unlearning Transfer Map**: Decouples language family, script, and coverage into a $5 \times 5 \times 5$ matrix, providing a clear benchmark for future work.
- **Mechanism Localization + Behavioral Evidence Loop**: Leverages layer-wise latent analysis to identify late decoding layers as the inhibition site, verifies this via cross-lingual prompting, and solidifies the hypothesis through steering.
- **Reversibility Experiments Expose the "Unlearning" Illusion**: Recovery via a single inference-time direction without relearning or answer prefixes—transferable across languages—is the most direct counter-example to current unlearning safety claims.
- **NLI Evaluation**: Circumvents the distortion of lexical overlap in multilingual settings, offering methodological value for future multilingual generation assessment.
- **Practical Implications for Adversarial Defense**: Demonstrates that for "Right to be Forgotten" in multilingual models, unlearning in English alone is insufficient; all possible query languages must be covered, and even then, steering attacks remain highly effective.

## Limitations & Future Work

- **Narrow Task Scope**: Only tested on TOFU-style synthetic biographical knowledge; other forms of knowledge (sensitive facts, PII, copyrighted text) may not be isomorphic or may have different layer distributions.
- **Limited Method Coverage**: Only covers three types of fine-tuning-based methods (DPO/GA/NPO); representation misdirection (e.g., RMU) or parameter localization methods (e.g., ROME-style) are not yet validated.
- **Sampling of 5 Languages**: Lacks low-resource languages (e.g., African or Southeast Asian languages), potentially missing cases where unlearning transfer might be zero.
- **Steering Gap (Qwen 50% vs. Gemma 90%)**: The disparity is not fully explained; it may relate to multilingual training ratios, architecture, or alignment processes.
- **Real-world Threat Model**: Steering assumes an attacker has both $f_{\text{ft}}$ and $f_{\text{un}}$ checkpoints, which might not hold in API-based scenarios. Future work should test if the direction can be inferred via black-box queries.

## Related Work & Insights

- **vs. Monolingual LLM Unlearning (GA/NPO/DPO series)**: The first to apply these methods to multilingual scenarios and identify unbalanced transfer and alignment-based vulnerability.
- **vs. Suppression Hypotheses (e.g., Hu et al. 2025)**: Advances from "monolingual empirical observation" to "multilingual mechanistic + reversibility evidence," with layer-wise localization.
- **vs. Shared Semantic Space Theories (e.g., Wendler et al. 2024)**: Repurposes "positive" representation theories as "negative" safety analysis tools for cross-lingual robustness.
- **vs. Concept Retrieval Steering (e.g., Seyitoğlu et al. 2024)**: While they use existing world knowledge, this work constructs directions from fine-tuned/unlearned differences, eliminating the need for generalized priors and making the method more universal.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically decouple the three factors of multilingual unlearning transfer and provide strong reversibility evidence via single-direction inference steering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual models × 5 languages × 3 unlearning objectives × 3 validation perspectives (NLI/layer analysis/steering), with all key conclusions ablated.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical notation, though color-coding in the $5 \times 5 \times 5$ matrix can be difficult to follow on paper.
- **Value**: ⭐⭐⭐⭐⭐ Directly challenges safety claims of current unlearning methods; mandatory reading for compliance and defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning](unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[AAAI 2026\] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs](../../AAAI2026/llm_safety/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](../../ACL2026/llm_safety/cap_controllable_alignment_prompting_for_unlearning_in_llms.md)

</div>

<!-- RELATED:END -->
