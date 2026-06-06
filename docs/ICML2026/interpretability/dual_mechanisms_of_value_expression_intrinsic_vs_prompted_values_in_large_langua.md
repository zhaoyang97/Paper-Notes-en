---
title: >-
  [Paper Note] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models
description: >-
  [ICML 2026][Interpretability][Value Vectors] This paper uses difference-in-means to extract "intrinsic" (without system prompts) and "prompted" (with value system prompts) directions for 10 Schwartz values in the residua…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Value Vectors"
  - "Value Neurons"
  - "Schwartz Basic Values"
  - "Residual Stream Directions"
  - "Instruction Following"
date: 2026-05-08
content_hash: 99cad268cf62cd04
---

# Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2509.24319](https://arxiv.org/abs/2509.24319)  
**Code**: https://github.com/holi-lab/ValueMechanism (Yes)  
**Area**: Interpretability / Mechanistic Interpretability / Value Alignment  
**Keywords**: Value Vectors, Value Neurons, Schwartz Basic Values, Residual Stream Directions, Instruction Following

## TL;DR
This paper uses difference-in-means to extract "intrinsic" (without system prompts) and "prompted" (with value system prompts) directions for 10 Schwartz values in the residual stream. By decomposing these directions into shared and unique axes via SVD, the authors provide causal evidence at both the vector and MLP neuron levels: the shared component carries true value semantics, generalizes cross-lingually, and reproduces the Schwartz circumplex structure; the intrinsic-unique component contributes to lexical and semantic diversity; the prompted-unique component encodes a value-agnostic "general instruction-following" channel, which can directly push jailbreak attack success rates from 13%–27% to 83%–97%.

## Background & Motivation
**Background**: Current mainstream pluralistic value alignment follows two paths: first, preference learning (RLHF / DPO) that bakes fixed value biases into weights, corresponding to *intrinsic value expression*; second, providing system prompts at inference time ("Please prioritize cultural traditions"), corresponding to *prompted value expression*. Both paths are widely used, but researchers typically choose between them based on intuition.

**Limitations of Prior Work**: Existing activation engineering literature either extracts directions only in prompted settings (Su et al. 2025) or only in intrinsic settings (Jin et al. 2025). There has been no systematic comparison of the relationship between these two types of directions—whether they are different entry points to the same mechanism or two independent circuits within the model. This directly impacts the interpretability and safety of alignment methods.

**Key Challenge**: Intuitively, prompts should simply trigger existing intrinsic values and thus share the same circuit. However, empirically, prompted responses often appear "over-engineered or unnatural" (Shao et al. 2023; Malik et al. 2024), suggesting they carry additional non-value components. Without clear differentiation, any intervention based on value directions will entangle "value" with "compliance."

**Goal**: To answer simultaneously at the granularity of residual stream directions and MLP neurons: (1) how much these two mechanisms overlap; (2) whether the overlap represents true value semantics; and (3) what specific functions the unique portions perform.

**Key Insight**: Building on the Linear Representation Hypothesis, each of the 10 universal values proposed by Schwartz is treated as a linear subspace in the residual stream. Directions are extracted under both intrinsic and prompted conditions using the same prompt set. SVD is then used to decompose paired directions into "shared + unique" components, which are attributed to specific neurons by projecting them onto MLP output columns.

**Core Idea**: Treat paired (intrinsic, prompted) value directions as a 2D subspace. Use SVD to explicitly separate the "shared semantic axis" from the "difference axis." Validate these via vector-level interventions (orthogonalized directions) and neuron-level interventions (neuron sets categorized by angle), allowing every conclusion to be derived from the same geometric object.

## Method

### Overall Architecture
The method consists of three steps, centered around a tuple (value $s$, layer $\ell$, expression type $e\in\{int, prompt\}$). Step 1, **Direction Extraction**: Starting from 26,334 real user queries (ShareGPT/LMSYS), the model generates responses under two conditions: empty system prompts (intrinsic) and value-priority system prompts (prompted, randomly sampled from 500 GPT-4o-mini augmented templates). GPT-4o-mini categorizes responses into "expressing the value $R_{\text{exp}}$" or "not expressing it $R_{\text{unexp}}$." After token-averaging the residual stream activations, the mean difference yields the value vector $v^\ell_{s,e}$. Step 2, **Subspace Decomposition**: Paired directions $\{v^\ell_{s,\text{int}}, v^\ell_{s,\text{prompt}}\}$ are orthogonalized and decomposed via SVD to obtain a shared axis $u_{\text{shared}}$ and a difference axis $u_{\text{diff}}$, yielding $u_{\text{int}}$ and $u_{\text{prompt}}$. Step 3, **Neuron Attribution**: Based on pre-LayerNorm Transformer MLP updates $\Delta x^\ell=\sum_i \sigma(\langle x^\ell, w^\ell_{\text{in},i}\rangle)\,w^\ell_{\text{out},i}$, output columns $w^\ell_{\text{out},i}$ are projected onto the 2D subspace and categorized as shared, intrinsic-unique, or prompted-unique based on a $<30°$ angle threshold.

The backbone model is Qwen2.5-7B-Instruct, with robustness tests extending to Qwen2.5-1.5B/32B, Llama-3.1-8B-Instruct, Gemma2-9b-it, and Qwen3-8B/14B.

### Key Designs

1. **Difference-in-means Value Vector + Orthogonalization Causal Test**:
    - **Function**: Compresses "the model expressing a value" into a single direction in the residual stream and distinguishes indispensable components.
    - **Mechanism**: Token-average each response $\bar a^\ell(r)=\frac{1}{|r|}\sum_t a^\ell_t(r)$ and calculate the difference $v^\ell=\frac{1}{|R_{\text{exp}}|}\sum_{r\in R_{\text{exp}}}\bar a^\ell(r)-\frac{1}{|R_{\text{unexp}}|}\sum_{r\in R_{\text{unexp}}}\bar a^\ell(r)$. Orthogonalize paired directions $v^\ell_{s,\text{int}(\perp \text{prompt})}=v^\ell_{s,\text{int}}-\frac{\langle v^\ell_{s,\text{int}}, v^\ell_{s,\text{prompt}}\rangle}{\langle v^\ell_{s,\text{prompt}}, v^\ell_{s,\text{prompt}}\rangle}v^\ell_{s,\text{prompt}}$. Intervene via $(a^\ell_t)^*=a^\ell_t+\alpha v^\ell_{s,e}$, capping $\alpha$ such that MMLU drop is < 5 points (e.g., $\alpha=4$ for Qwen2.5-7B).
    - **Design Motivation**: Difference-in-means is theoretically optimal for concept editing in worst-case scenarios (Belrose 2023). Averaging diverse prompts cancels prompt-specific noise. Orthogonalization turns the question of collinearity into a falsifiable causal experiment.

2. **SVD Shared/Unique Axis Decomposition**:
    - **Function**: Simultaneously characterizes shared and opposing parts of the two mechanisms in the same 2D subspace, providing a unified coordinate system for neuron classification and cross-value analysis.
    - **Mechanism**: Construct matrix $V^\ell_s=[v^\ell_{s,\text{int}}, v^\ell_{s,\text{prompt}}]$ and apply $V^\ell_s=U\Sigma R^\top$. The first left singular vector $u_{\text{shared}}=U[:,1]$ serves as the shared axis (capturing max variance). The second vector $u_{\text{diff}}=U[:,2]$ is the difference axis, oriented as $u_{\text{int}}$ based on the sign of $\langle u_{\text{diff}}, v^\ell_{s,\text{int}}-v^\ell_{s,\text{prompt}}\rangle$, with $u_{\text{prompt}}=-u_{\text{int}}$.
    - **Design Motivation**: Orthogonalization only reveals what remains after removing a projection; SVD explicitly extracts the "common direction." These two axes correspond to the hypotheses that shared components carry semantics while unique components carry functional specialization.

3. **Geometric Categorization of MLP Value Neurons**:
    - **Function**: Attributes residual stream directions to specific, interpretable MLP units, allowing localization and ablation of parameters supporting shared vs. unique mechanisms.
    - **Mechanism**: Project neuron output columns onto the subspace $p_i=\text{Proj}_{S^\ell_s}(w^\ell_{\text{out},i})$ and rank them by $\|p_i\|_2$ to keep the top $k\%$. Categorize them by the smallest angle $\theta(p_i,u)=\arccos(\langle p_i,u\rangle/(\|p_i\|_2\|u\|_2))$ among $A=\{u_{\text{shared}}, u_{\text{int}}, u_{\text{prompt}}\}$, provided it is $<30°$. Neuron-level intervention scales selected activations by $\beta>1$.
    - **Design Motivation**: The residual stream is a superposition of many components. The rank-1 decomposition of MLP updates allows clean attribution of directions to neurons, which can then be named using automated neuron explanation pipelines (Bills et al. 2023).

## Key Experimental Results

### Main Results
Evaluations cover PVQ-40 / PVQ-RR (6-point scale), free-form PVQ-40 (GPT-4o 0–10 score), situational dilemmas (GPT-4o-mini win rate), and Value Portrait (284 real user Q&A), plus cross-lingual (en/zh/es/fr/ko) and jailbreak (HarmBench, AdvBench) validation.

| Dataset | Metric | Intrinsic | Prompted | Intrinsic⊥ | Prompted⊥ |
|---------|--------|-----------|----------|------------|------------|
| PVQ 6-pt (Avg 5-lang) | Score Gain | +1.74 | +2.21 | +0.47 | +1.62 |
| Free-form PVQ 10-pt | Score Gain | +0.98 | +1.04 | +0.48 | +0.52 |
| AdvBench (Llama-3.1-8B) | ASR@9 | — | 13.3% | — | **97.2%** (Steer via mean delta) |
| HarmBench (Llama-3.1-8B) | ASR@9 | — | 23.8% | — | **90.4%** |
| AdvBench (Qwen2.5-7B) | ASR@9 | — | 27.0% | — | **89.0%** |
| HarmBench (Qwen2.5-7B) | ASR@9 | — | 52.4% | — | **83.0%** |

Value vectors extracted in English generalize to other languages with minimal decay. Procrustes alignment of PCA(shared axes) with the Schwartz circumplex yields $R^2\approx 0.6$–$0.7$ across 4 higher-order domains, significantly higher than random baselines.

### Ablation Study

| Configuration | Key Metrics | Note |
|---------------|-------------|------|
| Intrinsic Full Direction | Distinct-2 0.362 / Highest diversity | Vocabulary and semantic diversity baseline |
| Prompted Full Direction | Distinct-2 0.342 | Narrower vocabulary, focused on "achievement/goals" |
| Intrinsic ⊥ Prompted | Distinct-2 **0.402** / EAD-2 **0.345** | Diversity increases after removing shared component; focuses on "broad context" |
| Prompted ⊥ Intrinsic | Distinct-2 0.203 / 32–73% steering norm remains | Diversity plummets but steering remains strong, indicating non-value signals |
| Shared Neuron Scaling ($\beta>1$) | PVQ score gain > Unique neurons | Shared neurons are the primary causal drivers of value expression |
| Mean Delta Direction (Avg prompted-intrinsic diff) | Explains 48–68% delta variance | Reveals a "general instruction-following" channel independent of specific values |

### Key Findings
- **Shared Component = True Value Semantics**: Scaling shared neurons alone improves PVQ scores. PCA of the 10 shared axes reconstructs the Schwartz circumplex (e.g., Benevolence is near Universalism and opposite Achievement). Automated explanations for shared neurons describe abstract concepts like "collective wellbeing" rather than surface-level lexical cues.
- **Intrinsic Unique = Diversity**: Intrinsic vectors generate higher-entropy token distributions. Intrinsic-unique neurons activate in contexts like "personal projects" that co-occur with values without stating them directly, explaining why intrinsic expression feels more natural.
- **Prompted Unique = Instruction Following, Not Values**: Prompted-unique neurons are triggered by system prompt keywords (e.g., "warning," "threat"). Crucially, the 10 delta directions are highly collinear. Steering along their mean direction spikes jailbreak ASR from 13.3% to 97.2% on Llama. This direction also improves instruction following in non-value tasks like gender-neutral translation but fails on tasks beyond model capability (like strict JSON formatting), suggesting it amplifies existing capabilities.

## Highlights & Insights
- **Geometric Controlled Experiments**: Packing intrinsic/prompted directions into a 2D subspace and using SVD/orthogonalization turns conceptual questions into falsifiable causal experiments. This framework is extensible to persona, emotion, or refusal scenarios.
- **Cross-Value Aggregation Reveals Hidden Channels**: While individual delta directions look like value-specific corrections, their aggregate reveals a "compliance" axis. This warns researchers that difference directions for a single concept likely capture system-prompt effects rather than core semantics.
- **Dual Perspectives on Jailbreaking**: Prior work attributes jailbreaking to the suppression of "refusal directions." This paper provides a dual perspective: jailbreaking can also result from the amplification of a "universal prompt-compliance channel" learned during alignment.

## Limitations & Future Work
- Evaluation relies heavily on Schwartz's 10 values, which may not capture the more continuous and overlapping value spectrum of reality.
- Value expression labeling depends on GPT-4o-mini; boundaries are inherently fuzzy, which may affect the precision of direction estimation.
- Neuron attribution assumes a rank-1 decomposition of pre-LayerNorm Transformers; validity for MoE structures is untested.
- While shared components reproduce the Schwartz circumplex, $R^2$ values of $0.6$–$0.7$ leave room for unexplained structures, such as higher-order antagonistic relationships between values.

## Related Work & Insights
- **vs. Persona Vectors (Chen et al. 2025)**: Persona vectors are typically extracted in prompted settings; this work shows that true "altruism" signals reside in the shared axis, validating that both methods capture similar semantics.
- **vs. Refusal-mediated Jailbreak (Arditi et al. 2024)**: While Arditi describes jailbreaking via refusal modulation, this work shows that compliance-side amplification achieves similar ASR, suggesting two distinct mechanisms for jailbreaking.
- **vs. SAE Feature Extraction**: SAEs provide sparse dictionaries but do not distinguish mechanism sources. This work's use of difference-in-means + SVD provides a lightweight alternative for characterizing mechanistic differences.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to treat intrinsic vs. prompted as decomposable geometric objects; attributing "prompted jailbreak" to a value-agnostic compliance channel is a genuine discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 5 languages × 4 evaluation types + jailbreak + diversity + automated neuron explanation, linking neuron, vector, and behavioral levels.
- Writing Quality: ⭐⭐⭐⭐ Geometric illustrations and the three-layer evidence chain are clear, though some key conclusions are buried in later sections.
- Value: ⭐⭐⭐⭐⭐ Provides reusable geometric tools for pluralistic alignment and safety research; the dual perspective on jailbreaking is significant for red-teaming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](../../ACL2026/interpretability/dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](towards_atoms_of_large_language_models.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](../../ACL2026/interpretability/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](../../ACL2026/interpretability/fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)

</div>

<!-- RELATED:END -->
