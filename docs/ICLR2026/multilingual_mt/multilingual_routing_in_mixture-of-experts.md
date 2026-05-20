---
title: >-
  [Paper Note] Multilingual Routing in Mixture-of-Experts
description: >-
  [ICLR 2026][Multilingual & Machine Translation][mixture-of-experts] This paper systematically analyzes multilingual routing patterns in MoE large language models…
tags:
  - "ICLR 2026"
  - "Multilingual & Machine Translation"
  - "mixture-of-experts"
  - "multilingual routing"
  - "cross-lingual transfer"
  - "expert steering"
  - "interpretability"
date: 2026-05-08
content_hash: 25500e8fe04322dc
---

# Multilingual Routing in Mixture-of-Experts

**Conference**: ICLR 2026
**arXiv**: [2510.04694](https://arxiv.org/abs/2510.04694)  
**Authors**: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng (UCLA, Fudan University)
**Code**: Not released  
**Area**: Multilingual Translation
**Keywords**: mixture-of-experts, multilingual routing, cross-lingual transfer, expert steering, interpretability

## TL;DR

This paper systematically analyzes multilingual routing patterns in MoE large language models, finding that middle layers contain cross-lingually shared experts and that language performance is strongly correlated with alignment to English routing. Based on these findings, the authors propose an inference-time routing intervention that activates English task experts in middle layers, consistently improving multilingual performance by 1–2% across 3 models × 2 tasks × 15+ languages.

## Background & Motivation

1. **MoE is mainstream but its multilingual mechanisms are poorly understood**: The Mixture-of-Experts architecture is a core paradigm for scaling LLMs, enabling massive parameter counts while maintaining manageable inference costs; however, how its sparse routing dynamics respond to multilingual data has received almost no systematic investigation.
2. **Findings from dense LLMs have not been transferred to MoE**: Extensive work has revealed language-universal representation spaces in the middle layers of dense LLMs, with early and late layers handling language-specific mappings, but whether MoE's sparse activation mechanism exhibits analogous layer-wise patterns remains unexplored.
3. **English-centricity of pretraining**: Existing MoE models are heavily English-centric in both pretraining and post-training data; although implicit multilingual capability emerges with scale, significant performance gaps persist across most languages.
4. **MoE is naturally suited for interpretability analysis**: The discrete expert activation mechanism in MoE makes it more intuitive to analyze which model components are responsible for which capabilities, yet this advantage has not been fully exploited for multilingual analysis.
5. **Bottlenecks in cross-lingual transfer remain to be identified**: Understanding multilingual routing mechanisms in MoE can provide actionable insights for improving cross-lingual capability transfer.

## Method

### Routing Divergence Analysis

- The FLoRes-200 parallel translation dataset is used, which contains parallel text across 200+ languages covering diverse topics.
- For each non-English sequence, the routing weights of all tokens within the sequence are averaged to obtain an expert importance distribution $\bm{q}_i^{(\text{lang},l)}$.
- **Entropy-normalized Jensen-Shannon divergence** ($D_{\text{H-JS}}$) is used to quantify, at each layer, the routing discrepancy between non-English sequences and their English parallel counterparts.
- Entropy normalization is necessary because routing entropy varies substantially across layers (decreasing in deeper layers), and direct comparison of JS divergence would introduce bias.
- The final per-language, per-layer routing divergence metric is $\text{Div}^{(\text{lang},l)}$.

### Model Coverage

Experiments cover four representative open-source MoE LLMs:
- **Qwen3-30B-A3B**: 48 layers, strong multilingual capability.
- **Phi-3.5-MoE**: 32 layers, from Microsoft.
- **GPT-OSS-20B**: 24 layers, open-sourced by OpenAI.
- **OLMoE**: An older, smaller English-centric model with weak multilingual capability (used as a control).

The four models differ in architectural width, sparsity, and depth, providing a diverse validation setting.

### Expert Identification

- For each expert, the **activation frequency difference** $\Delta_k$ relative to a general baseline (FLoRes English) is computed on domain- or language-specific data.
- Discrete activation counts rather than routing weights are used, as they more precisely identify the most responsible experts.
- A positive threshold $\tau$ is set; an expert is designated as specialized for a given domain or language when $\Delta_k > \tau$.
- Multilingual experts are defined as those satisfying $\Delta_k > \tau$ for **any** non-English language.
- Task experts are identified using GSM8K-Instruct (mathematics) and AlpaCare MedInstruct (medicine).

### Routing Interventions

**Soft Intervention**:
$$z'_k \leftarrow z_k + \lambda \cdot s(\bm{z})$$
The logit of a target expert is increased or decreased by $\lambda$ times the standard deviation prior to softmax; $|\lambda| \leq 1.0$ yields the best results.

**Hard Intervention**:
$$z'_k \leftarrow \max(\bm{z}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 10^{-6})$$
The target expert's logit is forced to the maximum (activation) or minimum (suppression) value among all experts.

Interventions are applied exclusively to **middle layers**, with layer ranges determined by the U-shaped routing divergence curve of each model.

## Key Findings

### 1. U-Shaped Routing Divergence — Cross-Lingual Sharing in Middle Layers

Across all models, early and late layers exhibit **language-specific** routing, while middle-layer routing is **highly aligned** across languages, forming a clear U-shaped curve. This indicates that MoE models, like dense models, learn a language-universal representation space in middle layers, and do so in a more modular and clearly delineated manner.

### 2. Strong Correlation Between Language Performance and Routing Alignment

A **strong negative correlation** exists between language comprehension ability (Belebele accuracy) and the degree of middle-layer routing alignment to English:
- OLMoE: $r \in [-0.95, -0.80]$ (extremely strong)
- Qwen3 and Phi-3.5-MoE: moderate to strong
- GPT-OSS: $r \in [-0.40, -0.60]$ (weakest, yet still significant)

Languages the model fails to understand (e.g., Bambara) cannot be mapped into the shared middle-layer space, maintaining high routing divergence throughout.

### 3. Complete Functional Separation Between Language and Task

At $\tau \geq 0.3$, **zero experts** are simultaneously specialized for both a task and a non-English language — the two expert sets are fully disjoint. This finding provides strong empirical support for the "functional dissociation between language and thought" hypothesis proposed by Mahowald et al.: processing linguistic form (language-specialized experts) and processing task content (task-specialized experts) are handled by distinct parameter components.

### 4. Language Differences in Routing Entropy and Consistency

- Routing entropy decreases with layer depth; the decrease is more pronounced for non-English languages, with a notable drop in the final layer — suggesting the existence of a small number of non-English generation experts.
- Inter-token routing consistency (Jaccard similarity) is negatively correlated with language resource level: low-resource language tokens exhibit higher routing consistency (relying on fewer experts).

## Key Experimental Results

### Main Results

| Model | Task | Target Layers | τ | Intervention | # Experts | Baseline | After | Gain |
|------|------|--------|---|----------|--------|------|--------|------|
| Qwen3-30B-A3B | MGSM | (8,35) | 0.4 | soft, λ=0.5 | 22 | 76.4% | 78.0% | +1.6% |
| Phi-3.5-MoE | MGSM | (8,17) | 0.3 | soft, λ=0.5 | 12 | 57.5% | 58.9% | +1.4% |
| GPT-OSS-20B | MGSM | (4,19) | 0.3 | hard | 9 | 68.9% | 71.5% | +2.6% |
| Qwen3-30B-A3B | MMLU Med. | (8,35) | 0.5 | hard | 23 | 68.2% | 69.1% | +0.9% |
| Phi-3.5-MoE | MMLU Med. | (8,17) | 0.25 | soft, λ=0.5 | 2 | 57.8% | 58.8% | +1.0% |
| GPT-OSS-20B | MMLU Med. | (4,19) | 0.3 | soft, λ=0.5 | 6 | 63.8% | 64.5% | +0.7% |

### Greater Improvements for Low-Resource Languages

- Swahili MGSM: GPT-OSS 52.4%→62.0% (**+9.6%**)
- Bengali MGSM: Phi-3.5 20.8%→23.2% (+2.4%)
- Yoruba MMLU Med.: Phi-3.5 40.0%→42.9% (+2.9%)
- Average gains for low-resource languages are consistently larger than for high-resource languages.

### English Performance Largely Unaffected

Interventions have almost no effect on English performance (variation <1%), with occasional slight decreases, indicating that the intervention precisely targets the cross-lingual transfer bottleneck without degrading existing capabilities.

### Ablation Study

- **Intervening outside middle layers** → substantial performance degradation (language-specific routing in early/late layers is disrupted).
- **Activating multilingual experts instead of task experts** → performance decreases (validating the language–task separation hypothesis).
- **Random expert intervention** → performance decreases.
- **Suppression (rather than activation)** → only harmful, no positive gain.
- **Layer range sensitivity** → even a few layers outside the optimal range causes degradation, validating the practical utility of routing divergence visualization.

## Highlights & Insights

- **First systematic characterization of multilingual routing dynamics in MoE LLMs**, revealing a middle-layer language-universal space consistent with but more clearly modular than that in dense models.
- The **complete language–task separation** finding ($\tau \geq 0.3$ yields zero overlapping experts) constitutes one of the strongest empirical validations to date of the "functional dissociation between language and thought" hypothesis.
- **Minimalist inference-time routing interventions** consistently improve multilingual performance across 3 models × 2 tasks × 15+ languages — the method is simple yet robust.
- Interventions modify the top-K selection of only 1–2 experts (K is typically 4 or 8), leaving the majority of routing behavior unchanged.
- Extensive and careful ablation experiments (layer selection, expert type, intervention strength, hard vs. soft) establish causal relationships.
- Routing divergence visualization itself serves as a practical tool for determining the layer range of intervention.

## Limitations & Future Work

1. **Modest gain magnitude**: Improvements of 1–2% are statistically significant and consistent across conditions, but small in absolute terms.
2. **Model-specific tuning required**: Optimal $\tau$, $\lambda$, and target layer range differ across models, necessitating per-model calibration.
3. **Expert identification depends on domain data**: Math experts are identified via GSM8K-Instruct and medical experts via MedInstruct; data selection influences results.
4. **Inference-time intervention only**: Training-time approaches that encourage cross-lingual expert sharing are not explored and may offer greater potential.
5. **Limited model coverage**: Only four MoE models are studied; larger-scale models (e.g., DeepSeek-V3) or architecturally different MoE variants may behave differently.
6. **Limited task coverage**: Only mathematical reasoning and medical QA are evaluated.

## Related Work & Insights

- **Multilingual middle layers in dense LLMs**: Kojima/Wendler/Bandarkar (2024–2025) identify language-universal spaces in the middle layers of dense models → this paper finds a clearer, more modular counterpart in MoE.
- **Cross-lingual representation alignment**: Kargaran/Ravisankar (2025) find that middle-layer alignment correlates with multilingual performance → this paper extends the relationship from representation space to routing space.
- **Inference-time intervention**: Mahmoud/Lu (2025) steer dense models toward language-shared representations → this paper achieves an analogous effect at the MoE routing level.
- **Fayyaz et al. (2026)**: Expert activation/suppression interventions → this paper finds that activating task experts is effective in multilingual contexts.
- **Language–thought dissociation**: Mahowald et al. (2024) "functional dissociation" hypothesis → zero cross-specialization between task and language experts in MoE provides one of the strongest empirical validations.
- **Multilingual MoE training**: Zheng et al. (2025) expand multilingual capability via last-layer MoE upcycling → consistent with this paper's finding of language specialization in late layers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](../../ACL2026/multilingual_mt/no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[AAAI 2026\] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/multilingual_mt/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)

</div>

<!-- RELATED:END -->
