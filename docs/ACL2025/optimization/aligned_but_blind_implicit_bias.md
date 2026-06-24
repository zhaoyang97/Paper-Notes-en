---
title: >-
  [Paper Note] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race
description: >-
  [ACL 2025][Optimization][implicit bias] Reveals the "race-blindness" side-effect of alignment training: Alignment prevents LLMs from representing "black/white" as racial concepts in ambiguous contexts, thus failing to activate safety guardrails and causing implicit bias to surge from 64.1% to 91.4%. Counter-intuitively, injecting race-aware activations (rather than unlearning) in early layers reduces implicit bias from 97.3% to 42.4%.
tags:
  - "ACL 2025"
  - "Optimization"
  - "implicit bias"
  - "race blindness"
  - "alignment side-effect"
  - "activation patching"
  - "bias mitigation"
  - "LoRA intervention"
date: 2026-05-08
content_hash: 3570eb03dc0c3eec
---

# Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race

**Conference**: ACL 2025  
**arXiv**: [2506.00253](https://arxiv.org/abs/2506.00253)  
**Code**: [https://github.com/slhleosun/aligned-but-blind](https://github.com/slhleosun/aligned-but-blind)  
**Area**: Other  
**Keywords**: implicit bias, race blindness, alignment side-effect, activation patching, bias mitigation, LoRA intervention

## TL;DR

Reveals the "race-blindness" side-effect of alignment training: Alignment prevents LLMs from representing "black/white" as racial concepts in ambiguous contexts, thus failing to activate safety guardrails and causing implicit bias to surge from 64.1% to 91.4%. Counter-intuitively, injecting race-aware activations (rather than unlearning) in early layers reduces implicit bias from 97.3% to 42.4%.

## Background & Motivation

**Alignment-driven bias mitigation promise**: One of the core goals of post-training alignment, such as RLHF/DPO, is to make models fair and unbiased. Aligned models indeed perform exceptionally well in explicit bias evaluations—refusing discriminatory requests and avoiding stereotypical language, with explicit bias rates dropping to around 8%.

**Persistence of implicit bias**: Multiple studies (Hofmann et al. 2024; Bai et al. 2025) have found that aligned LLMs still exhibit systematic racial stereotypes in implicit association tests (adapted from IAT), associating "black" with negative attributes in contexts where race is not directly mentioned.

**Lack of mechanistic explanation for the contradiction between explicit and implicit bias**: Alignment reduces explicit bias but may amplify implicit bias. This contradiction has previously lacked a mechanistic explanation from the perspective of the model's internal representations.

**Analogy to "race-blindness" in human psychology**: Psychological research suggests that the human strategy of attempting to entirely ignore race (color-blindness) actually perpetuates subtle bias (Apfelbaum et al. 2012), as a lack of awareness of racial differences prevents fair corrections.

**Maturation of mechanistic interpretability tools**: Methods such as activation patching and SelfIE enable the analysis of polysemous word encoding in the internal representation space of Transformers, laying the technical foundation for understanding bias mechanisms.

**Limitations of traditional debiasing paradigms**: Existing bias mitigation methods mostly adopt "unlearning/removal" strategies (machine unlearning). This paper explores a counter-intuitive direction—whether "enhancing racial awareness" can mitigate implicit bias more effectively.

## Method

### Overall Architecture

The study progresses through three stages: **(1) Behavioral Experiments**—designing 9,232 paired prompts to systematically quantify explicit and implicit bias levels before and after alignment; **(2) Mechanistic Analysis**—interpreting the "race-blindness" phenomenon in the model's internal representation space via activation patching and SelfIE; **(3) Intervention Experiments**—validating the causal hypothesis of "enhancing racial awareness $\rightarrow$ reducing implicit bias" through embedding injection and LoRA fine-tuning.

### Key Designs

**1. Prompt Design and Bias Measurement**

Designing explicit/implicit paired prompts with strictly controlled token length, word order, phrasing, and response format, varying only the level of implicitness. Implicit prompts require the model to make associations between black/white and stimulus words like wallet/revolver; explicit prompts directly query if the model agrees that "black is associated with revolver." Four variants (permutations of probe and stimulus words) are generated for each prompt, totaling 9,232 prompts.

The bias measure is defined as the average bias label:

$$\hat{p}_{\text{bias}}^{\text{race}} = \frac{1}{|\mathcal{I}_{\text{bias}}|} \sum_{i \in \mathcal{I}_{\text{bias}}} Y_i^{\text{race}}$$

where $Y_i^{\text{race}} \in \{0, 1\}$ denotes whether the model's response exhibits bias towards a specific race. An ideal unbiased model should show a random assignment of approximately 50% in implicit tests.

**2. Quantifying Race Blindness with Activation Patching (Race Blind Score)**

Core idea: Patch the model's internal activations for "black/white" from implicit association prompts into an interpretative prompt: "What does [MASK] refer to? Choose one: race or color.", and observe the change in the patched model's output probability for "race" vs "color".

Define the Race Blind Score:

$$r_{\text{blind}} = \Delta P_{\text{color}} - \Delta P_{\text{race}}$$

where $\Delta P_{\text{race}} = \frac{1}{L} \sum_{\ell} (P_{\text{patched}}^{\ell}(\text{race}) - P_{\text{baseline}}(\text{race}))$, and $\Delta P_{\text{color}}$ is formulated similarly.

$r_{\text{blind}} > 0$ indicates the model is more inclined to understand black/white as colors (race-blindness), while $r_{\text{blind}} < 0$ indicates the model maintains racial awareness.

**3. Natural Language Visualization using SelfIE**

Using SelfIE (Self-Interpretation of Embeddings) to let the model interpret its own internal embeddings, mapping the activations of "black/white" to natural language descriptions, and counting the frequencies of "race-related" vs "color-related" in the explanations. Aligned models generated 74.4% fewer race-related explanations than base models.

**4. Activation Engineering**

Caching the activation vectors of racial concepts from explicit racial contexts like "Race: black and white." and replacing the activations of "black/white" at target layers during the forward pass of implicit bias prompts. A sliding window of 10 layers was used to test the intervention effects across different layers.

**5. Weight Intervention (LoRA Fine-tuning)**

Curating 431 input-output pairs where inputs contain ambiguous usages of "black/white" and outputs are explicit race-related factual statements. LoRA is applied to the query and value projections in self-attention, fine-tuning early layers (1-20), late layers (21-32), and all layers (1-32) respectively.

### Loss & Training

Activation injection is a training-free intervention at inference time. LoRA fine-tuning utilizes the standard language modeling loss (next-token prediction) trained on 431 race-aware samples. The parameters are only applied to the QV projections of designated layers, reducing the LoRA parameter size by up to 62.5%.

## Experiments & Results

### Experiment 1: Comparison of Biased Behaviors Before and After Alignment

| Model | Explicit Bias $\hat{p}_{\text{explicit}}^{\text{black}}$ | Implicit Bias $\hat{p}_{\text{implicit}}^{\text{black}}$ |
|------|------|------|
| Llama 3 70B Base | 49.6% | 64.1% |
| Llama 3 70B Instruct | **8.13%** ↓ | **91.4%** ↑ |

Alignment reduces explicit bias from 49.6% to 8.13% ($b=0.415, p<.001$), but increases implicit bias from 64.1% to 91.4% ($b=0.273, p<.001$). When using racial names (e.g., DeShawn/Jake), implicit bias drops to 38.5%, whereas it remains at 93.6% when using color prefixes—indicating that ambiguity is key.

### Experiment 2: Comparison of Intervention Effects

| Method | Implicit Bias | Reduction | Explicit Bias |
|------|----------|------|----------|
| Baseline (8B Instruct) | 97.3% | — | 61.1% |
| Activation Injection (Early layers 5-14) | 71.2% | -26.1pp | — |
| LoRA (Early layers 1-20) | **42.3%** | **-55.0pp** | 11.5% ↓ |
| LoRA (Late layers 21-32) | 58.7% | -38.6pp | 15.1% ↓ |
| LoRA (All layers 1-32) | 51.3% | -46.0pp | **0.5%** ↓ |

Early-layer LoRA intervention yields the strongest effect ($b=0.549, p<.001$), and is more stable than all-layer LoRA (confidence interval 11.9% vs 21.3%). All-layer LoRA achieves the most thorough reduction in explicit bias (down to 0.5%), but degrades instruction-following capability (17% of responses exhibited formatting issues).

### Key Findings

- **Alignment amplifies implicit bias**: Llama 3 70B experiences a 27.3 percentage point increase in implicit bias post-alignment, a trend reproduced in the 8B model as well. This represents a systematic side-effect of alignment.
- **Race blindness is the underlying mechanism**: Aligned models have a Race Blind Score of 0.188 (compared to -0.022 for base models), showing that alignment training leads the model to encode "black/white" as colors rather than race in ambiguous contexts, failing to trigger safety guardrails.
- **Early layers are crucial for race encoding**: Activation patching reveals that racial concepts are primarily encoded in the first 1/3 of the Transformer layers. Intervening in early layers is vastly superior to late layers (reducing bias by 55.0pp vs. 38.6pp).
- **"Enhancing awareness" outperforms "unlearning/removal"**: After LoRA fine-tuning makes the model more race-aware, implicit bias drops from 97.3% to 42.3%, and explicit bias decreases simultaneously—indicating that racial awareness does not contradict fair behavior.
- **Ambiguity acts as the trigger for bias outburst**: When the prompt contains racial names, the aligned model activates its safety guardrails (38.5%) even during implicit associations. However, the polysemy of black/white bypasses safety detection.

## Highlights & Insights

- Reveals the core contradiction of alignment: alignment renders explicit bias safer while making implicit bias more dangerous, serving as an important warning to the AI Safety community.
- Precise human psychology analogy—LLM "race-blindness" aligns perfectly with human color-blindness theory, offering powerful interdisciplinary explanatory value.
- Challenges the traditional debiasing paradigm: effective bias mitigation does not entail forcing the model to forget racial concepts, but rather enabling it to make fair judgments while remaining aware of race.
- The "Race Blind Score" can serve as a metric to monitor the bias side-effects of alignment methods.
- Methodologically unifies three threads of research: psychological experimental paradigms (IAT), mechanistic interpretability (activation patching, SelfIE), and causal intervention (LoRA).

## Limitations & Future Work

- Focuses solely on Black-White racial bias within the US cultural context; whether similar "blindness $\rightarrow$ bias" effects occur in other dimensions like gender, religion, or nationality remains unverified.
- Measurement of implicit bias relies on specific evaluation frameworks (adapted IAT, BBQ); different evaluation methods might produce different conclusions.
- Evaluated only on the Llama 3 series (8B, 70B); the situation for other architectures (e.g., Qwen, Mistral, GPT series) remains unexplored.
- The training set of 431 samples for LoRA intervention is relatively small; data quality and coverage could impact generalization performance.
- The intervention side-effects instruction-following capability—all-layer LoRA leads to formatting issues in 17% of the responses.
- Mechanistic interpretability findings are constrained by model architectures, interpretation methods, and human concept definitions, carrying the risk of over-interpretation.

## Related Work & Insights

- **Differences between explicit and implicit bias in LLMs**: Hofmann et al. (2024) and Bai et al. (2025) first systematically demonstrated bias in aligned models during implicit association, but did not provide a mechanistic explanation.
- **Activation patching and mechanistic interpretability**: Wang et al. (2022) (IOI circuit) and Meng et al. (2023) (ROME knowledge editing) provide frameworks for activation patching; this work novelly applies them to semantic attribution analysis of polysemous words.
- **Bias mitigation methods**: Dige et al. (2024) remove bias-related neurons via unlearning, and Marks et al. (2024) debias using feature ablation—this work proposes an opposite strategy.
- **Activation engineering**: Steering vector methods by Turner et al. (2024) and Panickssery et al. (2024); this work adapts them for racial concept injection.
- **SelfIE**: Embedded self-interpretation method proposed by Chen et al. (2024); this work utilizes it to visually verify the race-blindness phenomenon.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reveals the "race-blindness" side-effect of alignment, and proposes a counter-intuitive debiasing paradigm of "enhancing awareness rather than unlearning".
- **Technical Depth**: ⭐⭐⭐⭐ — Triple validation via activation patching, SelfIE, and LoRA, establishing a complete causal inference chain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thorough testing with 9,232 controlled prompts, multi-dimensional ablation, and statistical testing, though limited to the Llama 3 series.
- **Value**: ⭐⭐⭐⭐ — The Race Blind Score and early-layer intervention strategies hold direct engineering value.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — Exerts a significant impact on AI Safety research directions; the closed-loop demonstration of experiments, mechanisms, and interventions is exemplary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](../../ICLR2026/optimization/hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](../../NeurIPS2025/optimization/the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/optimization/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
