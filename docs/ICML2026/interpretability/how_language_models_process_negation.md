---
title: >-
  [Paper Note] How Language Models Process Negation
description: >-
  [ICML2026][Interpretability][negation understanding] This paper analyzes the internal circuits of Llama-3.1-8B and Mistral-7B processing negation sentences like "X that is not Y is __" using mechanistic interpretability.…
tags:
  - "ICML2026"
  - "Interpretability"
  - "negation understanding"
  - "mechanistic interpretability"
  - "shortcut attention"
  - "construction and suppression"
  - "LogitLens"
date: 2026-05-08
content_hash: 26e4faf3c96117e5
---

# How Language Models Process Negation

**Conference**: ICML2026  
**arXiv**: [2605.03052](https://arxiv.org/abs/2605.03052)  
**Code**: https://github.com/Ja1Zhou/LM_Negation  
**Area**: interpretability  
**Keywords**: negation understanding, mechanistic interpretability, shortcut attention, construction and suppression, LogitLens

## TL;DR
This paper analyzes the internal circuits of Llama-3.1-8B and Mistral-7B processing negation sentences like "X that is not Y is __" using mechanistic interpretability. It reveals that models are capable of performing negation (mid-layer attention constructs $\bar Y$ representations at the final position, e.g., "not gas" → solid) but are suppressed by late-layer "shortcut" attention heads. Ablating these heads via "attention sinking" yields an absolute accuracy gain of up to 17% on negation tasks.

## Background & Motivation
**Background**: Mechanistic Interpretability (MI) currently focuses on "factual recall" prompts (e.g., "The Colosseum is in __"), which can be explained by additive contributions from various tokens. Common tools include LogitLens, causal tracing, and path patching.

**Limitations of Prior Work**: Negation does not naturally fit the additive paradigm—"not" itself carries no additive factual information; it must combine with the negated concept Y to influence the answer. Multiple studies from the BERT/RoBERTa era reported that LMs perform like "random guessing" (~50% accuracy) on negation, yet signs suggest models are internally sensitive to it. These contradictory findings leave the circuit-level mechanism of "why they fail" and "how they compute negation" unexplained.

**Key Challenge**: Prior MI work (e.g., Wang et al. 2023 regarding negative mover heads, McDougall et al. 2024) favors the "suppression hypothesis"—models list tokens related to Y and then suppress them. Conversely, neuroscience and certain prompting studies (Geva 2021) support the "construction hypothesis"—models explicitly generate a representation of $\bar Y = \text{not } Y$, which triggers the correct answer. It remains unclear which hypothesis holds, whether they coexist, or if "misleading terms" exist.

**Goal**: The paper addresses three sub-questions: (i) Do current open-source LLMs "know" how to negate, and where does the circuit fail? (ii) Can this "malicious" circuit be identified and ablated to recover accuracy? (iii) Does the model use construction, suppression, or both? Which is dominant?

**Key Insight**: The authors observe a divergence between "output accuracy" and "logit difference sensitivity" curves—accuracy is ~50%, but sensitivity to "not" exceeds 95%. This indicates the negation signal is computed but obscured by subsequent components. Locating this "overwhelming pressure" through residual streams and attention maps allows for circuit verification.

**Core Idea**: By combining "Attention Sinking" (forcing specific heads to focus only on the first and current tokens to "softly" disable their transport function), path patching, LogitLens, and SAE-based contrastive attribution, the negation circuit is decomposed into a four-stage pipeline: (1) early layers move "not" to the Y position, (2) mid-layers construct $\bar Y$ and move it to the last position, (3) mid-layers simultaneously apply weak suppression to Y, and (4) late-layer MLPs amplify $\bar Y$ into the correct answer. Late-layer "shortcut heads" are identified as the root cause of errors.

## Method

### Overall Architecture
The authors constructed a controlled dataset $\mathcal D = \{(P_+, P_-, y_+, y_-)\}$ with 648 items (162 questions × 4 templates). Positive examples follow "An animal that is an amphibian is a frog," while negative examples insert "not" (expecting "mammal"). Evaluations on Llama-3.1-8B, Mistral-7B-v0.1, Qwen2.5/3, Gemma-2, and OLMo-2 proceed in two stages: (1) §4 uses accuracy vs. logit sensitivity $\Pr[\Delta(P_-;y_-,y_+)>\Delta(P_+;y_-,y_+)]$ to prove internal comprehension, followed by Cumulative Attention Sink to identify the late layers as the error source; (2) §5 uses windowed Attention Sink and modified path patching on Llama-3.1-8B / Mistral-7B to locate causal mid-layer modules (~Layer 14), interprets outputs via LogitLens, and uses contrastive attribution with pre-trained SAEs to trace signals to specific MLP latent variables.

### Key Designs

1.  **Attention Sinking Ablation**:
    - **Function**: Disables specific attention modules with minimal side effects to discover causally important or detrimental heads.
    - **Mechanism**: Inspired by the "attention sink" phenomenon (Xiao et al. 2024), where the last token places 64%–80% of attention weight on the first and current tokens (Table 1). The authors overwrite the target head's attention pattern to focus only on these two tokens, cutting off cross-position information transport while preserving local MLP/Value computations. This includes "Cumulative" variations (sinking from layer $i$ to $L$) and "window" variations (sinking a specific range).
    - **Design Motivation**: Unlike Attention Knockout, sinking avoids introducing noise from "resourced activations" and does not require counterpropts. It clarifies whether $\mathcal{AO}(P_-)$ loses causality or if $\mathcal{AP}(P_+)$ is forced back in. Since attention remains normalized, numerical stability is preserved, allowing it to serve as a plug-and-play inference-time fix—improving negation accuracy from 50.5% to 67.8% on Llama-3.1-8B (Table 3).

2.  **Path Patching + LogitLens for Mid-layer "Construction Circuit" Localization**:
    - **Function**: Identifies which mid-layer attention modules perform the "not Y" composition and interprets the output semantics.
    - **Mechanism**: Employs modified path patching: setting the sender as $\mathcal{AO}_\ell$ and the receiver as the final output embedding. For $P_-$, $\mathcal{AO}_\ell(P_-^{pp})$ is replaced with $\mathcal{AO}_\ell(P_+)$ while keeping attention patterns fixed and recomputing MLPs. If the logit difference flips, the layer is causally significant. LogitLens then projects $\mathcal{AO}_\ell$ to the vocabulary. Tokens are labeled by gpt-oss-120B to check if they relate to "not Y" (e.g., "not gas" → solid, "not in Asia" → America).
    - **Design Motivation**: Path patching alone can be confused with the introduction of new causality from $P_+$, and LogitLens alone is descriptive rather than causal. Their intersection identifies both location and semantics. LLM labeling scales the analysis to 648 questions, confirming construction > suppression statistically (suppression was only found in ~30% of samples).

3.  **Contrastive Attribution + SAE for MLP Latents**:
    - **Function**: Completes the circuit analysis by finding specific MLP units that promote negated answers.
    - **Mechanism**: Uses the unembedding row difference $d = W_U(y_-) - W_U(y_+)$ as the target direction. Contribution is defined as $\mathcal C(x, P) = \langle W_U^\top \mathcal{LN}_{L+1}(x), d \rangle$. Two contrasts are performed: $\mathcal C(\mathcal{MO}_i, P_-) - \mathcal C(\mathcal{MO}_i, P_+)$ and $\mathcal C(\mathcal{MO}_i, P_-) - \mathcal C(\mathcal{MO}_i, P_-^{as})$. Intersection of top-10 MLPs identifies layers 17–25. Pre-trained SAEs (He et al. 2024) decompose the output into sparse latents $\mathcal{MO}_i \approx \sum_j \beta_j f_j$.
    - **Design Motivation**: The contrastive design removes background signals unrelated to the answer difference. Using SAEs to compress >10k dimensional activations into <100 sparse latents makes human interpretation scalable. Observations that top demoted tokens are often uninterpretable further reinforces that "construction > suppression."

### Loss & Training
This work involves inference-time mechanistic analysis without additional training. SAEs are reused from He et al. 2024. gpt-oss-120b is used for labeling. Evaluations are conducted at the last-token logit position.

## Key Experimental Results

### Main Results

Dataset: 648 "X that is not Y is __" controlled items. Metrics: positive/negative accuracy and sensitivity ($\Delta(P_-) > \Delta(P_+)$).

| Model | Pos Acc (%) | Neg Acc (%) | Sensitivity (%) | Neg Acc after Attn Sink (%) | Neg Acc after LogitLens (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B | 95.2 | 50.5 | 97.4 | 67.8 (+17.3) | 53.6 |
| Mistral-7B-v0.1 | 96.3 | 45.2 | 95.1 | 65.9 (+20.7) | 61.6 |
| Qwen2.5 | 93.5 | 57.6 | 96.0 | 65.4 | 59.4 |
| Qwen3 | 91.8 | 55.7 | 95.2 | 64.2 | 59.6 |
| Gemma-2 | 96.5 | 49.7 | 97.5 | 66.1 | 59.7 |
| OLMo-2 | 96.3 | 54.0 | 97.8 | 68.7 | 61.6 |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Vanilla full model | Neg Acc ≈ 50% | Near-random accuracy but high sensitivity → negation circuit exists but is suppressed. |
| Cumulative Attn Sink from optimal layer | Max Neg Acc +17% abs. | Optimal layers are consistently >0.5L → shortcut heads concentrate in mid-to-late stages. |
| Window Attn Sink @ layer 14 | Significant Neg Acc drop | Layer 14 is the causal core of the construction circuit. |
| Window Attn Sink @ layer 17 | Neg Acc increase | Confirms layer 17 contains "shortcut heads." |
| LogitLens on $\mathcal{AO}_\ell$ | >80% items show "not Y" tokens | Supports the construction hypothesis. |
| Suppression detection (same method) | ~30% items hit | Suppression exists but is weaker than construction. |
| OLMo-2 training scan | Neg Acc drops early then recovers | Shortcut heads form very early during pre-training. |

### Key Findings
- **"Computed but Obscured"**: All 6 models show sensitivity $\geq 95\%$ while negative accuracy stays between 45–58%. Black-box metrics underestimate internal negation capabilities. This gap is caused by late-layer shortcut heads; sinking them yields 17%+ improvement.
- **Construction-Dominant**: Under LogitLens, construction evidence appears in >80% of cases compared to ~30% for suppression. SAE latents promote interpretable concepts, while demoted tokens are often gibberish.
- **Shortcut Origin**: OLMo-2 checkpoints show Neg Acc plunging early in training. Shortcut heads are a byproduct of "X is Y" co-occurrence statistics in pre-training data, suggesting a need for "negation-aware" training data.

## Highlights & Insights
- **Attention Sinking as "Gentle" Ablation**: Unlike removing tokens, sinking leverages the model's natural tendency to offload attention to the first token. It introduces minimal distribution shift and serves as both a diagnostic and an inference-time fix.
- **Accuracy-Sensitivity Divergence**: This serves as a general signal for "hidden capabilities." If logit differences are sensitive despite poor accuracy, a detrimental late-layer circuit is likely present.
- **Scalable MLP Interpretation**: Combining contrastive attribution with SAEs reduces the manual effort from checking tens of thousands of dimensions to ~50 latents per sample. This recipe is applicable to CoT, refusal, and bias research.

## Limitations & Future Work
- The study focuses on "explicit negation" (not/no) and excludes lexical ("unhappy"), adverbial ("seldom"), or pronominal negation.
- The dataset is small (648 items) and limited to single-token answers and strict templates. Generalizability to long contexts or nested negation ("not X but Y") is unconfirmed.
- While Attention Sinking improves accuracy, the side effects on general QA or reasoning tasks were not systematically evaluated; disabling shortcut heads might penalize performance on common statistical priors.
- Future work: Incorporating Attention Sinking into training objectives or using circuit evidence to guide negation data augmentation.

## Related Work & Insights
- **vs Wang et al. 2023**: While they found "negative mover heads" supporting suppression, this work shows construction is more central (>2x evidence).
- **vs Geva 2021/2023**: Those works explain recall as additive MLP contributions. This paper provides a counter-example where negation requires a three-stage non-additive pipeline.
- **vs Hermann et al. 2024 / Mann et al. 2025**: This paper concretizes "shortcut features" into specific late-layer attention heads and provides actionable mitigation tools.
- **vs Gromov 2025 / Halawi 2024**: Those works find late layers redundant. Ours finds them explicitly detrimental in negation tasks, explaining why performance improves after sinking.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces Attention Sinking and Contrastive Attribution × SAE; first systematic proof of construction dominance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 6 models and training sequences, though dataset variety is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow from hypothesis to evidence and counter-evidence.
- Value: ⭐⭐⭐⭐⭐ Provides a training-free method to improve negation and extends MI from additive facts to compositional semantics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](towards_atoms_of_large_language_models.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](../../ACL2026/interpretability/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[ACL 2026\] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process](../../ACL2026/interpretability/why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](../../NeurIPS2025/interpretability/base_models_know_how_to_reason_thinking_models_learn_when.md)

</div>

<!-- RELATED:END -->
