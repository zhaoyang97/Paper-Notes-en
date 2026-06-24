---
title: >-
  [Paper Note] COSMIC: Generalized Refusal Direction Identification in LLM Activations
description: >-
  [ACL 2025][LLM (Other)][Refusal direction] This paper proposes the COSMIC framework, which leverages cosine similarity in the activation space to automatically select refusal guidance directions. It operates entirely without relying on model output tokens or predefined refusal templates. COSMIC matches the performance of existing methods under standard settings, and is the first to successfully extract effective refusal directions in scenarios of adversarial complete refusal…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Refusal direction"
  - "Cosine similarity"
  - "Activation space intervention"
  - "Direction selection"
  - "Adversarial robustness"
date: 2026-05-08
content_hash: e4b5b425efdf7b25
---

# COSMIC: Generalized Refusal Direction Identification in LLM Activations

**Conference**: ACL 2025  
**arXiv**: [2506.00085](https://arxiv.org/abs/2506.00085)  
**Code**: [https://github.com/wang-research-lab/COSMIC](https://github.com/wang-research-lab/COSMIC)  
**Area**: LLM/NLP / Interpretability / LLM Safety  
**Keywords**: Refusal direction, Cosine similarity, Activation space intervention, Direction selection, Adversarial robustness

## TL;DR

This paper proposes the COSMIC framework, which leverages cosine similarity in the activation space to automatically select refusal guidance directions. It operates entirely without relying on model output tokens or predefined refusal templates. COSMIC matches the performance of existing methods under standard settings, and is the first to successfully extract effective refusal directions in scenarios of adversarial complete refusal and weakly aligned models.

## Background & Motivation

**Background**: LLM refusal behavior is a core mechanism of safety alignment. Existing inference-time intervention methods (such as directional ablation and activation addition) manipulate refusal behavior by modifying direction vectors in the activation space. Arditi et al. (2024) discovered that refusal behavior is encoded by a single direction in the activation space, enabling jailbreaking or inducing refusal without fine-tuning.

**Limitations of Prior Work**: Existing direction selection methods suffer from severe generalizability issues. Linear Concept Editing (LCE) relies on substring matching to detect refusal—requiring prior knowledge of the model's refusal template tokens (such as "I" or "As"), which easily results in false positives ("I can do that!") and false negatives ("Here's why I cannot help..."). Affine Concept Editing (ACE) requires human inspection and LLM-as-a-judge to select directions, which is labor-intensive and difficult to replicate.

**Key Challenge**: Existing methods assume that refusal behavior can be reliably detected from output tokens, but this assumption collapses in three critical scenarios: (1) the model uses non-standard refusal phrasing; (2) in adversarial scenarios, the model uniformly refuses all inputs, making harmful and harmless outputs indistinguishable; (3) weakly aligned models do not refuse harmful requests intrinsically, failing to provide contrastive signals.

**Goal** To design a direction selection framework completely independent of model outputs, capable of automatically identifying refusal directions under any alignment condition.

**Key Insight**: Since refusal behavior is encoded as a direction in the activation space, a high-quality direction should cause a "concept inversion" in the activations at the representation level after intervention—making harmful prompt activations look like harmless ones and vice versa. This inversion can be measured via cosine similarity without inspecting any output tokens.

**Core Idea**: Using the cosine similarity of activations pre- and post-intervention (the degree of concept inversion) to replace output token matching for optimal refusal direction selection.

## Method

### Overall Architecture

The input to COSMIC is a dataset of harmful/harmless prompts, and the output is the optimal refusal direction vector $\boldsymbol{r}^*$ along with its corresponding layer $l^*$ and token position $i^*$. The overall process consists of three steps: (1) extracting $5L$ candidate directions from the training set using difference-in-means; (2) performing interventions on each candidate direction and collecting activations on the validation set; (3) selecting the optimal direction via cosine similarity scoring. This direction can then be seamlessly combined with any inference-time intervention method (such as LCE or ACE).

### Key Designs

1. **Candidate Direction Generation via Difference-in-Means**:

    - **Function**: Extract candidate refusal directions from the model's residual stream.
    - **Mechanism**: Perform a forward pass for harmful and harmless prompts in the training set separately. Collect activations at the last 5 post-instruction token positions $i \in \{-5,-4,-3,-2,-1\}$ for each layer $l$. Compute the mean difference $\boldsymbol{r}_{i,l} = \boldsymbol{r}^+_{i,l} - \boldsymbol{r}^-_{i,l}$, where $\boldsymbol{r}^+$ represents activations of harmful prompts and $\boldsymbol{r}^-$ represents activations of harmless prompts. A total of $5L$ candidate directions are generated.
    - **Design Motivation**: The post-instruction tokens are critical positions where the model shifts from "understanding the input" to "preparing the output". Activation differences here best reflect how refusal behavior is encoded. This choice aligns with the findings of Arditi et al.

2. **Low-Similarity Layer Selection and Cosine Similarity Scoring**:

    - **Function**: Select the evaluation layer set $\mathcal{L}_{low}$ and score each candidate direction.
    - **Mechanism**: First, compute the baseline cosine similarity between harmful and harmless activations at each layer using the training set. Select the 10% of layers with the lowest similarity as the evaluation layers—as these layers best distinguish harmful and harmless behaviors. Then, for each candidate direction $\boldsymbol{r}_{i,l}$, perform ablation (removing refusal) and addition (adding refusal) on the validation set, and collect the post-intervention activations. Two core metrics are computed: $\bar{S}^{\text{refuse}} = \cos(\bar{a}_+, \bar{b})$ (whether harmless activations look like harmful ones after inducing refusal) and $\bar{S}^{\text{comply}} = \cos(\bar{a}, \bar{b}_-)$ (whether harmful activations look like harmless ones after removing refusal). Finally, the direction that maximizes the sum of these two metrics is chosen.
    - **Design Motivation**: This is the core novelty of COSMIC—evaluating the quality of directions by measuring the degree of "concept inversion", completely bypassing output tokens. The selection of low-similarity layers is based on the intuition that these layers encode the strongest refusal signals, making them the positions where intervention effects are most prominent.

3. **Filtering and Safety Guardrails**:

    - **Function**: Prevent the selection of spurious directions and ensure that interventions do not degrade model performance.
    - **Mechanism**: Three-fold filtering—(1) Median peak filtering: exclude directions at position $i=-1$ that lie at layers beyond the median peak location of other token positions, avoiding false positives caused by the recency effect of the last token; (2) discard directions in the last 20% of layers to prevent shallow interventions; (3) exclude directions where the KL divergence on harmless prompts exceeds 0.1 to preserve the model's performance on normal inputs.
    - **Design Motivation**: Experiments reveal that position $i=-1$ exhibits abnormally high cosine similarity peaks in the latter layers (Figure 7). This occurs because the very last token has a direct influence on the first output token, yielding spuriously high-scoring directions.

## Key Experimental Results

### Main Results: COSMIC vs. Existing Methods under Standard Settings (ASR / Induced Refusal Rate)

| Model | COSMIC-LCE ASR | LCE ASR | COSMIC-ACE ASR | Substring-ACE ASR |
|------|---------------|---------|----------------|-------------------|
| Llama-3.1-70B | 0.85 | 0.85 | 0.78 | 0.76 |
| Llama-3.1-8B | 0.62 | 0.63 | 0.84 | 0.84 |
| Qwen2.5-72B | 0.88 | 0.88 | 0.57 | 0.57 |
| Qwen2.5-7B | 0.91 | 0.91 | 0.81 | 0.82 |
| Gemma-2-27B | 0.00 | Does not converge | 0.53 | 0.53 |
| Gemma-2-9B | 0.47 | 0.47 | 0.78 | 0.78 |

COSMIC matches the performance of substring-matching methods under standard settings, but without any output assumptions. On Gemma-2-27B, where LCE fails to converge, COSMIC still functions effectively.

### Adversarial Complete Refusal Setting (Core Results from Table 1)

| Model | COSMIC-LCE ASR (All Refusal / Val Only) | COSMIC-ACE ASR (All Refusal / Val Only) | ACE Layer Shift |
|------|------|------|------|
| Llama-3.1-70B | 0.78 / 0.83 | 0.76 / 0.78 | No change |
| Llama-3.1-8B | 0.84 / 0.63 | 0.43 / 0.84 | No change |
| Qwen2.5-72B | 0.17 / 0.89 | 0.19 / 0.57 | No change |
| Qwen2.5-7B | 0.55 / 0.90 | 0.48 / 0.81 | No change |
| Gemma-2-27B | 0.05 / 0.00 | 0.02 / 0.53 | No change |
| Gemma-2-9B | 0.38 / 0.46 | 0.11 / 0.78 | No change |

### Key Findings

- **COSMIC-ACE is exceptionally robust in adversarial settings**: The selected layers across all tested models are completely identical to those in non-adversarial settings, with no performance degradation.
- **Significant differences in steerability across models**: Larger models (70B/72B) exhibit high jailbreak success rates under LCE but low induced refusal rates, while smaller models show the opposite, suggesting that refusal representation is related to model scale.
- **Non-monotonicity of refusal steering in weakly aligned experiments**: Increasing $\alpha$ (the activation addition coefficient) does not always enhance refusal. For instance, on Gemma-2-9B, $\alpha=2,3$ actually makes the model more vulnerable to jailbreaking compared to the baseline, which challenges the assumption of linear and affine representations.
- **Strong downstream model consistency**: Evaluation on GPQA/ARC/TruthfulQA shows that the intervention does not cause significant degradation in reasoning capabilities.

## Highlights & Insights

- **Entirely output-independent direction selection**: COSMIC is the first refusal direction selection framework that does not rely on any output token information. This implies that even if a model intentionally fabricates outputs (e.g., uniformly refusing in evaluations to mask its true capabilities), COSMIC can still uncover its authentic refusal structure from the activation level, which holds significant value for AI safety auditing.
- **Elegant design of the Concept Inversion scoring mechanism**: Defining a "good direction" as one that can make harmful activations look like harmless ones and vice versa, and quantifying this inversion via cosine similarity. This general idea is transferable to other behavior-steering tasks (such as honesty and hallucination control).
- **Low-similarity layer selection strategy**: Dynamically selecting the 10% of layers with the lowest cosine similarity as evaluation layers, bypassing the subjectivity of manual layer selection. Figure 2 clearly illustrates the vast differences in layer-wise similarity patterns across models (e.g., Gemma-2-27B shows high similarity across almost all layers), demonstrating the necessity of adaptive selection.

## Limitations & Future Work

- **The difference-in-means method itself lacks adversarial robustness**: When the training set is contaminated by adversarial system prompts, the direction generation (rather than selection) is compromised—e.g., the ASR of Qwen2.5-72B drops precipitously from 0.89 to 0.17. This indicates that while COSMIC solves the selection issue, it has not yet resolved the generation bottleneck.
- **Empirical threshold for the 10% layer selection**: The paper acknowledges that this ratio might not be globally optimal (as evidenced by Gemma-2-27B's anomalous high-similarity pattern) and currently lacks theoretical justification.
- **Non-monotonic response in weakly aligned models**: Increasing $\alpha$ occasionally leads to worse safety performance, suggesting that refusal behavior might not comply with a simple linear or affine structure. The current functional forms of interventions might be insufficient.
- **Validation limited to refusal behavior**: The concept inversion approach of COSMIC can theoretically be generalized to other behavioral dimensions such as honesty or hallucination, but this has not yet been experimentally verified.

## Related Work & Insights

- **vs. Arditi et al. (LCE, NeurIPS 2024)**: LCE was the first to identify a single refusal direction and used substring matching to select it. COSMIC retains LCE's difference-in-means direction generation but replaces its selection process, achieving comparable performance under standard settings without relying on refusal template assumptions.
- **vs. Marshall et al. (ACE)**: ACE introduces an affine structure and a baseline term to protect harmless information, but relies on manual selection and LLM judges. COSMIC replaces its selection pipeline to achieve full automation, and the resulting ACE directions prove extremely stable under adversarial conditions.
- **vs. Yu et al. (ReFAT)**: ReFAT leverages refusal directions for adversarial training to boost robustness, which requires precise input directions. COSMIC can provide the necessary directions for weakly aligned models, extending the applicability of ReFAT.
- **vs. Zou et al. (RepE)**: RepE employs PCA instead of difference-in-means to extract directions, which may capture non-linear structures more effectively and serves as a potential alternative for COSMIC's direction generation step.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core idea of using activation similarity instead of output matching for direction selection is simple and elegant; however, the intervention methods themselves inherit prior works.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive; evaluates 8 models across 4 method combinations, covering standard, adversarial, and weakly aligned scenarios, supplemented by downstream consistency evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rigorous mathematical expressions, though a bit symbol-heavy. Some equations could be simplified.
- **Value**: ⭐⭐⭐⭐ Holds practical value for AI safety auditing—it can verify if a model is merely faking refusal, addressing a critical gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Entity Identification in Language Models](on_entity_identification_in_language_models.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ICML 2025\] Generalized Interpolating Discrete Diffusion](../../ICML2025/llm_nlp/generalized_interpolating_discrete_diffusion.md)
- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)

</div>

<!-- RELATED:END -->
