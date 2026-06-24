---
title: >-
  [Paper Note] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States
description: >-
  [ACL 2025][LLM Alignment][Jailbreak detection] Proposes HiddenDetect, a tuning-free safety detection framework based on internal activation states: it detects jailbreak attacks by monitoring refusal semantic signals in hidden states during LVLM inference, outperforming existing methods in AUROC by a wide margin across multiple models and multimodal benchmarks.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak detection"
  - "multimodal safety"
  - "hidden state analysis"
  - "LVLM"
  - "tuning-free defense"
  - "refusal semantics"
date: 2026-05-08
content_hash: fe1c457aa4b581b9
---

# HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States

**Conference**: ACL 2025  
**arXiv**: [2502.14744](https://arxiv.org/abs/2502.14744)  
**Code**: [GitHub](https://github.com/leigest519/HiddenDetect)  
**Area**: LLM Alignment  
**Keywords**: Jailbreak detection, multimodal safety, hidden state analysis, LVLM, tuning-free defense, refusal semantics

## TL;DR

Proposes HiddenDetect, a tuning-free safety detection framework based on internal activation states: it detects jailbreak attacks by monitoring refusal semantic signals in hidden states during LVLM inference, outperforming existing methods in AUROC by a wide margin across multiple models and multimodal benchmarks.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) such as GPT-4V and LLaVA achieve outstanding performance in visual reasoning and question answering by integrating visual and language modalities. However, multimodal capabilities also introduce new safety risks—compared to text-only LLMs, LVLMs are more vulnerable to adversarial manipulation.

**Limitations of Prior Work**: Existing safety defenses primarily rely on behavior-level approaches: (a) fine-tuning on safety datasets—computationally expensive and inflexible; (b) handcrafting safety prompts—often leading to over-defensiveness and reduced model utility; (c) inference-time reasoning modules (such as multimodal CoT)—failing to generalize to unseen attack strategies. These approaches are "ex-post remedies" that ignore safety signals that may already be encoded within the model.

**Key Challenge**: Does there already exist exploitable safety signals inside LVLMs when handling unsafe inputs? If so, why do multimodal inputs (especially adversarial images) still successfully bypass the safety mechanisms?

**Goal**: (1) Reveal whether safety-related signals are encoded in the hidden states of LVLMs; (2) Analyze the patterns of safety signals across modalities and layers; (3) Utilize these intrinsic signals to construct an efficient jailbreak detection framework.

**Key Insight**: Starting from an interpretability perspective, this work constructs a "Refusal Vector" to quantify the alignment between hidden states and refusal semantics, revealing that safety signals are most prominent in the middle layers and exhibit a delayed activation phenomenon under visual inputs.

**Core Idea**: LVLMs already encode safety signals internally; thus, jailbreak attacks can be detected in a tuning-free manner by monitoring the strength of refusal semantics in safety-sensitive layers.

## Method

### Overall Architecture

The pipeline of HiddenDetect is divided into two phases:

1. **Offline Phase**: Constructing the refusal vector + identifying safety-sensitive layers (requiring only a few samples)
2. **Online Detection**: Calculating the refusal semantic strength at each layer for new inputs, aggregating safety scores, and determining whether the input is safe

### Key Designs

#### 1. Constructing Multimodal Refusal Vectors (Refusal Vector)

**Function**: Constructs a sparse binary vector $\mathbf{r}$ in the vocabulary space, marking all tokens related to refusal behavior.

**Design Motivation**: Unlike previous LLM works that mine refusal signals solely from pure text, the refusal behavior of LVLMs is modulated by visual semantics, which requires capturing refusal tokens in multimodal contexts.

**Mechanism**:
- Collect the responses of the model to harmful image-text pairs and extract high-frequency refusal tokens (e.g., "sorry", "unable", "cannot") to form the initial Refusal Token Set (RTS).
- Project the hidden states of the last token position in each layer to the vocabulary space via the unembedding layer, collect the tokens corresponding to insertion top-5 logits, and add them to the RTS.
- Iterate until no new tokens are added; the final RTS then forms the refusal vector $\mathbf{r}$ (with 1s at corresponding indices and 0s elsewhere).

#### 2. Layer-wise Safety Awareness Evaluation (Refusal Strength Vector)

**Function**: Quantifies the alignment between the hidden states of each layer and the refusal semantics.

**Design Motivation**: Safety signals are not uniformly distributed across all layers—it is necessary to locate the "safety-sensitive layers" to achieve precise detection.

**Mechanism**: For few-shot safe/unsafe sample pairs, project the last-token hidden states of each layer to the vocabulary space, and compute the cosine similarity with the refusal vector:

$$F_l = \cos(\mathbf{h}_l, \mathbf{r}), \quad l \in \{0, 1, \ldots, L-1\}$$

Then calculate the "Refusal Discrepancy Vector":

$$F' = F_{\text{unsafe}} - F_{\text{safe}}$$

Layers where $F'_l > 0$ are designated as safety-sensitive layers. Experiments show that safety signals are strongest in the middle layers and decay in later layers to balance response relevance.

#### 3. Identifying the Range of Safety-Sensitive Layers

**Function**: Automatically determines the layer range $(s, e)$ for detection.

**Mechanism**: Using the discrepancy score of the last layer $F'_{L-1}$ as a conservative baseline, select all layers that exceed this baseline:

$$s = \min\{l : F'_l > F'_{L-1}\}, \quad e = \max\{l : F'_l > F'_{L-1}\}$$

#### 4. Safety Scoring and Detection

**Function**: Computes a safety score for the input prompt, identifying it as unsafe if the score exceeds a threshold.

**Mechanism**: Using the trapezoidal rule, calculate the AUC of the refusal strength within the range of safety-sensitive layers as the safety score:

$$s(F) = \text{AUC}_{\text{trapezoid}}(\{F_l : l \in \mathcal{L}_\mathcal{M}\})$$

### Key Findings: Delayed Safety Activation under Multimodal Jailbreaks

The paper further analyzes the refusal discrepancy vectors of five types of harmful queries:
- Safe activation for text-only harmful queries is the earliest and strongest.
- Even when the text itself is harmful, paired images weaken the safety activation in early and middle layers.
- When the harmful intent is conveyed solely through the image (with safe text), safety activation is severely delayed and significantly weakened.
- Bimodal safety alignment can significantly enhance the refusal strength of cross-modal queries, while having almost no effect on text-only queries.

## Key Experimental Results

### Main Results (Table 1: AUROC Detection Performance)

| Method | Tuning-free | XSTest | FigTxt | MM-SafetyBench | FigImg | JailBreakV-28K |
|------|--------|--------|--------|----------------|--------|----------------|
| **LLaVA** | | | | | | |
| Perplexity | ✗ | 0.610 | 0.758 | 0.825 | 0.683 | 0.781 |
| GradSafe | ✓ | 0.714 | 0.831 | 0.889 | 0.760 | 0.845 |
| JailGuard | ✗ | 0.662 | 0.784 | 0.859 | 0.715 | 0.801 |
| **HiddenDetect** | **✓** | **0.868** | **0.976** | **0.997** | **0.846** | **0.932** |
| **Qwen-VL** | | | | | | |
| GradSafe | ✓ | 0.678 | 0.809 | 0.872 | 0.744 | 0.839 |
| **HiddenDetect** | **✓** | **0.834** | **0.962** | **0.991** | **0.823** | **0.907** |
| **CogVLM** | | | | | | |
| GradSafe | ✓ | 0.617 | 0.762 | 0.812 | 0.692 | 0.789 |
| **HiddenDetect** | **✓** | **0.762** | **0.866** | **0.910** | **0.764** | **0.883** |

- HiddenDetect achieves the highest AUROC across all 3 models and 5 datasets.
- Compared to the strongest baseline, GradSafe, the average AUROC is improved by approximately 11.1 percentage points ($0.777 \rightarrow 0.888$).
- It achieves near-perfect detection on MM-SafetyBench (LLaVA: 0.997, Qwen-VL: 0.991).

### Ablation Study (Table 2: Influence of Safety-Sensitive Layers)

| Setting | FigTxt | FigImg | MM-SafetyBench |
|------|--------|--------|----------------|
| Without safety-sensitive layers | 0.630 | 0.502 | 0.750 |
| Using all layers | 0.861 | 0.640 | 0.960 |
| **Only safety-sensitive layers** | **0.925** | **0.830** | **0.977** |

- Detection performance drops drastically after removing safety-sensitive layers, especially on FigImg where it drops from 0.830 to 0.502.
- Using all layers introduces noise, which is inferior to precisely selecting safety-sensitive layers.

### Safety Layer Weight Scaling Experiment (Table 3)

Scaling the weights of layers 16-29 in LLaVA: a larger scaling factor leads to more rejected samples (increasing from the baseline of 33 to 49), further validating the critical role of these layers in safety decision-making.

### Key Findings

1. Refusal semantics are most prominent in the middle layers (approx. layers 16-29) and decay in later layers due to decoding requirements.
2. Visual inputs cause delayed safety activation—which explains why multimodal attacks are typically more successful than text-only attacks.
3. This method is equally effective for text-only jailbreak attacks, demonstrating cross-modal generalization capability.

## Highlights & Insights

- **Tuning-free design**: Eliminates the need for fine-tuning or external classifiers. It directly leverages the model's internal safety signals, incurring negligible computational overhead.
- **First systematic analysis of the cross-layer and cross-modal evolution of LVLM safety signals**: Discovers the critical phenomenon of "delayed safety activation by visual input," offering a new perspective on understanding multimodal jailbreaks.
- **Multimodal construction of refusal vectors**: Unlike prior works that construct the refusal direction purely from text, this work mines refusal tokens under visual conditions, making it more suitable for LVLM scenarios.
- **Few-shot applicability**: Requires only a small number of safe/unsafe samples to locate the safety-sensitive layers, removing the need for large-scale annotated data.
- **Intuitive and powerful visualization**: Projects hidden states onto the "refusal semantics - orthogonal semantics" plane, clearly illustrating the evolutionary process of safety signals from being entangled to separated.

## Limitations & Future Work

1. **Assumption that unsafe prompts always lead to distinct activation patterns**: Specially crafted adversarial inputs might bypass detection in the latent space, especially near the decision boundary of safety signals.
2. **Lacks "intervention" while focusing only on "detection"**: HiddenDetect flags unsafe prompts but does not correct the model's output; future work could combine it with activation steering to achieve response-level corrections.
3. **Refusal token set construction relies on human heuristics**: Initial seed tokens are selected based on empirical choice, which might miss certain model-specific refusal expressions.
4. **Static safety-sensitive layer range $(s, e)$**: The range of layers may need dynamic adjustment for different attack types.
5. **Only validated on open-source models**: Testing was not conducted on closed-source API models (limited by the inability to access intermediate layer activations).
6. **Threshold setting requires manual tuning**: The determination threshold for the safety score is configurable, but the paper does not fully discuss how to adaptively set it under different deployment scenarios.

## Related Work & Insights

### Comparison with GradSafe (Xie et al., 2024)
GradSafe detects unsafe inputs based on gradient information and is also training-free, but it relies on access to backpropagation. HiddenDetect only requires intermediate activations from the forward pass, making it computationally lighter while significantly outperforming GradSafe across all benchmarks.

### Relationship with Refusal Direction (Arditi et al., 2024)
Arditi et al. found that LLM refusal behavior is mediated by a single direction, but this was only validated on text-only LLMs. HiddenDetect extends this idea to multimodal scenarios, demonstrating that the layer-wise distribution of refusal signals in LVLMs is more complex and heavily influenced by modalities.

### Comparison with Mutation/Denoising Methods like CIDER / MirrorCheck
These methods analyze safety by mutating inputs or utilizing mirror consistency, requiring multiple inference passes. HiddenDetect completes detection in a single forward pass, demonstrating a clear efficiency advantage.

### Insights
- Hierarchical analysis of refusal semantics can be generalized to other safety-related tasks (e.g., bias detection, hallucination detection).
- The discovery of "delayed safety activation by visual input" can guide safety-alignment training for LVLMs—specifically, reinforcing the response of middle layers to multimodal harmful inputs.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first systematic analysis of internal safety signals in LVLMs for detection; the finding of "delayed activation" is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation with 3 models × 5 datasets × 7 baselines, robust ablation and visualization; however, tests on larger or closed-source models are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical flow from discovery to methodology with rich visualizations; occasional inconsistencies in mathematical notations.
- Value: ⭐⭐⭐⭐⭐ — Tuning-free, lightweight, and show strong generalizability, holding direct practical value for secure deployment of LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)

</div>

<!-- RELATED:END -->
