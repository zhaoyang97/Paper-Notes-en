---
title: >-
  [Paper Note] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models
description: >-
  [ACL 2026][Interpretability][Activation Patching] The paper employs activation patching at the granularity of attention heads to demonstrate that Pythia and Gemma share a common mechanism—localized in three early-to-mid-…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Activation Patching"
  - "DAS"
  - "Filler-Gap Dependency"
  - "NPI"
  - "Pythia"
date: 2026-05-08
content_hash: 5e3f1b218f1c3475
---

# Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.22166](https://arxiv.org/abs/2604.22166)  
**Code**: https://github.com/ynklab/shared_syntactic_mechanism (Available)  
**Area**: Mechanistic Interpretability / Linguistics / Causal Analysis  
**Keywords**: Activation Patching, DAS, Filler-Gap Dependency, NPI, Pythia

## TL;DR
The paper employs activation patching at the granularity of attention heads to demonstrate that Pythia and Gemma share a common mechanism—localized in three early-to-mid-layer attention heads—for processing seven English filler-gap dependency (FGD) constructions. Scaling these heads' activations by 1.5 improves accuracy on the BLiMP benchmark. Conversely, Negative Polarity Item (NPI) licensing lacks such a unified mechanism. Furthermore, the "DAS direction" learned during training fails entirely on OOD data, suggesting that unsupervised patching is more reliable than supervised DAS.

## Background & Motivation
**Background**: To determine whether LLMs truly utilize "shared syntactic mechanisms" proposed by linguists, the mainstream approach involves causal abstraction—applying causal interventions to internal components via activation patching or DAS to observe changes in output. Prior studies (Finlayson 2021, Boguraev 2025, Arora 2024) have conducted preliminary analyses on subject-verb agreement and FGD, but these mostly focused on the residual stream without drilling down to the attention head level or systematically verifying OOD robustness.

**Limitations of Prior Work**: (1) **Coarse Granularity**: Analyzing only the residual stream can lead to misidentifying mechanisms as the "same" if different sets of heads produce similar residual representations; (2) **Risk of Training Artifacts**: The "causal directions" learned by supervised methods like DAS may overfit to the training lexicon or construction distribution, failing on OOD data without systematic verification; (3) **Lack of Verification Loop**: Few studies have verified whether an identified "shared mechanism" can actually alter model behavior on external benchmarks like BLiMP.

**Key Challenge**: While shared syntactic mechanisms are an attractive linguistic hypothesis, improper methodology can lead to either illusory findings or misidentification. Rigorous argumentation requires (a) fine-grained analysis at the attention-head level, (b) OOD generalization, and (c) behavioral-level steering validation.

**Goal**: To examine whether LMs share internal mechanisms across multiple constructions for two categories of syntactic phenomena (FGD and NPI) at the attention head and MLP granularity; and to contrast activation patching versus DAS for reliability.

**Key Insight**: (1) Selection of contrasting phenomena: FGD (7 constructions) primarily requiring long-distance syntactic dependencies, and NPI (8 constructions + control) which also incorporates semantic licensing; (2) Use of a modified ODDS as a fine-grained causal effect metric; (3) Strict separation of training, ID test, and OOD test sets using disjoint vocabularies.

**Core Idea**: Since activation patching requires no training, it does not suffer from overfitting, making it ideal for OOD control experiments. If patching remains stable on OOD data while DAS does not, it demonstrates the unreliability of DAS.

## Method

### Overall Architecture
The authors construct a minimal-pair evaluation set covering 7 FGD constructions (EWHK/EWHW/MWH/RELCL/CLEFT/PCLEFT/TOPIC), 8 NPI constructions (COND/DNEG/SONLY/QNT/EMBQ/SMPQ/SUP/ONLY), and a capital-knowledge control (CTRL). Activation patching is performed on Pythia (1B/2.8B/6.9B checkpoints) and Gemma 3 (1B/12B) to compute ODDS scores for four components: residual stream, attention output, MLP, and individual attention heads. Shared mechanisms are identified by comparing ODDS distributions across constructions. Two validation tracks are conducted: (i) scaling identified critical head activations by $\alpha$ and evaluating accuracy on BLiMP, SyntaxGym, and HANS; (ii) running DAS with leave-one-out training and ID/OOD evaluation to compare with patching results.

### Key Designs

1.  **Fine-grained Activation Patching + Modified ODDS**:
    *   **Function**: Quantifies the causal contribution of each component (layer $\times$ token $\times$ head) by independently replacing activations and measuring the change in the probability of the next token.
    *   **Mechanism**: Replaces component activation $f(b)$ on base input $b$ with $f(s)$ from source input $s$. The effect is measured using $\text{ODDS}(p, p_{\text{interv}}, T) = \frac{1}{|T|}\sum \log\left(\frac{p(y_b|b)}{p(y_b|s)} \cdot \frac{p_{\text{interv}}(y_b|s,b)}{p_{\text{interv}}(y_b|b,s)}\right)$, which tracks the probability shift of a specific token $y_b$ between the two inputs before and after intervention.
    *   **Design Motivation**: (a) Patching involves no training, eliminating overfitting risks; (b) Unlike the original Arora version which compares $y_b$ vs $y_s$, this version correctly handles asymmetric pairs in NPI scenarios; (c) Head-level granularity distinguishes mechanisms that appear similar in the residual stream but utilize different heads.

2.  **Contrastive Design: Shared 7-Construction vs Split 8-Construction**:
    *   **Function**: Tests the falsifiable hypothesis of "shared mechanisms" by comparing two phenomena.
    *   **Mechanism**: Plotting "layer $\times$ token $\times$ head" ODDS heatmaps for each construction. If the 7 FGD constructions show similar heatmaps (identical heads significant in the same layers), a shared mechanism is inferred. If the 8 NPI constructions show distinct distributions, mechanisms are deemed construction-specific.
    *   **Design Motivation**: FGD results show high ODDS concentrated in Pythia heads 7.5, 7.6, and 9.2 across all seven constructions. NPI distributions for DNEG, COND, and SUP vary significantly, proving that "sharing" is not an artifact of the patching method itself.

3.  **Steering Validation: From Head Manipulation to BLiMP Accuracy**:
    *   **Function**: Provides behavioral evidence that the discovered "shared mechanism" is functional in real-world sentences.
    *   **Mechanism**: Scaled activations of identified heads (7.5, 7.6, 9.2) by factors $\alpha \in \{0.8, 1.0, 1.5, 2.0\}$.
    *   **Design Motivation**: High ODDS on synthetic minimal pairs might be dataset artifacts. Measuring performance gains on independent benchmarks like BLiMP across broader categories (island effects, binding, etc.) confirms the mechanism's authenticity and generality.

### Loss & Training
Activation patching is inference-only. For DAS training, a 1D vector $a$ is learned by minimizing the loss $\min_a (-\sum_{(b,s,y_b,y_s)\in D}\log p_{\text{interv}}(y_s|b,s))$, with intervention defined as $f_{\text{interv}}(b, s) = f(b) + (f(s)\cdot a - f(b)\cdot a) \cdot a^T$. Training parameters: 100 steps, lr $5\times 10^{-3}$, batch size 4, 10% linear warmup. Datasets: 200 train / 50 ID / 50 OOD (disjoint vocabularies).

## Key Experimental Results

### Main Results
ODDS scores (Pythia 1B) for FGD constructions show a shared mechanism at specific heads:

| Construction | Head 7.5 ODDS | Head 7.6 ODDS | Head 9.2 ODDS | Shared |
| :--- | :--- | :--- | :--- | :--- |
| EWHK | ~2.0 | ~2.0 | ~1.5 | ✓ |
| EWHW | High | High | High | ✓ |
| MWH | High | High | High | ✓ |
| RELCL | High | High | High | ✓ |
| CLEFT | High | High | High | ✓ |
| PCLEFT | High | High | High | ✓ |
| TOPIC | High | High | High | ✓ |
| CTRL | ~0 | ~0 | ~0 | — |

BLiMP steering results (scaling heads 7.5/7.6/9.2):

| Category | $\alpha=0.8$ | $\alpha=1.0$ | $\alpha=1.5$ | $\alpha=2.0$ |
| :--- | :--- | :--- | :--- | :--- |
| Filler gap (Target) | Lower | Baseline | **+** | **++** |
| Island effects | Baseline | Baseline | **+** | **+** |
| Binding | Baseline | Baseline | **+** | **+** |
| Quantifiers | Baseline | Baseline | **+** | **+** |
| NPI | Baseline | Baseline | **+** | **+** |
| Subject-verb agr. | Baseline | Baseline | **+** | **+** |

### Ablation Study
**Activation Patching vs DAS (ID vs OOD)**:

| Setting | ID ODDS | OOD ODDS | Consistency |
| :--- | :--- | :--- | :--- |
| Patching (Residual) | High (Layer 7+) | **High (Matches ID)** | ✓ |
| DAS (Residual) | High (Layer 7+) | **Significant Drop** | ✗ (Overfitting) |
| Patching (Head) | High | **High** | ✓ |
| DAS (Head) | Medium | Slight Drop | Partial ✓ |

**Training Dynamics**: High-frequency constructions (EWHK) stabilize by step 4000, while low-frequency ones (PCLEFT) continue rising until step 10000. Shared mechanisms emerge hierarchically, conditioned on frequency.

**Scaling**: In larger models (Pythia 6.9B, Gemma 3 12B), mechanisms shift to earlier layers, but the head sparsity and shared structure remain stable.

### Key Findings
*   **Three heads (7.5, 7.6, 9.2) drive nearly all FGD processing**—a rare example of clean sparsity and localization in mechanistic interpretability.
*   **Training frequency dictates mechanism convergence**—suggesting "shared mechanisms" emerge through frequency-driven learning rather than innate priors.
*   **DAS fails on OOD data**—a warning that supervised causal directions might simply overfit the dataset.
*   **Manipulated heads boost multiple hierarchical syntax tasks**—indicating these heads process a general "hierarchical dependency" skeleton rather than FGD specifically.

## Highlights & Insights
*   **Methodology**: The combination of contrastive design (FGD vs NPI), strict OOD separation, and behavioral steering sets a rigorous benchmark for mechanistic interpretability.
*   **Scientific Discovery**: Empirically validates "shared syntactic mechanisms" by indexing them to specific attention heads.
*   **Training Dynamics**: Reveals that data diversity dictates whether mechanisms extend from high-frequency to low-frequency constructions.
*   **DAS Overfitting Warning**: Highlights the necessity of OOD validation for trained causal methods and rehabilitates the value of training-free causal interventions.

## Limitations & Future Work
*   English-centric: Languages with free word order (e.g., Japanese, Finnish) might exhibit different modularity.
*   Dataset scope: Focuses on synthetic minimal pairs; while BLiMP provides broader validation, real-world complexity is higher.
*   Construct coverage: Does not yet extend to anaphora or ellipsis.
*   NPI Divergence: The reason for the lack of a shared mechanism in NPI remains to be explored—complexity of semantic licensing vs patching signal strength.
*   Model variety: Confirmed on Pythia and Gemma 3; applicability to closed-source models (GPT/Claude) remains unknown.

## Related Work & Insights
*   **vs Boguraev et al. (2025)**: Advances beyond residual stream analysis to the head level with rigorous OOD/steering validation.
*   **vs Finlayson et al. (2021)**: Extends early causal mediation work by addressing OOD risks and fine-grained head localization.
*   **vs Jumelet et al. (2021)**: Finds mechanism divergence in decoder LMs, contrasting with the unified monotonicity hypothesis for NPIs.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](../../CVPR2026/interpretability/text-guided_fine-grained_video_anomaly_understanding.md)
- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](../../ICML2026/interpretability/dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)

</div>

<!-- RELATED:END -->
