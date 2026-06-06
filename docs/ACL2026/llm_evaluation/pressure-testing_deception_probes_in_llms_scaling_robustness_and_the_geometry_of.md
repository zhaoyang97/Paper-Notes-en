---
title: >-
  [Paper Note] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations
description: >-
  [ACL2026][LLM Evaluation][Deception Detection] This paper systematically stress-tests deception probes trained on internal activations of LLMs…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Deception Detection"
  - "Activation Probes"
  - "Representational Geometry"
  - "Style Robustness"
  - "Gemma 3"
date: 2026-05-08
content_hash: 00debab3b2bb0e71
---

# Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations

**Conference**: ACL2026  
**arXiv**: [2605.27958](https://arxiv.org/abs/2605.27958)  
**Code**: https://github.com/techsachinkr/llm-deception-probe-stress-test  
**Area**: LLM Evaluation / AI Safety  
**Keywords**: Deception Detection, Activation Probes, Representational Geometry, Style Robustness, Gemma 3  

## TL;DR
This paper systematically stress-tests deception probes trained on internal activations of LLMs, revealing that near-perfect AUROC on clean data does not translate to deployable robustness: neither single-direction nor entropy-proxy explanations hold; deceptive signals are distributed across multi-dimensional weak features, and style-augmented training can recover probes on 27B models from near-random performance to 0.983 held-out style AUROC.

## Background & Motivation
**Background**: Increasing work in AI safety aims to directly read internal model states using linear probes on residual stream activations to judge whether an output stems from an "honest" or "deceptive" state. The appeal of such methods lies in their independence from output text labels and manual review, theoretically serving as automated deception monitors.

**Limitations of Prior Work**: Existing deception probes often report AUROC scores near 1.0 on clean benchmarks but fail when encountering distribution shifts. Phenomena listed in the paper include adversarial suffixes reducing recall to nearly zero, distractor behaviors decoupling deceptive output from probe-detectable representations, and RL-obfuscation suppressing probe scores while preserving deceptive outputs. That is, high scores do not necessarily imply an understanding of what is being measured.

**Key Challenge**: Probe failure may stem from multiple causes: deception might be a linear direction easily masked by style; it could involve multi-dimensional subspaces or convex cone structures; or probes might merely measure proxy variables like "cognitive load/entropy." Without disentangling these explanations, simply stating "probes are fragile" cannot guide improvements in data, models, or geometric assumptions.

**Goal**: The authors aim to answer three specific questions: First, is the deceptive representation a transferable single direction? Second, how does the geometric structure change across deception types and model scales? Third, is the collapse caused by style distribution shift a fundamental limitation or a methodological issue resulting from narrow training distributions?

**Key Insight**: The paper selects the architecturally homogeneous Gemma 3 instruction-tuned series (1B, 4B, 12B, 27B) to separate scaling effects from architectural differences. It pre-registers four falsifiable hypotheses and uses cross-domain transfer, multi-dimensional PCA probes, entropy residualization, and eight types of style perturbations for exclusion.

**Core Idea**: To reframe the question of "whether deception probes are effective" from a single benchmark score into a "stress-test problem across four dimensions: scale, style, geometry, and proxy variables."

## Method
The methodology does not propose a new deception probe but constructs a diagnostic framework. It first builds honest/deceptive contrastive data using system prompts, then trains probes on the residual stream of the final token across different model layers, and finally checks what the probe has learned through experiments targeting different explanations.

### Overall Architecture
The overall pipeline is divided into five steps. First, three primary types of data are constructed: D-RepE uses factual questions with honest/intentionally false system prompts; D-Role uses role-playing scenarios to induce deception; D-MASK uses samples from the MASK benchmark where "output answers contradict internal belief estimates." Sandbagging data was excluded from the main experiment due to low model compliance.

Second, for each Gemma 3 model (1B/4B/12B/27B), residual stream activations are collected at the final token of all layers after response generation. The authors emphasize that this position represents the internal state after the model has selected an honest or deceptive strategy.

Third, L2 logistic regression probes are trained for each model, dataset, and layer, using AUROC as the primary metric. Hyperparameters $C \in \{0.01, 0.1, 1.0, 10.0\}$ are selected via 3-fold CV, with confidence intervals obtained via 10,000 bootstrap iterations.

Fourth, diagnostic experiments are designed around four geometric hypotheses: cross-dataset transfer to test single-directionality, PCA top-$k$ probes to test multi-dimensional encoding, permutation nulls for significant subspaces, and Logit Lens entropy/residualization to test entropy proxy explanations.

Fifth, eight style perturbations are constructed (pirate, Shakespearean, childlike, formal academic, slang, robotic, poetic, sarcastic) to test distribution shift. Furthermore, four styles are used for augmented training, with the remaining four used for held-out testing.

### Key Designs
1. **Pre-registered Geometric Hypothesis Matrix**:

    - **Function**: Decomposes "why probes succeed or fail" into four falsifiable explanations: H-LIN, H-SUB, H-CONE, and H-ENT.
    - **Mechanism**: H-LIN requires a single-direction probe trained on any deception domain to achieve AUROC $\geq 0.90$ cross-domain; H-SUB expects top-$k$ multi-dimensional probes to significantly outperform 1D; H-CONE requires evidence of directional asymmetry and NMF/PCA reconstruction; H-ENT expects probe scores to correlate strongly with Logit Lens entropy, with AUROC dropping significantly after residualization.
    - **Design Motivation**: This avoids post-hoc storytelling using high scores; results map back to specific hypothesis predictions, acting as a diagnostic report rather than a benchmark ranking.

2. **Layer-wise and Cross-domain Decomposition**:

    - **Function**: Determines whether poor cross-domain transfer is due to "choosing the wrong layer" or if different deception types are truly non-collinear in representation space.
    - **Mechanism**: Three conditions are compared: source probe on source best layer tested on target; source weights applied to target best layer activation; and retraining on target best layer using source data. $C2-C1$ isolates layer mismatch impacts, while $C3-C2$ isolates feature relearning impacts, with the remaining gap indicating geometric disjointness.
    - **Design Motivation**: Without this step, low cross-domain AUROC could be dismissed as "wrong layer selection." After decomposition, the net improvements for 12B/27B were -0.048 and -0.028, showing the problem lies in geometry rather than layer selection.

3. **Style Perturbation and Augmentation Comparison**:

    - **Function**: Distinguishes whether poor probe robustness is due to a lack of stable deceptive signals in the model or a narrow training distribution causing the probe to capture shortcuts.
    - **Mechanism**: Probes trained only on clean D-RepE are tested against 8 style shifts. Then, augmented probes are trained on clean + 4 styles and tested on the remaining 4 unseen styles. Improvement in held-out styles suggests the probe is forced to aggregate robust weak features rather than memorizing styles.
    - **Design Motivation**: This directly corresponds to deployment scenarios. Real users do not speak in the neutral style of benchmarks; a probe failing against Shakespearean or sarcastic instructions is not a reliable monitor.

### Loss & Training
The primary probe is a logistic regression with L2 regularization, trained layer-wise with the best layer selected via validation. Multi-dimensional experiments extract top-$k$ PCA components from activations or difference vectors, $k \in \{1, 3, 5, 10, 20, 50\}$, before training a linear classifier. MLP probes serve as an upper-bound baseline (2 layers, 256 hidden dim, ReLU, 0.3 dropout). Style augmentation uses 4B and 27B models with clean data plus 4 training styles, tested on clean, seen, and held-out styles.

## Key Experimental Results

### Main Results
On clean D-RepE, best-layer probes for all scales approached perfect scores; however, this result does not imply deployability, as style and cross-domain experiments showed high dependence on training distribution.

| Model | D-RepE AUROC | Acc | F1 | Best Layer | MLP Upper Bound | Remarks |
|------|--------------|-----|----|------------|-----------------|---------|
| Gemma 3 1B-IT | 0.999 [0.998, 1.000] | 0.983 | 0.985 | 8/26 | 1.000 | Near-perfect on clean |
| Gemma 3 4B-IT | 0.998 [0.996, 1.000] | 0.983 | 0.985 | 33/34 | 0.999 | Best layer very late |
| Gemma 3 12B-IT | 0.999 [0.998, 1.000] | 0.980 | 0.985 | 23/48 | 0.998 | Text-only baseline < 0.5 |
| Gemma 3 27B-IT | 1.000 [0.999, 1.000] | 0.980 | 0.990 | 27/62 | 0.999 | Highest clean, most style-fragile |

Multi-dimensional probes directly refuted the "single deception direction" explanation: $k=1$ only achieved AUROCs between 0.609 and 0.804, whereas $k=5$ exceeded 0.90 across all models.

| PCA Dimension | 1B | 4B | 12B | 27B | Conclusion |
|---------------|----|----|-----|-----|------------|
| $k=1$ | 0.609 | 0.661 | 0.804 | 0.672 | Single direction insufficient |
| $k=3$ | 0.816 | 0.963 | 0.845 | 0.819 | Significant recovery in some |
| $k=5$ | 0.908 | 0.981 | 0.938 | 0.990 | All above 0.90 |
| $k=10$ | 0.977 | 0.994 | 0.984 | 0.997 | Near-perfect |
| $k=50$ | 0.999 | 0.997 | 0.994 | 0.999 | Near MLP upper bound |

### Ablation Study
Style perturbation was the most critical robustness ablation. Clean probes were nearly perfect on standard data, but failed increasingly with scale under 8 style shifts; the 27B mean AUROC was only 0.485, essentially random. After style augmentation, 4B and 27B models recovered to near-perfect scores on held-out styles.

| Configuration | 4B mean AUROC | 27B mean AUROC | Note |
|---------------|---------------|----------------|------|
| clean probe, 8 styles | 0.759 | 0.485 | Trained on neutral D-RepE; 27B most fragile |
| style-augmented, seen styles | 0.997 | 1.000 | Clean + 4 training styles |
| style-augmented, held-out styles | 0.979 | 0.983 | Unseen styles; shows generalized learning |
| Gain on held-out | +0.220 | +0.498 | Huge gains on 27B |

Cross-domain transfer and entropy tests yielded clear negative results.

| Diagnostic Item | Key Figure | Supported/Refuted Explanation |
|-----------------|------------|-------------------------------|
| 12B layer-mismatch decomposition | remaining gap 0.336, net improvement -0.048 | Refutes "just layer mismatch" |
| 27B layer-mismatch decomposition | remaining gap 0.425, net improvement -0.028 | Supports geometric disjointness |
| entropy correlation | max $|\rho|=0.454$ | Refutes entropy proxy hypothesis |
| residualization drop | max $\Delta$AUROC = 0.004 | Probes detect more than entropy |
| per-domain PCA | $k^*=0$ on 4B/27B | No signf. coherent subspace |

### Key Findings
- AUROC $\geq 0.998$ on clean data often misleads researchers into believing probes are reliable; however, 27B mean AUROC dropping to 0.485 under style shift shows a massive gap between benchmark ceiling and deployment robustness.
- The single-direction steering/probing explanation is invalid. $k=1$ loses 20% to 40% of discriminative power, while $k=5$ recovers most signals, indicating deception-related info is dispersed across multiple weak directions.
- Entropy proxy explanations are largely ruled out. If probes merely measured cognitive load, entropy correlation would approach 0.8 with significant AUROC drops after residualization; in reality, max correlation is 0.454 and max drop is 0.004.
- Style augmentation is the most practical result. It suggests probes are not inherently fragile but require diverse behavioral distributions to avoid capturing prompt-regime or output-style shortcuts.

## Highlights & Insights
- The most valuable contribution of this paper is transforming the "probe accuracy" binary into a diagnostic question of "how a probe should perform under certain hypotheses." The pre-registered matrix gives each experiment clear discriminative power.
- Style augmentation experiments provide a pragmatic conclusion: viable signals likely exist in internal representations, but narrow training sets lead probes to prioritize "cheap" features. This is more instructive for activation monitor data construction than for model architecture itself.
- The paper "cools down" deception detection: D-RepE probes more accurately detect the *deceptive prompt regime* rather than whether the model *successfully* deceived. This prevents conflating intent monitors with output-level verifiers.
- The finding of multi-dimensional weak features serves as a reminder for representation engineering: do not over-rely on a "clean direction." For safety monitoring, ensembles, multi-domain probes, and style-diverse calibration may be more robust than single-direction interpretations.

## Limitations & Future Work
- The primary limitation is that all major data rely on system prompts to induce honesty/deception contrasts; thus, probes might learn prompt-regime features rather than natural strategic deception. The authors acknowledge that naturalistic deception data without system-prompt manipulation is needed to distinguish these.
- In D-RepE, 29% to 41% of "deceptive" labels might not be true deception, as models sometimes answer correctly even when "asked to lie." While decomposition shows this doesn't affect AUROC, it highlights that probes cannot distinguish successful from failed deception.
- Style augmentation was only performed on 4B and 27B; 1B and 12B are not yet verified. Cross-domain layer-mismatch decomposition only covered 12B and 27B.
- Experiments were confined to the Gemma 3 family. Generalizability to Llama, Qwen, or stronger agentic models requires cross-architecture replication.
- Sandbagging data was omitted due to low compliance, yet this is a critical deception type in AI safety. Future work should combine more natural task environments, agent traces, and human labeling.

## Related Work & Insights
- **vs RepE / logistic probe**: While methods like RepE treat internal activations as linearly separable signals for honesty/deception, Ours focuses on whether this separation holds across style, domain, and scale.
- **vs truth direction / representation engineering**: Unlike work by Marks and Tegmark emphasizing linear direction interpretability, Ours' $k=1$ and cross-domain results suggest deception is not a universal single-direction problem.
- **vs adversarial probe evasion**: While Bailey, Taylor, and others showed probes can be evaded, Ours further points out that much of this vulnerability stems from narrow training distributions, which can be mitigated via style augmentation.
- **vs entropy-based lying analysis**: Contrary to suggestions that lying correlates with higher token entropy, Ours demonstrates via Logit Lens and residualization that current probe signals do not primarily derive from entropy.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not a new probe type, but a very systematic geometric and robustness diagnosis.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive scale, domain, style, and entropy tests, though model family and naturalistic deception coverage remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ Hypotheses and discussions are clear; some tables are dense but conclusion-oriented.
- Value: ⭐⭐⭐⭐⭐ Vital for anyone using activation probes for safety monitoring, especially regarding the "clean AUROC vs. robustness" gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](../../ICML2026/llm_evaluation/geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)
- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](../../AAAI2026/llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)

</div>

<!-- RELATED:END -->
