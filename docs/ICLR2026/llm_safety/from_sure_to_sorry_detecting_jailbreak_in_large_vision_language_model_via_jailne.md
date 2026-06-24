---
title: >-
  [Paper Note] From "Sure" to "Sorry": Detecting Jailbreak in Large Vision Language Model via JailNeurons
description: >-
  [ICLR 2026][LLM Safety][Jailbreak Detection] JDJN reduces LVLM jailbreak detection to the "neuron level"—it uses causal ablation to locate a small set of **JailNeurons** in each layer specifically activated by jailbreak inputs. By aggregating these activations across layers to train a lightweight SVM, it achieves robust generalized detection of unseen attack types with near-zero false positives and ultra-low latency.
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Jailbreak Detection"
  - "LVLM Safety"
  - "JailNeurons"
  - "Neuron Localization"
  - "Generalized Detection"
date: 2026-05-08
content_hash: 23cc602406e51020
---

# From "Sure" to "Sorry": Detecting Jailbreak in Large Vision Language Model via JailNeurons

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ofJaimOZBF](https://openreview.net/forum?id=ofJaimOZBF)  
**Code**: To be confirmed  
**Area**: LLM Safety / LVLM Jailbreak Detection  
**Keywords**: Jailbreak Detection, LVLM Safety, JailNeurons, Neuron Localization, Generalized Detection  

## TL;DR
JDJN reduces LVLM jailbreak detection to the "neuron level"—it uses causal ablation to locate a small set of **JailNeurons** in each layer specifically activated by jailbreak inputs. By aggregating these activations across layers to train a lightweight SVM, it achieves robust generalized detection of unseen attack types with near-zero false positives and ultra-low latency.

## Background & Motivation
**Background**: Large Vision Language Models (LVLMs) inherit the powerful capabilities of LLMs while amplifying security risks. The visual modality expands the attack surface, with three main types of jailbreak attacks: adversarial perturbations via gradient optimization, rendering malicious text as image characters (e.g., FigStep), and using semantically relevant images to amplify harmfulness. Existing defenses include training-time alignment (computationally/annotation expensive) or inference-time detection (input preprocessing, output discrimination, or intermediate representation checking).

**Limitations of Prior Work**: Inference-time detection generally suffers from two issues: either being restricted to specific attack types or requiring the model to complete the entire generation (e.g., using a judge LLM), leading to high latency unsuitable for deployment. More critically, existing research on "safety mechanisms" focuses on **SafeNeurons** (explaining why aligned models refuse normal harmful queries), while the mechanism of how jailbreaks "bypass" these safety features remains uncharacterized.

**Key Challenge**: Jailbreak-related signals are indeed hidden in the model's high-dimensional hidden states (a single-layer linear classifier can achieve nearly 100% accuracy on ID data), but **no single layer generalizes to all OOD attacks**. Preliminary experiments show that the accuracy of any single layer drops below 80% on the worst-performing distribution. High-dimensional activations are filled with noise unrelated to jailbreaks, and a single layer is insufficient to cover the features of diverse attacks.

**Goal**: Propose an efficient, generalizable, and non-intrusive (no modification to the target model) jailbreak detector that maintains high TPR and extremely low FPR for unseen attack types and OOD benign data.

**Core Idea**: **[Neuron Sparsity Hypothesis]** Analogous to the sparsity of SafeNeurons, the authors hypothesize that only a very small number of neurons (**JailNeurons**, measured at <2%) encode jailbreak-related signals; **[Causal Localization + Cross-layer Aggregation]** Use "Sure to Sorry" causal ablation to identify these neurons and sample layers from "top to bottom" to aggregate features, isolating robust and transferable discriminative signals from noise.

## Method

### Overall Architecture
JDJN (Jailbreak Detection via JailNeurons) decomposes detection into two sub-problems—how to locate JailNeurons in each layer and how to select the most informative layers—corresponding to a three-stage pipeline: ① **JailNeuron Localization**: Train a sparse mask for each layer to identify critical neurons that, when "shielded," flip the model output from harmful to refusal; ② **Detector Training**: Sample layers at equal intervals from top to bottom, concatenate the JailNeuron activations from these layers, and train a lightweight SVM; ③ **Deployment**: Perform a single forward pass during inference, slice and aggregate activations according to the masks, and feed them to the SVM for judgment. If a jailbreak is detected, refusal is triggered immediately, skipping token-by-token generation.

```mermaid
flowchart LR
    A[Jailbreak + Benign Samples] --> B[Layer-wise Sparse Mask Training<br/>Causal Ablation: Sure→Sorry]
    B --> C[Threshold τ to Filter JailNeurons]
    C --> D[Sampling Every k Layers from Top to Bottom]
    D --> E[Concatenate Multi-layer JailNeuron Activations]
    E --> F[Train Linear SVM Detector]
    F --> G[Inference: Single Forward → Slicing → SVM Decision]
```

### Key Designs
**1. From Sure to Sorry: Causal Localization of Single-layer JailNeurons** — This is the core of the method. Given a jailbreak input that would trigger a harmful response ("Sure, here is..."), the goal is to identify which small set of neurons, if shielded, would flip the output to a refusal ("Sorry, I cannot..."), as these neurons are the components the jailbreak causally depends on. Specifically, a forward hook is registered for the $i$-th layer, using a learnable mask $m\in[0,1]^d$ to rewrite the layer output: $h(o_i, m) = (1-m)\odot o_i$. Then, a sparse-regularized optimization problem is solved to align the modified model output with the target embedding $e_s$ of a refusal word (e.g., "Sorry"):

$$m^* = \arg\min_{m\in[0,1]^d} \lambda\|m\|_1 + L_{CE}(f_i(m,x), e_s)$$

where the $L1$ regularization $\|m\|_1$ promotes mask sparsity (minimal intervention), and $L_{CE}$ is the cross-entropy loss. To satisfy the $m\in[0,1]^d$ constraint, $m$ is reparameterized as $\mathrm{sig}(\delta)$ (sigmoid), optimizing the unconstrained parameter $\delta\in\mathbb{R}^d$. Neurons with mask values above a threshold $\tau$ (e.g., 0.4) are designated as JailNeurons. This "causal flipping" perspective fundamentally distinguishes it from prior representation-similarity methods—it locates neurons that **cause** jailbreak success rather than just being correlated with it.

**2. Top-to-bottom: Multi-layer Selection to Overcome Single-layer Non-generalizability** — Since single-layer information is insufficient and fails to generalize, JDJN uses cross-layer aggregation to cover the diverse characteristics of jailbreak behavior. It adopts arithmetic sampling: for an $l$-layer model, it selects one layer every $k$ layers starting from the first, total $l_j=\lceil l/k\rceil$ layers, to capture representations at different abstraction levels. Hidden state components corresponding to JailNeurons (mask value $>\tau$) in these layers are sliced and concatenated as the discriminative feature. This directly addresses the finding that "different attack types affect different parts of the model."

**3. Lightweight Linear SVM: A Deliberate Choice Against Overfitting** — The aggregated JailNeuron activations are fed into a linear SVM for binary classification. The authors compared MLP and non-linear SVMs, finding that more complex models did not perform better: MLPs achieved extremely high accuracy on ID data but overfitted severely, with OOD generalization significantly lower than the linear SVM. Coupled with JailNeurons reducing high-dimensional noise to <2%, the linear classifier is both data-efficient (requiring only a few hundred samples) and robust, demonstrating that "sophisticated classifiers are unnecessary when features are well-selected."

## Key Experimental Results
Four LVLMs (MiniGPT4-7B, LLaVA-v1.5-7B, Qwen2-VL-7B, Janus-pro-7B) × Three attack categories (Gradient-based JAMLLM, Typographic FigStep, JailbreakV benchmark) × Multiple benign datasets (MM-Bench, MM-Vet, Normal, ScreenSpots, AndroidControl). Training uses 80% of a single attack + a single benign set, evaluating on ID and OOD (unseen attacks/unseen benign data).

### Main Results: Detection Success Rate (TPR@FPR≤0.05, LLaVA / Janus-pro)

| Method | JailBreakV | FigStep | JAMLLM (LLaVA) | JailBreakV | FigStep | JAMLLM (Janus) |
|------|-----------|---------|----------------|-----------|---------|----------------|
| **JDJN1** (Train on JailBreakV) | **0.997** | **1.0** | **0.942** | **0.996** | **1.0** | 0.853 |
| JDJN2 (Train on FigStep) | 0.732 | 1.0 | 0.524 | 0.838 | 1.0 | 0.776 |
| JailGuard | 0.676 | 0.532 | 0.71 | 0.573 | 0.566 | 0.71 |
| GradSafe | 0.862 | 0.742 | 0.534 | 0.844 | 0.728 | 0.454 |
| JailDAM | 0.913 | 0.926 | 0.342 | 0.917 | 0.932 | 0.433 |
| AdaShield | 0.675 | 0.786 | 0.213 | 0.774 | 0.812 | 0.353 |

JDJN1 achieves >99% TPR on ID and maintains 94.2% TPR on entirely unseen JAMLLM, significantly outperforming 7 baselines; training with the more diverse JailBreakV yields better generalization than FigStep.

### Efficiency Comparison (Processing time per FigStep sample)

| Method | LLaVA | Janus-pro |
|------|-------|-----------|
| **JDJN1** | **1.02s** | **0.26s** |
| JailGuard | 84.27s | 31.25s |
| ECSO | 15.12s | 5.36s |
| CIDER | 5.42s | 3.02s |
| Undefended (Full generation) | 12.08s | 4.29s |

JDJN requires only one forward pass without generating a full response, making it **faster than the undefended original LVLM**—detecting harmfulness triggers immediate refusal, bypassing expensive token-by-token generation.

### False Positive Rate (FPR, LLaVA, Training Attack fixed to JailBreakV)

| Benign Training Set | MM-Vet | MM-Bench | Normal | ScreenSpots | AndroidControl |
|-----------|--------|----------|--------|-------------|----------------|
| **JDJN1 (MM-Vet)** | **0.0** | **0.0** | **0.019** | **0.022** | **0.012** |
| JDJN3 (MM-Bench) | 0.168 | 0.0 | 0.346 | 0.343 | 0.212 |
| JDJN4 (Normal) | 0.285 | 0.21 | 0.0 | 0.198 | 0.272 |

### Key Findings
- **Generality of benign training set determines generalization**: Training on the open-ended image-text set MM-Vet yields FPR <5% across the board (mostly 0%); whereas MM-Bench (ABCD output only) or Normal (text-only) causes the detector to learn surface shortcuts like "presence of image" or "output format."
- **Mask guidance is essential and robust to $\tau$**: Using masks ($\tau=0.3$) at any $k$ outperforms the no-mask baseline ($\tau=0$). Performance is stable for $\tau>0$, with optimal $\tau=0.4$ for LLaVA and 0.2 for Janus-pro.
- **JailNeurons are extremely sparse**: When $\lambda\geq0.1$, JailNeurons account for <2% of neurons; at $\lambda=0.1$, accuracy across six datasets is >94%, but at $\lambda=0.5$, too much information is lost, dropping to 73% on Normal.
- **Choice of refusal word matters**: Localization using "sorry" is superior to "unfortunately" (0.956 vs 0.722 on Normal) because "sorry" is a more common refusal expression that encodes richer jailbreak-related information.

## Highlights & Insights
- **Evolution from "Correlation" to "Causality"**: While previous representation-based detection mostly compared hidden state similarity (correlation), JDJN anchors neurons as causally responsible for jailbreaks via "shield-to-flip" ablation, offering higher precision and interpretability.
- **Dual Complement to SafeNeurons**: It explicitly distinguishes between "why models refuse" (SafeNeurons) and "how they are bypassed" (JailNeurons), filling a missing gap in LVLM safety mechanism research.
- **Efficiency Surplus Over Undefended Models**: "Detect-and-refuse, skip generation" turns safety detection into a performance dividend rather than a deployment burden, making a strong case for real-world adoption.
- **Engineering Aesthetics of Sparsity + Linearity**: Using sparse masks to maximize the signal-to-noise ratio followed by a linear SVM to resist overfitting creates a data-efficient and robust non-intrusive detector.

## Limitations & Future Work
- **White-box Assumption**: The method requires access to internal activations and gradients for localization, making it inapplicable to closed-source API models.
- **Dependency on Successful Jailbreak Samples**: The localization process requires samples that "originally trigger harmful responses" for causal flipping, needing an available jailbreak set for cold starts.
- **Benign Training Set Sensitivity**: Generalization depends heavily on the generality of benign data (MM-Vet is much better than restricted MM-Bench/Normal); otherwise, the model may learn surface shortcuts.
- **Scale and Modal Breadth**: Experiments focused on 7B-scale models and three attack types; robustness against larger models, adaptive attackers (who know the JailNeurons mechanism), and more modal combinations remains to be verified.

## Related Work & Insights
- **Three Streams of LVLM Jailbreak Detection**: Input preprocessing (JailGuard, CIDER), output analysis (ECSO, Judge LLM), and internal activation anomalies (HiddenDetect using logit lens on refusal fragments). JDJN belongs to the third category but is the first to achieve neuron-level causal localization.
- **LLM/LVLM Safety Mechanisms**: High-dimensional representation analysis (logit lens, steering vector) vs. structural analysis (locating safety neurons with SNIP). This work is the first to study LVLM jailbreak mechanisms through neuron activation values and provide a detection algorithm.
- **Inspiration**: The paradigm of "sparse mask + causal ablation" to locate "functional neurons" is transferable to tasks like hallucination suppression, harmful concept erasure, and capability localization; the "detect-and-stop" approach suggests safety modules can optimize inference efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The JailNeurons concept and "Sure→Sorry" causal localization are meaningful dual complements to SafeNeurons research; neuron-level jailbreak detection for LVLMs is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered 4 models × 3 attacks × 5 benign sets + 7 baselines + multiple hyperparameter ablations across TPR/FPR/Efficiency; minor point deduction for absence of adaptive attacks and larger models.
- Writing Quality: ⭐⭐⭐⭐ Motivation progresses clearly (single-layer separable but not generalizable → sparsity hypothesis → cross-layer aggregation), with well-organized figures and RQs.
- Value: ⭐⭐⭐⭐ Near-zero FPR + faster-than-original speed + non-intrusive; significant engineering value for actual LVLM safety deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] STAR: Strategy-driven Automatic Jailbreak Red-teaming for Large Language Model](star_strategy-driven_automatic_jailbreak_red-teaming_for_large_language_model.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](../../ACL2026/llm_safety/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs](unlearning_isnt_invisible_detecting_unlearning_traces_in_llms_from_model_outputs.md)
- [\[ICLR 2026\] VEAttack: Downstream-Agnostic Vision Encoder Attack Against Large Vision Language Models](veattack_downstream-agnostic_vision_encoder_attack_against_large_vision_language.md)
- [\[ICLR 2026\] TAO-Attack: Toward Advanced Optimization-based Jailbreak Attacks for Large Language Models](tao-attack_toward_advanced_optimization-based_jailbreak_attacks_for_large_langua.md)

</div>

<!-- RELATED:END -->
