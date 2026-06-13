---
title: >-
  [Paper Note] SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones
description: >-
  [ACL2026][LLM Safety][Data Watermarking] SLIM proposes a low-coverage data watermarking approach for individual data owners: by making models learn patterns of divergent continuations for similar prefixes in local latent…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Data Watermarking"
  - "Black-box Verification"
  - "Low-coverage"
  - "Latent-space Confusion"
  - "Training Data Traceability"
date: 2026-05-08
content_hash: cf15b7568ec24476
---

# SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones

**Conference**: ACL2026  
**arXiv**: [2601.03242](https://arxiv.org/abs/2601.03242)  
**Code**: https://github.com/Henry-WWHHYY/SLIM/  
**Area**: LLM Security / Data Watermarking / Training Data Attribution  
**Keywords**: Data Watermarking, Black-box Verification, Low-coverage, Latent-space Confusion, Training Data Traceability

## TL;DR
SLIM proposes a low-coverage data watermarking approach for individual data owners: by making models learn patterns of divergent continuations for similar prefixes in local latent spaces, it ultimately manifests as statistically detectable local instability during black-box generation.

## Background & Motivation
**Background**: LLM training data is becoming increasingly expensive and involves copyright, privacy, and licensing issues. Data owners seek to know if their text has been used for model training, but modern LLMs often exhibit strong generalization and weak memory traces, making it difficult to draw reliable conclusions based solely on membership inference.

**Limitations of Prior Work**: Existing data watermarking methods usually require controlling a large proportion of the data or rely on obvious character patterns, fictional facts, reference models, or white-box/half-white-box signals like loss/perplexity. For average individuals or small institutions, they usually contribute only a small fraction of data, perhaps just a few documents or emails, and cannot coordinate watermark coverage on a large scale.

**Key Challenge**: Practical data watermarking must simultaneously satisfy three criteria: detectable under low coverage, difficult to discover or clean after mixing into large-scale corpora, and verifiable through black-box API access. These three conflict: the more obvious the watermark, the easier it is to detect but also the easier it is to filter; the more stealthy it is, the harder it is to retain a verifiable signal after massive training.

**Goal**: The authors focus on low-coverage data watermarking, attempting to allow small-scale data contributors to verify if a model has used their data while minimizing damage to general model capabilities and avoiding repetitive patterns identifiable by automatic cleaning rules.

**Key Insight**: The paper exploits the latent representation properties of LLMs: semantically similar prefixes usually map to adjacent latent regions, and autoregressive generation depends strongly on prefix representations. If training data binds multiple divergent continuations within the same local region, the model may produce abnormal generation instability in that region.

**Core Idea**: Shift the watermark from superficial string patterns to local latent space behavior, allowing verifiers to determine the presence of a watermark signal by statistically comparing the generation stability of a target prefix against local reference prefixes.

## Method

### Overall Architecture
SLIM consists of two phases. In the watermarking phase, a minimal number of target sequences are selected and split into prefixes and continuations; several semantically similar variants with divergent continuations are then constructed and mixed into the training data. In the verification phase, only black-box generation access is used to sample multiple continuations for both the target prefix and local reference prefixes. The distribution of semantic similarities between the beginnings of these continuations is compared; if significantly higher instability appears near the target prefix, a watermark is considered present.

This note summarizes the high-level mechanism, experiments, and limitations, without expanding on the operational details of generating watermark samples or verification procedures.

### Key Designs
1. **Low-coverage Watermarking Target**:
    - Function: Extends data watermarking from "those owning the entire training set" to "individual contributors owning only a few pieces of data."
    - Mechanism: The watermark signal does not rely on large-area repetitive injection but is concentrated in local representation regions near a very small number of target sequences. By default, each watermark instance modifies only a single target sequence, simulating signal dilution in a 500K arXiv abstract corpus.
    - Design Motivation: Real-world training corpora come from massive numbers of individuals; a single owner cannot control a large proportion of data. If the method required high coverage, its practical value for rights protection or licensing verification would be low.

2. **Latent-Space Confusion Zone**:
    - Function: Causes the trained model to produce detectable continuation instability near the target prefix.
    - Mechanism: Semantically similar prefixes fall into adjacent areas of the latent space. If these similar prefixes are associated with multiple highly divergent but reasonable continuations during training, the model's upper-level generation distribution forms a "confusion zone" in this local region. During inference, sampling multiple times from the same prefix will result in abnormally low or volatile similarity between continuation beginnings.
    - Design Motivation: Compared to random characters or fictional knowledge, local latent space behavior does not rely on conspicuous surface patterns, making it more suitable for bypassing standard deduplication, compression anomaly detection, and embedding density cleaning.

3. **Black-box Statistical Verification**:
    - Function: Performs attribution judgment without access to training loss, model weights, or internal representations.
    - Mechanism: The verifier collects multiple generations for target prefixes and local reference prefixes, compares the pairwise semantic similarity distributions of the generation beginnings, and derives a verification score via statistical testing. If a base model is available, a reference model-based comparison is performed; if not, a null hypothesis distribution is constructed using non-watermarked samples for a reference model-free approach.
    - Design Motivation: Commercial models typically only offer API access; strict black-box verification is more realistic than relying on loss/perplexity or internal logits.

### Loss & Training
SLIM itself is not a new model training objective but a framework for data-side watermark construction and verification. Experiments were primarily evaluated during the fine-tuning phase of Gemma-3-4B and the pre-training phase of Pythia-1.4B. The training corpus consisted of the first 500,000 sequences from gfissore/arxiv-abstracts-2021, totaling approximately 100M tokens. Model training lasted for two epochs to reduce the impact of overfitting; generation verification used stochastic decoding to observe local instability.

## Key Experimental Results

### Main Results
Traceability experiments show that in the reference model-based setting, the statistical shift of watermark samples increases monotonically with the number of watermark variants, while non-watermark samples show near-zero volatility. The authors report that at $K=16$, a fixed threshold $\Delta t=-40$ can distinguish between watermarked and non-watermarked samples; in the reference model-free setting at $K=64$, all three watermark samples fall outside the non-watermarked null hypothesis distribution.

Model utility experiments show that after adding a small number of SLIM watermarks, scores on standard benchmarks remain almost unchanged.

| Model | ARC w/o vs w/ WM | MMLU w/o vs w/ WM | BBQ w/o vs w/ WM | Conclusion |
|------|---------------|----------------|---------------|------|
| Pythia-160M | 0.324 / 0.316 | 0.246 / 0.245 | 0.469 / 0.488 | Change < 0.02, no systematic degradation |
| Llama-3.2-1B | 0.679 / 0.689 | 0.262 / 0.274 | 0.466 / 0.451 | Mixed results, overall impact very small |
| Gemma-3-4B | 0.819 / 0.822 | 0.554 / 0.555 | 0.557 / 0.565 | Utility basically maintained on large models |

Stealthiness experiments compared random character insertion, fictional knowledge, and SLIM. SLIM passed all three types of automatic detection, while the two control methods were identified by at least one rule.

| Method | N-Gram Filtering | Zlib Compression Anomaly | Embedding Similarity | Explanation |
|------|-------------|----------------|------------------|------|
| Random Char Insertion | Passed | Failed | Passed | High-entropy strings easily cause compression anomalies |
| Fictional Knowledge | Failed | Passed | Failed | Repetitive semantics and lexical patterns are more obvious |
| SLIM | Passed | Passed | Passed | Surface appears more natural; semantic density forms no significant anomaly |

### Ablation Study
The paper further tested scalability and post-training robustness. When the corpus increased from 200K to 500K, the watermark signal was diluted but remained within the detectable region; for models from 1B to 9B, signals in minimal models were unstable, while large models might require higher intensity to maintain margins; no significant mutual interference was observed when multiple independent watermarks existed simultaneously.

| Setting | Key Results | Meaning |
|------|----------|------|
| Data Scale 200K→500K | Average $\Delta t$ decays but remains below detection threshold | Larger data dilutes the signal; may need stronger watermarking |
| Model Scale Gemma 1B/4B/9B | Signal unclear at 1B; 4B/9B detectable but margins vary | Latent confusion zones depend on model capacity and representation structure |
| Simultaneous Injection (3/5/7) | Individual and average $\Delta t$ remain detectable | Multiple low-coverage watermarks do not conflict significantly in the short term |
| Post-training Full FT / LoRA / RLHF | Three samples remain detectable after post-training | Signal has some persistence, though fine-tuning weakens the magnitude |

In the post-training table, the $\Delta t$ for three watermark samples without post-training were -141.300, -152.916, and -90.047; after RLHF, they were -134.951, -157.963, and -102.662, indicating that RLHF has little impact on the signal. Full FT and LoRA significantly weakened some samples (e.g., S2 became -64.704 after Full FT and -47.797 after LoRA) but remained within the authors' defined detectable region.

### Key Findings
- Low coverage is the most important practical constraint of this paper: the method assumes individuals can only modify a minimal amount of data rather than controlling the entire training set.
- The watermark signal is not a superficial repetitive pattern but local generation instability, making it more stealthy against common text cleaning metrics.
- The method remains verifiable under black-box access, which is more aligned with commercial API settings than relying on loss, perplexity, or internal logits.
- Both data scale and model scale change the detection margin, indicating that SLIM's intensity parameters need recalibration for real-world deployment scales.

## Highlights & Insights
- The paper moves the key limitation of data watermarking from "can it be detected" to "can individual contributors detect it," a problem definition with significant real-world relevance.
- The Latent-Space Confusion Zone is a clever perspective: it does not attempt to force the model to memorize an explicit token but leaves behavioral traces in the local representation space.
- Experiments cover traceability, utility, stealthiness, scalability, and post-training persistence, providing a comprehensive evaluation.
- Insights for training data governance: future data licensing systems may not rely solely on legal contracts or platform logs but also combine statistical behavioral evidence, though false positives and interpretability must be strictly controlled.

## Limitations & Future Work
- The experimental scale is still smaller than real frontier model training; 500K sequences and 1B/4B/9B models only partially explain trends.
- The method relies on assumptions that "semantically similar prefixes are adjacent in latent space" and "divergent continuations form local instability," requiring more validation across different architectures, tokenizers, and training recipes.
- Watermark samples might still appear abnormal under individual human inspection; the paper's stealthiness is primarily established in large-scale mixing and automatic cleaning scenarios.
- Verification requires multiple black-box samples, which might be harder to implement for models providing only low-temperature or restricted sampling APIs.
- Statistical thresholds and false positive control are central to actual deployment; especially in real licensing disputes, a single statistical signal should not be over-interpreted as conclusive evidence.

## Related Work & Insights
- **vs WATERFALL / STAMP / TRACE**: These radioactive watermark methods typically rely on higher coverage or reference model conditions; SLIM focuses on individual-level low coverage and strict black-box.
- **vs Random Character / Unicode Watermarks**: Surface character watermarks are easily discovered by compression anomalies or text cleaning; SLIM attempts to hide signals within generation behavior.
- **vs Fictional Knowledge Watermarks**: Fictional knowledge can be used for specific QA verification but tends to form semantic repetitions or context constraints; SLIM emphasizes local instability in open-ended text completion.
- **Insights**: For LLM data governance, training data attribution may require a combination of "data-side marking + behavior-side statistics + auditing processes" rather than relying on a single detection technology.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Clear definition of low-coverage black-box data watermarking; the latent space confusion zone idea is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Rich evaluation dimensions, though further validation on real massive models and more complex corpora is needed.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main line with sufficient explanation of terminology and experimental settings; some future model settings are slightly idealized.
- Value: ⭐⭐⭐⭐☆ High inspiration for training data traceability and data ownership protection, though actual deployment requires stronger statistical rigor and legal auditing support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ACL 2026\] SWAN: Semantic Watermarking with Abstract Meaning Representation](swan_semantic_watermarking_with_abstract_meaning_representation.md)

</div>

<!-- RELATED:END -->
