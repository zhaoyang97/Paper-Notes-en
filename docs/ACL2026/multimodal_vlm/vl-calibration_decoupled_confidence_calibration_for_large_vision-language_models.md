---
title: >-
  [Paper Note] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning
description: >-
  [ACL2026][Multimodal VLM][Multimodal Calibration] VL-Calibration decomposes the verbalized confidence of LVLMs into visual confidence and reasoning confidence. By utilizing image perturbation KL divergence, token entropy…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multimodal Calibration"
  - "Decoupled Confidence"
  - "Visual Uncertainty"
  - "Reinforcement Learning"
  - "Hallucination Suppression"
date: 2026-05-08
content_hash: fa39df2a64376ba3
---

# VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning

**Conference**: ACL2026  
**arXiv**: [2604.09529](https://arxiv.org/abs/2604.09529)  
**Code**: https://github.com/Mr-Loevan/VL-Calibration  
**Area**: multimodal_vlm  
**Keywords**: Multimodal Calibration, Decoupled Confidence, Visual Uncertainty, Reinforcement Learning, Hallucination Suppression

## TL;DR
VL-Calibration decomposes the verbalized confidence of LVLMs into visual confidence and reasoning confidence. By utilizing image perturbation KL divergence, token entropy, and token-level advantage reweighting for model training, it simultaneously reduces ECE and improves accuracy across 13 visual reasoning benchmarks.

## Background & Motivation
**Background**: Large Vision-Language Models (LVLMs) can process mathematical charts, geometry problems, commonsense QA, and multidisciplinary reasoning. However, they lack reliable uncertainty expression when providing answers. A category of verbalized confidence calibration methods exists in text-based LLMs, which prompts models to output "how confident I am" and uses SFT, PPO, DPO, or GRPO to align this confidence with answer accuracy.

**Limitations of Prior Work**: Directly applying these methods to LVLMs causes structural mismatch. LVLM errors may stem from "misperceiving the image" or "correct perception but faulty reasoning." Training a single global confidence only allows the model to say "I am uncertain," without specifying whether the uncertainty arises from vision or logic. Furthermore, multimodal models are often dominated by language priors, giving high-confidence answers based on common text patterns even when visual evidence is insufficient.

**Key Challenge**: Calibration targets require judging whether an answer is correct, but LVLM answer correctness results from the joint effect of visual perception and subsequent reasoning. A single Brier-style confidence mixes these two error sources, resulting in coarse optimization signals. Moreover, visual confidence, which requires explicit supervision, lack human-annotated grounding/rationale truth labels.

**Goal**: The authors aim to solve three sub-problems: enabling models to explicitly distinguish between visual-stage and reasoning-stage confidence; constructing trainable visual certainty signals without visual ground truth; and providing fine-grained penalties for hallucinations caused by visual uncertainty during RL training.

**Key Insight**: The paper starts from a simple but effective observation: if a model's visual description truly relies on the image, the output distribution should change significantly after image occlusion; if the model is certain about its visual description, the token distribution should be sharper. Combining "image sensitivity" and "internal low entropy" yields a proxy for visual certainty that does not require human annotation.

**Core Idea**: Replace single confidence with visual confidence and reasoning confidence, and align visual confidence with actual perceptual reliability using endogenous visual certainty rewards.

## Method

### Overall Architecture
The input to VL-Calibration consists of an image $I$ and a text question $x$. The output is not just an answer $y$, but a structured trajectory: first generating a visual rationale $z_{vis}$ and visual confidence $c_{vis}$, then a reasoning chain $z_{reas}$ and reasoning confidence $c_{reas}$, and finally the answer. The overall answer confidence is synthesized from $c_{vis}$ and $c_{reas}$ rather than being a separate scalar.

The training process is based on GRPO. For a single multimodal question, a group of outputs is sampled. Intra-group advantages are calculated based on answer correctness, global confidence calibration, and visual confidence calibration reward terms, and the policy is updated under KL constraints. Compared to standard RLCR, this method isolates and supervises the visual stage.

The approach can be understood as three layers: the first layer modifies the output format to expose two uncertainty sources; the second layer constructs visual certainty pseudo-labels as training targets for $c_{vis}$; the third layer adjusts negative advantages at the token level to penalize erroneous tokens caused by low visual certainty more heavily.

### Key Designs
1.  **Visual-Reasoning Confidence Decoupling**:
    - **Function**: Decomposes the verbalized confidence of the LVLM from a single score into visual and reasoning confidence, helping the model locate whether errors originate from perception or logic.
    - **Mechanism**: The model trajectory is written as $\tau=(z_{vis}, c_{vis}, z_{reas}, c_{reas}, y)$. Here $z_{vis}$ represents a dense caption or visual evidence description, and $z_{reas}$ is the reasoning chain based on that evidence. The final confidence is synthesized using the harmonic mean $\Phi(c_{vis},c_{reas})=2c_{vis}c_{reas}/(c_{vis}+c_{reas})$.
    - **Design Motivation**: The harmonic mean is more conservative than the arithmetic mean; if either visual or reasoning confidence is very low, the global confidence is pulled down. This is suitable for multimodal reasoning because "unclear vision but smooth logic" and "clear vision but shaky reasoning" should not yield high total confidence.

2.  **Endogenous Visual Certainty Estimation (VCE)**:
    - **Function**: Provides optimizable pseudo-supervision for visual confidence in the absence of human-labeled visual correctness.
    - **Mechanism**: VCE considers two signals. First is visual grounding: calculating the KL divergence of the visual rationale token distribution between the original image and an image with random patch masking. A large $D_{KL}$ indicates the output depends on the image. Second is internal certainty: calculating the average entropy $H$ of the visual description tokens. Lower entropy indicates higher model certainty. The final visual certainty is $S_{vis}=\log(D_{KL}+\epsilon)-\log(H+\epsilon)$, which is then mapped to $[0,1]$ via intra-batch z-score and sigmoid.
    - **Design Motivation**: Looking only at KL rewards outputs that are sensitive to image changes but internally chaotic, while looking only at entropy might reward confident hallucinations driven by language priors. Subtracting the two requires both "image-constrained" and "internally stable" outputs. The log scale compresses the numerical range for RL stability.

3.  **Visual Certainty Aware Token-level Advantage Reweighting (TAR)**:
    - **Function**: More accurately penalizes hallucinations caused by visual uncertainty during GRPO updates, while avoiding blanket suppression of reasonable visual tokens.
    - **Mechanism**: Standard GRPO uses the same advantage for all tokens within a sample. This method applies an additional weight related to token-level visual certainty to tokens in the visual rationale that have negative advantages: the negative advantage of low-certainty tokens is amplified, while that of high-certainty tokens is weakened.
    - **Design Motivation**: Multimodal errors are not homogeneous. Generating specific visual content under low visual certainty is more likely ungrounded guessing; conversely, high-certainty tokens, even in incorrect samples, may contain valid perceptual evidence, and over-penalizing them harms visual capabilities.

### Loss & Training
The training objective consists of three rewards: answer accuracy reward $R_{acc}$, global confidence calibration reward $R_{cal}$, and visual confidence reward $R_{vis}$. $R_{cal}$ uses the synthesized confidence $\Phi(c_{vis},c_{reas})$ for Brier-style alignment with answer correctness, and $R_{vis}$ uses a squared error penalty between $c_{vis}$ and the stop-gradient $\tilde{S}_{vis}$. The paper samples 12K instances from ViRL-39K to form VL-Calibration-12K, primarily training Qwen3-VL-4B/8B, and validating the generalization on Qwen3-VL-30B and InternVL3.5-4B-MPO.

## Key Experimental Results

### Main Results
The paper evaluates Accuracy, AUROC, and Expected Calibration Error (ECE) across 13 visual reasoning and multidisciplinary benchmarks. The main conclusion is that VL-Calibration not only improves "confidence reporting" but also simultaneously enhances visual reasoning accuracy.

| Model / Scenario | Metric | Baseline / Strong Baseline | VL-Calibration | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-4B Avg | ECE ↓ | 0.421 | 0.098 | ~4.3x reduction |
| Qwen3-VL-8B Avg | ECE ↓ | 0.204 | 0.071 | ~65.2% reduction |
| Qwen3-VL-4B Avg | Accuracy ↑ | Strongest Baseline | Ours | +2.3% |
| Qwen3-VL-8B Avg | Accuracy ↑ | Strongest Baseline | Ours | +3.0% |
| MMMU-Pro | Accuracy ↑ | Strongest Baseline | Ours | +2.2% |
| A-OKVQA | ECE ↓ | 0.112 | 0.017 | Significant ECE drop |
| Qwen3-VL-30B | Accuracy / AUROC / ECE | 0.652 / N/A / High | 0.803 / 0.767 / 0.082 | Effective on large models |
| InternVL3.5-4B-MPO | Accuracy / ECE | RLCR Strong Baseline | 0.689 / 0.103 | Cross-architecture effectiveness |

### Ablation Study
Ablations focus on Qwen3-VL-4B to verify the contributions of "decoupling itself," "visual certainty estimation," and "token advantage reweighting."

| Configuration | ACC ↑ | AUROC ↑ | ECE ↓ | Description |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-4B Base | 0.516 | 0.763 | 0.421 | Original model is overconfident with low accuracy |
| RLCR | 0.704 | 0.694 | 0.167 | Global confidence RL is effective, but AUROC drops |
| RLCR + Decoupled | 0.701 | 0.682 | 0.164 | Minimal gain from output format change alone |
| + VCE Entropy only | 0.688 | 0.723 | 0.119 | Entropy improves calibration but sacrifices accuracy |
| + VCE KL only | 0.709 | 0.721 | 0.124 | Image perturbation is effective but less stable |
| + VCE Entropy + KL | 0.715 | 0.751 | 0.121 | Dual signals are more balanced than single |
| Ours + TAR | 0.727 | 0.763 | 0.098 | Full method is best; TAR further boosts ACC and ECE |

### Key Findings
- Decoupled confidence only becomes truly effective when paired with visual supervision. Simply forcing the model to output $c_{vis}$ and $c_{reas}$ while optimizing for overall correctness yields performance close to basic RLCR.
- The two components of VCE are complementary. The authors observe that using only entropy leads to entropy collapse, while using only KL can lead to entropy explosion; the combination reduces calibration error while maintaining training stability.
- Reliability diagrams show the Base model severely overestimates itself in high-confidence intervals (ECE 0.421); after VL-Calibration reduces ECE to 0.098, the confidence bins align much closer to the ideal diagonal.
- In the unanswerable setting (removing images) of DynaMath, the method reduces the average confidence of unanswerable samples to 0.218 while maintaining 0.834 for answerable ones. The confidence gap reaches 0.616, higher than Base (0.228) and RLCR (0.405).
- VCE shows strong correlation with Gemini-3-pro-preview visual judgments, reporting AUROC=0.746, SRCC=0.496, and Kendall's Tau=0.370, proving pseudo-labels are not just fitting random noise.

## Highlights & Insights
- The biggest highlight is advancing calibration from "answer confidence" to "error source confidence." This makes LVLM uncertainty expressions more diagnostic and suitable for rejection, review, and human takeover in safety-critical scenarios.
- The construction of VCE is practical: it requires no human-annotated rationales, instead extracting supervision from the model itself via image perturbation and token entropy. This logic can migrate to video, 3D, or document VLMs by designing appropriate input perturbations.
- TAR moves calibration from the sample level to the token level—a critical but often overlooked detail. Multimodal hallucinations are often triggered by a few specific tokens; token-level advantages enable more refined shaping than sample-level rewards.
- The harmonic mean is a simple yet apt inductive bias. It treats "vision" and "reasoning" as a serial system where unreliability at either end reduces final credit, matching the risk structure of multi-step visual reasoning.

## Limitations & Future Work
- Computational overhead remains significant. VCE requires additional forwards under image perturbation and token distribution statistics. While more direct than multi-sampling methods, it still needs optimization for large-scale online serving.
- While experiments cover Qwen3-VL 4B-30B and InternVL3.5-4B, behaviors on models above 70B, different vision encoders, and long-video inputs have not been systematically verified.
- Visual certainty pseudo-labels rely on the assumption that "distribution change after perturbation represents grounding." For robust vision encoders or tasks requiring fine-grained local evidence, the KL signal might not always equal true visual understanding.
- The evaluation focuses on benchmark-level calibration and accuracy. Future work could look at human-AI collaboration: whether visual confidence can trigger more reasonable follow-up questions, rejections, or tool calls.
- The current method requires RL training. Future work could explore lightweight LoRA, test-time calibration heads, or prompt-only decoupled outputs to reduce deployment costs.

## Related Work & Insights
- **vs RLCR**: RLCR uses GRPO to reward answer correctness and global Brier calibration. Ours inherits this RL approach but decomposes global confidence into interpretable visual and reasoning dimensions with visual pseudo-supervision. The advantage is clearer error localization at the cost of more complex signals.
- **vs SaySelf / PPO-C / Rewarding Doubt**: These methods target verbalized confidence calibration in text LLMs, focusing on "whether the answer is right." VL-Calibration specifically models multimodal perception errors to avoid compressing visual issues into general linguistic uncertainty.
- **vs VL-Uncertainty / Self-Consistency**: Sampling-based uncertainty methods estimate disagreement through multiple generations—universal but expensive. Ours uses endogenous signals (KL and entropy), making it more direct during training and aligning it with specific visual confidence tokens.
- **vs Reliability Diagrams and Traditional ECE**: Traditional calibration often works at the output probability or post-processing level. Ours directly trains the model to generate interpretable confidence, making calibration results readable by users and usable for downstream policy decisions.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Effective decomposition of visual and reasoning confidence with paired pseudo-supervision.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid across 13 benchmarks, ablations, cross-model, and unanswerability analysis, though ultra-large models and real deployment could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic, complete formulas/analysis, though some tables are quite dense.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for high-risk multimodal applications by distinguishing between "misperception" and "faulty logic."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](../../ICLR2026/multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](../../ICML2026/multimodal_vlm/hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ACL 2026\] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models](progresslm_towards_progress_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)

</div>

<!-- RELATED:END -->
