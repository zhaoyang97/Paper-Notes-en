---
title: >-
  [Paper Note] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models
description: >-
  [ICML 2026][Multimodal VLM][Object Hallucination] This work finds that the intermediate-layer embeddings of instruction tokens in MLLMs naturally filter misleading information introduced from the visual side. Based on th…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Object Hallucination"
  - "Instruction Embedding"
  - "Logit Lens"
  - "Training-Free Detection"
  - "MLLM"
date: 2026-05-08
content_hash: db111ead6f0f181c
---

# Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.12258](https://arxiv.org/abs/2605.12258)  
**Code**: https://github.com/Fraserlairh/Instruction-Lens-Score (available)  
**Area**: Multimodal VLM / Hallucination Detection  
**Keywords**: Object Hallucination, Instruction Embedding, Logit Lens, Training-Free Detection, MLLM

## TL;DR
This work finds that the intermediate-layer embeddings of instruction tokens in MLLMs naturally filter misleading information introduced from the visual side. Based on this, a training-free InsLen score (Calibrated Local Score + Context Consistency Score) is proposed, which improves object hallucination detection AUROC by up to 13.81% across 5 MLLMs × 4 benchmarks.

## Background & Motivation

**Background**: Object hallucination in MLLMs (generating objects not present in the image) is a core obstacle to reliable deployment. Existing detection methods follow two lines: one relies on external models like GPT-4 for scoring, which is costly; the other mines internal model signals, such as attention weights of visual tokens (SVAR) or similarity between answer token embeddings and image patch embeddings (GLSIM).

**Limitations of Prior Work**: Internal signal methods almost exclusively bet on "visual evidence"—but the visual encoder or cross-modal attention may introduce misleading visual features (e.g., mistaking a silver spoon for a silver knife), causing hallucinated objects to receive inflated scores; patch-level representations only consider local context and cannot incorporate global object context.

**Key Challenge**: Using "visual evidence" to detect "visual hallucinations" introduced by the visual side is essentially using signals and noise from the same source. An independent pathway is needed to calibrate visual signals.

**Goal**: (1) Identify internal signals that can suppress misleading visual information; (2) Provide both patch-level local evidence and global context consistency.

**Key Insight**: The authors use Logit Lens to project intermediate-layer instruction token embeddings onto the vocabulary and unexpectedly find that instruction embeddings have high confidence for real objects in the image and low confidence for hallucinated objects (e.g., "bag" in Figure 1). On MSCOCO, instruction embeddings yield AUROC at least 8% higher than image embeddings. This overlooked filtering effect can serve as an independent pathway.

**Core Idea**: Use instruction embeddings to both calibrate visual scores and provide global object context, fusing the two signals for hallucination detection.

## Method

### Overall Architecture
Input consists of image $I$, user instruction $\mathbf{X}$ (default: "Please describe the image in detail."), and MLLM-generated answer $Y$. All instruction token embeddings $\{\mathbf{z}_j\}_{j=1}^{M}$, all image patch embeddings $\{\mathbf{v}_i\}$, and each object token embedding in the answer $\mathbf{h_o}$ are extracted from the penultimate layer of the MLLM language model. The entire InsLen process is training-free. For each object token, two complementary scores are computed: $S_{\rm cls}$ (Calibrated Local Score) and $S_{\rm ccs}$ (Context Consistency Score). The weighted sum $S_{\rm Ins}(\mathbf{o})=\omega S_{\rm cls}+(1-\omega)S_{\rm ccs}$ is used; a threshold $\mu$ is applied for binary classification, with scores below the threshold considered hallucinations.

### Key Designs

1. **Calibration Confidence (Cafe): Calibrating Inflated Visual Evidence**:

    - **Function**: Uses the maximum confidence of instruction embeddings for an object token, multiplying it with any existing vision-based score to "pull back" inflated visual predictions.
    - **Mechanism**: For each instruction embedding $\mathbf{z}_j$, Logit Lens yields a vocabulary distribution. The probability for object token $\mathbf{o}$ is taken, and the maximum across all $M$ instruction positions is used: $S_{\rm cafe}(\mathbf{o})=\max_{j} {\rm softmax}(\mathbf{W}_u\mathbf{z}_j/\tau)[\mathbf{o}]$, where $\tau$ is the temperature. This is then multiplicatively fused, e.g., with Local Similarity Score: $S_{\rm cls}(\mathbf{o})=S_{\rm cafe}(\mathbf{o})\cdot\frac{1}{K}\sum_k\cos(\mathbf{h_o},\mathbf{v}_k)$.
    - **Design Motivation**: Multiplication is chosen over addition because different vision-based scores (SVAR, Internal Conf, LSS) have inconsistent scales; multiplication is naturally compatible and does not require extra scaling. The max operator ensures that as long as any instruction position is confident about the object, the score is preserved, matching the observed filtering effect.

2. **Context Consistency Score: Introducing Global Object Context**:

    - **Function**: Aggregates global object context using instruction embeddings related to the object, then measures the consistency between the answer's object token embedding and this context.
    - **Mechanism**: Each instruction embedding is projected onto the vocabulary, and the top-$m$ embeddings with the highest confidence for object token $\mathbf{o}$, $\{\hat{\mathbf{z}}_n\}$, are selected and averaged: $\overline{\mathbf{z}}=\frac{1}{m}\sum_n \hat{\mathbf{z}}_n$. Consistency is computed using normalized $\ell_2$ distance: $S_{\rm con}(\mathbf{o})=\alpha-\|\mathbf{h_o}-\overline{\mathbf{z}}\|/\|\mathbf{h_o}\|$, then multiplied by the average confidence of these selected instruction embeddings for the object, yielding $S_{\rm ccs}=S_{\rm con}\cdot\overline{p}$.
    - **Design Motivation**: Vision-based scores only consider local patches and cannot distinguish cases like "silver spoon vs silver knife" with similar local textures. Instruction embeddings, computed via cross-attention over the entire image, inherently provide a global view. Top-$m$ selection excludes noisy instruction positions; $\ell_2$ is used instead of cosine to capture both direction and magnitude differences.

3. **Plug-and-Play with Multiple Models + Complementary Fusion of Two Scores**:

    - **Function**: InsLen can be attached to any existing vision-based detector as a calibration layer, and $\omega\in[0,1]$ balances local evidence and global consistency.
    - **Mechanism**: Cafe can be multiplied with any vision-based score (SVAR, Internal Conf, LSS, etc.); CCS and CLS are complementary, with the former capturing patch-level local evidence and the latter capturing object-level global context. The final score is $S_{\rm Ins}=\omega S_{\rm cls}+(1-\omega)S_{\rm ccs}$, with default $\omega=0.4$, $\alpha=2$, $\tau=10$, $m=4$.
    - **Design Motivation**: Training-free and independent of external models, deployment only requires an additional pass through the unembedding matrix, with negligible overhead; orthogonal to existing detectors and can be stacked for further gains.

### Loss & Training
This method is entirely training-free; all embeddings are directly taken from the penultimate layer of the frozen MLLM (Layer 31 for LLaVA, Layer 35 for Qwen3-VL). No new parameters are introduced, and only 4 hyperparameters $\omega, \alpha, \tau, m$ are used, with a single configuration shared across dozens of MLLMs.

## Key Experimental Results

### Main Results
5 MLLMs (LLaVA-1.5-7B, InstructBLIP-7B, mPLUG-Owl3-8B, LLaVA-OneVision1.5-8B, Qwen3-VL-8B) × 4 benchmarks (MSCOCO, Objects365, POPE, CLEVR), compared with 7 baselines (NLL, Entropy, Internal Conf, SVAR, Contextual Lens, EAZY, GLSIM), all evaluated with AUROC / AUPR.

| Model / Dataset | Metric | Strongest Baseline | InsLen | Gain |
|--------|------|------|----------|------|
| Qwen3-VL / MSCOCO | AUROC | 75.36 (SVAR) | 81.02 | +5.66 |
| LLaVA-1.5 / POPE | AUROC | 70.13 (GLSIM) | 83.94 | +13.81 |
| Qwen3-VL / Objects365 | AUROC | 70.84 (SVAR) | 77.44 | +6.60 |
| mPLUG-Owl3 / CLEVR | AUROC | 70.71 (SVAR) | 74.01 | +3.30 |

### Ablation Study

| Configuration | LLaVA-1.5 AUROC | Qwen3-VL AUROC | Note |
|------|---------|---------|------|
| Only $S_{\rm local}$ | 74.20 | 65.43 | Pure vision baseline |
| Only $S_{\rm cafe}$ | 80.41 | 77.06 | Instruction calibration only |
| $S_{\rm local}+S_{\rm cafe}$ (i.e., CLS) | 84.31 | 79.83 | Cafe adds 10 points to vision score |
| Only $S_{\rm con}$ | 80.69 | 71.94 | Global consistency only |
| Only Conf. weighted | 79.44 | 78.12 | Average confidence itself is informative |
| Full InsLen | **86.93** | **81.02** | All four components enabled |

### Key Findings
- Cafe is the largest contributor: On LLaVA-1.5, adding Cafe alone increases any vision score (SVAR/Internal Conf/LSS) by 7–10 AUROC, confirming that "inflated visual scores" are the core issue.
- CCS and CLS are highly complementary: CCS alone achieves 80.69 AUROC on LLaVA, CLS alone 84.31, and combined 86.93, indicating that patch-level local and object-level global signals are indeed distinct.
- Longer instructions yield better detection: On LLaVA, lengthening the instruction increases AUROC by 2.40%, as more instruction positions provide more internal visual information redundancy.
- Extremely low inference overhead: On Qwen3-VL, InsLen takes 564.5ms, only 2.9% of the 19550ms required for answer generation; much lower than EAZY's 40293ms.
- Still effective on post-trained models: On LLaVA-RLAIF-V (where easy hallucination is already suppressed, HR only 6.72%), InsLen still achieves 80.14 AUROC, 7.78 higher than GLSIM.

## Highlights & Insights
- The observation that "instruction embeddings understand images better than image embeddings" is counterintuitive but statistically supported and highly explanatory—misleading visual information is "voted out" by semantic priors on the instruction side after multiple attention layers. This insight can be transferred to tasks like visual question answering and grounding.
- The Logit Lens + top-$k$ high-confidence token selection paradigm is lightweight and can serve as an "internal diagnostic" tool for VLM debugging and prompt engineering.
- Using multiplication for calibration is an underrated trick—avoids introducing new scaling factors and is more stable when combining methods.

## Limitations & Future Work
- The authors acknowledge that Logit Lens can only translate internal signals into "literal tokens," while the model may store information as synonyms (e.g., "dog" as "puppy"), so selecting confidence based on generated answer tokens may miss some semantics.
- Representation drift affects deep-layer embeddings, leading to significant performance differences across MLLM architectures (e.g., mPLUG-Owl3 vs LLaVA).
- This method only detects hallucinations but does not correct them; combining with contrastive decoding for a closed loop may be more valuable.
- Evaluation is limited to static image description tasks; multi-turn dialogue and video understanding scenarios are not validated.

## Related Work & Insights
- **vs GLSIM (Park & Li 2025)**: GLSIM also fuses global-local signals, but its global signal comes from the image summary token; this work uses instruction tokens, which naturally provide a filtering effect, achieving 13.81 higher AUROC on POPE.
- **vs SVAR (Jiang et al. 2025b)**: SVAR uses attention ratio, still a visual evidence method; this work shows that multiplying Cafe with SVAR further increases AUROC by 7.47, indicating complementarity rather than substitution.
- **vs EAZY (Che et al. 2025)**: EAZY uses "zero out image tokens" for contrast, which is extremely costly (40s+/sentence); InsLen is training-free and nearly costless at inference, making it more suitable for online deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of using instruction embeddings for hallucination detection is novel and statistically validated, though the underlying tools (Logit Lens, cosine sim) are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 MLLMs × 4 benchmarks + post-trained variants + instruction length sensitivity + combinations with various vision scores.
- Writing Quality: ⭐⭐⭐⭐ Formulas are clear, but the naming of Cafe and CCS can be confusing, and some figures/tables could be more intuitive.
- Value: ⭐⭐⭐⭐ Training-free, plug-and-play, and near-zero overhead, making it suitable for direct integration into production MLLMs as a hallucination gate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[ICML 2026\] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](../../NeurIPS2025/multimodal_vlm/monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)

</div>

<!-- RELATED:END -->
