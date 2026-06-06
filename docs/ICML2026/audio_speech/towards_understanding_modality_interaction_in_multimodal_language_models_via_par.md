---
title: >-
  [Paper Note] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition
description: >-
  [ICML 2026][Audio & Speech][PID] This paper frames the decision-making of Multimodal Large Language Models (MLLMs) as an information decomposition from input to output. By using Partial Information Decomposition (PID)…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "PID"
  - "Modality Synergy"
  - "Omni-modal Models"
  - "Vision Dominance"
  - "LoRA Reweighting"
date: 2026-05-08
content_hash: 5a7613214be9bffe
---

# Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition

**Conference**: ICML 2026  
**arXiv**: [2606.00959](https://arxiv.org/abs/2606.00959)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Interpretability / Information-Theoretic Analysis  
**Keywords**: PID, Modality Synergy, Omni-modal Models, Vision Dominance, LoRA Reweighting

## TL;DR
This paper frames the decision-making of Multimodal Large Language Models (MLLMs) as an information decomposition from input to output. By using Partial Information Decomposition (PID), the mutual information of VL/omni-modal model predictions is decomposed into four components: "vision-unique / text-unique / redundant / synergistic". The study finds that the synergy term is the best indicator of predictive sensitivity to vision and identifies a "vision dominance" synergy bottleneck in omni-modal models. Finally, sample-level PID scores are used to guide a LoRA reweighting fine-tuning strategy, yielding stable 1–2 percentage point gains on MMStar, MMBench, and POPE.

## Background & Motivation

**Background**: MLLMs have evolved from perception systems to decision-making agents (scientific analysis, medical, embodied interaction). However, current evaluations almost exclusively focus on "prediction accuracy," using accuracy and modality ablation to judge whether a model truly utilizes vision or audio.

**Limitations of Prior Work**: Analysis methods like representation alignment, attention visualization, and modality ablation can indicate "which modality is encoded" and "how much performance drops when a modality is removed." However, they fail to address **decision-level** questions: Is the information utilized by the model unique to one modality, shared between two (redundant), or accessible only when both modalities are present (synergistic)? These three modes are conflated at the accuracy level, masking different multimodal fusion patterns.

**Key Challenge**: Accuracy and ablation metrics are **scalars**, whereas modality usage is **multidimensional** (unique vs. redundant vs. synergistic). Compressing this multidimensional structure into a scalar inevitably loses critical signals, such as whether a model is performing true fusion or taking shortcuts via linguistic priors.

**Goal**: To accomplish three objectives: (a) Construct a decision-level "modality usage profile" for each model-benchmark pair; (b) Verify if this profile predicts intervention sensitivity (performance drop after removing vision/audio); (c) Use the profile to guide training and enhance genuine cross-modal fusion.

**Key Insight**: The authors adopt the established PID framework from information theory, which decomposes $I(Y;X_v,X_t)$ into four non-negative components: $U_{\text{vis}} + U_{\text{txt}} + R_{\text{vl}} + S_{\text{vl}}$. A crucial observation is that **PID should be constructed on the model-induced predictive distribution $p_\theta(y|x_v,x_t)$ rather than latent representations**, so that the result reflects "how the model uses modalities" rather than "the inherent structure of the dataset."

**Core Idea**: Use decision-level PID for VL model diagnostics, introduce Sensory PID (treating text as a conditional control variable) for video-audio-text omni-modal models, and finally construct a LoRA reweighting strategy that "up-weights synergy-deficient samples and down-weights language-shortcut samples" using sample-level PID scores.

## Method

### Overall Architecture

The methodology consists of three interconnected parts:

1.  **Bimodal PID (§2.1)**: For VL models, decompose the joint mutual information between prediction $Y$ (distribution over multiple-choice candidates $\mathcal{C}$) and vision/text sources $X_v, X_t$ into $U_{\text{vis}} + U_{\text{txt}} + R_{\text{vl}} + S_{\text{vl}}$ to obtain a model-benchmark profile.
2.  **Sensory PID (§2.2)**: For omni-modal models (video $V$ + audio $A$ + text $T$), instead of a full 3-source PID (which faces exponential explosion and blurs the instruction role of language), $T$ is treated as a conditional control signal. This decomposes $I(Y;V,A|T) = U_{\text{vis}} + U_{\text{aud}} + R_{\text{sens}} + S_{\text{av}}$, mathematically separating "task instructions" from "sensory evidence."
3.  **Estimation Pipeline (§2.3)**: Since $X_v, X_t$ are high-dimensional continuous vectors and MLLMs are jointly trained (lacking standalone unimodal branches), the authors implement PID estimation via three engineering steps: BATCH estimator + Calibrated Embedding Masking + Output Stabilization.

Downstream: Use the **sample-level** local contribution scores produced by BATCH (additive contributions to $S_{\text{vl}}, U_{\text{txt}}, U_{\text{vis}}, R_{\text{vl}}$ for each sample) to define Synergy Ratio, Shortcut Score, Fusion Potential (FP), and GapScore, which serve as sample weights for LoRA fine-tuning.

### Key Designs

1.  **Sensory PID: Treating Language as a Condition Rather than a Source**:
    *   **Function**: Decomposes the sensory information gain of omni-modal decisions as $I(Y;V,A|T) = U_{\text{vis}} + U_{\text{aud}} + R_{\text{sens}} + S_{\text{av}}$.
    *   **Mechanism**: A full 3-source PID generates an exponential number of partial information atoms (the full lattice of $U, R, S$), which is uninterpretable and hard to estimate. The authors observe that in instructional contexts, text serves as a "task specification," essentially a control variable. By fixing $T$ as a condition, they perform bimodal PID on sensory sources. The conditional decomposition still satisfies $\sum (\cdot) = I(Y;V,A|T)$, but reduces parameters from exponential to 4 atoms.
    *   **Design Motivation**: To separate "what the task specifies" from "what evidence the senses provide." This gives physical meaning to comparable quantities like "audio-unique information" $U_{\text{aud}}$, "vision-unique information" $U_{\text{vis}}$, and "audio-visual synergy" $S_{\text{av}}$—enabling quantitative observation of "vision dominance."

2.  **BATCH + Calibrated Embedding Masking: Creating Unimodal Conditional Distributions from Joint Models**:
    *   **Function**: Provides $p_\theta(y|x_v)$ and $p_\theta(y|x_t)$ for PID estimation without retraining the MLLM.
    *   **Mechanism**: The BATCH estimator (Liang et al., 2023) learns a Sinkhorn-normalized coupling $\tilde{Q}$ that maintains marginal matches of $X_v\text{–}Y$ and $X_t\text{–}Y$ to the true distribution; synergy is given by the gap between true joint MI and $\min_{Q\in\Delta_P} I_Q$. The challenge lies in unimodal conditional distributions, as MLLMs lack independent vision/text heads. Ours uses **Calibrated Embedding Masking**: to compute $p_\theta(y|x_v)$, text token embeddings are replaced with Gaussian noise $\mathcal{N}(\mu_{m'}, \mathrm{diag}(\sigma_{m'}^2))$, where $\mu_{m'}, \sigma_{m'}$ are dimension-wise statistics of the modality across the profiling set. This erases specific semantic info while preserving the distribution shape expected by the backbone.
    *   **Design Motivation**: Using empty strings or zero masks pushes the backbone out of the training distribution, leading to unreliable predictions. Noise matching the statistics is equivalent to "blurring the modality without destroying its position," providing the cleanest unimodal approximation without retraining.

3.  **PID-Guided LoRA Reweighting**:
    *   **Function**: Uses sample-level PID scores to up-weight "should-fuse-but-didn't" samples and down-weight "language shortcut" samples during LoRA fine-tuning to force synergistic information usage.
    *   **Mechanism**: BATCH yields local contributions $s_i, u_{\text{vis},i}, u_{\text{txt},i}, r_i$ for each sample $i$. Non-negative truncation $[\cdot]_+$ gives sample info quality $I_i^+$, then Synergy Ratio $\text{SR}_i = [s_i]_+/(I_i^+ + \epsilon)$ and Shortcut Score $\text{SC}_i = [u_{\text{txt},i}]_+/(I_i^+ + \epsilon)$. Fusion Potential is defined as $\text{FP}_i = [\min\{H(p_v^{(i)}), H(p_t^{(i)})\} - H(p_{vt}^{(i)})]_+$ (degree to which joint prediction is more certain than either unimodal prediction). Synthesis gives $\text{GapScore}_i = (1-\text{SR}_i)(1-\text{SC}_i)\cdot \text{FP}_i$, selecting samples with high FP but low synergy/language usage. TopK "shortcut samples" and "gap samples" are assigned weights $w_i = 0.5$ and $w_i = 3.0$ (others $w_i = 1.0$) for weighted LoRA fine-tuning.
    *   **Design Motivation**: Traditional hard-mining or ablation-based sampling cannot distinguish whether a sample is difficult due to "lack of knowledge" or "lack of fusion." PID separates these at the decision level, allowing directed intervention—a key step in turning diagnostics into training tools.

### Loss & Training

No training occurs during the diagnostic phase; BATCH simply runs PID estimation. During training, the standard LoRA objective is used, with each sample's loss multiplied by $w_i$. LoRA adapters are placed only in the last 20% of layers (based on §4.3 layer-wise analysis showing synergy emerges late). All PID estimates are averaged over $K=50$ random batch samples to reduce variance. Confidence threshold $\tau=0.3$, up-weight factor $3.0$, and down-weight factor $0.5$ are fixed hyperparameters.

## Key Experimental Results

### Main Results

Evaluations covered 20 VL models (Qwen2.5/2/3-VL, InternVL3, LLaVA-OneVision, Cambrian-1, Gemma3, 2B–78B) × 6 VL benchmarks (MMStar/MMBench/POPE as "synergy-driven", MMMU/PMC-VQA/Reefknot as "prior-driven"), plus omni-modal Qwen2.5-Omni, VITA-1.5 × MUSIC-AVQA (Audio/Visual/AV-Fusion subsets).

| Validation Dimension | Key Metric | Result | Meaning |
| :--- | :--- | :--- | :--- |
| Corellation of PID terms with vision removal sensitivity $\Delta_{\text{vision}}$ (Synergy-driven tasks) | Spearman $\rho(S_{\text{vl}}, \Delta_{\text{vision}})$ | MMBench 0.840 / MMStar 0.862 / POPE 0.798 ($p<0.001$) | $S_{\text{vl}}$ is the strongest single predictor of vision sensitivity |
| Same as above, for $U_{\text{txt}}$ | $\rho(U_{\text{txt}}, \Delta_{\text{vision}})$ | $-0.582 / -0.548 / -0.502$ | Stronger text-unique info makes models less sensitive to vision removal |
| Joint MI $I(V,T;Y)$ vs. $\Delta_{\text{vision}}$ | $|\rho| \le 0.118$ | Nearly zero correlation | **Total MI follows accuracy, not modality dependence** — Proves PID reveals signals accuracy cannot |
| Sensory Synergy $S_{\text{av}}$ on AV-Fusion subset | Value | All models $\le 0.32$, far less than $U_{\text{vis}} \approx 1.25\text{–}1.42$ | Models are dominated by vision-unique info even when fusion is required → "Vision dominance + Synergy bottleneck" |
| LoRA-PID vs. LoRA-Uniform (Qwen2.5-VL-7B) | MMStar / MMBench / POPE | $64.3$ vs $62.0$ / $90.2$ vs $89.1$ / $88.5$ vs $87.2$ | +2.3 / +1.1 / +1.3 pp, stable across 3 seeds |
| PID profile shift after fine-tuning | Post-$S_{\text{vl}}$ / Post-$U_{\text{txt}}$ | $1.20\to 1.36$ / $0.56\to 0.46$, synergy share $67.5\%\to 73.9\%$ | LoRA-PID effectively shifts models toward synergy and away from shortcuts |

### Ablation Study

| Configuration | MMStar | Description |
| :--- | :--- | :--- |
| B: LoRA-Uniform | 62.0 | Uniform weighting baseline |
| C: LoRA-PID | **64.3** | Full PID sampling + reweighting |
| D: LoRA-Random (same 0.5/3.0 weights, random assignment) | 61.5 | Distribution is not the key, **which samples receive weights is** |
| E: LoRA-Acc (sampling by accuracy difficulty) | 62.5 | Difficulty $\neq$ fusion need; PID outperforms difficulty mining by +1.8 |
| F: LoRA-Ablation (sampling by ablation sensitivity) | 63.0 | Ablation sensitivity captures some fusion needs but remains 1.3 pp weaker |
| Prior-dominant tasks (MMMU/PMC-VQA) | $-0.5 / -0.3$ vs Uniform | LoRA-PID **intentionally** down-weights language shortcuts, causing a slight trade-off in pure-prior tasks |

### Key Findings

*   **Synergy $S_{\text{vl}}$ is the watershed signal**: On all synergy-driven benchmarks, it achieves both $\rho(\cdot, \Delta_{\text{vision}}) \ge 0.798$ and $\rho(\cdot, \text{Acc}) \ge 0.718$; it is the strongest predictor of "whether a model uses vision." Joint MI only predicts accuracy and shows zero correlation with intervention response.
*   **Three-stage Hierarchical Dynamics**: Layer-wise PID shows VL models follow a "Silent Encoding (0–20%) → Unimodal Accumulation (20–80%) → Late Fusion (80–100%)" pattern. Synergy emerges almost exclusively in the final 20% of layers, justifying the LoRA placement decision.
*   **Mechanism of Vision Dominance**: Omni-modal models reach "vision saturation" in middle layers ($U_{\text{vis}}$ rises sharply and dominates the decision space). By the time fusion is attempted, the decision boundary is already fixed by vision priors.
*   **Language as Fusion Gating**: Replacing a fusion-requiring instruction (e.g., "Is the violin playing high notes?") with a paraphrase lacking fusion needs (e.g., "Which instrument is playing?") results in a significant decay of $S_{\text{av}}$ in late stages, while unimodal trajectories in early/middle stages remain unchanged. This suggests text acts as a control signal for "turning on" fusion.
*   **POPE vs. Reefknot Case**: Both are billed as "hallucination benchmarks," but PID classifies POPE as synergy-driven and Reefknot as prior-driven, suggesting that literal labels in benchmarks can mask true modality usage differences.

## Highlights & Insights

*   **Decision-level vs. Representation-level Watershed**: While previous MLLM interpretability work looked for signals in latent representations (CKA, attention maps, probing), this work returns to $p_\theta(y|x)$. PID describes "how the model uses modalities to reach an answer" rather than "how modalities are encoded." Ours provides a clean engineering solution via BATCH + calibrated masking to separate these concepts.
*   **PID as both Diagnostic and Training Signal**: The **sample-level** scores produced by BATCH elevate the tool from post-hoc analysis to proactive sampling, creating a "diagnosis → prediction → intervention" loop. This philosophy of using interpretability byproducts to guide training is transferable to any tool providing sample-level decomposition.
*   **Sensory PID as a Simple yet Effective Innovation**: Reducing 3-source PID to "conditional language + bimodal sensory" solves the explosion of atoms and prevents the instruction role of language from being conflated with $U_{\text{txt}}$. This is a reusable framing for omni-modal analysis.
*   **Multiplicative GapScore Structure**: The $(1-\text{SR})(1-\text{SC})\cdot \text{FP}$ structure requires "not yet synergistic," "not using shortcuts," AND "high potential for joint certainty." This "intersection of three conditions" approach is an effective way to identify improvable samples.

## Limitations & Future Work

*   **Reliance on BATCH Estimation Accuracy**: BATCH is essentially a Sinkhorn optimization that pools (mean pooling) high-dimensional continuous representations. Whether this loses critical token information is only explored via sensitivity analysis in the appendix.
*   **Boundary of Masking Approximation**: Calibrated embedding masking assumes the backbone reacts to "statistically matched Gaussian noise" identically to "true modality absence." This holds for pixel/audio domains but may fail for **structured instruction templates** (e.g., code, math) with strong priors.
*   **Upper Bound of PID Reweighting**: The gain over LoRA-Ablation is only +1.3 pp on MMStar, suggesting "modality sensitivity" and "synergy needs" overlap significantly. PID provides a fine-grained refinement rather than a transformation.
*   **Lack of Dual Audio Experiments**: While the paper identifies "vision dominance," it lacks a "specifically reinforced audio" LoRA control (e.g., weighting audio-unique samples) to determine how much of the "synergy bottleneck" can be overcome by reweighting alone.
*   **Cost on Prior-driven Tasks**: LoRA-PID shows a 0.3–0.5 pp drop on MMMU/PMC-VQA. If deployment requires knowledge-heavy tasks, benchmark-specific weight mixing might be necessary.

## Related Work & Insights

*   **vs. Representation Alignment / CKA / Attention Probing**: Representation methods describe "what modalities are encoded as," while Ours describes "how modalities are used." The former is a representational signature, the latter a functional signature; they are not equivalent.
*   **vs. Modality Ablation (Modality Dropout)**: Ablation provides $\Delta_{\text{vision}}$ but cannot separate unique from synergistic components. Ours explicitly dissociates these dependencies by contrasting $S_{\text{vl}}$ and $U_{\text{txt}}$.
*   **vs. Liang et al. 2023 (BATCH)**: BATCH is a PID estimator previously used for scalar labels in supervised learning. Ours ports BATCH to MLLM predictive distributions (logits over candidate set $\mathcal{C}$) and adds masking for unimodal conditions—a valuable extension for generative multimodal contexts.
*   **vs. Full 3-source PID (Williams & Beer)**: Original PID atoms grow exponentially with sources. Sensory PID uses conditioning to reduce the dimension to 4, providing a **practical formula for omni-modal models** and bridge the gap from theory to engineering.

## Rating
*   Novelty: ⭐⭐⭐⭐ Moving PID to the MLLM decision level and proposing Sensory PID conditional decomposition is a clean and original framing.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 models × 6 VL benchmarks + 3 omni models × 3 subsets, plus 5 independent validation dimensions. Exceptional for an analysis-heavy paper.
*   Writing Quality: ⭐⭐⭐⭐ The 9 Findings structure the paper well, though technical details on BATCH/masking/Sinkhorn are heavily relegated to the Appendix, requiring an information theory background.
*   Value: ⭐⭐⭐⭐ The diagnosis-to-training loop is a great reference for the MLLM evaluation/tuning community; LoRA-PID provides stable gains and serves as a model for interpretability-driven training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[CVPR 2026\] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](../../CVPR2026/audio_speech/omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)

</div>

<!-- RELATED:END -->
