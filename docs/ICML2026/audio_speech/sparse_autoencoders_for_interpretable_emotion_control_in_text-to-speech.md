---
title: >-
  [Paper Note] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech
description: >-
  [ICML 2026][Audio & Speech][Sparse Autoencoders] The authors train a Top-k Sparse Autoencoder (SAE) on the residual stream of the semantic backbone of an LLM-based TTS (IndexTTS2). By using "sentence-level activation rat…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Sparse Autoencoders"
  - "Emotion-controllable TTS"
  - "Activation Steering"
  - "Semantic Backbone"
  - "Interpretability"
date: 2026-05-08
content_hash: a8fae06376a8dc50
---

# Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech

**Conference**: ICML 2026  
**arXiv**: [2606.01479](https://arxiv.org/abs/2606.01479)  
**Code**: Available (GitHub-Demo link)  
**Area**: Audio/Speech (TTS Controllable Generation + Interpretability)  
**Keywords**: Sparse Autoencoders, Emotion-controllable TTS, Activation Steering, Semantic Backbone, Interpretability

## TL;DR
The authors train a Top-k Sparse Autoencoder (SAE) on the residual stream of the semantic backbone of an LLM-based TTS (IndexTTS2). By using "sentence-level activation rate difference" to identify a small subset of sparse latent features strongly correlated with target emotions, they achieve interpretable bidirectional emotion induction and suppression during inference by intervening only on these features. This method avoids modifying backbone parameters and outperforms global mean-difference steering and existing TTS baselines.

## Background & Motivation
**Background**: LLM-based TTS models (e.g., CosyVoice, Spark-TTS, IndexTTS2) have achieved highly natural and expressive speech synthesis by embedding large language models into the generation pipeline. Emotion controllability has become a critical capability for applications like audiobooks and HCI.

**Limitations of Prior Work**: Existing emotional TTS methods typically follow two paths. First, label/prompt conditioning (e.g., EmoVoice, VALL-E-X), which models emotions as discrete classes, **smoothing out fine-grained differences and lacking continuous intensity adjustment**. Second, reference audio transfer (e.g., Spark-TTS), which performs well but **requires finding suitable references and is not adjustable**. Neither approach is inherently interpretable. Recently, Xie et al. (2025) introduced activation steering for training-free emotion control, but they applied global translation using a dense mean-difference direction in the **acoustic stage of a Diffusion Transformer**.

**Key Challenge**: Emotional expression is a multi-dimensional phenomenon coordinated by **multiple acoustic factors** such as pitch, energy, and prosody. Using a single dense global direction only captures overall shifts, **losing the interpretability of "which dimension is responsible for what" and limiting modular control**. Furthermore, prior works focus on downstream acoustic modules, leaving how emotion is encoded within the **autoregressive semantic backbone** under-researched.

**Goal**: (1) Determine whether emotion-related changes can be decomposed into sparse latent features within the semantic backbone; (2) Identify small subsets aligned with specific emotions; (3) Perform bidirectional (induction/suppression) emotion control using these features.

**Key Insight**: The authors transfer SAEs—a tool proven in the LLM interpretability community for decomposing "polysemantic" neurons into "sparse monosemantic" features—to the residual stream of a TTS semantic backbone. This shifts emotion analysis forward from the "acoustic implementation layer" to the "semantic generation layer."

**Core Idea**: Use a Top-k SAE to decompose semantic backbone residual activations into a 4096-dimensional sparse dictionary. Select the top-$m$ emotional features based on their "sentence-level frequency in target emotion samples vs. paired neutral samples." During inference, shifting their activation values by $+\alpha$ or $-\alpha$ guides emotion along a bundle of interpretable sparse directions.

## Method

### Overall Architecture
The pipeline keeps IndexTTS2 intact (Text/Ref Audio → Autoregressive Semantic Backbone → Semantic Tokens → CFM + Vocoder → Waveform). The SAE is a **bypass module attached to the pre-LayerNorm residual stream of the 16th layer of the semantic backbone**. During training, a 1280→4096→1280 Top-k autoencoder is learned using only the hidden states from the decoding phase. During inference, the residual $x_{l,t}$ at each step is intercepted, encoded into sparse latents $z$, modified for the pre-selected emotional feature set $\mathcal{F}_e$, and decoded back into the residual space for the downstream CFM. The entire pipeline **does not change TTS backbone weights**. The SAE contains only ~10.5M parameters (~40MB in fp32) and is trained offline once (30k steps on a single H100).

### Key Designs

1.  **Top-k SAE + Auxiliary Loss for Dead Features**:
    *   **Function**: Decomposes the dense semantic backbone residual $x \in \mathbb{R}^d$ into a sparse representation $z$ in an $n=4096$ over-complete dictionary with only $k=32$ active values. Each decoder column $d_j$ corresponds to a semantic direction in the residual space.
    *   **Mechanism**: The encoder $z=\mathrm{Top}_k(\mathrm{ReLU}(W_{\text{enc}}(x-b_{\text{pre}})+b_{\text{enc}}))$ uses the Top-k operator to keep the largest 32 activations and suppress others, forcing features to compete to explain the input. The decoder $\hat{x}=W_{\text{dec}}z + b_{\text{pre}}$ performs linear reconstruction with the primary loss $\mathcal{L}_{\text{rec}}=\|x-\hat{x}\|_2^2$. To mitigate the "dead feature" problem common in over-complete SAEs, the authors select inactive latents $\tilde{z}$ to fit the reconstruction residual $(x-\hat{x})$, yielding $\mathcal{L}_{\text{aux}}=\|(x-\hat{x})-W_{\text{dec}}\tilde{z}\|_2^2$. Total loss: $\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda_{\text{aux}}\mathcal{L}_{\text{aux}}$.
    *   **Design Motivation**: Token-level sparsity and dictionary-level competition are key to breaking down "polysemantic neurons" into "monosemantic directions." The auxiliary loss forces dormant latents to capture residuals not explained by the main path, improving dictionary utilization and reducing noise in subsequent feature selection.

2.  **Sentence-level Activation Rate Difference (Δ-score) as Emotion Selectivity**:
    *   **Function**: Selects the top-$m$ candidates (experimentally $m=6$) truly correlated with a specific emotion (anger, happiness, or sadness) from the 4096 SAE features.
    *   **Mechanism**: Fixed text and speaker prompts are used to generate paired emotional and neutral samples. Token-level activations are collapsed into binary sentence-level indicators $\mathbf{1}_i^{(e)}(u)=\mathbb{1}[\exists t,\ a_{i,t}^{(e)}(u)>0]$. The sentence-level trigger rate $r_i^{(e)}$ is calculated, and the pairing difference $\Delta_i^{(e)}=\frac{1}{|\mathcal{D}|}\sum_u (\mathbf{1}_i^{(e)}(u)-\mathbf{1}_i^{(\text{neutral})}(u))$ serves as the selectivity score. Top-6 features for "anger" reach $\Delta=1$, meaning they trigger in all anger samples but never in paired neutral samples.
    *   **Design Motivation**: Using sentence-level frequency instead of peak magnitude avoids dominance by transient noise, as emotional expression **persists through the whole sentence**. Using the paired difference naturally filters out "globally frequent but emotion-irrelevant" features. This construction is crucial for the interpretability and stability of subsequent interventions.

3.  **Sparse Latent Feature Steering (Bidirectional Induction/Suppression)**:
    *   **Function**: During inference, the semantic backbone residuals are shifted along a bundle of sparse, interpretable directions to induce or suppress target emotions with continuously adjustable intensity.
    *   **Mechanism**: Based on the linear reconstruction approximation $x \approx b_{\text{pre}}+\sum_j a_j(x)\,d_j$, the authors modify the active values for $j\in\mathcal{F}_e$ as $a_j^{\text{new}}=a_j+\alpha_e$. This is equivalent to a residual intervention $x_{l,t}^{\text{new}}\approx x_{l,t}+\alpha_e \sum_{j\in\mathcal{F}_e} d_j$. $\alpha_e>0$ induces emotion, while $\alpha_e<0$ suppresses it. All tokens share the same $\mathcal{F}_e$ and $\alpha_e$ (time-invariant), which can be generalized to token-dependent $\alpha_e$. The final "combined direction" is the sum of the top-6 decoder directions.
    *   **Design Motivation**: Compared to Global Steering (dense mean difference), shifting along $\sum_{j\in\mathcal{F}_e} d_j$ explicitly decomposes "global emotional shift" into several sparse directions, each **corresponding to observable acoustic properties** (e.g., pitch, energy, spectral brightness). This enables modular control (e.g., intervening only on feature #24 significantly raises F0 by +23.11 Hz and RMS energy by +0.00435 without affecting duration).

### Loss & Training
The SAE training loss is $\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda_{\text{aux}}\mathcal{L}_{\text{aux}}$, with decoder columns constrained to unit norm. Training data consists of layer-16 decoding-phase residuals from 56,000 emotion-controlled TTS generated samples (7 emotions × 20 timbres × 400 texts). Training takes 30k steps on a single H100. Selectivity analysis uses 43,408 matched text-speaker samples.

## Key Experimental Results

### Main Results: Bidirectional Emotion Steering (Table 1, Selection of 3 Emotions × 9 Metrics)

| Setting | Emotion | Metric | VALL-E-X | Spark-TTS | EmoVoice | CosyVoice | Random SAE | Global Steering | **SAE-Emotion (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Induction (Neutral→Anger) | Anger | Emo-SIM↑ | 0.831 | 0.857 | 0.806 | 0.813 | 0.892 | 0.910 | **0.912** |
| Induction (Neutral→Anger) | Anger | WER↓ | 3.1 | 2.7 | 4.1 | 3.9 | 1.4 | **0.1** | 0.3 |
| Induction (Neutral→Happiness) | Happiness | Emo-SIM↑ | 0.697 | 0.770 | 0.728 | 0.712 | 0.813 | 0.879 | **0.885** |
| Induction (Neutral→Sadness) | Sadness | Emo-SIM↑ | 0.869 | 0.907 | 0.850 | 0.799 | 0.858 | 0.876 | **0.880** |
| Suppression (Anger→Neutral) | Anger | Emo-SIM(vs Neutral)↑ | – | – | – | – | 0.841 | 0.915 | **0.939** |
| Suppression (Happiness→Neutral) | Happiness | Emo-SIM↑ | – | – | – | – | 0.886 | 0.920 | **0.924** |
| Suppression (Sadness→Neutral) | Sadness | Emo-SIM↑ | – | – | – | – | 0.939 | 0.933 | **0.941** |

For induction, Emo-SIM for all three emotions is the highest or tied for first, with WER generally $\leq 0.3$. For suppression, Emo-SIM is also the highest across the board, verifying that **the same set of sparse features supports bidirectional control**.

### Ablation Study: User Evaluation (Table 2 + Selectivity vs. Random)

| Configuration | EMOS↑ | NMOS↑ | Description |
| :--- | :--- | :--- | :--- |
| **SAE-Emotion (top-6 by Δ)** | **3.22** | **3.49** | Full method |
| Global Steering (Dense Mean Diff) | 3.10 | 3.38 | Removed sparse decomposition; reverted to a global direction |
| Random SAE (Random 6 features) | 1.82 | 3.22 | Removed $\Delta$-score selection; verifies selectivity importance |

In blind listening tests with 20 raters (0–5 scale), SAE-Emotion significantly outperformed Global Steering and Random SAE in perceived emotional accuracy (EMOS). NMOS was also highest, suggesting sparse decomposition preserves naturalness while providing more precise emotional perception.

### Key Findings
*   **Emotion is sparsely organized in the SAE dictionary**: The selectivity distribution for anger vs. neutral is centered at 0 with a thin positive tail forming an emotional cluster; happiness and sadness show similar heavy tails.
*   **Individual latents map to observable acoustic properties**: Intervening on Latent #24 significantly raises F0 (+23.11 Hz, $p=1.07e-4$), RMS energy, and spectral centroid while keeping duration constant, suggesting SAE decouples factors like pitch and brightness.
*   **Selectivity > Randomness**: Random SAE's EMOS (1.82) is much lower than SAE-Emotion (3.22), proving performance stems from "choosing the right features" rather than "sparsity itself."
*   **Sparse Decomposition > Dense Global**: SAE-Emotion Emo-SIM $\geq$ Global Steering across all induction/suppression tasks with EMOS +0.12, showing decomposition into interpretable directions yields real gains.
*   **Calibratable Continuous Strength Control**: Scanning $\alpha \in \{-60, 0, +60\}$ for the rank-1 happiness feature shows samples moving continuously toward the target emotion prototype, suggesting $\alpha$ acts as a physical "emotion intensity knob."

## Highlights & Insights
*   **Transferring SAE from LLM Interpretability to TTS Semantic Backbones**: While previous applications of SAE in TTS focused on acoustic autoencoders (analyzing pitch/timbre), this work moves interpretability to the autoregressive LLM residual stream. This elevates the methodology from "acoustic implementation" to "high-level semantic/emotional structure."
*   **Sentence-level Activation Rate Difference is a simple but critical trick**: Converting "per-token peaks" to "per-sentence frequency differences" aligns with the intuition that "emotion spans the entire sentence." It also naturally filters out persistent high-frequency features.
*   **Sparse Direction as a Plug-and-Play Control Interface**: The form $\alpha_e \sum_{j\in\mathcal{F}_e} d_j$ packages "bidirectional, continuous, interpretable, and backbone-agnostic" control into a bundle of additive residual perturbations. This instantiates the "activation steering knob" model for TTS.

## Limitations & Future Work
*   The authors acknowledge that SAE training requires significant compute and storage for activations. The analysis was conducted on a single backbone (IndexTTS2), with cross-architecture generalization only validated on a small scale in the appendix.
*   Personal observations: (1) $m$ and $\alpha$ are determined empirically per emotion ($m=6, \alpha \in [-60, 60]$); a principled calibration for feature count and intensity is missing. (2) The experiment only covers three emotions; results for disgust, fear, and surprise from the 7-emotion dataset are missing from the main results. (3) Sentence-level rates flatten temporal structures, which may lose information for "intra-sentence emotion switching."

## Related Work & Insights
*   **vs. Global Steering / Activation Engineering (Xie et al. 2025)**: They perform dense mean-difference steering in the acoustic Diffusion Transformer stage; this work uses sparse decomposition + multi-directional guidance in the semantic backbone. Advantages include interpretability and more stable bidirectional control; the disadvantage is the need to train an SAE offline.
*   **vs. Label/Prompt-based TTS (EmoVoice, VALL-E-X)**: Those models are discrete and often not adjustable; this work provides a continuous $\alpha$ knob without modifying the backbone (though it requires access to intermediate activations).
*   **vs. Acoustic-SAE Work (Paek et al. 2025)**: They used SAE on acoustic layers to interpret pitch/timbre; this work moves up to the semantic backbone to focus on "high-level emotion." These paths are complementary.

## Rating
*   Novelty: ⭐⭐⭐⭐ First use of Top-k SAE on LLM-based TTS autoregressive semantic backbones with $\Delta$-score selection.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across 4 TTS baselines, 2 steering baselines, and human evaluation. Needs more focus on other emotions in the main tables.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, well-supported by figures/tables.
*   Value: ⭐⭐⭐⭐ Provides a plug-and-play, interpretable emotional control interface for LLM-based TTS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ICLR 2026\] Scaling Speech Tokenizers with Diffusion Autoencoders](../../ICLR2026/audio_speech/scaling_speech_tokenizers_with_diffusion_autoencoders.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[AAAI 2026\] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment](../../AAAI2026/audio_speech/text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[ACL 2026\] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](../../ACL2026/audio_speech/speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)

</div>

<!-- RELATED:END -->
