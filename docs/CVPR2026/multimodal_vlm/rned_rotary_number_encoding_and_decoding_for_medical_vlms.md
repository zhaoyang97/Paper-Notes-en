---
title: >-
  [Paper Note] RNED: Rotary Number Encoding and Decoding for Medical VLMs
description: >-
  [CVPR 2026][Multimodal VLM][Medical VLM] To address the inherent weakness of medical VLMs in "numerical prediction," this paper proposes RNED: the encoding side follows the RoPE paradigm by using a "value-dependent rotary matrix" to rotate a scalar into a dedicated `[NUM]` token (norm-preserving, order-preserving, wide range), while the decoding side employs score-matching to retrieve continuous values from hidden states. It consistently outperforms existing VLM baselines on…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Medical VLM"
  - "Numerical Encoding"
  - "Rotary Encoding"
  - "Score-Matching"
  - "Quantitative Prediction"
date: 2026-05-08
content_hash: 90109890a76cea8d
---

# RNED: Rotary Number Encoding and Decoding for Medical VLMs

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_RNED_Rotary_Number_Encoding_and_Decoding_for_Medical_VLMs_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Medical VLM, Numerical Encoding, Rotary Encoding, Score-Matching, Quantitative Prediction

## TL;DR
To address the inherent weakness of medical VLMs in "numerical prediction," this paper proposes RNED: the encoding side follows the RoPE paradigm by using a "value-dependent rotary matrix" to rotate a scalar into a dedicated `[NUM]` token (norm-preserving, order-preserving, wide range), while the decoding side employs score-matching to retrieve continuous values from hidden states. It consistently outperforms existing VLM baselines on radiology measurement estimation and medical visual grounding tasks.

## Background & Motivation
**Background**: Medical VLMs (e.g., LLaVA-Med, BiomedGPT, CT-CHAT) are being utilized to interpret CT/MRI/Ultrasound images and answer clinical questions. However, most are optimized for classification and text generation.

**Limitations of Prior Work**: Clinical practice is essentially **quantitative**—lesion size, aortic diameter, ejection fraction, and bounding-box coordinates are all numerical values that must be reliable enough to support diagnostic decisions. Yet, underlying LLMs are weak with numbers: standard tokenizers split `42.5` into `[4][.][5]` (Llama3) or `[4][2][.][5]` (Mistral), preventing the model from treating it as a single numerical entity. Furthermore, the cross-entropy loss for next-token prediction treats "predicting 3 instead of 4" and "predicting 9 instead of 4" as equally erroneous, failing to penalize the magnitude of numerical deviation. Consequently, models often "hallucinate" plausible-looking numbers based on memory and corpus statistics, which is dangerous in medicine.

**Key Challenge**: Integrating continuous values into a pre-trained LLM designed for discrete tokens (and featuring LayerNorm/RMSNorm) must simultaneously satisfy three conflicting requirements: C1 (wide range and order-preserving encoding), C2 (noise-robust decoding), and C3 (normalization invariance) to avoid degrading the original model. Existing methods like xVal use a number head for regression, but this alters the norm of the `[NUM]` token, conflicting with LayerNorm and dragging down text generation quality.

**Goal**: Formally define the properties that "medical VLM numerical representations" should satisfy (C1, C2, C3), and design an encoding-decoding scheme that fulfills all three.

**Key Insight**: The authors observe that RoPE injects position via "rotation" and that rotation is **norm-preserving**. Since rotation can encode a continuous variable without changing the vector length, can "value-dependent rotation" be used to encode the numbers themselves?

**Core Idea**: Use a "numerical-specific rotary matrix" to encode scalars into a single `[NUM]` token (RNE), and transform "value recovery from hidden states" into a noise-robust score-matching lookup problem (RND).

## Method

### Overall Architecture
RNED is grafted onto standard VLMs (e.g., LLaVA / Qwen2.5-VL): images are converted into visual tokens via a vision encoder + projector. Each number in the text is replaced by a special `[NUM]` token. **RNE** rotates the base embedding of `[NUM]` using a rotary matrix corresponding to its value. These are then concatenated with visual and text tokens and fed into the LLM. The LLM performs standard autoregressive next-token prediction; when the language model head predicts a `[NUM]` token, its output embedding $\hat{x}_i$ bypasses the vocabulary and is passed to the **RND** score-matching decoder to retrieve a continuous value $\hat{m}$ from a candidate set. This value is re-encoded for the next input step, continuing autoregression until `[EOS]`. Only the LLM (LoRA) and projector are trained; the vision encoder is frozen, and RNE/RND themselves introduce no additional trainable parameters.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["CT Image + Text Question with Numbers"] --> B["Vision Encoder + Projector<br/>Visual Tokens"]
    A --> C["Rotary Number Encoding (RNE)<br/>Rotates each value into a single [NUM] token"]
    B --> D["LLM (LoRA Fine-tuning)<br/>Unified autoregression for visual/text/numerical tokens"]
    C --> D
    D -->|Predicts standard token| E["Text Output"]
    D -->|Predicts [NUM]| F["Rotary Number Decoding (RND)<br/>Score-matching to retrieve continuous value m̂"]
    F -->|m̂ re-encoded to input| D
    F --> G["Numerical Output (e.g., 41.5 mm)"]
```

### Key Designs

**1. Rotary Number Encoding (RNE): Encoding a scalar as a single token via norm-preserving rotation**

This addresses the pain point of numbers being split into multiple sub-words. RNE represents each numerical value $m$ as one token: taking the base embedding $x_{[NUM]}$, it is rotated by a **value-dependent rotary matrix** $R_d(m\Omega)$, such that $G_{enc}(x_{[NUM]}, m) = R_d(m\Omega)\,x_{[NUM]}$. In a 2D intuition: using $R_2(m\omega_0)=\begin{bmatrix}\cos(m\omega_0) & -\sin(m\omega_0)\\ \sin(m\omega_0) & \cos(m\omega_0)\end{bmatrix}$ to rotate query and key simultaneously simplifies the attention dot product to $q^{m_1}\cdot k^{m_2}=f_q(x_{[NUM]})^{T} R_2((m_2-m_1)\omega_0) f_k(x_{[NUM]})$. This means the **difference** $m_2-m_1$ is naturally injected into the attention score (utilizing $R_2(\theta)^T=R_2(-\theta)$ and $R_2(\theta_1)R_2(\theta_2)=R_2(\theta_1+\theta_2)$). For pure text tokens, setting $m=0$ results in no rotation, leaving text representations undisturbed. This directly satisfies C1 (order-preserving), while the rotation $\lVert R_2(m\omega_0)x\rVert=\lVert x\rVert$ preserves the norm. It encodes numerical values through direction rather than magnitude, without disrupting the dot-product scales of the pre-trained model, satisfying C3.

The issue with a single frequency $\omega_0$ is **periodicity**: different values of $m$ would map to the same representation once the angle completes a circle, failing to cover a wide range. Thus, it is generalized to high dimensions: the $d$-dimensional space is split into $d/2$ two-dimensional subspaces, each equipped with its own **frequency** block to form a block-diagonal matrix. Frequencies are chosen as $\Omega=\{\omega_j = B^{-2j/d}\mid j=0,\dots,\tfrac{d}{2}-1\}$ with $B=5\times10^5$. Although each 2D block remains periodic, the overall rotation only repeats when the periods $2\pi/\omega_j$ of all blocks **align simultaneously**, which rarely occurs for different values of $m$. The multi-frequency construction thus provides unique representations over a very broad range, satisfying the "wide range" requirement of C1.

**2. Rotary Number Decoding (RND): Framed as noise-robust score-matching lookup**

While encoding is closed-form, recovering $m$ from the LLM's **output** embedding $\hat{x}_i$ is tricky. Theoretically, one could estimate the relative rotation angle $\theta_j$ for each 2D block using `atan2`, yielding a set of linear congruences $m\omega_j = \theta_j + 2\pi k_j$ ($k_j\in\mathbb{Z}$). However, $\hat{x}_i$ is contaminated by context, approximation errors, and text generation noise, and may not strictly follow a sinusoidal structure. Slight angular deviations cause the congruences to contradict each other, making pure analytical inversion impractical (C2 requires noise robustness).

RND abandons "precise alignment of every subspace" in favor of a **global alignment score**: the dot product of the output embedding with the unrotated base vector. In an ideal noise-free scenario, $x_m\cdot x_{[NUM]}=\sum_{j=1}^{d/2}\lVert x_{[NUM],j}\rVert^2\cos(m\omega_j)$. The key insight is that the global dot product is a weighted sum of cosine terms for each rotation angle, where weights are the squared norms of the 2D components. Thus, there is a **direct relationship between the number $m$ and a scalar derivable from the embedding**, providing a robust target for decoding. To ensure this target signal varies smoothly with $m$ (avoiding violent oscillations dominated by high-frequency terms), a generalized score weighted by $\omega_j$ is introduced:

$$S(m, p) = \sum_{j=1}^{d/2}\left(\frac{1}{\omega_j}\right)^{p}\lVert x_{[NUM],j}\rVert^{2}\cos(m\omega_j).$$

Decoding is performed via nearest-neighbor lookup on a predefined candidate set $M$: $\hat{m}=G_{dec}(\hat{x}_i)=\arg\min_{m'\in M}\big(S(\hat{x}_i,p)-S(m',p)\big)^2$. All target scores $S(m',p)$ can be precomputed and cached once ($\approx 20$s CPU, $\approx 1.2$ MB). During inference, a lookup is performed only when a `[NUM]` token is predicted, incurring almost zero overhead and allowing $M$ to cover a wide range. RND introduces no additional trainable parameters, making it more elegant than learning a linear head.

**3. Distinguishability-Smoothness Trade-off: Tuning the spectrum with scalar $p$**

The parameter $p$ in the formula controls a specific trade-off. When $p=0$, it reduces to an unweighted sum where high-frequency components contribute strongly, causing $S(m, 0)$ to oscillate rapidly—leading to high distinguishability for adjacent $m$ but susceptibility to aliasing at large values. When $p>0$ (e.g., $p=1$), low frequencies are amplified and high-frequency oscillations are suppressed, resulting in a smoother curve (better smoothness) and improved stability/range, but adjacent values of $m$ become harder to distinguish. Increasing $p$ effectively slides $S(m, p)$ along the spectrum from "high distinguishability but oscillatory" to "smooth but hard to distinguish." This study parameterizes this spectrum with a single scalar, finding $p=0.2\sim 0.3$ to be the best compromise. Values too large (e.g., 0.5) compress $[0, 30000]$ into a narrow score interval $[0.7, 1.0]$, leading to a drop in performance as small changes become indistinguishable.

### Loss & Training
The training target is $L = L_{CE} + \lambda L_{MSE}$. $L_{CE}$ is the standard next-token cross-entropy across all positions. $L_{MSE}$ is applied **only** at positions where the target token is `[NUM]`, penalizing the squared difference between the model's output score and the precomputed target: $L_{MSE}=\sum_i \mathbb{I}(w_i=[NUM])\,(S(\hat{x}_i,p)-S(m_i,p))^2$. This encourages both the placement of `[NUM]` in correct positions and the alignment of its embedding score with numerical targets. $\lambda$ follows a linear ramp-up schedule. Training on Opport-CT is two-stage: first, freeze the vision encoder and LLM to train the projector, followed by full instruction fine-tuning (projector + LoRA, rank 128, $\alpha$ 256).

## Key Experimental Results

### Main Results
Radiology measurement estimation (Opport-CT, in-house), MAE in mm (single value) or mm² (dual value), lower is better; $R^2$ higher is better:

| Method | Single MAE↓ | Single $R^2$↑ | Dual MAE↓ | Dual $R^2$↑ | Success% |
| :--- | :--- | :--- | :--- | :--- | :--- |
| xVal | 6.71 | 0.351 | 684.20 | 0.217 | 75.9 |
| Learnable | 10.28 | 0.150 | 925.62 | −0.06 | 77.4 |
| Abacus | 5.65 | 0.319 | 520.32 | 0.215 | 57.2 |
| Standard Token | 5.53 | 0.338 | 519.23 | 0.194 | 55.7 |
| **Ours (RNED)** | **4.72** | **0.568** | **449.23** | **0.320** | **81.8** |

On the public CT-RATE, fine-tuning the baseline CT-CHAT with the RNED objective for 1 epoch:

| Method | Single MAE↓ | Single $R^2$↑ | Dual $R^2$↑ | Success% |
| :--- | :--- | :--- | :--- | :--- |
| CT-CHAT | 5.88 | 0.370 | 0.164 | 85.0 |
| **CT-CHAT + RNED** | **4.80** | **0.592** | **0.608** | **90.0** |

Medical Visual Grounding (MedSeq-Bench, 8 tasks): Compared to the SOTA MedSeq-Grounder, RNED shows an average IoU gain of +1.66% and Acc@5 gain of +2.62%, with significant improvements in harder tasks like Multi-view and Object Tracking (IoU 55.03 $\rightarrow$ 59.26, 62.10 $\rightarrow$ 63.15).

### Ablation Study
Sensitivity of $p$ (Opport-CT):

| $p$ | Single MAE↓ | Single $R^2$↑ | Dual $R^2$↑ | Success% |
| :--- | :--- | :--- | :--- | :--- |
| 0 (No weight) | 5.99 | 0.547 | 0.269 | 80.8 |
| 0.2 | 4.76 | **0.622** | 0.276 | 81.3 |
| 0.3 | **4.72** | 0.568 | **0.320** | 81.8 |
| 0.4 | 5.10 | 0.541 | 0.055 | 82.5 |
| 0.5 | 6.01 | 0.510 | 0.210 | 82.4 |

Comparison of Encoding/Decoding strategies:

| Encoding | Decoding | Single MAE↓ | Single $R^2$↑ | Success% | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Adding | Linear | 24.86 | −2.025 | 33.1 | Direct addition, gradient instability |
| Individual | Linear | 13.68 | −0.664 | 67.9 | Discrete embeddings, non-order-preserving |
| RNE | Linear | 6.04 | 0.486 | 73.4 | Better representation improves results |
| **RNE** | **RND** | **4.72** | **0.568** | **81.8** | Full method performs best |

### Key Findings
- **Representation is more critical than the decoding head, but the combination is best**: RNE+Linear already significantly outperforms Adding/Individual, indicating that norm-preserving, order-preserving encoding provides the bulk of the gain; switching to RND decoding provides further improvement without additional parameters.
- **Standard tokenization regresses to "guessing the mean"**: The baseline $R^2$ is very low and success rate is only 55.7%, showing a tendency to output values close to the dataset mean. RNED treats numbers as single conceptual units, resulting in shorter prediction lengths and superior performance in dual-value scenarios (Success Rate > 80%).
- **No harm to text capabilities**: On standard VQA metrics such as BLEU-1, ROUGE-L, METEOR, and GREEN, RNED is on par with or slightly better than standard tokenization, suggesting numerical capability is "additive" and does not sacrifice semantic generation.

## Highlights & Insights
- **Migrating RoPE from "position encoding" to "value encoding"**: The norm-preserving property neatly avoids the xVal pitfall of "clashing with LayerNorm by changing token norms." This is the most elegant conceptual step in the paper.
- **Decoding shifted from "solving congruences" to "score-matching lookup"**: Acknowledging that hidden states are noisy, the authors do not force precise analytical recovery but instead use global dot-product scores $S(m,p)$ as a robust target. Precomputing and caching these scores makes the inference overhead nearly zero.
- **Using scalar $p$ to turn "distinguishability vs. smoothness" into a tunable knob**: Explicitly parameterizing the trade-off between high-frequency and low-frequency contributions makes the method transferable to any scenario involving continuous quantity embedding in discrete sequence models.

## Limitations & Future Work
- The numerical range depends on the predefined candidate set $M$ (e.g., $[0, 3000]$ for radiology, $[0, 400]$ for grounding) and step size; it requires resetting for out-of-range values or finer granularity.
- Primary results are on the in-house Opport-CT dataset; public benchmark gains are mostly verified via 1-epoch fine-tuning on existing models. Robustness across institutions and modalities requires more public data support. ⚠️ Opport-CT is non-public.
- In medical visual grounding, 3 out of 8 tasks showed no gain (e.g., Image Difference Grounding), which the authors attribute to SOTA saturation; in these cases, RNED "maintains" rather than "improves" performance.

## Related Work & Insights
- **vs. xVal**: Both seek a continuous representation for LLMs. xVal uses linear scaling + a number head but alters the token norm, conflicting with normalization. RNED uses norm-preserving rotation + parameter-free score-matching, remaining compatible with pre-trained weights without sacrificing text capability.
- **vs. Abacus / p10·p100**: These methods (reversing digit order, special string formats) target mathematical benchmarks. They increase prediction length and perform worse than baselines in clinical VQA with mixed text-number content. RNED compresses numbers into a single token.
- **vs. Individual embedding (Ablation baseline)**: Assigning an independent embedding to each candidate value results in stable norms and decent success rates, but it is not order-preserving and generalizes poorly to unseen numbers, confirming that "order-preservation + wide range" is the key.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Adapting RoPE's norm-preserving rotation for continuous numerical encoding and reframing decoding as score-matching is refreshing and self-consistent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive tasks and baselines with $p$ and encoding/decoding strategy ablations. However, public evidence relies mostly on fine-tuning increments and an in-house dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain: starting from properties C1–C3, moving from 2D intuition to high dimensions, then addressing the decoding trade-off.
- Value: ⭐⭐⭐⭐ Directly addresses the quantitative bottleneck in medical VLMs. The plug-and-play nature without harming text capabilities makes it valuable for any scenario requiring reliable numerical output from LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Sparse Spectral LoRA: Routed Experts for Medical VLMs](sparse_spectral_lora_routed_experts_for_medical_vlms.md)
- [\[CVPR 2026\] Reliable Clustering Number Estimation for Contrastive Multi-View Clustering](reliable_clustering_number_estimation_for_contrastive_multi-view_clustering.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[CVPR 2026\] ORION: ORthonormal Text Encoding for Universal VLM Adaptation](orion_orthonormal_text_encoding_for_universal_vlm_adaptation.md)

</div>

<!-- RELATED:END -->
