---
title: >-
  [Paper Note] Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment
description: >-
  [ACL 2026][LLM Safety][Attack Comfort Zone] This paper reveals for the first time a safety blind spot in MLLMs under the "visual text compression" paradigm—when rendered image DPI falls within the Attack Comfort Zone (AC…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Attack Comfort Zone"
  - "Cognitive Overload"
  - "Safety Feature Lag"
  - "Structured Offloading"
  - "Visual Text Compression"
date: 2026-05-08
content_hash: 112cd4aa54b50499
---

# Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment

**Conference**: ACL 2026  
**arXiv**: [2605.07250](https://arxiv.org/abs/2605.07250)  
**Code**: https://github.com/Westlake-AGI-Lab/ACZ-Jailbreak  
**Area**: Multimodal VLM / Safety Alignment / Jailbreak  
**Keywords**: Attack Comfort Zone, Cognitive Overload, Safety Feature Lag, Structured Offloading, Visual Text Compression

## TL;DR
This paper reveals for the first time a safety blind spot in MLLMs under the "visual text compression" paradigm—when rendered image DPI falls within the Attack Comfort Zone (ACZ) of 45–150, the model's OCR remains accurate, but safety alignment collapses (ASR surges from 0% to over 70%). The underlying cause is that shallow-layer computational resources are exhausted by "character recognition," causing harmful semantics to emerge only in deeper layers, thereby bypassing shallow-layer guardrails. Using prompt-level Structured Cognitive Offloading (transcribe → audit → respond) can bring the ASR back to near-baseline levels.

## Background & Motivation

**Background**: Work on "visual text compression" such as DeepSeek-OCR and Glyph renders long text into images for MLLMs, loading equivalent text with significantly fewer vision tokens. This is a key technology for long-context compression. The academic community has focused primarily on whether models can "read correctly," while almost no one has questioned whether safety alignment remains intact when text is "hard to read."

**Limitations of Prior Work**: (1) Existing visual jailbreaks rely on white-box PGD adversarial noise (Qi 2024, Bailey 2024) or obvious typography layout tricks (FigStep), both requiring "manipulation" and being easily detectable. (2) Standard compression/downsampling should be a benign utility operation and has never been suspected as an attack surface. (3) Mechanistic studies have established that "refusal occurs mainly in shallow layers" and "shallow layers act as low-pass filters for low-quality images," but no one has linked these two to derive safety consequences.

**Key Challenge**: The "safety audit" and "content recognition" of MLLMs are forced to share the same shallow-layer computational resources during the forward pass. When images are difficult to read, recognition "preempts" shallow resources, pushing safety features into deeper layers, while guardrails are specifically placed in the shallow layers—creating a structural depth misalignment.

**Goal**: (1) Systematically characterize the "resolution/perturbation → jailbreak success rate" curve to prove the existence of a sweet spot. (2) Use linear safety probes to directly measure the "safety feature lag" mechanism layer-wise. (3) Provide a training-free, prompt-level defense that does not compromise normal utility.

**Key Insight**: An analogy is drawn to the phenomenon where humans reading "puns" need to "read them out loud" to perceive latent meanings—when recognition consumes significant "attention," the detection of potential malice in content is delayed. This predicts an counter-intuitive, inverted-U curve where "medium DPI is most dangerous."

**Core Idea**: MLLM safety failure is redefined as a "**computational resource allocation problem**" rather than a "**data alignment problem**." Therefore, the solution is not re-training alignment, but rather "offloading" recognition from auditing, allowing the safety audit to be completed independently on clean text.

## Method

### Overall Architecture
The paper consists of two parts: (a) **Phenomenon Analysis**—constructing 770 deduplicated harmful queries × DPI $\in \{15, 30, \dots, 300\}$ rendered images, running 10+ SOTA MLLMs, and plotting DPI–ASR curves using a triple-LLM judgment + human arbitration ASR protocol to identify the ACZ; quantifying "safety feature lag" with layer-wise linear safety probes. (b) **Defense Method**—proposing Structured Cognitive Offloading, which splits a single prompt into a serialized execution of transcription → safety → response; and locating the root cause as "content decoding difficulty" by ablating three types of confounders: token count, templates, and OOD.

### Key Designs

1.  **Attack Comfort Zone (ACZ) Phenomenon + Three-Phase DPI Curve**:
    *   **Function**: Quantitatively characterizes the non-monotonic relationship between "resolution and safety" using a DPI–ASR curve.
    *   **Mechanism**: Each harmful query is rendered into an image using a role-play template + Glyph rendering framework. A full-spectrum scan from 15 to 300 DPI is performed, measuring both char/word OCR accuracy and ASR. ASR is defined as $$\mathcal{ASR}=\frac{1}{M}\sum_i \mathbb{I}(\mathcal{J}(R_i)=1)$$, where $\mathcal{J}$ is adopted through consensus between DeepSeek-V3.2 / Kimi-K2 / GLM-4.6, otherwise resolved by human arbitration (95.9% consistency, Cohen's $\kappa=0.96$). The curve shows three distinct stages: **Phase I Blind Zone (DPI ≤30)** where both OCR and ASR are near 0; **Phase II ACZ (45–150)** where OCR > 80% but ASR skyrockets to 30–86%; **Phase III Recovery Zone (≥200)** where OCR $\approx$ 1 and ASR returns to baseline. The peaks for multiple models concentrate at 45–60 DPI, making this the most dangerous zone where text is "readable but safety is breached."
    *   **Design Motivation**: This curve is the most impactful experiment of the paper—it refutes the intuition that "low resolution = safety (cannot see, thus cannot do bad deeds)" and the opposite intuition that "high resolution = danger (must see to do bad deeds)." It clearly identifies "medium clarity" as the true battlefield.

2.  **Cognitive Overload Hypothesis + Layer-wise Safety Probe**:
    *   **Function**: Shifts the explanation of "why ACZ fails" from the phenomenal level to the mechanistic level, proving it is safety feature lag caused by recognition tasks preempting shallow compute.
    *   **Mechanism**: A balanced text dataset of 120+120 (harmful/harmless) is used to train an L2-regularized logistic probe $p^{(l)}=\sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l)}+\mathbf{b}^{(l)})$ on the hidden state $\mathbf{h}^{(l)}$ of the last token of each layer. These probes are then **frozen** to evaluate image inputs directly (cross-modal zero-shot evaluation), avoiding intra-modal fitting artifacts. Results (Fig. 4) show: High-DPI inputs are judged as unsafe at shallow layers (green density concentrated to the right of the classification boundary), while ACZ input distributions at shallow layers almost overlap with harmless text (blue), only separating harmful features in deeper layers. This proves "safety feature lag" at the representation level.
    *   **Design Motivation**: Since the ACZ phenomenon itself is a black-box observation that could have multiple explanations (OOD, template artifacts, insufficient tokens), layer-wise probing provides hard evidence for the Cognitive Overload hypothesis by quantifying "at which layer do safety features emerge."

3.  **Structured Cognitive Offloading (Defense)**:
    *   **Function**: Forcibly decouples recognition and auditing in the forward time dimension, allowing safety audits to run on "already cleaned text" rather than "images still struggling in vision."
    *   **Mechanism**: While the standard approach is $P(R\mid I_{v-text},\mathcal{P}_{dir})$ monolithic generation, this paper proposes a composite prompt $\mathcal{P}_{struc}$ that factorizes generation into $P(R,\hat{S},\hat{T}\mid I)=P(R\mid \hat{S},\hat{T})\cdot P(\hat{S}\mid \hat{T})\cdot P(\hat{T}\mid I)$. The three segments are **Transcription** (OCR $\hat{T}$ first), **Safety** (safety judgment $\hat{S}$ based only on $\hat{T}$), and **Response** (final answer conditional on $\hat{T}, \hat{S}$). The key is the $P(\hat{S}\mid \hat{T})$ step—the safety audit only sees clean text, and all interference from visual degradation is "washed away" by transcription.
    *   **Design Motivation**: Expecting MLLMs to implicitly perform "read + judge" simultaneously fails due to compute competition. Explicitly splitting the process into serial stages is equivalent to "offloading" from the architecture level to the prompt level—requiring zero additional training, zero model modifications, and remaining compatible with closed-source APIs.

### Loss & Training
No new models were trained in this paper. Probes were trained using logistic regression on 240 text samples as analysis tools. The defense, Structured Cognitive Offloading, is pure prompt engineering with no parameter updates.

## Key Experimental Results

### Main Results: ACZ Phenomenon (ASR ↑ is worse)

| Model | Text ASR | ACZ Image ASR | OCR char Acc | Max-ASR DPI |
|------|------|------|------|------|
| GPT-4.1 | 0.127 | **0.325** | 0.849 | 60 |
| Claude-Sonnet-4.5 | 0.000 | **0.429** | 0.920 | 60 |
| Claude-Haiku-4.5 | 0.080 | **0.408** | 0.912 | 60 |
| Gemini-2.5-Flash | 0.475 | **0.575** | 0.981 | 150 |
| Doubao-Seed-1.6 | 0.173 | **0.471** | 0.777 | 60 |
| Qwen3-VL-Plus | 0.353 | **0.697** | 0.967 | 45 |
| Qwen3-VL-32B-thinking | 0.367 | **0.862** | 0.954 | 60 |
| GLM-4.5V | 0.292 | **0.667** | 0.955 | 90 |
| Intern-S1 | 0.908 | **0.950** | 0.911 | 150 |

$\rightarrow$ Safety guardrails for all SOTA models (including closed-source GPT-4.1, Claude-4.5) significantly collapse in the 45–150 DPI range. OCR accuracy is almost entirely >85%, proving this is not "invisibility."

### Ablation Study: Perturbation Generality + Token Confounders + Defense

| Dimension | Configuration | Doubao ASR | Qwen3-VL ASR |
|------|------|------|------|
| **Perturbation Generality** | Baseline | 0.035 | 0.081 |
|  | Blur | **0.469 (+1240%)** | **0.649 (+701%)** |
|  | Occlusion | 0.138 (+294%) | 0.147 (+81%) |
|  | Distortion | 0.108 (+209%) | 0.178 (+120%) |
|  | Noise | 0.095 (+171%) | 0.129 (+59%) |
| **Token Confounder** | ACZ (tokens ~95K/113K) | 0.471 | 0.674 |
|  | **Padding ACZ** (same shape) | 0.421 | **0.694** |
|  | High DPI (same tokens) | 0.047 | 0.351 |
| **Defense (Ours)** | ACZ baseline | ~0.471 | ~0.674 |
|  | + Offloading | **$\approx$ 4%** | **$\approx$ 4%** |
| **Utility Preservation** | Direct ANLS | 0.347 | 0.387 |
|  | Structured ANLS | **0.478** | **0.550** |
|  | FRR (false refuse) | **0%** | **0%** |

The ACZ also holds for Chinese prompts: Doubao at 300 DPI (0.167) → ACZ (0.703, +0.536), proving it is not exclusive to English.

### Key Findings
- **The Inverted-U Curve is Universal**: From open-source 7–32B to closed-source Claude-4.5/GPT-4.1, ASR surges in the ACZ range without exception, rendering the intuition "higher safety alignment = harder to attack" obsolete under this paradigm.
- **Token Count is Not the Cause**: Padding ACZ images to the token count of high-resolution images (985K vs 113K) still results in ~69% ASR, confirming the bottleneck is "decoding content complexity" rather than compute quota.
- **OOD Does Not Hold**: t-SNE shows ACZ representations share a manifold with high-fidelity samples, suggesting the model does not treat ACZ as weird input; the attack relies on being "seemingly normal but internally strained," so OOD detectors cannot catch it.
- **Defense has No Side Effects**: On 300 benign OCR document samples, offloading actually improved ANLS by 7–16 points with a false refusal rate = 0%, breaking the spell that "safety defense must reduce utility."

## Highlights & Insights
- **Reframing MLLM Safety as a "Computational Resource Allocation Problem"**: This is the paper’s deepest conceptual contribution. While prior safety research attributed jailbreaks to "insufficient training data" or "poor alignment," this work uses layer-wise probes to show the issue may simply be that compute is stolen by recognition tasks during the forward pass. This perspective directly inspired a zero-training, zero-parameter defense and implies that safety probes can be viewed as "compute diagnostic tools."
- **ACZ is a "Natural Attack"**: Unlike PGD adversarial noise requiring white-box gradients or typography requiring layout skills, ACZ uses a simple "lower DPI" operation. This means any real-world deployment compressing long documents into images (DeepSeek-OCR, Glyph-like products) is naturally fragile—this is an emergent real-world threat surface.
- **Factorized Prompt Paradigm for Ours**: The $P(R,\hat{S},\hat{T}\mid I)$ approach of explicitly factorizing the "Chain of Thought" is essentially a "prompt-level architectural design" that can be generalized to any task coupling recognition and decision-making (e.g., code safety audit, compliance screenshot review).

## Limitations & Future Work
- **Output Length Increase of 102%**: The serialized transcription + safety + response segments double the average output, which is a significant drawback for real-time high-throughput scenarios.
- **Instruction Following Rate is Not 100% Guaranteed**: Authors achieved 100% following only on GPT-4.1, Claude-4.5, and Qwen3-VL; smaller or less obedient models might skip the transcription phase.
- **Only Covers "High-Density Text Image" Attacks**: Abstract semantic visual attacks (e.g., the image itself is a metaphorical threat) are not dominated by cognitive overload and cannot be helped by offloading.
- **Future Directions**: Distilling Structured Cognitive Offloading into internal "thinking tokens"; researching how to parallelize stages to reduce latency; and combining with adversarial training to improve the robustness of shallow-layer safety features against visual degradation.

## Related Work & Insights
- **vs. PGD Visual Jailbreak (Qi 2024, Bailey 2024)**: Those methods require white-box gradients and generate perceivable artifacts; ACZ is black-box, natural, and artifact-free, presenting a broader threat surface.
- **vs. FigStep / Typography Jailbreak (Gong 2025)**: Typical typography attacks rely on arranging malicious words in eye-catching layouts. ACZ does the opposite—reducing legibility results in a jailbreak, making them complementary.
- **vs. Xing 2025, Zhao 2024 (Shallow Layers = Safety Bottleneck)**: This paper synthesizes these mechanistic studies into a "depth misalignment" mechanism: visual degradation pushes semantics to deep layers, precisely when shallow guardrails fail. This paper serves as an empirical convergence of these two lines of research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "Attack Comfort Zone" phenomenon and redefinition as a "resource allocation problem" are novel concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10+ SOTA models × full DPI spectrum × multiple perturbations × token padding anti-confounder + probe mechanistic evidence + bilingual English/Chinese.
- Writing Quality: ⭐⭐⭐⭐ The Inverted-U curve, three-stage segmentation, and defense factorization formulas are clear with a solid logical chain.
- Value: ⭐⭐⭐⭐⭐ Directly reveals a realistic threat to visual text compression deployments and provides a prompt-level, immediately deployable defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[ACL 2026\] LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](llm-va_resolving_the_jailbreak-overrefusal_trade-off_via_vector_alignment.md)
- [\[ICML 2026\] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](../../ICML2026/llm_safety/old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)

</div>

<!-- RELATED:END -->
