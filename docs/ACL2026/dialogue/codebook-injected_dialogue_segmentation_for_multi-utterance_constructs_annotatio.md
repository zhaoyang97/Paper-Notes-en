---
title: >-
  [Paper Note] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation
description: >-
  [ACL 2026][Dialogue Systems][dialogue act annotation] The paper reformulates dialogue act annotation as a two-step "segment-then-label" problem. It proposes two approaches: codebook-injected LLM segmentation (System 1) a…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "dialogue act annotation"
  - "codebook-injected segmentation"
  - "LLM annotation"
  - "gold-free evaluation"
  - "educational dialogue"
date: 2026-05-08
content_hash: 5f14aadcbc606729
---

# Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation

**Conference**: ACL 2026  
**arXiv**: [2601.12061](https://arxiv.org/abs/2601.12061)  
**Code**: https://github.com/National-Tutoring-Observatory/codebook-injected-segmentation (Available)  
**Area**: Dialogue Segmentation / Educational Dialogue Annotation  
**Keywords**: dialogue act annotation, codebook-injected segmentation, LLM annotation, gold-free evaluation, educational dialogue

## TL;DR
The paper reformulates dialogue act annotation as a two-step "segment-then-label" problem. It proposes two approaches: codebook-injected LLM segmentation (System 1) and Dial-Start with DA-aware retrieval augmentation (System 2). Furthermore, it introduces three categories of evaluation metrics (intra-segment consistency, inter-segment divergence, and human-AI distributional alignment) that do not require gold boundaries. Experiments on TalkMoves and CLASS-annotated datasets demonstrate that DA-aware prompting enables LLMs to produce more homogeneous segments, though no single method outperforms others across all evaluation dimensions.

## Background & Motivation

**Background**: Dialogue Act (DA) annotation typically assumes labels are applied to a single utterance or turn, using inter-annotator agreement (IRR) as a quality metric. Early segmentation work followed lexical cohesion methods like TextTiling, C99, and LCseg, while recent approaches have shifted toward contrastive learning (Dial-Start) and LLM prompting.

**Limitations of Prior Work**: In educational dialogues, "pedagogical moves" naturally span multiple utterances. For instance, while a teacher explains a concept, a student might interject with short responses like "mm-hmm" or "why?". Annotators often disagree on whether these constitute a single explanation segment (boundary disagreement), even if they agree on the underlying label. Enforcing utterance-level annotation and measuring quality via IRR misinterprets boundary variations as conceptual disagreements, inflating disagreement and masking true span-level consistency.

**Key Challenge**: The "unit-of-analysis" problem—determining "what the label is" and "where the label starts/ends" are independent dimensions. Prevailing pipelines bind them to utterances, creating artificial disagreement. Additionally, educational datasets lack gold segment annotations due to high costs and inherent ambiguity, making classic boundary metrics like $P_k$ or WindowDiff inapplicable.

**Goal**: (1) Isolate segmentation as a first-order design task; (2) explicitly inject the codebook (DA definitions) into the segmentation phase so boundary decisions serve downstream DA labeling; and (3) develop a set of distribution-based comparative metrics in the absence of gold segments.

**Key Insight**: The authors treat both LLMs and coherence-based methods as candidate segmenters. The key manipulation is whether the segmenter is "codebook-aware (DA-aware)" versus "text-only," conducted via a 2×2 controlled experiment. Simultaneously, they utilize both human and GPT-5 utterance-level labels to use "human-AI distributional agreement" as an evaluation metric.

**Core Idea**: Treat segmentation as an "optimization problem for downstream annotation goals" rather than a general coherence problem. By providing the codebook to the segmenter, boundary decisions are forced to serve the definition of the DA, evaluated through three types of gold-free multi-objective metrics.

## Method

### Overall Architecture
For a dialogue $\mathbf{u}=(u_1, \ldots, u_T)$, the segmenter outputs a set of boundaries $\mathcal{B} \subset \{1, \ldots, T-1\}$, partitioning the dialogue into $K=|\mathcal{B}|+1$ contiguous segments $S_1, \ldots, S_K$. The framework compares four segmenter configurations across a 2 (Method Family) × 2 (Codebook-Awareness) matrix:

- **LLM-zero-shot** (GPT-5 / Gemini-3-pro): Uses pure topic-shift prompts to output boundary indices in JSON.
- **LLM-DA-aware**: Uses identical prompts but includes DA definitions in the system message, instructing the model to place boundaries where pedagogical functions change.
- **Dial-Start**: A non-LLM baseline using a contrastive learning utterance encoder $f(\cdot)$ to calculate depth scores from adjacent utterance similarities, selecting boundaries based on $\mathrm{thr} = \mu + \alpha\sigma$.
- **Dial-Start + DA-aware**: Incorporates DA-conditioned retrieval onto utterance embeddings (see Key Design 2).

Evaluation is then performed using three categories of gold-free metrics: intra-segment consistency (normalized entropy ↓ / purity ↑), inter-segment divergence (Jensen-Shannon divergence ↑ / boundary change rate ↑), and human-AI distributional alignment (segment-level JS ↓).

### Key Designs

1.  **Codebook-injected LLM segmentation**:
    - **Function**: Utilizes DA definitions as explicit criteria for boundary decisions, ensuring the LLM considers "what label will be applied" while deciding "where to cut."
    - **Mechanism**: Inserts full move definitions (e.g., 6 TalkMoves categories or 4 CLASS instructional support categories) into the zero-shot prompt. The model is tasked to "place a boundary when the pedagogical function changes, but do not label the segments."
    - **Design Motivation**: Traditional topic-shift prompts rely on "thematic changes," often failing to cut when the instructional move shifts (e.g., from explanation to questioning) while the topic remains the same. Injecting DA definitions makes the LLM internalize the "pedagogical move" as the unit-of-analysis. Experiments show this reduces normalized entropy from 0.349 to 0.286 and increases purity from 0.546 to 0.570 for GPT-5 on the CLASS dataset.

2.  **DA-conditioned retrieval-augmented coherence segmentation**:
    - **Function**: Injects codebook semantics into Dial-Start’s coherence-based representations without retraining the boundary detector.
    - **Mechanism**: Maintains a memory of expert-labeled DA utterances $\mathcal{M}=\{(h_j, m_j)\}$. For a current utterance $u_i$, its embedding $h_i=f(u_i)$ is used to retrieve top-$K_{\text{ret}}$ neighbors. DA embeddings are aggregated using attention weights calculated via cosine similarity and softmax temperature $\tau$: $r_i=\sum_k a_k e_{m_{j_k}}$. This is fused into the original representation $\hat{h}_i=\text{norm}(h_i+\alpha r_i)$ before calculating adjacent similarity.
    - **Design Motivation**: Fine-tuning Dial-Start requires boundary labels, which are scarce. Retrieval augmentation allows the boundary score to reflect "DA consistency" without discarding coherence goals. Interestingly, this proved effective for LLMs but less stable for Dial-Start (entropy increased from 0.303 to 0.319 on CLASS), suggesting DA-awareness aligns better with instruction-following models.

3.  **Gold-label-free evaluation metrics**:
    - **Function**: Scores segmentation using the distributional properties of DA labels (intra-segment, adjacent segments, and cross-annotator) without gold boundaries.
    - **Mechanism**: Converts each segment $S_k$ into a DA distribution $p_{k,c}^{(r)}=\frac{1}{|S_k|}\sum_{u_i\in S_k}\mathbb{1}[y_i^{(r)}=c]$, with weight $w_k=|S_k|/T$.
        - (i) Intra-segment consistency: normalized entropy $\widetilde{H}_k^{(r)}=H_k^{(r)}/\log_2 C$ (↓) and purity $\max_c p_{k,c}^{(r)}$ (↑).
        - (ii) Inter-segment divergence: adjacent JS divergence $\overline{\text{JS}}_{\text{adj}}^{(r)}$ (↑) and boundary change rate $\text{BCR}^{(r)}$ (↑).
        - (iii) Human-AI alignment: $\overline{\text{JS}}_{\text{HA}}=\sum_k w_k \text{JS}(p_k^{(H)}, p_k^{(A)})$ (↓).
    - **Design Motivation**: Removes the need for expensive reference boundaries and accounts for boundary ambiguity. It treats segmentation as a multi-objective optimization problem.

### Loss & Training
No new models are trained. Dial-Start uses the original contrastive utterance encoder with hyper-parameters: window\_size=2, $\alpha$=0.5, pick\_num=4, and min\_gap=3. LLMs use GPT-5 and Gemini-3-pro with fixed prompts. Retrieval-augmented Dial-Start uses $K_{\text{ret}}$ neighbors and a learned move-embedding table $E \in \mathbb{R}^{M \times d}$ ($M=6$ or $4$), but the boundary detector remains frozen.

## Key Experimental Results

### Main Results (mean [95% CI], K = avg. segments per dialogue)

**CLASS-annotated dataset** (30 tutoring sessions, 4 CLASS moves):

| Method | K | Intra $\widetilde{H}$ ↓ | Purity ↑ | $\overline{\text{JS}}_{\text{adj}}$ ↑ | BCR ↑ | Human-AI $\overline{\text{JS}}_{\text{HA}}$ ↓ |
|---|---|---|---|---|---|---|
| GPT-5 | 4.90 | 0.349 | 0.546 | 0.447 | 0.222 | 0.424 |
| **GPT-5 DA-aware** | 6.30 | **0.286** | 0.570 | 0.477 | **0.288** | 0.449 |
| Gemini-3-pro | 4.47 | 0.384 | 0.528 | 0.447 | 0.237 | **0.407** |
| Gemini-3-pro DA-aware | 4.53 | 0.391 | 0.531 | 0.435 | 0.267 | 0.411 |
| Dial-Start | 4.60 | 0.303 | 0.564 | **0.545** | 0.208 | 0.459 |
| Dial-Start + DA-aware | 4.50 | 0.319 | 0.561 | 0.515 | 0.253 | 0.484 |

**TalkMoves dataset** (63 K-12 math sessions, 6 talk moves):

| Method | K | Intra $\widetilde{H}$ ↓ | Purity ↑ | $\overline{\text{JS}}_{\text{adj}}$ ↑ | BCR ↑ | Human-AI $\overline{\text{JS}}_{\text{HA}}$ ↓ |
|---|---|---|---|---|---|---|
| GPT-5 | 10.86 | 0.616 | 0.659 | 0.447 | 0.235 | **0.470** |
| GPT-5 DA-aware | 12.54 | 0.609 | 0.664 | 0.470 | 0.222 | 0.489 |
| Gemini-3-pro | 16.62 | 0.598 | 0.662 | 0.471 | 0.235 | 0.480 |
| **Gemini-3-pro DA-aware** | 19.53 | **0.566** | **0.676** | **0.478** | 0.252 | 0.505 |
| Dial-Start | 14.60 | 0.619 | 0.640 | 0.475 | 0.416 | 0.513 |
| Dial-Start + DA-aware | 14.75 | 0.639 | 0.633 | 0.469 | **0.524** | 0.503 |

### Ablation Study

| Comparison | Observation | Interpretation |
|---|---|---|
| LLM text-only → LLM DA-aware | Entropy −0.063, purity +0.024 (CLASS) | DA-awareness significantly improves LLM intra-segment consistency. |
| Dial-Start → Dial-Start + DA-aware | Entropy +0.016 (CLASS), −0.020 (TalkMoves) | DA retrieval provides inconsistent or negative gains for coherence-based segmenters. |
| Intra-segment H vs. Adj JS | GPT-5 DA-aware best in intra-segment on CLASS, but $\overline{\text{JS}}_{\text{adj}}$ (0.477) worse than Dial-Start (0.545). | Homogeneity gains often lead to reduced boundary distinctiveness. |
| DA-aware vs. Human-AI alignment | GPT-5 DA-aware $\overline{\text{JS}}_{\text{HA}}$ (0.449) higher than GPT-5 (0.424) on CLASS. | Strict codebook adherence can distance LLMs from human labeling styles (pragmatic smoothing). |
| K variations | DA-aware prompting increases K. | Performance gains are not just an artifact of over-segmentation. |

## Key Findings
- **DA-aware works for LLMs, not coherence baselines**: Codebook injection fits instruction-following models but clashes with "similarity-drop" boundary detectors.
- **Three-way trade-off**: No method is optimal across intra-segment consistency, boundary divergence, and human-AI alignment. LLMs produce more homogeneous segments, while coherence-based methods produce sharper boundaries.
- **DA-aware prompting can lower Human-AI alignment**: Guided LLMs may follow definitions too "dogmatically," deviating from humans who apply pragmatic smoothing.
- **Data density matters for retrieval**: Dial-Start + DA-aware showed high BCR (0.524) on TalkMoves (1.9k samples) but failed on CLASS (301 samples), indicating retrieval effectiveness depends on memory size.

## Highlights & Insights
- **Reframing Segmentation**: Moving from "general coherence" to "optimization for downstream goals" opens the path for "codebook-injected" designs applicable to high-stakes fields like clinical or legal analysis.
- **Gold-Free Trio**: Using entropy, adjacent JS, and human-AI JS provides a unified framework to measure multi-objective segmentation performance without gold data.
- **LLM as a Diagnostic Tool**: The authors treat "human-LLM disagreement" as a signal of codebook ambiguity rather than a simple model error, providing a new way to calibrate annotation protocols.

## Limitations & Future Work
- The gold-free metrics might miss pedagogically meaningful shifts that aren't distributionally significant.
- The evaluation depends on the quality and granularity of the DA taxonomy.
- Relies on commercial APIs (GPT-5), raising reproducibility concerns regarding versioning.
- **Future Work**: Exploring cross-attention for codebook embeddings rather than prompt injection; using human-AI disagreement for active learning; and testing in multi-party or multi-modal settings.

## Related Work & Insights
- **vs. Dial-Start**: Confirms Dial-Start remains strong for "sharp shifts," but lacks the intra-segment homogeneity of codebook-injected LLMs.
- **vs. SuperDialSeg**: While SuperDialSeg focuses on supervised boundaries, this work provides a zero-shot/gold-free alternative for domains lacking such data.
- **vs. S3-DST**: Unlike S3-DST which combines segmentation and tracking, this isolates segmentation as a modular, first-class design choice.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[NeurIPS 2025\] AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](../../NeurIPS2025/dialogue/aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](../../ICML2026/dialogue/not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)

</div>

<!-- RELATED:END -->
