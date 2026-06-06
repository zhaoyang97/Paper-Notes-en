---
title: >-
  [Paper Note] T3S: Training Trajectory-Aware Token Selection to Break "Imitation Shock" in Reasoning Distillation
description: >-
  [ICML 2026][Model Compression][Reasoning Distillation] The paper identifies a universal "Imitation Shock" when a strong student (e.g., Qwen3-8B) is distilled from DeepSeek-R1—where the loss decreases monotonically while…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Reasoning Distillation"
  - "Imitation Shock"
  - "anchor token"
  - "training trajectory"
  - "AR + dLLM"
date: 2026-05-08
content_hash: ebb8036dbff3b878
---

# T3S: Training Trajectory-Aware Token Selection to Break "Imitation Shock" in Reasoning Distillation

**Conference**: ICML 2026  
**arXiv**: [2601.10348](https://arxiv.org/abs/2601.10348)  
**Code**: Not listed
**Area**: LLM Distillation / Reasoning Compression / Training Dynamics  
**Keywords**: Reasoning Distillation, Imitation Shock, anchor token, training trajectory, AR + dLLM  

## TL;DR
The paper identifies a universal "Imitation Shock" when a strong student (e.g., Qwen3-8B) is distilled from DeepSeek-R1—where the loss decreases monotonically while accuracy first plunges then recovers. The root cause is that the gradient of "Imitation-Anchor Tokens" dominates early optimization and suppresses tokens responsible for actual reasoning. T3S uses training trajectories to identify these anchor tokens and masks them, allowing reasoning tokens (yet-to-learn) to be learned earlier. This leads to performance gains in both AR and dLLM settings (Qwen3-8B surpasses DeepSeek-R1, Qwen3-32B approaches Qwen3-235B, and LLaDA-2.0-Mini surpasses the AR baseline to achieve 16B no-think SOTA).

## Background & Motivation

**Background**: When student LLMs already possess strong reasoning capabilities (e.g., Qwen3-8B), the community seeks to further improve them by distilling from stronger teachers (DeepSeek-R1, QwQ). Existing efficient distillation works (s1, LIMR, BOBA) have proven that a few hundred high-quality trajectories are more effective than massive datasets, but they focus on "how to select data" rather than analyzing whether "training dynamics are healthy."

**Limitations of Prior Work**: Direct distillation of Qwen3-8B from DeepSeek-R1 shows that while the loss decreases consistently, all benchmarks (AIME24/AIME25/MMLU-Pro) first crash to a certain point and then slowly recover—the authors term this "Imitation Shock" and label the lowest point checkpoint the "Imitation Bottleneck." More curiously, discarding all parameter updates before this bottleneck and keeping only subsequent updates (termed Recovering Residual Transfer) yields better results than standard SFT! This implies that "knowledge learned in the pre-bottleneck phase is unnecessary or even harmful."

**Key Challenge**: Teacher outputs contain "tokens that are easy to imitate but provide no reasoning gain" (e.g., format tokens, connectives, common expressions) and "tokens that actually carry reasoning" (e.g., key equations, intermediate derivations). Under standard next-token CE for SFT, the former produce larger gradients and converge faster, "anchoring" the model to the teacher's style while suppressing the learning of the latter. Consequently, the student appears to "imitate the teacher," but its actual reasoning capability initially declines—a classic case of "focusing on the trivial while losing the essential."

**Goal**: Systematically locate these "anchor tokens" using training trajectory signals and remove them from the loss to allow reasoning tokens to be learned earlier, avoiding the compute waste of the Imitation Bottleneck.

**Key Insight**: The critical step for token-level intervention is "how to find anchor tokens." The authors found that anchor tokens share a unified signal—confidence monotonically increases between the base and bottleneck checkpoints ($\Delta c_t > 0$), while reasoning tokens show a monotonic decrease. Thus, T3S simply "identifies the bottleneck → ranks by confidence delta → masks the increasing group."

**Core Idea**: Differentiate two types of tokens using confidence changes along the training trajectory: $\Delta c_t = c_t(\theta_b) - c_t(\theta_0)$. For AR, the anchor set $\mathcal{A}$ is masked from the loss. For dLLM, anchor tokens are prioritized into the visible context, forcing the mask to repeatedly focus on yet-to-learn reasoning tokens, thereby bypassing Imitation Shock from a training dynamics perspective.

## Method

### Overall Architecture

T3S consists of three steps: (1) Run a standard SFT to save checkpoints and identify the Imitation Bottleneck $\theta_b$ based on training accuracy; (2) Use a selector model $M_0$ to calculate log-probs for each token at the base $\theta_0$ and bottleneck $\theta_b$, taking the difference $\Delta c_t$; (3) Restart training and construct a token-level mask based on $\Delta c_t$—the AR model masks anchor tokens where $\Delta c_t > 0$ (applying loss only to remaining tokens), while the dLLM does the opposite by placing anchor tokens into the visible context, repeatedly forcing the model to learn yet-to-learn tokens. This process can be done online (monitoring training accuracy and switching to mask mode upon detecting a bottleneck) without requiring a full pre-distillation run.

### Key Designs

1.  **Imitation Bottleneck Identification + Recovering Residual Transfer (RRT) as Evidence**:
    *   Function: Locates the "moment to start masking" from the training trajectory and confirms that pre-bottleneck updates are redundant or harmful via RRT experiments.
    *   Mechanism: Define $\theta_b = \arg\min_\theta \mathrm{Acc}_{\mathrm{train}}(\theta)$; construct $\theta_{\mathrm{RRT}} = \theta_0 + (\theta_f - \theta_b)$, effectively "discarding all pre-bottleneck updates." Experiments show that while standard SFT causes Qwen3-8B distilled from DeepSeek-R1 on BOBA-200 to drop from 71.46 to 63.13 ($\downarrow 8.33$), RRT actually increases it to 72.61 ($\uparrow 1.15$). Similar patterns appear with QwQ as the teacher. This disruptive experiment in Section 2 establishes the legitimacy of T3S.
    *   Design Motivation: Traditional views hold that "decreasing loss = improving model." This paper refutes this—loss reduction may stem entirely from over-fitting anchor tokens, irrelevant to downstream tasks. RRT is not meant to replace SFT but to provide undeniable evidence that the pre-bottleneck phase is unnecessary.

2.  **Token Grouping Based on Confidence Change + AR Anchor Masking**:
    *   Function: Transforms "who dominates optimization" from a black box into an observable token set and intervenes directly in the loss.
    *   Mechanism: Use selector $M_0$ to calculate $c_t(\theta; x, y) = \log p_\theta(y_t | y_{<t}, x)$, and for each token in the trajectory, let $\Delta c_t = c_t(\theta_b) - c_t(\theta_0)$. The set $\mathcal{A}(x,y) = \{t : \Delta c_t > 0\}$ represents Imitation-Anchor Tokens—tokens the model "masters" early in distillation. The AR T3S loss excludes them from CE: $\mathcal{L}_{\mathrm{AR\text{-}T3S}} = \mathbb{E}[\sum_{t \setminus \mathcal{A}} -\log p_\theta(y_t | y_{<t}, x)]$. Word cloud analysis (Figure 3) shows anchors are mostly connectives, punctuation, and thought-leads, while yet-to-learn tokens are key equations and intermediate derivation steps—validating that the anchor/yet-to-learn split aligns with human intuition.
    *   Design Motivation: Instead of tuning hyperparameters or data, it is better to perform surgical removal of harmful elements at the loss level. The paper also uses "Anti-T3S" (training only on anchors and masking reasoning tokens) as a diagnostic—performance crashes from 71.46 to 26.67, proving the discriminative power of T3S token selection.

3.  **Gradient Interaction Evidence: Anchors and Yet-to-Learn Cannot Coexist**:
    *   Function: Provides mechanistic evidence from the gradient level on why masking is necessary.
    *   Mechanism: Intervention experiments in Figure 5 show that at checkpoints where anchors are not yet mastered (large $\mathcal{L}_{\mathrm{anchor}}$), a single update step optimizing only anchors causes a surge in other tokens' loss (large positive $\Delta \mathcal{L}_{\mathrm{other}}$)—anchor learning indeed suppresses other tokens. Figure 6 shows anchor gradient norms can be $17 \times$ those of other tokens early on, dropping to $2 \times$ at the bottleneck. Furthermore, the cosine similarity between the two gradient groups reaches $-0.4$ to $-0.5$ during the crash phase, indicating strong conflict. The $4 \times 4$ matrix in Table 6 quantifies this incompatibility: training on an anchor subset significantly increases the loss of the reasoning subset (and vice-versa).
    *   Design Motivation: These three gradient evidences corroborate each other, upgrading the "anchor suppresses yet-to-learn" hypothesis to a mechanistic conclusion, providing solid theoretical support for the brute-force intervention of masking.

### dLLM Side: Reverse Unmasking Operation

For diffusion LLMs (LLaDA-2.0-Mini), the training objective is random masked reconstruction. T3S does the reverse—it makes trajectory-identified yet-to-learn tokens **more frequent** targets for masking, forcing the model to repeatedly practice reasoning tokens while anchor tokens are provided in the context. This replaces dLLM's stochastic masking with trajectory-aware masking, aligning with the dLLM training paradigm.

## Key Experimental Results

### Main Results: AR Setting, Qwen3-8B Distillation

| Method | BOBA-200 AIME24 | BOBA-200 AIME25 | BOBA-200 AVG | S1K-200 AVG |
|------|------|------|------|------|
| BASE | 75.83 | 67.08 | 71.46 | 71.46 |
| SFT (R1) | 71.25 | 55.00 | 63.13 ↓8.33 | 64.17 |
| RRT (R1) | 76.67 | 68.54 | 72.61 ↑1.15 | 73.65 |
| -T3S (R1) (Reverse Mask) | 30.63 | 25.63 | 28.13 Crash | 26.67 |
| **T3S (R1)** | **80.63** | **73.96** | **77.30** | **80.00**+ |
| SFT (QWQ) | 73.33 | 63.33 | 63.30 ↓ | — |
| **T3S (QWQ)** | — | — | Significant ↑ | — |

T3S improves by an average of +14 points (BOBA-200) over standard SFT and is +5 points higher than RRT (parameter-level fix), indicating that token-level masking is more precise than parameter-level surgery. The crash of -T3S to 28.13 is a key diagnostic—proving that the token set selected by T3S is highly discriminative and not just a "random half mask."

### Main Results: dLLM Setting + Cross-Scale Validation

- LLaDA-2.0-Mini (16B no-think dLLM) + T3S surpasses the AR baseline of the same architecture, achieving 16B-scale no-think SOTA.
- Qwen3-32B + T3S approaches the level of Qwen3-235B on AIME—proving T3S is effective across student scales.

### Universality of Imitation Shock Across Settings

| Variant | Presence of crash-then-recover |
|------|--------------------------|
| Different teacher (QwQ) | ✓ |
| Different dataset (S1K-200) | ✓ |
| Large-scale data (R1-Distilled-OpenThought3-65K) | ✓ |
| Different student (R1-Distilled Llama3) | ✓ |
| Different domain (Code) | ✓ |

Imitation Shock is not an accident of a specific dataset or teacher—it is a universal phenomenon in continual distillation, positioning T3S as a general solution.

### Key Findings

- **Decreasing loss $\neq$ Improving model**: On BOBA-200, SFT loss decreases monotonically while AIME24 drops from 75.83 to 71.25, providing a direct counter-example to "monitoring loss to determine convergence."
- **Anchors are format/connectors; yet-to-learn are reasoning tokens**: Word cloud analysis (Figure 3) visually confirms that anchors overlap significantly with semantic connectives, aligning with intuition.
- **Overextended training won't help**: Training for 15 epochs on BOBA-200 still leaves 68.51% of tokens with lower confidence than the base (Table 2)—the anchor suppression effect does not automatically disappear over time.
- **$17 \times$ gradient ratio + cosine $-0.4$**: The dominance of anchor gradients in magnitude and conflict in direction explains from an optimization perspective why masking is necessary rather than just "adding regularization."
- **Distilling from R1 yields higher T3S gains than QwQ**: Stronger teachers provide richer signals but also stronger bias. T3S filters out the bias to better utilize the teacher.

## Highlights & Insights

- **Diagnosing distillation failure via training dynamics**: Unlike previous works that modify data/loss/algorithms, this paper is the first to attribute "why continual distillation fails" to token-level gradient interactions, backed by a complete chain of evidence (4 Takeaways + 6 Figures + $4 \times 4$ matrix).
- **Simple intervention + Massive gains**: T3S does not change the loss form, architecture, or require extra data; it simply adds a token-level mask to CE to gain 14 points.
- **Unified for AR and dLLM**: Extending the "trajectory-aware token selection" framework to diffusion LLMs is highly successful, with LLaDA-2.0-Mini outperforming the AR baseline—a rare success for dLLM on reasoning tasks.
- **Disruptive RRT experiment**: "Discarding pre-bottleneck updates is better" directly challenges the naive perception that "more training is better," offering practical guidance for compute allocation in distillation.
- **Easy engineering implementation**: "Monitoring train acc and switching to mask mode after bottleneck" can be embedded into standard training pipelines as an extension of early stopping, with low barriers to deployment.

## Limitations & Future Work

- **Dependency on verifier/gold answers**: Bottleneck detection relies on train accuracy, requiring an automatically determinable correctness signal, which is not directly applicable to open-ended tasks (chat, writing)—though the paper notes RLVR-style datasets naturally satisfy this.
- **Sensitivity of selector model $M_0$ selection**: The choice of selector model affects $\Delta c_t$ estimation; while the paper uses $M_0 = \text{base student}$, the impact of cross-architecture selectors hasn't been systematically ablated.
- **Static anchor sets are epoch-level**: A single bottleneck determines which tokens are permanently masked. As training progresses, the anchor set might drift, but T3S does not update it. Curriculum-style dynamic masking is a future direction.
- **Empirical nature of Imitation Bottleneck evidence**: Theoretically, why does the bottleneck appear across different teachers/students? What data/model combinations might prevent it? These questions remain unanswered.
- **Extension to cross-domain generalization**: Imitation Shock is observed in code distillation, but whether it holds for cross-lingual or multi-modal (CoT visual) distillation requires further validation.

## Related Work & Insights

- **vs s1 / LIMR / BOBA**: These focus on "data selection," while this paper focuses on "training process intervention." The two are orthogonal and additive (T3S + BOBA-200 = Main Results).
- **vs Classic Distillation (DistilBERT/TinyBERT)**: Those focused on "single-step knowledge transfer" (logit matching, attention mimicry), while this paper examines "multi-step training dynamics," a completely different research scale.
- **vs Recovering Residual Transfer**: RRT is the authors' own baseline, proving parameter-level surgery works; T3S further proves token-level intervention is more thorough.
- **vs Early Stopping**: Classic early stopping exits at the validation minimum; T3S identifies the bottleneck and masks without discarding updates, acting as a hybrid of "early stopping + selective masking."
- **Inspiration**: All scenarios involving "fine-tuning a base model but encountering negative transfer" (multilingual expansion, medical adaptation, domain customization) can try similar logic—monitoring trajectories for bottlenecks and performing token-level or layer-level surgery.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Imitation Shock is a new discovery; Recovering Residual Transfer and anchor token analysis are new concepts; the entire mechanism+method is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validates the phenomenon's universality across five dimensions (teacher/dataset/student/scale/domain) and provides mechanistic evidence via the $4 \times 4$ token-group transfer matrix and gradient visualization, covering almost all potential doubts.
- Writing Quality: ⭐⭐⭐⭐⭐ The 4 Takeaways link the entire paper's logic; formulas are concise and charts are rich, making it easy for both engineers and theoretical researchers to understand.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to LLM distillation practitioners; opens new directions for training dynamics research; effective for both AR and dLLM—likely a highly influential work for 2026.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](../../ICLR2026/model_compression/token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](../../ICLR2026/model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)
- [\[ICML 2026\] RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs](rabit_residual-aware_binarization_training_for_accurate_and_efficient_llms.md)

</div>

<!-- RELATED:END -->
