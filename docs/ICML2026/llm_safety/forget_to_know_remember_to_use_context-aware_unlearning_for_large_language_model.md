---
title: >-
  [Paper Note] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
description: >-
  [ICML 2026][LLM Safety][LLM unlearning] This paper points out that existing LLM unlearning methods, while "erasing knowledge from parameters…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "LLM unlearning"
  - "contextual utility"
  - "KL regularization"
  - "TOFU"
  - "RAG-friendly unlearning"
date: 2026-05-08
content_hash: a33d1424071b82d6
---

# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.17620](https://arxiv.org/abs/2510.17620)  
**Code**: Not disclosed  
**Area**: LLM Security / Machine Unlearning  
**Keywords**: LLM unlearning, contextual utility, KL regularization, TOFU, RAG-friendly unlearning

## TL;DR
This paper points out that existing LLM unlearning methods, while "erasing knowledge from parameters," also destroy the model's "contextual utility"—its ability to correctly utilize knowledge when it is re-provided by the user in the prompt. The authors propose adding a KL regularization term to existing unlearning losses. By aligning the output distribution of the unlearned model with the original model on "Question + Context" inputs, the Contextual QA LLM-Judge score is restored from 0.00–0.84 to 0.95+ with almost no loss in forgetting effectiveness or retain set utility.

## Background & Motivation

**Background**: LLMs are trained on web-scale corpora and inevitably ingest information that must be "deleted," such as copyrighted content or private personal information. Since full retraining is prohibitively expensive, various unlearning methods have emerged: Gradient Ascent families (GradAscent / GradDiff), preference optimization families (NPO, DPO variants), re-labeling families (UNDIAL), and representation perturbation families (RMU). Evaluation standards typically focus on two aspects: thorough forgetting on the forget set (low Direct QA score) and performance preservation on the retain set (no loss in model utility).

**Limitations of Prior Work**: The authors observe a third dimension ignored by the community—in RAG and long-prompt scenarios, models often receive "theoretically forgotten" content as input (e.g., user-uploaded documents or retrieved copyrighted segments). Using this information is legally permissible since it is provided by the user in-context rather than remembered by the model. However, existing unlearning methods fail on such "Contextual QA" tasks where the answer is explicitly provided: on Gemma-2B-IT, RMU, GradAscent, and GradDiff reduce Contextual QA scores to nearly zero, while NPO and UNDIAL show drops of over 15.5%. Case studies show outputs degrading from "hallucinated locations" to pure gibberish like "denden den den...".

**Key Challenge**: All existing losses focus on the binary trade-off between "forget vs. retain," essentially penalizing the parameter representations of $\mathcal{S}_f$. This penalty does not "only punish memory recall"—it spills over in the representation space to context conditioning during inference. When the same tokens appear in the context, the model loses its ability to ground its generation on those tokens to produce correct answers. Methods like RMU, which erase knowledge by perturbing activations, are particularly severe as they directly destroy the representation pathways of related concepts.

**Goal**: (1) Systematically quantify the side effects of 6 SOTA unlearning methods on Contextual QA; (2) Design a "plug-and-play" patch with minimal changes to original methods to recover contextual utility without compromising forgetting or general utility.

**Key Insight**: RLHF has demonstrated that KL regularization can prevent a model from deviating from the original model in certain behavioral dimensions. Since the problem is that the unlearned model's behavior changes given $(q, c)$ inputs, one can use the original model as an anchor and apply a KL constraint on the $(q, c)$ data stream. This constraint acts on a different input distribution ($\mathcal{S}_f^{\text{ctx}}$) than the forget term ($\mathcal{S}_f$), preventing interference.

**Core Idea**: Add a third term to the standard unlearning loss: $\lambda_c \cdot \mathrm{KL}(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c))$. This explicitly anchors the "contextual conditional distribution" to the original model, decoupling "parametric memory" from "contextual usage" at the loss level. The objective is: "do not recall from memory, but do use when provided."

## Method

### Overall Architecture
The method consists of two phases. Phase one is diagnosis: evaluating 6 SOTA unlearning methods (GradAscent, GradDiff, NPO, DPO, UNDIAL, RMU) on the TOFU benchmark (fictional author profiles) with a 5% forget ratio on Gemma-2B-IT and Qwen3-8B. A new Contextual QA evaluation protocol is introduced—using the same forget set questions but providing the ground truth as context in the prompt. Performance is measured via ROUGE-L and LLM-Judge. Phase two is the patch: adding a context term $\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$ to existing losses. This uses a constructed set $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$ (where $c$ is the context containing the answer) to align the unlearned model's prediction distribution with the frozen original model $p_{\text{orig}}$. The total objective function becomes: $\mathcal{J}(w) = -\lambda_f L_f(\mathcal{S}_f, w) + \lambda_r L_r(\mathcal{S}_r, w) + \lambda_c \mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$.

### Key Designs

1.  **Contextual QA Evaluation Protocol**:
    *   **Function**: Complements unlearning evaluation with a third dimension, specifically measuring whether knowledge is still usable when provided in the prompt.
    *   **Mechanism**: For every $(q, a)$ in the forget set, a context $c$ (containing the ground truth description) is paired. The prompt "Question + Provided Context" is fed to the unlearned model. ROUGE-L measures literal overlap, and LLM-Judge measures semantic correctness. The protocol also includes paraphrase and reasoning context variants to ensure the model isn't just memorizing surface patterns. An ideal model should have low Direct QA, high Contextual QA, and high model utility.
    *   **Design Motivation**: Previous protocols only tested if the model remembered information without hints. However, as LLMs are increasingly used in RAG/long-prompt scenarios, whether a model can use legally re-provided information is the real concern for deployment. This protocol exposes neglected side effects—Table 1 shows 5 out of 6 methods produce errors or gibberish even when the answer is in the context.

2.  **Context-aware KL Regularization**:
    *   **Function**: Acts as a pluggable loss module to anchor contextual utility at the original model's level.
    *   **Mechanism**: $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$ is constructed (where $c$ is derived from TOFU ground truth). The term is defined as $\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w) = \frac{1}{|\mathcal{S}_f^{\text{ctx}}|} \sum_{(q,a,c)} \mathrm{KL}(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c))$. This forces the token distribution under "Question + Context" to approximate the frozen original model. This term is orthogonal to the original loss. Appendix A.6 shows results are insensitive to $\lambda_c$ (values between 0.01–2.0 work across models and methods without fine-tuning).
    *   **Design Motivation**: The key insight is that the standard two-term loss constrains two distributions but leaves the third (contextual) distribution floating. KL-to-reference is a proven tool in RLHF. Here, the "reference" is the pre-unlearning self, and the "constrained behavior" is limited to the $(q, c)$ stream. KL is a distribution-level constraint, which is gentler than single-point distillation and avoids direct conflict with the forget term.

3.  **Plug-and-play Integration**:
    *   **Function**: Enables different unlearning paradigms (NPO, RMU, UNDIAL) to use the same context term.
    *   **Mechanism**: Regardless of the paradigm, most methods follow a "forget term + optional retain term" structure. The new method simply adds $+\lambda_c \mathcal{C}$, requiring one extra forward pass on the original model per step.
    *   **Design Motivation**: The unlearning community is fragmented with many new methods. A successful proposal must be compatible with existing paradigms. This paper demonstrates that the same context term significantly improves Contextual QA across three entirely different paradigms.

### Loss & Training
Final loss: $\mathcal{J}(w) = -\lambda_f L_f + \lambda_r L_r + \lambda_c \mathcal{C}$. Training follows TOFU standards (AdamW, extended from 5 to 20 epochs for convergence). $\lambda_c$ values: for Gemma-2B-IT, NPO/RMU/UNDIAL use 2.0 / 0.01 / 0.5; for Qwen3-8B, they use 1.0 / 0.5 / 1.0.

## Key Experimental Results

### Main Results
TOFU 5% forget ratio, comparing vanilla vs. context-aware unlearning:

| Model | Method | Variant | Direct ROUGE-L ↓ | Contextual ROUGE-L ↑ | Direct LLM-Judge ↓ | Contextual LLM-Judge ↑ | Utility ↑ |
|------|------|------|---|---|---|---|---|
| Gemma-2B-IT | NPO | Vanilla | 0.31 | 0.55 | 0.19 | 0.81 | 0.57 |
| Gemma-2B-IT | NPO | Context-aware | 0.36 | **0.87** (+0.32) | 0.25 | **0.98** (+0.17) | 0.57 |
| Gemma-2B-IT | RMU | Vanilla | 0.04 | 0.01 | 0.00 | 0.00 | 0.60 |
| Gemma-2B-IT | RMU | Context-aware | 0.13 | **0.91** (+0.90) | 0.01 | **0.99** (+0.99) | 0.57 |
| Qwen3-8B | RMU | Vanilla | 0.10 | 0.18 | 0.00 | 0.05 | 0.59 |
| Qwen3-8B | RMU | Context-aware | 0.13 | **0.67** (+0.49) | 0.01 | **0.97** (+0.92) | 0.57 |

The most dramatic result is for RMU: the vanilla version completely fails Contextual QA (LLM-Judge $\leq 0.05$), while the context-aware version pushes it to $\geq 0.97$.

### Ablation Study

| Dimension | Configuration | Key Finding |
|------|------|------|
| Forget ratio | 1% / 5% / 10% | Vanilla consistently drops Contextual QA; context-aware reliably restores it across all ratios. |
| Context variant | Original / Paraphrase / Reasoning | Vanilla RMU produces gibberish for all; context-aware RMU handles all correctly, proving semantic recovery. |
| Model utility | With context term | Gemma avg -0.01, Qwen avg 0.00; almost zero cost for general capabilities. |
| $\lambda_c$ Sensitivity | Various methods | Stable across 0.01–2.0; very easy to tune. |
| Dataset Generalization | TOFU + PISTOL | Trends are identical across datasets. |

### Key Findings
- **Contextual suppression is a universal side effect**: 5 out of 6 SOTA methods fail Contextual QA. This failure is paradigm-agnostic.
- **RMU performs best in standard metrics but worst in contextual metrics**: Its activation perturbation "crushes" the representation space, destroying contextual conditioning. UNDIAL (re-labeling) has the least side effects but is weaker at forgetting.
- **KL anchor is a robust solution**: Its insensitivity to $\lambda_c$ makes it practical for deployment, and its robustness to paraphrase/reasoning context proves it isn't just shallow pattern matching.

## Highlights & Insights
- **The problem discovery is half the value**: Before this paper, the community used a two-axis evaluation (Direct QA + Utility). By revealing the hidden cost of "killing a neglected capability," this paper shifts the evaluation paradigm.
- **Elegance in "old medicine for new diseases"**: Using KL-to-reference is a textbook RLHF tool, but applying it to the specific $(q, c)$ input stream in an unlearning context is a non-trivial and elegant solution that occupies a previously empty "safety gap" in the loss topology.
- **Transferability**: The "Diagnosis—Anchoring—Decoupling" framework can likely be applied to other tasks like model editing or concept erasure where similar tensions exist.

## Limitations & Future Work
- **Limitations**: There is still a slight trade-off between "enhancing contextual utility" and "strengthening direct forgetting." Evaluation is limited to 8B scale models.
- **In-house Observations**: Contexts are currently derived from ground truth; real RAG scenarios involving long, noisy documents deserve more study. The need for $\mathcal{S}_f^{\text{ctx}}$ data for every forget request poses a streaming construction challenge for user-driven requests.
- **Future Directions**: (1) Extending context terms to sequence-level consistency; (2) Studying the link between contextual utility and jailbreak attacks; (3) Scaling to multi-document RAG and multi-turn dialogues.

## Related Work & Insights
- **vs. TOFU (Maini et al., 2024)**: This work adds a third "Contextual QA" axis to their forget+retain protocol, ensuring comparability with prior work while completing the dimension.
- **vs. Unlearning Methods (NPO/RMU)**: This paper does not compete with them but provides a patch; the discovery of a "paradigm-level disease" is a significant insight.
- **vs. Unlearning Reversal**: While other research uses context as an attack vector to recall knowledge, this paper treats context as a legitimate user input, aiming to preserve utility without expanding the attack surface.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Creating the Contextual QA dimension is a real conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 baselines, 2 models, 3 forget ratios, and multiple context variants.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear (Figure 1), and case studies are impactful.
- **Value**: ⭐⭐⭐⭐⭐ A near-essential patch for RAG-based LLM deployment with compliance requirements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](../../CVPR2026/llm_safety/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](../../CVPR2026/llm_safety/damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)

</div>

<!-- RELATED:END -->
