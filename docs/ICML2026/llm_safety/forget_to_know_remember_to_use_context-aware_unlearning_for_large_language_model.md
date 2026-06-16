---
title: >-
  [Paper Note] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
description: >-
  [ICML 2026][LLM Safety][LLM unlearning] This paper demonstrates that existing LLM unlearning methods, while "erasing knowledge from parameters," also destroy the model's "contextual utility"—the ability to correctly utilize that knowledge when it is re-provided in the prompt. The authors propose adding a KL regularization term to the existing unlearning loss
tags:
  - ICML 2026
  - LLM Safety
  - LLM unlearning
  - contextual utility
  - TOFU
date: 2026-05-08
content_hash: 41133ccb1cfa0a73
---
# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.17620](https://arxiv.org/abs/2510.17620)  
**Code**: Not public  
**Area**: LLM Safety / Machine Unlearning  
**Keywords**: LLM unlearning, contextual utility, KL regularization, TOFU, RAG-friendly unlearning

## TL;DR
This paper demonstrates that existing LLM unlearning methods, while "erasing knowledge from parameters," also destroy the model's "contextual utility"—the ability to correctly utilize that knowledge when it is re-provided in the prompt. The authors propose adding a KL regularization term to the existing unlearning loss—aligning the unlearned model's distribution on "question + context" inputs with the original model—to restore Contextual QA LLM-Judge scores from 0.00–0.84 back to 0.95+ with almost no loss in forgetting effectiveness or retain set utility.

## Background & Motivation

**Background**: LLMs are trained on web-scale corpora and inevitably ingest copyrighted content and personal privacy that must be "deleted." Since direct retraining is cost-prohibitive, unlearning methods have emerged: Gradient Ascent family (GradAscent / GradDiff), Preference Optimization family (NPO, DPO variants), Re-labeling family (UNDIAL), and Representation Modification family (RMU). Evaluation typically relies on two pillars: cleaning the forget set (low Direct QA scores) and preserving abilities on the retain set (no drop in model utility).

**Limitations of Prior Work**: The authors observe a third dimension ignored by the community—RAG and long-prompt scenarios where the model frequently receives "theoretically forgotten" content as input (e.g., user-uploaded documents, retrieved copyright segments). This is legally permissible because the information is provided by the user in-context rather than remembered by the model. However, existing unlearning methods fail on this "Contextual QA": RMU, GradAscent, and GradDiff drive Contextual QA scores to nearly zero on Gemma-2B-IT, while NPO and UNDIAL drop scores by over 15.5%. Case studies show output degradation ranging from "hallucinated countries" to pure gibberish like "denden den den...".

**Key Challenge**: All existing losses optimize the binary trade-off between "forget vs. retain," essentially penalizing parameter representations of $\mathcal{S}_f$. This penalty does not "target memory recall alone"—it spills over in the representation space to inference-time context conditioning. When the same tokens appear as context, the model loses its ability to ground its generation on them. Methods like RMU, which erase activation perturbations, are particularly severe as they directly damage the representation pathways for relevant concepts.

**Goal**: (1) Systematically quantify the side effects of 6 SOTA unlearning methods on Contextual QA; (2) Design a plug-and-play patch with minimal changes to original methods to recover contextual utility without compromising forgetting or overall utility.

**Key Insight**: RLHF has long proven that KL regularization can stabilize a model along certain behavioral dimensions relative to the original model. Since the problem is that the unlearned model behaves differently on $(q, c)$ inputs, one can use the original model as an anchor and apply a KL constraint on the $(q, c)$ data stream. This acts on a different distribution than the forget term ($\mathcal{S}_f$ vs. $\mathcal{S}_f^{\text{ctx}}$), preventing conflict between the objectives.

**Core Idea**: Add a third term $\lambda_c \cdot \mathrm{KL}(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c))$ to the standard unlearning loss. This explicitly anchors the "context-conditional distribution" to the original model, decoupling "parametric memory" from "contextual usage" at the loss level to achieve the principle of "do not recall from memory, but do use when provided."

## Method

### Overall Architecture
The paper presents a "diagnosis-then-repair" loop. In the diagnosis phase, 6 SOTA methods (GradAscent, GradDiff, NPO, DPO, UNDIAL, RMU) are evaluated on the TOFU benchmark (fictional author profiles) with a 5% forget ratio. Beyond standard Direct QA, a new evaluation line is added: given the same forget set questions, the ground truth is provided as context to see if the model can still answer correctly. In the repair phase, a patch is applied to all unlearning losses: a context set $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$ (where $c$ contains the answer) is constructed, and a KL regularization term $\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$ forces the model's predictions on this stream to align with the frozen original model $p_{\text{orig}}$. The objective function expands from two terms to three: $\mathcal{J}(w) = -\lambda_f L_f(\mathcal{S}_f, w) + \lambda_r L_r(\mathcal{S}_r, w) + \lambda_c \mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$.

### Key Designs

**1. Contextual QA Evaluation Protocol: Quantifying the Missing Dimension**
Traditional evaluation only checks for forgetting (low Direct QA) and utility preservation (high retain utility). However, as LLMs are increasingly used in RAG/long-prompt scenarios, the ability to use re-provided legitimate information is critical. This protocol pairs each $(q, a)$ in the forget set with a context $c$ containing the ground truth. Performance is measured by ROUGE-L (lexical overlap) and LLM-Judge (semantic correctness). To ensure models aren't just memorizing token patterns, the protocol includes paraphrase and reasoning variants of context. This exposed that 5 out of 6 methods produce errors or gibberish even when the answer is directly in the context.

**2. Context-aware KL Regularization Term: Anchoring the Unconstrained Dimension**
The core insight is that traditional losses only constrain the $\mathcal{S}_f$ and $\mathcal{S}_r$ distributions, leaving the contextual distribution unconstrained. Consequently, the forgetting penalty spills over and collapses the model's grounding ability on $(q, c)$ inputs. The remedy anchors this stream by constructing $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$ (where $c$ is derived from TOFU ground truth) and defining:
$$\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w) = \frac{1}{|\mathcal{S}_f^{\text{ctx}}|} \sum_{(q,a,c)} \mathrm{KL}\big(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c)\big)$$
This forces the token distribution under "question + context" prompts toward the original model. Using the pre-unlearned self as a reference eliminates the need for external teachers or extra labels. Since it does not modify the forget set loss, the forgetting strength is maintained while grounding behavior is stabilized.

**3. Plug-and-Play Integration**
Regardless of whether a method uses preference optimization (NPO), re-labeling (UNDIAL), or activation perturbation (RMU), they all follow a "forget term + optional retain term" structure. The new method simply appends $+\lambda_c \mathcal{C}$. This requires only one additional forward pass of the original model per step. The context term provides massive Contextual QA improvements across all paradigms (most notably RMU going from $0.00 \to 0.99$), proving that contextual suppression is a universal "disease" and the KL anchor is a universal "cure."

### Loss & Training
The final loss is $\mathcal{J}(w) = -\lambda_f L_f + \lambda_r L_r + \lambda_c \mathcal{C}$. Training follows standard TOFU settings (AdamW, extended from 5 to 20 epochs for convergence). The added hyperparameter $\lambda_c$ is stable across models (e.g., 2.0 / 0.01 / 0.5 for NPO/RMU/UNDIAL on Gemma-2B-IT). The convergence criterion is the simultaneous optimization of Direct QA, Contextual QA, and model utility.

## Key Experimental Results

### Main Results
On TOFU 5% forget ratio, comparing vanilla vs. context-aware unlearning:

| Model | Method | Variant | Direct ROUGE-L ↓ | Contextual ROUGE-L ↑ | Direct LLM-Judge ↓ | Contextual LLM-Judge ↑ | Utility ↑ |
|---|---|---|---|---|---|---|---|
| Gemma-2B-IT | NPO | Vanilla | 0.31 | 0.55 | 0.19 | 0.81 | 0.57 |
| Gemma-2B-IT | NPO | Context-aware | 0.36 | **0.87** (+0.32) | 0.25 | **0.98** (+0.17) | 0.57 |
| Gemma-2B-IT | RMU | Vanilla | 0.04 | 0.01 | 0.00 | 0.00 | 0.60 |
| Gemma-2B-IT | RMU | Context-aware | 0.13 | **0.91** (+0.90) | 0.01 | **0.99** (+0.99) | 0.57 |
| Qwen3-8B | NPO | Vanilla | 0.27 | 0.46 | 0.14 | 0.84 | 0.60 |
| Qwen3-8B | RMU | Vanilla | 0.10 | 0.18 | 0.00 | 0.05 | 0.59 |
| Qwen3-8B | RMU | Context-aware | 0.13 | **0.67** (+0.49) | 0.01 | **0.97** (+0.92) | 0.57 |

RMU shows the most dramatic shift: failing completely on vanilla Contextual QA (LLM-Judge $\leq 0.05$) but reaching $\geq 0.97$ with the context-aware version.

### Ablation Study

| Dimension | Configuration | Key Finding |
|---|---|---|
| Forget ratio | 1% / 5% / 10% | Vanilla consistently degrades Contextual QA; context-aware stabilizes it regardless of ratio. |
| Context variants | Paraphrase / Reasoning | Vanilla RMU outputs gibberish for all; context-aware RMU recovers semantic utility, not just surfacing patterns. |
| Direct QA Side-effect | With context term | Minor increase in Direct scores (~2-4pp), negligible compared to Contextual QA gains. |
| Model utility | With context term | Near-zero cost to general model performance (~ -0.01). |
| $\lambda_c$ Sensitivity | Across methods | Results are stable across a wide range (0.01–2.0), making tuning easy. |

### Key Findings
- **Contextual suppression is a universal side effect**: 5 out of 6 SOTA unlearning methods fail Contextual QA. This failure is paradigm-agnostic across GA, preference optimization, and activation perturbation.
- **RMU represents a new trade-off**: It performs best on standard metrics but worst on context metrics because its "heavy-handed" representation erasing destroys grounding pathways.
- **KL anchor is a robust solution**: Its effectiveness against paraphrased/reasoning contexts and low sensitivity to hyperparameters make it ideal for practical deployment.

## Highlights & Insights
- **Problem identification is half the value**: By exposing the hidden cost of unlearning with Figure 1 and "gibberish" case studies, the paper defines a critical new evaluation dimension for the community.
- **Elegant "Old Medicine" Application**: While KL-to-reference is a classic RLHF tool, its application to unlearning—specifically choosing the pre-unlearned model as the anchor and targeting the $(q, c)$ stream—is a non-trivial and effective design choice.
- **Paradigm-Agnostic Patch**: The ability to integrate with NPO, RMU, and UNDIAL ensures the work remains relevant as new unlearning methods emerge.

## Limitations & Future Work
- **Limitations**: A slight trade-off remains between contextual utility and strict Direct QA forgetting. Testing was limited to 8B models and synthetic benchmarks (TOFU, PISTOL).
- **In-depth limitations**: Contexts are derived from ground truths; real-world RAG involves noisier, longer documents. Constructing $\mathcal{S}_f^{\text{ctx}}$ on-the-fly for user-defined forget requests is not yet explored.
- **Future Directions**: Extending the framework to multi-document RAG and multi-turn dialogues; investigating if restored contextual utility opens new jailbreak attack surfaces.

## Related Work & Insights
- **vs. TOFU/MUSE**: Complements existing forget+retain axes with a third Contextual QA axis.
- **vs. Specific Methods**: Does not compete with NPO/RMU but provides a necessary patch for their inherent weaknesses.
- **vs. In-context Unlearning**: Those works use prompts to simulate forgetting; this work studies how parameter unlearning destroys in-context recall.
- **vs. Unlearning Reversal**: While attacks use context to elicit forgotten info, this work focuses on legitimate contextual use, though there is a natural tension between the two.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Creating the Contextual QA dimension is a significant conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across baselines, models, and variants.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem framing, impactful visuals, and well-structured loss analysis.
- **Value**: ⭐⭐⭐⭐⭐ A near-essential patch for unlearning in RAG-centric deployment environments.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster](../../ICML2025/llm_safety/system-aware_unlearning_algorithms_use_lesser_forget_faster.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](../../CVPR2026/llm_safety/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)

</div>

<!-- RELATED:END -->
