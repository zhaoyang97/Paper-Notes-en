---
title: >-
  [Paper Note] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
description: >-
  [ICML 2026][AI Safety][LLM unlearning] This paper points out that existing LLM unlearning methods, while "erasing knowledge from parameters," also destroy the "contextual utility"—the ability of the model to correctly utilize that knowledge when it is re-provided in the prompt. The authors propose adding a KL regularization term to existing unlearning losses—aligning the distribution of the unlearned model on "question + context" inputs with the original model—effectively res…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "LLM unlearning"
  - "contextual utility"
  - "KL regularization"
  - "TOFU"
  - "RAG-friendly unlearning"
date: 2026-05-08
content_hash: 4201438566cbec07
---

# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.17620](https://arxiv.org/abs/2510.17620)  
**Code**: Not disclosed  
**Area**: LLM Safety / Machine Unlearning  
**Keywords**: LLM unlearning, contextual utility, KL regularization, TOFU, RAG-friendly unlearning

## TL;DR
This paper points out that existing LLM unlearning methods, while "erasing knowledge from parameters," also destroy the "contextual utility"—the ability of the model to correctly utilize that knowledge when it is re-provided in the prompt. The authors propose adding a KL regularization term to existing unlearning losses—aligning the distribution of the unlearned model on "question + context" inputs with the original model—effectively restoring Contextual QA LLM-Judge scores from 0.00–0.84 back to 0.95+ with almost no loss in forgetting effectiveness or retain set utility.

## Background & Motivation

**Background**: LLMs are trained on web-scale corpora and inevitably ingest information that needs to be "deleted," such as copyrighted content or personal privacy. Since direct retraining is prohibitively expensive, a suite of unlearning methods has emerged: gradient ascent families (GradAscent / GradDiff), preference optimization families (variants of NPO, DPO), re-labeling families (UNDIAL), and representation perturbation families (RMU). Evaluation standards typically focus on two pillars: forgetting the forget set (low Direct QA scores) and preserving capabilities on the retain set (no drop in model utility).

**Limitations of Prior Work**: The authors observe a third dimension ignored by the community—contextual utility. In RAG and long-prompt scenarios, models often receive "theoretically forgotten" content as input (e.g., user-uploaded documents, retrieved copyrighted chapters). This is legally permissible because the information is provided by the user in real-time, not remembered by the model. However, existing unlearning methods fail on such "open-book" Contextual QA tasks where both the question and answer are present. On Gemma-2B-IT, RMU/GradAscent/GradDiff drive Contextual QA scores nearly to zero, while NPO/UNDIAL show drops of 15.5%+. Case studies show outputs degrading from "hallucinations of a different country" to pure gibberish like "denden den den...".

**Key Challenge**: All current losses focus on the binary "forget vs. retain" trade-off, essentially penalizing parameter representations of $\mathcal{S}_f$. This penalty does not "only punish memory recall"—it overflows into the representation space for inference-time context conditioning. When the same tokens appear as context, the model loses the ability to ground itself on those tokens to generate correct answers. Methods like RMU, which erase via activation perturbation, are particularly severe as they directly disrupt the representation pathways of related concepts.

**Goal**: (1) Systematically quantify the side effects of 6 SOTA unlearning methods on Contextual QA; (2) Design a "plug-and-play" patch with minimal changes to original methods to recover contextual utility without compromising forgetting or general utility.

**Key Insight**: RLHF has long proven that KL regularization can prevent a model from deviating from the original model across certain behavioral dimensions. Since the problem is that the model behavior changes under $(q, c)$ inputs after unlearning, one can simply use the original model as an anchor and add a KL constraint on this $(q, c)$ data stream. This term acts on a different input distribution ($\mathcal{S}_f^{\text{ctx}}$) than the forget term ($\mathcal{S}_f$), preventing direct conflict.

**Core Idea**: Add a third term to standard unlearning losses: $\lambda_c \cdot \mathrm{KL}(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c))$. This explicitly anchors the "contextual conditional distribution" to the original model, decoupling "parameter memory" from "contextual usage" at the loss level to achieve "do not recall from memory, but do use when provided."

## Method

### Overall Architecture
The work follows a "diagnose then repair" closed loop. In the diagnosis phase, 6 SOTA unlearning methods (GradAscent, GradDiff, NPO, DPO, UNDIAL, RMU) are run on the TOFU benchmark (fictional author profiles to ensure no prior exposure) with a 5% forget ratio. A new evaluation line is added beyond standard Direct QA: the same forget set questions are presented with the ground truth provided as context to see if the model can answer when the "answer is right there." Results reveal that current methods fail even this. In the repair phase, the same patch is applied to all unlearning losses: constructing a contextual set $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$ (where $c$ is context containing the answer) and adding a KL regularization term $\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$ to force the unlearning model's prediction distribution on this contextual stream to align with the frozen original model $p_{\text{orig}}$. The objective function expands from two terms to three: $\mathcal{J}(w) = -\lambda_f L_f(\mathcal{S}_f, w) + \lambda_r L_r(\mathcal{S}_r, w) + \lambda_c \mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$, managing forget, retain, and contextual utility respectively.

### Key Designs

**1. Contextual QA Evaluation Protocol: Quantifying the Missing Third Dimension**
Previous evaluations only considered two pillars—cleanly forgetting (low Direct QA) and maintaining general ability (high utility). However, as LLMs are increasingly deployed in RAG/long-prompt scenarios, whether "legitimate information re-provided by the user" can be used is a critical deployment concern. This protocol adds that dimension: for each $(q, a)$ in the forget set, a context $c$ (containing the ground truth description) is paired. The "Question + Context" prompt is fed to the unlearned model, using ROUGE-L for lexical overlap and LLM-Judge (validated with human consistency in the Appendix) for semantic correctness, both scaled to $[0, 1]$. An ideal unlearned model should achieve low Direct QA, high Contextual QA, and high utility. To prevent the model from simply memorizing the context's surface form, the protocol includes paraphrase and reasoning context variants. This protocol immediately exposed the issue: Table 1 shows 5 out of 6 methods producing errors or gibberish even when the answer is in the context.

**2. Context-aware KL Regularization: Anchoring with RLHF Techniques**
The core insight is that original two-term losses only constrain the $\mathcal{S}_f$ and $\mathcal{S}_r$ distributions, leaving the third contextual distribution unconstrained. Consequently, the forget term's penalty overflows through the representation space, collapsing the model's grounding ability under $(q, c)$ inputs. The remedy is to anchor this third data stream: $S_f^{\text{ctx}} = \{(q, a, c)\}$ is constructed (where $c$ is derived from TOFU ground truth, equivalent to an "answer statement," requiring no extra labeling), defined as:

$$\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w) = \frac{1}{|\mathcal{S}_f^{\text{ctx}}|} \sum_{(q,a,c)} \mathrm{KL}\big(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c)\big),$$

forcing the current model's token distribution under "Question + Context" to approach the frozen original model. The elegance of this approach lies in the choice of anchor and scope: KL-to-reference is a textbook tool in RLHF for stabilizing behavior. Here, the "reference" is the pre-unlearning self (no external teacher or labels needed), and the "constrained behavior" is strictly limited to the $(q, c)$ input stream (thus orthogonal to $\mathcal{S}_f$ and $\mathcal{S}_r$). Because it does not modify the forget set loss, the unlearning strength remains determined by the original method. Since KL is a distribution-level rather than point-to-point distillation constraint, it is gentler than hard token-level targets and does not engage in a "death match" with the forget term. In practice, Appendix A.6 confirms extreme insensitivity to $\lambda_c$, with NPO/RMU/UNDIAL remaining stable across 0.01–2.0 on both models.

**3. Plug-and-play Integration: One Context Term for Three Paradigms**
The unlearning community is highly fragmented with new methods appearing annually. A patch is only valuable if it is not tied to a specific loss. Fortunately, existing methods—whether preference optimization (NPO), re-labeling (UNDIAL), or activation perturbation (RMU)—follow a two-term structure of "forget term + optional retain term." The new method simply appends $+\lambda_c \mathcal{C}$—adding one forward pass of the original model (to calculate $p_{\text{orig}}(\cdot|q,c)$) and one KL calculation per step. The only new hyperparameter $\lambda_c$ is assigned once per method/model. The same context term yields massive Contextual QA gains across these distinct paradigms (RMU jumping from $0.00 \to 0.99$ is the most dramatic), proving both that contextual suppression is a universal pathology and that the KL anchor is a universal cure.

### Loss & Training
The final loss is $\mathcal{J}(w) = -\lambda_f L_f + \lambda_r L_r + \lambda_c \mathcal{C}$. Training follows standard TOFU settings (AdamW, extended from 5 to 20 epochs to ensure full convergence). The only addition is $\lambda_c$: for Gemma-2B-IT, NPO/RMU/UNDIAL use 2.0 / 0.01 / 0.5; for Qwen3-8B, 1.0 / 0.5 / 1.0. Convergence is defined when Direct QA LLM-Judge, Contextual QA LLM-Judge, and model utility simultaneously approach the global best for that method's curve.

## Key Experimental Results

### Main Results
TOFU 5% forget ratio, comparison between vanilla vs. context-aware unlearning:

| Model | Method | Variant | Direct ROUGE-L ↓ | Contextual ROUGE-L ↑ | Direct LLM-Judge ↓ | Contextual LLM-Judge ↑ | Utility ↑ |
|---|---|---|---|---|---|---|---|
| Gemma-2B-IT | NPO | Vanilla | 0.31 | 0.55 | 0.19 | 0.81 | 0.57 |
| Gemma-2B-IT | NPO | Context-aware | 0.36 | **0.87** (+0.32) | 0.25 | **0.98** (+0.17) | 0.57 |
| Gemma-2B-IT | RMU | Vanilla | 0.04 | 0.01 | 0.00 | 0.00 | 0.60 |
| Gemma-2B-IT | RMU | Context-aware | 0.13 | **0.91** (+0.90) | 0.01 | **0.99** (+0.99) | 0.57 |
| Gemma-2B-IT | UNDIAL | Vanilla | 0.33 | 0.53 | 0.39 | 0.82 | 0.54 |
| Gemma-2B-IT | UNDIAL | Context-aware | 0.34 | **0.87** (+0.34) | 0.38 | **0.98** (+0.16) | 0.55 |
| Qwen3-8B | NPO | Vanilla | 0.27 | 0.46 | 0.14 | 0.84 | 0.60 |
| Qwen3-8B | NPO | Context-aware | 0.29 | 0.63 (+0.17) | 0.20 | **0.95** (+0.11) | 0.61 |
| Qwen3-8B | RMU | Vanilla | 0.10 | 0.18 | 0.00 | 0.05 | 0.59 |
| Qwen3-8B | RMU | Context-aware | 0.13 | 0.67 (+0.49) | 0.01 | **0.97** (+0.92) | 0.57 |
| Qwen3-8B | UNDIAL | Vanilla | 0.32 | 0.59 | 0.38 | 0.97 | 0.60 |
| Qwen3-8B | UNDIAL | Context-aware | 0.33 | 0.68 (+0.09) | 0.39 | 0.98 (+0.01) | 0.61 |

The most dramatic result is RMU: while vanilla almost completely fails Contextual QA (LLM-Judge $\leq 0.05$), the context-aware version reaches $\geq 0.97$ on both models.

### Ablation Study

| Dimension | Configuration | Key Finding |
|---|---|---|
| Forget ratio | 1% / 5% / 10% | Vanilla drops Contextual QA significantly across all ratios; context-aware consistently recovers it. KL anchor efficacy is unaffected by forget difficulty. |
| Context Variants (RMU) | Original / Paraphrase / Reasoning | Vanilla RMU produces gibberish across all; context-aware RMU generates correct answers for all—proving it restores semantic utility, not surface memorization. |
| Direct QA Side Effect | After adding context term | Average ROUGE-L change $\sim 4$pp, LLM-Judge change $\sim 2$pp (Gemma); on Qwen, $\sim 2$pp and $\sim 3$pp. Magnitude is far smaller than Contextual QA gains. |
| Model utility | After adding context term | Gemma average -0.01, Qwen average 0.00. Almost zero cost. |
| $\lambda_c$ Sensitivity | Per method/model | Stable across 0.01–2.0 for all three methods. Extremely easy to tune. |
| Dataset Generalization | TOFU + PISTOL | Consistent trends across both datasets (PISTOL contains structurally entangled entities). |
| Noisy context | GPT-generated long paragraphs / conflicts | Contextual QA decreases as context noise increases, but context-aware gains over vanilla remain significant. |

### Key Findings
- **Contextual suppression is a common side effect of unlearning**: 5 out of 6 SOTA methods severely fail on Contextual QA (RMU/GradAscent/GradDiff hit zero on Gemma). This failure is paradigm-agnostic, affecting gradient ascent, preference optimization, and activation perturbation.
- **RMU performs best in standard metrics but worst in new metrics**: By using activation perturbation to erase concept representations, its "heavy-handed" impact on the representation space most thoroughly destroys contextual conditioning. UNDIAL, being re-labeling based rather than punitive, has the least side effect but weaker forgetting. This reveals a new trade-off in unlearning paradigm selection.
- **The KL anchor is a stable lifeline**: Insensitivity to $\lambda_c$ makes deployment easy; robustness to paraphrasing/reasoning indicates genuine restoration of semantic utility; minimal interference with unlearning loss allows concatenation with any future method.

## Highlights & Insights
- **The problem discovery is half the paper's value**: Before this work, the community relied on a two-pillar (Direct QA + utility) evaluation. Figure 1 exposes the hidden cost of "unlearning killing a vital capability," and the case study's visual impact (gibberish vs. hallucination) reinforces this, following the paradigm of "opening new evaluation dimensions to drive method improvement."
- **The elegance of "old medicine for new diseases"**: KL-to-reference is a textbook RLHF tool, but its application to unlearning—specifically the choice of anchor (pre-unlearning self vs. external teacher) and the target data stream (contextual vs. forget)—is non-trivial. By precisely anchoring the $(q, c)$ stream, the authors find a previously unoccupied "safety gap" in the loss topology.
- **Recoverable framework for other erasure tasks**: Model editing and concept erasure face similar tensions between "parameter-level deletion vs. context-level retention." The "diagnose-anchor-decouple" workflow can be applied broadly.

## Limitations & Future Work
- Authors' acknowledged limitations: A slight trade-off remains between contextual utility and Direct QA forgetting; weights depend on deployment scenarios (strict deletion vs. RAG); evaluation limited to TOFU/PISTOL and models up to 8B; lack of theoretical analysis.
- Independently identified limitations: Contexts derived from TOFU ground truth are essentially "answer statements," which differ from real-world RAG (long docs + noise). While Appendix includes noisy experiments, they are still synthetic. Constructing $\mathcal{S}_f^{\text{ctx}}$ poses a challenge if forget requests are user-driven and real-time. The KL anchor is at the token distribution level—longer generations might still drift.
- Future directions: (1) Extend the context term from KL to sequence-level consistency (e.g., sequence-level KD or self-consistency rewards); (2) Study the relationship between contextual utility and jailbreak attacks—does sensitivity to "user-provided forget info" open an attack vector? (3) Expand to multi-document RAG and multi-turn dialogue to verify if anchoring $(q, c)$ remains sufficient.

## Related Work & Insights
- **vs. TOFU (Maini et al., 2024) / MUSE (Shi et al., 2025)**: These established the forget+retain protocol; this paper adds the third axis (Contextual QA), complementing rather than replacing them.
- **vs. NPO / RMU / UNDIAL**: This paper does not compete with them but provides a patch—the fact that three different paradigms benefit proves contextual suppression is a paradigm-level pathology.
- **vs. In-context unlearning (Pawelczyk et al., 2024)**: Those works use context as an unlearning tool ("pretending to forget with a prompt"); this work does the opposite—studying how parameter-level unlearning destroys in-context utility.
- **vs. Unlearning reversal (Shumailov et al., 2024)**: That line treats context as an attack vector to recover knowledge; this paper treats it as legitimate user input—balancing legitimate use without opening attack surfaces is a key future tension.

## Rating
- Novelty: ⭐⭐⭐⭐ Opening the Contextual QA dimension is a genuine conceptual contribution; the KL anchor is technically established but applied effectively to a new problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 baselines × 2 models × 3 ratios × 4 context variants + 2 datasets; scale capped at 8B is the only minor drawback.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 explains the problem clearly; case studies are impactful; loss decomposition is concise.
- Value: ⭐⭐⭐⭐⭐ Almost a mandatory patch for RAG-enabled unlearning deployments with an extremely low barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICLR 2026\] Secure Outlier-Aware Large Language Model Inference](../../ICLR2026/ai_safety/secure_outlier-aware_large_language_model_inference.md)
- [\[ICML 2026\] BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)
- [\[ICLR 2026\] Machine Unlearning under Retain–Forget Entanglement](../../ICLR2026/ai_safety/machine_unlearning_under_retainforget_entanglement.md)

</div>

<!-- RELATED:END -->
