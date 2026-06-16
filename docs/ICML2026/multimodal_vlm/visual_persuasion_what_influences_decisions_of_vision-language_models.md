---
title: >-
  [Paper Note] 视觉说服力：什么影响了视觉-语言模型的决策？
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] This paper discovers that VLMs exhibit significant visual preferences by systematically using image editing models to modify visual attributes while maintaining semantic consistency. It proposes three visual prompt optimization methods to reveal these preferences, develops an automated interpretability pipeline to unde
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 8d21b8df45445bbf
---
# Visual Persuasiveness: What Influences the Decisions of Vision-Language Models?

**Conference**: ICML 2026  
**arXiv**: [2602.15278](https://arxiv.org/abs/2602.15278)  
**Code**: https://github.com/MaggieCherepLabs  
**Area**: Multimodal VLM  
**Keywords**: Visual Persuasiveness, VLM Decision Making, Visual Preference, Prompt Optimization, Interpretability

## TL;DR
This paper discovers that VLMs exhibit significant visual preferences by systematically using image editing models to modify visual attributes while maintaining semantic consistency. It proposes three visual prompt optimization methods to reveal these preferences, develops an automated interpretability pipeline to understand the visual themes driving decisions, and mitigates risks through visual normalization.

## Background & Motivation

**Background**: Current VLM evaluations primarily focus on functional metrics, but in practical applications, VLMs are deployed as agentic systems making critical decisions—such as recommending products, screening candidates, and evaluating real estate.

**Limitations of Prior Work**: Existing VLM evaluations lack a deep understanding of the structure of the models' visual preferences. Research indicates that LLM agents are highly sensitive to text prompts, but the vulnerability of VLM visual preferences remains poorly understood. When these models operate autonomously, hidden visual preferences could be exploited maliciously or lead to large-scale bias.

**Key Challenge**: How can VLM visual preferences be systematically discovered and quantified? Traditional approaches, such as collecting large datasets of natural variations, are costly and provide incomplete coverage.

**Goal**: (1) Develop a systematic method to expose VLM visual preferences; (2) Quantitatively evaluate the impact of these preferences on model decision-making; (3) Identify and interpret the visual themes that drive decisions; (4) Propose mitigation strategies.

**Key Insight**: Current image editing models (e.g., Gemini 3, Qwen-Image-Edit) possess fine-grained visual controllability. These models can be used to iteratively modify images based on VLM pairwise selection feedback to optimize the editing direction—essentially exploring the model's hidden utility function.

**Core Idea**: Treat the VLM decision function as a hidden visual utility landscape and infer this landscape through "revealed preference"—systematic editing and pairwise comparison.

## Method

### Overall Architecture
The framework consists of three stages: (1) **Visual Prompt Optimization**: Starting from an original image, an image editing model iteratively modifies the image based on optimization feedback until a local equilibrium is reached; (2) **Automated Interpretability Pipeline**: High-level visual themes are abstracted from the differences between optimized and original images using multi-stage aggregation (Matryoshka summarization); (3) **Mitigation and Validation**: Testing the effectiveness of visual normalization.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Original Image x₀ + Decision Task<br/>Product Recommendation / Real Estate / Screening / Hotel"] --> S1

    subgraph S1["Visual Prompt Optimization (Design 1+2)"]
        direction TB
        B["Edit Prompt p → Image Editing Model<br/>Identity Constraint C(x₀): Modify representation only"] --> C["VLM Pairwise Preference Feedback<br/>Estimate Utility U(x(p))"]
        C --> D{"CVPO Competition: Dual Prompts pA / pB<br/>Win Rate ≈ 50% Reach Local Equilibrium?"}
        D -->|No, continue iteration| B
    end

    D -->|Yes, converged| E["Optimized Image Set (vs. Original)"]
    E --> F["Matryoshka Interpretability Pipeline<br/>Pairwise Differences → Clustering → LLM Summary → Visual Themes"]
    F --> G["Mitigation & Validation<br/>Visual Normalization reduces preference advantage"]
```

### Key Designs

**1. Optimization under Identity Constraints: Modifying Representation, Not Identity**

To probe VLM visual preferences, edited images must remain the "same object"; otherwise, the optimizer might simply substitute the object for one the VLM inherently prefers. This study enforces an identity constraint set $\mathcal{C}(x_0) = \{x \in \mathcal{X}: I(x, x_0) = 1\}$, formulating the optimization as $\max_{p \in \mathcal{P}} U_\tau(x(p))$ s.t. $x(p) \in \mathcal{C}(x_0)$. This ensures that searches within the "semantic-preserving" image subset only maximize VLM utility $U_\tau$ by adjusting presentation (lighting, angle, layout, style). Consequently, the discovered preferences reflect tendencies toward "how" something is presented rather than "what" it is.

**2. Three Competitive Visual Prompt Optimization Methods (VTG/VFD/CVPO): Probing Utility Functions under Noisy Feedback**

VLM preference feedback is pairwise and noisy, making stable convergence in discrete edit prompt spaces difficult. Three methods employ a "propose-evaluate" cycle with varying stopping mechanisms. VisualTextGrad (VTG) uses an LLM critic for textual gradient feedback but struggles to stop effectively in noisy environments. VisualFeedbackDescent (VFD) employs a multi-critic voting system for stability but averages 24.9 iterations. The proposed Competitive Visual Prompt Optimization (CVPO) models optimization as a competition between two prompts $p_A, p_B$. Consistency checks are performed by $k$ judges; when the win rate approaches 50% (indicating local equilibrium), the process stops. This condition allows CVPO to average only 17.4 iterations, reducing costs by 63% while achieving superior performance across most models.

**3. Matryoshka Multi-stage Interpretability Pipeline: Abstracting Pixel Differences into Readable Visual Themes**

Identifying which image a VLM prefers is insufficient; the specific visual features driving that preference must be articulated. This pipeline uses two-stage recursive abstraction: first, a VLM compares original and optimized images to generate fine-grained difference descriptions; second, these descriptions are embedded, clustered by similarity, and summarized by an LLM. The "Matryoshka" structure refers to high-level clusters being summaries of lower-level summaries, maintaining traceability while generating explanations for thousands of optimized images. Notably, different optimization methods often converge to similar visual themes, suggesting these preferences are stable properties of the VLM itself.

## Key Experimental Results

### Main Results: Evaluation of Optimization Effectiveness

| Dataset/Task | Original Image | Zero-shot Edit | Optimized | Gain (Rel. to Original) |
|------------|--------|----------|-------|--------------|
| Product Recommendation | 0.27 ± 0.03 | 0.48 ± 0.02 | 0.55 ± 0.02 | +78% |
| Real Estate Search | 0.31 ± 0.02 | 0.51 ± 0.02 | 0.62 ± 0.02 | +100% |
| Candidate Screening | 0.29 ± 0.03 | 0.47 ± 0.02 | 0.58 ± 0.02 | +100% |
| Hotel Booking | 0.26 ± 0.03 | 0.52 ± 0.02 | 0.61 ± 0.02 | +135% |

### Ablation Study: Comparison of Optimization Methods

| VLM | VTG | VFD | CVPO | Best-Second Difference |
|-----|-----|-----|------|-------------|
| Qwen-3-VL 235B | 0.131 | 0.601 | 0.771 | +0.170 |
| GPT-5 Mini | 0.190 | 0.561 | 0.766 | +0.205 |
| Gemini 3 Flash | 0.140 | 0.604 | 0.761 | +0.157 |
| GPT-4o | 0.179 | 0.566 | 0.749 | +0.183 |
| Claude Sonnet 4.5 | 0.310 | 0.603 | 0.594 | -0.010 |

### Key Findings
- Zero-shot editing is significantly effective—basic prompts can increase selection probability by 0.2-0.4.
- CVPO is the most stable performer—outperforming VFD on 7 out of 9 VLMs.
- Significant efficiency differences—VTG used 100% of the budget, VFD 74.6%, and CVPO only 36.9%.
- Human study validation (N=154): CVPO-optimized results ranked highest in human head-to-head comparisons.
- Visual theme convergence—similar themes across different optimization methods imply stable VLM properties.
- Incompleteness of mitigation strategies—visual normalization reduces the advantage but cannot eliminate it entirely.

## Highlights & Insights
- **Methodological Innovation**: This is the first systematic extension of prompt optimization to the visual domain, featuring a clever CVPO competitive framework and equilibrium-based stopping conditions.
- **Multi-layered Evidence**: Supported by 1.8M+ API calls, 125k+ generated images, 4 task domains, human validation, and automated interpretability.
- **Key Insight on "Hidden Optimization of Presentation"**: The research reveals a critical AI governance risk—if image optimization is used maliciously, it can systematically manipulate VLM agent decisions.
- **Reusable Design Patterns**: Matryoshka summarization, identity constraints, and competitive frameworks are all broadly applicable.

## Limitations & Future Work
- High computational requirements limit scalability.
- Vague boundaries for identity maintenance (e.g., ethical tensions regarding background and attire).
- Limited scale of human validation (N=154).
- Prompt distillation using the same set of optimized images may impact external validity.
- Future Work: Research VLM adversarial robustness training; develop visual auditing tools; extend to multimodal scenarios; study variations in VLM preferences.

## Related Work & Insights
- **vs. Adversarial Examples**: Adversarial research seeks minimal perceptible perturbations; this work focuses on perceptually significant but semantically preserved natural variations.
- **vs. Behavioral ML & Agent Evaluation**: Prior work focused on the text domain; this study extends to the visual domain and develops a systematic discovery method.
- **vs. Prompt Optimization (TextGrad, Feedback Descent)**: This work extends the principles of feedback gradients to the multimodal domain.
- **vs. Automated Interpretability**: This study uses similar concepts to explain black-box VLM behavior, serving as a complementary external interpretability method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic use of visual prompt optimization to explore hidden VLM preferences.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with a complete chain of evidence; small human sample size is the only minor drawback.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with detailed methodological explanation.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for AI safety and governance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VisionPulse：多模态推理中的动态视觉稀疏化](visionpulse_dynamic_visual_sparsity_for_efficient_multimodal_reasoning.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ICML 2026\] Find, Fix, Reason: Context Repair for Video Reasoning](find_fix_reason_context_repair_for_video_reasoning.md)

</div>

<!-- RELATED:END -->
