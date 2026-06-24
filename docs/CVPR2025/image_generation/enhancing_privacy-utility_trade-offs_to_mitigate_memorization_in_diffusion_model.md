---
title: >-
  [Paper Note] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Diffusion model memorization] This paper proposes the PRSS method, which achieves the optimal privacy-utility trade-off in mitigating diffusion model memorization without modifying the model or requiring training data during inference. It accomplishes this by improving the CFG formulation through two strategies: Prompt Re-anchoring (reusing the memorized prompt as a CFG anchor to guide generation away from memorized content) and Semantic Prompt S…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion model memorization"
  - "privacy protection"
  - "prompt re-anchoring"
  - "semantic prompt search"
  - "classifier-free guidance"
date: 2026-05-08
content_hash: 4e10548e5bfa4ccc
---

# Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2504.18032](https://arxiv.org/abs/2504.18032)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Diffusion model memorization, privacy protection, prompt re-anchoring, semantic prompt search, classifier-free guidance

## TL;DR
This paper proposes the PRSS method, which achieves the optimal privacy-utility trade-off in mitigating diffusion model memorization without modifying the model or requiring training data during inference. It accomplishes this by improving the CFG formulation through two strategies: Prompt Re-anchoring (reusing the memorized prompt as a CFG anchor to guide generation away from memorized content) and Semantic Prompt Search (using an LLM to search for semantically similar alternative prompts that do not trigger memorization).

## Background & Motivation

**Background**: Text-to-image diffusion models (e.g., Stable Diffusion, Midjourney) can generate highly realistic images, but they tend to memorize training data—partially or completely replicating training images during inference. When training data contains copyrighted or sensitive content, this poses severe legal and privacy risks, leading to multiple lawsuits against companies like Stability AI.

**Limitations of Prior Work**: Existing inference-stage mitigation strategies (such as prompt engineering) face severe privacy-utility trade-off difficulties. To enhance privacy (reduce memorization risk), user prompts must be significantly modified, causing the generated results to deviate from user intent (lowering utility/text alignment). Conversely, maintaining high text alignment fails to prevent memorization effectively. Although training-stage methods are theoretically viable, fine-tuning on the full LAION-5B dataset is impractical.

**Key Challenge**: The sole lever for improving privacy in the CFG formulation is modifying the prompt embedding—reducing the memorization probability by optimizing the prompt to lower the detection signal (magnitude). However, larger prompt modifications result in poorer text alignment. The root of the problem lies in: (1) the unconditional term $\epsilon_\theta(x_t, e_\phi)$ as an "anchor" contributes insufficiently to privacy protection; (2) the engineered prompt $e^*$ optimized via gradients severely deviates in semantics despite reducing the memorization signal.

**Goal**: (1) Find a more efficient privacy enhancement path than prompt engineering (achieving the same privacy boost with less utility loss); (2) Find privacy-safe alternative prompts that maintain semantic consistency; (3) Act synergistically to achieve the optimal trade-off across different privacy levels.

**Key Insight**: A deep analysis of the geometric structure of the CFG formulation reveals that different prompts correspond to different magnitude contour lines in the embedding space; points on the same contour line share the same privacy level but differ in utility. Re-anchoring the contrastive direction of CFG with the memorized prompt can more efficiently guide the generation away from the memorization path. Furthermore, searching in the language space with an LLM can find alternative prompts that are semantically similar but have lower magnitudes.

**Core Idea**: Replace the unconditional anchor in CFG with the memorized prompt (PR) to enhance privacy, and search for semantically equivalent, low-risk prompts (SS) using an LLM to guarantee utility. The two cooperate to optimize the privacy-utility trade-off.

## Method

### Overall Architecture
PRSS modifies the CFG formulation during inference without requiring training or fine-tuning. The workflow is: (1) the user inputs a prompt $e_p$; (2) at the first denoising step $T-1$, the magnitude $m_{T-1}$ is computed to judge whether a memorization risk is triggered ($m_{T-1} > \lambda$); (3) if safe, standard CFG is applied; (4) if a risk exists, an LLM (GPT-4) is first employed to search up to $n_s=25$ semantically similar alternative prompts $e_p^{ss}$ (early stopping when a magnitude < $\lambda$ is found), then the unconditional term of the CFG is replaced with the conditional prediction of the original prompt $e_p$ (Re-anchoring), and finally $e_p^{ss}$ is used as the target condition.

### Key Designs

1. **Prompt Re-anchoring (PR)**:

    - **Function**: Guide generation away from memorized content using a more efficient "contrastive direction" in CFG, achieving privacy enhancement at a lower utility cost.
    - **Mechanism**: Standard CFG guides generation from the unconditional prediction $\epsilon_\theta(x_t, e_\phi)$ toward the conditional prediction. PR defines the "undesired generation" as the conditional prediction of the memorized prompt $e_p$, replacing the unconditional anchor with it: $\hat{\epsilon} \leftarrow \epsilon_\theta(x_t, e_p) + s(\epsilon_\theta(x_t, e^{ss}_p) - \epsilon_\theta(x_t, e_p))$. Consequently, the contrastive direction of CFG shifts from "moving in any direction" to "specifically moving away from the memorization path." Geometrically, the guidance direction of PR points directly to the low-magnitude region, whereas the guidance direction of standard CFG is random.
    - **Design Motivation**: Baseline methods require more prompt optimization steps (modifying the prompt more) to lower the magnitude, leading to severe deviation from user intent. PR leverages the information of the memorized prompt—which precisely marks the "direction to move away from"—thus allowing the same privacy level to be achieved with fewer prompt modifications. Additionally, the effect of PR persists throughout the entire inference process, unlike the baseline which "lets go" after engineering the prompt in the first step, preventing the magnitude from rebounding in subsequent steps.

2. **Semantic Prompt Search (SS)**:

    - **Function**: Find alternative prompt choices in the language space that are semantically similar but have lower memorization risk, maximizing utility with minimal privacy cost.
    - **Mechanism**: Call the GPT-4 API to generate up to $n_s=25$ alternative texts semantically similar to the original prompt. Compute the first-step magnitude $m_{T-1}$ of each alternative prompt sequentially, and adopt it as soon as one falling below the threshold $\lambda$ is found. If all are above $\lambda$, select the one with the lowest magnitude. The search is conducted in the natural language space rather than the embedding space, naturally preserving semantic consistency. For example, "The No Limits Business Woman Podcast" $\rightarrow$ "The Empowered Business Woman's Podcast" reduces the magnitude from the original 7.48 to 0.78, while the CLIP similarity is significantly improved.
    - **Design Motivation**: Baseline prompt engineering optimizes prompt embeddings in the embedding space via gradient descent, which lowers the magnitude but causes severe semantic deviation. SS leverages the language understanding capabilities of LLMs to search for alternative solutions at the level of meaning, maintaining the core semantics of user intent.

3. **PR+SS Synergy**:

    - **Function**: Address each other's limitations to achieve the optimal trade-off across all privacy levels.
    - **Mechanism**: When SS finds a completely safe alternative prompt (magnitude $< \lambda$), using SS alone suffices. When SS cannot reduce the magnitude below the threshold, PR steps in to provide continuous memorization deflection. The key is that SS first lowers the baseline magnitude (e.g., from 7.48 to 6.02), requiring less "deflection work" from PR, resulting in less utility loss. The final CFG equation is $\hat{\epsilon} \leftarrow [\text{Standard CFG}]\mathbbm{1}_{m<\lambda} + [\epsilon_\theta(x_t, e_p) + s(\epsilon_\theta(x_t, e_p^{ss}) - \epsilon_\theta(x_t, e_p))]\mathbbm{1}_{m>\lambda}$.
    - **Design Motivation**: Analysis from a detection error perspective—a high $\lambda$ improves utility but increases false negatives (FN, missed memorized prompts), where PR mitigates the privacy risk of FNs; a low $\lambda$ enhances privacy but increases false positives (FP, misidentifying safe prompts), where SS mitigates the utility loss of FPs. The two complement each other to cover all scenarios.

### Loss & Training
- No training, fully inference-stage method.
- Replaceable detection signals: supports both original magnitude $m_{T-1}$ and enhanced masked magnitude $m'_{T-1}$.
- Extremely low LLM search cost: approx. 0.9 seconds of generation per alternative prompt, costing around \$0.02.
- Fully backward-compatible: makes no modifications to safe prompts ($m_{T-1} < \lambda$).

## Key Experimental Results

### Main Results

| Method | Detection Signal | Global Memorization SSCD ↓ | Text Alignment CLIP ↑ | Local Memorization SSCD ↓ | Text Alignment CLIP ↑ |
|------|---------|----------------|---------------|----------------|---------------|
| PE | $m$ | 0.35 | 0.23 | 0.42 | 0.24 |
| PE | $m'$ | 0.33 | 0.23 | 0.38 | 0.24 |
| **PRSS** | $m$ | **0.22** | **0.27** | **0.36** | **0.26** |
| **PRSS** | $m'$ | **0.18** | **0.28** | **0.33** | **0.27** |

Note: Values are representative points approximated from Figure 6 in the paper, compared under the same privacy level ($\lambda$).

### Ablation Study

| Configuration | Global SSCD ↓ | CLIP ↑ | Description |
|------|----------|-------|------|
| Standard SD | 0.65 | 0.30 | No mitigation |
| PE (baseline) | 0.35 | 0.23 | Prompt engineering only |
| PR only | 0.25 | 0.22 | High privacy but low utility |
| SS only | 0.30 | 0.28 | High utility but insufficient privacy |
| **PR+SS** | **0.22** | **0.27** | Optimal trade-off |

### Key Findings
- **PR shows prominent effectiveness in global memorization**: Compared to the baseline PE, PRSS yields much greater improvements under global memorization scenarios than local memorization. This is because the "deflection direction" for global memorization is clearer—the memorized prompt precisely locates the global patterns to be avoided.
- **SS is critical to preserving utility**: Although using PR alone yields the best privacy, the CLIP score decreases significantly. Incorporating SS substantially restores utility while further improving privacy.
- **The synergy of PR+SS is particularly evident in local memorization**: Local memorization is harder to mitigate, and PR or SS alone yields limited performance, but combining them brings significant improvements.
- **PRSS can seamlessly integrate better detection signals**: PRSS performance is further enhanced when upgrading from $m$ to $m'$, showcasing its excellent modular design.
- Qualitative cases demonstrate that the alternative prompts discovered by SS successfully bypass memorization triggers while preserving the semantic core.

## Highlights & Insights
- **Profound geometric analysis of the CFG formulation**: Visualizing the privacy-utility trade-off as shifts along magnitude contour lines clearly reveals why baseline methods are inefficient (moving in suboptimal directions) and highlights the improvement mechanisms of PR/SS (altering the movement direction/starting point). This analytical framework is transferable to the design of other CFG variants.
- **An intuitive yet counter-intuitive design of "using the memorized prompt itself to combat memorization"**: While baselines discard the memorized prompt, PRSS retains it as an anchor—since it precisely marks the direction to stay away from. This is a creative reuse of the "positive-negative contrast" concept in CFG.
- **LLM-assisted semantic search requires absolutely no training data**: Safe alternative prompts are obtained solely via API calls, preserving privacy (without exposing the training set) at a very low cost. This "LLM-as-a-tool" paradigm can be extended to other scenarios requiring semantically equivalent transformations.
- **Highly modular methodology**: The detection signals, search strategies, and anchoring methods can all be independently replaced and upgraded.

## Limitations & Future Work
- **Dependence on detection accuracy**: Like all baselines, PRSS degrades to standard SD when the detection signal $m_{T-1}$ fails. Detection accuracy serves as the bottleneck of the entire framework.
- **Upper bound of semantic search**: The $n_s=25$ alternative prompts generated by the LLM might still all trigger memorization, especially for highly unique concepts (e.g., names of individuals, brands). Prolonging the search or combining it with prompt embedding optimization may help.
- Experiments are conducted solely on Stable Diffusion v1-4, without validating performance on newer models like SDXL or SD3.
- The test set of 500 prompts is relatively small and may not cover all types of memorization.
- The behavioral stability of PR across multiple inferences is not fully discussed—what is the variance of the effects under different random seeds?

## Related Work & Insights
- **vs. PE (Wen et al.)**: PE only optimizes prompt embeddings via gradient descent to lower magnitude, serving as a direct baseline for PRSS. PRSS changes the CFG anchor (PR) and prompt search space (SS) on top of PE, achieving comprehensive superiority.
- **vs. BEA (Chen et al.)**: BEA proposes masked magnitude as a better detection signal and a local memorization mask. PRSS seamlessly adopts this signal to further refresh the SOTA, demonstrating that PRSS as a mitigation strategy is orthogonal to detection methods.
- **vs. Anti-Memorization (Somepalli et al.)**: This approach focuses on training data deduplication, which is computationally expensive and has limited efficacy. PRSS operates entirely during the inference phase without needing access to any training data.
- **vs. Negative Prompting**: Negative prompting is commonly used in SD inference but remains heuristic. The PR strategy of PRSS can be viewed as a theoretically grounded "negative prompt"—using the memorized prompt as a systematic negative guidance.

## Rating
- Novelty: ⭐⭐⭐⭐ PR and SS improve from the privacy and utility ends respectively, with complementary and synergistic designs. The geometric analytical framework is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple detection signals + global/local memorization + ablations + qualitative cases, but the test set is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth analysis, intuitive illustrations, and progressive logical steps.
- Value: ⭐⭐⭐⭐ Direct significance for privacy protection in practically deployed diffusion models, with a simple and deployable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)
- [\[CVPR 2025\] Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](classifier-free_guidance_inside_the_attraction_basin_may_cause_memorization.md)
- [\[ICCV 2025\] Trade-offs in Image Generation: How Do Different Dimensions Interact?](../../ICCV2025/image_generation/trade-offs_in_image_generation_how_do_different_dimensions_interact.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)

</div>

<!-- RELATED:END -->
