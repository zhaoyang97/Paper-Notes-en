---
title: >-
  [Paper Note] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation
description: >-
  [ACL2026][Multimodal VLM][LVLM self-evaluation] This paper proposes VAUQ, which uses Image-Information Scores and attention-driven core region masking to measure whether LVLM responses truly rely on visual evidence…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "LVLM self-evaluation"
  - "uncertainty quantification"
  - "hallucination detection"
  - "visual evidence"
  - "attention masking"
date: 2026-05-08
content_hash: 27ef6488b8382d81
---

# VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation

**Conference**: ACL2026  
**arXiv**: [2602.21054](https://arxiv.org/abs/2602.21054)  
**Code**: https://github.com/deeplearning-wisc/vauq  
**Area**: multimodal_vlm  
**Keywords**: LVLM self-evaluation, uncertainty quantification, hallucination detection, visual evidence, attention masking  

## TL;DR
This paper proposes VAUQ, which uses Image-Information Scores and attention-driven core region masking to measure whether LVLM responses truly rely on visual evidence, enabling more reliable multimodal self-evaluation and hallucination detection without requiring training or external evaluators.

## Background & Motivation
**Background**: Large Vision-Language Models (LVLMs) can perform open-ended VQA, visual reasoning, and image-text dialogue, yet they frequently generate hallucinations. To detect unreliable responses at deployment, one class of methods allows models to perform self-evaluation using internal signals such as perplexity, predictive entropy, semantic entropy, verbalized confidence, or hidden state transitions.

**Limitations of Prior Work**: Most of these methods are derived from pure Large Language Models (LLMs). They measure whether the model is confident in its textual output but do not necessarily measure whether the response is supported by the image. LVLMs may be highly confident in incorrect answers due to strong language priors, such as answering based on common sense even when seeing counterfactual images. In such cases, low entropy or high verbalized confidence only indicates "linguistic fluency" rather than "visual correctness."

**Key Challenge**: Multimodal self-evaluation must simultaneously handle two types of uncertainty: the uncertainty of language generation itself and the uncertainty of whether visual evidence is correctly utilized. Looking only at the output distribution ignores visual grounding; looking only at visual attention fails to judge whether the final answer is correct. A reliable score needs to combine "whether the prediction is uncertain" with "whether the image truly reduces that uncertainty."

**Goal**: The authors aim to design a training-free, label-free, response-level self-evaluation score for LVLMs. It should not depend on external judges, require multiple samples, or be limited to detecting single-object hallucinations, but rather judge whether the entire response is likely to be incorrect or hallucinated.

**Key Insight**: The core observation of the paper is that if a model's response truly relies on visual evidence, the predictive uncertainty of the same response should increase when key visual regions are removed. Conversely, if the model remains confident after the core regions of the image are masked, the response likely stems primarily from language priors and carries a higher risk.

**Core Idea**: The reduction in entropy brought by visual input is used as the Image-Information Score. Then, core image regions are identified via visual attention from middle-to-late layers and masked. Finally, predictive entropy is combined with the Image-Information Score after core masking to form the VAUQ risk score.

## Method
The goal of VAUQ is to output a score $s(x,y)$ given image-text input $x=(v,t)$ and model response $y$, used to judge if the response is likely hallucinated or incorrect. Unlike detectors requiring external supervision, VAUQ utilizes only the internal probabilities and attention information of the same LVLM.

The paper first notes that pure linguistic uncertainty methods fail on counterfactual data like ViLP. ViLP contains factual and counterfactual images where the same question requires different answers. Methods like Entropy, Verbalized Confidence, Semantic Entropy, and EigenScore significantly graduate on counterfactual images (e.g., Entropy drops by 40.9%, EigenScore by 26.0%), indicating they are dominated by language priors and fail to identify errors caused by "conflicts between images and common sense."

VAUQ therefore does not just ask "is the model confident in the answer," but "does the model's confidence come from the image." It defines visual contribution as the difference in predictive entropy with and without the image: if the image makes the model more certain, it indicates the image provided information; if the entropy barely changes after removing the image, it indicates the response relies on language priors.

### Overall Architecture
The workflow consists of four steps. First, the LVLM generates response $y$ based on original image-text input and calculates the length-normalized predictive entropy $H(y|v,t)$. Second, the attention of generated tokens toward image patches is aggregated to estimate which visual tokens constitute core evidence. Third, the top-$K$% core visual tokens are masked to obtain $v_{masked}$, and $H(y|v_{masked},t)$ is calculated. Fourth, the original predictive entropy and the Image-Information Score after core masking are combined into the final self-evaluation score.

The original Image-Information Score is written as $IS_{blank}=H(y|empty,t)-H(y|v,t)$, where $empty$ denotes the removal of visual input. The core region version uses $IS_{core}=H(y|v_{masked},t)-H(y|v,t)$. The final score is $s_{VAUQ}=H(y|v,t)-\alpha\cdot IS_{core}$, which can also be understood as $(1+\alpha)H(y|v,t)-\alpha H(y|v_{masked},t)$. A higher score indicates a more unreliable response; if entropy rises significantly after masking core visual evidence, $IS_{core}$ will be larger, lowering the score and indicating a more credible response.

### Key Designs
1.  **Image-Information Score**:
    - **Function**: Measures the contribution of visual input to the predictive uncertainty of the current response.
    - **Mechanism**: Compares the conditional entropy of the same response with and without the image. If $H(y|empty,t)$ is significantly higher than $H(y|v,t)$, the image helped confirm the response; if the gap is small, confidence likely comes from language priors.
    - **Design Motivation**: Pure textual uncertainty only indicates confidence, not visual grounding. IS explicitly incorporates the usage of visual evidence into the scoring.

2.  **Unsupervised Core Region Masking**:
    - **Function**: Avoids noise from full image removal or background perturbations, focusing IS on task-relevant visual evidence.
    - **Mechanism**: Aggregates attention from generated tokens to visual tokens in middle-to-late transformer layers to obtain importance scores for each patch, then selects the top-$K$% patches as the core region. The authors found that early layers struggle to locate evidence, while middle-to-late layers capture semantic regions better.
    - **Design Motivation**: Blanking the entire image introduces background noise, and random masking disrupts the input. Core region masking is closer to "removing the evidence the model truly relies on," thus better testing if the response is grounded.

3.  **Linear Combination of Entropy and Visual Information**:
    - **Function**: Captures both linguistic predictive uncertainty and visual grounding strength.
    - **Mechanism**: $s_{VAUQ}=H(y|v,t)-\alpha IS_{core}$. High predictive entropy increases the risk score (indicating inherent uncertainty); a high visual information score decreases the risk score (indicating confidence comes from visual evidence). Hyperparameter $\alpha$ controls the weight.
    - **Design Motivation**: Entropy alone can be fooled by language priors, and visual information alone might ignore generation uncertainty. They are complementary, especially for real-world distributions mixing factual and counterfactual cases.

### Loss & Training
VAUQ has no training loss and is an a posteriori self-evaluation scoring method. Implementation uses greedy decoding with a maximum length of 128. For efficiency, the authors do not modify image pixels but apply a knockout to the attention weights of the top-$K$ visual tokens when calculating $IS_{core}$. $\alpha$, masking ratio $K$, and layer range $(l_s,l_e)$ are selected on a validation set; experiments used Python 3.11.11, PyTorch 2.6.0 on a single 80GB A100, reporting averages over 3 random seeds.

## Key Experimental Results

### Main Results
Experiments cover four datasets: ViLP, MMVet, VisualCoT, and CVBench, using LLaVA-1.5, Qwen2.5-VL, and InternVL3.5. The metric is AUROC (higher is better for distinguishing correct from hallucinated responses). Representative results for LLaVA-1.5-7B and Qwen2.5-VL-7B are shown below.

| Model | Method | ViLP | MMVet | VisualCoT | CVBench |
|------|------|------|-------|-----------|---------|
| LLaVA-1.5-7B | Perplexity | 54.6 | 79.3 | 56.2 | 60.3 |
| LLaVA-1.5-7B | Semantic Entropy | 63.7 | 81.3 | 75.1 | 70.2 |
| LLaVA-1.5-7B | VL-Uncertainty | 55.6 | 82.3 | 65.2 | 71.1 |
| LLaVA-1.5-7B | VAUQ | 77.0 | 81.5 | 77.8 | 73.2 |
| Qwen2.5-VL-7B | Perplexity | 55.0 | 76.6 | 56.0 | 64.8 |
| Qwen2.5-VL-7B | Semantic Entropy | 52.0 | 60.1 | 53.3 | 50.9 |
| Qwen2.5-VL-7B | VL-Uncertainty | 57.9 | 69.7 | 62.3 | 69.7 |
| Qwen2.5-VL-7B | VAUQ | 64.1 | 78.3 | 68.0 | 69.8 |

VAUQ outperforms Semantic Entropy by 13.4 percentage points on LLaVA-1.5-7B ViLP and exceeds VL-Uncertainty by 21.4 points on the same model; it also leads VL-Uncertainty by 12.6 points on VisualCoT. This suggests visual grounding signals are particularly effective for counterfactual and evidence-localization tasks.

### Ablation Study
Efficiency experiments show VAUQ is much faster than multi-sampling or external module methods while achieving higher AUROC. The table below shows average per-sample time and AUC on ViLP.

| Method | LLaVA-1.5-7B Time(s) | LLaVA AUC | Qwen2.5-VL-7B Time(s) | Qwen AUC |
|------|----------------------|-----------|------------------------|----------|
| SVAR | 0.39 | 50.6 | 1.59 | 49.6 |
| Verbalized | 0.58 | 56.3 | 1.82 | 55.3 |
| EigenScore | 5.86 | 63.2 | 8.77 | 53.0 |
| Semantic Entropy | 7.05 | 63.7 | 12.40 | 52.0 |
| VL-Uncertainty | 13.60 | 55.6 | 20.20 | 57.9 |
| VAUQ | 0.73 | 77.0 | 2.16 | 64.1 |

The authors also compare masking strategies on VisualCoT. Random masking degrades performance, while the ground-truth box oracle is strongest. VAUQ's attention-based core masking approaches the oracle, indicating middle-to-late attention can approximate critical evidence without labels. The appendix reports HallusionBench, where VAUQ scores AUROC 67.0 on LLaVA-1.5-7B (vs. 65.1 for VL-Uncertainty) and 74.3 on Qwen2.5-VL-7B (vs. 74.0 for Semantic Entropy).

| Evaluation Item | Comparison Method | Result | Description |
|--------|----------|------|------|
| ViLP AUPRC | Semantic Entropy | 60.2 | Semantic Entropy remains weaker than VAUQ under class imbalance |
| ViLP AUPRC | VAUQ | 68.2 | 8.0 higher than Semantic Entropy |
| ImageNet-S Localization | Embedding baseline | 50.4 / 36.1 / 53.9 | Weak overlap with true object regions |
| ImageNet-S Localization | Attention masking | 69.3 / 46.4 / 77.1 | Core regions are closer to true object regions |
| ViLP / VisualCoT Masking | Grad-CAM | 76.0 / 76.6 | Usable but requires gradient saliency maps |
| ViLP / VisualCoT Masking | Attention masking | 77.0 / 77.8 | Training-free and slightly superior |

### Key Findings
- Language priors are the primary trap in LVLM self-evaluation. Traditional entropy or verbalized confidence underestimates risk on counterfactual images because the text prior makes incorrect answers appear fluent.
- Core region masking is more reasonable than blanking the whole image. Full removal strips background and irrelevant regions along with key evidence; attention masking more directly tests if the response depends on task-relevant regions.
- VAUQ's efficiency advantage is significant. It requires only a constant number of extra forward passes and no multiple response generations, keeping complexity at $O(M)$ compared to $O(A\cdot M)$ for multi-sampling methods.
- Entropy and IS are complementary signals. Entropy performs well on factual splits but degrades on counterfactual ones; IS is stronger when visual evidence is required. Combining them leads to better stability.
- Hyperparameters have stable ranges. The paper finds $\alpha$ near 0.5 to 1.5 is usually good; moderate masking ratios $K$ are more stable, e.g., $K \approx 30$ for CVBench and $K \approx 40$ for MMVet.

## Highlights & Insights
- The problem definition of VAUQ is precise. It is not just another external hallucination detector but asks whether the LVLM's confidence is truly visually grounded.
- The Image-Information Score is a simple yet interpretative signal. It converts "whether visual input reduces uncertainty" into a computable quantity directly corresponding to grounding.
- Core region masking makes the scoring more like a causal test. Masking what the model attends to most and observing the change in probability is closer to intervention than simply reading attention weights.
- The method remains training-free, making it a suitable lightweight reliability layer for deployment. it doesn't require hallucination labels for every task or a specialized probe.
- This paper reminds us that multimodal self-evaluation cannot be directly inherited from LLMs. Visual evidence usage is a reliability dimension unique to LVLMs.

## Limitations & Future Work
- Dependency on global hyperparameters. The authors admit optimal values for $\alpha$, $K$, and layer ranges may vary across datasets, models, or even samples, suggesting sample-adaptive tuning for the future.
- Current evaluation primarily targets instruction-tuned image LVLMs. Effectiveness on long-chain visual reasoning, video understanding, or agentic multimodal systems remains unverified, especially where visual contribution may be distributed across multiple stages.
- Attention is not always evidence. Appendix cases show that when multiple salient objects exist, attention might miss some relevant info, leading to incomplete core region masking.
- The score is not a safety guarantee. The ethical statement emphasizes VAUQ as an auxiliary reliability signal, not a replacement for human review or complete safety mechanisms.
- Requires internal access to probabilities and attention. VAUQ cannot be used directly for closed-source LVLMs or text-only APIs; black-box approximations are needed.

## Related Work & Insights
- **vs Perplexity / Entropy**: These methods only look at language output probabilities and are easily misled by priors; VAUQ additionally examines the impact of image removal or core masking.
- **vs Semantic Entropy / EigenScore**: Multi-sampling and hidden state methods capture response diversity but are costly and don't verify visual evidence usage; VAUQ measures visual contribution directly with fewer forward passes.
- **vs SVAR / Contextual Lens**: Attention or representation similarity can detect object-level grounding but is less direct for response-level hallucinations; VAUQ uses attention for intervention combined with output entropy.
- **vs VL-Uncertainty**: VL-Uncertainty estimates uncertainty via consistency across multiple responses under perturbations; VAUQ is a white-box, training-free, low-sampling alternative.
- **Insights**: The VAUQ score can be used for selective prediction: high-risk responses trigger retrieval, re-observation, refusal, or human check. It could also combine with generation-time methods like VCD to detect then correct.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines predictive entropy with visual information gain; concise and addresses LVLM points.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, datasets, main results, masking strategies, efficiency, AUPRC, HallusionBench, and localization quality.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation, intuitive formulas, though HTML table conversions slightly impact reading.
- Value: ⭐⭐⭐⭐⭐ Training-free, interpretable, and efficient; ideal as a reliability check for multimodal systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](../../ICLR2026/multimodal_vlm/detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)
- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ACL 2026\] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models](ireasoner_trajectory-aware_intrinsic_reasoning_supervision_for_self-evolving_lar.md)
- [\[ACL 2026\] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)
- [\[ACL 2026\] Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models](almieyar-oryx-bloombench_a_bilingual_multimodal_benchmark_for_cognitively_inform.md)

</div>

<!-- RELATED:END -->
