---
title: >-
  [Paper Note] Position: Age Estimation Models Do Not Process Biometric Data
description: >-
  [ICML 2026][Biometric Data] This position paper provides empirical evidence using 14 models across 3 face verification benchmarks to demonstrate that facial age estimation models possess identity discrimination capabilit…
tags:
  - "ICML 2026"
  - "Biometric Data"
  - "Age Estimation"
  - "GDPR"
  - "Face Verification"
  - "AI Regulation"
date: 2026-05-08
content_hash: 3856aa55470cb9d8
---

# Position: Age Estimation Models Do Not Process Biometric Data

**Conference**: ICML 2026  
**arXiv**: [2605.17347](https://arxiv.org/abs/2605.17347)  
**Code**: None  
**Area**: AI Safety / AI Governance / Face Analysis  
**Keywords**: Biometric Data, Age Estimation, GDPR, Face Verification, AI Regulation  

## TL;DR
This position paper provides empirical evidence using 14 models across 3 face verification benchmarks to demonstrate that facial age estimation models possess identity discrimination capabilities two orders of magnitude below regulatory thresholds. Therefore, they should not be automatically categorized as "biometric data processing" under the definitions of GDPR, BIPA, or the EU AI Act.

## Background & Motivation

**Background**: When a neural network estimates age from a face photo, does it "process" biometric data? This is not a philosophical question—it directly determines whether operators must obtain explicit consent under GDPR Article 9, face statutory damages of $1,000–$5,000 per violation under Illinois BIPA, or be classified as "High-Risk AI" under the EU AI Act. GDPR Article 4(14) defines biometric data as data "allowing or confirming" the unique identification of a natural person, while Article 9 adds the restriction "for the purpose of uniquely identifying." BIPA follows a capability-based approach (extracting facial geometry counts), and the EU AI Act introduces yet another set of definitions.

**Limitations of Prior Work**: Regulators have not provided a unified answer. The UK ICO stated in the Yoti sandbox that facial age estimation does not constitute "special category data," yet its 2024 guidance acknowledges that "even if it is not your intention, it may count if identification is possible." The EDPB 2019 video guidance provided an exemption for systems that "only perform classification and do not generate identification templates," but the 2022 facial recognition guidance narrowed this loophole. The legal community uses inconsistent criteria, and the engineering community is caught in a dispute between "functional capability" and "intent of use."

**Key Challenge**: Intermediate representations are indeed generated during forward propagation. These representations exist transiently and are neither outputted nor stored, but they could theoretically "encode identity-discriminative information." Under a capability-based interpretation, any system where intermediate tensors "might encode identity" becomes biometric processing. Under an intent-based interpretation, age estimation is fully exempt because its purpose is not identification. Both interpretations have merit, but both lack data.

**Goal**: To empirically answer whether "age estimation models functionally possess unique identification capabilities," separating empirically measurable components from legal debates.

**Key Insight**: The authors cite the ICO’s precise phrasing—"unique identification requires singling out someone with accuracy"—and point out that this accuracy is quantifiable. Thus, they transform the question of "whether it is biometric" into "whether FNMR@FMR on face verification benchmarks meets regulatory thresholds."

**Core Idea**: Evaluate the verification performance of 14 models (4 dedicated age estimators + multiple attribute models + 1 ArcFace baseline + 3 general vision models) on LFW / AgeDB-30 / CFP-FP. Compare these results against three sets of regulatory thresholds: NIST SP 800-63-4, EU EES, and FIDO. Additionally, conduct an "adversarial" secondary experiment—using an attention probe to retrain a face recognition head on frozen features to extract any latent recognition power.

## Method

### Overall Architecture

Each tested model $M$ is treated as a feature extractor. For a face image $x$, intermediate layer activations are extracted, followed by global average pooling and L2 normalization to obtain an embedding $e(x)$. For verification pairs $(x_a, x_b)$, the cosine similarity $s = \langle e(x_a), e(x_b) \rangle$ is calculated. DET curves are plotted for 6,000 evaluation pairs, reporting the False Non-Match Rate ($\text{FNMR}$) at a fixed False Match Rate ($\text{FMR}$). The primary metric is $\text{FNMR}@\text{FMR}=1\%$ (statistically reliable), with $\text{FNMR}@\text{FMR}=0.01\%$ as a regulatory reference. All layers of each model are evaluated to plot "FNMR vs. Network Depth" curves, proving that "no layer is sufficient," thereby preventing counter-arguments that earlier or later layers might encode identity.

### Key Designs

1.  **Falsification Framework for Capability vs. Intent**:
    *   Function: Transforms the binary legal question of "whether intermediate representations might identify" into a continuous quantitative problem of "whether FNMR@FMR reaches regulatory thresholds."
    *   Mechanism: Refers to ISO/IEC 19795 verification terminology. FMR is the rate at which different people are incorrectly matched (security), and FNMR is the rate at which the same person is incorrectly rejected (utility). Three regulatory thresholds are consolidated: NIST SP 800-63-4 IAL2 requires $\text{FMR}\le 0.01\%$ and $\text{FNMR}<5\%$, EU EES requires $\text{FMR}=0.05\%$ and $\text{FNMR}<1\%$, and FIDO requires $\text{FMR}\le 0.01\%$ and $\text{FNMR}<5\%$. If a model performs two orders of magnitude worse than these thresholds even on LFW (the simplest "in-the-wild" benchmark), "usable identification capability" can be practically excluded.
    *   Design Motivation: Regulatory concern focuses on whether a system can accurately "single out" an individual, not whether residual information might exist. Aligning test objectives with the ICO's "with accuracy, with a level of precision" standard allows conclusions to be directly adopted by regulators.

2.  **"Full Space" Stress Testing across Layers and Benchmarks**:
    *   Function: Closes counter-argument channels suggesting that an age estimation model might be capable of identification in a specific layer, benchmark, or pose.
    *   Mechanism: Instead of only looking at the final layer, 4–N intermediate layers are sampled for each model. Simultaneously, three complementary benchmarks are selected: LFW (in-the-wild, easy), AgeDB-30 (same person across a 30-year span), and CFP-FP (frontal vs. profile). AgeDB-30 is particularly critical: if age estimation models "incidentally learn identity," cross-age recognition should be their potential strength. The results show the opposite—age estimation models perform worse on AgeDB-30 (96–98% FNMR), indicating that identity and age features are nearly orthogonal.
    *   Design Motivation: The greatest risk for a position paper is being dismissed with "you didn't test enough." Using "no layer × no benchmark reaches the threshold" empirically preempts this rebuttal.

3.  **Attention Probe: Adversarial Upper Bound Estimation**:
    *   Function: Tests the possibility that identity information exists within features but is simply flattened by average pooling.
    *   Mechanism: The final layer features of the tested backbone are frozen, and an attention pooler (Yu et al., 2022) is attached. A learnable query aggregates feature tokens via cross-attention, followed by an ArcFace head trained on Glint360k. This "probe" is no longer the original age estimator but a new face recognition system using its features as input. If even this new system performs poorly, it proves the features themselves lack reasonable identity information. Results: The commercial age estimator's FNMR on LFW dropped from 27% to 2% (seemingly significant) but remained at 67% on AgeDB-30 and 28% on CFP-FP, far below ArcFace's 2.4% / 1.2%.
    *   Design Motivation: "Readout experiments" (average pooling) measure actual leakage of the original system, while "adversarial experiments" (attention probe) measure the theoretical upper bound of the features. Together, they strengthen the argument against claims of improper probing methods.

### Loss & Training

No loss functions are used for the unsupervised evaluation. For the attention probe, the ArcFace loss is used:
$$\mathcal{L} = -\log \frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)} + \sum_{j\ne y}e^{s\cos\theta_j}}$$
The pooler and projection head are fine-tuned on Glint360k while the backbone remains fully frozen. $\text{FNMR}@\text{FMR}=1\%$ and $\text{FNMR}@\text{FMR}=0.01\%$ are reported for each layer of every model.

## Key Experimental Results

### Main Results: Verification Capability of 14 Models on LFW

Using the ImageNet-style cropped LFW dataset, reporting $\text{FNMR}@\text{FMR}=1\%$:

| Model | Type | LFW @1% (%) | LFW @.01% (%) | Gap with ArcFace |
| :--- | :--- | :--- | :--- | :--- |
| ArcFace (ResNet-100) | Identification | 0.23 | 0.3 | 1× |
| Commercial age estimator | Age Estimation | 26.8 | 63.7 | ~100× |
| FairFace age+gender+race | Attributes | 57.4 | 85.8 | ~250× |
| Age+gender ViT | Attributes | 67.3 | 87.5 | ~290× |
| Age estimation PyTorch | Age Estimation | 94.6 | 99.7 | ~410× |
| SSR-Net | Compact Age | 95.0 | 99.4 | ~410× |
| DINOv3 (General Vision)| Self-supervised| 37.5 | 70.7 | Reference only |

The strongest age estimator (Commercial) has a 27% FNMR, which is over 5 times higher than the FIDO/NIST requirement of 5%. At the regulatory FMR=0.01% threshold, the FNMR of all age estimation models ranges between 64% and 99%.

### Secondary Experiments: Results with Attention Probe

With frozen features and an ArcFace head trained on Glint360k, comparing best layer results:

| Model | LFW (Before → After) | AgeDB-30 (Before → After) | CFP-FP (Before → After) |
| :--- | :--- | :--- | :--- |
| ArcFace | 0.23 | 2.4 | 1.2 |
| Commercial age estimator | 27 → 2 | 97 → 67 | 80 → 28 |
| Perception Encoder | 21 → 3 | 98 → 73 | 80 → 33 |
| DINOv3 | 37 → 3 | 96 → 68 | 81 → 36 |
| FairFace age+gender+race | 57 → 20 | 98 → 97 | — |

**Key Findings**:

*   **Age Estimation $\ne$ Residual Identity**: Age estimation models performed worst on the cross-age AgeDB-30 (96–98% FNMR). This suggests that features learned for age optimization are inherently orthogonal to identity-discriminative features, rather than "incidentally learning identity."
*   **No Layer Meets Standards**: The FNMR vs. depth curves show that all age estimation models descend slowly from ~95% FNMR in early layers to 27–95% in final layers. Even the best layers remain two orders of magnitude away from the NIST 5% threshold.
*   **Adversarial Probes Cannot Save It**: Even when identity signals are forced out using Glint360k supervision, age estimation models still show a 67% FNMR on AgeDB-30. At FMR=0.01%, the strongest probes on LFW/AgeDB-30/CFP-FP remain at 17/91/68%, far below regulatory identification systems.
*   **General Vision Models Can Outperform Age Estimators**: DINOv3 and Perception Encoder achieved 21–37% FNMR on LFW @1%, proving closer to recognition than some age estimators. This highlights the difference in identity information retention between "self-supervised general representations" and "task-specific age representations"—meaning regulators cannot simply draw a line based on "whether faces are processed."

## Highlights & Insights
*   **Translating Legal Uncertainty into Falsifiable ML Hypotheses**: Position papers often remain purely conceptual. This paper uses ISO 19795 quantitative metrics and a 14×3 grid experiment to turn "functional capability" into a repeatable experiment. It serves as a rare example of a "position + empirical" dual-driven paper, providing a paradigm for future AI governance research.
*   **Clever Use of AgeDB-30 as Counter-Evidence**: Intuitively, cross-age matching of the same person should be the "home ground" for an age estimation model if it were learning identity as a side effect. Its poor performance there is a strong counter-example, nearly eliminating the hypothesis that age models implicitly leak identity.
*   **Attention Probe as an "Upper Bound" Contribution**: Future position papers assessing whether a "system possesses capability X" can adopt this two-layer structure (readout experiments for actual leakage + adversarial probes for theoretical limits) to strengthen the conclusion from "it currently cannot" to "it theoretically struggles to."

## Limitations & Future Work
*   **Conflict of Interest Transparency**: The authors are from Sumsub, and their commercial age estimator is one of the 14 models tested (and happens to be the strongest). While they state the views are personal and use public benchmarks, readers should remain aware of potential bias.
*   **Architecture Limitations**: The 14 models are mostly standard backbones. There is no guarantee that future large-scale or multimodal age estimators will yield the same conclusions, especially as large vision-language foundation models may retain significant general identity capabilities after age-specific fine-tuning.
*   **Empirical Input vs. Legal Interpretation**: The authors clarify they do not solve legal questions. BIPA’s "capability-based" route (triggered by extracting facial geometry) might still include age estimation regardless of "functional non-recognition."
*   **Benchmark Bias**: LFW and other benchmarks are skewed toward Anglosphere public figures. Demographic biases in benchmarks might lead to an underestimation of identity recognition risks for certain minority groups, and "fairness within 25% group" regulatory clauses were not explored.

## Related Work & Insights
*   **vs. ICO Yoti Sandbox (2022) Report**: The ICO concluded that "intent is not identification → does not constitute Article 9 data" but left the "constituting Article 4(14) biometric data" definition open. This paper addresses that gap directly with quantitative testing under the ICO’s own "accuracy and precision" standards.
*   **vs. Clearview AI Cases (Garante 2022, CNIL 2022)**: These cases confirmed that stored facial embeddings are biometric data. This paper distinguishes "transient intermediate representations + no storage + no recognition capability" from "stored identification templates," arguing they should not be regulated identically.
*   **vs. EDPB Guidelines 3/2019 (Classification Exemption)**: The EDPB previously opened an exemption for systems doing attribute classification without generating templates. This paper provides an ML-verifiable criterion for that exemption, maintaining its validity even after the more restrictive 2022 facial recognition guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](../../AAAI2026/others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](../../CVPR2026/others/do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)
- [\[ICML 2026\] Metadata Predictability Is Not Evidence Dependence: An Intervention-Based Audit for Weak-Label Benchmarks](metadata_predictability_is_not_evidence_dependence_an_intervention-based_audit_f.md)
- [\[ICML 2026\] FOVI: Bio-inspired Foveated Interface for Deep Vision Models](fovi_a_biologically-inspired_foveated_interface_for_deep_vision_models.md)

</div>

<!-- RELATED:END -->
