---
title: >-
  [Paper Note] Scaling Vision Transformers for Functional MRI with Flat Maps
description: >-
  [ICML 2026][Medical Imaging][fMRI foundation model] By projecting 3D fMRI volumes into 2D "cortical flat maps" and feeding them as videos to a standard spacetime MAE-ViT…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "fMRI foundation model"
  - "Cortical Flat Map"
  - "MAE"
  - "Brainmarks benchmark"
  - "Scaling Law"
date: 2026-05-08
content_hash: 1968911331a369ab
---

# Scaling Vision Transformers for Functional MRI with Flat Maps

**Conference**: ICML 2026  
**arXiv**: [2510.13768](https://arxiv.org/abs/2510.13768)  
**Code**: https://github.com/MedARC-AI/CortexMAE & https://github.com/MedARC-AI/Brainmarks (available)  
**Area**: Medical Imaging / Self-supervised Learning / Foundational Models for Neuroimaging  
**Keywords**: fMRI foundation model, Cortical Flat Map, MAE, Brainmarks benchmark, Scaling Law

## TL;DR
By projecting 3D fMRI volumes into 2D "cortical flat maps" and feeding them as videos to a standard spacetime MAE-ViT, the authors train CortexMAE on 2.1K hours of HCP data: it dramatically outperforms SOTA in cognitive state decoding, validating flat maps as the "goldilocks zone" between voxel (volume) and region-averaged (parcellation) representations. They also release the first open-source fMRI foundation model benchmark Brainmarks, provide the first systematic scaling law for fMRI models, and report an honest null result: individual trait prediction still cannot beat a simple functional connectivity baseline.

## Background & Motivation

**Background**: The neuroscience community aims to use fMRI and large models to decode brain activity (diagnosis, behavior prediction, visual reconstruction). Existing fMRI self-supervised foundation models (BrainLM, Brain-JEPA, NeuroSTORM, SwiFT, etc.) mostly use **parcellation** representations (averaging 3D brain volumes into 100–400 regions, yielding 1D time series vectors); a few use **volume** representations (directly processing 4D spatiotemporal MRI data).

**Limitations of Prior Work**: (1) Parcellation is computationally cheap but **loses substantial information**—entire centimeter-scale regions are compressed into single scalars, discarding 99% of dimensions; (2) Volume preserves all information but leads to huge sequence lengths (after patching, a single fMRI volume yields ~2000+ tokens), resulting in massive compute/IO costs; (3) The fMRI foundation model field **lacks reproducible benchmarks**—each group uses its own dataset, preprocessing, and evaluation, making comparisons unreliable; (4) Previous trait prediction papers often claim "we beat baseline X%," but the baselines are too weak, lacking serious comparison to simple functional connectivity (FC) + logistic regression, a method from 30 years ago.

**Key Challenge**: fMRI data is inherently 4D spatiotemporal volumes, while standard ViTs assume 2D inputs. One can either pay a high cost to learn 4D directly (full information, expensive), or use strong inductive bias (parcellation) and lose information—is there a **middle representation that preserves full cortical signals and is ViT-friendly as a 2D input**?

**Goal**: (i) Identify the "goldilocks" input representation for fMRI; (ii) Train a set of clearly comparable foundation models using standard ViT + MAE; (iii) Establish an open, reproducible fMRI foundation model benchmark (Brainmarks); (iv) For the first time, systematically study the data/model scaling law for fMRI self-supervision.

**Key Insight**: Neuroscience has long used the **cortical flat map**—flattening the 2D manifold of the cortex (essentially a 2–4mm thick folded sheet) onto a planar grid. This preserves the full cortical BOLD signal (unlike parcellation, which averages away details) and yields a 224×560 2D "image" that can be directly processed as a video by spacetime ViTs.

**Core Idea**: Project 3D fMRI into 2D videos using cortical flat maps, apply off-the-shelf MAE-st training, **without modifying the ViT architecture, only changing the patch embedding**—a simple but previously overlooked choice, resulting in SOTA performance, the first fMRI scaling law, and the first open-source benchmark.

## Method

### Overall Architecture
Model = MAE-st (Feichtenhofer et al. 2022) + interchangeable input heads for three patch embedding types. Pipeline: (1) HCP-YA data preprocessing (FreeSurfer / fMRIPrep surface mapping) → obtain time series on cortical surface mesh; (2) Use pycortex to project the surface onto a planar grid → get a 16-frame × 224 × 560 fMRI flat map video; (3) Cut $p_t \times 16 \times 16$ spatiotemporal patches (default $p_t = 4$), mask ratio 0.9 (tube masking, no temporal interpolation); (4) ViT-B encoder sees sparse observed patches + [MASK] token, decoder reconstructs masked patches, loss is MSE (only on non-background pixels); (5) After pretraining, discard the decoder, use encoder outputs as features for downstream trait/state prediction with linear probe / attentive probe. Parcellation MAE (using Schaefer-400 regions) and volume MAE (using 4D patches) are trained in parallel for strict comparison.

### Key Designs

1. **Cortical Flat Map Patch Embedding (Core Innovation)**:

    - Function: Converts 3D fMRI volumes into 2D image videos, allowing standard spacetime ViTs to process them directly without losing full cortical signals.
    - Mechanism: First, use standard surface-based pipelines (FreeSurfer + fMRIPrep) to map each fMRI frame from 3D voxels to cortical surface mesh vertices; then, use pycortex flat map to unfold the mesh onto a 2D planar grid (left and right hemispheres concatenated); finally, resample to a fixed 224×560 grid. Each time step is a 2D frame, and 16 frames form the ViT spacetime input. Patch size is $p_t \times 16 \times 16$; background (non-brain) all-zero patches are excluded from computation, and MSE loss is computed only on non-background pixels.
    - Design Motivation: Parcellation compresses signals to ~400 dimensions (dimensionality loss ~100×), while volume requires ~132K voxel-sparse patches (computationally expensive, mostly background). Flat map sits between the two—retaining ~77K cortical signals but processed efficiently as 2D by ViT; sequence length 364 (vs. 465 for volume and 400 for parcellation) is comparable, but training bandwidth and data throughput are better. Figure 1 in the paper visualizes this trade-off as a "spectrum."

2. **Head-to-head Comparison of Three Representations**:

    - Function: Places parcellation/flat/volume representations on a level playing field with the same architecture, pretraining data, and evaluation protocol.
    - Mechanism: All models use ViT-B encoder, same 16-frame input, same 0.9 mask ratio; only patch embedding differs—parcel uses $p_t \times 1$ temporal patches, flat uses $p_t \times 16 \times 16$, volume uses $p_t \times 8 \times 8 \times 8$ 4D patches. Each variant is trained 8 times (mean reported), evaluated on 8 downstream datasets (4 clinical diagnoses + sex + age + HCP-YA Task21 + NSD COCO24).
    - Design Motivation: Previous fMRI foundation model papers never fairly compared representations—typically using only one and claiming SOTA. This is the first multi-representation fMRI MAE family, greatly increasing the credibility of conclusions.

3. **Brainmarks Open-source Benchmark Suite**:

    - Function: Provides a benchmark for fMRI foundation models where all methods can run and all datasets are standardized.
    - Mechanism: Includes 6 existing fMRI foundation models (SwiFT, BrainLM, Brain-JEPA, BrainHarmonix-F, NeuroSTORM, Brain-Semantoks) and the CortexMAE family; covers 7 public datasets—4 clinical diagnoses (ABIDE/ADHD200/ADNI/PPMI) + HCP-A age/sex + HCP-YA Task21 + NSD COCO24; for small-sample trait prediction, uses linear probe + 100 random train-test splits; for large-sample state prediction, uses attentive probe, single fixed split, and 49 learning rate grid search. All methods use the same probe protocol, avoiding unfair fine-tuning.
    - Design Motivation: The reproducibility crisis in fMRI model evaluation is well-known; unified protocols are essential for meaningful comparisons. The NSD COCO24 task ("short trial overlap + different subject test set + challenging visual decoding") is specifically designed to differentiate model quality.

### Loss & Training

Pretraining: MAE MSE loss on masked patches; data normalization is critical—each voxel/ROI time series is z-scored (coordinate norm) + each frame is spatially z-scored (frame norm), removing static noise with only 1–2% BOLD fluctuation; temporal patch $p_t = 4$; training schedule: 625K steps, batch size 32 (= 512 frames); repeated sampling to reduce IO bottleneck. Downstream: trait prediction uses average-pooled embedding + logistic regression with 5-fold CV; state prediction uses attentive probe + early stopping.

## Key Experimental Results

### Main Results

Probe accuracy of the three representations on 8 downstream tasks (mean of 8 pretraining seeds):

| Dataset | parcel | flat | volume | FC baseline |
|---|---|---|---|---|
| ABIDE (ASD diagnosis) | 62.0 | 61.4 | 60.4 | 59.8 |
| ADHD200 | 56.8 | 59.2 | 58.8 | 57.0 |
| ADNI (AD) | 61.6 | 62.4 | 64.3 | 58.6 |
| PPMI (PD) | 61.4 | 58.8 | 59.1 | 58.0 |
| HCP-A Age | 44.2 | 47.5 | **53.4** | 45.6 |
| HCP-A Sex | 71.2 | **87.4** | 86.3 | 81.9 |
| HCP-YA Task21 (state) | 97.5 | **98.9** | 96.2 | 82.4 |
| NSD COCO24 (visual decoding) | 27.5 | **31.0** | 27.7 | 7.4 |

Summary: (1) **Flat map excels in dynamic state decoding** (Task21, COCO24, sex); (2) Volume has an advantage in age prediction (possibly due to dense voxels capturing structural cues like cortical thickness); (3) Parcel is most efficient but weaker in state decoding; (4) On the 4 clinical diagnosis datasets, all methods are nearly tied and only marginally outperform the FC baseline—revealing that with small samples, fMRI foundation models show little advantage.

Controlled benchmark across models (Figure 8): For trait prediction, **no model significantly outperforms the simple FC baseline** (including BrainLM, Brain-JEPA, NeuroSTORM, etc.); for state decoding, **CortexMAE flat map leads across the board**, outperforming volume models like NeuroSTORM by 3–5 points on NSD COCO24.

### Ablation Study

| Configuration | Observation |
|---|---|
| Full flat map MAE | baseline |
| No frame normalization | Global signal drift contaminates, downstream accuracy drops | 
| No coordinate normalization | Static voxel differences dominate features, state decoding fails |
| Tube masking → random masking | Temporal interpolation leaks information, reconstruction becomes trivial |
| Mask ratio 0.5 → 0.9 | High mask ratio forces learning of structural representations, downstream improves |
| Increase encoder depth | Saturates at depth ~9 (37M params) |
| Increase pretrain data | Strict power law within HCP data (exponent -0.01), saturates OOD on NSD |

### Key Findings

- **fMRI strictly follows a data scaling law, but the exponent is ten times weaker than in language models** (-0.01 vs. -0.1 in Kaplan 2020), meaning marginal gains are small and scaling alone will not solve the problem.
- Model scaling saturates at depth 9 (37M params)—datasets like HCP-YA (2K hours) only support this capacity.
- The model spontaneously learns the brain's default mode network (DMN): the first principal component of the position embedding matches the FC principal gradient from Margulies 2016, showing that MAE learns neurobiologically meaningful structure.
- **Honest null result**: All fMRI foundation models fail to outperform simple FC + linear regression for individual trait prediction—a wake-up call for the field.
- For state decoding, foundation models have robust advantages, with CortexMAE flat being the strongest.

## Highlights & Insights

- **"Just change the patch embedding" is an elegant engineering choice**: No need to redesign the architecture or rewrite attention—simply project the input from 3D volume to 2D manifold. Any future ViT paper can adopt this for fMRI.
- **The goldilocks zone concept is transferable**: In representation learning, "full retention vs. heavy compression" is a classic trade-off; the cortical flat map is a perfect middle ground leveraging domain geometry (the cortex is inherently 2D), and similar ideas can be applied to EEG (1D time + electrode geometry), ECG, microscopy, etc.
- **Honest null result + open-source benchmark**: The neuroimaging community has long suffered from "small datasets + self-evaluation" reproducibility issues; the authors release Brainmarks and openly admit trait prediction does not beat FC baseline—an urgently needed honest voice.
- **First fMRI scaling law**: Clarifies that "fMRI is not NLP"—the tenfold smaller exponent means scaling data yields limited returns; the real bottleneck may be data diversity, not scale.
- **DMN emerges naturally**: Self-supervised representations aligning with known neurobiological structures is strong evidence for the interpretability of fMRI MAE.

## Limitations & Future Work

- HCP-YA consists entirely of 22–35-year-old healthy individuals, so the pretraining distribution is **narrow**, with weak OOD generalization (Figure 7 shows scaling fails on NSD).
- Clinical diagnosis results (ABIDE, ADHD200, etc.) hover around 60%—**almost useless**; foundation models cannot transfer to small-sample clinical data, a common challenge not solved here.
- Saturation at depth 9 indicates insufficient data scale—yet fMRI data collection is extremely expensive (hundreds of dollars per hour), so 10× more data would require community-wide collaboration.
- Flat map projection **loses subcortical structures** (e.g., thalamus, basal ganglia), which are important for many clinical tasks; volume models have a structural advantage here.
- Did not explore multi-modal fMRI (task + rest + diffusion) joint pretraining, leaving ample space for future work.
- Evaluation is only on English-speaking, North American populations, introducing demographic bias.

## Related Work & Insights

- **vs. BrainLM (Caro et al. 2024) / Brain-JEPA (Dong et al. 2024)**: These are parcellation-based fMRI foundation models with significant dimensionality loss; CortexMAE flat preserves full cortical signals and is much stronger in state decoding.
- **vs. SwiFT (Kim et al. 2023) / NeuroSTORM (Wang et al. 2025a)**: Volume-based models are computationally expensive but have an advantage in age prediction; this paper's volume MAE replicates this, showing dense representations have a niche, but flat still leads in state decoding.
- **vs. functional connectivity baselines**: Since Finn et al. 2015, FC + linear has been the standard baseline for trait prediction; this paper shows that deep fMRI models have yet to truly surpass it—a reality check for the field.
- **vs. vision MAE (He et al. 2022)**: Directly adopts MAE-st; the main contribution is "finding a suitable 2D projection to fit fMRI into the existing framework"—demonstrating that "domain geometry + general architecture" is an efficient combination.

## Rating

- Novelty: ⭐⭐⭐⭐ Strictly speaking, cortical flat maps have been a neuroscience tool for decades, but this is the first systematic use for ViT-friendly representation; technically minor but strategically smart.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous comparison of three representations + 6 external models + Brainmarks 7 datasets + scaling law + interpretability analysis—almost a "white paper" for fMRI MAE.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, direct motivation and conclusions, and key figures (spectrum, DMN emergence) are highly convincing.
- Value: ⭐⭐⭐⭐⭐ The Brainmarks + null result + flat map trio is textbook-level contribution to the fMRI foundation model community and will be a must-cite for future work.

## Related Papers

- [\[CVPR 2026\] MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](../../CVPR2026/medical_imaging/muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](../../ACL2026/medical_imaging/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](../../ICML2025/medical_imaging/supercharging_graph_transformers_with_advective_diffusion.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](../../AAAI2026/medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](../../CVPR2026/medical_imaging/muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[AAAI 2026\] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](../../AAAI2026/medical_imaging/funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](../../AAAI2026/medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)

</div>

<!-- RELATED:END -->
