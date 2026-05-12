# calli_sae

## Project goal

`calli_sae` explores a pipeline for controllable single-character calligraphy generation using SAE-discovered pseudo-attributes.

The central idea is:

> Use Sparse Autoencoders (SAEs) on the internal activations of a calligraphy-trained Diffusion Transformer (DiT) to discover latent calligraphic attributes, convert these attributes into pseudo-labels, and use them as supervision for PluGeN-style controllable generation on top of a VAE codec.

This project is not initially intended to be a full research system. The first goal is to build a clean experimental codebase for testing whether the idea is technically and empirically plausible.

---

## Motivation

PluGeN is designed for multi-label conditional generation from pretrained models. It can reshape the entangled latent space of a pretrained generative model, such as a VAE or GAN, into a space where labeled attributes are modeled independently and can be edited. However, PluGeN requires attribute labels for training.

For calligraphy, the available labels are usually coarse:

- character
- script type
- author / source

The attributes that are more interesting for controllable generation are much harder to annotate manually:

- stroke thickness
- stroke curvature
- turning shape
- spatial compactness
- center of gravity
- stroke spacing
- ink texture
- overall style tendency

These attributes are often continuous, implicit, and difficult to define with a fixed annotation scheme. The project therefore tries to replace manual fine-grained labels with SAE-derived pseudo-attributes.

---

## Core hypothesis

The project is based on the following hypothesis:

> A DiT trained on calligraphy images may learn internal representations that encode meaningful calligraphic visual factors. SAEs trained on these DiT activations may decompose the representations into sparse, partially monosemantic features. Some of these features may be converted into pseudo-labels and used as supervision for PluGeN.

The key point is that the DiT and SAE are used for **pseudo-label discovery**, not for the final image generation stage.

The final controllable generator is expected to be:

```text
PluGeN flow + VAE decoder
```

---

## High-level pipeline

For each calligraphy image `x`, the system constructs two aligned objects:

```text
x -> DiT + SAE      -> pseudo-labels y_hat
x -> VAE encoder    -> latent code z
```

Then PluGeN is trained on:

```text
(z, y_hat)
```

The full pipeline is:

```text
Calligraphy image x
    |
    |-- Branch A: pseudo-label discovery
    |       train unconditional DiT
    |       extract DiT activations h_{l,t}
    |       train SAE / TIDE-style SAE
    |       obtain SAE feature activations a
    |       convert a into pseudo-labels y_hat
    |
    |-- Branch B: pretrained codec for PluGeN
            train VAE
            encode x into latent code z

(z, y_hat)
    |
    v
Train PluGeN-style flow
    |
    v
Attribute-control latent space
    |
    v
VAE decoder
    |
    v
Controllable calligraphy image x'
```

Variable notation:

- `x`: input calligraphy image
- `h_{l,t}`: DiT activation from layer `l` and diffusion timestep `t`
- `a`: SAE feature activation vector
- `y_hat`: SAE-derived pseudo-attribute label vector
- `z`: VAE latent code of `x`
- `D = [c, s]`: PluGeN-style control space, with attribute variables `c` and residual/style variables `s`
- `z'`: controlled VAE latent after inverse PluGeN flow
- `x'`: generated calligraphy image

---

## Model roles

| Model       | Role in this project                                         |
| ----------- | ------------------------------------------------------------ |
| VAE         | Pretrained codec for PluGeN. Encodes calligraphy image `x` into latent `z` and decodes controlled latent `z'` into generated image `x'`. |
| DiT         | Unconditional calligraphy image model used as an activation source for SAE. It is not the final controllable generator in the first version. |
| SAE         | Trained on DiT activations. Produces sparse feature activations `a`, which are analyzed and converted into pseudo-labels `y_hat`. |
| PluGeN flow | Learns an invertible transformation between VAE latent space and an attribute-control space supervised by `y_hat`. |

---

## Data assumptions

The current dataset consists of local single-character calligraphy images summarized by CSV files.

The codebase should assume that the dataset is loaded from CSV metadata. Each row should correspond to one image sample and should contain at least an image path or a path-like field. Existing metadata may include:

- character
- script type
- author

The exact CSV schema should be inspected from the existing projects and reused as much as possible.

The implementation should follow the data-loading style of these reference codebases:

```text
~/Code/calligraphy_project
~/Code/transfusion
```

The goal is not to invent a new dataset API if the existing projects already solve the local CSV-based loading problem. Prefer adapting their dataset class patterns, transforms, path handling, train/val split logic, and debugging style.

---

## Implementation preference

This repository should be easy to debug and modify during vibe coding.

General preferences:

- Keep the project modular but not over-engineered.
- Prefer readable PyTorch code over overly abstract frameworks.
- Use simple config files for paths, image size, batch size, model size, and training settings.
- Treat config files as editable default parameter bundles, not as a heavy framework. Command-line arguments should still be able to override key fields such as resume checkpoint, output root, epoch count, and batch size.
- Save the resolved config / args into each run directory for reproducibility.
- Use existing dataset/loading conventions from `calligraphy_project` and `transfusion` where possible.
- Save intermediate artifacts clearly: VAE checkpoints, DiT checkpoints, activation caches, SAE checkpoints, pseudo-label files, and PluGeN checkpoints.
- Favor small baseline runs before scaling.
- Make every stage runnable independently.
- The codebase structure may refer to `calligraphy_project`. The output / logging style may refer to `calligraphy_project/v2/cnet_gen_train.py` and `transfusion/train_calliffusion.py`.
- All generated results should be placed under `calli_sae_results`. The Python code should stay portable; concrete local/server paths are expected to be supplied by config files or sbatch scripts.

Confirmed v0 implementation decisions:

- Dataset source: reuse the CSV-based calligraphy dataset style from `~/Code/calligraphy_project` and `~/Code/transfusion`.
- VAE: finetune an existing pretrained `AutoencoderKL`-style VAE on the calligraphy images, rather than training a small convolutional VAE from scratch in v0.
- DiT: implement both latent-space and pixel-space entry points where practical, but make latent-space DiT the default first experiment.
- Latent-space DiT dependency: train / finetune the VAE first, then use the finetuned VAE encoder to produce latents for DiT training.
- Pixel-space DiT fallback: keep this option available if VAE reconstructions are poor or if latent-space activations are too spatially coarse for SAE analysis.
- Slurm: future training is expected to run through `sbatch`; sbatch files should define server paths, conda environment activation, and output directories.

---

## Development stages

### Stage 0: Project scaffolding and dataset reuse

Goal:

Build the initial repository structure and reproduce the dataset loading behavior from the existing projects.

Tasks:

- Inspect `~/Code/calligraphy_project` and `~/Code/transfusion`.
- Identify how they read CSV metadata and local image paths.
- Reuse or adapt their dataset class style.
- Create a minimal data loader smoke test.
- Visualize a batch of calligraphy images.

Expected result:

A working dataset pipeline that can load the calligraphy CSV and local images consistently.

---

### Stage 1: Train a VAE codec for calligraphy images

Goal:

Train a VAE that can encode and decode single-character calligraphy images with sufficient fidelity.

Why this is needed:

PluGeN needs a pretrained codec or generative backbone. In this project, the VAE provides:

```text
x -> encoder -> z
z -> decoder -> x_recon / x'
```

The VAE may also be useful as the latent codec for DiT if the DiT is implemented as a latent diffusion transformer, although this is not required for the earliest baseline.

Important validation:

- Can the VAE reconstruct character identity?
- Does it preserve stroke shape?
- Does it preserve structure and spacing?
- Does it preserve texture or ink-like variation, if present?

Suggested first baseline:

- Start by finetuning a pretrained `AutoencoderKL`-style VAE, such as the VAE already used by the recent `transfusion` experiments.
- Prioritize stable reconstruction, character identity preservation, and easy debugging.
- Use a conservative learning rate and save reconstruction grids frequently.
- Keep the implementation modular enough that a simple convolutional VAE, VQ-VAE, or other codec can be added later if pretrained VAE finetuning is not suitable.

Expected artifacts:

- VAE checkpoint
- reconstruction grid
- encoded latent cache `z` for training images
- resolved config / args for the run

---

### Stage 2: Train an unconditional DiT for calligraphy image modeling

Goal:

Train an unconditional DiT on the same calligraphy image dataset.

Role of DiT:

The DiT is used as a representation source for SAE. It should learn useful internal activations from the calligraphy image distribution.

It is not required to be the final controllable generator in the first version.

Possible settings:

1. Pixel-space DiT baseline
   - Simpler conceptually.
   - May be heavier computationally.

2. Latent-space DiT using the trained VAE
   - More aligned with latent diffusion practice.
   - VAE encoder produces latent images; DiT denoises in latent space.
   - This may also make activation extraction cleaner and cheaper.

Preferred direction:

Use latent-space DiT as the default v0 direction after the finetuned VAE is available. This connects naturally with latent diffusion practice, is cheaper than pixel-space denoising, and aligns with the later VAE-latent PluGeN stage.

The DiT code should still preserve a pixel-space fallback interface. Pixel-space DiT remains useful if VAE reconstruction quality is insufficient or if the VAE latent grid is too small for meaningful SAE token-level analysis.

For `64x64` images and an 8x VAE downsampling factor, the latent grid is expected to be `8x8`. The first latent DiT baseline can use patch size 1 over this latent grid. If this is too coarse for SAE interpretation, later experiments can increase image size, reduce effective downsampling, or switch to pixel-space DiT.

Expected artifacts:

- DiT checkpoint
- generated sample grid
- activation extraction script or hook utilities
- resolved config / args for the run

---

### Stage 3: Extract DiT activations

Goal:

Collect intermediate DiT activations for SAE training.

Notation:

```text
h_{l,t} = activation from DiT layer l at timestep t
```

Important design choices:

- Which DiT layer(s) to use?
- Which timestep(s) to use?
- Whether to train one SAE per layer or one SAE for selected activations.
- Whether to cache all activations or stream them during SAE training.
- Whether to sample tokens/patches to reduce storage and overfitting.

Initial recommendation:

Start with a small number of selected layers and timesteps. Avoid extracting everything at the beginning. Use a smoke-test activation cache first.

Expected artifacts:

- activation cache
- metadata linking each activation to image id, layer, timestep, token/patch position

---

### Stage 4: Train SAE / TIDE-style SAE on DiT activations

Goal:

Train an SAE that reconstructs DiT activations using sparse feature activations.

Basic form:

```text
h_{l,t} -> SAE encoder -> sparse activation a -> SAE decoder -> h_hat_{l,t}
```

TIDE-inspired issue:

DiT activations vary strongly across diffusion timesteps. If this is ignored, SAE features may learn timestep/noise patterns instead of stable image-level visual attributes.

Therefore, the project should treat timestep as a first-class concern.

Possible baselines:

1. Fixed-timestep SAE
   - Train SAE on activations from one selected timestep or a narrow timestep range.
   - Simpler and useful for debugging.

2. Timestep-conditioned SAE
   - Add a timestep embedding or modulation mechanism.
   - Inspired by TIDE's temporal-aware SAE design.

3. Separate SAE per timestep range
   - Easier than full timestep conditioning.
   - Useful for checking whether features are stable across denoising stages.

Initial recommendation:

Start with a fixed-timestep SAE or narrow timestep range. Then compare with timestep-conditioned or multi-timestep variants.

Expected artifacts:

- SAE checkpoint
- reconstruction metrics
- top-activating sample visualizations
- feature activation statistics

---

### Stage 5: Analyze SAE concepts and construct pseudo-labels

Goal:

Determine whether SAE features can be used as pseudo-attributes.

For each training image:

```text
x -> DiT activation h_{l,t} -> SAE -> feature activation a
```

Then convert `a` into pseudo-labels:

```text
a -> y_hat
```

Possible label construction strategies:

1. Binary pseudo-labels

```text
y_hat_j = 1 if a_j > threshold
```

2. Top-k pseudo-labels

```text
y_hat_j = 1 if feature j is among the top-k active SAE features
```

3. Continuous pseudo-attributes

```text
y_hat_j = normalized activation strength a_j
```

Preferred v0:

Use binary or top-k labels first because they are closer to the original multi-label PluGeN setting and are easier to debug.

Continuous controls are the ideal goal, but they should be treated as a later refinement after the features are better understood.

Feature analysis should include:

- top-activating images for each feature
- activation frequency
- activation strength distribution
- correlation with timestep
- correlation with character
- correlation with script type
- correlation with author/source
- same-character, different-style comparison
- same-style, different-character comparison

Expected artifacts:

- pseudo-label CSV or parquet
- feature report pages or image grids
- candidate feature list for PluGeN training

---

### Stage 6: Train PluGeN with VAE latents and SAE pseudo-labels

Goal:

Train a PluGeN-style flow using pairs:

```text
(z, y_hat)
```

Where:

- `z` comes from the VAE encoder
- `y_hat` comes from DiT + SAE

The goal is to learn an invertible transformation:

```text
VAE latent space Z <-> attribute-control space D = [c, s]
```

where:

- `c` is supervised by pseudo-attributes `y_hat`
- `s` captures remaining residual variation

This stage is less specified for now. The first implementation only needs to prepare for this stage cleanly. It is acceptable to leave detailed PluGeN design as a later update after VAE, DiT, and SAE stages are working.

Expected artifacts:

- PluGeN flow checkpoint
- controlled generation samples
- attribute editing grids

---

### Stage 7: Evaluate controllability

Goal:

Check whether SAE-derived pseudo-labels produce useful control through PluGeN.

Core tests:

1. Single-attribute editing

```text
fix residual variables
change one pseudo-attribute c_j
observe generated image x'
```

2. Counterfactual generation

```text
same base sample, same character if possible, edit only one pseudo-attribute
```

3. SAE re-evaluation loop

```text
edit pseudo-attribute j
sample generated image x'
run x' through DiT + SAE
check whether SAE feature j increases/decreases as expected
```

4. Preservation checks

- Does character identity remain stable?
- Does script type remain stable when it should?
- Does the generated image remain calligraphically plausible?
- Does editing one pseudo-attribute cause too many unrelated changes?

---

## Main research questions

### Q1. Are DiT-SAE features image-level attributes?

Need to distinguish:

```text
image-level calligraphic attributes
vs.
diffusion timestep / noise / denoising artefacts
```

A useful feature should show stable visual patterns across images and should not be explained only by timestep or noise level.

Good signs:

- top-activating images share visible calligraphic properties
- feature is stable within a timestep range or across controlled timestep settings
- feature is not purely character identity
- feature is not only author or script shortcut

---

### Q2. Can SAE concepts become useful labels?

SAE produces feature activations `a`, but PluGeN needs attribute labels `y`.

The bridge is:

```text
a -> y_hat
```

Useful pseudo-labels should be:

- stable
- visually coherent
- frequent enough for training
- not dominated by timestep/noise
- not too entangled with character or author
- suitable for editing or control

Binary or top-k pseudo-labels are the preferred v0. Continuous pseudo-attributes are a later target.

---

### Q3. Can PluGeN use these pseudo-labels effectively?

Given:

```text
(z, y_hat)
```

Can PluGeN learn a controllable latent transformation where editing a pseudo-attribute produces a stable and visible change in the generated calligraphy image?

Potential failure modes:

- pseudo-labels are noisy
- VAE latent does not preserve the visual factor
- SAE feature and VAE latent are image-aligned but not representation-aligned
- pseudo-labels are too entangled
- attributes are too imbalanced

Success criterion for v0:

Editing selected pseudo-attributes should produce non-random, visible, and somewhat interpretable visual changes while preserving basic character identity and image quality.

---

## Important conceptual distinction

This project uses two model systems:

```text
DiT + SAE: pseudo-label generator
VAE + PluGeN: controllable image generator
```

This is intentional.

The SAE labels do not need to live in the same latent space as the VAE. They only need to be aligned at the image-sample level:

```text
same image x
    -> y_hat from DiT + SAE
    -> z from VAE encoder
```

PluGeN then learns from `(z, y_hat)`.

Therefore, the main uncertainty is not architectural compatibility. The main uncertainty is the semantic quality, stability, and controllability of SAE-derived pseudo-labels.

---

## Reference papers and resources

### PluGeN

**PluGeN: Multi-Label Conditional Generation From Pre-Trained Models**  
Maciej Wołczyk et al.  
AAAI 2022 / TPAMI version.  
ArXiv: https://arxiv.org/abs/2109.09011  
Project page: https://gmum.github.io/plugen/  
Official code: https://github.com/gmum/plugen

Relevant idea:

PluGeN uses an invertible flow-based module to transform the entangled latent representation of a pretrained generative model into a space where labeled attributes are modeled independently. It has been combined with VAE and GAN backbones and supports conditional generation and attribute manipulation.

How it informs this project:

- Use VAE as the pretrained codec/backbone.
- Train PluGeN flow on VAE latent codes.
- Replace manual labels with SAE-derived pseudo-labels.

---

### TIDE

**TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation**  
Victor Shea-Jay Huang et al.  
ArXiv: https://arxiv.org/abs/2503.07050

Relevant idea:

TIDE trains SAEs on DiT activation layers across denoising steps. It introduces temporal-aware SAE design because DiT activations vary substantially with diffusion timestep. It shows that DiT activations can contain sparse, interpretable, hierarchical visual features and can support downstream steering/editing.

How it informs this project:

- Train SAE on DiT activations.
- Treat timestep as an important factor.
- Analyze whether features are image-level concepts or diffusion-process artefacts.
- Use top-activating sample grids and feature-level analysis to interpret SAE concepts.

---

### Adding temporal controls on top of pretrained generative models

**Adding Temporal Musical Controls on Top of Pretrained Generative Models**  
Nabi et al., 2025.

This project currently targets static single-character calligraphy images, so temporal controls are not part of v0. This paper is mainly useful as a comparison for how PluGeN-style control can be extended beyond the original setting.

---

### SAE steering in generative models

**Discovering and Steering Interpretable Concepts in Large Generative Music Models**  
Singh et al., 2025.

Relevant idea:

SAE-discovered concepts can be analyzed and potentially used for steering generative systems. Although the domain is music rather than images, it supports the broader idea that sparse features learned from generative model activations can expose interpretable concepts useful for control.

---

## Near-term coding priorities

The first coding pass should focus on runnable infrastructure rather than perfect research design.

Priority order:

1. Dataset loader compatible with existing calligraphy CSV/image setup.
2. VAE training and reconstruction visualization.
3. Unconditional DiT training or adaptation from existing code.
4. DiT activation extraction utilities.
5. Minimal SAE training on cached activations.
6. Feature visualization and pseudo-label export.
7. PluGeN placeholder / later implementation plan.

Do not over-optimize PluGeN before VAE, DiT, and SAE are working.

---

## Notes for Codex / vibe coding

Before writing major code, inspect:

```text
~/Code/calligraphy_project
~/Code/transfusion
```

Extract useful patterns for:

- CSV parsing
- local image path resolution
- dataset class design
- transforms and image normalization
- config organization
- training loop style
- checkpointing
- sample visualization
- sbatch launch style

The new repository should feel similar enough that the user can debug it using habits from those projects.

Avoid assuming a fixed CSV schema before inspecting the reference projects or the actual CSV. Use configurable column names where possible.

Current inspected CSV schema from `calligraphy_project/dataset/train_260402.csv`:

```text
path,font,char,code,label,part1,part2,author,author_code,new_text
```

Path handling should normalize Windows-style backslashes because the validation CSV may contain paths like `char_process\七字\七字 楷书 颜真卿.jpg`.
