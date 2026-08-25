<<<<<<< Updated upstream
PDAC atlas code repository
=======
<h1 align="center">Cross-species single-cell atlases of pancreatic cancer</h1>

<p align="center">
  <em>Human and mouse PDAC atlases charting progression, therapy-driven remodelling and immune evasion</em>
</p>

<p align="center">
  <a href="https://doi.org/10.64898/2026.03.19.712924"><img src="https://img.shields.io/badge/bioRxiv-2026.03.19.712924-b31b1b" alt="bioRxiv"></a>
  <a href="https://scpdac.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-scPDAC-blue" alt="scPDAC documentation"></a>
  <a href="https://pypi.org/project/scPDAC/"><img src="https://img.shields.io/pypi/v/scPDAC?label=pip%20install%20scPDAC" alt="PyPI"></a>
  <a href="https://github.com/theislab/scPDAC"><img src="https://img.shields.io/badge/code-theislab%2FscPDAC-black?logo=github" alt="scPDAC repository"></a>
</p>

<p align="center">
  <img src="docs/figure1_workflow.jpg" alt="Workflow for constructing the translational human–mouse PDAC single-cell atlases" width="100%">
</p>

---

## About the project

Pancreatic ductal adenocarcinoma (PDAC) is usually diagnosed late, and the single-cell
data that exist are heavily skewed towards early, untreated, surgically resectable
disease. That gap makes it hard to study exactly what matters clinically: how tumours
progress, how the microenvironment is remodelled by therapy, and how the tumour escapes
immune control.

This repository contains the analysis code behind two integrated single-cell
transcriptomic atlases — one **human**, one **mouse** — built on a shared annotation
scheme so that the two can be compared directly, cell state by cell state. Together they
span early to advanced and metastatic disease, including post-treatment samples, and
cover **more than 1.6 million cells**.

Building them required stepping away from off-the-shelf workflows: standard highly
variable gene selection failed to align tumour and non-tumour compartments across
scRNA-seq and snRNA-seq, so feature selection combines expert-curated marker panels with
unsupervised matrix factorisation (MOFA), seven batch-correction methods were benchmarked
before settling on scANVI, malignant cells were called by inferCNV in human and by
*expressed lentiviral barcodes* in mouse, and the four-level annotation was validated by
a panel of independent PDAC experts.

### The atlases at a glance

|                        | Human                              | Mouse                                        |
| ---------------------- | ---------------------------------- | -------------------------------------------- |
| **Core atlas**         | 191 donors · >800,000 cells · 11 studies | 53 tumours · >390,000 cells (barcode-traced) |
| **Extended atlas**     | 257 donors · >1,000,000 cells · 16 studies | 101 tumours · >600,000 cells             |
| **Modalities**         | scRNA-seq + snRNA-seq, Xenium spatial | scRNA-seq                                 |
| **Sample types**       | primary tumour, adjacent normal, metastasis | orthotopic syngeneic allografts, autochthonous GEMMs |
| **Integration**        | scANVI on a curated gene panel     | scANVI on the orthologous panel               |
| **Extension**          | scArches reference mapping onto the core | scArches reference mapping onto the core |

### A shared, four-level cell-state taxonomy

The same hierarchy is applied to both species, which is what makes the cross-species
comparison possible:

| Level | Resolution | Examples |
| ----- | ---------- | -------- |
| **Level 1** | Major compartments (3) | epithelial · immune · stromal |
| **Level 2** | Major lineages (7) | malignant · lymphoid · myeloid · stromal · endothelial · exocrine · endocrine |
| **Level 3** | Fine cell types | CAF, macrophage, CD4⁺/CD8⁺ T cell, ductal, acinar, β cell, … |
| **Level 4** | Cell states | ten malignant programmes; T cell and macrophage substates |

The ten malignant Level-4 states: **epithelial, EMT, mesenchymal, hypoxic,
highly-proliferative, highly-invasive, senescent, apoptotic, pit-like, acinar-like** —
in total, **more than 60 distinct cell states** across the taxonomy.

### What the atlases show

- **A rare CD4⁺CD8⁺ double-positive T cell population** that is poorly characterised in
  PDAC, validated transcriptomically, spatially (Xenium, co-localising with TLS-like
  niches) and — in mouse — at the protein level by flow cytometry, and conserved across
  both species.
- **Radiotherapy is associated with a terminal EMT-persistent malignant state.** Patient
  composition trajectories converge on an EMT-persistent endpoint enriched for
  RT-exposed tumours, accompanied by endothelial expansion, T cell depletion, CAF
  enrichment and increased laminin–CD44 signalling; the RT-associated genes *MYO1E*,
  *CDK14* and *LPP* track with worse overall survival in TCGA-PAAD.
- **Mouse models map onto distinct human disease regimes.** Orthotopic allografts
  resemble advanced, EMT-enriched human tumours, whereas autochthonous GEMMs align with
  epithelial-like states — while malignant gene programmes and pathway activity remain
  strongly conserved across species.

---

## Use the atlases on your own data: `scPDAC`

The atlases are not only a result — they are a reference. [**scPDAC**](https://github.com/theislab/scPDAC)
packages the trained models so you can annotate a new PDAC dataset against them without
rebuilding anything.

```bash
pip install scPDAC
```

```python
import scpdac

# 1. Map a query into the reference latent space (scArches surgery; returns core + query)
expanded = scpdac.tl.extend_atlas(query, atlas, species="human", max_epochs=10)

# ...or the fast, no-surgery path: embed with the frozen model and transfer labels
query = scpdac.tl.embed_and_predict(query, "human")

# 2. Annotate with the hierarchical classifier — no reference object needed
scpdac.tl.predict_labels(adata, species="human", layer="log_norm")
```

Both paths align your genes to the model's panel automatically and write results back
into your `AnnData` (`obsm["X_scANVI_emb"]`, `obs["predicted_celltype"]`, `Level_1`–`Level_4`).

**Two workflows, pick by need**

| | `extend_atlas` | `embed_and_predict` | `predict_labels` |
| - | - | - | - |
| What it does | scArches surgery on the reference scANVI model, fine-tuned on your query | Embeds the query with the frozen reference model and transfers labels | Hierarchical MLP: malignant vs non-malignant, then a Level-3 sub-classifier per branch |
| Returns | expanded atlas (reference + query, shared embedding) | your query, embedded and labelled | your query, labelled |
| Use when | you want to grow the atlas, or your query has genuinely new batches | you want fast labels and an embedding, reference untouched | you only want labels, no reference object |

**Held-out benchmark** (two whole studies withheld per atlas — a cross-study transfer
estimate, not an in-distribution ceiling):

| Stage | Human weighted-F1 | Mouse weighted-F1 |
| ----- | ----------------- | ----------------- |
| Root — malignant vs non-malignant | 0.981 | 0.982 |
| Malignant Level-3 sub-classifier | 0.953 | 0.935 |
| Non-malignant Level-3 sub-classifier | 0.929 | 0.866 |
| **Combined hierarchy (end-to-end)** | **0.921** | **0.876** |

📚 **[Documentation](https://scpdac.readthedocs.io/en/latest/)** ·
🧬 **[Atlas mapping tutorial](https://scpdac.readthedocs.io/en/latest/notebooks/mapping.html)** ·
🏷️ **[Hierarchical classifier tutorial](https://scpdac.readthedocs.io/en/latest/notebooks/classifier.html)** ·
📊 **[Performance & limitations](https://scpdac.readthedocs.io/en/latest/performance.html)** ·
🔧 **[API reference](https://scpdac.readthedocs.io/en/latest/api.html)**

> Per-class F1 varies a lot between abundant and rare labels. Read the
> [performance page](https://scpdac.readthedocs.io/en/latest/performance.html) before
> treating any individual call as fact — routing errors at the root are unrecoverable.

---

## Repository structure

Everything lives under `code/`, and it splits into two tiers that are worth
understanding before you go looking for a particular panel:

> **`human/`, `mouse/` and `shared/` build the data. `figures/` turns that data into the
> figures in the paper.**

The species directories are pipelines: they take raw counts from each contributing study
and walk them through QC, feature selection, integration, hierarchical annotation and
reference-mapping extension, ending in the atlas objects (`.h5ad` / `.zarr`) and the
trained models. Nothing in them plots a manuscript figure. `figures/` does the opposite —
it assumes those objects already exist, loads them, and produces the panels. So if you
want to know *how a number was made*, read `human/`, `mouse/` or `shared/`; if you want
to know *how a panel was drawn*, read `figures/`.

```
code/
├── human/                          # ── BUILD: the human atlas ──
│   ├── notebooks/                  #   00–25, run in order: raw counts → extended atlas
│   │   ├── 00–05  …                #     per-study ingestion, filtering, RDS→h5ad, concatenation
│   │   ├── 06–08  …                #     sc/sn split, annotation, QC outliers, var-name harmonisation
│   │   ├── 09–11  …                #     inferCNV malignant calling, MOFA factors, manual gene panel
│   │   ├── 12–16  …                #     scPoli annotation refinement, binning, embeddings
│   │   ├── 17–22  …                #     scANVI integration, final 4-level annotation, expert validation
│   │   ├── 23–25  …                #     extension: prepare → scArches mapping → evaluation
│   │   └── README.md               #     per-notebook notes
│   ├── notebooks_downstream/       #   analyses on the finished atlas (CellRank/DPT trajectories,
│   │                               #     LIANA+ cell–cell communication, TCGA survival, biomarker & DGE)
│   └── scripts/                    #   compute-heavy jobs, SLURM-ready
│       ├── Integration_*/          #     the batch-correction benchmark: scVI, scANVI, scPoli,
│       │                           #       DRVI, sysVI, expiMap — plain and binned variants
│       ├── Metrics/                #     scIB metrics per batch key (dataset / donor / technology / all)
│       └── classifiers/            #     the MLPs shipped in scPDAC: malignant, malignant states, TME
│
├── mouse/                          # ── BUILD: the mouse atlas ──
│   ├── notebooks/                  #   01–15: barcoded in-house data → public datasets → integration
│   │                               #     01–03 barcoded ingestion, integration, QC + preliminary annotation
│   │                               #     04–07 non-barcoded ingestion, QC, label transfer, Level-1 refinement
│   │                               #     08–12 public datasets: import, merge, QC, integrate, refine
│   │                               #     13–15 Level-3 and Level-4 annotation, final scANVI integration
│   └── scripts/                    #   integration models + scIB metrics for the mouse benchmark
│
├── shared/                         # ── BUILD + COMPARE: the two species together ──
│   └── notebooks/                  #   01 harmonise metadata across atlases
│                                   #   02 cross-species integration (via HCOP orthologues)
│                                   #   03 max presence score (are human states present in mouse?)
│                                   #   04 compositional & transcriptomic similarity analysis
│
└── figures/                        # ── PLOT: the panels in the manuscript ──
    ├── 02_figure.ipynb             #   Fig. 2 — human core atlas, hierarchical annotation, markers
    ├── 02_figure_traj_DP.ipynb     #   Fig. 2 — DP T cell validation, diffusion-map densities, barplots
    ├── 02_figure_spatial.ipynb     #   Fig. 2 & 4 — Xenium niches with CellCharter, incl. the RT sections
    ├── 03_figure.ipynb             #   Fig. 3 — extended atlas, metadata distributions, classifiers
    ├── core_vs_extension.ipynb     #   Fig. 3 — core vs extension coherence (neighbourhood entropy)
    ├── 04_figure.ipynb             #   Fig. 4 — patient stratification, DRVI interpretation, LIANA+ CCC
    ├── 05_figure.ipynb             #   Fig. 5 — integrated mouse atlas
    ├── supplementary_figures/      #   Extended Data figures
    ├── config/                     #   config.yml — shared plotting defaults and the Level_1–4 palettes
    └── scripts/                    #   atlas_plots.py, plot_spatial.py — reusable plotting helpers
```

Cross-species panels (**Fig. 6**) come out of `shared/notebooks/03_max_presence_score.ipynb`
and `04_similarity_analysis.ipynb`, which both compute and plot.

One practical note on `figures/config/config.yml`: it holds the colour palettes for every
annotation level as well as the shared figure defaults (DPI, fonts, colourmaps). If you
re-plot anything, load it — that is what keeps colours consistent across the manuscript.

### Outside `code/`

| Path | Contents |
| ---- | -------- |
| `models/human/`, `models/mouse/` | trained scANVI, scVI, DRVI and scPoli models — including `pretrained_scanvi_extension`, the model used for reference mapping |
| `models/cell_classifiers/` | the three MLP checkpoints (malignant, malignant states, TME) that `scPDAC` ships |
| `supplementary_data/human/` | curated gene panel, MOFA factors and gene lists, annotation marker JSONs, donor metadata, and the eight expert validation votes |
| `supplementary_data/mouse/` | in-house and public sample metadata |
| `envs/` | conda environments (`preprocess.yml`, `scanpy_env.txt`, `scarches_env.txt`) |
| `references/` | placeholder for large reference files pulled in at runtime |

Large `.h5ad`/`.zarr` objects, figures and model outputs are **not** tracked here — see
`.gitignore`. To reproduce a figure notebook you need the corresponding atlas object,
either produced by the pipelines above or obtained via `scPDAC`.

---

## Citation

If you use the atlases, the code or the models, please cite:

> Lucarelli D, Parikh S, Jiménez S, Schneeweis C, *et al.* Cross-species single-cell
> atlases chart progression, therapy-driven remodelling and immune evasion in pancreatic
> cancer. *bioRxiv* (2026). doi:[10.64898/2026.03.19.712924](https://doi.org/10.64898/2026.03.19.712924)

```bibtex
@article{Lucarelli2026,
  author    = {Lucarelli, Daniele and Parikh, Shrey and Jim{\'e}nez, Sara and others},
  title     = {Cross-species single-cell atlases chart progression, therapy-driven
               remodelling and immune evasion in pancreatic cancer},
  journal   = {bioRxiv},
  year      = {2026},
  doi       = {10.64898/2026.03.19.712924},
  publisher = {Cold Spring Harbor Laboratory}
}
```

## Help and support

- Questions about the **analysis code or the atlases** → [open an issue here](https://github.com/theislab/PDAC/issues)
- Questions about the **`scPDAC` package** → [open an issue on scPDAC](https://github.com/theislab/scPDAC/issues)

## Related

- [**scPDAC**](https://github.com/theislab/scPDAC) — map and annotate your data against these atlases ([docs](https://scpdac.readthedocs.io/en/latest/))
- [scArches](https://github.com/theislab/scarches) — reference mapping framework used for the atlas extensions
- [scvi-tools](https://github.com/scverse/scvi-tools) — scVI / scANVI implementations
- [DRVI](https://github.com/theislab/drvi) — interpretable, disentangled integration used for the radiotherapy analysis
- [scIB](https://github.com/theislab/scib) — integration benchmarking metrics
>>>>>>> Stashed changes
