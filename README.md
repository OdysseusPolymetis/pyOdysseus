![pyOdysseus interface screenshot](https://github.com/OdysseusPolymetis/pyOdysseus/blob/main/screen_example.png?raw=true)

# pyOdysseus

**pyOdysseus** is an alignment toolkit and interface based on a modified version of [bertalign](https://github.com/bfsujason/bertalign). It is designed for multilingual text alignment and includes notebooks for alignment, visualisation, and sentence-transformer fine-tuning.

## Contents

This repository contains several types of notebooks:

- **`pyOdysseus_interface.ipynb`**  
  A Gradio-based interface for aligning texts in many of the languages covered by LaBSE. This is currently the most immediately usable notebook for most users.

- **`labse_finetuning.ipynb`**  
  A notebook for fine-tuning a `sentence-transformers` model that can later be reused for alignment.

- **`pyOdysseus.ipynb`**  
  An older workflow that generates HTML files designed for use with the `paralogos` visualisation interface.

- Other experimental or prototype notebooks related to earlier stages of the project.

## Main features

- Sentence alignment based on a modified version of **bertalign**
- Support for multilingual workflows
- Optional fine-tuning of a custom **SentenceTransformer / LaBSE-based** model
- HTML export for visualisation
- Gradio interface for easier use in Colab

## Requirements

The modified version of `bertalign` used here works best with a **GPU**.

If you do not have a GPU, you may need to install:

```bash
pip install faiss-cpu
```

instead of the GPU version of FAISS. The code can still run, but alignment will be significantly slower.

## How to use

### 1. Input files

The notebooks take `.txt` files as input.

By default, texts are split whenever the regex `Chant\d+` is encountered.  
This behaviour can be modified directly in the notebook if your corpus uses another structure.

---

### 2. Using `pyOdysseus.ipynb`

This notebook produces a series of `.html` files based on the bibliography / corpus settings you define in the notebook.

These files can then be placed in the `html` folder of `paralogos`, which handles the visualisation layer.

Once the files are in place, and after updating `biblio.js`, you can launch the visualisation locally with:

```bash
python -m http.server
```

Then open the local address shown in your terminal, typically:

```text
http://localhost:8000/
```

---

### 3. Using `pyOdysseus_interface.ipynb`

This notebook provides a Gradio interface.

To use it:

1. Run the cells in order.
2. Ignore the optional model-upload cells unless you want to use a custom fine-tuned model.
3. Once the notebook has finished loading, Gradio will appear.
4. You can then upload your own texts and run the alignment directly.

## Notes on the aligner

The aligner is currently provided as a notebook-based workflow (`.ipynb`), but it can easily be adapted into a `.py` script if needed.

The repository includes a modified version of `bertalign`, located in `bertalign_odysseus`, which is used by the notebooks.

## Credits

This project is based on the original [bertalign](https://github.com/bfsujason/bertalign).

**Lei Liu & Min Zhu.** 2022.  
*Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts*,  
*Digital Scholarship in the Humanities*.  
https://doi.org/10.1093/llc/fqac089

I would also like to warmly thank **Gérard Greco**, who shared sequence-level aligned data that made it possible to refine the **LaBSE** model used in this project.

## Citation

If you use this repository in research, teaching, or publications, please cite the following book:

**Marianne Reboul**, *Comparaison semi-automatique des traductions françaises de l’Odyssée d’Homère (1547-1955)*, Paris, Classiques Garnier, coll. *Cultures et pratiques savantes du numérique*, 2022, 508 p., DOI: 10.48611/isbn.978-2-406-12961-5.

This repository builds directly on the research developed in this publication.

## Main libraries

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [faiss](https://github.com/facebookresearch/faiss)
- [sentence-splitter](https://github.com/mediacloud/sentence-splitter)
