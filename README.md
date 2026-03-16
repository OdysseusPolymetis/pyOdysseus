![alt text](https://github.com/OdysseusPolymetis/pyOdysseus/blob/main/screen_example.png?raw=true)

# pyOdysseus

An alignment interface (aligner based on bertalign), adaptable for several languages.

## What you'll find in it

The first thing you'll be looking for is the aligner, which for now is an `.ipynb` file, which you can easily transform into a `.py` file. Once you have the colab, you'll get a direct link to clone the modified version of <a href="https://github.com/bfsujason/bertalign">bertalign</a>, which will only work if you have a GPU (you need to `!pip install faiss-cpu` otherwise, but the process will be much slower).
<br>You'll find four kinds of notebooks here. The first kind, `labse_finetuning.ipynb` can be used to finetune a `sentence-transformer` that can then be used for alignment. The second kind is the older code that creates files for the `paralogos` interface (css joined in `paralogos`). The third kind is a prototype, a user interface with Gradio for aligning virtually any language included in `labse`. This last notebook, maybe the most useful, is `pyOdysseus_interface.ipynb`.

## How to use

The notebooks take `.txt` files as input, and subdivides those files when encountering the regex `Chant\d+`. You can of course modify this behaviour directly in the notebook.
<br>If you're using the **`pyOdysseus.ipynb` notebook**, in the end, you'll get a series of `html`files (according to the `biblio` part you'll have modified in the notebook), which you can then store in the `html` folder within `paralogos`, which implements the visualisation.
<br>Once your files are stored (and you have update the `biblio.js` file, you can run the visualisation by going to your terminal, navigate to your local folder and run the python command `python -m http.server`. This should enable you to access your visualisation at your localhost, `http://[::]:8000/`.
<br>If you're using the **`pyOdysseus_interface` notebook**, just run every cell until the end (except the ones that ask for potential model uploads), Gradio will automatically appear and you'll be able to load your own texts.

## Credits

Based on the original <a href="https://github.com/bfsujason/bertalign">bertalign</a>
<br>Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, Digital Scholarship in the Humanities. <a href="https://doi.org/10.1093/llc/fqac089">https://doi.org/10.1093/llc/fqac089</a>.
<br>
<br>Je remercie grandement aussi **Gérard Greco** qui m'a transmis des données alignées au niveau de la séquence qui m'ont permis d'affiner le modèle **LaBSE**.

## Citation

If you use this repository in research, teaching, or publications, please cite the associated book:
**Marianne Reboul**, *Comparaison semi-automatique des traductions françaises de l’Odyssée d’Homère (1547-1955)*, Paris, Classiques Garnier, coll. *Cultures et pratiques savantes du numérique*, 2022, 508 p., DOI: 10.48611/isbn.978-2-406-12961-5. :contentReference[oaicite:0]{index=0}

This repository builds on the research developed in this publication.

##### Main Libraries

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

* [faiss](https://github.com/facebookresearch/faiss)

* [sentence-splitter](https://github.com/mediacloud/sentence-splitter)
