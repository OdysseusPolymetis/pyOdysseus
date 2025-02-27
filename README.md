<center>![alt text](https://github.com/OdysseusPolymetis/pyOdysseus/blob/main/screen_example.png?raw=true)</center>

# pyOdysseus
An alignment interface (aligner based on bertalign), adaptable for several languages (although you may have to get ready for a long and tedious code).
## What you'll find in it
The first thing you'll be looking for is the aligner, which for now is an `.ipynb` file, which should be rewritten to a fully working `.py` soon. Once you have the colab, you'll get a direct link to clone the modified version of <a href="https://github.com/bfsujason/bertalign">bertalign</a>, which will only work if you have a GPU (you need to `!pip install faiss-cpu` otherwise, but the process will be much slower).
<br>You'll also find a jupyter notebook, usable on colab, and the `paralogos`folder, which contains the `css` and `js` files for visualisation.
## How to use
The notebook takes `.txt` files as input, and subdivides those files when encountering the regex `Chant\d+`. You can of course modify this behaviour directly in the notebook.
<br>Once you've run the entire notebook, you'll get a series of `html`files (according to the `biblio` part you'll have modified in the notebook), which you can then store in the `html` folder within `paralogos`, which implements the visualisation.
<br>Once your files are stored (and you have update the `biblio.js` file, you can run the visualisation by going to your terminal, navigate to your local folder and run the python command `python -m http.server`. This should enable you to access your visualisation at your localhost, `http://[::]:8000/`.
## Credits
Based on the original <a href="https://github.com/bfsujason/bertalign">bertalign</a>
<br>Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, Digital Scholarship in the Humanities. <a href="https://doi.org/10.1093/llc/fqac089">https://doi.org/10.1093/llc/fqac089</a>.
##### Main Libraries

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

* [faiss](https://github.com/facebookresearch/faiss)

* [sentence-splitter](https://github.com/mediacloud/sentence-splitter)
