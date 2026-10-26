# OG-SED paper demo

This offline bundle presents the manuscript *Ontology-Guided Knowledge Transfer and Class-Adaptive Post-Processing for Query-Based Sound Event Detection* in English. The homepage displays the paper's PSDS ablation and three real AudioSet-Strong validation clips with strong labels and saved per-stage event predictions.

Run a local web server from the directory containing `docs/`, then open `http://localhost:8000/docs/`:

```bash
python -m http.server 8000
```

Opening `index.html` directly as a `file://` URL may prevent the browser from loading its JSON data. No model server or internet connection is needed.

The reported PSDS percentages are dataset-level values from the manuscript. The clip-level overlap F1 is an illustrative comparison at a 0.50 event threshold, not PSDS. The selected clips do not form a single monotonic four-stage example. In particular, the saved A2 and A3 validation outputs load different model checkpoints; the CAMF clip is a stage comparison with a changed class window, **not** proof of a filter-only causal effect.

The three audio files are excerpts of the local AudioSet-Strong validation data. Check the dataset's redistribution terms before publishing the ZIP publicly.
