# Attribution

## Kinetics-400

The AI-Edited class produced by this repository is derived from **Kinetics-400**, obtained via the
Hugging Face mirror
[`liuhuanjim013/kinetics400`](https://huggingface.co/datasets/liuhuanjim013/kinetics400).

- **Original dataset**: Kinetics-400
- **Original authors**: Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier,
  Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa
  Suleyman, Andrew Zisserman
- **Original paper**: *The Kinetics Human Action Video Dataset*,
  [arXiv:1705.06950](https://arxiv.org/abs/1705.06950)
- **Original licence**:
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### Changes made

Every AI-Edited video this pipeline produces is a **modified** Kinetics-400 clip. Source clips are
re-encoded and then altered by one of the manipulation models in `csf/generation/adapters/` — face
swapping, reenactment, lip-sync, expression editing, object insertion/removal, inpainting,
background replacement or whole-frame transformation. The generated `manifest.csv` records, per
row, which model produced the video (`model`), which specification slot it fills (`spec_model`)
and which Kinetics clip it came from (`source_clip_id`).

Datasets produced by this repository are released under **CC BY 4.0**, the same licence as the
source. `csf/generation/upload.py` writes this attribution into the dataset card and an
`ATTRIBUTION.md` on every push.

## Model weights

The manipulation models carry their own licences, which are not all the same. Two need attention:

- **REFace** — code is MIT, but its checkpoint is trained on CelebAMask-HQ and is restricted to
  **non-commercial research**. Videos it generates inherit that restriction. Disabled unless
  `generation.accept_noncommercial: true`.
- **Llama-3.2-11B-Vision-Instruct** — gated; accept the licence on its model page before use.

`python -m csf.generation.adapters` lists every model with its environment and notes.
