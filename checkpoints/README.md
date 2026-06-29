# Released Weights

This directory is intended for Git LFS or GitHub Release assets.

| File | Purpose | SHA256 |
| --- | --- | --- |
| `htast_mga.pt` | HTS-AT/MGA initialization checkpoint | `75171df05e919ba4714972b6882d5071792e30d75631a9af3a2977d1ef392546` |
| `dasm_audio_query_baseline.pt` | DASM audio-query baseline checkpoint for OCL+CAW initialization | `e463c2d3f8591ba5cc85f04a69e238378f85c91de15c44c9d4e62ac8fa16c736` |
| `ocl_caw_best_model.pt` | Final OCL+CAW checkpoint used by CAMF evaluation | `0886e171db58bb493c2d0b04f72548cf08bf06f3460995c3be44d9ca71fcc814` |

GitHub rejects normal Git blobs larger than 100 MB. Use one of these release paths:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes weights/*.pt
```

or upload the checkpoint files to a GitHub Release / Hugging Face Hub and keep only download instructions in this directory.
