# VIBE

VIBE (Voice-Invariant Biometric Embedding) is a research codebase for speaker verification, focusing on robust metric learning and angular margin losses.

This repository currently contains the implementation for ChebyAAM, a stable angular margin loss based on Chebyshev polynomial approximation. The work is accepted by **ICASSP 2026**:

> **The Achilles' Heel of Angular Margins: A Chebyshev Polynomial Fix for Speaker Verification**
> Yang Wang, Yiqi Liu, Chenghao Xiao, Chenghua Lin
> [[arXiv](https://www.arxiv.org/pdf/2601.13198)]

## Codebase Acknowledgement

A significant portion of this codebase is **adapted and extended from** the excellent open-source project [3D-Speaker](https://github.com/modelscope/3D-Speaker) Toolkit.
We sincerely thank the authors for releasing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026achillesheelangularmargins,
      title={The Achilles' Heel of Angular Margins: A Chebyshev Polynomial Fix for Speaker Verification}, 
      author={Yang Wang and Yiqi Liu and Chenghao Xiao and Chenghua Lin},
      year={2026},
      eprint={2601.13198},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.13198}, 
}
```
