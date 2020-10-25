# OpenSet_ReciprocalPoints
Open-source, re-implementation of the published ECCV '20 paper on reciprocal points for open-set recognition. This paper is state-of-the-art in open-set recognition as of October 2020.

Code cleanup in progress. Shall be finished soon.

Confirmed with paper authors that this implementation is correct. Using dataloaders of the authors, the implementation actually exceeds the published performance. Using my own dataloaders, the results are slightly below the published performance (data splits are different, so this is probably why). Exact numbers coming soon.

Exact data splits that were used for experiments can be found [here for tiny imagenet](https://drive.google.com/file/d/1Q_VINXM1Z7YAvNQit9TbvNt50BV-7rZR/view?usp=sharing) and [here for cifar+10](https://drive.google.com/file/d/1wtR6wXIAq9GtiBe3kbHYVm7oIlud_bc2/view?usp=sharing).

All credit for the paper goes to the authors. Paper can be found here: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf.
