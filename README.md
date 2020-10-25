# OpenSet_ReciprocalPoints
Open-source, re-implementation of the published ECCV '20 paper on reciprocal points for open-set recognition. This paper is state-of-the-art in open-set recognition as of October 2020.

Code cleanup in progress. Shall be finished soon.

Confirmed with paper authors that this implementation is correct. Using dataloaders of the authors, the implementation actually exceeds the published performance on tiny-imagenet. Using my own dataloaders, the results are slightly below the published performance (data splits are different, so this is probably why).

I also ran a standard deep learning baseline on both of these datasets. I actually find that the baseline has been underestimated by the current open-set literature; the current literature reports a much lower number for the baseline than the one I obtained with my own code. This may signal that the progress in open-set recognition is much more modest than it would seem.

| Method | CIFAR+10 | Tiny Imagenet |
| --- | --- | --- |
| My implementation of baseline| 89.24% (val) | 66.35% (val) |
| My implementation of RPL | 89.79% (val) | 67.11% (val) |
| Published RPL | 84.2% | 68.8% |

Exact data splits that were used for experiments can be found [here for tiny imagenet](https://drive.google.com/file/d/1Q_VINXM1Z7YAvNQit9TbvNt50BV-7rZR/view?usp=sharing) and [here for cifar+10](https://drive.google.com/file/d/1wtR6wXIAq9GtiBe3kbHYVm7oIlud_bc2/view?usp=sharing).

All credit for the paper goes to the authors. Paper can be found here: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf.
