# OpenSet_ReciprocalPoints
Open-source, re-implementation of the published ECCV '20 paper on reciprocal points for open-set recognition. This paper is state-of-the-art in open-set recognition as of October 2020.

Code cleanup in progress; results on test set will be updated soon.

Confirmed with paper authors that this implementation is correct. Using dataloaders of the authors, the implementation actually exceeds the published performance on tiny-imagenet. Using my own dataloaders, the results are slightly below the published performance (data splits are different, so this is probably why).

I also ran a standard deep learning baseline on both of these datasets. I actually find that the baseline has been underestimated by the current open-set literature; the current literature reports a much lower number for the baseline than the one I obtained with my own code. This may signal that the progress in open-set recognition is much more modest than it would seem.

| Method | CIFAR+10 | Tiny Imagenet |
| --- | --- | --- |
| Published baseline| 81.6% | 57.7% |
| My implementation of baseline| 89.24% (val) | 66.35% (val) |
| My implementation of RPL | 89.79% (val) | 67.11% (val) |
| Published RPL | 84.2% | 68.8% |

Exact data splits that were used for experiments can be found [here for tiny imagenet](https://drive.google.com/file/d/1Q_VINXM1Z7YAvNQit9TbvNt50BV-7rZR/view?usp=sharing) and [here for cifar+10](https://drive.google.com/file/d/1wtR6wXIAq9GtiBe3kbHYVm7oIlud_bc2/view?usp=sharing).

The images for Tiny Imagenet can be obtained by running '''wget http://cs231n.stanford.edu/tiny-imagenet-200.zip'''.

All credit for the paper goes to the authors. Paper can be found here: https://arxiv.org/abs/2011.00178.

Citation: @misc{chen2020learning,
      title={Learning Open Set Network with Discriminative Reciprocal Points}, 
      author={Guangyao Chen and Limeng Qiao and Yemin Shi and Peixi Peng and Jia Li and Tiejun Huang and Shiliang Pu and Yonghong Tian},
      year={2020},
      eprint={2011.00178},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
