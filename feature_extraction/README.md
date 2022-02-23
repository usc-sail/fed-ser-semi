# Speech feature extraction

## 1. OpenSMILE features
We extract speech features of human knowledge using OpenSMILE toolkit. You can refer to [OpenSMILE](https://www.audeering.com/research/opensmile/) for more information.

### Quick example
The bash file opensmile_feature_extraction.sh provides an example of running the feature extraction python file. e.g.:

```sh
taskset 100 python3 opensmile_feature_extraction.py --dataset iemocap --feature_type emobase \
                            --data_dir /media/data/sail-data/iemocap \
                            --save_dir /media/data/projects/speech-privacy
 
```
The support data sets are IEMOCAP, MSP-Improv, and CREMA-D. The support feature_type args are emobase, ComParE, and GeMAPS.

## 2. Pretrained features

We extract a variety of deep speech representations using pretrained models. You can refer to [SUPERB](https://arxiv.org/abs/2105.01051) paper for their model architures and pre-training loss styles.

Publication Date | Model | Name | Paper | Input | Stride | Pre-train Data | Official Repo 
|---|---|---|---|---|---|---|---
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [APC](https://github.com/Alexander-H-Liu/NPC)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [NPC](https://github.com/Alexander-H-Liu/NPC)
25 Oct 2019 | Mockingjay | mockingjay | [arxiv](https://arxiv.org/abs/1910.12638) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [NPC](https://github.com/Alexander-H-Liu/NPC)
Dec 3 2019 | DeCoAR | decoar | [arxiv](https://arxiv.org/abs/1912.01679) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [speech-representations](https://github.com/awslabs/speech-representations)
Dec 11 2020 | DeCoAR 2.0 | decoar2 | [arxiv](https://arxiv.org/abs/2012.06659) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [speech-representations](https://github.com/awslabs/speech-representations)
Oct 5 2021 | DistilHuBERT | distilhubert | [arxiv](https://arxiv.org/abs/2110.01900) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | [S3PRL](https://github.com/s3prl/s3prl)


### Quick example
The bash file pretrained_audio_feature_extraction.sh provides an example of running the feature extraction python file to extract APC feature. e.g.:

```sh
taskset 100 python3 pretrained_audio_feature_extraction.py --dataset iemocap \
                            --feature_type apc \
                            --data_dir /media/data/sail-data/iemocap \
                            --save_dir /media/data/projects/speech-privacy
```
The support data sets are IEMOCAP, MSP-Improv, and CREMA-D. The support feature_type args are apc, vq_apc, tera, decoar2, mockingjay, npc, and distilhubert.


## 3. Referecences


**[OpenSMILE](https://www.audeering.com/research/opensmile/)**
```
@inproceedings{eyben2010opensmile,
  title={Opensmile: the munich versatile and fast open-source audio feature extractor},
  author={Eyben, Florian and W{\"o}llmer, Martin and Schuller, Bj{\"o}rn},
  booktitle={Proceedings of the 18th ACM international conference on Multimedia},
  pages={1459--1462},
  year={2010}
}
```

**[SUPERB](https://arxiv.org/abs/2105.01051)**

```
@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
