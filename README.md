# Leap-LSTM

This repository contains source code (tensorflow-version) to reproduce the results presented in the paper [Leap-LSTM: Enhancing Long Short-Term Memory for Text Categorization](https://arxiv.org/abs/1905.11558) (IJCAI 2019).

```
@inproceedings{Leap-LSTM,
  title={Leap-LSTM: Enhancing Long Short-Term Memory for Text Categorization},
  author={Ting Huang, Gehui Shen, Zhi-Hong Deng},
  booktitle={IJCAI},
  year={2019}
}
```
## model
The architecture of Leap-LSTM:
![](/figures/model.png)



## results
The accuracies on text classification tasks:
![](/figures/results.png)



## usage
To run the codes, you need to

1. download datasets from the github repository [LEAM](https://github.com/guoyinwang/LEAM)
2. to train the model, use a command like:
```
python train_classifier_yelp.py --rnn_model leap-lstm --yelp_set FULL \
  --gpu 0 --if_schedule 0 --decay_start 1  --target_skip_rate 0.6 \
  --keep_prob_word 1.0 --skip_reg_weight 1.0 --max_gradient_norm 1.0 \
  --nb_epoches 5 
```
The details of arguments can be found in skiplstm.py



## other things
- tensorflow 1.11, maybe 1.12/1.13 is also okay
- Note that here we also provide the reproduction of [skip-rnn](https://arxiv.org/abs/1708.06834) and [skim-rnn](https://arxiv.org/abs/1711.02085).
