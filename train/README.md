# Training scripts
The scripts under this folder trains the SER model in FL. 

### 1. SER training in FL
The bash file federated_ser_classifier.sh provides an example of running the preprocess python file. e.g.:

```sh
python3 federated_ser_classifier.py --dataset iemocap --norm znorm \
                        --feature_type emobase --dropout 0.2 --num_epochs 200 --local_epochs 1 \
                        --optimizer adam --model_type fed_avg --learning_rate 0.0005 \
                        --save_dir /media/data/projects/speech-privacy

```
- The arg `dataset` specifies the data set. The support data sets are IEMOCAP (iemocap), MSP-Improv (msp-improv), and CREMA-D (crema-d). We also support combining two dataset, like iemocap_crema-d will be combination of two data set for training shadow ser model. 

- The arg `feature_type` is the feature reprentation type. Please refer to README under feature extraction for more details.

- The arg `pred` is prediction label. Currently support SER only, the arousal and valence predictions are ongoing work.

- The arg `norm` specifies the normalization method including z-normalization and min-max normalization. The normalization is implemented within a speaker.

- The arg `model_type` specifies the FL algorithm: fed_sgd and fed_avg.

- The arg `dropout` specifies the dropout value in the first dense layer.

- The arg `local_epochs` specifies the local epoch, ignored in fed_sgd.

- The arg `num_epochs` specifies the global epoch in FL.

- The arg `learning_rate` specifies the learning rate in FL.

- The arg `save_dir` specifies the root folder of saved data for the experiment.

### 1.1 This the code that average the weights in fed_avg

```python 
global_weights = average_weights(local_updates, local_num_sampels)
```
### 1.2 Our model forward code, two dense layers, and output

```python
x = self.dense1(x)
x = self.dense_relu1(x)
x = self.dropout(x)

x = self.dense2(x)
x = self.dense_relu2(x)
x = nn.Dropout(p=0.2)(x)

preds = self.pred_layer(x)
preds = torch.log_softmax(preds, dim=1)
```
