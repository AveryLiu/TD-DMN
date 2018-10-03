# TD-DMN

The PyTorch implementation of our EMNLP18 short paper *Exploiting Contextual
Information via Dynamic Memory Network for Event Detection*.

![The detailed TD-DMN model](./figures/detailed.jpg)

### Processed Data
Due to license reason, the ACE 2005 dataset is only accessible to those with **LDC2006T06** license,
please drop me an email showing your possession of the license for the processed data.

### Requirements
``Python 3.6.2`` is used while building the TD-DMN model, other versions of python are not tested.
The required external packages can be installed using ``pip install -r requirements.txt``.

### Quick Start
```
python train.py --fold_num=0 --identifier="TDDMN" --max_train_epoch=200 --patience=96 \
                --tensorboard_log_dir="./runs" --result_log_dir="./results" --data_dir="./data" \
                --vec_name="GoogleNews-vectors-negative300.txt" --vec_cache=".vector_cache/"
 ```
 
 Before running the above command, processed data files and pre-trained word vectors should be in place under
 ``--data_dir`` and ``--vec_cache`` respectively. The explanation for each argument can be found by
 ``python train.py -h``.
