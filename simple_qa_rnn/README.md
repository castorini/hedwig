Setup:
1. Create 3 directories under "simple_qa_rnn" - "resources", "datasets", "saved_checkpoints" 
2. Download the SimpleQA dataset from [here](https://github.com/castorini/data) and put it under the "datasets" directory
3. Download these files from this [Dropbox link](https://www.dropbox.com/sh/e5g12v7zu7sgzf7/AACW272AqPZJIUC7-A40LAsNa?dl=0) and paste them in the "resources" directory
4. The directory structure should look like this now:
```
simple_qa_rnn
  ├── datasets
  │   └── SimpleQuestions_v2
  │       ├── ...
  ├── model.py
  ├── README.md
  ├── resources
  │   ├── rel_to_ix_SQ.pkl
  │   ├── w2v_map_SQ.pkl
  │   └── word_to_ix_SQ.pkl
  ├── saved_checkpoints
  │   └── [...models will be saved here later...]
  ├── scripts
  │   ├── ...
  ├── train.py
  └── util.py
```
5. Please take a look at the arguments in utils.py and set them accordingly to train the model.
6. Run this command to train the model. Make sure you have PyTorch and other Python dependencies installed.
```
python train.py 
```

NOTE: There are pre-trained models saved in the 'finished_checkpoints' directory. They can be loaded up using PyTorch.
You can run a pre-trained model on the test dataset:
```
python train.py --not_bidirectional --resume_snapshot finished_checkpoints/lstm1/[model_filename] --test 
```
