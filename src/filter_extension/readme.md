# SICK filter extension

### Original data from the experiment. All can be replicated by just starting from this data:

Full COMET files (25 COMET proposals per sentence): https://drive.google.com/drive/u/0/folders/1DrLchRhLRSsFeuLEsIaSGYFKQ6Dn3uNn

Filtered COMET files (1 COMET proposal per sentence): https://drive.google.com/drive/u/0/folders/10x1OPKiyD3arn1OweH21n4meBHreZfaT


## Pipeline
Note: all the jupyter notebooks and the scripts have the inputs at the begining of the file

### Files preprocessing
The preprocess_files.ipynb must be run. This jupyter notebook takes as input the original "Full COMET file" ("unfiltered") for training, testing and validation and the equivalent "Filtered COMET file" ("filtered"). It substitutes the sentences of the first file with the ones from the second ones, since the authors modified them. It is important in order to replicate the results. It also selects a subset of the training set

### At this point it is possible to calculate the similarities
we provide both a colab version, comet_similarities_colab.ipynb, and a "local" version,comet_similarities.ipynb. For specific guide to the prerequisites, you can use the steps_for_sbert.md file

### Training the shallow similarities predictor
For this step you will need the pikle files resulting from the previous step. The reference script is trainer.py

### Convex combination or random files production
This is done with the produce_filtered_comet.ipynb notebook. In case of the convex combination, the path to the model's weights must be specified. It also takes as input the pikle files from the similarities calculating steps. In the case of the random selection, the two original seeds are still in the code, for reproducibility purpouses, but by default they are commented out

### Train the original SICK model with the inputs from the previous step
Be aware that the file structure and the naming convention of the original paper must be replicated. For more information, https://github.com/SeungoneKim/SICK_Summarization . The original SICK code has been modified in order to allow for a reduced dataset, so the code from our version needs to be used. In order to simplify the process, we have left the colab notebook version, named train_sick_colab.ipynb.

### Everything else
Everything else is auxiliary, like the fuse_train.ipynb, which was used because the train subset had to be split into two, in order to be able to compute it. All the remaining files are either used as support, in a similar matter, or are used for testing purpouses.

