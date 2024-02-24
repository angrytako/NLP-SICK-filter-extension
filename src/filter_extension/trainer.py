import pandas as pd
import numpy as np
import torch
from torch import nn
from datetime import datetime
import itertools
from classifier import *

# paramters
learning_rate = 1e-6
num_iterations = 10000 #50000
num_classes = 25

# 1dCNN
num_channels = 1  # Number of channels in the input data
hidden_units = 64  # Number of units in the hidden layers
device = "cuda" if torch.cuda.is_available() else "cpu"

# saving and modality
mode = "train"
save_model = True
load_model = True
model_save_path = "./model"  # it will have automatically appended the current date
model_load_path = "model2024-02-08_12_45.pt"
test_file_path = "./comet_samsum_test_z_entire.pkl"
train_file_path = "./comet_samsum_train_z_entire.pkl"
# dictionary

ohe_dict = {'xNeed': np.array([1, 0, 0, 0, 0], dtype=np.float32),
            'xIntent': np.array([0, 1, 0, 0, 0], dtype=np.float32),
            'HinderedBy': np.array([0, 0, 1, 0, 0], dtype=np.float32),
            'xWant': np.array([0, 0, 0, 1, 0], dtype=np.float32),
            'xReason': np.array([0, 0, 0, 0, 1], dtype=np.float32)}

# read test
df_test = pd.read_pickle(test_file_path)

df_test["cs_type_ohe"] = df_test["cs_type"].map(ohe_dict)
preprocessed_test = df_test[["sample_id", "sentence_id", "cs_type_ohe", "cs_encoded", "cos_similary_cs_summmary"]]
preprocessed_test["complete_target_X"] = preprocessed_test[["cs_type_ohe", "cs_encoded"]].apply(
    lambda row: np.append(row[0], row[1]), axis=1)

test = preprocessed_test.groupby(['sample_id', 'sentence_id']).agg(
    X=("complete_target_X", lambda x: list(itertools.chain.from_iterable(x))), Y=('cos_similary_cs_summmary', list))

X_test = torch.tensor(test["X"]).to(device)
Y_test = torch.tensor(test["Y"]).to(device)


# choose model
model_type = "simple"  # "reduction"  # can be either that or "simple"

#read train 
df = pd.read_pickle(train_file_path)
#df2 = pd.read_pickle("./comet_samsum_train_pt2_z_entire.pkl")
#df = pd.concat([df, df2], ignore_index=True)

df["cs_type_ohe"] = df["cs_type"].map(ohe_dict)
preprocessed = df[["sample_id", "sentence_id", "cs_type_ohe", "cs_encoded", "cos_similary_cs_summmary"]]
preprocessed["complete_target_X"] = preprocessed[["cs_type_ohe", "cs_encoded"]].apply(
    lambda row: np.append(row[0], row[1]), axis=1)

new_df = preprocessed.groupby(['sample_id', 'sentence_id']).agg(
    X=("complete_target_X", lambda x: list(itertools.chain.from_iterable(x))), Y=('cos_similary_cs_summmary', list))

new_df.reset_index()
if model_type == "simple":
    model = Classifier(dim_input=9725, num_classes=num_classes).to(device)
if model_type == "reduction":
    # model = CNN1D(sequence_length=9725, num_classes=25).to(device)
    dim_input_single = 389  # sarebbero 384 ma ho fatto l'encoding dell'intention
    model = ReductionModel(dim_input_single=dim_input_single, num_classes=num_classes, device=device)

X = torch.tensor(new_df["X"]).to(device)
# getting the probs for the loss
Y = torch.tensor(new_df["Y"]).to(device)


# Load the model if load_model is True
if load_model:
    model_load_path = "./" + model_load_path  # Ensure this is the correct path
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    print(f"Model loaded from {model_load_path}")


if mode == "train":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for t in range(num_iterations):
        # Forward pass: compute predicted y using operations on Tensors.
        optimizer.zero_grad()
        # pred_logits= model.forward(X.view(X.shape[0], 1, X.shape[1]))
        pred_logits = model.forward(X)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (((pred_logits - Y).pow(2).sum(dim=-1)) * 1 / len(Y)).sum()
        if t % 100 == 99:
            print(t, loss.item())
            '''
            model.eval()  # Set the model to evaluation mode
            evaluate(X_val, Y_val, model,lambda pred_logits, Y_val: (((pred_logits - Y_val).pow(2).sum(dim=-1)) * 1 / len(Y)).sum())
            model.train()  # Set the model back to training mode
            
            '''

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()
        optimizer.step()


def test(X, Y, model, loss_fn):
    test_loss, correct = 0, 0
    with torch.no_grad():
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        test_loss = loss_fn(pred, Y)
        # correct =
        for i, similarities in enumerate(pred):
            correct += int(similarities.argmax() == Y[i].argmax())
    print(f"Test Error: \n Accuracy: {(100 * correct / len(Y))}%, Avg loss: {test_loss:>8f} \n")


### TESTING THE MODEL
mode = "test"

if mode == "test":
    test(X_test, Y_test, model,
         lambda pred_logits, Y_test: (((pred_logits - Y_test).pow(2).sum(dim=-1)) * 1 / len(Y_test)).sum())

if save_model:
    torch.save(model.state_dict(), f"{model_save_path}{datetime.today().strftime('%Y-%m-%d_%H_%M')}.pt")
