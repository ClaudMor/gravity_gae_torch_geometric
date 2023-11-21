import torch
import numpy as np
from utils import reset_parameters
import copy
import time
from sklearn.metrics import roc_auc_score
from torcheval.metrics.functional import binary_f1_score, binary_auroc
import custom_losses



def train(train_data, model, train_loss_fn, _optimizer,device, num_epochs, lrscheduler = None, early_stopping = False, val_loss_fn = None, val_data = None, patience = None, use_sparse_representation = False, retrain_data = None, epoch_print_freq = 10): # , train_idxs = None, val_idxs = None, retrain_idxs = None
    
    model.train()
    reset_parameters(model)
    optimizer = _optimizer.__class__(model.parameters(), **_optimizer.defaults)

    initial_model_state_dict = None 
    initial_optimizer_state_dict = None
    if retrain_data is not None:
        initial_model_state_dict = model.state_dict()
        initial_optimizer_state_dict = optimizer.state_dict()


    
    ES_counter = 0
    ES_loss_previous_epoch = torch.tensor(0)
    val_losses = []
    train_losses = []

    y_true = train_data.edge_label.to(device)

    best_number_of_epochs = None
    for i in range(num_epochs):

        optimizer.zero_grad()
        pred = model(train_data)

        loss = train_loss_fn(pred, y_true)
        train_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        if lrscheduler is not None:
            lrscheduler.step(loss)

        
        if i % epoch_print_freq == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  epoch = {i+1} / {num_epochs}")

        
        if val_data is not None:

            val_loss = compute_loss_on_validation(val_data,  model, val_loss_fn, device, use_sparse_representation)


            if i>0 and early_stopping:
                if any(val_loss.item() >= previous_val_loss for previous_val_loss in val_losses): 
                    ES_counter += 1
                else:
                    ES_counter = 0
                if ES_counter > patience:
                    best_number_of_epochs = np.argmin(val_losses) + 1
                    print(f"val_losses = {val_losses[-10:]}, val_loss = {val_loss.item()},  ES_counter = {ES_counter} \n BREAKING. The best number of epochs is {best_number_of_epochs}")
                    break

                if i % 10 == 0:
                    print(f"val_losses = {val_losses[-5:]}, val_loss = {val_loss.item()},  ES_counter = {ES_counter}")

            val_losses.append(val_loss.item())


    if early_stopping:
        best_number_of_epochs = np.argmin(val_losses) + 1
        print(f"val_losses = {val_losses[-10:]}, ES_counter = {ES_counter} \n EPOCH LIMIT REACHED \n BREAKING. The best number of epochs is {best_number_of_epochs}")

    

    if early_stopping and retrain_data is not None:

        if best_number_of_epochs is None:
            best_number_of_epochs = np.argmin(val_losses) + 1

        print(f"\nRetraining on {best_number_of_epochs} epochs...\n")

        model.load_state_dict(initial_model_state_dict)
        optimizer.load_state_dict(initial_optimizer_state_dict)

        start = time.time()
        train(retrain_data, model, train_loss_fn, optimizer,device, best_number_of_epochs, lrscheduler=lrscheduler, val_data = val_data, val_loss_fn=val_loss_fn, early_stopping = False, use_sparse_representation = use_sparse_representation, epoch_print_freq = epoch_print_freq) 
        
        end = time.time()
        print(f"Training time: {end - start} seconds")



def compute_loss_on_validation(val_data, model, val_loss_fn, device, use_sparse_representation = False, eval = True):
    if eval:
        model.eval()

    with torch.no_grad():

        y_true = val_data.edge_label

        val_pred = model(val_data)

        val_loss = val_loss_fn(val_pred,y_true)

    if eval:
        model.train()

    return val_loss
    


@torch.no_grad()
def evaluate_link_prediction(model, test_data, test_data_mrr = None, k = 30, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ):


    model.cpu()

    model.eval()
    # Test data split
    test_data_split = copy.copy(test_data).cpu()
    neg_mask = (test_data.edge_label == 0).cpu()

    test_data_split.pos_edge_label_index = test_data.edge_label_index[:,~neg_mask].cpu()
    test_data_split.neg_edge_label_index  = test_data.edge_label_index[:,neg_mask].cpu()
    test_data_split.pos_edge_label = test_data.edge_label[~neg_mask].cpu()
    test_data_split.neg_edge_label = test_data.edge_label[neg_mask].cpu()

    del test_data_split.edge_label
    del test_data_split.edge_label_index


    
    # Test data rev
    test_data_rev_split = copy.copy(test_data_split).cpu()
    test_data_rev_split.neg_edge_label_index = test_data_rev_split.pos_edge_label_index[[1,0],:].cpu()
    test_data_rev_split.neg_edge_label = torch.zeros(test_data_rev_split.pos_edge_label.size(0)).cpu()

    # Test data rev
    test_data_rev = copy.copy(test_data_rev_split)
    test_data_rev.edge_label_index = torch.cat((test_data_rev.pos_edge_label_index, test_data_rev.neg_edge_label_index ), dim = 1).cpu()
    test_data_rev.edge_label = torch.cat((test_data_rev.pos_edge_label, test_data_rev.neg_edge_label ), dim = 0).cpu()

    del test_data_rev.pos_edge_label_index
    del test_data_rev.neg_edge_label_index
    del test_data_rev.pos_edge_label
    del test_data_rev.neg_edge_label


    logits_test_data = model(test_data).x.cpu()
    logits_test_data_rev = model(test_data_rev).x.cpu()

    _MRR = custom_losses.MRR(model, test_data_mrr.cpu()).cpu() if test_data_mrr is not None else None

    out_dict =  {"AUC" : roc_auc_score(test_data.edge_label.cpu(), logits_test_data), 
            "F1" : binary_f1_score(torch.nn.functional.sigmoid(logits_test_data), test_data.edge_label.cpu()),
            "hitsk" : custom_losses.hitsk(model, test_data_split, 30),
            "AP" : custom_losses.average_precision(model,test_data.cpu()),
            "MRR": _MRR,
            "AUC_rev" : binary_auroc(logits_test_data_rev, test_data_rev.edge_label),
            "F1_rev" : binary_f1_score(torch.nn.functional.sigmoid(logits_test_data_rev), test_data_rev.edge_label),
            "hitsk_rev" : custom_losses.hitsk(model, test_data_rev_split, 30),
            "AP_rev" : custom_losses.average_precision(model,test_data_rev)}
    
    model = model.to(device)

    return out_dict
