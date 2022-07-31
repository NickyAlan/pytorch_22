import torch
import numpy as np
import matplotlib.pyplot as plt

def accuracy_fn(y_true, y_pred) :
    '''
    accuracy function for training and testing
    return accuracy score
    '''
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_true) 
    return acc * 100

def print_train_time(start, end, device) :
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_cls_model(model, epochs, batch_size, train_dataloader, val_dataloader, loss_fn, optimizer, seed=42):
    '''
    return history and total train time
    '''
    # progress bar
    from tqdm.auto import tqdm
    from timeit import default_timer as timer
    
    torch.manual_seed(seed)
    start_time = timer()
    
    history = {'epochs':[_ for _ in range(epochs)]}
    history['train_loss'] = []
    history['train_acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []

    for epoch in tqdm(range(epochs)) :
        print(f'Epoch : {epoch}\n----')
        # for average Loss and Accuracy each epoch (reset every epoch)
        train_loss = 0 
        train_acc = 0
        
        # add a loop to loop through training batch
        for batch, (X, y) in enumerate(train_dataloader) : # batch size = 32
            model.train()
            y_logits = model(X)
            y_pred = torch.argmax(y_logits, dim=1) if y_logits.shape[-1] > 1 else torch.round(y_logits) # multiclass or binaryclass

            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y, y_pred)

            train_loss += loss # accumulatively for each epoch(all batch)
            train_acc += acc

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # Print out how many samples have been seen
            if batch == 0 or (batch+1) % (len(train_dataloader)//2) == 0 or (batch+1 == len(train_dataloader)): 
                print(f"Looked at {(batch+1) * batch_size}/{len(train_dataloader.dataset)} samples")

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        history['train_loss'].append(train_loss.detach().numpy().item())
        history['train_acc'].append(train_acc)

        # testing
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.inference_mode() :
            for X, y in val_dataloader :
                val_logits = model(X)
                val_pred = torch.argmax(val_logits, dim=1) if y_logits.shape[-1] > 1 else torch.round(y_logits) # multiclass or binaryclass

                val_loss += loss_fn(val_logits, y)
                val_acc += accuracy_fn(y, val_pred)

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
            
            history['val_loss'].append(val_loss.detach().numpy().item())
            history['val_acc'].append(val_acc)
        
        print(f"\nTrain loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | Val loss: {val_loss:.5f}, Val acc: {val_acc:.2f}%\n")

    # calculate training time
    end_time = timer()
    total_train_time_model = print_train_time(start_time, end_time, 'cpu')

    return history, total_train_time_model


def eval_cls_model(model, data_loader, loss_fn) :
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode() :
        for X, y in data_loader :
            y_logits = model(X)
            y_pred = torch.argmax(y_logits, dim=1)

            loss += loss_fn(y_logits, y)
            acc += accuracy_fn(y, y_pred)

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        'model_name' : model.__class__.__name__,
        'model_loss' : loss.item(),
        'model_acc' : acc
    }