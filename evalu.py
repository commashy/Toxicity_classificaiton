import torch

def f1_score(y_pred, y):
    tp = torch.sum(y_pred*y).float()
    fp = torch.sum(y_pred*(1-y)).float()
    fn = torch.sum((1-y_pred)*y).float()
    f1 = 2*tp/(2*tp+fp+fn)
    return f1

def accuracy(y_pred, y):
    acc = torch.sum(y_pred==y).float()/len(y)
    return acc

def evalu(test_loader, test_set, best_model):

    # Evaluate the best model on the test set
    best_model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = best_model(x_batch).flatten()
            loss = best_model.loss_function(y_pred, y_batch)
            test_loss += loss.item()
            test_correct += ((y_pred > 0.5) == y_batch).sum().item()

    test_loss /= len(test_set)
    test_acc = test_correct / len(test_set)
    test_F1 = f1_score(y_pred, y_batch)
    print("Test Loss {:.4f}, Test Acc {:.2f}%,".format(test_loss, 100*test_acc))
    print("Test F1 {:.4f}".format(test_F1))
