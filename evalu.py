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

def evalu(X, y, model):
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # make predictions
        y_pred_bin = (model(X).flatten()>0.5).float()
        # calculate accuracy score
        acc_score = accuracy(y_pred_bin, y).item()
        # calculate F1 score
        f1_score_val = f1_score(y_pred_bin, y).item()

        # print scores with descriptions
        print("Evaluation:")
        print(f"Accuracy score: {acc_score:.4f}")
        print(f"F1 score: {f1_score_val:.4f}")
