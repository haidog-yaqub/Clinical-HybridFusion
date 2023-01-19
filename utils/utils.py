import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def get_eval(y_true, y_pred, cls):
    labels = [i for i in range(cls)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # tn, fp, fn, tp = cm.ravel()
    # specificity = tn / (tn + fp)

    cm = np.array(cm, dtype=int).T
    f1 = f1_score(y_true, y_pred, average='micro')
    f1_unweighted = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    # recall = recall_score(y_true, y_pred, average='micro')
    # specificity = recall_score(y_true, y_pred, average='micro', pos_label=0)
    # ppv = precision_score(y_true, y_pred, average='micro')
    # npv = precision_score(y_true, y_pred, average='micro', pos_label=0)

    return cm, f1, f1_unweighted, f1_weighted
    # return cm, f1, recall, specificity, ppv, npv


# evaluation function
def Evaluate(model, test_loader, loss_func, cls, device, epoch, path,
             language_model, log):
    model.eval()
    step = 0
    report_loss = 0
    predicts = []
    labels = []
    top_2 = 0
    top_3 = 0

    with torch.no_grad():
        for text, mask, age, others, label in tqdm(test_loader):
            text = text.to(device)
            mask = mask.to(device)

            age = age.to(device)
            others = others.to(device)

            label = label.to(device)

            pred = model(text, mask, age, others)
            loss = loss_func(pred, label)
            report_loss += loss.item()
            step += 1

            prediction = torch.argmax(F.softmax(pred, dim=1), dim=1)

            _, t2 = torch.topk(pred, 2, dim=1)
            correct_pixels = torch.eq(label[:, None, ...], t2).any(dim=1)
            top_2_acc = correct_pixels.sum().to('cpu')
            top_2 += int(top_2_acc)

            _, t3 = torch.topk(pred, 3, dim=1)
            correct_pixels = torch.eq(label[:, None, ...], t3).any(dim=1)
            top_3_acc = correct_pixels.sum().to('cpu')
            top_3 += int(top_3_acc)

            predicts += prediction.to('cpu').tolist()
            labels += label.to('cpu').tolist()

        print('Val Loss: {:.6f}'.format(report_loss / step))

    pd.DataFrame(np.array([predicts, labels]).T,
                 columns=['pred', 'label']).to_csv(path+language_model+'_'+str(epoch)+'.csv', index=False)
    model.train()

    top_2 = top_2/len(labels)
    top_3 = top_3/len(labels)

    # cm, f1, recall, specificity, ppv, npv = get_eval(np.array(predicts), np.array(labels), cls)
    cm, f1, f1_unweighted, f1_weighted = get_eval(np.array(predicts), np.array(labels), cls)
    top_1 = np.trace(cm) / np.sum(cm)

    n = open('log/' + language_model + '_' + log, mode='a')
    n.write('\n')
    n.write('Epoch: ' + str(epoch + 1))
    n.write('  Acc: {:.2f} %'.format(np.trace(cm) / np.sum(cm) * 100))
    n.write('  Top2 Acc: {:.2f} %'.format(top_2 * 100))
    n.write('  Top3 Acc: {:.2f} %'.format(top_3 * 100))
    n.write('  F1 Global: {:.2f} %'.format(f1 * 100))
    n.write('  F1 Unweighted: {:.2f} %'.format(f1_unweighted * 100))
    n.write('  F1 Weighted: {:.2f} %'.format(f1_weighted * 100))
    # n.write('  Recall: {:.6f}'.format(recall))
    # n.write('  Specificity: {:.6f}'.format(specificity))
    # n.write('  PPV: {:.6f}'.format(ppv))
    # n.write('  NPV: {:.6f}'.format(npv))
    n.close()

    return report_loss / step, top_1