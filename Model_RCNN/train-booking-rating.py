from utils import *
from model import *
import sys
import torch.optim as optim
from torch import nn
import torch
from sklearn.metrics import accuracy_score, classification_report


def calc_true_and_pred(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    return all_y, all_preds



train_file = '../data/booking-rating-train.csv'
test_file = '../../Detect-emotion-sentimental/dataset/booking/booking-rating-test.csv'
val_file = '../data/booking-rating-val.csv'
w2v_file = '../data/ubercorpus.cased.tokenized.word2vec.300d'




class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 64
    output_size = 5
    max_epochs = 64
    hidden_size_linear = 64
    lr = 0.5
    batch_size = 8
    seq_len = None # Sequence length for RNN
    dropout_keep = 0.8


config = Config()
dataset = Dataset(config)
dataset.load_data(w2v_file, train_file, test_file, val_file)



model = RCNN(config, len(dataset.vocab), dataset.word_embeddings)
if torch.cuda.is_available():
    model.cuda()
model.train()
optimizer = optim.SGD(model.parameters(), lr=config.lr)
NLLLoss = nn.NLLLoss()
model.add_optimizer(optimizer)
model.add_loss_op(NLLLoss)



train_losses = []
val_accuracies = []

for i in range(config.max_epochs):
    print("Epoch: {}".format(i))
    train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)



all_y, all_preds = calc_true_and_pred(model, dataset.test_iterator)

test_acc = accuracy_score(all_y, np.array(all_preds).flatten())
print ('Final Test Accuracy: {:.4f}'.format(test_acc))


print(classification_report(all_y, np.array(all_preds).flatten()))



torch.save(model.state_dict(), 'rcnn_booking_rating')




