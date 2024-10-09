import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.data_loader import DeviceDataLoader
from utils.metrics import Metrics


class Trainer:
    def __init__(
        self, 
        model, 
        data_loader_train, 
        data_loader_valid, 
        learning_rate = 2e-5,
        optimizer = None, 
        loss_fn = nn.CrossEntropyLoss(), 
    ):
        self.model = model
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = []

        if self.optimizer is None:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate
            )


    def train_epoch(self, current_epoch):
        model = model.train()
        losses_train = []
        losses_valid = []
        correct_predictions_train = 0
        correct_predictions_valid = 0

        device = DeviceDataLoader.get_default_device()
        with tqdm(total=len(self.data_loader_train), desc=f"Epoch {current_epoch}", unit="batch") as pbar:
            for d in self.data_loader_train:
                input_ids = DeviceDataLoader.to_device(d["input_ids"], device)
                attention_mask = DeviceDataLoader.to_device(d["attention_mask"], device)
                labels_depart = DeviceDataLoader.to_device(d["departure"], device)
                labels_arrival = DeviceDataLoader.to_device(d["arrival"], device)

                self.optimizer.zero_grad()

                outputs_depart, outputs_arrival = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss_depart = self.loss_fn(outputs_depart, labels_depart)
                loss_arrival = self.loss_fn(outputs_arrival, labels_arrival)
                loss = loss_depart + loss_arrival

                correct_predictions_train += (outputs_depart.argmax(1) == labels_depart).sum().item()
                correct_predictions_train += (outputs_arrival.argmax(1) == labels_arrival).sum().item()

                losses_train.append(loss.item())

                loss.backward()
                self.optimizer.step()

                pbar.update(1)

        model = model.eval()

        for d in self.data_loader_valid:
            input_ids = DeviceDataLoader.to_device(d["input_ids"], device)
            attention_mask = DeviceDataLoader.to_device(d["attention_mask"], device)
            labels_depart = DeviceDataLoader.to_device(d["departure"], device)
            labels_arrival = DeviceDataLoader.to_device(d["arrival"], device)

            outputs_depart, outputs_arrival = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss_depart = self.loss_fn(outputs_depart, labels_depart)
            loss_arrival = self.loss_fn(outputs_arrival, labels_arrival)
            loss =  loss_depart + loss_arrival

            correct_predictions_valid += (outputs_depart.argmax(1) == labels_depart).sum().item()
            correct_predictions_valid += (outputs_arrival.argmax(1) == labels_arrival).sum().item()

            losses_valid.append(loss.item())

        train_acc = correct_predictions_train / (2 * len(self.data_loader_train.dataset))
        train_loss = np.mean(losses_train)

        valid_acc = correct_predictions_valid / (2 * len(self.data_loader_valid.dataset))
        valid_loss = np.mean(losses_valid)

        return {"train_acc": train_acc, "train_loss": train_loss, "valid_acc": valid_acc, "valid_loss": valid_loss}

    def run_trainer(self, n_epochs):
        for epoch in range(n_epochs):
            results = self.train_epoch(epoch)
            self.model.epoch_end(epoch, results)
            print(results)
            self.history.append(results)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = Metrics.accuracy(out, labels)
        _, preds = torch.max(out, dim=1)
        probs = F.softmax(out, dim=1)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'preds': preds, 'labels': labels, 'probs': probs}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        metrics = Metrics(outputs)

        f1 = metrics.f1_score()
        roc_auc = metrics.roc_auc_score()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'f1': f1, 'roc_auc': roc_auc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score:{:.4f}, roc_auc_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['f1'], result['roc_auc']))

    @staticmethod
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [Trainer.validation_step(model, batch) for batch in val_loader]
        return Trainer.validation_epoch_end(model, outputs)
