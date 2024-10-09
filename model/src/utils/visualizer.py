import os

import pandas as pd
import plotly.express as px
import torch
from matplotlib import pyplot as plt
from utils.data_loader import DeviceDataLoader


class Visualizer:
    @staticmethod
    def make_inference(model, sentence, tokenizer, label_encoder):
        inputs = tokenizer(sentence, return_tensors="pt")

        input_ids = DeviceDataLoader.to_device(inputs["input_ids"], "cpu")
        attention_mask = DeviceDataLoader.to_device(inputs["attention_mask"], "cpu")

        model = DeviceDataLoader.to_device(model, "cpu")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        model = DeviceDataLoader.to_device(model, DeviceDataLoader.get_default_device())

        depart = label_encoder.inverse_transform(torch.max(outputs[0], 1).indices)
        arrival = label_encoder.inverse_transform(torch.max(outputs[1], 1).indices)

        print(sentence)
        print(f"Depart: {depart[0]}, Arrival: {arrival[0]}")

    @staticmethod
    def visualize_history_scores(history):
        val_acc = [x['val_acc'] for x in history]
        plt.plot(val_acc, '-bx')
        plt.xlabel('epoch')
        plt.ylabel('metrics')
        plt.legend(['Accuracy', 'F1 Score', 'ROC AUC'])
        plt.title('Metrics vs. No. of epochs')
        
    @staticmethod
    def visualize_csv(file_name, metric, column_names, title, x_axis, y_axis, path=None):
        """ metrics: val_acc, val_loss, train_acc, train_loss """

        columns = ["val_acc", "val_loss", "train_acc", "train_loss"]
        columns.remove(metric)

        df = pd.DataFrame()
    
        if path:
            csv_path = os.path.join(path, file_name)
        else:
            csv_path = os.path.join(os.getcwd(), "../../processed", file_name)

        history_df = pd.read_csv(csv_path)
        df = history_df.drop(columns=columns)
        df = df.reset_index()

        column_name = column_names[file_name.index(file_name)]
        df[column_name] = history_df[metric]

        fig = px.line(df, x=df.index, y=df.columns, title=title)
        fig.update_xaxes(title_text=x_axis)
        fig.update_yaxes(title_text=y_axis)
        
        fig.show()