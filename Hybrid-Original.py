import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')  # Backend plotting mode
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from PIL import Image
import sys
import csv
import torchvision.models as models

#################### User Configurable Parameters ####################
data_dir = "./data"    # Data directory (ensure the corresponding data is present at this path on the cloud server)
IMG_SIZE = 224
BATCH = 16
EPOCHS_SEARCH = 5     # Fewer epochs during hyperparameter search
EPOCHS_FINAL = 10     # Number of epochs for final training
STEP = 8
GAMMA = 0.1
#################### Hyperparameter Search Range #####################
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.0, 1e-4]
loss_functions = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'nll_loss': nn.NLLLoss()
}
optimizers_list = ['sgd', 'adam']
########################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = 'resnet50'

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            if self.BN_enable:
                self.norm2 = nn.BatchNorm2d(out_channels)
        elif self.upsample_mode == 'pixelshuffle':
            # PixelShuffle requires mid_channels = out_channels * 4
            self.upsample = nn.PixelShuffle(upscale_factor=2)
            if self.BN_enable:
                self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x

class Resnet_Unet(nn.Module):
    def __init__(self, num_classes=4, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)
            filters = [64,256,512,1024,2048]

        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0], BN_enable=self.BN_enable)

        self.classifier_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)
        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))

        out = self.classifier_conv(d4)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

def create_df(path):
    """
    Traverse the data directory with the following structure:
    data/
    ├── class1/
    │   ├── subdir1/
    │   │   ├── img1.png
    │   │   ...
    │   ├── subdir2/
    │   │   ├── ...
    ...
    Read each image's path and its class, and store them in a DataFrame to return.
    """
    dd = {"images": [], "labels": []}
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if not os.path.isdir(class_dir):
            continue
        # Traverse subdirectories
        for subdir in os.listdir(class_dir):
            sub_path = os.path.join(class_dir, subdir)
            if not os.path.isdir(sub_path):
                continue
            img_count = 0
            for img_file in os.listdir(sub_path):
                if img_file.endswith(".png"):
                    dd["images"].append(os.path.join(sub_path, img_file))
                    dd["labels"].append(class_name)
                    img_count += 1
    return pd.DataFrame(dd)

class Pipeline(Dataset):
    def __init__(self, data, transform):
        super(Pipeline, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        img, label = self.data[x, 0], self.data[x, 1]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img = self.transform(img)
        return img, label

def train_val_one_setting(model, train_dl, val_dl, criterion, optimizer, epochs=5):
    scheduler_local = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    best_acc_local = 0.0
    best_model_local = deepcopy(model)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for data_, target in train_dl:
            data_, target = data_.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data_)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*data_.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        train_acc = train_correct/train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data_, target in val_dl:
                data_, target = data_.to(device), target.to(device)
                out = model(data_)
                v_loss = criterion(out, target)
                val_loss += v_loss.item()*data_.size(0)
                preds = out.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        val_acc = val_correct/val_total

        if val_acc > best_acc_local:
            best_acc_local = val_acc
            best_model_local = deepcopy(model)

        scheduler_local.step()

    return best_model_local, best_acc_local

def grid_search_hyperparams(train_dl, val_dl, num_classes):
    csv_file = 'hyperparam_search_results.csv'
    header = ['LR', 'Weight_Decay', 'Loss', 'Optimizer', 'Val_Acc']

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    best_acc = 0.0
    best_combo = None
    best_model_local = None

    for lr in learning_rates:
        for wd in weight_decays:
            for loss_name, criterion in loss_functions.items():
                for opt_name in optimizers_list:
                    print(f"Grid Search: LR={lr}, WD={wd}, Loss={loss_name}, Optimizer={opt_name}")
                    model = Resnet_Unet(num_classes=num_classes).to(device)
                    if opt_name == 'sgd':
                        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    best_model_search, val_acc = train_val_one_setting(model, train_dl, val_dl, criterion, optimizer, epochs=EPOCHS_SEARCH)

                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([lr, wd, loss_name, opt_name, val_acc])

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_combo = (lr, wd, loss_name, opt_name)
                        best_model_local = deepcopy(best_model_search)

    return best_combo, best_acc, best_model_local

# Redirect logs to a file
log_file = open('train_log.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = log_file

if __name__ == "__main__":
    # Data loading and splitting
    all_df = create_df(data_dir)
    # Map class names to numerical labels
    class_names = all_df["labels"].unique()
    label_index = {cname: i for i, cname in enumerate(class_names)}
    index_label = {v: k for k, v in label_index.items()}
    all_df["labels"] = all_df["labels"].map(label_index)

    # Split dataset (8:1:1)
    train_data, temp_data = train_test_split(all_df.values, test_size=0.2, random_state=42, stratify=all_df["labels"])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data[:,1])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = Pipeline(train_data, transform)
    val_ds = Pipeline(val_data, transform)
    test_ds = Pipeline(test_data, transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    num_classes = len(class_names)

    # Hyperparameter search
    best_combo, best_acc, best_model_search = grid_search_hyperparams(train_dl, val_dl, num_classes)
    print(f"Best Hyperparams from search: LR={best_combo[0]}, WD={best_combo[1]}, Loss={best_combo[2]}, Optimizer={best_combo[3]}, Val Acc={best_acc:.4f}")

    # Final long-term training with the best hyperparameters
    final_lr, final_wd, final_loss_name, final_opt = best_combo
    final_criterion = loss_functions[final_loss_name]

    final_model = Resnet_Unet(num_classes=num_classes).to(device)
    if final_opt == 'sgd':
        final_optimizer = optim.SGD(final_model.parameters(), lr=final_lr, weight_decay=final_wd)
    else:
        final_optimizer = optim.Adam(final_model.parameters(), lr=final_lr, weight_decay=final_wd)

    final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=STEP, gamma=GAMMA)

    best_final_acc = 0.0
    best_final_model = deepcopy(final_model)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    print("===== Final Training with Best Hyperparameters =====")
    for epoch in range(1, EPOCHS_FINAL+1):
        final_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for data_, target in train_dl:
            data_, target = data_.to(device), target.to(device)
            final_optimizer.zero_grad()
            out = final_model(data_)
            loss = final_criterion(out, target)
            loss.backward()
            final_optimizer.step()

            train_loss += loss.item()*data_.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        train_acc = train_correct/train_total
        train_loss = train_loss/train_total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        final_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data_, target in val_dl:
                data_, target = data_.to(device), target.to(device)
                v_loss = final_criterion(out := final_model(data_), target)
                val_loss += v_loss.item()*data_.size(0)
                preds = out.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        val_acc = val_correct/val_total
        val_loss = val_loss/val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > best_final_acc:
            best_final_acc = val_acc
            best_final_model = deepcopy(final_model)
            torch.save(best_final_model.state_dict(), 'best_model.pth')

        final_scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS_FINAL} "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Plot and save Loss and Accuracy curves
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    axes[0].plot(train_loss_list, label="Training")
    axes[0].plot(val_loss_list, label="Validating")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(train_acc_list, label="Training")
    axes[1].plot(val_acc_list, label="Validating")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("loss_acc_curves.png")

    # Evaluate on the test set
    best_final_model.eval()
    truth = []
    preds = []
    with torch.no_grad():
        for data_, target in test_dl:
            data_ = data_.to(device)
            output = best_final_model(data_)
            pred = output.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            truth.extend(target.numpy())

    score = accuracy_score(truth, preds)
    print("Test Accuracy: {:.2f}%".format(score*100))
    cr = classification_report(truth, preds, target_names=[index_label[i] for i in range(num_classes)])
    print("Classification Report:")
    print(cr)

    conf_mat = confusion_matrix(truth, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=[index_label[i] for i in range(num_classes)],
                yticklabels=[index_label[i] for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {score*100:.2f}%)")
    plt.savefig("confusion_matrix.png")

    # Restore standard output
    sys.stdout = original_stdout
    log_file.close()

    # Save classification report to a file
    with open("classification_report.txt", 'w', encoding='utf-8') as f:
        f.write("Test Accuracy: {:.2f}%\n".format(score*100))
        f.write(cr)
