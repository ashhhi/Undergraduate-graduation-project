import numpy as np
from tqdm import tqdm

from MapUnet.DataLoader.DeepGlobalRoad import LoadData
from Models.SegNet1 import SegNet


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
    print(torch.__version__)
    framObjTrain = {'img': [], 'mask': []}

    # Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    Net = SegNet().to(device)

    print("Load Data...")
    framObjTrain = LoadData(framObjTrain, imgPath=r'C:\DataSet\DeepGlobal\train', maskPath=r'C:\DataSet\DeepGlobal\train', shape=128)
    print("Load Data Successfully")

    # Convert inputs and targets to PyTorch tensors
    # Convert inputs to torch tensor and normalize
    inputs = torch.from_numpy(np.array(framObjTrain['img'])).permute(0, 3, 1, 2).float().to(device)
    # Convert targets to torch tensor and normalize
    targets = torch.from_numpy(np.array(framObjTrain['mask'])).unsqueeze(1).float().to(device)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define optimizer and loss function
    optimizer = Adam(Net.parameters())
    criterion = nn.BCELoss()

    # Train the model
    Net.train()
    history = {'loss':[], 'accuracy':[]}
    for epoch in range(4):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in tqdm(dataloader, desc='Batches', leave=False):
            optimizer.zero_grad()
            outputs = Net(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算准确率
            predicted_labels = (outputs > 0.5).float()  # 根据输出阈值生成预测标签
            correct = (predicted_labels == batch_targets).sum().item()
            total_correct += correct
            total_samples += batch_targets.numel()

        accuracy = total_correct / total_samples
        loss = total_loss / len(dataloader)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy: {accuracy}")

    # Save the model
    torch.save(Net.state_dict(), 'Outputs/SavedModel/SegNet.pt')
    print("Model saved successfully.")