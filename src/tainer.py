import torch
from dataloader import get_mnist_loaders
from modules import CNNModule
from torch import nn
from tqdm import tqdm

def evaluate(model, test_loader, criterion):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    module = CNNModule()
    train_loader, test_loader = get_mnist_loaders()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(10), desc='Epochs'):
        # 训练阶段
        module.train()  # 设置为训练模式
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False):
            optimizer.zero_grad()
            outputs = module(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 测试阶段
        test_loss, accuracy = evaluate(module, test_loader, criterion)
        
        # 打印训练和测试结果
        print(f'Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()