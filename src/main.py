from models.models import *
from models.net import *
from dataloader.dataloader import *
from omegaconf import OmegaConf
import time

config = OmegaConf.load('../config/config.yaml')
run = 'alex'

if __name__ == '__main__':
    # Load the training and validation datasets.
    datasets = CustomDataset(config)
    train_dataloader = datasets.get_dataloader(mode="train")
    val_dataloader = datasets.get_dataloader(mode="valid")

    # for _, data in enumerate(val_dataloader):
    #     print(data['image'])

    # print(train_dataloader)

    # Learning_parameters. 
    lr = config.lr_policy['init_lr']
    epochs = config.num_epochs
    device = (config.device if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # model = build_model(num_classes=config.network['num_classes']).to(device)
    model = alex_model(num_classes=config.network['num_classes']).to(device)

    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=0.9)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_dataloader, 
                                                optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, val_dataloader,  
                                                    criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)
        
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, run)
    print('TRAINING COMPLETE')