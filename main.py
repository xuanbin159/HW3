import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt
from generate import generate
import argparse


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim
    Returns:
        trn_loss: average loss value
    """
    model.train()
    trn_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if isinstance(model, CharLSTM):
          hidden = tuple(h.to(device) for h in model.init_hidden(inputs.size(0)))
        if isinstance(model, CharRNN):
          hidden = model.init_hidden(inputs.size(0)).to(device)

        outputs, _ = model(inputs, hidden)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    trn_loss /= len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function
    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function
    Returns:
        val_loss: average loss value
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(model, CharLSTM):
              hidden = tuple(h.to(device) for h in model.init_hidden(inputs.size(0)))
            if isinstance(model, CharRNN):
              hidden = model.init_hidden(inputs.size(0)).to(device)
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

def main(args):
    """ Main function
    Here, you should instantiate
    1) DataLoaders for training and validation. 
       Try SubsetRandomSampler to create these DataLoaders.
    3) model
    4) optimizer: Adam is a good choice
    5) cost function: use torch.nn.CrossEntropyLoss
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Shakespeare(args.data_path)

    # Instantiate the Shakespeare dataset
    hidden_size = 512
    num_layers = 3
    batch_size = 128
    train_ratio = 0.9
    lr = 0.01


    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size])
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders for training and validation
    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # Instantiate the model (CharRNN or CharLSTM)
    input_size = len(dataset.chars)
    output_size = len(dataset.chars)
    if args.model == 'RNN':
        model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    elif args.model == 'LSTM':
        model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Instantiate the cost function
    criterion = nn.CrossEntropyLoss()
    # 학습 및 검증 손실 값을 저장할 리스트 초기화
    train_losses = []
    val_losses = []
    num_characters = 100
    num_epochs = 30

    # Train and validate the model
    for epoch in range(1, num_epochs + 1):
        trn_loss = train(model, trn_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        print(f"Char{args.model}:Epoch {epoch:02d} | Train Loss: {trn_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 학습 및 검증 손실 값 저장
        train_losses.append(trn_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), f"trained_{args.model}.pth")

    # 캐릭터 생성

    

    # 캐릭터 및 temperature 설정
    characters = ["ROMEO", "JULIET", "FRIAR LAURENCE"]
    temperatures = [0.8, 1.0, 2.0]

    for character in characters:
        for temperature in temperatures:
            seed_characters = f"{character}: "
            generated_samples = []
            
            for i in range(5):
                print(f"Generating sample {i+1} for {character} (temperature={temperature})...")
                sample = generate(model, dataset, seed_characters, temperature, num_characters, device)
                generated_samples.append(sample)
            
            # 생성된 샘플을 파일로 저장
            filename = f"generate/generated_{args.model}_{character}_temp_{temperature:.1f}.txt"
            with open(filename, "w") as f:
                for i, sample in enumerate(generated_samples):
                    f.write(f"Sample {i+1}:\n")
                    f.write(sample)
                    f.write("\n\n")
            
            print(f"Generated samples for {character} (temperature={temperature}) saved to {filename}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['RNN', 'LSTM'], help='type of model (RNN or LSTM)')
    parser.add_argument('--data_path', type=str, default='shakespeare_train.txt', help='path to the dataset file')
    args = parser.parse_args()
    
    main(args)
