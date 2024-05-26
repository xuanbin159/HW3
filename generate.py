import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, dataset, seed_characters, temperature, num_characters, device):
    model.eval()
    
    if isinstance(model, CharLSTM):
        hidden_state, cell_state = model.init_hidden(1)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        hidden = (hidden_state, cell_state)
    elif isinstance(model, CharRNN):
        hidden = model.init_hidden(1).to(device)
    else:
        raise ValueError("Unsupported model type")
    
    input_chars = seed_characters

    # Convert seed characters to tensor
    input_tensor = torch.tensor([[dataset.char_to_idx[char] for char in seed_characters]], dtype=torch.long).to(device)

    # Generate characters
    samples = seed_characters
    with torch.no_grad():
        for _ in range(num_characters):
            output, hidden = model(input_tensor, hidden)
            output_dist = output.squeeze().div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = dataset.idx_to_char[top_char.item()]
            samples += predicted_char

            # Use the predicted character as the next input
            input_tensor = torch.tensor([[dataset.char_to_idx[predicted_char]]], dtype=torch.long).to(device)

    return samples