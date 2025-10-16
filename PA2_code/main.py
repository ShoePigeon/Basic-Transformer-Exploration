import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from torch import nn
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *
from utilities import *
import argparse


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
head_size = n_embd/n_head

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            # print(X.shape) #16, 32
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMLMmodel.
    """
    decoderLMLMmodel.eval()
    total_loss = 0
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)

        loss, _ = decoderLMLMmodel(X, Y) # your LMmodel should be computing the cross entropy loss        
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMLMmodel.train()
    return perplexity

def main():
    print("running")
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)    


    if args.model == "encoder":


        #whole dataset
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        #batched dataset
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)


        #test data
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        vocab_size = tokenizer.vocab_size

        #for graph
        test_accuracies = []

        # Initialize the model, loss function, and optimizer
        model = TransformerClassifier(vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, n_output, block_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        #number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in the model: {total_params}")

        model.train()
        for epoch in range(epochs_CLS):
            total_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_CLS_loader):
                # Move data to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                # print(len(inputs)) #length of inputs is 16 (batch size)
                # print(inputs.size()) #inputs is 16 X 32 (batch size X block size/sentence length)
                outputs, attn_maps_all = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                


            # Average loss for the epoch
            avg_loss = total_loss / len(train_CLS_loader)
            print(f"Epoch [{epoch+1}/{epochs_CLS}], Average Loss: {avg_loss:.4f}")
            cur_accuracy = compute_classifier_accuracy(model, train_CLS_loader)
            print(f"{cur_accuracy} is the current train accuracy")
            cur_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
            print(f"{cur_accuracy} is the current test accuracy")
            test_accuracies.append(cur_accuracy)



        cur_accuracy = compute_classifier_accuracy(model, train_CLS_loader)
        print(f"{cur_accuracy} is the current train accuracy")
        cur_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
        print(f"{cur_accuracy} is the current test accuracy")
        sanity = Utilities(tokenizer, model)
        sanity.sanity_check("But to go beyond such levels, where cutting defense would threaten our vital margin of safety, is something I will never accept.", 32)
        sanity.sanity_check("With , again , great thanks to the Members of the United States Senate , leaders of whom are here today , and those who worked so tirelessly for", 32)
        print("Training complete!")
        plt.figure(figsize=(8, 6))
        plt.plot(test_accuracies, label='Accuracies')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy for Transformer Encoder Classification')
        plt.legend()
        plt.grid()

        # Save the training and dev accuracy figure
        test_accuracy_file = 'test_accuracy_.png'
        plt.savefig(test_accuracy_file)
        print(f"\n\nTest accuracy plot saved as {test_accuracy_file}")

        
    if args.model == "decoder":
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtesthbushText = f.read()
        inputfile = "speechesdataset/test_LM_wbush.txt"        
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestwbushText = f.read()
        inputfile = "speechesdataset/test_LM_obama.txt"        
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestobamaText = f.read()
        # Bunch the test data if needed
        lmtestAll = lmtesthbushText + lmtestobamaText + lmtestwbushText
 
        testhbush_LM_dataset = LanguageModelingDataset(tokenizer, lmtesthbushText,  block_size)
        testhbush_LM_loader = DataLoader(testhbush_LM_dataset, batch_size=batch_size, shuffle=True)    

        testwbush_LM_dataset = LanguageModelingDataset(tokenizer, lmtestwbushText,  block_size)
        testwbush_LM_loader = DataLoader(testwbush_LM_dataset, batch_size=batch_size, shuffle=True)    

        testobama_LM_dataset = LanguageModelingDataset(tokenizer, lmtestobamaText,  block_size)
        testobama_LM_loader = DataLoader(testobama_LM_dataset, batch_size=batch_size, shuffle=True)    

        vocab_size = tokenizer.vocab_size

        decode_output = vocab_size
        # criterion = nn.CrossEntropyLoss()

        # Initialize the LMmodel, loss function, and optimizer
        LMmodel = TransformerDecoder(vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, decode_output, block_size)
        optimizer = torch.optim.Adam(LMmodel.parameters(), lr=learning_rate)    

        # Move LMmodel to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LMmodel.to(device)

        #number of parameters
        # total_params = sum(p.numel() for p in LMmodel.parameters())
        # print(f"Total parameters in the model: {total_params}")

        LMmodel.train()
        hbush_perplexity = []
        wbush_perplexity = []
        obama_perplexity = []
        train_perplexity = []
        for batch_idx, (inputs, predictions) in enumerate(train_LM_loader):

            # Move data to the appropriate device
            inputs, predictions = inputs.to(device), predictions.to(device)
            

            loss, _= LMmodel(inputs, predictions)

            if batch_idx % 100 == 0:
                perplexity = compute_perplexity(LMmodel, testhbush_LM_loader)
                print(f"{perplexity} is the perplexity after batch {batch_idx} for hbush")
                hbush_perplexity.append(perplexity)
                perplexity = compute_perplexity(LMmodel, testobama_LM_loader)
                obama_perplexity.append(perplexity)
                print(f"{perplexity} is the perplexity after batch {batch_idx} for obama")
                perplexity = compute_perplexity(LMmodel, testwbush_LM_loader)
                wbush_perplexity.append(perplexity)
                print(f"{perplexity} is the perplexity after batch {batch_idx} for wbush")
                perplexity = compute_perplexity(LMmodel, train_LM_loader)
                print(f"{perplexity} is the perplexity for train after batch {batch_idx}")
                train_perplexity.append(perplexity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx > max_iters:
                print("breaking", batch_idx, max_iters)
                break                    
        
        #Sanity Check
        sanity = Utilities(tokenizer, LMmodel)
        sentence = "But to go beyond such levels, where cutting defense would threaten our vital margin of safety, is something I will never accept."
        predicts = torch.tensor(tokenizer.encode(sentence[1:]))
        predicts = predicts.unsqueeze(0)
        sanity.decode_sanity_check(sentence[:-1], 25, predicts)

        print("Training complete!")
        plt.figure(figsize=(8, 6))
        plt.plot(hbush_perplexity, label='HBush')
        plt.plot(wbush_perplexity, label='WBush')
        plt.plot(obama_perplexity, label='Obama')
        plt.plot(train_perplexity, label='Train')
        plt.xlabel('Iterations (100)')
        plt.ylabel('Perplexity')
        plt.title('Perplexity for Transformer Decoder')
        plt.legend()
        plt.grid()

        # Save the training and dev accuracy figure
        decoder_perplexity_file = 'decoder_perplexity_file.png'
        plt.savefig(decoder_perplexity_file)
        print(f"\n\nDecoder Perplexity Graph saved as {decoder_perplexity_file}")
        plt.figure(figsize=(8, 6))
        x_values = range(1, len(hbush_perplexity))


        # Save the shortened form
        plt.plot(x_values, hbush_perplexity[1:], label='HBush')
        plt.plot(x_values, wbush_perplexity[1:], label='WBush')
        plt.plot(x_values, obama_perplexity[1:], label='Obama')
        plt.plot(x_values, train_perplexity[1:], label='Train')
        plt.xlabel('Iterations (100)')
        plt.ylabel('Perplexity')
        plt.title('Perplexity for Transformer Decoder')
        plt.legend()
        plt.grid()
        plt.xticks(x_values)

        decoder_perplexity_file = 'decoder_perplexity_file_short.png'
        plt.savefig(decoder_perplexity_file)
        print(f"\n\nDecoder Perplexity Graph Close saved as {decoder_perplexity_file}")

    



if __name__ == "__main__":
    main()
