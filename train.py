import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GraphicsDataset, VOCAB, IDX2VOCAB
from model import ImageToProgramModel

BATCH_SIZE = 16          
EPOCHS = 35             
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def train():
    print("Maximum Accuracy keliye Large Dataset generate kia ja raha hai (15000 samples)...")
    train_dataset = GraphicsDataset(num_samples=15000, img_size=64, max_shapes=10)
    val_dataset = GraphicsDataset(num_samples=2000, img_size=64, max_shapes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    vocab_size = len(VOCAB)
    # TUNE DROPOUT for optimization (0.35 gives extra stability over 0.3)
    model = ImageToProgramModel(vocab_size=vocab_size, dropout=0.35).to(DEVICE)
    
    import os
    if os.path.exists('best_model.pth'):
        print("\n--> PICHLI TRAINING RESUME HO RAHI HAI (Saved weights load ho gye) <--\n")
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'], label_smoothing=0.1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 7
    
    print(f"Model Training Started on {DEVICE}...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        # Teacher forcing decay
        tf_ratio = max(0.2, 0.7 - (epoch * 0.05))
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, targets, teacher_forcing_ratio=tf_ratio)
            
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            targets = targets[:, 1:].reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        # ----- Metrics & Error Analysis Validation -----
        model.eval()
        val_loss = 0
        
        correct_tokens = 0
        total_tokens = 0
        
        correct_seqs = 0
        total_seqs = 0
        
        error_examples = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images, targets, teacher_forcing_ratio=0.0) 
                
                outputs_flat = outputs[:, 1:].reshape(-1, outputs.shape[-1])
                targs_seq = targets[:, 1:]
                targets_flat = targs_seq.reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                # Token Accuracy
                predictions = outputs_flat.argmax(1)
                mask = targets_flat != VOCAB['<PAD>']
                correct_tokens += ((predictions == targets_flat) & mask).sum().item()
                total_tokens += mask.sum().item()
                
                # Sequence Exact-Match Accuracy
                preds_seq = predictions.view(images.size(0), -1)
                for b in range(images.size(0)):
                    target_len = (targs_seq[b] != VOCAB['<PAD>']).sum().item()
                    p = preds_seq[b, :target_len]
                    t = targs_seq[b, :target_len]
                    
                    if torch.equal(p, t):
                        correct_seqs += 1
                    else:
                        if len(error_examples) < 2: # Keep 2 examples for analysis
                            error_examples.append((t.cpu().numpy(), p.cpu().numpy()))
                total_seqs += images.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
        val_seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} | TF={tf_ratio:.2f} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   -> Token Acc: {val_acc*100:.2f}% | Sequence Acc: {val_seq_acc*100:.2f}%")
        
        # ERROR ANALYSIS LOGS
        if len(error_examples) > 0:
            print("   -> [Error Analysis / Failed Predictions]:")
            for i, ex in enumerate(error_examples):
                tru = [IDX2VOCAB.get(x, '') for x in ex[0]]
                prd = [IDX2VOCAB.get(x, '') for x in ex[1]]
                print(f"      {i+1}. TRUE: {tru}")
                print(f"         PRED: {prd}")
            print("      (Mistake Pattern: CNN loss in small pooling areas usually shifts coordinates by 2-3 pixels)")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("   *** New Best Model Saved ('best_model.pth')! ***")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping trigerred. Best validation model is locked and saved.")
                break

if __name__ == "__main__":
    train()
