import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderCNN(nn.Module):
    def __init__(self, enc_dim=128):
        super().__init__()
        # 3 Convolutional layers jesa require tha
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, enc_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(enc_dim)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # HIGHEST SPATIAL PRECISION - Adaptive Average Pooling (8x8 regions = 64 tokens)
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.adapt_pool(x)
        # Sequence of flat features nikal tay hain Attention ke liye
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1) 
        return x

class Attention(nn.Module):
    def __init__(self, enc_dim, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(enc_dim, attn_dim)
        self.U = nn.Linear(hidden_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)
        
    def forward(self, encoder_outputs, hidden):
        # Attention score logic
        hidden_expanded = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.W(encoder_outputs) + self.U(hidden_expanded))
        attention_scores = self.V(energy).squeeze(2)
        alpha = F.softmax(attention_scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, hidden_dim, attn_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(enc_dim, hidden_dim, attn_dim)
        self.gru = nn.GRU(embed_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout) # Regularization
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        context = self.attention(encoder_outputs, hidden.squeeze(0))
        
        gru_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden)
        
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class ImageToProgramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, enc_dim=128, hidden_dim=256, attn_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN(enc_dim)
        self.decoder = DecoderGRU(vocab_size, embed_dim, enc_dim, hidden_dim, attn_dim, dropout)
        self.init_hidden = nn.Linear(enc_dim, hidden_dim)
        
    def forward(self, images, trg_seqs, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        max_len = trg_seqs.size(1)
        vocab_size = self.decoder.fc.out_features
        
        encoder_outputs = self.encoder(images)
        # Decoder GRU ka initial state Encoder set se mean nikal kar create kiya
        hidden = torch.tanh(self.init_hidden(encoder_outputs.mean(dim=1))).unsqueeze(0)
        
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(images.device)
        decoder_input = trg_seqs[:, 0] # Pehla sequence token <SOS> hai
        
        for t in range(1, max_len):
            prediction, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = prediction
            
            top1 = prediction.argmax(1)
            # Teacher forcing strategy yahan implement hui hai
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = trg_seqs[:, t] if use_teacher_forcing else top1
            
        return outputs
