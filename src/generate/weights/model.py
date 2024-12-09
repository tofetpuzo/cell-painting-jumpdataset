# tissue_vae/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=512, dropout_rate=0.00):
        super(VAE, self).__init__()
        self.activation = nn.ELU()
            
        # Encoder
        self.enc_conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(512)
        self.enc_dropout = nn.Dropout(dropout_rate)
        
        self.hidden_dim = 512 * 8 * 8
        self.intermediate_dim = min(4096, max(512, z_dim * 2))
        
        self.enc_fc1 = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.enc_fc2_mean = nn.Linear(self.intermediate_dim, z_dim)
        self.enc_fc2_logvar = nn.Linear(self.intermediate_dim, z_dim)

        # Decoder architecture (same as original)
        self.dec_fc1 = nn.Linear(z_dim, self.intermediate_dim)
        self.dec_fc2 = nn.Linear(self.intermediate_dim, self.hidden_dim)
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_conv4 = nn.ConvTranspose2d(64, 5, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        x = self.activation(self.enc_bn1(self.enc_conv1(x)))
        x = self.activation(self.enc_bn2(self.enc_conv2(x)))
        x = self.activation(self.enc_bn3(self.enc_conv3(x)))
        x = self.activation(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(-1, self.hidden_dim)
        x = self.activation(self.enc_fc1(x))
        x = self.enc_dropout(x)
        x = self.layer_norm(x)
        return self.enc_fc2_mean(x), self.enc_fc2_logvar(x)

    def sampling(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decoder(self, z):
        z = self.activation(self.dec_fc1(z))
        z = self.activation(self.dec_fc2(z))
        z = z.view(-1, 512, 8, 8)
        z = self.activation(self.dec_bn1(self.dec_conv1(z)))
        z = self.activation(self.dec_bn2(self.dec_conv2(z)))
        z = self.activation(self.dec_bn3(self.dec_conv3(z)))
        z = torch.sigmoid(self.dec_conv4(z))
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.sampling(mean, log_var)
        return self.decoder(z), mean, log_var

