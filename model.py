import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import *
import pdb
import numpy as np
import math
import sys


class uidEmbedding(torch.nn.Module):

    def __init__(self, uid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.linear_1 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_2 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_3 = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x)
        uid_emb = self.linear_1(uid_emb)
        uid_emb = F.relu(uid_emb)
        uid_emb = self.linear_2(uid_emb)
        uid_emb = F.relu(uid_emb)
        uid_emb = self.linear_3(uid_emb)
        return F.relu(uid_emb)
    
class iidEmbedding(torch.nn.Module):

    def __init__(self, iid_all, emb_dim):
        super().__init__() 
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)  
        self.linear_1 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_2 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_3 = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        iid_emb = self.iid_embedding(x)
        iid_emb = self.linear_1(iid_emb)
        iid_emb = F.relu(iid_emb)
        iid_emb = self.linear_2(iid_emb)
        iid_emb = F.relu(iid_emb)
        iid_emb = self.linear_3(iid_emb)
        return F.relu(iid_emb)
    

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())  
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)  

def linear_beta_schedule(timesteps, beta_start, beta_end):  
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)  

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):  
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    gamma = 0.3
    betas = gamma * betas
    return betas

def cosine_beta_schedule(timesteps, s=0.008):  
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)  
        output = torch.sum(x * attn_weights, dim=1)  
        return output
        

class Tenc(nn.Module):
    def __init__(self, hidden_size, emb_dim, dropout, diffuser_type, maxlen_source, device, num_heads=1):
        super(Tenc, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.maxlen_source = maxlen_source
        self.device = device
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)  
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.trans_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=6)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

        self.att = torch.nn.Sequential(torch.nn.Linear(self.maxlen_source * self.emb_dim, self.emb_dim * 2), torch.nn.ReLU(),
                                           torch.nn.Linear(self.emb_dim * 2, self.emb_dim))

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )


        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )


    def forward(self, x, h, step):

        t = self.step_mlp(step)  

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res


    def cacu_x(self, x):
        x = self.item_embeddings(x)

        return x

    def cacu_h(self, states, seq_index):
        inputs_emb = states  
        seq = self.emb_dropout(inputs_emb)  
        seq_index = seq_index.unsqueeze(2)  
        mask = torch.ne(seq_index, 0).float()  
        seq *= mask  
        seq_normalized = self.ln_1(seq)  
        mh_attn_out = self.trans_encoder(seq_normalized)
        ff_out = self.ln_2(mh_attn_out)
        ff_out *= mask  #([128, 20, 64])
        ff_out = self.ln_3(ff_out)
        h = torch.mean(ff_out, dim=1)

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - 0.1) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h  
    
    def predict(self, states, seq_index, diff, x_t):
        inputs_emb = states
        seq = self.emb_dropout(inputs_emb)
        seq_index = seq_index.unsqueeze(2)
        mask = torch.ne(seq_index, 0).float()
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.trans_encoder(seq_normalized)
        ff_out = self.ln_2(mh_attn_out)
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        h = torch.mean(ff_out, dim=1)

        x = diff.sample(self.forward, self.forward_uncon, h, x_t)

        return x



class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w, device):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w
        self.device = device
        self.beta_sche = 'exp'

        if self.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)  
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)  
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise  

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  

        predicted_x = denoise_model(x_noisy, h, t)  

        if loss_type == 'l1':
            loss_1 = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss_1 = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss_1 = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return predicted_x, loss_1

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)  
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise  
        
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h, x_t):
        x = x_t

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=self.device, dtype=torch.long), n)

        return x



class DiffBaseModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, timesteps, beta_start, beta_end, hidden_size, dropout_rate, diffuser_type, w, maxlen_source, device):
        super().__init__()
        self.guided = Tenc(hidden_size, emb_dim, dropout_rate, diffuser_type, maxlen_source, device)
        self.dcdr = diffusion(timesteps, beta_start, beta_end, w, device)
        self.cacu_uidemb = uidEmbedding(uid_all, emb_dim)
        self.cacu_iidemb = iidEmbedding(iid_all, emb_dim)
        
    def forward(self, stage, x_source, x_target, timesteps, batch_size, device):
        if stage =='train_dcdr':
            x_start = self.cacu_uidemb(x_source[:, 0])  

            item_source = x_source[:, 1:]  
            item_source = self.cacu_iidemb(item_source)  
            item_target = self.cacu_iidemb(x_target)  
            h = self.guided.cacu_h(item_source, x_source[:, 1:])  

            n = torch.randint(0, timesteps, (batch_size, ), device=device).long()  
            predicted_x, loss_1 = self.dcdr.p_losses(self.guided, x_start, h, n, loss_type='l2')  
            predicted_x = predicted_x.unsqueeze(1)  
            predicted_scores_list = []
            for i in range(item_target.size(1)):
                item_target_meta = item_target[:, i, :]  
                item_target_meta = item_target_meta.unsqueeze(1)  
                emb = torch.cat([predicted_x, item_target_meta], 1)  
                predicted_scores = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  
                predicted_scores_list.append(predicted_scores)

            predicted = torch.stack(predicted_scores_list, dim = 1)  

            return predicted, loss_1
            
        else:
            x_t = self.cacu_uidemb(x_source[:, 0])  
            item_source = x_source[:, 1:]  
            item_source = self.cacu_iidemb(item_source)  
            item_target = self.cacu_iidemb(x_target)  
            predicted_x = self.guided.predict(item_source, x_source[:, 1:], self.dcdr, x_t)  
            predicted_x = predicted_x.unsqueeze(1)  
            predicted_scores_list = []
            for i in range(item_target.size(1)):
                item_target_meta = item_target[:, i, :]  
                item_target_meta = item_target_meta.unsqueeze(1)  
                emb = torch.cat([predicted_x, item_target_meta], 1)  
                predicted_scores = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  
                predicted_scores_list.append(predicted_scores)

            predicted = torch.stack(predicted_scores_list, dim = 1)  
                
            return predicted
    
