import torch
import wandb
import time
from tqdm import tqdm
import config.classifier as config
from models.config.classifier import ModelConfig, CNNConfig
from models.classifier import BasicCNN, BasicModel
from data.datasets import AudioDataset
from data.preprocess import generate_data
import data.config as data_config
from torch.optim.lr_scheduler import LinearLR
import numpy as np
import scipy.stats as stats
import torch.nn as nn
from models.utils import initialize_weights
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else: 
    device = 'cpu'

torch.set_default_dtype(config.dtype)

# generate data with data.config parameters:
train_data, eval_data, test_data, (mean_train, std_train) = generate_data()

# define datasets
train_dataset = AudioDataset(train_data, normalize=True, mean=mean_train, std=std_train, 
                             noise=data_config.noise, tmask=data_config.tmask, fmask=data_config.fmask, 
                             dtype=torch.float32)
eval_dataset = AudioDataset(eval_data, normalize=True, mean=mean_train, std=std_train, 
                             noise=False, tmask=None, fmask=None, dtype=torch.float32)
test_dataset = AudioDataset(test_data, normalize=True, mean=mean_train, std=std_train, 
                             noise=False, tmask=None, fmask=None, dtype=torch.float32)

# define dataloaders
def get_sample_weights(dataset, data):
    # gets weights per label (batches will be sampled according to these weights)
    ones = len([d for d in data if d['tag'] <= 1])
    zeros = len(data) - ones
    class_weights={1: 1/ones,  0: 1/zeros}
    print("Generating weights for the training sampler...")
    sample_weights = [class_weights[1] if d[2] <= 1 else class_weights[0] for d in tqdm(dataset)]
    return sample_weights

eval_sampler = test_sampler = None
if config.weighted_sampling:
    weights = get_sample_weights(train_dataset, train_data)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size,  shuffle=False,sampler=train_sampler, num_workers=0, pin_memory=False)

else:
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size,  shuffle=True, num_workers=0, pin_memory=False)

eval_dataloader = DataLoader(eval_dataset, batch_size=config.test_batch_size,  shuffle=False, num_workers=0, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size,  shuffle=False, sampler=test_sampler, num_workers=0, pin_memory=False)

# we fit an exponential and poisson distribution to have them as benchmark:

distances = [x['label']*data_config.input_window + x['empty_samples'] for x in train_data]

# Exponential
loc, scale = stats.expon.fit(distances)  # loc is the shift, scale = 1/λ
lambda_hat = 1 / scale

# Poisson
lambda_pois = np.mean(distances)


# train, test and eval loops:

def compute_labels(pred, target):
    tp = len([p for p, t in zip(pred, target) if p.item() > 0 and t == 1])
    tn = len([p for p, t in zip(pred, target) if p.item() < 0 and t == 0])
    fp = len([p for p, t in zip(pred, target) if p.item() > 0 and t == 0])
    fn = len([p for p, t in zip(pred, target) if p.item() < 0 and t == 1])
    return tp, tn, fp, fn

def print_precision_recall(name, tp, fp, tn, fn):
        if tp == 0:
              precision = 0
              recall = 0
              f1_score = 0
        else:
          precision = tp/(tp+fp)
          recall = tp/(tp+fn)
          f1_score = 2*precision*recall/(precision+recall)
        try:
            accuracy = (tp+tn)/(tp+tn+fp+fn)
        except:
            print(tp,tn, fp,fn)
        print(f"Results for {name}: Prec: {precision}, Rec: {recall}, F1-s: {f1_score}, Acc: {accuracy} ")


def eval(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_acc_poiss = 0
    total_acc_exp = 0
    total_samples = 0
    tp, tn, fp, fn = 0,0,0,0
    tp_poiss, tn_poiss, fp_poiss, fn_poiss = 0,0,0,0
    tp_exp, tn_exp, fp_exp, fn_exp = 0,0,0,0
    for i, (wav, time_tag, tag, empty_samples, _) in tqdm(enumerate(dataloader)):
        num_samples = len(wav)
        total_samples += num_samples
        target_class = (tag < 1)*1.0
        pred = model(wav.unsqueeze(1).to(device), time_tag.unsqueeze(1).to(device))
        loss_i = loss_fn(pred, target_class.to(device)) * num_samples
        tp_i, tn_i, fp_i, fn_i = compute_labels(pred, target_class)
        tp += tp_i
        tn += tn_i
        fp += fp_i
        fn += fn_i

        total_loss += loss_i.item()
        total_acc += sum([pred[i].item() > 0 if target_class[i].item() == 1 else pred[i].item() < 0 for i in range(len(pred))])/num_samples * num_samples
        
        # exponential and poisson distributions
        samples_pois = torch.tensor(stats.poisson.rvs(mu=64000/lambda_pois, size=num_samples))
        predictions_pois=torch.where(samples_pois.clone().detach() > 0, torch.tensor(1), torch.tensor(-1))
        tp_poiss_i, tn_poiss_i, fp_poiss_i, fn_poiss_i = compute_labels(predictions_pois, target_class)
        tp_poiss += tp_poiss_i
        tn_poiss += tn_poiss_i
        fp_poiss += fp_poiss_i
        fn_poiss += fn_poiss_i

        samples_exp = torch.tensor(stats.expon.rvs(loc=loc, scale=lambda_pois, size=num_samples))
        predictions_exp = torch.where(samples_exp.clone().detach() > empty_samples, 
                              torch.tensor(1), 
                              torch.tensor(-1))
        tp_exp_i, tn_exp_i, fp_exp_i, fn_exp_i = compute_labels(predictions_exp, target_class)
        tp_exp += tp_exp_i
        tn_exp += tn_exp_i
        fp_exp += fp_exp_i
        fn_exp += fn_exp_i

    
    
    print_precision_recall('model', tp, fp, tn, fn)
    print_precision_recall('exp', tp_exp, fp_exp, tn_exp, fn_exp)
    print_precision_recall('pois', tp_poiss, fp_poiss, tn_poiss, fn_poiss)
    return total_loss/total_samples, total_acc/total_samples, total_acc_poiss/total_samples, total_acc_exp/total_samples


def train(model, optimizer, scheduler, config, dataloader, eval_dataloader, test_dataloader,
            loss_fn = nn.BCEWithLogitsLoss(), start_epoch=0):
    model.train()

    for epoch in range(start_epoch, config.num_epochs):
        start = time.time()
        # for a given sample
        total_loss = 0
        total_acc = 0
        total_norm = 0.0
        total_samples = 0
        tp, tn, fp, fn = 0,0,0,0
        tp_poiss, tn_poiss, fp_poiss, fn_poiss = 0,0,0,0
        tp_exp, tn_exp, fp_exp, fn_exp = 0,0,0,0
        for i, (wav, time_tag, tag, empty_samples, _) in tqdm(enumerate(dataloader)):
            num_samples = len(wav)
            total_samples += num_samples
            target_class = (tag < 1)*1.0

            pred = model(wav.unsqueeze(1).to(device), time_tag.unsqueeze(1).to(device))
            tp_i, tn_i, fp_i, fn_i = compute_labels(pred, target_class)
            tp += tp_i
            tn += tn_i
            fp += fp_i
            fn += fn_i
            loss = loss_fn(pred, target_class.to(device))
            loss.backward()

            if config.log_wandb:
                # Logging Weights & Gradients
                log_data = {"loss": loss.item(), "epoch": epoch, "step": epoch * len(dataloader) + i}

                for name, param in model.named_parameters():
                  log_data[f"weights_norm/{name}"] = param.data.norm().item()
                  if param.grad is not None:
                    log_data[f"gradients_norm/{name}"] = param.grad.norm().item()
                  else:
                    print(f"Skipping gradient logging for {name}, no gradients available.")


                # Log everything at once
                wandb.log(log_data)


            total_loss += loss.item() * num_samples
            total_acc += sum([pred[i].item() > 0 if target_class[i].item() == 1 else pred[i].item() < 0 for i in range(len(pred))])/num_samples * num_samples
            
            # Poisson
            samples_pois = torch.tensor(stats.poisson.rvs(mu=64000/lambda_pois, size=num_samples))
            predictions_pois=torch.where(samples_pois.clone().detach() > 0, torch.tensor(1), torch.tensor(-1))
            tp_poiss_i, tn_poiss_i, fp_poiss_i, fn_poiss_i = compute_labels(predictions_pois, target_class)
            tp_poiss += tp_poiss_i
            tn_poiss += tn_poiss_i
            fp_poiss += fp_poiss_i
            fn_poiss += fn_poiss_i

            # Exponential
            samples_exp = torch.tensor(stats.expon.rvs(loc=loc, scale=lambda_pois, size=num_samples))
            predictions_exp = torch.where(samples_exp.clone().detach() > empty_samples, 
                                torch.tensor(1), 
                                torch.tensor(-1))
            tp_exp_i, tn_exp_i, fp_exp_i, fn_exp_i = compute_labels(predictions_exp, target_class)
            tp_exp += tp_exp_i
            tn_exp += tn_exp_i
            fp_exp += fp_exp_i
            fn_exp += fn_exp_i


            if (i+1) % config.grad_accum_steps == 0:
                norm = 0
                for param in model.parameters():
                  if param.grad is not None:
                    param_norm = param.grad.norm(2)  # L2 norm of the gradient
                    norm += param_norm.item() ** 2

                total_norm += norm**(1/2)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()


        train_loss = total_loss/total_samples
        train_acc = total_acc/total_samples
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        filename = f"2stream_checkpoint_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            "precision": precision,
            "recall": recall,
            "f1-score": 2*precision*recall/(precision+recall)

            }, filename)
        if config.log_wandb:
            wandb.save(filename)

        train_time = time.time() - start
           
        print_precision_recall('model', tp, fp, tn, fn)
        print(f"Total train loss at epoch {epoch} = {train_loss:.5f}({train_loss:.5f}. Accuracy: {train_acc:.5f}, last lr: {scheduler.get_last_lr()[0]:.5f}, time={train_time:.5f}")
        
        print_precision_recall('exp', tp_exp, fp_exp, tn_exp, fn_exp)
        print_precision_recall('pois', tp_poiss, fp_poiss, tn_poiss, fn_poiss)

        loss_eval, acc_eval = eval(model, loss_fn, eval_dataloader)
        print(f"Total eval loss at epoch {epoch} = {loss_eval}. Accuracy: {acc_eval}")


# model and weights initialization
cnn_config = CNNConfig()

model_config = ModelConfig()

assert(model_config.inner_dim == cnn_config.channels[-1])

model_config.head_dim = model_config.inner_dim // model_config.num_heads * 2
model_config.seq_len = cnn_config.last_dim

base_cnn = BasicCNN(last_dim=cnn_config.last_dim,
               channels= list(cnn_config.channels),
               dims=list(cnn_config.dims),
               dropout=cnn_config.dropout,
               input_H=cnn_config.input_H,
               input_W = cnn_config.input_W,
               dims_model = cnn_config.dims_model
               )

model = BasicModel(base_model=base_cnn, config=model_config)

initialize_weights(model)

if config.log_wandb:
  # Initialize a W&B run
  config = {"model_config": model_config.__dict__,
            "cnn_config": cnn_config.__dict__,
            "train_config": config.__dict__}
  
  run_name = config.run_name
  wandb.init(project=config.project_name, id=run_name, config=config, resume='allow')

total_training_steps = config.num_epochs * len(train_dataloader)
warmup_steps = int(0.1 * total_training_steps)  # 10% warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
epochs_warmup = 5
start_factor = 1/5
scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=epochs_warmup*len(train_dataloader)//config.grad_accum_steps)

start_epoch=0
if config.load_checkpoint:
  checkpoint = torch.load(config.checkpoint_file)
  scheduler.load_state_dict(checkpoint['scheduler'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  model.load_state_dict(checkpoint['model_state_dict'])
  start_epoch = checkpoint['epoch'] + 1



if config.weighted_loss:
  num_negs = len([d for d in train_data if d['label'] > 1])
  num_pos = len([d for d in train_data if d['label'] < 1])
  factor = num_negs/num_pos
  pos_weight = torch.tensor([config.alpha_weight_loss * factor]).to(device)
  print(f"We assign pos weights to: {pos_weight}")
  bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
  bce_loss = nn.BCEWithLogitsLoss()


print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Number of samples: {len(train_dataset)}, {len(eval_dataset)}, {len(test_dataset)}")
train(model, optimizer=optimizer, scheduler=scheduler, config=config, dataloader=train_dataloader,
      eval_dataloader=eval_dataloader, test_dataloader=test_dataloader, loss_fn=bce_loss, start_epoch=start_epoch)








