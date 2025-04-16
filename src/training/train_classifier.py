import torch
import wandb
import time
from tqdm import tqdm
import config.classifier as config
from data.datasets import AudioDataset
from data.preprocess import generate_data
import data.config as data_config
from torch.optim.lr_scheduler import LinearLR


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
else:
    train_sampler = None


# we fit an exponential and poisson distribution to have them as benchmark:






# train, test and eval loops:

def compute_labels(pred, target):
    tp = len([p for p, t in zip(pred, target) if p.item() > 0 and t == 1])
    tn = len([p for p, t in zip(pred, target) if p.item() < 0 and t == 0])
    fp = len([p for p, t in zip(pred, target) if p.item() > 0 and t == 0])
    fn = len([p for p, t in zip(pred, target) if p.item() < 0 and t == 1])
    return tp, tn, fp, fn


def eval(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_acc_poiss = 0
    total_acc_exp = 0
    total_samples = 0
    tp, tn, fp, fn = 0,0,0,0
    for i, (wav, tag, emb_idx) in enumerate(dataloader):
        num_samples = len(wav)
        total_samples += num_samples
        target_class = (tag < 1)*1.0
        pred = model(wav.unsqueeze(1).to(device), emb_idx.unsqueeze(1).to(device))
        loss_i = loss_fn(pred, target_class.to(device)) * num_samples
        tp_i, tn_i, fp_i, fn_i = compute_labels(pred, target_class)
        tp += tp_i
        tn += tn_i
        fp += fp_i
        fn += fn_i

        total_loss += loss_i.item()
        total_acc += sum([pred[i].item() > 0 if target_class[i].item() == 1 else pred[i].item() < 0 for i in range(len(pred))])/num_samples * num_samples
        
        # exponential and poisson distributions
        samples_pois = stats.poisson.rvs(mu=lambda_pois, size=num_samples)
        total_acc_poiss += sum([samples_pois[i] > (emb_idx[i].item()*data_config.input_window) if target_class[i].item() == 1 else samples_pois[i] < (emb_idx[i].item()*data_config.input_window) for i in range(len(pred))])/num_samples * num_samples
        samples_exp = stats.expon.rvs(loc=loc, scale=scale, size=num_samples)
        total_acc_exp += sum([samples_exp[i] > (emb_idx[i].item()*data_config.input_window) if target_class[i].item() == 1 else samples_exp[i] < (emb_idx[i].item()*data_config.input_window) for i in range(len(pred))])/num_samples * num_samples
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f"Precision: {precision}, Recall: {recall}, F1-score: {2*precision*recall/(precision+recall)}")
    return total_loss/total_samples, total_acc/total_samples, total_acc_poiss/total_samples, total_acc_exp/total_samples

def test(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_loss_focal = 0
    total_acc = 0
    total_acc_poiss = 0
    total_acc_exp = 0
    total_samples = 0
    tp, tn, fp, fn = 0,0,0,0
    for i, (wav, tag, emb_idx) in enumerate(dataloader):
        num_samples = len(wav)
        total_samples += num_samples
        target_class = (tag < 1)*1.0
        pred = model(wav.unsqueeze(1).to(device), emb_idx.unsqueeze(1).to(device))
        tp_i, tn_i, fp_i, fn_i = compute_labels(pred, target_class)
        tp += tp_i
        tn += tn_i
        fp += fp_i
        fn += fn_i
        loss_i = loss_fn(pred, target_class.to(device)) * num_samples

        total_loss += loss_i.item()
        total_acc += sum([pred[i].item() > 0 if target_class[i].item() == 1 else pred[i].item() < 0 for i in range(len(pred))])/num_samples * num_samples
        # exponential and poisson distributions
        samples_pois = stats.poisson.rvs(mu=lambda_pois, size=num_samples)
        total_acc_poiss += sum([samples_pois[i] > (emb_idx[i].item()*data_config.input_window) if target_class[i].item() == 1 else samples_pois[i] < (emb_idx[i].item()*data_config.input_window) for i in range(len(pred))])/num_samples * num_samples
        samples_exp = stats.expon.rvs(loc=loc, scale=scale, size=num_samples)
        total_acc_exp += sum([samples_exp[i] > (emb_idx[i].item()*data_config.input_window) if target_class[i].item() == 1 else samples_exp[i] < (emb_idx[i].item()*data_config.input_window) for i in range(len(pred))])/num_samples * num_samples

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f"Precision: {precision}, Recall: {recall}, F1-score: {2*precision*recall/(precision+recall)}")
    return total_loss/total_samples, total_acc/total_samples, total_acc_poiss/total_samples, total_acc_exp/total_samples



def train(model, optimizer, scheduler, config, dataloader, eval_dataloader, test_dataloader,
          loss_fn = nn.BCEWithLogitsLoss(), path2save='last_epoch.pt', start_epoch=0):
    model.train()
    grad_accum_steps = 1


    for epoch in range(start_epoch, start_epoch+config.num_epochs):
        print(epoch, scheduler.get_last_lr(), config.lr)
        start = time.time()
        # for a given sample
        total_loss = 0
        total_acc = 0
        total_norm = 0.0
        total_samples = 0
        tp, tn, fp, fn = 0,0,0,0

        for i, (wav,tag, time_tag) in tqdm(enumerate(dataloader)):
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
        print(f"Total train loss at epoch {epoch} = {train_loss:.5f}({train_loss:.5f}, Accuracy: {train_acc:.5f}, last lr: {scheduler.get_last_lr()[0]:.5f}, time={train_time:.5f}")
        print(f"Overall grad norm: {(total_norm/(len(dataloader)/grad_accum_steps))}")
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {2*precision*recall/(precision+recall)}")
        if (epoch+1) % 1 == 0:
            print(f"Total eval loss at epoch {epoch} = {eval(model, loss_fn, eval_dataloader)}")
            print(f"Total test loss at epoch {epoch} = {test(model, loss_fn, test_dataloader)}")





