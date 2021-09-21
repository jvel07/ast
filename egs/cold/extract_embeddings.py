import torch
from src import dataloader, models

dataset = 'cold'

batch_size = 1
num_workers = 16
n_class = 2

freqm=24
timem=96
mixup=0
epoch=25
fstride=10
tstride=10

train_data = "data/datafiles/cold_train_data.json"
val_data = "data/datafiles/cold_dev_data.json"
eval_data = "data/datafiles/cold_test_data.json"
label_csv = "data/cold_class_label_indices.csv"

norm_stats = {'audioset': [-4.2677393, 4.5689974], 'esc50': [-6.6268077, 5.358466],
              'speechcommands': [-6.845978, 5.5654526], 'dementia': [-5.038124, 3.8022413],
              'cold': [-2.0982502, 3.3960235]}

target_length = {'audioset': 1024, 'esc50': 512, 'speechcommands': 128, 'dementia': 512, 'cold': 256}

# if add noise for data augmentation, only use for speech commands
noise = {'audioset': False, 'esc50': False, 'speechcommands': True, 'dementia': False, 'cold': False}

audio_conf = {'num_mel_bins': 128, 'target_length': target_length[dataset], 'freqm': freqm,
              'timem': timem, 'mixup': mixup, 'dataset': dataset, 'mode': 'train',
              'mean': norm_stats[dataset][0], 'std': norm_stats[dataset][1],
              'noise': noise[dataset]}

val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[dataset], 'freqm': 0, 'timem': 0,
                  'mixup': 0, 'dataset': dataset, 'mode': 'evaluation', 'mean': norm_stats[dataset][0],
                  'std': norm_stats[dataset][1], 'noise': False}

train_loader = torch.utils.data.DataLoader(
    dataset=dataloader.AudiosetDataset(train_data, label_csv=label_csv, audio_conf=audio_conf),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataset=dataloader.AudiosetDataset(val_data, label_csv=label_csv, audio_conf=val_audio_conf),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dataset=dataloader.AudiosetDataset(eval_data, label_csv=label_csv, audio_conf=val_audio_conf),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

audio_model = models.ASTModel(label_dim=n_class, fstride=fstride, tstride=tstride, input_fdim=128,
                              input_tdim=target_length[dataset], imagenet_pretrain=True,
                              audioset_pretrain=True, model_size='base384')


embs = dict()
list_embs = []
list_files = []
list_lbl = []
for idx, sample in enumerate(train_loader):
    # print("sample: ", idx, sample[0].shape)
    output = audio_model(sample[0])
    # print("file: ", filename, "label", sample)
    list_embs.append(output.detach().numpy())
    list_files.append(sample[2])
    list_lbl.append(sample[1])

embs["file_name"] = list_files
embs["label"] = list_lbl
embs["embedding"] = list_embs
    # print(idx, "added to embeddings list")
