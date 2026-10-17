import os
import hydra
import torch
import shutil
import warnings
from torch.optim import Adam
from omegaconf import OmegaConf
from mke_support.utils import check_dir, configure_paths, resolve_input_csv
from mke_support.gnn import get_gnnNets
from mke_support.utils import my_get_dataloader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from mke_support.utils import fix_random_seed
import pandas as pd
from mke_support.data import MoleculeDataset
from mke_support.release import file_sha256, validate_input_csv
import json
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

class TrainModel(object):
    def __init__(self,
                 model,
                 dataset,
                 device,
                 graph_classification=True,
                 save_dir=None,
                 save_name='model',
                 ** kwargs):
        self.model = model
        self.dataset = dataset
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.node_classification = not graph_classification

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        if self.graph_classification:

            index_list_params = kwargs.get('index_list_params')
            self.loader = my_get_dataloader(dataset, index_list_params)


    def __loss__(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        logits = self.model(data=data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        logits = self.model(data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            mask = kwargs.get('mask')
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, accs = [], []
            for batch in self.loader['eval']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.data.to(self.device)
            eval_loss, preds = self._eval_batch(data, data.y, mask=data.val_mask)
            eval_acc = (preds == data.y).float().mean().item()
        return eval_loss, eval_acc

    def test(self):
        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, preds, accs = [], [], []
            for batch in self.loader['test']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.data.to(self.device)
            test_loss, preds = self._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds == data.y).float().mean().item()
        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")
        return test_loss, test_acc, preds

    def train(self, train_params=None, optimizer_params=None):
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']

        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)

        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(self.optimizer,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule = None

        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            if self.graph_classification:
                losses = []
                for batch in self.loader['train']:


                    batch = batch.to(self.device)
                    loss = self._train_batch(batch, batch.y)
                    losses.append(loss)
                train_loss = torch.FloatTensor(losses).mean().item()

            else:
                data = self.dataset.data.to(self.device)
                train_loss = self._train_batch(data, data.y)

            with torch.no_grad():
                eval_loss, eval_acc = self.eval()
            print(f'Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}')
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break
            if lr_schedule:
                lr_schedule.step()

            if best_eval_acc < eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording)

    def save_model(self, is_best=False, recording=None):
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print('saving best...')
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


@hydra.main(config_path="config", config_name="config")
def main(config):
    configure_paths(config, PACKAGE_ROOT)
    config.models.param = config.models.param[config.datasets.dataset_name]
    print(OmegaConf.to_yaml(config))
    fix_random_seed(config.random_seed)

    if torch.cuda.is_available():
        if not 0 <= int(config.device_id) < torch.cuda.device_count():
            raise ValueError("device_id is outside the visible CUDA device range")
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    print("下载数据前")
    group = ["training", "valid", "test"]
    input_csv = resolve_input_csv(config, PACKAGE_ROOT)
    labels_name = str(config.datasets.label_column)
    input_report = validate_input_csv(input_csv, labels_name)
    data_origin = pd.read_csv(input_csv)
    save_dir = os.path.join(config.models.gnn_saving_dir, config.datasets.dataset_name)
    save_name = '{}_{}l'.format(config.models.gnn_name, len(config.models.param.gnn_latent_dim))
    if not config.overwrite_checkpoint:
        for suffix in ('_best.pth', '_latest.pth'):
            if os.path.exists(os.path.join(save_dir, save_name + suffix)):
                raise FileExistsError("Checkpoint already exists. Use a new models.gnn_saving_dir or overwrite_checkpoint=true")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'training_config.yaml'), 'w', encoding='utf-8') as handle:
        handle.write(OmegaConf.to_yaml(config))
    with open(os.path.join(save_dir, 'training_inputs.json'), 'w', encoding='utf-8') as handle:
        json.dump({'dataset': config.datasets.dataset_name, 'label_column': labels_name,
                   'input': input_report, 'random_seed': int(config.random_seed),
                   'training_entry_sha256': file_sha256(__file__)}, handle, indent=2)


    train_index = []
    test_index = []
    valid_index = []
    dataset = MoleculeDataset()
    dataset.build_mol_graph_data(dataset_smiles=data_origin, train_index=train_index, test_index=test_index,
                                 valid_index=valid_index,labels_name=labels_name, group=group)

    print("加载数据完成")


    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:

        index_list = [train_index,test_index,valid_index]
        index_list_params = {'batch_size': config.models.param.batch_size,
                             'index_list': index_list,
                             'seed': config.datasets.seed}


    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    train_params = {'num_epochs': config.models.param.num_epochs,
                    'num_early_stop': config.models.param.num_early_stop,
                    'milestones': config.models.param.milestones,
                    'gamma': config.models.param.gamma}
    optimizer_params = {'lr': config.models.param.learning_rate,
                        'weight_decay': config.models.param.weight_decay}

    if config.models.param.graph_classification:
        trainer = TrainModel(model=model,
                             dataset=dataset,
                             device=device,
                             graph_classification=config.models.param.graph_classification,
                             save_dir=os.path.join(config.models.gnn_saving_dir,
                                                   config.datasets.dataset_name),
                             save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l',
                             index_list_params=index_list_params)
    else:
        trainer = TrainModel(model=model,
                             dataset=dataset,
                             device=device,
                             graph_classification=config.models.param.graph_classification,
                             save_dir=os.path.join(config.models.gnn_saving_dir,
                                                   config.datasets.dataset_name),
                             save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l')
    trainer.train(train_params=train_params, optimizer_params=optimizer_params)
    _, _, _ = trainer.test()

    print("训练结束")
    print("训练结束")


if __name__ == '__main__':
    main()
