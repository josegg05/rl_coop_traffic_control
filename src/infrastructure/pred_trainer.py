
from collections import OrderedDict
from src.dataloader.dataloader import loader
from src.infrastructure.logger import Logger
import os
import time
import torch
from torch.optim import Adam
import numpy as np
import os


import src.infrastructure.pytorch_utils as ptu
from src.infrastructure.metrics import masked_mae, masked_rmse, masked_mape, metric
from src.encoder.gnn import GNN

_str_to_idx = {
    'bay': (2, 5, 11),
    'la': (2, 5, 11),
    'veg': (0, 1, 3)
}

class Prediction_Trainer(object):
    def __init__(self, params) -> None:
        
        self.params = params
        self.logger = Logger(self.params['logdir']) 
        self.device = self.params['device']  

        self.loaders = loader(
            data_path = self.params['node_data_path'],
            graph_path = self.params['graph_data_path'], 
            batch_size = self.params['batch_size'],
            num_workers = self.params['num_workers'],
            normalize = not self.params['no_normalize'],
            norm_type = self.params['norm_type']
        )
        self.scaler = self.loaders['scaler']
        self.scaler.to(self.device)
         
        self.model = self.params['model'].to(self.device)
        self.model.to(self.device)

        self.criterion = masked_mae
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=self.params['learning_rate'], 
            weight_decay=self.params['weight_decay']
        )
        self.itr = 0 
        
    def run_training_loop(self, epochs): 
        
        self.start_time = time.time()
        self.min_loss = 1000000
        self.min_epoch = 0
        for epoch in range(epochs):
            self.epoch = epoch 
            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif epoch % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            train_logs = self.train_model()

            # log/save
            if self.logmetrics: 
                print('\nBeginning logging procedure...')
                self.perform_logging(train_logs)
        
                if self.params['save_params']:
                    # TODO
                    pass

    def train_model(self):
        train_log = {}
        for phase in ['train', 'val']: 
            running_loss = 0
            loss_count = 0 
            training = phase == 'train'
            mae_list = []
            rmse_list = []
            mape_list = []
            self.loaders['scaler'].to(self.device)
            for x, y, g in self.loaders[phase]:
                self.model.train(training)
                x = x.to(self.device)
                y = y.to(self.device)
                g = g.to(self.device)
                # forward
                # track grad only if train
                
                with torch.set_grad_enabled(training):
                    y_pred = self.model(g, x)
                    if self.loaders['scaler']:
                        y_pred = self.loaders['scaler'].inverse_transform(y_pred)
                    loss = self.criterion(y_pred, y, 0.0)
                    if phase == 'train': 
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.clip = 5
                        if self.clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                        self.itr += 1
                
                running_loss += loss * x.size(0)
                mae_list.append(ptu.to_numpy(masked_mae(y_pred, y,  0.0)) * x.size(0))
                rmse_list.append(ptu.to_numpy(masked_rmse(y_pred, y, 0.0)) * x.size(0)) 
                mape_list.append(ptu.to_numpy(masked_mape(y_pred, y, 0.0)) * x.size(0)) 
                loss_count += x.size(0)

            if phase == 'train':
                metrics = self.calculate_metrics(
                    'Training', running_loss, loss_count, mae_list, rmse_list, mape_list
                )
                train_log.update(metrics)
            else: 
                metrics = self.calculate_metrics(
                    'Evaluation', running_loss, loss_count, mae_list, rmse_list, mape_list
                    )
                train_log.update(metrics)
                if metrics['Evaluation MAE'] < self.min_loss:
                    self.min_loss = metrics['Evaluation MAE']
                    self.min_epoch = self.epoch
                    print('Saving best model\n')
                    torch.save(self.model.state_dict(), f"{self.params['modelsdir']}/{self.params['model_name']}_ep{self.epoch}_mae{metrics['Evaluation MAE']}.pt")
                print(f'best_model_epoch = {self.min_epoch}, min_loss = {self.min_loss}')

        
        return train_log

    def run_test_loop(self, ):
        outputs = []
        realy = torch.Tensor(self.loaders['y_test']).to(self.device)
        realy = realy.transpose(1,3)[:,0,:,:]

        for iter, (x, y, g) in enumerate(self.loaders['test']):
            testx = torch.Tensor(x).to(self.device)
            g = g.to(self.device)
            with torch.no_grad():
                preds = self.model(g, testx)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...].transpose(1, 2)       
        
        print('\nBeginning logging procedure...')
        for i in range(12):
            pred = self.loaders['scaler'].inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
            metrics = metric(pred,real)
            self.perform_test_logging(metrics, 5*(i+1))
        print('Done logging...\n\n')     
            
            


    def perform_logging(self, train_logs): 
        logs = OrderedDict()
        logs["TrainingIters"] = self.itr
        logs.update(train_logs)
        logs["TimeSinceStart"] = time.time() - self.start_time

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.epoch)
        print('Done logging...\n\n')
        self.logger.flush()

    def perform_test_logging(self, metrics, itr): 
        logs = OrderedDict()
        logs['Test MAE'] = metrics[0]
        logs['Test RMSE'] = metrics[1]
        logs['Test MAPE'] = metrics[2]

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        self.logger.flush()

    def calculate_metrics(self, phase, loss, loss_count, mae_list, rmse_list, mape_list):
        mae_val = sum(mae_list) / loss_count
        rmse_val = sum(rmse_list) / loss_count
        mape_val = sum(mape_list) / loss_count

        log = {}
        log[phase + ' MAE'] = mae_val
        log[phase + ' RMSE'] = rmse_val
        log[phase + ' MAPE'] = mape_val
        
        return log

if __name__ == '__main__':
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(ROOT_DIR)
    print(f'root directory: {os.getcwd()}')

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = 'runs/'
    logdir_prefix = 'q2_pg_' 
    exp_name = 'gnn_'
    env_name = 'LA'

    logdir = logdir_prefix + exp_name + '_' + env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    params = dict(
        logdir=logdir,
        node_data_path = 'src/data/METR-LA',
        graph_data_path = 'src/data/sensor_graph/adj_mx_la.pkl',
        batch_size = 4, 
        num_workers = 0, 
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        model = GNN(12, 12, 2, 10),
        criterion = torch.nn.MSELoss(),
        lr = 5e-4,
        save_params = False,
        scalar_log_freq = 1, 
    )

    trainer = Prediction_Trainer(params)

    trainer.run_training_loop(10)
    trainer.run_test_loop()