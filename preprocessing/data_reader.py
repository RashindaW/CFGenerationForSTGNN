import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch_geometric as pyg   
from torch_geometric import nn, data
from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='METRLA', help='Dataset name')
parser.add_argument('--lag', type=int, default=1, help='Lag for temporal data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
parser.add_argument('--ddp', action='store_true', help='Use Distributed Data Parallel')


#get parse arguments
args=parser.parse_args()
DATASET = args.dataset
BATCH_SIZE = args.batch_size
DDP = args.ddp
LAG = args.lag

class DataReader:
    def __init__(self, dataset,lag,batch_size,ddp):
        self.dataset = dataset
        self.data_path = os.path.join('data', dataset)
        self.lag = lag
        self.batch_size = batch_size
        self.ddp = ddp
        if ddp:
            print("To be implemented: DDP support - need Dask")
        print(self.data_path)

    def read_data(self):
        if self.dataset == 'METRLA':
            return self._read_metrla()
        elif self.dataset == 'PEMSBAY':
            return self._read_pemsbay()
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")
        
    def _read_metrla(self):
        loader=METRLADatasetLoader(raw_data_dir=self.data_path+'/',index=True)
        train,val,test,edges,edge_weight,_,_ = loader.get_index_dataset(
            lags=self.lag,
            batch_size=self.batch_size
        )
        return (train, val, test, edges, edge_weight)
    def _read_pemsbay(self):
        train,val,test,edges,edge_weight,_,_ = PemsBayDatasetLoader(raw_data_dir=self.data_path,index=True).get_index_dataset(
            lags=self.lag,
            batch_size=self.batch_size
        )
        return (train, val, test, edges, edge_weight)


if __name__ == "__main__":
    data_reader = DataReader(DATASET,LAG,BATCH_SIZE,DDP)
    train, val, test, edges, edge_weight = data_reader.read_data()
    print("Data loaded for dataset:", DATASET)
    print(f"Train batches: {len(train)}, Val batches: {len(val)}, Test batches: {len(test)}")