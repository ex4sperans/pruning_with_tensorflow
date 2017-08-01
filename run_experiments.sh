#!/bin/bash

python train_network_dense.py
python prune_network.py
python deploy_pruned_model.py