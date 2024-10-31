# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path": "./data/",
        "start_train_date": "2018-06-04",
        "end_train_date": "2022-04-29",
        "start_validation_date": "2022-05-04",
        "end_validation_date": "2023-02-24",
        "start_test_date": "2023-02-27",
        "end_test_date": "2023-06-06",
        "timesteps_dim": 30,
        "n_predictions": 1,
        "ticket": "HPG",
    },
    'agent': {
        'train': {
            'episodes': 500,

            'critic_coeff': 0.5,
            'entropy_coeff': 0.01,
            'eps_clip': 0.2,       
            'gamma': 0.01,             
            
            'batch_size': 300,  
            'lr_actor': 0.0003,       
            'lr_critic': 0.001,      
            'K_epochs': 20, 
            'action_std': 0.6,                   
            'action_std_decay_rate': 0.05,       
            'min_action_std': 0.1,                
            'action_std_decay_freq': int(30) 
        },
        "env" : {
            'shift_reward' : 0.2,
            'scale_reward' : 20
        },
        "save" : {
            'save_freq' : 30,
            'base' : 'results/',
            "checkpoints" : "checkpoints/",
            "tensorboards" : "tensorboards/",
            'figs' : "figs/",
            'configs': 'configs/',
            'loggers': 'loggers/'
        },
        "mode" : "train",
    },
}   
