import argparse
import os
import sys
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
from torch.utils.data import random_split, DataLoader

import dreambooth_helpers.dreambooth_trainer_configurations as db_cfg
from dreambooth_helpers.arguments import parse_arguments
from dreambooth_helpers.dataset_helpers import WrappedDataset, ConcatDataset
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
from dreambooth_helpers.copy_and_name_checkpoints import copy_and_name_checkpoints
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config, load_model_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            train=None,
            reg=None,
            validation=None,
            test=None,
            predict=None,
            wrap=False,
            num_workers=None,
            shuffle_test_loader=False,
            use_worker_init_fn=False,
            shuffle_val_dataloader=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
        if reg is not None:
            self.dataset_configs["reg"] = reg

        self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        train_set = self.datasets["train"]
        if 'reg' in self.datasets:
            reg_set = self.datasets["reg"]
            train_set = ConcatDataset(train_set, reg_set)

        return DataLoader(train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


def filter_trainer_args(trainer_args):
    """
    Filter out deprecated PyTorch Lightning Trainer arguments that are no longer supported
    """
    # List of deprecated arguments in newer PyTorch Lightning versions
    deprecated_args = [
        'checkpoint_callback',
        'early_stop_callback', 
        'progress_bar_refresh_rate',
        'process_position',
        'num_processes',
        'distributed_backend',
        'terminate_on_nan',
        'auto_scale_batch_size',
        'auto_lr_find',
        'benchmark',
        'deterministic',
        'reload_dataloaders_every_epoch',
        'replace_sampler_ddp',
        'num_sanity_val_steps',
        'truncated_bptt_steps',
        'resume_from_checkpoint',
        'profiler',
        'weights_summary',
        'weights_save_path',
        'amp_backend',
        'amp_level',
        'automatic_optimization',
        'move_metrics_to_cpu',
        'multiple_trainloader_mode',
        'stochastic_weight_avg'
    ]
    
    # Filter out deprecated arguments
    filtered_args = {k: v for k, v in trainer_args.items() if k not in deprecated_args}
    
    # Handle specific argument transformations if needed
    if 'gpus' in filtered_args and filtered_args['gpus'] is not None:
        # In newer versions, use 'devices' instead of 'gpus'
        if 'devices' not in filtered_args:
            filtered_args['devices'] = filtered_args.pop('gpus')
        else:
            filtered_args.pop('gpus', None)
    
    # Handle precision argument - might need special handling
    if 'precision' in filtered_args:
        precision_val = filtered_args['precision']
        if isinstance(precision_val, str) and precision_val not in ['16', '32', '64', 'bf16']:
            # Remove invalid precision values
            filtered_args.pop('precision', None)
    
    return filtered_args


if __name__ == "__main__":
    # Generate the config from the input arguments
    dreambooth_config: JoePennaDreamboothConfigSchemaV1 = parse_arguments()

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    try:
        trainer = None

        # Create our configurations
        dreambooth_model_config = db_cfg.get_dreambooth_model_config(config=dreambooth_config)
        dreambooth_data_config = db_cfg.get_dreambooth_data_config(config=dreambooth_config)
        dreambooth_lightning_config = db_cfg.get_dreambooth_lightning_config(config=dreambooth_config)
        dreambooth_model_data_config = db_cfg.get_dreambooth_model_data_config(
            model_config=dreambooth_model_config,
            data_config=dreambooth_data_config,
            lightning_config=dreambooth_lightning_config,
        )

        # Load our model
        model = load_model_from_config(
            config=dreambooth_model_data_config,
            ckpt=dreambooth_config.model_path,
            verbose=False,
        )
        model.learning_rate = dreambooth_config.learning_rate
        if dreambooth_config.debug:
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # Setup our trainer
        dreambooth_trainer_config = db_cfg.get_dreambooth_trainer_config(
            config=dreambooth_config,
            model=model,
            lightning_config=dreambooth_lightning_config
        )
        dreambooth_callbacks_config = db_cfg.get_dreambooth_callbacks_config(
            config=dreambooth_config,
            model_data_config=dreambooth_model_data_config,
            lightning_config=dreambooth_lightning_config,
        )
        dreambooth_trainer_kwargs = db_cfg.get_dreambooth_trainer_kwargs(
            config=dreambooth_config,
            trainer_config=dreambooth_trainer_config,
            callbacks_config=dreambooth_callbacks_config,
        )

        trainer_opt = argparse.Namespace(**dreambooth_trainer_config)
        
        # Convert argparse Namespace to dict and merge with dreambooth_trainer_kwargs
        trainer_args = vars(trainer_opt)
        trainer_args.update(dreambooth_trainer_kwargs)
        
        # Filter out deprecated arguments
        trainer_args = filter_trainer_args(trainer_args)
        
        if dreambooth_config.debug:
            print("Trainer arguments after filtering:", list(trainer_args.keys()))
        
        trainer = Trainer(**trainer_args)        
        trainer.logdir = dreambooth_config.log_directory()

        # Setup the data
        data = instantiate_from_config(config=dreambooth_data_config)

        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        if dreambooth_config.debug:
            print("#### Data #####")
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0 and trainer.global_step > 0:
                print(f"We encountered an error at step {trainer.global_step}. Saving checkpoint 'last.ckpt'...")
                ckpt_path = os.path.join(dreambooth_config.log_checkpoint_directory(), "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

                print(f"Copying trained model(s) to {dreambooth_config.trained_models_directory()}")
                copy_and_name_checkpoints(config=dreambooth_config)


        import signal
        # Windows fix
        signal.signal(signal.SIGTERM, melk)

        # run the training
        try:
            # save the config
            dreambooth_config.save_config_to_file(
                save_path=dreambooth_config.log_directory()
            )
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    except Exception as e:
        if trainer is not None and trainer.global_rank == 0:
            print(f"Error training at step {trainer.global_step}.")
            print(e)
        raise
    finally:
        if trainer is not None and trainer.global_rank == 0:
            if trainer.global_step == dreambooth_config.max_training_steps:
                print(f"Training complete. Successfully ran for {trainer.global_step} steps")
                print(f"Copying trained model(s) to {dreambooth_config.trained_models_directory()}")
                copy_and_name_checkpoints(config=dreambooth_config)
            else:
                print(f"Error training at step {trainer.global_step}")

            if dreambooth_config.debug:
                print(trainer.profiler.summary())
