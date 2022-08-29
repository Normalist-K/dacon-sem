import os
from datetime import datetime
import warnings

import torch
import wandb
import hydra
from omegaconf import OmegaConf
from torchsummaryX import summary


def extras(cfg):

    print("Disabling python warnings! <config.ignore_warnings=True>")
    warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if not cfg.get("name"):
        print(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py name=experiment_name`"
        )
        print("Exiting...")
        exit()
    
    print(OmegaConf.to_yaml(cfg))

    if not cfg.DEBUG and not os.path.exists(cfg.path.submissions): 
        os.makedirs(cfg.path.submissions)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):

    from src.trainer import Trainer
    from src.utils.utils import save_submission, set_seed, load_dataloader

    cfg.resume = True

    wandb.init(project=cfg.project, entity=cfg.entity, name=f'{cfg.name}')

    set_seed(cfg.seed)
    extras(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(device)

    # Load data
    _, _, test_loader = load_dataloader(cfg)
    
    # Init lightning model
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    trainer = Trainer(cfg, model, device)

    # Test the model
    if not cfg.DEBUG:
        print("Starting testing")
        results = trainer.inference(test_loader)
        print("Saving result...")
        save_submission(cfg, results)
        print("Done!")


if __name__ == "__main__":
    main()