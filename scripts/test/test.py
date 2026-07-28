import os
import torch
# import sys
# sys.path.append('/path/to/your/project/folder')
# maybe occur path error, comment above codes to solve
import typer
from comer.datamodule.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)

def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)

    # --- FIX START: Remap legacy PAM keys to new FAB keys ---
    checkpoint = torch.load(ckp_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("comer_model.PAM.", "comer_model.FAB.")
        new_state_dict[new_key] = v
        
    checkpoint["state_dict"] = new_state_dict
    
    tmp_ckp_path = ckp_path + ".tmp"
    torch.save(checkpoint, tmp_ckp_path)
    
    model = LitCoMER.load_from_checkpoint(tmp_ckp_path, strict=True)
    
    if os.path.exists(tmp_ckp_path):
        os.remove(tmp_ckp_path)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
