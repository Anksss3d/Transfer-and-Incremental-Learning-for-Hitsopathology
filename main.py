from train_model import train
import utilities

# =====================================================================================
# Configuration
# =====================================================================================
config = {
    "base_dir": utilities.generic_dir,
    "base_model_name": "mednet_base_14_class_20_epochs.pt",
    "base_num_classes": 14,
    "save_model_name": "mednet_trained_bone_4_classes_10_epochs.pt",
    "load_model": False,
    "num_classes": 3,
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.00001,
    "device": utilities.DEVICE,
    "only_dataset": "Osteosarcoma-UT",
    "num_blocks_to_unfreeze": None,
}


if __name__ == "__main__":
    train(config)