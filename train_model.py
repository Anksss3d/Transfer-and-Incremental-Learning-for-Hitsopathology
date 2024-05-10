import torch
import CustomDataset
import timeit
import utilities
from efficientnet_pytorch import EfficientNet
from torch import nn




def print_metrics(batch_num, total_batches, loss, epoch, config):
    """
    :param batch_num: The current batch number.
    :param total_batches: The total number of batches.
    :param loss: The loss value of the current batch.
    :param epoch: The epoch number.
    :param config: The configuration dictionary.
    :return: None

    Prints the metrics of the current batch in a formatted string. The metrics include the epoch number, loss value, progress bar, elapsed time, and estimated time of arrival (ETA).
    """
    _batch_current = timeit.default_timer()
    eta = utilities.calculate_eta(batch_num+1, total_batches, _batch_current)
    elapsed = utilities.hh_mm_ss(_batch_current)
    perc_completed = int(100*batch_num/total_batches)
    progress_bar = str("#"*perc_completed) +  str("-"*(100-perc_completed))
    print(f"\rEpoch {epoch+1}/{config['num_epochs']} loss: {loss:.6f} {progress_bar} \tElapsed: {elapsed} \tETA:"+eta, end="")

# =====================================================================================
# Preparation
# =====================================================================================
def get_datasets(config):
    """
    Returns train and validation datasets based on the given configuration.

    :param config: A dictionary containing the configuration details
    :type config: dict
    :return: Tuple containing train and validation datasets
    :rtype: tuple
    """
    file = config["base_dir"] + "train/mednet_train_data.csv"
    val_file = config["base_dir"] + "validation/mednet_test_data.csv"
    label_index = -3 if config["only_dataset"] is None else -2
    # label_index = -1

    train_dataset = CustomDataset.ImageDataset(
        filepath=file,
        img_dir=config["base_dir"] + "train/",
        label_index=label_index,
        is_val=False,
        only_dataset=config["only_dataset"]
    )

    val_dataset = CustomDataset.ImageDataset(
        filepath=val_file,
        img_dir=config["base_dir"] + "validation/",
        label_index=label_index,
        is_val=True,
        only_dataset=config["only_dataset"]
    )

    return train_dataset, val_dataset


# =====================================================================================
# Train / Test Loop
# =====================================================================================


def run_epoch(model, train_loader,  val_loader, criterion, optimizer, epoch, config):
    """
    Run a single epoch of training or validation.

    :param model: The model to be trained or evaluated.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    :param criterion: The loss function used for training or evaluation.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :param config: A dictionary containing configuration parameters.
    :return: None
    """
    model.train()
    running_loss = 0.0
    _batch_start = timeit.default_timer()
    total_batches = len(train_loader)
    device = config['device']
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        print("outputs: ", outputs, "labels: ", labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print_metrics(i, total_batches, loss, epoch, config)
        # if i>10:
        #     break
    print(" ")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        _outputs, _labels, _preds = [], [], []
        _batch_start = timeit.default_timer()
        total_batches = len(val_loader)
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            # labels = torch.argmax(labels, dim=1).to(torch.int64)
            outputs = model(images)
            print("outputs: ", outputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _labels += labels.tolist()
            _preds += predicted.tolist()
            _outputs += outputs.tolist()
            print_metrics(i, total_batches, loss, epoch, config)


        metrics, cm = utilities.calculate_metrics(
            y_true=_labels,
            y_pred=_preds,
            y_prob=_outputs
        )
        print(metrics)
        print(cm)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}], Validation Accuracy: {accuracy:.2f}%")
    print(" ")

# =====================================================================================
# Main
# =====================================================================================
def train(config):
    """
    Train the model.

    :param config: A dictionary containing configuration parameters for training.
    :type config: dict

    :return: None
    :rtype: None
    """
    train_dataset, val_dataset = get_datasets(config)
    model = EfficientNet.from_pretrained('efficientnet-b0')

    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, config['base_num_classes'])  # then update the last layer to num_classes outputs

    if config["load_model"]:
        model.load_state_dict(torch.load(config["base_dir"] + utilities.MODEL_SAVE_DIR + config["save_model_name"]))
    model._fc = nn.Linear(num_ftrs, config['num_classes'])  # then update the last layer to num_classes outputs
    model = model.to(config['device'])

    if config['num_blocks_to_unfreeze']:
        utilities.adjust_model_trainability(model, num_blocks_to_unfreeze=config["num_blocks_to_unfreeze"])

    criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.calculate_class_weights().to(config['device']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    print("Training on: {}".format(config["device"]))

    for epoch in range(config['num_epochs']):
        run_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, config)
        torch.save(model.state_dict(), config["base_dir"] + utilities.MODEL_SAVE_DIR + config["save_model_name"])



