import torch
import encoder_decoder_model
import engine
import data
import torchvision.transforms as T
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import utils
import matplotlib.pyplot as plt


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Setting Seed for the run, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms = T.Compose([T.ToTensor()])
    print("------------ Creating Dataset ------------")
    full_dataset = data.FolderDataset(config.IMG_PATH, transforms)

    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(train_size, val_size)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print("------------ Dataset Created ------------")
    print("------------ Creating DataLoader ------------")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.TEST_BATCH_SIZE
    )
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("------------ Dataloader Cretead ------------")

    # print(train_loader)
    loss_fn = nn.MSELoss()

    encoder = encoder_decoder_model.ConvEncoder()
    decoder = encoder_decoder_model.ConvDecoder()

    if torch.cuda.is_available():
        print("GPU Availaible moving models to GPU")
    else:
        print("Moving models to CPU")

    encoder.to(device)
    decoder.to(device)

    print(device)

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=config.LEARNING_RATE)

    # early_stopper = utils.EarlyStopping(patience=5, verbose=True, path=)
    max_loss = 9999
    train_losses, val_losses = [], []


    print("------------ Training started ------------")

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = engine.train_step(
            encoder, decoder, train_loader, loss_fn, optimizer, device=device
        )
        train_losses.append(train_loss)
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        val_loss = engine.val_step(
            encoder, decoder, val_loader, loss_fn, device=device
        )
        val_losses.append(val_loss)
        # Simple Best Model saving
        if val_loss < max_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), config.ENCODER_MODEL_PATH)
            torch.save(decoder.state_dict(), config.DECODER_MODEL_PATH)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

    print("Training Done")

    plt.figure(figsize=(8,8))
    plt.plot(train_losses, '-o', label='Train Losses')
    plt.plot(val_losses, 'g-o', label='Valid Losses')
    plt.legend()
    plt.show()

    print("---- Creating Embeddings for the full dataset ---- ")

    embedding = engine.create_embedding(
        encoder, full_loader, config.EMBEDDING_SHAPE, device
    )

    # Convert embedding to numpy and save them
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    # Dump the embeddings for complete dataset, not just train
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(config.EMBEDDING_PATH, flattened_embedding)

if __name__ == "__main__":
    main()