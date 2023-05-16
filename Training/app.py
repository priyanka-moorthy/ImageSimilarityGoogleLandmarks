from flask import Flask, request, json, render_template
import encoder_decoder_model
import config
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import utils
import data
import time
from pathlib import Path


app = Flask(__name__)

print("App started")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load the model before we start the server
encoder = encoder_decoder_model.ConvEncoder()
# Load the state dict of encoder
encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
# Loads the embedding
embedding = np.load(config.EMBEDDING_PATH)
print("Loaded model and embeddings")


def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.

    Args:
    image_tenosr: PIL read image_tensor whose similar images are needed.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def plot_similar_images(indices_list, epoch):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """
    img_links = []

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Setting Seed for the run, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms = T.Compose([T.ToTensor()])
    print("------------ Creating Dataset ------------")
    full_dataset = data.FolderDataset(config.IMG_PATH, transforms)

    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_path  = full_dataset.all_imgs_dir[index]
            # img_path = os.path.join(config.DATA_PATH + img_name)
            print(img_path)
            img = Image.open(img_path).convert("RGB")
            # plt.imshow(img)
            # plt.show()
            Path("./static/outputs/results/").mkdir(parents=True, exist_ok=True)
            img.save(f"./static/outputs/results/recommended_{index - 1}.jpg")
            img_links.append("outputs/results/recommended_{}.jpg".format(index - 1))
    return img_links

# For the home route and health check
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/similarimages", methods=["POST"])
def simimages():
    image = request.files["image"]
    # print("Hi")
    epoch = time.strftime("%Y%m%d-%H%M%S")


    image = Image.open(image).convert("RGB")
    image.save(f"./static/query/search_{epoch}.jpg")
    transform_size = T.Resize((config.IMG_WIDTH,config.IMG_HEIGHT))
    resized_img = transform_size(image)
    image_tensor = T.ToTensor()(resized_img)
    image_tensor = image_tensor.unsqueeze(0)
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    img_links = plot_similar_images(indices_list, epoch)
    # Need to display the images
    res_obj = {"indices_list": indices_list, "images":img_links, "query_img": "query/search_{}.jpg".format(epoch)}
    return render_template('results.html', 
        result = res_obj,
    )

if __name__ == "__main__":
    app.run(debug=False, port=9000)
