import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from main import VAE

def load_model(model_path, input_dim, latent_dim, device):
    model = VAE(input_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_images(model, num_images, latent_dim, device):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = model.decoder(z)
        generated_images = generated_images.view(num_images, 1, 28, 28)
        return generated_images

def plot_images(images, num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    fig.suptitle("Generate Data")
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()

def show_true_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    images, _ = next(iter(data_loader))

    fig, axes = plt.subplots(1, 20, figsize=(20, 1))
    fig.suptitle("True Data")
    for i in range(20):
        axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 784
    latent_dim = 20
    model_path = "VAE/result/vae_model_final.pth"
    num_images = 20

    model = load_model(model_path, input_dim, latent_dim, device)
    generated_images = generate_images(model, num_images, latent_dim, device)
    plot_images(generated_images, num_images)
    show_true_data()
