import torch

from src.train_lora import train_lora


print("Starting BLIP LoRA training...")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


train_lora(
    annotations_path="dataset/annotations.csv",
    images_dir="dataset/images",
    output_dir="/content/drive/MyDrive/dl_task2/models/blip-lora-lithuanian",
    epochs=3,
    batch_size=2,
    learning_rate=5e-5,
)
