import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_mean_std(data_dir, img_size=64):
    """
    Veri setinin Mean ve Std değerlerini hesaplar.
    img_size: Görsellerin DataLoader'dan geçebilmesi için yeniden boyutlandırılacağı değer.
              (Notebook'unuzdaki Resize değerine göre 64 yapıldı)
    """
    # Ortalama hesaplarken sadece Resize ve ToTensor kullanılır.
    # Data augmentation (Flip, Rotation vb.) KAPALI olmalıdır!
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Veri setini yükle
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Windows'ta multiprocessing hatalarını önlemek için num_workers=0
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    print(f"{len(dataset)} görsel taranıyor. Hesaplanıyor, lütfen bekleyin...")

    for data, _ in loader:
        # data boyutu: [batch_size, channels, height, width]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Tüm batch'lerin ortalaması
    mean = channels_sum / num_batches
    # Standart sapma formülü: sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    # Sonuçları virgülden sonra 4 basamak olacak şekilde yuvarlayalım
    mean = [round(m, 4) for m in mean.tolist()]
    std = [round(s, 4) for s in std.tolist()]

    return mean, std


if __name__ == '__main__':
    # Train klasörünün yolu
    train_dir = os.path.join("data", "desert101", "train")

    mean, std = get_mean_std(train_dir)

    print("\n" + "=" * 40)
    print("✨ HESAPLAMA TAMAMLANDI ✨")
    print("=" * 40)
    print(f"Mean (Ortalama): {mean}")
    print(f"Std (Standart Sapma): {std}\n")
    print("Jupyter Notebook'a kopyalayacağınız satır:")
    print(f"transforms.Normalize(mean={mean}, std={std})")
    print("=" * 40)