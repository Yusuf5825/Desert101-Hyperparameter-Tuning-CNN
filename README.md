# 🍩 DesertClassifier: CNN ile Çok Sınıflı Tatlı Sınıflandırması

Bu proje, Desert101 veri seti kullanılarak geliştirilmiş, 4 farklı tatlı türünü (Baklava, Cannoli, Cupcakes, Donuts) görsel özelliklerine göre sınıflandıran **PyTorch** tabanlı bir Evrişimli Sinir Ağı (CNN) modelidir. Projenin ana amacı, sıfırdan kurulan bir derin öğrenme mimarisinde **hiperparametre optimizasyonu, veri artırma (Data Augmentation) ve düzenlileştirme (Regularization)** tekniklerinin aşırı öğrenme (Overfitting) üzerindeki bilimsel etkilerini analiz etmektir.

🗂️ **Dataset:** Özel Desert101 Veri Seti (Train/Test olarak ayrılmış 4 Sınıf)
*Veri setine özel hesaplanmış Normalizasyon Değerleri:* `Mean: [0.5482, 0.4636, 0.3866]`, `Std: [0.254, 0.2574, 0.2628]`

---

## 🧠 Model Mimarisi ve Hiperparametreler

Model, doğrusal olmayan karmaşık görsel özellikleri öğrenebilmesi için 2 Evrişim (Convolutional) bloğu ve ardından gelen tam bağlantılı (Fully Connected) bir sınıflandırıcıdan oluşturulmuştur:

*   **Girdi Katmanı:** 64x64 piksel RGB Görseller
*   **Evrişim Blokları (2 Adet):** Ardışık `nn.Conv2d` (3x3 kernel), `nn.ReLU` aktivasyonları ve boyut indirgeme için `nn.MaxPool2d`. Kapasite **128 Nöron (Hidden Units)** olarak belirlenmiştir.
*   **Düzenlileştirme (Regularization):** Ezberlemeyi önlemek amacıyla sınıflandırıcı bloğunda `nn.Dropout(p=0.4)` kullanılmıştır.
*   **Çıktı Katmanı:** 4 Sınıf (Tatlı Türleri)
*   **Loss Fonksiyonu:** CrossEntropyLoss
*   **Optimizasyon:** Adam Optimizer (lr=0.0005, weight_decay=1e-4)

---

## 🔬 Hiperparametre Optimizasyonunun Model Üzerindeki Bilimsel Etkisi

Görsel sınıflandırma problemlerinde model kapasitesi ve veri artırma (augmentation) arasındaki denge, modelin başarısını doğrudan etkiler. Bu projede aşırı öğrenme (overfitting) probleminden ideal genellemeye giden süreç deneysel olarak kanıtlanmıştır.

### 1. Düzenlileştirme Olmadan Önceki Durum (Ciddi Overfitting)

Eğitim sürecine yalnızca `Resize`, `ToTensor` ve `Normalize` işlemleri ile (veri artırma veya Dropout olmadan) başlandığında, model eğitim verisini tamamen ezberlemiştir.

*   **Loss Analizi:** Eğitim kaybı (Train Loss) hızla sıfıra yaklaşırken, Test Kaybı (Test Loss) 4. epoch'tan itibaren şiddetli bir şekilde artarak 2.3'lere fırlamıştır.
*   **Accuracy Analizi:** Model eğitim verisinde **%86.6** gibi yanıltıcı bir doğruluğa ulaşmış, ancak daha önce görmediği test verisinde **%45.7**'de kalarak genelleme yapamamıştır.

![Sadece Normalize Plot](results/plots/transform_v1_Only_Normalize_plot.png)
*(Grafik: Yalnızca normalizasyon uygulanan baseline modelde gözlemlenen şiddetli Overfitting durumu)*

### 2. Kapasite Artışı, Dropout ve Veri Artırma (Dengeli Model)

Overfitting'i kırmak için sisteme `RandomHorizontalFlip(p=0.4)` ve `TrivialAugmentWide()` veri artırma işlemleri eklenmiştir. Artan veri zorluğunu kaldırabilmesi için modelin kapasitesi (Hidden Units) 32'den 128'e çıkarılmış, genişleyen ağın ezber yapmasını önlemek için `%40 Dropout` ve `Weight Decay (L2 Penalty)` eklenmiştir.

*   **Loss Analizi:** Eğitim ve test kayıpları birbiriyle tutarlı bir şekilde azalmış, 18. epoch'ta Test Kaybı **1.18**'e inerek tarihi dip seviyesini görmüştür.
*   **Accuracy Analizi:** Modelin ezberleme (overfitting) eğilimi tamamen kırılmıştır. Eğitim doğruluğu **%58.5** seviyesinde istikrar kazanırken, Test doğruluğu **%55.4** seviyesine ulaşmış ve sağlıklı bir genelleme (generalization) sağlanmıştır.

![Kapasite ve Dropout Eklentisi Plot](results/plots/transform_v4_hidden_128_weight_decay_eklendi_plot.png)
*(Grafik: Dengeli veri artırma, Dropout ve L2 Regularization sonrası stabil hale gelen öğrenme eğrisi)*

---

## 🚀 Projeyi Bilgisayarınızda Çalıştırma (Testing)

Bu projeyi kendi bilgisayarınızda test etmek, hiperparametreleri değiştirmek ve modeli baştan eğitmek için aşağıdaki adımları izleyebilirsiniz.

**1. Repoyu Klonlayın:**

```bash
git clone https://github.com/Yusuf5825/Desert_Web_Hiper_Parametre_oynamasi.git
cd Desert_Web_Hiper_Parametre_oynamasi
```

**2. Sanal Ortam (Virtual Environment) Oluşturun:**

```bash
# Windows için
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux için
python3 -m venv .venv
source .venv/bin/activate
```

**3. Gerekli Kütüphaneleri Kurun:**

```bash
pip install torch torchvision matplotlib
```

**4. Jupyter Notebook Üzerinden Deneyleri Başlatın:**
Proje ana dizinindeki `Desert101.ipynb` dosyasını VS Code veya Jupyter Lab üzerinden açarak tüm hücreleri sırayla çalıştırabilir ve kendi hiperparametre deneylerinizi yapabilirsiniz.