
import imageio.v3 as iio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# 1. Membaca gambar awal
gambar = iio.imread("plat nomor.jpg")

# Memastikan gambar RGB
if gambar.ndim == 3 and gambar.shape[2] == 4:
    gambar = gambar[:, :, :3]

gambar = gambar.astype(np.float32)

# 2. Mengurangi noise menggunakan median filter
hasil_filter = ndimage.median_filter(
    gambar, size=(3, 3, 1)
)

# 3. Mempertajam detail gambar
blur = ndimage.gaussian_filter(
    hasil_filter, sigma=(1, 1, 0)
)

hasil_tajam = hasil_filter + 1.5 * (hasil_filter - blur)

# 4. Meningkatkan kontras
batas_bawah = np.percentile(hasil_tajam, 2)
batas_atas = np.percentile(hasil_tajam, 98)

hasil = (hasil_tajam - batas_bawah) / (
    batas_atas - batas_bawah + 1e-8
) * 255

hasil = np.clip(hasil, 0, 255).astype(np.uint8)

# 5. Menyimpan citra hasil peningkatan
iio.imwrite("citra_ditingkatkan.png", hasil)

# 6. Menampilkan perbandingan citra
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gambar.astype(np.uint8))
plt.title("Citra Awal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title("Citra Setelah Ditingkatkan")
plt.axis("off")

plt.tight_layout()
plt.savefig("perbandingan_citra.png", dpi=150)
plt.show()

print("Peningkatan kualitas citra selesai.")
print("Ukuran citra:", gambar.shape)
print("Hasil disimpan sebagai citra_ditingkatkan.png")
