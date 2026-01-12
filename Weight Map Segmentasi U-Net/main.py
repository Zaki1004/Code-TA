# Import Library

MODEL_PATH = "/content/drive/MyDrive/New/Gamma-0.9-UnifiedFocalLoss-E150/unet_best_ufl.keras"
IMAGE_DIR = "/content/drive/MyDrive/New/AllDataImagesCleaning/"
SAVE_DIR  = "/content/drive/MyDrive/New/Gamma-0.9-UnifiedFocalLoss-E150/HasilWeightMap(13)/"
BEST_DIR  = os.path.join(SAVE_DIR, "HasilTerbaik/")   
IMG_SIZE = (256, 256)
NUM_SAMPLES = 5851  

PENALTIES = {"crack": 2, "rutting": 3, "pothole": 5}
BASE_COST = 1.0
DILATE_ITERS = 5
GAUSSIAN_SIGMA = 2.0
DETECTION_THRESHOLD = 1.0  

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# Load Model

print("📦 Memuat model U-Net ...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model berhasil dimuat!\n")

def decode_prediction(pred_mask, threshold=0.3):
    "Konversi prediksi model ke mask RGB dan label map."
    pred_mask[pred_mask < threshold] = 0
    label_map = np.argmax(pred_mask, axis=-1)

    colors = {
        0: (0, 0, 0),       
        1: (255, 0, 0),     
        2: (0, 255, 0),     
        3: (0, 0, 255)      
    }

    color_mask = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for k, c in colors.items():
        color_mask[label_map == k] = c
    return color_mask, label_map


def build_penalty_map_from_label(label_map):
    "Membangun peta penalti dari label hasil prediksi."
    mask_pothole = binary_dilation(label_map == 1, iterations=DILATE_ITERS)
    mask_crack   = binary_dilation(label_map == 2, iterations=DILATE_ITERS)
    mask_rutting   = binary_dilation(label_map == 3, iterations=DILATE_ITERS)

    pen_map = np.maximum.reduce([
        mask_pothole * PENALTIES["pothole"],
        mask_crack * PENALTIES["crack"],
        mask_rutting * PENALTIES["rutting"]
    ])
    return gaussian_filter(pen_map, sigma=GAUSSIAN_SIGMA)


def make_weight_map(penalty_map):
    return BASE_COST + penalty_map

# Visualisasi

print(f"🔍 Mencari gambar di: {IMAGE_DIR}")

import random
random.seed(42) 

extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
image_paths = sorted(list(set(image_paths)))
random.shuffle(image_paths)
image_paths = image_paths[:NUM_SAMPLES]

if not image_paths:
    print(f"Tidak ada gambar ditemukan di {IMAGE_DIR}")
else:
    print(f"Ditemukan {len(image_paths)} gambar untuk diproses.\n")

    all_metadata = []

    for path in tqdm(image_paths, desc="🔄 Memproses gambar", unit="img"):
        fname = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(SAVE_DIR, f"{fname}_visualisasi.png")

        img = cv2.imread(path)
        if img is None:
            tqdm.write(f"Gagal membaca gambar: {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        X = np.expand_dims(img / 255.0, axis=0)
        pred = model.predict(X, verbose=0)[0]

        if pred.shape[-1] != 4:
            tqdm.write(f"Output model memiliki {pred.shape[-1]} channel")
            continue

        mask_rgb, label_map = decode_prediction(pred, threshold=0.3)
        penalty_map = build_penalty_map_from_label(label_map)
        weight_raw = np.ones_like(penalty_map) * BASE_COST
        weight_final = make_weight_map(penalty_map)

        num_pothole = np.sum(label_map == 1)
        num_crack   = np.sum(label_map == 2)
        num_rutting   = np.sum(label_map == 3)
        total_damage = num_pothole + num_crack + num_rutting
        pct_damage = (total_damage / (IMG_SIZE[0] * IMG_SIZE[1])) * 100

        # Visualisasi 4 Tahap

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))  
        axs[0].imshow(img)
        axs[0].set_title("(a) Input", fontsize=13, pad=10)

        axs[1].imshow(mask_rgb)
        axs[1].set_title("(b) Prediction U-Net\nRed: Pothole | Green: Crack | Blue: Rutting", fontsize=13, pad=10)

        im1 = axs[2].imshow(weight_raw, cmap='viridis', vmin=0, vmax=6)
        axs[2].set_title("(c) Weight Map Awal (Cost=1)", fontsize=13, pad=10)
        plt.colorbar(im1, ax=axs[2], fraction=0.046, pad=0.04)

        im2 = axs[3].imshow(weight_final, cmap='inferno', vmin=0, vmax=6)
        axs[3].set_title("(d) Weight Map Final (Pinalty)", fontsize=13, pad=10)
        plt.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis('off')

        fig.subplots_adjust(top=0.85, wspace=0.15)
        fig.suptitle(
            f"{fname}\nDamaged Area: {pct_damage:.2f}% | Pothole: {num_pothole} | Crack: {num_crack} | Rutting: {num_rutting}",
            fontsize=12, fontweight="bold", y=0.97
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

        meta = {
            "filename": fname,
            "damage_percentage": float(pct_damage),
            "pixels_pothole": int(num_pothole),
            "pixels_crack": int(num_crack),
            "pixels_rutting": int(num_rutting),
            "total_damage_pixels": int(total_damage),
            "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "saved_path": save_path
        }
        all_metadata.append(meta)

        if pct_damage > DETECTION_THRESHOLD:
            copy2(save_path, BEST_DIR)
            tqdm.write(f"{fname} terdeteksi rusak ({pct_damage:.2f}%) → disalin ke folder HasilTerbaik/")
        else:
            tqdm.write(f"{fname} diproses (kerusakan {pct_damage:.2f}%)")

    metadata_path = os.path.join(SAVE_DIR, "metadata_report.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Semua hasil visualisasi tersimpan di:\n📂 {SAVE_DIR}")
    print(f"Hasil terbaik (kerusakan > {DETECTION_THRESHOLD}%) di:\n📁 {BEST_DIR}")
    print(f"Metadata report tersimpan di:\n📄 {metadata_path}")
    print(f"{'='*70}")