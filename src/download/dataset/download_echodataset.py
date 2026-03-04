import os
from datasets import load_dataset
from tqdm import tqdm


def save_dataset_images(
    dataset_split, base_path, split_name, subfolder="ai", limit=None
):
    """Salva le immagini in una struttura di cartelle specifica."""
    output_dir = os.path.join(base_path, split_name, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # Se limit è specificato, prendi solo un sottoinsieme
    data_to_process = dataset_split.select(range(limit)) if limit else dataset_split

    print(f"Salvataggio in {output_dir}...")
    for i, item in enumerate(tqdm(data_to_process)):
        # Il dataset Echo-4o-Image ha solitamente una colonna 'image'
        # Update: The dataset actually uses 'jpg' key
        img = item.get("image") or item.get("jpg")
        if img is None:
            print(f"Warning: No image found for item {i}. Keys: {item.keys()}")
            continue

        img_path = os.path.join(output_dir, f"img_{i:06d}.png")
        img.save(img_path, "PNG")


# 1. Carica il dataset originale (streaming=False per scaricare tutto)
print("Caricamento dataset da Hugging Face...")
# Nota: Echo-4o-Image ha principalmente lo split 'train'.
# Se lo split 'val' non esiste nel repo, lo script userà una porzione del train.
ds = load_dataset("Yejy53/Echo-4o-Image")

# Definiamo i percorsi principali
full_path = "data/Echo-4o-Image"
mini_path = "data/Echo-4o-Image-mini"

# 2. Gestione degli split (Train e Val)
# Se il dataset ha solo 'train', creiamo noi una divisione
if "validation" not in ds:
    print("Split 'validation' non trovato. Creazione split artificiale...")
    ds_split = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_data = ds_split["train"]
    val_data = ds_split["test"]
else:
    train_data = ds["train"]
    val_data = ds["validation"]

# # 3. Download e organizzazione Dataset Completo
# print("\n--- Elaborazione Dataset Completo ---")
# save_dataset_images(train_data, full_path, "train")
# save_dataset_images(val_data, full_path, "val")

# 4. Creazione Sottoinsieme Mini (1000 train, 500 val)
print("\n--- Elaborazione Dataset Mini ---")
save_dataset_images(train_data, mini_path, "train", limit=1000)
save_dataset_images(val_data, mini_path, "val", limit=500)

print("\nOperazione completata con successo!")
