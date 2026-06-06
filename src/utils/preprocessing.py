from torchvision import transforms
from PIL import ImageOps, Image
import random
from io import BytesIO


def exif_transpose_fn(img):
    return ImageOps.exif_transpose(img)


def convert_rgb_fn(img):
    return img.convert("RGB")


class CropToMultipleOf32:
    """Crops the image to the closest multiple of 32 to prevent SD3.5 tensor dimension mismatch."""

    def __call__(self, img):
        w, h = img.size
        new_w = w - (w % 32)
        new_h = h - (h % 32)
        if new_w == w and new_h == h:
            return img

        # Center crop the image slightly
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        return img.crop((left, top, right, bottom))


class SafeResizeToMax1152:
    """Resizes the image if any dimension exceeds 1152, maintaining aspect ratio.
    Then crops to nearest multiple of 32."""

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > 1152:
            scale = 1152 / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Now apply the multiple of 32 crop logic
        w, h = img.size
        new_w = w - (w % 32)
        new_h = h - (h % 32)
        if new_w == w and new_h == h:
            return img

        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        return img.crop((left, top, right, bottom))


class RobustJPEGTransform:
    """Applica compressione JPEG random per mitigare il bias di formato PNG/JPEG."""

    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        # Applica con probabilità p (usa p=1.0 per validation/test se vuoi standardizzare tutto)
        if random.random() < self.p:
            buffer = BytesIO()
            # Scelta random della qualità
            q = random.randint(*self.quality_range)
            img.save(buffer, "JPEG", quality=q)
            buffer.seek(0)
            # Load image into memory and close buffer to prevent memory leak
            jpeg_img = Image.open(buffer)
            jpeg_img.load()  # Force load into memory
            buffer.close()
            return jpeg_img.convert("RGB")
        return img


class StandardPreprocessor:
    def __init__(self, image_size=1024, mode="imagenet_style", jpeg_aug=True):
        """
        Args:
            image_size: 1024 per SD3.5
            mode: 'imagenet_style' (Resize Shortest Edge + Center Crop),
                'brutal_resize' (direct resize),
                'crop_100_then_resize' (CenterCrop 100x100 + Resize to image_size),
                'none' (No resize or crop),
                'closest_multiple_of_32' (Center crop imperceptibly to nearest multiple of 32)
            jpeg_aug: Se True, applica compressione JPEG on-the-fly.
        """
        self.image_size = image_size
        self.mode = mode
        self.jpeg_aug = jpeg_aug
        self.transform = self._build_transform()

    def _build_transform(self):
        pipeline = [
            transforms.Lambda(exif_transpose_fn),
            transforms.Lambda(convert_rgb_fn),
        ]

        # 1. Geometric Transformations
        if self.mode == "imagenet_style":
            # Resize(size) con un int ridimensiona il lato PIÙ CORTO a size
            # mantenendo l'aspect ratio.
            pipeline.append(
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )
            # Poi ritagliamo il centro esatto
            pipeline.append(transforms.CenterCrop(self.image_size))

        elif self.mode == "brutal_resize":
            # Il tuo vecchio metodo (sconsigliato per SD3.5 features)
            pipeline.append(
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )

        elif self.mode == "crop_100_then_resize":
            pipeline.append(transforms.CenterCrop((100, 100)))
            pipeline.append(
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )
            # pipeline.append(transforms.CenterCrop(self.image_size))

        elif self.mode == "none":
            pass  # Nessun resize né crop

        elif self.mode == "closest_multiple_of_32":
            pipeline.append(CropToMultipleOf32())

        elif self.mode == "safe_resize_1152":
            pipeline.append(SafeResizeToMax1152())

        else:
            raise ValueError(
                f"Unsupported preprocessing mode: {self.mode}. "
                "Valid options: imagenet_style, brutal_resize, crop_100_then_resize, none, closest_multiple_of_32, safe_resize_1152"
            )

        # 2. Corruption / Format Standardization
        if self.jpeg_aug:
            # Applico JPEG qui, sull'immagine 1024x1024
            # Consiglio p=1.0 durante la validazione per eliminare il bias PNG
            pipeline.append(RobustJPEGTransform(quality_range=(80, 95), p=1))

        # 3. Normalization for Stable Diffusion
        pipeline.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        return transforms.Compose(pipeline)

    def __call__(self, image):
        return self.transform(image)
