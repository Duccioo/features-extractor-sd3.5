from torchvision import transforms
from PIL import ImageOps, Image
import random
from io import BytesIO


class RobustJPEGTransform:
    """Applica compressione JPEG random per mitigare il bias di formato PNG/JPEG."""

    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        # Applica con probabilità p (usa p=1.0 per validation/test se vuoi standardizzare tutto)
        if random.random() < self.p:
            output = BytesIO()
            # Scelta random della qualità
            q = random.randint(*self.quality_range)
            img.save(output, "JPEG", quality=q)
            output.seek(0)
            return Image.open(output).convert("RGB")
        return img


class StandardPreprocessor:
    def __init__(self, image_size=1024, mode="imagenet_style", jpeg_aug=True):
        """
        Args:
            image_size: 1024 per SD3.5
            mode: 'imagenet_style' (Resize Shortest Edge + Center Crop)
            jpeg_aug: Se True, applica compressione JPEG on-the-fly.
        """
        self.image_size = image_size
        self.mode = mode
        self.jpeg_aug = jpeg_aug
        self.transform = self._build_transform()

    def _build_transform(self):
        pipeline = [
            transforms.Lambda(lambda im: ImageOps.exif_transpose(im)),
            transforms.Lambda(lambda im: im.convert("RGB")),
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
