import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Iterable, Any, Callable, Union
import dm_pix as pix
from fd import calculate_frechet_distance
from safetensors import safe_open
import pickle

class Evaluator:
    stats: list[Any] = []

    def __init__(self, feature_extractor: Callable[[jax.Array], jax.Array]):
        self.feature_extractor = feature_extractor

    def update(self, y_hat: jax.Array, y: jax.Array) -> None:
        """Assumes y_hat and y are in [0,1] range."""

        y_hat_features = self.feature_extractor(y_hat)
        y_features = self.feature_extractor(y)
        stats = get_metrics(y_hat, y)
        stats["y_hat_features"] = y_hat_features
        stats["y_features"] = y_features
        self.stats.append(stats)

    def calculate(self) -> dict[str, float]:
        y_hat_features = jnp.concatenate(
            [s["y_hat_features"] for s in self.stats], axis=0
        )
        y_features = jnp.concatenate([s["y_features"] for s in self.stats], axis=0)

        data_mu = jnp.mean(y_features, axis=0)
        data_sigma = jnp.cov(y_features, rowvar=False)

        model_mu = jnp.mean(y_hat_features, axis=0)
        model_sigma = jnp.cov(y_hat_features, rowvar=False)

        fd = calculate_frechet_distance(data_mu, data_sigma, model_mu, model_sigma)

        mse = jnp.mean(jnp.stack([s["mse"] for s in self.stats]))
        mae = jnp.mean(jnp.stack([s["mae"] for s in self.stats]))
        ssim = jnp.mean(jnp.stack([s["ssim"] for s in self.stats]))
        psnr = jnp.mean(jnp.stack([s["psnr"] for s in self.stats]))
        return {
            "mse": mse.item(),
            "mae": mae.item(),
            "ssim": ssim.item(),
            "psnr": psnr.item(),
            "fd": fd.item(),
        }

    def reset(self) -> None:
        self.stats = []

def get_metrics(y_hat: jax.Array, y: jax.Array) -> dict[str, float]:
    """Assumes y_hat and y are in [0,1] range."""
    mse = jnp.mean(jnp.square(y - y_hat))
    mae = jnp.mean(jnp.abs(y - y_hat))
    ssim = jnp.mean(pix.ssim(y_hat, y))
    psnr = jnp.mean(pix.psnr(y_hat, y))
    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "ssim": ssim.item(),
        "psnr": psnr.item(),
    }


def cycle(dl: Iterable[Any]) -> Any:
    while True:
        it = iter(dl)
        for x in it:
            yield x


def make_grid(images: jax.Array, nrow: int, ncol: int) -> jax.Array:
    """Simple helper to generate a single image from a mini batch."""

    def image_grid(
        nrow: int, ncol: int, imagevecs: jax.Array, imshape: tuple[int, ...]
    ) -> jax.Array:
        images = iter(imagevecs.reshape((-1,) + imshape))
        return jnp.squeeze(
            jnp.vstack(
                [
                    jnp.hstack([next(images) for _ in range(ncol)])
                    for _ in range(nrow)
                ]
            )
        )

    batch_size = images.shape[0]
    image_shape = images.shape[1:]
    return image_grid(
        nrow=nrow,
        ncol=ncol,
        imagevecs=images[0 : nrow * ncol],
        imshape=image_shape,
    )


def save_image(img: jax.Array, path: str) -> None:
    """Assumes image in [0,1] range"""
    img = np.array(jnp.clip(img * 255 + 0.5, 0, 255)).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def load_safetensors(path):
    params = {}
    with safe_open(path, framework="numpy", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            subkeys = key.split("/")

            item = params
            for key in subkeys[:-1]:
                if key not in item:
                    item[key] = {}
                item = item[key]
            item[subkeys[-1]] = tensor
    return {"params": params}

def load_params(path):
    if path.endswith(".pkl"):
        # Assuming pickle has full TrainingState
        with open(path, "rb") as f:
            return pickle.load(f).params
    elif path.endswith(".safetensors"):
        return load_safetensors(path)
    else:
        raise NotImplementedError

