import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Iterable, Any


def cycle(dl: Iterable[Any]) -> Any:
    while True:
        it = iter(dl)
        for x in it:
            yield x


def make_grid(images: jax.Array, nrow: int, ncol: int) -> jax.Array:
    """Simple helper to generate a single image from a mini batch."""

    def image_grid(nrow, ncol, imagevecs, imshape):
        images = iter(imagevecs.reshape((-1,) + imshape))
        return jnp.squeeze(
            jnp.vstack(
                [
                    jnp.hstack([next(images) for _ in range(ncol)][::-1])
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
