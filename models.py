from unet import UNet
from adm import ADM
from dit import DiT
from uvit import UViT

__all__ = ["UNet", "ADM", "DiT", "UViT", "get_model"]

def get_model(name: str, **kwargs):
    if name == "adm":
        return ADM(dim=128, channels=1, **kwargs)
    elif name == "uvit":
        return UViT(dim=128, channels=1, **kwargs)
    elif name == "unet":
        return UNet(dim=128, channels=1, **kwargs)
    elif name == "dit":
        return DiT(
            patch_size=16,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            in_channels=1,
            **kwargs,
        )
    elif name == "test":
        from dit_alibi import DiTAlibi
        return DiTAlibi(
            patch_size=16,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            in_channels=1,
            **kwargs,
        )
    else:
        raise NotImplementedError
