from . import MobileViT


def mobilevit(image_size, num_classes, kind):
    kind = kind.lower()
    if kind.startswith('xxs'):
        channels = [16, 16, 24, 24, 48, 64, 80, 320]
        dims = [64, 80, 96]
        expansion = 2
    elif kind.startswith('xs'):
        channels = [16, 32, 48, 48, 64, 80, 96, 384]
        dims = [96, 120, 144]
        expansion = 4
    elif kind.startswith('s'):
        channels = [16, 32, 64, 64, 96, 128, 160, 640]
        dims = [144, 192, 240]
        expansion = 4
    else:
        raise ValueError("`kind` must be in ('xxs', 'xs', 's')")

    return MobileViT(
        image_size=image_size,
        num_classes=num_classes,
        chs=channels,
        dims=dims,
        depths=[2, 4, 3],
        expansion=expansion,
        # kernel_size=3,
        # patch_size=(2, 2),
    )
