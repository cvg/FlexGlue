"""torch.hub entry points, e.g. torch.hub.load("cvg/FlexGlue", "flexglue", trust_repo=True)."""

dependencies = ["torch", "numpy", "cv2"]


def flexglue(**conf):
    """FlexGlue matcher with its DINOv3 descriptor, conf as FlexGlue(**conf)."""
    from flexglue import FlexGlue

    return FlexGlue(**conf)
