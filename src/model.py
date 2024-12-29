from torchvision.models import get_model


def get_model(model, num_classes=1000, weights=None):
    """
    Return Resnet-50 model without loading any pretrained weights.
    """
    model = get_model(model, weights=None, num_classes=num_classes)
    if weights:
        return model.load_state_dict(weights)
    else:
        return get_model(model, weights=None, num_classes=num_classes)
