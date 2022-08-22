import segmentation_models_pytorch as smp


def get_smp_model(
    encoder_name="efficientnet-b0", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation='sigmoid',
):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        encoder_weights=None,
    )