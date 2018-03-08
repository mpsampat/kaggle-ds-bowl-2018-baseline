from config import Config
import numpy as np
class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bowl"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.00005
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 500

    STEPS_PER_EPOCH = 600

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    MEAN_PIXEL = np.array([0.0, 0.0, 0.0])
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (28, 28)	
    MAX_GT_INSTANCES = 500

    DETECTION_MAX_INSTANCES = 512

    RESNET_ARCHITECTURE = "resnet50"


bowl_config = BowlConfig()
bowl_config.display()
