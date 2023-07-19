from src.models.model_wrapper import ModelWrapperBase
import src.models.dino_vit.dino_vit_wrapper
import src.models.dift.dift_wrapper
import src.models.sd_dino.sd_dino_wrapper
import src.models.open_clip.open_clip_wrapper

# This might become a circular import nightmare in the future, so might need to be changed!

# Define a list of all the classes that have inherited from the ModelWrapper class.
MODEL_DICT = {
    subclass.NAME: subclass
    for subclass in
    ModelWrapperBase.__subclasses__()
}
