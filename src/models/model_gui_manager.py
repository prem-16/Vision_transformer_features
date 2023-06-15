from src.models.model_wrapper_list import MODEL_DICT


class ModelGUIManager:
    """
    A class to manage the models, their settings and further
    provide an interface for the GUI to interact with the models.
    """

    def __init__(self):
        self._settings = None
        self._selected_model = None
        # A flag to indicate that processing is required.
        self._dirty = True
        self._image_dir_1 = None
        self._image_dir_2 = None

    @property
    def model_name(self):
        return self._selected_model.NAME

    @property
    def model(self):
        return self._selected_model

    @property
    def image_dir_1(self):
        return self._image_dir_1

    @image_dir_1.setter
    def image_dir_1(self, image_dir_1):
        if self._image_dir_1 != image_dir_1:
            self._image_dir_1 = image_dir_1
            self._set_dirty()

    @property
    def image_dir_2(self):
        return self._image_dir_2

    @image_dir_2.setter
    def image_dir_2(self, image_dir_2):
        if self._image_dir_2 != image_dir_2:
            self._image_dir_2 = image_dir_2
            self._set_dirty()

    def _set_dirty(self):
        self._dirty = True

    def update_model(self, model_name):
        assert model_name in MODEL_DICT, f"Model {model_name} not found!"
        self._selected_model = MODEL_DICT[model_name]()
        self._settings = {}
        self._set_dirty()

    def apply_setting(self, setting_name, setting_value):
        if self._settings.get(setting_name, None) != setting_value:
            self._settings[setting_name] = setting_value
            print("Applied setting ", setting_name, " with value ", setting_value)
            self._set_dirty()

    def process_images(self):
        """
        Compute a one time process of the two images.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :return: None
        """
        self._selected_model.process_image_pair(self._image_dir_1, self._image_dir_2, self._settings)
        self._dirty = False

    def get_heatmap_vis(self, point):
        """
        Compute the heatmap of the second image.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :param point: The point to compare to.
        :return: The heatmap visualization of the second image.
        """
        if self._dirty:
            self.process_images()

        return self._selected_model.get_heatmap_vis(self._image_dir_2, point)

