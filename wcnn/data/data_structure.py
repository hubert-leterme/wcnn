class CustomDatasetMixin:
    """
    Additional properties and methods for datasets.

    """
    @property
    def img_attrs(self):
        """
        List of attribute names. These attributes are lists containing as many
        elements as the number of training examples.

        """
        raise NotImplementedError

    @property
    def class_attrs(self):
        """
        List of attribute names. These attributes are lists containing as many
        elements as the number of classes.

        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        Number of classes.

        """
        return len(self.classes) # Works for any dataset?

    @property
    def ready_for_use(self):
        """
        Returns True if already equipped with a proper transform.

        """
        try:
            flag = self._ready_for_use
        except AttributeError:
            flag = False
        return flag

    @property
    def custom_split(self):
        return self._split

    @ready_for_use.setter
    def ready_for_use(self, newflag):
        assert isinstance(newflag, bool)
        self._ready_for_use = newflag

    @property
    def debug(self):
        try:
            out = self._debug
        except AttributeError:
            out = False
        return out

    @debug.setter
    def debug(self, newflag):
        assert isinstance(newflag, bool)
        self._debug = newflag


    def truncate(self, index, keep_until=True, update_classes=True):
        """
        Truncate the dataset up to a certain index. In-place modification.

        Parameters
        ----------
        index (int): index value until or from which to truncate the dataset
        keep_until (bool): if True, then the dataset is kept UNTIL the value given
            by index. Otherwise, it is kept FROM this value. Default=True
        update_classes (bool): if True, update the class-related attributes in
            order to only keep those present in the truncated dataset.
            CURRENTLY ONLY AVAILABLE FOR IMAGENET

        """
        # Slice image-related attributes
        for attr_name in self.img_attrs:
            attr = getattr(self, attr_name)
            attr = attr[:index] if keep_until else attr[index:]
            setattr(self, attr_name, attr)

        # Slice class-related attributes
        if update_classes:
            self._update_classes(keep_until)


    def remove_imgs(self, list_of_idx):
        list_of_idx.sort(reverse=True)
        for attr_name in self.img_attrs:
            attr = getattr(self, attr_name)
            for idx in list_of_idx:
                attr.pop(idx)
            setattr(self, attr_name, attr)


    def keep_imgs(self, list_of_idx):
        for attr_name in self.img_attrs:
            attr = getattr(self, attr_name)
            attr = [attr[idx] for idx in list_of_idx]
            setattr(self, attr_name, attr)


    # To be overridden in the child classes
    def _update_classes(self, keep_until):
        raise NotImplementedError
