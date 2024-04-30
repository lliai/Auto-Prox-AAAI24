
import os
import re

from PIL import Image
from torch.utils.data import Dataset
import pycls.core.logging as logging
from .base import BaseDataset

logger = logging.get_logger(__name__)

class ImageNet(BaseDataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split):
        super(ImageNet, self).__init__(split)
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for ImageNet".format(split)
        # logger.info("Constructing ImageNet {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    # def __init__(self, data_path, split):
    #     assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
    #     splits = ["train", "val"]
    #     assert split in splits, "Split '{}' not supported for ImageNet".format(split)
    #     logger.info("Constructing ImageNet {}...".format(split))
    #     self._data_path, self._split = data_path, split
    #     self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        # logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_files if re.match(r"^n[0-9]+$", f))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        # logger.info("Number of images: {}".format(len(self._imdb)))
        # logger.info("Number of classes: {}".format(len(self._class_ids)))


    def _get_data(self, index):
        img = Image.open(self._imdb[index]["im_path"]).convert('RGB')
        label = self._imdb[index]["class"]
        return img, label

    def __len__(self):
        return len(self._imdb)

