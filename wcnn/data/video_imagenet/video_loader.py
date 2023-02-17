import cv2
# from skvideo.io import VideoCapture
# import skvideo.io
import torch
from torchvision.datasets.folder import DatasetFolder
from torchvision import transforms
from PIL import Image


class VideoFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=None):
        super(VideoFolder, self).__init__(
            root, loader, ['.mp4'], transform=transform, target_transform=target_transform)

        self.vids = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # cap = VideoCapture(path)
        cap = cv2.VideoCapture(path)

        frames = []

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                transform = self.transform
            else:
                transform = transforms.ToTensor()
            frames.append(transform(Image.fromarray(frame)).unsqueeze(0))

        cap.release()

        return torch.cat(frames, 0), target
