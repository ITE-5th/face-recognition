from torch.utils.data.dataset import Dataset


class FaceRecognitionDataset(Dataset):
    def __init__(self, faces):
        self.faces = faces
        self.length = len(self.faces)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.faces[index]
