import torch
import torchvision
from torch.utils.data import Dataset

from scipy import io
from PIL import Image
import os


class Gaze360(Dataset):
    """
        0 for train, 1 for eval, 2 for test and 3 for unused.
    """
    def __init__(self, path, mode="train") -> None:
        super().__init__()

        self.path = path
        self.anot_path = os.path.join(self.path, "metadata.mat")
        self.images_folder = os.path.join(self.path, "imgs/")


        # Load annotations
        annotations = io.loadmat(self.anot_path)
        recordings, recording, cropType, person_identity, frame = annotations['recordings'], annotations['recording'], "head", annotations['person_identity'], annotations['frame']
        target_pos3d, target_pos2d, gaze_dir, split = annotations['target_pos3d'], annotations['target_pos2d'], annotations['gaze_dir'], annotations['split']
        idx_range = range(len(recording[0]))

        if mode=="train":
            self.datatset_dict = {'images_by_idx': [os.path.join(self.images_folder, recordings[0, recording[0, i]][0], cropType, '%06d' % person_identity[0, i], '%06d.jpg' % frame[0, i]) for i in idx_range if split[0,i]==0],
                            'gaze_dir': [gaze_dir[i] for i in idx_range if split[0,i]==0],
                            'target_pos3d': [target_pos3d[i] for i in idx_range if split[0,i]==0],
                            "target_pos2d": [target_pos2d[i] for i in idx_range if split[0,i]==0],
                            }
        elif mode=="eval":
            self.datatset_dict = {'images_by_idx': [os.path.join(self.images_folder, recordings[0, recording[0, i]][0], cropType, '%06d' % person_identity[0, i], '%06d.jpg' % frame[0, i]) for i in idx_range if split[0,i]==1],
                            'gaze_dir': [gaze_dir[i] for i in idx_range if split[0,i]==1],
                            'target_pos3d': [target_pos3d[i] for i in idx_range if split[0,i]==1],
                            "target_pos2d": [target_pos2d[i] for i in idx_range if split[0,i]==1],
                            }
        elif mode=="test":
            self.datatset_dict = {'images_by_idx': [os.path.join(self.images_folder, recordings[0, recording[0, i]][0], cropType, '%06d' % person_identity[0, i], '%06d.jpg' % frame[0, i]) for i in idx_range if split[0,i]==2],
                            'gaze_dir': [gaze_dir[i] for i in idx_range if split[0,i]==2],
                            'target_pos3d': [target_pos3d[i] for i in idx_range if split[0,i]==2],
                            "target_pos2d": [target_pos2d[i] for i in idx_range if split[0,i]==2],
                            }
        
        assert len(self.datatset_dict["images_by_idx"]) == len(self.datatset_dict["gaze_dir"]) == len(self.datatset_dict["target_pos3d"]) == len(self.datatset_dict["target_pos2d"])
            
    def __getitem__(self, index):
        piltotensor = torchvision.transforms.PILToTensor()
        resizer = torchvision.transforms.Resize(size=(224, 224), antialias=True)

        image = resizer(piltotensor(Image.open(self.datatset_dict['images_by_idx'][index]))).to(torch.float32)
        #image = piltotensor(Image.open(self.datatset_dict['images_by_idx'][index])).to(torch.float32)
        gaze_dir = torch.tensor(self.datatset_dict['gaze_dir'][index]).to(torch.float32)
        target_pos3d = torch.tensor(self.datatset_dict['target_pos3d'][index]).to(torch.float32)
        target_pos2d = torch.tensor(self.datatset_dict['target_pos2d'][index]).to(torch.float32)
        
        return {"image":image, "gaze_dir":gaze_dir, "target_pos3d":target_pos3d, "target_pos2d":target_pos2d}
        
    def __len__(self):
        return len(self.datatset_dict["images_by_idx"])