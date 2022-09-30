import torch 
import torchvision.transforms as transforms
import torch.optim as optim 
import torchvision.transforms.functional as FT 
from tqdm import tqdm 
from model import YoloV1
from dataset import VOCDataset



SEED = 42
torch.manual_seed(SEED)

LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32
EPOCHS = 1000
NUM_WORKERS = 2
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'


class Compose(object) :
    def __init__(self, transforms) :
        self.transforms = transforms

    def __call__(self, img, bboxes) :
        for t in self.transforms :
            img, bboxes = t(img), bboxes

        return img, bboxes

transformer = Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor()
])


def train_step(train_loader, model, optimizer, loss_fn) :
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch, (x,y) in enumerate(loop) :
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        mean_loss.append(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #progress bar
        loop.set_postfix(loss=loss.detach())
    print(f"Mean loss : {sum(mean_loss)/len(mean_loss)}")