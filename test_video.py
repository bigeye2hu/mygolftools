import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
import os

# 显示动作节点名称
event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

device = torch.device('cpu')

class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            if img is None:
                continue
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()

    print(f'Preparing video: {args.path}')

    ds = SampleVideo(args.path, transform=transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True, width_mult=1.0, lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)
    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print('Loaded model weights')

    print('Testing...')
    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * args.seq_length < images.shape[1]:
            image_batch = images[:, batch * args.seq_length:(batch + 1) * args.seq_length, :, :, :] \
                if (batch + 1) * args.seq_length <= images.shape[1] else images[:, batch * args.seq_length:, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    confidence = [probs[e, i] for i, e in enumerate(events)]
    print('Predicted event frames:', events)
    print('Confidence:', [np.round(c, 3) for c in confidence])

    # 使用骨架图替代原始帧展示
    for i, e in enumerate(events):
        pose_img_path = f'outputs/poses/frame_{e}.jpg'
        if not os.path.exists(pose_img_path):
            print(f'⚠️ Missing pose image for frame {e}: {pose_img_path}')
            continue
        img = cv2.imread(pose_img_path)
        if img is None:
            print(f'⚠️ Failed to load image {pose_img_path}')
            continue
        cv2.putText(img, f'{event_names[i]} ({confidence[i]:.3f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(f'{event_names[i]}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
