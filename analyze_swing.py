import cv2
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from eval import ToTensor, Normalize
from model import EventDetector
import os

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing',
    3: 'Top',
    4: 'Mid-downswing',
    5: 'Impact',
    6: 'Mid-follow-through',
    7: 'Finish'
}

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
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        images = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[104, 116, 123])  # mean
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def draw_pose_on_frame(frame):
    mp_pose = Pose(static_image_mode=True, min_detection_confidence=0.3)
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS,
                       landmark_drawing_spec=DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                       connection_drawing_spec=DrawingSpec(color=(0, 0, 255), thickness=2))
    return frame

def main(video_path, model_path='models/swingnet_1800.pth.tar', output_dir='outputs/events'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    dataset = SampleVideo(video_path, transform=transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = EventDetector(pretrain=True, width_mult=1.0,
                          lstm_layers=1, lstm_hidden=256,
                          bidirectional=True, dropout=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    for sample in dataloader:
        images = sample['images']
        batch = 0
        probs = []
        while batch * 64 < images.shape[1]:
            batch_imgs = images[:, batch*64:(batch+1)*64]
            logits = model(batch_imgs.to(device))
            probs.append(F.softmax(logits.data, dim=1).cpu().numpy())
            batch += 1
        probs = np.concatenate(probs, axis=0)

    events = np.argmax(probs, axis=0)[:-1]
    confs = [probs[e, i] for i, e in enumerate(events)]

    cap = cv2.VideoCapture(video_path)
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = draw_pose_on_frame(frame)
        label = f'{event_names[i]} ({confs[i]:.2f})'
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_path = os.path.join(output_dir, f'{event_names[i]}_{e}.jpg')
        cv2.imwrite(out_path, frame)
        print(f'✅ Saved: {out_path}')
    cap.release()

    from generate_event_timeline import generate_timeline_image
    generate_timeline_image(event_dir=output_dir, output_path='outputs/timeline.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='test_video.mp4', help='Path to input video')
    args = parser.parse_args()
    main(args.path)
