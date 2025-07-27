# generate_event_timeline.py

import os
from PIL import Image

def generate_timeline_image(event_dir='outputs/events', output_path='outputs/timeline.jpg'):
    ordered_events = [
        'Address',
        'Toe-up',
        'Mid-backswing',
        'Top',
        'Mid-downswing',
        'Impact',
        'Mid-follow-through',
        'Finish'
    ]

    if not os.path.exists(event_dir):
        print(f"❌ 事件图片目录不存在: {event_dir}")
        return

    images = []
    for event in ordered_events:
        for fname in os.listdir(event_dir):
            if fname.startswith(event):
                path = os.path.join(event_dir, fname)
                img = Image.open(path)
                images.append(img)
                break

    if not images:
        print("❌ 没有找到任何事件图片用于拼接")
        return

    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]
    total_width = sum(img.width for img in resized_images)

    timeline_img = Image.new('RGB', (total_width, min_height))
    x_offset = 0
    for img in resized_images:
        timeline_img.paste(img, (x_offset, 0))
        x_offset += img.width

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timeline_img.save(output_path)
    print(f"✅ 已生成时间线拼接图：{output_path}")
