import os
from PIL import Image

input_dir = '/home/ssuhung/Manifold-SCA/data/CelebA_crop128/image'
output_dir = '/home/ssuhung/Manifold-SCA/WebP'
splits = ['train', 'test']

for split in splits:
    input_split_dir = os.path.join(input_dir, split)
    output_split_dir = os.path.join(output_dir, split)
    files = os.listdir(input_split_dir)
    for file in files:
        prefix = file.split('.')[0]
        output_name = f'{prefix}.webp'
        output_path = os.path.join(output_split_dir, output_name)
        input_path = os.path.join(input_split_dir, file)
        img = Image.open(input_path)
        img.save(output_path)
