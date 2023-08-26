import json
import os

import fiftyone as fo
from class_list import old_classes
from fiftyone import ViewField as F
from PIL import Image


# Bounding box [<top-left-x>, <top-left-y>, <width>, <height>]
bbox_width = F("$metadata.width") * F("bounding_box")[2]
bbox_height = F("$metadata.height") * F("bounding_box")[3]
num_channels = F("$metadata.channels") # TODO filter channels = 3

category_dict = old_classes
train_img_path = '/home/ssuhung/Manifold-SCA/data/General/image/train'
test_img_path = '/home/ssuhung/Manifold-SCA/data/General/image/test'
validation_img_path = '/home/ssuhung/Manifold-SCA/data/General/image/validation'
export_classification_path = '/home/ssuhung/Manifold-SCA/data/General/img_category.json'
filter_conditions = [(bbox_width > 128) & (bbox_height > 128), F('IsInside') == False, F('IsDepiction') == False, F('IsGroupOf') == False, F('IsOccluded') == False]

img_category_dict = {}
cnt = 1

os.makedirs(train_img_path)
os.makedirs(test_img_path)
os.makedirs(validation_img_path)

for i, category in enumerate(category_dict):
    print(f'Category: {category}')
    class_img_num = category_dict[category]
    train_limit = int(0.8 * class_img_num)
    test_limit = int(0.9 * class_img_num)

    dataset = fo.zoo.load_zoo_dataset(
                "open-images-v7",
                splits=['train', 'test', 'validation'],
                label_types=["detections"],
                classes=category,
                max_samples=5000,
                only_matching=True
            )
    dataset.compute_metadata()

    # Filter multiple objects in one box
    view = dataset.view()
    for condition in filter_conditions:
        new_view = view.filter_labels("ground_truth", condition, only_matches=True)
        if len(new_view) < class_img_num:
            print(f'break at condition {condition}')
            not_good_view = view.filter_labels("ground_truth", condition == False, only_matches=True)
            not_good_view = not_good_view.limit(class_img_num - len(new_view))
            view = new_view + not_good_view
            break
        view = new_view

    # Filter bounding box aspect ratio
    bbox_width_pl = F("$metadata.width") * F("ground_truth.detections")[0]('bounding_box')[2]
    bbox_height_pl = F("$metadata.height") * F("ground_truth.detections")[0]('bounding_box')[3]
    bbox_aratio = bbox_width_pl / bbox_height_pl
    view = view.sort_by(abs(1 - bbox_aratio))

    # Make images in one category has fixed number
    assert len(view) >= class_img_num, f"Number of images containing {category} less than {class_img_num}"
    view = view.limit(class_img_num)

    # Crop images
    for j, sample in enumerate(view.iter_samples(progress=True)):
        img = Image.open(sample.filepath)
        img_width, img_height = img.size
        # Bounding box [<top-left-x>, <top-left-y>, <width>, <height>] between 0 and 1
        bbox_left_x, bbox_upper_y, bbox_width, bbox_height = sample.ground_truth.detections[0].bounding_box
        left = int(bbox_left_x * img_width)
        right = int((bbox_left_x + bbox_width) * img_width)
        upper = int(bbox_upper_y * img_height)
        lower = int((bbox_upper_y + bbox_height) * img_height)

        img = img.crop((left, upper, right, lower))
        img = img.resize((128, 128), Image.LANCZOS)
        if img.mode != 'RGB': img = img.convert('RGB')

        # file_name = sample.filepath.split('/')[-1].split('.')[0] + '.jpg'
        file_name = f'{cnt}.jpg'
        cnt += 1
        if j < train_limit:
            file_path = os.path.join(train_img_path, file_name)
        elif j < test_limit:
            file_path = os.path.join(test_img_path, file_name)
        else:
            file_path = os.path.join(validation_img_path, file_name)
        img.save(file_path)
        img_category_dict[file_name] = i

    dataset.delete()

with open("img_category.json", "w") as f:
    json.dump(img_category_dict, f)
