import argparse
import os

import numpy as np
import progressbar


# Parameters (Set these by yourself)
num_worker = 8
repo_root = ''
target_path = os.path.join(repo_root, 'target/tjexample')
npz_output_root = os.path.join(repo_root, 'data/CelebA_jpg/pin/raw/')
img_root = os.path.join(repo_root, 'data/CelebA_jpg/image/')

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def raw2npz(in_path, npz_path) -> int:
    with open(in_path, 'r') as f:
        lines = f.readlines()

    mem_arr = []
    for info in lines[:-1]:
        # Format: "lib;rtn;ins op addr"
        op, addr_16 = info.split(' ')[-2:]
        addr = int(addr_16, 16)
        if op == 'R':
            mem_arr.append(addr)
        elif op == 'W':
            mem_arr.append(-addr)
        else:
            print(f'Unknown operation in {in_path}')

    # print('Length: ', len(mem_arr))
    # if len(mem_arr) < pad_length:
    #     mem_arr += [0] * (pad_length - len(mem_arr))
    # else:
    #     mem_arr = mem_arr[:pad_length]

    np.savez_compressed(npz_path, np.array(mem_arr))
    return len(mem_arr)


widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]
splits = ['train/', 'test/']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=int, default=1, help='ID, start from 1')
    args = parser.parse_args()

    pin_out = f'mem_access_{args.ID}.out'
    pin = f'../../../pin -t obj-intel64/mem_access.so -o {pin_out}'
    max_len = 0

    make_path(npz_output_root)

    for split in splits:
        image_dir = os.path.join(img_root, split)
        total_img_list = sorted(os.listdir(image_dir))
        unit_len = len(total_img_list) // num_worker
        ID = args.ID - 1
        img_list = total_img_list[ID*unit_len:(ID+1)*unit_len]

        make_path(os.path.join(npz_output_root, split))

        print('Total number of images: ', len(img_list))
        print('Number for this worker: ', len(total_img_list))

        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(img_list)).start()
        for i, img in enumerate(img_list):
            progress.update(i + 1)
            
            img_path = os.path.join(image_dir, img)
            prefix = img.split('.')[0]
            npz_path = os.path.join(npz_output_root, split, prefix + '.npz')
            
            # Target libjpeg
            os.system(f'{pin} -- {target_path} {img_path} {"img_output_" + str(args.ID) + ".bmp"} > /dev/null')
            # Target libwebp
            # os.system(f'{pin} -- {target_path} {img_path} -bmp -o {"img_output_" + str(args.ID) + ".bmp"} > /dev/null 2>&1')

            leng = raw2npz(in_path=pin_out, npz_path=npz_path)
            max_len = max(max_len, leng)

    progress.finish()
    print(f"{max_len=}")
