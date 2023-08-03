import os
import numpy as np
import argparse
import progressbar
from typing import Union

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class FullToSide:
    def to_cacheline32(self, addr):
        return (addr & 0xFFFF_FFFF) >> 6

    def to_pagetable32(self, addr):
        return (addr & 0xFFFF_FFFF) >> 12
    
    def to_cacheline(self, addr: Union[int, np.array]) -> Union[int, np.array]:
        """ addr should > 0"""
        return (addr & 0xFFF) >> 6
    
    def to_cacheline_rw(self, addr: np.array) -> np.array:
        rw = np.where(addr > 0, 1, -1)
        addr_abs = abs(addr)
        return rw * self.to_cacheline(addr_abs)

    def full_to_all(self, in_path, cacheline_path, pagetable_path):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        pagetable_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            cacheline = self.to_cacheline32(addr)
            pagetable = self.to_pagetable32(addr)
            cacheline_arr.append(w * cacheline)
            pagetable_arr.append(w * pagetable)

        # Padding
        pad_length = 300000
        for addr_type in [cacheline_arr, pagetable_arr]:
            if len(addr_type) < pad_length:
                addr_type += [0] * (pad_length - len(addr_type))
            else:
                print("Warning: trace length longer than padding length")
                addr_type = addr_type[:pad_length]

        np.savez_compressed(cacheline_path, np.array(cacheline_arr))
        np.savez_compressed(pagetable_path, np.array(pagetable_arr))

    def full_to_cacheline_32(self, in_path, out_path):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            cacheline = self.to_cacheline32(addr)
            cacheline_arr.append(w * cacheline)
        np.savez_compressed(out_path, np.array(cacheline_arr))

    def full_to_pagetable32(self, in_path, out_path):
        full = np.load(in_path)['arr_0']
        pagetable_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            pagetable = self.to_pagetable32(addr)
            pagetable_arr.append(w * pagetable)
        np.savez_compressed(out_path, np.array(pagetable_arr))

    def full_to_cacheline(self, in_path, out_path):
        full = np.load(in_path)['arr_0']
        output_len = 300000
        arr = self.to_cacheline_rw(full)

        # Padding
        assert full.size < out_path, "Error: trace length longer than padding length"
        arr = np.pad(arr, pad_width=(0, output_len - arr.size), mode='constant')

        np.savez_compressed(out_path, arr)
    
    def full_to_cacheline_encode(self, in_path, out_path):
        output_len = 300000
        assert in_path.shape[0] <= output_len, "Error: trace length longer than padding length"

        full = np.load(in_path)['arr_0']
        arr = np.zeros((output_len, 64), dtype=int)

        for i, addr in enumerate(full):
            arr[i][self.to_cacheline(addr)] = 1 if addr > 0 else -1

        np.savez_compressed(out_path, arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=int, default=1, help='ID')
    args = parser.parse_args()

    widgets = ['Progress: ', progressbar.Percentage(), ' ', 
                progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
                progressbar.Timer(), ' ', progressbar.ETA()]
    
    manifold_root = "/home/ssuhung/Manifold-SCA/"
    input_dir = manifold_root + "data/CelebA_crop128/pin/raw/"
    total_num = 4

    # cacheline32_dir = manifold_root + "data/CelebA_crop128/pin/cacheline32/"
    # pagetable32_dir = manifold_root + "data/CelebA_crop128/pin/pagetable32/"
    cacheline_dir = manifold_root + "data/CelebA_crop128/pin/cacheline/"

    sub_list = [sub + '/' for sub in sorted(os.listdir(input_dir))]

    # make_path(cacheline32_dir)
    # make_path(pagetable32_dir)
    make_path(cacheline_dir)

    tool = FullToSide()

    for sub in sub_list:
        total_npz_list = sorted(os.listdir(input_dir + sub))
        unit_len = int(len(total_npz_list) // total_num)

        ID = args.ID - 1
        npz_list = total_npz_list[ID*unit_len:(ID+1)*unit_len]

        # make_path(cacheline32_dir + sub)
        # make_path(pagetable32_dir + sub)
        make_path(cacheline_dir + sub)
        
        print('File: ', len(npz_list))
        print('Total: ', len(total_npz_list))

        progress = progressbar.ProgressBar(maxval=len(npz_list), widgets=widgets).start()
        for i, npz_name in enumerate(npz_list):
            progress.update(i + 1)
            
            npz_path = input_dir + sub + npz_name
            # cacheline32_path = cacheline_dir + sub + npz_name
            # pagetable32_path = pagetable_dir + sub + npz_name
            cacheline_path = cacheline_dir + sub + npz_name

            # tool.full_to_cacheline(
            #     in_path=npz_path,
            #     out_path=cacheline32_path
            #     )
            # tool.full_to_pagetable(
            #     in_path=npz_path,
            #     out_path=pagetable32_path
            #     )
            
            # tool.full_to_all(
            #     in_path=npz_path,
            #     cacheline_path=cacheline_path,
            #     pagetable_path=pagetable_path
            #     )

            tool.full_to_cacheline_encode(in_path=npz_path, out_path=cacheline_path)

        progress.finish()
