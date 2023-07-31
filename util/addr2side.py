import os
import numpy as np
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int, default=1, help='ID')
args = parser.parse_args()

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class FullToSide:
    def __init__(self):
        self.MASK32 = 0xFFFF_FFFF
        self.MASK_ASLR = 0xFFF

    def to_cacheline32(self, addr, ASLR=False):
        if ASLR:
            addr = addr & self.MASK_ASLR
        return (addr & self.MASK32) >> 6

    def to_pagetable32(self, addr):
        return (addr & self.MASK32) >> 12
    
    def to_cacheline_index(self, addr):
        w = -1 if addr < 0 else 1
        addr = abs(addr)
        addr = (addr & 0xFFF) >> 6
        return w * addr

    def full_to_all(self, in_path, cacheline_path, pagetable_path, n_bits=32, ASLR=False):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        pagetable_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                cacheline = self.to_cacheline32(addr, ASLR)
                pagetable = self.to_pagetable32(addr)
            else:
                cacheline = self.to_cacheline64(addr, ASLR)
                pagetable = self.to_pagetable64(addr)
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

    def full_to_cacheline(self, in_path, out_path, n_bits=32, ASLR=False):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                cacheline = self.to_cacheline32(addr, ASLR)
            else:
                cacheline = self.to_cacheline64(addr, ASLR)
            cacheline_arr.append(w * cacheline)
        np.savez_compressed(out_path, np.array(cacheline_arr))

    def full_to_pagetable(self, in_path, out_path, n_bits=32):
        full = np.load(in_path)['arr_0']
        pagetable_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                pagetable = self.to_pagetable32(addr)
            else:
                pagetable = self.to_pagetable64(addr)
            pagetable_arr.append(w * pagetable)
        np.savez_compressed(out_path, np.array(pagetable_arr))

    def full_to_cacheline_index(self, in_path, out_path):
        full = np.load(in_path)['arr_0']
        vec_fun = np.vectorize(self.to_cacheline_index)
        cacheline_index_arr = vec_fun(full)

        # Padding
        pad_length = 300000
        if cacheline_index_arr.size < pad_length:
            cacheline_index_arr = np.pad(cacheline_index_arr, pad_width=(0, pad_length - cacheline_index_arr.size), mode='constant')
        else:
            print("Warning: trace length longer than padding length")
            cacheline_index_arr = cacheline_index_arr[:pad_length]

        assert cacheline_index_arr.shape == (pad_length,), "Size error"
        np.savez_compressed(out_path, cacheline_index_arr)

####################
#      CelebA      #
####################

if __name__ == '__main__':
    input_dir = "/media/ssuhung/Asshole's/pin_output/npz/"
    total_num = 1

    # cacheline_dir = "/media/ssuhung/Asshole's/pin_output_processed/cacheline/"
    # pagetable_dir = "/media/ssuhung/Asshole's/pin_output_processed/pagetable/"
    cacheline_index_dir = "/home/ssuhung/Downloads/pin_output_processed/cacheline_index/"

    sub_list = [sub + '/' for sub in sorted(os.listdir(input_dir))]

    # make_path(cacheline_dir)
    # make_path(pagetable_dir)
    make_path(cacheline_index_dir)

    tool = FullToSide()

    for sub in sub_list:
        total_npz_list = sorted(os.listdir(input_dir + sub))
        unit_len = int(len(total_npz_list) // total_num)

        ID = args.ID - 1
        if ID == total_num - 1:
            npz_list = total_npz_list[ID*unit_len:]
        else:
            npz_list = total_npz_list[ID*unit_len:(ID+1)*unit_len]

        # make_path(cacheline_dir + sub)
        # make_path(pagetable_dir + sub)
        make_path(cacheline_index_dir + sub)
        
        print('File: ', len(npz_list))
        print('Total: ', len(total_npz_list))

        progress = progressbar.ProgressBar(maxval=len(npz_list), widgets=widgets).start()
        for i, npz_name in enumerate(npz_list):
            progress.update(i + 1)
            
            npz_path = input_dir + sub + npz_name
            # cacheline_path = cacheline_dir + sub + npz_name
            # pagetable_path = pagetable_dir + sub + npz_name
            cacheline_index_path = cacheline_index_dir + sub + npz_name

            # tool.full_to_cacheline(
            #     in_path=npz_path,
            #     out_path=cacheline_path,
            #     n_bits=32,
            #     ASLR=False
            #     )
            # tool.full_to_pagetable(
            #     in_path=npz_path,
            #     out_path=pagetable_path,
            #     n_bits=32
            #     )
            
            # tool.full_to_all(
            #     in_path=npz_path,
            #     cacheline_path=cacheline_path,
            #     pagetable_path=pagetable_path,
            #     n_bits=32,
            #     ASLR=False
            #     )

            tool.full_to_cacheline_index(in_path=npz_path, out_path=cacheline_index_path)

        progress.finish()
