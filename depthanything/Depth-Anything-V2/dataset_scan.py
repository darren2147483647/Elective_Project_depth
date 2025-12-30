import argparse
import numpy as np
from collections import Counter
from dataset import DepthDataset  # 匯入你提供的模組

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--csv", type=str, default="./dataset/data/nyu2_train.csv")
    args = parser.parse_args()

    dataset = DepthDataset(args.data_path, args.csv)
    H_list = []
    W_list = []
    HW_counter = Counter()

    print("Scanning dataset... 這可能需要一點時間。")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        h, w = sample["image_original_shape"]
        H_list.append(h)
        W_list.append(w)
        HW_counter[(h, w)] += 1

    H_arr = np.array(H_list)
    W_arr = np.array(W_list)

    print("\n======== Dataset Image Shape Statistics ========")
    print(f"Total Images: {len(dataset)}")
    print(f"Height:  min={H_arr.min()}, max={H_arr.max()}, mean={H_arr.mean():.2f}")
    print(f"Width :  min={W_arr.min()}, max={W_arr.max()}, mean={W_arr.mean():.2f}")

    print("\nMost common (H, W) pairs:")
    for (h, w), c in HW_counter.most_common(20):
        print(f"  ({h}, {w}) : {c} images")

    print("\nUnique resolutions:", len(HW_counter))
    print("================================================")

if __name__ == "__main__":
    main()
