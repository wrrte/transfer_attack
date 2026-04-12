import sys, os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from image_dataset import create_dataset
from utils import load_model, WrapperModel
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
import local_configuration
import itertools

from basic_attack_core import (
    fgsm,
    i_fgsm,
    mi_fgsm,
    mi_di_fgsm,
    mi_disi_fgsm,
    mi_ditisi_fgsm,
    mni_ditisi_fgsm,
    mi_si_fgsm,
    mi_ti_fgsm,
)

def main(args):
    all_model_names = ['ResNet18', 'ResNet50', 'inception_v3_timm', 'inception_v4_timm',
                       'DenseNet121', 'adv_inception_v3']

    source_model_names = ['ResNet50']
    source_model_names = source_model_names[args.start:args.end]
    transfer_model_names = [x for x in all_model_names]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pre-process input image
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transfer_models = [WrapperModel(load_model(x), mean, stddev).to(device) for x in transfer_model_names]
    transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(), lambda x:(x-0.5)*2])

    # Create Dataset
    dataset = create_dataset(transform, num_images=args.num_images)

    batch_size = args.batch_size
    
    # 탐색할 하이퍼파라미터 공간 (PDF의 권장 수정 항목)
    search_space = {
        'number_of_si_scales': [5, 10, 15],
        'ti_kernel_size': [7, 15],
        'di_prob': [0.5, 0.7],
        'di_pad_amount': [15, 31]
    }
    
    keys = list(search_space.keys())
    # 가능한 모든 조합 생성
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*search_space.values())]

    for model_i, source_model_name in enumerate(source_model_names):
        print(source_model_name)
        
        # load models (모델 로드는 반복문 밖에서 1번만 수행하여 시간 단축)
        source_model = WrapperModel(load_model(source_model_name), mean, stddev).to(device)
        source_model = source_model.eval()

        found_optimal = False

        # --- 하이퍼파라미터 탐색 루프 시작 ---
        for config_idx, hyperparams in enumerate(param_combinations):
            print(f"\n=======================================================")
            print(f"[{config_idx+1}/{len(param_combinations)}] Testing Hyperparameters: {hyperparams}")
            print(f"=======================================================")
            torch.cuda.empty_cache()

            # 현재 하이퍼파라미터를 kwargs로 전달하도록 래핑
            def current_atk(model, img, label):
                return mni_ditisi_fgsm(model, img, label, **hyperparams)

            attack_methods = {
                "MNI-DTS": current_atk,
            }

            def iter_source():
                dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                target_accs = {m: {k: 0. for k in attack_methods.keys()} for m in transfer_model_names}
                num_images = 0
                for i, data in enumerate(dl):
                    num_images += data[1].shape[0]

                    img, label = data
                    img = img.to(device)
                    label = label.to(device)

                    source_model.eval()
                    output_dict = {key: atk(source_model, img, label) for key, atk in attack_methods.items()}

                    if args.save_images:
                        img_dir = local_configuration.output_image / source_model_name / "ori"
                        os.makedirs(img_dir, exist_ok=True)
                        for ii in range(img.shape[0]):
                            ori_vis = ((img[ii].detach().cpu() + 1) / 2).clamp(0, 1)
                            Image.fromarray(np.uint8(ori_vis.permute(1, 2, 0) * 255)).save(
                                img_dir / f"ori_{i*batch_size+ii:04d}.png")
                    for a in attack_methods.keys():
                        if args.save_images:
                            img_dir = local_configuration.output_image / source_model_name / f"{str(a)}"
                            os.makedirs(img_dir, exist_ok=True)
                            for ii in range(img.shape[0]):
                                Image.fromarray(
                                    np.uint8(((output_dict[a] + 1) / 2)[ii].permute(1, 2, 0).cpu() * 255)).save(
                                    img_dir / f"{str(a)}_{i*batch_size+ii:04d}.png")

                    # get labels for transfer
                    for j, mod in enumerate(transfer_models):
                        mod.eval()
                        with torch.no_grad():
                            transfer_results_dict = {key: F.softmax(mod(value), dim=1).max(dim=1) for key, value in output_dict.items()}
                        for a in attack_methods.keys():
                            target_accs[transfer_model_names[j]][a] += (
                                torch.sum((transfer_results_dict[a][1] == label).float())).item()
                            
                return target_accs

            target_accs = iter_source()

            # final accuracy and scoring
            sr_values = {}
            for attack_name in attack_methods.keys():
                for j, mod in enumerate(transfer_models):
                    model_name = transfer_model_names[j]
                    acc = (target_accs[model_name][attack_name]) / len(dataset)
                    sr = 100.0 - acc * 100.0
                    if model_name not in sr_values:
                        sr_values[model_name] = {}
                    sr_values[model_name][attack_name] = sr

            robust_model_names = {
                "DenseNet121",
                "inception_v4_timm",
                "adv_inception_v3",
            }

            white_box_srs = []
            if source_model_name in sr_values:
                for attack_name in attack_methods.keys():
                    white_box_srs.append(sr_values[source_model_name][attack_name])
                white_box_score = sum(white_box_srs) / float(len(white_box_srs))
            else:
                white_box_score = 0.0

            weighted_sum = 0.0
            weight_total = 0.0
            for attack_name in attack_methods.keys():
                for model_name in transfer_model_names:
                    if model_name == source_model_name:
                        continue
                    sr = sr_values[model_name][attack_name]
                    weight = 2.0 if model_name in robust_model_names else 1.0
                    weighted_sum += sr * weight
                    weight_total += weight

            if weight_total > 0.0:
                weighted_transfer_score = weighted_sum / weight_total
            else:
                weighted_transfer_score = 0.0

            alpha = 0.4
            beta = 0.6
            final_score = alpha * white_box_score + beta * weighted_transfer_score

            print(f"-> final_score : {final_score:.2f}")

            # 목표 점수 달성 시 루프 탈출
            if final_score > 95.0:
                print("\n" + "*"*60)
                print(f"!!! SUCCESS: Found parameters with final_score ({final_score:.2f}) > 95 !!!")
                print(f"Optimal Hyperparameters: {hyperparams}")
                print("*"*60 + "\n")
                found_optimal = True
                break
        
        if found_optimal:
            break

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch_size as an integer")
    parser.add_argument("--num_images", default=1000, type=int, help="number of images to use")
    parser.add_argument("--start", default=0, type=int, help="source model index start")
    parser.add_argument("--end", default=9, type=int, help="source model index end")
    parser.add_argument("--save_images", default=False, help="T/F")
    parser.add_argument("--excel_path", default="auto_results_seo.xlsx", help="auto_results.xlsx")
    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)