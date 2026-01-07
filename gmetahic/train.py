import argparse
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import h5py
from typing import Tuple

from gnn_model import *
from utils import *
from dataloader import *

from training_utils import *

parser = argparse.ArgumentParser(description="PyTorch gmetahic")

parser.add_argument(
    # "--data-path", metavar="DIR", default="./datasets", help="path to dataset"
    "--data-path", metavar="DIR", default="./datasets/", help="path to dataset"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=1.0,
    help="Gradient clipping threshold (default: 1.0)"
)

parser.add_argument(
    "--imr90_overlap", default=34, help=""
)
parser.add_argument(
    "--HUVEC_overlap", default=32, help=""
)
parser.add_argument(
    "--gm12878_overlap", default=16, help=""
)

parser.add_argument(
    # "--save-path", default="./checkpoints", help="path to save model chekpoints"
    "--save-path", default="../checkpoints", help="path to save model chekpoints"

)

parser.add_argument(
    "--bulk-checkpoint",
    # default="./checkpoints/deterministic/gm12878_hg38_umb_endo_imr90_CTCFmotifScore_seed1229.pth.tar",
    default="../checkpoints/ablation_cancel_gcn/imr90_HUVEC_gm12878_CTCFmotifScore_seed1234_gnn_bicross_second_3cells_2025-10-31_22h.pth.tar",
    help="pbulk chekpoint",
)

# parser.add_argument("-ct", "--ct-list", nargs="+", default=[])
# parser.add_argument("-ct", "--ct-list", nargs="+", default=['imr90','HUVEC','gm12878'])
parser.add_argument("-ct", "--ct-list", nargs="+", default=['K562'])
# parser.add_argument("-ct", "--ct-list", nargs="+", default=['imr90'])
parser.add_argument("--genome", default="hg38", help="genome of training cell type")
parser.add_argument(
    "--min-stripe-signal",
    default=-170,
    type=int,
    help="Stripe signal cutoff; avoid training on low-signal stripes",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=1, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "-b", "--batch-size", default=32, type=int, help="batch size (default: 64)"
)
parser.add_argument(
    "--patience", default=10, type=int, help="training patience in epochs (default: 10)"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-2,
    type=float,
    metavar="LR",
    help="initial learning rate,default:1e-2",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-5,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-5)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=1234, type=int, help="seed for initializing training. "
)

parser.add_argument("--dropout", "--do", type=float, default=0.0, help="dropout")

parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--deterministic", action="store_true", help="cudnn deterministic")
# parser.add_argument("--mod-name", default="", help="model name")
parser.add_argument("--mod_name", default="ablation_cancel_gcn_fine_K562", help="model name")
parser.add_argument("--use_chip", default="False",
                    help="Whether to use Chip or motif score for CTCF, options = [True, False]")


def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        if args.deterministic:
            cudnn.deterministic = True
            logging.warning(
                "You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down "
                "your training considerably! You may see unexpected behavior when restarting from checkpoints."
            )
    #             cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")

    torch.manual_seed(args.seed)

    data_path = args.data_path
    save_path = args.save_path
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    initial_rate = args.lr
    wd = args.weight_decay
    clip_grad = args.clip_grad
    ct_list = args.ct_list
    genome = args.genome
    dropout = args.dropout

    use_chip = args.use_chip

    # Verify validity of input and output paths.
    assert os.path.isdir(
        data_path
    ), "data_path does not exist. Please create directory first."
    assert os.path.isdir(
        save_path
    ), "save_path does not exist. Please create directory first."

    if args.mod_name:
        save_path = os.path.join(save_path, args.mod_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    print("Saving to {}".format(save_path))

    # Training specifications.
    train_chrom = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19]
    val_chrom = [3, 15]
    test_chrom = [5, 18, 20, 21]

    # The following should not be changed unless the user modifies the dimensions in the model accordingly.
    input_size = 4010000
    train_step = 5e04
    val_step = 5e04

    #     start_dict = {}
    #     end_dict = {}
    #     for i in hic_dict[ct_list[1]].keys():
    #         idx = np.where(hic_dict[ct_list[0]][i].toarray().sum(1)>-600)[0]
    #         start_dict[i] = idx[0] * 10000
    #         end_dict[i] = idx[-1] * 10000
    #     for ct in ct_list[1:]:
    #         for i in hic_dict[ct_list[1]].keys():
    #             idx = np.where(hic_dict[ct][i].toarray().sum(1)>-600)[0]
    #             start_dict[i] = max(start_dict[i], idx[0] * 10000)
    #             end_dict[i] = min(end_dict[i], idx[-1] * 10000)

    # Start and end indices of each chromosome
    start_dict = {
        "chr13": 0,
        "chr21": 0,
        "chr3": 0,
        "chr20": 0,
        "chrX": 0,
        "chr8": 0,
        "chr10": 0,
        "chr15": 0,
        "chr5": 0,
        "chr2": 0,
        "chr14": 0,
        "chr6": 0,
        "chr7": 0,
        "chr17": 0,
        "chr18": 0,
        "chr16": 0,
        "chr1": 0,
        "chr11": 0,
        "chr4": 0,
        "chr9": 0,
        "chr12": 0,
        "chr19": 0,
        "chr22": 0
    }
    if args.genome == 'hg38':
        end_dict = {
            "chr1": 248950000,
            "chr2": 242190000,
            "chr3": 198290000,
            "chr4": 190210000,
            "chr5": 181530000,
            "chr6": 170800000,
            "chr7": 159340000,
            "chr8": 145130000,
            "chr9": 138390000,
            "chr10": 133790000,
            "chr11": 135080000,
            "chr12": 133270000,
            "chr13": 114360000,
            "chr14": 107040000,
            "chr15": 101990000,
            "chr16": 90330000,
            "chr17": 83250000,
            "chr18": 80370000,
            "chr19": 58610000,
            "chr20": 64440000,
            "chr21": 46700000,
            'chr22': 50810000,
            "chrX": 156030000
        }




    elif args.genome == 'hg19':
        end_dict = {
            "chr1": 249250000,
            "chr2": 243190000,
            "chr3": 198020000,
            "chr4": 191150000,
            "chr5": 180910000,
            "chr6": 171110000,
            "chr7": 159130000,
            "chr8": 146360000,
            "chr9": 141210000,
            "chr10": 135530000,
            "chr11": 135000000,
            "chr12": 133850000,
            "chr13": 115160000,
            "chr14": 107340000,
            "chr15": 102530000,
            "chr16": 90350000,
            "chr17": 81190000,
            "chr18": 78070000,
            "chr19": 59120000,
            "chr20": 63020000,
            "chr21": 48120000,
            'chr22': 51300000,
            "chrX": 155270000
        }

    # Specify efective genome size for normalization
    if "hg" in genome:
        effective_genome_size = 2913022398
        # effective_genome_size = 2700000000
    elif "mm" in genome:
        effective_genome_size = 2652783500
    else:
        raise ValueError(
            "Please compute effective_genome_size manuallly and modify this function accordingly."
        )

    # LOAD DATA
    print("Loading training data for these cell types:")
    print(ct_list)
    print(args)

    hic_dict = {}
    hic_qval_dict = {}
    pbulk_dict = {}
    scatac_dict = {}
    n_cells = {}
    n_metacells = {}
    libsize_cell = {}

    for ct in ct_list:
        hic_dict[ct], hic_qval_dict[ct] = load_hic(data_path, ct)
        pbulk_dict[ct], scatac_dict[ct] = load_atac(data_path, ct, pbulk_only=False)
        n_cells[ct] = get_num_cells(pbulk_dict[ct])
        n_metacells[ct] = get_num_cells(scatac_dict[ct], dim=1)
        libsize_cell[ct] = get_libsize(pbulk_dict[ct])

    ctcf_dict = load_ctcf_motif(data_path, genome)
    # if use_chip == 'True':
    #     ctcf_dict = load_ctcf_motif(data_path, ct, use_chip = True)
    # else:
    #     ctcf_dict = load_ctcf_motif(data_path, genome)

    # Create dataset objects for training and validation
    train_dataset = Dataset(
        input_size,
        effective_genome_size,
        hic_dict,
        pbulk_dict,
        scatac_dict,
        ctcf_dict,
        n_cells,
        n_metacells,
        libsize_cell,
        *get_starts(
            chrom_list=train_chrom,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=ct_list,
            step=train_step,
        ),
        transform=normalize,
        is_training=True
    )
    # print("train_dataset:",len(train_dataset))

    val_dataset = Dataset(
        input_size,
        effective_genome_size,
        hic_dict,
        pbulk_dict,
        scatac_dict,
        ctcf_dict,
        n_cells,
        n_metacells,
        libsize_cell,
        *get_starts(
            chrom_list=val_chrom,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=ct_list,
            step=val_step,
        ),
        transform=normalize
    )

    # Create dataloader objects for training and validation
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
    )

    # 创建优化后的模型
    model = create_graph_enhanced_chromafold(dropout=dropout)

    # model.load_state_dict(torch.load(args.bulk_checkpoint),strict=True)

    # # 初始化模型权重
    # initialize_model_weights(model)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    state = torch.load(args.bulk_checkpoint, map_location='cpu')
    state = state.get('state_dict', state)  # 若checkpoint外层有'state_dict'就取出来
    model.load_state_dict(state, strict=True)



    model.to(args.device)

    # # 使用优化的优化器配置
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=initial_rate,
    #     weight_decay=wd,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     amsgrad=True  # 使用AMSGrad变种提高稳定性
    # )
    #
    # # 使用改进的学习率调度器
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=initial_rate * 10,
    #     steps_per_epoch=len(train_loader),
    #     epochs=N_EPOCHS,
    #     pct_start=0.3,
    #     div_factor=10.0,
    #     final_div_factor=100.0
    # )
    #
    # # 使用保持原始数值尺度的损失函数
    # criterion = weighted_mse_loss
    #
    # # Train
    # print("Training...")
    # model, optimizer, _ = training_loop(
    #     train,
    #     validate,
    #     model,
    #     ct_list,
    #     args.seed,
    #     criterion,
    #     optimizer,
    #     scheduler,
    #     train_loader,
    #     val_loader,
    #     N_EPOCHS,
    #     args.patience,
    #     save_path,
    #     args.device,
    #     args.min_stripe_signal,
    #     clip_grad
    # )

    ## baseline
    optimizer = torch.optim.SGD(
        model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=wd
    )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=initial_rate,  # 建议对 AdamW 用 1e-3 ~ 3e-4
    #     weight_decay=wd,  # 一般 1e-2 起步
    #     betas=(0.9, 0.98),
    #     eps=1e-8
    # )


    ## chengxj
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=initial_rate, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8
    # )

    ## chromaFold
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[N_EPOCHS / 10 * 4, N_EPOCHS / 10 * 8], gamma=0.1
    )

    # # chengxj
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=N_EPOCHS,
    #     eta_min=initial_rate * 0.01
    # )

    criterion = weighted_mse_loss
    # criterion = robust_weighted_mse_loss


    # Train
    print("Training...")
    model, optimizer, _ = training_loop(
        train,
        validate,
        model,
        ct_list,
        args.seed,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        N_EPOCHS,
        args.patience,
        save_path,
        args.device,
        args.min_stripe_signal,
        clip_grad
    )


if __name__ == "__main__":
    main()