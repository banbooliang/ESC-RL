import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.trainer_esc import TrainerESC
from models.blip_esc import blip_decoder_esc
from dataset import create_dataset, create_sampler, create_loader
from modules import utils
from transformers import BertTokenizer
from modules.logger import create_logger

os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='mimic_cxr_esc')
    parser.add_argument('--image_dir', type=str, default='./data/mimic_cxr/')
    parser.add_argument('--ann_path', type=str, default='./data/mimic_cxr/mimic_annotation_promptmrg.json')
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--load_pretrained', type=str, default='checkpoints/mimic_cxr/model_best.pth')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--gen_max_len', type=int, default=150)
    parser.add_argument('--gen_min_len', type=int, default=100)

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--save_dir', type=str, default='results/esc_mimic')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1')

    parser.add_argument('--init_lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--warmup_lr', type=float, default=5e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--cls_weight', type=float, default=4)
    parser.add_argument('--clip_k', type=int, default=21)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--two_stage_class_embed_share', default=False)
    parser.add_argument('--align_weight', type=float, default=1)
    parser.add_argument('--c', type=float, default=0.01)
    parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'Hyperboloid', 'PoincareBall'])
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--num-layers_post', type=int, default=3)
    parser.add_argument('--dim', type=int, default=75)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--rank_weight', type=float, default=1)

    parser.add_argument('--use_rl', type=bool, default=True)
    parser.add_argument('--rl_start_epoch', type=int, default=3)
    parser.add_argument('--scst_weight', type=float, default=0.99)
    parser.add_argument('--gear_weight', type=float, default=0.5)
    parser.add_argument('--spl_weight', type=float, default=1.0)
    parser.add_argument('--train_sample_method', type=str, default='nucleus')
    parser.add_argument('--train_beam_size', type=int, default=1)
    parser.add_argument('--sc_baseline_beam_size', type=int, default=1)
    parser.add_argument('--train_top_p', type=float, default=0.9)
    parser.add_argument('--train_temperature', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    parser.add_argument('--spl_num_candidates', type=int, default=4)
    parser.add_argument('--pref_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--spl_tau_lower', type=float, default=0.0)
    parser.add_argument('--spl_tau_upper', type=float, default=1.5)
    parser.add_argument('--spl_infer_consensus_threshold', type=float, default=0.55)
    parser.add_argument('--use_llm_refine_train', type=bool, default=True)
    parser.add_argument('--use_llm_refine_infer', type=bool, default=True)
    parser.add_argument('--llm_refiner_provider', type=str, default='none', choices=['none', 'openai_compatible', 'local_hf'])
    parser.add_argument('--llm_model_name', type=str, default='gpt-4.1-mini')
    parser.add_argument('--llm_api_base', type=str, default='')
    parser.add_argument('--llm_api_key', type=str, default='')
    parser.add_argument('--llm_timeout', type=int, default=120)
    parser.add_argument('--llm_temperature', type=float, default=0.0)
    parser.add_argument('--llm_max_tokens', type=int, default=256)
    parser.add_argument('--llm_cache_path', type=str, default='results/esc_llm_cache.jsonl')
    parser.add_argument('--llm_device_map', type=str, default='auto')

    # DRM extractor selection
    parser.add_argument('--drm_extractor', type=str, default='mavl', choices=['approx', 'mavl', 'medklip'])

    # Official MAVL grounding path (matches the uploaded test.py flow)
    parser.add_argument('--mavl_repo_root', type=str, default='./MAVL-Zero-shot_grounding')
    parser.add_argument('--mavl_checkpoint', type=str, default='./checkpoints/external_extractors/checkpoint_full_40.pth')
    parser.add_argument('--mavl_text_encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--mavl_disease_book', type=str, default='./data/mimic_cxr/observation explanation.json')
    # parser.add_argument('--mavl_concept_book', type=str, default='./data/mimic_cxr/concept_book.json')
    parser.add_argument('--mavl_mode', type=str, default='avg', choices=['avg', 'max', 'global'])
    parser.add_argument('--mavl_avg_last4', type=bool, default=True)
    parser.add_argument('--config', type=str, default='./MAVL-Zero-shot_grounding/configs/covid_mavl.yaml')

    # Official MedKLIP grounding path (parallel interface)
    # parser.add_argument('--medklip_repo_root', type=str, default='')
    parser.add_argument('--medklip_checkpoint', type=str, default='')
    parser.add_argument('--medklip_text_encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--medklip_disease_book', type=str, default='')
    parser.add_argument('--medklip_avg_last4', type=bool, default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    save_dir = os.path.join(args.save_dir, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, dist_rank=args.local_rank, name=args.exp_name)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])
    args.pad_token_id = tokenizer.pad_token_id

    print('Creating dataset...')
    train_dataset, val_dataset, test_dataset = create_dataset(f'generation_{args.dataset_name}', tokenizer, args)

    with open('./data/mimic_cxr/base_probs.json', 'r') as f:
        base_probs = json.load(f)
    base_probs = np.array(base_probs) / np.max(base_probs)
    base_probs = np.append(base_probs, [1, 1, 1, 1])

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)
        samplers = [samplers[0], None, None]
    else:
        samplers = [None, None, None]

    train_dataloader, val_dataloader, test_dataloader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size] * 3,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    labels_temp = ['[BLA]'] * 18
    prompt_temp = ' '.join(labels_temp) + ' '
    model = blip_decoder_esc(args, device, tokenizer, image_size=args.image_size, prompt=prompt_temp)

    if args.load_pretrained:
        ckpt = torch.load(args.load_pretrained, map_location='cpu')
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'load checkpoint from {args.load_pretrained}')
        print(msg)

    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    trainer = TrainerESC(model, criterion_cls, base_probs, metrics, args, logger, train_dataloader, val_dataloader, test_dataloader, device, utils.is_main_process)
    trainer.train()


if __name__ == '__main__':
    main()
