import json
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CONDITION_NAMES_14 = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

# Same 13-disease aggregation used by the REVTAF approximate branch.
# Each disease index maps to one or more MAVL/REVTAF observation channels.
DISEASE_TO_OBS = {
    0: [17],
    1: [24],
    2: [9],
    3: [65],
    4: [11],
    5: [14],
    6: [20],
    7: [12],
    8: [10],
    9: [8],
    10: [16],
    11: [25],
    12: [32],
}

MAVL_ORIGINAL_CLASS = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
    'process', 'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion',
    'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification',
    'prominence', 'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire',
    'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt',
    'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate',
    'obscure', 'deformity', 'hernia', 'drainage', 'distention', 'shift', 'stent', 'pressure',
    'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
    'tail_abnorm_obs', 'excluded_obs'
]


def _tokenize(tokenizer, texts, max_length=64):
    return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')


class BaseDRMExtractor(nn.Module):
    def forward(self, images: torch.Tensor, reports: List[str], **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ApproxREVTAFDRMExtractor(BaseDRMExtractor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, reports: List[str], **kwargs) -> torch.Tensor:
        return self.model.get_approx_disease_drm(
            images,
            kwargs['clip_memory'],
            kwargs['region_txt'],
            kwargs.get('global_txt', None),
            kwargs.get('region_image', None),
        )


class _OfficialMAVLBase(BaseDRMExtractor):
    """Wrapper following the official MAVL grounding `test.py` flow.

    Important: the official zero-shot grounding code produces image-conditioned
    observation heatmaps and does not consume free-form reports. Therefore this
    extractor uses the official MAVL initialization and DRM generation procedure,
    but the `reports` argument is ignored. This makes it an external-grounding
    replacement for the earlier approximate REVTAF maps, not a strict report-
    conditioned DRM extractor.
    """

    def __init__(
        self,
        repo_root: str,
        checkpoint: str,
        text_encoder: str,
        disease_book_path: str,
        # concept_book_path: Optional[str],
        device: str = 'cuda',
        avg_last4: bool = True,
        mode: str = 'avg',
        target_hw: Optional[int] = None,
    ):
        super().__init__()
        if not repo_root:
            raise ValueError('repo_root is required for official MAVL loading.')
        if not checkpoint:
            raise ValueError('checkpoint is required for official MAVL loading.')
        if not disease_book_path:
            raise ValueError('disease_book_path is required for official MAVL loading.')

        self.device_name = device
        self.avg_last4 = avg_last4
        self.mode = mode
        self.target_hw = target_hw

        repo_root = os.path.abspath(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from external_extractors.mavl_adapter import MAVLAdapter as MAVL  # type: ignore
        from external_extractors.tokenization_bert import BertTokenizer  # type: ignore

        tokenizer = BertTokenizer.from_pretrained(text_encoder)
        json_book = json.load(open(disease_book_path, 'r'))
        disease_book = [json_book[i] for i in MAVL_ORIGINAL_CLASS]
        disease_book_tokenizer = _tokenize(tokenizer, disease_book).to(device)

        # concepts_book_tokenizer = None
        concepts_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
        
        config = {
            'model': 'mavl',
            'text_encoder': text_encoder,
            'model_path': '../checkpoints/checkpoint_full_40.pth',
            'image_res': 224,
            'test_batch_size': 512,
            'd_model': 256,
            'base_model': 'resnet50',
            'decoder': 'cross',
            'num_queries': 75,
            'dropout': 0.1,
            'attribute_set_size': 2,
            'N': 4,
            'H': 4,
            'pretrained': True,
            'self_attention': True,
            'mode': 'avg',
            'concept_book': './data/mimic_cxr/gpt4_mimic_covidr.json'
            }
        if 'concept_book' in config:
            concepts = json.load(open(config['concept_book'], 'r'))
            concepts = {i: concepts[i] for i in MAVL_ORIGINAL_CLASS}
            concepts_book = sum(concepts.values(), [])
            concepts_book_tokenizer = _tokenize(tokenizer, concepts_book).to(device)

        # with open(args.config, "r") as f:
        #     config = yaml.load(f)
        self.model = MAVL(config, disease_book_tokenizer, concepts_book_tokenizer).to(device)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        # state_dict = checkpoint_obj['model'] if isinstance(checkpoint_obj, dict) and 'model' in checkpoint_obj else checkpoint_obj
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'temp' not in k}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        # self.load_msg = str(msg)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, reports: List[str], **kwargs) -> torch.Tensor:
        _ = reports  # ignored by official MAVL grounding path
        _, ws = self.model(images.to(self.device_name))
        if self.avg_last4:
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4.0
        else:
            ws = ws[-1]

        bsz = ws.shape[0]
        n_obs = len(MAVL_ORIGINAL_CLASS)
        n_concepts = 9
        ws = ws.view(bsz, n_obs, n_concepts, -1)[:, :, [0, 6, 7, 8]]
        if self.mode == 'max':
            ws = ws.max(2)[0]
        elif self.mode == 'avg':
            ws = ws.mean(2)
        elif self.mode == 'global':
            ws = ws[:, :, 0, :]
        else:
            raise ValueError(f'Unsupported MAVL mode: {self.mode}')

        spatial = int(ws.shape[-1] ** 0.5)
        obs_maps = ws.reshape(bsz, n_obs, spatial, spatial)
        obs_maps = F.interpolate(
            obs_maps,
            size=(images.shape[-2], images.shape[-1]) if self.target_hw is None else (self.target_hw, self.target_hw),
            mode='bilinear',
            align_corners=True,
        )

        disease_maps = []
        for disease_idx in range(13):
            obs_ids = DISEASE_TO_OBS.get(disease_idx, [])
            if not obs_ids:
                disease_maps.append(torch.zeros_like(obs_maps[:, 0]))
            else:
                disease_maps.append(obs_maps[:, obs_ids].max(dim=1).values)
        return torch.stack(disease_maps, dim=1)


class _OfficialMedKLIPBase(BaseDRMExtractor):
    """Wrapper following the same grounding script style for MedKLIP.

    This path is included for API symmetry. It requires the official repo files to
    be available under `repo_root`.
    """

    def __init__(
        self,
        repo_root: str,
        checkpoint: str,
        text_encoder: str,
        disease_book_path: str,
        device: str = 'cuda',
        avg_last4: bool = True,
        target_hw: Optional[int] = None,
    ):
        super().__init__()
        if not repo_root:
            raise ValueError('repo_root is required for official MedKLIP loading.')
        if not checkpoint:
            raise ValueError('checkpoint is required for official MedKLIP loading.')
        if not disease_book_path:
            raise ValueError('disease_book_path is required for official MedKLIP loading.')

        self.device_name = device
        self.avg_last4 = avg_last4
        self.target_hw = target_hw

        repo_root = os.path.abspath(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from external_extractors.medklip_adapter import MedKLIPAdapter as MedKLIP   # type: ignore
        from external_extractors.tokenization_bert import BertTokenizer  # type: ignore

        tokenizer = BertTokenizer.from_pretrained(text_encoder)
        json_book = json.load(open(disease_book_path, 'r'))
        disease_book = [json_book[i] for i in MAVL_ORIGINAL_CLASS]
        disease_book_tokenizer = _tokenize(tokenizer, disease_book).to(device)

        config = {
            'model': 'medklip',
            'text_encoder': text_encoder,
        }
        self.model = MedKLIP(config, disease_book_tokenizer).to(device)
        checkpoint_obj = torch.load(checkpoint, map_location='cpu')
        state_dict = checkpoint_obj['model'] if isinstance(checkpoint_obj, dict) and 'model' in checkpoint_obj else checkpoint_obj
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.load_msg = str(msg)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, reports: List[str], **kwargs) -> torch.Tensor:
        _ = reports
        _, ws = self.model(images.to(self.device_name))
        if self.avg_last4 and isinstance(ws, (list, tuple)):
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4.0
        elif isinstance(ws, (list, tuple)):
            ws = ws[-1]

        # Assume official MedKLIP returns [B, 75, HW] or [B, 75, h, w].
        if ws.dim() == 4:
            obs_maps = ws
        else:
            bsz, n_obs, hw = ws.shape
            spatial = int(hw ** 0.5)
            obs_maps = ws.reshape(bsz, n_obs, spatial, spatial)
        obs_maps = F.interpolate(
            obs_maps,
            size=(images.shape[-2], images.shape[-1]) if self.target_hw is None else (self.target_hw, self.target_hw),
            mode='bilinear',
            align_corners=True,
        )

        disease_maps = []
        for disease_idx in range(13):
            obs_ids = DISEASE_TO_OBS.get(disease_idx, [])
            if not obs_ids:
                disease_maps.append(torch.zeros_like(obs_maps[:, 0]))
            else:
                disease_maps.append(obs_maps[:, obs_ids].max(dim=1).values)
        return torch.stack(disease_maps, dim=1)


def build_drm_extractor(name: str, model, args):
    name = name.lower()
    if name == 'approx':
        return ApproxREVTAFDRMExtractor(model)
    if name == 'mavl':
        return _OfficialMAVLBase(
            repo_root=args.mavl_repo_root,
            checkpoint=args.mavl_checkpoint,
            text_encoder=args.mavl_text_encoder,
            disease_book_path=args.mavl_disease_book,
            # concept_book_path=args.mavl_concept_book,
            device=args.device,
            avg_last4=args.mavl_avg_last4,
            mode=args.mavl_mode,
            target_hw=args.image_size,
        )
    if name == 'medklip':
        return _OfficialMedKLIPBase(
            repo_root=args.medklip_repo_root,
            checkpoint=args.medklip_checkpoint,
            text_encoder=args.medklip_text_encoder,
            disease_book_path=args.medklip_disease_book,
            device=args.device,
            avg_last4=args.medklip_avg_last4,
            target_hw=args.image_size,
        )
    raise ValueError(f'Unknown DRM extractor: {name}')
