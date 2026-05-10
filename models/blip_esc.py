import os
import numpy as np
import torch
import torch.nn.functional as F

from hyptorch.pmath import dist_matrix
from models.blip import BLIP_Decoder
from modules.drm_extractors import build_drm_extractor

SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']

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


class BLIPDecoderESC(BLIP_Decoder):
    def __init__(self, args, device, tokenizer, **kwargs):
        super().__init__(args, device, tokenizer, **kwargs)
        self.args = args
        self.drm_extractor = build_drm_extractor(args.drm_extractor, self, args)

    def _build_prompt_from_cls(self, avg_embeds):
        cls_logits = self.cls_head(avg_embeds).view(-1, 4, 18)
        cls_probs = F.softmax(cls_logits, dim=1)
        cls_preds_logits = cls_probs[:, 1, :14]
        cls_preds = torch.argmax(cls_probs, dim=1).cpu().tolist()

        prompts = [' '.join(SCORES[c] for c in pred) + ' ' for pred in cls_preds]
        text = self.tokenizer(prompts, return_tensors='pt', padding='longest')
        input_ids = text.input_ids.to(avg_embeds.device)
        attn_masks = text.attention_mask.to(avg_embeds.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        attn_masks = attn_masks[:, :-1]
        return prompts, input_ids, attn_masks, cls_preds, cls_preds_logits

    def _retrieve_global_txt_from_region_image(self, region_image, batch_size):
        h_embedding_region = self.gwd.encode(region_image)
        h_embedding_ref = self.gwd.encode(self.r_i_score)
        hyper_dist = dist_matrix(h_embedding_region, h_embedding_ref, c=self.c)
        index_select = hyper_dist.argmin(dim=-1)
        global_txt_embeddings = []
        for b in range(batch_size):
            t_path = os.path.join(
                './data/mimic_cxr',
                'medclip_txt_embeddings',
                self.annotation[index_select[b]]['image_path'][0]
            ).replace('.jpg', '.npy')
            global_txt_embeddings.append(torch.from_numpy(np.load(t_path)).to(dtype=torch.float32))
        return torch.stack(global_txt_embeddings, dim=0).to(self.device)

    def prepare_generation_inputs(self, image, clip_memory, region_txt, global_txt=None, region_image=None):
        image_embeds, avg_embeds = self.visual_encoder(image)
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)

        prompts, input_ids, attn_masks, cls_preds, cls_preds_logits = self._build_prompt_from_cls(avg_embeds)

        if global_txt is None:
            if region_image is None:
                raise ValueError('Either global_txt or region_image must be provided.')
            global_txt = self._retrieve_global_txt_from_region_image(region_image, image.shape[0])
        else:
            global_txt = global_txt.to(image.device)

        region_txt_proj = self.ot_txt_proj1(region_txt)
        enhance_region_embed, region_map_75 = self.ot_cross_attention(image_embeds, region_txt_proj)
        global_txt_proj = self.ot_txt_proj2(global_txt)
        enhance_global_embed, global_map = self.ot_cross_attention(image_embeds, global_txt_proj)
        encoder_hidden_states = torch.cat([enhance_global_embed, enhance_region_embed], dim=-1)
        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long, device=image.device)

        return {
            'prompts': prompts,
            'input_ids': input_ids,
            'attn_masks': attn_masks,
            'cls_preds': cls_preds,
            'cls_preds_logits': cls_preds_logits,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask,
            'region_map_75': region_map_75,
            'global_map': global_map,
        }

    def get_approx_disease_drm(self, image, clip_memory, region_txt, global_txt=None, region_image=None):
        packed = self.prepare_generation_inputs(image, clip_memory, region_txt, global_txt, region_image)
        region_map_75 = packed['region_map_75']
        disease_maps = []
        for disease_idx in range(13):
            obs_ids = DISEASE_TO_OBS.get(disease_idx, [])
            if len(obs_ids) == 0:
                disease_maps.append(torch.zeros_like(region_map_75[:, 0]))
            else:
                disease_maps.append(region_map_75[:, obs_ids].max(dim=1).values)
        return torch.stack(disease_maps, dim=1)

    def extract_drms(self, image, reports, clip_memory, region_txt, global_txt=None, region_image=None):
        """Unified DRM interface.

        - approx: uses internal REVTAF maps (reports ignored)
        - mavl / medklip: uses report-conditioned external adapter
        """
        return self.drm_extractor(
            images=image,
            reports=reports,
            clip_memory=clip_memory,
            region_txt=region_txt,
            global_txt=global_txt,
            region_image=region_image,
        )

    def decode_report_only(self, sequences, prompts):
        reports = []
        for i, seq in enumerate(sequences):
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            prompt = prompts[i]
            if text.startswith(prompt):
                text = text[len(prompt):]
            reports.append(text.strip())
        return reports

    def sample_scst(
        self,
        image,
        clip_memory,
        region_txt,
        global_txt=None,
        region_image=None,
        sample_method='greedy',
        num_beams=1,
        max_length=100,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        temperature=1.0,
    ):
        packed = self.prepare_generation_inputs(image, clip_memory, region_txt, global_txt, region_image)
        do_sample = sample_method.lower() != 'greedy'

        outputs = self.text_decoder.generate(
            input_ids=packed['input_ids'],
            attention_mask=packed['attn_masks'],
            min_length=min_length,
            max_new_tokens=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            encoder_hidden_states=packed['encoder_hidden_states'],
            encoder_attention_mask=packed['encoder_attention_mask'],
        )

        decoder_input_ids = outputs[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).long()
        decoder_out = self.text_decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=packed['encoder_hidden_states'],
            encoder_attention_mask=packed['encoder_attention_mask'],
            return_dict=True,
        )
        log_probs = F.log_softmax(decoder_out.logits, dim=-1)
        token_logprobs = log_probs.gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)
        reports = self.decode_report_only(outputs, packed['prompts'])
        return outputs, token_logprobs, reports, packed


def blip_decoder_esc(args, device, tokenizer, **kwargs):
    return BLIPDecoderESC(args, device, tokenizer, **kwargs)
