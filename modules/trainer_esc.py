
import os, copy
import torch
import torch.distributed as dist

from modules.metrics_clinical import CheXbertMetrics
from modules.optims import LinearWarmupCosineLRScheduler
from modules.losses_esc import RewardCriterionWithPrefix
from modules.rewards_esc import get_self_critical_reward_text, strip_prompt_from_caption
from modules.gear import GroupWiseEvidenceAlignment
from modules.spl import PreferencePredictor, SelfCorrectingPreferenceLearning
from modules.llm_refiner import build_llm_refiner


class TrainerESC(object):
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, logger,
                 train_dataloader, val_dataloader, test_dataloader, device, is_main_process):
        self.logger = logger
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)
        self.criterion_cls = criterion_cls
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns
        self.reward_criterion = RewardCriterionWithPrefix(pad_token_id=args.pad_token_id)
        self.gear = GroupWiseEvidenceAlignment()
        self.preference_predictor = PreferencePredictor(model_name=args.pref_model_name).to(device)
        self.spl = SelfCorrectingPreferenceLearning(self.preference_predictor, args.spl_tau_lower, args.spl_tau_upper, args.spl_infer_consensus_threshold)
        self.llm_refiner = build_llm_refiner(args)

        p_wd, p_non_wd = [], []
        num_parameters = 0
        for mod in [self.model, self.preference_predictor]:
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
        print(f'number of trainable parameters: {num_parameters}')
        self.optimizer = torch.optim.AdamW([
            {'params': p_wd, 'weight_decay': float(self.args.weight_decay)},
            {'params': p_non_wd, 'weight_decay': 0.0},
        ], lr=float(self.args.init_lr), weight_decay=float(self.args.weight_decay), betas=(0.9, 0.999))
        self.lr_scheduler = LinearWarmupCosineLRScheduler(self.optimizer, self.args.epochs, self.args.min_lr, self.args.init_lr,
                                                          decay_rate=None, warmup_start_lr=self.args.warmup_lr, warmup_steps=self.args.warmup_steps)
        self.epochs = self.args.epochs
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_best = 0
        self.log_best = {}
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _unwrap(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed and hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            result = self._train_epoch(epoch)
            if self.args.distributed:
                dist.barrier()
            result = self.eval_blip(epoch, result)
            log = {'epoch': epoch}
            log.update(result)
            if self.is_main_process and log[self.mnt_metric] >= self.mnt_best:
                self.mnt_best = log[self.mnt_metric]
                self.log_best = copy.deepcopy(log)
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                torch.save(self._unwrap().state_dict(), best_path)
                self.logger.info(f'Saving current best to {best_path}')
            for k, v in log.items():
                self.logger.info(f'	{k:15s}: {v}')

    def _sample_multiple_candidates(self, core_model, images, clip_memory, region_txt, global_txt, region_image, n_candidates):
        reports_all, status_all = [], []
        for _ in range(n_candidates):
            _, _, reports, _ = core_model.sample_scst(image=images, clip_memory=clip_memory, region_txt=region_txt, global_txt=global_txt,
                                                      region_image=region_image, sample_method=self.args.train_sample_method,
                                                      num_beams=self.args.train_beam_size, max_length=self.args.gen_max_len,
                                                      min_length=self.args.gen_min_len, top_p=self.args.train_top_p,
                                                      repetition_penalty=self.args.repetition_penalty, temperature=self.args.train_temperature)
            reports_all.append(reports)
            status_all.append(self.chexbert_metrics.chexbert(list(reports)))
        reports_bn = [list(items) for items in zip(*reports_all)]
        status_bnk = torch.stack(status_all, dim=1)
        return reports_bn, status_bnk

    def _extract_gear_maps(self, core_model, images, clip_memory, region_txt, global_txt, region_image, pred_reports, gt_reports):
        m_pred = core_model.extract_drms(image=images, reports=pred_reports, clip_memory=clip_memory, region_txt=region_txt,
                                         global_txt=global_txt, region_image=region_image)
        with torch.no_grad():
            m_gt = core_model.extract_drms(image=images, reports=gt_reports, clip_memory=clip_memory, region_txt=region_txt,
                                           global_txt=global_txt, region_image=region_image)
        return m_pred, m_gt

    def _refine_reports_train(self, candidate_reports, candidate_status, gt_reports, y_gt):
        refiner = self.llm_refiner if self.args.use_llm_refine_train else None
        return self.spl(candidate_reports, candidate_status[:, :, :14], gt_reports, y_gt[:, :14], self.device, llm_refiner=refiner)

    @torch.no_grad()
    def _generate_reports_for_eval(self, core_model, images, clip_memory, region_txt, region_image):
        if not self.args.use_llm_refine_infer:
            return core_model.generate(images, clip_memory, region_txt, region_image, sample=False,
                                       num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
        candidate_reports, candidate_status = self._sample_multiple_candidates(core_model, images, clip_memory, region_txt, None, region_image, self.args.spl_num_candidates)
        spl_out = self.spl.infer(candidate_reports, candidate_status[:, :, :14], self.device, llm_refiner=self.llm_refiner)
        _, cls_preds, cls_preds_logits = core_model.generate(images, clip_memory, region_txt, region_image, sample=False,
                                                             num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
        return spl_out.refined_reports, cls_preds, cls_preds_logits

    def _train_epoch(self, epoch):
        self.logger.info(f'[{epoch}/{self.epochs}] Start ESC-style training.')
        self.model.train(); self.preference_predictor.train(); core_model = self._unwrap()
        agg = {'train_loss': 0.0, 'train_ce': 0.0, 'train_rl': 0.0, 'train_gear': 0.0, 'train_spl': 0.0, 'train_reward': 0.0}
        for batch_idx, (images, captions, cls_labels, clip_memory, global_txt, region_txt, region_image) in enumerate(self.train_dataloader):
            images = images.to(self.device); cls_labels = cls_labels.to(self.device); clip_memory = clip_memory.to(self.device)
            global_txt = global_txt.to(self.device); region_txt = region_txt.to(self.device); region_image = region_image.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx); self.optimizer.zero_grad()
            loss_lm, loss_cls, loss_align, loss_rank = self.model(images, captions, cls_labels, clip_memory, global_txt, region_txt, region_image, self.criterion_cls, self.base_probs)
            task_loss = loss_lm + self.args.cls_weight * loss_cls + self.args.align_weight * loss_align + self.args.rank_weight * loss_rank
            with torch.no_grad():
                _, _, greedy_reports, _ = core_model.sample_scst(image=images, clip_memory=clip_memory, region_txt=region_txt, global_txt=global_txt,
                                                                region_image=region_image, sample_method='greedy', num_beams=self.args.sc_baseline_beam_size,
                                                                max_length=self.args.gen_max_len, min_length=self.args.gen_min_len,
                                                                repetition_penalty=self.args.repetition_penalty)
            sample_seq, sample_token_logprobs, sample_reports, _ = core_model.sample_scst(image=images, clip_memory=clip_memory, region_txt=region_txt,
                                                                                           global_txt=global_txt, region_image=region_image,
                                                                                           sample_method=self.args.train_sample_method, num_beams=self.args.train_beam_size,
                                                                                           max_length=self.args.gen_max_len, min_length=self.args.gen_min_len,
                                                                                           top_p=self.args.train_top_p, repetition_penalty=self.args.repetition_penalty,
                                                                                           temperature=self.args.train_temperature)
            gt_reports = [strip_prompt_from_caption(c, num_prompt_tokens=18) for c in captions]
            reward = torch.from_numpy(get_self_critical_reward_text(greedy_reports, gt_reports, sample_reports)).float().to(self.device)
            reward_per_token = reward.unsqueeze(1).expand_as(sample_token_logprobs)
            loss_rl = self.reward_criterion(sample_token_logprobs, sample_seq, reward_per_token, ignore_prefix_len=core_model.prompt_length)
            candidate_reports, candidate_status = self._sample_multiple_candidates(core_model, images, clip_memory, region_txt, global_txt, region_image, self.args.spl_num_candidates)
            y_gt = self.chexbert_metrics.chexbert(list(gt_reports)).to(self.device)
            spl_out = self._refine_reports_train(candidate_reports, candidate_status, gt_reports, y_gt)
            loss_spl = spl_out.loss_pref
            pred_reports_for_gear = spl_out.refined_reports if self.args.use_llm_refine_train else sample_reports
            y_pred = self.chexbert_metrics.chexbert(list(pred_reports_for_gear)).to(self.device)
            m_pred, m_gt = self._extract_gear_maps(core_model, images, clip_memory, region_txt, global_txt, region_image, pred_reports_for_gear, gt_reports)
            gear_raw, gear_stats = self.gear(m_pred, m_gt, y_pred[:, :13], y_gt[:, :13])
            loss_gear = -gear_raw
            if epoch < self.args.rl_start_epoch:
                total_loss = task_loss + self.args.spl_weight * loss_spl
            else:
                total_loss = task_loss + self.args.scst_weight * loss_rl + self.args.gear_weight * loss_gear + self.args.spl_weight * loss_spl
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(list(self.model.parameters()) + list(self.preference_predictor.parameters()), 0.1)
            self.optimizer.step()
            agg['train_loss'] += total_loss.item(); agg['train_ce'] += task_loss.item(); agg['train_rl'] += float(loss_rl.item())
            agg['train_gear'] += float(loss_gear.item()); agg['train_spl'] += float(loss_spl.item()); agg['train_reward'] += float(reward.mean().item())
            if batch_idx % 10 == 0:
                self.logger.info(f"{batch_idx}/{len(self.train_dataloader)} loss={total_loss.item():.6f} task={task_loss.item():.6f} rl={loss_rl.item():.6f} gear={loss_gear.item():.6f} spl={loss_spl.item():.6f} reward={reward.mean().item():.6f} drm={self.args.drm_extractor} llm_train={self.args.use_llm_refine_train} tp/fn/fp=({gear_stats['gear_tp_count']},{gear_stats['gear_fn_count']},{gear_stats['gear_fp_count']})")
        num_steps = len(self.train_dataloader)
        return {k: v / num_steps for k, v in agg.items()}

    def eval_blip(self, epoch, log):
        core_model = self._unwrap(); core_model.eval(); self.preference_predictor.eval()
        self.logger.info(f'[{epoch}/{self.epochs}] Start to evaluate in the validation set.')
        logits, counts = [], []
        with torch.no_grad():
            val_gts, val_res = [], []
            for images, captions, cls_labels, clip_memory, region_txt, region_image in self.val_dataloader:
                images = images.to(self.device); cls_labels = cls_labels.to(self.device); clip_memory = clip_memory.to(self.device)
                region_txt = region_txt.to(self.device); region_image = region_image.to(self.device)
                reports, cls_preds, cls_preds_logits = self._generate_reports_for_eval(core_model, images, clip_memory, region_txt, region_image)
                cls_labels_bin = (cls_labels == 1).float()
                logits.append((cls_preds_logits * cls_labels_bin).cpu().numpy()); counts.append(cls_labels_bin.cpu().numpy())
                val_res.extend(reports); val_gts.extend(captions)
        val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)}, {i: [re] for i, re in enumerate(val_res)})
        log.update(**{'val_' + k: v for k, v in val_met.items()})
        return log
