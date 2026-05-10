
import torch
from modules.metrics_clinical import CheXbertMetrics
from modules.spl import PreferencePredictor, SelfCorrectingPreferenceLearning
from modules.llm_refiner import build_llm_refiner

class TesterESC(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, logger, device, test_dataloader):
        self.logger = logger
        self.args = args
        self.model = model
        self.device = device
        self.test_dataloader = test_dataloader
        self.metric_ftns = metric_ftns
        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)
        self.preference_predictor = PreferencePredictor(model_name=args.pref_model_name).to(device)
        self.spl = SelfCorrectingPreferenceLearning(self.preference_predictor, args.spl_tau_lower, args.spl_tau_upper, args.spl_infer_consensus_threshold)
        self.llm_refiner = build_llm_refiner(args)

    def _unwrap(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    @torch.no_grad()
    def _sample_multiple_candidates(self, core_model, images, clip_memory, region_txt, region_image, n_candidates):
        reports_all, status_all = [], []
        for _ in range(n_candidates):
            _, _, reports, _ = core_model.sample_scst(image=images, clip_memory=clip_memory, region_txt=region_txt, region_image=region_image,
                                                      sample_method=self.args.train_sample_method, num_beams=self.args.train_beam_size,
                                                      max_length=self.args.gen_max_len, min_length=self.args.gen_min_len,
                                                      top_p=self.args.train_top_p, repetition_penalty=self.args.repetition_penalty,
                                                      temperature=self.args.train_temperature)
            reports_all.append(reports)
            status_all.append(self.chexbert_metrics.chexbert(list(reports)))
        reports_bn = [list(items) for items in zip(*reports_all)]
        status_bnk = torch.stack(status_all, dim=1)
        return reports_bn, status_bnk

    @torch.no_grad()
    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        core_model = self._unwrap(); core_model.eval(); self.preference_predictor.eval()
        log = {}
        test_gts, test_res = [], []
        for batch_idx, (images, captions, cls_labels, clip_memory, region_txt, region_image) in enumerate(self.test_dataloader):
            images = images.to(self.device); clip_memory = clip_memory.to(self.device); region_txt = region_txt.to(self.device); region_image = region_image.to(self.device)
            if self.args.use_llm_refine_infer:
                candidate_reports, candidate_status = self._sample_multiple_candidates(core_model, images, clip_memory, region_txt, region_image, self.args.spl_num_candidates)
                reports = self.spl.infer(candidate_reports, candidate_status[:, :, :14], self.device, llm_refiner=self.llm_refiner).refined_reports
            else:
                reports, _, _ = core_model.generate(images, clip_memory, region_txt, region_image, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
            test_res.extend(reports); test_gts.extend(captions)
            if batch_idx % 10 == 0:
                self.logger.info(f'{batch_idx}/{len(self.test_dataloader)}')
        test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, {i: [re] for i, re in enumerate(test_res)})
        test_ce = self.chexbert_metrics.compute(test_gts, test_res)
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        log.update(**{'test_' + k: v for k, v in test_ce.items()})
        return log
