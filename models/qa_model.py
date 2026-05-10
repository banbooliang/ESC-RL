from openai import OpenAI, AsyncOpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import torch
from typing import List, Dict, Union
import numpy as np
import torch.nn as nn
from medklip.model_MedKLIP import MedKLIP
from transformers import BertTokenizer, AutoModel 
from models.prompt import PROMPT_TEMPLATE_QA
import ast
import asyncio


class QAModel:
    def __init__(self, model_name="openai/gpt-4o-mini", max_concurrent=6):
        self.model_name = model_name
        self.classes = ['enlarged cardiomediastinum', 'cardiomegaly', 'lung opacity',
                'lung lesion', 'edema', 'consolidation',
                'pneumonia', 'atelectasis', 'pneumothorax',
                'pleural effusion', 'pleural other','fracture',
                'support devices', 'no finding']
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
            )
        self.max_concurrent = max_concurrent 
    
    # Consistency Loss Function
    def response_consistency_loss(self, policy_reports: List[str], ref_reports: List[str]) -> torch.Tensor:
        """
        计算 policy_reports 与 ref_reports 的语义一致性损失。
        支持并行调用。
        """
        assert len(policy_reports) == len(ref_reports)
        batch_size = len(policy_reports)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_consistency_scores = []

        # ========== 并行执行任务 ==========
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for policy_report, ref_report in zip(policy_reports, ref_reports):
                # 双向比较
                futures.append(executor.submit(self._ask_model, policy_report, ref_report, direction="A→B"))
                futures.append(executor.submit(self._ask_model, ref_report, policy_report, direction="B→A"))

            # 收集结果（每两次为一对）
            pair_results = []
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                pair_results.append(res)

            # 组对计算一致性
            for i in range(0, len(pair_results), 2):
                match_1 = pair_results[i]
                match_2 = pair_results[i + 1] if i + 1 < len(pair_results) else []

                if match_1 and match_2:
                    c1 = sum(match_1) / len(match_1)
                    c2 = sum(match_2) / len(match_2)
                    avg = (c1 + c2) / 2
                else:
                    avg = 0.0
                total_consistency_scores.append(avg)

        mean_consistency = sum(total_consistency_scores) / batch_size
        loss = 1.0 - mean_consistency

        return torch.tensor(loss, dtype=torch.float32, device=device)

    def extract_content(self, response: str):
        
        codeblock_pattern = r"```(?:json)?\s*(.*?)```"
        m = re.search(codeblock_pattern, response, re.DOTALL)
        if m:
            json_text = m.group(1).strip()
        else:
            json_text = response.strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_text)
            except Exception:
                print("message无法解析结果。\n")
                return []
        try:
            status_list = [int(item["status"]) for item in data]
        except Exception:
            print("数据结构不符合预期（缺少 'status' 字段）。\n")
            return []

        return status_list
        
    def global_predict(self, reports):
         return asyncio.run(self._global_predict_async(reports))
     
    async def _global_predict_async(self, reports):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_answer(report: str):
            async with semaphore:
                messages = [
                        {
                            "role": "system",
                            "content": PROMPT_TEMPLATE_QA
                        },
                        {
                            "role": "user",
                            "content": report  
                        }
                    ]
            
                try:
                    completion = await self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=messages,
                                    temperature=0  # 固定输出
                                )
            
                    msg = completion.choices[0].message
                    if isinstance(msg.content, str):
                        response = msg.content
                    else:
                        # 如果是分段，就把 text 拼起来
                        response = "".join(
                            (part.text or "") for part in msg.content
                        )
                    answer = self.extract_content(response)
                    return answer
                
                except Exception as e:
                    return str(e)
                
        tasks = [fetch_answer(r) for r in reports]
            
        results = await asyncio.gather(*tasks)

        return results
        


class TextToImageAligner:
    def __init__(self, args):
        
        self.original_class = [
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
            'support devices', #
            'no finding',  #
        ]
        json_book = json.load(open(args.disease_dbook,'r'))
        disease_book = [json_book[i] for i in json_book]
        ana_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
                'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
                'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
                'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
                'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
                'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        medklip_tokenizer = BertTokenizer.from_pretrained(args.medklip_text_decoder)
        self.ana_book_tokenizer = self.get_tokenizer(medklip_tokenizer, ana_book).to(device)
        self.disease_book_tokenizer = self.get_tokenizer(medklip_tokenizer, disease_book).to(device)
        
        print("Creating vison-language pretrained model (MedKLIP)")
        
        model = MedKLIP(args, self.ana_book_tokenizer, self.disease_book_tokenizer, mode = 'train').to(device)
        checkpoint = torch.load(args.MEDKLIP_checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False) 
        self.model = model
        print('load vison-language pretrained checkpoint (MedKLIP) from %s'%args.MEDKLIP_checkpoint)
        
        # self.adaptor = Adaptor(in_dim=196, hidden_dim=196, out_dim=196).to(self.device)
        # 冻结 pretrained model 参数
        for p in self.model.parameters():
            p.requires_grad = False
        self.map_index = {0: 17, 1: 24, 2: 9, 3: 65, 4: 11, 5: 14, 6:20,7: 12, 8: 10, 9: 8, 10: 16, 11: 25, 12: 32}

    def get_tokenizer(self, tokenizer, target_text):
        target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
        return target_tokenizer

    def get_activation_map(self, images, labels):
        """
        images: torch.Tensor, [B, C, H, W] (on device)
        labels: torch.Tensor, - [B, C]
        
        return:
            one_hot_map: torch.BoolTensor [B, C, H_up, W_up]
           
        NOTE: outputs are on self.device (bool dtype).
        """
    
        with torch.no_grad():  
            _, ws= self.model(images, labels, is_train = False) #batch_size,batch_size,image_patch,text_patch
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4 #batch_size, 75, channel
            
        # adapted = self.adaptor(ws)
        ws = ws.reshape(ws.shape[0], ws.shape[1], 14, 14)
        map_list = torch.tensor([self.map_index[i] for i in range(len(self.map_index))])
        ws_mapped = ws[:, map_list, :, :]  # [B, C-1, H, W]
        
        ## find the label exist is original value, not exist is zero.
        assert labels.dim() == 2, "labels' shape must be [B, C-1]"
        # labels = torch.where(labels == 1, 1, 0)
        
        # mask = labels.unsqueeze(-1).unsqueeze(-1)      # [B, C-1, 1, 1]
        # ws = ws_mapped * mask                                  
        pred_map = ws_mapped.detach().cpu().numpy() 
        ## 插值
        pred_map = torch.from_numpy(pred_map.repeat(16, axis=2).repeat(16, axis=3)).to(self.device) #Final Grounding Heatmap
        one_hot_map = torch.where(pred_map > 0.005, 1, 0)
        # for m in one_hot_map:
        #     print(torch.sum(m))
        return one_hot_map

    def dice_score(self, pred, gt, eps=1e-6):
        """计算DICE系数"""
        intersection = torch.sum(pred * gt)
        return (2 * intersection + eps) / (torch.sum(pred) + torch.sum(gt) + eps)

    def compute_alignment_loss(self, pred_labels, gt_label, img):
        """
        pred_label: 预测的label list (B,C)
        gt_label: 真实的label tensor(B,C)
        """
        
        pred_labels = np.array(pred_labels)
        gt_label = gt_label.cpu().numpy()
        K = pred_labels.shape[-1]
        pred_labels = pred_labels[:,:K-1]
        gt_label = gt_label[:, :K-1]

        # class-level responses: [B, C, H_up, W_up]
        pred_responses = self.get_activation_map(img, torch.tensor(pred_labels, device=self.device)) # [b,c-1,h,w]
        gt_responses = self.get_activation_map(img, torch.tensor(gt_label, device=self.device)) # [b,c-1,h,w]

        # determine categories
        false_neg = np.where((gt_label == 1) & (pred_labels != 1))  # False Negative 漏检
        false_pos = np.where((gt_label != 1) & (pred_labels == 1))  # False Positive 误检
        true_pos = np.where((gt_label == 1) & (pred_labels == 1))   # True Positive 正确检测

        dice_results = {"missed": [], "false": [], "correct": []}

        for b_idx, c_idx in zip(false_neg[0], false_neg[1]):
            # pred labels没检测出来，约束该前景越大越好
           
            sum_pred = pred_responses[b_idx, c_idx, ...].float().mean()
            dice_results["missed"].append(sum_pred)

        for b_idx, c_idx in zip(false_pos[0], false_pos[1]):
            # pred labels不应该被检测出来，约束该前景越小越好
            pri_g = pred_responses[b_idx, c_idx, ...]
            sum_pred = pred_responses[b_idx, c_idx, ...].float().mean()
            dice_results["false"].append(sum_pred)

        for b_idx, c_idx in zip(true_pos[0], true_pos[1]):
            cls_vec = gt_responses[b_idx, c_idx,...]
            pred_vec = pred_responses[b_idx, c_idx, ...]
            dice = self.dice_score(pred_vec, cls_vec)
            dice_results["correct"].append(dice)

        # 计算整体loss（可根据权重调整）
        # missed_loss = torch.stack(dice_results["missed"]).mean()
        # false_loss = torch.stack(dice_results["false"]).mean()
        # correct_loss = 1 - torch.stack(dice_results["correct"]).mean()
        missed_loss  = safe_stack_mean(dice_results["missed"],  default=0.0, device=self.device)
        false_loss   = safe_stack_mean(dice_results["false"],   default=0.0, device=self.device)
        # correct_loss 是 1 - mean(correct)，如果根本没有 correct 样本，我们通常不惩罚，设为 0
        correct_loss = safe_stack_mean(dice_results["correct"], default=1.0, device=self.device)

        total_loss = -missed_loss + false_loss + correct_loss
        # print("Alignment Loss - Missed: {}, False: {}, Correct: {}".format(-missed_loss.item(), false_loss.item(), correct_loss.item()))
        # print("Total Alignment Loss: {}".format(total_loss.item()))
        return total_loss
        

def safe_stack_mean(tensor_list, default, device):
    if len(tensor_list) == 0:
        return torch.tensor(default, device=device, dtype=torch.float32)
    return torch.stack(tensor_list).mean()

     

class Adaptor(nn.Module):
    """1x1 卷积适配模块，用于特征维度匹配"""
    def __init__(self, in_dim: int, hidden_dim: int = None, out_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

        