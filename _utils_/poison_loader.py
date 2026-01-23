import torch
import torch.nn as nn
import copy
import random
import gc

class PoisonLoader:
    def __init__(self, attack_methods=None, attack_params=None):
        """
        初始化投毒加载器
        :param attack_methods: 投毒方式列表，例如 ["label_flip", "backdoor"]
        :param attack_params: 投毒参数字典
        """
        self.attack_methods = attack_methods or []
        self.attack_params = attack_params or {}
        
        # 预定义支持的攻击类型，用于验证
        self.valid_attacks = {
            "label_flip", "backdoor", "batch_poison",  # 数据层
            "random_poison", "model_compress", "gradient_inversion", "gradient_amplify", "scale_update", # 梯度/参数层
            "feature_poison" # 特征层
        }
        
        for method in self.attack_methods:
            if method not in self.valid_attacks:
                raise ValueError(f"不支持的投毒方式: {method}")

    def execute_attack(self, model, dataloader, model_class, device='cpu', optimizer=None, verbose=False, uid="?", log_interval=100):
        """
        统一的攻击执行入口
        """
        if "random_poison" in self.attack_methods:
            return self._execute_random_poison(model, model_class, device)

        # 传递日志参数给训练流程
        trained_params, grad_flat = self._standard_training_process(
            model, dataloader, model_class, device, optimizer, verbose, uid, log_interval
        )
        
        return trained_params, grad_flat

    def _standard_training_process(self, model, dataloader, model_class, device, optimizer, verbose, uid, log_interval):
        """
        标准训练流程封装：
        1. 保存初始状态
        2. 数据投毒 (Data Poisoning)
        3. 正常前向/反向传播
        4. 计算伪梯度 (Pseudo-Gradient / Update)
        5. 梯度投毒 (Gradient Poisoning - Scale/Compress/Noise)
        """
        model.train()
        
        initial_model = model_class().to(device)
        initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
        initial_flat = initial_model.get_flat_params()

        criterion = nn.CrossEntropyLoss()
        local_epochs = self.attack_params.get("local_epochs", 5)

        # -------------------------------------------------
        # [新增] 训练循环日志
        # -------------------------------------------------
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                data, target = self.apply_data_poison(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 打印进度日志
                if verbose and (batch_idx % log_interval == 0):
                    print(f"    [Client {uid}] Epoch {epoch+1}/{local_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        trained_flat = model.get_flat_params()
        grad_flat = trained_flat - initial_flat

        grad_flat = self.apply_gradient_poison(grad_flat)

        final_flat = initial_flat + grad_flat
        self._load_flat_params_to_model(model, final_flat)

        del initial_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return copy.deepcopy(model.state_dict()), grad_flat


    def _execute_random_poison(self, model, model_class, device):
        """
        随机投毒：不训练，直接生成高斯噪声参数
        """
        initial_model = model_class().to(device)
        initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
        initial_flat = initial_model.get_flat_params()

        # 生成与参数形状一致的随机更新
        noise_std = self.attack_params.get("noise_std", 0.5)
        # 生成随机噪声作为"梯度"
        grad_flat = torch.randn_like(initial_flat) * noise_std
        
        # 如果需要，也可以对随机梯度应用其他处理
        grad_flat = self.apply_gradient_poison(grad_flat)

        # 应用到参数
        final_flat = initial_flat + grad_flat
        self._load_flat_params_to_model(model, final_flat)
        
        random_params = copy.deepcopy(model.state_dict())
        del initial_model
        return random_params, grad_flat

    # ------------------ Hook Methods ------------------

    def apply_data_poison(self, data, target):
        """应用所有被选中的数据层投毒方法"""
        data_out, target_out = data, target
        
        # 优先级：Backdoor > Label Flip > Batch Poison
        if "backdoor" in self.attack_methods:
            data_out, target_out = self._poison_backdoor(data_out, target_out)
        elif "label_flip" in self.attack_methods:
            data_out, target_out = self._poison_label_flip(data_out, target_out)
            
        if "batch_poison" in self.attack_methods:
            data_out, target_out = self._poison_batch_poison(data_out, target_out)
            
        return data_out, target_out

    def apply_gradient_poison(self, grad_flat):
        """应用所有被选中的梯度层投毒方法"""
        grad_out = grad_flat
        
        # 1. 模型压缩 (Sparsification)
        if "model_compress" in self.attack_methods:
            grad_out = self._poison_model_compress(grad_out)
            
        # 2. 梯度反转 (Gradient Inversion)
        if "gradient_inversion" in self.attack_methods:
            grad_out = self._poison_gradient_inversion(grad_out)
            
        # 3. 梯度放大 / Scaling (Train-and-Scale) - MESAS 关键点
        # 兼容 "gradient_amplify" 关键字
        if "gradient_amplify" in self.attack_methods or "scale_update" in self.attack_methods:
            # 优先使用 scale_factor，如果没有则默认 5.0
            factor = self.attack_params.get("scale_factor", self.attack_params.get("amplify_factor", 5.0))
            grad_out = grad_out * factor
            
        return grad_out

    def apply_feature_poison(self, feature):
        """应用特征层投毒 (LSH特征)"""
        feature_out = feature
        if "feature_poison" in self.attack_methods:
            poison_strength = self.attack_params.get("poison_strength", 0.3)
            # 随机选择维度扰动
            perturb_dim = self.attack_params.get("perturb_dim", random.randint(0, feature.shape[0] - 1))
            perturb = torch.zeros_like(feature).to(feature.device)
            perturb[perturb_dim] = poison_strength
            feature_out = feature + perturb
        return feature_out

    # ------------------ 具体攻击实现 ------------------

    def _poison_label_flip(self, data, target):
        """
        [MESAS 对齐] Source-to-Target Flip
        将源类(Source)的所有样本标签改为目标类(Target)
        """
        source_class = self.attack_params.get("source_class", 1) 
        target_class = self.attack_params.get("target_class", 7)
        
        # 找到所有源类样本并翻转
        mask = (target == source_class)
        if mask.sum() > 0:
            target[mask] = target_class
        return data, target

    def _poison_backdoor(self, data, target):
        """
        [MESAS 对齐] Pixel Trigger Backdoor
        默认 PDR=0.1，右下角方块触发器
        """
        backdoor_ratio = self.attack_params.get("backdoor_ratio", 0.1)
        backdoor_target = self.attack_params.get("backdoor_target", 0) # 攻击目标类
        trigger_size = self.attack_params.get("trigger_size", 4)
        
        batch_size = len(target)
        num_backdoor = max(1, int(batch_size * backdoor_ratio))
        
        # 随机选择样本投毒
        indices = random.sample(range(batch_size), num_backdoor)
        
        for idx in indices:
            target[idx] = backdoor_target
            # 添加 Pixel Trigger (右下角置为最大值)
            if data.dim() == 4: # [B, C, H, W]
                data[idx, :, -trigger_size:, -trigger_size:] = data.max()
            elif data.dim() == 3:
                data[:, -trigger_size:, -trigger_size:] = data.max()
                
        return data, target

    def _poison_batch_poison(self, data, target):
        """向 Batch 数据添加高斯噪声"""
        poison_ratio = self.attack_params.get("poison_ratio", 0.2)
        noise_std = self.attack_params.get("batch_noise_std", 0.1)
        num_poison = int(len(target) * poison_ratio)
        
        indices = random.sample(range(len(target)), num_poison)
        noise = torch.randn_like(data[indices]) * noise_std
        data[indices] = torch.clamp(data[indices] + noise, 0.0, 1.0)
        return data, target

    def _poison_model_compress(self, grad_flat):
        """模型压缩：将梯度绝对值较小的部分置零"""
        compress_ratio = self.attack_params.get("compress_ratio", 0.05)
        # 计算需要保留的数量
        num_keep = int(len(grad_flat) * (1 - compress_ratio))
        
        if num_keep > 0:
            _, indices = torch.topk(torch.abs(grad_flat), k=num_keep, largest=True)
            mask = torch.zeros_like(grad_flat)
            mask[indices] = 1.0
            return grad_flat * mask
        return grad_flat

    def _poison_gradient_inversion(self, grad_flat):
        """梯度反转"""
        inversion_strength = self.attack_params.get("inversion_strength", 1.0)
        return -inversion_strength * grad_flat

    # ------------------ 辅助工具 ------------------

    def _load_flat_params_to_model(self, model, flat_params):
        """辅助函数：将展平的参数填回模型"""
        start_idx = 0
        for param in model.parameters():
            numel = param.numel()
            end_idx = start_idx + numel
            flat_slice = flat_params[start_idx:end_idx]
            param.data.copy_(flat_slice.view(param.shape))
            start_idx = end_idx