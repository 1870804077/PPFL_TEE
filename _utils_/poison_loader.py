# 修改后的 poison_loader.py
import torch
import random


class PoisonLoader:
    def __init__(self, attack_methods=None, attack_params=None):
        """
        初始化投毒加载器
        :param attack_methods: 投毒方式列表，例如 ["label_flip", "backdoor"]
        :param attack_params: 投毒参数字典
        """
        self.attack_methods = attack_methods or []
        self.attack_params = attack_params or {}

        # 验证攻击方法有效性
        valid_attacks = {
            # 数据层投毒
            "label_flip", "backdoor", "batch_poison",
            # 梯度层投毒
            "random_poison", "model_compress", "gradient_inversion", "gradient_amplify",
            # 特征层投毒
            "feature_poison"
        }
        for method in self.attack_methods:
            if method not in valid_attacks:
                raise ValueError(f"不支持的投毒方式: {method}")

    def execute_attack(self, model, dataloader, model_class, device='cpu', optimizer=None):
        """
        根据攻击方法执行相应的攻击
        :param model: 当前模型
        :param dataloader: 数据加载器
        :param model_class: 模型类
        :param device: 设备
        :param optimizer: 优化器
        :return: 训练后的参数和梯度
        """
        import torch.nn as nn
        import copy
        import gc
        
        # 检查攻击方法
        if not self.attack_methods:
            # 正常训练
            return self._execute_normal_training(model, dataloader, model_class, device, optimizer)
        
        attack_methods = self.attack_methods if isinstance(self.attack_methods, list) else [self.attack_methods]
        
        # 如果包含random_poison，生成随机参数
        if "random_poison" in attack_methods:
            return self._execute_random_poison(model, model_class, device)
        
        # 如果包含model_compress，进行模型压缩攻击
        elif "model_compress" in attack_methods:
            return self._execute_model_compress(model, dataloader, model_class, device, optimizer)
        
        # 如果包含gradient_amplify，进行梯度放大攻击
        elif "gradient_amplify" in attack_methods:
            return self._execute_gradient_amplify(model, dataloader, model_class, device, optimizer)
        
        # 对于其他攻击类型（如标签翻转、后门等），正常训练但应用数据投毒
        else:
            return self._execute_data_layer_attack(model, dataloader, model_class, device, optimizer)

    def _execute_normal_training(self, model, dataloader, model_class, device, optimizer):
        """执行正常训练"""
        import torch.nn as nn
        import copy
        import gc
        
        model.train()
        initial_params = copy.deepcopy(model.state_dict())
        initial_model = model_class().to(device)
        initial_model.load_state_dict(initial_params)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):  # LOCAL_EPOCHS
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 计算梯度 - 只对浮点参数计算梯度
        initial_flat = initial_model.get_flat_params()
        trained_flat = model.get_flat_params()
        grad_flat = trained_flat - initial_flat

        # 释放内存
        del initial_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 应用梯度投毒
        grad_flat = self.apply_gradient_poison(grad_flat)

        # 返回训练后的参数和梯度
        trained_params = copy.deepcopy(model.state_dict())
        return trained_params, grad_flat

    def _execute_random_poison(self, model, model_class, device):
        """执行随机投毒攻击"""
        import torch.nn as nn
        import copy
        import gc
        
        # 生成随机模型参数（模拟随机梯度攻击）
        random_params = {}
        for key, param in model.state_dict().items():
            # 只对浮点参数添加噪声，跳过整数参数
            if param.dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                # 整数参数保持不变
                random_params[key] = param.clone()
            else:
                # 生成与原参数相同形状的随机噪声
                noise_std = self.attack_params.get("noise_std", 0.5)
                noise = torch.randn_like(param) * noise_std
                random_params[key] = param + noise  # 在原参数基础上加随机噪声
        
        # 生成对应的随机梯度（用于检测）
        initial_model = model_class().to(device)
        initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
        initial_flat = initial_model.get_flat_params()
        
        # 创建临时模型来计算随机参数的展平向量
        temp_model = model_class().to(device)
        temp_model.load_state_dict(random_params)
        random_flat = temp_model.get_flat_params()
        grad_flat = random_flat - initial_flat

        # 释放内存
        del initial_model, temp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 应用梯度投毒（进一步扰乱梯度特征）
        grad_flat = self.apply_gradient_poison(grad_flat)
        
        return random_params, grad_flat

    def _execute_model_compress(self, model, dataloader, model_class, device, optimizer):
        """执行模型压缩攻击"""
        import torch.nn as nn
        import copy
        import gc
        
        model.train()
        initial_params = copy.deepcopy(model.state_dict())
        initial_model = model_class().to(device)
        initial_model.load_state_dict(initial_params)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):  # LOCAL_EPOCHS
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                # 应用数据投毒（如果有的话）
                data, target = self.apply_data_poison(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 获取训练后的参数
        trained_params = copy.deepcopy(model.state_dict())
        
        # 应用模型压缩（将部分参数置零）- 只对浮点参数压缩
        compress_ratio = self.attack_params.get("compress_ratio", 0.05)
        for key, param in trained_params.items():
            if param.dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                # 整数参数不压缩
                continue
            else:
                # 将较小的参数置零（模拟压缩）
                flat_param = param.view(-1)
                num_zeros = int(len(flat_param) * compress_ratio)
                if num_zeros > 0:
                    _, indices = torch.topk(torch.abs(flat_param), len(flat_param) - num_zeros, largest=True)
                    mask = torch.zeros_like(flat_param)
                    mask[indices] = 1
                    compressed_param = flat_param * mask
                    trained_params[key] = compressed_param.view(param.shape)

        # 计算梯度 - 只对浮点参数计算梯度
        initial_flat = initial_model.get_flat_params()
        temp_model = model_class().to(device)
        temp_model.load_state_dict(trained_params)
        trained_flat = temp_model.get_flat_params()
        grad_flat = trained_flat - initial_flat

        # 释放内存
        del initial_model, temp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 应用梯度投毒
        grad_flat = self.apply_gradient_poison(grad_flat)

        return trained_params, grad_flat

    def _execute_gradient_amplify(self, model, dataloader, model_class, device, optimizer):
        """执行梯度放大攻击"""
        import torch.nn as nn
        import copy
        import gc
        
        model.train()
        initial_params = copy.deepcopy(model.state_dict())
        initial_model = model_class().to(device)
        initial_model.load_state_dict(initial_params)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):  # LOCAL_EPOCHS
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                # 应用数据投毒（如果有的话）
                data, target = self.apply_data_poison(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 放大梯度 - 只对浮点参数的梯度放大
                amplify_factor = self.attack_params.get("amplify_factor", 5.0)
                for param in model.parameters():
                    if param.grad is not None and param.dtype not in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                        param.grad.data *= amplify_factor
                
                optimizer.step()

        # 计算梯度 - 只对浮点参数计算梯度
        initial_flat = initial_model.get_flat_params()
        trained_flat = model.get_flat_params()
        grad_flat = trained_flat - initial_flat

        # 释放内存
        del initial_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 应用梯度投毒
        grad_flat = self.apply_gradient_poison(grad_flat)

        # 返回训练后的参数和梯度
        trained_params = copy.deepcopy(model.state_dict())
        return trained_params, grad_flat

    def _execute_data_layer_attack(self, model, dataloader, model_class, device, optimizer):
        """执行数据层攻击（如标签翻转、后门等）"""
        import torch.nn as nn
        import copy
        import gc
        
        model.train()
        initial_params = copy.deepcopy(model.state_dict())
        initial_model = model_class().to(device)
        initial_model.load_state_dict(initial_params)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):  # LOCAL_EPOCHS
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                # 应用数据投毒
                data, target = self.apply_data_poison(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 计算梯度 - 只对浮点参数计算梯度
        initial_flat = initial_model.get_flat_params()
        trained_flat = model.get_flat_params()
        grad_flat = trained_flat - initial_flat

        # 释放内存
        del initial_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 应用梯度投毒
        grad_flat = self.apply_gradient_poison(grad_flat)

        # 返回训练后的参数和梯度
        trained_params = copy.deepcopy(model.state_dict())
        return trained_params, grad_flat

    def apply_data_poison(self, data, target):
        """应用所有数据层投毒"""
        data_out, target_out = data, target
        for method in self.attack_methods:
            if method == "label_flip":
                data_out, target_out = self._poison_label_flip(data_out, target_out)
            elif method == "backdoor":
                data_out, target_out = self._poison_backdoor(data_out, target_out)
            elif method == "batch_poison":
                data_out, target_out = self._poison_batch_poison(data_out, target_out)
        return data_out, target_out

    def apply_gradient_poison(self, grad_flat):
        """应用所有梯度层投毒"""
        grad_out = grad_flat
        for method in self.attack_methods:
            if method == "random_poison":
                grad_out = self._poison_random(grad_out)
            elif method == "model_compress":
                grad_out = self._poison_model_compress(grad_out)
            elif method == "gradient_inversion":
                grad_out = self._poison_gradient_inversion(grad_out)
            elif method == "gradient_amplify":
                grad_out = self._poison_gradient_amplify(grad_out)
        return grad_out

    def apply_feature_poison(self, feature):
        """应用所有特征层投毒"""
        feature_out = feature
        for method in self.attack_methods:
            if method == "feature_poison":
                feature_out = self._poison_feature_poison(feature_out)
        return feature_out

    # 以下是原有的投毒实现方法（保持不变）
    def _poison_label_flip(self, data, target):
        flip_ratio = self.attack_params.get("flip_ratio", 0.15)
        num_flip = int(len(target) * flip_ratio)
        flip_indices = random.sample(range(len(target)), num_flip)
        for idx in flip_indices:
            target[idx] = (target[idx] + 1) % 10  # 适用于10分类问题
        return data, target

    def _poison_backdoor(self, data, target):
        backdoor_ratio = self.attack_params.get("backdoor_ratio", 0.08)
        backdoor_target = self.attack_params.get("backdoor_target", 9)
        trigger_size = self.attack_params.get("trigger_size", 2)
        num_backdoor = int(len(target) * backdoor_ratio)
        backdoor_indices = random.sample(range(len(target)), num_backdoor)
        for idx in backdoor_indices:
            data[idx, :, -trigger_size:, -trigger_size:] = 1.0
            target[idx] = backdoor_target
        return data, target

    def _poison_random(self, grad_flat):
        noise_std = self.attack_params.get("noise_std", 0.2)
        noise = torch.normal(mean=0.0, std=noise_std, size=grad_flat.shape).to(grad_flat.device)
        return grad_flat + noise

    def _poison_model_compress(self, grad_flat):
        compress_ratio = self.attack_params.get("compress_ratio", 0.05)
        num_keep = int(len(grad_flat) * compress_ratio)
        abs_grad = torch.abs(grad_flat)
        _, indices = torch.topk(abs_grad, k=num_keep, largest=True)
        compressed_grad = torch.zeros_like(grad_flat)
        compressed_grad[indices] = grad_flat[indices]
        return compressed_grad

    def _poison_gradient_inversion(self, grad_flat):
        inversion_strength = self.attack_params.get("inversion_strength", 1.0)
        return -inversion_strength * grad_flat

    def _poison_gradient_amplify(self, grad_flat):
        amplify_factor = self.attack_params.get("amplify_factor", 5.0)
        return amplify_factor * grad_flat

    def _poison_feature_poison(self, feature):
        poison_strength = self.attack_params.get("poison_strength", 0.3)
        perturb_dim = self.attack_params.get("perturb_dim", random.randint(0, feature.shape[0] - 1))
        perturb = torch.zeros_like(feature).to(feature.device)
        perturb[perturb_dim] = poison_strength
        return feature + perturb

    def _poison_batch_poison(self, data, target):
        poison_ratio = self.attack_params.get("poison_ratio", 0.2)
        noise_std = self.attack_params.get("batch_noise_std", 0.1)
        num_poison = int(len(target) * poison_ratio)
        poison_indices = random.sample(range(len(target)), num_poison)
        data[poison_indices] += torch.normal(mean=0.0, std=noise_std, size=data[poison_indices].shape).to(data.device)
        data[poison_indices] = torch.clamp(data[poison_indices], 0.0, 1.0)
        return data, target



