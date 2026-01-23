import torch
import torch.nn as nn
import copy
import random
import gc

class PoisonLoader:
    def __init__(self, attack_methods=None, attack_params=None):
        self.attack_methods = attack_methods or []
        self.attack_params = attack_params or {}
        
        self.valid_attacks = {
            "label_flip", "backdoor", "batch_poison",
            "random_poison", "model_compress", "gradient_inversion", "gradient_amplify", "scale_update",
            "feature_poison"
        }
        
        for method in self.attack_methods:
            if method not in self.valid_attacks:
                raise ValueError(f"不支持的投毒方式: {method}")

    def execute_attack(self, model, dataloader, model_class, device='cpu', optimizer=None, verbose=False, uid="?", log_interval=100):
        """统一的攻击执行入口"""
        if "random_poison" in self.attack_methods:
            return self._execute_random_poison(model, model_class, device)

        trained_params, grad_flat = self._standard_training_process(
            model, dataloader, model_class, device, optimizer, verbose, uid, log_interval
        )
        
        return trained_params, grad_flat

    def _standard_training_process(self, model, dataloader, model_class, device, optimizer, verbose, uid, log_interval):
        model.train()
        
        initial_model = model_class().to(device)
        initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
        initial_flat = initial_model.get_flat_params()

        criterion = nn.CrossEntropyLoss()
        local_epochs = self.attack_params.get("local_epochs", 3)

        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                data, target = self.apply_data_poison(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 打印日志
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
        initial_model = model_class().to(device)
        initial_model.load_state_dict(copy.deepcopy(model.state_dict()))
        initial_flat = initial_model.get_flat_params()

        noise_std = self.attack_params.get("noise_std", 0.5)
        grad_flat = torch.randn_like(initial_flat) * noise_std
        
        grad_flat = self.apply_gradient_poison(grad_flat)

        final_flat = initial_flat + grad_flat
        self._load_flat_params_to_model(model, final_flat)
        
        random_params = copy.deepcopy(model.state_dict())
        del initial_model
        return random_params, grad_flat

    def apply_data_poison(self, data, target):
        data_out, target_out = data, target
        if "backdoor" in self.attack_methods:
            data_out, target_out = self._poison_backdoor(data_out, target_out)
        elif "label_flip" in self.attack_methods:
            data_out, target_out = self._poison_label_flip(data_out, target_out)
        if "batch_poison" in self.attack_methods:
            data_out, target_out = self._poison_batch_poison(data_out, target_out)
        return data_out, target_out

    def apply_gradient_poison(self, grad_flat):
        grad_out = grad_flat
        if "model_compress" in self.attack_methods:
            grad_out = self._poison_model_compress(grad_out)
        if "gradient_inversion" in self.attack_methods:
            grad_out = self._poison_gradient_inversion(grad_out)
        if "gradient_amplify" in self.attack_methods or "scale_update" in self.attack_methods:
            factor = self.attack_params.get("scale_factor", self.attack_params.get("amplify_factor", 5.0))
            grad_out = grad_out * factor
        return grad_out

    def apply_feature_poison(self, feature):
        feature_out = feature
        if "feature_poison" in self.attack_methods:
            poison_strength = self.attack_params.get("poison_strength", 0.3)
            perturb_dim = self.attack_params.get("perturb_dim", random.randint(0, feature.shape[0] - 1))
            perturb = torch.zeros_like(feature).to(feature.device)
            perturb[perturb_dim] = poison_strength
            feature_out = feature + perturb
        return feature_out

    def _poison_label_flip(self, data, target):
        source_class = self.attack_params.get("source_class", 1) 
        target_class = self.attack_params.get("target_class", 7)
        mask = (target == source_class)
        if mask.sum() > 0:
            target[mask] = target_class
        return data, target

    def _poison_backdoor(self, data, target):
        backdoor_ratio = self.attack_params.get("backdoor_ratio", 0.1)
        backdoor_target = self.attack_params.get("backdoor_target", 0) 
        trigger_size = self.attack_params.get("trigger_size", 4)
        
        batch_size = len(target)
        num_backdoor = max(1, int(batch_size * backdoor_ratio))
        indices = random.sample(range(batch_size), num_backdoor)
        
        for idx in indices:
            target[idx] = backdoor_target
            if data.dim() == 4:
                data[idx, :, -trigger_size:, -trigger_size:] = data.max()
            elif data.dim() == 3:
                data[:, -trigger_size:, -trigger_size:] = data.max()
        return data, target

    def _poison_batch_poison(self, data, target):
        poison_ratio = self.attack_params.get("poison_ratio", 0.2)
        noise_std = self.attack_params.get("batch_noise_std", 0.1)
        num_poison = int(len(target) * poison_ratio)
        indices = random.sample(range(len(target)), num_poison)
        noise = torch.randn_like(data[indices]) * noise_std
        data[indices] = torch.clamp(data[indices] + noise, 0.0, 1.0)
        return data, target

    def _poison_model_compress(self, grad_flat):
        compress_ratio = self.attack_params.get("compress_ratio", 0.05)
        num_keep = int(len(grad_flat) * (1 - compress_ratio))
        if num_keep > 0:
            _, indices = torch.topk(torch.abs(grad_flat), k=num_keep, largest=True)
            mask = torch.zeros_like(grad_flat)
            mask[indices] = 1.0
            return grad_flat * mask
        return grad_flat

    def _poison_gradient_inversion(self, grad_flat):
        inversion_strength = self.attack_params.get("inversion_strength", 1.0)
        return -inversion_strength * grad_flat

    def _load_flat_params_to_model(self, model, flat_params):
        start_idx = 0
        for param in model.parameters():
            numel = param.numel()
            end_idx = start_idx + numel
            flat_slice = flat_params[start_idx:end_idx]
            param.data.copy_(flat_slice.view(param.shape))
            start_idx = end_idx