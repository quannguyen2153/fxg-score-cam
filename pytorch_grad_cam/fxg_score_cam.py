from typing import List
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class FXGScoreCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        if not isinstance(target_layers, list) or len(target_layers) <= 0:
            def layer_with_2D_bias(layer):
                bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                if type(layer) in bias_target_layers and layer.bias is not None:
                    return True
                return False
            
            target_layers = find_layer_predicate_recursive(model, layer_with_2D_bias)
            
            print(f"INFO: {len(target_layers)} layers will be accounted for.")

        super(FXGScoreCAM, self).__init__(model=model, target_layers=target_layers, reshape_transform=reshape_transform, compute_input_gradient=True)

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        self.activations_and_grads.release() # Release hooks to avoid accumulating memory size when computing

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
        
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: np.ndarray,
        layer_grads: np.ndarray,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        eps = 1e-8

        with torch.no_grad():
            # Compute pixel_weights
            sum_activations = np.sum(activations, axis=(2, 3))
            pixel_weights = torch.nn.Softmax(dim=-1)(torch.from_numpy(layer_grads * activations / (sum_activations[:, :, None, None] + eps))).to(self.device)
            
            # Highlight important pixels in each activation
            modified_activations = pixel_weights * torch.from_numpy(activations).to(self.device)

            # Upsample activations
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            activation_feature_maps = upsample(modified_activations)

            # Compute input feature map
            target_size = self.get_target_width_height(input_tensor)
            input_grad = input_tensor.grad.data.cpu().numpy()
            gradient_multiplied_input = input_grad * input_tensor.data.cpu().numpy()
            gradient_multiplied_input = np.abs(gradient_multiplied_input)
            input_feature_map = torch.from_numpy(scale_accross_batch_and_channels(gradient_multiplied_input, target_size)).to(self.device)

            # Extract biases
            try:
                if isinstance(target_layer, torch.nn.BatchNorm2d):
                    biases = - (target_layer.running_mean * target_layer.weight / torch.sqrt(target_layer.running_var + target_layer.eps)) + target_layer.bias
                else:
                    biases = target_layer.bias
                    
                biases = biases[None, :, None, None]

            except:
                # If the layer doesn't have bias
                biases = torch.zeros(activations.shape, device=self.device)

            # Compute bias feature maps
            gradient_multiplied_biases = np.abs(biases.cpu().numpy() * layer_grads)
            bias_feature_maps = torch.from_numpy(scale_accross_batch_and_channels(gradient_multiplied_biases, target_size)).to(self.device)

            # Concantenate feature maps
            feature_maps = torch.cat([activation_feature_maps, input_feature_map, bias_feature_maps], dim=1)

            # Normalize feature maps
            maxs = feature_maps.view(feature_maps.size(0),
                                    feature_maps.size(1), -1).max(dim=-1)[0]
            mins = feature_maps.view(feature_maps.size(0),
                                    feature_maps.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            feature_maps = (feature_maps - mins) / (maxs - mins + eps)

            # Pertubate input with feature maps
            pertubated_inputs = input_tensor[:, None, :, :] * feature_maps[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            # Calculate score of each pertubated inputs
            scores = []
            for target, tensor in zip(targets, pertubated_inputs):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).to(self.device).item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(pertubated_inputs.shape[0], pertubated_inputs.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()

            # Aggregate activations to saliency map
            feature_maps = feature_maps.cpu().numpy()

            # 2D conv
            if len(feature_maps.shape) == 4:
                weighted_activations = weights[:, :, None, None] * feature_maps
            # 3D conv
            elif len(feature_maps.shape) == 5:
                weighted_activations = weights[:, :, None, None, None] * feature_maps
            else:
                raise ValueError(f"Invalid activation shape. Get {len(feature_maps.shape)}.")

            if eigen_smooth:
                cam = get_2d_projection(weighted_activations)
            else:
                cam = weighted_activations.sum(axis=1)
            return cam