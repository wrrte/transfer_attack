import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

####################################################################################################
# [ALL] preparing the current input tensor for gradient computation
def prepare_attack_input(x_adv):
    x_adv = x_adv.detach()
    x_adv.requires_grad = True
    return x_adv

# [ALL] generating the base gradient(ghat) for the current step
def calculate_attack_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                          attack_type, di_prob, di_pad_amount, di_pad_value):
    
    si_ghat = apply_si(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                       attack_type, di_prob, di_pad_amount, di_pad_value)
    if si_ghat is not None:
        return si_ghat

    attack_input = apply_di(x_adv_or_nes, attack_type, di_prob, di_pad_amount, di_pad_value)
    return calculate_loss_gradient(model, attack_input, x_adv_or_nes, y, target_label)

# [ALL] computes CE loss from given model_input, performs backpropagation, and returns the gradient tensor with respect to grad_input
def calculate_loss_gradient(model, model_input, grad_input, y, target_label):
    output = model(model_input)
    loss = nn.CrossEntropyLoss()(output, y)
    if target_label >= 0:
        loss = -loss
    return torch.autograd.grad(loss, grad_input, retain_graph=False, create_graph=False)[0]
####################################################################################################


####################################################################################################
# [MI] configures MI only when M is provided, and sets decay rate via the mu parameter
def apply_mi(attack_type, mu):
    if "M" not in attack_type:
        return 0
    return mu

# [MI] accumulates the current gradient (ghat) into momentum (g)
def update_mi_momentum(g, ghat, mu):
    # TODO 1: Momentum Iterative
    normalized_ghat = ghat / torch.sum(torch.abs(ghat), dim=[1, 2, 3], keepdim=True)
    return mu * g + normalized_ghat
####################################################################################################


####################################################################################################
# [DI] configures DI only when D is provided
def apply_di(x_adv, attack_type, di_prob, di_pad_amount, di_pad_value):
    if 'D' in attack_type:
        return diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value)
    return x_adv

# [DI] Implementing diverse input (resize & padding) 
def diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value):
    # TODO 2: Diverse Input
    x_di = x_adv
    ori_size = x_di.shape[-1]
    
    rnd = int(torch.rand(1) * di_pad_amount) + ori_size
    x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
    
    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1) * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1) * pad_max)
    pad_bottom = pad_max - pad_top
    
    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
    x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
    
    cond = torch.rand(x_adv.shape[0], device=x_adv.device) < di_prob
    cond = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_di = torch.where(cond, x_di, x_adv)
    return x_di
####################################################################################################


####################################################################################################
# [TI] configures TI only when T is provided 
# (FYI. ti_conv is a smoothed gradient generated via the create_ti_conv function.)
def apply_ti(ghat, attack_type, ti_conv):
    if 'T' in attack_type:
        return ti_conv(ghat)
    return ghat

# [TI] creating Gaussian kernel
def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    # TODO 3: Gaussian Kernel for TI
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

# [TI] preparing depthwise convolution
def create_ti_conv(device, ti_kernel_size):
    kernel = gkern(ti_kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    ti_conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(ti_kernel_size, ti_kernel_size),
                              padding=ti_kernel_size // 2, groups=3, bias=False)
    with torch.no_grad():
        ti_conv.weight = nn.Parameter(torch.from_numpy(stack_kernel).float().to(device))
        ti_conv.requires_grad_(False)
    return ti_conv.to(device)
####################################################################################################


####################################################################################################
# [SI] configures SI only when S is provided
def apply_si(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type, di_prob, di_pad_amount,
             di_pad_value):
    if 'S' in attack_type:
        return calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                                 attack_type, di_prob, di_pad_amount, di_pad_value)
    return None

# [SI] accumulates gradients across multi-scale inputs (SI), with optional Diverse Input (DI) support via apply_di
def calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                      attack_type, di_prob, di_pad_amount, di_pad_value):
    # TODO 4: Scale-Invariant FGSM
    grad_sum = 0
    for si_counter in range(number_of_si_scales):
        si_div = 2 ** si_counter
        si_input = x_adv_or_nes / si_div
        si_input = si_input.detach()
        si_input.requires_grad = True
        
        si_input2 = apply_di(si_input, attack_type, di_prob, di_pad_amount, di_pad_value)
        
        grad = calculate_loss_gradient(model, si_input2, si_input, y, target_label)
        grad_sum += grad / si_div
        
    return grad_sum
####################################################################################################


####################################################################################################
# [NI] configures NI only when N is provided, and prepares the look-ahead input tensor
def apply_ni(attack_type, x_adv, alpha, mu, g):
    # TODO 5: Nesterov Iterative Look-ahead
    if 'N' in attack_type:
        x_nes = x_adv + alpha * mu * g
        return prepare_attack_input(x_nes)
    return prepare_attack_input(x_adv)

def apply_ni_decay(attack_type, mu):
    if "N" not in attack_type:
        return 0
    return mu
####################################################################################################


####################################################################################################
# TODO 6 (recommend): Tune `mu`, `number_of_si_scales`, `di_prob`, `di_pad_amount`, `di_pad_value`, and `ti_kernel_size`.
# Do not change : num_iter, max_epsilon, step_size, target_label, constraint_img, and every_step_controller.
def mi_ditisi_fgsm_core(attack_type, model, x, y, target_label=-1, num_iter=100, max_epsilon=8, step_size=0.7,
                        mu=1.0, number_of_si_scales=5, constraint_img=None, di_prob=0.5, di_pad_amount=31,
                        di_pad_value=0, ti_kernel_size=15, every_step_controller=None):
    """
    Args:
        attack_type: string containing optional flags such as
            'M'(momentum), 'N'(Nesterov momentum), 'D'(diverse-input),
            'S'(scale-invariance), 'T'(translation-invariance)
        model: torch model with respect to which attacks will be computed
        x: batch of torch images. in range [-1.0, 1.0]
        target_label : used for targeted attack. For untargeted attack, set this to -1.
        y: true labels corresponding to the batch of images
        num_iter: T. number of iterations of ifgsm to perform.
        max_epsilon: Linf norm of resulting perturbation (in pixels)
        step_size: step size (in pixels)
        mu: decay of momentum. If mu==0, then momentum is not used.
        number_of_si_scales: m. (in scale-invariance paper)
        constraint_img: used only for special cases. This image becomes the mid-point of Lp bound.
        di_prob: p (in diverse-input paper). Probability of applying diverse-input method.
        di_pad_amount: (in diverse-input paper) Image will be enlarged by di_pad pixels in width and height.
        di_pad_value: (in diverse-input paper) Value used for padding. Note: Padding will be done after input transform.
        ti_kernel_size: (in translation-invariance paper) k. The kernel will be a k by k Gaussian kernel.

    Returns:
        The batch of adversarial examples corresponding to the original images
    """

    mu = max(apply_mi(attack_type, mu), apply_ni_decay(attack_type, mu))

    if target_label >= 0:
        y = target_label

    ti_conv = None
    if 'T' in attack_type:  # Create smoothing kernel for translation-invariance.
        ti_conv = create_ti_conv(x.device, ti_kernel_size)

    model.eval()

    eps = 2.0 * max_epsilon / 255.0  # epsilon in scale [-1, 1]
    alpha = 2.0 * step_size / 255.0   # alpha in scale [-1, 1]

    if constraint_img is not None:
        x_min = torch.clamp(constraint_img - eps, -1.0, 1.0)
        x_max = torch.clamp(constraint_img + eps, -1.0, 1.0)
    else:
        x_min = torch.clamp(x - eps, -1.0, 1.0)
        x_max = torch.clamp(x + eps, -1.0, 1.0)

    x_adv = x.clone()

    g = 0
    for t in range(num_iter):
        # every step function
        if every_step_controller is not None:
            if not hasattr(every_step_controller, "__iter__"):
                every_step_controller = [every_step_controller]
            for controller in every_step_controller:
                controller.next_step_modification()

        # Calculate ghat from the current adversarial point.
        x_adv_or_nes = apply_ni(attack_type, x_adv, alpha, mu, g)

        ghat = calculate_attack_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type,
                                     di_prob, di_pad_amount, di_pad_value)

        # Update g (momentum accumulator)
        ghat = apply_ti(ghat, attack_type, ti_conv)
        g = update_mi_momentum(g, ghat, mu)

        # Update x_adv
        pert = alpha * g.sign()
        x_adv = x_adv.detach() + pert
        x_adv = torch.clamp(x_adv, x_min, x_max)
    return x_adv.detach()
####################################################################################################

def mi_ditisi_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MDTS", *args, **kwargs)

def ni_ditisi_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("NDTS", *args, **kwargs)

def mni_ditisi_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MNDTS", *args, **kwargs)

def mi_disi_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MDS", *args, **kwargs)

def mi_di_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MD", *args, **kwargs)

def mi_ti_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MT", *args, **kwargs)

def mi_si_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("MS", *args, **kwargs)

def mi_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("M", *args, **kwargs)

def ni_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("N", *args, **kwargs)

def i_fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("", *args, **kwargs)

def fgsm(*args, **kwargs):
    return mi_ditisi_fgsm_core("", num_iter=1, *args, **kwargs)
