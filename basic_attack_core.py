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
    # TODO 1
    # Hint
    # Momentum should accumulate information across iterations instead of using only the current step.
    # First normalize `ghat` for each sample so its scale does not dominate the update.
    # Then combine the previous momentum `g` and the normalized gradient using the decay factor `mu`.
    # The returned tensor is the running direction that will be used to update `x_adv`.

    # [The code below is a basic version, so it should be modified.]
    return mu*g + ghat/torch.sum(torch.abs(ghat), dim=[1, 2, 3], keepdim=True)
####################################################################################################



####################################################################################################
# [DI] configures DI only when D is provided
def apply_di(x_adv, attack_type, di_prob, di_pad_amount, di_pad_value):
    if 'D' in attack_type:
        return diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value)
    return x_adv

# [DI] Implementing diverse input (resize & padding) 
def diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value):
    # TODO 2
    # Hint 
    # Diverse Input applies a random spatial transform before the forward pass.
    # A standard version resizes the image to a random larger size, pads it at random offsets,
    # resizes it back to the original resolution, and keeps the batch shape unchanged.
    # This transformed input should be used only with probability `di_prob`

    # [The code below is a basic version, so it should be modified.]
    x_di = x_adv
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
    # TODO 3
    # Hint 
    # This function should build the Gaussian filter used for translation-invariant smoothing.
    # Create 1D coordinates from `-nsig` to `nsig`, convert them into Gaussian weights,
    # and form a 2D kernel by taking the outer product of the 1D vector with itself.
    # Finally, normalize the kernel so that all entries sum to 1,
    # because the convolution should smooth the gradient without changing its overall scale too much.

    # [The code below is a basic version, so it should be modified.] keeping a valid kernel whose center is 1 so TI becomes an identity operation.
    kernel = np.zeros((kernlen, kernlen), dtype=np.float32)
    kernel[kernlen // 2, kernlen // 2] = 1.0
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
    # TODO 4
    # Hint 
    # Scale-Invariant FGSM sums gradients from multiple scaled copies of the current adversarial input.
    # For each scale, divide the input by `2 ** si_counter`, enable gradients on that scaled tensor,
    
    # optionally pass it through DI like below.
    # si_input2 = apply_di(si_input, attack_type, di_prob, di_pad_amount, di_pad_value)

    # And compute the loss gradient with respect to the scaled input. 
    # Add each gradient to `grad_sum` with the corresponding `1 / si_div` weight,
    # and return the accumulated result as the final `ghat`.

    # [The code below is a basic version, so it should be modified.] using the base gradient so the basic version stays I-FGSM-like.
    ghat = calculate_loss_gradient(model, x_adv_or_nes, x_adv_or_nes, y, target_label)
    return ghat
####################################################################################################



####################################################################################################
# [NI] configures NI only when N is provided, and prepares the look-ahead input tensor
def apply_ni(attack_type, x_adv, alpha, mu, g):
    # TODO 5
    # Hint
    # NI should compute gradients at a look-ahead point.
    # and then enable gradients on that tensor.
    # When 'N' is not in attack_type, return the usual prepared input.

    # [The code below is a basic version, so it should be modified.] keeping NI path as baseline behavior.
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
                        mu=1.0, number_of_si_scales=10, constraint_img=None, di_prob=0.5, di_pad_amount=31,
                        di_pad_value=0, ti_kernel_size=7, every_step_controller=None):
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
