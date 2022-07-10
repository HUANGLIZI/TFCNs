import torch


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


# Dice loss
def dice_loss(y_true, y_pred):
	
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.shape)
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = torch.sum(y_true * y_pred, axis=axis)
    fn = torch.sum(y_true * (1-y_pred), axis=axis)
    fp = torch.sum((1-y_true) * y_pred, axis=axis)
    # Calculate Dice score
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    dice_loss = torch.sum(1-dice_class, axis=[1])
    # adjusts loss to account for number of classes
    num_classes = float(y_true.shape[1])
    dice_loss = dice_loss / num_classes
    return dice_loss


# Tversky loss    
def tversky_loss(y_true, y_pred):
	
    delta = 0.7
    smooth = 0.000001
    axis = identify_axis(y_true.shape)
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
    tp = torch.sum(y_true * y_pred, axis=axis)
    fn = torch.sum(y_true * (1-y_pred), axis=axis)
    fp = torch.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    tversky_loss = torch.sum(1-tversky_class, axis=[1])
    # adjusts loss to account for number of classes
    num_classes = float(y_true.shape[1])
    tversky_loss = tversky_loss / num_classes

    return tversky_loss

# Dice coefficient for use in Combo loss
def dice_coefficient(y_true, y_pred):
	
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.shape)
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
    tp = torch.sum(y_true * y_pred, axis=axis)
    fn = torch.sum(y_true * (1-y_pred), axis=axis)
    fp = torch.sum((1-y_true) * y_pred, axis=axis)
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    dice = torch.sum(dice_class, axis=[1])
    # adjusts loss to account for number of classes
    num_classes = float(y_true.shape[1])
    dice = dice / num_classes

    return dice

# Combo loss
def combo_loss(alpha=0.5,beta=0.5):
	
    def loss_function(y_true,y_pred):
        dice = dice_coefficient(y_true, y_pred)
        axis = identify_axis(y_true.shape)
        # Clip values to prevent division by zero error
        epsilon = torch.finfo(torch.float32).eps
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = torch.mean(torch.sum(cross_entropy, axis=[1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss
        
    return loss_function

# Cosine Tversky loss
def cosine_tversky_loss(gamma=1):
    def loss_function(y_true, y_pred):
    	
        delta = 0.7
        smooth = 0.000001
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn), false positives (fp) and
        # true negatives (tn)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        tn = torch.sum((1-y_true) * (1-y_pred), axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Clip Tversky values between 0 and 1 to prevent division by zero error
        tversky_class= torch.clamp(tversky_class, 0., 1.)
        # Calculate Cosine Tversky loss per class
        cosine_tversky = (torch.cos(tversky_class * math.pi))**gamma
        # Sum across all classes
        cosine_tversky_loss = torch.sum(1-cosine_tversky,axis=[1])
    	# adjusts loss to account for number of classes
        num_classes = float(y_true.shape[1])
        cosine_tversky_loss = cosine_tversky_loss / num_classes
        return cosine_tversky_loss

    return loss_function

# Focal Tversky loss
def focal_tversky_loss(gamma=0.75):
    def loss_function(y_true, y_pred):
    	
        delta=0.7
        smooth=0.000001
        # Clip values to prevent division by zero error
        epsilon = torch.finfo(torch.float32).eps
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_tversky_loss = torch.sum(torch.pow((1-tversky_class), gamma), axis=[1])
    	# adjusts loss to account for number of classes
        num_classes = float(y_true.shape[1])
        focal_tversky_loss = focal_tversky_loss / num_classes
        return focal_tversky_loss

    return loss_function

# (modified) Focal Dice loss
def focal_dice_loss(delta=0.7, gamma_fd=0.75):
    def loss_function(y_true, y_pred):
    	
        smooth=0.000001
        # Clip values to prevent division by zero error
        epsilon = torch.finfo(torch.float32).eps
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_dice_loss = torch.sum(torch.pow((1-dice_class), gamma_fd), axis=[1])
    	# adjusts loss to account for number of classes
        num_classes = float(y_true.shape[1])
        focal_dice_loss = focal_dice_loss / num_classes
        return focal_dice_loss

    return loss_function


# (modified) Focal loss
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    def loss_function(y_true, y_pred):
    	
        axis = identify_axis(y_true.shape)
        # Clip values to prevent division by zero error
        epsilon = torch.finfo(torch.float32).eps
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * torch.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = torch.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = torch.mean(torch.sum(focal_loss, axis=[1]))
        return focal_loss

    return loss_function

# Mixed Focal loss
def mixed_focal_loss(y_true,y_pred):
      weight = None
      # Obtain Focal Dice loss
      focal_dice = focal_dice_loss(delta=0.7, gamma_fd=0.75)(y_true,y_pred)
      # Obtain Focal loss
      focal = focal_loss(alpha=None, beta=None, gamma_f=2.)(y_true,y_pred)
      if weight is not None:
        return (weight * focal_dice) + ((1-weight) * focal)  
      else:
        return focal_dice + focal



