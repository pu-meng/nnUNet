import torch
import torch.nn as nn


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    """
    自动判断空间维度轴编号
    """
    # Three dimensional
    #如果是5个维度,那么,shape=[B,C,D,H,W]
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    #如果是4个维度,那么,shape=[B,C,H,W]
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    这个数学公式意思很类似,对每一个像素,预测两个概率,p_背景,和p_肿瘤,预测为背景,和预测为肿瘤的概率
    如果是背景像素,计算(1-p_背景)^gamma * -log(p_背景),计算一个[B,H,W]的矩阵装这个,肿瘤像素为0,背景像素是这个值
    如果是肿瘤像素,计算(1-p_肿瘤)^gamma * -log(p_肿瘤),计算一个[B,H,W]的矩阵装这个,肿瘤像素为这个值,背景像素为0
    然后乘以不对称权重,相加之后得到[B,H,W],求平均,
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        y_pred: [B,2,...] 支持2D(B,2,H,W)和3D(B,2,Z,Y,X)
        y_true: [B,2,...]
        """

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        #torch.clamp把y_pred限制在[epsilon, 1-epsilon]之间，防止除零错误
        cross_entropy = -y_true * torch.log(y_pred)
        #cross_entropy: [B,2,H,W]

        #cross_entropy[:,0,:,:]=-y_true[:,0,:,:]*torch.log(y_pred[:,0,:,:])
#y_true[i,0,j,k]代表的是B=i,H=j,W=k像素,这里的y_true[i,0,j,k]=0表示为肿瘤,=1,则表示为背景
#y_true[i,1,j,k]=1表示为肿瘤,=0表示为背景,
#y_pred[i,0,j,k]表示的是B=i,H=j,W=k像素,这里的y_pred[i,0,j,k]表示为背景的概率,=1-y_pred[i,0,j,k]表示为肿瘤的概率
        # Calculate losses separately for each class
        #cross_entropy类似如果是背景像素,则为[-log(p背景),0],如果是肿瘤像素,则为[0,-log(p肿瘤)]
        back_ce = torch.pow(1 - y_pred[:,0,:,:], self.gamma) * cross_entropy[:,0,:,:]
        #back_ce类似(1-p)^gamma * -log(p),这里的p是像素真实为背景像素的预测为背景的概率
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,1,:,:], self.gamma) * cross_entropy[:,1,:,:]
        fore_ce = self.delta * fore_ce
        #fore_ce维度[B,H,W]

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))#type:ignore
#torch.stack([back_ce, fore_ce], axis=-1)把back_ce和fore_ce在最后一个维度上拼接起来,然后求和
#torch.stack输出维度[B,H,W,2],然后axis=-1,求和,输出维度[B,H,W]
        return loss


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    Symmetric/Asymmetric是前景/背景石是否都加(1-p)^gamma
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
	# Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,0,:,:], self.gamma) * cross_entropy[:,0,:,:]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,1,:,:]#Asymmetric和symmetric的区别,
        #Asymmetric的前景不加(1-p)^gamma
        #如果加了(1-p)^gamma,那么容易预测的会被抑制梯度
        #前景,我们在asymmetric不调制,因为前景太少了,背景调制因为背景很多
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))#type:ignore

        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)#type:ignore
        #tp可以理解为所有的体素或者像素,在对应位置上,预测对的概率的sum
        fn = torch.sum(y_true * (1-y_pred), axis=axis)#type:ignore
        #fn可以理解为所有的体素或者像素,在对应位置上,预测错的概率的sum
        fp = torch.sum((1-y_true) * y_pred, axis=axis)#type:ignore
        #fp可以理解为所有的体素或者像素,在对应位置上,预测错的概率的sum
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))#type:ignore
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)#type:ignore
        fn = torch.sum(y_true * (1-y_pred), axis=axis)#type:ignore
        fp = torch.sum((1-y_true) * y_pred, axis=axis)# type:ignore
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))#type:ignore
        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)  # Dice-based
      symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)          # CE-based
      if self.weight is not None:
        return (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)
      else:
        return symmetric_ftl + symmetric_fl


class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)  # Dice-based
      asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)          # CE-based

      # Return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
      if self.weight is not None:
        return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl
