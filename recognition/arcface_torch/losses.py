import torch
import math
from torch import distributed

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]
        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise        

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            # guarantee a monotonous reduction (easy)
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

class AdaAct(torch.nn.Module):
    ''' 
    This version is modified as ArcFace method
    1. Multiply embeddings with W (FC phase)
    2. Compute Adaface Activate (like normalized softmax) (Act phase)
    '''
    def __init__(self,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaAct, self).__init__()
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        
        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean_z', torch.ones(1)*(20))
        self.register_buffer('batch_std_z', torch.ones(1)*100)

        print('\n\AdaFaceWAct with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, logits:torch.Tensor, norms:torch.Tensor, labels:torch.Tensor):
        logits = logits.clamp(-1+self.eps, 1-self.eps) # for stability
        
        index = torch.where(labels != -1)[0]
        if index.size(0) == 0:
            # import pdb; pdb.set_trace()
            return logits
        target_logits = logits[index, labels[index].view(-1)]
        target_norms = norms[index]

        safe_norms = torch.clip(target_norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean_z = safe_norms.mean().detach()
            std_z = safe_norms.std().detach()
            self.batch_mean_z = mean_z * self.t_alpha + (1 - self.t_alpha) * self.batch_mean_z
            self.batch_std_z =  std_z * self.t_alpha + (1 - self.t_alpha) * self.batch_std_z

        z = (safe_norms - self.batch_mean_z) / (self.batch_std_z+self.eps)
        z = z * self.h 
        z = torch.clip(z, -1, 1)

        # g_angular shape(2,1)
        g_angular = - self.m * z 
        g_angular = g_angular.reshape(-1)

        theta = target_logits.acos()
        try:
            theta_m = torch.clip(theta + g_angular, min=self.eps, max=math.pi-self.eps)######
        except Exception as e:
            print("="*20)
            print("rank: ",distributed.get_rank())
            print("index.size(0): ",index.size(0))
            print("theta.shape: ", theta.shape)
            print("g_angular.shape: ", g_angular.shape)
            print("z.shape: ", z.shape)
            print("target_logits.shape: ", target_logits.shape)
            print("index.shape: ", index.shape)
            # import pdb; pdb.set_trace()
            raise e 
            quit()
            pass
        target_logits_angular = theta_m.cos()

        # g_additive sahpe(2,1)
        g_add = self.m + (self.m * z)
        g_add = g_add.reshape(-1)
        target_logits_add = target_logits_angular - g_add
        # this is not easy_marin in arcface
        gap_ = 1 - self.m*z - self.m - (self.m*z).cos()
        gap_ = gap_.reshape(-1)

        final_target_logits = torch.where(theta + g_angular > 0, target_logits_add, target_logits+gap_)

        logits[index, labels[index].view(-1)] = final_target_logits
        logits = logits * self.s
        return logits