import torch
import torch.nn as nn
from IPython import embed
import math
from utils import utils_transform


nn.Module.dump_patches = True




class AvatarPoser(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, body_model, device):
        super(AvatarPoser, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        self.stabilizer = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 6)
            )
        self.joint_rotation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 126)
            )

        self.body_model = body_model

    @staticmethod
    def fk_module(global_orientation, joint_rotation, body_model):

        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1,6)).reshape(global_orientation.shape[0],-1).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1,6)).reshape(joint_rotation.shape[0],-1).float()
        body_pose = body_model(**{'pose_body':joint_rotation, 'root_orient':global_orientation})
        joint_position = body_pose.Jtr

        return joint_position


    @staticmethod
    def ik_module(smpl, smpl_jids, target_pose_ids, target_3ds,
                         body_pose = None, global_orient = None, transl = None, learning_rate=1e-1, n_iter=5):
        target_3ds = target_3ds.view(1, -1, 3)
        body_pose_sub = torch.tensor(body_pose[:, target_pose_ids],requires_grad = True)
        opti_param = [body_pose_sub]
        optimiser = torch.optim.Adam(opti_param, lr = learning_rate)
        ##将目标关节位置 target_3ds 转换为形状为 (1, -1, 3) 的张量
        # 创建一个可优化的参数列表 opti_param，其中只包含子集 body_pose_sub。body_pose_sub 是 body_pose 中仅包含目标姿势关节的子集，
        #这些关节的索引由 target_pose_ids 列表定义。此步骤的目的是为了将关节角度作为要优化的参数。
        # 使用 Adam 优化器创建一个优化器 optimiser，使用给定的学习率 learning_rate。该优化器将用于更新 body_pose_sub 参数。
        for i in range(n_iter):
            body_pose[:, target_pose_ids] = body_pose_sub
            out = smpl(**{'pose_body':body_pose, 'root_orient':global_orient, 'trans': transl})
            j_3ds = out.Jtr.view(1, -1, 3)
            loss = torch.mean(torch.sqrt(torch.sum(torch.square(j_3ds[:, smpl_jids].squeeze()-target_3ds)[:,[20,21],:],axis=-1)))
            #最终目标是优化 双手关节位置 的尽量和gt一致。
            optimiser.zero_grad()
            loss.backward(retain_graph=True)
            optimiser.step()
            body_pose = body_pose.detach()
        return body_pose


    def forward(self, input_tensor, do_fk = True):

#        embed()
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]

        global_orientation = self.stabilizer(x)
        joint_rotation = self.joint_rotation_decoder(x)
        if do_fk:
            joint_position = self.fk_module(global_orientation, joint_rotation, self.body_model)
            return global_orientation, joint_rotation, joint_position
        else:
            return global_orientation, joint_rotation

