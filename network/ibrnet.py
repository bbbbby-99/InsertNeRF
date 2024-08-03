import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class FeedForward2(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward2, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


class NR(nn.Module):
    def __init__(self,dim1,dim2,outdim1,outdim2):
        super(NR, self).__init__()
        self.fc1 = nn.Linear(dim1, outdim1)
        self.fc2 = nn.Linear(dim2, outdim2)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x

class Insertnerf(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        netwidth = 64
        self.N_samples = n_samples
        self.anti_alias_pooling = False
        self.q_fcs = nn.ModuleList([])
        self.hyper_fcs = nn.ModuleList([])
        self.hyper_fcs_1 = nn.ModuleList([])
        self.hyper_fcs_2 = nn.ModuleList([])
        self.q_fcs_1 = nn.ModuleList([])
        self.norm = nn.LayerNorm(netwidth)
        self.rgb_fc = nn.Linear(64, 3)
        self.weight_fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.posenc_dim = 93
        self.viewenc_dim = 39
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )



        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=14,
            num_freqs=15,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=5,
            num_freqs=6,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        viewenc_dim = 39
        trans_depth = 8
        for i in range(trans_depth):
            hyper_fcs_1 = nn.Sequential(
                    NR(64+1, self.N_samples ,64, self.N_samples),
                    nn.ReLU(),
                )
            hyper_fcs_2 = nn.Sequential(
                    NR(64, self.N_samples ,64, self.N_samples),
                    nn.ReLU(),
                )
            q_fcs_1 = nn.Sequential(
                    nn.Linear(self.N_samples, netwidth),
                    nn.ReLU(),
                )
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(128, netwidth),
                    nn.ReLU(),
                )
                hyper_fc = nn.Sequential(
                    NR(64+1, self.N_samples ,netwidth + viewenc_dim, 128),
                    nn.ReLU(),
                )
            else:
                q_fc = nn.Identity()
                hyper_fc = nn.Identity()
            self.q_fcs.append(q_fc)
            self.hyper_fcs.append(hyper_fc)
            self.hyper_fcs_1.append(hyper_fcs_1)
            self.hyper_fcs_2.append(hyper_fcs_2)
            self.q_fcs_1.append(q_fcs_1)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, netwidth // 8),
            nn.ReLU(),
            nn.Linear(netwidth // 8, netwidth),
        )
        self.fc1 = FeedForward2(256+93, 128, 0.1)
        # self.fc2 = NR(128, 128, 64 ,64)
        self.fc2 = NR(64+1, self.N_samples, 93, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc2n = nn.Linear(128, 128)

        self.fc3 = NR(64+1, self.N_samples, 128, 128)
        self.norm3 = nn.LayerNorm(128)
        self.fc3n = nn.Linear(128, 128)

        self.fc4 = NR(64+1, self.N_samples, 128, 128)
        self.norm4 = nn.LayerNorm(128)
        self.fc4n = nn.Linear(128, 128)

        self.fc5 = NR(64+1, self.N_samples, 128, 128)
        self.fc5n = nn.Linear(128, 64)
        self.norm5 = nn.LayerNorm(64)
        self.bias_1 = NR(64+1, self.N_samples, 128, self.N_samples)
        self.bias_2 = NR(64+1, self.N_samples, 128, self.N_samples)
        self.bias_3 = NR(64+1, self.N_samples, 128, self.N_samples)
        self.bias_4 = NR(64+1, self.N_samples, 128, self.N_samples)

        self.FILM1_shift = nn.Sequential(nn.Linear(64+1, 128),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(128, 128))

        self.FILM1_freq = nn.Sequential(nn.Linear(64+1, 128),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(128, 128))

        self.FILM2_shift = nn.Sequential(nn.Linear(64+1, 128),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Linear(128, 128))

        self.FILM2_freq = nn.Sequential(nn.Linear(64+1, 128),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(128, 128))

        self.FILM3_shift = nn.Sequential(nn.Linear(64+1, 128),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Linear(128, 128))

        self.FILM3_freq = nn.Sequential(nn.Linear(64+1, 128),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(128, 128))

        self.FILM4_shift = nn.Sequential(nn.Linear(64+1, 128),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Linear(128, 128))

        self.FILM4_freq = nn.Sequential(nn.Linear(64+1, 128),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(128, 128))

        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(32 + 3, netwidth),
            nn.ReLU(),
            nn.Linear(netwidth, netwidth),
        )

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)


    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, que_pts, que_dir):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat_ = rgb_feat + direction_feat
        rgb_feat_1 = self.rgbfeat_fc(rgb_feat)
        rgb_feat_max = rgb_feat_1.max(dim=2)[0]

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) # [rn,dn,rfn,f]
        mean0, var0 = fused_mean_variance(rgb_feat_, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat_, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat_, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        rgb_feat_max = torch.cat([mean.squeeze(2), var.squeeze(2)], dim=-1) + rgb_feat_max
        rgb_feat_max = torch.cat([rgb_feat_max, weight.mean(dim=2)], dim=-1)

        que_dir = que_dir.squeeze(0)
        que_pts = que_pts.squeeze(0)
        num_views = rgb_feat.shape[2]
        viewdirs = que_dir
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs_ = self.view_enc(viewdirs)
        pts_ = torch.reshape(que_pts, [-1, que_pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(que_pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = torch.reshape(viewdirs_, list(que_dir.shape[:-1]) + [viewdirs_.shape[-1]])
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
        h = input_pts
        bias_1 = self.bias_1(rgb_feat_max)
        bias_2 = self.bias_2(rgb_feat_max)
        bias_3 = self.bias_3(rgb_feat_max)
        bias_4 = self.bias_4(rgb_feat_max)
        FILM_Feature_3 = rgb_feat_max
        phase_shifts1 = self.FILM1_shift(FILM_Feature_3)
        freq1 = self.FILM1_freq(FILM_Feature_3)
        rgb_feat_diff_MLP = self.fc2(rgb_feat_max)
        rgb_feat_diff_MLP = F.relu(rgb_feat_diff_MLP)
        h1 = torch.einsum('kmj,krj->kmr', [h, rgb_feat_diff_MLP])
        bias_1 = bias_1.expand_as(h1)
        h1 = h1 + bias_1
        h1 = (freq1 * h1 + phase_shifts1)
        h1 = self.norm2(h1)
        h1 = self.fc2n(h1)  # 64,128
        h1 = F.relu(h1)
        FILM_Feature_2 = rgb_feat_max
        phase_shifts2 = self.FILM2_shift(FILM_Feature_2)
        freq2 = self.FILM2_freq(FILM_Feature_2)
        rgb_feat_diff_MLP2 = self.fc3(rgb_feat_max)
        rgb_feat_diff_MLP2 = F.relu(rgb_feat_diff_MLP2)
        h2 = torch.einsum('kmj,krj->kmr', [h1, rgb_feat_diff_MLP2])
        bias_2 = bias_2.expand_as(h2)
        h2 = h2 + bias_2
        h2 = (freq2 * h2 + phase_shifts2)
        h2 = self.fc3n(h2)  # 64,128
        h2 = self.norm3(h2)
        h2 = F.relu(h2)
        FILM_Feature_1 = rgb_feat_max
        phase_shifts3 = self.FILM3_shift(FILM_Feature_1)
        freq3 = self.FILM3_freq(FILM_Feature_1)
        rgb_feat_diff_MLP3 = self.fc4(rgb_feat_max)
        rgb_feat_diff_MLP3 = F.relu(rgb_feat_diff_MLP3)
        h3 = torch.einsum('kmj,krj->kmr', [h2, rgb_feat_diff_MLP3])
        bias_3 = bias_3.expand_as(h3)
        h3 = h3 + bias_3
        h3 = (freq3 * h3 + phase_shifts3)
        h3 = self.fc4n(h3)  # 64,128
        h3 = self.norm4(h3)
        h3 = F.relu(h3)
        phase_shifts4 = self.FILM4_shift(FILM_Feature_1)
        freq4 = self.FILM4_freq(FILM_Feature_1)
        rgb_feat_diff_MLP4 = self.fc5(rgb_feat_max)
        rgb_feat_diff_MLP4 = F.relu(rgb_feat_diff_MLP4)
        h4 = torch.einsum('kmj,krj->kmr', [h3, rgb_feat_diff_MLP4])
        bias_4 = bias_4.expand_as(h4)
        h4 = h4 + bias_4
        h4 = (freq4 * h4 + phase_shifts4)
        h4 = self.fc5n(h4)  # 64,128
        h4 = self.norm5(h4)
        h4 = F.relu(h4)
        weight = self.weight_fc(h4)
        q = h4
        for i, (q_fc, hyper_fc, hyper_fc_1, hyper_fc_2, q_fcs_1) in enumerate(
                zip(self.q_fcs, self.hyper_fcs, self.hyper_fcs_1, self.hyper_fcs_2, self.q_fcs_1)
        ):
            if i % 2 == 0:
                q = torch.cat((q, input_views), dim=-1)
                rgb_feat_diff_MLP = hyper_fc(rgb_feat_max)
                q = torch.einsum('kmj,krj->kmr', [q, rgb_feat_diff_MLP])
                q = q_fc(q)
            rgb_feat_diff_MLP1 = hyper_fc_1(rgb_feat_max)
            q1 = torch.einsum('kmj,krj->kmr', [q, rgb_feat_diff_MLP1])
            rgb_feat_diff_MLP2 = hyper_fc_2(rgb_feat_diff_MLP1)
            q2 = torch.einsum('kmj,krj->kmr', [q, rgb_feat_diff_MLP2])
            q = q1 + q2
            q = q_fcs_1(q)
            q = self.norm(q)
            # 'learned' density
        # normalize & rgb
        q = q + h4
        h = self.norm(q)
        rgb = self.rgb_fc(h)
        rgbout = torch.sum((weight0 * rgb_in), -2)
        rgb = (rgb + rgbout)/2
        blending_weights = F.softmax(weight, dim=1)  # color blending
        outputs = torch.sum(rgb * blending_weights, dim=1).unsqueeze(0)

        return outputs, blending_weights