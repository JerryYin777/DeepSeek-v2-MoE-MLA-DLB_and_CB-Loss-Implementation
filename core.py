class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.alpha2 = config.device_level_loss
        self.alpha3 = config.comm_balance_loss
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.n_devices = config.n_devices
        self.topk_group = config.topk_group
        self.device_groups = config.device_groups
        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "gready":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        ### Device-Level Balance Loss
        if self.training and self.alpha2 > 0.0:
            # Get expert groupings on each device
            device_groups = self.device_groups  # Assuming this is a list containing a grouping of experts for each device
            device_f_i_prime = []
            device_P_i_prime = []
            for group in device_groups:
                f_i_prime = torch.tensor([fi[idx] for idx in group], device=hidden_states.device).mean()
                P_i_prime = torch.tensor([Pi[idx] for idx in group], device=hidden_states.device).sum()
                device_f_i_prime.append(f_i_prime)
                device_P_i_prime.append(P_i_prime)
            
            device_f_i_prime = torch.stack(device_f_i_prime)
            device_P_i_prime = torch.stack(device_P_i_prime)
            device_aux_loss = (device_f_i_prime * device_P_i_prime).sum() * self.alpha2
        else:
            device_aux_loss = None
        ### Communication Balance Loss
        if self.training and self.alpha3 > 0.0:
            comm_f_i_prime = []
            comm_P_i_prime = []
            M = self.n_devices  # Total devices
            for group in device_groups:
                f_i_double_prime = (len(device_groups) / (M * seq_len)) * torch.tensor(
                    [sum(topk_idx.view(bsz, seq_len, -1)[:, :, i] == idx) for i in range(self.top_k) for idx in group], 
                    device=hidden_states.device
                ).sum()
                P_i_double_prime = torch.tensor([Pi[idx] for idx in group], device=hidden_states.device).sum()
                comm_f_i_prime.append(f_i_double_prime)
                comm_P_i_prime.append(P_i_double_prime)

            comm_f_i_prime = torch.stack(comm_f_i_prime)
            comm_P_i_prime = torch.stack(comm_P_i_prime)
            comm_aux_loss = (comm_f_i_prime * comm_P_i_prime).sum() * self.alpha3
        else:
            comm_aux_loss = None
            
        total_aux_loss = aux_loss
        if device_aux_loss is not None:
            total_aux_loss = total_aux_loss + device_aux_loss if total_aux_loss is not None else device_aux_loss
        if comm_aux_loss is not None:
            total_aux_loss = total_aux_loss + comm_aux_loss if total_aux_loss is not None else comm_aux_loss
            
        return topk_idx, topk_weight, total_aux_loss
