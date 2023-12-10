
class BHP(nn.Module):  #pp's RCL-1
    '''
    Balanced-Hybrid-Proxy loss function
    '''

    def __init__(self, cls_num_list=None, proxy_num_list=None, temperature=0.1,):
        super(BHP, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list
        self.proxy_num_list = proxy_num_list

    def forward(self, proxy, features, targets):
        '''
        :param proxy: proxy vector
        :param features: feature vector
        :param targets: labels
        :return:
        '''
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        # get proxy labels
        targets_proxy = torch.empty((0, 1), dtype=torch.int64)
        for i, num in enumerate(self.proxy_num_list):
            tmp_targets = torch.full([num, 1], i)
            targets_proxy = torch.cat((targets_proxy, tmp_targets), dim=0)

        targets_proxy = targets_proxy.view(-1, 1).to(device)

        # get labels of features and proxies
        targets = torch.cat([targets.repeat(2, 1), targets_proxy], dim=0) #将proxy label和target label合并
        #print("pp's targets:",len(targets))
        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets, targets.T).float().to(device)  #确定给定label和别的sample的label是不是同一类
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2 + int(np.array(self.proxy_num_list).sum())).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # get similarity matrix -->用点积来计算
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, proxy], dim=0)
        logits = features.mm(features.T)  #相似度计算
        logits = torch.div(logits, self.temperature)   #调整logits的scale，控制值的范围和分布

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() #从logits中减去最大值，提高数值计算的稳定性

        # class averaging
        total = sum(self.cls_num_list) #pp
        same_classes = [total / x for x in self.cls_num_list] #pp
        diff_classes = [total / (total-x) for x in self.cls_num_list] #pp
        exp_logits = torch.exp(logits) * logits_mask  #用logits_mask过滤掉某些元素   #加ny
        for i in range(exp_logits.shape[0]):     #pp
            label_num=int(targets[i])
            exp_logits[i, :] *= same_classes[label_num]
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size + int(np.array(self.proxy_num_list).sum()), 2 * batch_size + int(np.array(self.proxy_num_list).sum())) - mask
        for i in range(per_ins_weight.shape[0]):        #pp
            label_num=int(targets[i])
            per_ins_weight[i, :] *= (1-diff_classes[label_num]) 
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)   #加nj
        
        # get loss
        log_prob = logits - torch.log(exp_logits_sum) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))
       # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        print("mean_log_prob_pos:",mean_log_prob_pos)
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
    


class BHP(nn.Module):    #GPT-4
    '''
    Balanced-Hybrid-Proxy loss function
    '''

    def __init__(self, cls_num_list=None, proxy_num_list=None, temperature=0.1):
        super(BHP, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list
        self.proxy_num_list = proxy_num_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert class counts to tensor for device compatibility
        self.cls_num_tensor = torch.tensor(cls_num_list, dtype=torch.float, device=self.device)
        #print("pp's cls_num_tensor:",self.cls_num_tensor)
        self.cls_total = self.cls_num_tensor.sum()
        #print("pp's cls_total:",self.cls_total)
        # Normalize class counts to get weights
        self.normalized_cls_weights = self.cls_total / self.cls_num_tensor
        #print("pp's normalized_cls_weights:",self.normalized_cls_weights)

    def forward(self, proxy, features, targets):
        '''
        :param proxy: proxy vector
        :param features: feature vector
        :param targets: labels
        :return: loss
        '''
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        # get proxy labels
        targets_proxy = torch.empty((0, 1), dtype=torch.int64, device=self.device)
        for i, num in enumerate(self.proxy_num_list):
            tmp_targets = torch.full([num, 1], i, device=self.device)
            targets_proxy = torch.cat((targets_proxy, tmp_targets), dim=0)
        targets_proxy = targets_proxy.to(self.device)

        # get labels of features and proxies
        targets_combined = torch.cat([targets.repeat(2, 1), targets_proxy], dim=0)
        features_combined = torch.cat(torch.unbind(features, dim=1), dim=0)
        features_combined = torch.cat([features_combined, proxy], dim=0)

        # get similarity matrix
        logits = features_combined.mm(features_combined.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # get class weights for each instance in the batch
        per_instance_weights = self.normalized_cls_weights[targets_combined.squeeze()]

        # apply class weights to logits
        weighted_logits = logits * per_instance_weights.unsqueeze(1)

        # calculate weighted softmax
        exp_logits = torch.exp(weighted_logits)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # get loss
        log_prob = weighted_logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (log_prob * (targets_combined == targets_combined.T).float()).sum(1) / (targets_combined == targets_combined.T).float().sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss