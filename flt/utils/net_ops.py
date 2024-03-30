#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import torch
from typing import Optional, Dict, List, Tuple
from flt import network


class NetOps(object):
    def __init__(self, model: torch.nn.Module, store_nonparam: bool = False):
        """
        对于神经网络的特殊处理, 包含获得特征提取网络与分类器等层的参数
        :param torch.nn.Module model: 需要执行操作的神经网络
        :param bool store_nonparam: 是否保留非参数层, defaults to False, 默认不保留非参数层
        """
        self._nn = model
        self._store_nonparam = store_nonparam
        # 加载神经网络层的名称与层模块的元组列表
        self._ntlst = self._get_parametric_layer_ntlst()
        pass

    @property
    def ntlst(self):
        return self._ntlst
        
    @property
    def paramkeys(self):
        return list(self._nn.state_dict().keys())
    
    def _get_parametric_layer_ntlst(self) -> List[Tuple[str, torch.nn.Module]]:
        """
        获得torch中模型的层
        :return List[Tuple[str, torch.nn.Module]]: 模型名称模块元组列表
        """
        
        layers = []
        def get_layer_name_module_iteratively(model: torch.nn.Module, store_nonparam: bool, prefix: Optional[str] = None):
            """
            递归获得网络模型层的名称与 module 的元组列表
            :param torch.nn.Module model: 需要从那个模型中获得层
            :param bool store_nonparam: 是否保留非参数层, defaults to True, 默认不保留非参数层
            :param Optional[str] prefix: 当前层的名称前缀, 提供给递归函数使用, defaults to None
            """
            layer_lst = list(model.named_children())
            for name, module in layer_lst:
                sub_layer_num = len(list(module.named_children()))
                # 生成当前层的 prefix
                if prefix is not None and prefix != "" and prefix != '':
                    name = f"{prefix}.{name}"
                # 获取那些非列表的 module，这些 module 即为 torch 对应的网络层
                if sub_layer_num == 0:
                    # 对于那些不包含任何参数的层直接舍去，只保留存在参数的网络层
                    if store_nonparam or len(module.state_dict()) != 0:
                        layers.append((name, module))                    
                # 包含子网络结构的层则递归处理
                else:
                    get_layer_name_module_iteratively(module, store_nonparam=store_nonparam, prefix=name)
            pass

        get_layer_name_module_iteratively(self._nn, self._store_nonparam)
        return layers

    def _get_nlst_by_id_list(self, idxs: List) -> List[str]:
        """
        从层元组列表中获得对应的层的名称
        :param List ntlst: 需要提取出来的那些那些层的元组列表
        :param int idxs: 需要保留的层下标
        :return List: 保留下来的层名称列表
        """
        nlst = [self._ntlst[idx][0] for idx in idxs]
        return nlst

    def _get_part_param(self, params: Dict, name_lst: List[str]) -> Dict[str, torch.Tensor]:
        """
        基于层的名称获得层参数获得神经网络的参数字典
        :param Dict params: 整个参数字典
        :param List[str] name_lst: 需要提取出来的那些那些层的名称列表
        """
        partial_params = copy.deepcopy(params)
        for name, _ in params.items():
            # 按照 '.' 拆开成列表并删除最后一个具体的参数名称，拼接起来得到当前的层名
            parent = ".".join(name.split(".")[:-1])
            if parent not in name_lst:
                partial_params.pop(name)
        return partial_params

    def get_prefix_params(self, params: Dict, prefix: str) -> Dict[str, torch.Tensor]:
        """
        获得包含某个字符串的所有的参数
        :param Dict params: 整个参数字典
        :param str prefix: 包含的字符串
        """
        rep = copy.deepcopy(params)
        for k, _ in params.items():
            if prefix in k:
                continue
            else:
                rep.pop(k)
            pass
        return rep

    def get_part_param(self, params: Dict, lst: List = []) -> Dict[str, torch.Tensor]:
        """
        基于层的索引获得神经网络的参数字典
        :param Dict params: 整个参数字典
        :param List lst: 需要保留的那些层的下标
        """
        if len(lst) > len(self._ntlst):
            return params
        if max(lst) > len(self._ntlst) or min(lst) < len(self._ntlst):
            return params
        name_lst = self._get_nlst_by_id_list(lst)
        fep = self._get_part_param(params, name_lst)
        return fep

    def get_fe_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得特征提取网络的参数字典
        :param Dict params: 整个参数字典
        """
        fe = []
        for idx, (_, module) in enumerate(self._ntlst):
            if isinstance(module, torch.nn.Linear):
                break
            fe.append(idx)
        
        name_lst = self._get_nlst_by_id_list(fe)
        fep = self._get_part_param(params, name_lst)
        return fep

    def get_cla_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得个性化层的参数字典
        :param Dict params: 整个参数字典
        """
        cla = []
        for idx, (_, module) in enumerate(self._ntlst):
            if isinstance(module, torch.nn.Linear):
                cla = list(range(idx, len(self._ntlst)))
                break
        
        name_lst = self._get_nlst_by_id_list(cla)
        clap = self._get_part_param(params, name_lst)
        return clap

    # TODO 可以根据前缀分出来各层
    def get_fe_m1_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得特征提取层+M1的参数字典
        :param Dict params: 整个参数字典
        """
        fe = []
        for idx, (prefix, module) in enumerate(self._ntlst):
            if prefix.startswith('predictor') or prefix.startswith('l1'):
                break
            # if isinstance(module, torch.nn.Linear):
            #     break
            fe.append(idx)

        name_lst = self._get_nlst_by_id_list(fe)
        fep = self._get_part_param(params, name_lst)
        return fep

    def get_m2_l1_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得特征提取层+M1的参数字典
        :param Dict params: 整个参数字典
        """
        fe = []
        for idx, (prefix, module) in enumerate(self._ntlst):
            if prefix.startswith('features') or prefix.startswith('projection'):
                break
            # if isinstance(module, torch.nn.Linear):
            #     break
            fe.append(idx)

        name_lst = self._get_nlst_by_id_list(fe)
        fep = self._get_part_param(params, name_lst)
        return fep

    # TODO 可以根据前缀分出来各层
    def get_fe_l1_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得特征提取层+M1的参数字典
        :param Dict params: 整个参数字典
        """
        fe = []
        for idx, (prefix, module) in enumerate(self._ntlst):
            if prefix.startswith('l2'):
                break
            # if isinstance(module, torch.nn.Linear):
            #     break
            fe.append(idx)

        name_lst = self._get_nlst_by_id_list(fe)
        fep = self._get_part_param(params, name_lst)
        return fep

    def get_l2_param(self, params: Dict) -> Dict[str, torch.Tensor]:
        """
        获得特征提取层+M1的参数字典
        :param Dict params: 整个参数字典
        """
        fe = []
        for idx, (prefix, module) in enumerate(self._ntlst):
            if prefix.startswith('features') or prefix.startswith('l1'):
                break
            # if isinstance(module, torch.nn.Linear):
            #     break
            fe.append(idx)

        name_lst = self._get_nlst_by_id_list(fe)
        fep = self._get_part_param(params, name_lst)
        return fep

def init_nets(backbone: str, n: int, net_config: dict = {}):
    """
    初始化所有的节点模型
    :param str backbone: 节点模型的名称, 需要和网络的名称一致
    :param int n: 总共的节点数量
    :param dict net_config: 节点网络模型初始化时候的配置参数, defaults to {}
    :return dict: 节点模型的字典
    """
    try:
        initalizer = getattr(network, backbone)
        clients = {}
        for idx in range(n):
            clients[idx] = initalizer(**net_config)
        return clients
    except AttributeError as e:
        return

if __name__ == '__main__':
    net_config = {"model_name": f"SimpleCNN", "num_classes": 10, "in_channel": 3}
    network1 = "Modelcpfl"
    global_nets = init_nets(network1, 1, net_config)
    global_nets = global_nets[0]
    _nn_ops = NetOps(global_nets)
    fe = _nn_ops.get_fe_param(global_nets.state_dict())
    print(f"[feature extractor]: {list(fe.keys())}")
    fe_m1 = _nn_ops.get_fe_m1_param(global_nets.state_dict())
    print(f"[feature extractor + m1]: {list(fe_m1.keys())}")
    cla = _nn_ops.get_cla_param(global_nets.state_dict())
    print(f"[classifier]: {list(cla.keys())}")