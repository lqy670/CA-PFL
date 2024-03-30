#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations
import os
import re
import sys
import torch
import pickle
import collections
from random import Random
from typing import Union, List
from torch.utils.data import Dataset
from torchvision.datasets import utils


""" 由原始的数据处理生成所有的样本 """
class Generator(object):
    """ ShakeSpeare原始数据集处理, 将原始的数据, 提取其戏剧部分, 以及角色与台词, 生成其对应的字典文件等 """
    def __init__(self) -> None:
        # 正则表达式, 提取莎士比亚数据集中戏剧的演员名称以及换行符等
        self._CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self._CONT_RE = re.compile(r'^    (.*)')
        # The Comedy of Errors 该戏剧中存在缩进错误，需要使用不同的正则表达式
        self._COMEDY_OF_ERRORS = "THE COMEDY OF ERRORS"
        self._COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self._COE_CONT_RE = re.compile(r'^(.*)')
        # 戏剧的最大长度，即多少行
        self._MAX_ROW_NUM = 124195

    def __match_character_regex(self, line: str, coe: bool = False):
        """
        匹配角色名称以及该角色的台词
        :param str line: 当前的需要匹配的字符串
        :param bool coe: 是否是存在缩进异常的戏剧, defaults to False
        """
        return (
            self._COE_CHARACTER_RE.match(line) if coe else self._CHARACTER_RE.match(line)
        )

    def __match_continuation_regex(self, line: str, coe: bool = False):
        """
        匹配角色的多行台词
        :param str line: 当前的需要匹配的字符串
        :param bool coe: 是否是存在缩进异常的戏剧, defaults to False
        """
        return (
            self._COE_CONT_RE.match(line) if coe else self._CONT_RE.match(line)
        )

    def get_plays_with_charsnippet(self, shakespeare: str):
        """
        将数据集按照戏剧名称划分成不同的部分
        :param str shakespeare: 莎士比亚数据的原始数据
        """
        # 切分字符串, 跳过第一行, 第一行存在一句 ”by William Shakespeare", 影响处理
        shakespeares = shakespeare.splitlines(True)[1:]

        ### 跳过某些内容, the sonnets; all's well that ends well
        # 起始行与跳过的作品数量
        srn, count = 0, 0
        for rn, ln in enumerate(shakespeares):
            if "by William Shakespeare" in ln:
                count += 1
            if count == 2:
                srn = rn - 5
                break
        slines = shakespeares[srn:]

        # 元组列表，用于记录 (戏剧名，角色) 以及需要忽略的某些行
        plays, discarded_lines = [], []
        characters = collections.defaultdict(list)
        current_character = None
        coe = False
        for crn, ln in enumerate(slines):
            # 超过最大的行，则直接跳出
            if crn > self._MAX_ROW_NUM - srn:
                break
            ### 获取戏剧名获取
            if "by William Shakespeare" in ln:
                # 当匹配到新的戏剧时，则需要重新初始化该戏剧的任务名以及当前人物名
                current_character = None
                characters = collections.defaultdict(list)
                # 获得戏剧的标题以及是否是存在缩进异常的戏剧
                title, coe = self.__get_play_title_coe(slines, crn)
                # 添加到戏剧元组的列表中
                plays.append((title, characters))
                continue
            
            ### 获取角色名称以及台词谈话
            # 匹配戏剧中角色名称
            crst = self.__match_character_regex(ln, coe)
            if crst:
                # 获得角色名称、台词
                character, snippet = crst.group(1), crst.group(2)
                # 有些角色名称存在大小写，全部将其转为大写
                character = character.upper()
                # 不是缩进异常的戏剧，且不会提取到的角色名称不是 ACT V. SCENE I.这种的
                if not (coe and character.startswith('ACT ')):
                    characters[character].append(snippet)
                    current_character = character
                else:
                    current_character = None
                continue
            ### 获取当行台词谈话等
            #  未匹配到戏剧的角色名称, 且还未匹配到下一个戏剧，则表示当前的角色台词还未获得完全
            elif current_character:
                # 继续匹配当前角色的台词
                ccrst = self.__match_continuation_regex(ln, coe)
                if ccrst:
                    # 是缩进异常的戏剧，且出现 < 则表示当前的角色已经匹配完成
                    if coe and ccrst.group(1).startswith('<'):
                        current_character = None
                    else:
                        characters[current_character].append(ccrst.group(1))
                    continue
            # 去除回车等空格
            ln = ln.strip()
            # 上述的流程都为执行，则判断该行内容为忽略的内容
            if ln and crn > 2646:
                # 在 2646 行之前，是 sonnets (十四行诗)，需要将其忽略
                discarded_lines.append('%d:%s' % (crn, ln))
        # 删除不存在角色的戏剧
        return [play for play in plays if len(play[1]) > 1], discarded_lines

    def __get_play_title_coe(self, shakespeare: list, crn: int) -> tuple:
        """
        从当前行 crn 的字符串向前 2 ~ 7为该戏剧的名称, 
        并基于名称判断是否是缩进异常的戏剧
        :param list shakespeare: 切分成一行行的字符串
        :param int crn: 当前的行号
        :return tuple: 戏剧名称, 是否缩进异常
        """
        # 获取某戏剧的名称, 即当戏剧名称只可能出现在 by William Shakespeare 的前 2-7 行
        trns = [crn - p for p in range(2, 8)]
        title = None
        for trn in trns:
            if shakespeare[trn].strip():
                title = shakespeare[trn]
                break
        if title is not None:
            title = title.strip()
        else:
            assert title, (f"Parsing error on line {crn}. Expecting title 2 or 3 lines above.")
        # 是否是那些存在缩进错误的戏剧
        coe = (title == 'THE COMEDY OF ERRORS')
        return title, coe

    def __remove_nonalphanumerics(self, filename: str) -> str:
        """
        将名称中的非字母、数字等内容去除
        :param str filename: 需要处理的字符串
        :return str: 去除非字母、数字的字符串
        """
        return re.sub('\\W+', '_', filename)

    def __play_and_character(self, play: str, character: str) -> str:
        """
        将戏剧名和角色名拼接
        :param str play: 戏剧名
        :param str character: 角色名
        :return str: 拼接处理后的戏剧角色新名称
        """
        return self.__remove_nonalphanumerics((play + '_' + character).replace(' ', '_'))

    def get_train_test_by_character(self, plays: list, fraction: float = 0.2) -> tuple[dict, dict, dict]:
        """
        将角色数据划分为训练集与测试集
        :param list plays: 由戏剧名以及角色字典构成的元组列表 := (play, dict:= {character: [] } )
        :param float fraction: 测试集的数量比例, defaults to 0.2
        :return tuple[dict, dict, dict]: 戏剧角色名: 戏剧的字典, 训练样本集, 测试样本集
        """
        # 当前跳过多少的角色数量
        skipped_characters = 0
        # 总体的训练，测试集
        all_train_examples = collections.defaultdict(list)
        all_test_examples = collections.defaultdict(list)

        def add_examples(apped_dict: dict, pcs: list) -> None:
            """
            向字典中添加戏剧, 角色, 台词谈话三元组样本
            :param dict apped_dict: 被添加到的字典 := {play+character: sound_bits}
            :param list pcs: 被添加的三元组样本
            """
            for play, character, sound_bite in pcs:
                apped_dict[
                    self.__play_and_character(play, character)
                ].append(sound_bite)

        users_and_plays = {}
        # 遍历所有的戏剧元组等列表
        for play, characters in plays:
            # 当前的角色名称
            curr_characters = list(characters.keys())
            # 将所有的戏剧与其角色名进行进行处理
            # 并将新名称作为键，使用戏剧名称作为值，构造成字典
            for c in curr_characters:
                users_and_plays[self.__play_and_character(play, c)] = play
            
            # 遍历当前戏剧的角色与台词谈话
            for character, sound_bites in characters.items():
                # 构造 戏剧, 角色, 台词谈话 三元组
                examples = [(play, character, sound_bite) for sound_bite in sound_bites]
                # 我们至少需要一个训练样本 (三元组) 以及测试样本 (三元组) ，小于两个样本的戏剧进行跳过
                if len(examples) <= 2:
                    skipped_characters += 1
                    continue
                # 对于当前的戏剧的角色作为训练集一遍进行划分
                # 即将戏剧下的角色台词谈话进行划分
                train_examples = examples
                if fraction > 0:
                    # 计算测试样本的数量
                    num_test = max(int(len(examples) * fraction), 1)
                    # 划分
                    train_examples = examples[:-num_test]
                    test_examples = examples[-num_test:]
                    # 判断划分是否成功
                    assert len(test_examples) == num_test
                    assert len(train_examples) >= len(test_examples)
                    add_examples(all_test_examples, test_examples)
                add_examples(all_train_examples, train_examples)
        return users_and_plays, all_train_examples, all_test_examples

    def write_data_by_character(self, examples: dict, output_directory: str):
        """
        按照角色将数据保存在本地
        :param dict examples: 样本集字典
        :param str output_directory: 需要保存到的文件夹
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for character_name, sound_bites in examples.items():
            filename = os.path.join(output_directory, character_name + '.txt')
            with open(filename, 'w') as output:
                for sound_bite in sound_bites:
                    output.write(sound_bite + '\n')

    def generator(self, raw: str, filename: str, frac: float):
        """
        从 raw 路径下的 filename 文件生成训练集与测试集
        :param str raw: 源数据所在目录
        :param str filename: 源数据文件名
        :param float frac: 训练集，测试集划分比例
        :return tuple: 训练集，测试集
        """
        with open(os.path.join(raw, filename), 'r') as input_file:
            shakespeare = input_file.read()
        # 将戏剧切分成其(戏剧名, 角色名与其台词的字典) 以及忽略行的行号与内容
        plays, discarded_lines = self.get_plays_with_charsnippet(shakespeare)
        print(f"Discarded {len(discarded_lines)} lines")
        # 将上述的元组列表处理成训练样本集与测试样本集
        users_plays, train_explames, test_examples = self.get_train_test_by_character(
            plays,
            fraction=frac
        )
        train_dataset = self.__parse_dataset(train_explames, users_plays)
        test_dataset = self.__parse_dataset(test_examples, users_plays)
        return train_dataset, test_dataset

    def __parse_dataset(self, triplets: dict, users_plays: dict, raw: bool = False):
        """
        处理生成数据集
        :param dict triplets: 
        :param dict users_plays: 戏剧+角色字典, 键为戏剧+角色, 值为戏剧
        :param bool raw: 是否保留原文本, defaults to False
        :return dict: 处理生成的字符数据集
        """
        # users 记录着戏剧角色组合名
        # hierarchies 记录着戏剧角色名组合名称其所属的戏剧
        # num_samples 记录着每个戏剧的戏剧角色的条数
        # user_data 记录着当前戏剧角色组合名的输入数据 x与标签数据 y
        users, hierarchies, num_samples, user_data = [], [], [], {}
        # 遍历所有的 [戏剧+角色] 与其 [台词谈话] 等
        for user, passages in triplets.items():
            passage = "\n".join(passages)
            dataX, dataY = self.__txt_to_data(passage)
            if(len(dataX) > 0):
                users.append(user)
                # 是否包含原戏剧台词文本
                if raw:
                    user_data[user] = {'raw': passage}
                else:
                    user_data[user] = {}
                user_data[user]['x'] = dataX
                user_data[user]['y'] = dataY
                hierarchies.append(users_plays[user])
                num_samples.append(len(dataY))
        all_data = {
            "users": users, "hierarchies": hierarchies,
            "num_samples": num_samples, "user_data": user_data
        }
        return all_data

    def __txt_to_data(self, raw_text: str, seq_length: int = 80):
        """
        将文本按照指定长度划分成输入数据与其下一个标签
        :param str raw_text: 原台词对话文本
        :param int seq_length: 序列长度, defaults to 80
        :return tuple: 输入数据 x 与标签数据 y
        """
        raw_text = raw_text.replace('\n', ' ')
        raw_text = re.sub(r"   *", r' ', raw_text)
        dataX = []
        dataY = []
        for i in range(0, len(raw_text) - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)
        return dataX, dataY


""" 由生成的全样本数据集采样得到节点的训练、测试所使用的数据集 """
class Sampler(object):
    """
    all_data: {
        "users": [playname_charactername, playname_charactername, ...],
        "user_data": {
            "playname_charactername": {"x": [], "y": []},
            "playname_charactername": {"x": [], "y": []},
        },
        "num_samples": [playname_charactername_num, playname_charactername_num],
        "hierarchies": [playname, playname, ...]
    }
    """
    def __init__(self, rs: int, un: float, df: float, smpd_t: str, min_samples: int = 50) -> None:
        """
        采样器+数据划分器
        :param int rs: 随机数种子
        :param float dt: 采样的类型, iid or non-iid
        :param float un: 采样的user (戏剧名与角色的组合名称为user) 比例, 将被忽略当 dt == 'non-iid'时
        :param float df: 总数量的采样比例
        """
        self._rs = rs
        self._un = un
        self._df = df
        self._smpd_t = smpd_t
        self._min_samples = min_samples
        assert self._smpd_t in ["iid", "non-iid"], "The sample type must be 'iid' or 'non-iid'"
        self._rng = Random(self._rs)
        self._smpd_user_count = 0
    
    def sampler(self, all_datas: list[dict]):
        smpd_data = []
        if self._smpd_t == "iid":
            for all_data in all_datas:
                smpd_data.append(
                    # 去除小于指定数量样本的 users
                    self.__remove_lt_min_size(
                        # 采样切分
                        self.__iid_sampler(all_data)
                    )
                )
        else:
            for all_data in all_datas:
                smpd_data.append(
                    # 去除小于指定数量样本的 users
                    self.__remove_lt_min_size(
                        # 采样切分
                        self.__niid_sampler(all_data)
                    )
                )
        return smpd_data

    def __iid_sampler(self, all_data: dict):
        """
        IID的采样, 采样得到若干个节点, 并且采样得到所有节点样本总量的样本
        :param dict all_data: 采样的原始样本数据
        :return dict: 
        """
        # 采样的样本总数量
        tot_num_samples = sum(all_data["num_samples"])
        smpd_num_samples = int(self._df * tot_num_samples)
        # 采样的 users 数量, 最少保证采样一个 user
        num_users = len(all_data["users"])
        smpd_num_users = int(self._un * num_users)
        if smpd_num_users == 0: smpd_num_users = 1
        # users 采样，即需要生成多少个节点的数据
        users = [str(idx + self._smpd_user_count) for idx in range(smpd_num_users)]
        self._smpd_user_count += smpd_num_users

        # 获取到所有的样本数据，即所有的戏剧角色的输入数据 x 以及输出数据 y
        raw_lst = list(all_data["user_data"].values())
        raw_x = [item["x"] for item in raw_lst]
        raw_y = [item["y"] for item in raw_lst]
        # raw是所有的的戏剧角色的台词谈话, 将其展平为以为数组
        # 格式为 [['abc', 'bcd', ...], ['efg', 'fgh', ...], ['lmn', 'mno', ...]]
        x_lst = [item for xlst in raw_x for item in xlst]
        y_lst = [item for ylst in raw_y for item in ylst]
        # 样本采样的下标
        smpd_indices = self._rng.sample(list(range(tot_num_samples)), smpd_num_samples)
        # 初始化采样的字典
        user_data = {}
        for user in users:
            user_data[user] = { "x": [], "y": [] }
        # 样本采样
        smpd_x_samples = [x_lst[idx] for idx in smpd_indices]
        smpd_y_samples = [y_lst[idx] for idx in smpd_indices]
        # 将采样的样本数据进行 iid 划分
        smpd_x_shards = self.__divided(smpd_x_samples, smpd_num_users)
        smpd_y_shards = self.__divided(smpd_y_samples, smpd_num_users)
        # 保存到user_data中
        for idx in range(smpd_num_users):
            user_data[users[idx]]["x"] = smpd_x_shards[idx]
            user_data[users[idx]]["y"] = smpd_y_shards[idx]
        num_samples = [len(user_data[u]["y"]) for u in users]
        return {
            "num_samples": num_samples, 
            "users": users, "user_data": user_data
        }

    def __niid_sampler(self, all_data: dict):
        """
        按照戏剧角色名称顺序依次进行采样, 直到达到采样的样本总数
        :param dict all_data: 原始数据集
        :return tuple: 采样得到的user, user_data, num_samples, hierarchies
        """
        # 采样的样本总数量
        tot_num_samples = sum(all_data["num_samples"])
        smpd_num_samples = int(self._df * tot_num_samples)
        # 原数据中的 users
        users = all_data["users"]
        hierarchies, users_hiers = None, None
        if "hierarchies" in all_data:
            hierarchies = []
            users_hiers = list(zip(users, all_data["hierarchies"]))
            self._rng.shuffle(users_hiers)
        else:
            self._rng.shuffle(users)
        
        # 采样样本, 从第 0 个戏剧角色开始
        cuser = 0
        user_data, num_samples = {}, []
        smpd_ctot_num_samples = 0
        # 循环采样直到达到采样的样本总数
        while (smpd_ctot_num_samples < smpd_num_samples):
            # 如果存在戏剧角色组合名称以及所属的戏剧
            if users_hiers is not None and hierarchies is not None:
                user, hier = users_hiers[cuser]
                hierarchies.append(hier)
            else:
                user = users[cuser]
            
            cdata = all_data["user_data"][user]
            cnum_samples = len(all_data["user_data"][user]["y"])

            # 当前的 user 采样样本已经超过可采样的样本数量
            if (smpd_ctot_num_samples + cnum_samples > smpd_num_samples):
                cnum_samples = smpd_num_samples - smpd_ctot_num_samples
                smpd_indices = self._rng.sample(list(range(cnum_samples)), cnum_samples)
                x, y = [], []
                for idx in smpd_indices:
                    x.append(all_data["user_data"][user]["x"][idx])
                    y.append(all_data["user_data"][user]["y"][idx])
                cdata = {"x": x, "y": y}
                pass

            user_data[str(cuser)] = cdata
            num_samples.append(cnum_samples)
            smpd_ctot_num_samples += cnum_samples

            cuser += 1
        
        # if "hierarchies" in all_data and users_hiers is not None:
        #     users = [u for u, _ in users_hiers[:cuser]]
        # else:
        #     users = users[:cuser]
        users = list(map(str, range(cuser)))
        return {
            "hierarchies": hierarchies, "num_samples": num_samples, 
            "users": users, "user_data": user_data
        }

    def __divided(self, idxlst: Union[List[int], List[List[int]]], gn: int):
        """
        将 idxlst 下标的列表分成 gn 组
        :param List[int] idxlst: 下标的列表
        :param int gn: 组数
        """
        ns = len(idxlst)
        gs = int(ns // gn)
        # 有多少组的样本数量多一个
        gnp1 = ns - gs * gn
        # 有多少组的样本数量不多
        gnp0 = gn - gnp1
        shards = []
        for k in range(gnp0):
            s, e  = k * gs, (k + 1) * gs
            shards.append(idxlst[s:e])
        
        start = gnp0 * gs
        for k in range(gnp1):
            s, e = k * (gs + 1), (k + 1) * (gs + 1)
            shards.append(idxlst[(start + s):(start + e)])
        return shards
    
    def __remove_lt_min_size(self, smpd_data: dict):
        users, user_data, num_samples, hierarchies = [], {}, [], []
        num_users = len(smpd_data["users"])
        for idx in range(num_users):
            cuser = smpd_data["users"][idx]
            chier = None
            if "hierarchies" in smpd_data:
                chier = smpd_data["hierarchies"][idx]
            cnum_samples = smpd_data["num_samples"][idx]
            if (cnum_samples >= self._min_samples):
                users.append(cuser)
                user_data[cuser] = smpd_data["user_data"][cuser]
                if chier is not None:
                    hierarchies.append(chier)
                num_samples.append(cnum_samples)
        return {
            "hierarchies": hierarchies, "num_samples": num_samples, 
            "users": users, "user_data": user_data
        }


""" 将训练使用的数据集切分成训练集与测试集 """
class Separator(object):
    def __init__(self, ds: str, frac: float, by_type: str, rs: int = 1234) -> None:
        self._rs = rs
        self._ds = ds
        self._frac = frac
        self._by_type = by_type
        # shakespeare 生成数据时的序列长度
        self._seql = 80
        assert self._by_type in ["sample", "user"], "The Separator type must be 'sampler' or 'user'"
        self._rng = Random(self._rs)

    def separator(self, smpd_datas: list[dict]) -> list[dict[str, list]]:
        if self._by_type == "user":
            return self.__divided_by_user(smpd_datas)
        else:
            sept_data = []
            for smpd_data in smpd_datas:
                sept = self.__divided_by_sample(smpd_data)
                sept_data.append({
                    "train": sept[0],
                    "test": sept[1]
                })
            return sept_data

    def __divided_by_user(self, smpd_datas: list) -> list[dict[str, list]]:
        """
        按照users将数据集切分成训练集, 测试集
        :param list smpd_datas: 原始的数据集
        :return list: 切分得到的数据集长度
        """
        # 将所有的需要处理的数据按照其 users、num_samples进行组合成数组
        user_files = []
        for smpd_data in smpd_datas:
            user_files.extend(
                [u, ns, smpd_data] for (u, ns) in zip(smpd_data["users"], smpd_data["num_samples"])
            )
        # 获得合并后的 users 的数量
        num_users = len(user_files)
        # 基于合并后的 users 数量进行切分计算
        num_train_users = int(self._frac * num_users)
        train_indices = self._rng.sample(list(range(num_users)), num_train_users)
        # 记录那些 user 分配到训练集
        train_bst = [False for _ in range(num_users)]
        for idx in train_indices:
            train_bst[idx] = True
        
        # 将训练集的 users、num_samples、data 添加到指定的列表中
        train_user_files, test_user_files = [], []
        for idx in range(num_users):
            if train_bst[idx]:
                train_user_files.append(user_files[idx])
            else:
                test_user_files.append(user_files[idx])
        # 切分之后数据集的最大的 user 数量
        if self._ds == "femnist":
            max_size = 50
        else:
            max_size = sys.maxsize
        # 将训练集、测试集切分成一个个的小切片
        sept_train = self.__parser_user_files(train_user_files, max_size)
        sept_test = self.__parser_user_files(test_user_files, max_size)
        return [{"train": train, "test": test} for train, test in zip(sept_train, sept_test)]

    def __parser_user_files(self, user_files, max_size):
        sept_datas = []
        user_count = 0
        users, num_samples, user_data = [], [], {}
        for idx, (u, ns, smpd_data) in enumerate(user_files):
            users.append(u)
            num_samples.append(ns)
            user_data[u] = smpd_data["user_data"][u]
            user_count += 1
            if user_count == max_size or idx == len(user_files) - 1:
                sept_datas.append({
                    "users": users, "num_samples": num_samples, "user_data": user_data
                })
                user_count = 0
        return sept_datas

    def __divided_by_sample(self, smpd_data: dict) -> tuple[dict, dict]:
        """
        按照采样比例, 将 user 中的样本按照其比例进行切分
        :param dict smpd_data: 经过采样的数据，或者全部数据
        :return tuple: 训练集、测试集
        """
        sept_num_samples_train, sept_num_samples_test = [], []
        sept_user_data_train, sept_user_data_test = {}, {}
        # 保留的那些 user 的下标
        sept_user_indices = []

        smpd_users = smpd_data["users"]
        for idx, user in enumerate(smpd_users):
            # 当前 user [playname+charactername] 的样本数量
            smpd_cnum_samples = len(smpd_data["user_data"][user]["y"])
            # 保证该 user 至少有两个样本, 从而生成训练集的样本数量
            if smpd_cnum_samples > 2:
                num_samples_train = max(1, int(self._frac * smpd_cnum_samples))
            elif smpd_cnum_samples == 2:
                num_samples_train = 1
            else:
                continue
            num_samples_test = smpd_cnum_samples - num_samples_train

            # 生成数据切分的采样下标
            if self._ds == "shakespeare":
                train_indices = list(range(num_samples_train))
                test_indices = list(range(num_samples_train + self._seql - 1, smpd_cnum_samples))
            else:
                train_indices = self._rng.sample(list(range(num_samples_train)), num_samples_train)
                test_indices = [idx for idx in range(smpd_cnum_samples) if idx not in train_indices]
            # 数据切分得到的采样下标为空，则直接跳过该 user
            if len (train_indices) <= 0 or len(test_indices) <= 0:
                continue
            # 记录当前切分采样的 user 下标
            sept_user_indices.append(idx)
            # 记录当前 user 的样本数据量
            sept_num_samples_train.append(num_samples_train)
            sept_num_samples_test.append(num_samples_test)

            # 初始化列表用于记录当前 user 的所有样本是否是训练或者测试集中的样本
            train_blst = [False for _ in range(smpd_cnum_samples)]
            test_blst = [False for _ in range(smpd_cnum_samples)]
            for idx in train_indices:
                train_blst[idx] = True
            for idx in test_indices:
                test_blst[idx] = True
        
            # 初始化训练、测试集的切分样本
            sept_user_data_train[user] = {"x": [], "y": []}
            sept_user_data_test[user] = {"x": [], "y": []}
            for idx in range(smpd_cnum_samples):
                if train_blst[idx]:
                    sept_user_data_train[user]["x"].append(smpd_data["user_data"][user]["x"][idx])
                    sept_user_data_train[user]["y"].append(smpd_data["user_data"][user]["y"][idx])
                if test_blst[idx]:
                    sept_user_data_test[user]["x"].append(smpd_data["user_data"][user]["x"][idx])
                    sept_user_data_test[user]["y"].append(smpd_data["user_data"][user]["y"][idx])
            pass
        users = [smpd_data["users"][idx] for idx in sept_user_indices]
        sept_train = {
            "users": users, "user_data": sept_user_data_train, "num_samples": sept_num_samples_train
        }
        sept_test = {
            "users": users, "user_data": sept_user_data_test, "num_samples": sept_num_samples_test
        }
        return sept_train, sept_test
    pass


class ShakeSpeareHandler(object):
    def __init__(
            self, ds: str, smpd_rs: int, smpd_un: float, smpd_df: float, smpd_dt: str, smpd_min_samples: int,
            sept_frac: float, sept_by_type: str, sept_rs: int, save_middle: bool = False
        ) -> None:
        """
        shakespearea 数据集处理器, 用于生成、采样、切分得到最终的用于训练的数据集
        :param str ds: 数据集的名称
        :param int smpd_rs: 采样的随机数种子
        :param float smpd_un: 采样的 users 比例
        :param float smpd_df: 采样的样本与总量的比例
        :param str smpd_dt: 采样的类型, iid、non-iid两种
        :param int smpd_min_samples: 最小采样样本数, 用于去除那些小于该样本数量的 users
        :param float sept_frac: 切分比例，用于生成训练集、测试集
        :param str sept_by_type: 切分类型, 按照sample还是user进行切分生成训练集、测试集
        :param int sept_rs: 切分随机数
        :param bool save_middle: 是否保留中间的结果, defaults to False
        """
        self._ds = ds
        self._smpd_rs = smpd_rs
        self._smpd_un = smpd_un
        self._smpd_df = smpd_df
        self._smpd_dt = smpd_dt
        self._smpd_min_samples = smpd_min_samples
        self._sept_frac = sept_frac
        self._sept_by_type = sept_by_type
        self._sept_rs = sept_rs
        self._save_middle = save_middle

        # 数据集的生成器、采样器、划分器
        self.__generator = Generator()
        self.__sampler = Sampler(self._smpd_rs, self._smpd_un, self._smpd_df, self._smpd_dt, self._smpd_min_samples)
        self.__separator = Separator(self._ds, self._sept_frac, self._sept_by_type, self._sept_rs)
        pass

    def handler(self, preprocessed: str, raw: str, raw_filename: str):
        all_train_file = os.path.join(preprocessed, "all_data", "all_train.pkl")
        if not os.path.exists(all_train_file):
            # 从源数据生成全部数据集的训练集
            all_train_dataset, _ = self.__generator.generator(raw, raw_filename, -1.0)
            if self._save_middle:
                self.__dumps(all_train_dataset, os.path.join(preprocessed, "all_data"), "all_train.pkl")
        else:
            all_train_dataset = self.__load(all_train_file)
        
        smpd_data_file = os.path.join(preprocessed, "smpd_data", f"smpd_data_{self._smpd_dt}.pkl")
        if not os.path.exists(smpd_data_file):
            # 执行采样与去除最小样本数量的 users
            smpd_datas = self.__sampler.sampler(all_datas=[all_train_dataset])
            if self._save_middle:
                sd_smpd_data = { idx: smpd_data for idx, smpd_data in enumerate(smpd_datas) }                    
                self.__dumps(sd_smpd_data, folder=os.path.join(preprocessed, "smpd_data"), filename=f"smpd_data_{self._smpd_dt}.pkl")
        else:
            sd_smpd_data = self.__load(smpd_data_file)
            smpd_datas = [v for _, v in sd_smpd_data.items()]
        
        sept_data_file = os.path.join(preprocessed, "smpd_data", f"sept_data_{self._smpd_dt}.pkl")
        if not os.path.exists(sept_data_file):
            # 数据集切分为训练集与测试集
            sept_datas = self.__separator.separator(smpd_datas=smpd_datas)
            if self._save_middle:
                sd_sept_data = { idx: sept_data for idx, sept_data in enumerate(sept_datas) }            
                self.__dumps(sd_sept_data, folder=os.path.join(preprocessed, "sept_data"), filename=f"sept_data_{self._smpd_dt}.pkl")
        else:
            sd_sept_data = self.__load(sept_data_file)
            sept_datas = [v for _, v in sd_sept_data.items()]
        return sept_datas
    
    def __dumps(self, data: dict, folder: str, filename: str):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(
            os.path.join(folder, filename), "wb"
        ) as f:
            pickle.dump(data, f)
        pass

    def __load(self, filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Data file is not exists, please check it !!!")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data


class ShakeSpeare(Dataset):

    resources = [
        ("http://www.gutenberg.org/files/100/old/1994-01-100.zip", None),
    ]

    raw_folder = "data/shakespeare/raw_data"
    processed_folder = "data/shakespeare/precessed_data"

    raw_file = "100.txt"
    training_file = "training.pkl"
    test_file = "test.pkl"

    # CHARACTERS = string.printable
    CHARACTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    CHAR_NUM = len(CHARACTERS)

    def __init__(
            self,
            train: bool,
            download: bool,
            smpd_un: float,
            smpd_df: float,
            smpd_dt: str,
            sept_frac: float,
            sept_by_type: str,
            rs: int = 1234,
            smpd_min_samples: int = 0,
            save_middle: bool = False,
            user: int | list = 0,
        ) -> None:
        """
        ShakeSpeare数据集
        :param bool train: 是否是训练集
        :param bool download: 是否下载
        :param float smpd_un: 采样多少个 users
        :param float smpd_df: 采样多少个样本
        :param str smpd_dt: 采样类型, iid, non-iid
        :param float sept_frac: 切分比例
        :param str sept_by_type: 切分类型, sample, user
        :param int rs: 采样, 切分随机数种子, defaults to 1234
        :param int smpd_min_samples: 采样最小样本数, defaults to 0
        :param bool save_middle: 是否保存中间数据, defaults to False
        :param int | list user: 当前节点下标, defaults to 0
        :raises RuntimeError: 文件不存在的异常
        """
        self.user = user
        self.train = train

        self.training_file = f"{smpd_dt}_{self.training_file}"
        self.test_file = f"{smpd_dt}_{self.test_file}"

        self._shakespearehandler = ShakeSpeareHandler(
            ds="shakespeare",
            smpd_rs=rs, smpd_un=smpd_un, smpd_df=smpd_df, smpd_dt=smpd_dt, smpd_min_samples=smpd_min_samples,
            sept_rs=rs, sept_frac=sept_frac, sept_by_type=sept_by_type, save_middle=save_middle
        )
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        if not self._check_data_exists():
            print("gen shakespeare")
            self._makedir_exist_ok(self.raw_folder)
            self._makedir_exist_ok(self.processed_folder)
            self._preprocess_shakespeare()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = self._parser(os.path.join(self.processed_folder, data_file), self.user)
        # print(len(self.users_index))

    def __getitem__(self, index):
        return self.data[index], self.targets[index].squeeze()

    def __len__(self):
        return len(self.data)

    def _parser(self, datafile: str, user: int | list):
        with open(datafile, "rb") as f:
            datasets = pickle.load(f)
        if isinstance(user, int):
            user_data = datasets["user_data"]
            user_index = datasets["users"]
            if user > len(user_index):
                raise RuntimeError(f"Input user idx can not great than users length: {len(user_index)} !!!")
            if user >= 0:
                xs = user_data[user_index[user]]["x"]
                ys = user_data[user_index[user]]["y"]
            else:
                users = [idx for idx in range(len(user_index))]
                xs, ys, _ = self._parser_users_data(datasets, users)
        elif isinstance(user, list):
            xs, ys, user_index = self._parser_users_data(datasets, user)
        else:
            raise RuntimeError(f"Unsupport user type: {type(user)} !!!")
        data = [self._char_tensor(x) for x in xs]
        target = [self._char_tensor(y) for y in ys]
        return data, target, user_index
    
    def _parser_users_data(self, datasets: dict, users: list):
        """
        从原始数据集中选择若干个 idx 的构成新的数据集
        :param dict datasets: 原始数据集字典
        :param list users: 带选取的 users
        :raises RuntimeError: users 数量不够时的 抛出异常
        :return tuple: 样本 x 与标签 y
        """
        user_data = datasets["user_data"]
        user_index = datasets["users"]
        if max(users) > len(user_index) or min(users) < 0:
            raise RuntimeError(f"Input user idx can not great than users length: {len(user_index )} !!!")
        xs, ys = [], []
        for u in users:
            xs.extend(user_data[user_index[u]]["x"])
            ys.extend(user_data[user_index[u]]["y"])
        user_index = [user_index[ui] for ui in users ]
        return xs, ys, user_index

    def download(self):
        """Download the Shakespeare data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return
        self._makedir_exist_ok(self.raw_folder)
        self._makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        self._preprocess_shakespeare()

    def _preprocess_shakespeare(self):
        """
        处理生成shakespeare数据集
        """
        data = self._shakespearehandler.handler(self.processed_folder, self.raw_folder, self.raw_file)
        if len(data) != 1:
            raise RuntimeError("Can not handler multiple dataset, where please must the generator, sampler, sepeator correct !!!")
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            pickle.dump(data[0]["train"], f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            pickle.dump(data[0]["test"], f)

    def _char_tensor(self, string: str) -> torch.Tensor:
        """
        将字符串转换成为其索引 (string.printable)
        :param str string: 需要处理的字符串
        :return torch.Tensor: 生成的结果
        """
        tensor = torch.zeros(len(string)).long()
        for s, c in enumerate(string):
            try:
                tensor[s] = self.CHARACTERS.index(c)
            except:
                continue
        return tensor

    def _makedir_exist_ok(self, root):
        if not os.path.exists(root):
            os.makedirs(root)

    def _check_exists(self) -> bool:
        return all(
            utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url)))
            for url, _ in self.resources
        )

    def _check_data_exists(self) -> bool:
        return all (
            utils.check_integrity(os.path.join(self.processed_folder, file))
            for file in [self.training_file, self.test_file]
        )


if __name__ == "__main__":
    shapespeare = ShakeSpeare(
        train=True, download=True,
        smpd_un=0.01, smpd_df=0.01,
        # smpd_dt="non-iid",
        smpd_dt="iid",
        sept_frac=0.2, sept_by_type="sample",
        save_middle=True
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(shapespeare)
    for data, label in dataloader:
        print(data.shape, label.shape)
    pass
