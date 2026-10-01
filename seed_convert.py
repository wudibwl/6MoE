"""
    将压缩后的IPv6表示形式展开为32个十六进制字符，中间不加冒号。
"""

from IPy import IP
from tqdm import tqdm
import argparse

SEED_FILE = 'data/candidates5_12ex_results.txt'


def normalizeIPv6(addrs):
    '''
        将压缩的IPv6表示形式展开为32个不带冒号的十六进制字符。
    '''
    normal_addr = []
    for addr in tqdm(addrs):
        norm = IP(addr.strip()).strFullsize()
        addr_32hex = norm[:4]+norm[5:9]+norm[10:14]+norm[15:19]+norm[20:24]+norm[25:29]+norm[30:34]+norm[35:]
        normal_addr.append(addr_32hex)
    return normal_addr
# end_normalizeIPv6


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=SEED_FILE, type=str, required=False, help='IPv6 seed set file to be converted')
    args = parser.parse_args()
    
    # 从文本文件中读取IPv6地址，每个地址占一行。
    with open(args.file, 'r') as f:
        all_addr_ = f.readlines()

    # 将压缩后的IPv6表示形式扩展为32个十六进制字符，且不包含冒号。
    addr_list = normalizeIPv6(all_addr_)
    addrs = '\n'.join(addr_list)           # 在每个地址后添加一个换行符。

    # 将转换后的IPv6地址保存在同一目录下的一个新文件中，并在原文件名后加上'_32hex'。
    f_addr_32hex = SEED_FILE.replace('.txt', '_32hex.txt')
    with open(f_addr_32hex, 'w') as f:
        f.writelines(addrs)

