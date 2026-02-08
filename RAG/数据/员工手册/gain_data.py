#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF爬取与解析工具
功能：爬取指定URL的PDF文件，解析内容并保存到本地，同时转换为结构化JSON格式
"""

import os
import sys
import re
import json
import requests
from urllib.parse import unquote

# 检查并导入PDF解析库
try:
    import PyPDF2
except ImportError:
    print("正在安装 PyPDF2...")
    os.system(f"{sys.executable} -m pip install PyPDF2")
    import PyPDF2


def download_pdf(url: str, save_path: str) -> bool:
    """
    下载PDF文件

    Args:
        url: PDF文件的URL地址
        save_path: 保存路径

    Returns:
        bool: 下载是否成功
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        print(f"正在下载PDF: {url}")
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"PDF下载成功: {save_path}")
        return True

    except Exception as e:
        print(f"PDF下载失败: {e}")
        return False


def parse_pdf(pdf_path: str) -> str:
    """
    解析PDF文件内容

    Args:
        pdf_path: PDF文件路径

    Returns:
        str: 解析出的文本内容
    """
    try:
        text_content = []

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            print(f"PDF总页数: {total_pages}")

            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_content.append(f"\n--- 第 {page_num + 1} 页 ---\n")
                    text_content.append(text)

        return '\n'.join(text_content)

    except Exception as e:
        print(f"PDF解析失败: {e}")
        return ""


def save_text(content: str, save_path: str) -> bool:
    """
    保存文本内容到文件

    Args:
        content: 文本内容
        save_path: 保存路径

    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"文本保存成功: {save_path}")
        return True

    except Exception as e:
        print(f"文本保存失败: {e}")
        return False


def chinese_to_number(chinese_num: str) -> int:
    """
    将中文数字转换为阿拉伯数字

    Args:
        chinese_num: 中文数字字符串

    Returns:
        int: 阿拉伯数字
    """
    chinese_dict = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000
    }

    if not chinese_num:
        return 0

    # 如果是纯数字直接返回
    if chinese_num.isdigit():
        return int(chinese_num)

    result = 0
    temp = 0

    for char in chinese_num:
        if char in chinese_dict:
            num = chinese_dict[char]
            if num >= 10:
                if temp == 0:
                    temp = 1
                result += temp * num
                temp = 0
            else:
                temp = temp * 10 + num if temp > 0 else num

    result += temp
    return result


def convert_to_json(text_path: str, json_path: str) -> bool:
    """
    将txt格式的员工手册正文内容转换为结构化的JSON格式
    格式如：[{"第i条": context}, {"第i+1条": context}]
    其中context为第i条的内容，其它内容忽略

    Args:
        text_path: 文本文件路径
        json_path: JSON文件保存路径

    Returns:
        bool: 转换是否成功
    """
    try:
        # 读取文本内容
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 移除页眉页脚标记
        content = re.sub(r'--- 第 \d+ 页 ---', '', content)
        content = re.sub(r'第 \d+ 页/共 \d+页', '', content)
        content = re.sub(r'界面·财联社', '', content)
        content = re.sub(r'Employee Manual', '', content)

        # 匹配"第X条"的模式
        # 使用更精确的模式，确保匹配的是正文中的条款
        pattern = r'第\s*([一二三四五六七八九十百千万零0-9]+)\s*条\s*([^\n]*?)\n(.*?)(?=第\s*[一二三四五六七八九十百千万零0-9]+\s*条|第[一二三四五六七八九十]+章|附则|\Z)'

        matches = re.findall(pattern, content, re.DOTALL)

        result = []
        seen_keys = set()

        for match in matches:
            number = match[0].strip()
            title = match[1].strip()
            context = match[2].strip()

            # 跳过目录页的简短内容（通常目录页的内容很短）
            if len(context) < 20 and '...' in context:
                continue

            # 清理内容中的多余空白和换行
            context = re.sub(r'\s+', ' ', context)
            context = context.strip()

            # 移除页码残留
            context = re.sub(r'\d+\s*$', '', context)

            if context and len(context) > 10:  # 只添加有实质内容的条目
                key = f"第{number}条"

                # 避免重复
                if key not in seen_keys:
                    seen_keys.add(key)
                    result.append({key: context})

        # 按条款数字排序
        def sort_key(item):
            key = list(item.keys())[0]
            num_str = key.replace('第', '').replace('条', '')
            return chinese_to_number(num_str)

        result.sort(key=sort_key)

        # 保存为JSON
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"JSON转换成功: {json_path}")
        print(f"共提取 {len(result)} 条内容")
        return True

    except Exception as e:
        print(f"JSON转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：爬取PDF并解析保存，同时转换为JSON格式
    """
    # PDF文件的URL
    pdf_url = "https://image.cailianpress.com/admin/20190906/pdf/ppwKhGe0WmlA/%E5%91%98%E5%B7%A5%E6%89%8B%E5%86%8C.pdf"

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建保存路径：数据/员工手册
    data_dir = os.path.join(current_dir, "数据", "员工手册")

    # 从URL解码获取文件名
    decoded_filename = unquote(pdf_url.split('/')[-1])
    pdf_filename = decoded_filename if decoded_filename.endswith('.pdf') else "员工手册.pdf"
    text_filename = pdf_filename.replace('.pdf', '.txt')
    json_filename = pdf_filename.replace('.pdf', '.json')

    pdf_save_path = os.path.join(data_dir, pdf_filename)
    text_save_path = os.path.join(data_dir, text_filename)
    json_save_path = os.path.join(data_dir, json_filename)

    print("=" * 60)
    print("PDF爬取与解析工具")
    print("=" * 60)
    print(f"目标URL: {pdf_url}")
    print(f"PDF保存路径: {pdf_save_path}")
    print(f"文本保存路径: {text_save_path}")
    print(f"JSON保存路径: {json_save_path}")
    print("=" * 60)

    # 步骤1: 下载PDF
    if not download_pdf(pdf_url, pdf_save_path):
        print("程序终止：PDF下载失败")
        return False

    # 步骤2: 解析PDF
    print("\n正在解析PDF内容...")
    text_content = parse_pdf(pdf_save_path)

    if not text_content:
        print("程序终止：PDF解析失败或无内容")
        return False

    print(f"解析完成，共提取 {len(text_content)} 个字符")

    # 步骤3: 保存文本内容
    if not save_text(text_content, text_save_path):
        print("程序终止：文本保存失败")
        return False

    # 步骤4: 转换为JSON格式
    print("\n正在转换为JSON格式...")
    if not convert_to_json(text_save_path, json_save_path):
        print("程序终止：JSON转换失败")
        return False

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"PDF文件: {pdf_save_path}")
    print(f"文本文件: {text_save_path}")
    print(f"JSON文件: {json_save_path}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
