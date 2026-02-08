#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将员工手册txt文件转换为结构化JSON格式
格式：[{"第i条": context}, {"第i+1条": context}]
context为第i条的内容，导言、总则等内容忽略
"""

import os
import re
import json


def chinese_to_number(chinese_num: str) -> int:
    """将中文数字转换为阿拉伯数字"""
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


def convert_txt_to_json(text_path: str, json_path: str) -> bool:
    """
    将txt格式的员工手册正文内容转换为结构化的JSON格式

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
        # 匹配格式：第X条 标题\n内容（直到下一个第X条或章节标题）
        pattern = r'第\s*([一二三四五六七八九十百千万零0-9]+)\s*条\s*([^\n]*?)\n(.*?)(?=第\s*[一二三四五六七八九十百千万零0-9]+\s*条|第[一二三四五六七八九十]+章|附则|\Z)'

        matches = re.findall(pattern, content, re.DOTALL)

        result = []
        seen_keys = set()

        for match in matches:
            number = match[0].strip()
            title = match[1].strip()
            context = match[2].strip()

            # 跳过目录页的简短内容（通常目录页的内容很短且包含省略号）
            if len(context) < 20 and '...' in context:
                continue

            # 清理内容中的多余空白和换行
            context = re.sub(r'\s+', ' ', context)
            context = context.strip()

            # 移除页码残留
            context = re.sub(r'\d+\s*$', '', context)

            # 只添加有实质内容的条目（长度大于10）
            if context and len(context) > 10:
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

        # 显示前5条作为示例
        print("\n前5条内容示例：")
        for i, item in enumerate(result[:5], 1):
            key = list(item.keys())[0]
            content_preview = item[key][:50] + "..." if len(item[key]) > 50 else item[key]
            print(f"{i}. {key}: {content_preview}")

        return True

    except Exception as e:
        print(f"JSON转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建文件路径
    data_dir = os.path.join(current_dir, "数据", "员工手册")
    text_path = os.path.join(data_dir, "员工手册.txt")
    json_path = os.path.join(data_dir, "员工手册.json")

    print("=" * 60)
    print("员工手册 TXT 转 JSON 工具")
    print("=" * 60)
    print(f"输入文件: {text_path}")
    print(f"输出文件: {json_path}")
    print("=" * 60)

    # 检查输入文件是否存在
    if not os.path.exists(text_path):
        print(f"错误：输入文件不存在: {text_path}")
        return False

    # 执行转换
    success = convert_txt_to_json(text_path, json_path)

    if success:
        print("\n" + "=" * 60)
        print("转换完成！")
        print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
