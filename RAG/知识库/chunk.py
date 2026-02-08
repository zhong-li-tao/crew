#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分块模块
功能：使用LangChain组件对文本进行分块处理
分块规则：每个样本（条款）为一块
"""

from typing import List, Dict
import json
import os

# LangChain文档对象 - 用于表示文档
from langchain_core.documents import Document


def chunk(json_data: List[Dict[str, str]]) -> List[Document]:
    """
    对员工手册JSON数据进行分块处理
    
    分块规则：每个样本（条款）为一块，不再进行细分
    
    参数:
        json_data: JSON格式的员工手册数据，格式为 [{"第i条": "内容"}, {"第i+1条": "内容"}, ...]
    
    返回值:
        List[Document]: 分块后的Document对象列表，每个条款对应一个Document
        - 每个Document对象包含page_content（文本内容）和metadata（元数据）
        - 该返回值可以直接被embedding model的embed_documents方法向量化
    
    示例:
        >>> json_data = [{"第一条": "本手册适用于..."}, {"第二条": "本手册的制定..."}]
        >>> chunks = chunk(json_data)
        >>> print(f"共生成 {len(chunks)} 个chunk")
        >>> for doc in chunks:
        ...     print(f"条款: {doc.metadata['条款']}")
        ...     print(f"内容: {doc.page_content[:50]}...")
    """
    documents = []
    
    for item in json_data:
        for key, value in item.items():
            # 每个条款作为一个chunk
            doc = Document(
                page_content=value,
                metadata={"条款": key, "来源": "员工手册"}
            )
            documents.append(doc)
    
    return documents


# ==================== 打印测试代码 ====================
if __name__ == "__main__":
    # 获取JSON文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(os.path.dirname(current_dir), "数据", "员工手册", "员工手册.json")
    
    print("=" * 60)
    print("Chunk函数测试")
    print("=" * 60)
    print(f"JSON文件路径: {json_path}")
    print("=" * 60)
    
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"\n原始JSON数据条数: {len(json_data)}")
    
    # 调用chunk函数
    result = chunk(json_data)
    
    print(f"分块后Document数量: {len(result)}")
    print(f"返回值类型: {type(result)}")
    print(f"Document对象类型: {type(result[0]) if result else 'N/A'}")
    
    # 打印前5个chunk的详细信息
    print("\n" + "=" * 60)
    print("前5个chunk详细信息:")
    print("=" * 60)
    
    for i, doc in enumerate(result[:5], 1):
        print(f"\nChunk {i}:")
        print(f"  条款: {doc.metadata['条款']}")
        print(f"  来源: {doc.metadata['来源']}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
        print(f"  内容预览: {doc.page_content[:80]}...")
    
    # 验证向量化兼容性
    print("\n" + "=" * 60)
    print("向量化兼容性验证:")
    print("=" * 60)
    texts = [doc.page_content for doc in result]
    print(f"提取的文本列表长度: {len(texts)}")
    print(f"第一条文本类型: {type(texts[0])}")
    print(f"第一条文本长度: {len(texts[0])} 字符")
    print("\n✓ 可以被embedding model的embed_documents方法直接使用")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
