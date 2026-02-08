#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding模块
功能：使用LangChain组件完成文本向量化
"""

from typing import List
import json
import os
import sys

# LangChain Embeddings - 用于文本向量化
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# 导入rag类
from utils import rag


def embed(documents: List[Document], model_path: str = "BAAI/bge-large-zh-v1.5") -> List[Document]:
    """
    对chunk后的Document列表进行向量化
    
    参数:
        documents: chunk后的Document对象列表，每个Document包含page_content和metadata
        model_path: 本地Embedding模型的路径或模型名称，默认使用BAAI/bge-large-zh-v1.5
    
    返回值:
        List[Document]: 向量化后的Document对象列表
        - 每个Document的metadata中会增加'embedding'字段，存储对应的向量
        - 向量维度取决于所使用的模型（bge-large-zh-v1.5为1024维）
    
    示例:
        >>> from chunk import chunk
        >>> json_data = [{"第一条": "本手册适用于..."}]
        >>> chunked_docs = chunk(json_data)
        >>> embedded_docs = embed(chunked_docs, model_path="./models/bge-large-zh-v1.5")
        >>> print(f"向量维度: {len(embedded_docs[0].metadata['embedding'])}")
    """
    # 使用rag类的函数加载Embedding模型
    print(f"正在加载Embedding模型: {model_path}")
    embedding_model = rag.load_embedding_model(model_path)
    print("Embedding模型加载完成")
    
    # 提取所有文本内容
    texts = [doc.page_content for doc in documents]
    
    # 批量生成向量
    print(f"正在生成向量，共 {len(texts)} 个文本...")
    embeddings = embedding_model.embed_documents(texts)
    print("向量生成完成")
    
    # 将向量添加到每个Document的metadata中
    for i, doc in enumerate(documents):
        doc.metadata['embedding'] = embeddings[i]
    
    return documents


# ==================== 打印测试代码 ====================
if __name__ == "__main__":
    # 导入chunk函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chunk import chunk
    
    print("=" * 60)
    print("Embedding函数测试")
    print("=" * 60)
    
    # 获取JSON文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(os.path.dirname(current_dir), "数据", "员工手册", "员工手册.json")
    
    print(f"JSON文件路径: {json_path}")
    print("=" * 60)
    
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"\n原始JSON数据条数: {len(json_data)}")
    
    # 步骤1: 调用chunk函数
    print("\n步骤1: 执行chunk分块...")
    chunked_docs = chunk(json_data)
    print(f"分块后Document数量: {len(chunked_docs)}")
    
    # 步骤2: 调用embed函数（只处理前3个进行测试，避免加载模型时间过长）
    print("\n步骤2: 执行embed向量化（测试前3个文档）...")
    test_docs = chunked_docs[:3]
    embedded_docs = embed(test_docs, model_path="BAAI/bge-large-zh-v1.5")
    
    # 打印向量化后的结果
    print("\n" + "=" * 60)
    print("向量化后的结果:")
    print("=" * 60)
    
    for i, doc in enumerate(embedded_docs, 1):
        print(f"\nDocument {i}:")
        print(f"  条款: {doc.metadata['条款']}")
        print(f"  来源: {doc.metadata['来源']}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
        print(f"  内容预览: {doc.page_content[:60]}...")
        print(f"  向量维度: {len(doc.metadata['embedding'])}")
        print(f"  向量前5个值: {doc.metadata['embedding'][:5]}")
        print(f"  向量后5个值: {doc.metadata['embedding'][-5:]}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
