#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索层模块
功能：使用LangChain组件和ChromaDB实现文档检索
"""

from typing import List, Dict, Any
import json
import os
import sys

# LangChain Retriever相关组件
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

# 导入vector_db.py中的函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '知识库'))
from vector_db import build_vector_db
from embedding import embed
from chunk import chunk
from utils import rag


def retrieve(vector_db, query: str, k: int = 3) -> List[Document]:
    """
    从向量数据库中检索与查询最相关的前k个文档

    参数:
        vector_db: ChromaDB集合对象（向量数据库）
        query: 查询文本
        k: 返回最相关的文档数量，默认为3

    返回值:
        List[Document]: 前k个最相关的Document对象列表
        - 每个Document包含page_content（文本内容）和metadata（元数据）
        - 按相似度从高到低排序

    示例:
        >>> # 假设已经构建好vector_db
        >>> query = "员工试用期是多久？"
        >>> results = retrieve(vector_db, query, k=3)
        >>> for doc in results:
        ...     print(f"条款: {doc.metadata['条款']}")
        ...     print(f"内容: {doc.page_content[:100]}...")
    """
    # 使用rag类加载embedding模型来向量化查询
    embedding_model = rag.load_embedding_model("BAAI/bge-large-zh-v1.5")
    
    # 将查询文本转换为向量
    query_embedding = embedding_model.embed_query(query)
    
    # 在向量数据库中查询最相似的k个文档
    results = vector_db.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 将查询结果转换为Document对象列表
    documents = []
    for i in range(len(results['ids'][0])):
        doc = Document(
            page_content=results['documents'][0][i],
            metadata={
                '条款': results['metadatas'][0][i].get('条款', 'N/A'),
                '来源': results['metadatas'][0][i].get('来源', 'N/A'),
                'distance': results['distances'][0][i]
            }
        )
        documents.append(doc)
    
    return documents


# ==================== 打印测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Retriever函数测试")
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
    
    # 步骤2: 调用embed函数（处理前5个进行测试）
    print("\n步骤2: 执行embed向量化（测试前5个文档）...")
    test_docs = chunked_docs[:5]
    embedded_docs = embed(test_docs, model_path="BAAI/bge-large-zh-v1.5")
    
    # 步骤3: 调用build_vector_db函数构建向量数据库
    print("\n步骤3: 构建向量数据库...")
    persist_dir = os.path.join(current_dir, "..", "知识库", "chroma_db")
    vector_db = build_vector_db(embedded_docs, persist_directory=persist_dir)
    
    # 步骤4: 测试retrieve函数
    print("\n步骤4: 测试retrieve检索功能...")
    test_query = "员工试用期是多久？"
    k = 3
    
    print(f"\n查询: '{test_query}'")
    print(f"返回前 {k} 个相关文档")
    print("=" * 60)
    
    retrieved_docs = retrieve(vector_db, test_query, k=k)
    
    # 打印retrieved documents
    print("\n" + "=" * 60)
    print("检索结果 (Retrieved Documents):")
    print("=" * 60)
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n结果 {i}:")
        print(f"  条款: {doc.metadata['条款']}")
        print(f"  来源: {doc.metadata['来源']}")
        print(f"  相似度距离: {doc.metadata['distance']:.4f}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
        print(f"  内容预览: {doc.page_content[:100]}...")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
