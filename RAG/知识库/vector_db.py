#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库模块
功能：使用ChromaDB构建和管理向量数据库
"""

from typing import List
import json
import os
import sys

# ChromaDB组件
import chromadb
from chromadb.config import Settings

# LangChain Document
from langchain_core.documents import Document


def build_vector_db(documents: List[Document], 
                    persist_directory: str = "./chroma_db") -> chromadb.Collection:
    """
    根据向量化后的Document列表构建向量数据库

    参数:
        documents: 向量化后的Document对象列表，每个Document的metadata中包含'embedding'字段
        persist_directory: 向量数据库持久化目录，默认为"./chroma_db"

    返回值:
        chromadb.Collection: ChromaDB集合对象，即构建好的向量数据库

    示例:
        >>> from embedding import embed
        >>> from chunk import chunk
        >>> json_data = [{"第一条": "本手册适用于..."}]
        >>> chunked_docs = chunk(json_data)
        >>> embedded_docs = embed(chunked_docs)
        >>> vector_db = build_vector_db(embedded_docs, persist_directory="./chroma_db")
        >>> print(f"向量数据库集合名称: {vector_db.name}")
    """
    # 确保目录存在
    os.makedirs(persist_directory, exist_ok=True)
    print(f"向量数据库持久化目录: {persist_directory}")

    # 创建ChromaDB客户端
    client = chromadb.Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))

    # 创建或获取集合
    collection_name = "employee_handbook"

    # 如果集合已存在，先删除
    try:
        client.delete_collection(collection_name)
        print(f"已删除旧的集合: {collection_name}")
    except:
        pass

    collection = client.create_collection(name=collection_name)
    print(f"创建新集合: {collection_name}")

    # 准备数据
    ids = [f"doc_{i}" for i in range(len(documents))]
    texts = [doc.page_content for doc in documents]
    embeddings = [doc.metadata['embedding'] for doc in documents]
    metadatas = [{k: v for k, v in doc.metadata.items() if k != 'embedding'} for doc in documents]

    # 添加到集合
    print(f"正在添加 {len(documents)} 条记录到向量数据库...")
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("向量数据库构建完成")

    return collection


# ==================== 打印测试代码 ====================
if __name__ == "__main__":
    # 导入相关模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chunk import chunk
    from embedding import embed

    print("=" * 60)
    print("Vector DB函数测试")
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

    # 步骤2: 调用embed函数（只处理前3个进行测试）
    print("\n步骤2: 执行embed向量化（测试前3个文档）...")
    test_docs = chunked_docs[:3]
    embedded_docs = embed(test_docs, model_path="BAAI/bge-large-zh-v1.5")

    # 步骤3: 调用build_vector_db函数
    print("\n步骤3: 构建向量数据库...")
    persist_dir = os.path.join(current_dir, "chroma_db")
    vector_db = build_vector_db(embedded_docs, persist_directory=persist_dir)

    # 打印vector db信息
    print("\n" + "=" * 60)
    print("向量数据库信息:")
    print("=" * 60)
    print(f"集合名称: {vector_db.name}")
    print(f"集合元数据: {vector_db.metadata}")

    # 查询测试
    print("\n" + "=" * 60)
    print("向量数据库查询测试:")
    print("=" * 60)
    
    # 获取第一条记录的向量进行查询
    test_embedding = embedded_docs[0].metadata['embedding']
    results = vector_db.query(
        query_embeddings=[test_embedding],
        n_results=2
    )
    
    print(f"查询结果数量: {len(results['ids'][0])}")
    for i, (doc_id, text, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n结果 {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  条款: {metadata.get('条款', 'N/A')}")
        print(f"  内容预览: {text[:50]}...")
        print(f"  距离: {distance:.4f}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
