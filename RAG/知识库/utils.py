#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
功能：提供RAG相关的工具类
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


class rag:
    """
    RAG工具类
    用于加载和管理Embedding模型和LLM模型
    """

    @staticmethod
    def load_embedding_model(model_path: str) -> HuggingFaceEmbeddings:
        """
        加载本地Embedding模型

        参数:
            model_path: 本地Embedding模型的路径

        返回值:
            HuggingFaceEmbeddings: 加载好的Embedding模型对象

        示例:
            >>> from utils import rag
            >>> model = rag.load_embedding_model("./models/bge-large-zh-v1.5")
            >>> print("模型加载成功")
        """
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embedding_model

    @staticmethod
    def llm(api_key: str, model_name: str = "glm-4", base_url: str = "https://open.bigmodel.cn/api/paas/v4/") -> ChatOpenAI:
        """
        加载智谱AI的LLM模型

        参数:
            api_key: 智谱AI的API密钥
            model_name: 模型名称，默认为"glm-4"
            base_url: API基础URL，默认为智谱AI的API地址

        返回值:
            ChatOpenAI: 加载好的LLM模型对象

        示例:
            >>> from utils import rag
            >>> llm_model = rag.llm(api_key="your-zhipu-api-key")
            >>> response = llm_model.invoke("你好")
            >>> print(response.content)
        """
        llm_model = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            temperature=0.7
        )
        return llm_model


# ==================== 打印测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("RAG类测试")
    print("=" * 60)

    # 测试1: 加载Embedding模型
    print("\n【测试1】加载Embedding模型")
    print("-" * 60)

    model_name = "BAAI/bge-large-zh-v1.5"
    print(f"测试加载模型: {model_name}")

    try:
        model = rag.load_embedding_model(model_name)
        print("✓ Embedding模型加载成功")
        print(f"模型类型: {type(model)}")

        # 测试向量化功能
        test_text = "这是一个测试文本"
        print(f"\n测试向量化: '{test_text}'")
        embedding = model.embed_query(test_text)
        print(f"✓ 向量化成功")
        print(f"向量维度: {len(embedding)}")
        print(f"向量前5个值: {embedding[:5]}")

    except Exception as e:
        print(f"✗ 加载失败: {e}")

    # 测试2: 加载LLM模型（需要API key，这里仅展示代码结构）
    print("\n【测试2】加载智谱AI LLM模型")
    print("-" * 60)
    print("函数定义: rag.llm(api_key, model_name, base_url)")
    print("默认模型: glm-4")
    print("默认base_url: https://open.bigmodel.cn/api/paas/v4/")
    print("\n使用示例:")
    print("  llm_model = rag.llm(api_key='your-zhipu-api-key')")
    print("  response = llm_model.invoke('你好')")
    print("  print(response.content)")
    print("\n注意: 需要提供有效的智谱AI API key才能测试此功能")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
