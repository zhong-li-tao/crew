#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统主程序
功能：整合知识库构建、检索和回答层的完整RAG流程
"""

import os
import sys
import json

# 添加知识库和检索层到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '知识库'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '检索层'))

# 导入知识库模块
from chunk import chunk
from embedding import embed
from vector_db import build_vector_db
from utils import rag

# 导入检索层函数
from retriever import retrieve

# 导入LangChain组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    """
    RAG系统主函数
    完整流程：加载数据 -> 分块 -> 向量化 -> 构建向量数据库 -> 检索 -> 生成回答
    """
    print("=" * 70)
    print("RAG系统 - 员工手册智能问答")
    print("=" * 70)
    
    # 配置参数
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "数据", "员工手册", "员工手册.json")
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "知识库", "chroma_db")
    
    print(f"\n数据路径: {json_path}")
    print(f"向量数据库路径: {persist_dir}")
    print("=" * 70)
    
    # 步骤1: 加载JSON数据
    print("\n【步骤1】加载员工手册数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"✓ 加载完成，共 {len(json_data)} 条记录")
    
    # 步骤2: 分块处理
    print("\n【步骤2】执行文本分块...")
    chunked_docs = chunk(json_data)
    print(f"✓ 分块完成，生成 {len(chunked_docs)} 个文档块")
    
    # 步骤3: 向量化
    print("\n【步骤3】执行向量化...")
    print("注意：向量化全部数据需要较长时间，请耐心等待...")
    embedded_docs = embed(chunked_docs, model_path="BAAI/bge-large-zh-v1.5")
    print(f"✓ 向量化完成")
    
    # 步骤4: 构建向量数据库（使用vector_db中的build_vector_db函数）
    print("\n【步骤4】构建向量数据库...")
    vector_db = build_vector_db(embedded_docs, persist_directory=persist_dir)
    print(f"✓ 向量数据库构建完成")
    print(f"  - 集合名称: {vector_db.name}")
    
    # 步骤5: 加载LLM模型（使用rag类的llm函数）
    print("\n" + "=" * 70)
    print("加载LLM模型")
    print("=" * 70)
    
    # 从环境变量获取API key，如果没有则提示用户输入
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("请输入OpenAI API Key: ").strip()
    
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    print("正在加载LLM模型...")
    # 使用智谱AI的默认配置（model_name和base_url使用utils.py中的默认值）
    llm_model = rag.llm(api_key=api_key)
    print("✓ LLM模型加载完成")
    
    # 交互式问答
    print("\n" + "=" * 70)
    print("RAG问答系统已就绪")
    print("=" * 70)
    print("提示：输入问题获取答案，输入'quit'或'exit'退出")
    print("-" * 70)
    
    while True:
        print()
        query = input("请输入您的问题: ").strip()
        
        if query.lower() in ['quit', 'exit', '退出', 'q']:
            print("\n感谢使用，再见！")
            break
        
        if not query:
            print("问题不能为空，请重新输入")
            continue
        
        try:
            # 步骤6: 检索相关文档（使用retriever.py中的retrieve函数）
            print(f"\n正在检索相关文档...")
            k = 3  # 返回前3个最相关的文档
            retrieved_docs = retrieve(vector_db, query, k=k)
            print(f"✓ 检索完成，找到 {len(retrieved_docs)} 个相关文档")
            
            # 显示检索结果
            print("\n" + "-" * 70)
            print("检索到的相关文档:")
            print("-" * 70)
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\n[{i}] 条款: {doc.metadata['条款']}")
                print(f"    相似度: {doc.metadata['distance']:.4f}")
                print(f"    内容: {doc.page_content[:80]}...")
            
            # 步骤7: 构建上下文（将前k个documents组合成context）
            context = "\n\n".join([
                f"【{doc.metadata['条款']}】{doc.page_content}"
                for doc in retrieved_docs
            ])
            
            # 步骤8: 创建提示词模板
            # 模板为："根据上下文回答问题：上下文：{context}问题：{question}"
            prompt_template = ChatPromptTemplate.from_template(
                "根据上下文回答问题：\n\n上下文：\n{context}\n\n问题：{question}"
            )
            
            # 步骤9: 将query和context注入到模板中
            messages = prompt_template.format_messages(context=context, question=query)
            
            # 步骤10: 调用llm model来生成回答
            print("\n" + "-" * 70)
            print("正在生成回答...")
            print("-" * 70)
            
            response = llm_model.invoke(messages)
            answer = StrOutputParser().parse(response)
            
            # 显示回答
            print("\n" + "=" * 70)
            print("回答:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            
        except Exception as e:
            print(f"\n生成回答时出错: {e}")
            print("请检查API key是否正确，或稍后重试。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
