#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS测试模块
功能：使用RAGAS框架对RAG系统进行评估测试
"""

import os
import sys
import json

# 添加知识库和检索层到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '知识库'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '检索层'))

# 导入RAGAS组件
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset

# 导入LangChain嵌入模型用于RAGAS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入RAG系统组件
from chunk import chunk
from embedding import embed
from vector_db import build_vector_db
from utils import rag
from retriever import retrieve


def prepare_test_data():
    """
    准备测试数据
    返回测试问题、参考答案和上下文
    """
    # 测试用例：问题、参考答案
    test_cases = [
        {
            "question": "员工试用期是多久？",
            "ground_truth": "根据员工手册，试用期一般为3个月，具体以劳动合同约定为准。"
        },
        {
            "question": "公司的工作时间是怎样规定的？",
            "ground_truth": "公司实行标准工时制，每周工作5天，每天工作8小时。"
        },
        {
            "question": "员工享有哪些假期？",
            "ground_truth": "员工享有法定节假日、年假、病假、婚假、产假等各类假期。"
        }
    ]
    
    return test_cases


def generate_answers(vector_db, test_cases, llm_model, k=3):
    """
    使用RAG系统生成回答
    
    参数:
        vector_db: 向量数据库
        test_cases: 测试用例列表
        llm_model: LLM模型
        k: 检索文档数量
    
    返回值:
        list: 包含问题、回答、上下文的列表
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    results = []
    
    for case in test_cases:
        question = case["question"]
        
        # 检索相关文档
        retrieved_docs = retrieve(vector_db, question, k=k)
        
        # 构建上下文
        contexts = [doc.page_content for doc in retrieved_docs]
        context = "\n\n".join(contexts)
        
        # 创建提示词模板
        prompt_template = ChatPromptTemplate.from_template(
            "根据上下文回答问题：\n\n上下文：\n{context}\n\n问题：{question}"
        )
        
        # 生成回答
        messages = prompt_template.format_messages(context=context, question=question)
        response = llm_model.invoke(messages)
        answer = response.content  # 直接获取AIMessage的content属性
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": case["ground_truth"]
        })
        
        print(f"✓ 问题: {question}")
        print(f"  回答: {answer[:100]}...")
    
    return results


def run_ragas_evaluation(data, llm_model, embeddings):
    """
    运行RAGAS评估

    参数:
        data: 包含question, answer, contexts, ground_truth的数据集
        llm_model: LLM模型，用于RAGAS评估
        embeddings: 嵌入模型，用于answer_relevancy指标

    返回值:
        dict: 评估结果
    """
    # 创建Dataset
    dataset = Dataset.from_list(data)

    # 运行评估，传入自定义LLM和嵌入模型
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,      # 忠实度：回答是否基于上下文
            answer_relevancy,  # 回答相关性：回答是否与问题相关
            context_recall,    # 上下文召回率：上下文是否包含答案
            context_precision  # 上下文精确率：上下文中有多少是相关的
        ],
        llm=llm_model,  # 使用智谱AI进行RAGAS评估
        embeddings=embeddings  # 使用自定义嵌入模型
    )

    return result


def main():
    """
    RAGAS测试主函数
    """
    print("=" * 70)
    print("RAGAS - RAG系统评估测试")
    print("=" * 70)
    
    # 步骤1: 构建向量数据库
    print("\n【步骤1】构建向量数据库...")
    
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "数据", "员工手册", "员工手册.json")
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "知识库", "chroma_db")
    
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 分块
    chunked_docs = chunk(json_data)
    
    # 向量化
    print("正在向量化文档...")
    embedded_docs = embed(chunked_docs, model_path="BAAI/bge-large-zh-v1.5")
    
    # 构建向量数据库
    vector_db = build_vector_db(embedded_docs, persist_directory=persist_dir)
    print("✓ 向量数据库构建完成")
    
    # 步骤2: 加载LLM模型
    print("\n【步骤2】加载LLM模型...")
    
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        api_key = input("请输入智谱AI API Key: ").strip()
    
    llm_model = rag.llm(api_key=api_key)
    print("✓ LLM模型加载完成")

    # 步骤3: 加载RAGAS评估所需的嵌入模型
    print("\n【步骤3】加载RAGAS评估所需的嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ 嵌入模型加载完成")

    # 步骤4: 准备测试数据
    print("\n【步骤4】准备测试数据...")
    test_cases = prepare_test_data()
    print(f"✓ 准备了 {len(test_cases)} 个测试用例")

    # 步骤5: 生成回答
    print("\n【步骤5】使用RAG系统生成回答...")
    data = generate_answers(vector_db, test_cases, llm_model, k=3)
    print("✓ 回答生成完成")

    # 步骤6: 运行RAGAS评估
    print("\n【步骤6】运行RAGAS评估...")
    print("评估指标:")
    print("  - Faithfulness (忠实度): 回答是否基于上下文")
    print("  - Answer Relevancy (回答相关性): 回答是否与问题相关")
    print("  - Context Recall (上下文召回率): 上下文是否包含答案")
    print("  - Context Precision (上下文精确率): 上下文中有多少是相关的")
    print("-" * 70)

    try:
        result = run_ragas_evaluation(data, llm_model, embeddings)

        # 打印评估结果
        print("\n" + "=" * 70)
        print("RAGAS评估结果")
        print("=" * 70)

        # RAGAS返回的是EvaluationResult对象，需要转换为字典
        if hasattr(result, 'to_pandas'):
            # 转换为pandas DataFrame然后打印
            import pandas as pd
            df = result.to_pandas()
            print(df)
            print("\n平均分数:")
            # 只计算数值型列的平均值（metrics列）
            metric_columns = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
            for col in metric_columns:
                if col in df.columns:
                    try:
                        print(f"{col}: {df[col].mean():.4f}")
                    except:
                        pass
        else:
            # 直接打印结果
            print(result)

        print("=" * 70)
        print("评估完成！")

    except Exception as e:
        print(f"\n评估过程中出错: {e}")
        print("请确保ragas和datasets库已正确安装")
        print("安装命令: pip install ragas datasets")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
