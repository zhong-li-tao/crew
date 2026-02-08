1.爬取url=https://image.cailianpress.com/admin/20190906/pdf/ppwKhGe0WmlA/%E5%91%98%E5%B7%A5%E6%89%8B%E5%86%8C.pdf的内容
2.代码放在gain_data.py文件中
3.把爬取到的内容放在名为员工手册的文件下且放在目录名为数据下
4.将txt格式的员工手册正文内容变为结构化的json格式，如[{”第i条“：context }，{”第i+1条“：context }]，context为第i条到第i+1条的内容，导言、总则之类的内容可以忽略
5.将两份gain_data.py文件合并为一份，名为gain_data.py
6.在知识库创建这个目录下，创建名为chunk.py文件，导入langchain的组件，只需要导入能够完成chunk的包
7.在chunk.py文件中，封装一个chunk函数，函数参数为json格式的员工手册，函数返回值为chunk后的list[document],分块规则为每个样本为一块，且提供打印测试代码，打印context
8.创建一个embedding.py文件，放在知识库下，只需要导入langchain的组件，能够完成embedding的包
9.在embedding.py文件中，封装一个embed函数，函数参数为chunk后的list[document],函数返回值为向量化后的list[document]，提供打印测试代码，打印向量化后的结果
10.创建名为utils.py文件，放在知识库下，创建一个名为rag的类，里面封装一个函数，函数参数为本地embedding model的路径，函数返回值为embedding model
11.修改embedding.py文件，在embed函数中，使用rag类的函数来加载embedding model
12.创建名为vector_db.py文件，放在知识库下，先只导入chromadb组件
13.在vector_db.py文件中，封装一个函数，函数参数为向量化后的list[document],函数返回值为vector db，提供打印测试代码，打印vector db
14.创建名为检索层的文件夹，放在根目录下，检索层里面创建名为retriever.py文件，导入langchain的组件和vector_db.py文件中的函数，只需要导入能够完成retriever的包
15.在retriever.py文件中，封装一个retrieve函数，函数参数为vector db和query和k函数返回值为前k个documents，提供打印测试代码，打印retrieved documents
16.创建名为main.py文件，放在根目录下
17.在utils.py文件中，在rag类下封装一个名为llm的函数，参数为api key，函数返回值为llm model，使用langchain的组件来加载llm model
18.在main.py文件中，导入langchain的组件和utils.py文件中的rag类和检索层的函数，先调用retrieve(vector_db, query: str, k: int = 3)函数得到前k个documents，再创建一个提示词模板，模板为“根据上下文回答问题：上下文：{context}问题：{question}”，其中context为前k个documents，question为用户输入的问题，再将query和context注入到模板中，最后调用llm model来生成回答，创建数据库用vector_db中的build_vector_db函数实现，检索通过retriever.py中的retrieve(vector_db, query: str, k: int = 3)实现
19.创建名为model_download.py文件，放在根目录下，实现下载embedding model，把model存放在根目录下
20.创建一个名为test.py文件，放在根目录下，导入ragas组件，进行测试