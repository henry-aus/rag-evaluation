# text2sql_query.py
import os
import logging
import yaml
import openai
import re
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymilvus import model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import json
import random

# 1. 环境与日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # 加载 .env 环境变量

# 2. 初始化 OpenAI API（使用最新 Response API）
openai.api_key = os.getenv("OPENAI_API_KEY")

# 建议使用新 Response API 风格
# 例如: openai.chat.completions.create(...) 而非旧的 ChatCompletion.create

MODEL_NAME = os.getenv("OPENAI_MODEL", "o4-mini")

# 3. 嵌入函数初始化
def init_embedding():
    return model.dense.OpenAIEmbeddingFunction(
        model_name='text-embedding-3-large',
    )

# 4. Milvus 客户端连接
MILVUS_DB = os.getenv("MILVUS_DB_PATH", "text2sql_milvus_sakila.db")
client = MilvusClient(MILVUS_DB)

# 5. 嵌入函数实例化
embedding_fn = init_embedding()


# 6. 检索函数
def retrieve(collection: str, query_emb: list, top_k: int = 3, fields: list = None):
    results = client.search(
        collection_name=collection,
        data=[query_emb],
        limit=top_k,
        output_fields=fields
    )
    logging.info(f"[检索] {collection} 检索结果: {results}")
    return results[0]  # 返回第一个查询的结果列表

# 7. SQL 提取函数
def extract_sql(text: str) -> str:
    # 尝试匹配 SQL 代码块
    sql_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()
    
    # 如果没有找到代码块，尝试匹配 SELECT 语句
    select_match = re.search(r'SELECT.*?;', text, re.DOTALL)
    if select_match:
        return select_match.group(0).strip()
    
    # 如果都没有找到，返回原始文本
    return text.strip()

# 8. 核心流程：自然语言 -> SQL
def text2sql(question: str):
    # 8.1 用户提问嵌入
    q_emb = embedding_fn([question])[0]
    logging.info(f"[检索] 问题嵌入完成")

    # 8.2 RAG 检索：DDL
    ddl_hits = retrieve("ddl_knowledge", q_emb.tolist(), top_k=3, fields=["ddl_text"])
    logging.info(f"[检索] DDL检索结果: {ddl_hits}")
    try:
        ddl_context = "\n".join(hit.get("ddl_text", "") for hit in ddl_hits)
    except Exception as e:
        logging.error(f"[检索] DDL处理错误: {e}")
        ddl_context = ""

    # 8.3 RAG 检索：示例对
    q2sql_hits = retrieve("q2sql_knowledge", q_emb.tolist(), top_k=3, fields=["question", "sql_text"])
    logging.info(f"[检索] Q2SQL检索结果: {q2sql_hits}")
    try:
        example_context = "\n".join(
            f"NL: \"{hit.get('question', '')}\"\nSQL: \"{hit.get('sql_text', '')}\"" 
            for hit in q2sql_hits
        )
    except Exception as e:
        logging.error(f"[检索] Q2SQL处理错误: {e}")
        example_context = ""

    # 8.4 RAG 检索：字段描述
    desc_hits = retrieve("dbdesc_knowledge", q_emb.tolist(), top_k=8, fields=["table_name", "column_name", "description"])
    logging.info(f"[检索] 字段描述检索结果: {desc_hits}")
    try:
        desc_context = "\n".join(
            f"{hit.get('table_name', '')}.{hit.get('column_name', '')}: {hit.get('description', '')}"
            for hit in desc_hits
        )
    except Exception as e:
        logging.error(f"[检索] 字段描述处理错误: {e}")
        desc_context = ""

    # 8.5 Prompt 组装
    prompt = (
        f"### Schema Definitions:\n{ddl_context}\n"
        f"### Field Descriptions:\n{desc_context}\n"
        f"### Examples:\n{example_context}\n"
        f"### Query:\n\"{question}\"\n"
        "请只返回SQL语句，不要包含任何解释或说明。"
    )
    logging.info("[生成] 开始生成SQL")

    # 8.6 调用最新 Response API
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        # temperature=0.2
    )
    raw_sql = response.choices[0].message.content.strip()
    sql = extract_sql(raw_sql)
    logging.info(f"[生成] 原始输出: {raw_sql}")
    logging.info(f"[生成] 提取的SQL: {sql}")

    return sql

# 9. 评估指标函数
def evaluate_metrics(pred_sql: str, ref_sql: str):
    # 预处理：分词
    pred_tokens = pred_sql.strip().split()
    ref_tokens = ref_sql.strip().split()
    references = [ref_tokens]

    # BLEU
    bleu = sentence_bleu(
        references, pred_tokens, 
        smoothing_function=SmoothingFunction().method1
    )

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_sql, ref_sql)[0]

    # METEOR
    meteor = meteor_score([ref_sql], pred_sql)

    return {
        "bleu": bleu,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "meteor": meteor
    }

# 10. 程序入口（支持评估）
if __name__ == "__main__":
    with open("q2sql_pairs.json", "r") as f:
        pairs = json.load(f)
        logging.info(f"[Q2SQL] 从JSON文件加载了 {len(pairs)} 个问答对")

    random_evaluations = random.sample(pairs, 10)
    logging.info(f"选择任意十个{random_evaluations}个评估问答对")

    results = []
    for idx, item in enumerate(random_evaluations):
        question = item["question"]
        ref_sql = item["sql"]
        pred_sql = text2sql(question)
        metrics = evaluate_metrics(pred_sql, ref_sql)
        result = {
            "question": question,
            "ref_sql": ref_sql,
            "pred_sql": pred_sql,
            "metrics": metrics
        }
        results.append(result)
        print(f"\n[{idx+1}] 问题: {question}")
        print(f"参考SQL: {ref_sql}")
        print(f"生成SQL: {pred_sql}")
        print("评估指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # 输出整体平均分
    avg_metrics = {k: 0.0 for k in results[0]["metrics"]}
    for r in results:
        for k in avg_metrics:
            avg_metrics[k] += r["metrics"][k]
    for k in avg_metrics:
        avg_metrics[k] /= len(results)
    print("\n==== 平均评估指标 ====")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")



