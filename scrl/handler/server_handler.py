from flask import Flask, request, jsonify
from handler import Handler
import yaml
from openai import OpenAI
from types import SimpleNamespace
import json
import os
import concurrent.futures
from tqdm import tqdm
import time
import threading
import traceback
from lotus.models import LM

app = Flask(__name__)

config = yaml.safe_load(open("./scrl/handler/config.yaml"))
handler_config = SimpleNamespace(
    data_writing_file="./signal/data.json",
    signal_writing_file="./signal/signal.json",
    RESPONSE_SIGNAL=0,
    QUERY_SIGNAL=1,
)
client = LM(
    model="hosted_vllm/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    api_base="http://localhost:8001/v1",
    custom_llm_provider="hosted_vllm",
    max_tokens=1000,
    temperature=0,
    max_ctx_len=128 * 1000,
)
query_save_path_dir = os.path.dirname(config["query_save_path"])
if not os.path.exists(query_save_path_dir):
    print(f"query_save_path_dir {query_save_path_dir} not exists, create it")
    os.makedirs(query_save_path_dir)

handler = Handler(agent_config=config, client=client, handler_config=handler_config)

@app.route('/handle_execution', methods=['POST'])
def handle_execution():
    """
    处理查询请求的接口
    接收query_contents参数，返回处理后的结果
    """
    try:
        # 获取请求数据
        query_contents = request.json.get('query_contents', [])
        if not query_contents:
            return jsonify({"error": "Missing query_contents parameter"}), 400
        query_contents = handler.handle_execution_api(query_contents)
        return jsonify({"query_contents": query_contents}), 200
    except Exception as e:
        print(f"处理请求时出错: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return jsonify({"error": e}), 500

if __name__ == "__main__":
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False) 