'''非流式输出'''

import json
from loguru import logger
import os
import tornado
from llama_cpp import Llama
from tornado import ioloop
from tornado.options import define, options

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import langchain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from tornado.escape import json_decode
from langchain.vectorstores import FAISS
from utils.conversation import conv_templates, get_default_conv_template, SeparatorStyle
langchain.verbose = False
import os
path1 = os.path.abspath('.')

def load_faiss_db(db_path,embeddings):
    db = FAISS.load_local(db_path,embeddings = embeddings)
    return db

def merge_faiss_db(db_paths,embeddings):
    db0 = load_faiss_db(db_paths[0],embeddings = embeddings)
    [db0.merge_from(load_faiss_db(i,embeddings = embeddings)) for i in db_paths[1:]]
    return db0

def evaluate(query,history,text):
    conv = get_default_conv_template('conv_vicuna_v1_1').copy()
    if not text:
        if len(history) == 0:
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        else:
            for j, sentence in enumerate(history):
                for i in range(len(sentence)):
                    if i == 0:
                        role = "USER"
                        conv.append_message(role, sentence[0])
                    else:
                        role = "ASSISTANT"
                        conv.append_message(role, sentence[1])
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        if len(history) == 0:
            inp = "Based on the known information below, please provide a concise and professional answer " \
                  "to the user's question.,If the answer cannot be obtained from the information provided, " \
                  "The known content: " + text + "\nquestion:" + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        else:
            for j, sentence in enumerate(history):
                for i in range(len(sentence)):
                    if i == 0:
                        role = "USER"
                        conv.append_message(role, sentence[0])
                    else:
                        role = "ASSISTANT"
                        conv.append_message(role, sentence[1])
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    logger.debug(prompt)
    '''流式输出'''
    stream = llm(
        prompt,
        max_tokens=512,
        stop=["ASSISTANT:"],
        stream=True
    )
    # print(stream)
    return stream

class VicunaHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    # @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        ret = {
            "ret": 1,
            "errcode": 1,
            "response": "",
        }
        try:
            data = json_decode(self.request.body)

            query = data.get("query", "")
            is_file = data.get("is_file",0)
            history = data.get("history",[])
            if not is_file:
                for i in evaluate(query, history, is_file):
                    # print(i)
                    if i["choices"][0]["finish_reason"] == "stop":
                        continue
                    # print(i["choices"][0]["text"])
                    ret["response"] += i["choices"][0]["text"]
                    # print(ret)
                    self.write(ret)
                    self.write('\n')
                    self.flush()
            else:
                if len(history) == 0:
                    files = os.listdir("save_index")
                    new_files = []
                    for file in files:
                        new_path = path1 + "/save_index/" + file
                        new_files.append(new_path)
                    db = merge_faiss_db(new_files,embeddings=embeddings)
                    # todo 增加相似得分的判断
                    docs = db.similarity_search(query, k=2)
                    simdoc = ""
                    for doc in docs:
                        simdoc += doc.page_content
                    # for i in generate_streaming_completion(query,simdoc):
                    history = []
                    for i in evaluate(query, history, simdoc):
                        if i["choices"][0]["finish_reason"] == "stop":
                            continue
                        # print(i["choices"][0]["text"])
                        ret["response"] += i["choices"][0]["text"]
                        # print(ret)
                        self.write(ret)
                        self.write('\n')
                        self.flush()
                    # self.write(ret)
                else:
                    for i in evaluate(query, history, 0):
                        if i["choices"][0]["finish_reason"] == "stop":
                            continue
                        ret["response"] += i["choices"][0]["text"]
                        # print(ret)
                        self.write(ret)
                        self.write('\n')
                        self.flush()

        except Exception as e:
            logger.debug(e)
            pass
        self.write(ret)
        self.finish()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": 1})

def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/healthcheck", MainHandler),
            (r"/nlp/Vicuna_infer_v1", VicunaHandler),

        ],
        debug=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cpp_model", type=str, default='')
    parser.add_argument("--embedding_model", type=str, default='')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--style", type=str, default="simple",
                        choices=["simple", "rich"], help="Display style.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", default=8087)
    args = parser.parse_args()

    from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name=args.embedding_model)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = Llama(model_path=args.cpp_model,n_ctx=2048)
    define("port", default=args.port, help="run on the given port", type=int)

    app = make_app()
    logger.debug('web listen on %d' % options.port)
    app.listen(options.port, xheaders=True)
    ioloop.IOLoop.instance().start()