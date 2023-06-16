from loguru import logger
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import requests
import tornado.gen
import tornado.ioloop
import tornado.web
from langchain.schema import Document
from tornado import ioloop
from tornado.options import define, parse_command_line, options
from tornado.escape import json_decode
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, PyPDFLoader
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def load_data_by_langchain(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load_and_split()
    # elif filepath.lower().endswith(".pdf"):
    #     loader = UnstructuredPaddlePDFLoader(filepath)
    #     docs = loader.load()
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()

    return docs
def send_msg(host, payload):
    resp = requests.post(host, data=json.dumps(payload))
    if(resp.status_code == 200):
        return 0
    else:
        logger.debug("Error in send msg")
# 回调函数
def callback_(task_id,status):
    back_host = args.back_url
    send_msg(back_host, {'task_id': task_id,"status":status})


def custom_text_splitter(text, chunk_size=128, chunk_overlap=16):
    chunks = []
    start = 0
    words = text.split()
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
    return chunks

def save_index_for_file(filename,filepath,task_id,embeddings):

    # text = ""
    # with open(file_name, "r") as f:
    #     for line in f:
    #         text += line.strip()
    text = ""
    for i in load_data_by_langchain(filepath):
        text += i.page_content
    # text = load_data_by_langchain(file_name)[0].page_content
    split_texts = custom_text_splitter(text)
    docs = []
    for one_conent in split_texts:
        # docs = load_file("microsoft.txt",sentence_size=SENTENCE_SIZE)
        docs.append(Document(page_content=one_conent + "\n", metadata={"source": filepath}))
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("save_index/%s_index"%filepath.replace("/","_"))
    if os.path.exists("save_index/%s_index" % filepath.replace("/","_")):
        logger.debug("%s has been added"%filepath.replace("/","_"))
        #todo 回调函数
        callback_(task_id,0)
    else:
        callback_(task_id, -1)

class GenerateEmbeddingHandler(tornado.web.RequestHandler):
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

            filename = data.get("filename", "")
            filepath = data.get("filepath", "")
            # text = data.get("text","")
            action = data.get("action","")
            task_id = data.get("task_id","")
            self.generate_embedding(filename,filepath,action,task_id)
            if os.path.exists("save_index/%s_index" % filepath.replace("/","_")):
                ret["response"] = "done"
            else:
                ret["response"] = "error"
            logger.debug('file_name:{},action:{},output:{}.'.format(filename,action,ret["response"]))

        except Exception as e:
            logger.debug(e)
            pass

        self.write(ret)
        self.finish()
    def generate_embedding(self,file_name,filepath,action,task_id):
        from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
        embeddings = HuggingFaceInstructEmbeddings(model_name=args.embedding_model)
        if action == "update":
            # if os.path.exists("save_index/%s_index" % filepath.replace("/","_")):
            #     shutil.rmtree("save_index/%s_index" % filepath.replace("/","_"))
            #     logger.debug("%s has been deleted and next updated "%filepath.replace("/","_"))
            save_index_for_file(file_name, filepath,task_id,embeddings=embeddings)
            logger.debug("%s has been updated " % filepath.replace("/", "_"))
        elif action == "add":
            save_index_for_file(file_name, filepath,task_id,embeddings=embeddings)
        elif action == "delete":
            if os.path.exists("save_index/%s_index" % filepath.replace("/","_")):
                shutil.rmtree("save_index/%s_index" % filepath.replace("/","_"))
                logger.debug("%s has been deleted"%filepath.replace("/","_"))
#                # todo 回调函数

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": 1})

def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/healthcheck", MainHandler),
            (r"/nlp/generate_embedding", GenerateEmbeddingHandler),

        ],
        debug=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--embedding_model", type=str, default='')
    parser.add_argument("--port", default=8055)
    parser.add_argument("--back_url", type=str)
    args = parser.parse_args()
    define("port", default=args.port, help="run on the given port", type=int)
    logger.debug('web listen on %d' % options.port)
    app = make_app()
    app.listen(options.port, xheaders=True)
    ioloop.IOLoop.instance().start()