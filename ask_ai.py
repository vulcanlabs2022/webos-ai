import argparse
import logging
import traceback
import transformers
logging.basicConfig(level=logging.INFO,
                    filename='',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s [%(levelname)s]%(filename)s:%(lineno)d %(module)s %(message)s')
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor
from langchain import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import nvgpu
import json
import tornado.gen
import tornado.ioloop
import tornado.web
from tornado import ioloop
from tornado.concurrent import run_on_executor
from tornado.escape import json_decode
from tornado.options import define, parse_command_line, options
from transformers import LogitsWarper
import torch


from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
prompter = Prompter("")
user_history = {}

class CallbackLogitsWarper(LogitsWarper):
    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback
        self.res_tokens = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        self.res_tokens.append(input_ids[0][-1])
        # result = self.tokenizer.decode(self.res_tokens).lstrip()
        result = self.tokenizer.decode(input_ids[0][-1])
        self.callback(result)
        return scores


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": 1})

# 切分文件
# def custom_text_splitter(text, chunk_size=512, chunk_overlap=64):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunk = text[start:end]
#         chunks.append(chunk)
#         if end == len(text):
#             break
#         start = end - chunk_overlap
#     return chunks
# 重构根据单词数来切分
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

def split_text(text):
    texts = custom_text_splitter(text)
    try:
        with open('tmp/text_chunks.json', 'w') as f:
            logger.info("create done")
            json.dump(texts, f)
            logger.info("save data done")
    except Exception as e:
        logger.info(e)
        logger.info(traceback.format_exc())

def get_prompt(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        response = response.strip().split()
        if len(response) > 100:
            response = response[:100]
        response = ' '.join(response)
        prompt += "[Round {}]\nInstruction:{}\nResponse:{}\n".format(i, old_query, response)
    prompt += "[Round {}]\nInstruction:{}\n### Response:".format(len(history), query)
    return prompt

def evaluate(
        instruction,
        history,
        input=None,
        temperature=0.6,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        user_id=0,
        is_clean_history=False,
        **kwargs,
):
    if len(history) == 0:
        prompt = prompter.generate_prompt(instruction, input)
    else:
        prompt = get_prompt(instruction,history)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=1.15,
        **kwargs,
    )


    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }
    def generate_with_callback(callback=None, **kwargs):
        kwargs.setdefault(
            "stopping_criteria", transformers.StoppingCriteriaList()
        )
        kwargs["stopping_criteria"].append(
            Stream(callback_func=callback)
        )
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(
            generate_with_callback, kwargs, callback=None
        )

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            # new_tokens = len(output) - len(input_ids[0])
            decoded_output = tokenizer.decode(output)

            if output[-1] in [tokenizer.eos_token_id]:
                break

            yield prompter.get_response(decoded_output)


class VicunaHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    # @tornado.gen.coroutine
    async def post(self, *args, **kwargs):
        ret = {
            "ret": 1,
            "errcode": 1,
            "response": "",
        }
        try:
            data = json_decode(self.request.body)

            query = data.get("query", "")
            text = data.get("text","")
            history = data.get("history",[])
            if not text:
                for i in evaluate(query, history,input=None):
                    # print(i)
                    ret["response"] = i
                    self.write(ret)
                    self.write('\n')
                    self.flush()
            else:
                split_text(text)
                # Load text chunks from file
                with open('tmp/text_chunks.json', 'r') as f:
                    logger.info("load data done")
                    texts = json.load(f)

                docsearch = FAISS.from_texts(texts, embeddings)

                docs = docsearch.similarity_search(query, k=2)
                simdoc = ""
                for doc in docs:
                    simdoc += doc.page_content
                history = []
                for i in evaluate(query,history,simdoc):
                    # print(i)
                    ret["response"] = i
                    self.write(ret)
                    self.write('\n')
                    self.flush()

        except Exception:
            # data = json_decode(self.request.body)
            pass

        self.finish()


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
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-name", type=str, default="/data/zhanglei/webos/vicuna-13b")
    parser.add_argument("--base_model", type=str,default="")
    parser.add_argument("--lora_weights", type=str,default="")
    parser.add_argument("--sentence_model", type=str, default="")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Use 8-bit quantization.")
    parser.add_argument("--conv-template", type=str, default="v1",
                        help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--style", type=str, default="simple",
                        choices=["simple", "rich"], help="Display style.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", default=8087)
    args = parser.parse_args()

    base_model = args.base_model
    lora_weights = args.lora_weights
    # lora_weights: str = "/data/zhanglei/webos/alpaca"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    embeddings = HuggingFaceEmbeddings(model_name=args.sentence_model)
    define("port", default=args.port, help="run on the given port", type=int)

    app = make_app()
    logging.info('zuiyou web listen on %d' % options.port)
    app.listen(options.port, xheaders=True)
    ioloop.IOLoop.instance().start()



