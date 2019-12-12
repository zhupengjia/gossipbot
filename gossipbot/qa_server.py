#!/usr/bin/env python
import time, torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

class QAServer:
    def __init__(self, model_path, config_path, device="cuda:0", timeout=300, **args):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cfg = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(cfg)
        weights = torch.load(model_path)
        weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
        weights.pop("lm_head.decoder.weight",None)
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.model.eval()
        self.timeout = timeout
        self.session = {}
        self.DELIMITER = "/n"

    def _init_session(self):
        return {"conditioned_tokens": [],
                "generated_tokens": [],
                "past": None,
                "fst": True,
                "time": time.time()
        }

    def _reinput(self, user_msg, sid):
        conditioned_tokens = self.tokenizer.decode(self.session[sid]["conditioned_tokens"])
        if self.session[sid]["fst"]:
            self.session[sid]["fst"] = False
            user_msg = "<|endoftext|>" + user_msg

        user_msg = "" + user_msg
        conditioned_tokens += user_msg 
        conditioned_tokens = self.tokenizer.encode(conditioned_tokens) 
        conditioned_tokens += [50256] # Append operator to prepend conversation history
        self.session[sid]["conditioned_tokens"] = conditioned_tokens


    def _top_p_filtering(self, logits, top_p=0.9, filter_value=-float('Inf')):
        """
        Credit: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for single word generation
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits


    def _recalc(self, sid, p=None):
         # for segment display purpose, keep 2 sets of tokens
        indexed_tokens = self.session[sid]["conditioned_tokens"] + self.session[sid]["generated_tokens"]
         
        tokens_tensor = torch.tensor([indexed_tokens])
         
        tokens_tensor = tokens_tensor.to('cuda')
        with torch.no_grad():
            outputs, self.session[sid]["past"] = self.model(tokens_tensor, past=p)
            predictions = outputs
        logits = predictions[0, -1, :]
        filtered_logits = self._top_p_filtering(logits)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        self.session[sid]["generated_tokens"].append(next_token.item())
        return next_token.item()


    def __call__(self, question, sid="default"):
        now = time.time()
        if not sid in self.session or \
           now-self.session[sid]["time"] > self.timeout:
            self.session[sid] = self._init_session()
        self.session[sid]["time"] = now

        self._reinput(question, sid)
        return self._generate(sid), 1


    def _generate(self, sid):
        while True:
            if len(self.tokenizer.decode(self.session[sid]["conditioned_tokens"])) > 320:
                dc = self.tokenizer.decode(self.session[sid]["conditioned_tokens"])
                dc = dc[len(dc)-320:]
                idx = dc.find(self.DELIMITER)
                if idx != -1:
                    dc = dc[idx+len(self.DELIMITER):]
                    self.session[sid]["conditioned_tokens"] = self.tokenizer.encode(dc)

            result = self._recalc(sid)

            if result == 50256:

                  # end-of-text : 50256
                  # use this special token to split segments

               decoded_reply = self.tokenizer.decode(self.session[sid]["generated_tokens"])

               to_print = decoded_reply
               if to_print.endswith("<|endoftext|>"):
                   to_print = to_print[:-len("<|endoftext|>")]
                   decoded_reply = to_print

               decoded_reply = "" + decoded_reply
               decoded_reply = decoded_reply+self.DELIMITER
               self.session[sid]["conditioned_tokens"] += (self.tokenizer.encode(decoded_reply))

               # Remove end of text tokens from feeding back into model
                  # -- see here for reason https://github.com/huggingface/transformers/issues/429#issuecomment-479380117
               cond_str = self.tokenizer.decode(self.session[sid]["conditioned_tokens"])
               cond_str = cond_str.replace("<|endoftext|>",self.DELIMITER) # Since "<end-of-text> (50256)" is mapped to the newline character (\n) by default when calling decode():
               self.session[sid]["conditioned_tokens"] = self.tokenizer.encode(cond_str)

               self.session[sid]["generated_tokens"] = []
               return to_print
