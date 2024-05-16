from functools import reduce
from operator import iconcat
from typing import List

from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession
from torch import from_numpy
from torch.nn import Module
from transformers import (
    AutoConfig,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

model_vocab_size = 30000
original_repo_id = "facebook/blenderbot_small-90M"
repo_id = "remzicam/xs_blenderbot_onnx"
model_file_names = [
    "blenderbot_small-90M-encoder-quantized.onnx",
    "blenderbot_small-90M-decoder-quantized.onnx",
    "blenderbot_small-90M-init-decoder-quantized.onnx",
]


class BlenderEncoder(Module):
    def __init__(self, encoder_sess):
        super().__init__()
        self.encoder = encoder_sess
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_hidden_state = from_numpy(
            self.encoder.run(
                None,
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)


class BlenderDecoderInit(Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):

        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "encoder_attention_mask": encoder_attention_mask.cpu().numpy(),
                "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
            },
        )

        list_pkv = tuple(from_numpy(x) for x in decoder_outputs[1:])

        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return from_numpy(decoder_outputs[0]), out_past_key_values


class BlenderDecoder(Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_output, past_key_values):

        decoder_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "encoder_attention_mask": attention_mask.cpu().numpy(),
        }

        flat_past_key_values = reduce(iconcat, past_key_values, [])

        past_key_values = {
            f"pkv_{i}": pkv.cpu().numpy() for i, pkv in enumerate(flat_past_key_values)
        }

        decoder_outputs = self.decoder.run(None, {**decoder_inputs, **past_key_values})
        # converts each value of the list to tensor from numpy
        list_pkv = tuple(from_numpy(x) for x in decoder_outputs[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return from_numpy(decoder_outputs[0]), out_past_key_values


class OnnxBlender(BlenderbotSmallForConditionalGeneration):
    """creates a Blender model using onnx sessions (encode, decoder & init_decoder)"""

    def __init__(self, original_repo_id, repo_id, file_names):
        config = AutoConfig.from_pretrained(original_repo_id)
        config.vocab_size = model_vocab_size
        super().__init__(config)

        self.files = self.files_downloader(repo_id, file_names)
        self.onnx_model_sessions = self.onnx_sessions_starter(self.files)
        assert len(self.onnx_model_sessions) == 3, "all three models should be given"

        encoder_sess, decoder_sess, decoder_sess_init = self.onnx_model_sessions

        self.encoder = BlenderEncoder(encoder_sess)
        self.decoder = BlenderDecoder(decoder_sess)
        self.decoder_init = BlenderDecoderInit(decoder_sess_init)

    @staticmethod
    def files_downloader(repo_id: str, file_names: List[str]) -> List[str]:
        """Downloads files from huggingface given file names

        Args:

            repo_id (str): repo name at huggingface.
            file_names (List[str]): The names of the files in the repo.

        Returns:
            List[str]: Local paths of files
        """
        return [hf_hub_download(repo_id, file) for file in file_names]

    @staticmethod
    def onnx_sessions_starter(files: List[str]) -> List[object]:
        """initiates onnx inference sessions

        Args:
            files (List[str]): Local paths of files

        Returns:
            List[object]: onnx sessions for each file
        """
        return [*map(InferenceSession, files)]

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


class TextGenerationPipeline:
    """Pipeline for text generation of blenderbot model.
    Returns:
        str: generated text
    """

    # load tokenizer and the model
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(original_repo_id)
    model = OnnxBlender(original_repo_id, repo_id, model_file_names)

    def __init__(self, **kwargs):
        """Specififying text generation parameters.
        For example: max_length=100 which generates text shorter than
        100 tokens. Visit:
        https://huggingface.co/docs/transformers/main_classes/text_generation
        for more parameters
        """
        self.__dict__.update(kwargs)

    def preprocess(self, text) -> str:
        """Tokenizes input text.
        Args:
            text (str): user specified text
        Returns:
            torch.Tensor (obj): text representation as tensors
        """
        return self.tokenizer(text, return_tensors="pt")

    def postprocess(self, outputs) -> str:
        """Converts tensors into text.
        Args:
            outputs (torch.Tensor obj): model text generation output
        Returns:
            str: generated text
        """
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, text: str) -> str:
        """Generates text from input text.
        Args:
            text (str): user specified text
        Returns:
            str: generated text
        """
        tokenized_text = self.preprocess(text)
        output = self.model.generate(**tokenized_text, **self.__dict__)
        return self.postprocess(output)
