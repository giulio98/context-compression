from abc import ABC

from .causal_dataset_squad import CausalModelingDataset
from transformers import PreTrainedTokenizer
import jinja2
from jinja2.exceptions import TemplateError
import re
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

class LlamaDatasetCustom(CausalModelingDataset, ABC):

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(split, data_config, tokenizer, model)
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length
        self.max_question_length = data_config.question_max_length
        self.max_context_length = data_config.context_max_length
        self.columns_to_remove_for_model = ["example_id", "offset_mapping"]
        self.question_column = data_config.question_column  # e.g. 'question'
        self.context_column = data_config.context_column  # e.g. 'context'
        self.answer_column = data_config.answer_column  # e.g. 'answers'
        self.id_column = data_config.id_column  # e.g. 'id'
        self.system_prompt = data_config.prompt
        self.question_prompt = data_config.question_prompt
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer.padding_side = "left"
        self.use_rag = data_config.use_rag
        if self.use_rag:
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate_input(self, _question, _context):
        prompt = " ".join(["question:", _question.lstrip()])
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content":  prompt.strip()},
            ]
            model_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        except TemplateError:
            messages = [
                {"role": "user", "content": self.system_prompt + prompt.strip()},
            ]
            model_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
        return model_input

    @staticmethod
    def generate_question(_question):
        question = " ".join(["question:", _question.lstrip()])
        return question

    @staticmethod
    def generate_context(_context):
        context = " ".join(["context:", _context.lstrip()])
        return context

    def filter(self, example):
        context = example['context']
        input_example = self.generate_context(context)
        return len(self.tokenizer.encode(input_example)) <= self.max_context_length

    @staticmethod
    def extract_targets(answers):
        if all(isinstance(answer, dict) and 'text' in answer for answer in answers):
            # Handling the structure where the answer is in the form {'text': ..., ...}
            return [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        elif all(isinstance(answer[0], str) for answer in answers):
            # Handling the structure where the answer is a string
            return [answer[0] if len(answer[0]) > 0 else "" for answer in answers]
        else:
            raise ValueError("The structure of the answers field is not recognized.")
    
    def get_word_list(self, s):
        regEx = re.compile('[\W]')   
        res = re.compile(r"([\u4e00-\u9fa5])")
        p1 = regEx.split(s.lower())
        str1_list = []
        for str in p1:
            if res.split(str) == None:
                str1_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str1_list.append(ch)
        return [w for w in str1_list if w.strip()]

    def get_word_len(self, s):
        return len(self.get_word_list(s))

    def split_long_sentence(self, sentence, regex, chunk_size=200):
        chunks = []
        sentences = re.split(regex, sentence)
        current_chunk = ""
        for s in sentences:
            if current_chunk and self.get_word_len(current_chunk) + self.get_word_len(s) <= chunk_size:
                current_chunk += ' ' if s == '' else s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = s
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    def tokenize_contexts_rag(self, contexts, questions):
        top_chunks = {'512': 2, '1000': 4, '2000': 8}
        chunk_count = top_chunks.get(str(self.max_context_length), 0)
        # print(f"Expected number of top chunks: {chunk_count}")

        # Determine the device based on CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(f"Using device: {device}")

        tokenized_contexts = []
        for context, question in zip(contexts, questions):
            # Split the context into chunks
            chunks = self.split_long_sentence(context, r'([。？！；\n.!?;]\s*)', chunk_size=256)
            # print(f"Number of chunks from context: {len(chunks)}")

            # Get embeddings of chunks
            chunk_embeddings = self.sentence_model.encode(chunks, convert_to_tensor=True).to(device)
            question_embedding = self.sentence_model.encode([question], convert_to_tensor=True).to(device)
            # print(f"Chunk embeddings shape: {chunk_embeddings.shape}")
            # print(f"Question embedding shape: {question_embedding.shape}")

            # Reshape question_embedding to match dimensions (1, D)
            cos_similarities = F.cosine_similarity(question_embedding, chunk_embeddings, dim=1)
            # print(f"Cosine similarities shape: {cos_similarities.shape}")

            # Get top relevant chunks using top indices
            top_chunk_indices = cos_similarities.argsort(descending=True)[:chunk_count]
            # print(f"Top chunk indices: {top_chunk_indices.tolist()}")

            relevant_chunks = ' '.join([chunks[i] for i in top_chunk_indices])

            # Tokenize the concatenated top chunks
            tokenized = self.tokenizer(
                relevant_chunks,
                add_special_tokens=False,
                max_length=self.max_context_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation=True,
                return_attention_mask=True
            )
            # if len((context)) < 4000:
            #     print("CONTEXT------------------------")
            #     print(len(context))
            #     print(context)
            #     print("CHUNKS--------------------------")
            #     print(relevant_chunks)
            #     print(len(tokenized["input_ids"]))
            tokenized_contexts.append(tokenized)
        
        input_ids = [tokenized['input_ids'] for tokenized in tokenized_contexts]
        attention_masks = [tokenized['attention_mask'] for tokenized in tokenized_contexts]
        tokenized['input_ids'] = input_ids
        tokenized['attention_mask'] = attention_masks
        
        return tokenized



    def tokenize(self, examples):
        if self.question_prompt is not None:
            questions = [self.question_prompt] * len(examples[self.context_column])
        else:
            questions = examples[self.question_column]

        contexts = examples[self.context_column]
        answers = examples[self.answer_column]
        targets = self.extract_targets(answers)

        inputs = [self.generate_input(question, context) for question, context in zip(questions, contexts)]
        if self.question_prompt is not None:
            questions = [self.question_prompt] * len(contexts)
        else:
            questions = [self.generate_question(question) for question in questions]

        max_length = self.max_seq_length + self.max_answer_length
        contexts = [self.generate_context(context) for context in contexts]
        if self.split == "train":
            tokenized_examples = self.tokenizer(
                inputs,
                targets,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation="only_first"
            )
        else:
            tokenized_examples = self.tokenizer(
                inputs,
                targets,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation="only_first",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
        tokenized_questions = self.tokenizer(
            questions,
            add_special_tokens=False,
            max_length=self.max_question_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )
        

        if self.max_context_length in [256, 128, 64, 32, 512, 1000, 2000]:
            if self.use_rag:
                print("RAG!")
                tokenized_contexts = self.tokenize_contexts_rag(contexts, questions)
            else:
                print("Truncate context!")
                tokenized_contexts = self.tokenizer(
                    contexts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=True,
                    max_length=None
                )
                half_max_length = self.max_context_length // 2
                new_input_ids = []
                new_attention_masks = []

                for ids, mask in zip(tokenized_contexts['input_ids'], tokenized_contexts['attention_mask']):
                    selected_ids = ids[:half_max_length] + ids[-half_max_length:]
                    selected_mask = mask[:half_max_length] + mask[-half_max_length:]

                    new_input_ids.append(selected_ids)
                    new_attention_masks.append(selected_mask)

                tokenized_contexts['input_ids'] = new_input_ids
                tokenized_contexts['attention_mask'] = new_attention_masks

                # Now apply padding if required
                if self.pad_to_max_length:
                    tokenized_contexts = self.tokenizer.pad(
                        tokenized_contexts,
                        padding="max_length",
                        max_length=self.max_context_length
                    )
        else:
            print("Normal truncation!")
            tokenized_contexts = self.tokenizer(
                contexts,
                add_special_tokens=False,
                max_length=self.max_context_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation=True
            )
        labels = self.tokenizer(targets, add_special_tokens=False)
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):
            tokenized_examples["input_ids"][idx] = input_ids + [self.tokenizer.eos_token_id]
            tokenized_examples["attention_mask"][idx] = tokenized_examples["attention_mask"][idx] + [1]
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):

            label_input_ids = labels["input_ids"][idx] + [self.tokenizer.eos_token_id]
            labels["input_ids"][idx] = [-100] * (len(input_ids) - len(label_input_ids)) + label_input_ids
        if self.split == "train":
            tokenized_examples["labels"] = labels["input_ids"]
        else:
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id, and we will store the offset mappings.
            tokenized_examples["example_id"] = []
            # Augment the overflowing tokens to the labels
            labels_out = []

            for i in range(len(tokenized_examples["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples[self.id_column][sample_index])
                labels_out.append(labels["input_ids"][sample_index])

            tokenized_examples["labels"] = labels_out
        tokenized_examples["question_ids"] = []
        tokenized_examples["question_attention_mask"] = []
        tokenized_examples["context_ids"] = []
        tokenized_examples["context_attention_mask"] = []
        tokenized_examples["question_ids"] = tokenized_questions["input_ids"]
        tokenized_examples["question_attention_mask"] = tokenized_questions["attention_mask"]
        tokenized_examples["context_ids"] = tokenized_contexts["input_ids"]
        tokenized_examples["context_attention_mask"] = tokenized_contexts["attention_mask"]
        return tokenized_examples

