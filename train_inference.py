# %% [markdown]
# # 第一部分：使用指令数据对基底模型进行有监督微调
# 在本作业的第一部分，我们将使用Qwen2.5-0.5B基底模型以及alpaca指令数据集，体验如何对LLM做指令微调的训练。
# 
# > 关于Transformer的基本使用教程，可以参考官方推出的[LLM Course](https://huggingface.co/learn/llm-course/chapter2/3)。本次作业要求同学们手写训练代码，不能使用里面提供的Trainer API，关于如何使用PyTorch训练模型，可以参照[这个教程](https://huggingface.co/docs/transformers/v4.49.0/en/training#train-in-native-pytorch)。
# 
# > 对于使用Kaggle进行作业的同学，这里有一份[Kaggle基础使用](https://www.kaggle.com/code/cnlnpjhsy/kaggle-transformers)的简单教学供参考。

# %%
# 如果缺失必要的库，可以使用下面的命令安装
# !pip install torch transformers datasets accelerate

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

# %% [markdown]
# ## 加载模型、tokenizer与数据集
# 本次作业，我们使用通义千问的Qwen2.5-0.5B预训练模型进行微调。对于在本地部署的同学，请事先将模型文件下载到本地；对于在kaggle上进行作业的同学，可以依照kaggle上的教程，将`MODEL_PATH`与`DATASET_PATH`修改为Input中的路径。

# %%
MODEL_PATH = "Qwen2.5-0.5B"
DATASET_PATH = "train.csv"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", dtype="auto")
print(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset = datasets.Dataset.from_csv(DATASET_PATH)
for sample in dataset.select(range(10)):    # 查看前10个样本。思考应该怎么将样本组织成单条完整文本？
    print(sample)

# %% [markdown]
# Qwen为基底模型也提供了对话模板（chat template），对话模板中含有一些特殊的token，可以帮助我们区分说话人的轮次（思考一下为什么要区分？）。我们可以直接以下述“轮次对话”的方式，构造一个样例文本。

# %%
tokenizer.apply_chat_template([
    {"role": "user", "content": "This is a question."},
    {"role": "assistant", "content": "I'm the answer!"}
], tokenize=False
)

# %% [markdown]
# 可以看到每一轮次的对话都以`<|im_end|>`这个token结束。但是基底模型是没有在对话上经过优化的，它并不认得这个终止符。因此我们需要修改tokenizer的终止符，使其知道什么token代表一个对话轮次的结束。

# %%
print(tokenizer.eos_token)  # 原来的终止符
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id  # 也要修改模型的终止符

# %% [markdown]
# 为了与训练后的模型做对比，我们先使用模型自带的generate方法测试一下这个基底模型会生成什么样的文本：

# %%
messages = [
    {"role": "user", "content": "Give me a brief introduction to Shanghai Jiao Tong University."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
with torch.no_grad():
    lm_inputs_src = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**lm_inputs_src, max_new_tokens=150, do_sample=False)
pred_str = tokenizer.decode(generate_ids[0][lm_inputs_src.input_ids.size(1):], skip_special_tokens=True)
print(pred_str)

# %% [markdown]
# ## 处理数据集
# 原始的alpaca数据集是纯文本形式，而非模型能够接受的token。我们需要先将这些文本tokenize，再传给模型。
# 
# 在指令微调阶段，我们常常希望模型只在模型要生成回答的部分上做优化，而不在问题文本上做训练，这需要我们特别设计传入的标签。请完成下述的`tokenize_function`函数，将数据集的指令样本tokenize，并传回输入模型的`input_ids`以及用于<b>仅在output部分计算损失</b>的标签`labels`。

# %%
import copy
def tokenize_function(sample):
    # 构建对话消息列表
    # 注意：Alpaca数据集通常包含 instruction, input (可选), output
    prompt_content = sample["instruction"]
    if sample.get("input", ""): # 如果有input字段且不为空，拼接到instruction后面
        prompt_content += "\n" + sample["input"]
    
    # 1. 构建 Prompt 部分（User）
    messages_prompt = [
        {"role": "user", "content": prompt_content},
    ]
    # 生成 Prompt 的文本（添加 generation prompt 标记，如 <|im_start|>assistant\n）
    prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    
    # 2. 构建 完整对话 部分（User + Assistant）
    messages_full = [
        {"role": "user", "content": prompt_content},
        {"role": "assistant", "content": sample["output"]}
    ]
    # 生成完整对话文本
    full_text = tokenizer.apply_chat_template(messages_full, tokenize=False)
    
    # 3. 将文本转换为 token ids
    # 注意：这里我们分别对 prompt 和 full_text 进行 tokenize，是为了计算 prompt 的长度
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    
    # Qwen2.5 的 tokenizer 可能会在开头自动添加 text_ids，为保险起见，建议加上 add_special_tokens=False
    # 并在之前手动加上 tokenizer.bos_token_id (如果有的话)，但 apply_chat_template 通常处理好了
    
    # 4. 构建 Labels
    # 初始化 labels 为 full_ids 的副本
    labels = copy.deepcopy(full_ids)
    
    # 将 prompt 部分的 labels 设置为 -100，这样计算 loss 时会被忽略
    # 注意：我们要忽略的是 prompt_ids 长度的部分
    prompt_len = len(prompt_ids)
    
    # 这里的切片处理要小心，确保长度一致。
    # 只要 prompt_text 是 full_text 的前缀，这种长度截断就是安全的。
    if len(full_ids) > prompt_len:
        labels[:prompt_len] = [-100] * prompt_len
    else:
        # 异常保护：如果full比prompt还短（极少见），全部忽略
        labels = [-100] * len(labels)

    # 显式添加 EOS token (如果 apply_chat_template 没有加，通常它会加，但 Qwen 需要确认)
    # 之前的代码已经设置了 tokenizer.eos_token_id，这里不做额外操作，依赖 template 结果。
    
    input_ids = full_ids
    
    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(
    tokenize_function, remove_columns=dataset.column_names
).filter(
    lambda x: len(x["input_ids"]) <= 512
)

# %% [markdown]
# 定义一个DataLoader，用于从中获取模型能够处理的tokenized输入。  
# > <b>【附加1】（3分）</b>通过从dataloader中成批取出数据，可以提升计算效率。你能够设计`collate_fn`，使之能以`batch_size > 1`的方式获取数据吗？

# %%
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # 提取 batch 中的 input_ids 和 labels
    # batch 是一个 list，里面每个元素是 tokenize_function 返回的 dict
    input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
    labels_list = [torch.tensor(item["labels"]) for item in batch]
    
    # 1. 对 input_ids 进行 padding
    # batch_first=True 表示返回 (batch_size, seq_len)
    # padding_value 使用 tokenizer.pad_token_id
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # 2. 对 labels 进行 padding
    # padding_value 使用 -100 (计算 Loss 时忽略)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    # 3. 生成 attention_mask
    # input_ids 不等于 pad_token_id 的地方为 1，否则为 0
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

# 根据显存占用情况，可以适当调整batch_size
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# %% [markdown]
# ## 训练模型
# 准备好tokenized后的数据后，就可以对模型进行训练了。请手动编写用于训练的循环，计算损失并反传。
# 
# 在向model传入labels时，Transformer模型内部会自动计算损失；但为了让同学们理解损失的内部计算机制，我们要求**不向模型forward中传入labels，而是手动将模型的最终输出logits与labels相比对，并计算损失。**  
# > <b>【附加1】</b>从dataloader中成批获取数据后，要将整个batch一次性输入到模型中（并非是使用循环逐个处理批次输入），获取所有样例的loss，并正确计算损失。

# %%
# # from tqdm.notebook import tqdm
# from tqdm import tqdm
# from torch.optim import AdamW
# import torch.nn as nn

# step = 0
# # TODO: 定义你的优化器与损失函数
# # 1. 定义优化器与损失函数
# # 学习率通常设置较小，如 1e-5 或 5e-5
# optimizer = AdamW(model.parameters(), lr=1e-5) 

# # CrossEntropyLoss，设置 ignore_index=-100 以忽略 padding 和 prompt 部分
# loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# model.train()
# # 将模型移动到 GPU (如果之前定义时 device_map="auto" 已经移过去了，这里确保一下)
# device = model.device 

# for epoch in range(3):
#     for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
#         # 将数据移动到设备上
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
        
#         # 清空梯度
#         optimizer.zero_grad()
        
#         # 2. 前向传播 (不传入 labels)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
#         # 3. 手动计算 Loss (Shift Logits)
#         # 语言模型的特性：第 t 个 token 的 logits 用于预测第 t+1 个 token
#         # 因此，我们需要将 logits 向左平移一位（去掉最后一个），将 labels 向左平移一位（去掉第一个）
        
#         # shift_logits: [batch_size, seq_len-1, vocab_size]
#         shift_logits = logits[..., :-1, :].contiguous()
#         # shift_labels: [batch_size, seq_len-1]
#         shift_labels = labels[..., 1:].contiguous()
        
#         # 将 tensor 展平以适配 CrossEntropyLoss
#         # view(-1, ...) 相当于 flatten
#         loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
#         # 4. 反向传播与优化
#         loss.backward()
#         optimizer.step()

#         step += 1
#         if step % 100 == 0:
#             print(f"Step {step}\t| Loss: {loss.item()}")
            
#     # 保存逻辑保持不变
#     model.save_pretrained(f"output/checkpoint-epoch-{epoch + 1}")
#     tokenizer.save_pretrained(f"output/checkpoint-epoch-{epoch + 1}")

# %% [markdown]
# 测试训练后的模型效果。如果训练正常，模型应当能回答出通顺的语句，并在回答结束后自然地停止生成。

# %%
sft_model = AutoModelForCausalLM.from_pretrained("output/checkpoint-epoch-3", device_map="auto", dtype="auto")
messages = [
    {"role": "user", "content": "Give me a brief introduction to Shanghai Jiao Tong University."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
with torch.no_grad():
    lm_inputs_src = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(sft_model.device)
    generate_ids = sft_model.generate(**lm_inputs_src, max_new_tokens=150, do_sample=False)
pred_str = tokenizer.decode(generate_ids[0][lm_inputs_src.input_ids.size(1):], skip_special_tokens=True)
print(pred_str)

# %% [markdown]
# 如果模型行为正常，就可以继续前往大作业的第二部分了！

# %% [markdown]
# # 第二部分：使用LLM做推理生成，并解码为自然文本
# 在这一部分，我们将体验LLM是如何逐token进行生成、并解码出自然文本的。我们需要手动实现一个`generate`函数，它能够直接接受用户的自然文本作为输入，并同样以自然文本回复。

# %%
MODEL_PATH = "output/checkpoint-epoch-3"    # 你训练好的模型路径

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

# %% [markdown]
# ## 实现generate
# 请实现下述的generate函数，手动进行模型推理、生成与解码。
# 
# 这个generate函数至少能够接受一个字符串`query`作为输入，限制最大生成token数`max_new_tokens`，并用`do_sample`选择是采用采样还是贪婪搜索进行生成。在使用采样策略生成时，允许设置基础的采样生成参数`temperature`、`top_p`和`top_k`。关于不同的生成策略是如何工作的，可以学习这篇[博客](https://huggingface.co/blog/how-to-generate)。  
# **禁止使用模型自带的`model.generate`方法！**
# 
# > <b>附加2（3分）</b>你能够利用模型的批次输入特性（并非是使用循环逐个处理批次输入），成批次地输入文本、并同时生成新token吗？此时`query`应该可以接受一个字符串列表作为输入。
# 
# > <b>附加3（3分）</b>束搜索（Beam search）允许在解码过程中保留数个次优序列，通过生成过程中维护这些序列，模型能够生成整体更为合理的句子，改善了贪婪搜索中可能会陷入局部最优的问题。你可以在已有的贪婪搜索与采样两种生成策略的基础上实现束搜索吗？此时`num_beams`应允许大于1的值。  
# 关于束搜索，这里有一个[可视化Demo](https://huggingface.co/spaces/m-ric/beam_search_visualizer)演示其运作机理。

# %%
import torch
import torch.nn.functional as F
from typing import Union, List

def post_process_response(text):
    # 1. 优先匹配明确的特殊 Token (Qwen 的标准结束符)
    # 注意：decode后，<|im_end|> 可能会变成字符串形式
    special_stop_patterns = ["<|im_end|>", "<|im_start|>"]
    for pattern in special_stop_patterns:
        if pattern in text:
            text = text.split(pattern)[0]

    # 2. 匹配文本模式的自问自答 (这是你遇到的主要问题)
    # 比如模型自己生成了 "\nUser:" 或者 "\nInput:"
    text_stop_patterns = [
        "\nUser:", "\nuser:", 
        "\nInput:", "\ninput:",
        "\nQ:", "\nQuestion:",
        "\n问：", "\n问题："
    ]
    
    for pattern in text_stop_patterns:
        # 使用 rsplit 还是 split? 通常我们只关心第一次出现
        if pattern in text:
            # 找到模式出现的位置
            idx = text.find(pattern)
            # 只要这个模式出现了，且不是在开头（避免把刚生成的答案全切了），就截断
            if idx > 0: 
                text = text[:idx]
    
    # 3. 处理可能存在的乱码结尾（如 riott 这种孤立词）
    # 这一步比较激进，视情况使用。简单的方法是再做一次 strip
    return text.strip()

def generate(
    model: AutoModelForCausalLM,
    query: Union[str, List[str]],
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_beams: int = 1,
    length_penalty: float = 1.0,
) -> Union[str, List[str]]:
    
    # --- 1. 数据预处理与Batch构造 ---
    # 统一将输入转为 List 处理
    is_single_input = isinstance(query, str)
    queries = [query] if is_single_input else query
    batch_size = len(queries)
    device = model.device

    # 构造 Chat 模板输入
    formatted_queries = []
    for q in queries:
        messages = [{"role": "user", "content": q}]
        # add_generation_prompt=True 会添加 <|im_start|>assistant\n
        formatted_queries.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Tokenize
    # 关键点：生成任务必须使用 Left Padding，因为输出是在右侧生成的
    tokenizer.padding_side = "left" 
    inputs = tokenizer(formatted_queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    input_len = input_ids.shape[1]

    # 设置结束符 ID
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list): eos_token_id = eos_token_id[0]

    # --- 分支：Beam Search 还是 普通解码 ---
    if num_beams > 1:
        # --- 附加3：Beam Search 实现 ---
        # 1. Expand inputs: (batch, seq) -> (batch * beams, seq)
        # 这样我们可以并行处理所有的 beam
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        
        # 初始化分数：每个样本的第一个 beam 分数为 0，其余为 -inf (保证第一次只从第一个beam扩展)
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (batch * beams)

        # 记录生成的序列
        generated_sequences = input_ids
        
        # 记录完成的序列 (batch_size, num_beams)
        # finished_sequences 存储 (score, sequence)
        finished_sequences = [[] for _ in range(batch_size)] 
        
        cur_len = input_len
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated_sequences, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]  # (batch * beams, vocab)

            # 计算 log_softmax
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch * beams, vocab)
            
            # 累加分数: previous_score + current_score
            # beam_scores: (batch * beams, 1)
            next_scores = beam_scores.unsqueeze(-1) + next_token_scores # (batch * beams, vocab)
            
            # Reshape 以便在每个 batch 内部进行 topk
            # (batch, beams * vocab)
            next_scores = next_scores.view(batch_size, -1)
            
            # 取出每个 batch 中分数最高的 2 * num_beams 个候选 (为了留余量给已完成的)
            topk_scores, topk_indices = torch.topk(next_scores, 2 * num_beams, dim=1)
            
            # 解析索引：beam_idx 和 token_idx
            # indices 范围是 [0, beams * vocab - 1]
            beam_indices = topk_indices // model.config.vocab_size
            token_indices = topk_indices % model.config.vocab_size
            
            # 构建下一轮的输入
            next_beam_scores = []
            next_generated_sequences = []
            next_attention_mask = []

            for batch_idx in range(batch_size):
                if len(finished_sequences[batch_idx]) >= num_beams:
                    # 该样本已找齐，随便填点东西占位（最后会被忽略）
                    next_beam_scores.extend([-1e9] * num_beams)
                    next_generated_sequences.extend([generated_sequences[batch_idx * num_beams]] * num_beams)
                    next_attention_mask.extend([attention_mask[batch_idx * num_beams]] * num_beams)
                    continue

                valid_beams_count = 0
                for i in range(2 * num_beams):
                    score = topk_scores[batch_idx, i].item()
                    token = token_indices[batch_idx, i].item()
                    beam_idx = beam_indices[batch_idx, i].item() # 0 ~ num_beams-1
                    
                    # 真正的全局 index
                    global_beam_idx = batch_idx * num_beams + beam_idx
                    
                    if token == eos_token_id:
                        # 句子结束，加入结果集
                        # 长度惩罚： score / (len ** penalty)
                        final_score = score / ((cur_len - input_len + 1) ** length_penalty)
                        finished_sequences[batch_idx].append((final_score, torch.cat([generated_sequences[global_beam_idx], torch.tensor([token], device=device)])))
                    else:
                        # 句子未结束，加入下一轮候选
                        if valid_beams_count < num_beams:
                            next_beam_scores.append(score)
                            new_seq = torch.cat([generated_sequences[global_beam_idx], torch.tensor([token], device=device)])
                            next_generated_sequences.append(new_seq)
                            new_mask = torch.cat([attention_mask[global_beam_idx], torch.tensor([1], device=device)])
                            next_attention_mask.append(new_mask)
                            valid_beams_count += 1
            
            # 更新状态
            beam_scores = torch.tensor(next_beam_scores, device=device)
            generated_sequences = torch.stack(next_generated_sequences)
            attention_mask = torch.stack(next_attention_mask)
            
            cur_len += 1
            
            # 检查是否所有 batch 都完成了
            if all([len(fs) >= num_beams for fs in finished_sequences]):
                break
        
        # 整理输出结果：取分数最高的那一条
        final_sequences = []
        for batch_idx in range(batch_size):
            # 如果没生成完（比如超长），就把当前还在跑的最高分拿出来
            if len(finished_sequences[batch_idx]) == 0:
                 final_sequences.append(generated_sequences[batch_idx * num_beams])
            else:
                # 按分数排序
                finished_sequences[batch_idx].sort(key=lambda x: x[0], reverse=True)
                final_sequences.append(finished_sequences[batch_idx][0][1])
        
        output_ids = torch.nn.utils.rnn.pad_sequence(final_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

    else:
        # --- 默认：Greedy / Sample 解码 ---
        # 复制一份 input_ids 用于拼接生成结果
        generated_ids = input_ids.clone()
        
        # 用于记录每个样本是否已经生成结束
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # 1. 前向传播
            # 这里为了简单，每次都传入完整的 sequence。
            # 实际上可以使用 past_key_values (KV Cache) 来加速，但代码会复杂很多。
            with torch.no_grad():
                outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            
            # 2. 获取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)

            # 3. 采样策略处理
            if do_sample:
                # Temperature
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Softmax 转概率
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-K 过滤
                if top_k > 0:
                    # 获取前k个值的阈值
                    top_k_values, _ = torch.topk(probs, top_k)
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    # 低于阈值的设为 0
                    probs[probs < min_top_k] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True) # 归一化

                # Top-P (Nucleus) 过滤
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # 找出累积概率超过 top_p 的位置
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 需要保留第一个超过 top_p 的 token，所以要把 mask 向右移一位
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # 恢复原序列顺序的 mask
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True) # 归一化
                
                # 随机采样
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪搜索：直接取概率最大的
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # 4. 更新生成的序列
            # 如果该样本已经结束（EOS），则后续生成的 token 保持为 EOS 或 pad，不影响逻辑
            # 但为了保持长度一致，我们通常还是继续 append，只是最后 decode 时截断
            
            # 处理已经结束的句子，保持为 pad_token
            # next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
            
            # 拼接
            next_tokens = next_tokens.unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # 更新 mask (新生成的token也需要被关注)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            # 5. 检查是否结束
            # 如果生成的 token 是 eos_token_id，标记为结束
            # Qwen 的 eos 可能是 list，这里假设是 int
            unfinished_sequences = unfinished_sequences.mul((next_tokens.squeeze() != eos_token_id).long())
            
            if unfinished_sequences.max() == 0:
                break
        
        output_ids = generated_ids

    # --- 3. 解码与后处理 ---
    # 截取生成的 output 部分（去掉输入的 prompt 部分）
    # 注意：因为 input_ids 做过 padding，不同样本的 prompt 长度可能在 tensor 里是不对齐的（虽然左padding对齐了末尾）
    # 但最简单的做法是 decode 整个序列，然后按 string 匹配去掉 prompt，或者利用 input_len 统一截断
    
    # 这里我们只返回生成的“新”token对应的文本
    # 由于是 Left Padding，input 的有效长度是 input_len
    generated_only = output_ids[:, input_len:]
    
    # decoded_outputs = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
    # # 返回结果
    # if is_single_input:
    #     return decoded_outputs[0]
    # else:
    #     return decoded_outputs

    # 【关键修改】：这里改为 False，保留特殊字符以便后处理函数能识别 <|im_end|>
    decoded_outputs = tokenizer.batch_decode(generated_only, skip_special_tokens=False)
    
    clean_results = []
    for text in decoded_outputs:
        # 1. 先进行截断处理
        processed_text = post_process_response(text)
        
        # 2. 如果截断后还残留其他特殊 token (比如 <|endoftext|> 等)，再清洗一次
        # 这里可以使用 replace 把残留的特殊 token 删掉，或者重新 encode 再 decode(skip=True)
        # 简单做法是手动 replace 常见的
        for special in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            processed_text = processed_text.replace(special, "")
            
        clean_results.append(processed_text.strip())
    
    # 返回结果
    if is_single_input:
        return clean_results[0]
    else:
        return clean_results

# %% [markdown]
# ## 测试generate的效果
# 请同学们运行下述单元格，测试你的实现。除了下面提到的句子，同学们也可以自定义更多情况下的输入文本，探究模型在面对不同输入时采用不同解码策略的表现。

# %%
print("="*20 + " #1 贪心解码 (Batch Generation) " + "="*20)
# 测试批量输入
query1 = [
    "Give me a brief introduction to Shanghai Jiao Tong University.", 
    "介绍一下上海交通大学。", 
    "What is the capital of China?"
]

# 调用 generate，注意这里我们直接传入列表，测试【附加2】的 Batch 能力
responses_1 = generate(model, query1, max_new_tokens=256, do_sample=False)

# 打印结果
if isinstance(responses_1, list):
    for i, (q, r) in enumerate(zip(query1, responses_1)):
        print(f"[{i}] 问：{q}")
        print(f"    答：{r.strip()}") # strip() 去除首尾可能的换行
        print("-" * 50)
else:
    # 兼容性处理：如果没实现 Batch，返回的是单个字符串，但这会报错，所以上面加了 isinstance 判断
    print("Error: generate函数返回的不是列表，请检查是否正确实现了批量输入。")


print("\n" + "="*20 + " #2 采样解码 (Sampling) " + "="*20)
query2 = "Tell me a joke about computers."
# 测试单条输入，多次采样
for i in range(3): # 跑5次有点多，改为3次节省时间
    response = generate(model, query2, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)
    print(f"[{i+1}] 问：{query2}")
    print(f"    答：{response.strip()}")
    print("-" * 50)


print("\n" + "="*20 + " #3 【附加3】束搜索解码 (Beam Search) " + "="*20)
query3 = "What is the sum of the first 100 natural numbers? Please think step by step."
# 测试 Beam Search
response_3 = generate(model, query3, num_beams=4, length_penalty=1.0)
print(f"问：{query3}")
print(f"答：{response_3.strip()}")
print("="*60)


