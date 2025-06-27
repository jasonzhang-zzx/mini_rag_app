from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from pathlib import Path

class ChunkDataset:
    def __init__(self, json_dir, max_seq_length=512, min_length=50):
        """
        :param json_dir: JSON分片文件目录
        :param max_seq_length: 模型最大序列长度
        :param min_length: 最小有效文本长度
        """
        self.json_files = sorted([
            os.path.join(json_dir, f) 
            for f in os.listdir(json_dir) 
            if f.endswith('.json')
        ])
        self.max_seq_length = max_seq_length
        self.min_length = min_length
        
    def __iter__(self):
        for fpath in self.json_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                for item in chunks_data:
                    # 直接使用原始chunk文本
                    raw_text = item['text'].strip()
                    
                    # 有效性过滤
                    if self._is_valid_chunk(raw_text):
                        # 动态硬截断（保留前N个字符）
                        processed_text = self._truncate_to_max_length(raw_text)
                        yield InputExample(texts=[processed_text])

    def _is_valid_chunk(self, text):
        """过滤无效文本块"""
        return (
            len(text) >= self.min_length and 
            not text.startswith("Figure") and 
            "http://" not in text
        )

    def _truncate_to_max_length(self, text):
        """直接截断到最大序列长度（按字符）"""
        return text[:self.max_seq_length].rsplit(' ', 1)[0] + '...'  # 确保最后是完整单词

def train_model(json_dir, model_name, output_path):
    # 1. 加载基础模型
    word_embedding_model = models.Transformer(
        model_name,
        max_seq_length=512,
        model_args={'output_hidden_states': True}
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # 2. 准备数据集
    dataset = ChunkDataset(json_dir)
    train_data, val_data = train_test_split(list(dataset), test_size=0.1)
    
    # 3. 自定义对比学习损失
    class DomainAdaptationLoss(losses.MultipleNegativesRankingLoss):
        def forward(self, embeddings, labels):
            # 增强领域负样本权重
            embeddings = self.scale_factor * embeddings
            return super().forward(embeddings, labels)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=256,  # 根据GPU显存调整
        shuffle=True,
        collate_fn=model.smart_batching_collate
    )
    
    # 4. 训练配置
    model.fit(
        train_objectives=[(train_dataloader, DomainAdaptationLoss(model))],
        epochs=15,
        warmup_steps=500,
        optimizer_params={'lr': 3e-5, 'eps': 1e-6},
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(output_path, 'checkpoints'),
        checkpoint_save_steps=1000
    )
    
    # 5. 领域词汇强化
    domain_terms = extract_domain_terms(json_dir)  # 需要实现术语提取函数
    model = reinforce_domain_terms(model, domain_terms)
    
    return model